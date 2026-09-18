"""Preset resolution and lazy backend orchestration for depth estimation.

The module preserves the current estimator request/response contract without
importing a model registry, Torch, simulator code, or benchmark paths.  Model
factories and backends are explicit injected values and are constructed only
when inference is requested.
"""

from __future__ import annotations

import copy
import importlib
import importlib.resources
import inspect
import json
import math
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ...model_assets.dvd_identity import (
    DVD_ASSET_ATTESTATION_FIELD,
    DVD_MODEL_FAMILIES,
    DVD_MODEL_FAMILY,
    DVD_OFFICIAL_MODEL_FAMILY,
    DVD_PROVENANCE_VALIDATION,
    normalize_dvd_model_identity,
    normalize_dvd_model_provenance,
)
from .attestation import attest_dvd_asset_files


SUPPORTED_DEPTH_BACKENDS = ("dvd", "vda")
DEPTH_MODEL_NAMES = SUPPORTED_DEPTH_BACKENDS
DEFAULT_DEPTH_PRESET = "dvd_lora_specific"
DEPTH_ESTIMATOR_PRESETS = (
    "dvd_official",
    "dvd_lora_shared",
    "dvd_lora_specific",
    "vda_metric",
    "vda_non_metric",
)
DEPTH_PRESET_ALIASES = {
    "dvd_base": "dvd_official",
    "video_depth_anything": "vda_non_metric",
    "video_depth_anything_metric": "vda_metric",
}
VDA_ENCODERS = ("vits", "vitb", "vitl")
INVALIDATE_MODES = ("none", "nan")
DVD_CHECKPOINT_SELECTOR_MODES = ("single_task_by_uid",)
DVD_CHECKPOINT_SELECTOR_MISSING_POLICIES = ("error",)


def _seed_torch_runtime(seed: int) -> None:
    """Mirror the maintained estimator's lazy Torch seeding behavior."""

    try:
        torch = importlib.import_module("torch")
    except ModuleNotFoundError:
        # Explicit non-Torch third-party backends remain usable. The built-in
        # DVD and VDA providers report their own dependency error when needed.
        return
    torch.manual_seed(int(seed))


BackendFactory = Callable[..., Any]
BackendRegistry = Mapping[str, Any]
_DEPTH_CONFIG_ALLOWED_KEYS = frozenset(
    {
        "calibration",
        "dvd",
        "dvd_checkpoint_selector",
        "model_family",
        "model_name",
        "model_provenance",
        "vda",
    }
)
_CALIBRATION_REQUIRED_KEYS = frozenset(
    {
        "calib_region",
        "calibration_solver",
        "if_calibrate_depth",
        "invalidate_mode",
        "multi_roi_strategy",
        "points_radius_px",
        "require_calibration_for_nonmetric",
        "roi_dilate_px",
    }
)
_CALIBRATION_ALLOWED_KEYS = frozenset(
    {
        *_CALIBRATION_REQUIRED_KEYS,
        "background_nearfield_quantile",
    }
)
_VDA_CONFIG_ALLOWED_KEYS = frozenset({"encoder", "metric", "ckpt_root"})
_DVD_CONFIG_ALLOWED_KEYS = frozenset(
    {
        "ckpt_root",
        "model_config_path",
        "resize_height",
        "resize_width",
        "window_size",
        "overlap",
        "scale_only_alignment",
        "channel_reduce",
        "invert_disparity_to_depth",
        "min_disparity",
        "allow_download",
    }
)
_DVD_SELECTOR_ALLOWED_KEYS = frozenset(
    {
        "mode",
        "root",
        "checkpoint_filename",
        "model_config_filename",
        "missing_policy",
    }
)
_DEPTH_RUNTIME_ALLOWED_KEYS = frozenset(
    {
        "config",
        "config_source",
        "preset",
        "model_name",
        "model_family",
        "model_provenance",
        "model_kwargs",
        "if_calibrate_depth",
        "calibration_solver",
        "multi_roi_strategy",
        "calib_region",
        "roi_dilate_px",
        "points_radius_px",
        "invalidate_mode",
        "require_calibration_for_nonmetric",
    }
)
_VDA_RUNTIME_KWARGS_ALLOWED_KEYS = frozenset(
    {"encoder", "metric", "device", "ckpt_root"}
)
_DVD_RUNTIME_KWARGS_ALLOWED_KEYS = frozenset(
    {*_DVD_CONFIG_ALLOWED_KEYS, "device", "model_family", "model_provenance"}
)


def _reject_unknown_mapping_fields(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    source: str,
) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"{source} contains unknown fields: {', '.join(unknown)}")


def _require_exact_bool(
    value: Any,
    *,
    field: str,
    source: str,
) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{field} must be a boolean in {source}")
    return value


def _require_integer(
    value: Any,
    *,
    field: str,
    source: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field} must be an integer in {source}")
    return value


def _require_finite_number(
    value: Any,
    *,
    field: str,
    source: str,
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field} must be a finite number in {source}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{field} must be a finite number in {source}")
    return number


def _unsupported_depth_backend_error(
    requested: Any,
    *,
    source: str,
) -> ValueError:
    return ValueError(
        f"Unsupported depth backend {requested!r} in {source}. "
        f"Supported backends: {list(SUPPORTED_DEPTH_BACKENDS)!r}. "
        "No fallback or remap is performed."
    )


def _validate_calibration_numeric_types(
    calibration: Mapping[str, Any],
    *,
    source: str,
) -> None:
    for field in (
        "if_calibrate_depth",
        "require_calibration_for_nonmetric",
    ):
        if field in calibration:
            _require_exact_bool(
                calibration[field],
                field=f"calibration.{field}",
                source=source,
            )
    for field in ("roi_dilate_px", "points_radius_px"):
        if field not in calibration:
            continue
        value = _require_integer(
            calibration[field],
            field=f"calibration.{field}",
            source=source,
        )
        if value < 0:
            raise ValueError(f"calibration.{field} must be >= 0 in {source}")
    if (
        "background_nearfield_quantile" in calibration
        and calibration["background_nearfield_quantile"] is not None
    ):
        value = _require_finite_number(
            calibration["background_nearfield_quantile"],
            field="calibration.background_nearfield_quantile",
            source=source,
        )
        if not 0.0 < value <= 1.0:
            raise ValueError(
                "calibration.background_nearfield_quantile must be "
                f"in (0,1] in {source}"
            )


def _validate_vda_options(
    options: Mapping[str, Any],
    *,
    source: str,
) -> None:
    if "metric" in options:
        _require_exact_bool(
            options["metric"],
            field="vda.metric",
            source=source,
        )
    if "encoder" in options and options["encoder"] not in VDA_ENCODERS:
        raise ValueError(f"Invalid vda.encoder in {source}: {options['encoder']}")


def _validate_dvd_options(
    options: Mapping[str, Any],
    *,
    source: str,
) -> None:
    for field in (
        "scale_only_alignment",
        "invert_disparity_to_depth",
        "allow_download",
    ):
        if field in options:
            _require_exact_bool(
                options[field],
                field=f"dvd.{field}",
                source=source,
            )
    integer_limits = {
        "resize_height": 1,
        "resize_width": 1,
        "window_size": 1,
        "overlap": 0,
    }
    for field, minimum in integer_limits.items():
        if field not in options:
            continue
        value = _require_integer(
            options[field],
            field=f"dvd.{field}",
            source=source,
        )
        if value < minimum:
            comparator = "> 0" if minimum == 1 else ">= 0"
            raise ValueError(f"dvd.{field} must be {comparator} in {source}")
    window_size = options.get("window_size", 81)
    overlap = options.get("overlap", 9)
    if (
        isinstance(window_size, int)
        and not isinstance(window_size, bool)
        and isinstance(overlap, int)
        and not isinstance(overlap, bool)
        and overlap >= window_size
    ):
        raise ValueError(
            f"dvd.overlap must satisfy 0 <= overlap < dvd.window_size in {source}"
        )
    if "channel_reduce" in options and options["channel_reduce"] not in {
        "first",
        "mean",
    }:
        raise ValueError(f"dvd.channel_reduce must be 'first' or 'mean' in {source}")
    if "min_disparity" in options:
        value = _require_finite_number(
            options["min_disparity"],
            field="dvd.min_disparity",
            source=source,
        )
        if value <= 0.0:
            raise ValueError(f"dvd.min_disparity must be > 0 in {source}")


def validate_depth_runtime_identity(
    runtime_config: Mapping[str, Any],
    *,
    source: str = "depth runtime",
) -> dict[str, Any]:
    """Validate an in-memory current implementation depth-family declaration.

    This gate deliberately performs no imports or filesystem access. For DVD,
    it validates either the official upstream or project-fine-tuned family,
    provenance, and optional complete hash declaration. File-backed asset
    validation is a separate pre-factory gate and is the only operation that
    can promote a declaration to ``verified``.
    """

    if not isinstance(runtime_config, Mapping):
        raise TypeError(f"{source} must be a mapping")
    runtime = copy.deepcopy(dict(runtime_config))
    _reject_unknown_mapping_fields(
        runtime,
        allowed=_DEPTH_RUNTIME_ALLOWED_KEYS,
        source=source,
    )
    model_name = str(runtime.get("model_name", "") or "").strip().lower()
    if model_name not in SUPPORTED_DEPTH_BACKENDS:
        raise _unsupported_depth_backend_error(
            model_name,
            source=f"{source}.model_name",
        )
    runtime["model_name"] = model_name
    nested_config = runtime.get("config")
    if nested_config is not None:
        if not isinstance(nested_config, Mapping):
            raise ValueError(f"{source}.config must be an object")
        validate_depth_estimator_config(
            nested_config,
            source=f"{source}.config",
        )
        normalized_config = normalize_depth_estimator_config(
            nested_config,
            source=f"{source}.config.model_name",
        )
        if normalized_config.get("model_name") != model_name:
            raise ValueError(
                f"{source}.config.model_name conflicts with {source}.model_name"
            )
        runtime["config"] = normalized_config
    for field in (
        "if_calibrate_depth",
        "require_calibration_for_nonmetric",
    ):
        if field in runtime:
            _require_exact_bool(
                runtime[field],
                field=field,
                source=source,
            )
    for field in ("roi_dilate_px", "points_radius_px"):
        if field not in runtime:
            continue
        value = _require_integer(
            runtime[field],
            field=field,
            source=source,
        )
        if value < 0:
            raise ValueError(f"{field} must be >= 0 in {source}")
    if model_name == "vda":
        model_kwargs = runtime.get("model_kwargs")
        if model_kwargs is not None:
            if not isinstance(model_kwargs, Mapping):
                raise ValueError(f"{source}.model_kwargs must be an object")
            _reject_unknown_mapping_fields(
                model_kwargs,
                allowed=_VDA_RUNTIME_KWARGS_ALLOWED_KEYS,
                source=f"{source}.model_kwargs",
            )
            _validate_vda_options(
                model_kwargs,
                source=f"{source}.model_kwargs",
            )
            runtime["model_kwargs"] = copy.deepcopy(dict(model_kwargs))
        return runtime

    provenance = runtime.get("model_provenance")
    try:
        model_family, normalized_provenance = normalize_dvd_model_identity(
            runtime.get("model_family"),
            provenance,
            source=source,
        )
    except ValueError as error:
        if provenance is None:
            raise ValueError(
                f"{source} requires model_provenance identifying an official "
                "or project fine-tuned DVD checkpoint."
            ) from error
        raise
    runtime["model_family"] = model_family
    runtime["model_provenance"] = normalized_provenance

    model_kwargs = runtime.get("model_kwargs")
    if model_kwargs is not None and not isinstance(
        model_kwargs,
        Mapping,
    ):
        raise ValueError(f"{source}.model_kwargs must be an object")
    if isinstance(model_kwargs, Mapping):
        _reject_unknown_mapping_fields(
            model_kwargs,
            allowed=_DVD_RUNTIME_KWARGS_ALLOWED_KEYS,
            source=f"{source}.model_kwargs",
        )
        _validate_dvd_options(
            model_kwargs,
            source=f"{source}.model_kwargs",
        )
        normalized_kwargs = copy.deepcopy(dict(model_kwargs))
        nested_family = str(
            normalized_kwargs.get("model_family", model_family) or ""
        ).strip()
        if nested_family != model_family:
            raise ValueError(
                f"{source}.model_kwargs.model_family conflicts with the "
                "top-level DVD model family."
            )
        nested_provenance = normalized_kwargs.get("model_provenance")
        normalized_nested = (
            None
            if nested_provenance is None
            else normalize_dvd_model_provenance(
                nested_provenance,
                source=(f"{source}.model_kwargs.model_provenance"),
            )
        )
        if normalized_nested is not None and normalized_nested != normalized_provenance:
            raise ValueError(
                f"{source}.model_kwargs.model_provenance conflicts with "
                "the top-level DVD declaration."
            )
        normalized_kwargs["model_family"] = model_family
        normalized_kwargs["model_provenance"] = copy.deepcopy(normalized_provenance)
        runtime["model_kwargs"] = normalized_kwargs
    return runtime


def _calibration_defaults(
    *,
    calibration_solver: str = "robust_affine",
    require_calibration_for_nonmetric: bool = True,
) -> dict[str, Any]:
    return {
        "if_calibrate_depth": True,
        "calibration_solver": calibration_solver,
        "multi_roi_strategy": "blend",
        "calib_region": "roi∧valid",
        "roi_dilate_px": 8,
        "points_radius_px": 3,
        "invalidate_mode": "nan",
        "require_calibration_for_nonmetric": (require_calibration_for_nonmetric),
    }


def normalize_depth_estimator_config(
    config: Mapping[str, Any],
    *,
    source: str = "depth_estimator.model_name",
) -> dict[str, Any]:
    """Normalize one canonical DVD or VDA estimator configuration."""

    output = copy.deepcopy(dict(config))
    unknown = sorted(set(output).difference(_DEPTH_CONFIG_ALLOWED_KEYS))
    if unknown:
        raise ValueError(
            f"Depth config contains unknown fields in {source}: {', '.join(unknown)}"
        )
    if str(output.get("model_name", "") or "").strip().lower() == "dvd":
        provenance = output.get("model_provenance")
        if isinstance(provenance, Mapping):
            output["model_provenance"] = normalize_dvd_model_provenance(
                provenance,
                source="DVD model_provenance",
            )
    return output


def resolve_depth_estimator_preset(preset: str) -> str:
    """Normalize an accepted preset or an unambiguous model-name alias."""

    requested = str(preset or DEFAULT_DEPTH_PRESET).strip().lower()
    if requested == "dvd":
        raise ValueError(
            "Plain 'dvd' names the backend but not a checkpoint identity. "
            "Select 'dvd_official', 'dvd_lora_shared', or "
            "'dvd_lora_specific', or provide an explicit DVD config."
        )
    canonical = DEPTH_PRESET_ALIASES.get(requested, requested)
    if canonical not in DEPTH_ESTIMATOR_PRESETS:
        available = ", ".join(DEPTH_ESTIMATOR_PRESETS)
        raise ValueError(
            f"Unsupported depth preset {preset!r}. "
            f"Available presets: {available}. "
            f"Supported backends: {list(SUPPORTED_DEPTH_BACKENDS)!r}. "
            "No fallback or remap is performed."
        )
    return canonical


def available_depth_estimator_presets(
    *,
    include_aliases: bool = False,
) -> tuple[str, ...]:
    """List deterministic current preset names and optional aliases."""

    values = set(DEPTH_ESTIMATOR_PRESETS)
    if include_aliases:
        values.update(DEPTH_PRESET_ALIASES)
    return tuple(sorted(values))


def _dvd_defaults(
    *,
    preset: str,
    checkpoint_root: str,
    model_config_path: str = "",
    resize_height: int = 480,
    resize_width: int = 640,
) -> dict[str, Any]:
    calibration = _calibration_defaults()
    calibration["background_nearfield_quantile"] = 0.5
    return {
        "model_name": "dvd",
        "model_family": DVD_MODEL_FAMILY,
        "model_provenance": {
            "kind": "project_finetuned",
            "preset": preset,
            "checkpoint_identity": checkpoint_root,
            "config_identity": model_config_path,
        },
        "dvd": {
            "ckpt_root": checkpoint_root,
            "model_config_path": model_config_path,
            "resize_height": resize_height,
            "resize_width": resize_width,
            "window_size": 81,
            "overlap": 9,
            "scale_only_alignment": False,
            "channel_reduce": "mean",
            "invert_disparity_to_depth": True,
            "min_disparity": 1e-4,
            "allow_download": False,
        },
        "calibration": calibration,
    }


def default_depth_estimator_config(
    preset: str = DEFAULT_DEPTH_PRESET,
) -> dict[str, Any]:
    """Return a path-portable copy of the current preset defaults.

    Relative local asset fields are resolved only against an explicit
    ``weights_root`` by :func:`resolve_depth_estimator_runtime`.
    """

    canonical = resolve_depth_estimator_preset(preset)
    if canonical in {"vda_metric", "vda_non_metric"}:
        metric = canonical == "vda_metric"
        config = {
            "model_name": "vda",
            "vda": {
                "encoder": "vitl",
                "metric": metric,
                "ckpt_root": "video_depth_anything",
            },
            "calibration": _calibration_defaults(
                require_calibration_for_nonmetric=not metric,
            ),
        }
    elif canonical in {"dvd_official", "dvd_lora_shared", "dvd_lora_specific"}:
        resource = importlib.resources.files("dream_exe.model_assets").joinpath(
            "configs", f"{canonical}.json"
        )
        config = json.loads(resource.read_text(encoding="utf-8"))
        if not isinstance(config, dict):  # pragma: no cover - packaged invariant
            raise ValueError(f"Packaged depth preset must be a JSON object: {canonical}")
        config = normalize_depth_estimator_config(
            config,
            source=f"package:dream_exe.model_assets/configs/{canonical}.json",
        )
    else:  # pragma: no cover - guarded by preset resolution
        raise AssertionError(canonical)
    return copy.deepcopy(config)


def load_depth_estimator_config(
    config_path: str | Path,
) -> dict[str, Any]:
    """Load one explicit JSON model configuration."""

    path = Path(config_path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(f"Depth config not found: {path.as_posix()}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"Depth config is not valid JSON: {path.as_posix()}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"Depth config must be a JSON object: {path.as_posix()}")
    return normalize_depth_estimator_config(
        payload,
        source=f"{path.as_posix()}:model_name",
    )


def preflight_depth_estimator_request(
    *,
    preset: str = DEFAULT_DEPTH_PRESET,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate model scope and config shape without resolving model assets."""

    if config is not None and config_path is not None:
        raise ValueError("Provide either depth config or config_path, not both.")
    requested_preset = str(preset or DEFAULT_DEPTH_PRESET).strip().lower()
    explicit_config = config is not None or config_path is not None
    if requested_preset in SUPPORTED_DEPTH_BACKENDS and explicit_config:
        canonical_preset = requested_preset
    else:
        canonical_preset = resolve_depth_estimator_preset(requested_preset)

    if config_path is not None:
        source_path = Path(config_path).expanduser().resolve()
        loaded_config = load_depth_estimator_config(source_path)
        config_source = source_path.as_posix()
    elif config is not None:
        loaded_config = normalize_depth_estimator_config(
            config,
            source="<mapping>:model_name",
        )
        config_source = "<mapping>"
    else:
        loaded_config = default_depth_estimator_config(canonical_preset)
        config_source = f"preset:{canonical_preset}"

    validate_depth_estimator_config(
        loaded_config,
        source=config_source,
    )
    if explicit_config:
        expected_model_name = (
            canonical_preset
            if canonical_preset in SUPPORTED_DEPTH_BACKENDS
            else str(default_depth_estimator_config(canonical_preset)["model_name"])
        )
        actual_model_name = str(loaded_config["model_name"])
        if actual_model_name != expected_model_name:
            raise ValueError(
                "Explicit depth config model family does not match the "
                f"requested preset: preset={preset!r} "
                f"expects {expected_model_name!r}, "
                f"config model_name={actual_model_name!r}. "
                "No fallback or remap is performed."
            )
    return {
        "config": loaded_config,
        "config_source": config_source,
        "preset": canonical_preset,
        "model_name": str(loaded_config["model_name"]),
    }


def validate_depth_estimator_config(
    config: Mapping[str, Any],
    *,
    source: str,
) -> None:
    """Validate accepted model and calibration-only configuration."""

    if not isinstance(config, Mapping):
        raise ValueError(f"Depth config must be a JSON object: {source}")
    config_dict = normalize_depth_estimator_config(
        config,
        source=f"{source}:model_name",
    )
    model_name = config_dict.get("model_name")
    if model_name not in DEPTH_MODEL_NAMES:
        raise _unsupported_depth_backend_error(
            model_name,
            source=source,
        )

    calibration = config_dict.get("calibration")
    if not isinstance(calibration, Mapping):
        raise ValueError(f"Depth config missing object 'calibration': {source}")
    for key in sorted(_CALIBRATION_REQUIRED_KEYS):
        if key not in calibration:
            raise ValueError(f"Depth config missing calibration.{key}: {source}")
    unknown_calibration_fields = sorted(
        set(calibration).difference(_CALIBRATION_ALLOWED_KEYS)
    )
    if unknown_calibration_fields:
        raise ValueError(
            "Depth calibration config contains unknown fields: "
            + ", ".join(unknown_calibration_fields)
        )
    _validate_calibration_numeric_types(
        calibration,
        source=source,
    )
    if calibration["calibration_solver"] not in (
        "robust_affine",
        "least_squares_scale",
        "least_squares_affine",
    ):
        raise ValueError(
            "Invalid calibration.calibration_solver in "
            f"{source}: {calibration['calibration_solver']}"
        )
    if calibration["multi_roi_strategy"] not in (
        "blend",
        "union",
    ):
        raise ValueError(
            "Invalid calibration.multi_roi_strategy in "
            f"{source}: {calibration['multi_roi_strategy']}"
        )
    if calibration["calib_region"] not in (
        "full",
        "valid",
        "roi",
        "roi∧valid",
        "tracks_points",
    ):
        raise ValueError(
            "Invalid calibration.calib_region in "
            f"{source}: {calibration['calib_region']}"
        )
    if calibration["invalidate_mode"] not in INVALIDATE_MODES:
        raise ValueError(
            "Invalid calibration.invalidate_mode in "
            f"{source}: {calibration['invalidate_mode']}"
        )

    model_config = config_dict.get(str(model_name))
    if not isinstance(model_config, Mapping):
        raise ValueError(f"Depth config missing object '{model_name}': {source}")
    if model_name == "vda":
        _reject_unknown_mapping_fields(
            model_config,
            allowed=_VDA_CONFIG_ALLOWED_KEYS,
            source="Depth config vda",
        )
        _validate_vda_options(model_config, source=source)
        if "metric" not in model_config:
            raise ValueError(f"Depth config missing vda.metric: {source}")
        if "ckpt_root" not in model_config:
            raise ValueError(f"Depth config missing vda.ckpt_root: {source}")
    elif model_name == "dvd":
        _reject_unknown_mapping_fields(
            model_config,
            allowed=_DVD_CONFIG_ALLOWED_KEYS,
            source="Depth config dvd",
        )
        _validate_dvd_options(model_config, source=source)
        provenance = config_dict.get("model_provenance")
        try:
            normalize_dvd_model_identity(
                config_dict.get("model_family"),
                provenance,
                source=f"DVD config in {source}",
            )
        except ValueError as error:
            if provenance is None:
                raise ValueError(
                    "DVD configs in dream-exe require model_provenance "
                    f"identifying an official or project fine-tuned checkpoint: {source}"
                ) from error
            raise
        selector = config_dict.get(
            "dvd_checkpoint_selector",
            None,
        )
        if selector is not None:
            if not isinstance(selector, Mapping):
                raise ValueError(
                    f"Depth config dvd_checkpoint_selector must be an object: {source}"
                )
            _reject_unknown_mapping_fields(
                selector,
                allowed=_DVD_SELECTOR_ALLOWED_KEYS,
                source="Depth config dvd_checkpoint_selector",
            )
            mode = str(selector.get("mode", "") or "").strip()
            if mode not in DVD_CHECKPOINT_SELECTOR_MODES:
                raise ValueError(
                    "Invalid dvd_checkpoint_selector.mode in "
                    f"{source}: {mode}. Expected one of "
                    f"{DVD_CHECKPOINT_SELECTOR_MODES}."
                )
            if not str(selector.get("root", "") or "").strip():
                raise ValueError(
                    f"Depth config missing dvd_checkpoint_selector.root: {source}"
                )
            missing_policy = str(
                selector.get("missing_policy", "error") or "error"
            ).strip()
            if missing_policy not in DVD_CHECKPOINT_SELECTOR_MISSING_POLICIES:
                raise ValueError(
                    "Invalid dvd_checkpoint_selector."
                    f"missing_policy in {source}: "
                    f"{missing_policy}. Expected one of "
                    f"{DVD_CHECKPOINT_SELECTOR_MISSING_POLICIES}."
                )


def _resolve_local_path(
    value: Any,
    *,
    weights_root: str | Path | None,
    field: str,
) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False).as_posix()
    if weights_root is None or not str(weights_root).strip():
        raise ValueError(
            f"{field} is relative ({text!r}); provide weights_root or an absolute path."
        )
    parts = list(path.parts)
    while parts and parts[0] in {".", ""}:
        parts = parts[1:]
    if parts and parts[0] == "checkpoints":
        parts = parts[1:]
    return (
        Path(weights_root)
        .expanduser()
        .resolve(strict=False)
        .joinpath(*parts)
        .resolve(strict=False)
        .as_posix()
    )


def _resolve_config_asset_paths(
    config: Mapping[str, Any],
    *,
    weights_root: str | Path | None,
) -> dict[str, Any]:
    output = copy.deepcopy(dict(config))
    if isinstance(output.get("vda"), dict):
        output["vda"]["ckpt_root"] = _resolve_local_path(
            output["vda"].get("ckpt_root", ""),
            weights_root=weights_root,
            field="vda.ckpt_root",
        )
    if isinstance(output.get("dvd"), dict):
        output["dvd"]["ckpt_root"] = _resolve_local_path(
            output["dvd"].get("ckpt_root", ""),
            weights_root=weights_root,
            field="dvd.ckpt_root",
        )
        model_config_path = str(
            output["dvd"].get("model_config_path", "") or ""
        ).strip()
        if model_config_path:
            output["dvd"]["model_config_path"] = _resolve_local_path(
                model_config_path,
                weights_root=weights_root,
                field="dvd.model_config_path",
            )
    if isinstance(
        output.get("dvd_checkpoint_selector"),
        dict,
    ):
        selector = output["dvd_checkpoint_selector"]
        selector["root"] = _resolve_local_path(
            selector.get("root", ""),
            weights_root=weights_root,
            field="dvd_checkpoint_selector.root",
        )
    return output


def _apply_dvd_checkpoint_selector(
    config: Mapping[str, Any],
    *,
    runtime_context: Mapping[str, Any] | None,
    source: str,
) -> dict[str, Any]:
    output = copy.deepcopy(dict(config))
    selector = output.get("dvd_checkpoint_selector", None)
    if not isinstance(selector, dict):
        return output
    mode = str(selector.get("mode", "") or "").strip()
    if mode != "single_task_by_uid":
        return output

    uid = str(dict(runtime_context or {}).get("uid", "") or "").strip()
    if not uid:
        raise ValueError(
            "DVD LoRA checkpoint selector requires "
            f"runtime_context['uid'] for {source}."
        )
    root = Path(str(selector.get("root", "") or "")).expanduser()
    root = root.resolve(strict=False)
    checkpoint_filename = str(
        selector.get(
            "checkpoint_filename",
            "model.safetensors",
        )
        or "model.safetensors"
    ).strip()
    model_config_filename = str(
        selector.get(
            "model_config_filename",
            "model_config.yaml",
        )
        or ""
    ).strip()
    checkpoint_directory = root / uid
    checkpoint_file = checkpoint_directory / checkpoint_filename
    dvd_config = dict(output.get("dvd", {}) or {})
    model_config_candidates: list[Path] = []
    if model_config_filename:
        model_config_candidates.append(checkpoint_directory / model_config_filename)
    configured_model_config = str(dvd_config.get("model_config_path", "") or "").strip()
    if configured_model_config:
        model_config_candidates.append(Path(configured_model_config).expanduser())
    model_config = next(
        (path.resolve() for path in model_config_candidates if path.exists()),
        None,
    )
    missing = [checkpoint_file.as_posix()] if not checkpoint_file.exists() else []
    if model_config is None:
        missing.append(
            "one of: "
            + ", ".join(
                path.resolve(strict=False).as_posix()
                for path in model_config_candidates
            )
        )
    if missing:
        raise FileNotFoundError(
            "DVD LoRA checkpoint for uid "
            f"{uid!r} is incomplete under "
            f"{checkpoint_directory.as_posix()}. Missing: " + ", ".join(missing)
        )
    dvd_config["ckpt_root"] = checkpoint_directory.as_posix()
    dvd_config["model_config_path"] = model_config.as_posix()
    output["dvd"] = dvd_config
    provenance = dict(output.get("model_provenance", {}) or {})
    provenance["uid"] = uid
    provenance["checkpoint_identity"] = checkpoint_file.resolve().as_posix()
    provenance["config_identity"] = model_config.as_posix()
    output["model_provenance"] = provenance
    output.pop("dvd_checkpoint_selector", None)
    return output


def resolve_depth_estimator_runtime(
    *,
    device: str,
    preset: str = DEFAULT_DEPTH_PRESET,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    fp32: bool = False,
    weights_root: str | Path | None = None,
    runtime_context: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve an explicit preset/config into current model kwargs."""

    _require_exact_bool(
        fp32,
        field="fp32",
        source="depth estimator request",
    )
    preflight = preflight_depth_estimator_request(
        preset=preset,
        config=config,
        config_path=config_path,
    )
    canonical_preset = str(preflight["preset"])
    loaded_config = dict(preflight["config"])
    config_source = str(preflight["config_source"])
    resolved_config = _resolve_config_asset_paths(
        loaded_config,
        weights_root=weights_root,
    )
    resolved_config = _apply_dvd_checkpoint_selector(
        resolved_config,
        runtime_context=runtime_context,
        source=config_source,
    )
    model_name = str(resolved_config["model_name"])
    calibration = dict(resolved_config["calibration"])
    runtime_device = str(device)

    if model_name == "vda":
        model_kwargs = {
            "encoder": str(resolved_config["vda"]["encoder"]),
            "metric": bool(resolved_config["vda"]["metric"]),
            "device": runtime_device,
            "ckpt_root": str(resolved_config["vda"]["ckpt_root"]),
        }
    elif model_name == "dvd":
        model_kwargs = dict(resolved_config.get(model_name, {}) or {})
        model_kwargs["model_family"] = str(resolved_config["model_family"])
        model_kwargs["model_provenance"] = copy.deepcopy(
            resolved_config["model_provenance"]
        )
        model_kwargs["device"] = runtime_device
    else:  # pragma: no cover - validation guards this path
        raise ValueError(f"Unsupported depth model: {model_name}")

    return {
        "config": resolved_config,
        "config_source": config_source,
        "preset": canonical_preset,
        "model_name": model_name,
        "model_family": (str(resolved_config.get("model_family", "") or "")),
        "model_provenance": copy.deepcopy(resolved_config.get("model_provenance", {})),
        "model_kwargs": model_kwargs,
        "if_calibrate_depth": bool(calibration["if_calibrate_depth"]),
        "calibration_solver": str(calibration["calibration_solver"]),
        "multi_roi_strategy": str(calibration["multi_roi_strategy"]),
        "calib_region": str(calibration["calib_region"]),
        "roi_dilate_px": int(calibration["roi_dilate_px"]),
        "points_radius_px": int(calibration["points_radius_px"]),
        "invalidate_mode": str(calibration["invalidate_mode"]),
        "require_calibration_for_nonmetric": bool(
            calibration["require_calibration_for_nonmetric"]
        ),
    }


def _require_directory(
    path_value: Any,
    *,
    label: str,
) -> Path:
    text = str(path_value or "").strip()
    if not text:
        raise FileNotFoundError(f"{label} is not configured.")
    path = Path(text).expanduser().resolve(strict=False)
    if not path.is_dir():
        raise FileNotFoundError(f"{label} not found: {path.as_posix()}")
    return path


def validate_depth_estimator_assets(
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate explicit local assets immediately before backend creation."""

    runtime = validate_depth_runtime_identity(
        runtime_config,
        source="depth runtime asset validation",
    )
    model_name = str(runtime.get("model_name", "") or "")
    kwargs = dict(runtime.get("model_kwargs", {}) or {})
    if model_name == "vda":
        checkpoint_root = _require_directory(
            kwargs.get("ckpt_root", ""),
            label="VDA checkpoint directory",
        )
        encoder = str(kwargs.get("encoder", "vitl"))
        prefix = (
            "metric_video_depth_anything"
            if bool(kwargs.get("metric", False))
            else "video_depth_anything"
        )
        checkpoint = checkpoint_root / f"{prefix}_{encoder}.pth"
        if not checkpoint.is_file():
            raise FileNotFoundError(
                f"VDA checkpoint file not found: {checkpoint.as_posix()}"
            )
    elif model_name == "dvd":
        checkpoint_root = _require_directory(
            kwargs.get("ckpt_root", ""),
            label="DVD checkpoint directory",
        )
        checkpoint = checkpoint_root / "model.safetensors"
        if not checkpoint.is_file():
            candidates = sorted(
                path for path in checkpoint_root.glob("*.safetensors") if path.is_file()
            )
            if len(candidates) == 1:
                checkpoint = candidates[0]
            else:
                raise FileNotFoundError(
                    "DVD checkpoint file not found under "
                    f"{checkpoint_root.as_posix()}. Expected "
                    "model.safetensors or exactly one "
                    "*.safetensors file."
                )
        configured_model_config = str(kwargs.get("model_config_path", "") or "").strip()
        model_config = (
            Path(configured_model_config).expanduser().resolve(strict=False)
            if configured_model_config
            else checkpoint_root / "model_config.yaml"
        )
        if not model_config.is_file():
            raise FileNotFoundError(
                f"DVD model_config.yaml not found. Looked at {model_config.as_posix()}."
            )
        kwargs["ckpt_root"] = checkpoint_root.as_posix()
        kwargs["model_config_path"] = model_config.as_posix()
        provenance = runtime["model_provenance"]
        if DVD_ASSET_ATTESTATION_FIELD in provenance:
            provenance = attest_dvd_asset_files(
                provenance,
                checkpoint_file=checkpoint,
                model_config_file=model_config,
            )
            runtime["model_provenance"] = provenance
            kwargs["model_provenance"] = copy.deepcopy(provenance)
            config = runtime.get("config")
            if isinstance(config, Mapping):
                normalized_config = copy.deepcopy(dict(config))
                normalized_config["model_provenance"] = copy.deepcopy(provenance)
                runtime["config"] = normalized_config
    else:
        raise ValueError(f"Unsupported depth model: {model_name}")
    runtime["model_kwargs"] = kwargs
    return runtime


def _resolve_backend_registry_entry(
    registry: BackendRegistry,
    *,
    model_name: str,
) -> Any:
    if model_name not in registry:
        available = ", ".join(sorted(str(key) for key in registry))
        raise ValueError(
            f"Depth backend '{model_name}' is not registered. "
            f"Available backends: {available or '<none>'}."
        )
    entry = registry[model_name]
    if isinstance(entry, str):
        module_name, separator, attribute = entry.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError(
                f"Depth backend import spec must be 'module:attribute', got {entry!r}."
            )
        try:
            module = importlib.import_module(module_name)
        except ModuleNotFoundError as error:
            missing_name = str(error.name or module_name)
            raise RuntimeError(
                f"Depth backend '{model_name}' is unavailable "
                f"because dependency '{missing_name}' is missing."
            ) from error
        try:
            return getattr(module, attribute)
        except AttributeError as error:
            raise RuntimeError(
                f"Depth backend '{model_name}' import spec "
                f"{entry!r} has no attribute {attribute!r}."
            ) from error
    return entry


def _filtered_factory_kwargs(
    factory: Callable[..., Any],
    kwargs: Mapping[str, Any],
) -> dict[str, Any]:
    signature = inspect.signature(factory)
    if any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    ):
        return dict(kwargs)
    allowed = {
        name
        for name, parameter in signature.parameters.items()
        if parameter.kind
        in {
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        }
    }
    return {key: value for key, value in kwargs.items() if key in allowed}


def _construct_backend(
    entry: Any,
    *,
    model_name: str,
    model_kwargs: Mapping[str, Any],
) -> Any:
    if hasattr(entry, "infer") and not inspect.isclass(entry):
        return entry
    if not callable(entry):
        raise TypeError(
            f"Depth backend '{model_name}' registry entry must "
            "be a factory, import spec, or object with infer()."
        )
    kwargs = dict(model_kwargs)
    if model_name == "vda" and "ckpt_path" in kwargs and "ckpt_root" not in kwargs:
        checkpoint_path = kwargs.pop("ckpt_path")
        checkpoint_root = checkpoint_path
        if isinstance(checkpoint_path, str) and os.path.splitext(checkpoint_path)[1]:
            checkpoint_root = os.path.dirname(checkpoint_path)
        kwargs["ckpt_root"] = checkpoint_root
    try:
        return entry(**kwargs)
    except ModuleNotFoundError as error:
        missing_name = str(error.name or "<unknown>")
        raise RuntimeError(
            f"Depth backend '{model_name}' is unavailable because "
            f"dependency '{missing_name}' is missing."
        ) from error
    except TypeError:
        filtered = _filtered_factory_kwargs(entry, kwargs)
        try:
            return entry(**filtered)
        except ModuleNotFoundError as error:
            missing_name = str(error.name or "<unknown>")
            raise RuntimeError(
                f"Depth backend '{model_name}' is unavailable "
                f"because dependency '{missing_name}' is missing."
            ) from error


def _prediction_field(
    prediction: Any,
    name: str,
    default: Any,
) -> Any:
    if isinstance(prediction, Mapping):
        return prediction.get(name, default)
    return getattr(prediction, name, default)


def _normalize_backend_prediction(
    prediction: Any,
    *,
    runtime_config: Mapping[str, Any],
    target_fps: float,
) -> tuple[
    np.ndarray,
    float,
    dict[str, Any],
    dict[str, Any],
]:
    raw_depths = _prediction_field(
        prediction,
        "depths",
        None,
    )
    if raw_depths is None:
        raise TypeError("Depth backend prediction is missing 'depths'.")
    if isinstance(raw_depths, np.ndarray):
        depth_stack = np.asarray(raw_depths, dtype=np.float32)
    else:
        depth_list = [np.asarray(depth, dtype=np.float32) for depth in list(raw_depths)]
        if not depth_list:
            raise RuntimeError("[Depth] Model returned empty depths.")
        depth_stack = np.stack(depth_list, axis=0).astype(np.float32)
    if depth_stack.ndim != 3:
        raise ValueError(
            f"Depth backend output must resolve to [T,H,W], got {depth_stack.shape}."
        )
    if depth_stack.shape[0] <= 0:
        raise RuntimeError("[Depth] Model returned empty depths.")

    prediction_fps = float(
        _prediction_field(
            prediction,
            "fps",
            target_fps,
        )
    )
    model_metadata = _prediction_field(
        prediction,
        "meta",
        {},
    )
    if not isinstance(model_metadata, Mapping):
        raise TypeError("Depth backend prediction meta must be a mapping.")
    depth_space = str(
        _prediction_field(
            prediction,
            "depth_space",
            "unknown",
        )
        or "unknown"
    )
    fps_source = str(
        _prediction_field(
            prediction,
            "fps_source",
            "unknown",
        )
        or "unknown"
    )
    valid_masks = _prediction_field(
        prediction,
        "valid_masks",
        None,
    )
    runtime = copy.deepcopy(dict(runtime_config))
    info = {
        "model": dict(model_metadata),
        "depth_config": copy.deepcopy(runtime.get("config", {})),
        "depth_config_source": str(runtime.get("config_source", "") or ""),
        "depth_space": depth_space,
        "fps_source": fps_source,
        "calibration": None,
    }
    aux = {
        "valid_masks": valid_masks,
        "init_ref_depth": None,
        "runtime_cfg": runtime,
        "fps": prediction_fps,
    }
    return depth_stack, prediction_fps, info, aux


def make_depth_estimator(
    *,
    runtime_config: Mapping[str, Any],
    backend_registry: BackendRegistry,
    fp32: bool = False,
    input_size: int = 512,
    validate_assets: bool = True,
    seed: int = 42,
    seed_hook: Callable[[int], Any] | None = None,
) -> Callable[..., Any]:
    """Create a closure that lazily constructs and reuses one backend."""

    _require_exact_bool(
        fp32,
        field="fp32",
        source="depth estimator construction",
    )
    _require_exact_bool(
        validate_assets,
        field="validate_assets",
        source="depth estimator construction",
    )
    parsed_input_size = _require_integer(
        input_size,
        field="input_size",
        source="depth estimator construction",
    )
    if parsed_input_size <= 0:
        raise ValueError("depth input_size must be > 0.")
    parsed_seed = _require_integer(
        seed,
        field="seed",
        source="depth estimator construction",
    )
    resolved_runtime = validate_depth_runtime_identity(
        runtime_config,
        source="depth estimator runtime",
    )
    model_name = str(resolved_runtime.get("model_name", "") or "")
    if model_name not in SUPPORTED_DEPTH_BACKENDS:
        raise _unsupported_depth_backend_error(
            model_name,
            source="depth estimator construction",
        )
    backend: Any | None = None
    enforce_attestation = (
        model_name == "dvd"
        and DVD_ASSET_ATTESTATION_FIELD in resolved_runtime.get("model_provenance", {})
    )

    def estimate(
        *,
        video_frames: Any,
        target_fps: float,
        intrinsics: np.ndarray | None = None,
        extrinsics: np.ndarray | None = None,
    ) -> tuple[
        np.ndarray,
        float,
        dict[str, Any],
        dict[str, Any],
    ]:
        nonlocal backend, resolved_runtime
        model_name = str(resolved_runtime.get("model_name", "") or "")
        if backend is None:
            if validate_assets or enforce_attestation:
                resolved_runtime = validate_depth_estimator_assets(resolved_runtime)
        np.random.seed(parsed_seed)
        _seed_torch_runtime(parsed_seed)
        if seed_hook is not None:
            seed_hook(parsed_seed)
        if backend is None:
            entry = _resolve_backend_registry_entry(
                backend_registry,
                model_name=model_name,
            )
            backend = _construct_backend(
                entry,
                model_name=model_name,
                model_kwargs=resolved_runtime.get(
                    "model_kwargs",
                    {},
                ),
            )
        infer = getattr(backend, "infer", None)
        if infer is None and callable(backend):
            infer = backend
        if not callable(infer):
            raise TypeError(
                f"Depth backend '{model_name}' has no callable infer method."
            )
        try:
            prediction = infer(
                video_frames,
                target_fps,
                fp32=fp32,
                input_size=parsed_input_size,
                intrinsics=intrinsics,
                extrinsics=extrinsics,
            )
        except ModuleNotFoundError as error:
            missing_name = str(error.name or "<unknown>")
            raise RuntimeError(
                f"Depth backend '{model_name}' is unavailable "
                f"because dependency '{missing_name}' is missing."
            ) from error
        return _normalize_backend_prediction(
            prediction,
            runtime_config=resolved_runtime,
            target_fps=target_fps,
        )

    return estimate


def prepare_depth_estimator(
    *,
    backend_registry: BackendRegistry,
    device: str,
    preset: str = DEFAULT_DEPTH_PRESET,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    fp32: bool = False,
    input_size: int = 512,
    weights_root: str | Path | None = None,
    runtime_context: Mapping[str, Any] | None = None,
    sample_uid: str = "",
    validate_assets: bool = True,
    seed: int = 42,
    seed_hook: Callable[[int], Any] | None = None,
) -> tuple[Callable[..., Any], dict[str, Any]]:
    """Resolve configuration and return a lazy estimator plus its identity."""

    context = dict(runtime_context or {})
    if str(sample_uid or "").strip():
        context["uid"] = str(sample_uid or "").strip()
    runtime = resolve_depth_estimator_runtime(
        device=device,
        preset=preset,
        config=config,
        config_path=config_path,
        fp32=fp32,
        weights_root=weights_root,
        runtime_context=context,
    )
    if runtime["model_name"] == "dvd" and DVD_ASSET_ATTESTATION_FIELD in runtime.get(
        "model_provenance", {}
    ):
        runtime = validate_depth_estimator_assets(runtime)
    estimator = make_depth_estimator(
        runtime_config=runtime,
        backend_registry=backend_registry,
        fp32=fp32,
        input_size=input_size,
        validate_assets=validate_assets,
        seed=seed,
        seed_hook=seed_hook,
    )
    return estimator, runtime


__all__ = [
    "BackendFactory",
    "BackendRegistry",
    "DEFAULT_DEPTH_PRESET",
    "DEPTH_ESTIMATOR_PRESETS",
    "DEPTH_MODEL_NAMES",
    "DEPTH_PRESET_ALIASES",
    "DVD_ASSET_ATTESTATION_FIELD",
    "DVD_MODEL_FAMILIES",
    "DVD_MODEL_FAMILY",
    "DVD_OFFICIAL_MODEL_FAMILY",
    "DVD_PROVENANCE_VALIDATION",
    "SUPPORTED_DEPTH_BACKENDS",
    "available_depth_estimator_presets",
    "default_depth_estimator_config",
    "load_depth_estimator_config",
    "make_depth_estimator",
    "normalize_depth_estimator_config",
    "preflight_depth_estimator_request",
    "prepare_depth_estimator",
    "resolve_depth_estimator_preset",
    "resolve_depth_estimator_runtime",
    "validate_depth_estimator_assets",
    "validate_depth_estimator_config",
    "validate_depth_runtime_identity",
]
