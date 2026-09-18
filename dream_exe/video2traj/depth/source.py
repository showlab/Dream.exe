"""Provider-neutral source composition for depth inputs.

This module deliberately owns no model backend.  Callers provide estimation
and calibration callbacks, explicit arrays, and explicit paths.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from .cache import (
    DEPTH_CACHE_METADATA_FILENAME,
    DEPTH_CACHE_SCHEMA,
    DEPTH_COMPAT_METADATA_FILENAME,
    LEGACY_METADATA_POLICY_REJECT,
    DepthCacheValidationError,
    load_depth_cache_metadata,
    normalize_depth_calibration_metadata,
    normalize_depth_model_provenance,
    read_validated_depth_cache,
    validate_canonical_depth_cache_metadata,
)

_ROLLOUT_DEPTH_SOURCE = "rollout_gt_depth"
_CACHE_DEPTH_SOURCE = "estimated_depth_cache"
_MODEL_DEPTH_SOURCE = "estimated_model"
_CANONICAL_CACHE_RUNTIME_INFO_FIELDS = frozenset(
    {
        "cache_signature",
        "calibration",
        "depth_config_source",
        "depth_model",
        "depth_space",
        "fps_source",
        "model_provenance",
        "source",
    }
)
_HISTORICAL_CACHE_RUNTIME_SCALAR_FIELDS = frozenset(
    {
        "cache_signature",
        "depth_config_source",
        "depth_model",
        "depth_space",
        "fps",
        "fps_source",
        "source",
    }
)

_ESTIMATOR_RESERVED_INPUTS = frozenset(
    {
        "video_frames",
        "target_fps",
    }
)
_CALIBRATION_RESERVED_INPUTS = frozenset(
    {
        "depths",
        "depth_source",
        "depth_model",
        "depth_base_cfg",
        "depth_info",
        "init_ref_depth",
        "valid_masks",
    }
)
_ESTIMATOR_INFO_FIELDS = frozenset(
    {
        "calibration",
        "depth_config",
        "depth_config_source",
        "depth_space",
        "fps_source",
        "model",
    }
)
_ESTIMATOR_AUX_FIELDS = frozenset(
    {
        "fps",
        "init_ref_depth",
        "runtime_cfg",
        "valid_masks",
    }
)
_VDA_MODEL_DIAGNOSTIC_FIELDS = frozenset(
    {
        "encoder",
        "input_color_order",
        "input_size",
        "metric",
        "model",
    }
)
_DVD_MODEL_DIAGNOSTIC_FIELDS = frozenset(
    {
        "allow_download",
        "channel_reduce",
        "checkpoint_file",
        "checkpoints_root",
        "ckpt_root",
        "fine_tuned",
        "input_color_order",
        "invert_disparity_to_depth",
        "min_disparity",
        "model",
        "model_config_path",
        "model_family",
        "model_provenance",
        "overlap",
        "raw_output_channels",
        "raw_output_representation",
        "resize_height",
        "resize_width",
        "scale_only_alignment",
        "window_size",
        "wan_asset_mode",
    }
)
_EXTERNAL_MODEL_DIAGNOSTIC_FIELDS = frozenset(
    {
        "backend",
        "backend_id",
        "contract_version",
        "model",
        "provider_kind",
    }
)
_CALIBRATED_INFO_FIELDS = frozenset(
    {
        "cache_rejected_reason",
        "cache_signature",
        "calibration",
        "depth_config_source",
        "depth_model",
        "depth_space",
        "estimator_fps",
        "fps_source",
        "model",
    }
)


def _json_ready(value: Any) -> Any:
    """Convert configuration values to the historical stable JSON shape."""

    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _mapping_copy(
    value: Mapping[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def normalize_depth_runtime_cache_identity(
    runtime_config: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Return the provider-neutral portion of a depth runtime identity.

    File identities are caller data.  They are serialized as supplied and
    are never discovered or statted here.
    """

    runtime = _mapping_copy(
        runtime_config,
        label="depth runtime config",
    )
    return {
        "config_source": runtime.get("config_source", ""),
        "preset": runtime.get("preset", ""),
        "model_name": runtime.get("model_name", ""),
        "model_kwargs": _json_ready(runtime.get("model_kwargs", {}) or {}),
        "config": _json_ready(runtime.get("config", {}) or {}),
        "files": _json_ready(runtime.get("files", {}) or {}),
    }


def build_depth_cache_signature(
    *,
    depth_model: str,
    depth_config_request: str,
    depth_base_cfg: Mapping[str, Any] | None,
    selected_video: str,
    depth_runtime_config: Mapping[str, Any] | None = None,
) -> str:
    """Build the stable current-compatible SHA-1 cache signature."""

    payload = {
        "depth_model": str(depth_model or ""),
        "depth_config_request": str(depth_config_request or ""),
        "depth_base_cfg": _json_ready(depth_base_cfg or {}),
        "selected_video": str(selected_video or ""),
        "depth_runtime": normalize_depth_runtime_cache_identity(depth_runtime_config),
    }
    serialized = json.dumps(
        _json_ready(payload),
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha1(serialized.encode("utf-8")).hexdigest()


def evaluate_depth_cache_compatibility(
    *,
    meta_payload: Mapping[str, Any] | None,
    expected_cache_signature: str,
    selected_video: str,
    expected_input_identity: Mapping[str, Any] | None = None,
    trust_cache_on_signature_mismatch: bool = False,
) -> tuple[bool, str]:
    """Require exact cache identity unless the caller explicitly trusts it."""

    metadata = _mapping_copy(
        meta_payload,
        label="depth cache metadata",
    )
    if not metadata:
        if expected_input_identity is None and str(selected_video or "") != "gen":
            return True, ""
        return False, "missing cache meta"

    cached_identity = metadata.get("input_identity", None)
    if str(metadata.get("schema", "") or "") == DEPTH_CACHE_SCHEMA:
        if cached_identity is None:
            return False, "missing cache decoded-input identity"
        if expected_input_identity is None:
            return False, "missing expected decoded-input identity"
        if _json_ready(cached_identity) != _json_ready(dict(expected_input_identity)):
            return False, "cache decoded-input identity mismatch"
    if str(selected_video or "") != "gen":
        return True, ""
    cached_signature = str(metadata.get("cache_signature", "") or "")
    if cached_signature == str(expected_cache_signature or ""):
        return True, ""
    if trust_cache_on_signature_mismatch:
        return True, ""
    return False, "cache signature mismatch"


def _default_cached_stage_stats() -> dict[str, dict[str, Any]]:
    return {
        "base": {
            "enabled": False,
            "applied": False,
            "reason": "loaded from depth cache",
            "cached": True,
        }
    }


def stage_stats_from_cached_calibration(
    calibration: Mapping[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Recover canonical calibration stage status from cache metadata."""

    if not isinstance(calibration, Mapping):
        return _default_cached_stage_stats()
    normalized = normalize_depth_calibration_metadata(
        calibration,
        label="cached depth calibration",
    )
    base = normalized.get("base")
    if not isinstance(base, Mapping) or not base:
        return _default_cached_stage_stats()
    result = dict(base)
    result["cached"] = True
    return {"base": result}


def load_depth_cache_meta(
    cache_path: str | os.PathLike[str],
    *,
    meta_path: str | os.PathLike[str] | None = None,
    legacy_metadata_policy: str = LEGACY_METADATA_POLICY_REJECT,
) -> dict[str, Any]:
    """Load safe JSON or explicitly trusted legacy metadata."""

    try:
        if meta_path is None:
            resolved_cache = Path(cache_path).expanduser().resolve(strict=False)
            safe_meta = resolved_cache.with_name(DEPTH_CACHE_METADATA_FILENAME)
            compatibility_meta = resolved_cache.with_name(
                DEPTH_COMPAT_METADATA_FILENAME
            )
            resolved_meta = (
                safe_meta
                if safe_meta.exists() or safe_meta.is_symlink()
                else compatibility_meta
            )
        else:
            resolved_meta = Path(meta_path).expanduser().resolve(strict=False)
        if not resolved_meta.is_file():
            return {}
        payload, _ = load_depth_cache_metadata(
            resolved_meta,
            legacy_metadata_policy=legacy_metadata_policy,
        )
        return payload
    except Exception:
        return {}


def _depth_stack(
    value: Any,
    *,
    label: str = "depths",
) -> np.ndarray:
    try:
        array = np.asarray(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must resolve to [T,H,W]") from exc
    if array.ndim != 3:
        raise ValueError(f"{label} must resolve to [T,H,W], got {array.shape}")
    if any(int(size) <= 0 for size in array.shape):
        raise ValueError(f"{label} must be a non-empty [T,H,W] stack")
    try:
        return np.asarray(array, dtype=np.float32)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must contain numeric depth values") from exc


def _frame_geometry(frames: Any) -> tuple[int, int, int]:
    try:
        frame_count = len(frames)
    except TypeError as exc:
        raise ValueError("frames must be a non-empty sequence") from exc
    if frame_count <= 0:
        raise ValueError("frames must be a non-empty sequence")
    first = np.asarray(frames[0])
    if first.ndim < 2 or first.shape[0] <= 0 or first.shape[1] <= 0:
        raise ValueError("frames must contain image-like arrays")
    height, width = int(first.shape[0]), int(first.shape[1])
    for frame in frames[1:]:
        shape = np.asarray(frame).shape
        if len(shape) < 2 or tuple(shape[:2]) != (height, width):
            raise ValueError("all frames must have the same spatial shape")
    return frame_count, height, width


def align_depth_stack_to_frames(
    depths: Any,
    *,
    frames: Any,
    depth_source: str,
    video_target_fps: float = -1,
    video_process_length: int = -1,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Align a depth stack to caller-provided frames with strict geometry."""

    array = _depth_stack(depths)
    frame_count, frame_height, frame_width = _frame_geometry(frames)
    input_count = int(array.shape[0])
    mode = "already_aligned"
    stride: int | None = None

    if input_count != frame_count:
        if str(depth_source) == _ROLLOUT_DEPTH_SOURCE and float(video_target_fps) > 0:
            nominal_stride = max(
                1,
                int(round(30.0 / float(video_target_fps))),
            )
            sampled = array[::nominal_stride]
            if int(sampled.shape[0]) < frame_count:
                nominal_stride = max(
                    1,
                    input_count // frame_count,
                )
                sampled = array[::nominal_stride]
            stride = nominal_stride
            array = sampled[:frame_count]
            mode = "rollout_target_fps_stride"
        elif (
            str(depth_source) == _ROLLOUT_DEPTH_SOURCE and int(video_process_length) > 0
        ):
            array = array[: int(video_process_length)]
            mode = "rollout_process_length_trim"

    if int(array.shape[0]) != frame_count:
        raise AssertionError(
            "depth/frame length mismatch: "
            f"depths={int(array.shape[0])} frames={frame_count}"
        )

    depth_height = int(array.shape[1])
    depth_width = int(array.shape[2])
    if (depth_height, depth_width) != (
        frame_height,
        frame_width,
    ):
        raise AssertionError(
            "Depth size "
            f"({depth_width}, {depth_height}) != frame size "
            f"({frame_width}, {frame_height})"
        )

    return array, {
        "input_num_frames": input_count,
        "num_frames": frame_count,
        "frame_shape": [frame_height, frame_width],
        "depth_shape": [depth_height, depth_width],
        "temporal_mode": mode,
        "stride": stride,
    }


def _reserved_kwargs(
    value: Mapping[str, Any] | None,
    *,
    label: str,
    reserved: frozenset[str],
) -> dict[str, Any]:
    options = _mapping_copy(value, label=f"{label} kwargs")
    overlap = sorted(reserved.intersection(options))
    if overlap:
        raise ValueError(
            f"{label} kwargs cannot replace reserved inputs: " + ", ".join(overlap)
        )
    return options


def _reject_unknown_diagnostic_fields(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    label: str,
) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"{label} contains unknown fields: " + ", ".join(unknown))


def _canonical_model_diagnostics(value: Any) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("depth estimator info.model must be a mapping")
    diagnostics = copy.deepcopy(dict(value))
    model = str(diagnostics.get("model", "") or "").strip().lower()
    if model == "vda":
        allowed = _VDA_MODEL_DIAGNOSTIC_FIELDS
    elif model == "dvd":
        allowed = _DVD_MODEL_DIAGNOSTIC_FIELDS
    else:
        allowed = _EXTERNAL_MODEL_DIAGNOSTIC_FIELDS
    _reject_unknown_diagnostic_fields(
        diagnostics,
        allowed=allowed,
        label="depth estimator info.model",
    )
    if model == "dvd" and "model_provenance" in diagnostics:
        diagnostics["model_provenance"] = normalize_depth_model_provenance(
            "dvd",
            diagnostics["model_provenance"],
            label="depth estimator info.model.model_provenance",
        )
    return diagnostics


def _canonical_estimator_info(value: Mapping[str, Any]) -> dict[str, Any]:
    info = copy.deepcopy(dict(value))
    _reject_unknown_diagnostic_fields(
        info,
        allowed=_ESTIMATOR_INFO_FIELDS,
        label="depth estimator info",
    )
    if "depth_config" in info and not isinstance(
        info["depth_config"],
        Mapping,
    ):
        raise TypeError("depth estimator info.depth_config must be a mapping")
    # The resolved config already has its own validated runtime contract. It is
    # intentionally not duplicated into pipeline artifacts.
    info.pop("depth_config", None)
    if "model" in info:
        info["model"] = _canonical_model_diagnostics(info["model"])
    calibration = info.get("calibration")
    if calibration is None:
        info["calibration"] = None
    else:
        info["calibration"] = normalize_depth_calibration_metadata(
            calibration,
            label="depth estimator info.calibration",
        )
    return info


def _canonical_estimator_aux(value: Mapping[str, Any]) -> dict[str, Any]:
    aux = dict(value)
    _reject_unknown_diagnostic_fields(
        aux,
        allowed=_ESTIMATOR_AUX_FIELDS,
        label="depth estimator aux",
    )
    if "runtime_cfg" in aux and not isinstance(
        aux["runtime_cfg"],
        Mapping,
    ):
        raise TypeError("depth estimator aux.runtime_cfg must be a mapping")
    return {
        field: aux[field] for field in ("init_ref_depth", "valid_masks") if field in aux
    }


def _canonical_calibrated_info(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    info = copy.deepcopy(dict(value))
    _reject_unknown_diagnostic_fields(
        info,
        allowed=_CALIBRATED_INFO_FIELDS,
        label="depth calibration info",
    )
    if "model" in info:
        info["model"] = _canonical_model_diagnostics(info["model"])
    calibration = info.get("calibration")
    if calibration is None:
        info["calibration"] = None
    else:
        info["calibration"] = normalize_depth_calibration_metadata(
            calibration,
            label="depth calibration info.calibration",
        )
    return info


def _estimator_result(
    result: Any,
    *,
    target_fps: float,
) -> tuple[np.ndarray, float, dict[str, Any], dict[str, Any]]:
    if not isinstance(result, tuple):
        return _depth_stack(result), float(target_fps), {}, {}
    if not 1 <= len(result) <= 4:
        raise TypeError(
            "depth estimator must return depths or (depths, fps, info[, aux])"
        )

    raw_depths = result[0]
    fps = float(result[1]) if len(result) >= 2 else float(target_fps)
    raw_info = result[2] if len(result) >= 3 else {}
    raw_aux = result[3] if len(result) >= 4 else {}
    if raw_info is None:
        raw_info = {}
    if not isinstance(raw_info, Mapping):
        raise TypeError("depth estimator info must be a mapping")
    if raw_aux is None:
        raw_aux = {}
    if not isinstance(raw_aux, Mapping):
        raise TypeError("depth estimator aux must be a mapping")
    return (
        _depth_stack(raw_depths),
        fps,
        _canonical_estimator_info(raw_info),
        _canonical_estimator_aux(raw_aux),
    )


def _run_calibration(
    calibration: Callable[..., Any] | None,
    *,
    depths: np.ndarray,
    depth_source: str,
    depth_model: str,
    depth_base_cfg: Mapping[str, Any],
    depth_info: Mapping[str, Any],
    init_ref_depth: Any,
    valid_masks: Any,
    calibration_kwargs: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    if calibration is None:
        return depths, dict(depth_info), {}
    result = calibration(
        depths=depths,
        depth_source=depth_source,
        depth_model=depth_model,
        depth_base_cfg=dict(depth_base_cfg),
        depth_info=dict(depth_info),
        init_ref_depth=init_ref_depth,
        valid_masks=valid_masks,
        **dict(calibration_kwargs),
    )
    if not isinstance(result, tuple) or len(result) != 3:
        raise TypeError(
            "depth calibration must return (depths, depth_info, stage_stats)"
        )
    calibrated, raw_info, raw_stats = result
    if not isinstance(raw_info, Mapping):
        raise TypeError("depth calibration info must be a mapping")
    if not isinstance(raw_stats, Mapping):
        raise TypeError("depth calibration stage stats must be a mapping")
    calibrated_info = _canonical_calibrated_info(raw_info)
    stage_stats = normalize_depth_calibration_metadata(
        raw_stats,
        label="depth calibration stage_stats",
    )
    return (
        _depth_stack(calibrated),
        calibrated_info,
        stage_stats,
    )


def _cache_metadata(
    *,
    cache_path: str | os.PathLike[str] | None,
    explicit_metadata: Mapping[str, Any] | None,
    meta_path: str | os.PathLike[str] | None,
    legacy_metadata_policy: str,
) -> dict[str, Any]:
    if explicit_metadata is not None:
        return _mapping_copy(
            explicit_metadata,
            label="estimated depth cache metadata",
        )
    if cache_path is None:
        return {}
    return load_depth_cache_meta(
        cache_path,
        meta_path=meta_path,
        legacy_metadata_policy=legacy_metadata_policy,
    )


def _cache_depths(
    *,
    explicit_depths: Any,
) -> np.ndarray | None:
    if explicit_depths is not None:
        return _depth_stack(
            explicit_depths,
            label="estimated depth cache",
        )
    return None


def _lexical_absolute_path(
    value: str | os.PathLike[str],
) -> Path:
    """Make a path absolute without resolving symlinks."""

    return Path(os.path.abspath(Path(value).expanduser()))


def _validated_path_cache(
    *,
    cache_path: str | os.PathLike[str],
    meta_path: str | os.PathLike[str] | None,
    frames: Any,
    depth_model: str,
    cache_signature: str,
    expected_input_identity: Mapping[str, Any] | None,
    legacy_metadata_policy: str,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]] | None:
    """Read one path-backed cache through the hardened canonical boundary."""

    cache = _lexical_absolute_path(cache_path)
    metadata = (
        (
            cache.with_name(DEPTH_CACHE_METADATA_FILENAME)
            if (
                cache.with_name(DEPTH_CACHE_METADATA_FILENAME).exists()
                or cache.with_name(DEPTH_CACHE_METADATA_FILENAME).is_symlink()
            )
            else cache.with_name(DEPTH_COMPAT_METADATA_FILENAME)
        )
        if meta_path is None
        else _lexical_absolute_path(meta_path)
    )
    if not (
        cache.exists()
        or cache.is_symlink()
        or metadata.exists()
        or metadata.is_symlink()
    ):
        return None
    frame_count, height, width = _frame_geometry(frames)
    depths, metadata_payload, report = read_validated_depth_cache(
        cache_path=cache,
        meta_path=metadata,
        expected_shape=(frame_count, height, width),
        expected_frame_count=frame_count,
        expected_input_identity=expected_input_identity,
        legacy_metadata_policy=legacy_metadata_policy,
    )
    if str(report.get("status", "") or "") == "valid":
        if str(metadata_payload.get("depth_model", "") or "") != str(depth_model or ""):
            raise DepthCacheValidationError(
                "depth cache validation failed: model_mismatch"
            )
        if str(metadata_payload.get("cache_signature", "") or "") != str(
            cache_signature or ""
        ):
            raise DepthCacheValidationError(
                "depth cache validation failed: cache_signature_mismatch"
            )
    return depths, metadata_payload, report


def _cache_info_and_stages(
    metadata: Mapping[str, Any],
    *,
    depth_model: str,
    depth_config_request: str,
    cache_signature: str,
    identity_verified: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    if str(metadata.get("schema", "") or "") == DEPTH_CACHE_SCHEMA:
        canonical_metadata = validate_canonical_depth_cache_metadata(
            metadata,
            label="depth cache metadata",
        )
        info = {
            key: copy.deepcopy(canonical_metadata[key])
            for key in sorted(_CANONICAL_CACHE_RUNTIME_INFO_FIELDS)
            if key in canonical_metadata
        }
    else:
        info = {
            key: copy.deepcopy(metadata[key])
            for key in sorted(_HISTORICAL_CACHE_RUNTIME_SCALAR_FIELDS)
            if key in metadata
            and (
                metadata[key] is None
                or isinstance(
                    metadata[key],
                    (str, int, float, bool),
                )
            )
        }
    info.setdefault("depth_model", str(depth_model or ""))
    info.setdefault(
        "depth_config_source",
        str(depth_config_request or ""),
    )
    info.setdefault("depth_space", "unknown")
    info.setdefault("cache_signature", cache_signature)
    info["identity_verified"] = bool(identity_verified)

    calibration = info.get("calibration")
    if isinstance(calibration, Mapping):
        info["calibration"] = {
            str(key): (dict(value) if isinstance(value, Mapping) else value)
            for key, value in calibration.items()
        }
        stages = stage_stats_from_cached_calibration(calibration)
    else:
        info["calibration"] = None
        stages = stage_stats_from_cached_calibration(None)
    return info, stages


def run_depth_source(
    *,
    frames: Any,
    target_fps: float,
    depth_model: str,
    depth_config_request: str,
    depth_base_cfg: Mapping[str, Any] | None,
    selected_video: str,
    use_rollout_gt_depth: bool = False,
    rollout_gt_depth: Any = None,
    rollout_gt_depth_path: str | os.PathLike[str] | None = None,
    use_depth_cache: bool = True,
    estimated_depth_cache: Any = None,
    estimated_depth_cache_path: str | os.PathLike[str] | None = None,
    estimated_depth_cache_meta: Mapping[str, Any] | None = None,
    estimated_depth_cache_meta_path: str | os.PathLike[str] | None = None,
    legacy_cache_metadata_policy: str = LEGACY_METADATA_POLICY_REJECT,
    expected_cache_input_identity: Mapping[str, Any] | None = None,
    force_recompute_depth: bool = False,
    trust_cache_on_signature_mismatch: bool = False,
    require_raw_model_depths: bool = False,
    depth_runtime_config: Mapping[str, Any] | None = None,
    init_ref_depth: Any = None,
    estimator: Callable[..., Any] | None = None,
    estimator_kwargs: Mapping[str, Any] | None = None,
    calibration: Callable[..., Any] | None = None,
    calibration_kwargs: Mapping[str, Any] | None = None,
    video_target_fps: float = -1,
    video_process_length: int = -1,
) -> tuple[np.ndarray, str, dict[str, Any], dict[str, Any]]:
    """Resolve rollout, cache, or model depth with explicit dependencies."""

    base_config = _mapping_copy(
        depth_base_cfg,
        label="depth base config",
    )
    estimator_options = _reserved_kwargs(
        estimator_kwargs,
        label="estimator",
        reserved=_ESTIMATOR_RESERVED_INPUTS,
    )
    calibration_options = _reserved_kwargs(
        calibration_kwargs,
        label="calibration",
        reserved=_CALIBRATION_RESERVED_INPUTS,
    )

    if use_rollout_gt_depth:
        cache_signature = build_depth_cache_signature(
            depth_model=depth_model,
            depth_config_request=depth_config_request,
            depth_base_cfg=base_config,
            selected_video=selected_video,
            depth_runtime_config=None,
        )
        if rollout_gt_depth is not None:
            source_depths = _depth_stack(
                rollout_gt_depth,
                label="rollout GT depth",
            )
        else:
            if not str(rollout_gt_depth_path or "").strip():
                raise ValueError(
                    "[Depth] depth.use_rollout_gt_depth=true but "
                    "depth.rollout_gt_depth_path is empty."
                )
            rollout_path = (
                Path(rollout_gt_depth_path).expanduser().resolve(strict=False)
            )
            if not rollout_path.is_file():
                raise FileNotFoundError(
                    f"[Depth] rollout GT depth not found: {rollout_path}"
                )
            source_depths = _depth_stack(
                np.load(rollout_path, allow_pickle=False),
                label="rollout GT depth",
            )

        source = _ROLLOUT_DEPTH_SOURCE
        info = {
            "depth_model": str(depth_model or ""),
            "depth_config_source": str(depth_config_request or ""),
            "depth_space": "metric",
            "calibration": None,
            "cache_signature": cache_signature,
        }
        source_depths, info, stages = _run_calibration(
            calibration,
            depths=source_depths,
            depth_source=source,
            depth_model=str(depth_model or ""),
            depth_base_cfg=base_config,
            depth_info=info,
            init_ref_depth=init_ref_depth,
            valid_masks=None,
            calibration_kwargs=calibration_options,
        )
        aligned, alignment = align_depth_stack_to_frames(
            source_depths,
            frames=frames,
            depth_source=source,
            video_target_fps=video_target_fps,
            video_process_length=video_process_length,
        )
        info["alignment"] = alignment
        return aligned, source, info, stages

    cache_signature = build_depth_cache_signature(
        depth_model=depth_model,
        depth_config_request=depth_config_request,
        depth_base_cfg=base_config,
        selected_video=selected_video,
        depth_runtime_config=depth_runtime_config,
    )
    cache_rejected_reason = ""
    if use_depth_cache and not force_recompute_depth:
        path_snapshot = None
        if (
            estimated_depth_cache is None
            and str(estimated_depth_cache_path or "").strip()
        ):
            if estimated_depth_cache_meta is not None:
                raise ValueError(
                    "path-backed depth cache cannot use explicit metadata; "
                    "provide estimated_depth_cache_meta_path or inject both "
                    "the cache array and metadata"
                )
            path_snapshot = _validated_path_cache(
                cache_path=estimated_depth_cache_path,
                meta_path=estimated_depth_cache_meta_path,
                frames=frames,
                depth_model=depth_model,
                cache_signature=cache_signature,
                expected_input_identity=expected_cache_input_identity,
                legacy_metadata_policy=legacy_cache_metadata_policy,
            )
        cached_depths = None if path_snapshot is None else path_snapshot[0]
        if estimated_depth_cache is not None:
            cached_depths = _cache_depths(
                explicit_depths=estimated_depth_cache,
            )
        if cached_depths is not None:
            metadata = (
                path_snapshot[1]
                if path_snapshot is not None
                else _cache_metadata(
                    cache_path=estimated_depth_cache_path,
                    explicit_metadata=estimated_depth_cache_meta,
                    meta_path=estimated_depth_cache_meta_path,
                    legacy_metadata_policy=legacy_cache_metadata_policy,
                )
            )
            compatible, reason = evaluate_depth_cache_compatibility(
                meta_payload=metadata,
                expected_cache_signature=cache_signature,
                selected_video=selected_video,
                expected_input_identity=expected_cache_input_identity,
                trust_cache_on_signature_mismatch=(trust_cache_on_signature_mismatch),
            )
            if compatible:
                if bool(require_raw_model_depths):
                    cache_rejected_reason = (
                        "raw model depths are required but the canonical "
                        "cache has no raw inference sidecar"
                    )
                else:
                    info, stages = _cache_info_and_stages(
                        metadata,
                        depth_model=depth_model,
                        depth_config_request=depth_config_request,
                        cache_signature=cache_signature,
                        identity_verified=bool(
                            expected_cache_input_identity is not None
                            and metadata.get("input_identity", None) is not None
                            and _json_ready(metadata.get("input_identity"))
                            == _json_ready(dict(expected_cache_input_identity))
                        ),
                    )
                    aligned, alignment = align_depth_stack_to_frames(
                        cached_depths,
                        frames=frames,
                        depth_source=_CACHE_DEPTH_SOURCE,
                        video_target_fps=video_target_fps,
                        video_process_length=video_process_length,
                    )
                    info["alignment"] = alignment
                    return (
                        aligned,
                        _CACHE_DEPTH_SOURCE,
                        info,
                        stages,
                    )
            else:
                cache_rejected_reason = reason

    if estimator is None:
        raise RuntimeError(
            "depth estimator is required when rollout GT depth "
            "and a compatible cache are unavailable"
        )
    result = estimator(
        video_frames=frames,
        target_fps=float(target_fps),
        **estimator_options,
    )
    source_depths, estimator_fps, info, aux = _estimator_result(
        result,
        target_fps=float(target_fps),
    )
    raw_model_depths = source_depths.copy()
    source = _MODEL_DEPTH_SOURCE
    info.setdefault("depth_model", str(depth_model or ""))
    info.setdefault(
        "depth_config_source",
        str(depth_config_request or ""),
    )
    info.setdefault("depth_space", "unknown")
    info.setdefault("calibration", None)
    info["cache_signature"] = cache_signature
    info["estimator_fps"] = estimator_fps
    if cache_rejected_reason:
        info["cache_rejected_reason"] = cache_rejected_reason

    source_depths, info, stages = _run_calibration(
        calibration,
        depths=source_depths,
        depth_source=source,
        depth_model=str(depth_model or ""),
        depth_base_cfg=base_config,
        depth_info=info,
        init_ref_depth=(
            init_ref_depth if init_ref_depth is not None else aux.get("init_ref_depth")
        ),
        valid_masks=aux.get("valid_masks"),
        calibration_kwargs=calibration_options,
    )
    aligned, alignment = align_depth_stack_to_frames(
        source_depths,
        frames=frames,
        depth_source=source,
        video_target_fps=video_target_fps,
        video_process_length=video_process_length,
    )
    aligned_raw, _ = align_depth_stack_to_frames(
        raw_model_depths,
        frames=frames,
        depth_source=source,
        video_target_fps=video_target_fps,
        video_process_length=video_process_length,
    )
    info["raw_model_depths"] = aligned_raw
    info["alignment"] = alignment
    return aligned, source, info, stages


__all__ = [
    "align_depth_stack_to_frames",
    "build_depth_cache_signature",
    "evaluate_depth_cache_compatibility",
    "load_depth_cache_meta",
    "normalize_depth_runtime_cache_identity",
    "run_depth_source",
    "stage_stats_from_cached_calibration",
]
