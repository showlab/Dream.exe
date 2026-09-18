"""Canonical configuration contracts for calibrated depth and target lifting.

This module is deliberately limited to side-effect-free configuration work.
Only the public current implementation schema is accepted, so misspelled or out-of-contract fields
fail before they can influence runtime behavior.
"""

from __future__ import annotations

import copy
import math
from typing import Any


_BASE_TEXT_RULES = (
    ("mode", None),
    ("calibration_solver", None),
    ("multi_roi_strategy", None),
    ("calib_region", None),
    ("calib_roi_mode", "joint"),
)
_BASE_OPTIONAL_INTS = ("roi_dilate_px", "points_radius_px")
_DISTRIBUTION_INTS = (
    "task_roi_margin_px",
    "support_exclusion_dilate_px",
    "blend_px",
    "min_valid_pixels",
)
_DISTRIBUTION_FLOATS = (
    "support_edge_band_fraction",
    "support_quantile_min",
    "support_quantile_max",
    "task_scale_ratio_limit",
    "task_bias_delta_limit_m",
    "support_bias_limit_m",
    "vis_threshold",
)

EEF_TRAJECTORY_DEPTH_MODES = {
    "static_eef",
    "dynamic_shift",
    "dynamic_affine",
}
OBJECT_TRAJECTORY_DEPTH_MODES = {"object_roi"}
GRIPPER_TRAJECTORY_DEPTH_MODES = {"pair_roi", "all_roi"}

_BASE_FIELDS = frozenset({"enabled", "sanitize", "init_calibration"})
_SANITIZE_FIELDS = frozenset({"enabled", "invalidate_mode"})
_INIT_CALIBRATION_FIELDS = frozenset(
    {
        "enabled",
        "if_calibrate_depth",
        "mode",
        "calibration_solver",
        "multi_roi_strategy",
        "calib_region",
        "calib_roi_mode",
        "roi_dilate_px",
        "points_radius_px",
        "background_nearfield_quantile",
        "require_calibration_for_nonmetric",
        "distribution_v1",
    }
)
_DISTRIBUTION_FIELDS = frozenset({*_DISTRIBUTION_INTS, *_DISTRIBUTION_FLOATS})
_TARGET_FIELDS = frozenset(
    {
        "enabled",
        "source_stage",
        "calibration_solver",
        "min_valid_pixels",
        "eef_traj",
        "object_traj",
        "gripper_traj",
        "save_debug_maps",
        "save_depth_maps",
        "save_depth_videos",
    }
)
_EEF_FIELDS = frozenset(
    {
        "enabled",
        "mode",
        "static_eef",
        "dynamic_shift",
        "dynamic_affine",
    }
)
_STATIC_EEF_FIELDS = frozenset({"calib_region"})
_DYNAMIC_AFFINE_FIELDS = frozenset(
    {
        "switch_strategy",
        "proximity_threshold",
        "consecutive_frames",
        "min_region_pixels",
        "max_pre_switch_motion_norm",
        "ramp_frames",
    }
)
_DYNAMIC_SHIFT_FIELDS = frozenset({*_DYNAMIC_AFFINE_FIELDS, "max_abs_delta"})
_OBJECT_FIELDS = frozenset({"enabled", "mode", "object_roi"})
_OBJECT_ROI_FIELDS = frozenset({"calib_region"})
_GRIPPER_FIELDS = frozenset({"enabled", "mode", "pair_roi", "all_roi"})
_PAIR_ROI_FIELDS = frozenset({"calib_region", "pair_granularity", "time_scope"})
_ALL_ROI_FIELDS = frozenset({"calib_region"})


def _new_base_template() -> dict[str, Any]:
    return {
        "enabled": True,
        "sanitize": {
            "enabled": True,
            "invalidate_mode": None,
        },
        "init_calibration": {
            "enabled": True,
            "if_calibrate_depth": None,
            "mode": None,
            "calibration_solver": None,
            "multi_roi_strategy": None,
            "calib_region": None,
            "calib_roi_mode": "joint",
            "roi_dilate_px": None,
            "points_radius_px": None,
            "background_nearfield_quantile": None,
            "require_calibration_for_nonmetric": None,
            "distribution_v1": {
                "task_roi_margin_px": 24,
                "support_exclusion_dilate_px": 24,
                "blend_px": 12,
                "min_valid_pixels": 1500,
                "support_edge_band_fraction": 0.15,
                "support_quantile_min": 0.15,
                "support_quantile_max": 0.60,
                "task_scale_ratio_limit": 0.10,
                "task_bias_delta_limit_m": 0.03,
                "support_bias_limit_m": 0.02,
                "vis_threshold": 0.40,
            },
        },
    }


DEFAULT_DEPTH_BASE_CONFIG = _new_base_template()


def _new_target_template() -> dict[str, Any]:
    return {
        "enabled": True,
        "source_stage": "raw_model",
        "calibration_solver": "robust_affine",
        "min_valid_pixels": 16,
        "eef_traj": {
            "enabled": True,
            "mode": "static_eef",
            "static_eef": {"calib_region": "roi∧valid"},
            "dynamic_shift": {
                "switch_strategy": "proximity_2d",
                "proximity_threshold": 1.5,
                "consecutive_frames": 2,
                "min_region_pixels": 50,
                "max_abs_delta": 1.0,
                "max_pre_switch_motion_norm": 2.0,
                "ramp_frames": 5,
            },
            "dynamic_affine": {
                "switch_strategy": "proximity_2d",
                "proximity_threshold": 1.5,
                "consecutive_frames": 2,
                "min_region_pixels": 50,
                "max_pre_switch_motion_norm": 2.0,
                "ramp_frames": 5,
            },
        },
        "object_traj": {
            "enabled": False,
            "mode": "object_roi",
            "object_roi": {"calib_region": "roi∧valid"},
        },
        "gripper_traj": {
            "enabled": True,
            "mode": "pair_roi",
            "pair_roi": {
                "calib_region": "roi∧valid",
                "pair_granularity": "runtime_object",
                "time_scope": "full_video",
            },
            "all_roi": {"calib_region": "roi∧valid"},
        },
        "save_debug_maps": False,
        "save_depth_maps": False,
        "save_depth_videos": False,
    }


def _layered_copy(
    foundation: dict[str, Any],
    additions: dict[str, Any],
) -> dict[str, Any]:
    """Return a detached recursive overlay without mutating either input."""

    result = copy.deepcopy(foundation)
    work: list[tuple[dict[str, Any], dict[str, Any]]] = [(result, additions)]
    while work:
        destination, source = work.pop()
        for key, value in source.items():
            current = destination.get(key)
            if isinstance(current, dict) and isinstance(value, dict):
                detached = copy.deepcopy(current)
                destination[key] = detached
                work.append((detached, value))
            else:
                destination[key] = copy.deepcopy(value)
    return result


def _mapping_or_empty(value: Any) -> dict[str, Any]:
    return dict(value or {}) if isinstance(value, dict) else {}


def _config_object(value: Any, *, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"{path} must be an object")
    return value


def _reject_unknown_fields(
    value: dict[str, Any],
    *,
    allowed: frozenset[str],
    path: str,
) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"{path} contains unknown fields: {', '.join(unknown)}")


def _validate_base_schema(config: dict[str, Any]) -> None:
    _reject_unknown_fields(
        config,
        allowed=_BASE_FIELDS,
        path="depth.base",
    )
    sanitize = _config_object(
        config.get("sanitize"),
        path="depth.base.sanitize",
    )
    _reject_unknown_fields(
        sanitize,
        allowed=_SANITIZE_FIELDS,
        path="depth.base.sanitize",
    )
    calibration = _config_object(
        config.get("init_calibration"),
        path="depth.base.init_calibration",
    )
    _reject_unknown_fields(
        calibration,
        allowed=_INIT_CALIBRATION_FIELDS,
        path="depth.base.init_calibration",
    )
    distribution = _config_object(
        calibration.get("distribution_v1"),
        path="depth.base.init_calibration.distribution_v1",
    )
    _reject_unknown_fields(
        distribution,
        allowed=_DISTRIBUTION_FIELDS,
        path="depth.base.init_calibration.distribution_v1",
    )


def _validate_target_schema(config: dict[str, Any]) -> None:
    prefix = "depth.target_calibrated_lift"
    _reject_unknown_fields(
        config,
        allowed=_TARGET_FIELDS,
        path=prefix,
    )
    eef = _config_object(
        config.get("eef_traj"),
        path=f"{prefix}.eef_traj",
    )
    _reject_unknown_fields(
        eef,
        allowed=_EEF_FIELDS,
        path=f"{prefix}.eef_traj",
    )
    static_eef = _config_object(
        eef.get("static_eef"),
        path=f"{prefix}.eef_traj.static_eef",
    )
    _reject_unknown_fields(
        static_eef,
        allowed=_STATIC_EEF_FIELDS,
        path=f"{prefix}.eef_traj.static_eef",
    )
    dynamic_shift = _config_object(
        eef.get("dynamic_shift"),
        path=f"{prefix}.eef_traj.dynamic_shift",
    )
    _reject_unknown_fields(
        dynamic_shift,
        allowed=_DYNAMIC_SHIFT_FIELDS,
        path=f"{prefix}.eef_traj.dynamic_shift",
    )
    dynamic_affine = _config_object(
        eef.get("dynamic_affine"),
        path=f"{prefix}.eef_traj.dynamic_affine",
    )
    _reject_unknown_fields(
        dynamic_affine,
        allowed=_DYNAMIC_AFFINE_FIELDS,
        path=f"{prefix}.eef_traj.dynamic_affine",
    )
    object_traj = _config_object(
        config.get("object_traj"),
        path=f"{prefix}.object_traj",
    )
    _reject_unknown_fields(
        object_traj,
        allowed=_OBJECT_FIELDS,
        path=f"{prefix}.object_traj",
    )
    object_roi = _config_object(
        object_traj.get("object_roi"),
        path=f"{prefix}.object_traj.object_roi",
    )
    _reject_unknown_fields(
        object_roi,
        allowed=_OBJECT_ROI_FIELDS,
        path=f"{prefix}.object_traj.object_roi",
    )
    gripper = _config_object(
        config.get("gripper_traj"),
        path=f"{prefix}.gripper_traj",
    )
    _reject_unknown_fields(
        gripper,
        allowed=_GRIPPER_FIELDS,
        path=f"{prefix}.gripper_traj",
    )
    pair = _config_object(
        gripper.get("pair_roi"),
        path=f"{prefix}.gripper_traj.pair_roi",
    )
    _reject_unknown_fields(
        pair,
        allowed=_PAIR_ROI_FIELDS,
        path=f"{prefix}.gripper_traj.pair_roi",
    )
    all_roi = _config_object(
        gripper.get("all_roi"),
        path=f"{prefix}.gripper_traj.all_roi",
    )
    _reject_unknown_fields(
        all_roi,
        allowed=_ALL_ROI_FIELDS,
        path=f"{prefix}.gripper_traj.all_roi",
    )


def _text_or_none(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _require_exact_bool(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")
    return value


def _require_integer(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _require_finite_number(value: Any, *, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{path} must be a finite number")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{path} must be a finite number")
    return number


def _optional_flag(value: Any, *, path: str) -> bool | None:
    if value is None:
        return None
    return _require_exact_bool(value, path=path)


def _validate_base_numeric_types(config: dict[str, Any]) -> None:
    _require_exact_bool(config.get("enabled"), path="depth.base.enabled")
    sanitize = _config_object(
        config.get("sanitize"),
        path="depth.base.sanitize",
    )
    _require_exact_bool(
        sanitize.get("enabled"),
        path="depth.base.sanitize.enabled",
    )
    calibration = _config_object(
        config.get("init_calibration"),
        path="depth.base.init_calibration",
    )
    _require_exact_bool(
        calibration.get("enabled"),
        path="depth.base.init_calibration.enabled",
    )
    for key in (
        "if_calibrate_depth",
        "require_calibration_for_nonmetric",
    ):
        value = calibration.get(key)
        if value is not None:
            _require_exact_bool(
                value,
                path=f"depth.base.init_calibration.{key}",
            )
    for key in _BASE_OPTIONAL_INTS:
        value = calibration.get(key)
        if value is not None:
            _require_integer(
                value,
                path=f"depth.base.init_calibration.{key}",
            )
    nearfield = calibration.get("background_nearfield_quantile")
    if nearfield is not None:
        _require_finite_number(
            nearfield,
            path=("depth.base.init_calibration.background_nearfield_quantile"),
        )
    distribution = _config_object(
        calibration.get("distribution_v1"),
        path="depth.base.init_calibration.distribution_v1",
    )
    for key in _DISTRIBUTION_INTS:
        _require_integer(
            distribution.get(key),
            path=(f"depth.base.init_calibration.distribution_v1.{key}"),
        )
    for key in _DISTRIBUTION_FLOATS:
        _require_finite_number(
            distribution.get(key),
            path=(f"depth.base.init_calibration.distribution_v1.{key}"),
        )


def _validate_target_numeric_types(config: dict[str, Any]) -> None:
    prefix = "depth.target_calibrated_lift"
    for key in (
        "enabled",
        "save_debug_maps",
        "save_depth_maps",
        "save_depth_videos",
    ):
        _require_exact_bool(
            config.get(key),
            path=f"{prefix}.{key}",
        )
    _require_integer(
        config.get("min_valid_pixels"),
        path=f"{prefix}.min_valid_pixels",
    )
    for section in ("eef_traj", "object_traj", "gripper_traj"):
        nested = _config_object(
            config.get(section),
            path=f"{prefix}.{section}",
        )
        _require_exact_bool(
            nested.get("enabled"),
            path=f"{prefix}.{section}.enabled",
        )
    eef = _config_object(
        config.get("eef_traj"),
        path=f"{prefix}.eef_traj",
    )
    for policy_name in ("dynamic_shift", "dynamic_affine"):
        policy = _config_object(
            eef.get(policy_name),
            path=f"{prefix}.eef_traj.{policy_name}",
        )
        for key in (
            "consecutive_frames",
            "min_region_pixels",
            "ramp_frames",
        ):
            _require_integer(
                policy.get(key),
                path=f"{prefix}.eef_traj.{policy_name}.{key}",
            )
        for key in (
            "proximity_threshold",
            "max_pre_switch_motion_norm",
        ):
            _require_finite_number(
                policy.get(key),
                path=f"{prefix}.eef_traj.{policy_name}.{key}",
            )
    _require_finite_number(
        eef["dynamic_shift"].get("max_abs_delta"),
        path=f"{prefix}.eef_traj.dynamic_shift.max_abs_delta",
    )


def default_depth_base_config_dict() -> dict[str, Any]:
    """Return an independent copy of the base depth defaults."""

    return copy.deepcopy(DEFAULT_DEPTH_BASE_CONFIG)


def normalize_depth_base_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize one canonical calibration configuration."""

    if raw_config is None:
        supplied: dict[str, Any] = {}
    elif isinstance(raw_config, dict):
        supplied = copy.deepcopy(raw_config)
    else:
        raise ValueError("depth.base must be an object")
    normalized = _layered_copy(DEFAULT_DEPTH_BASE_CONFIG, supplied)
    _validate_base_schema(normalized)
    _validate_base_numeric_types(normalized)

    normalized["enabled"] = normalized.get("enabled", True)

    sanitize = dict(normalized["sanitize"])
    sanitize["enabled"] = sanitize.get("enabled", True)
    invalidate_mode = _text_or_none(sanitize.get("invalidate_mode"))
    sanitize["invalidate_mode"] = (
        None if invalidate_mode is None else invalidate_mode.lower()
    )
    normalized["sanitize"] = sanitize

    calibration = dict(normalized["init_calibration"])
    calibration["enabled"] = calibration.get("enabled", True)
    for key in (
        "if_calibrate_depth",
        "require_calibration_for_nonmetric",
    ):
        calibration[key] = _optional_flag(
            calibration.get(key),
            path=f"depth.base.init_calibration.{key}",
        )
    for key, fallback in _BASE_TEXT_RULES:
        calibration[key] = _text_or_none(calibration.get(key, fallback))
    for key in _BASE_OPTIONAL_INTS:
        value = calibration.get(key)
        calibration[key] = None if value is None else int(value)
    nearfield = calibration.get("background_nearfield_quantile")
    calibration["background_nearfield_quantile"] = (
        None if nearfield is None else float(nearfield)
    )

    distribution = dict(calibration["distribution_v1"])
    distribution_defaults = DEFAULT_DEPTH_BASE_CONFIG["init_calibration"][
        "distribution_v1"
    ]
    distribution = _layered_copy(
        distribution_defaults,
        distribution,
    )
    for key in _DISTRIBUTION_INTS:
        distribution[key] = int(distribution[key])
    for key in _DISTRIBUTION_FLOATS:
        distribution[key] = float(distribution[key])
    calibration["distribution_v1"] = distribution
    normalized["init_calibration"] = calibration

    return normalized


def bind_depth_base_runtime_calibration(
    base_config: dict[str, Any] | None,
    *,
    depth_runtime_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Resolve nullable base-calibration fields from the selected backend.

    Pipeline values remain authoritative when explicitly set.  A ``None``
    value means the pipeline delegates that one calibration choice to the
    selected depth runtime.  This keeps model selection and numeric depth
    calibration independently configurable through one canonical schema.
    """

    resolved = normalize_depth_base_config(base_config)
    if depth_runtime_config is None:
        return resolved
    if not isinstance(depth_runtime_config, dict):
        raise ValueError("depth_runtime_config must be an object")

    runtime = copy.deepcopy(depth_runtime_config)
    raw_model_config = runtime.get("config", {})
    if raw_model_config is None:
        raw_model_config = {}
    if not isinstance(raw_model_config, dict):
        raise ValueError("depth_runtime_config.config must be an object")
    raw_calibration = raw_model_config.get("calibration", {})
    if raw_calibration is None:
        raw_calibration = {}
    if not isinstance(raw_calibration, dict):
        raise ValueError("depth_runtime_config.config.calibration must be an object")
    provider = copy.deepcopy(raw_calibration)
    for field in (
        "if_calibrate_depth",
        "calibration_solver",
        "multi_roi_strategy",
        "calib_region",
        "roi_dilate_px",
        "points_radius_px",
        "invalidate_mode",
        "require_calibration_for_nonmetric",
    ):
        if runtime.get(field) is not None:
            provider[field] = copy.deepcopy(runtime[field])

    init = dict(resolved["init_calibration"])
    for field in (
        "if_calibrate_depth",
        "calibration_solver",
        "multi_roi_strategy",
        "calib_region",
        "roi_dilate_px",
        "points_radius_px",
        "background_nearfield_quantile",
        "require_calibration_for_nonmetric",
    ):
        if init.get(field) is None and provider.get(field) is not None:
            init[field] = copy.deepcopy(provider[field])
    resolved["init_calibration"] = init

    sanitize = dict(resolved["sanitize"])
    if (
        sanitize.get("invalidate_mode") is None
        and provider.get("invalidate_mode") is not None
    ):
        sanitize["invalidate_mode"] = copy.deepcopy(provider["invalidate_mode"])
    resolved["sanitize"] = sanitize
    return normalize_depth_base_config(resolved)


def _unsupported(
    *,
    source: str,
    path: str,
    value: Any,
) -> ValueError:
    return ValueError(f"Unsupported {path} in {source}: {value}")


def validate_depth_base_config(
    *,
    base_config: dict[str, Any],
    source: str,
) -> None:
    """Validate a normalized base-depth configuration."""

    if not isinstance(base_config, dict):
        raise ValueError(f"depth.base must be an object in {source}")
    _validate_base_schema(base_config)
    _validate_base_numeric_types(base_config)

    sanitize = _mapping_or_empty(base_config.get("sanitize"))
    invalidate_mode = sanitize.get("invalidate_mode")
    if invalidate_mode not in {None, "none", "nan"}:
        raise _unsupported(
            source=source,
            path="depth.base.sanitize.invalidate_mode",
            value=invalidate_mode,
        )

    calibration = _mapping_or_empty(base_config.get("init_calibration"))
    enumerations = (
        (
            "mode",
            {None, "standard", "distribution_v1"},
        ),
        (
            "calibration_solver",
            {
                None,
                "robust_affine",
                "least_squares_scale",
                "least_squares_affine",
            },
        ),
        (
            "multi_roi_strategy",
            {None, "blend", "union"},
        ),
        (
            "calib_region",
            {
                None,
                "full",
                "valid",
                "roi",
                "roi∧valid",
                "tracks_points",
                "background_nearfield",
                "task_firstframe_neighborhood",
                "task_motion_envelope",
            },
        ),
        (
            "calib_roi_mode",
            {
                "joint",
                "eef",
                "eef_only",
                "obj",
                "object",
                "obj_only",
                "object_only",
                "split",
                "per_roi",
                "blend",
            },
        ),
    )
    for key, choices in enumerations:
        value = calibration.get(key)
        if value not in choices:
            raise _unsupported(
                source=source,
                path=f"depth.base.init_calibration.{key}",
                value=value,
            )

    for key in _BASE_OPTIONAL_INTS:
        value = calibration.get(key)
        if value is not None and int(value) < 0:
            raise ValueError(
                f"depth.base.init_calibration.{key} must be >= 0 in {source}"
            )

    nearfield = calibration.get("background_nearfield_quantile")
    if nearfield is not None and not (0.0 < float(nearfield) <= 1.0):
        raise ValueError(
            "depth.base.init_calibration."
            "background_nearfield_quantile must be in (0,1] "
            f"in {source}"
        )

    distribution = _mapping_or_empty(calibration.get("distribution_v1"))
    prefix = "depth.base.init_calibration.distribution_v1"
    for key in (
        "task_roi_margin_px",
        "support_exclusion_dilate_px",
        "blend_px",
    ):
        if int(distribution.get(key, 0)) < 0:
            raise ValueError(f"{prefix}.{key} must be >= 0 in {source}")
    if int(distribution.get("min_valid_pixels", 0)) <= 0:
        raise ValueError(f"{prefix}.min_valid_pixels must be > 0 in {source}")

    edge_fraction = float(distribution.get("support_edge_band_fraction", 0.0))
    if not 0.0 <= edge_fraction < 0.5:
        raise ValueError(
            f"{prefix}.support_edge_band_fraction must be in [0,0.5) in {source}"
        )

    quantile_min = float(distribution.get("support_quantile_min", 0.0))
    quantile_max = float(distribution.get("support_quantile_max", 0.0))
    if not 0.0 <= quantile_min < quantile_max <= 1.0:
        raise ValueError(
            f"{prefix}.support_quantile_min/max must satisfy "
            f"0 <= min < max <= 1 in {source}"
        )

    for key in (
        "task_scale_ratio_limit",
        "task_bias_delta_limit_m",
        "support_bias_limit_m",
    ):
        if float(distribution.get(key, 0.0)) < 0.0:
            raise ValueError(f"{prefix}.{key} must be >= 0 in {source}")
    visibility = float(distribution.get("vis_threshold", 0.0))
    if not 0.0 <= visibility <= 1.0:
        raise ValueError(f"{prefix}.vis_threshold must be in [0,1] in {source}")


def default_target_calibrated_lift_config_dict() -> dict[str, Any]:
    """Return independent defaults for target-specific depth lifting."""

    return _new_target_template()


def normalize_target_calibrated_lift_config(
    raw_config: dict[str, Any] | None,
) -> dict[str, Any]:
    """Normalize one canonical target-lift configuration."""

    if raw_config is None:
        supplied: dict[str, Any] = {}
    elif isinstance(raw_config, dict):
        supplied = copy.deepcopy(raw_config)
    else:
        raise ValueError("depth.target_calibrated_lift must be an object")
    normalized = _layered_copy(_new_target_template(), supplied)
    _validate_target_schema(normalized)
    _validate_target_numeric_types(normalized)

    normalized["enabled"] = normalized.get("enabled", True)
    normalized["source_stage"] = str(
        normalized.get("source_stage") or "raw_model"
    ).strip()
    normalized["calibration_solver"] = str(
        normalized.get("calibration_solver") or "robust_affine"
    ).strip()
    normalized["min_valid_pixels"] = int(normalized.get("min_valid_pixels", 16))
    for key in (
        "save_debug_maps",
        "save_depth_maps",
        "save_depth_videos",
    ):
        normalized[key] = normalized.get(key, False)

    eef = dict(normalized["eef_traj"])
    eef["enabled"] = eef.get("enabled", True)
    eef["mode"] = str(eef.get("mode") or "static_eef").strip()
    eef["static_eef"] = _layered_copy(
        {"calib_region": "roi∧valid"},
        dict(eef["static_eef"]),
    )
    dynamic_shift = _layered_copy(
        _new_target_template()["eef_traj"]["dynamic_shift"],
        dict(eef["dynamic_shift"]),
    )
    dynamic_shift["switch_strategy"] = str(
        dynamic_shift.get("switch_strategy") or "proximity_2d"
    ).strip()
    for key in (
        "consecutive_frames",
        "min_region_pixels",
        "ramp_frames",
    ):
        dynamic_shift[key] = int(dynamic_shift[key])
    for key in (
        "proximity_threshold",
        "max_abs_delta",
        "max_pre_switch_motion_norm",
    ):
        dynamic_shift[key] = float(dynamic_shift[key])
    eef["dynamic_shift"] = dynamic_shift
    dynamic_affine = _layered_copy(
        _new_target_template()["eef_traj"]["dynamic_affine"],
        dict(eef["dynamic_affine"]),
    )
    dynamic_affine["switch_strategy"] = str(
        dynamic_affine.get("switch_strategy") or "proximity_2d"
    ).strip()
    for key in (
        "consecutive_frames",
        "min_region_pixels",
        "ramp_frames",
    ):
        dynamic_affine[key] = int(dynamic_affine[key])
    for key in (
        "proximity_threshold",
        "max_pre_switch_motion_norm",
    ):
        dynamic_affine[key] = float(dynamic_affine[key])
    eef["dynamic_affine"] = dynamic_affine
    normalized["eef_traj"] = eef

    object_traj = dict(normalized["object_traj"])
    object_traj["enabled"] = object_traj.get("enabled", False)
    object_traj["mode"] = str(object_traj.get("mode") or "object_roi").strip()
    object_traj["object_roi"] = _layered_copy(
        {"calib_region": "roi∧valid"},
        dict(object_traj["object_roi"]),
    )
    normalized["object_traj"] = object_traj

    gripper = dict(normalized["gripper_traj"])
    gripper["enabled"] = gripper.get("enabled", True)
    gripper["mode"] = str(gripper.get("mode") or "pair_roi").strip()
    pair = _layered_copy(
        {
            "calib_region": "roi∧valid",
            "pair_granularity": "runtime_object",
            "time_scope": "full_video",
        },
        dict(gripper["pair_roi"]),
    )
    pair["pair_granularity"] = str(
        pair.get("pair_granularity") or "runtime_object"
    ).strip()
    pair["time_scope"] = str(pair.get("time_scope") or "full_video").strip()
    gripper["pair_roi"] = pair
    gripper["all_roi"] = _layered_copy(
        {"calib_region": "roi∧valid"},
        dict(gripper["all_roi"]),
    )
    normalized["gripper_traj"] = gripper
    return normalized


def validate_target_calibrated_lift_config(
    *,
    config: dict[str, Any],
    source: str,
) -> None:
    """Validate the static target-depth contract."""

    if not isinstance(config, dict):
        raise ValueError(f"depth.target_calibrated_lift must be an object in {source}")
    _validate_target_schema(config)
    _validate_target_numeric_types(config)

    prefix = "depth.target_calibrated_lift"
    source_stage = config.get("source_stage", "raw_model")
    if source_stage not in {"raw_model", "canonical"}:
        raise _unsupported(
            source=source,
            path=f"{prefix}.source_stage",
            value=source_stage,
        )

    solver = config.get("calibration_solver", "robust_affine")
    if solver not in {
        "robust_affine",
        "least_squares_scale",
        "least_squares_affine",
    }:
        raise _unsupported(
            source=source,
            path=f"{prefix}.calibration_solver",
            value=solver,
        )

    eef = _mapping_or_empty(config.get("eef_traj"))
    eef_mode = eef.get("mode", "static_eef")
    if eef_mode not in EEF_TRAJECTORY_DEPTH_MODES:
        raise _unsupported(
            source=source,
            path=f"{prefix}.eef_traj.mode",
            value=eef_mode,
        )
    if eef_mode == "dynamic_affine" and source_stage != "raw_model":
        raise ValueError(
            f"{prefix}.eef_traj.mode=dynamic_affine requires "
            f"source_stage='raw_model' in {source}"
        )
    for policy_name in ("dynamic_shift", "dynamic_affine"):
        policy = _mapping_or_empty(eef.get(policy_name))
        if policy.get("switch_strategy") != "proximity_2d":
            raise _unsupported(
                source=source,
                path=(f"{prefix}.eef_traj.{policy_name}.switch_strategy"),
                value=policy.get("switch_strategy"),
            )
        if float(policy.get("proximity_threshold", 0.0)) <= 0.0:
            raise ValueError(
                f"{prefix}.eef_traj.{policy_name}."
                f"proximity_threshold must be > 0 in {source}"
            )
        for key in (
            "consecutive_frames",
            "min_region_pixels",
            "ramp_frames",
        ):
            if int(policy.get(key, 0)) <= 0:
                raise ValueError(
                    f"{prefix}.eef_traj.{policy_name}.{key} must be > 0 in {source}"
                )
        if float(policy.get("max_pre_switch_motion_norm", -1.0)) < 0.0:
            raise ValueError(
                f"{prefix}.eef_traj.{policy_name}."
                "max_pre_switch_motion_norm must be >= 0 "
                f"in {source}"
            )
    if float(eef["dynamic_shift"].get("max_abs_delta", -1.0)) < 0.0:
        raise ValueError(
            f"{prefix}.eef_traj.dynamic_shift.max_abs_delta must be >= 0 in {source}"
        )

    object_traj = _mapping_or_empty(config.get("object_traj"))
    object_mode = object_traj.get("mode", "object_roi")
    if object_mode not in OBJECT_TRAJECTORY_DEPTH_MODES:
        raise _unsupported(
            source=source,
            path=f"{prefix}.object_traj.mode",
            value=object_mode,
        )

    gripper = _mapping_or_empty(config.get("gripper_traj"))
    gripper_mode = gripper.get("mode", "pair_roi")
    if gripper_mode not in GRIPPER_TRAJECTORY_DEPTH_MODES:
        raise _unsupported(
            source=source,
            path=f"{prefix}.gripper_traj.mode",
            value=gripper_mode,
        )

    pair = _mapping_or_empty(gripper.get("pair_roi"))
    granularity = pair.get(
        "pair_granularity",
        "runtime_object",
    )
    if granularity != "runtime_object":
        raise _unsupported(
            source=source,
            path=(f"{prefix}.gripper_traj.pair_roi.pair_granularity"),
            value=granularity,
        )

    if int(config.get("min_valid_pixels", 0)) <= 0:
        raise ValueError(f"{prefix}.min_valid_pixels must be > 0 in {source}")
