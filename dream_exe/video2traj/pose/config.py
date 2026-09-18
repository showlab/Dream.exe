"""Configuration contract for simulator-independent pose estimation."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_POSE_CONFIG_PATH = (
    Path(__file__).resolve().parents[1] / "configs" / "pose.default.json"
)
POINTCLOUD_KABSCH_BACKEND = "pointcloud_kabsch"
EXTERNAL_POSE_SELECTION_PREFIX = "external:"
MODEL_POSE_BACKENDS = ("foundationpose", "freepose", "sinref6d")
SUPPORTED_POSE_BACKENDS = (
    POINTCLOUD_KABSCH_BACKEND,
    *MODEL_POSE_BACKENDS,
)


def external_pose_backend_id(value: Any) -> str | None:
    """Return a validated external backend ID or ``None`` for built-ins."""

    selection = str(value or "").strip().lower()
    if not selection.startswith(EXTERNAL_POSE_SELECTION_PREFIX):
        return None
    backend_id = selection[len(EXTERNAL_POSE_SELECTION_PREFIX) :].strip()
    if (
        not backend_id
        or backend_id.startswith(".")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in backend_id
        )
    ):
        raise ValueError(
            "external pose selection must be 'external:<backend_id>' using [a-z0-9._-]"
        )
    return backend_id


def external_pose_selection(backend_id: Any) -> str:
    """Build the canonical identity for an external pose provider."""

    selection = EXTERNAL_POSE_SELECTION_PREFIX + str(backend_id or "").strip().lower()
    parsed = external_pose_backend_id(selection)
    assert parsed is not None
    return f"{EXTERNAL_POSE_SELECTION_PREFIX}{parsed}"


def pose_backend_identity_is_known(value: Any) -> bool:
    """Return whether an identity is built-in or explicitly external."""

    normalized = str(value or "").strip().lower()
    return (
        normalized in SUPPORTED_POSE_BACKENDS
        or external_pose_backend_id(normalized) is not None
    )


@dataclass(frozen=True)
class PoseConfig:
    backend: str = POINTCLOUD_KABSCH_BACKEND
    weights_root: str = ""
    mesh_path: str = ""
    model_kwargs: Dict[str, Any] = field(default_factory=dict)
    init_refine_iter: int = 5
    track_refine_iter: int = 2
    debug: int = 0
    force_recompute: bool = False
    foundationpose_enabled: bool = False
    foundationpose_fallback: bool = False
    kabsch_enabled: bool = True
    use_config_init_pose: bool = False
    foundationpose_force_register_frame0: bool = False
    pose_correction_path: str = ""
    pose_correction_matrix: Optional[Any] = None
    pose_correction_side: str = "right"
    anchor_rotation_constraint_enabled: bool = False
    anchor_rotation_min_quality: float = 0.45
    anchor_rotation_max_backend_kabsch_delta_deg: float = 15.0
    min_correspondences: int = 12
    min_inlier_correspondences: int = 8
    inlier_threshold_m: float = 0.02
    trim_quantile_scale: float = 2.5
    min_shape_ratio: float = 0.01
    max_angle_jump_deg: float = 12.0
    temporal_guard_enabled: bool = True
    temporal_guard_max_angle_deg: float = 12.0
    temporal_guard_mode: str = "clamp"
    foundationpose_max_angle_jump_deg: float = 20.0
    max_step_deg: float = 8.0
    stationary_position_epsilon_m: float = 0.004
    stationary_max_angle_deg: float = 4.0
    small_rotation_epsilon_deg: float = 2.0
    ema_alpha: float = 0.6
    min_pose_quality: float = 0.45


_INTEGER_DEFAULTS = {
    "init_refine_iter": 5,
    "track_refine_iter": 2,
    "debug": 0,
    "min_correspondences": 12,
    "min_inlier_correspondences": 8,
}
_BOOLEAN_DEFAULTS = {
    "force_recompute": False,
    "foundationpose_enabled": False,
    "foundationpose_fallback": False,
    "kabsch_enabled": True,
    "use_config_init_pose": False,
    "foundationpose_force_register_frame0": False,
    "anchor_rotation_constraint_enabled": False,
    "temporal_guard_enabled": True,
}
_FLOAT_DEFAULTS = {
    "anchor_rotation_min_quality": 0.45,
    "anchor_rotation_max_backend_kabsch_delta_deg": 15.0,
    "inlier_threshold_m": 0.02,
    "trim_quantile_scale": 2.5,
    "min_shape_ratio": 0.01,
    "max_angle_jump_deg": 12.0,
    "foundationpose_max_angle_jump_deg": 20.0,
    "max_step_deg": 8.0,
    "stationary_position_epsilon_m": 0.004,
    "stationary_max_angle_deg": 4.0,
    "small_rotation_epsilon_deg": 2.0,
    "ema_alpha": 0.6,
    "min_pose_quality": 0.45,
}


def _read_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"pose config must be a JSON object: {path}")
    return payload


def _default_payload(
    config_path: Optional[str],
) -> tuple[Dict[str, Any], frozenset[str]]:
    defaults = _read_object(DEFAULT_POSE_CONFIG_PATH)
    if not config_path:
        return defaults, frozenset(defaults)
    override = _read_object(Path(config_path))
    defaults.update(override)
    return defaults, frozenset(override)


def default_pose_config_dict(
    *,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    defaults, _ = _default_payload(config_path)
    return defaults


def _coerce_config_values(raw: Dict[str, Any]) -> Dict[str, Any]:
    values: Dict[str, Any] = {
        "backend": str(raw.get("backend") or PoseConfig.backend).strip().lower(),
        "weights_root": str(raw.get("weights_root") or ""),
        "mesh_path": str(raw.get("mesh_path") or ""),
        "model_kwargs": dict(raw.get("model_kwargs") or {}),
        "pose_correction_path": str(raw.get("pose_correction_path") or ""),
        "pose_correction_matrix": raw.get("pose_correction_matrix"),
        "pose_correction_side": str(raw.get("pose_correction_side", "right"))
        .strip()
        .lower(),
        "temporal_guard_mode": str(raw.get("temporal_guard_mode", "clamp"))
        .strip()
        .lower(),
    }
    values.update(
        {
            name: int(raw.get(name, default))
            for name, default in _INTEGER_DEFAULTS.items()
        }
    )
    values.update(
        {
            name: bool(raw.get(name, default))
            for name, default in _BOOLEAN_DEFAULTS.items()
        }
    )
    values.update(
        {
            name: float(raw.get(name, default))
            for name, default in _FLOAT_DEFAULTS.items()
        }
    )
    temporal_angle = (
        raw["temporal_guard_max_angle_deg"]
        if "temporal_guard_max_angle_deg" in raw
        else values["max_angle_jump_deg"]
    )
    values["temporal_guard_max_angle_deg"] = float(temporal_angle)
    return values


def _validate(config: PoseConfig) -> None:
    checks = (
        (
            pose_backend_identity_is_known(config.backend),
            f"Unsupported pose backend: {config.backend}",
        ),
        (
            config.init_refine_iter > 0,
            "pose.init_refine_iter must be > 0",
        ),
        (
            config.track_refine_iter > 0,
            "pose.track_refine_iter must be > 0",
        ),
        (
            config.min_correspondences >= 3,
            "pose.min_correspondences must be >= 3",
        ),
        (
            config.min_inlier_correspondences >= 3,
            "pose.min_inlier_correspondences must be >= 3",
        ),
        (
            config.min_inlier_correspondences <= config.min_correspondences,
            ("pose.min_inlier_correspondences cannot exceed pose.min_correspondences"),
        ),
        (
            config.inlier_threshold_m > 0,
            "pose.inlier_threshold_m must be > 0",
        ),
        (
            config.trim_quantile_scale >= 1,
            "pose.trim_quantile_scale must be >= 1",
        ),
        (
            config.min_shape_ratio >= 0,
            "pose.min_shape_ratio must be >= 0",
        ),
        (
            config.pose_correction_side in {"right", "left"},
            "pose.pose_correction_side must be 'right' or 'left'",
        ),
        (
            0 <= config.anchor_rotation_min_quality <= 1,
            "pose.anchor_rotation_min_quality must be in [0, 1]",
        ),
        (
            config.anchor_rotation_max_backend_kabsch_delta_deg > 0,
            ("pose.anchor_rotation_max_backend_kabsch_delta_deg must be > 0"),
        ),
        (
            config.max_angle_jump_deg > 0,
            "pose.max_angle_jump_deg must be > 0",
        ),
        (
            config.temporal_guard_max_angle_deg > 0,
            "pose.temporal_guard_max_angle_deg must be > 0",
        ),
        (
            config.temporal_guard_mode in {"clamp", "freeze"},
            "pose.temporal_guard_mode must be 'clamp' or 'freeze'",
        ),
        (
            config.foundationpose_max_angle_jump_deg > 0,
            "pose.foundationpose_max_angle_jump_deg must be > 0",
        ),
        (
            config.max_step_deg > 0,
            "pose.max_step_deg must be > 0",
        ),
        (
            config.stationary_position_epsilon_m >= 0,
            "pose.stationary_position_epsilon_m must be >= 0",
        ),
        (
            config.stationary_max_angle_deg > 0,
            "pose.stationary_max_angle_deg must be > 0",
        ),
        (
            config.small_rotation_epsilon_deg >= 0,
            "pose.small_rotation_epsilon_deg must be >= 0",
        ),
        (
            0 < config.ema_alpha <= 1,
            "pose.ema_alpha must be in (0, 1]",
        ),
        (
            0 <= config.min_pose_quality <= 1,
            "pose.min_pose_quality must be in [0, 1]",
        ),
    )
    for accepted, message in checks:
        if not accepted:
            raise ValueError(message)
    if (
        config.backend == "foundationpose"
        and not config.kabsch_enabled
        and not config.foundationpose_enabled
        and not config.foundationpose_fallback
    ):
        raise ValueError("foundationpose backend must enable at least one solver path.")
    if config.backend == POINTCLOUD_KABSCH_BACKEND and not config.kabsch_enabled:
        raise ValueError("pointcloud_kabsch backend requires pose.kabsch_enabled=true.")


def load_pose_config(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    default_config_path: Optional[str] = None,
) -> PoseConfig:
    raw, explicit_default_fields = _default_payload(default_config_path)
    overrides = dict(cfg or {})
    raw.update(overrides)
    if (
        "temporal_guard_max_angle_deg" not in explicit_default_fields
        and "temporal_guard_max_angle_deg" not in overrides
    ):
        raw.pop("temporal_guard_max_angle_deg", None)
    config = PoseConfig(**_coerce_config_values(raw))
    _validate(config)
    return config


def pose_config_to_dict(config: PoseConfig) -> Dict[str, Any]:
    return asdict(config)


def effective_pose_backend(
    config: PoseConfig | Dict[str, Any],
) -> str:
    """Return the solver that the normalized pose policy will execute.

    Historical bench configs label the backend ``foundationpose`` while
    disabling both FoundationPose execution flags and enabling Kabsch.  Those
    files remain untouched; this compatibility resolver reports the truthful
    point-cloud solver identity used by that policy.
    """

    policy = load_pose_config(config) if isinstance(config, dict) else config
    requested = str(policy.backend or POINTCLOUD_KABSCH_BACKEND).strip().lower()
    if requested == POINTCLOUD_KABSCH_BACKEND:
        return POINTCLOUD_KABSCH_BACKEND
    if (
        requested == "foundationpose"
        and not policy.foundationpose_enabled
        and not policy.foundationpose_fallback
        and policy.kabsch_enabled
    ):
        return POINTCLOUD_KABSCH_BACKEND
    return requested


__all__ = [
    "DEFAULT_POSE_CONFIG_PATH",
    "EXTERNAL_POSE_SELECTION_PREFIX",
    "MODEL_POSE_BACKENDS",
    "POINTCLOUD_KABSCH_BACKEND",
    "SUPPORTED_POSE_BACKENDS",
    "PoseConfig",
    "default_pose_config_dict",
    "effective_pose_backend",
    "external_pose_backend_id",
    "external_pose_selection",
    "load_pose_config",
    "pose_config_to_dict",
    "pose_backend_identity_is_known",
]
