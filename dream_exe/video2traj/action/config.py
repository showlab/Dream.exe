"""Configuration values for simulator-independent action planning."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_ACTION_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "action.default.json"
)


@dataclass(frozen=True)
class ActionConfig:
    """Normalized controls for producing an ``action`` plan."""

    enabled: bool = True
    eef_key: str = "eef_controller"
    obj_key: str = "obj_visual_center"
    reference_frame: str = "base"
    controller: str = "OSC_POSE"
    policy_hz: int = 20
    translation_step_budget_m: Optional[float] = None
    grasped_translation_step_budget_m: Optional[float] = 0.012
    rotation_step_budget_rad: float = 0.0
    zero_motion_epsilon_m: float = 5e-4
    max_motion_steps_per_segment: int = 16
    gripper_actuation_mode: str = "serial"
    emit_initial_noop_step: bool = True
    embed_gripper_settle_steps: bool = True
    settle_steps_after_close: int = 2
    settle_steps_after_open: int = 2
    insert_close_completion_gate: bool = True
    insert_open_completion_gate: bool = True
    force_gripper_completion_gates: bool = True
    enforce_close_before_object_motion: bool = False
    object_motion_onset_threshold_m: float = 0.004
    object_motion_onset_min_run: int = 2
    close_lead_frames_before_object_motion: int = 1
    max_eef_object_distance_for_motion_close_m: Optional[float] = 0.12
    max_close_shift_frames_before_object_motion: Optional[int] = 24
    compress_grasped_motion: bool = False
    grasped_motion_stride: int = 1
    grasped_motion_keep_event_neighbors: int = 1
    rotation_delta_guard_enabled: bool = True
    rotation_delta_guard_max_rad: float = 0.21
    grasp_rotation_guard_enabled: bool = True
    grasp_rotation_guard_max_rad: float = 0.21
    grasp_rotation_guard_mode: str = "freeze"
    suppress_gripper_z_bounce: bool = True
    post_close_z_bounce_window: int = 18
    post_close_z_bounce_max_drop_m: float = 0.010
    pre_open_z_bounce_window: int = 18
    pre_open_z_bounce_min_rise_m: float = 0.006
    pre_open_z_bounce_max_lift_m: float = 0.035
    z_bounce_epsilon_m: float = 0.001


_BOOL_FIELDS = frozenset(
    {
        "enabled",
        "emit_initial_noop_step",
        "embed_gripper_settle_steps",
        "insert_close_completion_gate",
        "insert_open_completion_gate",
        "force_gripper_completion_gates",
        "enforce_close_before_object_motion",
        "compress_grasped_motion",
        "rotation_delta_guard_enabled",
        "grasp_rotation_guard_enabled",
        "suppress_gripper_z_bounce",
    }
)
_INT_FIELDS = frozenset(
    {
        "policy_hz",
        "max_motion_steps_per_segment",
        "settle_steps_after_close",
        "settle_steps_after_open",
        "object_motion_onset_min_run",
        "close_lead_frames_before_object_motion",
        "grasped_motion_stride",
        "grasped_motion_keep_event_neighbors",
        "post_close_z_bounce_window",
        "pre_open_z_bounce_window",
    }
)
_FLOAT_FIELDS = frozenset(
    {
        "rotation_step_budget_rad",
        "zero_motion_epsilon_m",
        "object_motion_onset_threshold_m",
        "rotation_delta_guard_max_rad",
        "grasp_rotation_guard_max_rad",
        "post_close_z_bounce_max_drop_m",
        "pre_open_z_bounce_min_rise_m",
        "pre_open_z_bounce_max_lift_m",
        "z_bounce_epsilon_m",
    }
)
_OPTIONAL_FLOAT_FIELDS = frozenset(
    {
        "translation_step_budget_m",
        "grasped_translation_step_budget_m",
        "max_eef_object_distance_for_motion_close_m",
    }
)
_OPTIONAL_INT_FIELDS = frozenset({"max_close_shift_frames_before_object_motion"})
_FALLBACK_STRINGS = {
    "eef_key": "eef_controller",
    "obj_key": "obj_visual_center",
    "reference_frame": "base",
    "controller": "OSC_POSE",
    "gripper_actuation_mode": "serial",
    "grasp_rotation_guard_mode": "freeze",
}
_LOWERCASE_STRINGS = frozenset(
    {
        "reference_frame",
        "gripper_actuation_mode",
        "grasp_rotation_guard_mode",
    }
)


def default_action_config_dict(
    *,
    config_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Read the packaged defaults, or an explicitly selected JSON object."""

    path = (
        Path(config_path).expanduser().resolve()
        if config_path
        else DEFAULT_ACTION_CONFIG_PATH
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"action config must be a JSON object: {path}")
    return asdict(_build_config(dict(payload)))


def _convert_value(name: str, value: Any) -> Any:
    if name in _BOOL_FIELDS:
        return bool(value)
    if name in _INT_FIELDS:
        return int(value)
    if name in _FLOAT_FIELDS:
        return float(value)
    if name in _OPTIONAL_FLOAT_FIELDS:
        return None if value is None else float(value)
    if name in _OPTIONAL_INT_FIELDS:
        return None if value is None else int(value)
    fallback = _FALLBACK_STRINGS[name]
    text = str(value or fallback)
    if name in _LOWERCASE_STRINGS:
        return text.strip().lower()
    return text.strip() if name == "controller" else text


def _normalize_values(raw: Dict[str, Any]) -> Dict[str, Any]:
    defaults = asdict(ActionConfig())
    supported = {field.name for field in fields(ActionConfig)}
    defaults.update({name: value for name, value in raw.items() if name in supported})
    return {name: _convert_value(name, value) for name, value in defaults.items()}


def _validate(config: ActionConfig) -> None:
    if config.gripper_actuation_mode not in {"serial", "parallel"}:
        raise ValueError("action.gripper_actuation_mode must be 'serial' or 'parallel'")
    if (
        config.max_close_shift_frames_before_object_motion is not None
        and config.max_close_shift_frames_before_object_motion < 0
    ):
        raise ValueError(
            "action.max_close_shift_frames_before_object_motion must be >= 0 or null"
        )
    if (
        config.max_eef_object_distance_for_motion_close_m is not None
        and config.max_eef_object_distance_for_motion_close_m <= 0
    ):
        raise ValueError(
            "action.max_eef_object_distance_for_motion_close_m must be > 0 or null"
        )
    if config.rotation_delta_guard_max_rad <= 0:
        raise ValueError("action.rotation_delta_guard_max_rad must be > 0")
    if (
        config.grasped_translation_step_budget_m is not None
        and config.grasped_translation_step_budget_m <= 0
    ):
        raise ValueError(
            "action.grasped_translation_step_budget_m must be > 0 when set"
        )
    if config.grasp_rotation_guard_max_rad <= 0:
        raise ValueError("action.grasp_rotation_guard_max_rad must be > 0")
    if config.grasp_rotation_guard_mode not in {"freeze", "clamp"}:
        raise ValueError("action.grasp_rotation_guard_mode must be 'freeze' or 'clamp'")
    nonnegative = (
        ("post_close_z_bounce_window", config.post_close_z_bounce_window),
        (
            "post_close_z_bounce_max_drop_m",
            config.post_close_z_bounce_max_drop_m,
        ),
        ("pre_open_z_bounce_window", config.pre_open_z_bounce_window),
        (
            "pre_open_z_bounce_min_rise_m",
            config.pre_open_z_bounce_min_rise_m,
        ),
        (
            "pre_open_z_bounce_max_lift_m",
            config.pre_open_z_bounce_max_lift_m,
        ),
        ("z_bounce_epsilon_m", config.z_bounce_epsilon_m),
    )
    for name, value in nonnegative:
        if value < 0:
            raise ValueError(f"action.{name} must be >= 0")


def _build_config(raw: Dict[str, Any]) -> ActionConfig:
    config = ActionConfig(**_normalize_values(raw))
    _validate(config)

    if config.gripper_actuation_mode == "parallel":
        return replace(
            config,
            insert_close_completion_gate=False,
            insert_open_completion_gate=False,
            settle_steps_after_close=0,
            settle_steps_after_open=0,
        )
    if config.force_gripper_completion_gates:
        return replace(
            config,
            insert_close_completion_gate=True,
            insert_open_completion_gate=True,
            settle_steps_after_close=max(
                2,
                config.settle_steps_after_close,
            ),
            settle_steps_after_open=max(
                2,
                config.settle_steps_after_open,
            ),
        )
    return config


def load_action_config(
    cfg: Optional[Dict[str, Any]] = None,
    *,
    default_config_path: Optional[str] = None,
) -> ActionConfig:
    """Merge overrides with defaults and return an immutable configuration."""

    raw = default_action_config_dict(config_path=default_config_path)
    raw.update(cfg or {})
    return _build_config(raw)


def action_config_to_dict(config: ActionConfig) -> Dict[str, Any]:
    """Return a detached JSON-compatible representation."""

    return asdict(config)


__all__ = [
    "ActionConfig",
    "DEFAULT_ACTION_CONFIG_PATH",
    "action_config_to_dict",
    "default_action_config_dict",
    "load_action_config",
]
