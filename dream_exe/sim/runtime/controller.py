"""Simulator-side controller configuration and action-limit helpers."""

from __future__ import annotations

import json
from collections.abc import Callable
from typing import Any, Dict, Optional, Tuple

import numpy as np


def _nested_config_value(
    config: Dict[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    current: Any = config
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def robot_name_from_config(config: Dict[str, Any]) -> str:
    """Resolve the robot identity with the current saved-config precedence."""

    robot = _nested_config_value(config, "raw.robot_names", "")
    if not robot:
        robot = _nested_config_value(
            config,
            "raw.env_kwargs.robots",
            "",
        )
    if not robot:
        robot = _nested_config_value(
            config,
            "raw.env_kwargs.robot_names",
            "",
        )
    if not robot:
        robot = _nested_config_value(config, "robot_names", "")
    if isinstance(robot, (list, tuple)) and robot:
        return str(robot[0])
    return str(robot)


def tcp_site_name_from_config(
    config: Dict[str, Any],
    override: str = "",
) -> Optional[str]:
    """Resolve the current TCP site name from saved environment metadata."""

    if override:
        return override
    value = _nested_config_value(
        config,
        "raw.eef.tcp_site_name",
        None,
    )
    if value:
        return str(value)
    value = _nested_config_value(
        config,
        "derived.eef.robot_eef_site_name",
        None,
    )
    return str(value) if value else None


def controller_reference_site_name_from_config(
    config: Dict[str, Any],
    override: str = "",
) -> Optional[str]:
    """Resolve the current controller reference site with TCP fallback."""

    if override:
        return override
    value = _nested_config_value(
        config,
        "raw.eef.controller_ref_site_name",
        None,
    )
    if value:
        return str(value)
    value = _nested_config_value(
        config,
        "derived.eef.controller_ref_site_name",
        None,
    )
    if value:
        return str(value)
    return tcp_site_name_from_config(config)


def load_default_controller_config(
    robot_name: str,
    controller: Optional[str] = None,
) -> Dict[str, Any]:
    """Load a RoboSuite controller config across current supported APIs."""

    try:
        from robosuite.controllers import load_composite_controller_config

        config = load_composite_controller_config(
            controller=controller,
            robot=robot_name,
        )
        if config is None:
            raise RuntimeError("load_composite_controller_config returned None")
        return config
    except Exception:
        pass

    try:
        from robosuite.controllers import load_controller_config

        return load_controller_config(default_controller=controller or "OSC_POSE")
    except Exception as exc:
        raise ImportError(f"Cannot load controller config from robosuite APIs: {exc}")


def set_arm_controller(
    composite_cfg: Dict[str, Any],
    controller_type: str,
    policy_hz: int,
    ref_site_name: str,
    input_ref_frame: str = "world",
    use_ori: bool = True,
    arm_name: str = "right",
) -> Dict[str, Any]:
    """Edit only the selected arm block using current compatibility rules."""

    config = json.loads(json.dumps(composite_cfg))
    body_parts = config.get("body_parts", {})
    if not isinstance(body_parts, dict):
        raise ValueError("Invalid composite cfg: missing body_parts dict")

    arm_config = None
    if "arms" in body_parts and isinstance(body_parts["arms"], dict):
        arms = body_parts["arms"]
        if arm_name in arms and isinstance(arms[arm_name], dict):
            arm_config = arms[arm_name]

    if (
        arm_config is None
        and arm_name in body_parts
        and isinstance(body_parts[arm_name], dict)
    ):
        arm_config = body_parts[arm_name]

    if arm_config is None:
        raise KeyError(
            f"Cannot find arm config for arm_name='{arm_name}'. "
            f"Keys={list(body_parts.keys())}"
        )

    arm_config["type"] = controller_type
    arm_config["policy_freq"] = policy_hz
    arm_config["control_freq"] = policy_hz

    if isinstance(ref_site_name, str) and len(ref_site_name) > 0:
        arm_config["ref_name"] = ref_site_name

    if controller_type.startswith("OSC_"):
        position_only = controller_type == "OSC_POSITION"
        arm_config["input_type"] = "delta"
        arm_config["input_ref_frame"] = input_ref_frame
        arm_config["control_ori"] = (controller_type != "OSC_POSITION") and bool(
            use_ori
        )
        arm_config["input_min"] = -1.0
        arm_config["input_max"] = 1.0

        def _slice_three(value: Any) -> Any:
            if isinstance(value, (list, tuple)) and len(value) == 6:
                return list(value[:3])
            if isinstance(value, (list, tuple)) and len(value) == 3:
                return list(value)
            return value

        if position_only:
            for key in (
                "output_max",
                "output_min",
                "output_max_delta",
                "output_min_delta",
            ):
                if key in arm_config:
                    arm_config[key] = _slice_three(arm_config[key])

    elif controller_type == "IK_POSE":
        arm_config["control_delta"] = True

    return config


def get_ref_site_name_from_controller_cfg(
    controller_cfg: Dict[str, Any],
    arm_name: str = "right",
) -> Optional[str]:
    """Read a scalar ref site from nested or flattened controller schemas."""

    body_parts = controller_cfg.get("body_parts", {})
    if not isinstance(body_parts, dict):
        return None

    if (
        "arms" in body_parts
        and isinstance(body_parts["arms"], dict)
        and arm_name in body_parts["arms"]
    ):
        arm_config = body_parts["arms"][arm_name]
        ref = arm_config.get("ref_name", None)
        if isinstance(ref, str):
            return ref
        if isinstance(ref, list) and len(ref) > 0:
            return ref[0]

    if arm_name in body_parts and isinstance(
        body_parts[arm_name],
        dict,
    ):
        ref = body_parts[arm_name].get("ref_name", None)
        if isinstance(ref, str):
            return ref
        if isinstance(ref, list) and len(ref) > 0:
            return ref[0]

    return None


def pre_step_max_from_controller_cfg(
    controller_cfg: dict,
    default: float = 0.05,
) -> float:
    """Return the current minimum xyz output bound for horizon planning."""

    body_parts = controller_cfg.get("body_parts", {})
    arm_config = None

    if (
        isinstance(body_parts, dict)
        and "arms" in body_parts
        and isinstance(body_parts["arms"], dict)
        and "right" in body_parts["arms"]
    ):
        arm_config = body_parts["arms"]["right"]
    if (
        arm_config is None
        and isinstance(body_parts, dict)
        and "right" in body_parts
        and isinstance(body_parts["right"], dict)
    ):
        arm_config = body_parts["right"]

    if not isinstance(arm_config, dict):
        return float(default)

    output_max = arm_config.get("output_max", None)
    if output_max is None:
        return float(default)

    output_max = np.asarray(output_max, np.float64).reshape(-1)
    return (
        float(np.min(np.abs(output_max[:3])))
        if output_max.size >= 3
        else float(default)
    )


def read_step_clips(
    part_ctrl: Any,
    controller_name: str,
    use_ori: bool,
):
    """Resolve the current positional and rotational per-step clip values."""

    output_max = getattr(part_ctrl, "output_max", None)
    if output_max is None:
        output_max = getattr(part_ctrl, "_output_max", None)
    output_max = (
        np.asarray(output_max, np.float64).reshape(-1)
        if output_max is not None
        else None
    )

    if output_max is None or output_max.size < 3:
        output_max = np.array([0.05, 0.05, 0.05], np.float64)

    step_clip_position = np.abs(output_max[:3]) * 0.98
    step_clip_orientation = None
    if controller_name != "OSC_POSITION" and bool(use_ori) and output_max.size >= 6:
        step_clip_orientation = np.abs(output_max[3:6]) * 0.98
    return output_max, step_clip_position, step_clip_orientation


def _arm_config(
    controller_config: Dict[str, Any],
    arm_name: str = "right",
) -> Dict[str, Any]:
    body_parts = dict(controller_config.get("body_parts", {}) or {})
    arms = body_parts.get("arms", None)
    if isinstance(arms, dict) and isinstance(arms.get(arm_name, None), dict):
        return dict(arms[arm_name])
    if isinstance(body_parts.get(arm_name, None), dict):
        return dict(body_parts[arm_name])
    return {}


def resolve_action_step_budgets(
    *,
    cfg: Dict[str, Any],
    controller_name: str,
    policy_hz: int,
    reference_frame: str,
    want_orientation: bool,
    controller_config_loader: Callable[..., Dict[str, Any]] = (
        load_default_controller_config
    ),
) -> Tuple[float, float]:
    """Resolve simulator controller limits for pure action construction.

    The returned ``(translation_m, rotation_rad)`` tuple is the explicit
    cross-domain value accepted by
    :func:`dream_exe.video2traj.build_action`.  This module never imports the
    algorithm package, keeping dependency direction one-way at composition
    time.
    """

    robot_name = robot_name_from_config(cfg)
    base_config = controller_config_loader(robot_name=robot_name)
    reference_site = (
        controller_reference_site_name_from_config(cfg)
        or tcp_site_name_from_config(cfg)
        or ""
    )
    controller_config = set_arm_controller(
        base_config,
        controller_type=str(controller_name),
        policy_hz=int(policy_hz),
        ref_site_name=reference_site,
        input_ref_frame=str(reference_frame),
        use_ori=bool(want_orientation),
        arm_name="right",
    )

    translation_budget = float(pre_step_max_from_controller_cfg(controller_config))
    rotation_budget = 0.0
    if bool(want_orientation) and str(controller_name) != "OSC_POSITION":
        arm_config = _arm_config(controller_config)
        output_max = np.asarray(
            arm_config.get("output_max", []),
            dtype=np.float64,
        ).reshape(-1)
        if output_max.size >= 6:
            rotation_budget = float(np.min(np.abs(output_max[3:6])))
        if rotation_budget <= 0.0:
            rotation_budget = 0.1
    return (
        max(translation_budget, 1e-6),
        max(rotation_budget, 0.0),
    )


__all__ = [
    "controller_reference_site_name_from_config",
    "get_ref_site_name_from_controller_cfg",
    "load_default_controller_config",
    "pre_step_max_from_controller_cfg",
    "read_step_clips",
    "resolve_action_step_budgets",
    "robot_name_from_config",
    "set_arm_controller",
    "tcp_site_name_from_config",
]
