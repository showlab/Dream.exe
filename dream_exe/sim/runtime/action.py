"""Environment-side normalization and packing of simulator robot actions."""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np


def normalize_arm_action(part_ctrl: Any, raw: np.ndarray) -> np.ndarray:
    """Convert raw meter/radian deltas into the current normalized action."""

    raw = np.asarray(raw, dtype=np.float64).reshape(-1)

    output_max = getattr(part_ctrl, "output_max", None)
    if output_max is None:
        output_max = getattr(part_ctrl, "_output_max", None)

    if output_max is not None:
        output_max = np.asarray(output_max, dtype=np.float64).reshape(-1)
        denominator = np.maximum(
            np.abs(output_max[: raw.shape[0]]),
            1e-9,
        )
        action = raw / denominator
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    ik_position_limit = getattr(part_ctrl, "ik_pos_limit", None)
    ik_orientation_limit = getattr(part_ctrl, "ik_ori_limit", None)
    if ik_position_limit is not None or ik_orientation_limit is not None:
        action = raw.copy()
        if action.shape[0] >= 3 and ik_position_limit is not None:
            action[:3] = action[:3] / max(
                float(ik_position_limit),
                1e-9,
            )
        if action.shape[0] >= 6 and ik_orientation_limit is not None:
            action[3:6] = action[3:6] / max(
                float(ik_orientation_limit),
                1e-9,
            )
        return np.clip(action, -1.0, 1.0).astype(np.float32)

    return np.clip(raw, -1.0, 1.0).astype(np.float32)


def infer_arm_and_gripper_parts(
    robot: Any,
) -> Tuple[str, Optional[str], Dict[str, Any]]:
    """Infer composite-controller part names using current fallback rules."""

    metadata: Dict[str, Any] = {}
    arm_part = "right"
    gripper_part = None

    composite_controller = getattr(
        robot,
        "composite_controller",
        None,
    ) or getattr(robot, "controller", None)
    part_controllers = (
        getattr(composite_controller, "part_controllers", None)
        if composite_controller is not None
        else None
    )

    if hasattr(robot, "arms") and len(robot.arms) > 0:
        arm_part = str(robot.arms[0])

    if part_controllers is not None:
        keys = (
            list(part_controllers.keys()) if hasattr(part_controllers, "keys") else []
        )
        metadata["part_keys"] = [str(key) for key in keys]

        for key in keys:
            key_lower = str(key).lower()
            if "gripper" in key_lower or "grip" in key_lower:
                gripper_part = str(key)
                break

        if (
            hasattr(part_controllers, "get")
            and part_controllers.get(arm_part, None) is None
        ):
            for key in keys:
                key_lower = str(key).lower()
                if "gripper" not in key_lower and "grip" not in key_lower:
                    arm_part = str(key)
                    metadata["arm_part_fallback"] = arm_part
                    break

    return str(arm_part), gripper_part, metadata


def pack_action(
    robot: Any,
    arm_part: str,
    arm_action: np.ndarray,
    gripper_part: Optional[str],
    gripper_cmd: float,
) -> np.ndarray:
    """Pack an arm command and optional gripper command for ``env.step``."""

    arm_action = np.asarray(arm_action, dtype=np.float32).reshape(-1)

    if hasattr(robot, "create_action_vector"):
        action_dict: Dict[str, np.ndarray] = {arm_part: arm_action}
        if gripper_part is not None:
            action_dict[gripper_part] = np.asarray(
                [gripper_cmd],
                dtype=np.float32,
            )
        vector = robot.create_action_vector(action_dict).astype(np.float32)

        composite_controller = getattr(
            robot,
            "composite_controller",
            None,
        ) or getattr(robot, "controller", None)
        split_indexes = (
            getattr(
                composite_controller,
                "_action_split_indexes",
                None,
            )
            if composite_controller is not None
            else None
        )
        if isinstance(split_indexes, dict):
            keep = {str(arm_part)}
            if gripper_part is not None:
                keep.add(str(gripper_part))
            for part_name, bounds in split_indexes.items():
                if len(bounds) != 2:
                    continue
                start, end = int(bounds[0]), int(bounds[1])
                if str(part_name) in keep:
                    continue
                if 0 <= start <= end <= vector.shape[0]:
                    vector[start:end] = 0.0
        return vector

    if hasattr(robot, "action_dim"):
        action_dimension = int(robot.action_dim)
        if action_dimension == arm_action.shape[0]:
            return arm_action.astype(np.float32)
        if action_dimension == arm_action.shape[0] + 1:
            return np.concatenate(
                [
                    arm_action,
                    np.asarray([gripper_cmd], np.float32),
                ],
                axis=0,
            )
        output = np.zeros((action_dimension,), dtype=np.float32)
        output[: arm_action.shape[0]] = arm_action
        output[-1] = float(gripper_cmd)
        return output

    return arm_action.astype(np.float32)


__all__ = [
    "infer_arm_and_gripper_parts",
    "normalize_arm_action",
    "pack_action",
]
