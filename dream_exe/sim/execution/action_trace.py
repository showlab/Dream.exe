"""Explicit-runtime action-trace execution.

This module extracts the concrete action loop from the current action
executor.  The caller supplies an already-restored environment, parsed action
payload, simulator metadata, execution settings, and an explicit output
directory.  No UID, bench, dataset, or config-root discovery occurs here.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
import json
import os
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

from dream_exe.artifacts.io import update_exec_assets_manifest
from dream_exe.artifacts.layout import execution_artifact_paths
from dream_exe.transforms import (
    parse_X_wb,
    world_to_base_R,
    world_to_base_point,
)

from ..runtime.controller import (
    controller_reference_site_name_from_config,
    read_step_clips,
    tcp_site_name_from_config,
)
from ..runtime.contacts import robot_non_support_contacts
from ..runtime.action import (
    infer_arm_and_gripper_parts,
    normalize_arm_action,
    pack_action,
)


@dataclass(frozen=True)
class ActionStepTrace:
    exec_step_index: int
    plan_step_index: int | None
    kind: str
    target_checkpoint_index: int
    target_frame: int
    raw_action: np.ndarray
    normalized_action: np.ndarray
    gripper_cmd: float
    action_vector: np.ndarray
    is_checkpoint_boundary: bool


class VideoRecorder:
    """Current frame capture behavior with lazy video encoding."""

    def __init__(
        self,
        enabled: bool,
        env: Any,
        camera_name: str,
        frame_size: tuple[int, int],
    ) -> None:
        self.enabled = bool(enabled)
        self.env = env
        self.camera_name = camera_name
        self.cam_name = camera_name
        self.frame_size = frame_size
        self.frames: list[np.ndarray] = []

    def grab(self) -> None:
        if not self.enabled:
            return
        image = self.env.sim.render(
            camera_name=self.cam_name,
            width=self.frame_size[0],
            height=self.frame_size[1],
            depth=False,
        )
        self.frames.append(np.flipud(image))

    def save(self, output_path: str, fps: int) -> None:
        if not self.enabled or len(self.frames) == 0:
            return
        import imageio

        imageio.mimsave(
            output_path,
            self.frames,
            fps=int(fps),
        )


def _quaternion_wxyz_from_rotation(
    rotation: np.ndarray,
) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(
        np.asarray(rotation, dtype=float).reshape(3, 3)
    ).as_quat()
    return np.asarray(
        [
            quaternion_xyzw[3],
            quaternion_xyzw[0],
            quaternion_xyzw[1],
            quaternion_xyzw[2],
        ],
        dtype=np.float64,
    )


def _target_pose_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    reference_frame: str,
    use_ori: bool,
    orientation_target_mode: str = "absolute",
    orientation_anchor_R_ref: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    key = "eef_target_ref_6d" if reference_frame == "base" else "eef_target_world_6d"
    values = checkpoint.get(key, None)
    if values is None:
        fallback = checkpoint.get("eef_target_world_6d", None)
        if fallback is None:
            raise KeyError(f"checkpoint missing {key} and fallback world target")
        values = fallback

    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 3:
        raise ValueError(f"checkpoint target has invalid shape: {values}")
    position = array[:3].copy()
    target_rotation = None
    if bool(use_ori) and array.size >= 6:
        rotation_vector = array[3:6]
        if np.all(np.isfinite(rotation_vector)):
            target_rotation = Rotation.from_rotvec(rotation_vector).as_matrix()
            if str(orientation_target_mode) == "relative_to_first":
                if orientation_anchor_R_ref is None:
                    raise ValueError(
                        "orientation_target_mode=relative_to_first "
                        "requires execution-time orientation anchor."
                    )
                target_rotation = np.asarray(
                    target_rotation,
                    dtype=np.float64,
                ).reshape(3, 3) @ np.asarray(
                    orientation_anchor_R_ref,
                    dtype=np.float64,
                ).reshape(3, 3)
    return position, target_rotation


def _pick_gripper_joint_candidates(model: Any) -> list[str]:
    joint_names = getattr(model, "joint_names", None)
    if joint_names is None:
        return []
    output: list[str] = []
    for name in joint_names:
        text = str(name)
        normalized = text.lower()
        if "finger" in normalized or "gripper" in normalized:
            output.append(text)
    return output


def _read_gripper_qpos(
    env: Any,
    robot: Any,
    joint_names: list[str],
) -> np.ndarray | None:
    try:
        gripper_object = getattr(robot, "gripper", None)
        if gripper_object is not None:
            if hasattr(gripper_object, "qpos"):
                qpos = np.asarray(
                    gripper_object.qpos,
                    dtype=np.float64,
                ).reshape(-1)
                if qpos.size > 0 and np.all(np.isfinite(qpos)):
                    return qpos.copy()
            if isinstance(gripper_object, dict):
                values: list[np.ndarray] = []
                for item in gripper_object.values():
                    if hasattr(item, "qpos"):
                        qpos = np.asarray(
                            item.qpos,
                            dtype=np.float64,
                        ).reshape(-1)
                        if qpos.size > 0 and np.all(np.isfinite(qpos)):
                            values.append(qpos.copy())
                if values:
                    return np.concatenate(values, axis=0)
    except Exception:
        pass

    if not joint_names:
        return None
    try:
        model = env.sim.model
        data = env.sim.data
        values: list[float] = []
        for joint_name in joint_names:
            joint_id = model.joint_name2id(joint_name)
            address = int(model.jnt_qposadr[joint_id])
            values.append(float(data.qpos[address]))
        if values:
            array = np.asarray(
                values,
                dtype=np.float64,
            ).reshape(-1)
            if array.size > 0 and np.all(np.isfinite(array)):
                return array.copy()
    except Exception:
        return None
    return None


def _pose_world_payload(
    position_world: np.ndarray,
    rotation_world: np.ndarray,
) -> dict[str, Any]:
    position = np.asarray(
        position_world,
        dtype=np.float64,
    ).reshape(3)
    rotation = np.asarray(
        rotation_world,
        dtype=np.float64,
    ).reshape(3, 3)
    return {
        "pos": position.astype(float).tolist(),
        "R": rotation.astype(float).tolist(),
        "quat_wxyz": _quaternion_wxyz_from_rotation(rotation)
        .reshape(4)
        .astype(float)
        .tolist(),
    }


def _target_world_payload_from_checkpoint(
    checkpoint: Mapping[str, Any],
    *,
    use_ori: bool,
) -> dict[str, Any] | None:
    values = checkpoint.get("eef_target_world_6d", None)
    if values is None:
        return None
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    if array.size < 3 or not np.all(np.isfinite(array[:3])):
        return None
    payload: dict[str, Any] = {"pos": array[:3].astype(float).tolist()}
    if bool(use_ori) and array.size >= 6 and np.all(np.isfinite(array[3:6])):
        rotation = Rotation.from_rotvec(array[3:6]).as_matrix()
        payload["R"] = rotation.astype(float).tolist()
        payload["quat_wxyz"] = (
            _quaternion_wxyz_from_rotation(rotation).reshape(4).astype(float).tolist()
        )
    return payload


def _site_pose_world(
    env: Any,
    site_name: str,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    site_id = env.sim.model.site_name2id(site_name)
    position = env.sim.data.site_xpos[site_id].copy()
    rotation = env.sim.data.site_xmat[site_id].reshape(3, 3).copy()
    return (
        position,
        rotation,
        {
            "source": "site",
            "site_name": site_name,
        },
    )


def _robot_eef_site_name(robot: Any, sim: Any) -> str | None:
    arm = robot.arms[0] if hasattr(robot, "arms") and len(robot.arms) > 0 else "right"
    site_identifier = getattr(robot, "eef_site_id", None)
    site_id = (
        site_identifier.get(arm, None)
        if isinstance(site_identifier, dict)
        else site_identifier
    )
    site_name_value = getattr(robot, "eef_site_name", None)
    site_name = (
        site_name_value.get(arm, None)
        if isinstance(site_name_value, dict)
        else site_name_value
    )
    if site_name is None and site_id is not None:
        try:
            site_name = sim.model.site_id2name(int(site_id))
        except Exception:
            site_name = None
    return site_name


def current_eef_pose_world(
    env: Any,
    site_name: str | None = None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Read current EEF pose with the current explicit-site/TCP fallback."""

    if site_name is not None and site_name in list(env.sim.model.site_names):
        return _site_pose_world(env, site_name)

    robot = env.robots[0]
    eef_site_name = _robot_eef_site_name(robot, env.sim)
    if eef_site_name is not None:
        position, rotation, _ = _site_pose_world(
            env,
            eef_site_name,
        )
        arm = (
            robot.arms[0] if hasattr(robot, "arms") and len(robot.arms) > 0 else "right"
        )
        return (
            position,
            rotation,
            {
                "source": "robot_eef_site",
                "arm": arm,
                "site_name": eef_site_name,
            },
        )
    return np.zeros(3), np.eye(3), {"source": "fallback"}


def pose_in_reference(
    position_world: np.ndarray,
    rotation_world: np.ndarray | None,
    reference_frame: str,
    rotation_world_base: np.ndarray | None,
    translation_world_base: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None]:
    if reference_frame == "world":
        return position_world, rotation_world
    assert rotation_world_base is not None and translation_world_base is not None
    position_base = world_to_base_point(
        rotation_world_base,
        translation_world_base,
        position_world,
    )
    rotation_base = (
        None
        if rotation_world is None
        else world_to_base_R(
            rotation_world_base,
            rotation_world,
        )
    )
    return position_base, rotation_base


def _execution_sections(
    execution_config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    config = dict(execution_config or {})
    if "execution" in config or "runtime" in config:
        return (
            dict(config.get("execution", {}) or {}),
            dict(config.get("runtime", {}) or {}),
        )
    return config, {}


def _execution_setting(
    execution: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    value = execution.get(key, default)
    return default if value is None else value


def validate_action_trace_preflight(
    action_payload: Mapping[str, Any],
    execution_config: Mapping[str, Any] | None,
) -> None:
    """Validate failures that precede environment setup in the baseline."""

    steps = action_payload.get("steps", None)
    checkpoints = action_payload.get("checkpoints", None)
    if not isinstance(steps, list) or len(steps) == 0:
        raise ValueError("action steps missing or empty")
    if not isinstance(checkpoints, list) or len(checkpoints) == 0:
        raise ValueError("action checkpoints missing or empty")

    execution, _runtime = _execution_sections(execution_config)
    correction_mode = (
        str(
            _execution_setting(
                execution,
                "pose_correction_mode",
                "coupled",
            )
            or "coupled"
        )
        .strip()
        .lower()
    )
    if correction_mode == "position_first":
        correction_mode = "position_dominate"
    if correction_mode not in {"position_dominate", "coupled"}:
        raise ValueError(
            f"Unsupported pose_correction_mode={correction_mode!r}; "
            "expected 'position_dominate' or 'coupled'."
        )

    action_space = dict(action_payload.get("meta", {}).get("action_space", {}) or {})
    orientation_target_mode = (
        str(
            action_space.get(
                "orientation_target_mode",
                "absolute",
            )
            or "absolute"
        )
        .strip()
        .lower()
    )
    if orientation_target_mode not in {
        "absolute",
        "relative_to_first",
    }:
        raise ValueError(
            "Unsupported action_space.orientation_target_mode: "
            f"{orientation_target_mode}"
        )


def _drive_site_name_from_config(
    simulator_config: Mapping[str, Any],
) -> str:
    raw = dict(simulator_config.get("raw", {}) or {})
    raw_eef = dict(raw.get("eef", {}) or {})
    if raw_eef.get("drive_site_name"):
        return str(raw_eef["drive_site_name"])
    derived = dict(simulator_config.get("derived", {}) or {})
    derived_eef = dict(derived.get("eef", {}) or {})
    if derived_eef.get("execution_site_name"):
        return str(derived_eef["execution_site_name"])
    return ""


def warm_start_hold(
    *,
    env: Any,
    robot: Any,
    arm_part: str,
    gripper_part: str | None,
    part_ctrl: Any,
    controller_name: str,
    use_ori: bool,
    gripper: float,
    steps: int,
) -> None:
    if steps <= 0:
        return
    if controller_name == "OSC_POSITION":
        arm_dimension = 3
    else:
        arm_dimension = 6 if bool(use_ori) else 3
    raw_zero = np.zeros(arm_dimension, dtype=np.float64)
    for _ in range(int(steps)):
        arm_action = normalize_arm_action(part_ctrl, raw_zero)
        action_vector = pack_action(
            robot=robot,
            arm_part=arm_part,
            arm_action=arm_action,
            gripper_part=gripper_part,
            gripper_cmd=float(gripper),
        )
        env.step(action_vector)


def execute_action_trace_loop(
    *,
    env: Any,
    robot: Any,
    part_ctrl: Any,
    arm_part: str,
    gripper_part: str | None,
    action_payload: Mapping[str, Any],
    execution_config: Mapping[str, Any] | None = None,
    ref_site_name: str = "",
    rotation_world_base: np.ndarray | None = None,
    translation_world_base: np.ndarray | None = None,
    pose_reader: Callable[..., Any] | None = None,
    frame_capture: Callable[[], Any] | None = None,
    contact_reader: Callable[..., Sequence[Any]] | None = None,
    checkpoint_observer: (Callable[[Any, Mapping[str, Any]], Any] | None) = None,
) -> dict[str, Any]:
    """Execute parsed action steps against one prepared simulator environment."""

    steps = action_payload.get("steps", None)
    checkpoints = action_payload.get("checkpoints", None)
    if not isinstance(steps, list) or len(steps) == 0:
        raise ValueError("action steps missing or empty")
    if not isinstance(checkpoints, list) or len(checkpoints) == 0:
        raise ValueError("action checkpoints missing or empty")

    planner_meta = dict(action_payload.get("meta", {}).get("planner", {}) or {})
    action_space = dict(action_payload.get("meta", {}).get("action_space", {}) or {})
    execution, runtime = _execution_sections(execution_config)

    controller_name = str(
        _execution_setting(
            execution,
            "controller",
            "OSC_POSITION",
        )
    )
    policy_hz = int(
        _execution_setting(
            runtime,
            "policy_hz",
            _execution_setting(execution, "policy_hz", 20),
        )
    )
    render = bool(
        _execution_setting(
            runtime,
            "render",
            _execution_setting(execution, "render", False),
        )
    )
    gripper = float(_execution_setting(execution, "gripper", 0.0))
    position_tolerance = float(_execution_setting(execution, "pos_tol", 5.0e-3))
    orientation_tolerance = float(_execution_setting(execution, "ori_tol", 3.0e-2))
    arm_position_gain = float(_execution_setting(execution, "arm_pos_gain", 2.0))
    arm_orientation_gain = float(_execution_setting(execution, "arm_ori_gain", 1.5))
    maximum_correction_steps = int(
        _execution_setting(
            execution,
            "max_correction_steps",
            3,
        )
    )
    must_reach_min_correction_steps = int(
        _execution_setting(
            execution,
            "must_reach_min_correction_steps",
            3,
        )
    )
    position_threshold = float(
        execution.get(
            "position_dominate_correction_threshold_m",
            0.02,
        )
    )
    position_first_threshold = execution.get(
        "position_first_correction_threshold_m",
        None,
    )
    if position_first_threshold is not None:
        position_threshold = float(position_first_threshold)
    correction_mode = (
        str(
            _execution_setting(
                execution,
                "pose_correction_mode",
                "coupled",
            )
            or "coupled"
        )
        .strip()
        .lower()
    )
    if correction_mode == "position_first":
        correction_mode = "position_dominate"
    if correction_mode not in {"position_dominate", "coupled"}:
        raise ValueError(
            f"Unsupported pose_correction_mode={correction_mode!r}; "
            "expected 'position_dominate' or 'coupled'."
        )

    reference_frame = str(action_space.get("reference_frame", "base") or "base")
    orientation_target_mode = (
        str(
            action_space.get(
                "orientation_target_mode",
                "absolute",
            )
            or "absolute"
        )
        .strip()
        .lower()
    )
    if orientation_target_mode not in {
        "absolute",
        "relative_to_first",
    }:
        raise ValueError(
            "Unsupported action_space.orientation_target_mode: "
            f"{orientation_target_mode}"
        )
    use_orientation = (
        bool(action_space.get("has_orientation", False))
        and controller_name != "OSC_POSITION"
    )

    checkpoint_orientation_skip_frozen = bool(
        planner_meta.get(
            "checkpoint_orientation_skip_frozen",
            True,
        )
    )
    checkpoint_orientation_min_quality = float(
        planner_meta.get(
            "checkpoint_orientation_min_quality",
            0.5,
        )
    )
    close_gate_meta = dict(planner_meta.get("close_completion_gate", {}) or {})
    enable_close_gate = bool(
        _execution_setting(
            execution,
            "enable_close_completion_gate",
            True,
        )
    ) and bool(close_gate_meta.get("enabled", True))
    open_gate_meta = dict(planner_meta.get("open_completion_gate", {}) or {})
    enable_open_gate = bool(
        _execution_setting(
            execution,
            "enable_open_completion_gate",
            True,
        )
    ) and bool(open_gate_meta.get("enabled", False))
    close_gate_min_hold_steps = int(
        _execution_setting(
            execution,
            "close_gate_min_hold_steps",
            2,
        )
    )
    close_gate_max_wait_steps = int(
        _execution_setting(
            execution,
            "close_gate_max_wait_steps",
            60,
        )
    )
    close_gate_qpos_delta_min = float(
        _execution_setting(
            execution,
            "close_gate_qpos_delta_min",
            1.0e-4,
        )
    )
    close_gate_qpos_settle_tol = float(
        _execution_setting(
            execution,
            "close_gate_qpos_settle_tol",
            1.0e-4,
        )
    )
    close_gate_settle_window = int(
        _execution_setting(
            execution,
            "close_gate_settle_window",
            3,
        )
    )
    close_gate_require_non_support_contact = bool(
        _execution_setting(
            execution,
            "close_gate_require_non_support_contact",
            False,
        )
    )
    close_gate_contact_settle_steps = int(
        _execution_setting(
            execution,
            "close_gate_contact_settle_steps",
            2,
        )
    )
    close_gate_failure_policy = (
        str(
            _execution_setting(
                execution,
                "close_gate_failure_policy",
                "continue",
            )
            or "continue"
        )
        .strip()
        .lower()
    )
    if close_gate_failure_policy not in {"continue", "terminate"}:
        close_gate_failure_policy = "continue"

    planner_controller = str(planner_meta.get("controller", "") or "").strip()
    if planner_controller:
        controller_name = planner_controller
        if planner_controller == "OSC_POSITION":
            use_orientation = False
        elif bool(action_space.get("has_orientation", False)):
            use_orientation = True
    if int(planner_meta.get("policy_hz", 0)) > 0:
        policy_hz = int(planner_meta.get("policy_hz"))
    use_orientation = (
        bool(action_space.get("has_orientation", False))
        and controller_name != "OSC_POSITION"
    )
    if not use_orientation and controller_name == "OSC_POSE":
        controller_name = "OSC_POSITION"

    _output_maximum, step_clip_position, step_clip_orientation = read_step_clips(
        part_ctrl,
        controller_name,
        bool(use_orientation),
    )
    if use_orientation and step_clip_orientation is None:
        fallback_orientation_clip = float(
            planner_meta.get(
                "rotation_step_budget_rad",
                0.0,
            )
            or 0.0
        )
        if fallback_orientation_clip <= 0.0:
            fallback_orientation_clip = 0.1
        step_clip_orientation = np.full(
            (3,),
            fallback_orientation_clip,
            dtype=np.float64,
        )

    gripper_joint_candidates = _pick_gripper_joint_candidates(env.sim.model)
    read_pose = current_eef_pose_world if pose_reader is None else pose_reader
    read_contacts = (
        robot_non_support_contacts if contact_reader is None else contact_reader
    )
    grab_frame = (lambda: None) if frame_capture is None else frame_capture

    def current_pose_reference() -> tuple[
        np.ndarray,
        np.ndarray | None,
    ]:
        position_world, rotation_world, _ = read_pose(
            env,
            site_name=ref_site_name,
        )
        position_reference, rotation_reference = pose_in_reference(
            np.asarray(
                position_world,
                dtype=np.float64,
            ).reshape(3),
            np.asarray(
                rotation_world,
                dtype=np.float64,
            ).reshape(3, 3),
            reference_frame,
            rotation_world_base,
            translation_world_base,
        )
        return (
            np.asarray(
                position_reference,
                dtype=np.float64,
            ).reshape(3),
            None
            if rotation_reference is None
            else np.asarray(
                rotation_reference,
                dtype=np.float64,
            ).reshape(3, 3),
        )

    total_env_steps = 0
    terminated = False
    action_trace: list[ActionStepTrace] = []
    checkpoint_trace: list[dict[str, Any]] = []
    dense_tcp_trace: list[dict[str, Any]] = []
    close_gate_trace: list[dict[str, Any]] = []
    open_gate_trace: list[dict[str, Any]] = []
    orientation_anchor_R_ref: np.ndarray | None = None

    def step_once(
        *,
        raw_action: np.ndarray,
        gripper_cmd: float,
        kind: str,
        plan_step_index: int | None,
        target_checkpoint_index: int,
        target_frame: int,
        is_checkpoint_boundary: bool,
    ) -> tuple[bool, int]:
        nonlocal total_env_steps
        execution_step_index = len(action_trace)
        arm_action = normalize_arm_action(part_ctrl, raw_action)
        action_vector = pack_action(
            robot=robot,
            arm_part=arm_part,
            arm_action=arm_action,
            gripper_part=gripper_part,
            gripper_cmd=float(gripper_cmd),
        )
        _, _, done, _ = env.step(action_vector)
        total_env_steps += 1
        if render:
            env.render()
        grab_frame()

        try:
            position_world, rotation_world, _ = read_pose(
                env,
                site_name=ref_site_name,
            )
            actual_tcp_world = _pose_world_payload(
                position_world,
                rotation_world,
            )
        except Exception as error:
            actual_tcp_world = {"missing_reason": (f"{type(error).__name__}: {error}")}
        checkpoint = (
            checkpoints[target_checkpoint_index]
            if 0 <= int(target_checkpoint_index) < len(checkpoints)
            else {}
        )
        dense_tcp_trace.append(
            {
                "exec_step_index": int(execution_step_index),
                "timestamp_s": (
                    float(execution_step_index) / float(max(int(policy_hz), 1))
                ),
                "plan_step_index": (
                    None if plan_step_index is None else int(plan_step_index)
                ),
                "kind": str(kind),
                "target_checkpoint_index": int(target_checkpoint_index),
                "target_frame": int(target_frame),
                "is_checkpoint_boundary": bool(is_checkpoint_boundary),
                "reference_frame": "world",
                "controller_reference_frame": str(reference_frame),
                "actual_tcp_world": actual_tcp_world,
                "target_tcp_world": (
                    _target_world_payload_from_checkpoint(
                        checkpoint,
                        use_ori=bool(use_orientation),
                    )
                ),
                "gripper_cmd": float(gripper_cmd),
            }
        )
        action_trace.append(
            ActionStepTrace(
                exec_step_index=int(execution_step_index),
                plan_step_index=(
                    None if plan_step_index is None else int(plan_step_index)
                ),
                kind=str(kind),
                target_checkpoint_index=int(target_checkpoint_index),
                target_frame=int(target_frame),
                raw_action=np.asarray(
                    raw_action,
                    dtype=np.float64,
                ).copy(),
                normalized_action=np.asarray(
                    arm_action,
                    dtype=np.float64,
                ).copy(),
                gripper_cmd=float(gripper_cmd),
                action_vector=np.asarray(
                    action_vector,
                    dtype=np.float64,
                ).copy(),
                is_checkpoint_boundary=bool(is_checkpoint_boundary),
            )
        )
        return bool(done), int(execution_step_index)

    def execute_gripper_gate(
        *,
        plan_step_index: int | None,
        target_checkpoint_index: int,
        target_frame: int,
        gripper_cmd: float,
        gate_kind: str,
        gate_enabled: bool,
        require_non_support_contact: bool,
    ) -> bool:
        nonlocal terminated
        arm_dimension = (
            3 if controller_name == "OSC_POSITION" else (6 if use_orientation else 3)
        )
        raw_zero = np.zeros(
            (arm_dimension,),
            dtype=np.float64,
        )
        minimum_hold = int(max(0, close_gate_min_hold_steps))
        maximum_wait = int(
            max(
                minimum_hold,
                close_gate_max_wait_steps,
                60,
            )
        )
        settle_window = int(max(1, close_gate_settle_window))
        qpos_settle_tolerance = float(max(close_gate_qpos_settle_tol, 1.0e-4))
        contact_settle_steps = int(max(1, close_gate_contact_settle_steps))
        qpos_start = _read_gripper_qpos(
            env,
            robot,
            gripper_joint_candidates,
        )
        qpos_previous = (
            None
            if qpos_start is None
            else np.asarray(
                qpos_start,
                dtype=np.float64,
            ).reshape(-1)
        )
        stable_count = 0
        contact_stable_count = 0
        wait_steps = 0
        completed = False
        completion_mode = "fixed_hold_fallback"
        qpos_ready = False
        contact_ready = not bool(require_non_support_contact)
        last_contact_pairs: list[Any] = []

        while wait_steps < maximum_wait:
            terminated, _execution_step_index = step_once(
                raw_action=raw_zero,
                gripper_cmd=gripper_cmd,
                kind=str(gate_kind),
                plan_step_index=plan_step_index,
                target_checkpoint_index=target_checkpoint_index,
                target_frame=target_frame,
                is_checkpoint_boundary=False,
            )
            wait_steps += 1
            if terminated:
                break

            if require_non_support_contact:
                last_contact_pairs = list(read_contacts(env, topk=5))
                if last_contact_pairs:
                    contact_stable_count += 1
                else:
                    contact_stable_count = 0
                contact_ready = bool(contact_stable_count >= contact_settle_steps)

            qpos_current = _read_gripper_qpos(
                env,
                robot,
                gripper_joint_candidates,
            )
            if qpos_start is None or qpos_current is None:
                qpos_ready = bool(wait_steps >= minimum_hold)
                if qpos_ready and contact_ready:
                    completed = True
                    completion_mode = "fixed_hold_fallback"
                    break
                continue

            completion_mode = "qpos_settle"
            current = np.asarray(
                qpos_current,
                dtype=np.float64,
            ).reshape(-1)
            step_delta = (
                float(np.linalg.norm(current - qpos_previous))
                if (qpos_previous is not None and qpos_previous.shape == current.shape)
                else float("inf")
            )
            total_delta = (
                float(np.linalg.norm(current - qpos_start))
                if np.asarray(qpos_start).shape == current.shape
                else float("inf")
            )
            if np.isfinite(step_delta) and step_delta <= qpos_settle_tolerance:
                stable_count += 1
            else:
                stable_count = 0

            moved_enough = np.isfinite(total_delta) and (
                total_delta >= close_gate_qpos_delta_min
            )
            qpos_ready = bool(
                wait_steps >= minimum_hold
                and stable_count >= settle_window
                and (moved_enough or wait_steps >= (minimum_hold + settle_window))
            )
            if qpos_ready and contact_ready:
                completed = True
                break
            qpos_previous = current

        gate_trace = (
            close_gate_trace if str(gate_kind) == "close_only_gate" else open_gate_trace
        )
        gate_trace.append(
            {
                "kind": str(gate_kind),
                "plan_step_index": (
                    None if plan_step_index is None else int(plan_step_index)
                ),
                "target_checkpoint_index": int(target_checkpoint_index),
                "target_frame": int(target_frame),
                "gripper_cmd": float(gripper_cmd),
                "enabled": bool(gate_enabled),
                "mode": str(completion_mode),
                "wait_steps": int(wait_steps),
                "completed": bool(completed),
                "terminated": bool(terminated),
                "qpos_available": bool(qpos_start is not None),
                "qpos_ready": bool(qpos_ready),
                "require_non_support_contact": bool(require_non_support_contact),
                "contact_settle_steps": int(contact_settle_steps),
                "contact_stable_count": int(contact_stable_count),
                "contact_ready": bool(contact_ready),
                "contact_pairs_last": list(last_contact_pairs),
            }
        )
        return bool(completed)

    def evaluate_checkpoint(
        checkpoint_index: int,
        *,
        target_frame: int,
        gripper_cmd: float,
        boundary_plan_step_index: int | None,
        boundary_exec_step_index: int | None,
    ) -> bool:
        nonlocal terminated
        nonlocal orientation_anchor_R_ref
        checkpoint = checkpoints[checkpoint_index]
        if (
            use_orientation
            and orientation_target_mode == "relative_to_first"
            and orientation_anchor_R_ref is None
        ):
            _anchor_position, anchor_rotation = current_pose_reference()
            if anchor_rotation is None:
                raise RuntimeError(
                    "Failed to read current EEF orientation for "
                    "relative_to_first checkpoint anchoring."
                )
            orientation_anchor_R_ref = np.asarray(
                anchor_rotation,
                dtype=np.float64,
            ).reshape(3, 3)

        target_reference, target_rotation_reference = _target_pose_from_checkpoint(
            checkpoint,
            reference_frame=reference_frame,
            use_ori=bool(use_orientation),
            orientation_target_mode=orientation_target_mode,
            orientation_anchor_R_ref=orientation_anchor_R_ref,
        )
        checkpoint_pose_valid_raw = checkpoint.get(
            "eef_pose_valid",
            True,
        )
        checkpoint_pose_valid = (
            bool(checkpoint_pose_valid_raw)
            if checkpoint_pose_valid_raw is not None
            else True
        )
        checkpoint_orientation_frozen = bool(
            checkpoint.get("eef_orientation_frozen", False)
        )
        checkpoint_pose_quality_raw = checkpoint.get(
            "eef_pose_quality",
            None,
        )
        try:
            checkpoint_pose_quality = (
                None
                if checkpoint_pose_quality_raw is None
                else float(checkpoint_pose_quality_raw)
            )
        except Exception:
            checkpoint_pose_quality = None

        checkpoint_orientation_control_active = True
        orientation_control_reason = "enabled"
        if use_orientation and target_rotation_reference is not None:
            if checkpoint_orientation_skip_frozen and (
                not checkpoint_pose_valid or checkpoint_orientation_frozen
            ):
                checkpoint_orientation_control_active = False
                orientation_control_reason = "checkpoint_frozen_or_invalid"
            elif (
                checkpoint_pose_quality is not None
                and checkpoint_pose_quality < checkpoint_orientation_min_quality
            ):
                checkpoint_orientation_control_active = False
                orientation_control_reason = "checkpoint_low_quality"

        correction_steps = 0
        success = False
        position_error_norm = None
        orientation_error_norm = None

        while correction_steps <= maximum_correction_steps:
            (
                current_reference,
                current_rotation_reference,
            ) = current_pose_reference()
            position_error = np.asarray(
                target_reference - current_reference,
                dtype=np.float64,
            ).reshape(3)
            position_error_norm = float(np.linalg.norm(position_error))

            orientation_ok = True
            rotation_step = np.zeros((3,), dtype=np.float64)
            if (
                use_orientation
                and checkpoint_orientation_control_active
                and target_rotation_reference is not None
                and current_rotation_reference is not None
                and step_clip_orientation is not None
            ):
                relative_rotation = (
                    np.asarray(target_rotation_reference)
                    @ np.asarray(current_rotation_reference).T
                )
                rotation_error = np.asarray(
                    Rotation.from_matrix(relative_rotation).as_rotvec(),
                    dtype=np.float64,
                ).reshape(3)
                orientation_error_norm = float(np.linalg.norm(rotation_error))
                orientation_ok = bool(orientation_error_norm <= orientation_tolerance)
                if correction_mode == "coupled" or position_error_norm <= max(
                    position_tolerance,
                    position_threshold,
                ):
                    rotation_step = np.clip(
                        rotation_error * arm_orientation_gain,
                        -step_clip_orientation,
                        step_clip_orientation,
                    )

            if correction_mode == "position_dominate":
                checkpoint_ok = bool(position_error_norm <= position_tolerance)
            else:
                checkpoint_ok = bool(
                    position_error_norm <= position_tolerance
                ) and bool(orientation_ok)
            if checkpoint_ok:
                success = True
                break
            if correction_steps == maximum_correction_steps:
                break

            position_step = np.clip(
                position_error * arm_position_gain,
                -step_clip_position,
                step_clip_position,
            )
            raw_correction = (
                np.concatenate(
                    [position_step, rotation_step],
                    axis=0,
                )
                if use_orientation
                else position_step
            )
            terminated, _ = step_once(
                raw_action=raw_correction,
                gripper_cmd=gripper_cmd,
                kind="correction",
                plan_step_index=None,
                target_checkpoint_index=checkpoint_index,
                target_frame=target_frame,
                is_checkpoint_boundary=False,
            )
            correction_steps += 1
            if terminated:
                break

        checkpoint_entry = {
            "checkpoint_index": int(checkpoint_index),
            "frame": int(checkpoint.get("frame", target_frame)),
            "boundary_plan_step_index": (
                None
                if boundary_plan_step_index is None
                else int(boundary_plan_step_index)
            ),
            "boundary_exec_step_index": (
                None
                if boundary_exec_step_index is None
                else int(boundary_exec_step_index)
            ),
            "correction_steps": int(correction_steps),
            "success": bool(success),
            "terminated": bool(terminated),
            "position_error_norm": position_error_norm,
            "orientation_error_norm": (orientation_error_norm),
            "orientation_within_tolerance": bool(orientation_ok),
            "target_ref_3d": target_reference.tolist(),
            "orientation_control_active": bool(checkpoint_orientation_control_active),
            "orientation_control_reason": str(orientation_control_reason),
            "pose_correction_mode": str(correction_mode),
            "position_dominate_correction_threshold_m": float(position_threshold),
            "eef_pose_valid": bool(checkpoint_pose_valid),
            "eef_orientation_frozen": bool(checkpoint_orientation_frozen),
            "eef_pose_quality": checkpoint_pose_quality,
            "close_gate_required": bool(
                checkpoint.get(
                    "close_gate_required",
                    False,
                )
            ),
            "close_gate_step_range": dict(
                checkpoint.get(
                    "close_gate_step_range",
                    {},
                )
                or {}
            ),
            "open_gate_required": bool(
                checkpoint.get(
                    "open_gate_required",
                    False,
                )
            ),
            "open_gate_step_range": dict(
                checkpoint.get(
                    "open_gate_step_range",
                    {},
                )
                or {}
            ),
        }
        checkpoint_trace.append(checkpoint_entry)
        if checkpoint_observer is not None:
            checkpoint_observer(env, checkpoint_entry)
        return bool(success)

    if (
        checkpoints
        and checkpoints[0].get(
            "boundary_step_index",
            None,
        )
        is None
    ):
        initial_frame = int(checkpoints[0].get("frame", 0))
        initial_gripper_command = float(checkpoints[0].get("gripper_cmd", gripper))
        evaluate_checkpoint(
            0,
            target_frame=initial_frame,
            gripper_cmd=initial_gripper_command,
            boundary_plan_step_index=None,
            boundary_exec_step_index=None,
        )

    for step in steps:
        if terminated:
            break
        step_index = int(step.get("step_index", len(action_trace)))
        step_kind = str(step.get("kind", "motion") or "motion")
        checkpoint_index = int(step.get("target_checkpoint_index", 0))
        target_frame = int(
            step.get(
                "target_frame",
                checkpoints[checkpoint_index].get(
                    "frame",
                    checkpoint_index,
                ),
            )
        )
        gripper_command = float(step.get("gripper_cmd", gripper))
        is_boundary = bool(step.get("is_checkpoint_boundary", False))

        if step_kind in {"close_only_gate", "open_only_gate"}:
            gate_enabled = (
                enable_close_gate
                if step_kind == "close_only_gate"
                else enable_open_gate
            )
            if gate_enabled:
                gate_completed = execute_gripper_gate(
                    plan_step_index=step_index,
                    target_checkpoint_index=checkpoint_index,
                    target_frame=target_frame,
                    gripper_cmd=gripper_command,
                    gate_kind=step_kind,
                    gate_enabled=gate_enabled,
                    require_non_support_contact=bool(
                        step_kind == "close_only_gate"
                        and close_gate_require_non_support_contact
                    ),
                )
                if terminated or not gate_completed:
                    terminated = bool(
                        terminated or close_gate_failure_policy == "terminate"
                    )
                    if not terminated:
                        continue
                    break
                continue
            action_6d = np.zeros((6,), dtype=np.float64)
            raw_action = (
                action_6d[:6]
                if (use_orientation and controller_name != "OSC_POSITION")
                else action_6d[:3]
            )
        else:
            action_6d = np.asarray(
                step.get("action_ref_6d", [0.0] * 6),
                dtype=np.float64,
            ).reshape(-1)
            raw_action = (
                action_6d[:6]
                if (use_orientation and controller_name != "OSC_POSITION")
                else action_6d[:3]
            )

        terminated, execution_step_index = step_once(
            raw_action=raw_action,
            gripper_cmd=gripper_command,
            kind=step_kind,
            plan_step_index=step_index,
            target_checkpoint_index=checkpoint_index,
            target_frame=target_frame,
            is_checkpoint_boundary=is_boundary,
        )
        if terminated or not is_boundary:
            continue
        evaluate_checkpoint(
            checkpoint_index,
            target_frame=target_frame,
            gripper_cmd=gripper_command,
            boundary_plan_step_index=step_index,
            boundary_exec_step_index=execution_step_index,
        )

    return {
        "controller": controller_name,
        "reference_frame": reference_frame,
        "use_ori": bool(use_orientation),
        "orientation_target_mode": orientation_target_mode,
        "policy_hz": int(policy_hz),
        "render": bool(render),
        "pos_tol": float(position_tolerance),
        "ori_tol": float(orientation_tolerance),
        "max_correction_steps": int(maximum_correction_steps),
        "must_reach_min_correction_steps": int(must_reach_min_correction_steps),
        "pose_correction_mode": str(correction_mode),
        "position_dominate_correction_threshold_m": float(position_threshold),
        "enable_close_completion_gate": bool(enable_close_gate),
        "enable_open_completion_gate": bool(enable_open_gate),
        "close_gate_min_hold_steps": int(close_gate_min_hold_steps),
        "close_gate_max_wait_steps": int(close_gate_max_wait_steps),
        "close_gate_qpos_delta_min": float(close_gate_qpos_delta_min),
        "close_gate_qpos_settle_tol": float(close_gate_qpos_settle_tol),
        "close_gate_settle_window": int(close_gate_settle_window),
        "close_gate_require_non_support_contact": bool(
            close_gate_require_non_support_contact
        ),
        "close_gate_contact_settle_steps": int(close_gate_contact_settle_steps),
        "close_gate_failure_policy": str(close_gate_failure_policy),
        "num_action_steps": int(len(steps)),
        "num_checkpoints": int(len(checkpoints)),
        "total_env_steps": int(total_env_steps),
        "terminated": bool(terminated),
        "action_trace": action_trace,
        "checkpoint_trace": checkpoint_trace,
        "dense_tcp_trace": dense_tcp_trace,
        "close_gate_trace": close_gate_trace,
        "open_gate_trace": open_gate_trace,
    }


def _to_serializable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


def _save_json(payload: Any, path: str) -> str:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as stream:
        json.dump(_to_serializable(payload), stream, indent=4)
    return path


def _action_trace_payload(
    result: Mapping[str, Any],
    *,
    uid: str,
    action_path: str,
    arm_part: str,
    gripper_part: str | None,
    drive_site: str,
) -> dict[str, Any]:
    return {
        "meta": {
            "source": "action_executor",
            "uid": uid,
            "action_path": action_path,
            "controller": result["controller"],
            "reference_frame": result["reference_frame"],
            "use_ori": bool(result["use_ori"]),
            "arm_part": arm_part,
            "gripper_part": gripper_part,
            "drive_site": drive_site,
        },
        "steps": [
            {
                "exec_step_index": int(item.exec_step_index),
                "plan_step_index": (
                    None if item.plan_step_index is None else int(item.plan_step_index)
                ),
                "kind": item.kind,
                "target_checkpoint_index": int(item.target_checkpoint_index),
                "target_frame": int(item.target_frame),
                "raw_action": item.raw_action.tolist(),
                "normalized_action": (item.normalized_action.tolist()),
                "gripper_cmd": float(item.gripper_cmd),
                "action_vector": item.action_vector.tolist(),
                "is_checkpoint_boundary": bool(item.is_checkpoint_boundary),
            }
            for item in result["action_trace"]
        ],
    }


def publish_action_trace_outputs(
    result: Mapping[str, Any],
    *,
    output_dir: str | os.PathLike[str],
    uid: str = "",
    action_path: str = "",
    traj_path: str = "",
    tcp_site: str = "",
    ctrl_ref_site: str = "",
    drive_site: str = "",
    arm_part: str = "right",
    gripper_part: str | None = None,
    save_video: bool = True,
    save_action_trace: bool = False,
    json_writer: Callable[[Any, str], Any] | None = None,
    manifest_updater: Callable[..., str] | None = None,
) -> dict[str, Any]:
    """Write current action executor JSON/manifest outputs explicitly."""

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    write_json = _save_json if json_writer is None else json_writer
    update_manifest = (
        update_exec_assets_manifest if manifest_updater is None else manifest_updater
    )

    execution_paths = execution_artifact_paths(output_path)
    video_path = execution_paths["execution_video"]
    action_trace_path = execution_paths["action_trace"]
    checkpoint_trace_path = execution_paths["checkpoint_trace"].as_posix()
    dense_tcp_trace_path = execution_paths["dense_tcp_trace"]

    checkpoint_payload = {
        "meta": {
            "source": "action_executor",
            "uid": uid,
            "action_path": action_path,
            "controller": result["controller"],
            "reference_frame": result["reference_frame"],
            "use_ori": bool(result["use_ori"]),
            "pos_tol": float(result["pos_tol"]),
            "ori_tol": float(result["ori_tol"]),
            "max_correction_steps": int(result["max_correction_steps"]),
            "must_reach_min_correction_steps": int(
                result["must_reach_min_correction_steps"]
            ),
            "pose_correction_mode": str(result["pose_correction_mode"]),
            "position_dominate_correction_threshold_m": float(
                result["position_dominate_correction_threshold_m"]
            ),
            "enable_close_completion_gate": bool(
                result["enable_close_completion_gate"]
            ),
            "enable_open_completion_gate": bool(result["enable_open_completion_gate"]),
            "close_gate_min_hold_steps": int(result["close_gate_min_hold_steps"]),
            "close_gate_max_wait_steps": int(result["close_gate_max_wait_steps"]),
            "close_gate_qpos_delta_min": float(result["close_gate_qpos_delta_min"]),
            "close_gate_qpos_settle_tol": float(result["close_gate_qpos_settle_tol"]),
            "close_gate_settle_window": int(result["close_gate_settle_window"]),
            "close_gate_require_non_support_contact": bool(
                result["close_gate_require_non_support_contact"]
            ),
            "close_gate_contact_settle_steps": int(
                result["close_gate_contact_settle_steps"]
            ),
            "close_gate_failure_policy": str(result["close_gate_failure_policy"]),
        },
        "checkpoints": result["checkpoint_trace"],
        "close_gates": result["close_gate_trace"],
        "open_gates": result["open_gate_trace"],
    }
    write_json(checkpoint_payload, checkpoint_trace_path)

    dense_tcp_payload = {
        "meta": {
            "source": "action_executor",
            "uid": uid,
            "action_path": action_path,
            "controller": result["controller"],
            "reference_frame": "world",
            "controller_reference_frame": (result["reference_frame"]),
            "use_ori": bool(result["use_ori"]),
            "tcp_site": tcp_site,
            "ctrl_ref_site": ctrl_ref_site,
            "drive_site": drive_site,
            "pos_unit": "m",
            "rotation_unit": "quat_wxyz",
            "timestamp_unit": "s",
            "policy_hz": int(result["policy_hz"]),
        },
        "steps": result["dense_tcp_trace"],
    }
    write_json(
        dense_tcp_payload,
        dense_tcp_trace_path.as_posix(),
    )
    if save_action_trace:
        write_json(
            _action_trace_payload(
                result,
                uid=uid,
                action_path=action_path,
                arm_part=arm_part,
                gripper_part=gripper_part,
                drive_site=drive_site,
            ),
            action_trace_path.as_posix(),
        )

    success_count = sum(
        1 for item in result["checkpoint_trace"] if bool(item.get("success", False))
    )
    kind_counts = Counter(item.kind for item in result["action_trace"])
    close_completed_count = sum(
        1 for item in result["close_gate_trace"] if bool(item.get("completed", False))
    )
    open_completed_count = sum(
        1 for item in result["open_gate_trace"] if bool(item.get("completed", False))
    )
    position_errors = [
        float(item["position_error_norm"])
        for item in result["checkpoint_trace"]
        if item.get("position_error_norm", None) is not None
    ]
    orientation_errors = [
        float(item["orientation_error_norm"])
        for item in result["checkpoint_trace"]
        if item.get("orientation_error_norm", None) is not None
    ]
    summary = {
        "uid": uid,
        "output_dir": output_path.as_posix(),
        "video_path": (video_path.as_posix() if video_path.exists() else None),
        "action_path": action_path,
        "action_trace_path": (
            action_trace_path.as_posix() if save_action_trace else None
        ),
        "checkpoint_trace_path": checkpoint_trace_path,
        "dense_tcp_trace_path": dense_tcp_trace_path.as_posix(),
        "executed": True,
        "executor": "action",
        "controller": result["controller"],
        "reference_frame": result["reference_frame"],
        "use_ori": bool(result["use_ori"]),
        "orientation_target_mode": result["orientation_target_mode"],
        "policy_hz": int(result["policy_hz"]),
        "num_action_steps": int(result["num_action_steps"]),
        "num_checkpoints": int(result["num_checkpoints"]),
        "checkpoints_evaluated": int(len(result["checkpoint_trace"])),
        "max_correction_steps": int(result["max_correction_steps"]),
        "pose_correction_mode": str(result["pose_correction_mode"]),
        "position_dominate_correction_threshold_m": float(
            result["position_dominate_correction_threshold_m"]
        ),
        "env_steps": int(result["total_env_steps"]),
        "checkpoint_successes": int(success_count),
        "checkpoint_failures": int(len(result["checkpoint_trace"]) - success_count),
        "checkpoint_max_position_error_norm": (
            max(position_errors) if position_errors else None
        ),
        "checkpoint_max_orientation_error_norm": (
            max(orientation_errors) if orientation_errors else None
        ),
        "step_kind_counts": {
            str(key): int(value) for key, value in sorted(kind_counts.items())
        },
        "correction_steps": int(kind_counts.get("correction", 0)),
        "close_gate_count": int(len(result["close_gate_trace"])),
        "close_gate_completed_count": int(close_completed_count),
        "close_gate_failed_count": int(
            len(result["close_gate_trace"]) - close_completed_count
        ),
        "open_gate_count": int(len(result["open_gate_trace"])),
        "open_gate_completed_count": int(open_completed_count),
        "open_gate_failed_count": int(
            len(result["open_gate_trace"]) - open_completed_count
        ),
        "close_completed_before_motion": (
            None
            if not result["close_gate_trace"]
            else bool(
                all(item.get("completed", False) for item in result["close_gate_trace"])
            )
        ),
        "open_completed_before_motion": (
            None
            if not result["open_gate_trace"]
            else bool(
                all(item.get("completed", False) for item in result["open_gate_trace"])
            )
        ),
        "terminated_early": bool(result["terminated"]),
    }
    if str(traj_path).strip():
        summary["traj_path"] = str(traj_path)
    summary_path = execution_paths["exec_summary"].as_posix()
    write_json(summary, summary_path)
    update_manifest(
        output_path.as_posix(),
        "execution",
        {
            "dir": output_path.as_posix(),
            "exec_video": (video_path.as_posix() if save_video else None),
            "checkpoint_trace_json": checkpoint_trace_path,
            "dense_tcp_trace_json": (dense_tcp_trace_path.as_posix()),
            "action_trace_json": (
                action_trace_path.as_posix() if save_action_trace else None
            ),
            "exec_summary_json": summary_path,
        },
    )
    summary["summary_path"] = summary_path
    return summary


def execute_action_trace_explicit(
    *,
    env: Any,
    action_payload: Mapping[str, Any],
    simulator_config: Mapping[str, Any],
    execution_config: Mapping[str, Any] | None,
    output_dir: str | os.PathLike[str],
    uid: str = "",
    action_path: str = "",
    traj_path: str = "",
    robot: Any | None = None,
    part_ctrl: Any | None = None,
    arm_part: str = "",
    gripper_part: str | None = None,
    ref_site_name: str = "",
    tcp_site_name: str = "",
    ctrl_ref_site_name: str = "",
    drive_site_name: str = "",
    camera_name: str = "",
    frame_size: tuple[int, int] | None = None,
    pose_reader: Callable[..., Any] | None = None,
    contact_reader: Callable[..., Sequence[Any]] | None = None,
    frame_recorder: Any | None = None,
    json_writer: Callable[[Any, str], Any] | None = None,
    manifest_updater: Callable[..., str] | None = None,
    close_env: bool = True,
    perform_warm_start: bool = True,
) -> dict[str, Any]:
    """Warm, execute, close, encode, and publish one explicit action trace."""

    output_text = os.fspath(output_dir)
    if not str(output_text).strip():
        raise ValueError("output_dir is required")
    validate_action_trace_preflight(
        action_payload,
        execution_config,
    )
    output_path = Path(output_text).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    execution_paths = execution_artifact_paths(output_path)

    execution, runtime = _execution_sections(execution_config)
    save_video = bool(
        _execution_setting(
            runtime,
            "save_video",
            _execution_setting(execution, "save_video", True),
        )
    )
    save_action_trace = bool(
        _execution_setting(
            execution,
            "save_action_trace",
            False,
        )
    )
    video_path = execution_paths["execution_video"]
    if not save_video and video_path.exists():
        video_path.unlink()
    action_trace_output_path = execution_paths["action_trace"]
    if not save_action_trace and action_trace_output_path.exists():
        action_trace_output_path.unlink()

    runtime_robot = env.robots[0] if robot is None else robot
    inferred_arm, inferred_gripper, _ = infer_arm_and_gripper_parts(runtime_robot)
    runtime_arm_part = arm_part or inferred_arm
    runtime_gripper_part = inferred_gripper if gripper_part is None else gripper_part
    if part_ctrl is None:
        composite_controller = getattr(
            runtime_robot,
            "composite_controller",
            None,
        ) or getattr(runtime_robot, "controller", None)
        if composite_controller is None:
            raise RuntimeError(
                "Cannot find robot controller "
                "(neither composite_controller nor controller)."
            )
        runtime_part_ctrl = (
            composite_controller.get_controller(runtime_arm_part)
            if hasattr(
                composite_controller,
                "get_controller",
            )
            else composite_controller
        )
    else:
        runtime_part_ctrl = part_ctrl

    tcp_site = (
        str(tcp_site_name)
        if tcp_site_name
        else (tcp_site_name_from_config(dict(simulator_config)) or "")
    )
    ctrl_ref_site = (
        str(ctrl_ref_site_name)
        if ctrl_ref_site_name
        else (controller_reference_site_name_from_config(dict(simulator_config)) or "")
    )
    if not ctrl_ref_site and tcp_site:
        ctrl_ref_site = tcp_site
    drive_site = (
        str(drive_site_name)
        if drive_site_name
        else (
            _drive_site_name_from_config(simulator_config)
            or ctrl_ref_site
            or tcp_site
            or ""
        )
    )
    runtime_ref_site = ref_site_name or ctrl_ref_site or tcp_site

    planner_meta = dict(action_payload.get("meta", {}).get("planner", {}) or {})
    action_space = dict(action_payload.get("meta", {}).get("action_space", {}) or {})
    controller_name = str(
        _execution_setting(
            execution,
            "controller",
            "OSC_POSITION",
        )
    )
    planner_controller = str(planner_meta.get("controller", "") or "").strip()
    if planner_controller:
        controller_name = planner_controller
    use_orientation = (
        bool(action_space.get("has_orientation", False))
        and controller_name != "OSC_POSITION"
    )
    if not use_orientation and controller_name == "OSC_POSE":
        controller_name = "OSC_POSITION"

    warm_start_steps = int(
        _execution_setting(
            execution,
            "warm_start_steps",
            10,
        )
    )
    gripper = float(_execution_setting(execution, "gripper", 0.0))
    if perform_warm_start:
        warm_start_hold(
            env=env,
            robot=runtime_robot,
            arm_part=runtime_arm_part,
            gripper_part=runtime_gripper_part,
            part_ctrl=runtime_part_ctrl,
            controller_name=controller_name,
            use_ori=bool(use_orientation),
            gripper=gripper,
            steps=warm_start_steps,
        )

    reference_frame = str(action_space.get("reference_frame", "base") or "base")
    rotation_world_base = None
    translation_world_base = None
    if reference_frame == "base":
        derived = dict(simulator_config.get("derived", {}) or {})
        derived_eef = dict(derived.get("eef", {}) or {})
        transform = derived_eef.get("X_wb", None)
        if transform is None:
            raise RuntimeError(
                "input_ref_frame=base requires "
                "cfg['derived']['eef']['X_wb'] (base->world)."
            )
        rotation_world_base, translation_world_base = parse_X_wb(transform)

    if frame_recorder is None:
        if save_video:
            if not camera_name or frame_size is None:
                raise ValueError(
                    "save_video requires frame_recorder or explicit "
                    "camera_name and frame_size"
                )
            recorder = VideoRecorder(
                True,
                env,
                camera_name,
                frame_size,
            )
        else:
            recorder = VideoRecorder(
                False,
                env,
                camera_name,
                frame_size or (0, 0),
            )
    else:
        recorder = frame_recorder

    result = execute_action_trace_loop(
        env=env,
        robot=runtime_robot,
        part_ctrl=runtime_part_ctrl,
        arm_part=runtime_arm_part,
        gripper_part=runtime_gripper_part,
        action_payload=action_payload,
        execution_config=execution_config,
        ref_site_name=runtime_ref_site,
        rotation_world_base=rotation_world_base,
        translation_world_base=translation_world_base,
        pose_reader=pose_reader,
        frame_capture=recorder.grab,
        contact_reader=contact_reader,
    )
    if close_env:
        env.close()
    if save_video:
        recorder.save(
            video_path.as_posix(),
            fps=int(result["policy_hz"]),
        )

    return publish_action_trace_outputs(
        result,
        output_dir=output_path.as_posix(),
        uid=str(uid),
        action_path=str(action_path),
        traj_path=str(traj_path),
        tcp_site=tcp_site,
        ctrl_ref_site=ctrl_ref_site,
        drive_site=drive_site,
        arm_part=runtime_arm_part,
        gripper_part=runtime_gripper_part,
        save_video=save_video,
        save_action_trace=save_action_trace,
        json_writer=json_writer,
        manifest_updater=manifest_updater,
    )


__all__ = [
    "ActionStepTrace",
    "VideoRecorder",
    "current_eef_pose_world",
    "execute_action_trace_explicit",
    "execute_action_trace_loop",
    "pose_in_reference",
    "publish_action_trace_outputs",
    "validate_action_trace_preflight",
    "warm_start_hold",
]
