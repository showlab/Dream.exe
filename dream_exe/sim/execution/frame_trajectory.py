"""Explicit-runtime frame-trajectory execution.

The caller supplies an already-restored simulator environment, parsed
trajectory frames, an aligned gripper schedule, execution settings, and an
explicit output directory. UID, bench, config-root, environment construction,
reset, restore, and horizon selection remain outer-shell responsibilities.
"""

from __future__ import annotations

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
from dream_exe.transforms import parse_X_wb

from .action_trace import (
    VideoRecorder,
    current_eef_pose_world,
    pose_in_reference,
    warm_start_hold,
)
from ..runtime.controller import (
    controller_reference_site_name_from_config,
    read_step_clips,
    tcp_site_name_from_config,
)
from ..runtime.contacts import (
    find_support_top_z,
    robot_contacts_any,
)
from .inputs import parse_pose_from_entry
from ..runtime.action import (
    infer_arm_and_gripper_parts,
    normalize_arm_action,
    pack_action,
)


@dataclass(frozen=True)
class RobustKnobs:
    """Current adaptive tracking iteration bounds."""

    TRACKING_MULT: float = 5.0
    MIN_ITERS_FLOOR: int = 40
    MAX_ITERS_CAP: int = 120


def _config_sections(
    config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    payload = dict(config or {})
    if any(key in payload for key in ("input", "execution", "runtime")):
        return (
            dict(payload.get("input", {}) or {}),
            dict(payload.get("execution", {}) or {}),
            dict(payload.get("runtime", {}) or {}),
        )
    return {}, payload, {}


def _setting(
    section: Mapping[str, Any],
    key: str,
    default: Any,
) -> Any:
    value = section.get(key, default)
    return default if value is None else value


def _rotation_vector(rotation: np.ndarray) -> np.ndarray:
    return Rotation.from_matrix(
        np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    ).as_rotvec()


def _validate_trajectory(
    trajectory: Any,
    *,
    traj_key: str,
    max_steps: int,
) -> list[dict[str, Any]]:
    if not isinstance(trajectory, list) or len(trajectory) == 0:
        raise ValueError(f"traj[{traj_key}] is empty or not a list")
    frames = list(trajectory)
    if int(max_steps) > 0:
        frames = frames[: int(max_steps)]
    return frames


def _aligned_gripper_schedule(
    *,
    frame_count: int,
    default_command: float,
    commands: Sequence[float] | np.ndarray | None,
    edge_flags: Sequence[int] | np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if commands is None:
        command_array = np.full(
            (frame_count,),
            float(default_command),
            dtype=np.float64,
        )
    else:
        command_array = np.asarray(
            commands,
            dtype=np.float64,
        ).reshape(-1)
        if command_array.size < frame_count:
            raise ValueError(
                "gripper_commands length "
                f"{command_array.size} is smaller than trajectory "
                f"length {frame_count}"
            )
        command_array = command_array[:frame_count].copy()

    if edge_flags is None:
        edge_array = np.zeros(
            (frame_count,),
            dtype=np.int8,
        )
    else:
        edge_array = np.asarray(
            edge_flags,
            dtype=np.int8,
        ).reshape(-1)
        if edge_array.size < frame_count:
            raise ValueError(
                "gripper_edge_flags length "
                f"{edge_array.size} is smaller than trajectory "
                f"length {frame_count}"
            )
        edge_array = edge_array[:frame_count].copy()
    return command_array, edge_array


def execute_frame_trajectory_loop(
    *,
    env: Any,
    robot: Any,
    part_ctrl: Any,
    arm_part: str,
    gripper_part: str | None,
    trajectory: list[dict[str, Any]],
    traj_key: str = "eef_controller",
    execution_config: Mapping[str, Any] | None = None,
    gripper_commands: Sequence[float] | np.ndarray | None = None,
    gripper_edge_flags: Sequence[int] | np.ndarray | None = None,
    ref_site_name: str = "",
    rotation_world_base: np.ndarray | None = None,
    translation_world_base: np.ndarray | None = None,
    support_top_z: float | None = None,
    pose_reader: Callable[..., Any] | None = None,
    frame_capture: Callable[[], Any] | None = None,
    contact_reader: Callable[..., Sequence[Any]] | None = None,
    pose_parser: Callable[..., Any] | None = None,
    logger: Callable[[str], Any] | None = print,
    knobs: RobustKnobs | None = None,
) -> dict[str, Any]:
    """Drive parsed trajectory frames in one prepared environment."""

    if not isinstance(trajectory, list) or len(trajectory) == 0:
        raise ValueError(f"traj[{traj_key}] is empty or not a list")
    _, execution, runtime = _config_sections(execution_config)
    controller_name = str(_setting(execution, "controller", "OSC_POSITION"))
    use_orientation = bool(_setting(execution, "use_ori", False))
    input_reference_frame = str(
        _setting(
            execution,
            "input_ref_frame",
            "world",
        )
        or "world"
    )
    default_gripper_command = float(_setting(execution, "gripper", 0.0))
    position_tolerance = float(_setting(execution, "pos_tol", 5.0e-3))
    orientation_tolerance = float(_setting(execution, "ori_tol", 3.0e-2))
    position_gain = float(_setting(execution, "arm_pos_gain", 2.0))
    orientation_gain = float(_setting(execution, "arm_ori_gain", 1.5))
    force_one_step_per_frame = bool(
        _setting(
            execution,
            "force_one_step_per_frame",
            True,
        )
    )
    render = bool(
        _setting(
            runtime,
            "render",
            _setting(execution, "render", False),
        )
    )
    close_hold_steps = int(
        max(
            2,
            int(
                _setting(
                    execution,
                    "hold_steps_after_gripper_close",
                    2,
                )
            ),
        )
    )
    open_hold_steps = int(
        max(
            2,
            int(
                _setting(
                    execution,
                    "hold_steps_after_gripper_open",
                    2,
                )
            ),
        )
    )
    legacy_edge_hold_steps = int(
        max(
            0,
            int(
                _setting(
                    execution,
                    "hold_steps_after_gripper_change",
                    0,
                )
            ),
        )
    )
    active_knobs = RobustKnobs() if knobs is None else knobs
    parse_pose = parse_pose_from_entry if pose_parser is None else pose_parser
    read_pose = current_eef_pose_world if pose_reader is None else pose_reader
    read_contacts = robot_contacts_any if contact_reader is None else contact_reader
    grab_frame = (lambda: None) if frame_capture is None else frame_capture
    emit = (lambda _message: None) if logger is None else logger

    commands, edges = _aligned_gripper_schedule(
        frame_count=len(trajectory),
        default_command=default_gripper_command,
        commands=gripper_commands,
        edge_flags=gripper_edge_flags,
    )
    _output_maximum, step_clip_position, step_clip_orientation = read_step_clips(
        part_ctrl,
        controller_name,
        bool(use_orientation),
    )

    total_env_steps = 0
    fail_frames = 0
    terminated = False
    frame_env_steps: list[int] = []
    checkpoint_trace: list[dict[str, Any]] = []
    z_margin = 0.01

    def step_once(
        raw_action: np.ndarray,
        gripper_command: float,
    ) -> bool:
        nonlocal total_env_steps
        arm_action = normalize_arm_action(
            part_ctrl,
            raw_action,
        )
        action_vector = pack_action(
            robot=robot,
            arm_part=arm_part,
            arm_action=arm_action,
            gripper_part=gripper_part,
            gripper_cmd=float(gripper_command),
        )
        _, _, done, _ = env.step(action_vector)
        total_env_steps += 1
        if render:
            env.render()
        grab_frame()
        return bool(done)

    def hold_gripper(
        steps: int,
        gripper_command: float,
    ) -> None:
        if steps <= 0:
            return
        arm_dimension = (
            3
            if controller_name == "OSC_POSITION"
            else (6 if bool(use_orientation) else 3)
        )
        raw_zero = np.zeros(
            arm_dimension,
            dtype=np.float64,
        )
        for _ in range(int(steps)):
            done = step_once(raw_zero, gripper_command)
            if done:
                # The current executor only interrupts this settle loop. It
                # does not propagate settle-time done to global termination.
                break

    def target_pose(
        index: int,
    ) -> tuple[np.ndarray, np.ndarray | None, bool]:
        target_position_world, target_rotation_world = parse_pose(
            trajectory[index],
            traj_key=traj_key,
        )
        target_position_world = np.asarray(
            target_position_world,
            dtype=np.float64,
        ).reshape(3)
        track_orientation = bool(
            controller_name != "OSC_POSITION"
            and use_orientation
            and target_rotation_world is not None
            and step_clip_orientation is not None
            and len(step_clip_orientation) == 3
        )
        if controller_name == "OSC_POSITION":
            target_rotation_world = None
            track_orientation = False
        elif track_orientation:
            target_rotation_world = np.asarray(
                target_rotation_world,
                dtype=np.float64,
            ).reshape(3, 3)
        return (
            target_position_world,
            target_rotation_world,
            track_orientation,
        )

    def pose_in_control_frame(
        position_world: np.ndarray,
        rotation_world: np.ndarray | None,
    ) -> tuple[np.ndarray, np.ndarray | None]:
        return pose_in_reference(
            position_world,
            rotation_world,
            input_reference_frame,
            rotation_world_base,
            translation_world_base,
        )

    def measure_target_error(index: int) -> dict[str, Any]:
        (
            target_position_world,
            target_rotation_world,
            track_orientation,
        ) = target_pose(index)
        (
            current_position_world,
            current_rotation_world,
            _pose_meta,
        ) = read_pose(env, site_name=ref_site_name)
        current_position_world = np.asarray(
            current_position_world,
            dtype=np.float64,
        ).reshape(3)
        if not track_orientation:
            current_rotation_world = None
        else:
            current_rotation_world = np.asarray(
                current_rotation_world,
                dtype=np.float64,
            ).reshape(3, 3)

        current_position_ref, current_rotation_ref = pose_in_control_frame(
            current_position_world,
            current_rotation_world,
        )
        target_position_ref, target_rotation_ref = pose_in_control_frame(
            target_position_world,
            target_rotation_world,
        )
        position_error = np.asarray(
            target_position_ref - current_position_ref,
            dtype=np.float64,
        ).reshape(3)
        orientation_error_norm = None
        if (
            track_orientation
            and current_rotation_ref is not None
            and target_rotation_ref is not None
        ):
            relative_rotation = (
                target_rotation_ref @ current_rotation_ref.T
                if controller_name.startswith("OSC_")
                else (
                    np.asarray(current_rotation_world).T
                    @ np.asarray(target_rotation_world)
                )
            )
            orientation_error_norm = float(
                np.linalg.norm(_rotation_vector(relative_rotation))
            )
        return {
            "position_error_norm": float(np.linalg.norm(position_error)),
            "orientation_error_norm": orientation_error_norm,
            "target_ref_3d": np.asarray(
                target_position_ref,
                dtype=np.float64,
            )
            .reshape(3)
            .tolist(),
        }

    def drive_to_target(
        index: int,
        gripper_command: float,
    ) -> tuple[bool, bool]:
        (
            target_position_world,
            target_rotation_world,
            track_orientation,
        ) = target_pose(index)
        (
            current_position_world,
            current_rotation_world,
            _pose_meta,
        ) = read_pose(env, site_name=ref_site_name)
        current_position_world = np.asarray(
            current_position_world,
            dtype=np.float64,
        ).reshape(3)
        if not track_orientation:
            current_rotation_world = None
        else:
            current_rotation_world = np.asarray(
                current_rotation_world,
                dtype=np.float64,
            ).reshape(3, 3)

        current_position_ref, current_rotation_ref = pose_in_control_frame(
            current_position_world,
            current_rotation_world,
        )
        target_position_ref, target_rotation_ref = pose_in_control_frame(
            target_position_world,
            target_rotation_world,
        )
        initial_position_delta = target_position_ref - current_position_ref
        position_need = float(
            np.max(np.abs(initial_position_delta) / (step_clip_position + 1.0e-12))
        )
        orientation_need = 0.0
        if (
            track_orientation
            and current_rotation_ref is not None
            and target_rotation_ref is not None
        ):
            initial_relative_rotation = (
                target_rotation_ref @ current_rotation_ref.T
                if controller_name.startswith("OSC_")
                else (
                    np.asarray(current_rotation_world).T
                    @ np.asarray(target_rotation_world)
                )
            )
            initial_rotation_vector = _rotation_vector(initial_relative_rotation)
            orientation_need = float(
                np.max(
                    np.abs(initial_rotation_vector) / (step_clip_orientation + 1.0e-12)
                )
            )

        need = max(position_need, orientation_need)
        maximum_iterations = int(np.ceil(need * active_knobs.TRACKING_MULT)) + 10
        maximum_iterations = max(
            active_knobs.MIN_ITERS_FLOOR,
            min(
                active_knobs.MAX_ITERS_CAP,
                maximum_iterations,
            ),
        )

        iteration = 0
        while iteration < maximum_iterations:
            (
                current_position_world,
                current_rotation_world,
                _pose_meta,
            ) = read_pose(env, site_name=ref_site_name)
            current_position_world = np.asarray(
                current_position_world,
                dtype=np.float64,
            ).reshape(3)
            if not track_orientation:
                current_rotation_world = None
            else:
                current_rotation_world = np.asarray(
                    current_rotation_world,
                    dtype=np.float64,
                ).reshape(3, 3)

            current_position_ref, current_rotation_ref = pose_in_control_frame(
                current_position_world,
                current_rotation_world,
            )
            target_position_ref, target_rotation_ref = pose_in_control_frame(
                target_position_world,
                target_rotation_world,
            )
            position_error = target_position_ref - current_position_ref
            position_ok = bool(
                float(np.linalg.norm(position_error)) <= position_tolerance
            )

            orientation_ok = True
            rotation_step = None
            if (
                track_orientation
                and current_rotation_ref is not None
                and target_rotation_ref is not None
            ):
                relative_rotation = (
                    target_rotation_ref @ current_rotation_ref.T
                    if controller_name.startswith("OSC_")
                    else (
                        np.asarray(current_rotation_world).T
                        @ np.asarray(target_rotation_world)
                    )
                )
                rotation_vector = _rotation_vector(relative_rotation)
                orientation_ok = bool(
                    float(np.linalg.norm(rotation_vector)) <= orientation_tolerance
                )
                rotation_step = np.clip(
                    np.asarray(
                        rotation_vector,
                        dtype=np.float64,
                    )
                    * orientation_gain,
                    -step_clip_orientation,
                    step_clip_orientation,
                )

            if position_ok and orientation_ok:
                if force_one_step_per_frame:
                    raw_zero = np.zeros(
                        (
                            3
                            if controller_name == "OSC_POSITION"
                            else (6 if bool(use_orientation) else 3)
                        ),
                        dtype=np.float64,
                    )
                    done = step_once(
                        raw_zero,
                        gripper_command,
                    )
                    if done:
                        return True, True
                return True, False

            position_step = np.clip(
                np.asarray(
                    position_error,
                    dtype=np.float64,
                )
                * position_gain,
                -step_clip_position,
                step_clip_position,
            )
            if controller_name == "OSC_POSITION":
                raw_action = position_step
            else:
                if not track_orientation:
                    rotation_step = np.zeros(
                        3,
                        dtype=np.float64,
                    )
                raw_action = np.concatenate(
                    [position_step, rotation_step],
                    axis=0,
                )
            done = step_once(
                raw_action,
                gripper_command,
            )
            if done:
                return False, True
            iteration += 1
        return False, False

    previous_gripper_command = (
        float(commands[0]) if len(trajectory) > 0 else default_gripper_command
    )
    for index in range(len(trajectory)):
        if terminated:
            emit("[TERM] episode terminated early, stop stepping.")
            break

        gripper_command = float(commands[index])
        edge_flag = int(edges[index])
        drive_gripper_command = (
            float(previous_gripper_command)
            if index > 0 and edge_flag != 0
            else float(gripper_command)
        )
        steps_before = int(total_env_steps)

        target_position_world, _ = parse_pose(
            trajectory[index],
            traj_key=traj_key,
        )
        if support_top_z is not None and float(target_position_world[2]) < float(
            support_top_z + z_margin
        ):
            emit(
                f"[Z-ALERT] frame {index}: "
                f"target_z={target_position_world[2]:.4f} < "
                "support_top_z+margin="
                f"{support_top_z + z_margin:.4f}"
            )

        ok, terminated = drive_to_target(
            index,
            gripper_command=drive_gripper_command,
        )
        if not ok:
            fail_frames += 1
            emit(
                f"[WARN] frame {index}/{len(trajectory)}: "
                "cannot reach target within auto iters."
            )
            pairs = list(read_contacts(env, topk=5))
            if pairs:
                emit("[ROBOT-CONTACT-TOP5]")
                for first_name, second_name, distance in pairs:
                    emit(
                        f"    pair: {first_name} <-> {second_name} dist={distance:.6f}"
                    )

        steps_after_drive = int(total_env_steps)
        error = measure_target_error(index)
        settle_steps = 0
        if not terminated:
            if edge_flag > 0:
                settle_steps = close_hold_steps
            elif edge_flag < 0:
                settle_steps = open_hold_steps
            elif legacy_edge_hold_steps > 0:
                settle_steps = 0
            if settle_steps <= 0 and legacy_edge_hold_steps > 0 and edge_flag != 0:
                settle_steps = legacy_edge_hold_steps
            if settle_steps > 0:
                hold_gripper(
                    int(settle_steps),
                    gripper_command=gripper_command,
                )

        frame = trajectory[index]
        frame_identifier = (
            int(frame.get("frame", index)) if isinstance(frame, dict) else int(index)
        )
        checkpoint_trace.append(
            {
                "checkpoint_index": int(index),
                "frame": frame_identifier,
                "success": bool(ok),
                "terminated": bool(terminated),
                "position_error_norm": float(error["position_error_norm"]),
                "orientation_error_norm": error["orientation_error_norm"],
                "target_ref_3d": list(error["target_ref_3d"]),
                "gripper_cmd": float(gripper_command),
                "drive_gripper_cmd": float(drive_gripper_command),
                "gripper_edge_flag": int(edge_flag),
                "drive_env_steps": int(steps_after_drive - steps_before),
                "settle_env_steps": int(total_env_steps - steps_after_drive),
                "env_steps_for_checkpoint": int(total_env_steps - steps_before),
            }
        )
        frame_env_steps.append(int(total_env_steps) - steps_before)
        previous_gripper_command = float(gripper_command)

    return {
        "controller": controller_name,
        "traj_key": str(traj_key),
        "input_ref_frame": input_reference_frame,
        "pos_tol": float(position_tolerance),
        "ori_tol": float(orientation_tolerance),
        "num_frames": int(len(trajectory)),
        "total_env_steps": int(total_env_steps),
        "fail_frames": int(fail_frames),
        "terminated": bool(terminated),
        "frame_env_steps": frame_env_steps,
        "checkpoint_trace": checkpoint_trace,
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


def _default_metrics_functions() -> tuple[
    Callable[..., dict[str, Any]],
    Callable[..., dict[str, str]],
    Callable[[dict[str, Any]], dict[str, Any]],
]:
    from dream_exe.evaluation.execution import (
        build_exec_metrics,
        metrics_summary_fields,
        save_exec_metrics,
    )

    return (
        build_exec_metrics,
        save_exec_metrics,
        metrics_summary_fields,
    )


def publish_frame_trajectory_outputs(
    result: Mapping[str, Any],
    *,
    output_dir: str | os.PathLike[str],
    uid: str = "",
    traj_path: str = "",
    save_video: bool = True,
    json_writer: Callable[[Any, str], Any] | None = None,
    manifest_updater: Callable[..., str] | None = None,
    metrics_builder: Callable[..., dict[str, Any]] | None = None,
    metrics_writer: Callable[..., Mapping[str, str]] | None = None,
    metrics_summary_builder: (
        Callable[[dict[str, Any]], Mapping[str, Any]] | None
    ) = None,
) -> dict[str, Any]:
    """Publish the current frame executor trace, metrics, summary, and assets."""

    output_path = Path(output_dir).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    write_json = _save_json if json_writer is None else json_writer
    update_manifest = (
        update_exec_assets_manifest if manifest_updater is None else manifest_updater
    )
    if (
        metrics_builder is None
        or metrics_writer is None
        or metrics_summary_builder is None
    ):
        (
            default_builder,
            default_writer,
            default_summary_builder,
        ) = _default_metrics_functions()
        metrics_builder = (
            default_builder if metrics_builder is None else metrics_builder
        )
        metrics_writer = default_writer if metrics_writer is None else metrics_writer
        metrics_summary_builder = (
            default_summary_builder
            if metrics_summary_builder is None
            else metrics_summary_builder
        )

    execution_paths = execution_artifact_paths(output_path)
    checkpoint_trace_path = execution_paths["checkpoint_trace"].as_posix()
    checkpoint_payload = {
        "meta": {
            "source": "frame_traj_executor",
            "uid": str(uid),
            "traj_path": str(traj_path),
            "controller": result["controller"],
            "traj_key": result["traj_key"],
            "input_ref_frame": result["input_ref_frame"],
            "pos_tol": float(result["pos_tol"]),
            "ori_tol": float(result["ori_tol"]),
        },
        "checkpoints": result["checkpoint_trace"],
    }
    write_json(checkpoint_payload, checkpoint_trace_path)

    frame_env_steps = list(result["frame_env_steps"])
    position_errors = [
        float(item["position_error_norm"])
        for item in result["checkpoint_trace"]
        if item.get("position_error_norm", None) is not None
    ]
    video_path = execution_paths["execution_video"]
    summary = {
        "uid": str(uid),
        "output_dir": output_path.as_posix(),
        "video_path": (
            video_path.as_posix() if save_video and video_path.exists() else None
        ),
        "checkpoint_trace_path": checkpoint_trace_path,
        "executed": True,
        "executor": "frame_traj",
        "controller": result["controller"],
        "traj_key": result["traj_key"],
        "pos_tol": float(result["pos_tol"]),
        "ori_tol": float(result["ori_tol"]),
        "max_correction_steps": None,
        "must_reach_min_correction_steps": None,
        "num_frames": int(result["num_frames"]),
        "num_checkpoints": int(result["num_frames"]),
        "checkpoints_evaluated": int(len(result["checkpoint_trace"])),
        "env_steps": int(result["total_env_steps"]),
        "env_steps_per_checkpoint_mean": (
            float(np.mean(frame_env_steps)) if frame_env_steps else None
        ),
        "env_steps_per_checkpoint_median": (
            float(np.median(frame_env_steps)) if frame_env_steps else None
        ),
        "env_steps_per_checkpoint_max": (
            int(np.max(frame_env_steps)) if frame_env_steps else None
        ),
        "fail_frames": int(result["fail_frames"]),
        "checkpoint_successes": int(
            len(result["checkpoint_trace"]) - int(result["fail_frames"])
        ),
        "checkpoint_failures": int(result["fail_frames"]),
        "checkpoint_max_position_error_norm": (
            max(position_errors) if position_errors else None
        ),
        "terminated_early": bool(result["terminated"]),
    }
    metrics = dict(
        metrics_builder(
            uid=str(uid),
            output_dir=output_path.as_posix(),
            executor="frame_traj",
            summary=summary,
            checkpoint_trace=result["checkpoint_trace"],
            traj_path=str(traj_path),
        )
    )
    metric_paths = dict(metrics_writer(metrics, output_path.as_posix()))
    metrics["exec_metrics_path"] = metric_paths["exec_metrics_json"]
    summary.update(dict(metrics_summary_builder(metrics)))

    summary_path = execution_paths["exec_summary"].as_posix()
    write_json(summary, summary_path)
    update_manifest(
        output_path.as_posix(),
        "execution",
        {
            "dir": output_path.as_posix(),
            "exec_video": (video_path.as_posix() if save_video else None),
            "checkpoint_trace_json": checkpoint_trace_path,
            "action_trace_json": None,
            "exec_summary_json": summary_path,
            "exec_metrics_json": metric_paths["exec_metrics_json"],
            "exec_metrics_per_frame_csv": metric_paths["exec_metrics_per_frame_csv"],
        },
    )
    summary["summary_path"] = summary_path
    return summary


def execute_frame_trajectory_explicit(
    *,
    env: Any,
    trajectory: list[dict[str, Any]],
    simulator_config: Mapping[str, Any],
    execution_config: Mapping[str, Any] | None,
    output_dir: str | os.PathLike[str],
    traj_key: str = "",
    uid: str = "",
    traj_path: str = "",
    robot: Any | None = None,
    part_ctrl: Any | None = None,
    arm_part: str = "",
    gripper_part: str | None = None,
    gripper_commands: Sequence[float] | np.ndarray | None = None,
    gripper_edge_flags: Sequence[int] | np.ndarray | None = None,
    ref_site_name: str = "",
    tcp_site_name: str = "",
    ctrl_ref_site_name: str = "",
    camera_name: str = "",
    frame_size: tuple[int, int] | None = None,
    pose_reader: Callable[..., Any] | None = None,
    contact_reader: Callable[..., Sequence[Any]] | None = None,
    support_height_reader: Callable[[Any], float | None] | None = None,
    frame_recorder: Any | None = None,
    json_writer: Callable[[Any, str], Any] | None = None,
    manifest_updater: Callable[..., str] | None = None,
    metrics_builder: Callable[..., dict[str, Any]] | None = None,
    metrics_writer: Callable[..., Mapping[str, str]] | None = None,
    metrics_summary_builder: (
        Callable[[dict[str, Any]], Mapping[str, Any]] | None
    ) = None,
    logger: Callable[[str], Any] | None = print,
    close_env: bool = True,
    perform_warm_start: bool = True,
    knobs: RobustKnobs | None = None,
) -> dict[str, Any]:
    """Warm, replay, close, encode, and publish one explicit frame trajectory."""

    output_text = os.fspath(output_dir)
    if not str(output_text).strip():
        raise ValueError("output_dir is required")
    output_path = Path(output_text).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    execution_paths = execution_artifact_paths(output_path)

    inputs, execution, runtime = _config_sections(execution_config)
    save_video = bool(
        _setting(
            runtime,
            "save_video",
            _setting(execution, "save_video", True),
        )
    )
    video_path = execution_paths["execution_video"]
    if not save_video and video_path.exists():
        video_path.unlink()
    stale_action_trace = execution_paths["action_trace"]
    if stale_action_trace.exists():
        stale_action_trace.unlink()

    runtime_traj_key = str(traj_key or inputs.get("traj_key", "") or "eef_controller")
    max_steps = int(
        _setting(
            runtime,
            "max_steps",
            _setting(execution, "max_steps", -1),
        )
    )
    frames = _validate_trajectory(
        trajectory,
        traj_key=runtime_traj_key,
        max_steps=max_steps,
    )

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

    controller_name = str(_setting(execution, "controller", "OSC_POSITION"))
    use_orientation = bool(_setting(execution, "use_ori", False))
    default_gripper_command = float(_setting(execution, "gripper", 0.0))
    warm_start_steps = int(_setting(execution, "warm_start_steps", 10))
    if perform_warm_start:
        warm_start_hold(
            env=env,
            robot=runtime_robot,
            arm_part=runtime_arm_part,
            gripper_part=runtime_gripper_part,
            part_ctrl=runtime_part_ctrl,
            controller_name=controller_name,
            use_ori=bool(use_orientation),
            gripper=default_gripper_command,
            steps=warm_start_steps,
        )

    commands, edges = _aligned_gripper_schedule(
        frame_count=len(frames),
        default_command=default_gripper_command,
        commands=gripper_commands,
        edge_flags=gripper_edge_flags,
    )
    input_reference_frame = str(
        _setting(
            execution,
            "input_ref_frame",
            "world",
        )
        or "world"
    )
    rotation_world_base = None
    translation_world_base = None
    if input_reference_frame == "base":
        derived = dict(simulator_config.get("derived", {}) or {})
        derived_eef = dict(derived.get("eef", {}) or {})
        transform = derived_eef.get("X_wb", None)
        if transform is None:
            raise RuntimeError(
                "input_ref_frame=base requires "
                "cfg['derived']['eef']['X_wb'] (base->world)."
            )
        rotation_world_base, translation_world_base = parse_X_wb(transform)

    read_support_height = (
        find_support_top_z
        if support_height_reader is None
        else support_height_reader
    )
    support_top_z = read_support_height(env)

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
    runtime_ref_site = ref_site_name or ctrl_ref_site or tcp_site

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

    result = execute_frame_trajectory_loop(
        env=env,
        robot=runtime_robot,
        part_ctrl=runtime_part_ctrl,
        arm_part=runtime_arm_part,
        gripper_part=runtime_gripper_part,
        trajectory=frames,
        traj_key=runtime_traj_key,
        execution_config=execution_config,
        gripper_commands=commands,
        gripper_edge_flags=edges,
        ref_site_name=runtime_ref_site,
        rotation_world_base=rotation_world_base,
        translation_world_base=translation_world_base,
        support_top_z=support_top_z,
        pose_reader=pose_reader,
        frame_capture=recorder.grab,
        contact_reader=contact_reader,
        logger=logger,
        knobs=knobs,
    )
    if close_env:
        env.close()
    if save_video:
        policy_hz = int(
            _setting(
                runtime,
                "policy_hz",
                _setting(execution, "policy_hz", 20),
            )
        )
        recorder.save(
            video_path.as_posix(),
            fps=policy_hz,
        )

    return publish_frame_trajectory_outputs(
        result,
        output_dir=output_path.as_posix(),
        uid=str(uid),
        traj_path=str(traj_path),
        save_video=save_video,
        json_writer=json_writer,
        manifest_updater=manifest_updater,
        metrics_builder=metrics_builder,
        metrics_writer=metrics_writer,
        metrics_summary_builder=metrics_summary_builder,
    )


__all__ = [
    "RobustKnobs",
    "execute_frame_trajectory_explicit",
    "execute_frame_trajectory_loop",
    "publish_frame_trajectory_outputs",
]
