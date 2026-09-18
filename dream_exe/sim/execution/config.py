"""Path-free execution configuration and runtime dispatch.

Concrete simulator runners are injected into :func:`dispatch_execution`. This
keeps configuration and branch behavior importable without loading RoboSuite,
RoboCasa, bench storage, or the trajectory algorithm.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...artifacts.layout import SIMULATOR_CONFIG_FILENAME

DEFAULT_EXECUTION_CONFIG_FILENAME = "execution.json"


def _defaults() -> dict[str, Any]:
    return {
        "input": {
            "traj_path": "",
            "action_path": "",
            "traj_key": "eef_controller",
        },
        "runtime": {
            "save_video": True,
            "render": False,
            "policy_hz": 20,
            "max_steps": -1,
        },
        "execution": {
            "mode": "action",
            "controller": "OSC_POSE",
            "use_ori": True,
            "input_ref_frame": "world",
            "gripper": 0.0,
            "tcp_site_name": "",
            "ctrl_ref_site_name": "",
            "pos_tol": 5e-3,
            "ori_tol": 3e-2,
            "arm_pos_gain": 2.0,
            "arm_ori_gain": 1.5,
            "warm_start_steps": 10,
            "enable_gripper_schedule": True,
            "grasp_invalid_policy": "hold",
            "gripper_cmd_open": -1.0,
            "gripper_cmd_close": 1.0,
            "hold_steps_after_gripper_change": 0,
            "hold_steps_after_gripper_close": 2,
            "hold_steps_after_gripper_open": 2,
            "enable_close_completion_gate": True,
            "enable_open_completion_gate": True,
            "close_gate_min_hold_steps": 2,
            "close_gate_max_wait_steps": 60,
            "close_gate_qpos_delta_min": 1e-4,
            "close_gate_qpos_settle_tol": 1e-4,
            "close_gate_settle_window": 3,
            "close_gate_require_non_support_contact": False,
            "close_gate_contact_settle_steps": 2,
            "close_gate_failure_policy": "continue",
            "force_one_step_per_frame": True,
            "save_action_trace": False,
            "max_correction_steps": 3,
            "must_reach_min_correction_steps": 3,
            "pose_correction_mode": "coupled",
            "position_dominate_correction_threshold_m": 0.02,
        },
    }


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def default_execution_config() -> dict[str, Any]:
    """Return a detached copy of the current execution defaults."""

    return copy.deepcopy(_defaults())


def normalize_execution_config(
    config: Mapping[str, Any] | None,
    *,
    source_config: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Normalize an in-memory execution config without resolving bench paths."""

    raw_config = dict(config or {})
    source = source_config if source_config is not None else raw_config
    source_execution = dict((source or {}).get("execution", {}) or {})
    normalized = _deep_merge(_defaults(), raw_config)

    inputs = normalized.setdefault("input", {})
    inputs["traj_path"] = str(inputs.get("traj_path", "") or "")
    inputs["action_path"] = str(inputs.get("action_path", "") or "")
    inputs["traj_key"] = str(
        inputs.get("traj_key", "eef_controller") or "eef_controller"
    )

    runtime = normalized.setdefault("runtime", {})
    runtime["save_video"] = bool(runtime.get("save_video", True))
    runtime["render"] = bool(runtime.get("render", False))
    runtime["policy_hz"] = int(runtime.get("policy_hz", 20))
    runtime["max_steps"] = int(runtime.get("max_steps", -1))

    execution = normalized.setdefault("execution", {})
    execution["mode"] = str(execution.get("mode", "action") or "action").strip().lower()
    execution["controller"] = str(
        execution.get("controller", "OSC_POSE") or "OSC_POSE"
    ).strip()
    execution["use_ori"] = bool(
        source_execution.get(
            "use_ori",
            execution["controller"] != "OSC_POSITION",
        )
    )
    if execution["controller"] == "OSC_POSITION":
        execution["use_ori"] = False
    elif execution["controller"] == "OSC_POSE":
        execution["use_ori"] = True

    execution["input_ref_frame"] = (
        str(execution.get("input_ref_frame", "world") or "world").strip().lower()
    )
    execution["gripper"] = float(execution.get("gripper", 0.0))
    execution["tcp_site_name"] = str(execution.get("tcp_site_name", "") or "")
    execution["ctrl_ref_site_name"] = str(execution.get("ctrl_ref_site_name", "") or "")
    execution["pos_tol"] = float(execution.get("pos_tol", 5e-3))
    execution["ori_tol"] = float(execution.get("ori_tol", 3e-2))
    execution["arm_pos_gain"] = float(execution.get("arm_pos_gain", 2.0))
    execution["arm_ori_gain"] = float(execution.get("arm_ori_gain", 1.5))
    execution["warm_start_steps"] = int(execution.get("warm_start_steps", 10))
    execution["enable_gripper_schedule"] = bool(
        execution.get("enable_gripper_schedule", True)
    )
    execution["grasp_invalid_policy"] = str(
        execution.get("grasp_invalid_policy", "hold") or "hold"
    ).strip()
    execution["gripper_cmd_open"] = float(execution.get("gripper_cmd_open", -1.0))
    execution["gripper_cmd_close"] = float(execution.get("gripper_cmd_close", 1.0))
    execution["hold_steps_after_gripper_change"] = int(
        execution.get("hold_steps_after_gripper_change", 0)
    )
    execution["hold_steps_after_gripper_close"] = int(
        execution.get("hold_steps_after_gripper_close", 2)
    )
    execution["hold_steps_after_gripper_open"] = int(
        execution.get("hold_steps_after_gripper_open", 2)
    )
    execution["enable_close_completion_gate"] = bool(
        execution.get("enable_close_completion_gate", True)
    )
    execution["enable_open_completion_gate"] = bool(
        execution.get("enable_open_completion_gate", True)
    )
    execution["close_gate_min_hold_steps"] = int(
        execution.get("close_gate_min_hold_steps", 2)
    )
    execution["close_gate_max_wait_steps"] = int(
        execution.get("close_gate_max_wait_steps", 60)
    )
    execution["close_gate_qpos_delta_min"] = float(
        execution.get("close_gate_qpos_delta_min", 1e-4)
    )
    execution["close_gate_qpos_settle_tol"] = float(
        execution.get("close_gate_qpos_settle_tol", 1e-4)
    )
    execution["close_gate_settle_window"] = int(
        execution.get("close_gate_settle_window", 3)
    )
    execution["close_gate_require_non_support_contact"] = bool(
        execution.get("close_gate_require_non_support_contact", False)
    )
    execution["close_gate_contact_settle_steps"] = int(
        execution.get("close_gate_contact_settle_steps", 2)
    )
    failure_policy = (
        str(execution.get("close_gate_failure_policy", "continue") or "continue")
        .strip()
        .lower()
    )
    execution["close_gate_failure_policy"] = (
        failure_policy if failure_policy in {"continue", "terminate"} else "continue"
    )
    execution["force_one_step_per_frame"] = bool(
        execution.get("force_one_step_per_frame", True)
    )
    execution["save_action_trace"] = bool(execution.get("save_action_trace", False))
    execution["max_correction_steps"] = int(execution.get("max_correction_steps", 3))
    execution["must_reach_min_correction_steps"] = int(
        execution.get("must_reach_min_correction_steps", 3)
    )
    correction_mode = (
        str(execution.get("pose_correction_mode", "coupled") or "coupled")
        .strip()
        .lower()
    )
    if correction_mode == "position_first":
        correction_mode = "position_dominate"
    execution["pose_correction_mode"] = (
        correction_mode
        if correction_mode in {"position_dominate", "coupled"}
        else "coupled"
    )
    if (
        "position_dominate_correction_threshold_m" not in source_execution
        and "position_first_correction_threshold_m" in source_execution
    ):
        threshold = source_execution.get(
            "position_first_correction_threshold_m",
            0.02,
        )
    else:
        threshold = execution.get(
            "position_dominate_correction_threshold_m",
            0.02,
        )
    execution["position_dominate_correction_threshold_m"] = float(threshold)
    execution.pop("position_first_correction_threshold_m", None)
    execution.pop("checkpoint_require_orientation", None)
    return normalized


def dispatch_execution(
    config: Mapping[str, Any] | None,
    *,
    frame_runner: Callable[..., Any],
    action_runner: Callable[..., Any],
    metrics_builder: Callable[[Path], Mapping[str, Any] | None],
    trajectory_path: str | Path | None = None,
    action_path: str | Path | None = None,
    simulator_config_path: str | Path = "",
    uid: str = "",
    output_root: str | Path = "",
    project_config_root: str | Path = "./configs",
    simulator_config_filename: str = SIMULATOR_CONFIG_FILENAME,
) -> Any:
    """Dispatch normalized execution to one concrete simulator runner.

    Paths that identify trajectory and action inputs may be provided directly;
    otherwise their normalized config values are used. Bench UID and run-key
    resolution deliberately stay outside this function.
    """

    source_config = dict(config or {})
    requested_mode = dict(source_config.get("execution", {}) or {}).get(
        "mode", "action"
    )
    normalized = normalize_execution_config(source_config)
    inputs = normalized["input"]
    runtime = normalized["runtime"]
    execution = normalized["execution"]

    traj_value = (
        inputs["traj_path"] if trajectory_path is None else str(trajectory_path)
    )
    action_value = inputs["action_path"] if action_path is None else str(action_path)

    shared = {
        "traj_path": traj_value,
        "config_path": str(simulator_config_path),
        "uid": str(uid),
        "out_root": str(output_root),
        "config_root": str(project_config_root),
        "config_filename": str(simulator_config_filename),
        "save_video": runtime["save_video"],
        "render": runtime["render"],
        "policy_hz": runtime["policy_hz"],
        "controller_name": execution["controller"],
        "gripper": execution["gripper"],
        "tcp_site_name": execution["tcp_site_name"],
        "ctrl_ref_site_name": execution["ctrl_ref_site_name"],
        "pos_tol": execution["pos_tol"],
        "ori_tol": execution["ori_tol"],
        "arm_pos_gain": execution["arm_pos_gain"],
        "arm_ori_gain": execution["arm_ori_gain"],
        "warm_start_steps": execution["warm_start_steps"],
    }

    if execution["mode"] == "frame_traj":
        return frame_runner(
            **shared,
            traj_key=inputs["traj_key"],
            max_steps=runtime["max_steps"],
            use_ori=execution["use_ori"],
            input_ref_frame=execution["input_ref_frame"],
            enable_gripper_schedule=execution["enable_gripper_schedule"],
            grasp_invalid_policy=execution["grasp_invalid_policy"],
            gripper_cmd_open=execution["gripper_cmd_open"],
            gripper_cmd_close=execution["gripper_cmd_close"],
            hold_steps_after_gripper_change=execution[
                "hold_steps_after_gripper_change"
            ],
            hold_steps_after_gripper_close=execution["hold_steps_after_gripper_close"],
            hold_steps_after_gripper_open=execution["hold_steps_after_gripper_open"],
            force_one_step_per_frame=execution["force_one_step_per_frame"],
        )

    if execution["mode"] != "action":
        raise ValueError(f"Unsupported exec_mode={requested_mode}")

    summary = action_runner(
        **shared,
        action_path=action_value,
        save_action_trace=execution["save_action_trace"],
        max_correction_steps=execution["max_correction_steps"],
        must_reach_min_correction_steps=execution["must_reach_min_correction_steps"],
        position_dominate_correction_threshold_m=execution[
            "position_dominate_correction_threshold_m"
        ],
        pose_correction_mode=execution["pose_correction_mode"],
        enable_close_completion_gate=execution["enable_close_completion_gate"],
        enable_open_completion_gate=execution["enable_open_completion_gate"],
        close_gate_min_hold_steps=execution["close_gate_min_hold_steps"],
        close_gate_max_wait_steps=execution["close_gate_max_wait_steps"],
        close_gate_qpos_delta_min=execution["close_gate_qpos_delta_min"],
        close_gate_qpos_settle_tol=execution["close_gate_qpos_settle_tol"],
        close_gate_settle_window=execution["close_gate_settle_window"],
        close_gate_require_non_support_contact=execution[
            "close_gate_require_non_support_contact"
        ],
        close_gate_contact_settle_steps=execution["close_gate_contact_settle_steps"],
        close_gate_failure_policy=execution["close_gate_failure_policy"],
    )
    metrics = metrics_builder(Path(summary["output_dir"]))
    if metrics:
        summary["exec_metrics_path"] = metrics.get(
            "exec_metrics_path",
            metrics.get("exec_metrics_json", None),
        )
    return summary


__all__ = [
    "DEFAULT_EXECUTION_CONFIG_FILENAME",
    "default_execution_config",
    "dispatch_execution",
    "normalize_execution_config",
]
