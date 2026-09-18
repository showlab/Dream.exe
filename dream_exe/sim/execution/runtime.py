"""Explicit outer runtime for one saved-simulator execution.

This module composes the current execution critical path without resolving a
bench sample, UID, run key, config root, or output root.  The caller supplies
the saved simulator config, execution config, concrete input or input path,
output directory, and may inject every simulator-facing dependency.

The lifecycle intentionally matches the current executors:

``create -> reset -> restore -> controller -> warm start -> engine -> close``

The concrete engines close only after a successful loop.  Setup or engine
errors therefore propagate without an implicit ``close``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
import os
from pathlib import Path
from typing import Any

import numpy as np

from dream_exe.artifacts.layout import execution_artifact_paths

from .action_trace import (
    execute_action_trace_explicit,
    validate_action_trace_preflight,
    warm_start_hold,
)
from ..runtime.controller import (
    controller_reference_site_name_from_config,
    get_ref_site_name_from_controller_cfg,
    load_default_controller_config,
    pre_step_max_from_controller_cfg,
    robot_name_from_config,
    set_arm_controller,
    tcp_site_name_from_config,
)
from ..runtime.environment import create_env, preflight_robocasa_runtime
from .config import normalize_execution_config
from .inputs import (
    build_gripper_schedule_from_sidecar,
    estimate_horizon_steps_from_traj,
    load_action_stream,
    load_traj_stream,
    parse_pose_from_entry,
    traj_motion_stats,
)
from .frame_trajectory import (
    RobustKnobs,
    execute_frame_trajectory_explicit,
)
from ..frozen.restore import restore_env_from_json
from ..runtime.action import infer_arm_and_gripper_parts


def _raw_root(config: Mapping[str, Any]) -> dict[str, Any]:
    if "raw" in config:
        return dict(config.get("raw", {}) or {})
    return dict(config)


def _camera_name_from_root(root: Mapping[str, Any]) -> str:
    render_camera_name = root.get("render_camera_name", None)
    if render_camera_name:
        return str(render_camera_name)
    camera = root.get("camera", {})
    camera = camera if isinstance(camera, Mapping) else {}
    return str(camera.get("name", "agentview"))


def _frame_size_from_root(root: Mapping[str, Any]) -> tuple[int, int]:
    camera = root.get("camera", {})
    camera = camera if isinstance(camera, Mapping) else {}
    return (
        int(camera.get("width", 512)),
        int(camera.get("height", 512)),
    )


def prepare_execution_env_kwargs(
    simulator_config: Mapping[str, Any],
    *,
    horizon: int,
    render: bool,
    save_video: bool,
    active_camera_name: str | None = None,
) -> dict[str, Any]:
    """Build the current execution-time environment kwargs from saved config.

    An explicit ``active_camera_name`` replaces the current bench-side scene
    override lookup.  No root reference or override JSON is discovered here.
    """

    root = _raw_root(simulator_config)
    saved_env_kwargs = dict(root.get("env_kwargs", {}) or {})

    backend = (
        str(saved_env_kwargs.get("backend", "robosuite") or "robosuite").strip().lower()
    )
    env_name = saved_env_kwargs.get(
        "env_name",
        root.get("env_name", None),
    )
    robots = saved_env_kwargs.get(
        "robots",
        root.get("robot_names", None),
    )
    if isinstance(robots, list) and len(robots) == 1:
        robots = robots[0]

    bootstrap_camera_name = str(
        saved_env_kwargs.get("camera_name", None)
        or _camera_name_from_root(root)
        or "agentview"
    ).strip()
    render_camera_name = bootstrap_camera_name
    scene_restore = dict(root.get("scene_restore", {}) or {})
    if (
        str(scene_restore.get("backend", "") or "").strip().lower()
        == "robocasa_frozen"
    ):
        render_camera_name = str(
            scene_restore.get(
                "active_camera_name",
                render_camera_name,
            )
            or render_camera_name
        ).strip()
    if active_camera_name is not None and str(active_camera_name).strip():
        render_camera_name = str(active_camera_name).strip()

    if "frame_size" in saved_env_kwargs and saved_env_kwargs["frame_size"] is not None:
        saved_frame_size = saved_env_kwargs["frame_size"]
        frame_size = (
            int(saved_frame_size[0]),
            int(saved_frame_size[1]),
        )
    else:
        frame_size = _frame_size_from_root(root)

    effective_render = bool(render)
    use_camera_obs = False
    camera_depths = False
    camera_segmentations = None
    offscreen = bool(
        save_video
        or effective_render
        or use_camera_obs
        or camera_depths
        or camera_segmentations
    )

    return {
        "backend": backend,
        "env_name": env_name,
        "robots": robots,
        "camera_name": bootstrap_camera_name,
        "active_camera_name": render_camera_name,
        "render_camera_name": render_camera_name,
        "camera_names": saved_env_kwargs.get("camera_names", None),
        "camera_widths": saved_env_kwargs.get(
            "camera_widths",
            None,
        ),
        "camera_heights": saved_env_kwargs.get(
            "camera_heights",
            None,
        ),
        "frame_size": frame_size,
        "num_steps": int(horizon),
        "seed": saved_env_kwargs.get("seed", None),
        "render": effective_render,
        "offscreen": offscreen,
        "use_camera_obs": use_camera_obs,
        "camera_depths": camera_depths,
        "camera_segmentations": camera_segmentations,
        "extra_make_kwargs": dict(saved_env_kwargs.get("extra_make_kwargs", {}) or {}),
    }


def _execution_sites(
    simulator_config: Mapping[str, Any],
    *,
    tcp_site_name: str,
    ctrl_ref_site_name: str,
) -> tuple[str, str, str]:
    config = dict(simulator_config)
    tcp_site = (
        tcp_site_name_from_config(
            config,
            override=str(tcp_site_name or ""),
        )
        or ""
    )
    ctrl_ref_site = (
        controller_reference_site_name_from_config(
            config,
            override=str(ctrl_ref_site_name or ""),
        )
        or ""
    )
    if not ctrl_ref_site and tcp_site:
        ctrl_ref_site = tcp_site

    raw = dict(config.get("raw", {}) or {})
    raw_eef = dict(raw.get("eef", {}) or {})
    derived = dict(config.get("derived", {}) or {})
    derived_eef = dict(derived.get("eef", {}) or {})
    drive_site = str(
        raw_eef.get("drive_site_name", "")
        or derived_eef.get("execution_site_name", "")
        or ctrl_ref_site
        or tcp_site
        or ""
    )
    return tcp_site, ctrl_ref_site, drive_site


def prepare_execution_controller(
    simulator_config: Mapping[str, Any],
    *,
    controller_name: str,
    policy_hz: int,
    ctrl_ref_site_name: str,
    input_ref_frame: str,
    use_ori: bool,
    tcp_site_name: str,
    controller_config_loader: Callable[..., dict[str, Any]] | None = None,
) -> tuple[dict[str, Any], str]:
    """Build the current controller config and effective reference site."""

    config = dict(simulator_config)
    raw = dict(config.get("raw", {}) or {})
    scene_restore = dict(raw.get("scene_restore", {}) or {})
    bootstrap_env_kwargs = dict(scene_restore.get("bootstrap_env_kwargs", {}) or {})
    scene_controller_config = bootstrap_env_kwargs.get(
        "controller_configs",
        None,
    )
    if isinstance(scene_controller_config, dict) and len(scene_controller_config) > 0:
        base_config = copy.deepcopy(scene_controller_config)
    else:
        loader = (
            load_default_controller_config
            if controller_config_loader is None
            else controller_config_loader
        )
        base_config = loader(
            robot_name=robot_name_from_config(config),
        )

    controller_config = set_arm_controller(
        base_config,
        controller_type=str(controller_name),
        policy_hz=int(policy_hz),
        ref_site_name=str(ctrl_ref_site_name or ""),
        input_ref_frame=str(input_ref_frame),
        use_ori=bool(use_ori),
        arm_name="right",
    )
    ref_site_name = (
        get_ref_site_name_from_controller_cfg(
            controller_config,
            arm_name="right",
        )
        or str(ctrl_ref_site_name or "")
        or str(tcp_site_name or "")
    )
    return controller_config, ref_site_name


def action_execution_horizon(
    simulator_config: Mapping[str, Any],
    *,
    action_payload: Mapping[str, Any],
    warm_start_steps: int,
    max_correction_steps: int,
) -> int:
    """Return the current action executor's maximum environment horizon."""

    raw = dict(simulator_config.get("raw", {}) or {})
    saved_env_kwargs = dict(raw.get("env_kwargs", {}) or {})
    configured_horizon = int(
        raw.get(
            "horizon",
            saved_env_kwargs.get("num_steps", 200),
        )
    )
    steps = action_payload.get("steps", None)
    checkpoints = action_payload.get("checkpoints", None)
    if not isinstance(steps, list) or len(steps) == 0:
        raise ValueError("action steps missing or empty")
    if not isinstance(checkpoints, list) or len(checkpoints) == 0:
        raise ValueError("action checkpoints missing or empty")
    required_horizon = (
        int(warm_start_steps)
        + len(steps)
        + int(max_correction_steps) * len(checkpoints)
        + 50
    )
    return max(configured_horizon, required_horizon)


def frame_execution_horizon(
    simulator_config: Mapping[str, Any],
    *,
    trajectory: list[dict[str, Any]],
    traj_key: str,
    controller_config: dict[str, Any],
    knobs: RobustKnobs | None = None,
) -> int:
    """Return the current frame executor's maximum environment horizon."""

    active_knobs = RobustKnobs() if knobs is None else knobs
    raw = dict(simulator_config.get("raw", {}) or {})
    saved_env_kwargs = dict(raw.get("env_kwargs", {}) or {})
    configured_horizon = int(
        raw.get(
            "horizon",
            saved_env_kwargs.get("num_steps", 200),
        )
    )
    frame_count = len(trajectory)
    if frame_count == 0:
        raise ValueError(f"traj[{traj_key}] is empty or not a list")
    estimated_horizon = estimate_horizon_steps_from_traj(
        trajectory,
        traj_key=str(traj_key),
        T=frame_count,
        step_max_pos=pre_step_max_from_controller_cfg(controller_config),
        parse_pose_fn=parse_pose_from_entry,
        safety=2.0,
    )
    cap_horizon = int(frame_count * active_knobs.MAX_ITERS_CAP + 50)
    return max(
        configured_horizon,
        int(estimated_horizon),
        cap_horizon,
    )


def _path_input(
    explicit_path: str | os.PathLike[str] | None,
    configured_path: Any,
) -> str:
    if explicit_path is not None:
        return os.fspath(explicit_path)
    return str(configured_path or "")


def _action_metric_trajectory_path(
    requested_path: str,
    action_payload: Mapping[str, Any],
) -> str:
    """Resolve the EE trajectory consumed by action-execution metrics.

    Action execution is driven by ``action.json``, but trajectory evaluation
    still needs an explicit EE trajectory. Bench callers may pass a colocated
    ``union_traj.json``; in that case use ``ee_traj.json`` or the action's
    recorded EE source instead of rediscovering it from the output directory.
    """

    requested = str(requested_path or "").strip()
    meta = action_payload.get("meta", {})
    meta = meta if isinstance(meta, Mapping) else {}
    source = meta.get("source", {})
    source = source if isinstance(source, Mapping) else {}
    action_ee_path = str(source.get("ee_traj_path", "") or "").strip()

    if requested:
        candidate = Path(requested).expanduser()
        if candidate.name == "union_traj.json":
            ee_candidate = candidate.with_name("ee_traj.json")
            if ee_candidate.is_file():
                return ee_candidate.resolve().as_posix()
            if action_ee_path:
                return action_ee_path
        else:
            return requested
    return action_ee_path


def _prepare_output_directory(
    output_path: Path,
    *,
    mode: str,
    save_video: bool,
    save_action_trace: bool,
) -> None:
    output_path.mkdir(parents=True, exist_ok=True)
    execution_paths = execution_artifact_paths(output_path)
    stale_video = execution_paths["execution_video"]
    if not bool(save_video) and stale_video.exists():
        stale_video.unlink()
    stale_action_trace = execution_paths["action_trace"]
    if (
        mode == "frame_traj" or not bool(save_action_trace)
    ) and stale_action_trace.exists():
        stale_action_trace.unlink()


def _preflight_concrete_robocasa_runtime(
    simulator_config: Mapping[str, Any],
    *,
    env_factory: Callable[..., Any] | None,
    robocasa_source_root: str | os.PathLike[str] | None = None,
) -> None:
    """Check the concrete RoboCasa runtime before execution can write."""

    if env_factory is not None:
        return
    root = _raw_root(simulator_config)
    saved_env_kwargs = dict(root.get("env_kwargs", {}) or {})
    backend = (
        str(saved_env_kwargs.get("backend", "robosuite") or "robosuite").strip().lower()
    )
    if backend == "robocasa":
        if robocasa_source_root is None:
            preflight_robocasa_runtime()
        else:
            preflight_robocasa_runtime(
                robocasa_source_root=robocasa_source_root,
            )


def _action_runtime_contract(
    normalized_config: dict[str, Any],
    action_payload: Mapping[str, Any],
) -> tuple[dict[str, Any], str, bool, int, str]:
    effective_config = copy.deepcopy(normalized_config)
    execution = effective_config["execution"]
    runtime = effective_config["runtime"]
    planner = dict(action_payload.get("meta", {}).get("planner", {}) or {})
    action_space = dict(
        action_payload.get("meta", {}).get(
            "action_space",
            {},
        )
        or {}
    )

    controller_name = str(execution["controller"])
    planner_controller = str(planner.get("controller", "") or "").strip()
    if planner_controller:
        controller_name = planner_controller
    use_orientation = (
        bool(action_space.get("has_orientation", False))
        and controller_name != "OSC_POSITION"
    )
    if not use_orientation and controller_name == "OSC_POSE":
        controller_name = "OSC_POSITION"

    policy_hz = int(runtime["policy_hz"])
    if int(planner.get("policy_hz", 0)) > 0:
        policy_hz = int(planner["policy_hz"])
    reference_frame = str(action_space.get("reference_frame", "base") or "base")

    execution["controller"] = controller_name
    execution["use_ori"] = bool(use_orientation)
    runtime["policy_hz"] = int(policy_hz)
    return (
        effective_config,
        controller_name,
        bool(use_orientation),
        int(policy_hz),
        reference_frame,
    )


def _prepare_runtime_environment(
    *,
    simulator_config: Mapping[str, Any],
    env_kwargs: Mapping[str, Any],
    controller_config: dict[str, Any],
    policy_hz: int,
    controller_name: str,
    use_ori: bool,
    gripper: float,
    warm_start_steps: int,
    env_factory: Callable[..., Any] | None,
    restorer: Callable[..., Any] | None,
    scene_restore_options: Mapping[str, Any] | None,
    robocasa_source_root: str | os.PathLike[str] | None,
) -> tuple[Any, Any, str, str | None, Any]:
    make_environment = create_env if env_factory is None else env_factory
    restore_environment = restore_env_from_json if restorer is None else restorer
    factory_kwargs = {
        key: value
        for key, value in dict(env_kwargs).items()
        if key not in {"active_camera_name", "render_camera_name"}
    }
    factory_kwargs["controller_cfg"] = controller_config
    factory_kwargs["policy_hz"] = int(policy_hz)
    if robocasa_source_root is not None:
        factory_kwargs["robocasa_source_root"] = robocasa_source_root

    env = make_environment(**factory_kwargs)
    env.reset()
    restore_kwargs: dict[str, Any] = {
        "zero_vel": False,
        "verify": True,
    }
    if scene_restore_options is not None:
        restore_kwargs["scene_restore_options"] = dict(scene_restore_options)
    restore_environment(
        env,
        simulator_config,
        **restore_kwargs,
    )

    robot = env.robots[0]
    composite_controller = getattr(
        robot,
        "composite_controller",
        None,
    ) or getattr(robot, "controller", None)
    if composite_controller is None:
        raise RuntimeError(
            "Cannot find robot controller "
            "(neither composite_controller nor controller)."
        )
    arm_part, gripper_part, _ = infer_arm_and_gripper_parts(robot)
    part_controller = (
        composite_controller.get_controller(arm_part)
        if hasattr(composite_controller, "get_controller")
        else composite_controller
    )
    warm_start_hold(
        env=env,
        robot=robot,
        arm_part=arm_part,
        gripper_part=gripper_part,
        part_ctrl=part_controller,
        controller_name=str(controller_name),
        use_ori=bool(use_ori),
        gripper=float(gripper),
        steps=int(warm_start_steps),
    )
    return (
        env,
        robot,
        arm_part,
        gripper_part,
        part_controller,
    )


def execute_simulation_explicit(
    *,
    simulator_config: Mapping[str, Any],
    execution_config: Mapping[str, Any] | None,
    output_dir: str | os.PathLike[str],
    trajectory: list[dict[str, Any]] | None = None,
    action_payload: Mapping[str, Any] | None = None,
    trajectory_path: str | os.PathLike[str] | None = None,
    action_path: str | os.PathLike[str] | None = None,
    uid: str = "",
    run_key: str | None = None,
    gen_model: str | None = None,
    gripper_commands: Sequence[float] | np.ndarray | None = None,
    gripper_edge_flags: Sequence[int] | np.ndarray | None = None,
    active_camera_name: str | None = None,
    scene_restore_options: Mapping[str, Any] | None = None,
    robocasa_source_root: str | os.PathLike[str] | None = None,
    env_factory: Callable[..., Any] | None = None,
    restorer: Callable[..., Any] | None = None,
    controller_config_loader: (Callable[..., dict[str, Any]] | None) = None,
    action_loader: Callable[..., Any] | None = None,
    trajectory_loader: Callable[..., Any] | None = None,
    gripper_schedule_builder: Callable[..., Any] | None = None,
    action_runner: Callable[..., Any] | None = None,
    frame_runner: Callable[..., Any] | None = None,
    metrics_builder: (Callable[[Path], Mapping[str, Any] | None] | None) = None,
    knobs: RobustKnobs | None = None,
) -> dict[str, Any]:
    """Run one explicit action or frame-trajectory simulator execution.

    Direct path arguments override paths in ``execution_config``.  No relative
    path is rebased against a config file because no config path is accepted.
    The selected engine's summary is returned unchanged apart from its current
    metrics-path augmentation. ``run_key`` and ``gen_model`` are optional
    provenance for redirected outputs; they do not resolve any bench path.
    """

    output_text = os.fspath(output_dir)
    if not str(output_text).strip():
        raise ValueError("output_dir is required")

    source_config = dict(execution_config or {})
    requested_mode = dict(source_config.get("execution", {}) or {}).get(
        "mode", "action"
    )
    normalized = normalize_execution_config(source_config)
    inputs = normalized["input"]
    execution = normalized["execution"]
    runtime = normalized["runtime"]
    mode = str(execution["mode"])
    if mode not in {"action", "frame_traj"}:
        raise ValueError(f"Unsupported exec_mode={requested_mode}")

    resolved_trajectory_path = _path_input(
        trajectory_path,
        inputs.get("traj_path", ""),
    )
    resolved_action_path = _path_input(
        action_path,
        inputs.get("action_path", ""),
    )
    output_path = Path(output_text).expanduser().resolve()

    active_knobs = RobustKnobs() if knobs is None else knobs
    runtime_trajectory: list[dict[str, Any]] | None = None
    runtime_action_payload: Mapping[str, Any] | None = None
    effective_config = normalized
    controller_name = str(execution["controller"])
    use_orientation = bool(execution["use_ori"])
    policy_hz = int(runtime["policy_hz"])
    input_reference_frame = str(execution["input_ref_frame"])

    if mode == "action":
        if action_payload is None:
            if not str(resolved_action_path).strip():
                raise ValueError(
                    "action_path or action_payload is required for action execution"
                )
            load_actions = (
                load_action_stream if action_loader is None else action_loader
            )
            loaded_payload, _steps, _checkpoints = load_actions(resolved_action_path)
            runtime_action_payload = loaded_payload
        else:
            runtime_action_payload = action_payload
        (
            effective_config,
            controller_name,
            use_orientation,
            policy_hz,
            input_reference_frame,
        ) = _action_runtime_contract(
            normalized,
            runtime_action_payload,
        )
        validate_action_trace_preflight(
            runtime_action_payload,
            effective_config,
        )
        _preflight_concrete_robocasa_runtime(
            simulator_config,
            env_factory=env_factory,
            robocasa_source_root=robocasa_source_root,
        )
        _prepare_output_directory(
            output_path,
            mode=mode,
            save_video=bool(runtime["save_video"]),
            save_action_trace=bool(execution["save_action_trace"]),
        )
    else:
        traj_key = str(inputs["traj_key"])
        max_steps = int(runtime["max_steps"])
        _preflight_concrete_robocasa_runtime(
            simulator_config,
            env_factory=env_factory,
            robocasa_source_root=robocasa_source_root,
        )
        _prepare_output_directory(
            output_path,
            mode=mode,
            save_video=bool(runtime["save_video"]),
            save_action_trace=False,
        )
        if trajectory is None:
            if not str(resolved_trajectory_path).strip():
                raise ValueError(
                    "trajectory_path or trajectory is required for frame_traj execution"
                )
            load_trajectory = (
                load_traj_stream if trajectory_loader is None else trajectory_loader
            )
            _trajectory_payload, loaded_frames, _frame_count = load_trajectory(
                resolved_trajectory_path,
                traj_key,
                max_steps,
            )
            runtime_trajectory = loaded_frames
        else:
            if not isinstance(trajectory, list) or len(trajectory) == 0:
                raise ValueError(f"traj[{traj_key}] is empty or not a list")
            runtime_trajectory = list(trajectory)
            if max_steps > 0:
                runtime_trajectory = runtime_trajectory[:max_steps]
        if not runtime_trajectory:
            raise ValueError(f"traj[{traj_key}] is empty or not a list")
        traj_motion_stats(
            runtime_trajectory,
            traj_key=traj_key,
            T=len(runtime_trajectory),
            parse_pose_fn=parse_pose_from_entry,
        )

    tcp_site, ctrl_ref_site, drive_site = _execution_sites(
        simulator_config,
        tcp_site_name=str(execution["tcp_site_name"]),
        ctrl_ref_site_name=str(execution["ctrl_ref_site_name"]),
    )
    controller_config, ref_site_name = prepare_execution_controller(
        simulator_config,
        controller_name=controller_name,
        policy_hz=policy_hz,
        ctrl_ref_site_name=ctrl_ref_site,
        input_ref_frame=input_reference_frame,
        use_ori=use_orientation,
        tcp_site_name=tcp_site,
        controller_config_loader=controller_config_loader,
    )

    if mode == "action":
        assert runtime_action_payload is not None
        horizon = action_execution_horizon(
            simulator_config,
            action_payload=runtime_action_payload,
            warm_start_steps=int(effective_config["execution"]["warm_start_steps"]),
            max_correction_steps=int(
                effective_config["execution"]["max_correction_steps"]
            ),
        )
    else:
        assert runtime_trajectory is not None
        horizon = frame_execution_horizon(
            simulator_config,
            trajectory=runtime_trajectory,
            traj_key=str(inputs["traj_key"]),
            controller_config=controller_config,
            knobs=active_knobs,
        )

    env_kwargs = prepare_execution_env_kwargs(
        simulator_config,
        horizon=horizon,
        render=bool(runtime["render"]),
        save_video=bool(runtime["save_video"]),
        active_camera_name=active_camera_name,
    )

    env, robot, arm_part, gripper_part, part_controller = _prepare_runtime_environment(
        simulator_config=simulator_config,
        env_kwargs=env_kwargs,
        controller_config=controller_config,
        policy_hz=policy_hz,
        controller_name=controller_name,
        use_ori=use_orientation,
        gripper=float(execution["gripper"]),
        warm_start_steps=int(execution["warm_start_steps"]),
        env_factory=env_factory,
        restorer=restorer,
        scene_restore_options=scene_restore_options,
        robocasa_source_root=robocasa_source_root,
    )

    if mode == "action":
        run_action = (
            execute_action_trace_explicit if action_runner is None else action_runner
        )
        assert runtime_action_payload is not None
        metric_trajectory_path = _action_metric_trajectory_path(
            str(resolved_trajectory_path),
            runtime_action_payload,
        )
        summary = run_action(
            env=env,
            action_payload=runtime_action_payload,
            simulator_config=simulator_config,
            execution_config=effective_config,
            output_dir=output_path.as_posix(),
            uid=str(uid),
            action_path=str(resolved_action_path),
            traj_path=metric_trajectory_path,
            robot=robot,
            part_ctrl=part_controller,
            arm_part=arm_part,
            gripper_part=gripper_part,
            ref_site_name=ref_site_name,
            tcp_site_name=tcp_site,
            ctrl_ref_site_name=ctrl_ref_site,
            drive_site_name=drive_site,
            camera_name=str(env_kwargs["render_camera_name"]),
            frame_size=tuple(env_kwargs["frame_size"]),
            close_env=True,
            perform_warm_start=False,
        )
        if metrics_builder is None:
            from dream_exe.evaluation.execution import (
                build_metrics_from_exec_dir,
            )

            build_metrics = build_metrics_from_exec_dir
        else:
            build_metrics = metrics_builder
        if metrics_builder is None:
            metrics = build_metrics(
                Path(summary["output_dir"]),
                run_key=run_key,
                gen_model=gen_model,
            )
        else:
            # Preserve the established single-argument injectable seam.
            metrics = build_metrics(Path(summary["output_dir"]))
        if metrics:
            summary["exec_metrics_path"] = metrics.get(
                "exec_metrics_path",
                metrics.get("exec_metrics_json", None),
            )
        return summary

    assert runtime_trajectory is not None
    commands = gripper_commands
    edge_flags = gripper_edge_flags
    if commands is None and edge_flags is None:
        if str(resolved_trajectory_path).strip():
            build_schedule = (
                build_gripper_schedule_from_sidecar
                if gripper_schedule_builder is None
                else gripper_schedule_builder
            )
            commands, edge_flags, _schedule_metadata = build_schedule(
                traj_path=str(resolved_trajectory_path),
                T=len(runtime_trajectory),
                enable=bool(execution["enable_gripper_schedule"]),
                default_cmd=float(execution["gripper"]),
                cmd_open=float(execution["gripper_cmd_open"]),
                cmd_close=float(execution["gripper_cmd_close"]),
                cmd_hold=0.0,
                invalid_policy=str(execution["grasp_invalid_policy"]),
            )
        else:
            commands = np.full(
                (len(runtime_trajectory),),
                float(execution["gripper"]),
                dtype=np.float64,
            )
            edge_flags = np.zeros(
                (len(runtime_trajectory),),
                dtype=np.int8,
            )

    run_frame = (
        execute_frame_trajectory_explicit if frame_runner is None else frame_runner
    )
    return run_frame(
        env=env,
        trajectory=runtime_trajectory,
        simulator_config=simulator_config,
        execution_config=effective_config,
        output_dir=output_path.as_posix(),
        traj_key=str(inputs["traj_key"]),
        uid=str(uid),
        traj_path=str(resolved_trajectory_path),
        robot=robot,
        part_ctrl=part_controller,
        arm_part=arm_part,
        gripper_part=gripper_part,
        gripper_commands=commands,
        gripper_edge_flags=edge_flags,
        ref_site_name=ref_site_name,
        tcp_site_name=tcp_site,
        ctrl_ref_site_name=ctrl_ref_site,
        camera_name=str(env_kwargs["render_camera_name"]),
        frame_size=tuple(env_kwargs["frame_size"]),
        close_env=True,
        perform_warm_start=False,
        knobs=active_knobs,
    )


__all__ = [
    "action_execution_horizon",
    "execute_simulation_explicit",
    "frame_execution_horizon",
    "prepare_execution_controller",
    "prepare_execution_env_kwargs",
]
