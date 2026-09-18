"""Explicit saved-action task-success replay.

This module binds the current task-success observation timing to the current implementation action
executor without resolving a UID, bench root, config root, action path, or
trajectory path.  Environment preparation, replay, success policy, and
artifact publication are independent injectable boundaries.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
import copy
import os
from typing import Any

import numpy as np

from dream_exe.evaluation.execution import (
    CURRENT_TASK_SUCCESS_SCHEMA,
    TaskSuccessCheckResult,
    TaskSuccessObservation,
    build_task_success_observation,
    normalize_task_success_payload,
)
from dream_exe.transforms import parse_X_wb

from ..execution.action_trace import (
    current_eef_pose_world,
    execute_action_trace_loop,
    validate_action_trace_preflight,
)
from ..execution.config import normalize_execution_config
from ..execution.runtime import (
    _action_runtime_contract,
    _execution_sites,
    _preflight_concrete_robocasa_runtime,
    _prepare_runtime_environment,
    action_execution_horizon,
    prepare_execution_controller,
    prepare_execution_env_kwargs,
)
from ..robocasa.task_runtime import (
    load_robocasa_task_runtime,
    prepare_robocasa_task_runtime_hooks,
)


_RUNTIME_ENV_META_KEYS = (
    "loaded_ep_meta",
    "loaded_ep_meta_post_restore",
    "task_refs_applied_post_restore",
    "obj_body_id_repair",
    "restore_skipped_set_joint_qpos",
    "sink_prefix_repair",
    "sink_runtime_patch",
)


@dataclass
class ActionTaskSuccessSession:
    """One caller-owned, restored action replay session."""

    env: Any
    robot: Any
    part_controller: Any
    arm_part: str
    gripper_part: str | None
    ref_site_name: str = ""
    rotation_world_base: np.ndarray | None = None
    translation_world_base: np.ndarray | None = None
    runtime_env_meta: dict[str, Any] = field(default_factory=dict)
    _closed: bool = field(default=False, init=False)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.env.close()


def _effective_action_config(
    action_payload: Mapping[str, Any],
    execution_config: Mapping[str, Any] | None,
) -> tuple[dict[str, Any], str, bool, int, str]:
    source = dict(execution_config or {})
    requested_mode = (
        str(
            dict(source.get("execution", {}) or {}).get(
                "mode",
                "action",
            )
        )
        .strip()
        .lower()
    )
    if requested_mode != "action":
        raise ValueError("task-success replay requires execution.mode='action'")
    normalized = normalize_execution_config(source)
    (
        effective,
        controller_name,
        use_orientation,
        policy_hz,
        reference_frame,
    ) = _action_runtime_contract(
        normalized,
        action_payload,
    )
    effective["runtime"]["render"] = False
    effective["runtime"]["save_video"] = False
    validate_action_trace_preflight(
        action_payload,
        effective,
    )
    return (
        effective,
        controller_name,
        use_orientation,
        policy_hz,
        reference_frame,
    )


def prepare_action_task_success_session(
    *,
    simulator_config: Mapping[str, Any],
    action_payload: Mapping[str, Any],
    effective_execution_config: Mapping[str, Any],
    controller_name: str,
    use_orientation: bool,
    policy_hz: int,
    reference_frame: str,
    active_camera_name: str | None = None,
    runtime_env_meta: Mapping[str, Any] | None = None,
    scene_restore_options: Mapping[str, Any] | None = None,
    robocasa_source_root: str | os.PathLike[str] | None = None,
    robocasa_task_runtime: Mapping[str, Any] | None = None,
    robocasa_task_runtime_loader: (
        Callable[..., Mapping[str, Any] | None] | None
    ) = None,
    env_factory: Callable[..., Any] | None = None,
    restorer: Callable[..., Any] | None = None,
    controller_config_loader: (Callable[..., dict[str, Any]] | None) = None,
) -> ActionTaskSuccessSession:
    """Create and restore one explicit current implementation action session.

    A setup failure closes an environment that was already created.  No bench
    discovery or artifact writer is reachable from this function; a RoboCasa
    manifest may resolve only its own explicit scene references.
    """

    _preflight_concrete_robocasa_runtime(
        simulator_config,
        env_factory=env_factory,
        robocasa_source_root=robocasa_source_root,
    )

    execution = dict(effective_execution_config.get("execution", {}) or {})
    tcp_site, controller_reference_site, _drive_site = _execution_sites(
        simulator_config,
        tcp_site_name=str(execution.get("tcp_site_name", "")),
        ctrl_ref_site_name=str(execution.get("ctrl_ref_site_name", "")),
    )
    controller_config, ref_site_name = prepare_execution_controller(
        simulator_config,
        controller_name=str(controller_name),
        policy_hz=int(policy_hz),
        ctrl_ref_site_name=controller_reference_site,
        input_ref_frame=str(reference_frame),
        use_ori=bool(use_orientation),
        tcp_site_name=tcp_site,
        controller_config_loader=controller_config_loader,
    )
    horizon = action_execution_horizon(
        simulator_config,
        action_payload=action_payload,
        warm_start_steps=int(execution.get("warm_start_steps", 10)),
        max_correction_steps=int(execution.get("max_correction_steps", 3)),
    )
    environment_kwargs = prepare_execution_env_kwargs(
        simulator_config,
        horizon=horizon,
        render=False,
        save_video=False,
        active_camera_name=active_camera_name,
    )

    rotation_world_base = None
    translation_world_base = None
    if str(reference_frame) == "base":
        derived = dict(simulator_config.get("derived", {}) or {})
        derived_eef = dict(derived.get("eef", {}) or {})
        transform = derived_eef.get("X_wb", None)
        if transform is None:
            raise RuntimeError(
                "input_ref_frame=base requires "
                "cfg['derived']['eef']['X_wb'] (base->world)."
            )
        (
            rotation_world_base,
            translation_world_base,
        ) = parse_X_wb(transform)

    task_runtime: Mapping[str, Any] | None = None
    if robocasa_task_runtime is not None:
        if not isinstance(
            robocasa_task_runtime,
            Mapping,
        ):
            raise TypeError("robocasa_task_runtime must be a mapping")
        task_runtime = copy.deepcopy(dict(robocasa_task_runtime))
    elif robocasa_task_runtime_loader is not None:
        if not callable(robocasa_task_runtime_loader):
            raise TypeError("robocasa_task_runtime_loader must be callable")
        loaded_runtime = robocasa_task_runtime_loader(
            simulator_config,
            scene_restore_options=scene_restore_options,
        )
        if loaded_runtime is not None and not isinstance(loaded_runtime, Mapping):
            raise TypeError(
                "robocasa_task_runtime_loader must return a mapping or None"
            )
        task_runtime = (
            None if loaded_runtime is None else copy.deepcopy(dict(loaded_runtime))
        )
    elif env_factory is None and restorer is None:
        task_runtime = load_robocasa_task_runtime(
            simulator_config,
            scene_restore_options=scene_restore_options,
        )

    effective_env_factory = env_factory
    effective_restorer = restorer
    task_runtime_meta: dict[str, Any] = {}
    if task_runtime is not None:
        if effective_env_factory is None:
            from ..runtime.environment import create_env

            effective_env_factory = create_env
        if effective_restorer is None:
            from ..frozen.restore import restore_env_from_json

            effective_restorer = restore_env_from_json
        (
            effective_env_factory,
            effective_restorer,
            task_runtime_meta,
        ) = prepare_robocasa_task_runtime_hooks(
            task_runtime,
            env_factory=effective_env_factory,
            restorer=effective_restorer,
        )

    created_environments: list[Any] = []

    def capture_environment(**kwargs: Any) -> Any:
        if effective_env_factory is None:
            from ..runtime.environment import create_env

            environment = create_env(**kwargs)
        else:
            environment = effective_env_factory(**kwargs)
        created_environments.append(environment)
        return environment

    try:
        (
            env,
            robot,
            arm_part,
            gripper_part,
            part_controller,
        ) = _prepare_runtime_environment(
            simulator_config=simulator_config,
            env_kwargs=environment_kwargs,
            controller_config=controller_config,
            policy_hz=int(policy_hz),
            controller_name=str(controller_name),
            use_ori=bool(use_orientation),
            gripper=float(execution.get("gripper", 0.0)),
            warm_start_steps=int(execution.get("warm_start_steps", 10)),
            env_factory=capture_environment,
            restorer=effective_restorer,
            scene_restore_options=scene_restore_options,
            robocasa_source_root=robocasa_source_root,
        )
    except BaseException:
        for environment in reversed(created_environments):
            try:
                environment.close()
            except Exception:
                pass
        raise

    selected_runtime_meta = dict(runtime_env_meta or {})
    selected_runtime_meta.update(task_runtime_meta)
    return ActionTaskSuccessSession(
        env=env,
        robot=robot,
        part_controller=part_controller,
        arm_part=str(arm_part),
        gripper_part=gripper_part,
        ref_site_name=str(ref_site_name),
        rotation_world_base=rotation_world_base,
        translation_world_base=translation_world_base,
        runtime_env_meta=selected_runtime_meta,
    )


def observe_environment_task_success(
    env: Any,
    *,
    task_name: str,
    evidence: Mapping[str, Any] | None = None,
    checker: Callable[[Any], Any] | None = None,
    evidence_reader: (Callable[[Any], Mapping[str, Any] | None] | None) = None,
) -> TaskSuccessObservation:
    """Evaluate the generic current ``env._check_success`` boundary."""

    checker_error = None
    raw_success: Any = False
    if checker is None:
        check_success = getattr(env, "_check_success", None)
        if check_success is None:
            checker_error = "env_missing__check_success"
        else:
            try:
                raw_success = check_success()
            except Exception as error:
                checker_error = f"{type(error).__name__}: {error}"
    else:
        try:
            checker_result = checker(env)
            if isinstance(
                checker_result,
                TaskSuccessCheckResult,
            ):
                raw_success = checker_result.raw_success
                checker_error = checker_result.error
            else:
                raw_success = checker_result
        except Exception as error:
            checker_error = f"{type(error).__name__}: {error}"

    combined_evidence = copy.deepcopy(dict(evidence or {}))
    if evidence_reader is not None:
        supplied = evidence_reader(env)
        if supplied is not None:
            if not isinstance(supplied, Mapping):
                raise TypeError("evidence_reader must return a mapping or None")
            combined_evidence.update(copy.deepcopy(dict(supplied)))
    try:
        environment_components = getattr(
            env,
            "_last_success_components",
            None,
        )
        if isinstance(environment_components, dict) and environment_components:
            if "task_success_components" in combined_evidence:
                combined_evidence["task_success_components_env"] = copy.deepcopy(
                    environment_components
                )
            else:
                combined_evidence["task_success_components"] = copy.deepcopy(
                    environment_components
                )
    except Exception:
        pass
    return build_task_success_observation(
        task_name,
        raw_success=raw_success,
        checker_error=checker_error,
        meta=combined_evidence,
    )


def _task_state(env: Any, task_name: str) -> dict[str, Any]:
    state: dict[str, Any] = {}
    if str(task_name or "").strip():
        state["task_name"] = str(task_name)
    for attribute in ("behavior", "init_sink_mode"):
        if hasattr(env, attribute):
            try:
                state[attribute] = getattr(env, attribute)
            except Exception:
                pass
    if hasattr(env, "sink"):
        try:
            sink = getattr(env, "sink")
            handle_state = sink.get_handle_state(env=env)
            if isinstance(handle_state, Mapping):
                state["sink_handle_state"] = {
                    key: handle_state.get(key)
                    for key in (
                        "spout_ori",
                        "spout_joint",
                        "water_on",
                        "handle_joint",
                        "water_pressure",
                    )
                    if key in handle_state
                }
        except Exception as error:
            state["sink_handle_state_error"] = f"{type(error).__name__}: {error}"
    return state


def set_qpos_qvel_state(
    env: Any,
    qpos: Any,
    qvel: Any,
    *,
    timestamp_s: float = 0.0,
) -> str | None:
    """Inject one explicit simulator state using the current flat-state ABI.

    The return value intentionally mirrors the current diagnostic replay
    behavior: malformed vector sizes and simulator failures are recorded per
    state instead of being mistaken for task failure.
    """

    simulation = env.sim
    expected_qpos = int(getattr(simulation.model, "nq", 0))
    expected_qvel = int(getattr(simulation.model, "nv", 0))
    try:
        qpos_array = np.asarray(
            qpos,
            dtype=np.float64,
        ).reshape(-1)
        qvel_array = np.asarray(
            qvel,
            dtype=np.float64,
        ).reshape(-1)
    except Exception:
        return "missing_qpos_or_qvel"
    if qpos_array.shape[0] != expected_qpos:
        return f"qpos_size_mismatch: got={qpos_array.shape[0]} expected={expected_qpos}"
    if qvel_array.shape[0] != expected_qvel:
        return f"qvel_size_mismatch: got={qvel_array.shape[0]} expected={expected_qvel}"
    flattened = np.concatenate(
        [
            np.asarray(
                [float(timestamp_s)],
                dtype=np.float64,
            ),
            qpos_array,
            qvel_array,
        ]
    )
    try:
        simulation.set_state_from_flattened(flattened)
        simulation.forward()
    except Exception as error:
        return f"set_state_failed: {type(error).__name__}: {error}"
    return None


def evaluate_saved_state_task_success(
    *,
    simulator_config: Mapping[str, Any],
    execution_config: Mapping[str, Any] | None,
    action_payload: Mapping[str, Any],
    state_trace: list[Mapping[str, Any]],
    uid: str,
    run_key: str,
    task_name: str,
    state_path: str,
    gen_model: str | None = None,
    state_generation: str = "qpos_qvel_injection",
    trajectory_summary: Mapping[str, Any] | None = None,
    trajectory_fallback: Mapping[str, Any] | None = None,
    task_success_traj_fallback_disabled: bool = True,
    runtime_env_meta: Mapping[str, Any] | None = None,
    active_camera_name: str | None = None,
    scene_restore_options: Mapping[str, Any] | None = None,
    robocasa_source_root: str | os.PathLike[str] | None = None,
    robocasa_task_runtime: Mapping[str, Any] | None = None,
    robocasa_task_runtime_loader: (
        Callable[..., Mapping[str, Any] | None] | None
    ) = None,
    env_factory: Callable[..., Any] | None = None,
    restorer: Callable[..., Any] | None = None,
    controller_config_loader: (Callable[..., dict[str, Any]] | None) = None,
    session_factory: (Callable[[], ActionTaskSuccessSession] | None) = None,
    state_setter: Callable[..., str | None] | None = None,
    pose_reader: Callable[..., Any] | None = None,
    success_checker: Callable[[Any], Any] | None = None,
    success_evidence_reader: (Callable[[Any], Mapping[str, Any] | None] | None) = None,
    success_rule: (Callable[[Any], TaskSuccessObservation] | None) = None,
    artifact_sink: (Callable[[Mapping[str, Any]], Any] | None) = None,
) -> dict[str, Any]:
    """Replay an explicit qpos/qvel trace as diagnostic success evidence.

    This callable owns no UID, bench, or external-policy path resolution.  It
    creates no artifact unless ``artifact_sink`` is supplied.  Controller
    replay remains the primary execution evaluation; state injection is a
    separate diagnostic mode and is labelled as such in the result.
    """

    normalized_uid = str(uid or "").strip()
    normalized_run_key = str(run_key or "").strip()
    normalized_state_path = str(state_path or "").strip()
    if not normalized_uid:
        raise ValueError("uid is required")
    if not normalized_run_key:
        raise ValueError("run_key is required")
    if not normalized_state_path:
        raise ValueError("state_path is required")
    if not isinstance(state_trace, list):
        raise TypeError("state_trace must be a list")

    (
        effective_config,
        controller_name,
        use_orientation,
        policy_hz,
        reference_frame,
    ) = _effective_action_config(
        action_payload,
        execution_config,
    )
    if session_factory is None:

        def make_session() -> ActionTaskSuccessSession:
            return prepare_action_task_success_session(
                simulator_config=simulator_config,
                action_payload=action_payload,
                effective_execution_config=effective_config,
                controller_name=controller_name,
                use_orientation=use_orientation,
                policy_hz=policy_hz,
                reference_frame=reference_frame,
                active_camera_name=active_camera_name,
                runtime_env_meta=runtime_env_meta,
                scene_restore_options=scene_restore_options,
                robocasa_source_root=robocasa_source_root,
                robocasa_task_runtime=robocasa_task_runtime,
                robocasa_task_runtime_loader=(robocasa_task_runtime_loader),
                env_factory=env_factory,
                restorer=restorer,
                controller_config_loader=(controller_config_loader),
            )

    else:
        make_session = session_factory

    apply_state = set_qpos_qvel_state if state_setter is None else state_setter
    read_pose = current_eef_pose_world if pose_reader is None else pose_reader
    frames: list[dict[str, Any]] = []
    set_state_errors: list[dict[str, Any]] = []
    final_observation: TaskSuccessObservation | None = None
    final_task_state: dict[str, Any] = {}
    session: ActionTaskSuccessSession | None = None

    def read_success(env: Any) -> TaskSuccessObservation:
        if success_rule is not None:
            observation = success_rule(env)
            if not isinstance(
                observation,
                TaskSuccessObservation,
            ):
                raise TypeError("success_rule must return TaskSuccessObservation")
            return observation
        evidence: dict[str, Any] = {}
        if trajectory_summary:
            evidence["trajectory_objects"] = copy.deepcopy(dict(trajectory_summary))
        if trajectory_fallback:
            evidence["trajectory_fallback"] = copy.deepcopy(dict(trajectory_fallback))
        return observe_environment_task_success(
            env,
            task_name=task_name,
            evidence=evidence,
            checker=success_checker,
            evidence_reader=success_evidence_reader,
        )

    try:
        session = make_session()
        for index, raw_state in enumerate(state_trace):
            if not isinstance(raw_state, Mapping):
                continue
            state = dict(raw_state)
            env_step = state.get("env_step", index)
            entry: dict[str, Any] = {
                "step": int(index),
                "env_step": (int(env_step) if env_step is not None else int(index)),
            }
            qpos = state.get("qpos", None)
            qvel = state.get("qvel", None)
            if qpos is None or qvel is None:
                state_error = "missing_qpos_or_qvel"
            else:
                state_error = apply_state(
                    session.env,
                    qpos,
                    qvel,
                    timestamp_s=float(state.get("timestamp_s", 0.0) or 0.0),
                )
            if state_error is not None:
                set_state_errors.append(
                    {
                        "step": int(index),
                        "error": str(state_error),
                    }
                )
                entry["task_check_success"] = None
                entry["set_state_error"] = str(state_error)
                frames.append(entry)
                continue

            observation = read_success(session.env)
            final_observation = observation
            entry["task_check_success"] = observation.task_check_success
            entry.update(copy.deepcopy(observation.meta))
            try:
                position_world, _rotation_world, _meta = read_pose(
                    session.env,
                    site_name=session.ref_site_name,
                )
                entry["eef_world_xyz"] = (
                    np.asarray(
                        position_world,
                        dtype=np.float64,
                    )
                    .reshape(3)
                    .tolist()
                )
            except Exception:
                pass
            if observation.error is not None:
                entry["task_check_success_error"] = str(observation.error)
            if state.get("gripper", None) is not None:
                try:
                    entry["gripper"] = float(state["gripper"])
                except Exception:
                    pass
            frames.append(entry)

        final_observation = read_success(session.env)
        final_task_state = _task_state(
            session.env,
            task_name,
        )
    finally:
        if session is not None:
            session.close()

    assert final_observation is not None
    selected_runtime_meta = dict(runtime_env_meta or {})
    selected_runtime_meta.update(dict(session.runtime_env_meta or {}))
    payload: dict[str, Any] = {
        "format": CURRENT_TASK_SUCCESS_SCHEMA,
        "uid": normalized_uid,
        "run_key": normalized_run_key,
        "gen_model": (
            str(gen_model) if gen_model is not None and str(gen_model).strip() else None
        ),
        "source": "dream_exe.sim.task_success_replay",
        "eval_mode": "state_replay",
        "state_generation": str(state_generation),
        "task_name": (str(task_name) if str(task_name or "").strip() else None),
        "state_path": normalized_state_path,
        "task_success_traj_fallback_disabled": bool(
            task_success_traj_fallback_disabled
        ),
        "runtime_env_meta": {
            key: selected_runtime_meta[key]
            for key in _RUNTIME_ENV_META_KEYS
            if key in selected_runtime_meta
        },
        "final_task_check_success": (final_observation.task_check_success),
        "final_strict_task_check_success": (
            final_observation.strict_task_check_success
        ),
        "final_calibrated_task_check_success": (
            final_observation.calibrated_task_check_success
        ),
        "final_task_meta": copy.deepcopy(final_observation.meta),
        "final_task_state": final_task_state or None,
        "frames": frames,
    }
    if set_state_errors:
        payload["set_state_errors"] = set_state_errors
    if final_observation.error is not None:
        payload["final_task_check_success_error"] = str(final_observation.error)
    payload = normalize_task_success_payload(payload)
    if artifact_sink is not None:
        artifact_sink(copy.deepcopy(payload))
    return payload


def evaluate_saved_action_task_success(
    *,
    simulator_config: Mapping[str, Any],
    execution_config: Mapping[str, Any] | None,
    action_payload: Mapping[str, Any],
    uid: str,
    run_key: str,
    task_name: str,
    action_path: str,
    gen_model: str | None = None,
    trajectory_summary: Mapping[str, Any] | None = None,
    trajectory_fallback: Mapping[str, Any] | None = None,
    task_success_traj_fallback_disabled: bool = True,
    runtime_env_meta: Mapping[str, Any] | None = None,
    active_camera_name: str | None = None,
    scene_restore_options: Mapping[str, Any] | None = None,
    robocasa_source_root: str | os.PathLike[str] | None = None,
    robocasa_task_runtime: Mapping[str, Any] | None = None,
    robocasa_task_runtime_loader: (
        Callable[..., Mapping[str, Any] | None] | None
    ) = None,
    env_factory: Callable[..., Any] | None = None,
    restorer: Callable[..., Any] | None = None,
    controller_config_loader: (Callable[..., dict[str, Any]] | None) = None,
    session_factory: (Callable[[], ActionTaskSuccessSession] | None) = None,
    loop_runner: Callable[..., Mapping[str, Any]] | None = None,
    pose_reader: Callable[..., Any] | None = None,
    contact_reader: Callable[..., Any] | None = None,
    success_checker: Callable[[Any], Any] | None = None,
    success_evidence_reader: (Callable[[Any], Mapping[str, Any] | None] | None) = None,
    success_rule: (Callable[[Any], TaskSuccessObservation] | None) = None,
    artifact_sink: (Callable[[Mapping[str, Any]], Any] | None) = None,
) -> dict[str, Any]:
    """Replay one explicit saved action and return current-schema evidence.

    The default path creates a simulator only from ``simulator_config`` and
    injected runtime dependencies.  Passing ``session_factory`` replaces that
    lifecycle boundary.  With ``artifact_sink=None`` (the default), the call
    performs no task-success or bench artifact writes.
    """

    normalized_uid = str(uid or "").strip()
    normalized_run_key = str(run_key or "").strip()
    normalized_action_path = str(action_path or "").strip()
    if not normalized_uid:
        raise ValueError("uid is required")
    if not normalized_run_key:
        raise ValueError("run_key is required")
    if not normalized_action_path:
        raise ValueError("action_path is required")

    (
        effective_config,
        controller_name,
        use_orientation,
        policy_hz,
        reference_frame,
    ) = _effective_action_config(
        action_payload,
        execution_config,
    )
    if session_factory is None:

        def make_session() -> ActionTaskSuccessSession:
            return prepare_action_task_success_session(
                simulator_config=simulator_config,
                action_payload=action_payload,
                effective_execution_config=effective_config,
                controller_name=controller_name,
                use_orientation=use_orientation,
                policy_hz=policy_hz,
                reference_frame=reference_frame,
                active_camera_name=active_camera_name,
                runtime_env_meta=runtime_env_meta,
                scene_restore_options=scene_restore_options,
                robocasa_source_root=robocasa_source_root,
                robocasa_task_runtime=(robocasa_task_runtime),
                robocasa_task_runtime_loader=(robocasa_task_runtime_loader),
                env_factory=env_factory,
                restorer=restorer,
                controller_config_loader=(controller_config_loader),
            )

    else:
        make_session = session_factory

    run_loop = execute_action_trace_loop if loop_runner is None else loop_runner
    read_pose = current_eef_pose_world if pose_reader is None else pose_reader
    frames: list[dict[str, Any]] = []
    final_observation: TaskSuccessObservation | None = None
    final_task_state: dict[str, Any] = {}
    replay_result: Mapping[str, Any] | None = None
    session: ActionTaskSuccessSession | None = None

    def read_success(env: Any) -> TaskSuccessObservation:
        if success_rule is not None:
            observation = success_rule(env)
            if not isinstance(
                observation,
                TaskSuccessObservation,
            ):
                raise TypeError("success_rule must return TaskSuccessObservation")
            return observation
        evidence: dict[str, Any] = {}
        if trajectory_summary:
            evidence.setdefault(
                "trajectory_objects",
                copy.deepcopy(dict(trajectory_summary)),
            )
        if trajectory_fallback:
            evidence.setdefault(
                "trajectory_fallback",
                copy.deepcopy(dict(trajectory_fallback)),
            )
        return observe_environment_task_success(
            env,
            task_name=task_name,
            evidence=evidence,
            checker=success_checker,
            evidence_reader=success_evidence_reader,
        )

    def observe_checkpoint(
        env: Any,
        checkpoint: Mapping[str, Any],
    ) -> None:
        observation = read_success(env)
        position_world, _rotation_world, _pose_meta = read_pose(
            env,
            site_name=("" if session is None else session.ref_site_name),
        )
        entry: dict[str, Any] = {
            "checkpoint_index": int(checkpoint["checkpoint_index"]),
            "frame": int(checkpoint["frame"]),
            "task_check_success": (observation.task_check_success),
            "eef_world_xyz": np.asarray(
                position_world,
                dtype=np.float64,
            )
            .reshape(3)
            .tolist(),
        }
        entry.update(copy.deepcopy(observation.meta))
        if observation.error is not None:
            entry["task_check_success_error"] = str(observation.error)
        frames.append(entry)

    try:
        session = make_session()
        replay_result = run_loop(
            env=session.env,
            robot=session.robot,
            part_ctrl=session.part_controller,
            arm_part=session.arm_part,
            gripper_part=session.gripper_part,
            action_payload=action_payload,
            execution_config=effective_config,
            ref_site_name=session.ref_site_name,
            rotation_world_base=session.rotation_world_base,
            translation_world_base=(session.translation_world_base),
            pose_reader=read_pose,
            contact_reader=contact_reader,
            checkpoint_observer=observe_checkpoint,
        )
        final_observation = read_success(session.env)
        final_task_state = _task_state(
            session.env,
            task_name,
        )
    finally:
        if session is not None:
            session.close()

    assert replay_result is not None
    assert final_observation is not None
    selected_runtime_meta = dict(runtime_env_meta or {})
    selected_runtime_meta.update(dict(session.runtime_env_meta or {}))
    payload: dict[str, Any] = {
        "format": CURRENT_TASK_SUCCESS_SCHEMA,
        "uid": normalized_uid,
        "run_key": normalized_run_key,
        "gen_model": (
            str(gen_model) if gen_model is not None and str(gen_model).strip() else None
        ),
        "source": "dream_exe.sim.task_success_replay",
        "task_name": (str(task_name) if str(task_name or "").strip() else None),
        "action_path": normalized_action_path,
        "controller": str(replay_result["controller"]),
        "reference_frame": str(replay_result["reference_frame"]),
        "use_ori": bool(replay_result["use_ori"]),
        "pose_correction_mode": str(replay_result["pose_correction_mode"]),
        "pos_tol": float(replay_result["pos_tol"]),
        "ori_tol": float(replay_result["ori_tol"]),
        "max_correction_steps": int(replay_result["max_correction_steps"]),
        "task_success_traj_fallback_disabled": bool(
            task_success_traj_fallback_disabled
        ),
        "env_steps": int(replay_result["total_env_steps"]),
        "runtime_env_meta": {
            key: selected_runtime_meta[key]
            for key in _RUNTIME_ENV_META_KEYS
            if key in selected_runtime_meta
        },
        "final_task_check_success": (final_observation.task_check_success),
        "final_strict_task_check_success": (
            final_observation.strict_task_check_success
        ),
        "final_calibrated_task_check_success": (
            final_observation.calibrated_task_check_success
        ),
        "final_task_meta": copy.deepcopy(final_observation.meta),
        "final_task_state": final_task_state or None,
        "frames": frames,
    }
    if final_observation.error is not None:
        payload["final_task_check_success_error"] = str(final_observation.error)

    payload = normalize_task_success_payload(payload)
    if artifact_sink is not None:
        artifact_sink(copy.deepcopy(payload))
    return payload


__all__ = [
    "ActionTaskSuccessSession",
    "evaluate_saved_action_task_success",
    "evaluate_saved_state_task_success",
    "observe_environment_task_success",
    "prepare_action_task_success_session",
    "set_qpos_qvel_state",
]
