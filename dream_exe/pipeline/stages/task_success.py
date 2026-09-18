"""Evaluate task success from a saved action execution.

All sample and run paths are passed explicitly by the formal pipeline layout
and benchmark runtime. The simulator evaluator receives loaded
inputs and remains unaware of bench roots, UIDs, layout discovery, or batch
supervision.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
import copy
import json
from pathlib import Path
import threading
from typing import Any

from ..runner.supervisor import (
    ClaimTask,
    Clock,
    ExecutorFactory,
    StatusSink,
    run_batch_supervisor,
)
from ..records.layout import (
    DEFAULT_FORMAL_RUN_KEY,
    build_run_id,
    expand_available_runs,
    formal_artifact_paths,
    normalize_run_key,
)
from .sim import resolve_bench_execution_request
from ...evaluation.execution import (
    publish_task_success_json,
    summarize_openblenderlid_trajectory,
    summarize_object_trajectories,
)
from ...sim.execution.inputs import load_action_stream
from ...sim.robocasa.task_success import (
    check_task_success_with_joint_aliases,
    observe_robocasa_cheesybread_success,
)
from ...sim.robocasa.task_components import (
    SUPPORTED_ROBOCASA_COMPONENT_TASKS,
    extract_robocasa_task_success_evidence,
)
from ...sim.task_success.replay import (
    evaluate_saved_action_task_success,
)


TaskSuccessRunner = Callable[..., Mapping[str, Any]]
ExecutionRequestResolver = Callable[..., Mapping[str, Any]]
ActionLoader = Callable[[str], Any]
TrajectorySummaryLoader = Callable[
    [str],
    Mapping[str, Any] | None,
]

_TASK_FIELDS = {
    "task_id",
    "uid",
    "sample_dir",
    "run_key",
    "run_id",
    "gen_model",
    "task_name",
}
_RUNNER_IDENTITY_FIELDS = {
    "sample_dir",
    "run_key",
    "gen_model",
    "task_name",
}
_SAMPLE_LOCKS_GUARD = threading.Lock()
_SAMPLE_LOCKS: dict[str, threading.Lock] = {}


def _clean_required(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is required")
    if any(marker in text for marker in ("\t", "\r", "\n")):
        raise ValueError(f"{label} must not contain control delimiters")
    return text


def _resolved_path(
    value: str | Path,
    *,
    label: str,
) -> str:
    return Path(_clean_required(value, label=label)).expanduser().resolve().as_posix()


def _load_json_mapping(
    path: str | Path,
    *,
    label: str,
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    with source.open("r", encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} must be a JSON object: {source}")
    return copy.deepcopy(dict(payload))


def load_object_trajectory_summary(
    path: str | Path,
) -> dict[str, Any] | None:
    """Read one optional explicit ``obj_trajs.json`` as current evidence."""

    text = str(path or "").strip()
    if not text:
        return None
    source = Path(text).expanduser().resolve()
    if not source.exists():
        return None
    try:
        payload = _load_json_mapping(
            source,
            label="object trajectory",
        )
    except Exception as error:
        return {
            "source": source.as_posix(),
            "error": f"{type(error).__name__}: {error}",
        }
    return summarize_object_trajectories(
        payload,
        source=source.as_posix(),
    )


def load_openblenderlid_trajectory_fallback(
    path: str | Path,
) -> dict[str, Any] | None:
    """Load optional OpenBlenderLid-only diagnostic trajectory evidence."""

    text = str(path or "").strip()
    if not text:
        return None
    source = Path(text).expanduser().resolve()
    if not source.exists():
        return None
    try:
        payload = _load_json_mapping(
            source,
            label="object trajectory",
        )
    except Exception as error:
        return {
            "source": source.as_posix(),
            "error": f"{type(error).__name__}: {error}",
        }
    return summarize_openblenderlid_trajectory(
        payload,
        source=source.as_posix(),
    )


def _task_name_from_request(
    request: Mapping[str, Any],
    *,
    metadata_path: Path,
    explicit_task_name: str,
) -> str:
    if str(explicit_task_name or "").strip():
        return str(explicit_task_name).strip()
    simulator_config = dict(request.get("simulator_config", {}) or {})
    root = dict(simulator_config.get("raw", simulator_config) or {})
    scene_restore = dict(root.get("scene_restore", {}) or {})
    task_name = str(
        scene_restore.get("dataset_env_name", "")
        or scene_restore.get("bootstrap_env_name", "")
        or ""
    ).strip()
    if task_name:
        return task_name
    try:
        metadata = _load_json_mapping(
            metadata_path,
            label="sample metadata",
        )
        identity = dict(metadata.get("identity", {}) or {})
        task_name = str(identity.get("task", "") or "").strip()
    except Exception:
        task_name = ""
    return task_name


def resolve_bench_task_success_request(
    *,
    sample_dir: str | Path,
    run_key: str = DEFAULT_FORMAL_RUN_KEY,
    gen_model: str = "",
    task_name: str = "",
    simulator_config_path: str | Path | None = None,
    execution_config_path: str | Path | None = None,
    action_path: str | Path | None = None,
    object_trajectories_path: str | Path | None = None,
    output_path: str | Path | None = None,
    scene_override_path: str | Path | None = None,
    execution_request_resolver: (
        ExecutionRequestResolver
    ) = resolve_bench_execution_request,
) -> dict[str, Any]:
    """Resolve one task-success request without creating or writing paths."""

    if not callable(execution_request_resolver):
        raise TypeError("execution_request_resolver must be callable")
    sample_root = (
        Path(
            _clean_required(
                sample_dir,
                label="sample_dir",
            )
        )
        .expanduser()
        .resolve()
    )
    normalized_run_key = normalize_run_key(run_key)
    clean_gen_model = str(gen_model or "").strip()
    formal = formal_artifact_paths(
        sample_root,
        normalized_run_key,
        gen_model=clean_gen_model,
    )
    resolved_execution = execution_request_resolver(
        sample_dir=sample_root,
        run_key=normalized_run_key,
        gen_model=clean_gen_model,
        simulator_config_path=simulator_config_path,
        execution_config_path=execution_config_path,
        action_path=action_path,
        scene_override_path=scene_override_path,
    )
    if not isinstance(resolved_execution, Mapping):
        raise TypeError("execution_request_resolver must return a mapping")
    request = copy.deepcopy(dict(resolved_execution))
    clean_uid = _clean_required(
        request.get("uid"),
        label="resolved uid",
    )
    if sample_root.name != clean_uid:
        raise ValueError(
            f"resolved uid/sample_dir mismatch: {clean_uid!r} != {sample_root.name!r}"
        )
    resolved_object_trajectories = (
        _resolved_path(
            object_trajectories_path,
            label="object_trajectories_path",
        )
        if object_trajectories_path is not None
        and str(object_trajectories_path).strip()
        else formal["obj_trajs"].as_posix()
    )
    resolved_output = (
        _resolved_path(
            output_path,
            label="output_path",
        )
        if output_path is not None and str(output_path).strip()
        else formal["task_success"].as_posix()
    )
    request.update(
        {
            "run_id": build_run_id(
                run_key=normalized_run_key,
                gen_model=clean_gen_model,
            ),
            "task_name": _task_name_from_request(
                request,
                metadata_path=formal["sample_metadata"],
                explicit_task_name=task_name,
            ),
            "object_trajectories_path": (resolved_object_trajectories),
            "task_success_output_path": (resolved_output),
        }
    )
    return request


def _action_payload(
    action_path: str,
    *,
    loader: ActionLoader,
) -> dict[str, Any]:
    loaded = loader(action_path)
    payload = loaded[0] if isinstance(loaded, tuple) and len(loaded) > 0 else loaded
    if not isinstance(payload, Mapping):
        raise TypeError(
            "action_loader must return an action mapping "
            "or a tuple whose first item is a mapping"
        )
    return copy.deepcopy(dict(payload))


def evaluate_bench_task_success(
    *,
    sample_dir: str | Path,
    run_key: str = DEFAULT_FORMAL_RUN_KEY,
    gen_model: str = "",
    task_name: str = "",
    simulator_config_path: str | Path | None = None,
    execution_config_path: str | Path | None = None,
    action_path: str | Path | None = None,
    object_trajectories_path: str | Path | None = None,
    output_path: str | Path | None = None,
    scene_override_path: str | Path | None = None,
    scene_restore_options: Mapping[str, Any] | None = None,
    robocasa_source_root: str | Path | None = None,
    task_success_traj_fallback_disabled: bool = False,
    publish: bool = True,
    artifact_sink: (Callable[[Mapping[str, Any]], Any] | None) = None,
    replay_options: Mapping[str, Any] | None = None,
    runner: TaskSuccessRunner = (evaluate_saved_action_task_success),
    execution_request_resolver: (
        ExecutionRequestResolver
    ) = resolve_bench_execution_request,
    action_loader: ActionLoader = load_action_stream,
    trajectory_summary_loader: (
        TrajectorySummaryLoader
    ) = load_object_trajectory_summary,
    trajectory_fallback_loader: (
        TrajectorySummaryLoader
    ) = load_openblenderlid_trajectory_fallback,
) -> dict[str, Any]:
    """Evaluate one resolved bench run through the bench-free replay."""

    if not callable(runner):
        raise TypeError("runner must be callable")
    if not callable(action_loader):
        raise TypeError("action_loader must be callable")
    if not callable(trajectory_summary_loader):
        raise TypeError("trajectory_summary_loader must be callable")
    if not callable(trajectory_fallback_loader):
        raise TypeError("trajectory_fallback_loader must be callable")
    request = resolve_bench_task_success_request(
        sample_dir=sample_dir,
        run_key=run_key,
        gen_model=gen_model,
        task_name=task_name,
        simulator_config_path=simulator_config_path,
        execution_config_path=execution_config_path,
        action_path=action_path,
        object_trajectories_path=(object_trajectories_path),
        output_path=output_path,
        scene_override_path=scene_override_path,
        execution_request_resolver=(execution_request_resolver),
    )
    action_payload = _action_payload(
        str(request["action_path"]),
        loader=action_loader,
    )
    trajectory_summary = (
        None
        if bool(task_success_traj_fallback_disabled)
        else trajectory_summary_loader(str(request["object_trajectories_path"]))
    )
    trajectory_fallback = (
        trajectory_fallback_loader(str(request["object_trajectories_path"]))
        if (
            not bool(task_success_traj_fallback_disabled)
            and str(request["task_name"]) == "OpenBlenderLid"
        )
        else None
    )

    selected_sink = artifact_sink
    published_path: str | None = None
    if selected_sink is None and bool(publish):
        published_path = str(request["task_success_output_path"])

        def selected_sink(
            payload: Mapping[str, Any],
        ) -> Any:
            return publish_task_success_json(
                payload,
                published_path,
            )

    selected_success_rule = None
    if str(request["task_name"]) == "CheesyBread":

        def selected_success_rule(env: Any) -> Any:
            return observe_robocasa_cheesybread_success(
                env,
                trajectory_objects=(
                    trajectory_summary
                    if isinstance(
                        trajectory_summary,
                        Mapping,
                    )
                    else None
                ),
            )

    extra = copy.deepcopy(dict(replay_options or {}))
    conflicting = sorted(
        set(extra).intersection(
            {
                "simulator_config",
                "execution_config",
                "action_payload",
                "uid",
                "run_key",
                "task_name",
                "action_path",
                "gen_model",
                "trajectory_summary",
                "trajectory_fallback",
                "task_success_traj_fallback_disabled",
                "active_camera_name",
                "scene_restore_options",
                "robocasa_source_root",
                "success_rule",
                "artifact_sink",
            }
        )
    )
    if conflicting:
        raise ValueError(
            "replay_options cannot override resolved fields: " + ", ".join(conflicting)
        )
    if selected_success_rule is None and runner is evaluate_saved_action_task_success:
        extra.setdefault(
            "success_checker",
            check_task_success_with_joint_aliases,
        )
        if request["task_name"] in SUPPORTED_ROBOCASA_COMPONENT_TASKS:
            if "success_evidence_reader" in extra:
                raise ValueError(
                    "replay_options cannot override detached task-success evidence"
                )
            extra["success_evidence_reader"] = (
                lambda env: extract_robocasa_task_success_evidence(env, request["task_name"])
            )
    runner_arguments: dict[str, Any] = {
        "simulator_config": request["simulator_config"],
        "execution_config": request["execution_config"],
        "action_payload": action_payload,
        "uid": request["uid"],
        "run_key": request["run_key"],
        "task_name": request["task_name"],
        "action_path": request["action_path"],
        "gen_model": (request["gen_model"] or None),
        "trajectory_summary": trajectory_summary,
        "trajectory_fallback": trajectory_fallback,
        "task_success_traj_fallback_disabled": bool(
            task_success_traj_fallback_disabled
        ),
        "active_camera_name": (request["active_camera_name"] or None),
        "scene_restore_options": scene_restore_options,
        "success_rule": selected_success_rule,
        "artifact_sink": selected_sink,
        **extra,
    }
    if robocasa_source_root is not None:
        runner_arguments["robocasa_source_root"] = robocasa_source_root
    raw_result = runner(
        **runner_arguments,
    )
    if not isinstance(raw_result, Mapping):
        raise TypeError("runner must return a mapping")
    result = copy.deepcopy(dict(raw_result))
    return {
        "ok": True,
        "returncode": 0,
        "uid": request["uid"],
        "run_id": request["run_id"],
        "run_key": request["run_key"],
        "gen_model": request["gen_model"],
        "task_name": request["task_name"],
        "artifact_path": published_path,
        "request": request,
        "result": result,
    }


def _task_id(
    *,
    uid: str,
    run_id: str,
) -> str:
    return (
        f"{_clean_required(uid, label='uid')}\t"
        f"{_clean_required(run_id, label='run_id')}\t"
        "task_success"
    )


def build_bench_task_success_tasks(
    *,
    samples: Iterable[Mapping[str, Any]],
    run_keys: Iterable[str],
    gen_models: Iterable[str] = (),
) -> list[dict[str, Any]]:
    """Expand explicit samples and formal runs into stable batch records."""

    raw_run_keys = [
        normalize_run_key(value) for value in run_keys if str(value or "").strip()
    ]
    if not raw_run_keys:
        raise ValueError("run_keys must contain an explicit run key")
    _abstract, resolved_runs = expand_available_runs(
        raw_run_keys,
        available_gen_models=[
            str(value or "").strip() for value in gen_models if str(value or "").strip()
        ],
    )
    if "gen" in raw_run_keys and not any(run.run_key == "gen" for run in resolved_runs):
        raise ValueError("gen run key requires explicit gen_models")

    tasks: list[dict[str, Any]] = []
    for index, raw_sample in enumerate(samples):
        if not isinstance(raw_sample, Mapping):
            raise TypeError(f"sample {index} must be a mapping")
        uid = _clean_required(
            raw_sample.get("uid"),
            label=f"sample {index} uid",
        )
        sample_dir = _resolved_path(
            raw_sample.get("sample_dir"),
            label=f"sample {index} sample_dir",
        )
        if Path(sample_dir).name != uid:
            raise ValueError(f"sample {index} uid/sample_dir mismatch")
        task_name = str(raw_sample.get("task_name", "") or "").strip()
        for run in resolved_runs:
            tasks.append(
                {
                    "task_id": _task_id(
                        uid=uid,
                        run_id=run.run_id,
                    ),
                    "uid": uid,
                    "sample_dir": sample_dir,
                    "run_key": run.run_key,
                    "run_id": run.run_id,
                    "gen_model": run.gen_model,
                    "task_name": task_name,
                }
            )
    return tasks


def _validated_task(
    task: Mapping[str, Any],
) -> dict[str, Any]:
    if not isinstance(task, Mapping):
        raise TypeError("task-success batch task must be a mapping")
    unknown = sorted(set(task) - _TASK_FIELDS)
    if unknown:
        raise ValueError(
            "task-success batch task has unsupported fields: " + ", ".join(unknown)
        )
    uid = _clean_required(
        task.get("uid"),
        label="task uid",
    )
    sample_dir = _resolved_path(
        task.get("sample_dir"),
        label="task sample_dir",
    )
    if Path(sample_dir).name != uid:
        raise ValueError("task uid/sample_dir mismatch")
    run_key = normalize_run_key(task.get("run_key", ""))
    gen_model = str(task.get("gen_model", "") or "").strip()
    run_id = build_run_id(
        run_key=run_key,
        gen_model=gen_model,
    )
    if str(task.get("run_id", "") or "") != run_id:
        raise ValueError("task run_id does not match run identity")
    task_id = _task_id(uid=uid, run_id=run_id)
    if str(task.get("task_id", "") or "") != task_id:
        raise ValueError("task_id does not match task-success identity")
    return {
        "task_id": task_id,
        "uid": uid,
        "sample_dir": sample_dir,
        "run_key": run_key,
        "run_id": run_id,
        "gen_model": gen_model,
        "task_name": str(task.get("task_name", "") or "").strip(),
    }


def _sample_lock(sample_dir: str) -> threading.Lock:
    key = Path(sample_dir).resolve().as_posix()
    with _SAMPLE_LOCKS_GUARD:
        lock = _SAMPLE_LOCKS.get(key)
        if lock is None:
            lock = threading.Lock()
            _SAMPLE_LOCKS[key] = lock
        return lock


def run_bench_task_success_task(
    *,
    task: Mapping[str, Any],
    runner: TaskSuccessRunner = (evaluate_bench_task_success),
    runner_options: Mapping[str, Any] | None = None,
    attempt: int = 1,
    is_cancelled: Callable[[], bool] | None = None,
) -> dict[str, Any]:
    """Run one validated task-success record."""

    record = _validated_task(task)
    if not callable(runner):
        raise TypeError("runner must be callable")
    if isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1:
        raise ValueError("attempt must be a positive integer")
    cancel = is_cancelled if is_cancelled is not None else (lambda: False)
    if not callable(cancel):
        raise TypeError("is_cancelled must be callable")
    if cancel():
        return {
            "ok": False,
            "status": "cancelled",
            "task_id": record["task_id"],
            "attempt": attempt,
        }
    options = copy.deepcopy(dict(runner_options or {}))
    conflicts = sorted(set(options).intersection(_RUNNER_IDENTITY_FIELDS))
    if conflicts:
        raise ValueError(
            "runner_options cannot override task identity: " + ", ".join(conflicts)
        )
    with _sample_lock(record["sample_dir"]):
        if cancel():
            return {
                "ok": False,
                "status": "cancelled",
                "task_id": record["task_id"],
                "attempt": attempt,
            }
        raw_result = runner(
            sample_dir=record["sample_dir"],
            run_key=record["run_key"],
            gen_model=record["gen_model"],
            task_name=record["task_name"],
            **options,
        )
    if not isinstance(raw_result, Mapping):
        raise TypeError("runner must return a mapping")
    result = copy.deepcopy(dict(raw_result))
    if not isinstance(result.get("ok"), bool):
        raise TypeError("runner result requires a boolean 'ok'")
    return {
        "ok": bool(result["ok"]),
        "status": (
            "completed"
            if bool(result["ok"])
            else str(result.get("status", "failed") or "failed")
        ),
        "task_id": record["task_id"],
        "uid": record["uid"],
        "run_id": record["run_id"],
        "run_key": record["run_key"],
        "gen_model": record["gen_model"],
        "attempt": attempt,
        "evaluation": result,
    }


def run_bench_task_success_batch(
    *,
    tasks: Iterable[Mapping[str, Any]],
    runner: TaskSuccessRunner = (evaluate_bench_task_success),
    runner_options: Mapping[str, Any] | None = None,
    completed_task_ids: Iterable[str] = (),
    prior_transitions: Sequence[Mapping[str, Any]] = (),
    max_attempts: int = 1,
    shard_count: int = 1,
    shard_index: int = 0,
    max_workers: int = 1,
    executor_factory: ExecutorFactory | None = None,
    claim_task: ClaimTask | None = None,
    status_sink: StatusSink | None = None,
    clock: Clock | None = None,
    is_cancelled: Callable[[], bool] | None = None,
    dry_run: bool = False,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """Run explicit task-success records through the common supervisor."""

    validated = [_validated_task(task) for task in tasks]
    if not callable(runner):
        raise TypeError("runner must be callable")
    common_options = copy.deepcopy(dict(runner_options or {}))

    def run_one(
        *,
        task: Mapping[str, Any],
        attempt: int,
        is_cancelled: Callable[[], bool],
    ) -> dict[str, Any]:
        return run_bench_task_success_task(
            task=task,
            runner=runner,
            runner_options=common_options,
            attempt=attempt,
            is_cancelled=is_cancelled,
        )

    return run_batch_supervisor(
        tasks=validated,
        run_one=run_one,
        completed_task_ids=completed_task_ids,
        prior_transitions=prior_transitions,
        max_attempts=max_attempts,
        shard_count=shard_count,
        shard_index=shard_index,
        max_workers=max_workers,
        executor_factory=executor_factory,
        claim_task=claim_task,
        status_sink=status_sink,
        clock=clock,
        is_cancelled=is_cancelled,
        dry_run=dry_run,
        continue_on_error=continue_on_error,
    )


__all__ = [
    "ActionLoader",
    "ExecutionRequestResolver",
    "TaskSuccessRunner",
    "TrajectorySummaryLoader",
    "build_bench_task_success_tasks",
    "evaluate_bench_task_success",
    "load_object_trajectory_summary",
    "load_openblenderlid_trajectory_fallback",
    "resolve_bench_task_success_request",
    "run_bench_task_success_batch",
    "run_bench_task_success_task",
]
