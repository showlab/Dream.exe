"""Compose Dream.exe stages without owning domain-specific behavior.

The pipeline layer owns ordering, selection, reuse, and structured failure
records only.  It does not discover paths, construct simulator environments,
choose model backends, or reinterpret stage inputs and outputs.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import copy
from typing import Any


PIPELINE_STAGE_ORDER = (
    "init",
    "video",
    "video2traj",
    "exec",
    "task_success",
    "eval",
)

StageRunner = Callable[..., Mapping[str, Any]]
StageOptions = (
    Mapping[str, Any]
    | Callable[
        [Mapping[str, Mapping[str, Any]]],
        Mapping[str, Any],
    ]
)
EventSink = Callable[[Mapping[str, Any]], Any]


def _stage_names(
    only_stages: Sequence[str] | None,
) -> tuple[str, ...]:
    if only_stages is None:
        return PIPELINE_STAGE_ORDER
    requested = {
        str(name or "").strip().lower()
        for name in only_stages
        if str(name or "").strip()
    }
    unknown = requested.difference(PIPELINE_STAGE_ORDER)
    if unknown:
        raise ValueError("unsupported pipeline stage(s): " + ", ".join(sorted(unknown)))
    return tuple(name for name in PIPELINE_STAGE_ORDER if name in requested)


def _options_for_stage(
    options: StageOptions | None,
    *,
    results: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    value = options(results) if callable(options) else options
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("stage options must be a mapping or return a mapping")
    return copy.deepcopy(dict(value))


def _emit(
    sink: EventSink | None,
    payload: Mapping[str, Any],
) -> None:
    if sink is not None:
        sink(copy.deepcopy(dict(payload)))


def run_stage_sequence(
    *,
    runners: Mapping[str, StageRunner | None],
    options: Mapping[str, StageOptions] | None = None,
    only_stages: Sequence[str] | None = None,
    completed_results: (Mapping[str, Mapping[str, Any]] | None) = None,
    continue_on_error: bool = False,
    event_sink: EventSink | None = None,
) -> dict[str, Any]:
    """Run selected stages in canonical order.

    ``completed_results`` makes resume explicit: supplied stages are recorded
    as ``reused`` and are not called.  A stage's options may be a mapping or a
    callable receiving the successful/reused result map, which lets later
    stages consume earlier artifact references without coupling this module to
    their schemas.
    """

    selected = _stage_names(only_stages)
    supplied_options = dict(options or {})
    completed = {
        str(name): copy.deepcopy(dict(result))
        for name, result in dict(completed_results or {}).items()
    }
    unknown_runners = set(runners).difference(PIPELINE_STAGE_ORDER)
    unknown_options = set(supplied_options).difference(PIPELINE_STAGE_ORDER)
    unknown_completed = set(completed).difference(PIPELINE_STAGE_ORDER)
    unknown = unknown_runners | unknown_options | unknown_completed
    if unknown:
        raise ValueError("unsupported pipeline stage(s): " + ", ".join(sorted(unknown)))

    stage_records: dict[str, dict[str, Any]] = {}
    results: dict[str, Mapping[str, Any]] = {}
    stopped = False

    for name in PIPELINE_STAGE_ORDER:
        if name not in selected:
            stage_records[name] = {"status": "not_selected"}
            continue
        if stopped:
            stage_records[name] = {
                "status": "blocked",
                "reason": "earlier_stage_failed",
            }
            continue
        if name in completed:
            result = completed[name]
            results[name] = result
            stage_records[name] = {
                "status": "reused",
                "result": result,
            }
            _emit(
                event_sink,
                {"stage": name, "status": "reused"},
            )
            continue

        runner = runners.get(name)
        if runner is None:
            stage_records[name] = {"status": "disabled"}
            continue

        _emit(
            event_sink,
            {"stage": name, "status": "started"},
        )
        try:
            stage_options = _options_for_stage(
                supplied_options.get(name),
                results=results,
            )
            raw_result = runner(**stage_options)
            if not isinstance(raw_result, Mapping):
                raise TypeError(f"{name} stage returned a non-mapping result")
            result = copy.deepcopy(dict(raw_result))
            results[name] = result
            stage_records[name] = {
                "status": "completed",
                "result": result,
            }
            _emit(
                event_sink,
                {"stage": name, "status": "completed"},
            )
        except Exception as error:
            stage_records[name] = {
                "status": "failed",
                "error_type": type(error).__name__,
                "error": str(error),
            }
            _emit(
                event_sink,
                {
                    "stage": name,
                    "status": "failed",
                    "error_type": type(error).__name__,
                    "error": str(error),
                },
            )
            if not bool(continue_on_error):
                stopped = True

    failed = [
        name for name, record in stage_records.items() if record["status"] == "failed"
    ]
    completed_names = [
        name
        for name, record in stage_records.items()
        if record["status"] in {"completed", "reused"}
    ]
    requested_names = list(selected)
    pending = [
        name
        for name in selected
        if stage_records[name]["status"] in {"blocked", "disabled"}
    ]
    status = "failed" if failed else "partial" if pending else "completed"
    return {
        "format": "dream_exe.pipeline",
        "status": status,
        "stage_order": list(PIPELINE_STAGE_ORDER),
        "requested_stages": requested_names,
        "completed_stages": completed_names,
        "failed_stages": failed,
        "stages": stage_records,
        "results": copy.deepcopy(results),
    }


def run_pipeline(
    *,
    initialize: StageRunner | None = None,
    acquire_video: StageRunner | None = None,
    video2traj: StageRunner | None = None,
    execute: StageRunner | None = None,
    task_success: StageRunner | None = None,
    evaluate: StageRunner | None = None,
    init_options: StageOptions | None = None,
    video_options: StageOptions | None = None,
    video2traj_options: StageOptions | None = None,
    exec_options: StageOptions | None = None,
    task_success_options: StageOptions | None = None,
    eval_options: StageOptions | None = None,
    only_stages: Sequence[str] | None = None,
    completed_results: (Mapping[str, Mapping[str, Any]] | None) = None,
    continue_on_error: bool = False,
    event_sink: EventSink | None = None,
) -> dict[str, Any]:
    """Run the standard Dream.exe stage sequence through public callables.

    Per-run task success is an explicit optional post-execution producer.  It
    joins the selected sequence only when a runner is supplied or callers name
    it in ``only_stages``; the default remains the existing four-stage
    ``video → video2traj → exec → eval`` workflow (plus optional ``init``).
    """

    selected_stages = only_stages
    if selected_stages is None:
        selected_stages = tuple(
            stage
            for stage in PIPELINE_STAGE_ORDER
            if (
                (stage != "init" or initialize is not None)
                and (stage != "task_success" or task_success is not None)
            )
        )

    return run_stage_sequence(
        runners={
            "init": initialize,
            "video": acquire_video,
            "video2traj": video2traj,
            "exec": execute,
            "task_success": task_success,
            "eval": evaluate,
        },
        options={
            "init": init_options or {},
            "video": video_options or {},
            "video2traj": video2traj_options or {},
            "exec": exec_options or {},
            "task_success": task_success_options or {},
            "eval": eval_options or {},
        },
        only_stages=selected_stages,
        completed_results=completed_results,
        continue_on_error=continue_on_error,
        event_sink=event_sink,
    )


__all__ = [
    "EventSink",
    "PIPELINE_STAGE_ORDER",
    "StageOptions",
    "StageRunner",
    "run_pipeline",
    "run_stage_sequence",
]
