"""Run one explicit materialized benchmark case through the formal pipeline.

This module does not own stage implementations or benchmark discovery. It
orders an optional sample-scoped initializer, explicit saved-video handoffs,
an optional formal reference run, and one formal candidate run through the
existing verified benchmark workflow.
"""

from __future__ import annotations

import inspect
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...artifacts.layout import sample_artifact_paths
from ...evaluation.trajectory import (
    TRAJECTORY_SIMILARITY_GROUPS,
)
from .sequence import StageRunner
from ..records.layout import (
    benchmark_pipeline_state_path,
    build_run_id,
    formal_artifact_paths,
    normalize_run_key,
    split_run_key,
)
from ..validation.artifacts import (
    validate_benchmark_stage_artifacts,
)
from ..planning.depth_artifacts import (
    build_video2traj_depth_artifact_contract,
)
from ..records.state import (
    benchmark_pipeline_state_lock,
    digest_existing_outputs,
    digest_output_file,
    load_benchmark_pipeline_state,
    verify_recorded_outputs,
)
from ..records.resume import collect_stage_outputs
from ..stages.sim import resolve_bench_execution_request
from .workflow import (
    _configured_task_success_options,
    resolve_benchmark_video_handoff,
    run_benchmark_workflow,
)

SINGLE_UID_WORKFLOW_SCHEMA = "dream-exe.single-case-result"
TECHNICAL_COMPLETENESS_PROFILE = "technical"

SingleUIDWorkflowRunner = Callable[..., Mapping[str, Any]]
_DEFAULT_WORKFLOW_RUNNER = run_benchmark_workflow

_PHASE_ORDER = (
    "init",
    "video_handoff",
    "reference",
    "candidate",
    "acceptance",
)
_REFERENCE_STAGES = ("video2traj",)
_CANDIDATE_STAGES = (
    "video2traj",
    "exec",
    "task_success",
    "eval",
)
_PIPELINE_STAGE_STATUSES = frozenset(
    {
        "completed",
        "reused",
        "failed",
        "blocked",
        "disabled",
    }
)
_PROTECTED_RUN_FIELDS = frozenset(
    {
        "acquire_video",
        "completed_results",
        "continue_on_error",
        "init_options",
        "initialize",
        "only_stages",
        "require_all_reused",
        "robocasa_source_root",
        "sample_dir",
        "source_video_path",
        "stage_state_policy",
        "task_success_rate_specs",
        "trajectory_path_comparison_reference_path",
        "uid",
        "video_options",
    }
)
_FAILED_INIT_STATUSES = frozenset(
    {
        "blocked",
        "cancelled",
        "failed",
        "failure",
        "incomplete",
        "partial",
    }
)
_SUCCESS_INIT_STATUSES = frozenset(
    {
        "completed",
        "ok",
        "passed",
        "success",
        "succeeded",
    }
)


def _phase_error(error: Exception) -> dict[str, str]:
    return {
        "status": "failed",
        "error_type": type(error).__name__,
        "error": str(error),
    }


def _failed_response(
    *,
    uid: str,
    sample_root: Path,
    phases: Mapping[str, Any],
    failed_phase: str,
    reference_trajectory_path: Path | None = None,
) -> dict[str, Any]:
    return {
        "format": SINGLE_UID_WORKFLOW_SCHEMA,
        "status": "failed",
        "status_scope": "technical_workflow",
        "uid": uid,
        "sample_dir": sample_root.as_posix(),
        "failed_phase": failed_phase,
        "phase_order": list(_PHASE_ORDER),
        "phases": dict(phases),
        "reference_trajectory_path": (
            None
            if reference_trajectory_path is None
            else reference_trajectory_path.as_posix()
        ),
        "technical_completion": {
            "status": "failed",
            "failed_phase": failed_phase,
        },
    }


def _normalize_run_spec(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    options = dict(value)
    source_video = options.pop("video_path", None)
    accepted = set(inspect.signature(_DEFAULT_WORKFLOW_RUNNER).parameters)
    unknown = sorted(set(options).difference(accepted))
    if unknown:
        raise ValueError(f"{label} has unsupported fields: " + ", ".join(unknown))
    protected = sorted(set(options).intersection(_PROTECTED_RUN_FIELDS))
    if protected:
        raise ValueError(
            f"{label} cannot override top-level fields: " + ", ".join(protected)
        )
    run_key = str(options.get("run_key", "") or "").strip()
    if not run_key:
        raise ValueError(f"{label}.run_key is required")
    runtime_config = options.get("runtime_config")
    if not isinstance(runtime_config, Mapping):
        raise TypeError(f"{label}.runtime_config must be an explicit mapping")
    options["run_key"] = normalize_run_key(run_key)
    options["gen_model"] = str(options.get("gen_model", "") or "").strip()
    options["_explicit_video_path"] = (
        None
        if not str(source_video or "").strip()
        else Path(str(source_video)).expanduser().absolute()
    )
    return options


def _robocasa_runtime_source_path(value: Any, *, label: str) -> Path:
    if not isinstance(value, (str, Path)):
        raise TypeError(f"{label} must be a path")
    if not str(value).strip():
        raise ValueError(f"{label} must be non-empty")
    source_root = Path(value).expanduser()
    if not source_root.is_absolute():
        raise ValueError(f"{label} must be an absolute path")
    return source_root.resolve(strict=False)


def _candidate_acquisition_path(
    value: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(value, Mapping):
        raise TypeError("candidate video acquirer must return a mapping")
    result = dict(value)
    raw_video = result.get("video_path")
    raw_output = result.get("output_video")
    video_text = str(raw_video or "").strip()
    output_text = str(raw_output or "").strip()
    if not video_text and not output_text:
        raise ValueError(
            "candidate video acquirer result must contain video_path or output_video"
        )
    video_path = (
        None if not video_text else Path(video_text).expanduser().resolve(strict=False)
    )
    output_path = (
        None
        if not output_text
        else Path(output_text).expanduser().resolve(strict=False)
    )
    if video_path is not None and output_path is not None and video_path != output_path:
        raise ValueError(
            "candidate video acquirer result video_path and output_video must match"
        )
    selected = video_path or output_path
    assert selected is not None
    result["video_path"] = selected.as_posix()
    if raw_output is not None:
        result["output_video"] = selected.as_posix()
    return selected, result


def _resolve_single_uid_similarity_specs(
    *,
    specs: list[Mapping[str, Any]] | tuple[Mapping[str, Any], ...],
    sample_root: Path,
    candidate_options: Mapping[str, Any],
    reference_trajectory: Path,
    reference_object_trajectory: Path | None = None,
) -> list[dict[str, Any]]:
    candidate_formal = formal_artifact_paths(
        sample_root,
        str(candidate_options["run_key"]),
        gen_model=str(candidate_options.get("gen_model", "") or ""),
    )
    reference_object = (
        reference_trajectory.parent / "obj_trajs.json"
        if reference_object_trajectory is None
        else reference_object_trajectory.expanduser().absolute()
    )

    def bind_selected_path(
        spec: dict[str, Any],
        *,
        field: str,
        expected: Path,
        index: int,
    ) -> None:
        selected = expected.expanduser().absolute()
        explicit = spec.get(field)
        if explicit is not None:
            supplied = Path(explicit).expanduser().absolute()
            if supplied != selected:
                raise ValueError(
                    f"trajectory_similarity_specs[{index}].{field} must "
                    "match the selected single-UID formal artifact "
                    f"{selected}, got {supplied}"
                )
        spec[field] = selected

    resolved: list[dict[str, Any]] = []
    for index, raw in enumerate(specs):
        if not isinstance(raw, Mapping):
            raise TypeError(f"trajectory_similarity_specs[{index}] must be a mapping")
        spec = dict(raw)
        group = str(spec.get("group", "") or "").strip()
        if not group:
            raise ValueError(f"trajectory_similarity_specs[{index}].group is required")
        if group in TRAJECTORY_SIMILARITY_GROUPS[:2]:
            predicted = candidate_formal["ee_traj"]
            reference = reference_trajectory
        elif group == TRAJECTORY_SIMILARITY_GROUPS[2]:
            predicted = candidate_formal["obj_trajs"]
            reference = reference_object
            if not reference.is_file():
                raise FileNotFoundError(
                    "cannot bind OBJ reference trajectory: expected "
                    f"{reference} beside {reference_trajectory}"
                )
        else:
            raise ValueError(
                "trajectory similarity accepts only the configured EEF/OBJ groups, "
                f"got {group!r}"
            )
        bind_selected_path(
            spec,
            field="predicted_path",
            expected=predicted,
            index=index,
        )
        bind_selected_path(
            spec,
            field="reference_path",
            expected=reference,
            index=index,
        )
        resolved.append(spec)
    return resolved


def _safe_run_summary(options: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "run_key": str(options["run_key"]),
        "gen_model": str(options.get("gen_model", "") or ""),
        "source_video_path": Path(options["_explicit_video_path"]).as_posix(),
    }


def _validate_workflow_result(
    value: Mapping[str, Any],
    *,
    uid: str,
    sample_root: Path,
    options: Mapping[str, Any],
    expected_stages: tuple[str, ...],
    stage_state_policy: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("workflow runner must return a mapping")
    result = dict(value)
    video_kind, _slot = split_run_key(str(options["run_key"]))
    expected = {
        "format": "dream_exe.benchmark_workflow",
        "uid": uid,
        "run_id": build_run_id(
            run_key=str(options["run_key"]),
            gen_model=str(options.get("gen_model", "") or ""),
        ),
        "sample_dir": sample_root.as_posix(),
        "run_key": str(options["run_key"]),
        "video_kind": video_kind,
        "gen_model": str(options.get("gen_model", "") or ""),
    }
    for field, expected_value in expected.items():
        if result.get(field) != expected_value:
            raise RuntimeError(
                f"workflow result changed {field}: "
                f"{result.get(field)!r} != {expected_value!r}"
            )
    pipeline = result.get("pipeline")
    if not isinstance(pipeline, Mapping):
        raise RuntimeError("workflow result is missing pipeline")
    if pipeline.get("format") != "dream_exe.pipeline":
        raise RuntimeError("workflow result has an invalid pipeline schema")
    if pipeline.get("requested_stages") != list(expected_stages):
        raise RuntimeError("workflow result changed the façade-bound stage selection")
    stage_records = pipeline.get("stages")
    if not isinstance(stage_records, Mapping):
        raise RuntimeError("workflow result is missing pipeline stage records")
    selected_statuses: dict[str, str] = {}
    for stage in expected_stages:
        record = stage_records.get(stage)
        if not isinstance(record, Mapping):
            raise RuntimeError(f"workflow result is missing the {stage} stage record")
        stage_status = str(record.get("status", "") or "")
        if stage_status not in _PIPELINE_STAGE_STATUSES:
            raise RuntimeError(f"workflow result has invalid {stage} status")
        selected_statuses[stage] = stage_status
    status = str(pipeline.get("status", "") or "")
    if status not in {"completed", "failed", "partial"}:
        raise RuntimeError("workflow result has invalid pipeline status")
    if status == "completed" and any(
        stage_status not in {"completed", "reused"}
        for stage_status in selected_statuses.values()
    ):
        raise RuntimeError("completed workflow contains an incomplete selected stage")
    if stage_state_policy == "off" and any(
        stage_status == "reused" for stage_status in selected_statuses.values()
    ):
        raise RuntimeError("non-resumed workflow cannot report reused stages")
    artifacts = result.get("artifacts")
    if not isinstance(artifacts, Mapping):
        raise RuntimeError("workflow result is missing formal artifacts")
    formal = formal_artifact_paths(
        sample_root,
        str(options["run_key"]),
        gen_model=str(options.get("gen_model", "") or ""),
    )
    for name, expected_path in formal.items():
        if name == "sample_root":
            continue
        returned_path = str(artifacts.get(name, "") or "").strip()
        if (
            not returned_path
            or Path(returned_path).expanduser().absolute() != expected_path.absolute()
        ):
            raise RuntimeError(f"workflow result changed formal artifact {name}")
    if stage_state_policy == "resume":
        stage_state = result.get("stage_state")
        if not isinstance(stage_state, Mapping):
            raise RuntimeError("resumed workflow result is missing stage state")
        if stage_state.get("policy") != "resume":
            raise RuntimeError("resumed workflow result changed stage state policy")
        reuse_plan = stage_state.get("reuse_plan")
        if not isinstance(reuse_plan, Mapping):
            raise RuntimeError("resumed workflow result is missing a reuse plan")
        selected = reuse_plan.get("selected_stages")
        reused = reuse_plan.get("reused_stages")
        executed = reuse_plan.get("run_stages")
        if (
            not isinstance(selected, list)
            or not isinstance(reused, list)
            or not isinstance(executed, list)
        ):
            raise RuntimeError(
                "resumed workflow reuse plan requires selected/run/reuse stage lists"
            )
        expected_list = list(expected_stages)
        if (
            any(not isinstance(stage, str) for stage in (*selected, *reused, *executed))
            or selected != expected_list
            or len(reused) != len(set(reused))
            or len(executed) != len(set(executed))
            or reused != [stage for stage in expected_stages if stage in set(reused)]
            or executed
            != [stage for stage in expected_stages if stage in set(executed)]
            or set(reused).intersection(executed)
            or set(reused).union(executed) != set(expected_stages)
        ):
            raise RuntimeError(
                "resumed workflow reuse plan is not an exact ordered "
                "partition of bound stages"
            )
        for stage in reused:
            if selected_statuses[stage] != "reused":
                raise RuntimeError(
                    "resumed workflow reuse plan conflicts with stage records"
                )
        if status == "completed":
            for stage in executed:
                if selected_statuses[stage] != "completed":
                    raise RuntimeError(
                        "resumed workflow run plan conflicts with stage records"
                    )
    return result


def _formal_validation_settings(
    *,
    uid: str,
    sample_root: Path,
    options: Mapping[str, Any],
    expected_stages: tuple[str, ...],
) -> dict[str, Any]:
    run_key = str(options["run_key"])
    gen_model = str(options.get("gen_model", "") or "")
    video_kind, _slot = split_run_key(run_key)
    formal = formal_artifact_paths(
        sample_root,
        run_key,
        gen_model=gen_model,
    )
    context = {
        "uid": uid,
        "sample_dir": sample_root,
        "run_id": build_run_id(
            run_key=run_key,
            gen_model=gen_model,
        ),
        "run_key": run_key,
        "video_kind": video_kind,
        "gen_model": gen_model,
        "formal": formal,
    }
    settings: dict[str, Any] = {
        "context": context,
        "trajectory_path_comparison_reference_path": options.get(
            "trajectory_path_comparison_reference_path"
        ),
        "trajectory_similarity_specs": options.get("trajectory_similarity_specs"),
        # The façade forbids caller replacement of this run-local producer.
        "task_success_rate_specs": [
            {
                "uid": uid,
                "path": Path(formal["task_success"]).as_posix(),
            }
        ],
        "task_success_rate_options": options.get("task_success_rate_options"),
        "vlm_request_manifest_path": options.get("vlm_request_manifest_path"),
    }
    if "video2traj" in expected_stages:
        trajectory_path = (
            Path(options["trajectory_config_path"]).expanduser().resolve()
            if options.get("trajectory_config_path") is not None
            and str(options.get("trajectory_config_path", "")).strip()
            else sample_artifact_paths(sample_root)["trajectory_config"]
        )
        settings["video2traj_depth_contract"] = (
            build_video2traj_depth_artifact_contract(
                sample_root=sample_root,
                run_key=run_key,
                gen_model=gen_model,
                trajectory_config_path=trajectory_path,
                traj_dir=formal["traj_dir"],
            )
        )
    if not {"exec", "task_success"}.intersection(expected_stages):
        return settings

    explicit_execution_path = (
        Path(options["execution_config_path"]).expanduser().resolve()
        if options.get("execution_config_path") is not None
        and str(options.get("execution_config_path", "")).strip()
        else None
    )
    execution_request = resolve_bench_execution_request(
        sample_dir=sample_root,
        run_key=run_key,
        gen_model=gen_model,
        simulator_config_path=options.get("simulator_config_path"),
        execution_config_path=explicit_execution_path,
        action_path=formal["action"],
        output_dir=formal["exec_dir"],
    )
    execution_config = dict(execution_request["execution_config"])
    execution_section = dict(execution_config.get("execution", {}) or {})
    execution_inputs = dict(execution_config.get("input", {}) or {})
    execution_runtime = dict(execution_config.get("runtime", {}) or {})
    raw_execution_mode = str(execution_section.get("mode", "") or "").strip()
    settings.update(
        {
            "execution_mode": (
                "frame" if raw_execution_mode == "frame_traj" else raw_execution_mode
            ),
            "execution_trajectory_path": str(
                execution_request.get("trajectory_path", "") or formal["ee_traj"]
            ),
            "execution_action_path": str(
                execution_request.get("action_path", "") or formal["action"]
            ),
            "execution_traj_key": str(
                execution_inputs.get(
                    "traj_key",
                    "eef_controller",
                )
                or "eef_controller"
            ),
            "execution_max_steps": int(execution_runtime.get("max_steps", -1)),
            "task_success_options": _configured_task_success_options(
                options.get("task_success_options"),
                context=context,
                simulator_config_path=options.get("simulator_config_path"),
                execution_config_path=explicit_execution_path,
                scene_restore_options=options.get("scene_restore_options"),
                robocasa_source_root=options.get("robocasa_source_root"),
            ),
            "robocasa_source_root": options.get("robocasa_source_root"),
        }
    )
    return settings


def _validate_formal_stage_artifacts(
    *,
    uid: str,
    sample_root: Path,
    options: Mapping[str, Any],
    expected_stages: tuple[str, ...],
    workflow_result: Mapping[str, Any],
    stage_state_policy: str,
    state_lock_held: bool = False,
) -> tuple[
    dict[str, Any],
    dict[str, Mapping[str, Any]],
    dict[str, Any],
]:
    settings = _formal_validation_settings(
        uid=uid,
        sample_root=sample_root,
        options=options,
        expected_stages=expected_stages,
    )
    formal = dict(settings["context"]["formal"])
    pipeline = workflow_result.get("pipeline")
    stage_records = pipeline.get("stages") if isinstance(pipeline, Mapping) else None
    if not isinstance(stage_records, Mapping):
        raise RuntimeError("workflow result has no validated stage records")

    def validate_and_digest() -> tuple[
        dict[str, Any],
        dict[str, Mapping[str, Any]],
        dict[str, list[dict[str, Any]]],
    ]:
        validated: dict[str, Mapping[str, Any]] = {}
        stage_outputs: dict[str, list[dict[str, Any]]] = {}
        summary: dict[str, Any] = {
            "status": "verified",
            "validator": "benchmark_stage_artifact_contracts",
            "stages": {},
        }
        for stage in expected_stages:
            raw_record = stage_records.get(stage)
            stage_result = (
                raw_record.get("result") if isinstance(raw_record, Mapping) else None
            )
            if not isinstance(stage_result, Mapping):
                raise RuntimeError(f"workflow result is missing {stage} stage output")
            artifacts: Mapping[str, Any] = {}
            if stage != "video":
                artifacts = validate_benchmark_stage_artifacts(
                    stage,
                    settings=settings,
                )
                validated[stage] = artifacts
            output_paths = collect_stage_outputs(
                stage,
                result=stage_result,
                settings=settings,
            )
            output_records = digest_existing_outputs(
                output_paths,
                sample_root=sample_root,
                run_root=formal["run_root"],
            )
            if not output_records:
                raise RuntimeError(f"{stage} has no verified stage-owned outputs")
            # Re-run semantic validation after hashing, then hash again. This
            # rejects cooperative or accidental mutation during evidence
            # construction instead of trusting a prior resume sidecar.
            if stage != "video":
                validate_benchmark_stage_artifacts(
                    stage,
                    settings=settings,
                )
            repeated = digest_existing_outputs(
                collect_stage_outputs(
                    stage,
                    result=stage_result,
                    settings=settings,
                ),
                sample_root=sample_root,
                run_root=formal["run_root"],
            )
            if repeated != output_records:
                raise RuntimeError(
                    f"{stage} outputs changed during artifact validation"
                )
            stage_outputs[stage] = output_records
            summary["stages"][stage] = {
                "status": "verified",
                "artifact_groups": (
                    sorted(artifacts)
                    if stage != "video"
                    else ["explicit_saved_video_handoff"]
                ),
                "output_count": len(output_records),
            }
        return summary, validated, stage_outputs

    def verified_provenance(
        stage_outputs: Mapping[str, list[dict[str, Any]]],
    ) -> dict[str, Any]:
        if stage_state_policy != "resume":
            return {"status": "unverified", "stages": {}}
        state_path = benchmark_pipeline_state_path(
            sample_root,
            str(options["run_key"]),
            gen_model=str(options.get("gen_model", "") or ""),
        )
        state = load_benchmark_pipeline_state(state_path)
        if not isinstance(state, Mapping):
            raise RuntimeError("verified resume state is unavailable")
        expected_identity = {
            "uid": uid,
            "run_id": build_run_id(
                run_key=str(options["run_key"]),
                gen_model=str(options.get("gen_model", "") or ""),
            ),
            "run_key": str(options["run_key"]),
            "gen_model": str(options.get("gen_model", "") or ""),
        }
        if state.get("run_identity") != expected_identity:
            raise RuntimeError("resume state changed formal run identity")
        state_stages = state.get("stages")
        if not isinstance(state_stages, Mapping):
            raise RuntimeError("resume state has no stage records")
        provenance_stages: dict[str, dict[str, str]] = {}
        for stage in expected_stages:
            state_record = state_stages.get(stage)
            if (
                not isinstance(state_record, Mapping)
                or state_record.get("status") != "completed"
            ):
                raise RuntimeError(f"resume state does not verify completed {stage}")
            recorded_outputs = state_record.get("outputs")
            if not isinstance(recorded_outputs, list):
                raise RuntimeError(f"resume state has invalid {stage} outputs")
            outputs_match, reason = verify_recorded_outputs(
                recorded_outputs,
                sample_root=sample_root,
                run_root=formal["run_root"],
            )
            if not outputs_match or recorded_outputs != stage_outputs[stage]:
                raise RuntimeError(
                    f"resume state does not match live {stage} outputs: {reason}"
                )
            provenance_stages[stage] = {
                "input_fingerprint": str(state_record["input_fingerprint"]),
                "implementation_fingerprint": str(
                    state_record["implementation_fingerprint"]
                ),
            }
        return {"status": "verified", "stages": provenance_stages}

    if stage_state_policy == "resume" and not state_lock_held:
        state_path = benchmark_pipeline_state_path(
            sample_root,
            str(options["run_key"]),
            gen_model=str(options.get("gen_model", "") or ""),
        )
        with benchmark_pipeline_state_lock(state_path):
            summary, validated, stage_outputs = validate_and_digest()
            provenance = verified_provenance(stage_outputs)
    else:
        summary, validated, stage_outputs = validate_and_digest()
        provenance = verified_provenance(stage_outputs)
    return (
        summary,
        validated,
        {
            "stage_outputs": stage_outputs,
            "provenance": provenance,
        },
    )


def _run_formal_phase(
    *,
    uid: str,
    sample_root: Path,
    options: Mapping[str, Any],
    only_stages: tuple[str, ...],
    stage_state_policy: str,
    workflow_runner: SingleUIDWorkflowRunner,
    verify_formal_artifacts: bool,
    require_all_reused: bool,
) -> tuple[
    dict[str, Any],
    dict[str, Any],
    dict[str, Mapping[str, Any]],
    dict[str, Any],
]:
    invocation = {
        key: value for key, value in options.items() if key != "_explicit_video_path"
    }
    invocation.update(
        {
            "uid": uid,
            "sample_dir": sample_root,
            "source_video_path": options["_explicit_video_path"],
            "only_stages": only_stages,
            "stage_state_policy": stage_state_policy,
            "continue_on_error": False,
        }
    )
    if require_all_reused:
        invocation["require_all_reused"] = True
    result = _validate_workflow_result(
        workflow_runner(**invocation),
        uid=uid,
        sample_root=sample_root,
        options=options,
        expected_stages=only_stages,
        stage_state_policy=stage_state_policy,
    )
    if result["pipeline"]["status"] != "completed":
        return (
            result,
            {
                "status": "not_verified",
                "reason": "workflow_not_completed",
            },
            {},
            {},
        )
    if not verify_formal_artifacts:
        return (
            result,
            {
                "status": "test_only_unverified",
                "reason": (
                    "an explicitly enabled injected test workflow runner "
                    "cannot establish production artifact trust"
                ),
            },
            {},
            {},
        )
    summary, validated, evidence = _validate_formal_stage_artifacts(
        uid=uid,
        sample_root=sample_root,
        options=options,
        expected_stages=only_stages,
        workflow_result=result,
        stage_state_policy=stage_state_policy,
    )
    return result, summary, validated, evidence


def _validate_reference_file(path: str | Path) -> Path:
    candidate = Path(path).expanduser().absolute()
    if candidate.is_symlink() or not candidate.is_file():
        raise FileNotFoundError(
            f"explicit reference trajectory is not a regular file: {candidate}"
        )
    return candidate


def _validate_init_result(
    value: Mapping[str, Any],
    *,
    uid: str,
    sample_root: Path,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError("initialize must return a mapping")
    result = dict(value)
    status = str(result.get("status", "") or "").strip().lower()
    if status in _FAILED_INIT_STATUSES:
        raise RuntimeError(f"initialize reported status={status!r}")
    if "ok" in result and result["ok"] is not True:
        raise RuntimeError("initialize reported ok=false")
    if "returncode" in result and result["returncode"] != 0:
        raise RuntimeError("initialize reported a non-zero returncode")
    positive_success = (
        result.get("ok") is True
        or result.get("returncode") == 0
        or status in _SUCCESS_INIT_STATUSES
        or result.get("published") is True
    )
    if not positive_success:
        raise RuntimeError("initialize result lacks explicit positive success evidence")
    result_uid = str(result.get("uid", "") or "").strip()
    if result_uid and result_uid != uid:
        raise RuntimeError("initialize result changed uid")
    request = result.get("request")
    if request is not None:
        if not isinstance(request, Mapping):
            raise TypeError("initialize result request must be a mapping")
        request_uid = str(request.get("uid", "") or "").strip()
        if request_uid and request_uid != uid:
            raise RuntimeError("initialize request changed uid")
        request_sample = str(request.get("sample_dir", "") or "").strip()
        if (
            request_sample
            and Path(request_sample).expanduser().resolve() != sample_root
        ):
            raise RuntimeError("initialize request changed sample_dir")
    if not sample_root.is_dir():
        raise RuntimeError("initialize removed the materialized sample directory")
    return result


def run_single_uid_workflow(
    *,
    uid: str,
    sample_dir: str | Path,
    candidate_run: Mapping[str, Any],
    reference_run: Mapping[str, Any] | None = None,
    reference_trajectory_path: str | Path | None = None,
    initialize: StageRunner | None = None,
    init_options: Mapping[str, Any] | None = None,
    candidate_video_acquirer: StageRunner | None = None,
    candidate_video_options: Mapping[str, Any] | None = None,
    robocasa_source_root: str | Path | None = None,
    include_task_success: bool = True,
    stage_state_policy: str = "resume",
    workflow_runner: SingleUIDWorkflowRunner = _DEFAULT_WORKFLOW_RUNNER,
    test_only_allow_unverified_workflow_runner: bool = False,
    require_all_reused: bool = False,
) -> dict[str, Any]:
    """Run one explicit UID through init/reference/candidate boundaries.

    ``candidate_run`` and optional ``reference_run`` are ordinary
    :func:`run_benchmark_workflow` options. ``video_path`` may be explicit;
    when omitted, the current bench layout resolves it from the bound sample,
    run key, and generated-model identity.
    The top level binds UID, sample directory, source-video handoff, resume
    policy, stage selection, fail-fast behavior, and path-comparison reference
    so those values cannot be hidden in a nested mapping. A formal reference
    run is fixed to ``video2traj``. The candidate is fixed to
    ``video2traj → exec → task_success → eval``. Source videos must
    already exist; their handoff is validated before either formal run.

    Exactly one reference source is required: either a formal reference run or
    one explicit saved trajectory file. The reference run's formal
    ``ee_traj.json`` is consumed directly from its verified workflow result;
    no bench root, UID, run key, or alternative reference is discovered.

    ``sample_dir`` must already contain the materialized frozen environment.
    The optional initializer is only a test or embedding seam scoped to that
    exact directory; the public benchmark path never reconstructs an episode
    from RoboCasa or RoboMimic source datasets.

    ``candidate_video_acquirer`` is the optional top-level generation/import
    seam for the candidate only. It runs after initialization and before any
    video handoff, receives the bound ``uid`` and candidate ``gen_model``, and
    must return ``video_path`` or ``output_video``. The returned path still
    passes the normal sample/run/model handoff validation; nested run configs
    cannot provide an acquirer. Strict all-reused review forbids this seam.

    ``robocasa_source_root`` is an optional absolute RoboCasa code-checkout
    root. It is bound consistently to initialization, reference,
    candidate and task-success replay; nested overrides are
    rejected. The path is runtime-only and is never persisted in bench
    artifacts or workflow state.

    Successful return reports formally validated technical orchestration. It
    never applies paper table membership, publication penalties, scientific
    thresholds, or owner quality decisions. Optional VLM evaluators remain
    ordinary evaluation consumers supplied in ``candidate_run``.

    ``workflow_runner`` injection is a test seam. A non-default runner is
    rejected unless ``test_only_allow_unverified_workflow_runner=True`` and
    can never produce production technical completion.

    ``require_all_reused`` is a fail-closed evidence-consumer mode. It
    requires verified reuse for the trusted init adapter and every selected
    formal stage, and refuses execution before any initializer, provider,
    simulator, evaluator, recovery, or formal state publication can run.
    """

    clean_uid = str(uid or "").strip()
    if not clean_uid:
        raise ValueError("uid is required")
    sample_root = Path(sample_dir).expanduser().resolve()
    if not sample_root.is_dir():
        raise FileNotFoundError(
            f"materialized sample directory not found: {sample_root}"
        )
    if sample_root.name != clean_uid:
        raise ValueError("uid does not match materialized sample directory name")
    if not callable(workflow_runner):
        raise TypeError("workflow_runner must be callable")
    if not isinstance(
        test_only_allow_unverified_workflow_runner,
        bool,
    ):
        raise TypeError("test_only_allow_unverified_workflow_runner must be bool")
    if not isinstance(require_all_reused, bool):
        raise TypeError("require_all_reused must be bool")
    production_workflow = workflow_runner is _DEFAULT_WORKFLOW_RUNNER
    if not production_workflow and not test_only_allow_unverified_workflow_runner:
        raise ValueError(
            "workflow_runner injection is test-only; explicitly set "
            "test_only_allow_unverified_workflow_runner=True"
        )
    if not isinstance(include_task_success, bool):
        raise TypeError("include_task_success must be bool")
    if not include_task_success:
        raise ValueError("complete single-UID workflow requires task_success")
    policy = str(stage_state_policy or "").strip().lower()
    if policy not in {"off", "resume"}:
        raise ValueError("stage_state_policy must be 'off' or 'resume'")
    if require_all_reused and policy != "resume":
        raise ValueError("require_all_reused requires stage_state_policy='resume'")
    if require_all_reused and not production_workflow:
        raise ValueError(
            "require_all_reused requires the production benchmark workflow"
        )
    if (reference_run is None) == (reference_trajectory_path is None):
        raise ValueError(
            "provide exactly one of reference_run or reference_trajectory_path"
        )
    if initialize is not None and not callable(initialize):
        raise TypeError("initialize must be callable")
    if init_options is not None and not isinstance(init_options, Mapping):
        raise TypeError("init_options must be a mapping")
    if candidate_video_acquirer is not None and not callable(candidate_video_acquirer):
        raise TypeError("candidate_video_acquirer must be callable")
    if candidate_video_options is not None and not isinstance(
        candidate_video_options,
        Mapping,
    ):
        raise TypeError("candidate_video_options must be a mapping")
    normalized_candidate_video_options = dict(candidate_video_options or {})
    protected_video_options = sorted(
        {"uid", "gen_model"}.intersection(normalized_candidate_video_options)
    )
    if protected_video_options:
        raise ValueError(
            "candidate_video_options cannot override top-level fields: "
            + ", ".join(protected_video_options)
        )
    if candidate_video_acquirer is None and normalized_candidate_video_options:
        raise ValueError("candidate_video_options requires candidate_video_acquirer")
    if require_all_reused and candidate_video_acquirer is not None:
        raise ValueError("require_all_reused forbids candidate_video_acquirer")
    protected_init = sorted(
        {"robocasa_source_root", "sample_dir", "uid"}.intersection(
            dict(init_options or {})
        )
    )
    if protected_init:
        raise ValueError(
            "init_options cannot override top-level fields: "
            + ", ".join(protected_init)
        )

    candidate_options = _normalize_run_spec(
        candidate_run,
        label="candidate_run",
    )
    if (
        candidate_video_acquirer is not None
        and candidate_options["_explicit_video_path"] is not None
    ):
        raise ValueError(
            "candidate_video_acquirer cannot be combined with candidate_run.video_path"
        )
    candidate_options.setdefault("task_success_options", {})
    reference_options = (
        None
        if reference_run is None
        else _normalize_run_spec(
            reference_run,
            label="reference_run",
        )
    )
    normalized_init_options = dict(init_options or {})
    normalized_robocasa_source_root = (
        None
        if robocasa_source_root is None
        else _robocasa_runtime_source_path(
            robocasa_source_root,
            label="robocasa_source_root",
        )
    )
    if normalized_robocasa_source_root is not None:
        normalized_init_options["robocasa_source_root"] = (
            normalized_robocasa_source_root
        )
        candidate_options["robocasa_source_root"] = normalized_robocasa_source_root
        if reference_options is not None:
            reference_options["robocasa_source_root"] = normalized_robocasa_source_root
    candidate_trajectory = formal_artifact_paths(
        sample_root,
        candidate_options["run_key"],
        gen_model=candidate_options["gen_model"],
    )["ee_traj"].absolute()
    if reference_options is not None:
        formal_reference = formal_artifact_paths(
            sample_root,
            reference_options["run_key"],
            gen_model=reference_options["gen_model"],
        )["ee_traj"].absolute()
        if formal_reference == candidate_trajectory:
            raise ValueError(
                "reference_run and candidate_run must use distinct "
                "formal run identities"
            )
    else:
        assert reference_trajectory_path is not None
        if (
            Path(reference_trajectory_path).expanduser().absolute()
            == candidate_trajectory
        ):
            raise ValueError(
                "reference trajectory cannot be the candidate formal trajectory"
            )
    phases: dict[str, Any] = {
        "init": {"status": "not_requested"},
        "video_handoff": {"status": "pending"},
        "reference": {"status": "pending"},
        "candidate": {"status": "pending"},
        "acceptance": {"status": "pending"},
    }

    def fail(
        phase: str,
        error: Exception | None = None,
        *,
        reference: Path | None = None,
    ) -> dict[str, Any]:
        if error is not None:
            phases[phase] = _phase_error(error)
        return _failed_response(
            uid=clean_uid,
            sample_root=sample_root,
            phases=phases,
            failed_phase=phase,
            reference_trajectory_path=reference,
        )

    if initialize is not None:
        try:
            if require_all_reused:
                raise RuntimeError(
                    "strict verified-reuse mode forbids an initializer; "
                    "the frozen environment must already be materialized"
                )
            raw_init = initialize(
                sample_dir=sample_root,
                **normalized_init_options,
            )
            phases["init"] = {
                "status": "completed",
                "result": _validate_init_result(
                    raw_init,
                    uid=clean_uid,
                    sample_root=sample_root,
                ),
                "scope": "materialized_sample",
                "resume_tracked": False,
            }
        except Exception as error:
            return fail("init", error)

    acquisition_result: dict[str, Any] | None = None
    try:
        if candidate_video_acquirer is not None:
            raw_acquisition = candidate_video_acquirer(
                uid=clean_uid,
                gen_model=candidate_options["gen_model"],
                **normalized_candidate_video_options,
            )
            acquired_path, acquisition_result = _candidate_acquisition_path(
                raw_acquisition
            )
            candidate_options["_explicit_video_path"] = acquired_path
        candidate_handoff = resolve_benchmark_video_handoff(
            uid=clean_uid,
            sample_dir=sample_root,
            run_key=candidate_options["run_key"],
            gen_model=candidate_options["gen_model"],
            video_path=candidate_options["_explicit_video_path"],
        )
        candidate_options["_explicit_video_path"] = Path(
            candidate_handoff["video_path"]
        )
        handoff_result: dict[str, Any] = {
            "status": "completed",
            "candidate": _safe_run_summary(candidate_options),
        }
        if acquisition_result is not None:
            handoff_result["acquisition"] = acquisition_result
        if reference_options is not None:
            reference_handoff = resolve_benchmark_video_handoff(
                uid=clean_uid,
                sample_dir=sample_root,
                run_key=reference_options["run_key"],
                gen_model=reference_options["gen_model"],
                video_path=reference_options["_explicit_video_path"],
            )
            reference_options["_explicit_video_path"] = Path(
                reference_handoff["video_path"]
            )
            handoff_result["reference"] = _safe_run_summary(reference_options)
        phases["video_handoff"] = handoff_result
    except Exception as error:
        return fail("video_handoff", error)

    selected_reference: Path
    selected_reference_object: Path | None = None
    reference_evidence: dict[str, Any] = {
        "stage_outputs": None,
        "provenance": {
            "status": "explicit_artifact_only",
            "stages": {},
        },
    }
    reference_result: dict[str, Any] | None = None
    if reference_options is None:
        try:
            assert reference_trajectory_path is not None
            selected_reference = _validate_reference_file(reference_trajectory_path)
            phases["reference"] = {
                "status": "provided",
                "trajectory_path": selected_reference.as_posix(),
            }
        except Exception as error:
            return fail("reference", error)
    else:
        try:
            (
                reference_result,
                reference_artifact_validation,
                _reference_validated_artifacts,
                reference_evidence,
            ) = _run_formal_phase(
                uid=clean_uid,
                sample_root=sample_root,
                options=reference_options,
                only_stages=_REFERENCE_STAGES,
                stage_state_policy=policy,
                workflow_runner=workflow_runner,
                verify_formal_artifacts=production_workflow,
                require_all_reused=require_all_reused,
            )
            reference_status = str(reference_result["pipeline"]["status"])
            phases["reference"] = {
                "status": reference_status,
                "request": _safe_run_summary(reference_options),
                "requested_stages": list(_REFERENCE_STAGES),
                "artifact_validation": (reference_artifact_validation),
                "workflow": reference_result,
            }
            if reference_status != "completed":
                return fail("reference")
            expected_reference_paths = formal_artifact_paths(
                sample_root,
                reference_options["run_key"],
                gen_model=reference_options["gen_model"],
            )
            expected_reference = expected_reference_paths["ee_traj"]
            selected_reference_object = expected_reference_paths["obj_trajs"].absolute()
            returned_reference = (
                Path(
                    str(
                        dict(reference_result.get("artifacts", {})).get(
                            "ee_traj",
                            "",
                        )
                        or ""
                    )
                )
                .expanduser()
                .absolute()
            )
            if returned_reference != expected_reference:
                raise RuntimeError("reference workflow did not bind its formal ee_traj")
            selected_reference = _validate_reference_file(returned_reference)
            phases["reference"]["trajectory_path"] = selected_reference.as_posix()
        except Exception as error:
            return fail("reference", error)

    if candidate_options.get("trajectory_similarity_specs") is not None:
        try:
            candidate_options["trajectory_similarity_specs"] = (
                _resolve_single_uid_similarity_specs(
                    specs=list(candidate_options["trajectory_similarity_specs"]),
                    sample_root=sample_root,
                    candidate_options=candidate_options,
                    reference_trajectory=selected_reference,
                    reference_object_trajectory=(selected_reference_object),
                )
            )
        except Exception as error:
            return fail(
                "candidate",
                error,
                reference=selected_reference,
            )

    try:
        candidate_options["trajectory_path_comparison_reference_path"] = (
            selected_reference
        )
        (
            candidate_result,
            candidate_artifact_validation,
            candidate_validated_artifacts,
            candidate_evidence,
        ) = _run_formal_phase(
            uid=clean_uid,
            sample_root=sample_root,
            options=candidate_options,
            only_stages=_CANDIDATE_STAGES,
            stage_state_policy=policy,
            workflow_runner=workflow_runner,
            verify_formal_artifacts=production_workflow,
            require_all_reused=require_all_reused,
        )
        candidate_status = str(candidate_result["pipeline"]["status"])
        phases["candidate"] = {
            "status": candidate_status,
            "request": _safe_run_summary(candidate_options),
            "requested_stages": list(_CANDIDATE_STAGES),
            "reference_trajectory_path": selected_reference.as_posix(),
            "artifact_validation": candidate_artifact_validation,
            "workflow": candidate_result,
        }
        if candidate_status != "completed":
            return fail("candidate", reference=selected_reference)
    except Exception as error:
        return fail(
            "candidate",
            error,
            reference=selected_reference,
        )

    technical_status = (
        "passed"
        if production_workflow
        and candidate_artifact_validation.get("status") == "verified"
        else "test_only_unverified"
    )
    technical_completion = {
        "status": technical_status,
        "workflow_status": "completed",
        "artifact_validation": candidate_artifact_validation,
    }
    candidate_artifact_evidence: dict[str, dict[str, Any]] = {}
    candidate_artifacts = dict(candidate_result.get("artifacts", {}))
    evidence_roles = (
        ("task_success", "task_success"),
        ("evaluation_result", "eval"),
    )
    if production_workflow and all(
        Path(candidate_artifacts.get(role, "")).is_file()
        for role, _stage in evidence_roles
    ):
        candidate_run_root = Path(candidate_artifacts["run_root"])
        for role, stage in evidence_roles:
            record = digest_output_file(
                candidate_artifacts[role],
                sample_root=sample_root,
                run_root=candidate_run_root,
            )
            stage_outputs = candidate_evidence.get("stage_outputs", {}).get(stage)
            if not isinstance(stage_outputs, list) or not any(
                isinstance(item, Mapping)
                and item.get("size") == record["size"]
                and item.get("sha256") == record["sha256"]
                for item in stage_outputs
            ):
                raise RuntimeError(
                    f"candidate {role} digest is not bound to verified {stage} outputs"
                )
            candidate_artifact_evidence[role] = record
    phases["acceptance"] = {
        "status": "not_requested",
        "scope": "technical_workflow",
    }
    top_status = (
        "technical_completed" if technical_status == "passed" else "test_only_completed"
    )

    return {
        "format": SINGLE_UID_WORKFLOW_SCHEMA,
        "status": top_status,
        "status_scope": "technical_workflow",
        "uid": clean_uid,
        "sample_dir": sample_root.as_posix(),
        "failed_phase": None,
        "phase_order": list(_PHASE_ORDER),
        "phases": phases,
        "reference_trajectory_path": selected_reference.as_posix(),
        "candidate_artifacts": candidate_artifacts,
        **(
            {"candidate_artifact_evidence": candidate_artifact_evidence}
            if candidate_artifact_evidence
            else {}
        ),
        "technical_completion": technical_completion,
        "technical_complete": technical_status == "passed",
    }


__all__ = [
    "SINGLE_UID_WORKFLOW_SCHEMA",
    "SingleUIDWorkflowRunner",
    "TECHNICAL_COMPLETENESS_PROFILE",
    "run_single_uid_workflow",
]
