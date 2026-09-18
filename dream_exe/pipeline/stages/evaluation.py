"""Run the trajectory, execution, task-success, and VLM evaluations."""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...artifacts.layout import execution_artifact_paths
from ...evaluation.execution import build_metrics_from_exec_dir
from ...evaluation.execution import aggregate_task_success_paths
from ...evaluation.trajectory import (
    evaluate_trajectory_path_comparison_paths,
    evaluate_trajectory_similarity_files,
)
from ...evaluation.contracts import (
    EvaluationResultValidationError,
    build_evaluation_plan,
    build_evaluation_result_bundle,
    formal_protected_output_paths,
    publish_evaluation_result_bundle,
)
from ..records.layout import (
    build_run_id,
    formal_artifact_paths,
    normalize_run_key,
    split_run_key,
)
from ...evaluation.vlm.requests import (
    VLMEvaluatorRegistry,
    evaluate_vlm_request_manifest,
)

MetricsBuilder = Callable[[Path], Mapping[str, Any]]
TrajectorySimilarityEvaluator = Callable[..., Mapping[str, Any]]
TaskSuccessRateBuilder = Callable[..., Mapping[str, Any]]
TrajectoryPathComparisonEvaluator = Callable[..., Mapping[str, Any]]

_TRAJECTORY_SIMILARITY_SPEC_FIELDS = {
    "predicted_path",
    "reference_path",
    "group",
    "protocol",
    "visibility_threshold",
    "fps",
    "normalization_overrides",
}
_EXEC_METRICS_TRANSPORT_FIELDS = frozenset(
    {
        "exec_metrics_json",
        "exec_metrics_path",
        "exec_metrics_per_frame_csv",
    }
)


def _sensitive_key(key: Any) -> bool:
    normalized = str(key).strip().lower().replace("-", "_")
    sensitive_names = {
        "api_key",
        "apikey",
        "access_token",
        "auth_token",
        "authorization",
        "bearer_token",
        "credential",
        "credentials",
        "password",
        "refresh_token",
        "secret",
        "token",
    }
    return (
        normalized in sensitive_names
        or normalized.endswith("_api_key")
        or normalized.endswith("_password")
        or normalized.endswith("_secret")
    )


def _reject_sensitive_payload(
    value: Any,
    *,
    label: str,
) -> None:
    if isinstance(value, Mapping):
        for key, nested in value.items():
            if _sensitive_key(key):
                raise ValueError(
                    f"{label} cannot contain credentials; bind secrets "
                    "inside the evaluator runtime"
                )
            _reject_sensitive_payload(nested, label=label)
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        for nested in value:
            _reject_sensitive_payload(nested, label=label)


def evaluate_benchmark_execution_stage(
    *,
    uid: str,
    sample_dir: str | Path,
    run_id: str,
    run_key: str,
    video_kind: str,
    gen_model: str,
    output_exec_root: str | Path,
    metrics_builder: MetricsBuilder | None = None,
    vlm_request_manifest_path: str | Path | None = None,
    vlm_evaluators: VLMEvaluatorRegistry | None = None,
    _publish_result_bundle: bool = True,
) -> dict[str, Any]:
    """Evaluate one explicit formal execution output.

    The deterministic evaluator writes or refreshes only the normal metrics
    files below ``output_exec_root``. Optional VLM evaluation is driven by an
    explicit request manifest whose outputs are declared before execution.
    """

    clean_uid = str(uid or "").strip()
    if not clean_uid:
        raise ValueError("uid is required")
    sample_root = Path(sample_dir).expanduser().resolve()
    if not sample_root.is_dir():
        raise FileNotFoundError(
            f"explicit bench sample directory not found: {sample_root}"
        )
    if sample_root.name != clean_uid:
        raise ValueError(
            f"benchmark uid/sample_dir mismatch: {clean_uid!r} != {sample_root.name!r}"
        )
    normalized_run_key = normalize_run_key(run_key)
    expected_video_kind, _slot = split_run_key(normalized_run_key)
    if str(video_kind or "").strip() != expected_video_kind:
        raise ValueError("benchmark video_kind does not match run_key")
    clean_gen_model = str(gen_model or "").strip()
    if expected_video_kind == "rollout" and clean_gen_model:
        raise ValueError("rollout benchmark runs cannot name a gen_model")
    expected_run_id = build_run_id(
        run_key=normalized_run_key,
        gen_model=clean_gen_model,
    )
    if str(run_id or "").strip() != expected_run_id:
        raise ValueError("benchmark run_id does not match run identity")

    execution_root = Path(output_exec_root).expanduser().resolve()
    formal = formal_artifact_paths(
        sample_root,
        normalized_run_key,
        gen_model=clean_gen_model,
    )
    expected_execution_root = formal["exec_dir"]
    if execution_root != expected_execution_root:
        raise ValueError("output_exec_root does not match the formal benchmark run")
    summary_path = execution_artifact_paths(execution_root)["exec_summary"]
    if not summary_path.is_file():
        raise FileNotFoundError(f"execution summary not found: {summary_path}")

    explicit_vlm_requested = vlm_request_manifest_path is not None or vlm_evaluators is not None
    if explicit_vlm_requested:
        if vlm_request_manifest_path is None:
            raise ValueError(
                "vlm_request_manifest_path is required with vlm_evaluators"
            )
        if vlm_evaluators is None:
            raise ValueError(
                "vlm_evaluators is required with vlm_request_manifest_path"
            )

    run_identity = {
        "uid": clean_uid,
        "run_id": expected_run_id,
        "run_key": normalized_run_key,
        "video_kind": expected_video_kind,
        "gen_model": clean_gen_model,
    }
    preflight_plan = build_evaluation_plan(
        formal_artifacts=formal,
        run_identity=run_identity,
        vlm_request_manifest_path=vlm_request_manifest_path,
    )

    if metrics_builder is None:
        returned_metrics = build_metrics_from_exec_dir(
            execution_root,
            run_key=normalized_run_key,
            gen_model=(clean_gen_model or None),
        )
        deterministic = {
            key: copy.deepcopy(value)
            for key, value in returned_metrics.items()
            if key not in _EXEC_METRICS_TRANSPORT_FIELDS
        }
    else:
        deterministic = metrics_builder(execution_root)
    if not isinstance(deterministic, Mapping):
        raise TypeError("metrics_builder must return a mapping")

    vlm_evaluations: list[dict[str, Any]] = []
    if explicit_vlm_requested:
        assert vlm_request_manifest_path is not None
        assert vlm_evaluators is not None
        vlm_evaluations = evaluate_vlm_request_manifest(
            vlm_request_manifest_path,
            expected_run_identity={
                "uid": clean_uid,
                "run_id": expected_run_id,
                "run_key": normalized_run_key,
                "video_kind": expected_video_kind,
                "gen_model": clean_gen_model,
            },
            exec_dir=execution_root,
            evaluators=vlm_evaluators,
            protected_output_paths=formal_protected_output_paths(formal),
            durable_run_root=formal["run_root"],
            expected_formal_run_root=formal["run_root"],
            expected_request_sha256s=[
                entry["request_sha256"] for entry in preflight_plan["vlm"]
            ],
            expected_manifest_fingerprint=(
                None
                if preflight_plan["vlm_request_manifest"] is None
                else {
                    "size": preflight_plan["vlm_request_manifest"]["size"],
                    "sha256": preflight_plan["vlm_request_manifest"]["sha256"],
                }
            ),
        )

    result = {
        "ok": True,
        "returncode": 0,
        "uid": clean_uid,
        "run_id": expected_run_id,
        "run_key": normalized_run_key,
        "video_kind": expected_video_kind,
        "gen_model": clean_gen_model,
        "exec_dir": execution_root.as_posix(),
        "deterministic": copy.deepcopy(dict(deterministic)),
        "vlm_evaluations": vlm_evaluations,
    }
    postflight_plan = build_evaluation_plan(
        formal_artifacts=formal,
        run_identity=run_identity,
        vlm_request_manifest_path=vlm_request_manifest_path,
    )
    if postflight_plan != preflight_plan:
        raise EvaluationResultValidationError(
            "evaluation_plan_changed_during_execution"
        )
    if _publish_result_bundle:
        bundle = build_evaluation_result_bundle(
            formal_artifacts=formal,
            run_identity=run_identity,
            result=result,
            plan=preflight_plan,
        )
        publish_evaluation_result_bundle(
            bundle,
            formal_artifacts=formal,
        )
    return result


def _validated_trajectory_similarity_specs(
    specs: Sequence[Mapping[str, Any]] | None,
) -> list[dict[str, Any]] | None:
    if specs is None:
        return None
    if isinstance(specs, (str, bytes, bytearray)):
        raise TypeError("trajectory_similarity_specs must be a sequence of mappings")
    validated: list[dict[str, Any]] = []
    for index, raw in enumerate(specs):
        if not isinstance(raw, Mapping):
            raise TypeError(f"trajectory similarity spec {index} must be a mapping")
        unknown = sorted(set(raw) - _TRAJECTORY_SIMILARITY_SPEC_FIELDS)
        if unknown:
            raise ValueError(
                f"trajectory similarity spec {index} has unsupported fields: "
                + ", ".join(unknown)
            )
        spec = copy.deepcopy(dict(raw))
        missing = [
            field
            for field in ("predicted_path", "reference_path", "group")
            if not str(spec.get(field, "") or "").strip()
        ]
        if missing:
            raise ValueError(
                f"trajectory similarity spec {index} is missing: " + ", ".join(missing)
            )
        validated.append(spec)
    return validated


def _validated_path_comparison_reference_path(
    value: str | Path | None,
) -> Path | None:
    if value is None:
        return None
    if not isinstance(value, (str, Path)):
        raise TypeError("trajectory_path_comparison_reference_path must be a path")
    text = str(value).strip()
    if not text:
        raise ValueError("trajectory_path_comparison_reference_path cannot be empty")
    return Path(text).expanduser().absolute()


def evaluate_benchmark_domains_stage(
    *,
    uid: str,
    sample_dir: str | Path,
    run_id: str,
    run_key: str,
    video_kind: str,
    gen_model: str,
    output_exec_root: str | Path,
    metrics_builder: MetricsBuilder | None = None,
    vlm_request_manifest_path: str | Path | None = None,
    vlm_evaluators: VLMEvaluatorRegistry | None = None,
    trajectory_path_comparison_reference_path: str | Path | None = None,
    trajectory_path_comparison_evaluator: (
        TrajectoryPathComparisonEvaluator | None
    ) = None,
    trajectory_similarity_specs: Sequence[Mapping[str, Any]] | None = None,
    trajectory_similarity_evaluator: (TrajectorySimilarityEvaluator | None) = None,
    task_success_rate_specs: Sequence[Mapping[str, Any]] | None = None,
    task_success_rate_builder: TaskSuccessRateBuilder | None = None,
    task_success_rate_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the three public domains beneath the existing ``eval`` stage.

    Similarity, path-comparison, task-SR, and VLM inputs are explicit; this
    adapter never guesses a reference trajectory or cohort. The compared
    trajectory is the current formal run artifact.
    """

    path_comparison_reference = _validated_path_comparison_reference_path(
        trajectory_path_comparison_reference_path
    )
    if trajectory_path_comparison_evaluator is not None and not callable(
        trajectory_path_comparison_evaluator
    ):
        raise TypeError("trajectory_path_comparison_evaluator must be callable")
    if (
        trajectory_path_comparison_evaluator is not None
        and path_comparison_reference is None
    ):
        raise ValueError(
            "trajectory_path_comparison_evaluator requires an explicit "
            "trajectory_path_comparison_reference_path"
        )

    similarity_specs = _validated_trajectory_similarity_specs(
        trajectory_similarity_specs
    )
    if task_success_rate_specs is not None and isinstance(
        task_success_rate_specs,
        (str, bytes, bytearray),
    ):
        raise TypeError("task_success_rate_specs must be a sequence of mappings")
    if task_success_rate_specs is not None:
        for index, spec in enumerate(task_success_rate_specs):
            if not isinstance(spec, Mapping):
                raise TypeError(f"task success-rate spec {index} must be a mapping")
    task_specs = (
        None
        if task_success_rate_specs is None
        else [copy.deepcopy(dict(spec)) for spec in task_success_rate_specs]
    )
    if task_success_rate_options is None:
        task_options: dict[str, Any] = {}
    elif isinstance(task_success_rate_options, Mapping):
        task_options = copy.deepcopy(dict(task_success_rate_options))
    else:
        raise TypeError("task_success_rate_options must be a mapping")
    _reject_sensitive_payload(
        task_options,
        label="task_success_rate_options",
    )

    sample_root = Path(sample_dir).expanduser().resolve()
    normalized_run_key = normalize_run_key(run_key)
    expected_video_kind, _slot = split_run_key(normalized_run_key)
    clean_gen_model = str(gen_model or "").strip()
    expected_run_id = build_run_id(
        run_key=normalized_run_key,
        gen_model=clean_gen_model,
    )
    formal = formal_artifact_paths(
        sample_root,
        normalized_run_key,
        gen_model=clean_gen_model,
    )
    preflight_run_identity = {
        "uid": str(uid or "").strip(),
        "run_id": expected_run_id,
        "run_key": normalized_run_key,
        "video_kind": expected_video_kind,
        "gen_model": clean_gen_model,
    }
    preflight_plan = build_evaluation_plan(
        formal_artifacts=formal,
        run_identity=preflight_run_identity,
        trajectory_path_comparison_reference_path=path_comparison_reference,
        trajectory_similarity_specs=similarity_specs,
        task_success_rate_specs=task_specs,
        task_success_rate_options=task_options,
        vlm_request_manifest_path=vlm_request_manifest_path,
    )

    execution_result = evaluate_benchmark_execution_stage(
        uid=uid,
        sample_dir=sample_dir,
        run_id=run_id,
        run_key=run_key,
        video_kind=video_kind,
        gen_model=gen_model,
        output_exec_root=output_exec_root,
        metrics_builder=metrics_builder,
        vlm_request_manifest_path=vlm_request_manifest_path,
        vlm_evaluators=vlm_evaluators,
        _publish_result_bundle=False,
    )

    similarity_results: list[dict[str, Any]] | None = None
    if similarity_specs is not None:
        evaluator = (
            evaluate_trajectory_similarity_files
            if trajectory_similarity_evaluator is None
            else trajectory_similarity_evaluator
        )
        if not callable(evaluator):
            raise TypeError("trajectory_similarity_evaluator must be callable")
        similarity_results = []
        for raw_spec in similarity_specs:
            spec = copy.deepcopy(raw_spec)
            predicted = spec.pop("predicted_path")
            reference = spec.pop("reference_path")
            raw_result = evaluator(predicted, reference, **spec)
            if not isinstance(raw_result, Mapping):
                raise TypeError("trajectory_similarity_evaluator must return a mapping")
            similarity_results.append(copy.deepcopy(dict(raw_result)))

    path_comparison: dict[str, Any] | None = None
    if path_comparison_reference is not None:
        path_evaluator = (
            evaluate_trajectory_path_comparison_paths
            if trajectory_path_comparison_evaluator is None
            else trajectory_path_comparison_evaluator
        )
        raw_path_comparison = path_evaluator(
            trajectory_path=formal["ee_traj"],
            reference_trajectory_path=path_comparison_reference,
        )
        if not isinstance(raw_path_comparison, Mapping):
            raise TypeError(
                "trajectory path comparison evaluator must return a mapping"
            )
        path_comparison = copy.deepcopy(dict(raw_path_comparison))
        _reject_sensitive_payload(
            path_comparison,
            label="trajectory path comparison result",
        )
    task_success_rate: dict[str, Any] | None = None
    if task_specs is not None:
        builder = (
            aggregate_task_success_paths
            if task_success_rate_builder is None
            else task_success_rate_builder
        )
        if not callable(builder):
            raise TypeError("task_success_rate_builder must be callable")
        raw_task_result = builder(task_specs, **task_options)
        if not isinstance(raw_task_result, Mapping):
            raise TypeError("task_success_rate_builder must return a mapping")
        task_success_rate = copy.deepcopy(dict(raw_task_result))
        _reject_sensitive_payload(
            task_success_rate,
            label="task_success_rate_builder result",
        )

    result = copy.deepcopy(dict(execution_result))
    result["trajectory"] = {
        "similarity": similarity_results,
        "executability": copy.deepcopy(result["deterministic"]),
    }
    if path_comparison_reference is not None:
        result["trajectory"]["path_comparison"] = path_comparison
    result["task_success_rate"] = task_success_rate
    run_identity = {
        "uid": result["uid"],
        "run_id": result["run_id"],
        "run_key": result["run_key"],
        "video_kind": result["video_kind"],
        "gen_model": result["gen_model"],
    }
    postflight_plan = build_evaluation_plan(
        formal_artifacts=formal,
        run_identity=run_identity,
        trajectory_path_comparison_reference_path=path_comparison_reference,
        trajectory_similarity_specs=similarity_specs,
        task_success_rate_specs=task_specs,
        task_success_rate_options=task_options,
        vlm_request_manifest_path=vlm_request_manifest_path,
    )
    if run_identity != preflight_run_identity or postflight_plan != preflight_plan:
        raise EvaluationResultValidationError(
            "evaluation_plan_changed_during_execution"
        )
    bundle = build_evaluation_result_bundle(
        formal_artifacts=formal,
        run_identity=run_identity,
        result=result,
        plan=preflight_plan,
    )
    publish_evaluation_result_bundle(
        bundle,
        formal_artifacts=formal,
    )
    return result


__all__ = [
    "MetricsBuilder",
    "TaskSuccessRateBuilder",
    "TrajectoryPathComparisonEvaluator",
    "TrajectorySimilarityEvaluator",
    "VLMEvaluatorRegistry",
    "evaluate_benchmark_domains_stage",
    "evaluate_benchmark_execution_stage",
]
