"""Aggregate saved single-case results without rerunning producers."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

from ...evaluation.vlm.batch import (
    RUBRIC_PHYSICAL_PLAUSIBILITY,
    RUBRIC_SUBJECT_STABILITY,
    RUBRIC_TASK_ADHERENCE,
)
from ...evaluation.execution import aggregate_execution_metrics
from ...evaluation.execution import aggregate_task_success_payloads
from ...evaluation.trajectory import (
    DEFAULT_TRAJECTORY_SIMILARITY_PROTOCOL,
    TRAJECTORY_SIMILARITY_METRICS,
    aggregate_trajectory_similarity,
)
from ...pipeline.runner.single_case import SINGLE_UID_WORKFLOW_SCHEMA

FORMAL_CANDIDATE_STAGES = frozenset(
    {"video2traj", "exec", "task_success", "eval"}
)

def _candidate_evaluation(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    phases = result.get("phases")
    candidate = phases.get("candidate") if isinstance(phases, Mapping) else None
    workflow = candidate.get("workflow") if isinstance(candidate, Mapping) else None
    pipeline = workflow.get("pipeline") if isinstance(workflow, Mapping) else None
    results = pipeline.get("results") if isinstance(pipeline, Mapping) else None
    evaluation = results.get("eval") if isinstance(results, Mapping) else None
    return evaluation if isinstance(evaluation, Mapping) else None


def _candidate_task_success(result: Mapping[str, Any]) -> Mapping[str, Any] | None:
    phases = result.get("phases")
    candidate = phases.get("candidate") if isinstance(phases, Mapping) else None
    workflow = candidate.get("workflow") if isinstance(candidate, Mapping) else None
    pipeline = workflow.get("pipeline") if isinstance(workflow, Mapping) else None
    results = pipeline.get("results") if isinstance(pipeline, Mapping) else None
    stage = results.get("task_success") if isinstance(results, Mapping) else None
    payload = stage.get("result") if isinstance(stage, Mapping) else None
    return payload if isinstance(payload, Mapping) else None


def _task_succeeded(
    result: Mapping[str, Any],
    *,
    uid: str | None = None,
) -> bool:
    if result.get("format") != SINGLE_UID_WORKFLOW_SCHEMA:
        return False
    if uid is not None and str(result.get("uid", "") or "") != uid:
        return False
    if result.get("failed_phase") is not None:
        return False
    technical = result.get("technical_completion")
    if not isinstance(technical, Mapping) or technical.get("status") != "passed":
        return False
    phases = result.get("phases")
    candidate = phases.get("candidate") if isinstance(phases, Mapping) else None
    workflow = candidate.get("workflow") if isinstance(candidate, Mapping) else None
    pipeline = workflow.get("pipeline") if isinstance(workflow, Mapping) else None
    stage_results = pipeline.get("results") if isinstance(pipeline, Mapping) else None
    return isinstance(stage_results, Mapping) and FORMAL_CANDIDATE_STAGES.issubset(
        stage_results
    )


def _finite_metric_value(value: Any) -> bool:
    if isinstance(value, bool):
        return False
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError, OverflowError):
        return False


def aggregate_technical_run_results(
    *,
    run: Mapping[str, Any],
    task_results: list[Mapping[str, Any]],
    task_failure_boundaries: list[Mapping[str, Any]],
    expected_uids: Sequence[str],
    require_vlm: bool = True,
) -> dict[str, Any]:
    """Aggregate public cohort metrics without selecting a paper table."""

    expected = list(expected_uids)
    by_uid = {
        str(result.get("uid", "") or ""): result
        for result in task_results
        if isinstance(result, Mapping)
    }
    missing = [uid for uid in expected if uid not in by_uid]
    unsuccessful = [
        uid
        for uid in expected
        if uid in by_uid and not _task_succeeded(by_uid[uid], uid=uid)
    ]
    failed_boundaries = [
        dict(record)
        for record in task_failure_boundaries
        if isinstance(record, Mapping) and str(record.get("uid", "") or "") in expected
    ]
    boundary_failed_uids = sorted(
        {
            str(record["uid"])
            for record in failed_boundaries
            if str(record.get("uid", "") or "")
        }
    )
    missing = [uid for uid in missing if uid not in boundary_failed_uids]
    if missing or unsuccessful or failed_boundaries:
        return {
            "status": "incomplete",
            "expected_uids": expected,
            "missing_uids": missing,
            "unsuccessful_uids": unsuccessful,
            "boundary_failed_uids": boundary_failed_uids,
            "failure_boundaries": failed_boundaries,
            "reason": (
                "task_boundary_failure"
                if failed_boundaries and not missing and not unsuccessful
                else "task_result_incomplete"
            ),
        }

    successful = [by_uid[uid] for uid in expected]
    task_specs = [
        {"uid": str(result["uid"]), "payload": payload}
        for result in successful
        if (payload := _candidate_task_success(result)) is not None
    ]
    if len(task_specs) != len(expected):
        return {
            "status": "incomplete",
            "expected_uids": expected,
            "reason": "task_success_payload_missing",
        }
    task_success = aggregate_task_success_payloads(
        task_specs,
        include_records=True,
        metadata={
            "suite_run_id": run["id"],
            "video_source": dict(run["video_source"]),
        },
    )

    similarity_by_group: dict[str, list[Mapping[str, Any]]] = {}
    similarity_uids_by_group: dict[str, list[str]] = {}
    execution_rows: list[Mapping[str, Any]] = []
    required_video_only_rubrics = {
        RUBRIC_SUBJECT_STABILITY,
        RUBRIC_PHYSICAL_PLAUSIBILITY,
        RUBRIC_TASK_ADHERENCE,
    }
    vlm_request_counts = {"video_only": 0, "video_trajectory": 0}
    vlm_uid_coverage_counts = {"video_only": 0, "video_trajectory": 0}
    vlm_coverage_failures: list[dict[str, Any]] = []
    for result in successful:
        uid = str(result["uid"])
        evaluation = _candidate_evaluation(result)
        if evaluation is None:
            return {
                "status": "incomplete",
                "expected_uids": expected,
                "reason": "evaluation_missing",
                "uid": uid,
            }
        trajectory = evaluation.get("trajectory")
        if not isinstance(trajectory, Mapping):
            return {
                "status": "incomplete",
                "expected_uids": expected,
                "reason": "trajectory_evaluation_missing",
                "uid": uid,
            }
        similarity = trajectory.get("similarity")
        if not isinstance(similarity, list) or not similarity:
            return {
                "status": "incomplete",
                "expected_uids": expected,
                "reason": "trajectory_similarity_missing",
                "uid": uid,
            }
        uid_groups: set[str] = set()
        for record in similarity:
            if not isinstance(record, Mapping):
                return {
                    "status": "incomplete",
                    "expected_uids": expected,
                    "reason": "trajectory_similarity_record_invalid",
                    "uid": uid,
                }
            group = str(record.get("group", "") or "").strip()
            if not group or group in uid_groups:
                return {
                    "status": "incomplete",
                    "expected_uids": expected,
                    "reason": "trajectory_similarity_group_invalid",
                    "uid": uid,
                    "group": group,
                }
            uid_groups.add(group)
            similarity_by_group.setdefault(group, []).append(record)
            similarity_uids_by_group.setdefault(group, []).append(uid)
        execution = trajectory.get("executability")
        if not isinstance(execution, Mapping):
            return {
                "status": "incomplete",
                "expected_uids": expected,
                "reason": "execution_metrics_missing",
                "uid": uid,
            }
        execution_rows.append(execution)

        if not require_vlm:
            continue
        vlm = evaluation.get("vlm_evaluations")
        if not isinstance(vlm, list):
            vlm_coverage_failures.append(
                {"uid": uid, "reason": "vlm_evaluations_missing"}
            )
            continue
        rubric_counts = {rubric: 0 for rubric in sorted(required_video_only_rubrics)}
        video_only_clean = True
        trajectory_count = 0
        trajectory_clean = True
        for record in vlm:
            if not isinstance(record, Mapping):
                video_only_clean = False
                trajectory_clean = False
                continue
            mode = str(record.get("mode", "") or "")
            raw_vlm_result = record.get("result")
            vlm_result = raw_vlm_result if isinstance(raw_vlm_result, Mapping) else {}
            if mode == "video_only":
                vlm_request_counts[mode] += 1
                rubric = str(vlm_result.get("rubric", "") or "")
                if rubric in rubric_counts:
                    rubric_counts[rubric] += 1
                else:
                    video_only_clean = False
                if (
                    vlm_result.get("status") != "completed"
                    or vlm_result.get("record_status") != "ok"
                ):
                    video_only_clean = False
            elif mode == "video_trajectory":
                vlm_request_counts[mode] += 1
                trajectory_count += 1
                if (
                    vlm_result.get("status") != "completed"
                    or vlm_result.get("record_status") != "ok"
                    or record.get("paper_metric_compatibility") != "not_claimed"
                    or record.get("metric_claims") != []
                ):
                    trajectory_clean = False
            else:
                video_only_clean = False
                trajectory_clean = False
        video_only_complete = video_only_clean and all(
            count == 1 for count in rubric_counts.values()
        )
        trajectory_complete = trajectory_clean and trajectory_count == 1
        if video_only_complete:
            vlm_uid_coverage_counts["video_only"] += 1
        if trajectory_complete:
            vlm_uid_coverage_counts["video_trajectory"] += 1
        if not video_only_complete or not trajectory_complete:
            vlm_coverage_failures.append(
                {
                    "uid": uid,
                    "reason": "canonical_vlm_request_set_incomplete",
                    "video_only_rubric_counts": rubric_counts,
                    "video_trajectory_count": trajectory_count,
                }
            )

    if any(
        len(records) != len(expected)
        or sorted(similarity_uids_by_group[group]) != sorted(expected)
        for group, records in similarity_by_group.items()
    ):
        return {
            "status": "incomplete",
            "expected_uids": expected,
            "reason": "trajectory_group_denominator_mismatch",
            "group_uid_coverage": {
                group: list(similarity_uids_by_group[group])
                for group in sorted(similarity_uids_by_group)
            },
        }
    if len(execution_rows) != len(expected):
        return {
            "status": "incomplete",
            "expected_uids": expected,
            "reason": "execution_metrics_denominator_mismatch",
        }
    if require_vlm and (
        vlm_coverage_failures
        or any(
            vlm_uid_coverage_counts[mode] != len(expected)
            for mode in vlm_uid_coverage_counts
        )
    ):
        return {
            "status": "incomplete",
            "expected_uids": expected,
            "reason": "vlm_mode_coverage_missing",
            "vlm_request_counts": vlm_request_counts,
            "vlm_uid_coverage_counts": vlm_uid_coverage_counts,
            "vlm_coverage_failures": vlm_coverage_failures,
        }

    similarity_aggregates: dict[str, Any] = {}
    for group, records in sorted(similarity_by_group.items()):
        first_protocol = records[0].get("protocol") if records else None
        protocol = (
            str(first_protocol.get("id"))
            if isinstance(first_protocol, Mapping) and first_protocol.get("id")
            else DEFAULT_TRAJECTORY_SIMILARITY_PROTOCOL
        )
        similarity_aggregates[group] = aggregate_trajectory_similarity(
            records,
            protocol=protocol,
            expected_count=len(expected),
            metadata={"suite_run_id": run["id"]},
        )
    execution_aggregate = aggregate_execution_metrics(
        execution_rows,
        expected_count=len(expected),
    )
    similarity_coverage: dict[str, Any] = {}
    for group, records in sorted(similarity_by_group.items()):
        metric_coverage: dict[str, Any] = {}
        for metric_name in TRAJECTORY_SIMILARITY_METRICS:
            computed_uids: list[str] = []
            missing_uids: list[str] = []
            for uid, record in zip(
                similarity_uids_by_group[group],
                records,
                strict=True,
            ):
                raw_metrics = record.get("metrics")
                metrics = raw_metrics if isinstance(raw_metrics, Mapping) else {}
                raw_metric = metrics.get(metric_name)
                metric = raw_metric if isinstance(raw_metric, Mapping) else {}
                if metric.get("status") == "computed" and _finite_metric_value(
                    metric.get("value")
                ):
                    computed_uids.append(uid)
                else:
                    missing_uids.append(uid)
            metric_coverage[metric_name] = {
                "expected_count": len(expected),
                "computed_count": len(computed_uids),
                "missing_count": len(missing_uids),
                "computed_uids": computed_uids,
                "missing_uids": missing_uids,
            }
        similarity_coverage[group] = {"metrics": metric_coverage}
    similarity_complete = all(
        metric["missing_count"] == 0
        for group in similarity_coverage.values()
        for metric in group["metrics"].values()
    )
    execution_complete = all(
        metric.get("missing_count") == 0
        for metric in execution_aggregate["metrics"].values()
    )
    metric_coverage_status = (
        "complete" if similarity_complete and execution_complete else "incomplete"
    )
    aggregate: dict[str, Any] = {
        "status": "passed",
        "status_scope": "pipeline_and_aggregation_completed",
        "metric_coverage_status": metric_coverage_status,
        "all_metrics_computable": similarity_complete and execution_complete,
        "aggregation_scope": (
            "full_with_vlm" if require_vlm else "technical_without_vlm"
        ),
        "video_source": dict(run["video_source"]),
        "tasks_total": len(task_results),
        "tasks_succeeded": len(successful),
        "trajectory": {
            "similarity": similarity_aggregates,
            "execution": execution_aggregate,
            "metric_coverage": {
                "status": metric_coverage_status,
                "expected_uid_count": len(expected),
                "similarity_complete": similarity_complete,
                "execution_complete": execution_complete,
                "similarity": similarity_coverage,
                "execution": execution_aggregate["metrics"],
            },
        },
        "task_success_rate": task_success,
    }
    if require_vlm:
        aggregate["vlm"] = {
            "modes": ["video_only", "video_trajectory"],
            "request_counts": vlm_request_counts,
            "uid_coverage_counts": vlm_uid_coverage_counts,
            "video_only_required_rubrics": sorted(required_video_only_rubrics),
            "video_trajectory_paper_metric_compatibility": "not_claimed",
        }
    else:
        aggregate["vlm"] = {
            "status": "not_in_scope",
            "reason": "technical aggregate excludes provider-backed VLM evidence",
        }
    return aggregate




__all__ = ["aggregate_technical_run_results"]
