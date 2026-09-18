"""Workspace-driven evaluation over one case or a complete collection.

The benchmark and published results are read-only. Evaluation reports are
written below the workspace outputs root with portable source identities.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..bench.data.repository import BenchRepository
from ..bench.data.workspace import Workspace, load_workspace
from ..bench.outputs.lifecycle import write_json_atomic
from .execution import (
    aggregate_execution_metrics,
    aggregate_task_success_payloads,
    task_success_record_from_payload,
)
from .trajectory import (
    DEFAULT_TRAJECTORY_SIMILARITY_PROTOCOL,
    GROUP_EEF_TCP,
    GROUP_EEF_VIS,
    GROUP_OBJ,
    TRAJECTORY_SIMILARITY_GROUPS,
    aggregate_trajectory_similarity,
    evaluate_trajectory_similarity_files,
)


EVALUATION_FAMILIES = ("visual", "trajectory", "executability", "task", "all")
_TRAJECTORY_FILES = {
    GROUP_EEF_VIS: "trajectory/eef.json",
    GROUP_EEF_TCP: "trajectory/eef.json",
    GROUP_OBJ: "trajectory/objects.json",
}
_TRAJECTORY_SLUGS = {
    GROUP_EEF_VIS: "eef_vis",
    GROUP_EEF_TCP: "eef_tcp",
    GROUP_OBJ: "obj",
}
_EXECUTABILITY_FIELDS = {
    "E-SR": "exec_sr",
    "nDTW": "tracking_ndtw",
    "Pos95": "pos_p95",
    "Rot95": "rot_p95",
    "Smth": "executed_smoothness",
}


@dataclass(frozen=True)
class EvaluationSelection:
    workspace: Workspace
    repository: BenchRepository
    uids: tuple[str, ...]
    candidate_model: str
    prompt_variant: str
    output_root: Path

    def candidate_bundle(self, uid: str) -> Path:
        return _first_bundle(
            self.workspace,
            uid=uid,
            model=self.candidate_model,
            variant=self.prompt_variant,
        )

    def reference_bundle(self, uid: str) -> Path:
        return _first_bundle(
            self.workspace,
            uid=uid,
            model="reference",
            variant="w_gt_depth",
        )


def _safe_name(value: str, *, label: str) -> str:
    text = str(value or "").strip()
    if not text or text in {".", ".."} or Path(text).name != text:
        raise ValueError(f"invalid {label}: {value!r}")
    return text


def _first_bundle(
    workspace: Workspace,
    *,
    uid: str,
    model: str,
    variant: str,
) -> Path:
    candidates = (
        workspace.results_root / uid / model / variant,
        workspace.published_results_root / "experiments" / uid / model / variant,
    )
    for path in candidates:
        if (path / "result.json").is_file():
            return path
    return candidates[0]


def portable_workspace_path(path: Path, workspace: Workspace) -> str:
    """Return a named-root path without leaking the local absolute prefix."""

    target = path.expanduser().resolve(strict=False)
    roots = (
        ("outputs", workspace.outputs_root),
        ("published_results", workspace.published_results_root),
        ("bench", workspace.bench_root),
        ("work", workspace.work_root),
    )
    for label, root in roots:
        try:
            relative = target.relative_to(root.resolve(strict=False))
        except ValueError:
            continue
        return f"{label}/{relative.as_posix()}"
    raise ValueError(f"evaluation source is outside workspace-owned roots: {target}")


def portable_error_message(error: Exception, workspace: Workspace) -> str:
    """Keep useful diagnostics without persisting machine-local root paths."""

    message = str(error)
    roots = (
        ("outputs", workspace.outputs_root),
        ("published_results", workspace.published_results_root),
        ("bench", workspace.bench_root),
        ("work", workspace.work_root),
    )
    for label, root in roots:
        resolved = root.expanduser().resolve(strict=False).as_posix()
        message = message.replace(resolved, label)
    return message


def resolve_evaluation_selection(
    *,
    workspace_path: str | Path,
    scope: str,
    case: str = "",
    candidate_model: str,
    prompt_variant: str,
) -> EvaluationSelection:
    workspace = load_workspace(workspace_path)
    repository = BenchRepository(workspace.bench_root)
    model = _safe_name(candidate_model, label="candidate model")
    if prompt_variant not in {"standard", "enhanced"}:
        raise ValueError("prompt_variant must be standard or enhanced")
    selected_scope = str(scope or "").strip()
    if selected_scope == "one":
        uid = _safe_name(case, label="case UID")
        repository.load_case(uid)
        uids = (uid,)
    elif selected_scope == "all":
        if case:
            raise ValueError("--case is not valid with --scope all")
        manifest = repository.load_collection()
        uids = tuple(str(item["uid"]) for item in manifest["cases"])
    else:
        raise ValueError("scope must be one or all")
    return EvaluationSelection(
        workspace=workspace,
        repository=repository,
        uids=uids,
        candidate_model=model,
        prompt_variant=prompt_variant,
        output_root=(
            workspace.outputs_root
            / "evaluations"
            / model
            / prompt_variant
        ),
    )


def _load_mapping(path: Path, *, label: str) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _error_report(
    *,
    family: str,
    uid: str,
    error: Exception,
    workspace: Workspace,
) -> dict[str, Any]:
    return {
        "format": "dream-exe.evaluation-case",
        "family": family,
        "status": "not_evaluated",
        "uid": uid,
        "error": {
            "type": type(error).__name__,
            "message": portable_error_message(error, workspace),
        },
    }


def evaluate_trajectory_selection(
    selection: EvaluationSelection,
    *,
    protocol: str = DEFAULT_TRAJECTORY_SIMILARITY_PROTOCOL,
) -> dict[str, Any]:
    records_by_group: dict[str, list[dict[str, Any]]] = {
        group: [] for group in TRAJECTORY_SIMILARITY_GROUPS
    }
    case_reports: list[dict[str, Any]] = []
    for uid in selection.uids:
        candidate = selection.candidate_bundle(uid)
        reference = selection.reference_bundle(uid)
        group_reports: dict[str, Any] = {}
        case_status = "completed"
        for group in TRAJECTORY_SIMILARITY_GROUPS:
            relative = _TRAJECTORY_FILES[group]
            try:
                report = evaluate_trajectory_similarity_files(
                    candidate / relative,
                    reference / relative,
                    group=group,
                    protocol=protocol,
                )
                report["uid"] = uid
                report["candidate"] = {
                    "model": selection.candidate_model,
                    "prompt_variant": selection.prompt_variant,
                }
                report["source"] = {
                    "predicted": portable_workspace_path(candidate / relative, selection.workspace),
                    "reference": portable_workspace_path(reference / relative, selection.workspace),
                }
                records_by_group[group].append(report)
            except Exception as error:
                case_status = "not_evaluated"
                report = _error_report(
                    family="trajectory",
                    uid=uid,
                    error=error,
                    workspace=selection.workspace,
                )
                report["group"] = group
            write_json_atomic(
                selection.output_root
                / "cases"
                / uid
                / "trajectory"
                / f"{_TRAJECTORY_SLUGS[group]}.json",
                report,
            )
            group_reports[group] = report
        case_reports.append({"uid": uid, "status": case_status, "groups": group_reports})

    aggregates: dict[str, Any] = {}
    for group in TRAJECTORY_SIMILARITY_GROUPS:
        aggregate = aggregate_trajectory_similarity(
            records_by_group[group],
            protocol=protocol,
            expected_count=len(selection.uids),
            metadata={
                "candidate_model": selection.candidate_model,
                "prompt_variant": selection.prompt_variant,
            },
        )
        write_json_atomic(
            selection.output_root
            / "aggregate"
            / "trajectory"
            / f"{_TRAJECTORY_SLUGS[group]}.json",
            aggregate,
        )
        aggregates[group] = aggregate
    summary = {
        "format": "dream-exe.evaluation-summary",
        "family": "trajectory",
        "status": "completed",
        "candidate": {
            "model": selection.candidate_model,
            "prompt_variant": selection.prompt_variant,
        },
        "expected_cases": len(selection.uids),
        "groups": aggregates,
        "case_status": [
            {"uid": item["uid"], "status": item["status"]} for item in case_reports
        ],
    }
    summary_path = write_json_atomic(
        selection.output_root / "aggregate" / "trajectory" / "summary.json",
        summary,
    )
    return {"aggregate": summary, "aggregate_path": summary_path, "cases": case_reports}


def _executability_case_record(
    payload: Mapping[str, Any],
    *,
    uid: str,
    source: str,
) -> dict[str, Any]:
    metrics = {
        public_name: payload.get(field)
        for public_name, field in _EXECUTABILITY_FIELDS.items()
    }
    return {
        "format": "dream-exe.trajectory-executability-result",
        "family": "executability",
        "status": "completed",
        "uid": uid,
        "source": source,
        "metrics": metrics,
    }


def evaluate_executability_selection(selection: EvaluationSelection) -> dict[str, Any]:
    payloads: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []
    for uid in selection.uids:
        source = selection.candidate_bundle(uid) / "evaluation" / "trajectory_executability.json"
        try:
            payload = _load_mapping(source, label="trajectory executability result")
            if payload.get("format") != "exec_metrics":
                raise ValueError("unsupported trajectory executability result format")
            if str(payload.get("uid", "")) != uid:
                raise ValueError("trajectory executability UID mismatch")
            payloads.append(payload)
            report = _executability_case_record(
                payload,
                uid=uid,
                source=portable_workspace_path(source, selection.workspace),
            )
        except Exception as error:
            report = _error_report(
                family="executability",
                uid=uid,
                error=error,
                workspace=selection.workspace,
            )
        write_json_atomic(
            selection.output_root / "cases" / uid / "executability.json",
            report,
        )
        case_reports.append(report)

    metrics = None
    evaluated_uids: list[str] = []
    if payloads:
        raw_aggregate = aggregate_execution_metrics(
            payloads,
            expected_count=len(payloads),
        )
        metrics = {
            public_name: raw_aggregate["metrics"][field]
            for public_name, field in _EXECUTABILITY_FIELDS.items()
        }
        evaluated_uids = [str(uid) for uid in raw_aggregate["uids"]]
    summary = {
        "format": "dream-exe.evaluation-summary",
        "family": "executability",
        "status": "completed" if payloads else "not_evaluated",
        "candidate": {
            "model": selection.candidate_model,
            "prompt_variant": selection.prompt_variant,
        },
        "expected_cases": len(selection.uids),
        "evaluated_cases": len(payloads),
        "coverage": len(payloads) / len(selection.uids) if selection.uids else 0.0,
        "metrics": metrics,
        "evaluated_uids": evaluated_uids,
        "case_status": [
            {"uid": item["uid"], "status": item["status"]} for item in case_reports
        ],
    }
    summary_path = write_json_atomic(
        selection.output_root / "aggregate" / "executability.json",
        summary,
    )
    return {"aggregate": summary, "aggregate_path": summary_path, "cases": case_reports}


def evaluate_task_selection(selection: EvaluationSelection) -> dict[str, Any]:
    payload_specs: list[dict[str, Any]] = []
    case_reports: list[dict[str, Any]] = []
    for uid in selection.uids:
        source = selection.candidate_bundle(uid) / "evaluation" / "task_success.json"
        try:
            payload = _load_mapping(source, label="task-level execution result")
            case = selection.repository.load_case(uid)
            level = str(case["task"]["level"])
            report = task_success_record_from_payload(
                payload,
                uid=uid,
                level=level,
                metadata={
                    "candidate_model": selection.candidate_model,
                    "prompt_variant": selection.prompt_variant,
                },
            )
            report.update(
                {
                    "format": "dream-exe.task-execution-result",
                    "family": "task",
                    "status": "completed",
                    "source": portable_workspace_path(source, selection.workspace),
                }
            )
            payload_specs.append({"uid": uid, "payload": payload, "level": level})
        except Exception as error:
            report = _error_report(
                family="task",
                uid=uid,
                error=error,
                workspace=selection.workspace,
            )
        write_json_atomic(
            selection.output_root / "cases" / uid / "task.json",
            report,
        )
        case_reports.append(report)

    aggregate = aggregate_task_success_payloads(
        payload_specs,
        by_level=True,
        include_records=True,
        metadata={
            "candidate_model": selection.candidate_model,
            "prompt_variant": selection.prompt_variant,
            "expected_cases": len(selection.uids),
        },
    )
    aggregate["expected_cases"] = len(selection.uids)
    aggregate["evaluated_cases"] = len(payload_specs)
    aggregate["coverage"] = (
        len(payload_specs) / len(selection.uids) if selection.uids else 0.0
    )
    aggregate["case_status"] = [
        {"uid": item["uid"], "status": item["status"]} for item in case_reports
    ]
    aggregate["missing_cases"] = [
        item["uid"] for item in case_reports if item["status"] != "completed"
    ]
    summary_path = write_json_atomic(
        selection.output_root / "aggregate" / "task.json",
        aggregate,
    )
    return {"aggregate": aggregate, "aggregate_path": summary_path, "cases": case_reports}


def selection_summary(selection: EvaluationSelection) -> dict[str, Any]:
    return {
        "candidate": {
            "model": selection.candidate_model,
            "prompt_variant": selection.prompt_variant,
        },
        "case_count": len(selection.uids),
        "cases": list(selection.uids),
        "output_root": portable_workspace_path(selection.output_root, selection.workspace),
    }


__all__ = [
    "EVALUATION_FAMILIES",
    "EvaluationSelection",
    "evaluate_executability_selection",
    "evaluate_task_selection",
    "evaluate_trajectory_selection",
    "portable_error_message",
    "portable_workspace_path",
    "resolve_evaluation_selection",
    "selection_summary",
]
