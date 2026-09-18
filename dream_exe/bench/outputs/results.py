"""Integrity-checked reading and aggregation of immutable experiment results."""

from __future__ import annotations

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...evaluation.execution import aggregate_execution_metrics
from .aggregation import aggregate_technical_run_results
from ..contracts.action import load_action_bundle
from .lifecycle import discover_run_roots, result_relative_path, sha256_file
from ..contracts.schemas import (
    RESOLVED_CONFIG_SCHEMA,
    RESULT_REQUEST_SCHEMA,
    RESULT_SCHEMA,
    RUN_SCHEMA,
    canonical_sha256,
    input_identity_key,
    load_and_validate,
    load_json_strict,
)


def _contained_file(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f"result artifact escapes its bundle: {relative}") from error
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"result artifact is not a regular file: {path}")
    return path


def _input_label(value: Mapping[str, Any]) -> str:
    if value["kind"] == "reference":
        return f"reference/{value['reference_id']}"
    return f"{value['model_id']}/{value['prompt_variant']}"


class ResultRepository:
    """Read case-first results and run metadata below one outputs root."""

    def __init__(self, outputs_root: str | Path) -> None:
        self.root = Path(outputs_root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"benchmark outputs root not found: {self.root}")

    def run_root(self, run_id: str) -> Path:
        try:
            return discover_run_roots(self.root)[run_id]
        except KeyError as error:
            raise FileNotFoundError(f"published run not found: {run_id}") from error

    def iter_bundle_paths(self, run_id: str):
        cases = self.root / "experiments"
        if not cases.is_dir():
            return
        run = load_and_validate(
            self.run_root(run_id) / "run.json",
            expected_schema=RUN_SCHEMA,
        )
        if run["destination"]["run_id"] != run_id:
            raise ValueError("run directory identity mismatch")
        result_paths: set[Path] = set()
        for input_document in run["inputs"]:
            if input_document["kind"] == "generated":
                branch = (
                    input_document["model_id"],
                    input_document["prompt_variant"],
                )
            else:
                branch = ("reference", input_document["reference_id"])
            result_paths.update(
                cases.glob(f"*/{branch[0]}/{branch[1]}/result.json")
            )
        for result_path in sorted(result_paths):
            if result_path.is_symlink():
                raise ValueError(f"result manifest must not be a symlink: {result_path}")
            document = load_and_validate(result_path, expected_schema=RESULT_SCHEMA)
            if document["run_id"] != run_id:
                raise ValueError(
                    "result run identity disagrees with canonical run input: "
                    f"{result_path}"
                )
            yield result_path.parent

    def load_result(
        self,
        bundle: str | Path,
        *,
        verify_artifacts: bool = True,
    ) -> dict[str, Any]:
        root = Path(bundle).expanduser().resolve()
        result = load_and_validate(root / "result.json", expected_schema=RESULT_SCHEMA)
        expected = self.root / "experiments" / result_relative_path(result)
        if root != expected.resolve(strict=False):
            raise ValueError(f"result identity does not match bundle path: {root}")
        if result["resolved_config_sha256"] is not None:
            resolved = load_and_validate(
                root / "resolved_config.json",
                expected_schema=RESOLVED_CONFIG_SCHEMA,
            )
            if canonical_sha256(resolved) != result["resolved_config_sha256"]:
                raise ValueError(f"resolved config digest mismatch: {root}")
            if input_identity_key(resolved["input"]) != input_identity_key(result["input"]):
                raise ValueError(f"resolved config input identity mismatch: {root}")
        elif (root / "resolved_config.json").exists():
            raise ValueError(f"unbound resolved config exists: {root}")
        request = load_and_validate(
            root / "request.json",
            expected_schema=RESULT_REQUEST_SCHEMA,
        )
        if canonical_sha256(request) != result["request_sha256"]:
            raise ValueError(f"request digest mismatch: {root}")
        if request["resolved_config_sha256"] != result["resolved_config_sha256"]:
            raise ValueError(f"request/result config identity conflict: {root}")
        artifact_records = {
            artifact["path"]: artifact for artifact in result["artifacts"]
        }
        if len(artifact_records) != len(result["artifacts"]):
            raise ValueError(f"result contains duplicate artifact paths: {root}")
        if "action" in result["completed_stages"]:
            required_action_paths = {"action/action.npy", "action/meta.json"}
            missing = sorted(required_action_paths - set(artifact_records))
            if missing:
                raise ValueError(
                    f"completed action stage is missing canonical artifacts {missing}: {root}"
                )
            load_action_bundle(
                root / "action",
                expected_uid=result["uid"],
                expected_kind="motion_plan",
            )
        if verify_artifacts:
            for artifact in result["artifacts"]:
                path = _contained_file(root, artifact["path"])
                if path.stat().st_size != artifact["size"]:
                    raise ValueError(f"result artifact size mismatch: {path}")
                if sha256_file(path) != artifact["sha256"]:
                    raise ValueError(f"result artifact digest mismatch: {path}")
        return result

    def aggregate_run(
        self,
        *,
        run_id: str,
        expected_uids: Sequence[str] | None = None,
        require_vlm: bool = False,
        verify_artifacts: bool = True,
    ) -> dict[str, Any]:
        expected = None if expected_uids is None else sorted(set(expected_uids))
        if expected_uids is not None and len(expected) != len(expected_uids):
            raise ValueError("expected_uids contains duplicates")
        denominator_source = "argument" if expected is not None else "unknown"
        if expected is None:
            run = load_and_validate(
                self.run_root(run_id) / "run.json",
                expected_schema=RUN_SCHEMA,
            )
            declared_cases = list(run["selection"]["cases"])
            if declared_cases:
                expected = sorted(declared_cases)
                denominator_source = "run.selection.cases"
        grouped: dict[tuple[str, str, str], list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
        all_results: list[dict[str, Any]] = []
        for bundle in self.iter_bundle_paths(run_id):
            result = self.load_result(bundle, verify_artifacts=verify_artifacts)
            if result["run_id"] != run_id:
                raise ValueError(f"result belongs to another run: {bundle}")
            grouped[input_identity_key(result["input"])].append((bundle, result))
            all_results.append(result)

        inputs: dict[str, Any] = {}
        for _key, records in sorted(grouped.items()):
            identity = records[0][1]["input"]
            label = _input_label(identity)
            observed_uids = [result["uid"] for _bundle, result in records]
            if len(set(observed_uids)) != len(observed_uids):
                raise ValueError(f"duplicate result UID for input {label!r}")
            denominator = sorted(observed_uids) if expected is None else expected
            missing = sorted(set(denominator) - set(observed_uids))
            unexpected = sorted(set(observed_uids) - set(denominator))
            closed = expected is not None and not missing and not unexpected
            workflows: list[Mapping[str, Any]] = []
            workflow_missing: list[str] = []
            execution_payloads: list[Mapping[str, Any]] = []
            execution_missing: list[str] = []
            execution_unsupported: list[str] = []
            for bundle, result in records:
                workflow_path = bundle / "logs" / "workflow.json"
                if not workflow_path.is_file():
                    workflow_missing.append(result["uid"])
                else:
                    workflows.append(load_json_strict(workflow_path))
                execution_path = (
                    bundle
                    / "evaluation"
                    / "trajectory_executability.json"
                )
                if not execution_path.is_file():
                    execution_missing.append(result["uid"])
                    continue
                execution_payload = load_json_strict(execution_path)
                if execution_payload.get("format") != "exec_metrics":
                    execution_unsupported.append(result["uid"])
                    continue
                execution_payloads.append(execution_payload)
            if closed and not workflow_missing:
                source = (
                    {"kind": "ground_truth", "model": ""}
                    if identity["kind"] == "reference"
                    else {"kind": "generated", "model": identity["model_id"]}
                )
                technical = aggregate_technical_run_results(
                    run={"id": f"{run_id}:{label}", "video_source": source},
                    task_results=workflows,
                    task_failure_boundaries=[],
                    expected_uids=denominator,
                    require_vlm=require_vlm,
                )
            else:
                technical = {
                    "status": "unavailable",
                    "reason": "open_denominator" if not closed else "workflow_evidence_missing",
                    "workflow_evidence_missing_uids": sorted(workflow_missing),
                }
            if closed and not execution_missing and not execution_unsupported:
                execution_aggregate = aggregate_execution_metrics(
                    execution_payloads,
                    expected_count=len(denominator),
                )
            else:
                execution_aggregate = {
                    "status": "unavailable",
                    "reason": (
                        "open_denominator"
                        if not closed
                        else "execution_metrics_missing_or_unsupported"
                    ),
                    "missing_uids": sorted(execution_missing),
                    "unsupported_uids": sorted(execution_unsupported),
                }
            inputs[label] = {
                "identity": identity,
                "observed_uids": sorted(observed_uids),
                "expected_uids": denominator,
                "missing_uids": missing,
                "unexpected_uids": unexpected,
                "closed_denominator": closed,
                "denominator_source": denominator_source,
                "status_counts": dict(sorted(Counter(result["status"] for _bundle, result in records).items())),
                "task_outcomes": dict(sorted(Counter(result["task_outcome"] for _bundle, result in records).items())),
                "execution_metrics_aggregate": execution_aggregate,
                "technical_aggregate": technical,
            }
        return {
            "format": "dream-exe.result-aggregate",
            "run_id": run_id,
            "result_count": len(all_results),
            "status_counts": dict(sorted(Counter(result["status"] for result in all_results).items())),
            "task_outcomes": dict(sorted(Counter(result["task_outcome"] for result in all_results).items())),
            "provenance": dict(sorted(Counter(result["provenance_status"] for result in all_results).items())),
            "inputs": inputs,
        }


__all__ = ["ResultRepository"]
