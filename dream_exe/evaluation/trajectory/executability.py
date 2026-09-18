"""Raw trajectory executability and policy-free path comparison.

The metric implementation remains owned by :mod:`dream_exe.evaluation.execution`.
This module gives those metrics a trajectory-domain surface and exposes raw
path facts needed by optional downstream consumers. Publication-specific
penalties, display projections, and table aggregation are intentionally not
part of the released package.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping
from copy import deepcopy
from pathlib import Path
from typing import Any

from ..execution import (
    build_exec_metrics,
    build_metrics_from_exec_dir,
    compute_executed_smoothness,
    metrics_summary_fields,
    save_exec_metrics,
)

TRAJECTORY_EXECUTABILITY_METRICS = (
    "E-SR",
    "nDTW",
    "Pos95",
    "Rot95",
    "Smth",
)
# Saved artifacts and older callers may still use this name. It is an alias
# for raw metric labels only and carries no table membership or policy.
TABLE3_EXECUTABILITY_METRICS = TRAJECTORY_EXECUTABILITY_METRICS

CURRENT_TRAJECTORY_PATH_COMPARISON_SCHEMA = "dream-exe.trajectory-path-comparison"

_METADATA: dict[str, dict[str, Any]] = {
    "E-SR": {
        "name": "E-SR",
        "direction": "higher_is_better",
        "implementation_field": "exec_sr",
        "stored_unit": "ratio",
    },
    "nDTW": {
        "name": "nDTW",
        "direction": "lower_is_better",
        "implementation_field": "tracking_ndtw",
        "stored_unit": "normalized_pose_distance",
        "case_sensitive_warning": (
            "Executability nDTW is planned-vs-executed TCP disagreement. "
            "It is not trajectory-similarity NDTW."
        ),
    },
    "Pos95": {
        "name": "Pos95",
        "direction": "lower_is_better",
        "implementation_field": "pos_p95",
        "stored_unit": "m",
    },
    "Rot95": {
        "name": "Rot95",
        "direction": "lower_is_better",
        "implementation_field": "rot_p95",
        "stored_unit": "rad",
    },
    "Smth": {
        "name": "Smth",
        "direction": "lower_is_better",
        "implementation_field": "executed_smoothness",
        "stored_unit": "dimensionless_path_smoothness",
    },
}


def trajectory_executability_metadata() -> dict[str, Any]:
    """Describe raw executability metrics without selecting a report table."""

    return {
        "report_type": "trajectory_executability",
        "metrics": deepcopy(_METADATA),
        "case_sensitive_names": {
            "similarity": "NDTW",
            "executability_tracking": "nDTW",
            "same_metric": False,
        },
        "implementation_owner": "dream_exe.evaluation.execution",
    }


def trajectory_planned_path_length_m(
    trajectory: Mapping[str, Any],
) -> float | None:
    """Return planned TCP path length from one explicit trajectory payload."""

    if not isinstance(trajectory, Mapping):
        return None
    entries = trajectory.get("eef_tcp")
    if not isinstance(entries, list) or not entries:
        entries = trajectory.get("eef_controller")
    if not isinstance(entries, list) or not entries:
        return None
    points = [
        point
        for point in (_trajectory_position(item) for item in entries)
        if point is not None
    ]
    if not points:
        return None
    if len(points) < 2:
        return 0.0
    return float(sum(math.dist(left, right) for left, right in zip(points, points[1:])))


def compare_trajectory_path_lengths(
    trajectory_path_length_m: Any,
    reference_path_length_m: Any,
) -> dict[str, Any]:
    """Report path lengths and their raw ratio without applying a policy."""

    trajectory_length = _finite_number(trajectory_path_length_m)
    reference_length = _finite_number(reference_path_length_m)
    ratio = (
        float(trajectory_length / reference_length)
        if trajectory_length is not None
        and reference_length is not None
        and reference_length > 1e-12
        else None
    )
    if trajectory_length is None or reference_length is None:
        status = "not_computable"
        reason = "missing_trajectory_or_reference_path_length"
    elif reference_length <= 1e-12:
        status = "not_computable"
        reason = "reference_path_length_not_positive"
    else:
        status = "computed"
        reason = "ok"
    return {
        "format": CURRENT_TRAJECTORY_PATH_COMPARISON_SCHEMA,
        "report_type": "trajectory_path_comparison",
        "status": status,
        "reason": reason,
        "trajectory_path_length_m": trajectory_length,
        "reference_path_length_m": reference_length,
        "path_length_ratio": ratio,
    }


def evaluate_trajectory_path_comparison(
    trajectory: Mapping[str, Any],
    reference_trajectory: Mapping[str, Any],
) -> dict[str, Any]:
    """Compare two explicitly loaded planned trajectories."""

    report = compare_trajectory_path_lengths(
        trajectory_planned_path_length_m(trajectory),
        trajectory_planned_path_length_m(reference_trajectory),
    )
    report["source"] = {"input_mode": "loaded_payloads"}
    return report


def evaluate_trajectory_path_comparison_paths(
    trajectory_path: str | Path,
    reference_trajectory_path: str | Path,
) -> dict[str, Any]:
    """Compare two explicit trajectory files without bench discovery."""

    generated_path = Path(trajectory_path).expanduser().resolve()
    reference_path = Path(reference_trajectory_path).expanduser().resolve()
    report = evaluate_trajectory_path_comparison(
        _read_mapping_json(generated_path, label="trajectory"),
        _read_mapping_json(reference_path, label="reference trajectory"),
    )
    report["source"] = {
        "input_mode": "explicit_trajectory_files",
        "trajectory_path": generated_path.name,
        "reference_trajectory_path": reference_path.name,
        "artifact_publication_enabled": False,
    }
    return report


def evaluate_trajectory_executability_dir(
    exec_dir: str | Path,
    *,
    run_key: str | None = None,
    gen_model: str | None = None,
    publish: bool = False,
) -> dict[str, Any]:
    """Read raw metrics from one explicit execution directory."""

    source = Path(exec_dir).expanduser().resolve()
    raw_metrics = build_metrics_from_exec_dir(
        source,
        run_key=run_key,
        gen_model=gen_model,
        publish=bool(publish),
    )
    if not raw_metrics:
        return {}
    return {
        "format": "dream-exe.trajectory-executability",
        "report_type": "trajectory_executability",
        "source_exec_dir": source.name,
        "artifact_publication_enabled": bool(publish),
        "raw_exec_metrics": raw_metrics,
    }


def _trajectory_position(item: Any) -> tuple[float, float, float] | None:
    value = item
    if isinstance(value, Mapping):
        value = value.get(
            "pos_world",
            value.get("pos", value.get("position")),
        )
    if value is None:
        return None
    try:
        coordinates = [float(component) for component in list(value)[:3]]
    except (TypeError, ValueError, OverflowError):
        return None
    if len(coordinates) != 3 or not all(
        math.isfinite(component) for component in coordinates
    ):
        return None
    return tuple(coordinates)  # type: ignore[return-value]


def _read_mapping_json(path: Path, *, label: str) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise ValueError(f"{label} JSON must contain an object: {path}")
    return payload


def _finite_number(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


__all__ = [
    "CURRENT_TRAJECTORY_PATH_COMPARISON_SCHEMA",
    "TABLE3_EXECUTABILITY_METRICS",
    "TRAJECTORY_EXECUTABILITY_METRICS",
    "build_exec_metrics",
    "build_metrics_from_exec_dir",
    "compare_trajectory_path_lengths",
    "compute_executed_smoothness",
    "evaluate_trajectory_executability_dir",
    "evaluate_trajectory_path_comparison",
    "evaluate_trajectory_path_comparison_paths",
    "metrics_summary_fields",
    "save_exec_metrics",
    "trajectory_executability_metadata",
    "trajectory_planned_path_length_m",
]
