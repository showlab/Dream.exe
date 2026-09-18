"""Deterministic metrics computed from saved execution evidence.

The evaluator is deliberately self-contained: callers provide decoded
evidence directly or point it at one execution artifact directory.  It does
not import the execution runtime, simulator code, or benchmark adapters.
"""

from __future__ import annotations

import csv
import json
import math
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Dict, Optional, Sequence

import numpy as np

from ...artifacts.action_bundle import motion_plan_payload_from_bundle
from ...artifacts.layout import (
    ACTION_ARRAY_FILENAME,
    execution_artifact_paths,
    run_artifact_paths,
    trajectory_artifact_paths,
)

_METRIC_SCOPE = "trajectory_to_execution_only"
_METRIC_VERSION = "exec_metrics"
EXECUTION_METRICS_COHORT_SCHEMA = "dream-exe.execution-metrics-cohort"
_EXECUTION_COHORT_METRICS = (
    "exec_sr",
    "tracking_ndtw",
    "pos_p95",
    "rot_p95",
    "executed_smoothness",
)
_SMOOTHNESS_SAMPLES = 100
_CSV_COLUMNS = (
    "checkpoint_index",
    "frame",
    "success",
    "position_error_norm",
    "position_within_tolerance",
    "orientation_error_norm",
    "orientation_within_tolerance",
    "pose_error_normalized",
    "orientation_control_active",
    "correction_steps",
    "terminated",
    "stage_id",
    "object_id",
)
_DIAGNOSTIC_NOTES = (
    "Exec-SR and checkpoint p95 errors evaluate controller tracking at "
    "planned checkpoints.",
    "Tracking-nDTW evaluates dense planned-vs-executed TCP alignment when "
    "dense_tcp_trace.json is available.",
    "Executed-Smoothness evaluates actual executed TCP path smoothness after "
    "arc-length progress resampling.",
)


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _integer(value: Any, fallback: int | None) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError, OverflowError):
        return fallback


def _vector(value: Any, width: int) -> np.ndarray | None:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError, OverflowError):
        return None
    if array.shape != (width,) or not np.all(np.isfinite(array)):
        return None
    return array


def _fraction(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator) / float(denominator)


def _percentile_95(values: Sequence[float]) -> float:
    ordered = np.sort(np.asarray(list(values), dtype=np.float64))
    location = 0.95 * float(ordered.size - 1)
    lower = int(math.floor(location))
    upper = int(math.ceil(location))
    upper_weight = location - float(lower)
    return float((1.0 - upper_weight) * ordered[lower] + upper_weight * ordered[upper])


def _distribution(values: Sequence[float]) -> Dict[str, Any]:
    ordered = sorted(float(value) for value in values if math.isfinite(float(value)))
    if not ordered:
        return {
            "count": 0,
            "mean": None,
            "median": None,
            "p95": None,
            "max": None,
        }
    return {
        "count": len(ordered),
        "mean": float(sum(ordered) / len(ordered)),
        "median": float(np.median(ordered)),
        "p95": _percentile_95(ordered),
        "max": float(ordered[-1]),
    }


def _metric_record(
    name: str,
    *,
    value: float | None,
    direction: str,
    unit: str,
    details: Mapping[str, Any] | None = None,
    missing: str | None = None,
) -> Dict[str, Any]:
    return {
        "name": name,
        "value": value,
        "status": "computed" if missing is None else "not_computable",
        "direction": direction,
        "unit": unit,
        "missing_inputs": [] if missing is None else [missing],
        "details": dict(details or {}),
    }


def _expanded_metric(record: Mapping[str, Any]) -> Dict[str, Any]:
    expanded = {
        "value": record.get("value"),
        "status": record.get("status"),
        "direction": record.get("direction"),
        "unit": record.get("unit"),
    }
    expanded.update(dict(record.get("details", {}) or {}))
    missing = list(record.get("missing_inputs", []) or [])
    if missing:
        expanded["missing_inputs"] = missing
        expanded["reason"] = missing[0]
    return expanded


def _table_metric(record: Mapping[str, Any]) -> Dict[str, Any]:
    return {
        "name": record.get("name"),
        "value": record.get("value"),
        "direction": record.get("direction"),
        "unit": record.get("unit"),
        "status": record.get("status"),
        "missing_inputs": list(record.get("missing_inputs", []) or []),
        "details": dict(record.get("details", {}) or {}),
    }


def _normalized_quaternion(value: Any) -> np.ndarray | None:
    quaternion = _vector(value, 4)
    if quaternion is None:
        return None
    magnitude = float(np.linalg.norm(quaternion))
    if not math.isfinite(magnitude) or magnitude <= 0.0:
        return None
    return quaternion / magnitude


def _rotvec_quaternion(value: Any) -> np.ndarray | None:
    rotation = _vector(value, 3)
    if rotation is None:
        return None
    angle = float(np.linalg.norm(rotation))
    if angle == 0.0:
        return np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    half_angle = 0.5 * angle
    vector = rotation * (math.sin(half_angle) / angle)
    return np.asarray(
        [math.cos(half_angle), vector[0], vector[1], vector[2]],
        dtype=np.float64,
    )


def _planned_pose(item: Any) -> tuple[np.ndarray, np.ndarray | None] | None:
    if not isinstance(item, Mapping):
        return None
    position = _vector(item.get("pos", item.get("pos_world")), 3)
    quaternion = _normalized_quaternion(item.get("quat_wxyz"))
    if position is None:
        pose = _vector(item.get("eef_target_world_6d"), 6)
        if pose is None:
            return None
        position = pose[:3]
        quaternion = _rotvec_quaternion(pose[3:])
    return position, quaternion


def _executed_pose(item: Any) -> tuple[np.ndarray, np.ndarray | None] | None:
    if not isinstance(item, Mapping):
        return None
    nested = item.get("actual_tcp_world")
    if not isinstance(nested, Mapping):
        return None
    position = _vector(
        nested.get(
            "pos_world",
            nested.get("pos", nested.get("position")),
        ),
        3,
    )
    if position is None:
        return None
    return position, _normalized_quaternion(nested.get("quat_wxyz"))


def _action_checkpoints(action_payload: Any) -> list[Any]:
    if not isinstance(action_payload, Mapping):
        return []
    checkpoints = action_payload.get("checkpoints", [])
    if isinstance(checkpoints, Sequence) and not isinstance(
        checkpoints,
        (str, bytes, bytearray),
    ):
        return list(checkpoints)
    return []


def _tracking_metric(
    *,
    planned_dense_tcp: Sequence[Dict[str, Any]] | None,
    action_payload: Optional[Dict[str, Any]],
    dense_tcp_trace: Sequence[Dict[str, Any]] | None,
    pos_tol: float | None,
    ori_tol: float | None,
) -> Dict[str, Any]:
    planned_source = list(planned_dense_tcp or ())
    if not planned_source:
        planned_source = _action_checkpoints(action_payload)
    if not planned_source:
        return _metric_record(
            "Tracking-nDTW",
            value=None,
            direction="lower_is_better",
            unit="normalized_pose_distance",
            missing="planned_dense_tcp_trajectory",
        )

    planned = [
        pose
        for pose in (_planned_pose(item) for item in planned_source)
        if pose is not None
    ]
    if not planned:
        return _metric_record(
            "Tracking-nDTW",
            value=None,
            direction="lower_is_better",
            unit="normalized_pose_distance",
            missing="planned_dense_pose",
        )

    executed_source = list(dense_tcp_trace or ())
    if not executed_source:
        return _metric_record(
            "Tracking-nDTW",
            value=None,
            direction="lower_is_better",
            unit="normalized_pose_distance",
            missing="dense_tcp_trace",
        )
    executed = [
        pose
        for pose in (_executed_pose(item) for item in executed_source)
        if pose is not None
    ]
    if not executed:
        return _metric_record(
            "Tracking-nDTW",
            value=None,
            direction="lower_is_better",
            unit="normalized_pose_distance",
            missing="executed_dense_pose",
        )
    if pos_tol is None or pos_tol <= 0.0:
        return _metric_record(
            "Tracking-nDTW",
            value=None,
            direction="lower_is_better",
            unit="normalized_pose_distance",
            missing="pos_tol",
        )

    use_orientation = bool(
        ori_tol is not None
        and ori_tol > 0.0
        and all(quaternion is not None for _, quaternion in planned)
        and all(quaternion is not None for _, quaternion in executed)
    )
    plan_count = len(planned)
    exec_count = len(executed)
    accumulated = np.full(
        (plan_count, exec_count),
        np.inf,
        dtype=np.float64,
    )
    path_lengths = np.zeros(
        (plan_count, exec_count),
        dtype=np.int64,
    )

    for plan_index, (plan_position, plan_quaternion) in enumerate(planned):
        for exec_index, (exec_position, exec_quaternion) in enumerate(executed):
            position_component = (
                float(np.linalg.norm(plan_position - exec_position)) / pos_tol
            )
            if use_orientation:
                assert plan_quaternion is not None
                assert exec_quaternion is not None
                normalized_plan = _normalized_quaternion(plan_quaternion)
                normalized_exec = _normalized_quaternion(exec_quaternion)
                assert normalized_plan is not None
                assert normalized_exec is not None
                dot = abs(float(np.dot(normalized_plan, normalized_exec)))
                rotation_error = 2.0 * math.acos(max(-1.0, min(1.0, dot)))
                rotation_component = rotation_error / float(ori_tol)
                local_cost = math.sqrt(position_component**2 + rotation_component**2)
            else:
                local_cost = position_component

            if plan_index == 0 and exec_index == 0:
                accumulated[0, 0] = local_cost
                path_lengths[0, 0] = 1
                continue

            predecessors: list[tuple[float, int]] = []
            if plan_index > 0:
                predecessors.append(
                    (
                        float(accumulated[plan_index - 1, exec_index]),
                        int(path_lengths[plan_index - 1, exec_index]),
                    )
                )
            if exec_index > 0:
                predecessors.append(
                    (
                        float(accumulated[plan_index, exec_index - 1]),
                        int(path_lengths[plan_index, exec_index - 1]),
                    )
                )
            if plan_index > 0 and exec_index > 0:
                predecessors.append(
                    (
                        float(
                            accumulated[
                                plan_index - 1,
                                exec_index - 1,
                            ]
                        ),
                        int(
                            path_lengths[
                                plan_index - 1,
                                exec_index - 1,
                            ]
                        ),
                    )
                )
            prior_cost, prior_length = min(
                predecessors,
                key=lambda candidate: candidate[0],
            )
            accumulated[plan_index, exec_index] = prior_cost + local_cost
            path_lengths[plan_index, exec_index] = prior_length + 1

    path_length = int(path_lengths[-1, -1])
    value = float(accumulated[-1, -1]) / float(path_length)
    mode = "pos_rot_normalized" if use_orientation else "position_only"
    return _metric_record(
        "Tracking-nDTW",
        value=value,
        direction="lower_is_better",
        unit="normalized_pose_distance",
        details={
            "mode": mode,
            "tracking_ndtw_mode": mode,
            "num_plan_points": plan_count,
            "num_exec_points": exec_count,
            "dtw_path_length": path_length,
            "coverage_rate": min(
                1.0,
                float(exec_count) / float(plan_count),
            ),
        },
    )


def _smoothness_sample(item: Any) -> tuple[float, np.ndarray] | None:
    if not isinstance(item, Mapping):
        return None
    timestamp = _finite_number(item.get("timestamp_s"))
    nested = item.get("actual_tcp_world")
    if timestamp is None or not isinstance(nested, Mapping):
        return None
    position = _vector(
        nested.get(
            "pos_world",
            nested.get("pos", nested.get("position")),
        ),
        3,
    )
    if position is None:
        return None
    return timestamp, position


def compute_executed_smoothness(
    dense_exec_steps: Sequence[Dict[str, Any]],
    *,
    resample_points: int = 100,
    normalize_by_path_length: bool = True,
) -> Dict[str, Any]:
    """Measure third-difference roughness on arc-length-resampled TCP motion."""

    source = list(dense_exec_steps or ())
    if not source:
        return _metric_record(
            "Executed-Smoothness",
            value=None,
            direction="lower_is_better",
            unit="dimensionless_path_smoothness",
            missing="dense_tcp_trace",
        )

    candidates = [
        sample
        for sample in (_smoothness_sample(item) for item in source)
        if sample is not None
    ]
    candidates.sort(key=lambda sample: sample[0])
    unique: list[tuple[float, np.ndarray]] = []
    last_timestamp: float | None = None
    for sample in candidates:
        if last_timestamp is not None and abs(sample[0] - last_timestamp) <= 1e-12:
            continue
        unique.append(sample)
        last_timestamp = float(sample[0])
    if len(unique) < 4:
        return _metric_record(
            "Executed-Smoothness",
            value=None,
            direction="lower_is_better",
            unit="dimensionless_path_smoothness",
            missing="at_least_4_unique_timestamped_samples",
        )

    positions = np.asarray(
        [sample[1] for sample in unique],
        dtype=np.float64,
    )
    segment_lengths = np.linalg.norm(
        np.diff(positions, axis=0),
        axis=1,
    )
    progress = np.concatenate(
        (
            np.asarray([0.0], dtype=np.float64),
            np.cumsum(segment_lengths),
        )
    )
    path_length = float(progress[-1])
    if not math.isfinite(path_length) or path_length <= 1e-12:
        return _metric_record(
            "Executed-Smoothness",
            value=None,
            direction="lower_is_better",
            unit="dimensionless_path_smoothness",
            missing="positive_executed_path_length",
        )

    keep = np.concatenate(
        (
            np.asarray([True]),
            np.diff(progress) > 1e-12,
        )
    )
    progress = progress[keep]
    positions = positions[keep]
    if positions.shape[0] < 4:
        return _metric_record(
            "Executed-Smoothness",
            value=None,
            direction="lower_is_better",
            unit="dimensionless_path_smoothness",
            missing="at_least_4_unique_progress_samples",
        )

    output_points = _integer(resample_points, None)
    if output_points is None or output_points < 4:
        return _metric_record(
            "Executed-Smoothness",
            value=None,
            direction="lower_is_better",
            unit="dimensionless_path_smoothness",
            missing="resample_points_at_least_4",
        )

    sample_progress = np.linspace(
        0.0,
        path_length,
        output_points,
        dtype=np.float64,
    )
    resampled = np.empty((output_points, 3), dtype=np.float64)
    for axis in range(3):
        resampled[:, axis] = np.interp(
            sample_progress,
            progress,
            positions[:, axis],
        )
    if normalize_by_path_length:
        resampled = resampled / path_length
    third_difference = (
        resampled[3:] - 3.0 * resampled[2:-1] + 3.0 * resampled[1:-2] - resampled[:-3]
    )
    magnitudes = np.linalg.norm(third_difference, axis=1)
    value = _percentile_95(magnitudes)

    return _metric_record(
        "Executed-Smoothness",
        value=value,
        direction="lower_is_better",
        unit="dimensionless_path_smoothness",
        details={
            "num_samples": len(unique),
            "num_resampled_samples": output_points,
            "num_smoothness_samples": int(magnitudes.size),
            "resample_points": output_points,
            "path_length_m": path_length,
            "path_length_normalized": bool(normalize_by_path_length),
            "finite_difference": "third_difference_without_ds_scaling",
            "smoothing": {
                "enabled": False,
                "method": None,
                "params": None,
            },
        },
    )


def _checkpoint_rows(
    trace: Sequence[Dict[str, Any]] | None,
    *,
    pos_tol: float | None,
    ori_tol: float | None,
) -> list[Dict[str, Any]]:
    rows: list[Dict[str, Any]] = []
    for fallback_index, raw in enumerate(list(trace or ())):
        item = dict(raw) if isinstance(raw, Mapping) else {}
        checkpoint_index = _integer(
            item.get("checkpoint_index"),
            fallback_index,
        )
        if checkpoint_index is None:
            checkpoint_index = fallback_index
        frame = _integer(item.get("frame"), checkpoint_index)
        if frame is None:
            frame = checkpoint_index

        position_error = _finite_number(item.get("position_error_norm"))
        orientation_error = _finite_number(item.get("orientation_error_norm"))
        orientation_active = bool(
            item.get(
                "orientation_control_active",
                orientation_error is not None,
            )
        )
        position_within = (
            bool(position_error <= pos_tol)
            if (position_error is not None and pos_tol is not None and pos_tol > 0.0)
            else None
        )
        orientation_within = (
            bool(orientation_error <= ori_tol)
            if (
                orientation_active
                and orientation_error is not None
                and ori_tol is not None
                and ori_tol > 0.0
            )
            else None
        )

        normalized_pose_error = None
        if position_error is not None and pos_tol is not None and pos_tol > 0.0:
            position_component = position_error / pos_tol
            if (
                orientation_active
                and orientation_error is not None
                and ori_tol is not None
                and ori_tol > 0.0
            ):
                orientation_component = orientation_error / ori_tol
                normalized_pose_error = math.sqrt(
                    position_component * position_component
                    + orientation_component * orientation_component
                )
            else:
                normalized_pose_error = position_component

        correction_steps = _integer(
            item.get("correction_steps"),
            0,
        )
        if correction_steps is None:
            correction_steps = 0
        rows.append(
            {
                "checkpoint_index": checkpoint_index,
                "frame": frame,
                "success": bool(item.get("success", False)),
                "position_error_norm": position_error,
                "position_within_tolerance": position_within,
                "orientation_error_norm": orientation_error,
                "orientation_within_tolerance": orientation_within,
                "pose_error_normalized": normalized_pose_error,
                "orientation_control_active": orientation_active,
                "correction_steps": correction_steps,
                "terminated": bool(item.get("terminated", False)),
                "stage_id": str(item.get("stage_id", "") or ""),
                "object_id": str(item.get("object_id", "") or ""),
            }
        )
    return rows


def _gate_summary(
    close_gate_trace: Sequence[Dict[str, Any]] | None,
    open_gate_trace: Sequence[Dict[str, Any]] | None,
) -> Dict[str, Any]:
    close_items = list(close_gate_trace or ())
    open_items = list(open_gate_trace or ())
    close_completed = sum(
        bool(item.get("completed", False))
        for item in close_items
        if isinstance(item, Mapping)
    )
    open_completed = sum(
        bool(item.get("completed", False))
        for item in open_items
        if isinstance(item, Mapping)
    )
    close_count = len(close_items)
    open_count = len(open_items)
    all_count = close_count + open_count
    return {
        "close_gate_count": close_count,
        "close_gate_completed_count": close_completed,
        "close_gate_failed_count": close_count - close_completed,
        "close_gate_success_rate": _fraction(
            close_completed,
            close_count,
        ),
        "open_gate_count": open_count,
        "open_gate_completed_count": open_completed,
        "open_gate_failed_count": open_count - open_completed,
        "open_gate_success_rate": _fraction(
            open_completed,
            open_count,
        ),
        "all_gate_success_rate": _fraction(
            close_completed + open_completed,
            all_count,
        ),
        "close_completed_before_motion": (
            bool(close_completed == close_count) if close_count else None
        ),
        "open_completed_before_motion": (
            bool(open_completed == open_count) if open_count else None
        ),
    }


def _infer_run_from_exec_dir(exec_dir: Path) -> Dict[str, Any]:
    parts = list(exec_dir.expanduser().resolve().parts)
    if "runs" not in parts:
        return {"run_key": "", "gen_model": None}
    index = parts.index("runs")
    tail = parts[index + 1 :]
    if len(tail) >= 3 and tail[0] == "gt_video":
        return {
            "run_key": f"gt_video/{tail[1]}",
            "gen_model": None,
        }
    if len(tail) >= 3 and tail[0] == "gen":
        return {"run_key": "gen", "gen_model": tail[1]}
    return {"run_key": "", "gen_model": None}


def build_exec_metrics(
    *,
    uid: str,
    output_dir: str,
    executor: str,
    summary: Dict[str, Any],
    checkpoint_trace: Sequence[Dict[str, Any]],
    close_gate_trace: Sequence[Dict[str, Any]] = (),
    open_gate_trace: Sequence[Dict[str, Any]] = (),
    action_path: str = "",
    traj_path: str = "",
    action_payload: Optional[Dict[str, Any]] = None,
    planned_dense_tcp: Sequence[Dict[str, Any]] = (),
    dense_tcp_trace: Sequence[Dict[str, Any]] = (),
) -> Dict[str, Any]:
    """Build the deterministic execution metric document."""

    run_info = _infer_run_from_exec_dir(Path(output_dir))
    pos_tol = _finite_number(summary.get("pos_tol"))
    ori_tol = _finite_number(summary.get("ori_tol"))
    max_correction_steps = _integer(
        summary.get("max_correction_steps"),
        None,
    )
    rows = _checkpoint_rows(
        checkpoint_trace,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
    )
    evaluated = len(rows)
    planned = _integer(
        summary.get("num_checkpoints"),
        evaluated,
    )
    if planned is None:
        planned = evaluated

    successes = sum(bool(row["success"]) for row in rows)
    failures = evaluated - successes
    position_errors = [
        row["position_error_norm"]
        for row in rows
        if row["position_error_norm"] is not None
    ]
    orientation_errors = [
        row["orientation_error_norm"]
        for row in rows
        if row["orientation_error_norm"] is not None
    ]
    pose_errors = [
        row["pose_error_normalized"]
        for row in rows
        if row["pose_error_normalized"] is not None
    ]
    position_flags = [
        bool(row["position_within_tolerance"])
        for row in rows
        if row["position_within_tolerance"] is not None
    ]
    orientation_flags = [
        bool(row["orientation_within_tolerance"])
        for row in rows
        if row["orientation_within_tolerance"] is not None
    ]
    correction_values = [int(row["correction_steps"]) for row in rows]

    gate_metrics = _gate_summary(
        close_gate_trace,
        open_gate_trace,
    )
    terminated_early = bool(
        summary.get("terminated_early", False)
        or any(bool(row["terminated"]) for row in rows)
    )
    evaluated_success_rate = _fraction(successes, evaluated)
    all_success_rate = _fraction(successes, planned)
    coverage_rate = _fraction(evaluated, planned)
    gate_factor = gate_metrics["all_gate_success_rate"]
    executability = all_success_rate
    if executability is not None and gate_factor is not None:
        executability *= gate_factor
    if executability is not None and terminated_early:
        executability *= 0.5

    tracking_record = _tracking_metric(
        planned_dense_tcp=planned_dense_tcp,
        action_payload=action_payload,
        dense_tcp_trace=dense_tcp_trace,
        pos_tol=pos_tol,
        ori_tol=ori_tol,
    )
    smoothness_record = compute_executed_smoothness(
        dense_tcp_trace,
        resample_points=_SMOOTHNESS_SAMPLES,
        normalize_by_path_length=True,
    )
    exec_record = _metric_record(
        "Exec-SR",
        value=all_success_rate,
        direction="higher_is_better",
        unit="ratio",
        details={
            "source_equivalent": "checkpoint_success_rate_all",
            "successful_checkpoints": successes,
            "planned_checkpoints": planned,
            "evaluated_checkpoints": evaluated,
            "pos_tol": pos_tol,
            "rot_tol": ori_tol,
        },
        missing=None if planned > 0 else "planned_checkpoints",
    )
    pos_record = _metric_record(
        "Pos-p95",
        value=(_percentile_95(position_errors) if position_errors else None),
        direction="lower_is_better",
        unit="m",
        details={"num_errors": len(position_errors)},
        missing=(None if position_errors else "evaluated_checkpoint_position_error"),
    )
    rot_record = _metric_record(
        "Rot-p95",
        value=(_percentile_95(orientation_errors) if orientation_errors else None),
        direction="lower_is_better",
        unit="rad",
        details={"num_errors": len(orientation_errors)},
        missing=(
            None if orientation_errors else "evaluated_checkpoint_orientation_error"
        ),
    )
    records = (
        exec_record,
        pos_record,
        rot_record,
        tracking_record,
        smoothness_record,
    )

    correction_exhausted = 0
    if max_correction_steps is not None and max_correction_steps > 0:
        correction_exhausted = sum(
            (not bool(row["success"]))
            and int(row["correction_steps"]) >= max_correction_steps
            for row in rows
        )

    summary_payload = {
        "planned_checkpoints": planned,
        "evaluated_checkpoints": evaluated,
        "unevaluated_checkpoints": max(planned - evaluated, 0),
        "coverage_rate": coverage_rate,
        "checkpoint_successes": successes,
        "checkpoint_failures": failures,
        "checkpoint_success_rate_evaluated": evaluated_success_rate,
        "checkpoint_success_rate_all": all_success_rate,
        "frame_success_rate": all_success_rate,
        "position_within_tolerance_rate": (
            _fraction(sum(position_flags), len(position_flags))
        ),
        "orientation_within_tolerance_rate": (
            _fraction(sum(orientation_flags), len(orientation_flags))
        ),
        "trajectory_executability_score": executability,
        "terminated_early": terminated_early,
    }
    missing_report = [
        {
            "metric": record["name"],
            "missing_inputs": list(record["missing_inputs"]),
        }
        for record in records
        if record["status"] != "computed"
    ]

    return {
        "format": _METRIC_VERSION,
        "metrics_scope": _METRIC_SCOPE,
        "metrics_version": _METRIC_VERSION,
        "uid": str(uid),
        "run_key": run_info["run_key"],
        "gen_model": run_info["gen_model"],
        "output_dir": str(output_dir),
        "executor": str(executor),
        "action_path": str(action_path or ""),
        "traj_path": str(traj_path or ""),
        "tolerances": {
            "pos_tol": pos_tol,
            "ori_tol": ori_tol,
            "max_correction_steps": max_correction_steps,
            "must_reach_min_correction_steps": _integer(
                summary.get("must_reach_min_correction_steps"),
                None,
            ),
            "pose_correction_mode": summary.get("pose_correction_mode"),
            "position_dominate_correction_threshold_m": (
                _finite_number(summary.get("position_dominate_correction_threshold_m"))
            ),
        },
        "config_report": {
            "pos_tol": pos_tol,
            "rot_tol": ori_tol,
            "executed_smoothness_resample_points": _SMOOTHNESS_SAMPLES,
            "executed_smoothness_path_length_normalized": True,
            "dtw_penalty": "none",
            "tracking_ndtw_orientation_fallback": "position_only",
            "smoothing": {
                "enabled": False,
                "method": None,
                "params": None,
            },
        },
        "tracking_ndtw": _expanded_metric(tracking_record),
        "exec_sr": _expanded_metric(exec_record),
        "pos_p95": _expanded_metric(pos_record),
        "rot_p95": _expanded_metric(rot_record),
        "executed_smoothness": _expanded_metric(smoothness_record),
        "metric_table": [_table_metric(record) for record in records],
        "missing_input_report": missing_report,
        "diagnostic_summary": list(_DIAGNOSTIC_NOTES),
        "tracking_ndtw_value": tracking_record["value"],
        "exec_sr_value": exec_record["value"],
        "pos_p95_m": pos_record["value"],
        "rot_p95_rad": rot_record["value"],
        "executed_smoothness_value": smoothness_record["value"],
        "summary": summary_payload,
        "error_stats": {
            "position_error_norm": _distribution(position_errors),
            "orientation_error_norm": _distribution(orientation_errors),
            "pose_error_normalized": _distribution(pose_errors),
        },
        "correction": {
            "total_correction_steps": int(sum(correction_values)),
            "correction_steps": _distribution(correction_values),
            "correction_exhausted_count": int(correction_exhausted),
            "correction_exhausted_rate": _fraction(
                correction_exhausted,
                evaluated,
            ),
        },
        "gripper_gates": gate_metrics,
        "per_frame": rows,
    }


def _cohort_distribution(values: Sequence[float]) -> Dict[str, float | None]:
    if not values:
        return {"mean": None, "median": None, "std": None}
    ordered = sorted(float(value) for value in values)
    mean = float(math.fsum(ordered) / len(ordered))
    variance = math.fsum((value - mean) ** 2 for value in ordered) / len(ordered)
    return {
        "mean": mean,
        "median": float(np.median(ordered)),
        "std": float(math.sqrt(variance)),
    }


def aggregate_execution_metrics(
    payloads: Sequence[Mapping[str, Any]],
    *,
    expected_count: int,
) -> Dict[str, Any]:
    """Aggregate a closed cohort of raw execution metric payloads.

    A value contributes only when its metric record reports ``computed`` and
    the value is finite. Units remain exactly as emitted by
    :func:`build_exec_metrics`; this consumer performs no unit conversion or
    score adjustment.
    """

    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count <= 0
    ):
        raise ValueError("expected_count must be a positive integer")
    rows = list(payloads)
    if len(rows) != expected_count:
        raise ValueError(
            "expected_count must equal payload count for a closed cohort: "
            f"expected {expected_count}, got {len(rows)}"
        )

    identified: list[tuple[str, Mapping[str, Any]]] = []
    observed_uids: set[str] = set()
    for index, payload in enumerate(rows):
        if not isinstance(payload, Mapping):
            raise ValueError(f"execution metric payload {index} must be a mapping")
        if payload.get("format") != _METRIC_VERSION:
            raise ValueError(
                f"execution metric payload {index} has an unsupported schema"
            )
        uid = str(payload.get("uid", "") or "").strip()
        if not uid:
            raise ValueError(f"execution metric payload {index} lacks uid")
        if uid in observed_uids:
            raise ValueError(f"duplicate execution metric uid: {uid}")
        observed_uids.add(uid)
        identified.append((uid, payload))
    identified.sort(key=lambda item: item[0])

    summaries: Dict[str, Any] = {}
    for metric_name in _EXECUTION_COHORT_METRICS:
        units: set[str] = set()
        values: list[float] = []
        computed_uids: list[str] = []
        missing_uids: list[str] = []
        for uid, payload in identified:
            metric = payload.get(metric_name)
            if not isinstance(metric, Mapping):
                raise ValueError(f"{uid} lacks execution metric {metric_name}")
            unit = str(metric.get("unit", "") or "").strip()
            if not unit:
                raise ValueError(f"{uid} execution metric {metric_name} lacks unit")
            units.add(unit)
            status = str(metric.get("status", "") or "").strip()
            if status not in {"computed", "not_computable"}:
                raise ValueError(
                    f"{uid} execution metric {metric_name} has invalid status: "
                    f"{status!r}"
                )
            value = _finite_number(metric.get("value"))
            if status == "computed" and value is not None:
                values.append(value)
                computed_uids.append(uid)
            else:
                missing_uids.append(uid)
        if len(units) != 1:
            raise ValueError(
                f"execution metric {metric_name} has inconsistent units: "
                f"{sorted(units)}"
            )
        summaries[metric_name] = {
            "status": "computed" if values else "not_computable",
            "unit": next(iter(units)),
            "expected_count": expected_count,
            "computed_count": len(values),
            "missing_count": expected_count - len(values),
            "computed_uids": computed_uids,
            "missing_uids": missing_uids,
            **_cohort_distribution(values),
        }

    return {
        "format": EXECUTION_METRICS_COHORT_SCHEMA,
        "report_type": "execution_metrics_cohort",
        "status": "completed",
        "records_total": len(identified),
        "expected_count": expected_count,
        "uids": [uid for uid, _payload in identified],
        "metrics": summaries,
    }


def metrics_summary_fields(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten commonly reported execution metrics."""

    summary = dict(metrics.get("summary", {}) or {})
    errors = dict(metrics.get("error_stats", {}) or {})
    position = dict(errors.get("position_error_norm", {}) or {})
    orientation = dict(errors.get("orientation_error_norm", {}) or {})
    correction = dict(metrics.get("correction", {}) or {})
    gates = dict(metrics.get("gripper_gates", {}) or {})
    tracking = dict(metrics.get("tracking_ndtw", {}) or {})
    exec_sr = dict(metrics.get("exec_sr", {}) or {})
    smoothness = dict(metrics.get("executed_smoothness", {}) or {})
    return {
        "exec_metrics_path": metrics.get("exec_metrics_path"),
        "frame_success_rate": summary.get("frame_success_rate"),
        "checkpoint_success_rate_all": summary.get("checkpoint_success_rate_all"),
        "checkpoint_success_rate_evaluated": summary.get(
            "checkpoint_success_rate_evaluated"
        ),
        "trajectory_executability_score": summary.get("trajectory_executability_score"),
        "coverage_rate": summary.get("coverage_rate"),
        "unevaluated_checkpoints": summary.get("unevaluated_checkpoints"),
        "position_error_mean_m": position.get("mean"),
        "position_error_p95_m": position.get("p95"),
        "position_error_max_m": position.get("max"),
        "orientation_error_mean_rad": orientation.get("mean"),
        "orientation_error_p95_rad": orientation.get("p95"),
        "orientation_error_max_rad": orientation.get("max"),
        "correction_steps_total": correction.get("total_correction_steps"),
        "correction_exhausted_rate": correction.get("correction_exhausted_rate"),
        "all_gate_success_rate": gates.get("all_gate_success_rate"),
        "close_gate_success_rate": gates.get("close_gate_success_rate"),
        "open_gate_success_rate": gates.get("open_gate_success_rate"),
        "tracking_ndtw": tracking.get("value"),
        "exec_sr": exec_sr.get("value"),
        "pos_p95_m": metrics.get("pos_p95_m"),
        "rot_p95_rad": metrics.get("rot_p95_rad"),
        "executed_smoothness": smoothness.get("value"),
    }


def save_exec_metrics(
    metrics: Dict[str, Any],
    output_dir: str,
) -> Dict[str, str]:
    """Write the JSON metric document and fixed-schema per-frame CSV."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    paths = execution_artifact_paths(root)
    json_path = paths["exec_metrics"]
    csv_path = paths["exec_metrics_per_frame"]

    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2)
        handle.write("\n")
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(_CSV_COLUMNS),
        )
        writer.writeheader()
        for row in list(metrics.get("per_frame", []) or []):
            item = dict(row) if isinstance(row, Mapping) else {}
            writer.writerow({column: item.get(column) for column in _CSV_COLUMNS})
    return {
        "exec_metrics_json": json_path.as_posix(),
        "exec_metrics_per_frame_csv": csv_path.as_posix(),
    }


def _read_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_action_payload(path: Path) -> Any:
    """Read either the legacy JSON action or canonical NPY bundle."""

    if path.name == ACTION_ARRAY_FILENAME:
        return motion_plan_payload_from_bundle(path)
    return _read_json(path)


def _evidence_path(
    raw_path: Any,
    fallback: Path,
    *,
    base_dir: Path,
) -> Path:
    text = str(raw_path or "").strip()
    if not text:
        return fallback
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        candidate = base_dir / candidate
    return candidate.resolve()


def _artifact_paths(
    exec_root: Path,
    summary: Mapping[str, Any],
) -> Dict[str, Path]:
    run_root = exec_root.parent
    run_paths = run_artifact_paths(run_root)
    trajectory_paths = trajectory_artifact_paths(run_paths["traj_dir"])
    execution_paths = execution_artifact_paths(exec_root)
    action_fallback = trajectory_paths["action"]
    trajectory_fallback = trajectory_paths["ee_traj"]
    return {
        "checkpoint": _evidence_path(
            summary.get("checkpoint_trace_path"),
            execution_paths["checkpoint_trace"],
            base_dir=exec_root,
        ),
        "dense": _evidence_path(
            summary.get("dense_tcp_trace_path"),
            execution_paths["dense_tcp_trace"],
            base_dir=exec_root,
        ),
        "action": _evidence_path(
            summary.get("action_path"),
            action_fallback,
            base_dir=exec_root,
        ),
        "trajectory": _evidence_path(
            summary.get("traj_path"),
            trajectory_fallback,
            base_dir=exec_root,
        ),
    }


def _cache_is_current(
    cache_path: Path,
    dependencies: Sequence[Path],
) -> bool:
    if not cache_path.is_file():
        return False
    cache_time = cache_path.stat().st_mtime_ns
    return all(
        (not dependency.is_file()) or dependency.stat().st_mtime_ns <= cache_time
        for dependency in dependencies
    )


def _returned_metrics(
    metrics: Mapping[str, Any],
    *,
    json_path: Path,
) -> Dict[str, Any]:
    output = dict(metrics)
    output["exec_metrics_path"] = json_path.as_posix()
    return output


def _cached_metrics_usable(
    payload: Any,
    *,
    trajectory_path: Path,
) -> bool:
    structurally_valid = bool(
        isinstance(payload, Mapping)
        and payload.get("format") == _METRIC_VERSION
        and payload.get("metrics_version") == _METRIC_VERSION
        and isinstance(payload.get("metric_table"), list)
        and isinstance(payload.get("per_frame"), list)
    )
    if not structurally_valid:
        return False
    if not trajectory_path.is_file():
        return True
    cached_path_text = str(payload.get("traj_path", "") or "").strip()
    if not cached_path_text:
        return False
    cached_path = Path(cached_path_text).expanduser()
    if not cached_path.is_absolute():
        return False
    return cached_path.resolve(strict=False) == trajectory_path.resolve(strict=False)


def _sequence_field(payload: Any, key: str) -> list[Dict[str, Any]]:
    if isinstance(payload, Mapping):
        value = payload.get(key, [])
    elif isinstance(payload, list):
        value = payload if key in {"checkpoints", "steps"} else []
    else:
        value = []
    return list(value) if isinstance(value, list) else []


def build_metrics_from_exec_dir(
    exec_dir: Path,
    *,
    run_key: str | None = None,
    gen_model: str | None = None,
    publish: bool = True,
) -> Dict[str, Any]:
    """Load one execution directory, reuse a fresh cache, or recompute it.

    ``publish=True`` preserves the current evaluator contract and writes or
    refreshes ``exec_metrics.json`` plus its per-frame CSV when needed.
    ``publish=False`` is the read-only consumer path: it may reuse a valid
    cache or recompute in memory, but it never creates or rewrites artifacts.
    """

    root = Path(exec_dir).expanduser().resolve()
    execution_paths = execution_artifact_paths(root)
    summary_path = execution_paths["exec_summary"]
    if not summary_path.is_file():
        return {}
    summary_payload = _read_json(summary_path)
    if not isinstance(summary_payload, Mapping):
        return {}
    summary = dict(summary_payload)
    paths = _artifact_paths(root, summary)
    cache_path = execution_paths["exec_metrics"]
    dependencies = [
        summary_path,
        paths["checkpoint"],
        paths["dense"],
        paths["action"],
        paths["trajectory"],
    ]

    if _cache_is_current(cache_path, dependencies):
        cached = _read_json(cache_path)
        if _cached_metrics_usable(
            cached,
            trajectory_path=paths["trajectory"],
        ):
            metrics = dict(cached)
            changed = False
            if run_key is not None:
                requested_run_key = str(run_key)
                if metrics.get("run_key") != requested_run_key:
                    metrics["run_key"] = requested_run_key
                    changed = True
            if gen_model is not None:
                requested_model = str(gen_model)
                if metrics.get("gen_model") != requested_model:
                    metrics["gen_model"] = requested_model
                    changed = True
            if changed and bool(publish):
                save_exec_metrics(metrics, root.as_posix())
            return _returned_metrics(
                metrics,
                json_path=cache_path,
            )

    checkpoint_payload: Any = {}
    if paths["checkpoint"].is_file():
        checkpoint_payload = _read_json(paths["checkpoint"])
    checkpoint_meta = (
        dict(checkpoint_payload.get("meta", {}) or {})
        if isinstance(checkpoint_payload, Mapping)
        else {}
    )
    metric_summary = dict(summary)
    for key, value in checkpoint_meta.items():
        if key not in metric_summary or metric_summary.get(key) is None:
            metric_summary[key] = value

    action_payload: Dict[str, Any] = {}
    if paths["action"].is_file():
        loaded_action = _read_action_payload(paths["action"])
        if isinstance(loaded_action, Mapping):
            action_payload = dict(loaded_action)
    trajectory_payload: Any = {}
    if paths["trajectory"].is_file():
        trajectory_payload = _read_json(paths["trajectory"])
    dense_payload: Any = {}
    if paths["dense"].is_file():
        dense_payload = _read_json(paths["dense"])
    resolved_traj_path = (
        paths["trajectory"].as_posix()
        if paths["trajectory"].is_file()
        else str(summary.get("traj_path", "") or "")
    )

    metrics = build_exec_metrics(
        uid=str(summary.get("uid", "") or ""),
        output_dir=root.as_posix(),
        executor=str(summary.get("executor", "") or ""),
        summary=metric_summary,
        checkpoint_trace=_sequence_field(
            checkpoint_payload,
            "checkpoints",
        ),
        close_gate_trace=_sequence_field(
            checkpoint_payload,
            "close_gates",
        ),
        open_gate_trace=_sequence_field(
            checkpoint_payload,
            "open_gates",
        ),
        action_path=str(summary.get("action_path", "") or ""),
        traj_path=resolved_traj_path,
        action_payload=action_payload,
        planned_dense_tcp=_sequence_field(
            trajectory_payload,
            "eef_tcp",
        ),
        dense_tcp_trace=_sequence_field(dense_payload, "steps"),
    )
    if run_key is not None:
        metrics["run_key"] = str(run_key)
    if gen_model is not None:
        metrics["gen_model"] = str(gen_model)
    if bool(publish):
        written = save_exec_metrics(metrics, root.as_posix())
        metrics.update(written)
        metrics["exec_metrics_path"] = written["exec_metrics_json"]
    return metrics


__all__ = [
    "EXECUTION_METRICS_COHORT_SCHEMA",
    "aggregate_execution_metrics",
    "build_exec_metrics",
    "build_metrics_from_exec_dir",
    "compute_executed_smoothness",
    "metrics_summary_fields",
    "save_exec_metrics",
]
