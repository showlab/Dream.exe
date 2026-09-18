"""Paper-defined 3D trajectory similarity evaluation.

This module owns the Table 2 trajectory metrics only:

* ``HSD``: symmetric Hausdorff shape similarity;
* ``DYN``: Wasserstein-1 speed-distribution similarity;
* ``NDTW``: normalized temporal-alignment similarity.

All three are normalized to ``[0, 1]`` and are higher-is-better.  They are
deliberately distinct from the lower-is-better executable-tracking ``nDTW``
reported by :mod:`dream_exe.evaluation.execution`.

The implementation is path-neutral.  File APIs read exactly the two paths
provided by the caller and never discover a bench root, UID, run key, model,
or instruction condition.
"""

from __future__ import annotations

import json
import math
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.spatial.distance import directed_hausdorff, pdist
from scipy.stats import wasserstein_distance

CURRENT_TRAJECTORY_SIMILARITY_SCHEMA = "dream-exe.trajectory-similarity"

PAPER_SPEC_PROTOCOL = "paper_spec"
PAPER_COMPATIBLE_PROTOCOL = "paper_compatible"
DEFAULT_TRAJECTORY_SIMILARITY_PROTOCOL = PAPER_COMPATIBLE_PROTOCOL
TRAJECTORY_SIMILARITY_PROTOCOLS = (
    PAPER_SPEC_PROTOCOL,
    PAPER_COMPATIBLE_PROTOCOL,
)

GROUP_EEF_VIS = "EEF vis"
GROUP_EEF_TCP = "EEF tcp"
GROUP_OBJ = "OBJ"
TRAJECTORY_SIMILARITY_GROUPS = (
    GROUP_EEF_VIS,
    GROUP_EEF_TCP,
    GROUP_OBJ,
)
TABLE2_TRAJECTORY_GROUPS = TRAJECTORY_SIMILARITY_GROUPS

METRIC_HSD = "HSD"
METRIC_DYN = "DYN"
METRIC_NDTW = "NDTW"
TRAJECTORY_SIMILARITY_METRICS = (
    METRIC_HSD,
    METRIC_DYN,
    METRIC_NDTW,
)
TABLE2_TRAJECTORY_METRICS = TRAJECTORY_SIMILARITY_METRICS

_METRIC_METADATA: dict[str, dict[str, Any]] = {
    METRIC_HSD: {
        "name": METRIC_HSD,
        "paper_role": "trajectory_shape_similarity",
        "raw_definition": "symmetric_hausdorff_distance",
        "raw_unit": "m",
        "value_range": [0.0, 1.0],
        "direction": "higher_is_better",
    },
    METRIC_DYN: {
        "name": METRIC_DYN,
        "paper_role": "trajectory_dynamics_similarity",
        "raw_definition": "wasserstein_1_distance_between_speed_distributions",
        "raw_unit": "m_per_frame_when_fps_is_1",
        "value_range": [0.0, 1.0],
        "direction": "higher_is_better",
    },
    METRIC_NDTW: {
        "name": METRIC_NDTW,
        "paper_role": "trajectory_temporal_alignment_similarity",
        "raw_definition": "dtw_total_cost_divided_by_alignment_path_length",
        "raw_unit": "m",
        "value_range": [0.0, 1.0],
        "direction": "higher_is_better",
        "case_sensitive_warning": (
            "Table 2 NDTW is a higher-is-better prediction-vs-reference "
            "similarity. It is not Table 3 nDTW tracking disagreement."
        ),
    },
}

_GROUP_METADATA: dict[str, dict[str, Any]] = {
    GROUP_EEF_VIS: {
        "name": GROUP_EEF_VIS,
        "trajectory_role": "end_effector_visual_center",
        "artifact_keys": ["eef_visual_center", "visual_center"],
    },
    GROUP_EEF_TCP: {
        "name": GROUP_EEF_TCP,
        "trajectory_role": "end_effector_tool_center_point",
        "artifact_keys": ["eef_tcp", "tcp"],
    },
    GROUP_OBJ: {
        "name": GROUP_OBJ,
        "trajectory_role": "manipulated_object_visual_center",
        "artifact_keys": ["obj_visual_center"],
    },
}

# HSD [m], DYN [m/frame at fps=1], NDTW [m].
_NORMALIZATION_FLOORS: dict[str, dict[str, float]] = {
    GROUP_EEF_VIS: {
        METRIC_HSD: 0.30,
        METRIC_DYN: 0.010,
        METRIC_NDTW: 0.20,
    },
    GROUP_EEF_TCP: {
        METRIC_HSD: 0.30,
        METRIC_DYN: 0.010,
        METRIC_NDTW: 0.20,
    },
    GROUP_OBJ: {
        METRIC_HSD: 0.30,
        METRIC_DYN: 0.012,
        METRIC_NDTW: 0.20,
    },
}

_PROTOCOL_METADATA: dict[str, dict[str, Any]] = {
    PAPER_SPEC_PROTOCOL: {
        "id": PAPER_SPEC_PROTOCOL,
        "purpose": "literal_paper_formula_with_explicit_validity",
        "length_policy": "preserve_each_trajectory_length",
        "dtw_backend": "exact_dynamic_programming",
        "dtw_tie_policy": "lowest_cost_then_shortest_path_then_diagonal",
        "raw_zero_policy": "valid_perfect_similarity_one",
        "nonfinite_raw_policy": "invalid_value_none",
        "invalid_series_policy": "invalid_value_none_and_excluded",
        "per_series_quantization_digits": None,
        "aggregate_quantization_digits": None,
        "aggregation": "arithmetic_mean_of_unrounded_valid_values",
    },
    PAPER_COMPATIBLE_PROTOCOL: {
        "id": PAPER_COMPATIBLE_PROTOCOL,
        "purpose": "reproduce_current_paper_table_behavior",
        "length_policy": "truncate_both_to_shorter_length",
        "dtw_backend": "clean_room_fastdtw_radius_1",
        "dtw_tie_policy": "legacy_predecessor_order",
        "raw_zero_policy": "legacy_zero_and_exclude_from_object_mean",
        "nonfinite_raw_policy": "legacy_zero_when_no_object_is_computable",
        "invalid_series_policy": "legacy_zero_when_no_object_is_computable",
        "per_series_quantization_digits": 3,
        "aggregate_quantization_digits": 3,
        "aggregation": ("arithmetic_mean_after_three_decimal_per_series_quantization"),
        "compatibility_defects": [
            (
                "An exact raw distance of zero is treated as legacy-invalid "
                "and does not receive the mathematically correct score of one."
            ),
            (
                "A group with no computable object receives numeric zero so "
                "that historical episode-level aggregation can include it."
            ),
            (
                "Prediction and reference are truncated to the shorter length "
                "before visibility repair and metric computation."
            ),
        ],
    },
}

_INVALID_SENTINEL = np.array([-1.0, -1.0, -1.0], dtype=np.float64)
_MAX_INVALID_FRACTION = 0.80
_DEFAULT_VISIBILITY_THRESHOLD = 0.1
_SPATIAL_NORMALIZATION_MULTIPLIER = 1.5
_SPEED_NORMALIZATION_MULTIPLIER = 2.0
_PAPER_FASTDTW_RADIUS = 1


def trajectory_similarity_metadata(
    protocol: str = DEFAULT_TRAJECTORY_SIMILARITY_PROTOCOL,
) -> dict[str, Any]:
    """Return the complete reader-facing Table 2 metric contract."""

    protocol_id = _validate_protocol(protocol)
    return {
        "format": CURRENT_TRAJECTORY_SIMILARITY_SCHEMA,
        "protocol": deepcopy(_PROTOCOL_METADATA[protocol_id]),
        "groups": deepcopy(_GROUP_METADATA),
        "metrics": deepcopy(_METRIC_METADATA),
        "normalization": {
            "score_formula": "clip(1 - raw_distance / normalizer, 0, 1)",
            "spatial_characteristic": (
                "max(reference_diameter_m, reference_arc_length_m, 1e-4)"
            ),
            "spatial_multiplier": _SPATIAL_NORMALIZATION_MULTIPLIER,
            "speed_characteristic": "reference_speed_p95",
            "speed_multiplier": _SPEED_NORMALIZATION_MULTIPLIER,
            "floors_by_group": deepcopy(_NORMALIZATION_FLOORS),
        },
        "case_sensitive_names": {
            "similarity": METRIC_NDTW,
            "executability_tracking": "nDTW",
            "same_metric": False,
        },
    }


def evaluate_trajectory_similarity(
    predicted: Sequence[Sequence[float]] | np.ndarray,
    reference: Sequence[Sequence[float]] | np.ndarray,
    *,
    group: str,
    predicted_visibility: Sequence[float] | np.ndarray | None = None,
    reference_visibility: Sequence[float] | np.ndarray | None = None,
    protocol: str = DEFAULT_TRAJECTORY_SIMILARITY_PROTOCOL,
    visibility_threshold: float = _DEFAULT_VISIBILITY_THRESHOLD,
    fps: float = 1.0,
    normalization_overrides: Mapping[str, float] | None = None,
    label: str = "trajectory",
) -> dict[str, Any]:
    """Evaluate one explicit predicted/reference 3D trajectory pair.

    No path or benchmark discovery occurs.  ``group`` must use one of the
    canonical labels in :data:`TRAJECTORY_SIMILARITY_GROUPS`.
    """

    group_name = _validate_group(group)
    protocol_id = _validate_protocol(protocol)
    threshold = _validate_visibility_threshold(visibility_threshold)
    frame_rate = _validate_fps(fps)
    overrides = _validate_normalization_overrides(normalization_overrides)

    series_report = _evaluate_series(
        predicted,
        reference,
        predicted_visibility=predicted_visibility,
        reference_visibility=reference_visibility,
        group=group_name,
        protocol=protocol_id,
        visibility_threshold=threshold,
        fps=frame_rate,
        normalization_overrides=overrides,
        label=str(label),
    )
    return _build_group_report(
        group=group_name,
        protocol=protocol_id,
        series_reports={str(label): series_report},
        source=None,
    )


def evaluate_trajectory_similarity_files(
    predicted_path: str | Path,
    reference_path: str | Path,
    *,
    group: str,
    protocol: str = DEFAULT_TRAJECTORY_SIMILARITY_PROTOCOL,
    visibility_threshold: float = _DEFAULT_VISIBILITY_THRESHOLD,
    fps: float = 1.0,
    normalization_overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Evaluate two explicit trajectory artifact paths.

    Supported payloads are current ``ee_traj.json``, ``obj_trajs.json``, and
    ``union_traj.json`` shapes.  The function never derives either path from a
    UID, run key, model name, or repository location.
    """

    group_name = _validate_group(group)
    protocol_id = _validate_protocol(protocol)
    threshold = _validate_visibility_threshold(visibility_threshold)
    frame_rate = _validate_fps(fps)
    overrides = _validate_normalization_overrides(normalization_overrides)
    pred_path = Path(predicted_path).expanduser()
    ref_path = Path(reference_path).expanduser()

    predicted_payload = _load_json_object(pred_path)
    reference_payload = _load_json_object(ref_path)
    predicted_series = _series_from_payload(predicted_payload, group_name)
    reference_series = _series_from_payload(reference_payload, group_name)
    pairs = _pair_loaded_series(
        predicted_series,
        reference_series,
        group=group_name,
    )

    reports: dict[str, dict[str, Any]] = {}
    for label, predicted_item, reference_item in pairs:
        reports[label] = _evaluate_series(
            predicted_item["positions"],
            reference_item["positions"],
            predicted_visibility=predicted_item.get("visibility"),
            reference_visibility=reference_item.get("visibility"),
            group=group_name,
            protocol=protocol_id,
            visibility_threshold=threshold,
            fps=frame_rate,
            normalization_overrides=overrides,
            label=label,
        )

    return _build_group_report(
        group=group_name,
        protocol=protocol_id,
        series_reports=reports,
        source={
            "predicted_path": pred_path.name,
            "reference_path": ref_path.name,
            "predicted_series_count": len(predicted_series),
            "reference_series_count": len(reference_series),
            "paired_series_count": len(pairs),
        },
    )


def aggregate_trajectory_similarity(
    records: Iterable[Mapping[str, Any]],
    *,
    protocol: str = DEFAULT_TRAJECTORY_SIMILARITY_PROTOCOL,
    expected_count: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate same-group reports with an explicit protocol.

    ``paper_compatible`` reproduces the historical order:

    1. quantize every episode value to three decimals;
    2. take the arithmetic mean;
    3. quantize the aggregate to three decimals.

    The explicit corrected ``paper_spec`` path aggregates unrounded valid
    values and keeps invalid records missing.  The default remains
    ``paper_compatible`` for saved comparison records. Both paths publish
    denominators.
    """

    protocol_id = _validate_protocol(protocol)
    rows = list(records)
    if expected_count is None:
        expected = len(rows)
    else:
        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count < len(rows)
        ):
            raise ValueError("expected_count must be at least records_total")
        expected = expected_count

    for row in rows:
        row_protocol = row.get("protocol") if isinstance(row, Mapping) else None
        if isinstance(row_protocol, Mapping):
            row_protocol_id = row_protocol.get("id")
            if row_protocol_id is not None and row_protocol_id != protocol_id:
                raise ValueError(
                    "record protocol does not match aggregation protocol: "
                    f"{row_protocol_id!r} != {protocol_id!r}"
                )

    groups = {
        str(row.get("group"))
        for row in rows
        if isinstance(row, Mapping) and row.get("group") is not None
    }
    if len(groups) > 1:
        raise ValueError("all trajectory similarity records must share a group")
    group = next(iter(groups), None)
    if group is not None:
        _validate_group(group)

    metric_summaries: dict[str, dict[str, Any]] = {}
    for metric_name in TABLE2_TRAJECTORY_METRICS:
        values: list[float] = []
        invalid_compatibility_count = 0
        for row in rows:
            metric = _record_metric(row, metric_name)
            if metric is None:
                continue
            value = _finite_number_or_none(metric.get("value"))
            if value is None:
                continue
            if protocol_id == PAPER_COMPATIBLE_PROTOCOL:
                value = _quantize(value, 3)
                if metric.get("status") != "computed":
                    invalid_compatibility_count += 1
            elif metric.get("status") != "computed":
                continue
            values.append(value)

        mean_unrounded = float(math.fsum(values) / len(values)) if values else None
        if protocol_id == PAPER_COMPATIBLE_PROTOCOL:
            value = _quantize(mean_unrounded, 3) if mean_unrounded is not None else 0.0
            status = "computed" if values else "legacy_empty_zero"
        else:
            value = mean_unrounded
            status = "computed" if values else "not_computable"

        metric_summaries[metric_name] = {
            **deepcopy(_METRIC_METADATA[metric_name]),
            "value": value,
            "unrounded_value": mean_unrounded,
            "status": status,
            "valid_count": len(values),
            "missing_count": expected - len(values),
            "expected_count": expected,
            "invalid_compatibility_count": invalid_compatibility_count,
        }

    return {
        "format": CURRENT_TRAJECTORY_SIMILARITY_SCHEMA,
        "report_type": "trajectory_similarity_aggregate",
        "protocol": deepcopy(_PROTOCOL_METADATA[protocol_id]),
        "group": group,
        "records_total": len(rows),
        "expected_count": expected,
        "metrics": metric_summaries,
        "metadata": deepcopy(dict(metadata or {})),
    }


def _evaluate_series(
    predicted: Sequence[Sequence[float]] | np.ndarray,
    reference: Sequence[Sequence[float]] | np.ndarray,
    *,
    predicted_visibility: Sequence[float] | np.ndarray | None,
    reference_visibility: Sequence[float] | np.ndarray | None,
    group: str,
    protocol: str,
    visibility_threshold: float,
    fps: float,
    normalization_overrides: Mapping[str, float],
    label: str,
) -> dict[str, Any]:
    predicted_array = _position_array(predicted, name="predicted")
    reference_array = _position_array(reference, name="reference")
    predicted_vis = _visibility_array(
        predicted_visibility,
        len(predicted_array),
        name="predicted_visibility",
    )
    reference_vis = _visibility_array(
        reference_visibility,
        len(reference_array),
        name="reference_visibility",
    )

    original_lengths = {
        "predicted": len(predicted_array),
        "reference": len(reference_array),
    }
    if protocol == PAPER_COMPATIBLE_PROTOCOL:
        common_length = min(len(predicted_array), len(reference_array))
        predicted_array = predicted_array[:common_length]
        reference_array = reference_array[:common_length]
        if predicted_vis is not None:
            predicted_vis = predicted_vis[:common_length]
        if reference_vis is not None:
            reference_vis = reference_vis[:common_length]

    predicted_prepared = _prepare_positions(
        predicted_array,
        visibility=predicted_vis,
        visibility_threshold=visibility_threshold,
    )
    reference_prepared = _prepare_positions(
        reference_array,
        visibility=reference_vis,
        visibility_threshold=visibility_threshold,
    )
    validity = {
        "original_lengths": original_lengths,
        "evaluated_lengths": {
            "predicted": len(predicted_array),
            "reference": len(reference_array),
        },
        "predicted": predicted_prepared["validity"],
        "reference": reference_prepared["validity"],
    }

    invalid_reason = _series_invalid_reason(
        predicted_prepared,
        reference_prepared,
    )
    if (
        protocol == PAPER_COMPATIBLE_PROTOCOL
        and min(len(predicted_array), len(reference_array)) < 3
    ):
        invalid_reason = "legacy_minimum_three_frames"

    if invalid_reason is not None:
        return _invalid_series_report(
            label=label,
            protocol=protocol,
            validity=validity,
            reason=invalid_reason,
        )

    predicted_xyz = predicted_prepared["positions"]
    reference_xyz = reference_prepared["positions"]
    normalization, normalization_evidence = _normalization_factors(
        reference_xyz,
        group=group,
        fps=fps,
        overrides=normalization_overrides,
    )

    raw_metrics: dict[str, float | None] = {
        METRIC_HSD: _symmetric_hausdorff(predicted_xyz, reference_xyz),
        METRIC_DYN: _speed_wasserstein(
            predicted_xyz,
            reference_xyz,
            fps=fps,
        ),
        METRIC_NDTW: _normalized_dtw(
            predicted_xyz,
            reference_xyz,
            protocol=protocol,
        ),
    }
    metrics: dict[str, dict[str, Any]] = {}
    for metric_name in TABLE2_TRAJECTORY_METRICS:
        metrics[metric_name] = _score_metric(
            metric_name,
            raw=raw_metrics[metric_name],
            normalizer=normalization[metric_name],
            protocol=protocol,
        )

    return {
        "label": label,
        "status": (
            "computed"
            if all(item["status"] == "computed" for item in metrics.values())
            else "partially_computable"
        ),
        "metrics": metrics,
        "validity": validity,
        "normalization": normalization_evidence,
    }


def _score_metric(
    metric_name: str,
    *,
    raw: float | None,
    normalizer: float,
    protocol: str,
) -> dict[str, Any]:
    metadata = deepcopy(_METRIC_METADATA[metric_name])
    finite_raw = _finite_number_or_none(raw)
    if finite_raw is None:
        return {
            **metadata,
            "value": (0.0 if protocol == PAPER_COMPATIBLE_PROTOCOL else None),
            "unrounded_value": None,
            "raw": raw,
            "normalizer": normalizer,
            "status": "invalid_nonfinite_raw",
            "included_in_object_mean": False,
        }

    score = float(np.clip(1.0 - finite_raw / normalizer, 0.0, 1.0))
    if protocol == PAPER_COMPATIBLE_PROTOCOL and finite_raw == 0.0:
        return {
            **metadata,
            "value": 0.0,
            "unrounded_value": score,
            "raw": finite_raw,
            "normalizer": normalizer,
            "status": "legacy_zero_raw",
            "included_in_object_mean": False,
        }

    value = _quantize(score, 3) if protocol == PAPER_COMPATIBLE_PROTOCOL else score
    return {
        **metadata,
        "value": value,
        "unrounded_value": score,
        "raw": finite_raw,
        "normalizer": normalizer,
        "status": "computed",
        "included_in_object_mean": True,
    }


def _invalid_series_report(
    *,
    label: str,
    protocol: str,
    validity: Mapping[str, Any],
    reason: str,
) -> dict[str, Any]:
    metrics = {}
    for metric_name in TABLE2_TRAJECTORY_METRICS:
        metrics[metric_name] = {
            **deepcopy(_METRIC_METADATA[metric_name]),
            "value": (0.0 if protocol == PAPER_COMPATIBLE_PROTOCOL else None),
            "unrounded_value": None,
            "raw": None,
            "normalizer": None,
            "status": reason,
            "included_in_object_mean": False,
        }
    return {
        "label": label,
        "status": reason,
        "metrics": metrics,
        "validity": deepcopy(dict(validity)),
        "normalization": None,
    }


def _build_group_report(
    *,
    group: str,
    protocol: str,
    series_reports: Mapping[str, Mapping[str, Any]],
    source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    group_metrics: dict[str, dict[str, Any]] = {}
    for metric_name in TABLE2_TRAJECTORY_METRICS:
        included: list[float] = []
        computed_unrounded: list[float] = []
        for report in series_reports.values():
            metric = _record_metric(report, metric_name)
            if metric is None:
                continue
            unrounded = _finite_number_or_none(metric.get("unrounded_value"))
            if unrounded is not None and metric.get("status") == "computed":
                computed_unrounded.append(unrounded)
            if not metric.get("included_in_object_mean"):
                continue
            value = _finite_number_or_none(metric.get("value"))
            if value is not None:
                included.append(value)

        mean_unrounded = (
            float(math.fsum(computed_unrounded) / len(computed_unrounded))
            if computed_unrounded
            else None
        )
        if included:
            effective_mean = float(math.fsum(included) / len(included))
            value = (
                _quantize(effective_mean, 3)
                if protocol == PAPER_COMPATIBLE_PROTOCOL
                else effective_mean
            )
            status = "computed"
        elif protocol == PAPER_COMPATIBLE_PROTOCOL:
            value = 0.0
            status = "legacy_empty_zero"
        else:
            value = None
            status = "not_computable"

        group_metrics[metric_name] = {
            **deepcopy(_METRIC_METADATA[metric_name]),
            "value": value,
            "unrounded_value": mean_unrounded,
            "status": status,
            "valid_series_count": len(included),
            "invalid_series_count": len(series_reports) - len(included),
            "series_count": len(series_reports),
        }

    return {
        "format": CURRENT_TRAJECTORY_SIMILARITY_SCHEMA,
        "report_type": "trajectory_similarity",
        "protocol": deepcopy(_PROTOCOL_METADATA[protocol]),
        "group": group,
        "group_metadata": deepcopy(_GROUP_METADATA[group]),
        "metrics": group_metrics,
        "series_count": len(series_reports),
        "series": deepcopy(dict(series_reports)),
        "source": deepcopy(dict(source)) if source is not None else None,
    }


def _prepare_positions(
    positions: np.ndarray,
    *,
    visibility: np.ndarray | None,
    visibility_threshold: float,
) -> dict[str, Any]:
    count = len(positions)
    if count == 0:
        return {
            "positions": positions.copy(),
            "validity": {
                "frame_count": 0,
                "valid_frame_count": 0,
                "invalid_frame_count": 0,
                "invalid_fraction": 1.0,
                "status": "empty",
            },
        }

    valid = np.all(np.isfinite(positions), axis=1)
    valid &= ~np.all(positions == _INVALID_SENTINEL, axis=1)
    if visibility is not None:
        valid &= np.isfinite(visibility)
        valid &= visibility >= visibility_threshold

    valid_count = int(np.count_nonzero(valid))
    invalid_count = count - valid_count
    invalid_fraction = invalid_count / count
    if valid_count == 0:
        status = "no_valid_frames"
        repaired = positions.copy()
    elif invalid_fraction > _MAX_INVALID_FRACTION:
        status = "too_many_invalid_frames"
        repaired = positions.copy()
    else:
        status = "computed"
        repaired = positions.astype(np.float64, copy=True)
        frame_indices = np.arange(count, dtype=np.float64)
        valid_indices = frame_indices[valid]
        for axis in range(3):
            repaired[:, axis] = np.interp(
                frame_indices,
                valid_indices,
                repaired[valid, axis],
            )

    return {
        "positions": repaired,
        "validity": {
            "frame_count": count,
            "valid_frame_count": valid_count,
            "invalid_frame_count": invalid_count,
            "invalid_fraction": invalid_fraction,
            "status": status,
        },
    }


def _series_invalid_reason(
    predicted: Mapping[str, Any],
    reference: Mapping[str, Any],
) -> str | None:
    predicted_status = predicted["validity"]["status"]
    reference_status = reference["validity"]["status"]
    if predicted_status != "computed":
        return f"invalid_predicted:{predicted_status}"
    if reference_status != "computed":
        return f"invalid_reference:{reference_status}"
    return None


def _normalization_factors(
    reference: np.ndarray,
    *,
    group: str,
    fps: float,
    overrides: Mapping[str, float],
) -> tuple[dict[str, float], dict[str, Any]]:
    if len(reference) >= 2:
        pairwise = pdist(reference, metric="euclidean")
        diameter = float(np.max(pairwise)) if pairwise.size else 0.0
        steps = np.linalg.norm(np.diff(reference, axis=0), axis=1)
        arc_length = float(np.sum(steps))
        speeds = steps * fps
        speed_p95 = float(np.percentile(speeds, 95))
    else:
        diameter = 0.0
        arc_length = 0.0
        speed_p95 = 0.0

    characteristic = max(diameter, arc_length, 1e-4)
    floors = _NORMALIZATION_FLOORS[group]
    factors = {
        METRIC_HSD: max(
            floors[METRIC_HSD],
            _SPATIAL_NORMALIZATION_MULTIPLIER * characteristic,
        ),
        METRIC_DYN: max(
            floors[METRIC_DYN],
            _SPEED_NORMALIZATION_MULTIPLIER * speed_p95,
        ),
        METRIC_NDTW: max(
            floors[METRIC_NDTW],
            _SPATIAL_NORMALIZATION_MULTIPLIER * characteristic,
        ),
    }
    for metric_name, value in overrides.items():
        factors[metric_name] = value

    evidence = {
        "reference_diameter_m": diameter,
        "reference_arc_length_m": arc_length,
        "reference_spatial_characteristic_m": characteristic,
        "reference_speed_p95": speed_p95,
        "fps": fps,
        "floors": deepcopy(floors),
        "overrides": dict(overrides),
        "factors": dict(factors),
    }
    return factors, evidence


def _symmetric_hausdorff(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> float | None:
    if len(predicted) == 0 or len(reference) == 0:
        return None
    forward = float(directed_hausdorff(predicted, reference)[0])
    reverse = float(directed_hausdorff(reference, predicted)[0])
    return max(forward, reverse)


def _speed_wasserstein(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    fps: float,
) -> float | None:
    if len(predicted) < 2 or len(reference) < 2:
        return None
    predicted_speed = np.linalg.norm(np.diff(predicted, axis=0), axis=1) * fps
    reference_speed = np.linalg.norm(np.diff(reference, axis=0), axis=1) * fps
    if predicted_speed.size == 0 or reference_speed.size == 0:
        return None
    return float(wasserstein_distance(predicted_speed, reference_speed))


def _normalized_dtw(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    protocol: str,
) -> float | None:
    if len(predicted) == 0 or len(reference) == 0:
        return None
    if protocol == PAPER_COMPATIBLE_PROTOCOL:
        total, path = _paper_fastdtw(
            predicted,
            reference,
            radius=_PAPER_FASTDTW_RADIUS,
        )
    else:
        total, path = _exact_dtw(predicted, reference)
    if not path or not math.isfinite(total):
        return None
    return float(total / len(path))


def _point_distance(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right))


def _exact_dtw(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> tuple[float, list[tuple[int, int]]]:
    """Exact DTW with deterministic minimum-cost/shortest-path ties."""

    rows = len(predicted)
    columns = len(reference)
    if rows == 0 or columns == 0:
        return float("inf"), []

    costs = np.full((rows + 1, columns + 1), np.inf, dtype=np.float64)
    lengths = np.full(
        (rows + 1, columns + 1),
        np.iinfo(np.int64).max,
        dtype=np.int64,
    )
    predecessor = np.full((rows, columns), -1, dtype=np.int8)
    costs[0, 0] = 0.0
    lengths[0, 0] = 0

    # Candidate code: 0 diagonal, 1 vertical, 2 horizontal.
    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            candidates = (
                (costs[row - 1, column - 1], lengths[row - 1, column - 1], 0),
                (costs[row - 1, column], lengths[row - 1, column], 1),
                (costs[row, column - 1], lengths[row, column - 1], 2),
            )
            best_cost, best_length, code = min(
                candidates,
                key=lambda item: (item[0], item[1], item[2]),
            )
            if not math.isfinite(float(best_cost)):
                continue
            costs[row, column] = best_cost + _point_distance(
                predicted[row - 1],
                reference[column - 1],
            )
            lengths[row, column] = best_length + 1
            predecessor[row - 1, column - 1] = code

    path = _reconstruct_dense_path(predecessor)
    return float(costs[rows, columns]), path


def _reconstruct_dense_path(
    predecessor: np.ndarray,
) -> list[tuple[int, int]]:
    if predecessor.size == 0:
        return []
    row = predecessor.shape[0] - 1
    column = predecessor.shape[1] - 1
    if predecessor[row, column] < 0:
        return []
    path: list[tuple[int, int]] = []
    while row >= 0 and column >= 0:
        path.append((row, column))
        if row == 0 and column == 0:
            break
        code = int(predecessor[row, column])
        if code == 0:
            row -= 1
            column -= 1
        elif code == 1:
            row -= 1
        elif code == 2:
            column -= 1
        else:
            return []
    path.reverse()
    return path


def _paper_fastdtw(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    radius: int,
) -> tuple[float, list[tuple[int, int]]]:
    """Clean-room implementation of the radius-bounded FastDTW algorithm."""

    minimum_size = radius + 2
    if len(predicted) < minimum_size or len(reference) < minimum_size:
        return _windowed_dtw(predicted, reference, window=None)

    coarse_predicted = _average_adjacent_pairs(predicted)
    coarse_reference = _average_adjacent_pairs(reference)
    _, coarse_path = _paper_fastdtw(
        coarse_predicted,
        coarse_reference,
        radius=radius,
    )
    window = _expand_fastdtw_window(
        coarse_path,
        predicted_length=len(predicted),
        reference_length=len(reference),
        radius=radius,
    )
    return _windowed_dtw(predicted, reference, window=window)


def _average_adjacent_pairs(values: np.ndarray) -> np.ndarray:
    even_length = len(values) - (len(values) % 2)
    return (values[:even_length:2] + values[1:even_length:2]) / 2.0


def _expand_fastdtw_window(
    path: Sequence[tuple[int, int]],
    *,
    predicted_length: int,
    reference_length: int,
    radius: int,
) -> list[tuple[int, int]]:
    neighborhood: set[tuple[int, int]] = set()
    for row, column in path:
        for row_offset in range(-radius, radius + 1):
            for column_offset in range(-radius, radius + 1):
                neighborhood.add((row + row_offset, column + column_offset))

    expanded: set[tuple[int, int]] = set()
    for row, column in neighborhood:
        for row_offset, column_offset in (
            (0, 0),
            (0, 1),
            (1, 0),
            (1, 1),
        ):
            candidate = (
                row * 2 + row_offset,
                column * 2 + column_offset,
            )
            if (
                0 <= candidate[0] < predicted_length
                and 0 <= candidate[1] < reference_length
            ):
                expanded.add(candidate)

    ordered: list[tuple[int, int]] = []
    start_column = 0
    for row in range(predicted_length):
        first_column: int | None = None
        for column in range(start_column, reference_length):
            if (row, column) in expanded:
                ordered.append((row, column))
                if first_column is None:
                    first_column = column
            elif first_column is not None:
                break
        if first_column is not None:
            start_column = first_column
    return ordered


def _windowed_dtw(
    predicted: np.ndarray,
    reference: np.ndarray,
    *,
    window: Sequence[tuple[int, int]] | None,
) -> tuple[float, list[tuple[int, int]]]:
    if len(predicted) == 0 or len(reference) == 0:
        return float("inf"), []
    if window is None:
        cells: Iterable[tuple[int, int]] = (
            (row, column)
            for row in range(len(predicted))
            for column in range(len(reference))
        )
    else:
        cells = window

    # Legacy-compatible predecessor priority is vertical, horizontal,
    # diagonal when cumulative costs tie.
    state: dict[tuple[int, int], tuple[float, tuple[int, int] | None]] = {
        (-1, -1): (0.0, None)
    }
    for row, column in cells:
        candidates = [
            (row - 1, column),
            (row, column - 1),
            (row - 1, column - 1),
        ]
        available = [candidate for candidate in candidates if candidate in state]
        if not available:
            continue
        previous = min(available, key=lambda candidate: state[candidate][0])
        state[(row, column)] = (
            state[previous][0] + _point_distance(predicted[row], reference[column]),
            previous,
        )

    final = (len(predicted) - 1, len(reference) - 1)
    if final not in state:
        return float("inf"), []
    path: list[tuple[int, int]] = []
    cursor: tuple[int, int] | None = final
    while cursor is not None and cursor != (-1, -1):
        path.append(cursor)
        cursor = state[cursor][1]
    path.reverse()
    return float(state[final][0]), path


def _series_from_payload(
    payload: Mapping[str, Any],
    group: str,
) -> dict[str, dict[str, Any]]:
    if group in (GROUP_EEF_VIS, GROUP_EEF_TCP):
        top_level = _top_level_eef_series(payload, group)
        if top_level is not None:
            return {"trajectory": top_level}

    objects = payload.get("objects")
    if not isinstance(objects, Mapping):
        return {}

    key = {
        GROUP_EEF_VIS: "eef_visual_center",
        GROUP_EEF_TCP: "eef_tcp",
        GROUP_OBJ: "obj_visual_center",
    }[group]
    result: dict[str, dict[str, Any]] = {}
    for object_id, raw_object in objects.items():
        if not isinstance(raw_object, Mapping):
            continue
        entries = raw_object.get(key)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        loaded = _dense_entries(entries)
        if loaded is None:
            continue
        loaded["runtime_object_key"] = _clean_optional_text(
            raw_object.get("runtime_object_key")
        )
        result[str(object_id)] = loaded
    return result


def _top_level_eef_series(
    payload: Mapping[str, Any],
    group: str,
) -> dict[str, Any] | None:
    keys = (
        ("visual_center", "eef_visual_center")
        if group == GROUP_EEF_VIS
        else ("eef_tcp", "tcp")
    )
    for key in keys:
        entries = payload.get(key)
        if not isinstance(entries, Sequence) or isinstance(entries, (str, bytes)):
            continue
        loaded = _dense_entries(entries)
        if loaded is not None:
            loaded["artifact_key"] = key
            return loaded
    return None


def _dense_entries(
    entries: Sequence[Any],
) -> dict[str, Any] | None:
    parsed: list[tuple[int, np.ndarray | None, float | None]] = []
    maximum_frame = -1
    for offset, raw_entry in enumerate(entries):
        if isinstance(raw_entry, Mapping):
            raw_frame = raw_entry.get("frame", offset)
            raw_position = raw_entry.get("pos_world")
            raw_visibility = raw_entry.get("vis")
        else:
            raw_frame = offset
            raw_position = raw_entry
            raw_visibility = None
        if isinstance(raw_frame, bool):
            continue
        try:
            frame = int(raw_frame)
        except (TypeError, ValueError):
            continue
        if frame < 0:
            continue
        position: np.ndarray | None
        try:
            candidate = np.asarray(raw_position, dtype=np.float64)
            position = candidate.reshape(-1)[:3]
            if position.shape != (3,):
                position = None
        except (TypeError, ValueError):
            position = None
        visibility = _finite_number_or_none(raw_visibility)
        parsed.append((frame, position, visibility))
        maximum_frame = max(maximum_frame, frame)

    if maximum_frame < 0:
        return None
    positions = np.full((maximum_frame + 1, 3), np.nan, dtype=np.float64)
    visibility = np.ones(maximum_frame + 1, dtype=np.float64)
    for frame, position, visible in parsed:
        if position is not None:
            positions[frame] = position
        if visible is not None:
            visibility[frame] = visible
    return {
        "positions": positions,
        "visibility": visibility,
    }


def _pair_loaded_series(
    predicted: Mapping[str, Mapping[str, Any]],
    reference: Mapping[str, Mapping[str, Any]],
    *,
    group: str,
) -> list[tuple[str, Mapping[str, Any], Mapping[str, Any]]]:
    if "trajectory" in predicted or "trajectory" in reference:
        if "trajectory" in predicted and "trajectory" in reference:
            return [
                (
                    "trajectory",
                    predicted["trajectory"],
                    reference["trajectory"],
                )
            ]
        return []

    if group != GROUP_OBJ:
        return [
            (key, predicted[key], reference[key])
            for key in sorted(set(predicted) & set(reference))
        ]

    pairs: list[tuple[str, Mapping[str, Any], Mapping[str, Any]]] = []
    used_predicted: set[str] = set()
    used_reference: set[str] = set()
    predicted_runtime = _runtime_key_index(predicted)
    reference_runtime = _runtime_key_index(reference)
    for runtime_key in sorted(set(predicted_runtime) & set(reference_runtime)):
        predicted_id = predicted_runtime[runtime_key]
        reference_id = reference_runtime[runtime_key]
        pairs.append(
            (
                runtime_key,
                predicted[predicted_id],
                reference[reference_id],
            )
        )
        used_predicted.add(predicted_id)
        used_reference.add(reference_id)

    for object_id in sorted(set(predicted) & set(reference)):
        if object_id in used_predicted or object_id in used_reference:
            continue
        pairs.append((object_id, predicted[object_id], reference[object_id]))
    return pairs


def _runtime_key_index(
    series: Mapping[str, Mapping[str, Any]],
) -> dict[str, str]:
    index: dict[str, str] = {}
    for object_id, item in series.items():
        runtime_key = _clean_optional_text(item.get("runtime_object_key"))
        if runtime_key is not None:
            index[runtime_key] = object_id
    return index


def _position_array(
    value: Sequence[Sequence[float]] | np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric (N, 3) array") from exc
    if array.ndim != 2 or array.shape[1] != 3:
        raise ValueError(f"{name} must have shape (N, 3)")
    return array


def _visibility_array(
    value: Sequence[float] | np.ndarray | None,
    length: int,
    *,
    name: str,
) -> np.ndarray | None:
    if value is None:
        return None
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a numeric one-dimensional array") from exc
    if array.ndim != 1 or len(array) != length:
        raise ValueError(f"{name} must have shape ({length},)")
    return array


def _load_json_object(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid trajectory JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"trajectory payload must be a JSON object: {path}")
    return payload


def _record_metric(
    record: Mapping[str, Any],
    metric_name: str,
) -> Mapping[str, Any] | None:
    metrics = record.get("metrics")
    if not isinstance(metrics, Mapping):
        return None
    metric = metrics.get(metric_name)
    return metric if isinstance(metric, Mapping) else None


def _validate_protocol(protocol: str) -> str:
    value = str(protocol)
    if value not in TRAJECTORY_SIMILARITY_PROTOCOLS:
        raise ValueError(
            "protocol must be one of " + ", ".join(TRAJECTORY_SIMILARITY_PROTOCOLS)
        )
    return value


def _validate_group(group: str) -> str:
    value = str(group)
    if value not in TABLE2_TRAJECTORY_GROUPS:
        raise ValueError(
            "group must use the exact paper label: "
            + ", ".join(TABLE2_TRAJECTORY_GROUPS)
        )
    return value


def _validate_visibility_threshold(value: float) -> float:
    result = _finite_number_or_none(value)
    if result is None:
        raise ValueError("visibility_threshold must be finite")
    return result


def _validate_fps(value: float) -> float:
    result = _finite_number_or_none(value)
    if result is None or result <= 0:
        raise ValueError("fps must be finite and positive")
    return result


def _validate_normalization_overrides(
    value: Mapping[str, float] | None,
) -> dict[str, float]:
    if value is None:
        return {}
    unknown = set(value) - set(TABLE2_TRAJECTORY_METRICS)
    if unknown:
        raise ValueError(
            "unknown normalization metric(s): " + ", ".join(sorted(unknown))
        )
    result = {}
    for metric_name, raw in value.items():
        number = _finite_number_or_none(raw)
        if number is None or number <= 0:
            raise ValueError(
                f"normalization override for {metric_name} must be positive"
            )
        result[metric_name] = number
    return result


def _finite_number_or_none(value: Any) -> float | None:
    if value is None or isinstance(value, bool):
        return None
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _clean_optional_text(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    return cleaned or None


def _quantize(value: float, digits: int) -> float:
    return float(f"{float(value):.{digits}f}")


__all__ = [
    "CURRENT_TRAJECTORY_SIMILARITY_SCHEMA",
    "DEFAULT_TRAJECTORY_SIMILARITY_PROTOCOL",
    "GROUP_EEF_TCP",
    "GROUP_EEF_VIS",
    "GROUP_OBJ",
    "METRIC_DYN",
    "METRIC_HSD",
    "METRIC_NDTW",
    "PAPER_COMPATIBLE_PROTOCOL",
    "PAPER_SPEC_PROTOCOL",
    "TABLE2_TRAJECTORY_GROUPS",
    "TABLE2_TRAJECTORY_METRICS",
    "TRAJECTORY_SIMILARITY_GROUPS",
    "TRAJECTORY_SIMILARITY_METRICS",
    "TRAJECTORY_SIMILARITY_PROTOCOLS",
    "aggregate_trajectory_similarity",
    "evaluate_trajectory_similarity",
    "evaluate_trajectory_similarity_files",
    "trajectory_similarity_metadata",
]
