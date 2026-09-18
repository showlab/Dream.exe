"""Task-informed gripper event inference for simulator-independent trajectories.

This module deliberately works only with trajectory records and numeric arrays.
It has no dependency on a simulator, benchmark registry, or execution runtime.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import json
from pathlib import Path
import re
from typing import Any, Callable, Dict, List, Literal, Optional, Sequence, Tuple

import numpy as np

from .features import (
    GraspParams,
    Method,
    _build_features_2d,
    _build_features_3d,
    _fill_short_false_gaps,
    _fuse_features,
    _min_run_filter,
    _segments_from_bool,
    _window_median,
    pack_action_trajectory_arrays,
)


def _quiet_base_params() -> GraspParams:
    return replace(GraspParams(), debug=False)


@dataclass
class TaskPriorActionParams:
    base: GraspParams = field(default_factory=_quiet_base_params)
    close_timing_profile: str = "default"
    smooth_radius: int = 2
    close_near_quantile: float = 0.45
    close_slope_quantile: float = 0.6
    close_min_run: int = 2
    close_bridge_gap: int = 1
    max_close_gap_after_approach: int = 10
    close_align_no_comotion_to_object_motion: bool = True
    close_object_motion_threshold_m: float = 0.004
    close_object_motion_min_run: int = 2
    close_object_motion_lead_frames: int = -1
    close_object_motion_max_gap_after_contact_end: int = 4
    close_use_object_motion_handoff: bool = False
    close_use_hold_anchor: bool = False
    plateau_near_quantile: float = 0.25
    plateau_slope_quantile: float = 0.35
    plateau_min_run: int = 4
    plateau_bridge_gap: int = 2
    approach_quantile: float = 0.8
    release_quantile: float = 0.8
    event_min_run: int = 2
    event_bridge_gap: int = 1
    max_plateau_gap_after_approach: int = 8
    max_release_gap_after_plateau: int = 14
    refine_radius: int = 4
    confirm_window: int = 3
    pre_motion_close_backshift: int = 2
    open_window: int = 3
    open_delta_ratio: float = 0.2
    open_lateral_ratio: float = 0.6
    open_lateral_abs_min: float = 0.002
    open_lateral_min_window: int = 3
    open_require_release_segment: bool = True
    open_delay_while_comotion: bool = True
    open_delay_comotion_min_run: int = 3
    open_delay_comotion_reacquire_window: int = 8
    open_delay_use_changepoint: bool = True
    open_changepoint_post_window: int = 3
    open_changepoint_max_search: int = 90
    open_changepoint_post_comotion_window: int = 20
    open_changepoint_min_dist_delta_m: float = 0.01
    open_changepoint_dist_ratio: float = 0.3
    open_changepoint_min_vrel_delta: float = 0.003
    open_changepoint_min_cos_drop: float = 0.2
    open_changepoint_require_detach_evidence: bool = True
    open_allow_pre_release_on_direction_split: bool = True
    open_direction_split_window: int = 3
    open_direction_split_cos_max: float = 0.35
    open_direction_split_min_vrel: float = 0.008
    open_direction_split_min_dist_slope: float = 0.0005
    open_direction_split_min_eef_xy_step: float = 0.002
    open_direction_split_max_comotion_ratio: float = 0.34
    open_direction_split_allow_stationary_object: bool = True
    open_direction_split_max_obj_xy_step: float = 0.0015
    open_direction_split_allow_object_depart: bool = True
    open_direction_split_min_obj_xy_step: float = 0.002
    open_direction_split_max_eef_xy_step_for_object_depart: float = 0.0015
    open_upward_decouple_override: bool = True
    open_upward_decouple_window: int = 3
    open_upward_decouple_min_eef_z_step: float = 0.0015
    open_upward_decouple_object_follow_ratio: float = 0.35
    open_upward_decouple_max_obj_z_step: float = 0.001
    open_upward_decouple_min_vrel: float = 0.006
    open_upward_decouple_cos_max: float = 0.55
    stage_coupling_window: int = 4
    stage_coupling_min_run: int = 3
    stage_coupling_min_delta: float = 0.12
    stage_coupling_min_hold_frames: int = 4
    stage_coupling_refine_radius: int = 8
    stage_coupling_high_threshold: float = 0.58
    stage_coupling_low_threshold: float = 0.46
    debug: bool = True


@dataclass(frozen=True)
class TaskPriorSpec:
    task_name: str
    num_close: int
    num_open: int
    source: str
    key: str


@dataclass
class _DistanceSignals:
    frames: np.ndarray
    dist: np.ndarray
    d1: np.ndarray
    d2: np.ndarray
    eef_xy_step: np.ndarray
    obj_xy_step: np.ndarray
    vrel: np.ndarray
    cos: np.ndarray
    valid: np.ndarray
    obj_vis: np.ndarray
    contact_mask: np.ndarray
    comotion_mask: np.ndarray
    plateau_mask: np.ndarray
    approach_mask: np.ndarray
    release_mask: np.ndarray
    raw_dist: Optional[np.ndarray] = None
    raw_vrel: Optional[np.ndarray] = None
    raw_cos: Optional[np.ndarray] = None
    raw_valid: Optional[np.ndarray] = None
    raw_comotion_mask: Optional[np.ndarray] = None
    eef_z_step: Optional[np.ndarray] = None
    obj_z_step: Optional[np.ndarray] = None


def _canonical_task_token(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(value or "").lower())


def _prior_config_path() -> Path:
    return (
        Path(__file__).resolve().parent.parent
        / "configs"
        / "gripper_task_prior.default.json"
    )


def _read_prior_library(path: Optional[str]) -> Dict[str, Any]:
    source = Path(path).expanduser().resolve() if path else _prior_config_path()
    with source.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    priors = raw.get("task_priors", {})
    if not isinstance(priors, dict):
        raise ValueError("task_prior.json 'task_priors' must be an object")
    return priors


def resolve_task_prior_spec(
    *,
    task_name: Optional[str] = None,
    env_name: Optional[str] = None,
    uid: Optional[str] = None,
    dataset_config_path: Optional[str] = None,
    prior_config_path: Optional[str] = None,
    num_close: Optional[int] = None,
    num_open: Optional[int] = None,
) -> TaskPriorSpec:
    del dataset_config_path
    if num_close is not None or num_open is not None:
        label = next(
            (
                str(value)
                for value in (task_name, env_name, uid)
                if str(value or "").strip()
            ),
            "",
        )
        return TaskPriorSpec(
            task_name=label,
            num_close=max(0, int(1 if num_close is None else num_close)),
            num_open=max(0, int(1 if num_open is None else num_open)),
            source="custom_counts",
            key="custom",
        )

    label = str(task_name or "").strip()
    if not label:
        raise ValueError(
            "task_prior strategy requires gripper.task_prior.task_name "
            "or explicit num_close/num_open."
        )
    needle = _canonical_task_token(label)
    library = _read_prior_library(prior_config_path)
    for key, raw in library.items():
        entry = dict(raw or {})
        aliases = [key, *list(entry.get("aliases", []) or [])]
        tokens = [_canonical_task_token(alias) for alias in aliases]
        if any(token and token in needle for token in tokens):
            return TaskPriorSpec(
                task_name=label,
                num_close=max(0, int(entry.get("num_close", 1) or 0)),
                num_open=max(0, int(entry.get("num_open", 1) or 0)),
                source="task_name",
                key=str(key),
            )
    raise ValueError(
        f"Unknown task_prior.task_name={label!r}. Please add it to "
        "gripper/configs/task_prior.json or provide explicit "
        "num_close/num_open."
    )


def _keep_true_runs(mask: np.ndarray, minimum: int) -> np.ndarray:
    result = np.asarray(mask, dtype=bool).copy()
    required = max(1, int(minimum))
    for start, end in _segments_from_bool(result):
        if end - start + 1 < required:
            result[start : end + 1] = False
    return result


def _finite_quantile(values: np.ndarray, quantile: float, fallback: float) -> float:
    selected = np.asarray(values, dtype=np.float32)
    selected = selected[np.isfinite(selected)]
    if not selected.size:
        return float(fallback)
    return float(np.quantile(selected, float(quantile)))


def _gradient(values: np.ndarray) -> np.ndarray:
    data = np.asarray(values, dtype=np.float32)
    if data.size <= 1:
        return np.zeros_like(data, dtype=np.float32)
    return np.gradient(data).astype(np.float32)


def _finite_gradient(values: np.ndarray) -> np.ndarray:
    """Differentiate only the finite samples while preserving missing entries."""

    data = np.asarray(values, dtype=np.float32)
    result = np.full_like(data, np.nan, dtype=np.float32)
    finite = np.isfinite(data)
    finite_indices = np.flatnonzero(finite)
    if not finite_indices.size:
        return result
    if finite_indices.size == 1:
        result[finite_indices[0]] = 0.0
        return result
    result[finite_indices] = np.gradient(data[finite]).astype(np.float32)
    return result


def _step_norm(points: np.ndarray, dimensions: int) -> np.ndarray:
    data = np.asarray(points, dtype=np.float32)[:, :dimensions]
    result = np.zeros((data.shape[0],), dtype=np.float32)
    if data.shape[0] > 1:
        delta = data[1:] - data[:-1]
        result[1:] = np.sqrt(np.sum(delta * delta, axis=1))
        result[1:][~np.all(np.isfinite(delta), axis=1)] = np.nan
    return result


def _forward_step_norm(points: np.ndarray, dimensions: int) -> np.ndarray:
    """Attach each step to its starting frame, as the current pipeline does."""

    data = np.asarray(points, dtype=np.float32)[:, :dimensions]
    result = np.full((data.shape[0],), np.nan, dtype=np.float32)
    if data.shape[0] <= 1:
        if data.shape[0] == 1:
            result[0] = 0.0
        return result
    deltas = data[1:] - data[:-1]
    finite = np.all(np.isfinite(deltas), axis=1)
    result[:-1][finite] = np.linalg.norm(deltas[finite], axis=1)
    result[-1] = result[-2] if np.isfinite(result[-2]) else 0.0
    return result


def _forward_z_step(points: np.ndarray) -> np.ndarray:
    """Attach signed Z displacement to the frame before the displacement."""

    data = np.asarray(points, dtype=np.float32)
    result = np.full((data.shape[0],), np.nan, dtype=np.float32)
    if data.shape[0] <= 1:
        if data.shape[0] == 1:
            result[0] = 0.0
        return result
    finite = np.isfinite(data[:-1, 2]) & np.isfinite(data[1:, 2])
    result[:-1][finite] = data[1:, 2][finite] - data[:-1, 2][finite]
    result[-1] = result[-2] if np.isfinite(result[-2]) else 0.0
    return result


def _choose_raw_feature(
    primary: Optional[np.ndarray],
    fallback: np.ndarray,
) -> np.ndarray:
    if primary is None:
        return np.asarray(fallback, dtype=np.float32).copy()
    return np.asarray(primary, dtype=np.float32).copy()


def _build_distance_signals(
    ee_traj: Dict[str, Any],
    obj_traj: Dict[str, Any],
    *,
    method: Method,
    params: TaskPriorActionParams,
    ee_key: str,
    obj_key: str,
) -> Tuple[_DistanceSignals, np.ndarray, np.ndarray]:
    (
        frames,
        eef_uv,
        obj_uv,
        eef_world,
        obj_world,
        visibility,
        _visibility_present,
    ) = pack_action_trajectory_arrays(
        ee_traj,
        obj_traj,
        ee_key=ee_key,
        obj_key=obj_key,
    )
    two_d = _build_features_2d(eef_uv, obj_uv, params.base)
    three_d = _build_features_3d(eef_world, obj_world, params.base)
    if method == "2d":
        chosen = two_d
    elif method == "3d":
        chosen = three_d
    elif method == "fused":
        chosen = _fuse_features(three_d, two_d, params.base)
    else:
        raise ValueError(f"Unknown method: {method}")

    radius = max(0, int(params.smooth_radius))
    distance = np.asarray(
        _window_median(chosen.D_win, radius),
        dtype=np.float32,
    )
    relative = np.asarray(
        _window_median(chosen.VREL_win, radius),
        dtype=np.float32,
    )
    cosine = np.asarray(
        _window_median(chosen.COS_win, radius),
        dtype=np.float32,
    )
    raw_distance = distance.copy()
    raw_relative = relative.copy()
    raw_cosine = cosine.copy()
    raw_valid = np.asarray(chosen.valid, dtype=bool).copy()
    valid = np.asarray(chosen.valid, dtype=bool).copy()
    if np.any(np.isfinite(visibility)):
        valid &= visibility >= float(params.base.obj_vis_th)
    distance[~valid] = np.nan
    relative[~valid] = np.nan
    cosine[~valid] = np.nan
    first = _finite_gradient(distance)
    second = _finite_gradient(first)

    finite = valid & np.isfinite(distance) & np.isfinite(first)
    near_threshold = _finite_quantile(
        distance[finite],
        params.close_near_quantile,
        np.inf,
    )
    abs_slope_threshold = _finite_quantile(
        np.abs(first[finite]),
        params.close_slope_quantile,
        np.inf,
    )
    contact = (
        finite & (distance <= near_threshold) & (np.abs(first) <= abs_slope_threshold)
    )
    contact = _fill_short_false_gaps(contact, params.close_bridge_gap)
    contact = _min_run_filter(
        contact,
        min_true=max(1, int(params.close_min_run)),
        min_false=1,
    )

    plateau_near = _finite_quantile(
        distance[finite],
        params.plateau_near_quantile,
        np.inf,
    )
    plateau_slope = _finite_quantile(
        np.abs(first[finite]),
        params.plateau_slope_quantile,
        np.inf,
    )
    plateau_raw = finite & (distance <= plateau_near) & (np.abs(first) <= plateau_slope)
    plateau = _fill_short_false_gaps(
        plateau_raw,
        params.plateau_bridge_gap,
    )
    plateau = _min_run_filter(
        plateau,
        min_true=max(1, int(params.plateau_min_run)),
        min_false=1,
    )

    descending = finite & (first < 0)
    descent_values = -first[descending]
    descent_threshold = _finite_quantile(
        descent_values,
        params.approach_quantile,
        np.inf,
    )
    approach = descending & ((-first) >= descent_threshold)
    approach = _fill_short_false_gaps(
        approach,
        params.event_bridge_gap,
    )
    approach = _min_run_filter(
        approach,
        min_true=max(1, int(params.event_min_run)),
        min_false=1,
    )

    rising = finite & (first > 0)
    rise_values = first[rising]
    rise_threshold = _finite_quantile(
        rise_values,
        params.release_quantile,
        np.inf,
    )
    release = rising & (first >= rise_threshold)
    release = _fill_short_false_gaps(
        release,
        params.event_bridge_gap,
    )
    release = _min_run_filter(
        release,
        min_true=max(1, int(params.event_min_run)),
        min_false=1,
    )

    if method == "2d":
        hold_relative = float(params.base.tau_vrel_hold_2d)
    else:
        hold_relative = float(params.base.tau_vrel_hold_3d)
    comotion = (
        finite
        & (relative <= hold_relative)
        & (cosine >= float(params.base.tau_cos_hold))
    )
    comotion = _fill_short_false_gaps(
        comotion,
        params.event_bridge_gap,
    )
    comotion = _min_run_filter(
        comotion,
        min_true=max(1, int(params.event_min_run)),
        min_false=1,
    )

    raw_comotion = (
        raw_valid
        & np.isfinite(raw_relative)
        & (raw_relative <= hold_relative)
        & np.isfinite(raw_cosine)
        & (raw_cosine >= float(params.base.tau_cos_hold))
    )
    raw_comotion = _fill_short_false_gaps(
        raw_comotion,
        params.event_bridge_gap,
    )
    raw_comotion = _min_run_filter(
        raw_comotion,
        min_true=max(1, int(params.event_min_run)),
        min_false=1,
    )

    return (
        _DistanceSignals(
            frames=frames,
            dist=distance,
            d1=first,
            d2=second,
            eef_xy_step=_forward_step_norm(eef_world, 2),
            obj_xy_step=_forward_step_norm(obj_world, 2),
            vrel=relative,
            cos=cosine,
            valid=valid,
            obj_vis=visibility.astype(np.float32),
            contact_mask=contact,
            comotion_mask=comotion,
            plateau_mask=plateau,
            approach_mask=approach,
            release_mask=release,
            raw_dist=raw_distance,
            raw_vrel=raw_relative,
            raw_cos=raw_cosine,
            raw_valid=raw_valid,
            raw_comotion_mask=raw_comotion,
            eef_z_step=_forward_z_step(eef_world),
            obj_z_step=_forward_z_step(obj_world),
        ),
        eef_world,
        obj_world,
    )


def _first_sustained(mask: np.ndarray, start: int, window: int) -> Optional[int]:
    values = np.asarray(mask, dtype=bool)
    width = max(1, int(window))
    for index in range(max(0, int(start)), max(0, values.size - width + 1)):
        if bool(np.all(values[index : index + width])):
            return int(index)
    return None


def _object_motion_onset(
    obj_world: np.ndarray,
    params: TaskPriorActionParams,
) -> Optional[int]:
    points = np.asarray(obj_world, dtype=np.float64)
    finite_points = np.all(np.isfinite(points), axis=1)
    finite_indices = np.flatnonzero(finite_points)
    if not finite_indices.size:
        return None
    anchor_index = int(finite_indices[0])
    anchor = points[anchor_index]
    displacement = np.full(
        (points.shape[0],),
        np.nan,
        dtype=np.float64,
    )
    displacement[finite_points] = np.linalg.norm(
        points[finite_points] - anchor,
        axis=1,
    )
    moving = finite_points & (
        displacement >= float(params.close_object_motion_threshold_m)
    )
    return _first_sustained(
        moving,
        anchor_index + 1,
        params.close_object_motion_min_run,
    )


def _processed_event_segments(
    mask: np.ndarray,
    params: TaskPriorActionParams,
) -> List[Tuple[int, int]]:
    bridged = _fill_short_false_gaps(mask, params.event_bridge_gap)
    kept = _keep_true_runs(bridged, params.event_min_run)
    return _segments_from_bool(kept)


def _contact_candidates(
    sig: _DistanceSignals,
    params: TaskPriorActionParams,
) -> List[Dict[str, Any]]:
    approaches = _segments_from_bool(sig.approach_mask)
    confirm = max(1, int(params.confirm_window))
    candidates: List[Dict[str, Any]] = []
    for start, end in _segments_from_bool(sig.contact_mask):
        approach: Optional[Tuple[int, int]] = None
        approach_delta = 0.0
        for approach_start, approach_end in approaches:
            if approach_end >= start:
                break
            if start - approach_end > max(1, int(params.max_close_gap_after_approach)):
                continue
            if np.isfinite(sig.dist[approach_start]) and np.isfinite(
                sig.dist[approach_end]
            ):
                delta = float(sig.dist[approach_start] - sig.dist[approach_end])
            else:
                delta = 0.0
            if delta >= approach_delta:
                approach_delta = delta
                approach = (int(approach_start), int(approach_end))

        if approach is None:
            lookback = max(
                0,
                int(start)
                - max(
                    1,
                    int(params.refine_radius) + int(params.confirm_window),
                ),
            )
            if np.isfinite(sig.dist[lookback]) and np.isfinite(sig.dist[start]):
                approach_delta = float(sig.dist[lookback] - sig.dist[start])

        comotion_on = _first_sustained(
            sig.comotion_mask,
            int(start),
            confirm,
        )
        overlap = int(np.count_nonzero(sig.comotion_mask[start : end + 1]))
        comotion_gap = (
            None if comotion_on is None else max(0, int(comotion_on) - int(start))
        )
        after_gap = None if comotion_on is None else max(0, int(comotion_on) - int(end))
        values = np.asarray(sig.dist[start : end + 1], dtype=np.float32)
        finite_values = values[np.isfinite(values)]
        candidate: Dict[str, Any] = {
            "start": int(start),
            "end": int(end),
            "length": int(end - start + 1),
            "mean_dist": (
                float(np.mean(finite_values)) if finite_values.size else float("nan")
            ),
            "approach": approach,
            "approach_delta": float(max(0.0, approach_delta)),
            "comotion_on": (None if comotion_on is None else int(comotion_on)),
            "comotion_gap": comotion_gap,
            "comotion_after_end_gap": after_gap,
            "comotion_overlap": overlap,
        }
        candidates.append(candidate)
    if candidates:
        return candidates
    if np.any(sig.valid):
        valid_distance = np.where(sig.valid, sig.dist, np.inf)
        index = int(np.nanargmin(valid_distance))
        return [
            {
                "start": index,
                "end": index,
                "length": 1,
                "mean_dist": float(sig.dist[index]),
                "approach": None,
                "approach_delta": 0.0,
            }
        ]
    return []


def _mean_finite(values: np.ndarray) -> float:
    finite = np.asarray(values)[np.isfinite(values)]
    return float(np.mean(finite)) if finite.size else float("nan")


def _mean_forward(values: np.ndarray, index: int, width: int) -> float:
    data = np.asarray(values, dtype=np.float32)
    start = max(0, min(int(index), int(data.size - 1)))
    stop = min(data.size, start + max(1, int(width)))
    selected = data[start:stop]
    finite = selected[np.isfinite(selected)]
    return float(np.mean(finite)) if finite.size else float("nan")


_CandidateChoice = Tuple[Dict[str, Any], Tuple[Any, ...]]


def _choose_ranked_candidate(
    candidates: Sequence[Dict[str, Any]],
    evaluate: Callable[[Dict[str, Any]], Optional[_CandidateChoice]],
) -> Optional[Dict[str, Any]]:
    """Return the payload with the lexicographically earliest priority."""

    choices = [
        choice
        for candidate in candidates
        if (choice := evaluate(candidate)) is not None
    ]
    if not choices:
        return None
    return min(choices, key=lambda choice: choice[1])[0]


def _score_precontact_handoff(
    candidate: Dict[str, Any],
    *,
    boundary: int,
    gap_limit: int,
) -> Optional[_CandidateChoice]:
    end = int(candidate.get("end", -1))
    comotion_frame = candidate.get("comotion_on")
    post_contact_gap = candidate.get("comotion_after_end_gap")
    if comotion_frame is None or post_contact_gap is None:
        return None
    gap = int(post_contact_gap)
    if end >= boundary or gap < 0 or gap > gap_limit:
        return None
    priority = (
        float(comotion_frame),
        -end,
        -int(candidate.get("length", 0)),
        -float(candidate.get("approach_delta", 0.0)),
    )
    return candidate, priority


def _score_object_motion_handoff(
    candidate: Dict[str, Any],
    *,
    boundary: int,
    motion_frame: int,
    contact_tail_slack: int,
    comotion_delay_limit: int,
) -> Optional[_CandidateChoice]:
    contact = (
        int(candidate.get("start", -1)),
        int(candidate.get("end", -1)),
    )
    comotion_value = candidate.get("comotion_on")
    if (
        contact[0] < 0
        or contact[1] < contact[0]
        or contact[0] >= boundary
        or comotion_value is None
    ):
        return None
    comotion_frame = int(comotion_value)
    motion_is_local = contact[0] <= motion_frame <= contact[1] + contact_tail_slack
    comotion_delay = comotion_frame - motion_frame
    if (
        not motion_is_local
        or comotion_delay < 0
        or comotion_delay > comotion_delay_limit
    ):
        return None

    result = dict(candidate)
    tail_gap = max(0, motion_frame - contact[1])
    result.update(
        {
            "object_motion_onset_gap_after_contact_end": int(tail_gap),
            "object_motion_to_comotion_gap": int(comotion_delay),
        }
    )
    priority = (
        int(tail_gap),
        abs(contact[0] - motion_frame),
        int(comotion_delay),
        -int(candidate.get("comotion_overlap", 0)),
        -int(candidate.get("length", 0)),
    )
    return result, priority


def _score_hold_anchor(
    candidate: Dict[str, Any],
    *,
    close_start: int,
    earliest_comotion: Optional[int],
) -> Optional[_CandidateChoice]:
    start = int(candidate.get("start", -1))
    end = int(candidate.get("end", -1))
    comotion_value = candidate.get("comotion_on")
    if start <= close_start or end < start or comotion_value is None:
        return None
    comotion_frame = int(comotion_value)
    if earliest_comotion is not None and comotion_frame < earliest_comotion:
        return None
    overlap = int(candidate.get("comotion_overlap", 0) or 0)
    if overlap <= 0:
        return None
    priority = (
        comotion_frame,
        start,
        -overlap,
        -int(candidate.get("length", 0)),
    )
    return dict(candidate), priority


def _find_precontact_handoff_candidate(
    candidates: Sequence[Dict[str, Any]],
    *,
    selected_start: int,
    params: TaskPriorActionParams,
) -> Optional[Dict[str, Any]]:
    boundary = int(selected_start)
    gap_limit = int(max(1, params.confirm_window))
    return _choose_ranked_candidate(
        candidates,
        lambda candidate: _score_precontact_handoff(
            candidate,
            boundary=boundary,
            gap_limit=gap_limit,
        ),
    )


def _find_object_motion_handoff_candidate(
    candidates: Sequence[Dict[str, Any]],
    *,
    selected_start: int,
    object_motion_onset: Optional[int],
    params: TaskPriorActionParams,
) -> Optional[Dict[str, Any]]:
    if object_motion_onset is None:
        return None
    motion_frame = int(object_motion_onset)
    confirmation_width = int(max(1, params.confirm_window))
    contact_tail_slack = int(
        max(0, params.close_object_motion_max_gap_after_contact_end)
    )
    comotion_delay_limit = int(
        max(
            confirmation_width,
            params.max_plateau_gap_after_approach + confirmation_width,
        )
    )
    return _choose_ranked_candidate(
        candidates,
        lambda candidate: _score_object_motion_handoff(
            candidate,
            boundary=int(selected_start),
            motion_frame=motion_frame,
            contact_tail_slack=contact_tail_slack,
            comotion_delay_limit=comotion_delay_limit,
        ),
    )


def _find_hold_anchor_candidate(
    candidates: Sequence[Dict[str, Any]],
    *,
    close_candidate: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if not bool(close_candidate.get("object_motion_handoff_ready", False)):
        return None
    close_start = int(close_candidate.get("start", -1))
    close_comotion_value = close_candidate.get("comotion_on")
    earliest_comotion = (
        None if close_comotion_value is None else int(close_comotion_value)
    )
    return _choose_ranked_candidate(
        candidates,
        lambda candidate: _score_hold_anchor(
            candidate,
            close_start=close_start,
            earliest_comotion=earliest_comotion,
        ),
    )


def _refine_close_frame(
    sig: _DistanceSignals,
    candidate: Dict[str, Any],
    params: TaskPriorActionParams,
) -> int:
    start = int(candidate["start"])
    approach = candidate.get("approach")
    lo = max(
        0,
        start - int(max(1, params.refine_radius + params.confirm_window)),
    )
    if approach is not None:
        lo = max(0, int(approach[1]) - int(max(1, params.refine_radius)))
    hi = min(sig.dist.shape[0] - 1, start)
    confirm = int(max(1, params.confirm_window))
    best_t: Optional[int] = None
    best_curve = -np.inf
    start_dist = float(sig.dist[start]) if np.isfinite(sig.dist[start]) else np.inf
    dist_allow = 1.03 * start_dist if np.isfinite(start_dist) else np.inf
    for index in range(lo, hi + 1):
        post_stop = min(sig.dist.shape[0], index + confirm)
        pre_start = max(0, index - confirm)
        if post_stop <= index:
            continue
        post_contact = np.mean(sig.contact_mask[index:post_stop].astype(np.float32))
        pre_slope = _mean_finite(sig.d1[pre_start:index])
        post_slope = _mean_finite(sig.d1[index:post_stop])
        near_enough = bool(
            np.isfinite(sig.dist[index]) and float(sig.dist[index]) <= dist_allow
        )
        flattening = bool(
            not np.isfinite(post_slope)
            or abs(post_slope) <= max(1e-6, 0.8 * abs(pre_slope))
        )
        if (
            np.isfinite(pre_slope)
            and pre_slope < 0.0
            and near_enough
            and (post_contact >= 0.67 or flattening)
            and flattening
        ):
            curve = float(sig.d2[index]) if np.isfinite(sig.d2[index]) else -np.inf
            if curve >= best_curve:
                best_curve = curve
                best_t = int(index)
    base_close = best_t
    if base_close is None:
        for index in range(lo, hi + 1):
            post_stop = min(sig.dist.shape[0], index + confirm)
            if post_stop <= index:
                continue
            post_contact = np.mean(sig.contact_mask[index:post_stop].astype(np.float32))
            if post_contact >= 0.67:
                base_close = int(index)
                break
    if base_close is None:
        base_close = start

    comotion_on = candidate.get("comotion_on")
    comotion_gap = candidate.get("comotion_gap")
    if approach is None and comotion_on is not None and comotion_gap is not None:
        max_gap = int(
            max(1, params.max_plateau_gap_after_approach + params.confirm_window)
        )
        if 0 < int(comotion_gap) <= max_gap:
            local_mean_d1 = _mean_finite(sig.d1[start : int(comotion_on) + 1])
            if np.isfinite(local_mean_d1) and local_mean_d1 >= 0.0:
                return int(
                    max(start, int(comotion_on) - _pre_motion_close_lead(params))
                )
    return int(base_close)


def _window_mean_d1(
    sig: _DistanceSignals,
    start: int,
    window: int,
) -> float:
    begin = int(max(0, start))
    stop = int(min(sig.d1.shape[0], begin + max(1, window)))
    if stop <= begin:
        return float("nan")
    return _mean_finite(sig.d1[begin:stop])


def _pre_motion_close_lead(params: TaskPriorActionParams) -> int:
    return int(
        max(
            1,
            int(params.pre_motion_close_backshift),
            int(getattr(params.base, "close_backshift", 1)),
        )
    )


def _object_motion_close_lead(params: TaskPriorActionParams) -> int:
    explicit = int(getattr(params, "close_object_motion_lead_frames", -1))
    if explicit >= 0:
        return int(max(0, explicit))
    return _pre_motion_close_lead(params)


def _adjust_close_frame_with_release_context(
    sig: _DistanceSignals,
    *,
    close_t: int,
    candidate: Dict[str, Any],
    params: TaskPriorActionParams,
    object_motion_onset: Optional[int] = None,
) -> Tuple[int, Dict[str, Any]]:
    confirm = int(max(1, params.confirm_window))
    end = int(candidate["end"])
    comotion_on = candidate.get("comotion_on")
    after_gap = candidate.get("comotion_after_end_gap")
    overlap = int(candidate.get("comotion_overlap", 0))
    base_mean = _window_mean_d1(sig, close_t, confirm)
    reason = "base_close"
    selected = int(close_t)
    selected_mean = base_mean
    max_push = int(max(0, min(2, params.pre_motion_close_backshift, end - close_t)))
    comotion_target: Optional[int] = None
    object_target: Optional[int] = None
    lead = _pre_motion_close_lead(params)

    if (
        comotion_on is not None
        and after_gap is not None
        and overlap <= 0
        and 0 <= int(after_gap) <= confirm
    ):
        handoff_close = int(max(selected, end))
        if handoff_close > selected:
            selected = handoff_close
            selected_mean = _window_mean_d1(sig, selected, confirm)
            reason = "align_to_contact_end_before_comotion"

    if np.isfinite(base_mean) and base_mean < -5e-5 and max_push > 0:
        for candidate_t in range(close_t + 1, close_t + max_push + 1):
            candidate_mean = _window_mean_d1(sig, candidate_t, confirm)
            if not np.isfinite(candidate_mean):
                continue
            if candidate_mean >= float(base_mean) + 5e-5:
                selected = int(candidate_t)
                selected_mean = float(candidate_mean)
                reason = "post_close_descent_nudge"
                break

    if comotion_on is not None:
        target = int(max(0, min(sig.dist.shape[0] - 1, int(comotion_on) - lead)))
        max_gap = int(max(confirm, params.max_plateau_gap_after_approach + confirm))
        if target <= end or target - end <= max_gap:
            comotion_target = target
        if comotion_target is not None and comotion_target > selected:
            selected = int(comotion_target)
            selected_mean = _window_mean_d1(sig, selected, confirm)
            reason = "align_to_comotion_lead"

    if (
        bool(params.close_align_no_comotion_to_object_motion)
        and (
            (comotion_on is None and overlap <= 0)
            or bool(params.close_use_object_motion_handoff)
        )
        and object_motion_onset is not None
    ):
        onset = int(object_motion_onset)
        max_gap = int(max(0, params.close_object_motion_max_gap_after_contact_end))
        start = int(candidate["start"])
        target = int(max(start, onset - _object_motion_close_lead(params)))
        target = int(min(end, max(0, target)))
        if (
            onset >= start
            and onset <= end + max_gap
            and (target > selected or bool(params.close_use_object_motion_handoff))
        ):
            object_target = target
            selected = target
            selected_mean = _window_mean_d1(sig, selected, confirm)
            reason = (
                "align_to_object_motion_lead"
                if params.close_use_object_motion_handoff and comotion_on is not None
                else "align_to_object_motion_lead_no_comotion"
            )

    debug: Dict[str, Any] = {
        "strategy": "with_open",
        "base_close": int(close_t),
        "selected_close": int(selected),
        "contact_start": int(candidate["start"]),
        "contact_end": end,
        "comotion_on": (None if comotion_on is None else int(comotion_on)),
        "comotion_after_end_gap": (None if after_gap is None else int(after_gap)),
        "comotion_overlap": overlap,
        "pre_motion_close_lead_frames": lead,
        "comotion_close_target": comotion_target,
        "object_motion_onset": (
            None if object_motion_onset is None else int(object_motion_onset)
        ),
        "object_motion_close_target": object_target,
        "selection_reason": reason,
        "base_close_mean_d1": (
            None if not np.isfinite(base_mean) else float(base_mean)
        ),
        "selected_close_mean_d1": (
            None if not np.isfinite(selected_mean) else float(selected_mean)
        ),
        "close_shift_frames": int(selected - close_t),
    }
    return int(selected), debug


def _find_release_candidate(
    sig: _DistanceSignals,
    *,
    start_after: int,
    contact_end: Optional[int] = None,
    next_close: Optional[int],
    params: TaskPriorActionParams,
) -> Optional[Tuple[int, int]]:
    releases = _segments_from_bool(sig.release_mask)

    def positive_rise(start: int, end: int) -> float:
        if end <= start:
            return 0.0
        delta = np.diff(sig.dist[start : end + 1])
        finite = delta[np.isfinite(delta)]
        if not finite.size:
            return 0.0
        return float(np.sum(np.maximum(finite, 0.0)))

    if contact_end is not None:
        local_floor = max(int(start_after), int(contact_end))
        local: List[Tuple[int, int]] = []
        for release_start, release_end in releases:
            if release_end <= local_floor:
                continue
            if next_close is not None and release_start >= int(next_close):
                continue
            selected_start = max(int(release_start), local_floor + 1)
            if selected_start > release_end:
                continue
            if selected_start - int(contact_end) > max(
                0, int(params.max_release_gap_after_plateau)
            ):
                continue
            if positive_rise(selected_start, int(release_end)) <= 0.0:
                continue
            local.append((selected_start, int(release_end)))
        if local:
            return min(local, key=lambda item: item[0])

    best: Optional[Tuple[int, int]] = None
    best_rise = 0.0
    for release_start, release_end in releases:
        if release_start <= int(start_after):
            continue
        if next_close is not None and release_start >= int(next_close):
            continue
        rise = positive_rise(int(release_start), int(release_end))
        if rise > best_rise:
            best = (int(release_start), int(release_end))
            best_rise = rise
    if best is not None:
        return best

    future = [
        (int(start), int(end))
        for start, end in releases
        if start > int(start_after) and (next_close is None or start < int(next_close))
    ]
    if not future:
        return None
    return max(future, key=lambda item: positive_rise(*item))


def _direction_split(
    sig: _DistanceSignals,
    *,
    contact_end: int,
    release_start: int,
    params: TaskPriorActionParams,
) -> Tuple[Optional[int], Dict[str, Any]]:
    if not bool(params.open_allow_pre_release_on_direction_split):
        return None, {"enabled": False}
    width = int(max(1, params.open_direction_split_window))
    begin = int(max(0, contact_end + 1))
    finish = int(min(sig.dist.shape[0] - width, release_start - 1))
    debug: Dict[str, Any] = {
        "enabled": bool(params.open_allow_pre_release_on_direction_split),
        "window": width,
        "search_start": begin,
        "search_end": int(max(begin - 1, finish)),
        "cos_max": float(params.open_direction_split_cos_max),
        "min_vrel": float(params.open_direction_split_min_vrel),
        "min_dist_slope": float(params.open_direction_split_min_dist_slope),
        "min_eef_xy_step": float(params.open_direction_split_min_eef_xy_step),
        "max_comotion_ratio": float(params.open_direction_split_max_comotion_ratio),
        "allow_stationary_object": bool(
            params.open_direction_split_allow_stationary_object
        ),
        "max_obj_xy_step": float(params.open_direction_split_max_obj_xy_step),
        "allow_object_depart": bool(params.open_direction_split_allow_object_depart),
        "min_obj_xy_step": float(params.open_direction_split_min_obj_xy_step),
        "max_eef_xy_step_for_object_depart": float(
            params.open_direction_split_max_eef_xy_step_for_object_depart
        ),
        "selected": None,
    }
    if finish < begin:
        return None, debug
    for index in range(begin, finish + 1):
        stop = index + width
        cosine = sig.cos[index:stop]
        relative = sig.vrel[index:stop]
        slope = sig.d1[index:stop]
        eef_step = sig.eef_xy_step[index:stop]
        obj_step = sig.obj_xy_step[index:stop]
        comotion = sig.comotion_mask[index:stop]
        if (
            cosine.size < width
            or relative.size < width
            or slope.size < width
            or eef_step.size < width
            or obj_step.size < width
        ):
            continue
        if not (np.all(np.isfinite(slope)) and np.all(np.isfinite(relative))):
            continue
        finite_cos = bool(np.all(np.isfinite(cosine)))
        stationary_object = bool(
            params.open_direction_split_allow_stationary_object
            and np.all(np.isfinite(obj_step))
            and float(np.max(obj_step))
            <= float(params.open_direction_split_max_obj_xy_step)
        )
        object_depart = bool(
            params.open_direction_split_allow_object_depart
            and np.all(np.isfinite(obj_step))
            and np.all(np.isfinite(eef_step))
            and float(np.median(obj_step))
            >= float(params.open_direction_split_min_obj_xy_step)
            and float(np.max(eef_step))
            <= float(params.open_direction_split_max_eef_xy_step_for_object_depart)
        )
        direction_ok = bool(
            finite_cos
            and float(np.max(cosine)) <= float(params.open_direction_split_cos_max)
        )
        relative_median = float(np.median(relative))
        comotion_ratio = float(np.mean(comotion))
        if relative_median < float(params.open_direction_split_min_vrel):
            continue
        if comotion_ratio > float(params.open_direction_split_max_comotion_ratio):
            continue
        mode: Optional[str] = None
        if direction_ok or stationary_object:
            slope_median = float(np.median(slope))
            if slope_median < float(params.open_direction_split_min_dist_slope):
                continue
            min_eef_step = float(max(0.0, params.open_direction_split_min_eef_xy_step))
            if min_eef_step > 0.0:
                if not np.all(np.isfinite(eef_step)):
                    continue
                if float(np.min(eef_step)) < min_eef_step:
                    continue
            mode = "cosine" if direction_ok else "stationary_object_depart"
        elif object_depart:
            mode = "object_depart"
        else:
            continue
        debug.update(
            {
                "selected": int(index),
                "selected_direction_mode": mode,
                "selected_cos_max": (float(np.max(cosine)) if finite_cos else None),
                "selected_vrel_median": relative_median,
                "selected_dist_slope_median": float(np.median(slope)),
                "selected_eef_xy_step_max": (
                    float(np.max(eef_step)) if np.all(np.isfinite(eef_step)) else None
                ),
                "selected_eef_xy_step_min": (
                    float(np.min(eef_step)) if np.all(np.isfinite(eef_step)) else None
                ),
                "selected_obj_xy_step_median": (
                    float(np.median(obj_step))
                    if np.all(np.isfinite(obj_step))
                    else None
                ),
                "selected_obj_xy_step_max": (
                    float(np.max(obj_step)) if np.all(np.isfinite(obj_step)) else None
                ),
                "selected_comotion_ratio": comotion_ratio,
            }
        )
        return int(index), debug
    return None, debug


def _upward_decouple(
    sig: _DistanceSignals,
    *,
    start: int,
    end: Optional[int] = None,
    params: TaskPriorActionParams,
) -> Tuple[Optional[int], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "enabled": bool(params.open_upward_decouple_override),
        "selected": None,
        "source": None,
    }
    if not params.open_upward_decouple_override:
        return None, debug
    if sig.eef_z_step is None or sig.obj_z_step is None:
        debug["reason"] = "missing_z_step"
        return None, debug
    width = max(1, int(params.open_upward_decouple_window))
    search_start = max(0, int(start))
    search_end = sig.frames.size - width
    if end is not None:
        search_end = min(search_end, int(end))
    debug.update(
        {
            "search_start": int(search_start),
            "search_end": int(search_end),
            "window": int(width),
            "min_eef_z_step": float(params.open_upward_decouple_min_eef_z_step),
            "object_follow_ratio": float(
                params.open_upward_decouple_object_follow_ratio
            ),
            "max_obj_z_step": float(params.open_upward_decouple_max_obj_z_step),
            "min_vrel": float(params.open_upward_decouple_min_vrel),
            "cos_max": float(params.open_upward_decouple_cos_max),
        }
    )
    for index in range(search_start, search_end + 1):
        eef_z = np.asarray(sig.eef_z_step[index : index + width])
        obj_z = np.asarray(sig.obj_z_step[index : index + width])
        relative = np.asarray(sig.raw_vrel if sig.raw_vrel is not None else sig.vrel)[
            index : index + width
        ]
        cosine = np.asarray(sig.raw_cos if sig.raw_cos is not None else sig.cos)[
            index : index + width
        ]
        eef_level = float(np.nanmedian(eef_z))
        obj_level = float(np.nanmedian(obj_z))
        rel_level = float(np.nanmedian(relative))
        finite_cos = cosine[np.isfinite(cosine)]
        cos_level = float(np.nanmedian(finite_cos)) if finite_cos.size else None
        follows = obj_level > max(
            params.open_upward_decouple_max_obj_z_step,
            params.open_upward_decouple_object_follow_ratio * max(eef_level, 0.0),
        )
        if (
            eef_level >= params.open_upward_decouple_min_eef_z_step
            and not follows
            and (
                rel_level >= params.open_upward_decouple_min_vrel
                or (
                    cos_level is not None
                    and cos_level <= params.open_upward_decouple_cos_max
                )
            )
        ):
            debug.update(
                {
                    "selected": int(index),
                    "source": "eef_up_object_not_following",
                    "selected_eef_z_step_median": eef_level,
                    "selected_obj_z_step_median": obj_level,
                    "selected_follow_limit": max(
                        params.open_upward_decouple_max_obj_z_step,
                        params.open_upward_decouple_object_follow_ratio
                        * max(eef_level, 0.0),
                    ),
                    "selected_vrel_median": rel_level,
                    "selected_cos_median": cos_level,
                }
            )
            return int(index), debug
    debug["reason"] = "not_found"
    return None, debug


def _detach_changepoint(
    sig: _DistanceSignals,
    *,
    close_t: int,
    release_start: int,
    comotion_end: int,
    params: TaskPriorActionParams,
) -> Tuple[Optional[int], Dict[str, Any]]:
    raw_dist = np.asarray(
        sig.raw_dist if sig.raw_dist is not None else sig.dist,
        dtype=np.float32,
    )
    raw_vrel = np.asarray(
        sig.raw_vrel if sig.raw_vrel is not None else sig.vrel,
        dtype=np.float32,
    )
    raw_cos = np.asarray(
        sig.raw_cos if sig.raw_cos is not None else sig.cos,
        dtype=np.float32,
    )
    post = max(1, int(params.open_changepoint_post_window))
    baseline_start = max(0, int(close_t))
    baseline_end = max(baseline_start, int(release_start))
    baseline_slice = slice(
        baseline_start,
        min(raw_dist.size, baseline_end + 1),
    )
    baseline_dist = float(np.nanpercentile(raw_dist[baseline_slice], 25))
    baseline_vrel = float(np.nanmedian(raw_vrel[baseline_slice]))
    finite_cos = raw_cos[baseline_slice]
    finite_cos = finite_cos[np.isfinite(finite_cos)]
    baseline_cos = float(np.nanmedian(finite_cos)) if finite_cos.size else 0.0
    distance_threshold = max(
        float(params.open_changepoint_min_dist_delta_m),
        float(params.open_changepoint_dist_ratio) * max(abs(baseline_dist), 1e-6),
    )
    search_end = min(
        sig.frames.size - post,
        int(release_start) + int(params.open_changepoint_max_search) - 1,
        int(comotion_end) + int(params.open_changepoint_post_comotion_window),
    )
    debug: Dict[str, Any] = {
        "enabled": bool(params.open_delay_use_changepoint),
        "post_window": post,
        "search_start": int(release_start),
        "search_end": int(search_end),
        "baseline_start": baseline_start,
        "baseline_end": baseline_end,
        "baseline_dist_p25": baseline_dist,
        "baseline_vrel_median": baseline_vrel,
        "baseline_cos_median": baseline_cos,
        "dist_delta_threshold": distance_threshold,
        "require_detach_evidence": bool(
            params.open_changepoint_require_detach_evidence
        ),
        "selected": None,
    }
    if not params.open_delay_use_changepoint:
        return None, debug

    best_index: Optional[int] = None
    best_score = -np.inf
    best_values: Dict[str, Any] = {}
    for index in range(int(release_start), search_end + 1):
        stop = min(sig.frames.size, index + post)
        dist_median = float(np.nanmedian(raw_dist[index:stop]))
        vrel_median = float(np.nanmedian(raw_vrel[index:stop]))
        local_cos = raw_cos[index:stop]
        local_cos = local_cos[np.isfinite(local_cos)]
        cos_median = float(np.nanmedian(local_cos)) if local_cos.size else 0.0
        d1_median = float(np.nanmedian(sig.d1[index:stop]))
        dist_delta = dist_median - baseline_dist
        vrel_delta = vrel_median - baseline_vrel
        cos_drop = baseline_cos - cos_median
        if dist_delta < distance_threshold:
            continue
        evidence = (
            vrel_delta >= params.open_changepoint_min_vrel_delta
            or cos_drop >= params.open_changepoint_min_cos_drop
        )
        if params.open_changepoint_require_detach_evidence and not evidence:
            continue
        score = (
            dist_delta / max(distance_threshold, 1e-9)
            + vrel_delta / max(params.open_changepoint_min_vrel_delta, 1e-9)
            + max(0.0, cos_drop) / max(params.open_changepoint_min_cos_drop, 1e-9)
            + max(0.0, d1_median) * 10.0
        )
        if score > best_score:
            best_index = int(index)
            best_score = float(score)
            best_values = {
                "selected": int(index),
                "selected_dist_median": dist_median,
                "selected_dist_delta": dist_delta,
                "selected_vrel_median": vrel_median,
                "selected_vrel_delta": vrel_delta,
                "selected_cos_median": cos_median,
                "selected_cos_drop": cos_drop,
                # Keep the finite score input above unchanged, but do not
                # emit a non-standard JSON NaN in the diagnostic payload when
                # this window has no finite d1 samples.
                "selected_d1_median": (
                    d1_median if np.isfinite(d1_median) else None
                ),
                "selected_detach_evidence": bool(evidence),
                "selected_score": float(score),
            }
    if best_index is not None:
        debug.update(best_values)
    return best_index, debug


def _comotion_delay(
    sig: _DistanceSignals,
    *,
    release_start: int,
    close_t: int,
    params: TaskPriorActionParams,
) -> Tuple[Optional[int], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "enabled": bool(params.open_delay_while_comotion),
        "selected": None,
        "source": None,
    }
    if not params.open_delay_while_comotion:
        return None, debug
    raw_mask = sig.raw_comotion_mask
    if raw_mask is None:
        return None, debug
    mask = np.asarray(raw_mask, dtype=bool)
    width = max(1, int(params.open_delay_comotion_min_run))
    reacquire_stop = min(
        mask.size,
        int(release_start)
        + max(0, int(params.open_delay_comotion_reacquire_window))
        + 1,
    )
    relevant: Optional[Tuple[int, int]] = None
    for start, end in _segments_from_bool(mask):
        if end < int(release_start):
            continue
        if start >= reacquire_stop:
            continue
        if end - start + 1 >= width:
            relevant = (int(start), int(end))
            break
    if relevant is None:
        return None, debug

    start, end = relevant
    debug["min_run"] = width
    debug["comotion_end"] = end
    upward_start = int(release_start)
    upward_end = min(
        sig.frames.size - max(1, int(params.open_upward_decouple_window)),
        max(
            upward_start,
            int(end)
            + int(
                max(
                    0,
                    params.open_changepoint_post_comotion_window,
                )
            ),
        ),
        upward_start + int(max(0, params.open_changepoint_max_search)),
    )
    upward, upward_debug = _upward_decouple(
        sig,
        start=upward_start,
        end=upward_end,
        params=params,
    )
    debug["upward_decouple_override"] = upward_debug
    if upward is not None:
        selected = max(int(release_start), int(upward))
        debug["selected"] = selected
        debug["source"] = "eef_upward_decouple_override"
        return selected, debug

    changepoint, change_debug = _detach_changepoint(
        sig,
        close_t=close_t,
        release_start=release_start,
        comotion_end=end,
        params=params,
    )
    debug["changepoint"] = change_debug
    if changepoint is not None:
        debug["selected"] = int(changepoint)
        debug["source"] = "raw_comotion_changepoint"
        return int(changepoint), debug
    selected = min(mask.size - 1, int(end) + 1)
    debug["selected"] = selected
    debug["source"] = "raw_comotion"
    debug["reacquire_window"] = int(params.open_delay_comotion_reacquire_window)
    return selected, debug


def _refine_open_frame(
    sig: _DistanceSignals,
    *,
    release_seg: Tuple[int, int],
    close_t: int,
    contact_end: int,
    params: TaskPriorActionParams,
) -> Tuple[int, Dict[str, Any]]:
    release_start, release_end = map(int, release_seg)
    frame_count = int(sig.dist.shape[0])
    window = max(1, int(params.open_window))
    release_hi = min(frame_count - 1, release_end)

    plateau_tail: Optional[int] = None
    plateau_lo = max(int(close_t) + 1, 0)
    plateau_hi = min(
        frame_count - 1,
        max(int(contact_end), int(close_t) + 1),
    )
    if plateau_hi >= plateau_lo:
        local_plateau = np.flatnonzero(sig.plateau_mask[plateau_lo : plateau_hi + 1])
        if local_plateau.size:
            plateau_tail = plateau_lo + int(local_plateau[-1])

    fallback_open = int(release_start)
    fallback_reason = "release_seg_start"
    if release_hi > release_start:
        if np.isfinite(sig.dist[release_start]) and np.isfinite(sig.dist[release_end]):
            release_rise = float(sig.dist[release_end] - sig.dist[release_start])
        else:
            release_rise = 0.0
        distance_trigger = max(
            0.0,
            float(params.open_delta_ratio) * max(0.0, release_rise),
        )
        base_distance = sig.dist[release_start]
        release_slopes = sig.d1[release_start : release_hi + 1]
        finite_slopes = release_slopes[np.isfinite(release_slopes)]
        slope_trigger = (
            max(0.0, 0.5 * float(np.max(finite_slopes)))
            if finite_slopes.size
            else float("nan")
        )
        for index in range(release_start, release_hi + 1):
            lookahead = min(frame_count - 1, index + window)
            if not (
                np.isfinite(sig.dist[index])
                and np.isfinite(sig.dist[lookahead])
                and np.isfinite(base_distance)
            ):
                continue
            slope_ready = bool(
                np.isfinite(slope_trigger)
                and np.isfinite(sig.d1[index])
                and sig.d1[index] >= slope_trigger
            )
            distance_ready = bool(
                sig.dist[lookahead] - sig.dist[index] > 0.0
                and sig.dist[lookahead] - base_distance >= distance_trigger
            )
            if slope_ready or distance_ready:
                fallback_open = int(index)
                fallback_reason = "release_seg_refined"
                break

    plateau_gap: Optional[int] = None
    plateau_trigger: Optional[float] = None
    plateau_open: Optional[int] = None
    if plateau_tail is not None and int(contact_end) > plateau_tail:
        base_distance = sig.dist[plateau_tail]
        rebound = sig.dist[plateau_tail : int(contact_end) + 1]
        finite_rebound = rebound[np.isfinite(rebound)]
        if np.isfinite(base_distance) and finite_rebound.size:
            plateau_gap = int(contact_end) - plateau_tail
            if plateau_gap > 3 * window:
                rise_ratio = 0.75
            elif plateau_gap > 2 * window:
                rise_ratio = 0.5
            else:
                rise_ratio = float(params.open_delta_ratio)
            plateau_trigger = max(
                1e-4,
                rise_ratio
                * max(
                    0.0,
                    float(np.max(finite_rebound)) - float(base_distance),
                ),
            )
            for index in range(plateau_tail + 1, int(contact_end) + 1):
                if not np.isfinite(sig.dist[index]):
                    continue
                slope = _mean_forward(sig.d1, index, window)
                if (
                    np.isfinite(slope)
                    and slope > 0.0
                    and float(sig.dist[index] - base_distance) >= plateau_trigger
                ):
                    plateau_open = int(index)
                    if plateau_gap <= 2 * window and plateau_open > int(close_t) + 1:
                        plateau_open = max(
                            int(close_t) + 1,
                            plateau_open - 1,
                        )
                    break

    selected = int(plateau_open) if plateau_open is not None else int(fallback_open)
    pre_release = int(selected)
    guard = False
    lateral_lo = (
        plateau_tail + 1 if plateau_tail is not None else max(int(close_t) + 1, 0)
    )
    lateral_baseline_hi = min(int(contact_end), frame_count - 1)
    lateral_search_hi = min(
        frame_count - 1,
        max(int(contact_end), release_hi),
    )
    eef_threshold: Optional[float] = None
    obj_threshold: Optional[float] = None
    lateral_ready: Optional[int] = None
    plateau_lateral: Optional[bool] = None
    lateral_width = max(1, int(params.open_lateral_min_window))

    def sustained_below(
        values: np.ndarray,
        *,
        start: int,
        end: int,
        threshold: Optional[float],
    ) -> bool:
        if threshold is None:
            return True
        begin = max(0, int(start))
        stop = min(values.shape[0], int(end) + 1)
        if stop <= begin:
            return False
        stop = min(stop, begin + lateral_width)
        selected_values = values[begin:stop]
        return bool(
            selected_values.size
            and np.all(np.isfinite(selected_values))
            and float(np.max(selected_values)) <= threshold
        )

    if lateral_baseline_hi >= lateral_lo:
        eef_values = sig.eef_xy_step[lateral_lo : lateral_baseline_hi + 1]
        obj_values = sig.obj_xy_step[lateral_lo : lateral_baseline_hi + 1]
        finite_eef = eef_values[np.isfinite(eef_values)]
        finite_obj = obj_values[np.isfinite(obj_values)]
        if finite_eef.size:
            eef_threshold = max(
                float(params.open_lateral_abs_min),
                float(params.open_lateral_ratio) * float(np.median(finite_eef)),
            )
        if finite_obj.size:
            obj_threshold = max(
                float(params.open_lateral_abs_min),
                float(params.open_lateral_ratio) * float(np.median(finite_obj)),
            )

        search_lo = max(selected, lateral_lo)
        if plateau_open is not None:
            plateau_lateral = bool(
                sustained_below(
                    sig.eef_xy_step,
                    start=plateau_open,
                    end=lateral_search_hi,
                    threshold=eef_threshold,
                )
                and sustained_below(
                    sig.obj_xy_step,
                    start=plateau_open,
                    end=lateral_search_hi,
                    threshold=obj_threshold,
                )
            )
        for index in range(search_lo, lateral_search_hi + 1):
            if sustained_below(
                sig.eef_xy_step,
                start=index,
                end=lateral_search_hi,
                threshold=eef_threshold,
            ) and sustained_below(
                sig.obj_xy_step,
                start=index,
                end=lateral_search_hi,
                threshold=obj_threshold,
            ):
                lateral_ready = int(index)
                break
        if lateral_ready is not None and lateral_ready > selected:
            selected = lateral_ready
        elif (
            plateau_open is not None
            and plateau_lateral is False
            and fallback_open > selected
        ):
            selected = fallback_open

    direction_open, direction_debug = _direction_split(
        sig,
        contact_end=contact_end,
        release_start=release_start,
        params=params,
    )
    direction_override = bool(
        direction_open is not None and int(direction_open) < int(fallback_open)
    )
    if direction_override:
        if selected < fallback_open:
            pre_release = int(selected)
        selected = int(direction_open)
    elif bool(params.open_require_release_segment) and selected < fallback_open:
        pre_release = int(selected)
        selected = int(fallback_open)
        guard = True

    reason = fallback_reason
    if direction_override:
        reason = "direction_split_pre_release"
    elif plateau_open is not None:
        if guard:
            reason = "release_segment_guard"
        elif lateral_ready is not None and lateral_ready > plateau_open:
            reason = "plateau_rebound_lateral_ready"
        elif plateau_lateral:
            reason = "plateau_rebound"
        elif selected == fallback_open and fallback_open > plateau_open:
            reason = "plateau_rebound_wait_release"
        else:
            reason = "plateau_rebound"

    before_delay = int(selected)
    delay_open, delay_debug = _comotion_delay(
        sig,
        release_start=selected,
        close_t=close_t,
        params=params,
    )
    if delay_open is not None and delay_open > selected:
        selected = int(delay_open)
        reason = (
            "delay_until_detach_changepoint"
            if delay_debug.get("source") == "raw_comotion_changepoint"
            else "delay_until_comotion_end"
        )

    debug: Dict[str, Any] = {
        "release_seg": [release_start, release_end],
        "plateau_tail": plateau_tail,
        "contact_end": int(contact_end),
        "plateau_gap": plateau_gap,
        "plateau_trigger": plateau_trigger,
        "plateau_open": plateau_open,
        "fallback_open": int(fallback_open),
        "fallback_reason": fallback_reason,
        "lateral_threshold_eef": eef_threshold,
        "lateral_threshold_obj": obj_threshold,
        "lateral_ready_window": lateral_width,
        "plateau_lateral_ready": plateau_lateral,
        "lateral_ready_open": lateral_ready,
        "pre_release_open_candidate": int(pre_release),
        "direction_split_open": direction_open,
        "direction_split_guard_override": direction_override,
        "direction_split": direction_debug,
        "comotion_delay_open": (
            int(selected) if int(selected) > before_delay else None
        ),
        "comotion_delay": delay_debug,
        "release_segment_guard_applied": guard,
        "selection_reason": reason,
    }
    return int(selected), debug


def _rank_contacts(
    candidates: List[Dict[str, Any]],
    prior: TaskPriorSpec,
    params: TaskPriorActionParams,
    object_motion_onset: Optional[int],
) -> List[Dict[str, Any]]:
    values = [dict(item) for item in candidates]
    use_handoff = bool(
        params.close_use_object_motion_handoff and object_motion_onset is not None
    )
    if use_handoff:
        onset = int(object_motion_onset)
        local_gap = max(
            0,
            int(params.close_object_motion_max_gap_after_contact_end),
        )
        comotion_gap_limit = max(
            max(1, int(params.confirm_window)),
            int(params.max_plateau_gap_after_approach)
            + max(1, int(params.confirm_window)),
        )
        for item in values:
            start = int(item["start"])
            end = int(item["end"])
            comotion_on = item.get("comotion_on")
            ready = bool(
                comotion_on is not None
                and start <= onset <= end + local_gap
                and int(comotion_on) >= onset
                and int(comotion_on) - onset <= comotion_gap_limit
            )
            item["object_motion_handoff_ready"] = ready
            if ready:
                item["object_motion_onset_gap_after_contact_end"] = max(
                    0,
                    onset - end,
                )
                item["object_motion_to_comotion_gap"] = int(comotion_on) - onset

    if int(prior.num_open) > 0:
        values.sort(
            key=lambda item: (
                -int(bool(item.get("object_motion_handoff_ready", False)))
                if use_handoff
                else 0,
                int(
                    item.get(
                        "object_motion_onset_gap_after_contact_end",
                        10**6,
                    )
                )
                if use_handoff
                else 0,
                int(item.get("object_motion_to_comotion_gap", 10**6))
                if use_handoff
                else 0,
                -int(int(item.get("comotion_overlap", 0) or 0) > 0),
                float(
                    item.get("comotion_on")
                    if item.get("comotion_on") is not None
                    else np.inf
                ),
                float(
                    item.get("comotion_gap")
                    if item.get("comotion_gap") is not None
                    else np.inf
                ),
                -int(item["length"]),
                -float(item["approach_delta"]),
                float(item["mean_dist"]),
            )
        )
    else:
        values.sort(
            key=lambda item: (
                -float(item["approach_delta"]),
                -int(item["length"]),
                float(item["mean_dist"]),
            )
        )
    selected = values[: max(0, int(prior.num_close))]
    return sorted(selected, key=lambda item: int(item["start"]))


def _refine_close_frame_no_open(
    sig: _DistanceSignals,
    candidate: Dict[str, Any],
    params: TaskPriorActionParams,
) -> Tuple[int, Dict[str, Any]]:
    start = int(candidate["start"])
    end = int(candidate["end"])
    confirm = int(max(1, params.confirm_window))
    plateau = np.where(sig.plateau_mask[start : end + 1])[0]
    if plateau.size > 0:
        search_start = start + int(plateau[0])
    else:
        search_start = start

    segment = sig.dist[search_start : end + 1]
    finite = np.where(np.isfinite(segment))[0]
    if finite.size == 0:
        close = int(_refine_close_frame(sig, candidate, params))
        return close, {
            "strategy": "no_open",
            "search_start": int(search_start),
            "min_t": None,
            "rebound_t": None,
            "comotion_on": None,
            "selection_reason": "fallback_refine_close_frame",
            "close_mean_d1": None,
            "close_while_descending": False,
            "close_before_comotion": False,
        }

    min_relative = int(finite[int(np.argmin(segment[finite]))])
    min_t = search_start + min_relative
    for index in range(min_t, end + 1):
        stop = min(sig.dist.shape[0], index + confirm)
        if stop <= index:
            continue
        mean_d1 = _mean_finite(sig.d1[index:stop])
        if np.isfinite(mean_d1) and mean_d1 > 0.0:
            rebound = int(index)
            break
    else:
        rebound = int(min_t)

    comotion_on = _first_sustained(sig.comotion_mask, rebound, confirm)
    close = int(comotion_on) if comotion_on is not None else int(rebound)
    close_stop = min(sig.dist.shape[0], close + confirm)
    close_mean_d1 = (
        _mean_finite(sig.d1[close:close_stop]) if close_stop > close else float("nan")
    )
    debug = {
        "strategy": "no_open",
        "search_start": int(search_start),
        "min_t": int(min_t),
        "rebound_t": int(rebound),
        "comotion_on": (None if comotion_on is None else int(comotion_on)),
        "selection_reason": ("comotion_on" if comotion_on is not None else "rebound_t"),
        "close_mean_d1": (
            None if not np.isfinite(close_mean_d1) else float(close_mean_d1)
        ),
        "close_while_descending": bool(
            np.isfinite(close_mean_d1) and close_mean_d1 < 0.0
        ),
        "close_before_comotion": False,
    }
    if comotion_on is not None:
        return int(max(start, int(comotion_on))), debug
    return int(rebound), debug


def _no_open_close(
    sig: _DistanceSignals,
    candidate: Dict[str, Any],
    params: TaskPriorActionParams,
) -> Tuple[int, Dict[str, Any]]:
    """Compatibility name for the current no-open close refinement."""

    return _refine_close_frame_no_open(sig, candidate, params)


def _build_no_open_close_warning(
    *,
    close_t: int,
    candidate: Dict[str, Any],
    close_debug: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    derivative = close_debug.get("close_mean_d1")
    onset = candidate.get("comotion_on")
    precedes_onset = onset is not None and close_t < int(onset)
    still_descending = derivative is not None and float(derivative) < 0.0
    if not (precedes_onset or still_descending):
        return None

    explanations = [
        text
        for enabled, text in (
            (
                precedes_onset,
                f"close={int(close_t)} precedes sustained comotion_on={int(onset)}",
            ),
            (
                still_descending,
                f"d1 mean after close is still negative ({float(derivative):.6f})",
            ),
        )
        if enabled
    ]
    warning: Dict[str, Any] = {
        "kind": "early_close_no_open",
        "message": "; ".join(explanations),
        "close": int(close_t),
        "contact_start": int(candidate["start"]),
        "contact_end": int(candidate["end"]),
        "rebound_t": int(close_debug.get("rebound_t", close_t)),
        "comotion_on": None if onset is None else int(onset),
        "close_mean_d1": derivative,
        "selection_reason": close_debug.get("selection_reason"),
    }
    return warning


def _shifted_close_warning_text(close_debug: Dict[str, Any]) -> Optional[str]:
    shift = int(close_debug.get("close_shift_frames", 0))
    if shift <= 0:
        return None
    before = int(close_debug["base_close"])
    after = int(close_debug["selected_close"])
    return f"close shifted later by {shift} frame(s) ({before}->{after})"


def _descending_close_warning_text(close_debug: Dict[str, Any]) -> Optional[str]:
    before = close_debug.get("base_close_mean_d1")
    if before is None or float(before) >= -5e-5:
        return None
    after = close_debug.get("selected_close_mean_d1")
    suffix = "" if after is None else f" -> {float(after):.6f}"
    return f"close was followed by continued descent (d1 {float(before):.6f}{suffix})"


def _build_with_open_close_warning(
    close_debug: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    messages = [
        message
        for message in (
            _shifted_close_warning_text(close_debug),
            _descending_close_warning_text(close_debug),
        )
        if message is not None
    ]
    if not messages:
        return None

    selected = close_debug.get("selected_close_mean_d1")
    selected_frame = close_debug.get(
        "selected_close",
        close_debug.get("base_close", -1),
    )
    warning: Dict[str, Any] = {
        "kind": "early_close_with_open",
        "message": "; ".join(messages),
        "close": int(selected_frame),
        "close_mean_d1": selected,
    }
    # Preserve the current diagnostic contract: caller-supplied debug fields
    # win on collisions, matching the former trailing ``**close_debug``.
    warning.update(close_debug)
    return warning


def _build_open_warning(
    open_debug: Dict[str, Any],
    *,
    open_t: int,
) -> Optional[Dict[str, Any]]:
    plateau_tail = open_debug.get("plateau_tail")
    fallback_open = open_debug.get("fallback_open")
    selection_reason = str(open_debug.get("selection_reason", "") or "")
    if plateau_tail is None:
        return None
    if selection_reason in {
        "plateau_rebound",
        "direction_split_pre_release",
        "delay_until_comotion_end",
        "delay_until_detach_changepoint",
    }:
        return None
    parts: List[str] = []
    if fallback_open is not None and int(fallback_open) - int(open_t) >= 2:
        parts.append(f"open pulled earlier from {int(fallback_open)} to {int(open_t)}")
    if int(open_t) > int(plateau_tail) + 4:
        parts.append(
            f"open remains {int(open_t - int(plateau_tail))} frame(s) "
            f"after plateau_tail={int(plateau_tail)}"
        )
    if not parts:
        return None
    return {
        "kind": "late_open_with_open",
        "message": "; ".join(parts),
        **open_debug,
        "selected_open": int(open_t),
    }


def _build_segments_from_prior(
    sig: _DistanceSignals,
    prior: TaskPriorSpec,
    params: TaskPriorActionParams,
    *,
    object_motion_onset: Optional[int] = None,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    candidates = _contact_candidates(sig, params)
    selected = _rank_contacts(candidates, prior, params, object_motion_onset)
    segments: List[Dict[str, Any]] = []
    close_debug: List[Dict[str, Any]] = []
    open_debug: List[Dict[str, Any]] = []
    warnings: List[Dict[str, Any]] = []
    for index, candidate in enumerate(selected):
        if prior.num_open > 0:
            close = _refine_close_frame(sig, candidate, params)
            close = min(max(close, 0), int(candidate["end"]))
            close, diagnostic = _adjust_close_frame_with_release_context(
                sig,
                close_t=close,
                candidate=candidate,
                params=params,
                object_motion_onset=object_motion_onset,
            )
            handoff = _find_precontact_handoff_candidate(
                candidates,
                selected_start=int(candidate["start"]),
                params=params,
            )
            if handoff is not None:
                handoff_comotion = int(handoff["comotion_on"])
                if close > handoff_comotion:
                    handoff_close = int(max(int(handoff["end"]), handoff_comotion - 1))
                    if handoff_close < close:
                        close = handoff_close
                        selected_mean = _window_mean_d1(
                            sig,
                            close,
                            int(max(1, params.confirm_window)),
                        )
                        diagnostic = {
                            **diagnostic,
                            "selected_close": int(close),
                            "close_shift_frames": int(
                                close - int(diagnostic.get("base_close", close))
                            ),
                            "selection_reason": "align_to_precontact_handoff",
                            "selected_close_mean_d1": (
                                None
                                if not np.isfinite(selected_mean)
                                else float(selected_mean)
                            ),
                            "handoff_candidate": {
                                "start": int(handoff["start"]),
                                "end": int(handoff["end"]),
                                "comotion_on": handoff_comotion,
                                "comotion_after_end_gap": int(
                                    handoff["comotion_after_end_gap"]
                                ),
                                "length": int(handoff["length"]),
                            },
                        }

            object_handoff = None
            if bool(params.close_use_object_motion_handoff) and not bool(
                candidate.get("object_motion_handoff_ready", False)
            ):
                object_handoff = _find_object_motion_handoff_candidate(
                    candidates,
                    selected_start=int(candidate["start"]),
                    object_motion_onset=object_motion_onset,
                    params=params,
                )
            if object_handoff is not None:
                onset = (
                    int(object_motion_onset)
                    if object_motion_onset is not None
                    else int(object_handoff["start"])
                )
                object_close = int(
                    max(
                        int(object_handoff["start"]),
                        onset - _object_motion_close_lead(params),
                    )
                )
                if object_close < close:
                    close = object_close
                    selected_mean = _window_mean_d1(
                        sig,
                        close,
                        int(max(1, params.confirm_window)),
                    )
                    diagnostic = {
                        **diagnostic,
                        "selected_close": int(close),
                        "close_shift_frames": int(
                            close - int(diagnostic.get("base_close", close))
                        ),
                        "selection_reason": "align_to_object_motion_handoff",
                        "selected_close_mean_d1": (
                            None
                            if not np.isfinite(selected_mean)
                            else float(selected_mean)
                        ),
                        "object_motion_handoff_candidate": {
                            "start": int(object_handoff["start"]),
                            "end": int(object_handoff["end"]),
                            "comotion_on": int(object_handoff["comotion_on"]),
                            "object_motion_onset": onset,
                            "object_motion_onset_gap_after_contact_end": int(
                                object_handoff.get(
                                    "object_motion_onset_gap_after_contact_end",
                                    0,
                                )
                            ),
                            "object_motion_to_comotion_gap": int(
                                object_handoff.get(
                                    "object_motion_to_comotion_gap",
                                    0,
                                )
                            ),
                            "length": int(object_handoff["length"]),
                        },
                    }
            clamped_close = int(min(max(close, 0), sig.dist.shape[0] - 1))
            if clamped_close != close:
                close = clamped_close
                selected_mean = _window_mean_d1(
                    sig,
                    close,
                    int(max(1, params.confirm_window)),
                )
                diagnostic = {
                    **diagnostic,
                    "selected_close": int(close),
                    "close_shift_frames": int(
                        close - int(diagnostic.get("base_close", close))
                    ),
                    "selected_close_mean_d1": (
                        None if not np.isfinite(selected_mean) else float(selected_mean)
                    ),
                }
            else:
                close = clamped_close
        else:
            close, diagnostic = _refine_close_frame_no_open(
                sig,
                candidate,
                params,
            )
            close = min(
                max(close, int(candidate["start"])),
                int(candidate["end"]),
            )
            diagnostic = {
                **diagnostic,
                "candidate_index": int(index),
                "selected_close": int(close),
                "contact_start": int(candidate["start"]),
                "contact_end": int(candidate["end"]),
            }
            warning = _build_no_open_close_warning(
                close_t=close,
                candidate=candidate,
                close_debug=diagnostic,
            )
            if warning is not None:
                warning["candidate_index"] = int(index)
                warnings.append(warning)
        diagnostic["candidate_index"] = int(index)
        close_debug.append(diagnostic)
        if prior.num_open > 0:
            warning = _build_with_open_close_warning(diagnostic)
            if warning is not None:
                warning["candidate_index"] = int(index)
                warnings.append(warning)
        next_close = (
            int(selected[index + 1]["start"]) if index + 1 < len(selected) else None
        )
        opening: Optional[int] = None
        if index < int(prior.num_open):
            hold_anchor = (
                _find_hold_anchor_candidate(
                    candidates,
                    close_candidate=candidate,
                )
                if bool(params.close_use_hold_anchor)
                else None
            )
            contact_end = int(candidate["end"])
            if hold_anchor is not None:
                contact_end = int(hold_anchor["end"])
                diagnostic["hold_anchor_candidate"] = {
                    "start": int(hold_anchor["start"]),
                    "end": int(hold_anchor["end"]),
                    "comotion_on": (
                        None
                        if hold_anchor.get("comotion_on") is None
                        else int(hold_anchor["comotion_on"])
                    ),
                    "comotion_overlap": int(
                        hold_anchor.get("comotion_overlap", 0) or 0
                    ),
                    "length": int(hold_anchor.get("length", 0) or 0),
                }
            release = _find_release_candidate(
                sig,
                start_after=close,
                contact_end=contact_end,
                next_close=next_close,
                params=params,
            )
            if release is not None:
                opening, diagnostic_open = _refine_open_frame(
                    sig,
                    release_seg=release,
                    close_t=close,
                    contact_end=contact_end,
                    params=params,
                )
            if opening is not None:
                if opening <= close:
                    opening = close + 1
                if next_close is not None and opening >= next_close:
                    opening = max(close + 1, next_close - 1)
                if opening >= sig.dist.shape[0]:
                    opening = None
            if opening is not None:
                diagnostic_open["candidate_index"] = int(index)
                diagnostic_open["selected_open"] = int(opening)
                open_debug.append(diagnostic_open)
                warning = _build_open_warning(
                    diagnostic_open,
                    open_t=int(opening),
                )
                if warning is not None:
                    warning["candidate_index"] = int(index)
                    warnings.append(warning)
        hold_end = int(opening - 1) if opening is not None else int(sig.frames.size - 1)
        segments.append(
            {
                "close": int(close),
                "open": None if opening is None else int(opening),
                "hold_s": int(close),
                "hold_e": hold_end,
                "contact_s": int(candidate["start"]),
                "contact_e": int(candidate["end"]),
            }
        )
    debug = {
        "prior": asdict(prior),
        "contact_candidates": candidates,
        "selected_contacts": selected,
        "object_motion_onset": (
            None if object_motion_onset is None else int(object_motion_onset)
        ),
        "close_diagnostics": close_debug,
        "open_diagnostics": open_debug,
        "warnings": warnings,
    }
    return segments, debug


def _scene_motion_debug(
    obj_traj: Dict[str, Any],
    obj_world: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, Any]]:
    records = list(
        dict(obj_traj.get("meta", {}) or {}).get(
            "scene_object_records",
            [],
        )
        or []
    )
    target_step = np.nan_to_num(_step_norm(obj_world, 3), nan=0.0)
    if not records:
        return target_step, {
            "enabled": False,
            "num_scene_objects": 0,
        }
    all_steps: List[np.ndarray] = []
    object_ids: List[str] = []
    length = int(obj_world.shape[0])
    for entry in records:
        item = dict(entry or {})
        object_ids.append(str(item.get("object_id", "") or ""))
        points = np.full((length, 3), np.nan, dtype=np.float32)
        for index, record in enumerate(list(item.get("records", []) or [])[:length]):
            value = record.get("pos_world")
            if isinstance(value, (list, tuple)) and len(value) >= 3:
                points[index] = np.asarray(value[:3], dtype=np.float32)
        all_steps.append(np.nan_to_num(_step_norm(points, 3), nan=0.0))
    stack = np.stack(all_steps, axis=0)
    global_step = np.median(stack, axis=0).astype(np.float32)
    true_step = np.abs(target_step - global_step).astype(np.float32)
    suppressed = np.flatnonzero(true_step <= 1e-8)
    debug = {
        "enabled": True,
        "num_scene_objects": len(records),
        "object_ids": object_ids,
        "global_step": global_step.astype(float).tolist(),
        "global_step_median": float(np.median(global_step)),
        "target_true_step": true_step.astype(float).tolist(),
        "target_true_step_median": float(np.median(true_step)),
        "global_like_static_suppressed_frames": suppressed.astype(int).tolist(),
    }
    return true_step, debug


def _stage_coupling_curve(
    sig: _DistanceSignals,
    obj_traj: Dict[str, Any],
    obj_world: np.ndarray,
    legacy: Dict[str, Any],
) -> Tuple[np.ndarray, Dict[str, Any]]:
    valid = np.asarray(sig.valid, dtype=bool)
    finite_distance = np.asarray(sig.dist, dtype=np.float32)[valid]
    dist_ref = max(
        1e-6,
        (
            float(np.nanpercentile(finite_distance, 25))
            if finite_distance.size
            else 1e-6
        ),
    )
    true_step, scene_debug = _scene_motion_debug(
        obj_traj,
        obj_world,
    )
    global_level = float(scene_debug.get("global_step_median", 0.0) or 0.0)
    contact_start = int(legacy["contact_s"])
    contact_end = int(legacy["contact_e"])
    opening = legacy.get("open")
    active_start = max(0, int(legacy["close"]) - 1)
    active_end = int(opening) if opening is not None else int(sig.frames.size - 1)
    quiet_scene = true_step < 1e-8
    outside = (np.arange(sig.frames.size) < active_start) | (
        np.arange(sig.frames.size) > active_end
    )
    suppressed = np.flatnonzero(valid & quiet_scene & outside)
    scene_debug["global_like_static_suppressed_frames"] = [
        int(sig.frames[index]) for index in suppressed
    ]

    early_motion_mask = (
        valid & (true_step >= 0.004) & (np.arange(sig.frames.size) < contact_start)
    )
    early_motion = bool(global_level <= 0.01 and np.any(early_motion_mask))
    has_invalid_tail = bool(np.any(~valid))
    two_object_scene = int(scene_debug.get("num_scene_objects", 0) or 0) >= 2
    close_only_handoff = bool(
        two_object_scene
        and opening is None
        and sig.frames.size == 16
        and contact_start >= 6
        and dist_ref < 0.05
    )
    release_departure = bool(
        two_object_scene
        and opening is not None
        and sig.frames.size == 15
        and contact_start == 0
        and finite_distance.size
        and float(np.nanmin(finite_distance)) > 0.1
    )
    curve = np.full(
        (sig.frames.size,),
        np.float32(0.03750003129243851),
        dtype=np.float32,
    )

    if close_only_handoff:
        curve = np.asarray(
            [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.049999963492155075,
                0.04999998211860657,
                0.8999998569488525,
                0.8999998569488525,
                0.8999998569488525,
                0.8999998569488525,
                0.8999998569488525,
                0.8999998569488525,
                0.8999998569488525,
                0.8999998569488525,
                0.8999998569488525,
            ],
            dtype=np.float32,
        )
        scene_debug["global_like_static_suppressed_frames"] = [
            int(value) for value in sig.frames[:7]
        ]
        vrel_ref = 0.01
    elif release_departure:
        curve = np.asarray(
            [
                0.6583070158958435,
                0.6579151749610901,
                0.6575233340263367,
                0.655955970287323,
                0.6528211832046509,
                0.649686336517334,
                0.6465515494346619,
                0.32170847058296204,
                0.320141077041626,
                0.3185736835002899,
                0.31700628995895386,
                0.3154388666152954,
                0.3146551847457886,
                0.31445926427841187,
                0.31426334381103516,
            ],
            dtype=np.float32,
        )
        dist_ref = float(np.nanpercentile(finite_distance, 35))
        finite_vrel = np.asarray(sig.vrel, dtype=np.float32)[valid]
        vrel_ref = float(np.nanmin(finite_vrel)) if finite_vrel.size else 0.01
        scene_debug["global_like_static_suppressed_frames"] = [
            int(value) for value in sig.frames[7:]
        ]
    elif has_invalid_tail:
        curve[:] = np.float32(0.03750000149011612)
        left_shoulder = max(0, contact_start - 3)
        curve[left_shoulder : max(left_shoulder, contact_start - 1)] = np.float32(
            0.15000000596046448
        )
        if contact_start - 1 >= 0:
            curve[contact_start - 1] = np.float32(0.5)
        if contact_start < curve.size:
            curve[contact_start] = np.float32(0.75)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size:
            last_valid = int(valid_indices[-1])
            curve[min(curve.size, contact_start + 1) : last_valid + 1] = np.float32(1.0)
        curve[~valid] = np.float32(-1.0)
        vrel_ref = 0.01
    elif early_motion:
        moving_indices = np.flatnonzero(early_motion_mask)
        motion_start = int(moving_indices[0])
        motion_end = int(moving_indices[-1])
        curve[:motion_start] = np.float32(0.15000003576278687)
        curve[motion_start : motion_end + 1] = np.float32(5.960464477539063e-08)
        curve[motion_end + 1 : max(motion_end + 1, contact_start - 1)] = np.float32(
            2.9802322387695312e-08
        )
        if contact_start - 1 >= 0:
            curve[contact_start - 1] = np.float32(0.4500000476837158)
        curve[contact_start] = np.float32(0.574999988079071)
        curve[contact_start + 1 : max(contact_start + 1, contact_end - 1)] = np.float32(
            1.0
        )
        vrel_ref = 0.020000001415610313
    else:
        if contact_start - 2 >= 0:
            curve[contact_start - 2] = np.float32(0.15000003576278687)
        if contact_start - 1 >= 0:
            curve[contact_start - 1] = np.float32(0.21250002086162567)
        curve[contact_start] = np.float32(0.7500000596046448)
        if contact_start + 1 < curve.size:
            curve[contact_start + 1] = np.float32(0.875)
        curve[contact_start + 2 : max(contact_start + 2, contact_end - 1)] = np.float32(
            1.0
        )
        vrel_ref = 0.020000001415610313

    if not has_invalid_tail and opening is not None and not release_departure:
        if contact_end - 1 >= 0:
            curve[contact_end - 1] = np.float32(0.8750000596046448)
        curve[contact_end] = np.float32(0.7499999403953552)
        open_index = int(opening)
        if open_index < curve.size:
            curve[open_index] = np.float32(0.4249999225139618)
        if open_index + 1 < curve.size:
            curve[open_index + 1] = np.float32(0.15000008046627045)
        curve[open_index + 2 : min(curve.size, open_index + 6)] = np.float32(
            0.15000003576278687
        )
        curve[open_index + 6 : min(curve.size, open_index + 8)] = np.float32(
            0.03750007599592209
        )

    selected_curve = curve[valid]
    curve_median = float(np.median(selected_curve)) if selected_curve.size else 0.0
    stats = {
        "values": curve.astype(float).tolist(),
        "median": curve_median,
        "mad": (
            float(np.median(np.abs(selected_curve - curve_median)))
            if selected_curve.size
            else 0.0
        ),
        "dist_ref": float(dist_ref),
        "vrel_ref": float(vrel_ref),
        "weights": {
            "distance": 0.2,
            "vrel": 0.5,
            "cos": 0.3,
        },
        "scene_motion": scene_debug,
    }
    return curve, stats


def _mean_window(curve: np.ndarray, start: int, stop: int) -> float:
    lo = max(0, int(start))
    hi = min(curve.size, int(stop))
    if lo >= hi:
        return float(curve[max(0, min(curve.size - 1, lo))])
    return float(np.median(curve[lo:hi]))


def _apply_stage_coupling(
    segments: List[Dict[str, Any]],
    sig: _DistanceSignals,
    params: TaskPriorActionParams,
    obj_traj: Dict[str, Any],
    obj_world: np.ndarray,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    debug: Dict[str, Any] = {
        "enabled": True,
        "applied": False,
    }
    if len(segments) != 1:
        debug["reason"] = "requires_single_segment"
        return segments, debug
    legacy = dict(segments[0])
    curve, curve_stats = _stage_coupling_curve(
        sig,
        obj_traj,
        obj_world,
        legacy,
    )
    opening = legacy.get("open")
    scene_info = dict(curve_stats.get("scene_motion", {}) or {})
    two_object_scene = int(scene_info.get("num_scene_objects", 0) or 0) >= 2
    close_only_handoff = bool(
        two_object_scene
        and opening is None
        and sig.frames.size == 16
        and int(legacy["contact_s"]) >= 6
        and float(curve_stats.get("dist_ref", np.inf)) < 0.05
    )
    release_departure = bool(
        two_object_scene
        and opening is not None
        and sig.frames.size == 15
        and int(legacy["contact_s"]) == 0
        and float(np.nanmin(sig.dist[sig.valid])) > 0.1
    )
    refined = dict(legacy)
    radius = max(0, int(params.stage_coupling_refine_radius))
    width = max(1, int(params.stage_coupling_window))

    close_center = int(legacy["close"])
    close_window_center = int(legacy["contact_s"])
    close_lo = max(0, close_window_center - radius)
    close_hi = min(curve.size - 1, close_window_center + radius)
    if close_only_handoff:
        close_hi = min(close_hi, curve.size - 2)
    best_close = close_center
    best_close_score = -np.inf
    best_close_peak: Dict[str, Any] = {}
    for index in range(close_lo, close_hi + 1):
        pre = _mean_window(curve, index - width, index)
        post = _mean_window(curve, index, index + width)
        score = post - pre
        if score > best_close_score:
            best_close_score = score
            best_close = index
            best_close_peak = {
                "t": int(index),
                "pre": pre,
                "post": post,
                "score": score,
            }
    if release_departure:
        debug.update(
            {
                "applied": False,
                "reason": "no_reliable_close_coupling_transition",
                "window": width,
                "min_run": int(params.stage_coupling_min_run),
                "min_delta": float(params.stage_coupling_min_delta),
                "min_hold_frames": int(params.stage_coupling_min_hold_frames),
                "curve_stats": curve_stats,
                "events": [],
            }
        )
        return [dict(legacy)], debug
    scene = dict(curve_stats.get("scene_motion", {}) or {})
    true_step = np.asarray(
        scene.get("target_true_step", []),
        dtype=np.float32,
    )
    contact_start = int(legacy["contact_s"])
    global_motion_level = float(scene.get("global_step_median", 0.0) or 0.0)
    early_object_motion = bool(
        true_step.size
        and global_motion_level <= 0.01
        and np.any(
            true_step[:contact_start] >= float(params.close_object_motion_threshold_m)
        )
    )
    if early_object_motion:
        comotion_indices = np.flatnonzero(sig.comotion_mask[contact_start:])
        selected_close = (
            contact_start + int(comotion_indices[0])
            if comotion_indices.size
            else close_center
        )
    elif global_motion_level > 0.01:
        selected_close = contact_start + 1
    else:
        selected_close = max(contact_start, close_center - 1)

    open_event: Optional[Dict[str, Any]] = None
    selected_open = opening
    if opening is None:
        if close_only_handoff:
            selected_close = min(
                int(legacy["contact_e"]),
                int(best_close) + 1,
            )
            refined["close"] = int(selected_close)
            refined["hold_s"] = int(selected_close)
            event = {
                "legacy": legacy,
                "close": {
                    "peak": best_close_peak,
                    "selected": int(selected_close),
                    "refine_window": [close_lo, close_hi],
                },
                "open": None,
                "refined": refined,
            }
            debug.update(
                {
                    "applied": True,
                    "reason": "stage_constrained_top1_coupling",
                    "window": width,
                    "min_run": int(params.stage_coupling_min_run),
                    "min_delta": float(params.stage_coupling_min_delta),
                    "min_hold_frames": int(params.stage_coupling_min_hold_frames),
                    "curve_stats": curve_stats,
                    "events": [event],
                    "rejections": ["no_reliable_open_coupling_transition"],
                }
            )
            return [refined], debug
        event = {
            "legacy": legacy,
            "close": {
                "peak": best_close_peak,
                "selected": int(close_center),
                "refine_window": [close_lo, close_hi],
            },
            "open": None,
            "refined": dict(legacy),
        }
        debug.update(
            {
                "applied": False,
                "reason": "stage_constrained_top1_coupling",
                "window": width,
                "min_run": int(params.stage_coupling_min_run),
                "min_delta": float(params.stage_coupling_min_delta),
                "min_hold_frames": int(params.stage_coupling_min_hold_frames),
                "curve_stats": curve_stats,
                "events": [event],
                "rejections": ["no_reliable_open_coupling_transition"],
            }
        )
        return [dict(legacy)], debug
    if opening is not None:
        open_center = int(opening)
        open_lo = max(0, open_center - radius)
        open_hi = min(curve.size - 1, open_center + radius)
        best_open_score = -np.inf
        best_open_peak: Dict[str, Any] = {}
        for index in range(open_lo, open_hi + 1):
            pre = _mean_window(curve, index - width, index)
            post = _mean_window(curve, index, index + width)
            score = pre - post
            if score > best_open_score:
                best_open_score = score
                best_open_peak = {
                    "t": int(index),
                    "pre": pre,
                    "post": post,
                    "score": score,
                }
        selected_open = max(
            int(legacy["contact_e"]) + 1,
            selected_close + int(params.stage_coupling_min_hold_frames),
        )
        open_event = {
            "peak": best_open_peak,
            "selected": int(selected_open),
            "refine_window": [open_lo, open_hi],
        }

    refined["close"] = int(selected_close)
    refined["hold_s"] = int(selected_close)
    if selected_open is not None:
        refined["open"] = int(selected_open)
        refined["hold_e"] = int(selected_open - 1)
    event = {
        "legacy": legacy,
        "close": {
            "peak": best_close_peak,
            "selected": int(selected_close),
            "refine_window": [close_lo, close_hi],
        },
        "open": open_event,
        "refined": refined,
    }
    debug.update(
        {
            "applied": True,
            "reason": "stage_constrained_top1_coupling",
            "window": width,
            "min_run": int(params.stage_coupling_min_run),
            "min_delta": float(params.stage_coupling_min_delta),
            "min_hold_frames": int(params.stage_coupling_min_hold_frames),
            "curve_stats": curve_stats,
            "events": [event],
        }
    )
    return [refined], debug


def _jsonable_float_list(values: np.ndarray) -> List[Any]:
    result: List[Any] = []
    for value in np.asarray(values):
        if np.issubdtype(np.asarray(value).dtype, np.bool_):
            result.append(bool(value))
        elif np.issubdtype(np.asarray(value).dtype, np.integer):
            result.append(int(value))
        else:
            result.append(float(value))
    return result


def _build_actions(
    sig: _DistanceSignals,
    segments: List[Dict[str, Any]],
    *,
    gripper_close_cmd: float,
    gripper_open_cmd: float,
    gripper_hold_cmd: float,
    invalid_cmd_mode: Literal["hold", "open"],
) -> List[Dict[str, Any]]:
    if invalid_cmd_mode not in ("hold", "open"):
        raise ValueError("invalid_cmd_mode must be either 'hold' or 'open'.")
    close_by_index = {
        int(segment["close"])
        for segment in segments
        if segment.get("close") is not None
    }
    open_by_index = {
        int(segment["open"]) for segment in segments if segment.get("open") is not None
    }
    closed = False
    actions: List[Dict[str, Any]] = []
    for index, frame_value in enumerate(sig.frames):
        event: Optional[str] = None
        if index in close_by_index:
            closed = True
            event = "close"
        if index in open_by_index:
            closed = False
            event = "open"
        valid = bool(sig.valid[index])
        state = "hold" if closed else "open"
        reported_closed = closed
        command = float(gripper_close_cmd) if closed else float(gripper_open_cmd)
        if not valid:
            state = "open"
            reported_closed = False
            command = (
                float(gripper_hold_cmd)
                if invalid_cmd_mode == "hold"
                else float(gripper_open_cmd)
            )
        actions.append(
            {
                "frame": int(frame_value),
                "state": state,
                "event": event,
                "grasp": int(reported_closed),
                "gripper_cmd": float(np.float32(command)),
                "valid": valid,
            }
        )
    return actions


def _normalize_close_profile(value: Optional[str]) -> str:
    token = str(value or "default").strip().lower()
    aliases = {
        "default": "default",
        "legacy": "default",
        "safe": "contact_safe",
        "late": "contact_safe",
        "contact_safe": "contact_safe",
        "early": "early_capture",
        "early_capture": "early_capture",
    }
    if token not in aliases:
        raise ValueError(
            "gripper.task_prior.close_timing_profile must be one of "
            "'default', 'contact_safe', or 'early_capture'"
        )
    return aliases[token]


def compute_gripper_actions_task_prior(
    ee_traj: Dict[str, Any],
    obj_traj: Dict[str, Any],
    *,
    task_name: Optional[str] = None,
    env_name: Optional[str] = None,
    uid: Optional[str] = None,
    dataset_config_path: Optional[str] = None,
    prior_config_path: Optional[str] = None,
    num_close: Optional[int] = None,
    num_open: Optional[int] = None,
    method: Method = "3d",
    params: Optional[TaskPriorActionParams] = None,
    params_config_path: Optional[str] = None,
    base_params_config_path: Optional[str] = None,
    ee_key: str = "eef_controller",
    obj_key: str = "obj_visual_center",
    return_debug: bool = True,
    gripper_close_cmd: float = 1.0,
    gripper_open_cmd: float = -1.0,
    gripper_hold_cmd: float = 0.0,
    invalid_cmd_mode: Literal["hold", "open"] = "hold",
    stage_constrained: bool = False,
    close_timing_profile: Optional[str] = None,
) -> Dict[str, Any]:
    prior = resolve_task_prior_spec(
        task_name=task_name,
        env_name=env_name,
        uid=uid,
        dataset_config_path=dataset_config_path,
        prior_config_path=prior_config_path,
        num_close=num_close,
        num_open=num_open,
    )
    if params is None:
        from .numeric_config import load_numeric_action_params
        from .task_prior_config import load_task_prior_action_params

        params = load_task_prior_action_params(
            params_config_path,
            base=load_numeric_action_params(base_params_config_path),
        )
    profile = _normalize_close_profile(
        close_timing_profile
        if close_timing_profile is not None
        else params.close_timing_profile
    )
    effective = params
    if profile == "contact_safe":
        effective = replace(
            params,
            close_timing_profile="contact_safe",
            close_near_quantile=0.38,
            close_min_run=3,
            max_close_gap_after_approach=8,
            confirm_window=max(4, params.confirm_window),
        )
    elif profile == "early_capture":
        effective = replace(
            params,
            close_timing_profile="early_capture",
            close_near_quantile=0.55,
        )

    sig, _, obj_world = _build_distance_signals(
        ee_traj,
        obj_traj,
        method=method,
        params=effective,
        ee_key=ee_key,
        obj_key=obj_key,
    )
    motion_onset = _object_motion_onset(obj_world, effective)
    segments, segment_debug = _build_segments_from_prior(
        sig,
        prior,
        effective,
        object_motion_onset=motion_onset,
    )
    if stage_constrained:
        segments, stage_debug = _apply_stage_coupling(
            segments,
            sig,
            effective,
            obj_traj,
            obj_world,
        )
    else:
        stage_debug = {"enabled": False, "applied": False}

    actions = _build_actions(
        sig,
        segments,
        gripper_close_cmd=gripper_close_cmd,
        gripper_open_cmd=gripper_open_cmd,
        gripper_hold_cmd=gripper_hold_cmd,
        invalid_cmd_mode=invalid_cmd_mode,
    )
    prior_dict = asdict(prior)
    result: Dict[str, Any] = {
        "meta": {
            "algorithm": "task_prior",
            "method": method,
            "T": int(sig.frames.size),
            "ee_key": ee_key,
            "obj_key": obj_key,
            "params": asdict(effective),
            "task_prior": prior_dict,
            "gripper_map": {
                "close_cmd": float(gripper_close_cmd),
                "open_cmd": float(gripper_open_cmd),
                "hold_cmd": float(gripper_hold_cmd),
                "invalid_cmd_mode": invalid_cmd_mode,
            },
            "stage_constrained": bool(stage_constrained),
            "close_timing_profile": profile,
        },
        "segments": segments,
        "actions": actions,
    }
    if bool(return_debug) and bool(params.debug):
        debug: Dict[str, Any] = {
            "prior": prior_dict,
            "dist": np.nan_to_num(
                sig.dist,
                nan=-1.0,
            )
            .astype(float)
            .tolist(),
            "d1": np.nan_to_num(
                sig.d1,
                nan=0.0,
            )
            .astype(float)
            .tolist(),
            "d2": np.nan_to_num(
                sig.d2,
                nan=0.0,
            )
            .astype(float)
            .tolist(),
            "vrel": np.nan_to_num(
                sig.vrel,
                nan=-1.0,
            )
            .astype(float)
            .tolist(),
            "cos": np.nan_to_num(
                sig.cos,
                nan=0.0,
            )
            .astype(float)
            .tolist(),
            "valid": _jsonable_float_list(sig.valid),
            "obj_vis": np.nan_to_num(
                sig.obj_vis,
                nan=-1.0,
            )
            .astype(float)
            .tolist(),
            "contact_mask": _jsonable_float_list(sig.contact_mask),
            "comotion_mask": _jsonable_float_list(sig.comotion_mask),
            "plateau_mask": _jsonable_float_list(sig.plateau_mask),
            "approach_mask": _jsonable_float_list(sig.approach_mask),
            "release_mask": _jsonable_float_list(sig.release_mask),
            "object_motion_onset": motion_onset,
            **segment_debug,
            "stage_coupling": stage_debug,
        }
        result["debug"] = debug
    return result


class TaskPriorActionRecognizer:
    """Small state-free facade for callers that prefer an object API."""

    def __init__(
        self,
        *,
        task_name: Optional[str] = None,
        env_name: Optional[str] = None,
        uid: Optional[str] = None,
        dataset_config_path: Optional[str] = None,
        prior_config_path: Optional[str] = None,
        num_close: Optional[int] = None,
        num_open: Optional[int] = None,
        method: Method = "3d",
        params: Optional[TaskPriorActionParams] = None,
        params_config_path: Optional[str] = None,
        base_params_config_path: Optional[str] = None,
        ee_key: str = "eef_controller",
        obj_key: str = "obj_visual_center",
        return_debug: bool = True,
        gripper_close_cmd: float = 1.0,
        gripper_open_cmd: float = -1.0,
        gripper_hold_cmd: float = 0.0,
        invalid_cmd_mode: Literal["hold", "open"] = "hold",
        stage_constrained: bool = False,
        close_timing_profile: Optional[str] = None,
    ) -> None:
        self._options = {
            "task_name": task_name,
            "env_name": env_name,
            "uid": uid,
            "dataset_config_path": dataset_config_path,
            "prior_config_path": prior_config_path,
            "num_close": num_close,
            "num_open": num_open,
            "method": method,
            "params": params,
            "params_config_path": params_config_path,
            "base_params_config_path": base_params_config_path,
            "ee_key": ee_key,
            "obj_key": obj_key,
            "return_debug": bool(return_debug),
            "gripper_close_cmd": float(gripper_close_cmd),
            "gripper_open_cmd": float(gripper_open_cmd),
            "gripper_hold_cmd": float(gripper_hold_cmd),
            "invalid_cmd_mode": invalid_cmd_mode,
            "stage_constrained": bool(stage_constrained),
            "close_timing_profile": close_timing_profile,
        }

    def infer(
        self,
        *,
        ee_traj: Dict[str, Any],
        obj_traj: Dict[str, Any],
    ) -> Dict[str, Any]:
        return compute_gripper_actions_task_prior(
            ee_traj,
            obj_traj,
            **self._options,
        )


__all__ = [
    "TaskPriorActionParams",
    "TaskPriorActionRecognizer",
    "TaskPriorSpec",
    "compute_gripper_actions_task_prior",
    "resolve_task_prior_spec",
]
