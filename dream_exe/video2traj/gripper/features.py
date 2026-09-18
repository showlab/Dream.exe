"""Numerical feature extraction for environment-independent gripper inference."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional, Tuple

import numpy as np


Method = Literal["2d", "3d", "fused"]


@dataclass
class GraspParams:
    win_radius: int = 3
    vel_delta: int = 2
    min_speed_2d: float = 0.6
    min_speed_3d: float = 0.002
    tau_near_2d: float = 35.0
    tau_hold_2d: float = 28.0
    tau_near_3d: float = 0.06
    tau_hold_3d: float = 0.035
    tau_cos_hold: float = 0.75
    tau_cos_detach: float = 0.35
    tau_vrel_hold_2d: float = 3.5
    tau_vrel_hold_3d: float = 0.01
    bridge_gap_near: int = 2
    min_run_near: int = 5
    bridge_gap_attached: int = 3
    min_on: int = 6
    min_off: int = 6
    refine_radius: int = 3
    close_backshift: int = 1
    w_cos: float = 1.0
    w_vrel: float = 0.6
    w_dist: float = 0.4
    prefer_3d_in_fused: bool = True
    min_detach_run: int = 6
    detach_dist_ratio_soft: float = 1.15
    detach_dist_ratio_hi: float = 2.0
    detach_vrel_ratio: float = 1.5
    detach_use_cos: bool = True
    enable_max_delta_fallback: bool = True
    max_delta_window: int = 8
    fused_open_need_both: bool = False
    obj_vis_th: float = 0.55
    obj_vis_min_run: int = 4
    obj_vis_as_soft_detach: bool = True
    debug: bool = True


@dataclass
class Features:
    D_win: np.ndarray
    VREL_win: np.ndarray
    COS_win: np.ndarray
    valid: np.ndarray
    D: Optional[np.ndarray] = None
    VREL: Optional[np.ndarray] = None
    COS: Optional[np.ndarray] = None


def _norm2(x: np.ndarray) -> np.ndarray:
    values = np.asarray(x)
    return np.sqrt(np.sum(values * values, axis=1))


def _central_velocity(p: np.ndarray, delta: int) -> np.ndarray:
    points = np.asarray(p)
    length, dimensions = points.shape
    velocity = np.full(
        (length, dimensions),
        np.nan,
        dtype=np.float32,
    )
    stride = max(1, int(delta))
    if length > 2 * stride:
        interior = (points[2 * stride :] - points[: -2 * stride]).astype(np.float32)
        interior[~np.all(np.isfinite(interior), axis=1)] = np.nan
        velocity[stride : length - stride] = interior
    return velocity


def _cosine(
    u: np.ndarray,
    v: np.ndarray,
    eps: float = 1e-6,
) -> np.ndarray:
    left = np.asarray(u)
    right = np.asarray(v)
    left_norm = _norm2(left)
    right_norm = _norm2(right)
    dot = np.sum(left * right, axis=1)
    denominator = left_norm * right_norm + float(eps)
    cosine = (dot / denominator).astype(np.float32)
    too_small = (left_norm < eps) | (right_norm < eps)
    cosine[too_small] = np.nan
    return np.clip(cosine, -1.0, 1.0)


def _window_median(x: np.ndarray, R: int) -> np.ndarray:
    values = np.asarray(x)
    length = int(values.shape[0])
    radius = int(R)
    if radius <= 0:
        return values.astype(np.float32).copy()

    result = np.full((length,), np.nan, dtype=np.float32)
    for index in range(length):
        start = max(0, index - radius)
        stop = min(length, index + radius + 1)
        window = values[start:stop]
        finite = window[np.isfinite(window)]
        if finite.size:
            result[index] = np.median(finite)
    return result


def _segments_from_bool(x: np.ndarray) -> List[Tuple[int, int]]:
    mask = np.asarray(x, dtype=bool)
    if mask.size == 0:
        return []

    transitions = np.diff(
        np.concatenate(
            (
                np.asarray([False], dtype=bool),
                mask,
                np.asarray([False], dtype=bool),
            )
        ).astype(np.int8)
    )
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1)
    return [(int(start), int(stop - 1)) for start, stop in zip(starts, stops)]


def _fill_short_false_gaps(
    x: np.ndarray,
    max_gap: int,
) -> np.ndarray:
    result = np.asarray(x, dtype=bool).copy()
    limit = int(max_gap)
    if result.size == 0 or limit <= 0:
        return result

    for start, stop in _segments_from_bool(~result):
        bounded = start > 0 and stop < result.size - 1
        if bounded and stop - start + 1 <= limit:
            result[start : stop + 1] = True
    return result


def _min_run_filter(
    x: np.ndarray,
    min_true: int,
    min_false: int,
) -> np.ndarray:
    del min_true
    result = np.asarray(x, dtype=bool).copy()
    false_threshold = int(min_false)
    if result.size == 0 or false_threshold <= 0:
        return result

    for start, stop in _segments_from_bool(~result):
        if stop - start + 1 < false_threshold:
            result[start : stop + 1] = True
    return result


def _first_sustained_true(
    mask: np.ndarray,
    start: int,
    min_run: int,
) -> Optional[int]:
    values = np.asarray(mask, dtype=bool)
    begin = max(0, int(start))
    required = max(1, int(min_run))
    last = int(values.size) - required
    for index in range(begin, last + 1):
        if bool(np.all(values[index : index + required])):
            return index
    return None


def _argmax_delta_after(
    D: np.ndarray,
    start: int,
    window: int,
) -> Optional[int]:
    values = np.asarray(D)
    begin = max(0, int(start))
    stride = max(1, int(window))
    stop = int(values.shape[0]) - stride
    if begin >= stop:
        return None

    deltas = values[begin + stride :] - values[begin:stop]
    finite = np.isfinite(deltas)
    if not bool(np.any(finite)):
        return None
    scores = np.where(finite, deltas, -np.inf)
    return begin + int(np.argmax(scores))


def _clip01(x: np.ndarray) -> np.ndarray:
    return np.minimum(1.0, np.maximum(0.0, np.asarray(x)))


def _to_np_xy(
    rec: Dict[str, Any],
    key: str,
) -> Optional[np.ndarray]:
    value = rec.get(key)
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    selected = value[:2]
    if any(item is None for item in selected):
        return None
    return np.asarray(selected, dtype=np.float32)


def _to_np_xyz(
    rec: Dict[str, Any],
    key: str,
) -> Optional[np.ndarray]:
    value = rec.get(key)
    if not isinstance(value, (list, tuple)) or len(value) < 3:
        return None
    selected = value[:3]
    if any(item is None for item in selected):
        return None
    return np.asarray(selected, dtype=np.float32)


def pack_action_trajectory_arrays(
    ee_traj: Dict[str, Any],
    obj_traj: Dict[str, Any],
    *,
    ee_key: str,
    obj_key: str,
) -> Tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    ee_records = ee_traj.get(ee_key)
    if ee_records is None:
        raise KeyError(f"ee_traj missing key={ee_key!r}")
    obj_records = obj_traj.get(obj_key)
    if obj_records is None:
        raise KeyError(f"obj_traj missing key={obj_key!r}")

    length = min(len(ee_records), len(obj_records))
    if length <= 0:
        raise RuntimeError("Empty trajectory for gripper action extraction.")

    frames = np.empty((length,), dtype=np.int32)
    eef_uv = np.full((length, 2), np.nan, dtype=np.float32)
    obj_uv = np.full((length, 2), np.nan, dtype=np.float32)
    eef_world = np.full((length, 3), np.nan, dtype=np.float32)
    obj_world = np.full((length, 3), np.nan, dtype=np.float32)
    obj_visibility = np.ones((length,), dtype=np.float32)
    obj_visibility_valid = np.zeros((length,), dtype=bool)

    for index in range(length):
        ee_record = ee_records[index]
        obj_record = obj_records[index]
        frames[index] = int(ee_record.get("frame", index))

        value = _to_np_xy(ee_record, "pos_uv")
        if value is not None:
            eef_uv[index] = value
        value = _to_np_xy(obj_record, "pos_uv")
        if value is not None:
            obj_uv[index] = value
        value = _to_np_xyz(ee_record, "pos_world")
        if value is not None:
            eef_world[index] = value
        value = _to_np_xyz(obj_record, "pos_world")
        if value is not None:
            obj_world[index] = value

        visibility = obj_record.get("vis")
        if visibility is not None and bool(np.isfinite(visibility)):
            obj_visibility[index] = float(visibility)
            obj_visibility_valid[index] = True

    return (
        frames,
        eef_uv,
        obj_uv,
        eef_world,
        obj_world,
        obj_visibility,
        obj_visibility_valid,
    )


def _build_features(
    eef: np.ndarray,
    obj: np.ndarray,
    params: GraspParams,
    *,
    min_speed: float,
) -> Features:
    eef_values = np.asarray(eef)
    obj_values = np.asarray(obj)
    distance = _norm2(obj_values - eef_values)

    eef_velocity = _central_velocity(
        eef_values,
        params.vel_delta,
    )
    obj_velocity = _central_velocity(
        obj_values,
        params.vel_delta,
    )
    relative_velocity = _norm2(obj_velocity - eef_velocity)
    cosine = _cosine(eef_velocity, obj_velocity)
    slow = (_norm2(eef_velocity) < float(min_speed)) | (
        _norm2(obj_velocity) < float(min_speed)
    )
    cosine[slow] = np.nan

    distance_window = _window_median(
        distance,
        params.win_radius,
    )
    relative_window = _window_median(
        relative_velocity,
        params.win_radius,
    )
    cosine_window = _window_median(
        cosine,
        params.win_radius,
    )
    valid = np.isfinite(distance_window) & np.isfinite(relative_window)
    return Features(
        D_win=distance_window,
        VREL_win=relative_window,
        COS_win=cosine_window,
        valid=valid,
        D=distance,
        VREL=relative_velocity,
        COS=cosine,
    )


def _build_features_2d(
    eef_uv: np.ndarray,
    obj_uv: np.ndarray,
    params: GraspParams,
) -> Features:
    return _build_features(
        eef_uv,
        obj_uv,
        params,
        min_speed=params.min_speed_2d,
    )


def _build_features_3d(
    eef_w: np.ndarray,
    obj_w: np.ndarray,
    params: GraspParams,
) -> Features:
    return _build_features(
        eef_w,
        obj_w,
        params,
        min_speed=params.min_speed_3d,
    )


def _fuse_features(
    f3: Features,
    f2: Features,
    params: GraspParams,
) -> Features:
    if params.prefer_3d_in_fused:
        use_3d = np.asarray(f3.valid, dtype=bool)
    else:
        use_3d = np.zeros_like(np.asarray(f2.valid, dtype=bool))

    def select(
        three_d: np.ndarray,
        two_d: np.ndarray,
        *,
        require_finite_3d: bool = False,
    ) -> np.ndarray:
        selector = use_3d
        if require_finite_3d:
            selector = selector & np.isfinite(three_d)
        return np.where(selector, three_d, two_d).astype(np.float32)

    return Features(
        D_win=select(f3.D_win, f2.D_win),
        VREL_win=select(f3.VREL_win, f2.VREL_win),
        COS_win=select(
            f3.COS_win,
            f2.COS_win,
            require_finite_3d=True,
        ),
        valid=(np.asarray(f3.valid, dtype=bool) | np.asarray(f2.valid, dtype=bool)),
    )
