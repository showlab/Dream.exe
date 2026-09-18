"""Dynamic target calibration used by EEF trajectory lifting.

This module is a pure numeric policy.  It neither selects a depth estimator nor
owns artifact paths, stage compilation, or lift geometry.
"""

from __future__ import annotations

import copy
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .calibration import solve_depth_calibration


DYNAMIC_AFFINE_LIFT_SCHEMA = "dream-exe.dynamic-affine-lift"
DYNAMIC_SHIFT_LIFT_SCHEMA = "dream-exe.dynamic-shift-lift"
_COMMON_POLICY_FIELDS = frozenset(
    {
        "switch_strategy",
        "proximity_threshold",
        "consecutive_frames",
        "min_region_pixels",
        "max_pre_switch_motion_norm",
        "ramp_frames",
    }
)
_AFFINE_POLICY_FIELDS = _COMMON_POLICY_FIELDS
_SHIFT_POLICY_FIELDS = frozenset({*_COMMON_POLICY_FIELDS, "max_abs_delta"})


def _depth_stack(value: Any, *, label: str) -> np.ndarray:
    array = np.asarray(value)
    if array.ndim != 3 or any(int(size) <= 0 for size in array.shape):
        raise ValueError(f"{label} must be non-empty [T,H,W], got {array.shape}")
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{label} must contain real numeric values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _tracking_arrays(
    tracks: Any,
    visibility: Any,
    *,
    frame_count: int,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(tracks, dtype=np.float32)
    visible = np.asarray(visibility)
    if points.ndim == 4 and points.shape[0] == 1:
        points = points[0]
    if visible.ndim == 3 and visible.shape[0] == 1:
        visible = visible[0]
    if points.ndim != 3 or points.shape[-1] != 2:
        raise ValueError(f"{label}.tracks_uv must be [T,N,2], got {points.shape}")
    if visible.ndim != 2 or visible.shape != points.shape[:2]:
        raise ValueError(
            f"{label}.visibility must align [T,N] with tracks, "
            f"got {visible.shape} and {points.shape}"
        )
    if int(points.shape[0]) != int(frame_count) or int(points.shape[1]) <= 0:
        raise ValueError(
            f"{label} tracking must contain {frame_count} frames and points"
        )
    return points, visible


def _visible_points(
    tracks: np.ndarray,
    visibility: np.ndarray,
    frame: int,
) -> np.ndarray:
    selected = (
        visibility[frame] if visibility.dtype == np.bool_ else visibility[frame] > 0.5
    )
    points = tracks[frame][np.asarray(selected, dtype=bool)]
    return points[np.all(np.isfinite(points), axis=1)]


def _median_center(points: np.ndarray) -> np.ndarray | None:
    if points.ndim != 2 or int(points.shape[0]) == 0:
        return None
    return np.median(points, axis=0).astype(np.float32)


def _point_bbox(
    points: np.ndarray,
    *,
    height: int,
    width: int,
    padding: int,
) -> tuple[int, int, int, int] | None:
    if points.ndim != 2 or int(points.shape[0]) == 0:
        return None
    x0 = max(0, int(math.floor(float(np.min(points[:, 0])))) - padding)
    y0 = max(0, int(math.floor(float(np.min(points[:, 1])))) - padding)
    x1 = min(
        width - 1,
        int(math.ceil(float(np.max(points[:, 0])))) + padding,
    )
    y1 = min(
        height - 1,
        int(math.ceil(float(np.max(points[:, 1])))) + padding,
    )
    return None if x1 < x0 or y1 < y0 else (x0, y0, x1, y1)


def _bbox_mask(
    bbox: tuple[int, int, int, int] | None,
    *,
    shape: tuple[int, int],
    erode_three_by_three: bool = False,
) -> np.ndarray:
    mask = np.zeros(shape, dtype=bool)
    if bbox is None:
        return mask
    x0, y0, x1, y1 = bbox
    if erode_three_by_three:
        x0, y0, x1, y1 = x0 + 1, y0 + 1, x1 - 1, y1 - 1
    if x0 <= x1 and y0 <= y1:
        mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def _normalized_distance(
    first: np.ndarray,
    second: np.ndarray,
    bbox: tuple[int, int, int, int],
) -> float:
    x0, y0, x1, y1 = bbox
    scale = max(float(x1 - x0 + 1), float(y1 - y0 + 1), 1.0)
    return float(np.linalg.norm(first - second) / scale)


def _switch_frame(
    distances: Sequence[float],
    *,
    start: int,
    threshold: float,
    consecutive_frames: int,
) -> tuple[int | None, float | None]:
    run_start: int | None = None
    run_length = 0
    best = math.inf
    for frame in range(max(0, int(start)), len(distances)):
        distance = float(distances[frame])
        if math.isfinite(distance):
            best = min(best, distance)
        if math.isfinite(distance) and distance < threshold:
            if run_length == 0:
                run_start = frame
            run_length += 1
            if run_length >= consecutive_frames:
                return run_start, distance
        else:
            run_start = None
            run_length = 0
    return None, (best if math.isfinite(best) else None)


def _stage_identity(
    stage: Mapping[str, Any],
    *,
    index: int,
    mode: str,
) -> dict[str, str]:
    stage_id = str(stage.get("stage_id", "") or f"s{index + 1}")
    object_id = str(stage.get("object_id", "") or "")
    runtime_key = str(stage.get("runtime_object_key", "") or "")
    if not object_id and not runtime_key:
        raise ValueError(
            f"{mode} stage {stage_id!r} requires object_id or runtime_object_key"
        )
    return {
        "stage_id": stage_id,
        "object_id": object_id,
        "runtime_object_key": runtime_key,
    }


def _apply_dynamic_lift_calibration(
    *,
    mode: str,
    static_eef_depths: Any,
    raw_depths: Any | None,
    init_reference_depth: Any,
    eef_tracks_uv: Any,
    eef_visibility: Any,
    stages: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    calibration_solver: str | None,
) -> dict[str, Any]:
    """Apply one stage-local dynamic calibration policy.

    Each stage mapping supplies ``stage_id``, an object identity,
    ``tracks_uv``, ``visibility``, and its first-frame ``interaction_mask``.
    Frames before a detected switch remain byte-for-byte equal to the static
    EEF target map.  The returned mapping contains no paths and performs no I/O.
    """

    if mode not in {"dynamic_shift", "dynamic_affine"}:
        raise ValueError(f"unsupported dynamic lift mode: {mode}")
    if not isinstance(config, Mapping):
        raise TypeError(f"{mode} config must be a mapping")
    policy = copy.deepcopy(dict(config))
    policy_fields = (
        _SHIFT_POLICY_FIELDS if mode == "dynamic_shift" else _AFFINE_POLICY_FIELDS
    )
    unknown = sorted(set(policy).difference(policy_fields))
    if unknown:
        raise ValueError(
            f"{mode} config contains unknown fields: " + ", ".join(unknown)
        )
    missing = sorted(policy_fields.difference(policy))
    if missing:
        raise ValueError(f"{mode} config is missing fields: " + ", ".join(missing))
    static = _depth_stack(static_eef_depths, label="static_eef_depths")
    raw: np.ndarray | None = None
    if mode == "dynamic_affine":
        raw = _depth_stack(raw_depths, label="raw_depths")
        if raw.shape != static.shape:
            raise ValueError(
                "raw_depths must align with static_eef_depths: "
                f"{raw.shape} != {static.shape}"
            )
    frame_count, height, width = (int(value) for value in static.shape)
    reference = np.asarray(init_reference_depth, dtype=np.float32)
    if reference.shape != (height, width):
        raise ValueError(
            "init_reference_depth must match depth frames: "
            f"{reference.shape} != {(height, width)}"
        )
    eef_tracks, eef_vis = _tracking_arrays(
        eef_tracks_uv,
        eef_visibility,
        frame_count=frame_count,
        label="eef",
    )
    if not isinstance(stages, Sequence) or isinstance(stages, (str, bytes)):
        raise TypeError("stages must be an ordered sequence of mappings")

    if not isinstance(policy.get("switch_strategy"), str):
        raise TypeError(f"{mode} switch_strategy must be a string")
    if policy["switch_strategy"] != "proximity_2d":
        raise ValueError(f"{mode} supports switch_strategy='proximity_2d'")
    integer_fields = (
        "consecutive_frames",
        "min_region_pixels",
        "ramp_frames",
    )
    numeric_fields = (
        "proximity_threshold",
        "max_pre_switch_motion_norm",
    ) + (("max_abs_delta",) if mode == "dynamic_shift" else ())
    for key in integer_fields:
        if type(policy[key]) is not int:
            raise TypeError(f"{mode} {key} must be an integer")
    for key in numeric_fields:
        if isinstance(policy[key], bool) or not isinstance(
            policy[key],
            (int, float),
        ):
            raise TypeError(f"{mode} {key} must be a finite number")
    threshold = float(policy["proximity_threshold"])
    consecutive = int(policy["consecutive_frames"])
    min_pixels = int(policy["min_region_pixels"])
    motion_limit = float(policy["max_pre_switch_motion_norm"])
    ramp_frames = int(policy["ramp_frames"])
    max_abs_delta = (
        float(policy["max_abs_delta"]) if mode == "dynamic_shift" else math.inf
    )
    if (
        not math.isfinite(threshold)
        or not math.isfinite(motion_limit)
        or (mode == "dynamic_shift" and not math.isfinite(max_abs_delta))
        or threshold <= 0.0
        or consecutive <= 0
        or min_pixels <= 0
        or motion_limit < 0.0
        or max_abs_delta < 0.0
        or ramp_frames <= 0
    ):
        raise ValueError(f"{mode} config contains invalid bounds")

    records: list[dict[str, Any]] = []
    applied: list[dict[str, Any]] = []
    search_start = 0
    for index, raw_stage in enumerate(stages):
        if not isinstance(raw_stage, Mapping):
            raise TypeError(f"{mode} stages[{index}] must be a mapping")
        stage = dict(raw_stage)
        identity = _stage_identity(stage, index=index, mode=mode)
        object_tracks, object_vis = _tracking_arrays(
            stage.get("tracks_uv"),
            stage.get("visibility"),
            frame_count=frame_count,
            label=f"stage {identity['stage_id']}",
        )
        interaction_mask = np.asarray(
            stage.get("interaction_mask"),
            dtype=bool,
        )
        if interaction_mask.shape != (height, width):
            raise ValueError(
                f"stage {identity['stage_id']!r} interaction_mask must "
                f"be {(height, width)}, got {interaction_mask.shape}"
            )
        record: dict[str, Any] = {
            **identity,
            "stage_start": int(search_start),
            "stage_end": int(frame_count - 1),
            "switch_strategy": "proximity_2d",
            "t_switch": None,
            "dist_norm": None,
            "valid_region_pixels": 0,
            "delta_ir": None,
            "ramp_frames": ramp_frames,
            "fallback": True,
            "fallback_reason": "",
        }

        distances = [math.nan] * frame_count
        object_bboxes: list[tuple[int, int, int, int] | None] = []
        eef_bboxes: list[tuple[int, int, int, int] | None] = []
        for frame in range(frame_count):
            eef_points = _visible_points(eef_tracks, eef_vis, frame)
            object_points = _visible_points(
                object_tracks,
                object_vis,
                frame,
            )
            eef_center = _median_center(eef_points)
            object_center = _median_center(object_points)
            object_bbox = _point_bbox(
                object_points,
                height=height,
                width=width,
                padding=4,
            )
            eef_bbox = _point_bbox(
                eef_points,
                height=height,
                width=width,
                padding=2,
            )
            object_bboxes.append(object_bbox)
            eef_bboxes.append(eef_bbox)
            if (
                eef_center is not None
                and object_center is not None
                and object_bbox is not None
            ):
                distances[frame] = _normalized_distance(
                    eef_center,
                    object_center,
                    object_bbox,
                )

        switch, confirmed_distance = _switch_frame(
            distances,
            start=search_start,
            threshold=threshold,
            consecutive_frames=consecutive,
        )
        record["dist_norm"] = confirmed_distance
        if switch is None:
            record["fallback_reason"] = "t_switch_unavailable"
            records.append(record)
            continue
        record["t_switch"] = int(switch)

        object_initial = _median_center(_visible_points(object_tracks, object_vis, 0))
        object_switch = _median_center(
            _visible_points(object_tracks, object_vis, switch)
        )
        switch_bbox = object_bboxes[switch]
        if (
            object_initial is not None
            and object_switch is not None
            and switch_bbox is not None
        ):
            pre_motion = _normalized_distance(
                object_initial,
                object_switch,
                switch_bbox,
            )
            record["pre_switch_motion_norm"] = pre_motion
            if pre_motion > motion_limit:
                record["fallback_reason"] = (
                    f"interaction_region_moved_before_switch:{pre_motion:.3f}"
                )
                records.append(record)
                continue

        region = interaction_mask & _bbox_mask(
            switch_bbox,
            shape=(height, width),
            erode_three_by_three=True,
        )
        without_eef = region & ~_bbox_mask(
            eef_bboxes[switch],
            shape=(height, width),
        )
        if int(np.sum(without_eef)) >= min_pixels:
            region = without_eef
            record["eef_overlap_excluded"] = True
        else:
            record["eef_overlap_excluded"] = False
        region &= np.isfinite(reference) & np.isfinite(static[switch])
        pixels = int(np.sum(region))
        record["valid_region_pixels"] = pixels
        if pixels < min_pixels:
            record["fallback_reason"] = f"insufficient_region_pixels:{pixels}"
            records.append(record)
            continue

        delta = float(np.median(reference[region] - static[switch][region]))
        record["delta_ir"] = delta if math.isfinite(delta) else None
        if mode == "dynamic_shift":
            if not math.isfinite(delta):
                record["fallback_reason"] = "delta_invalid"
                records.append(record)
                continue
            if abs(delta) > max_abs_delta:
                record["fallback_reason"] = f"delta_exceeds_limit:{delta:.6f}"
                records.append(record)
                continue
            record["fallback"] = False
        else:
            assert raw is not None
            try:
                _, parameters = solve_depth_calibration(
                    raw[switch],
                    reference,
                    valid_mask=region,
                    calibration_solver=str(calibration_solver),
                )
                scale = float(parameters["s"])
                bias = float(parameters["b"])
                if not math.isfinite(scale) or not math.isfinite(bias):
                    raise ValueError("affine parameters are not finite")
            except Exception as error:
                record["fallback_reason"] = f"dynamic_affine_fit_failed:{error}"
                records.append(record)
                continue
            record.update(
                {
                    "affine_s": scale,
                    "affine_b": bias,
                    "calibration_solver": str(parameters["calibration_solver"]),
                    "fallback": False,
                }
            )
        records.append(record)
        applied.append(record)
        search_start = min(frame_count - 1, switch + 1)

    ordered = sorted(
        applied,
        key=lambda item: int(item["t_switch"]),
    )
    for current, following in zip(ordered, ordered[1:]):
        current["stage_end"] = max(
            int(current["t_switch"]),
            int(following["t_switch"]) - 1,
        )

    output = static.copy()
    for record in ordered:
        replacement: np.ndarray | None = None
        if mode == "dynamic_affine":
            assert raw is not None
            replacement = (
                float(record["affine_s"]) * raw + float(record["affine_b"])
            ).astype(np.float32)
        start = int(record["t_switch"])
        end = int(record["stage_end"])
        for frame in range(start, end + 1):
            alpha = (
                1.0
                if ramp_frames == 1
                else min(
                    1.0,
                    max(
                        0.0,
                        (frame - start) / float(ramp_frames - 1),
                    ),
                )
            )
            if mode == "dynamic_shift":
                output[frame] = output[frame] + (float(record["delta_ir"]) * alpha)
            else:
                assert replacement is not None
                output[frame] = (1.0 - alpha) * output[frame] + alpha * replacement[
                    frame
                ]

    metadata = {
        "schema": (
            DYNAMIC_SHIFT_LIFT_SCHEMA
            if mode == "dynamic_shift"
            else DYNAMIC_AFFINE_LIFT_SCHEMA
        ),
        "enabled": True,
        "mode": mode,
        "policy": policy,
        "stages": records,
        "applied": bool(ordered),
        "reason": "" if ordered else "no_stage_applied",
    }
    if mode == "dynamic_affine":
        metadata["calibration_solver"] = str(calibration_solver)
    return {
        "eef_depths": output,
        "meta": metadata,
    }


def apply_dynamic_affine_lift_calibration(
    *,
    static_eef_depths: Any,
    raw_depths: Any,
    init_reference_depth: Any,
    eef_tracks_uv: Any,
    eef_visibility: Any,
    stages: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    calibration_solver: str,
) -> dict[str, Any]:
    """Apply stage-local affine calibration after an interaction switch."""

    return _apply_dynamic_lift_calibration(
        mode="dynamic_affine",
        static_eef_depths=static_eef_depths,
        raw_depths=raw_depths,
        init_reference_depth=init_reference_depth,
        eef_tracks_uv=eef_tracks_uv,
        eef_visibility=eef_visibility,
        stages=stages,
        config=config,
        calibration_solver=calibration_solver,
    )


def apply_dynamic_shift_lift_calibration(
    *,
    static_eef_depths: Any,
    init_reference_depth: Any,
    eef_tracks_uv: Any,
    eef_visibility: Any,
    stages: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply a bounded stage-local depth shift after object interaction."""

    return _apply_dynamic_lift_calibration(
        mode="dynamic_shift",
        static_eef_depths=static_eef_depths,
        raw_depths=None,
        init_reference_depth=init_reference_depth,
        eef_tracks_uv=eef_tracks_uv,
        eef_visibility=eef_visibility,
        stages=stages,
        config=config,
        calibration_solver=None,
    )


__all__ = [
    "DYNAMIC_AFFINE_LIFT_SCHEMA",
    "DYNAMIC_SHIFT_LIFT_SCHEMA",
    "apply_dynamic_affine_lift_calibration",
    "apply_dynamic_shift_lift_calibration",
]
