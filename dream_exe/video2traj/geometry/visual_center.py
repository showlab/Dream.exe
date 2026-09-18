"""Geometry for visual-center estimation from tracked and lifted point trajectories.

The current class wrapper only stores parameters before forwarding to these
algorithms. current implementation exposes the computations directly so EEF and object callers can
select their different carry/interpolation policies without a service class.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .camera import Camera


def select_globally_moving_tracks(
    tracks_uv: np.ndarray,
    visibility: Optional[np.ndarray] = None,
    visibility_threshold: float = 0.5,
    min_visible: int = 2,
    topk_fallback: int = 8,
    metric: str = "hybrid",
    sample_count: int = 12,
    min_lag: int = 3,
) -> np.ndarray:
    """Return the current global motion-selection mask for tracked points."""

    frame_count, point_count, _ = tracks_uv.shape
    displacement_score = np.zeros(point_count, dtype=float)
    horizontal = tracks_uv[..., 0]
    vertical = tracks_uv[..., 1]

    for point_id in range(point_count):
        if visibility is None:
            point_visible = np.ones(frame_count, dtype=bool)
        else:
            point_visibility = visibility[:, point_id]
            point_visible = (
                point_visibility > visibility_threshold
                if point_visibility.dtype != np.bool_
                else point_visibility
            )

        visible_indices = np.nonzero(point_visible)[0]
        if visible_indices.size < min_visible:
            displacement_score[point_id] = 0.0
            continue

        visible_u = horizontal[visible_indices, point_id]
        visible_v = vertical[visible_indices, point_id]

        if metric in ("extent", "hybrid"):
            u_q05, u_q95 = np.percentile(visible_u, (5, 95))
            v_q05, v_q95 = np.percentile(visible_v, (5, 95))
            extent = float(np.hypot(u_q95 - u_q05, v_q95 - v_q05))
        else:
            extent = 0.0

        if metric in ("maxpair", "hybrid"):
            selected_count = min(sample_count, visible_indices.size)
            selected = visible_indices[
                np.linspace(
                    0,
                    visible_indices.size - 1,
                    selected_count,
                ).astype(int)
            ]
            points = np.stack(
                [
                    horizontal[selected, point_id],
                    vertical[selected, point_id],
                ],
                axis=1,
            )
            squared_distance = np.sum(
                (points[None, :, :] - points[:, None, :]) ** 2,
                axis=-1,
            )
            lag = np.abs(selected[None, :] - selected[:, None])
            squared_distance[lag < min_lag] = 0.0
            max_pair = (
                float(np.sqrt(squared_distance.max())) if selected_count >= 2 else 0.0
            )
        else:
            max_pair = 0.0

        if metric == "pathlen":
            steps = np.hypot(
                np.diff(visible_u),
                np.diff(visible_v),
            )
            if steps.size:
                p95 = np.percentile(steps, 95.0)
                steps = np.clip(steps, 0.0, p95)
            path_length = float(steps.sum())
        else:
            path_length = 0.0

        if metric == "extent":
            score = extent
        elif metric == "maxpair":
            score = max_pair
        elif metric == "pathlen":
            score = path_length
        elif metric == "hybrid":
            score = 0.7 * extent + 0.3 * max_pair
        else:
            score = extent
        displacement_score[point_id] = score

    median = np.median(displacement_score)
    mad = (
        1.4826 * np.median(np.abs(displacement_score - median))
        if np.any(displacement_score != median)
        else 0.0
    )
    cutoff = max(0.0, median - 2.0 * mad)
    keep = displacement_score > cutoff

    if not np.any(keep):
        fallback_count = min(topk_fallback, point_count)
        fallback_indices = np.argsort(displacement_score)[-fallback_count:]
        keep[fallback_indices] = True
    return keep


def _tukey_biweight(distances: np.ndarray, cutoff: float) -> np.ndarray:
    weights = np.zeros_like(distances)
    inside = distances < cutoff
    ratio = distances[inside] / cutoff
    weights[inside] = (1 - ratio**2) ** 2
    return weights


def estimate_visual_centers(
    point_trajectory: List[Dict[str, Any]],
    globally_kept: np.ndarray,
    camera: Camera,
    carry_previous: bool = True,
    fallback_to_all: bool = True,
    min_points: int = 1,
) -> List[Dict[str, Any]]:
    """Estimate the current robust world/pixel center for every frame."""

    centers: List[Dict[str, Any]] = []
    previous_center = None

    for frame_index, frame_data in enumerate(point_trajectory):
        points_world = np.asarray(frame_data["points_world"], float)
        point_ids = np.asarray(frame_data["point_ids"], int)
        required_points = int(max(1, min_points))

        if points_world.size == 0 or int(len(point_ids)) < required_points:
            if carry_previous and previous_center is not None:
                pixel = camera.world_to_pixel(
                    previous_center,
                    clip_inside=False,
                    return_float=True,
                )
                centers.append(
                    {
                        "frame": frame_index,
                        "vis_center_world": previous_center.tolist(),
                        "vis_center_uv": pixel,
                    }
                )
            else:
                centers.append(
                    {
                        "frame": frame_index,
                        "vis_center_world": [None, None, None],
                        "vis_center_uv": None,
                    }
                )
            continue

        selected = globally_kept[point_ids]
        candidates = points_world[selected]
        if (
            candidates.size == 0
            and fallback_to_all
            and int(len(point_ids)) >= required_points
        ):
            candidates = points_world

        if candidates.size == 0:
            if carry_previous and previous_center is not None:
                pixel = camera.world_to_pixel(
                    previous_center,
                    clip_inside=False,
                    return_float=True,
                )
                centers.append(
                    {
                        "frame": frame_index,
                        "vis_center_world": previous_center.tolist(),
                        "vis_center_uv": pixel,
                    }
                )
            else:
                centers.append(
                    {
                        "frame": frame_index,
                        "vis_center_world": [None, None, None],
                        "vis_center_uv": None,
                    }
                )
            continue

        median_center = np.median(candidates, axis=0)
        distances = np.linalg.norm(
            candidates - median_center[None, :],
            axis=1,
        )
        distance_median = np.median(distances)
        distance_mad = (
            1.4826 * np.median(np.abs(distances - distance_median))
            if np.any(distances != distance_median)
            else 1e-6
        )
        scale = max(
            1e-6,
            distance_median + 2.0 * distance_mad,
        )
        weights = _tukey_biweight(
            distances,
            cutoff=4.685 * scale,
        )
        center = (
            np.average(candidates, axis=0, weights=weights)
            if np.sum(weights) > 1e-8
            else median_center
        )

        previous_center = center.copy()
        pixel = camera.world_to_pixel(
            center,
            clip_inside=False,
            return_float=True,
        )
        centers.append(
            {
                "frame": frame_index,
                "vis_center_world": center.tolist(),
                "vis_center_uv": pixel,
            }
        )
    return centers


def interpolate_visual_centers(
    centers: List[Dict[str, Any]],
    camera: Optional[Camera] = None,
    max_gap: int = 10,
    fill_ends: bool = False,
    end_max: int = 3,
) -> List[Dict[str, Any]]:
    """Apply the current bounded linear interpolation policy."""

    values: List[Optional[np.ndarray]] = []
    valid_indices: List[int] = []

    for index, center in enumerate(centers):
        center_world = center.get("vis_center_world")
        if center_world and None not in center_world:
            values.append(np.asarray(center_world, float))
            valid_indices.append(index)
        else:
            values.append(None)

    if len(valid_indices) < 2:
        return centers

    if fill_ends:
        first_valid = valid_indices[0]
        if first_valid > 0 and first_valid <= end_max:
            for index in range(0, first_valid):
                values[index] = values[first_valid].copy()

        last_valid = valid_indices[-1]
        trailing_count = len(values) - 1 - last_valid
        if last_valid < len(values) - 1 and trailing_count <= end_max:
            for index in range(last_valid + 1, len(values)):
                values[index] = values[last_valid].copy()

    for left, right in zip(valid_indices[:-1], valid_indices[1:]):
        gap = right - left
        if gap <= 1 or gap - 1 > max_gap:
            continue
        left_value, right_value = values[left], values[right]
        for offset in range(1, gap):
            alpha = offset / gap
            values[left + offset] = (1 - alpha) * left_value + alpha * right_value

    output: List[Dict[str, Any]] = []
    for index, center in enumerate(centers):
        output_center = dict(center)
        if values[index] is not None:
            output_center["vis_center_world"] = values[index].tolist()
            if camera is not None:
                output_center["vis_center_uv"] = camera.world_to_pixel(
                    values[index],
                    clip_inside=False,
                    return_float=True,
                )
        output.append(output_center)

    missing_before = sum(
        1
        for center in centers
        if (not center.get("vis_center_world") or None in center["vis_center_world"])
    )
    missing_after = sum(
        1
        for center in output
        if (not center.get("vis_center_world") or None in center["vis_center_world"])
    )
    print(
        "[interp] missing centers: "
        f"{missing_before} → {missing_after} "
        f"(max_gap={max_gap}, fill_ends={fill_ends})"
    )
    return output


def compute_visual_center_trajectory(
    tracks_uv: np.ndarray,
    visibility: Optional[np.ndarray],
    point_trajectory: List[Dict[str, Any]],
    camera: Camera,
    *,
    visibility_threshold: float = 0.5,
    interpolate: bool = True,
    max_gap: int = 10,
    fill_ends: bool = False,
    end_max: int = 3,
    carry_previous: bool = True,
    min_points: int = 1,
) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Compose current track selection, robust centers, and interpolation."""

    normalized_visibility_threshold = float(visibility_threshold)
    normalized_interpolate = bool(interpolate)
    normalized_max_gap = int(max_gap)
    normalized_fill_ends = bool(fill_ends)
    normalized_end_max = int(end_max)
    normalized_carry_previous = bool(carry_previous)
    normalized_min_points = int(max(1, min_points))

    globally_kept = select_globally_moving_tracks(
        tracks_uv,
        visibility=visibility,
        visibility_threshold=normalized_visibility_threshold,
    )
    centers = estimate_visual_centers(
        point_trajectory,
        globally_kept,
        camera,
        carry_previous=normalized_carry_previous,
        min_points=normalized_min_points,
    )
    if normalized_interpolate:
        centers = interpolate_visual_centers(
            centers,
            camera=camera,
            max_gap=normalized_max_gap,
            fill_ends=normalized_fill_ends,
            end_max=normalized_end_max,
        )
    return centers, globally_kept


__all__ = [
    "compute_visual_center_trajectory",
    "estimate_visual_centers",
    "interpolate_visual_centers",
    "select_globally_moving_tracks",
]
