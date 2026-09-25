"""Pure depth calibration primitives.

The functions in this module operate only on caller-owned arrays.  They do
not resolve repository data, initialize a model or simulator, or publish
artifacts.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation

from .config import normalize_depth_base_config


_MIN_DEPTH = 1.0e-6


def bilateral_smooth_in_roi(
    depth: np.ndarray, roi_mask: np.ndarray, *, sigma_r: float = 0.02,
) -> np.ndarray:
    """Preserve the pre-refactor ROI filter, including invalid-pixel filling."""
    import cv2

    out = np.asarray(depth, dtype=np.float32).copy()
    roi_mask = np.asarray(roi_mask, dtype=bool)
    bad = ~np.isfinite(out) | (out <= 0)
    bad_in_roi = bad & roi_mask
    if np.any(bad_in_roi):
        valid = roi_mask & ~bad
        out[bad_in_roi] = float(np.median(out[valid])) if np.any(valid) else 1.0
    out = np.ascontiguousarray(out, dtype=np.float32)
    denom = float(max(1e-6, sigma_r))
    scaled = np.ascontiguousarray((out / denom).clip(0, 65535), dtype=np.float32)
    try:
        filtered = cv2.bilateralFilter(scaled, d=5, sigmaColor=10.0, sigmaSpace=5.0)
        out[roi_mask] = (filtered * denom)[roi_mask]
    except Exception:
        # The historical filter returned the filled input on OpenCV failure.
        pass
    return out.astype(np.float32)


def _ellipse_footprint(radius: int) -> np.ndarray:
    """Return the integer ellipse used for pixel-radius expansion."""

    radius = max(0, int(radius))
    if radius == 0:
        return np.ones((1, 1), dtype=bool)
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    half_widths = np.rint(
        radius
        * np.sqrt(
            np.maximum(
                0.0,
                1.0 - (offsets / float(radius)) ** 2,
            )
        )
    ).astype(np.int64)
    footprint = np.zeros(
        (2 * radius + 1, 2 * radius + 1),
        dtype=bool,
    )
    for row, half_width in enumerate(half_widths.tolist()):
        footprint[
            row,
            radius - half_width : radius + half_width + 1,
        ] = True
    return footprint


def _convex_boundary(points: np.ndarray) -> list[tuple[int, int]]:
    """Compute a counter-clockwise integer convex boundary."""

    vertices = sorted({(int(point[0]), int(point[1])) for point in np.asarray(points)})
    if len(vertices) <= 1:
        return vertices

    def turn(
        origin: tuple[int, int],
        first: tuple[int, int],
        second: tuple[int, int],
    ) -> int:
        return (first[0] - origin[0]) * (second[1] - origin[1]) - (
            first[1] - origin[1]
        ) * (second[0] - origin[0])

    lower: list[tuple[int, int]] = []
    for vertex in vertices:
        while len(lower) >= 2 and turn(lower[-2], lower[-1], vertex) <= 0:
            lower.pop()
        lower.append(vertex)
    upper: list[tuple[int, int]] = []
    for vertex in reversed(vertices):
        while len(upper) >= 2 and turn(upper[-2], upper[-1], vertex) <= 0:
            upper.pop()
        upper.append(vertex)
    return lower[:-1] + upper[:-1]


def _fill_convex_boundary(
    height: int,
    width: int,
    boundary: Sequence[tuple[int, int]],
) -> np.ndarray:
    """Rasterize a convex integer polygon at integer pixel centers."""

    if len(boundary) < 3:
        return np.ones((height, width), dtype=bool)
    rows, columns = np.indices((height, width), dtype=np.int64)
    nonnegative = np.ones((height, width), dtype=bool)
    nonpositive = np.ones((height, width), dtype=bool)
    vertices = list(boundary)
    for first, second in zip(
        vertices,
        vertices[1:] + vertices[:1],
    ):
        cross = (second[0] - first[0]) * (rows - first[1]) - (second[1] - first[1]) * (
            columns - first[0]
        )
        nonnegative &= cross >= 0
        nonpositive &= cross <= 0
    filled = nonnegative | nonpositive
    for first, second in zip(
        vertices,
        vertices[1:] + vertices[:1],
    ):
        x0, y0 = first
        x1, y1 = second
        delta_x = abs(x1 - x0)
        step_x = 1 if x0 < x1 else -1
        delta_y = -abs(y1 - y0)
        step_y = 1 if y0 < y1 else -1
        error = delta_x + delta_y
        while True:
            if 0 <= x0 < width and 0 <= y0 < height:
                filled[y0, x0] = True
            if x0 == x1 and y0 == y1:
                break
            doubled = 2 * error
            if doubled >= delta_y:
                error += delta_y
                x0 += step_x
            if doubled <= delta_x:
                error += delta_x
                y0 += step_y
    return filled


def make_roi_mask_from_tracks(
    height: int,
    width: int,
    uv: np.ndarray,
    *,
    dilate_px: int = 0,
) -> np.ndarray:
    """Build a convex track ROI, falling back to the full image."""

    coordinates = np.asarray(uv)
    finite = np.isfinite(coordinates[:, 0]) & np.isfinite(coordinates[:, 1])
    vertices = coordinates[finite, :2].astype(np.int64)
    if vertices.shape[0] < 3:
        return np.ones((height, width), dtype=bool)
    mask = _fill_convex_boundary(
        int(height),
        int(width),
        _convex_boundary(vertices),
    )
    radius = int(dilate_px)
    if radius > 0:
        mask = binary_dilation(
            mask,
            structure=_ellipse_footprint(radius),
        )
    return np.asarray(mask, dtype=bool)


def make_points_disk_mask(
    height: int,
    width: int,
    uv: np.ndarray,
    *,
    radius_px: int = 3,
) -> np.ndarray:
    """Rasterize disks around finite, in-frame track points."""

    coordinates = np.asarray(uv)
    finite = np.isfinite(coordinates[:, 0]) & np.isfinite(coordinates[:, 1])
    points = np.rint(coordinates[finite]).astype(np.int64)
    if points.shape[0] == 0:
        return np.ones((height, width), dtype=bool)
    output = np.zeros((height, width), dtype=bool)
    radius = max(1, int(radius_px))
    for center_x, center_y in points:
        if not (0 <= center_x < int(width) and 0 <= center_y < int(height)):
            continue
        left = max(0, int(center_x) - radius)
        right = min(int(width), int(center_x) + radius + 1)
        top = max(0, int(center_y) - radius)
        bottom = min(int(height), int(center_y) + radius + 1)
        rows, columns = np.ogrid[top:bottom, left:right]
        output[top:bottom, left:right] |= (columns - int(center_x)) ** 2 + (
            rows - int(center_y)
        ) ** 2 <= radius**2
    return output


def _visible_track_points(
    tracks: np.ndarray | None,
    visibility: np.ndarray | None,
    *,
    threshold: float,
) -> np.ndarray:
    if tracks is None:
        return np.empty((0, 2), dtype=np.float32)
    coordinates = np.asarray(tracks, dtype=np.float32)
    if coordinates.size == 0:
        return np.empty((0, 2), dtype=np.float32)
    if coordinates.ndim == 2:
        coordinates = coordinates[None, ...]
    if coordinates.ndim < 3 or coordinates.shape[-1] < 2:
        return np.empty((0, 2), dtype=np.float32)
    flattened = coordinates[..., :2].reshape(-1, 2)
    selected = np.isfinite(flattened).all(axis=1)
    if visibility is not None:
        scores = np.asarray(
            visibility,
            dtype=np.float32,
        ).reshape(-1)
        if scores.size == selected.size:
            selected &= np.isfinite(scores)
            selected &= scores > float(threshold)
    return flattened[selected]


def compute_static_nearfield_anchor_mask(
    predicted_depth: np.ndarray,
    *,
    eef_tracks_uv: np.ndarray | None = None,
    eef_visibility: np.ndarray | None = None,
    obj_tracks_uv: np.ndarray | None = None,
    obj_visibility: np.ndarray | None = None,
    near_quantile: float = 0.5,
    tracks_radius_px: int = 12,
    vis_threshold: float = 0.4,
    min_pixels: int = 256,
) -> np.ndarray:
    """Select valid near-field background away from visible tracks."""

    depth = np.asarray(predicted_depth, dtype=np.float32)
    if depth.ndim != 2:
        raise ValueError(f"depth0 must be [H,W], got {depth.shape}")
    height, width = depth.shape
    valid = np.isfinite(depth) & (depth > _MIN_DEPTH)
    if not np.any(valid):
        return valid
    quantile = float(np.clip(float(near_quantile), 0.0, 1.0))
    cutoff = float(np.quantile(depth[valid], quantile))
    output = valid & (depth <= cutoff)
    point_groups = [
        _visible_track_points(
            eef_tracks_uv,
            eef_visibility,
            threshold=vis_threshold,
        ),
        _visible_track_points(
            obj_tracks_uv,
            obj_visibility,
            threshold=vis_threshold,
        ),
    ]
    present = [group for group in point_groups if group.shape[0]]
    if present:
        excluded = make_points_disk_mask(
            height,
            width,
            np.concatenate(present, axis=0),
            radius_px=int(tracks_radius_px),
        )
        output &= ~excluded
    if int(np.sum(output)) < int(min_pixels):
        return valid
    return output


def _pair_validity(
    predicted: np.ndarray,
    reference: np.ndarray,
) -> np.ndarray:
    return (
        np.isfinite(predicted)
        & np.isfinite(reference)
        & (predicted > _MIN_DEPTH)
        & (reference > _MIN_DEPTH)
    )


def build_calib_mask(
    predicted_depth: np.ndarray,
    reference_depth: np.ndarray,
    *,
    calib_region: str,
    valid_mask0: np.ndarray | None,
    roi_mask0: np.ndarray | None,
    tracks_uv0: np.ndarray | None,
    custom_calib_masks: Mapping[str, np.ndarray] | None = None,
    points_radius_px: int = 3,
    background_nearfield_quantile: float = 0.5,
) -> np.ndarray:
    """Resolve a named calibration region against valid depth pairs."""

    predicted = np.asarray(predicted_depth, dtype=np.float32)
    reference = np.asarray(reference_depth, dtype=np.float32)
    assert predicted.shape == reference.shape
    base = _pair_validity(predicted, reference)

    def optional_mask(value: np.ndarray | None) -> np.ndarray | None:
        if value is None:
            return None
        converted = np.asarray(value, dtype=bool)
        return converted if converted.shape == base.shape else None

    valid = optional_mask(valid_mask0)
    roi = optional_mask(roi_mask0)
    region = str(calib_region)
    if region == "valid":
        return base if valid is None else base & valid
    if region == "roi":
        return base if roi is None else base & roi
    if region == "roi∧valid":
        output = base.copy()
        if roi is not None:
            output &= roi
        if valid is not None:
            output &= valid
        return output
    if region == "tracks_points":
        if tracks_uv0 is None:
            return base
        disks = make_points_disk_mask(
            predicted.shape[0],
            predicted.shape[1],
            np.asarray(tracks_uv0),
            radius_px=int(points_radius_px),
        )
        return base & disks
    if region == "background_nearfield":
        eligible = base if valid is None else base & valid
        if not np.any(eligible):
            return eligible
        cutoff = float(
            np.quantile(
                predicted[eligible],
                float(
                    np.clip(
                        background_nearfield_quantile,
                        0.0,
                        1.0,
                    )
                ),
            )
        )
        nearfield = eligible & (predicted <= cutoff)
        return nearfield if roi is None else nearfield & ~roi
    if region in {
        "task_firstframe_neighborhood",
        "task_motion_envelope",
    }:
        custom = dict(custom_calib_masks or {}).get(region)
        selected = optional_mask(custom)
        return base if selected is None else base & selected
    return base


def _calibration_pairs(
    predicted_depth: np.ndarray,
    reference_depth: np.ndarray,
    valid_mask: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    predicted = np.asarray(predicted_depth, dtype=np.float32)
    reference = np.asarray(reference_depth, dtype=np.float32)
    assert predicted.shape == reference.shape
    selected = _pair_validity(predicted, reference)
    if valid_mask is not None:
        selected = np.asarray(valid_mask, dtype=bool) & selected
    return (
        predicted,
        predicted[selected].astype(np.float64),
        reference[selected].astype(np.float64),
    )


def _apply_affine(
    predicted: np.ndarray,
    scale: float,
    bias: float,
) -> np.ndarray:
    return (
        float(scale) * np.asarray(predicted, dtype=np.float64) + float(bias)
    ).astype(np.float32)


def least_squares_calib(
    predicted_depth: np.ndarray,
    reference_depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    *,
    use_affine: bool = True,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit scale-only or affine depth calibration by least squares."""

    predicted, samples, targets = _calibration_pairs(
        predicted_depth,
        reference_depth,
        valid_mask,
    )
    if samples.size == 0:
        scale, bias = 1.0, 0.0
    elif bool(use_affine):
        design = np.column_stack((samples, np.ones_like(samples)))
        solution = np.linalg.lstsq(
            design,
            targets,
            rcond=None,
        )[0]
        scale, bias = float(solution[0]), float(solution[1])
    else:
        denominator = float(np.vdot(samples, samples))
        scale = (
            float(np.vdot(samples, targets)) / denominator if denominator > 0.0 else 1.0
        )
        bias = 0.0
    calibrated = (
        _apply_affine(predicted, scale, bias)
        if bool(use_affine)
        else (float(scale) * np.asarray(predicted, dtype=np.float32)).astype(np.float32)
    )
    return calibrated, {
        "s": scale,
        "b": bias,
    }


def robust_affine_calib(
    predicted_depth: np.ndarray,
    reference_depth: np.ndarray,
    valid_mask: np.ndarray | None = None,
    clip_percentile: float = 99.0,
    huber_delta: float = 0.02,
    iters: int = 5,
) -> tuple[np.ndarray, dict[str, float]]:
    """Fit a bounded-influence affine calibration."""

    predicted, samples, targets = _calibration_pairs(
        predicted_depth,
        reference_depth,
        valid_mask,
    )
    if samples.size == 0:
        scale, bias = 1.0, 0.0
    else:
        upper_target = float(np.percentile(targets, float(clip_percentile)))
        # Preserve the current reference extractor's scientific contract:
        # percentile rejection is defined by metric reference depth only.
        # Filtering the model prediction independently changes the paired
        # sample set and therefore the fitted scale and bias.
        retained = targets < upper_target
        fit_samples = samples[retained]
        fit_targets = targets[retained]
        if fit_samples.size < 10:
            scale, bias = 1.0, 0.0
        else:
            design = np.stack(
                (fit_samples, np.ones_like(fit_samples)),
                axis=1,
            )
            solution = np.linalg.lstsq(
                design,
                fit_targets,
                rcond=None,
            )[0]
            delta = float(huber_delta)
            for _ in range(max(1, int(iters))):
                residual = design @ solution - fit_targets
                absolute_residual = np.abs(residual)
                weights = np.where(
                    absolute_residual <= delta,
                    1.0,
                    delta / (absolute_residual + 1.0e-12),
                )
                solution = np.linalg.lstsq(
                    design * weights[:, None],
                    fit_targets * weights,
                    rcond=None,
                )[0]
            scale, bias = float(solution[0]), float(solution[1])
    return _apply_affine(predicted, scale, bias), {
        "s": scale,
        "b": bias,
    }


def solve_depth_calibration(
    predicted_depth: np.ndarray,
    reference_depth: np.ndarray,
    *,
    valid_mask: np.ndarray | None,
    calibration_solver: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Dispatch one supported calibration solver."""

    if calibration_solver == "robust_affine":
        output, parameters = robust_affine_calib(
            predicted_depth,
            reference_depth,
            valid_mask,
        )
    elif calibration_solver == "least_squares_scale":
        output, parameters = least_squares_calib(
            predicted_depth,
            reference_depth,
            valid_mask,
            use_affine=False,
        )
    elif calibration_solver == "least_squares_affine":
        output, parameters = least_squares_calib(
            predicted_depth,
            reference_depth,
            valid_mask,
            use_affine=True,
        )
    else:
        raise ValueError(f"Unknown calibration_solver: {calibration_solver}")
    parameters["calibration_solver"] = calibration_solver
    return output, parameters


def _depth_stack(depths: Sequence[np.ndarray] | np.ndarray) -> list[np.ndarray]:
    if isinstance(depths, np.ndarray) and depths.ndim == 3:
        values: Sequence[np.ndarray] = list(depths)
    else:
        values = list(depths)
    return [np.asarray(value, dtype=np.float32).copy() for value in values]


def _optional_frame_masks(
    masks: Sequence[np.ndarray] | np.ndarray | None,
) -> list[np.ndarray] | None:
    if masks is None:
        return None
    if isinstance(masks, np.ndarray) and masks.ndim == 3:
        values: Sequence[np.ndarray] = list(masks)
    else:
        values = list(masks)
    if not values:
        return None
    return [np.asarray(value, dtype=bool) for value in values]


def _first_track_groups(
    value: np.ndarray | Sequence[np.ndarray] | None,
) -> list[np.ndarray]:
    """Normalize first-frame ROI tracks without stacking ragged groups."""

    if value is None:
        return []
    if isinstance(value, np.ndarray):
        candidates: list[Any] = [value]
    elif isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes),
    ):
        candidates = list(value)
    else:
        raise TypeError(
            "first_tracks_uv must be an [N,2] array or a sequence of [N,2] arrays"
        )

    groups: list[np.ndarray] = []
    for index, candidate in enumerate(candidates):
        if candidate is None:
            continue
        group = np.asarray(candidate)
        if group.ndim != 2 or group.shape[1] != 2:
            raise ValueError(
                "first_tracks_uv group "
                f"{index} must have shape [N,2], got {group.shape}"
            )
        groups.append(group)
    return groups


def _region_masks(
    *,
    shape: tuple[int, int],
    first_region_masks: np.ndarray | Sequence[np.ndarray] | None,
    first_tracks_uv: np.ndarray | Sequence[np.ndarray] | None,
    dilate_px: int,
) -> tuple[list[np.ndarray], str]:
    height, width = shape
    masks: list[np.ndarray] = []
    if first_region_masks is not None:
        raw_regions: list[np.ndarray]
        if (
            isinstance(first_region_masks, np.ndarray)
            and first_region_masks.shape == shape
        ):
            raw_regions = [first_region_masks]
        elif isinstance(first_region_masks, Sequence):
            raw_regions = list(first_region_masks)
        else:
            raw_regions = []
        masks = [
            np.asarray(mask, dtype=bool)
            for mask in raw_regions
            if np.asarray(mask).shape == shape
        ]
        if masks:
            return masks, "mask"
    if first_tracks_uv is None:
        return [], "none"
    groups = _first_track_groups(first_tracks_uv)
    masks = [
        make_roi_mask_from_tracks(
            height,
            width,
            group,
            dilate_px=int(dilate_px),
        )
        for group in groups
    ]
    return (masks, "tracks") if masks else ([], "none")


def _union(masks: Sequence[np.ndarray], shape: tuple[int, int]) -> np.ndarray:
    output = np.zeros(shape, dtype=bool)
    for mask in masks:
        output |= np.asarray(mask, dtype=bool)
    return output


def _distance_outside_roi(roi_mask: np.ndarray) -> np.ndarray:
    import cv2

    outside = (~np.asarray(roi_mask, dtype=bool)).astype(np.uint8)
    return cv2.distanceTransform(
        outside,
        cv2.DIST_L2,
        5,
    ).astype(np.float32)


def _soft_weights_from_rois(
    roi_masks: Sequence[np.ndarray],
    *,
    blend_sigma_px: float = 12.0,
    eps: float = 0.001,
) -> list[np.ndarray]:
    import cv2

    distances = [_distance_outside_roi(mask) for mask in roi_masks]
    weights = [
        (1.0 / (distance + float(eps))).astype(np.float32) for distance in distances
    ]
    denominator = np.sum(
        np.stack(weights, axis=0),
        axis=0,
    )
    weights = [(weight / denominator).astype(np.float32) for weight in weights]
    sigma = float(blend_sigma_px)
    if sigma > 0.0:
        radius = max(1, int(round(3.0 * sigma)))
        kernel = 2 * radius + 1
        weights = [
            cv2.GaussianBlur(
                weight,
                (kernel, kernel),
                sigmaX=sigma,
                sigmaY=sigma,
            ).astype(np.float32)
            for weight in weights
        ]
        denominator = np.sum(
            np.stack(weights, axis=0),
            axis=0,
        )
        weights = [(weight / denominator).astype(np.float32) for weight in weights]
    return weights


def _mask_fraction(mask: np.ndarray | None) -> float | None:
    return None if mask is None else float(np.mean(mask))


def _sanitize_depths(
    depths: list[np.ndarray],
    valid_masks: list[np.ndarray] | None,
    *,
    enabled: bool,
    invalidate_mode: str,
) -> tuple[list[np.ndarray], dict[str, Any], bool]:
    if not enabled:
        return (
            depths,
            {
                "enabled": False,
                "applied": False,
                "reason": "disabled",
            },
            False,
        )
    if valid_masks is None or invalidate_mode == "none":
        return (
            depths,
            {
                "enabled": True,
                "applied": False,
                "reason": ("no valid_masks or invalidate_mode=none"),
            },
            False,
        )
    output = [depth.copy() for depth in depths]
    if invalidate_mode == "nan":
        for index, valid in enumerate(valid_masks):
            if index >= len(output):
                break
            output[index][~valid] = np.nan
    return (
        output,
        {
            "enabled": True,
            "applied": True,
            "reason": "",
        },
        True,
    )


def _standard_calibration(
    depths: list[np.ndarray],
    reference: np.ndarray,
    *,
    valid0: np.ndarray | None,
    roi_masks: list[np.ndarray],
    tracks0: np.ndarray | None,
    custom_masks: Mapping[str, np.ndarray] | None,
    calib_region: str,
    points_radius_px: int,
    background_nearfield_quantile: float,
    calibration_solver: str,
    multi_roi_strategy: str,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    shape = depths[0].shape
    union_roi = _union(roi_masks, shape) if roi_masks else None
    calibration_mask = build_calib_mask(
        depths[0],
        reference,
        calib_region=calib_region,
        valid_mask0=valid0,
        roi_mask0=union_roi,
        tracks_uv0=tracks0,
        custom_calib_masks=custom_masks,
        points_radius_px=points_radius_px,
        background_nearfield_quantile=(background_nearfield_quantile),
    )
    _, union_parameters = solve_depth_calibration(
        depths[0],
        reference,
        valid_mask=calibration_mask,
        calibration_solver=calibration_solver,
    )
    union_scale = float(union_parameters["s"])
    union_bias = float(union_parameters["b"])
    calibrated = [
        (union_scale * np.asarray(depth, dtype=np.float32) + union_bias).astype(
            np.float32
        )
        for depth in depths
    ]
    details: dict[str, Any] = {
        "base_frac": float(np.mean(np.isfinite(reference) & (reference > _MIN_DEPTH))),
        "vm0_frac": _mask_fraction(valid0),
        "scheme_b": False,
        "calib_mask_frac": float(np.mean(calibration_mask)),
        "s0": union_scale,
        "b0": union_bias,
    }
    if len(roi_masks) > 1 and multi_roi_strategy == "blend":
        records: list[dict[str, Any]] = []
        for index, roi in enumerate(roi_masks):
            local_mask = build_calib_mask(
                depths[0],
                reference,
                calib_region=calib_region,
                valid_mask0=valid0,
                roi_mask0=roi,
                tracks_uv0=tracks0,
                custom_calib_masks=custom_masks,
                points_radius_px=points_radius_px,
                background_nearfield_quantile=(background_nearfield_quantile),
            )
            _, local_parameters = solve_depth_calibration(
                depths[0],
                reference,
                valid_mask=local_mask,
                calibration_solver=calibration_solver,
            )
            records.append(
                {
                    "roi_index": index,
                    "mask_frac": float(np.mean(local_mask)),
                    "s": float(local_parameters["s"]),
                    "b": float(local_parameters["b"]),
                }
            )
        weights = _soft_weights_from_rois(roi_masks)
        for frame_index, depth in enumerate(depths):
            blended = np.zeros_like(depth, dtype=np.float32)
            for weight, record in zip(weights, records):
                local_depth = (
                    float(record["s"]) * np.asarray(depth, dtype=np.float32)
                    + float(record["b"])
                ).astype(np.float32)
                blended += weight * local_depth
            calibrated[frame_index][union_roi] = blended[union_roi]
        details = {
            "base_frac": details["base_frac"],
            "vm0_frac": details["vm0_frac"],
            "scheme_b": True,
            "calib_mask_frac": details["calib_mask_frac"],
            "calib_mask_equiv": (f"scheme_b: per-roi + blend ({calibration_solver})"),
            "s_union": union_scale,
            "b_union": union_bias,
            "per_roi": records,
        }
    return calibrated, details


def _distribution_defaults(
    raw: Mapping[str, Any] | None,
) -> dict[str, Any]:
    defaults: dict[str, Any] = {
        "task_roi_margin_px": 24,
        "support_exclusion_dilate_px": 24,
        "support_edge_band_fraction": 0.15,
        "support_quantile_min": 0.15,
        "support_quantile_max": 0.60,
        "task_scale_ratio_limit": 0.10,
        "task_bias_delta_limit_m": 0.03,
        "support_bias_limit_m": 0.02,
        "blend_px": 12,
        "min_valid_pixels": 1500,
        "vis_threshold": 0.4,
    }
    defaults.update(dict(raw or {}))
    return defaults


def _track_rois_by_frame(
    shape: tuple[int, int],
    tracks: np.ndarray | None,
    visibility: np.ndarray | None,
    *,
    margin: int,
    threshold: float,
    frame_count: int,
) -> list[np.ndarray]:
    height, width = shape
    if tracks is None:
        if visibility is not None:
            raise ValueError("tracking visibility requires aligned tracks")
        return [np.zeros(shape, dtype=bool) for _ in range(frame_count)]
    coordinates = np.asarray(tracks, dtype=np.float32)
    if (
        coordinates.ndim != 3
        or coordinates.shape[-1] != 2
        or int(coordinates.shape[0]) != int(frame_count)
    ):
        raise ValueError(
            "tracking coordinates must be aligned [T,N,2], "
            f"got {coordinates.shape} for T={frame_count}"
        )
    scores = None
    if visibility is not None:
        scores = np.asarray(visibility, dtype=np.float32)
        if scores.shape != coordinates.shape[:2]:
            raise ValueError(
                "tracking visibility must align [T,N] with coordinates, "
                f"got {scores.shape} and {coordinates.shape}"
            )

    def region_mask(points: np.ndarray) -> np.ndarray:
        usable = np.asarray(points, dtype=np.float32).reshape(-1, 2)
        if int(usable.shape[0]) >= 3:
            return make_roi_mask_from_tracks(
                height,
                width,
                usable,
                dilate_px=margin,
            )
        if int(usable.shape[0]) == 0:
            return np.zeros(shape, dtype=bool)
        mask = make_points_disk_mask(
            height,
            width,
            usable,
            radius_px=max(2, int(margin) // 2),
        )
        if int(margin) > 0:
            mask = binary_dilation(
                mask,
                structure=_ellipse_footprint(int(margin)),
            )
        return np.asarray(mask, dtype=bool)

    output: list[np.ndarray] = []
    for frame_index in range(frame_count):
        frame_points = coordinates[frame_index]
        selected = np.isfinite(frame_points[..., :2]).all(axis=-1)
        if scores is not None:
            selected &= np.isfinite(scores[frame_index])
            selected &= scores[frame_index] >= threshold
        usable = frame_points[selected, :2]
        output.append(region_mask(usable))
    return output


def _distribution_task_rois(
    shape: tuple[int, int],
    *,
    eef_tracks_uv: np.ndarray | None,
    eef_visibility: np.ndarray | None,
    obj_tracks_uv: np.ndarray | None,
    obj_visibility: np.ndarray | None,
    obj_track_groups: Sequence[Mapping[str, Any]] | None,
    margin: int,
    threshold: float,
    frame_count: int,
) -> list[np.ndarray]:
    """Union per-stream task ROIs without bridging distinct objects."""

    sources: list[tuple[np.ndarray | None, np.ndarray | None]] = [
        (eef_tracks_uv, eef_visibility)
    ]
    if obj_track_groups:
        if isinstance(obj_track_groups, (str, bytes)):
            raise TypeError("obj_track_groups must be a sequence of mappings")
        for index, value in enumerate(obj_track_groups):
            if not isinstance(value, Mapping):
                raise TypeError(f"obj_track_groups[{index}] must be a mapping")
            if value.get("tracks_uv") is None:
                raise ValueError(f"obj_track_groups[{index}].tracks_uv is required")
            sources.append(
                (
                    value.get("tracks_uv"),
                    value.get("visibility"),
                )
            )
    elif obj_tracks_uv is not None or obj_visibility is not None:
        sources.append((obj_tracks_uv, obj_visibility))

    output = [np.zeros(shape, dtype=bool) for _ in range(frame_count)]
    for tracks, visibility in sources:
        stream_rois = _track_rois_by_frame(
            shape,
            tracks,
            visibility,
            margin=margin,
            threshold=threshold,
            frame_count=frame_count,
        )
        for index, mask in enumerate(stream_rois):
            output[index] |= mask
    return output


def _edge_ring(shape: tuple[int, int], fraction: float) -> np.ndarray:
    height, width = shape
    margin = max(
        1,
        int(round(min(height, width) * float(fraction))),
    )
    output = np.ones(shape, dtype=bool)
    if 2 * margin >= min(height, width):
        return output
    output[
        margin : height - margin,
        margin : width - margin,
    ] = False
    return output


def _feather_mask(
    mask: np.ndarray,
    *,
    blend_px: int,
) -> np.ndarray:
    import cv2

    distance = cv2.distanceTransform(
        np.asarray(mask, dtype=np.uint8),
        cv2.DIST_L2,
        5,
    )
    width = max(1, int(blend_px))
    return np.clip(
        distance / float(width),
        0.0,
        1.0,
    ).astype(np.float32)


def _distribution_calibration(
    depths: list[np.ndarray],
    reference: np.ndarray,
    *,
    valid0: np.ndarray | None,
    roi_masks: list[np.ndarray],
    tracks0: np.ndarray | None,
    custom_masks: Mapping[str, np.ndarray] | None,
    calib_region: str,
    points_radius_px: int,
    background_nearfield_quantile: float,
    calibration_solver: str,
    distribution_cfg: Mapping[str, Any] | None,
    eef_tracks_uv: np.ndarray | None,
    eef_visibility: np.ndarray | None,
    obj_tracks_uv: np.ndarray | None,
    obj_visibility: np.ndarray | None,
    obj_track_groups: Sequence[Mapping[str, Any]] | None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    config = _distribution_defaults(distribution_cfg)
    shape = depths[0].shape
    union_roi = _union(roi_masks, shape) if roi_masks else None
    global_mask = build_calib_mask(
        depths[0],
        reference,
        calib_region=calib_region,
        valid_mask0=valid0,
        roi_mask0=union_roi,
        tracks_uv0=tracks0,
        custom_calib_masks=custom_masks,
        points_radius_px=points_radius_px,
        background_nearfield_quantile=(background_nearfield_quantile),
    )
    _, global_parameters = solve_depth_calibration(
        depths[0],
        reference,
        valid_mask=global_mask,
        calibration_solver=calibration_solver,
    )
    global_scale = float(global_parameters["s"])
    global_bias = float(global_parameters["b"])

    task_source = dict(custom_masks or {}).get("task_firstframe_neighborhood")
    task_mask = (
        np.asarray(task_source, dtype=bool)
        if (task_source is not None and np.asarray(task_source).shape == shape)
        else np.zeros(shape, dtype=bool)
    )
    task_fit_mask = _pair_validity(depths[0], reference) & task_mask
    task_pixels = int(np.sum(task_fit_mask))
    minimum = int(config["min_valid_pixels"])
    fallback = task_pixels < minimum
    if fallback:
        task_scale, task_bias = global_scale, global_bias
    else:
        _, task_parameters = solve_depth_calibration(
            depths[0],
            reference,
            valid_mask=task_fit_mask,
            calibration_solver=calibration_solver,
        )
        raw_scale = float(task_parameters["s"])
        scale_limit = float(config["task_scale_ratio_limit"])
        task_scale = float(
            np.clip(
                raw_scale,
                global_scale * (1.0 - scale_limit),
                global_scale * (1.0 + scale_limit),
            )
        )
        raw_bias = float(task_parameters["b"])
        bias_limit = float(config["task_bias_delta_limit_m"])
        task_bias = float(
            np.clip(
                raw_bias,
                global_bias - bias_limit,
                global_bias + bias_limit,
            )
        )

    task_rois = _distribution_task_rois(
        shape,
        eef_tracks_uv=eef_tracks_uv,
        eef_visibility=eef_visibility,
        obj_tracks_uv=obj_tracks_uv,
        obj_visibility=obj_visibility,
        obj_track_groups=obj_track_groups,
        margin=int(config["task_roi_margin_px"]),
        threshold=float(config["vis_threshold"]),
        frame_count=len(depths),
    )
    task_counts = [int(np.sum(mask)) for mask in task_rois]

    base_valid = _pair_validity(depths[0], reference)
    reference_valid = np.isfinite(reference) & (reference > _MIN_DEPTH)
    if valid0 is not None:
        reference_valid &= np.asarray(valid0, dtype=bool)
    task_exclusion = (
        binary_dilation(
            task_mask,
            structure=_ellipse_footprint(int(config["support_exclusion_dilate_px"])),
        )
        if np.any(task_mask)
        else np.zeros(shape, dtype=bool)
    )
    support_candidates = (
        reference_valid
        & ~task_exclusion
        & ~_edge_ring(
            shape,
            float(config["support_edge_band_fraction"]),
        )
    )
    if np.any(support_candidates):
        lower = float(
            np.quantile(
                reference[support_candidates],
                float(config["support_quantile_min"]),
            )
        )
        upper = float(
            np.quantile(
                reference[support_candidates],
                float(config["support_quantile_max"]),
            )
        )
        support_frame = support_candidates & (reference >= lower) & (reference <= upper)
    else:
        support_frame = support_candidates
    support_pixels = int(np.sum(support_frame))
    fallback_reason = ""
    if support_pixels < minimum:
        support_frame = support_candidates
        support_pixels = int(np.sum(support_frame))
        fallback_reason = "insufficient_edge_support"

    global_frame0 = (global_scale * depths[0] + global_bias).astype(np.float32)
    support_residual = reference[support_frame] - global_frame0[support_frame]
    if support_residual.size:
        support_bias = float(np.median(support_residual))
        support_bias = float(
            np.clip(
                support_bias,
                -float(config["support_bias_limit_m"]),
                float(config["support_bias_limit_m"]),
            )
        )
        centered = support_residual.astype(np.float64) - support_bias
        support_rmse = float(np.sqrt(np.mean(centered**2)))
    else:
        support_bias = 0.0
        support_rmse = 0.0

    calibrated: list[np.ndarray] = []
    support_counts: list[int] = []
    blend_pixels = max(1, int(config["blend_px"]))
    for index, depth in enumerate(depths):
        global_depth = (global_scale * depth + global_bias).astype(np.float32)
        task_depth = (task_scale * depth + task_bias).astype(np.float32)
        task_roi = (
            task_rois[index] if index < len(task_rois) else np.zeros(shape, dtype=bool)
        )
        task_weight = _feather_mask(
            task_roi,
            blend_px=blend_pixels,
        )
        frame_support = support_frame.copy()
        if index < len(task_rois):
            frame_support &= ~task_roi
        support_counts.append(int(np.sum(frame_support)))
        support_weight = _feather_mask(
            frame_support,
            blend_px=blend_pixels,
        )
        output = (
            global_depth
            + task_weight * (task_depth - global_depth)
            + support_weight * support_bias
        ).astype(np.float32)
        calibrated.append(output)

    def summary(values: list[int]) -> dict[str, Any]:
        return {
            "min": int(min(values)) if values else 0,
            "mean": float(np.mean(values)) if values else 0.0,
            "max": int(max(values)) if values else 0,
        }

    details = {
        "base_frac": float(np.mean(base_valid)),
        "vm0_frac": _mask_fraction(valid0),
        "scheme_b": False,
        "distribution_v1": {
            "mode": "distribution_v1",
            "applied": True,
            "global_branch": {
                "s": global_scale,
                "b": global_bias,
                "pixel_count": int(np.sum(global_mask)),
            },
            "task_branch": {
                "s": task_scale,
                "b": task_bias,
                "pixel_count": task_pixels,
                "fallback_to_global": fallback,
                "task_scale_ratio_limit": float(config["task_scale_ratio_limit"]),
                "task_bias_delta_limit_m": float(config["task_bias_delta_limit_m"]),
            },
            "support_branch": {
                "bias_m": support_bias,
                "pixel_count": support_pixels,
                "rmse_m": support_rmse,
                "support_bias_limit_m": float(config["support_bias_limit_m"]),
            },
            "support_frame0": {
                "pixel_count": support_pixels,
                "edge_band_fraction": float(config["support_edge_band_fraction"]),
                "support_quantile_min": float(config["support_quantile_min"]),
                "support_quantile_max": float(config["support_quantile_max"]),
                "fallback_reason": fallback_reason,
            },
            "task_roi_pixels": summary(task_counts),
            "support_roi_pixels": summary(support_counts),
            "task_roi_margin_px": int(config["task_roi_margin_px"]),
            "blend_px": int(config["blend_px"]),
            "vis_threshold": float(config["vis_threshold"]),
        },
    }
    return calibrated, details


def run_depth_calibration(
    depths: Sequence[np.ndarray] | np.ndarray,
    *,
    depth_space: str,
    init_ref_depth: np.ndarray | None = None,
    valid_masks: Sequence[np.ndarray] | np.ndarray | None = None,
    first_tracks_uv: np.ndarray | Sequence[np.ndarray] | None = None,
    first_region_masks: np.ndarray | Sequence[np.ndarray] | None = None,
    custom_calib_masks: Mapping[str, np.ndarray] | None = None,
    calib_region: str = "valid",
    roi_dilate_px: int = 8,
    points_radius_px: int = 3,
    background_nearfield_quantile: float = 0.5,
    invalidate_mode: str = "none",
    calibration_solver: str = "robust_affine",
    multi_roi_strategy: str = "blend",
    require_calibration_for_nonmetric: bool = False,
    run_sanitize: bool = True,
    run_init_calibration: bool = True,
    run_final_smooth: bool = False,
    sigma_r: float = 0.02,
    init_calibration_mode: str = "standard",
    distribution_cfg: Mapping[str, Any] | None = None,
    eef_tracks_uv: np.ndarray | None = None,
    eef_visibility: np.ndarray | None = None,
    obj_tracks_uv: np.ndarray | None = None,
    obj_visibility: np.ndarray | None = None,
    obj_track_groups: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Sanitize and calibrate an explicit depth stack."""

    output = _depth_stack(depths)
    height, width = output[0].shape
    frame_masks = _optional_frame_masks(valid_masks)
    output, sanitize_stats, invalidated = _sanitize_depths(
        output,
        frame_masks,
        enabled=bool(run_sanitize),
        invalidate_mode=str(invalidate_mode),
    )
    roi_masks, roi_source = _region_masks(
        shape=(height, width),
        first_region_masks=first_region_masks,
        first_tracks_uv=first_tracks_uv,
        dilate_px=int(roi_dilate_px),
    )
    union_roi = _union(roi_masks, (height, width)) if roi_masks else None
    stats: dict[str, Any] = {
        "depth_space": str(depth_space),
        "T": len(output),
        "H": int(height),
        "W": int(width),
        "calib_region": str(calib_region),
        "init_calibration_mode": str(init_calibration_mode),
        "invalidate_mode": str(invalidate_mode),
        "calibration_solver": str(calibration_solver),
        "multi_roi_strategy": str(multi_roi_strategy),
        "sanitize": sanitize_stats,
    }
    calibrated = False
    multi_roi = False
    scheme_b = False
    if not run_init_calibration:
        init_stats: dict[str, Any] = {
            "enabled": False,
            "applied": False,
            "reason": "disabled",
        }
    elif init_ref_depth is None:
        init_stats = {
            "enabled": True,
            "applied": False,
            "reason": "init_ref_depth unavailable",
        }
    else:
        reference = np.asarray(
            init_ref_depth,
            dtype=np.float32,
        )
        assert reference.shape == (height, width), (
            f"init_ref_depth shape {reference.shape} != {(height, width)}"
        )
        valid0 = (
            frame_masks[0]
            if frame_masks is not None and frame_masks[0].shape == (height, width)
            else None
        )
        tracks0 = None
        if first_tracks_uv is not None:
            track_groups = _first_track_groups(first_tracks_uv)
            tracks0 = track_groups[0] if track_groups else None
        if init_calibration_mode == "distribution_v1":
            output, details = _distribution_calibration(
                output,
                reference,
                valid0=valid0,
                roi_masks=roi_masks,
                tracks0=tracks0,
                custom_masks=custom_calib_masks,
                calib_region=str(calib_region),
                points_radius_px=int(points_radius_px),
                background_nearfield_quantile=float(background_nearfield_quantile),
                calibration_solver=str(calibration_solver),
                distribution_cfg=distribution_cfg,
                eef_tracks_uv=eef_tracks_uv,
                eef_visibility=eef_visibility,
                obj_tracks_uv=obj_tracks_uv,
                obj_visibility=obj_visibility,
                obj_track_groups=obj_track_groups,
            )
            mode = "distribution_v1"
        else:
            output, details = _standard_calibration(
                output,
                reference,
                valid0=valid0,
                roi_masks=roi_masks,
                tracks0=tracks0,
                custom_masks=custom_calib_masks,
                calib_region=str(calib_region),
                points_radius_px=int(points_radius_px),
                background_nearfield_quantile=float(background_nearfield_quantile),
                calibration_solver=str(calibration_solver),
                multi_roi_strategy=str(multi_roi_strategy),
            )
            mode = "standard"
            multi_roi = bool(details.get("scheme_b", False))
        calibrated = True
        scheme_b = bool(details.get("scheme_b", False))
        init_stats = {
            "enabled": True,
            "applied": True,
            "reason": "",
            "mode": mode,
        }
        if "distribution_v1" in details:
            init_stats["distribution_v1"] = details.pop("distribution_v1")
        stats.update(details)
    stats["init_calibration"] = init_stats

    if (
        init_ref_depth is None
        and str(depth_space) != "metric"
        and bool(run_init_calibration)
    ):
        warning = (
            "non-metric depth without calibration: "
            f"depth_space={str(depth_space)!r}, but "
            "init_ref_depth is None. Depth scale/offset "
            "undefined; downstream lift can be wrong."
        )
        if require_calibration_for_nonmetric:
            raise RuntimeError(f"[DepthCalibration] {warning}")
        stats["warning"] = warning

    stats["roi_used"] = bool(roi_masks)
    stats["roi_source"] = roi_source
    stats["roi_count"] = len(roi_masks)
    if union_roi is not None:
        stats["roi0_frac"] = float(np.mean(union_roi))
        stats["roi_dilate_px"] = int(roi_dilate_px)
    stats["calibrated"] = calibrated
    stats["multi_roi"] = multi_roi
    stats["scheme_b"] = scheme_b
    stats["invalidated"] = invalidated
    smooth_applied = bool(run_final_smooth and (depth_space == "metric" or calibrated))
    if smooth_applied:
        roi = union_roi if union_roi is not None else np.ones((height, width), dtype=bool)
        output = [bilateral_smooth_in_roi(frame, roi, sigma_r=sigma_r) for frame in output]
    stats["final_smooth"] = {"enabled": bool(run_final_smooth), "applied": smooth_applied}
    stats["applied"] = bool(calibrated or invalidated or smooth_applied)
    return output, stats


def run_depth_calibration_runtime(
    *,
    depths: Sequence[np.ndarray] | np.ndarray,
    depth_source: str,
    depth_model: str,
    depth_base_cfg: Mapping[str, Any],
    depth_info: Mapping[str, Any],
    init_ref_depth: np.ndarray | None,
    valid_masks: Sequence[np.ndarray] | np.ndarray | None,
    first_tracks_uv: np.ndarray | Sequence[np.ndarray] | None = None,
    first_region_masks: np.ndarray | Sequence[np.ndarray] | None = None,
    custom_calib_masks: Mapping[str, np.ndarray] | None = None,
    eef_tracks_uv: np.ndarray | None = None,
    eef_visibility: np.ndarray | None = None,
    obj_tracks_uv: np.ndarray | None = None,
    obj_visibility: np.ndarray | None = None,
    obj_track_groups: Sequence[Mapping[str, Any]] | None = None,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Adapt the built-in numeric calibration to the depth-source runtime.

    GT input skips the entire base postprocess, including final smoothing,
    as in the pre-refactor source pipeline.
    ``depth_model`` does not select policy; ``depth_base_cfg`` owns settings.
    """

    del depth_model
    if not isinstance(depth_info, Mapping):
        raise TypeError("depth_info must be a mapping")
    base = normalize_depth_base_config(dict(depth_base_cfg))
    sanitize = dict(base["sanitize"])
    init = dict(base["init_calibration"])
    base_enabled = bool(base["enabled"])
    is_gt = depth_source == "rollout_gt_depth"
    smooth = dict(base["final_smooth"])
    if_calibrate = init.get("if_calibrate_depth")
    run_init = base_enabled and not is_gt and bool(init["enabled"]) and if_calibrate is not False
    calibrated, stats = run_depth_calibration(
        depths,
        depth_space=str(depth_info.get("depth_space", "unknown")),
        init_ref_depth=init_ref_depth,
        valid_masks=valid_masks,
        first_tracks_uv=first_tracks_uv,
        first_region_masks=first_region_masks,
        custom_calib_masks=custom_calib_masks,
        calib_region=str(init.get("calib_region") or "valid"),
        roi_dilate_px=int(
            8 if init.get("roi_dilate_px") is None else init["roi_dilate_px"]
        ),
        points_radius_px=int(
            3 if init.get("points_radius_px") is None else init["points_radius_px"]
        ),
        background_nearfield_quantile=float(
            init.get("background_nearfield_quantile") or 0.5
        ),
        invalidate_mode=str(sanitize.get("invalidate_mode") or "none"),
        calibration_solver=str(init.get("calibration_solver") or "robust_affine"),
        multi_roi_strategy=str(init.get("multi_roi_strategy") or "blend"),
        require_calibration_for_nonmetric=bool(
            init.get("require_calibration_for_nonmetric") or False
        ),
        run_sanitize=(base_enabled and not is_gt and bool(sanitize["enabled"])),
        run_init_calibration=run_init,
        run_final_smooth=(
            base_enabled
            and not is_gt
            and bool(smooth["enabled"])
            and smooth["bilateral_mode"] == "on"
        ),
        sigma_r=float(smooth["sigma_r"]),
        init_calibration_mode=str(init.get("mode") or "standard"),
        distribution_cfg=dict(init.get("distribution_v1", {}) or {}),
        eef_tracks_uv=eef_tracks_uv,
        eef_visibility=eef_visibility,
        obj_tracks_uv=obj_tracks_uv,
        obj_visibility=obj_visibility,
        obj_track_groups=obj_track_groups,
    )
    if is_gt:
        reason = "skipped because depth source is rollout_gt_depth"
        stats["applied"] = False
        stats["reason"] = reason
        stats["sanitize"] = {
            "enabled": base_enabled and bool(sanitize["enabled"]),
            "applied": False,
            "reason": reason,
        }
        stats["init_calibration"] = {
            "enabled": base_enabled and bool(init["enabled"]),
            "applied": False,
            "reason": reason,
        }
        stats["final_smooth"] = {
            "enabled": (
                base_enabled
                and bool(smooth["enabled"])
                and smooth["bilateral_mode"] == "on"
            ),
            "applied": False,
            "reason": reason,
        }
    info = copy.deepcopy(dict(depth_info))
    previous_calibration = info.get("calibration")
    if previous_calibration is None:
        calibration_info: dict[str, Any] = {}
    elif isinstance(previous_calibration, Mapping):
        calibration_info = copy.deepcopy(dict(previous_calibration))
    else:
        raise TypeError("depth_info.calibration must be a mapping or None")
    calibration_info["base"] = copy.deepcopy(stats)
    info["calibration"] = calibration_info
    stage_stats = {"base": copy.deepcopy(stats)}
    return np.asarray(calibrated, dtype=np.float32), info, stage_stats


__all__ = [
    "build_calib_mask",
    "compute_static_nearfield_anchor_mask",
    "least_squares_calib",
    "make_points_disk_mask",
    "make_roi_mask_from_tracks",
    "robust_affine_calib",
    "run_depth_calibration",
    "run_depth_calibration_runtime",
    "solve_depth_calibration",
]
