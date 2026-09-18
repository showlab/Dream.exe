"""Built-in, model-free region query sampler backends."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from ..contract import (
    REGION_QUERY_SAMPLER_CONTRACT_VERSION,
    RegionPointSample,
    RegionSamplingPrediction,
    RegionSamplingRequest,
)


def _bbox_mask(
    shape_hw: tuple[int, int],
    bbox_xyxy: Sequence[int | float],
) -> np.ndarray:
    height, width = (int(value) for value in shape_hw)
    if len(bbox_xyxy) != 4:
        raise ValueError(f"Expected bbox [x0,y0,x1,y1], got {bbox_xyxy}")
    coordinates = [int(round(float(component))) for component in bbox_xyxy]
    left, right = sorted((coordinates[0], coordinates[2]))
    top, bottom = sorted((coordinates[1], coordinates[3]))
    left = max(0, min(left, max(0, width - 1)))
    right = max(0, min(right, max(0, width - 1)))
    top = max(0, min(top, max(0, height - 1)))
    bottom = max(0, min(bottom, max(0, height - 1)))
    mask = np.zeros((height, width), dtype=bool)
    mask[top : bottom + 1, left : right + 1] = True
    return mask


def farthest_point_indices(
    points: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select deterministic farthest-point-sampling indices."""

    normalized = np.asarray(points, dtype=np.float32)
    point_count = int(normalized.shape[0])
    if point_count == 0:
        return np.zeros((0,), dtype=np.int64)
    if count >= point_count:
        return np.arange(point_count, dtype=np.int64)
    if count <= 1:
        return np.array(
            [int(rng.integers(0, point_count))],
            dtype=np.int64,
        )

    first = int(rng.integers(0, point_count))
    selected = [first]
    minimum_distance_squared = np.sum(
        (normalized - normalized[first]) ** 2,
        axis=1,
    )
    minimum_distance_squared[first] = -1.0
    for _ in range(1, count):
        index = int(np.argmax(minimum_distance_squared))
        selected.append(index)
        distance_squared = np.sum(
            (normalized - normalized[index]) ** 2,
            axis=1,
        )
        minimum_distance_squared = np.minimum(
            minimum_distance_squared,
            distance_squared,
        )
        minimum_distance_squared[selected] = -1.0
    return np.asarray(selected, dtype=np.int64)


class BBoxGaussianQuerySampler:
    """Generate deterministic pixel queries around a resolved bbox center."""

    provider_kind = "builtin"
    backend_id = "bbox_gaussian"
    algorithm_id = "clipped_center_gaussian"
    contract_version = REGION_QUERY_SAMPLER_CONTRACT_VERSION

    def sample(
        self,
        request: RegionSamplingRequest,
    ) -> RegionSamplingPrediction:
        """Return current bbox-Gaussian points without loading a model."""

        if request.bbox_xyxy is None:
            raise ValueError("BBoxGaussianQuerySampler requires request.bbox_xyxy")
        x_min, y_min, x_max, y_max = [int(value) for value in request.bbox_xyxy]
        rng = np.random.default_rng(int(request.seed))
        center_x = (x_min + x_max) / 2.0
        center_y = (y_min + y_max) / 2.0
        sigma_x = max((x_max - x_min) / 6.0, 1e-6)
        sigma_y = max((y_max - y_min) / 6.0, 1e-6)
        x_coordinates = np.clip(
            rng.normal(
                center_x,
                sigma_x,
                int(request.num_points),
            ),
            x_min,
            x_max,
        )
        y_coordinates = np.clip(
            rng.normal(
                center_y,
                sigma_y,
                int(request.num_points),
            ),
            y_min,
            y_max,
        )
        sample = RegionPointSample(
            points_xy=np.stack(
                (x_coordinates, y_coordinates),
                axis=1,
            ).astype(np.float32),
            sampling_mask=_bbox_mask(
                request.frame_shape_hw,
                request.bbox_xyxy,
            ),
        )
        return RegionSamplingPrediction(
            sample=sample,
            backend_id=self.backend_id,
            algorithm_id=self.algorithm_id,
            coordinate_frame="pixel_xy",
            contract_version=self.contract_version,
            metadata={
                "requested_points": int(request.num_points),
                "selected_points": int(sample.points_xy.shape[0]),
                "seed": int(request.seed),
            },
        )


class MaskFarthestPointQuerySampler:
    """Select image- or world-space FPS queries from a resolved mask."""

    provider_kind = "builtin"
    backend_id = "mask_fps"
    algorithm_id = "farthest_point_sampling"
    contract_version = REGION_QUERY_SAMPLER_CONTRACT_VERSION

    def __init__(self, *, space: str = "3d_fps") -> None:
        normalized = str(space or "3d_fps").strip().lower()
        aliases = {
            "mask_2d_fps": "2d_fps",
            "mask_3d_fps": "3d_fps",
        }
        self.space = aliases.get(normalized, normalized)
        if self.space not in {"2d_fps", "3d_fps"}:
            raise ValueError(
                "MaskFarthestPointQuerySampler.space must be '2d_fps' or '3d_fps'"
            )

    def sample(
        self,
        request: RegionSamplingRequest,
    ) -> RegionSamplingPrediction:
        """Return current mask FPS points and explicit fallback identity."""

        if request.mask is None:
            raise ValueError("MaskFarthestPointQuerySampler requires request.mask")
        input_mask = np.asarray(request.mask, dtype=bool)
        if not np.any(input_mask):
            raise RuntimeError("Cannot sample CoTracker queries from an empty mask.")
        rng = np.random.default_rng(int(request.seed))
        sampling_mask = input_mask.copy()
        normalized_depth = None
        if request.init_depth is not None:
            normalized_depth = np.asarray(
                request.init_depth,
                dtype=np.float32,
            )
            if normalized_depth.shape == sampling_mask.shape:
                valid_depth_mask = np.isfinite(normalized_depth) & (
                    normalized_depth > 1e-6
                )
                depth_constrained_mask = sampling_mask & valid_depth_mask
                if np.any(depth_constrained_mask):
                    sampling_mask = depth_constrained_mask

        ys, xs = np.nonzero(sampling_mask)
        pixel_coordinates = np.stack(
            (xs, ys),
            axis=1,
        ).astype(np.float32)
        sample: RegionPointSample
        if (
            self.space == "3d_fps"
            and normalized_depth is not None
            and request.camera is not None
            and normalized_depth.shape == sampling_mask.shape
        ):
            depth_values = normalized_depth[ys, xs]
            valid = np.isfinite(depth_values) & (depth_values > 1e-6)
            if np.any(valid):
                valid_pixel_coordinates = pixel_coordinates[valid]
                points_3d = np.stack(
                    [
                        request.camera.pixel_to_world(
                            float(x),
                            float(y),
                            normalized_depth,
                            bilinear=False,
                        )
                        for x, y in valid_pixel_coordinates
                    ],
                    axis=0,
                ).astype(np.float32)
                keep = farthest_point_indices(
                    points_3d,
                    count=min(
                        int(request.num_points),
                        points_3d.shape[0],
                    ),
                    rng=rng,
                )
                sample = RegionPointSample(
                    points_xy=valid_pixel_coordinates[keep],
                    sampling_mask=sampling_mask,
                    points_xyz=points_3d[keep],
                    used_depth_for_sampling=True,
                )
            else:
                keep = farthest_point_indices(
                    pixel_coordinates,
                    count=min(
                        int(request.num_points),
                        pixel_coordinates.shape[0],
                    ),
                    rng=rng,
                )
                sample = RegionPointSample(
                    points_xy=pixel_coordinates[keep],
                    sampling_mask=sampling_mask,
                    points_xyz=None,
                    used_depth_for_sampling=False,
                )
        else:
            keep = farthest_point_indices(
                pixel_coordinates,
                count=min(
                    int(request.num_points),
                    pixel_coordinates.shape[0],
                ),
                rng=rng,
            )
            sample = RegionPointSample(
                points_xy=pixel_coordinates[keep],
                sampling_mask=sampling_mask,
                points_xyz=None,
                used_depth_for_sampling=False,
            )
        requested_frame = "world_xyz" if self.space == "3d_fps" else "pixel_xy"
        effective_frame = "world_xyz" if sample.used_depth_for_sampling else "pixel_xy"
        metadata: dict[str, Any] = {
            "requested_space": requested_frame,
            "effective_space": effective_frame,
            "requested_points": int(request.num_points),
            "selected_points": int(sample.points_xy.shape[0]),
            "input_mask_candidates": int(np.count_nonzero(input_mask)),
            "effective_mask_candidates": int(np.count_nonzero(sample.sampling_mask)),
            "depth_validity_filter_applied": bool(
                not np.array_equal(
                    sample.sampling_mask,
                    input_mask,
                )
            ),
            "seed": int(request.seed),
        }
        if requested_frame == "world_xyz" and effective_frame == "pixel_xy":
            normalized_depth = (
                None if request.init_depth is None else np.asarray(request.init_depth)
            )
            if normalized_depth is None:
                fallback_reason = "depth_unavailable"
            elif normalized_depth.shape != input_mask.shape:
                fallback_reason = "depth_shape_mismatch"
            elif request.camera is None:
                fallback_reason = "camera_unavailable"
            elif not np.any(
                input_mask & np.isfinite(normalized_depth) & (normalized_depth > 1e-6)
            ):
                fallback_reason = "no_valid_depth_in_mask"
            else:
                fallback_reason = "world_lift_unavailable"
            metadata["fallback_reason"] = fallback_reason
        return RegionSamplingPrediction(
            sample=sample,
            backend_id=self.backend_id,
            algorithm_id=self.algorithm_id,
            coordinate_frame=effective_frame,
            contract_version=self.contract_version,
            metadata=metadata,
        )


def builtin_region_query_samplers() -> dict[str, Any]:
    """Return one fresh backend per current sampling-method identity."""

    return {
        "bbox_gaussian": BBoxGaussianQuerySampler(),
        "mask_fps": MaskFarthestPointQuerySampler(space="2d_fps"),
        "mask_2d_fps": MaskFarthestPointQuerySampler(space="2d_fps"),
        "mask_3d_fps": MaskFarthestPointQuerySampler(space="3d_fps"),
    }


__all__ = [
    "BBoxGaussianQuerySampler",
    "MaskFarthestPointQuerySampler",
    "builtin_region_query_samplers",
    "farthest_point_indices",
]
