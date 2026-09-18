"""Path-neutral tracking backend template."""

from __future__ import annotations

from typing import Any

from dream_exe.video2traj.tracking import BaseTrackingPredictionBackend


class ExampleTrackingBackend(BaseTrackingPredictionBackend):
    backend_id = "example_tracker"

    def predict(
        self,
        *,
        video_frames: Any,
        region_bbox_xyxy: Any = None,
        num_points: int = 50,
        seed: int = 42,
        query_points_xy: Any = None,
        segmentation_mask: Any = None,
        query_mode: str = "auto",
        grid_size: int = 0,
    ) -> tuple[Any, Any, Any, str]:
        del (
            video_frames,
            region_bbox_xyxy,
            num_points,
            seed,
            query_points_xy,
            segmentation_mask,
            query_mode,
            grid_size,
        )
        raise NotImplementedError(
            "Return tracks[T,N,2], visibility[T,N], query_points[N,2], and mode"
        )


__all__ = ["ExampleTrackingBackend"]
