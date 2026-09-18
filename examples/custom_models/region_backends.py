"""Text-conditioned detector and box-conditioned segmenter templates."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from dream_exe.video2traj.region import (
    BaseRegionDetector,
    BaseRegionSegmenter,
    RegionProposal,
    RegionSegmentationPrediction,
)


class ExampleRegionDetector(BaseRegionDetector):
    backend_id = "example_detector"
    algorithm_id = "example_text_detection"

    def detect(
        self,
        image_rgb: np.ndarray,
        prompt: str,
    ) -> RegionProposal | Mapping[str, Any]:
        del image_rgb, prompt
        raise NotImplementedError("Return one pixel-coordinate xyxy proposal")


class ExampleRegionSegmenter(BaseRegionSegmenter):
    backend_id = "example_segmenter"
    algorithm_id = "example_box_segmentation"

    def segment_from_bbox(
        self,
        image_rgb: np.ndarray,
        bbox_xyxy: list[int],
        prompt: str | None = None,
    ) -> np.ndarray | RegionSegmentationPrediction:
        del image_rgb, bbox_xyxy, prompt
        raise NotImplementedError("Return one [H,W] mask aligned to image_rgb")


__all__ = ["ExampleRegionDetector", "ExampleRegionSegmenter"]
