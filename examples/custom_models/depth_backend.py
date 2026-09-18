"""Frame-aligned monocular or video-depth backend template."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from dream_exe.video2traj.depth import BaseDepthBackend


class ExampleDepthBackend(BaseDepthBackend):
    backend_id = "example_depth"

    def infer(
        self,
        video_frames: Sequence[Any],
        target_fps: float,
        *,
        fp32: bool,
        input_size: int,
        intrinsics: np.ndarray | None = None,
        extrinsics: np.ndarray | None = None,
    ) -> Any:
        del video_frames, target_fps, fp32, input_size, intrinsics, extrinsics
        raise NotImplementedError(
            "Return depths normalizable to [T,H,W] and optional fps/provenance"
        )


__all__ = ["ExampleDepthBackend"]
