"""Frame-aligned simulator-independent pose backend template."""

from __future__ import annotations

from dream_exe.video2traj.pose import (
    BasePoseBackend,
    PoseBackendRequest,
    PosePrediction,
)


class ExamplePoseBackend(BasePoseBackend):
    backend_id = "example_pose"

    def infer(self, request: PoseBackendRequest) -> PosePrediction:
        del request
        raise NotImplementedError(
            "Return PosePrediction with candidates aligned to request.video_frames"
        )


__all__ = ["ExamplePoseBackend"]
