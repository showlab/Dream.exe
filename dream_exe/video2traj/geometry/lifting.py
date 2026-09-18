"""Geometry for pure 2D-track and depth lifting into camera/world 3D trajectories.

The current implementation mixes this computation with JSON and NPZ writes.
current implementation keeps the scientific behavior here and leaves artifact persistence to an
external writer. The result value groups arrays that share the same ``[T, N]``
frame/track alignment; it is not a configurable service or class hierarchy.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from .camera import Camera, CameraExtrinsics


@dataclass(frozen=True)
class LiftedPointTrajectory:
    """Aligned outputs of one track/depth lifting invocation.

    ``frames`` preserves the current JSON record schema. Dense arrays preserve
    the current flow-NPZ payload needed by downstream diagnostics and future
    artifact writers.
    """

    frames: List[Dict[str, Any]]
    positions_camera: np.ndarray
    positions_world: np.ndarray
    query_tracks_uv: np.ndarray
    tracker_visibility: np.ndarray
    tracker_visible_mask: np.ndarray
    depth_samples_override: np.ndarray
    depth_valid_mask: np.ndarray
    valid_mask: np.ndarray


def lift_tracks_to_3d(
    tracks_uv: np.ndarray,
    visibility: np.ndarray,
    depths: np.ndarray,
    camera: Camera,
    extrinsics_by_frame: Optional[Sequence[CameraExtrinsics]] = None,
    bilinear_depth: bool = True,
    visibility_threshold: float = 0.5,
    *,
    depth_samples_override: Optional[np.ndarray] = None,
) -> LiftedPointTrajectory:
    """Lift visible in-frame tracks with depth into camera and world points.

    The coordinate, threshold, filtering, dtype, and alignment behavior matches
    the current trajectory geometry implementation:

    - visibility is strict ``> visibility_threshold`` unless already boolean;
    - out-of-frame tracks are omitted;
    - a finite override depth greater than ``1e-9`` wins per track/frame;
    - invalid or non-positive lifted depth is omitted;
    - every input frame still produces one frame record;
    - dense arrays remain indexed by the original ``[frame, point_id]``.
    """

    depth_maps = np.asarray(depths)
    frame_count, height, width = depth_maps.shape
    assert camera.K.width == width and camera.K.height == height, (
        f"Depth size {(width, height)} != K {(camera.K.width, camera.K.height)}"
    )
    assert tracks_uv.shape[0] == frame_count == visibility.shape[0], (
        f"T mismatch: tracks {tracks_uv.shape[0]}, depths {frame_count}, "
        f"vis {visibility.shape[0]}"
    )
    assert tracks_uv.shape[1] == visibility.shape[1], (
        "N mismatch between tracks and vis"
    )

    if extrinsics_by_frame is not None:
        assert len(extrinsics_by_frame) == frame_count, (
            f"E_seq length {len(extrinsics_by_frame)} != T {frame_count}"
        )

    visible_mask = (
        visibility > visibility_threshold
        if visibility.dtype != np.bool_
        else visibility
    )
    point_count = int(tracks_uv.shape[1])

    override = None
    if depth_samples_override is not None:
        override = np.asarray(depth_samples_override, dtype=np.float32)
        if override.shape != (frame_count, point_count):
            raise ValueError(
                "depth_samples_override must be [T,N] aligned to tracks; "
                f"got {override.shape}, expected "
                f"{(frame_count, point_count)}"
            )

    positions_camera = np.full(
        (frame_count, point_count, 3),
        np.nan,
        dtype=np.float32,
    )
    positions_world = np.full(
        (frame_count, point_count, 3),
        np.nan,
        dtype=np.float32,
    )
    depth_valid_mask = np.zeros(
        (frame_count, point_count),
        dtype=bool,
    )
    valid_mask = np.zeros(
        (frame_count, point_count),
        dtype=bool,
    )

    frames: List[Dict[str, Any]] = []
    for frame_index in range(frame_count):
        frame_camera = Camera(
            K=camera.K,
            E=(
                extrinsics_by_frame[frame_index]
                if extrinsics_by_frame is not None
                else camera.E
            ),
        )

        frame_points_camera: List[List[float]] = []
        frame_points_world: List[List[float]] = []
        point_ids: List[int] = []
        frame_tracks = tracks_uv[frame_index]
        frame_visible = visible_mask[frame_index]

        for point_id in range(frame_tracks.shape[0]):
            if not frame_visible[point_id]:
                continue

            u = float(frame_tracks[point_id, 0])
            v = float(frame_tracks[point_id, 1])
            if u < 0 or v < 0 or u >= width or v >= height:
                continue

            override_depth = (
                float(override[frame_index, point_id])
                if (
                    override is not None
                    and np.isfinite(override[frame_index, point_id])
                    and override[frame_index, point_id] > 1e-9
                )
                else float("nan")
            )
            if np.isfinite(override_depth):
                point_camera = np.array(
                    [
                        (u - frame_camera.cx) / frame_camera.fx * override_depth,
                        -(v - frame_camera.cy) / frame_camera.fy * override_depth,
                        override_depth,
                    ],
                    dtype=float,
                )
            else:
                point_camera = frame_camera.pixel_to_cam(
                    u,
                    v,
                    depth_maps[frame_index],
                    bilinear=bilinear_depth,
                )

            if not np.isfinite(point_camera).all() or point_camera[2] <= 1e-9:
                continue
            depth_valid_mask[frame_index, point_id] = True

            point_world = frame_camera.cam_to_world(point_camera)
            positions_camera[frame_index, point_id] = np.asarray(
                point_camera,
                dtype=np.float32,
            )
            positions_world[frame_index, point_id] = np.asarray(
                point_world,
                dtype=np.float32,
            )
            valid_mask[frame_index, point_id] = True

            frame_points_camera.append(point_camera.tolist())
            frame_points_world.append(point_world.tolist())
            point_ids.append(point_id)

        frames.append(
            {
                "frame": int(frame_index),
                "points_camera": frame_points_camera,
                "points_world": frame_points_world,
                "point_ids": point_ids,
                "valid_mask": valid_mask[frame_index].tolist(),
            }
        )

    override_payload = (
        np.asarray(override, dtype=np.float32).copy()
        if override is not None
        else np.full(
            (frame_count, point_count),
            np.nan,
            dtype=np.float32,
        )
    )
    return LiftedPointTrajectory(
        frames=frames,
        positions_camera=positions_camera,
        positions_world=positions_world,
        query_tracks_uv=np.asarray(tracks_uv, dtype=np.float32).copy(),
        tracker_visibility=np.asarray(visibility).copy(),
        tracker_visible_mask=np.asarray(visible_mask, dtype=bool).copy(),
        depth_samples_override=override_payload,
        depth_valid_mask=depth_valid_mask,
        valid_mask=valid_mask,
    )


__all__ = [
    "LiftedPointTrajectory",
    "lift_tracks_to_3d",
]
