"""Prepared-array geometry composition for video-to-trajectory."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .camera import Camera, CameraExtrinsics
from .eef_projection import (
    derive_tcp_orientation_from_controller,
    fuse_controller_orientation,
    mean_visibility_over_points,
    pack_visual_center_records,
    project_eef_trajectories,
)
from .lifting import lift_tracks_to_3d
from .visual_center import compute_visual_center_trajectory


def build_visual_geometry_from_tracks(
    *,
    tracks_uv: Any,
    visibility: Any,
    depths: Any,
    camera: Camera,
    extrinsics_by_frame: Sequence[CameraExtrinsics] | None = None,
    depth_samples_override: Any = None,
    bilinear_depth: bool = True,
    visibility_threshold: float = 0.5,
    center_options: Mapping[str, Any] | None = None,
    rotation_world_base: Any = None,
    translation_world_base: Any = None,
) -> dict[str, Any]:
    """Lift one aligned track stream and compute its robust visual center."""

    tracks = np.asarray(tracks_uv, dtype=np.float32)
    point_visibility = np.asarray(visibility)
    depth_stack = np.asarray(depths, dtype=np.float32)
    lifted = lift_tracks_to_3d(
        tracks,
        point_visibility,
        depth_stack,
        camera,
        extrinsics_by_frame=extrinsics_by_frame,
        bilinear_depth=bool(bilinear_depth),
        visibility_threshold=float(visibility_threshold),
        depth_samples_override=depth_samples_override,
    )
    options = dict(center_options or {})
    overlap = sorted(
        {
            "visibility_threshold",
        }.intersection(options)
    )
    if overlap:
        raise ValueError(
            "center_options cannot replace reserved inputs: " + ", ".join(overlap)
        )
    centers, globally_kept = compute_visual_center_trajectory(
        tracks,
        point_visibility,
        lifted.frames,
        camera,
        visibility_threshold=float(visibility_threshold),
        **options,
    )
    frame_visibility = mean_visibility_over_points(point_visibility)
    has_world_base = (
        rotation_world_base is not None and translation_world_base is not None
    )
    packed_centers = pack_visual_center_records(
        centers,
        has_X_wb=has_world_base,
        rotation_world_base=(
            None
            if rotation_world_base is None
            else np.asarray(
                rotation_world_base,
                dtype=np.float64,
            ).reshape(3, 3)
        ),
        translation_world_base=(
            None
            if translation_world_base is None
            else np.asarray(
                translation_world_base,
                dtype=np.float64,
            ).reshape(3)
        ),
        visibility=frame_visibility,
    )
    return {
        "points": lifted.frames,
        "positions_camera": lifted.positions_camera,
        "positions_world": lifted.positions_world,
        "query_tracks_uv": lifted.query_tracks_uv,
        "tracker_visibility": lifted.tracker_visibility,
        "tracker_visible_mask": lifted.tracker_visible_mask,
        "depth_samples_override": lifted.depth_samples_override,
        "depth_valid_mask": lifted.depth_valid_mask,
        "valid_mask": lifted.valid_mask,
        "visual_centers": centers,
        "visual_center_records": packed_centers,
        "globally_kept": globally_kept,
        "frame_visibility": frame_visibility,
    }


def build_eef_trajectory_from_tracks(
    *,
    tracks_uv: Any,
    visibility: Any,
    depths: Any,
    camera: Camera,
    config: Mapping[str, Any],
    extrinsics_by_frame: Sequence[CameraExtrinsics] | None = None,
    depth_samples_override: Any = None,
    bilinear_depth: bool = True,
    visibility_threshold: float = 0.5,
    center_options: Mapping[str, Any] | None = None,
    projection_method: str = "translation",
    smooth_alpha: float = -1,
    rotation_world_base: Any = None,
    translation_world_base: Any = None,
    pose_records: Sequence[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build current EEF center/TCP/controller records from prepared arrays."""

    geometry = build_visual_geometry_from_tracks(
        tracks_uv=tracks_uv,
        visibility=visibility,
        depths=depths,
        camera=camera,
        extrinsics_by_frame=extrinsics_by_frame,
        depth_samples_override=depth_samples_override,
        bilinear_depth=bilinear_depth,
        visibility_threshold=visibility_threshold,
        center_options=center_options,
        rotation_world_base=rotation_world_base,
        translation_world_base=translation_world_base,
    )
    return assemble_eef_trajectory_from_geometry(
        geometry=geometry,
        camera=camera,
        config=config,
        projection_method=projection_method,
        smooth_alpha=smooth_alpha,
        rotation_world_base=rotation_world_base,
        translation_world_base=translation_world_base,
        pose_records=pose_records,
        metadata=metadata,
    )


def assemble_eef_trajectory_from_geometry(
    *,
    geometry: Mapping[str, Any],
    camera: Camera,
    config: Mapping[str, Any],
    projection_method: str = "translation",
    smooth_alpha: float = -1,
    rotation_world_base: Any = None,
    translation_world_base: Any = None,
    pose_records: Sequence[Mapping[str, Any]] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Project one already-built EEF geometry payload without relifting it."""

    geometry_payload = dict(geometry)
    missing = sorted(
        {
            "visual_centers",
            "visual_center_records",
        }.difference(geometry_payload)
    )
    if missing:
        raise ValueError("geometry is missing required fields: " + ", ".join(missing))
    projected = project_eef_trajectories(
        geometry_payload["visual_centers"],
        dict(config),
        camera,
        method=str(projection_method),
        smooth_alpha=float(smooth_alpha),
    )
    controller = list(projected["controller"])
    tcp = list(projected["tcp"])
    pose_payload = [dict(record) for record in list(pose_records or [])]
    if pose_records is not None:
        if len(pose_payload) != len(controller):
            raise RuntimeError(
                "Pose sidecar length mismatch: "
                f"pose={len(pose_payload)} "
                f"ctrl_traj={len(controller)}"
            )
        controller = fuse_controller_orientation(
            position_trajectory=controller,
            pose_records=pose_payload,
        )
        if bool(projected["same_tcp_and_controller"]):
            tcp = list(controller)
        else:
            tcp = derive_tcp_orientation_from_controller(
                controller_trajectory=controller,
                config=dict(config),
                has_X_wb=(
                    rotation_world_base is not None
                    and translation_world_base is not None
                ),
                rotation_world_base=(
                    None
                    if rotation_world_base is None
                    else np.asarray(
                        rotation_world_base,
                        dtype=np.float64,
                    ).reshape(3, 3)
                ),
            )

    trajectory = {
        "meta": dict(metadata or {}),
        "visual_center": geometry_payload["visual_center_records"],
        "eef_controller": controller,
        "eef_tcp": tcp,
    }
    return {
        "trajectory": trajectory,
        "geometry": geometry_payload,
        "pose_records": pose_payload,
        "same_tcp_and_controller": bool(projected["same_tcp_and_controller"]),
    }


__all__ = [
    "assemble_eef_trajectory_from_geometry",
    "build_eef_trajectory_from_tracks",
    "build_visual_geometry_from_tracks",
]
