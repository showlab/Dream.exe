"""Built-in rigid-pose provider for tracked camera-frame point clouds."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from ..contract import (
    RIGID_POSE_CANDIDATE_CONTRACT_VERSION,
    PoseCandidate,
    PosePrediction,
    RigidPoseRequest,
    validate_tracked_point_trajectory,
)

if TYPE_CHECKING:
    from ..config import PoseConfig


def _transform_from_rotation_translation(
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = np.asarray(
        rotation,
        dtype=np.float64,
    ).reshape(3, 3)
    transform[:3, 3] = np.asarray(
        translation,
        dtype=np.float64,
    ).reshape(3)
    return transform


def _invert_transform(transform: np.ndarray) -> np.ndarray:
    normalized = np.asarray(
        transform,
        dtype=np.float64,
    ).reshape(4, 4)
    rotation = normalized[:3, :3]
    translation = normalized[:3, 3]
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = rotation.T
    inverse[:3, 3] = -rotation.T @ translation
    return inverse


def fit_rigid_transform(
    source_points: np.ndarray,
    destination_points: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Fit one proper rigid transform with the Kabsch/SVD algorithm."""

    source = np.asarray(
        source_points,
        dtype=np.float64,
    ).reshape(-1, 3)
    destination = np.asarray(
        destination_points,
        dtype=np.float64,
    ).reshape(-1, 3)
    source_center = np.mean(source, axis=0)
    destination_center = np.mean(destination, axis=0)
    source_zero = source - source_center.reshape(1, 3)
    destination_zero = destination - destination_center.reshape(1, 3)
    left, _singular_values, right_t = np.linalg.svd(source_zero.T @ destination_zero)
    rotation = right_t.T @ left.T
    if np.linalg.det(rotation) < 0.0:
        right_t[-1, :] *= -1.0
        rotation = right_t.T @ left.T
    translation = destination_center - rotation @ source_center
    return rotation, translation


def _valid_point_mask(
    points: np.ndarray,
    valid_mask: Any = None,
) -> np.ndarray:
    normalized = np.asarray(
        points,
        dtype=np.float64,
    ).reshape(-1, 3)
    mask = np.all(np.isfinite(normalized), axis=1)
    if valid_mask is not None:
        external = np.asarray(
            valid_mask,
            dtype=bool,
        ).reshape(-1)
        if external.shape[0] == mask.shape[0]:
            mask &= external
    return mask


def _point_shape_ratio(points: np.ndarray) -> float:
    normalized = np.asarray(
        points,
        dtype=np.float64,
    ).reshape(-1, 3)
    if normalized.shape[0] < 3 or not np.all(np.isfinite(normalized)):
        return 0.0
    centered = normalized - np.mean(
        normalized,
        axis=0,
        keepdims=True,
    )
    try:
        singular_values = np.linalg.svd(
            centered,
            compute_uv=False,
        )
    except Exception:
        return 0.0
    if singular_values.size < 3 or float(singular_values[0]) <= 1e-9:
        return 0.0
    return float(
        np.clip(
            singular_values[-1] / singular_values[0],
            0.0,
            1.0,
        )
    )


def _compute_kabsch_candidates(
    *,
    point_traj: Any,
    seed_pose_cam: Optional[np.ndarray],
    num_frames: int,
    min_correspondences: int,
    min_inlier_correspondences: int,
    inlier_threshold_m: float,
    min_shape_ratio: float,
) -> list[Optional[PoseCandidate]]:
    """Generate frame-aligned rigid-pose candidates from tracked 3D points."""

    if point_traj is None or seed_pose_cam is None:
        return [None] * int(num_frames)
    records = list(point_traj or [])
    if not records:
        return [None] * int(num_frames)
    seed_pose = np.asarray(
        seed_pose_cam,
        dtype=np.float64,
    ).reshape(4, 4)
    first = dict(records[0] or {})
    first_points = np.asarray(
        first.get("points_camera", []),
        dtype=np.float64,
    ).reshape(-1, 3)
    first_point_ids_raw = list(first.get("point_ids", []) or [])
    first_point_ids = (
        [int(value) for value in first_point_ids_raw]
        if len(first_point_ids_raw) == int(first_points.shape[0])
        else list(range(int(first_points.shape[0])))
    )
    first_id_to_index = {
        point_id: index for index, point_id in enumerate(first_point_ids)
    }
    first_mask = _valid_point_mask(
        first_points,
        first.get("valid_mask", None),
    )
    if int(np.count_nonzero(first_mask)) < int(min_correspondences):
        return [None] * int(num_frames)

    seed_inverse = _invert_transform(seed_pose)
    object_points = (seed_inverse[:3, :3] @ first_points.T).T + seed_inverse[
        :3, 3
    ].reshape(1, 3)
    output: list[Optional[PoseCandidate]] = []
    for frame_index in range(int(num_frames)):
        if frame_index >= len(records):
            output.append(None)
            continue
        record = dict(records[frame_index] or {})
        frame_points = np.asarray(
            record.get("points_camera", []),
            dtype=np.float64,
        ).reshape(-1, 3)
        frame_point_ids_raw = list(record.get("point_ids", []) or [])
        if len(frame_point_ids_raw) == int(frame_points.shape[0]):
            common_pairs = [
                (
                    first_id_to_index[int(point_id)],
                    index,
                )
                for index, point_id in enumerate(frame_point_ids_raw)
                if int(point_id) in first_id_to_index
            ]
            if len(common_pairs) < int(min_correspondences):
                output.append(None)
                continue
            object_index = np.asarray(
                [pair[0] for pair in common_pairs],
                dtype=np.int64,
            )
            frame_point_index = np.asarray(
                [pair[1] for pair in common_pairs],
                dtype=np.int64,
            )
            object_points_i = object_points[object_index]
            frame_points_i = frame_points[frame_point_index]
            mask = first_mask[object_index] & _valid_point_mask(
                frame_points_i,
                None,
            )
        else:
            if frame_points.shape != object_points.shape:
                output.append(None)
                continue
            object_points_i = object_points
            frame_points_i = frame_points
            mask = first_mask & _valid_point_mask(
                frame_points_i,
                record.get("valid_mask", None),
            )

        num_correspondences = int(np.count_nonzero(mask))
        if num_correspondences < int(min_correspondences):
            output.append(None)
            continue
        shape_ratio = min(
            _point_shape_ratio(object_points_i[mask]),
            _point_shape_ratio(frame_points_i[mask]),
        )
        if shape_ratio < float(min_shape_ratio):
            output.append(
                PoseCandidate(
                    pose_cam=None,
                    valid=False,
                    source="kabsch_rejected",
                    pose_quality=0.0,
                    meta={
                        "reason": "degenerate_shape",
                        "num_correspondences": num_correspondences,
                        "num_inliers": 0,
                        "residual_mean_m": None,
                        "residual_median_m": None,
                        "shape_ratio": float(shape_ratio),
                    },
                )
            )
            continue

        rotation, translation = fit_rigid_transform(
            object_points_i[mask],
            frame_points_i[mask],
        )
        pose_cam = _transform_from_rotation_translation(
            rotation,
            translation,
        )
        residual = np.linalg.norm(
            (rotation @ object_points_i[mask].T).T
            + translation.reshape(1, 3)
            - frame_points_i[mask],
            axis=1,
        )
        inliers = residual <= float(inlier_threshold_m)
        num_inliers = int(np.count_nonzero(inliers))
        quality = float(
            np.clip(
                num_inliers / max(num_correspondences, 1),
                0.0,
                1.0,
            )
        )
        if num_inliers < int(min_inlier_correspondences):
            output.append(
                PoseCandidate(
                    pose_cam=None,
                    valid=False,
                    source="kabsch_rejected",
                    pose_quality=quality,
                    meta={
                        "reason": "insufficient_inliers",
                        "num_correspondences": num_correspondences,
                        "num_inliers": num_inliers,
                        "residual_mean_m": (
                            float(np.mean(residual)) if residual.size else None
                        ),
                        "residual_median_m": (
                            float(np.median(residual)) if residual.size else None
                        ),
                        "shape_ratio": float(shape_ratio),
                    },
                )
            )
            continue
        output.append(
            PoseCandidate(
                pose_cam=pose_cam,
                valid=True,
                source="kabsch",
                pose_quality=quality,
                meta={
                    "num_correspondences": num_correspondences,
                    "num_inliers": num_inliers,
                    "residual_mean_m": (
                        float(np.mean(residual)) if residual.size else None
                    ),
                    "residual_median_m": (
                        float(np.median(residual)) if residual.size else None
                    ),
                    "shape_ratio": float(shape_ratio),
                },
            )
        )
    return output


def compute_kabsch_candidates(
    *,
    point_traj: Any,
    seed_pose_cam: Optional[np.ndarray],
    num_frames: int,
    config: PoseConfig,
) -> list[Optional[PoseCandidate]]:
    """Compatibility wrapper over the narrow rigid-candidate implementation."""

    return _compute_kabsch_candidates(
        point_traj=point_traj,
        seed_pose_cam=seed_pose_cam,
        num_frames=num_frames,
        min_correspondences=int(config.min_correspondences),
        min_inlier_correspondences=int(config.min_inlier_correspondences),
        inlier_threshold_m=float(config.inlier_threshold_m),
        min_shape_ratio=float(config.min_shape_ratio),
    )


class TrackedPointCloudRigidPoseBackend:
    """Generate framewise rigid-pose candidates with Kabsch/SVD.

    This provider deliberately stops at candidate generation.  Correction,
    anchor constraints, candidate fusion, temporal guards, and artifact
    packing remain responsibilities of the pose trajectory orchestrator.
    """

    provider_kind = "builtin"
    backend_id = "tracked_pointcloud_rigid"
    algorithm_id = "kabsch_svd"
    contract_version = RIGID_POSE_CANDIDATE_CONTRACT_VERSION

    def infer(self, request: RigidPoseRequest) -> PosePrediction:
        """Return candidates aligned to ``request.num_frames``."""

        tracked_frames = validate_tracked_point_trajectory(
            request.point_trajectory,
            max_frames=int(request.num_frames),
            source="TrackedPointCloudRigidPoseBackend.point_trajectory",
        )

        candidates = _compute_kabsch_candidates(
            point_traj=(
                None
                if tracked_frames is None
                else [frame.as_record() for frame in tracked_frames]
            ),
            seed_pose_cam=request.reference_pose_cam,
            num_frames=int(request.num_frames),
            min_correspondences=int(request.min_correspondences),
            min_inlier_correspondences=int(request.min_inlier_correspondences),
            inlier_threshold_m=float(request.inlier_threshold_m),
            min_shape_ratio=float(request.min_shape_ratio),
        )
        return PosePrediction(
            candidates=candidates,
            meta={
                "provider_kind": self.provider_kind,
                "backend_id": self.backend_id,
                "algorithm_id": self.algorithm_id,
                "contract_version": self.contract_version,
            },
        )


__all__ = [
    "TrackedPointCloudRigidPoseBackend",
    "compute_kabsch_candidates",
    "fit_rigid_transform",
]
