"""Shared, environment-free contracts for pose backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence as SequenceABC
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Protocol, Sequence

import numpy as np

from .config import external_pose_selection

if TYPE_CHECKING:
    from .config import PoseConfig


POSE_BACKEND_CONTRACT_VERSION = "pose_backend"
RIGID_POSE_CANDIDATE_CONTRACT_VERSION = "rigid_pose_candidate"
TRACKED_POINT_COORDINATE_FRAME = "camera_xyz_m"


def _normalized_identity_token(
    value: Any,
    *,
    label: str,
) -> str:
    token = str(value or "").strip().lower()
    if (
        not token
        or token.startswith(".")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in token
        )
    ):
        raise ValueError(f"{label} must use [a-z0-9._-] and be non-empty")
    return token


def normalize_rigid_pose_backend_identity(
    *,
    provider_kind: Any,
    backend_id: Any,
    algorithm_id: Any,
    contract_version: Any,
    source: str,
) -> dict[str, str]:
    """Return the complete identity of one rigid-pose provider."""

    normalized_kind = str(provider_kind or "").strip().lower()
    if normalized_kind not in {"builtin", "external"}:
        raise ValueError(f"{source}.provider_kind must be 'builtin' or 'external'")
    normalized_contract = str(contract_version or "").strip()
    if normalized_contract != RIGID_POSE_CANDIDATE_CONTRACT_VERSION:
        raise ValueError(
            f"{source}.contract_version must be "
            f"{RIGID_POSE_CANDIDATE_CONTRACT_VERSION!r}"
        )
    return {
        "provider_kind": normalized_kind,
        "backend_id": _normalized_identity_token(
            backend_id,
            label=f"{source}.backend_id",
        ),
        "algorithm_id": _normalized_identity_token(
            algorithm_id,
            label=f"{source}.algorithm_id",
        ),
        "contract_version": normalized_contract,
    }


def rigid_pose_backend_identity(
    backend: Any,
    *,
    source: str = "rigid pose backend",
) -> dict[str, str]:
    """Read the versioned provider and algorithm identity of a rigid solver."""

    if not callable(getattr(backend, "infer", None)) and not callable(backend):
        raise TypeError(f"{source} must be callable or expose infer(request)")
    return normalize_rigid_pose_backend_identity(
        provider_kind=getattr(backend, "provider_kind", None),
        backend_id=getattr(backend, "backend_id", None),
        algorithm_id=getattr(backend, "algorithm_id", None),
        contract_version=getattr(backend, "contract_version", None),
        source=source,
    )


def validate_external_pose_runtime_identity(
    runtime_config: Mapping[str, Any],
    *,
    source: str,
    require_effective_backend: bool = False,
) -> dict[str, Any]:
    """Validate and normalize one external pose-backend declaration."""

    if not isinstance(runtime_config, Mapping):
        raise TypeError(f"{source} must be a mapping")
    runtime = dict(runtime_config)
    if str(runtime.get("provider_kind", "") or "").strip().lower() != ("external"):
        raise ValueError(f"{source}.provider_kind must be 'external'")

    effective_backend = external_pose_selection(runtime.get("backend_id", ""))
    backend_id = effective_backend.removeprefix("external:")
    contract_version = str(runtime.get("contract_version", "") or "").strip()
    if contract_version != POSE_BACKEND_CONTRACT_VERSION:
        raise ValueError(
            f"{source}.contract_version must be {POSE_BACKEND_CONTRACT_VERSION!r}"
        )

    declared_effective = str(runtime.get("effective_backend", "") or "").strip().lower()
    if require_effective_backend and not declared_effective:
        raise ValueError(f"{source}.effective_backend must be explicitly provided")
    if declared_effective and declared_effective != effective_backend:
        raise ValueError(
            f"{source}.effective_backend {declared_effective!r} conflicts "
            f"with backend_id {backend_id!r}"
        )

    runtime["provider_kind"] = "external"
    runtime["backend_id"] = backend_id
    runtime["contract_version"] = contract_version
    runtime["effective_backend"] = effective_backend
    return runtime


def pose_backend_identity(
    backend: Any,
    *,
    source: str = "pose backend",
) -> dict[str, str]:
    """Read and validate the identity declared by an external provider."""

    if not callable(getattr(backend, "infer", None)) and not callable(backend):
        raise TypeError(f"{source} must be callable or expose infer(request)")
    runtime = validate_external_pose_runtime_identity(
        {
            "provider_kind": getattr(
                backend,
                "provider_kind",
                None,
            ),
            "backend_id": getattr(backend, "backend_id", None),
            "contract_version": getattr(
                backend,
                "contract_version",
                None,
            ),
        },
        source=source,
    )
    return {
        "provider_kind": str(runtime["provider_kind"]),
        "backend_id": str(runtime["backend_id"]),
        "contract_version": str(runtime["contract_version"]),
    }


@dataclass
class PoseCandidate:
    """One backend pose hypothesis expressed in the camera frame."""

    pose_cam: Optional[np.ndarray]
    valid: bool
    source: str
    pose_quality: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class PosePrediction:
    """Ordered pose hypotheses and backend-level diagnostics."""

    candidates: List[Optional[PoseCandidate]]
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class TrackedPointTrajectoryFrame:
    """Sparse lifted points with a dense, point-id-aligned validity mask.

    ``points_camera[row]`` belongs to ``point_ids[row]``.  ``valid_mask`` is
    indexed by the stable tracker point id, so its true indices must exactly
    match ``point_ids``.  Coordinates use the normalized camera frame in
    metres: +x right, +y up, and finite +z in front of the camera.
    """

    points_camera: np.ndarray
    point_ids: np.ndarray
    valid_mask: np.ndarray
    coordinate_frame: str = TRACKED_POINT_COORDINATE_FRAME

    def as_record(self) -> dict[str, Any]:
        """Return the current JSON-compatible sparse trajectory fields."""

        return {
            "points_camera": self.points_camera.tolist(),
            "point_ids": self.point_ids.tolist(),
            "valid_mask": self.valid_mask.tolist(),
        }


def _tracked_points_array(value: Any, *, source: str) -> np.ndarray:
    try:
        points = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as error:
        raise TypeError(f"{source}.points_camera must be numeric") from error
    if points.size == 0:
        points = np.empty((0, 3), dtype=np.float64)
    if points.ndim != 2 or points.shape[1:] != (3,):
        raise ValueError(
            f"{source}.points_camera must have shape (N, 3), got {points.shape}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError(f"{source}.points_camera must contain only finite values")
    if points.shape[0] and np.any(points[:, 2] <= 1e-9):
        raise ValueError(
            f"{source}.points_camera must use camera_xyz_m with positive z"
        )
    output = np.asarray(points, dtype=np.float64).copy()
    output.setflags(write=False)
    return output


def _tracked_point_ids(
    value: Any,
    *,
    expected_count: int,
    source: str,
) -> np.ndarray:
    try:
        raw_ids = list(value)
    except TypeError as error:
        raise TypeError(
            f"{source}.point_ids must be a one-dimensional integer sequence"
        ) from error
    if any(
        isinstance(point_id, (bool, np.bool_))
        or not isinstance(point_id, (int, np.integer))
        for point_id in raw_ids
    ):
        raise TypeError(f"{source}.point_ids must contain only integers")
    point_ids = np.asarray(raw_ids, dtype=np.int64)
    if point_ids.ndim != 1 or point_ids.shape[0] != expected_count:
        raise ValueError(
            f"{source}.point_ids must have shape ({expected_count},), "
            f"got {point_ids.shape}"
        )
    if np.any(point_ids < 0):
        raise ValueError(f"{source}.point_ids must be non-negative")
    if np.unique(point_ids).shape[0] != point_ids.shape[0]:
        raise ValueError(f"{source}.point_ids must be unique")
    output = point_ids.copy()
    output.setflags(write=False)
    return output


def _tracked_valid_mask(
    value: Any,
    *,
    point_ids: np.ndarray,
    source: str,
) -> np.ndarray:
    try:
        raw_mask = list(value)
    except TypeError as error:
        raise TypeError(
            f"{source}.valid_mask must be a one-dimensional bool sequence"
        ) from error
    if any(not isinstance(flag, (bool, np.bool_)) for flag in raw_mask):
        raise TypeError(f"{source}.valid_mask must contain only bool values")
    valid_mask = np.asarray(raw_mask, dtype=bool)
    if valid_mask.ndim != 1:
        raise ValueError(f"{source}.valid_mask must be one-dimensional")
    if point_ids.size and valid_mask.shape[0] <= int(np.max(point_ids)):
        raise ValueError(
            f"{source}.valid_mask must be indexed by point id and cover "
            f"id {int(np.max(point_ids))}"
        )
    valid_ids = np.flatnonzero(valid_mask)
    if not np.array_equal(
        np.sort(valid_ids),
        np.sort(point_ids),
    ):
        raise ValueError(
            f"{source}.valid_mask true indices must exactly match point_ids"
        )
    output = valid_mask.copy()
    output.setflags(write=False)
    return output


def validate_tracked_point_trajectory_frame(
    value: Any,
    *,
    source: str = "tracked point trajectory frame",
) -> TrackedPointTrajectoryFrame:
    """Validate one current-compatible sparse camera-point record."""

    if isinstance(value, TrackedPointTrajectoryFrame):
        raw = {
            "points_camera": value.points_camera,
            "point_ids": value.point_ids,
            "valid_mask": value.valid_mask,
            "coordinate_frame": value.coordinate_frame,
        }
    elif isinstance(value, Mapping):
        raw = value
    else:
        raise TypeError(f"{source} must be a mapping")
    for required in ("points_camera", "point_ids", "valid_mask"):
        if required not in raw:
            raise ValueError(f"{source}.{required} is required")

    coordinate_frame = (
        str(
            raw.get(
                "coordinate_frame",
                TRACKED_POINT_COORDINATE_FRAME,
            )
            or ""
        )
        .strip()
        .lower()
    )
    if coordinate_frame != TRACKED_POINT_COORDINATE_FRAME:
        raise ValueError(
            f"{source}.coordinate_frame must be {TRACKED_POINT_COORDINATE_FRAME!r}"
        )
    points = _tracked_points_array(
        raw["points_camera"],
        source=source,
    )
    point_ids = _tracked_point_ids(
        raw["point_ids"],
        expected_count=int(points.shape[0]),
        source=source,
    )
    valid_mask = _tracked_valid_mask(
        raw["valid_mask"],
        point_ids=point_ids,
        source=source,
    )
    return TrackedPointTrajectoryFrame(
        points_camera=points,
        point_ids=point_ids,
        valid_mask=valid_mask,
        coordinate_frame=coordinate_frame,
    )


def validate_tracked_point_trajectory(
    value: Any,
    *,
    max_frames: int | None = None,
    source: str = "tracked point trajectory",
) -> tuple[TrackedPointTrajectoryFrame, ...] | None:
    """Validate frame alignment while preserving missing trailing frames."""

    if value is None:
        return None
    if isinstance(value, (str, bytes, bytearray, Mapping)) or not isinstance(
        value,
        SequenceABC,
    ):
        raise TypeError(f"{source} must be a sequence of frame mappings")
    records = tuple(
        validate_tracked_point_trajectory_frame(
            frame,
            source=f"{source}[{frame_index}]",
        )
        for frame_index, frame in enumerate(value)
    )
    if max_frames is not None and len(records) > int(max_frames):
        raise ValueError(
            f"{source} has {len(records)} frames but the request has only "
            f"{int(max_frames)} video frames"
        )
    return records


@dataclass(frozen=True)
class PoseBackendRequest:
    """All explicit inputs needed by a simulator-independent pose backend."""

    video_frames: Sequence[Any]
    depths: np.ndarray
    cam: Any
    eef_mask: np.ndarray
    mesh_path: str
    seed_pose_cam: Optional[np.ndarray]
    force_register_frame0: bool
    config: PoseConfig
    device: str
    point_trajectory: (
        Sequence[TrackedPointTrajectoryFrame | Mapping[str, Any]] | None
    ) = None
    reference_pose_cam: Optional[np.ndarray] = None


@dataclass(frozen=True)
class RigidPoseRequest:
    """Only the inputs needed to generate tracked-point rigid candidates."""

    point_trajectory: Sequence[TrackedPointTrajectoryFrame | Mapping[str, Any]] | None
    reference_pose_cam: Optional[np.ndarray]
    num_frames: int
    min_correspondences: int
    min_inlier_correspondences: int
    inlier_threshold_m: float
    min_shape_ratio: float


class PoseBackend(Protocol):
    """Structural interface implemented by concrete pose providers."""

    def infer(self, request: PoseBackendRequest) -> PosePrediction:
        """Return pose candidates for the supplied video."""


class BasePoseBackend(ABC):
    """Minimal simulator-independent external pose provider template."""

    provider_kind = "external"
    backend_id = ""
    contract_version = POSE_BACKEND_CONTRACT_VERSION

    @abstractmethod
    def infer(self, request: PoseBackendRequest) -> PosePrediction:
        """Return frame-aligned pose candidates for ``request``."""


class RigidPoseCandidateBackend(Protocol):
    """Structural interface for model-free rigid candidate providers."""

    provider_kind: str
    backend_id: str
    algorithm_id: str
    contract_version: str

    def infer(self, request: RigidPoseRequest) -> PosePrediction:
        """Return candidates aligned to ``request.num_frames``."""


def normalize_pose_matrix(pose_cam: Any) -> Optional[np.ndarray]:
    """Return a finite 4x4 pose with a plausibly rigid rotation block."""

    if pose_cam is None:
        return None
    try:
        matrix = np.asarray(pose_cam, dtype=np.float64)
    except (TypeError, ValueError):
        return None
    if matrix.shape != (4, 4) or not np.all(np.isfinite(matrix)):
        return None
    try:
        determinant = float(np.linalg.det(matrix[:3, :3]))
    except np.linalg.LinAlgError:
        return None
    if not 0.95 < determinant < 1.05:
        return None
    return matrix.copy()


def invalid_candidate(
    source: str,
    *,
    reason: str,
    pose_quality: float = 0.0,
    **meta: Any,
) -> PoseCandidate:
    """Create an invalid candidate while retaining diagnostic context."""

    details = dict(meta)
    details["reason"] = str(reason)
    return PoseCandidate(
        pose_cam=None,
        valid=False,
        source=str(source),
        pose_quality=float(pose_quality),
        meta=details,
    )


def pose_candidate_from_cam(
    pose_cam: Any,
    *,
    source: str,
    pose_quality: float = 0.55,
    **meta: Any,
) -> PoseCandidate:
    """Normalize a camera-frame pose into a valid or diagnostic candidate."""

    matrix = normalize_pose_matrix(pose_cam)
    if matrix is None:
        return invalid_candidate(
            source,
            reason="invalid_pose_matrix",
            **meta,
        )
    return PoseCandidate(
        pose_cam=matrix,
        valid=True,
        source=str(source),
        pose_quality=float(pose_quality),
        meta=dict(meta),
    )


def _finite_number(value: Any) -> Optional[float]:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if np.isfinite(number) else None


def score_to_unit_interval(
    score: Any,
    *,
    default: float = 0.55,
) -> float:
    """Map a finite score from [-1, 1] to [0, 1]."""

    value = _finite_number(score)
    if value is None:
        return float(default)
    midpoint = 0.5 * (value + 1.0)
    if midpoint <= 0.0:
        return 0.0
    if midpoint >= 1.0:
        return 1.0
    return float(midpoint)


def mask_to_bbox(mask: np.ndarray) -> Optional[np.ndarray]:
    """Return an exclusive-max XYXY box around a non-empty 2D mask."""

    if mask is None:
        return None
    array = np.asarray(mask)
    if array.ndim != 2:
        return None
    rows, columns = np.nonzero(array)
    if rows.size == 0:
        return None
    return np.asarray(
        [
            int(columns.min()),
            int(rows.min()),
            int(columns.max()) + 1,
            int(rows.max()) + 1,
        ],
        dtype=np.int32,
    )


def project_points(
    pose_cam: np.ndarray,
    points_obj: np.ndarray,
    K: np.ndarray,
) -> np.ndarray:
    """Project object-frame points with positive camera depth into pixels."""

    pose = np.asarray(pose_cam, dtype=np.float64)
    points = np.asarray(points_obj, dtype=np.float64).reshape(-1, 3)
    intrinsics = np.asarray(K, dtype=np.float64)
    camera_points = points @ pose[:3, :3].T + pose[:3, 3]
    camera_points = camera_points[camera_points[:, 2] > 1e-6]
    if camera_points.shape[0] == 0:
        return np.empty((0, 2), dtype=np.float64)
    homogeneous = camera_points @ intrinsics.T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def project_mesh_bbox(
    *,
    pose_cam: np.ndarray,
    points_obj: np.ndarray,
    K: np.ndarray,
    image_shape: tuple[int, int],
    padding_px: int = 8,
) -> Optional[np.ndarray]:
    """Project mesh vertices and clip their padded box to an image."""

    pixels = project_points(pose_cam, points_obj, K)
    finite = pixels[np.all(np.isfinite(pixels), axis=1)]
    if finite.shape[0] == 0:
        return None

    height, width = (int(image_shape[0]), int(image_shape[1]))
    padding = int(padding_px)
    lower = np.floor(np.min(finite, axis=0)).astype(np.int64) - padding
    upper = np.ceil(np.max(finite, axis=0)).astype(np.int64) + padding
    x0 = int(np.clip(lower[0], 0, width - 1))
    y0 = int(np.clip(lower[1], 0, height - 1))
    x1 = int(np.clip(upper[0], 0, width - 1))
    y1 = int(np.clip(upper[1], 0, height - 1))
    if x1 <= x0 or y1 <= y0:
        return None
    return np.asarray([x0, y0, x1, y1], dtype=np.int32)


__all__ = [
    "BasePoseBackend",
    "POSE_BACKEND_CONTRACT_VERSION",
    "RIGID_POSE_CANDIDATE_CONTRACT_VERSION",
    "TRACKED_POINT_COORDINATE_FRAME",
    "PoseBackend",
    "PoseBackendRequest",
    "PoseCandidate",
    "PosePrediction",
    "RigidPoseCandidateBackend",
    "RigidPoseRequest",
    "TrackedPointTrajectoryFrame",
    "invalid_candidate",
    "mask_to_bbox",
    "normalize_pose_matrix",
    "normalize_rigid_pose_backend_identity",
    "pose_backend_identity",
    "pose_candidate_from_cam",
    "project_mesh_bbox",
    "project_points",
    "rigid_pose_backend_identity",
    "score_to_unit_interval",
    "validate_external_pose_runtime_identity",
    "validate_tracked_point_trajectory",
    "validate_tracked_point_trajectory_frame",
]
