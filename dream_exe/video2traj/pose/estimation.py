"""Pure pose trajectory processing with injectable backend inference."""

from __future__ import annotations

import json
from collections.abc import Mapping
from copy import deepcopy
from typing import Any, Callable, Dict, List, Optional

import numpy as np
from scipy.spatial.transform import Rotation

from ..geometry.camera import Camera
from ..geometry.eef_projection import pose_from_bundle
from .backends.tracked_pointcloud_rigid import (
    compute_kabsch_candidates,
    fit_rigid_transform,
)
from .config import (
    MODEL_POSE_BACKENDS,
    POINTCLOUD_KABSCH_BACKEND,
    PoseConfig,
    effective_pose_backend,
    external_pose_backend_id,
    load_pose_config,
    pose_config_to_dict,
)
from .contract import (
    POSE_BACKEND_CONTRACT_VERSION,
    PoseBackend,
    PoseBackendRequest,
    PoseCandidate,
    PosePrediction,
    RigidPoseCandidateBackend,
    RigidPoseRequest,
    normalize_pose_matrix,
    normalize_rigid_pose_backend_identity,
    pose_backend_identity,
    rigid_pose_backend_identity,
    validate_tracked_point_trajectory,
)


PoseBackendCallable = Callable[
    [PoseBackendRequest],
    PosePrediction,
]
RigidPoseBackendCallable = Callable[
    [RigidPoseRequest],
    PosePrediction,
]
_POSE_CORRECTION_NOT_LOADED = object()
POSE_CACHE_INPUT_IDENTITY_SCHEMA = "dream-exe.pose-cache-input-identity"


def pose_backend_execution_requested(
    config: PoseConfig | Dict[str, Any],
) -> bool:
    """Return whether the configured pose policy must invoke a model backend."""

    policy = load_pose_config(config) if isinstance(config, dict) else config
    return effective_pose_backend(policy) != POINTCLOUD_KABSCH_BACKEND


def controller_pose_camera_from_config(
    config: Dict[str, Any],
    camera: Camera,
) -> Optional[np.ndarray]:
    """Convert the saved controller-reference world pose to camera space."""

    try:
        position_world, rotation_world = pose_from_bundle(
            config,
            "controller_ref",
        )
    except (KeyError, TypeError, ValueError):
        return None
    if rotation_world is None:
        return None
    rotation_world_camera = np.asarray(
        camera.E.R_w2c,
        dtype=np.float64,
    ).reshape(3, 3)
    translation_world_camera = np.asarray(
        camera.E.t_w2c,
        dtype=np.float64,
    ).reshape(3)
    pose_camera = np.eye(4, dtype=np.float64)
    pose_camera[:3, :3] = rotation_world_camera @ np.asarray(
        rotation_world,
        dtype=np.float64,
    ).reshape(3, 3)
    pose_camera[:3, 3] = (
        rotation_world_camera
        @ np.asarray(
            position_world,
            dtype=np.float64,
        ).reshape(3)
        + translation_world_camera
    )
    return pose_camera


def coerce_pose_correction_matrix(
    payload: Any,
) -> Optional[np.ndarray]:
    if payload is None:
        return None
    if isinstance(payload, str) and not payload.strip():
        return None
    if isinstance(payload, (list, tuple, dict)) and len(payload) == 0:
        return None
    if isinstance(payload, dict):
        for key in (
            "pose_correction_matrix",
            "X_backend_to_controller",
            "X_backend_to_output",
            "X_model_to_controller",
            "matrix",
            "transform",
            "X",
        ):
            if key in payload:
                return coerce_pose_correction_matrix(payload[key])
        return None
    array = np.asarray(payload, dtype=np.float64)
    if array.shape != (4, 4):
        raise ValueError(f"pose correction matrix must be 4x4, got shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("pose correction matrix contains non-finite values")
    rotation = array[:3, :3]
    determinant = float(np.linalg.det(rotation))
    if abs(determinant - 1.0) > 5e-2:
        raise ValueError(
            f"pose correction rotation determinant is not near 1: {determinant:.6f}"
        )
    return array.reshape(4, 4)


def resolve_pose_correction(
    config: PoseConfig,
    *,
    loaded_payload: Any = _POSE_CORRECTION_NOT_LOADED,
    loaded_source: str = "",
) -> tuple[Optional[np.ndarray], Dict[str, Any]]:
    """Resolve inline or explicitly preloaded correction data without I/O."""

    matrix = coerce_pose_correction_matrix(config.pose_correction_matrix)
    source = "inline" if matrix is not None else ""
    payload_was_loaded = loaded_payload is not _POSE_CORRECTION_NOT_LOADED
    if matrix is None and payload_was_loaded:
        matrix = coerce_pose_correction_matrix(loaded_payload)
        if matrix is None:
            raise ValueError(
                "No 4x4 pose correction matrix found in "
                f"{loaded_source or 'explicit payload'}"
            )
        source = str(loaded_source or "explicit")
    if matrix is None and str(config.pose_correction_path or "").strip():
        raise ValueError(
            "pose.pose_correction_path must be loaded by an outer "
            "adapter and passed as loaded_payload"
        )
    return matrix, {
        "pose_correction_applied": bool(matrix is not None),
        "pose_correction_side": str(config.pose_correction_side or "right"),
        "pose_correction_source": source,
    }


def apply_pose_correction(
    pose_cam: np.ndarray,
    correction: Optional[np.ndarray],
    *,
    side: str,
) -> np.ndarray:
    pose_matrix = np.asarray(
        pose_cam,
        dtype=np.float64,
    ).reshape(4, 4)
    if correction is None:
        return pose_matrix

    correction_matrix = np.asarray(
        correction,
        dtype=np.float64,
    ).reshape(4, 4)
    operands = (
        (correction_matrix, pose_matrix)
        if side == "left"
        else (pose_matrix, correction_matrix)
    )
    return operands[0] @ operands[1]


def apply_pose_correction_to_candidate(
    candidate: Optional[PoseCandidate],
    correction: Optional[np.ndarray],
    *,
    side: str,
) -> Optional[PoseCandidate]:
    if candidate is None or candidate.pose_cam is None or correction is None:
        return candidate

    normalized_input = normalize_pose_matrix(candidate.pose_cam)
    if normalized_input is None:
        return candidate

    adjusted_pose = apply_pose_correction(
        normalized_input,
        correction,
        side=side,
    )
    normalized_output = normalize_pose_matrix(adjusted_pose)
    quality = float(candidate.pose_quality)
    if normalized_output is None:
        rejected_metadata = dict(candidate.meta)
        rejected_metadata["reason"] = "pose_correction_invalid_pose"
        return PoseCandidate(
            pose_cam=None,
            valid=False,
            source=candidate.source,
            pose_quality=quality,
            meta=rejected_metadata,
        )

    metadata = dict(candidate.meta)
    metadata.update(
        {
            "pose_correction_applied": True,
            "pose_correction_side": side or "right",
        }
    )
    return PoseCandidate(
        pose_cam=normalized_output,
        valid=candidate.valid,
        source=candidate.source,
        pose_quality=quality,
        meta=metadata,
    )


def rotation_angle_rad(
    previous: Optional[np.ndarray],
    current: Optional[np.ndarray],
) -> float:
    if previous is None or current is None:
        return 0.0
    delta = (
        np.asarray(
            current,
            dtype=np.float64,
        ).reshape(3, 3)
        @ np.asarray(
            previous,
            dtype=np.float64,
        )
        .reshape(3, 3)
        .T
    )
    cosine = float((np.trace(delta) - 1.0) * 0.5)
    return float(np.arccos(np.clip(cosine, -1.0, 1.0)))


def _rotmat_from_rotvec(
    rotvec: np.ndarray,
) -> np.ndarray:
    vector = np.asarray(
        rotvec,
        dtype=np.float64,
    ).reshape(3)
    theta = float(np.linalg.norm(vector))
    if theta <= 1e-12:
        return np.eye(3, dtype=np.float64)
    axis = vector / theta
    x, y, z = axis.tolist()
    skew = np.array(
        [
            [0.0, -z, y],
            [z, 0.0, -x],
            [-y, x, 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + np.sin(theta) * skew
        + (1.0 - np.cos(theta)) * (skew @ skew)
    )


def clamp_rotation_delta(
    previous: np.ndarray,
    current: np.ndarray,
    max_angle_rad: float,
) -> np.ndarray:
    previous = np.asarray(
        previous,
        dtype=np.float64,
    ).reshape(3, 3)
    current = np.asarray(
        current,
        dtype=np.float64,
    ).reshape(3, 3)
    angle = rotation_angle_rad(previous, current)
    if angle <= float(max_angle_rad) or angle <= 1e-12:
        return current
    delta = current @ previous.T
    sine = float(np.sin(angle))
    if abs(sine) <= 1e-8:
        return previous
    axis = np.array(
        [
            delta[2, 1] - delta[1, 2],
            delta[0, 2] - delta[2, 0],
            delta[1, 0] - delta[0, 1],
        ],
        dtype=np.float64,
    ) / (2.0 * sine)
    axis_norm = float(np.linalg.norm(axis))
    if axis_norm <= 1e-12:
        return previous
    step = _rotmat_from_rotvec(axis / axis_norm * float(max_angle_rad))
    return step @ previous


def apply_anchor_rotation_constraint(
    *,
    backend_candidate: Optional[PoseCandidate],
    kabsch_candidate: Optional[PoseCandidate],
    config: PoseConfig,
) -> Optional[PoseCandidate]:
    if not bool(config.anchor_rotation_constraint_enabled):
        return backend_candidate
    if (
        kabsch_candidate is None
        or not bool(kabsch_candidate.valid)
        or kabsch_candidate.pose_cam is None
    ):
        return backend_candidate

    kabsch_pose = normalize_pose_matrix(kabsch_candidate.pose_cam)
    if kabsch_pose is None:
        return backend_candidate
    kabsch_meta = dict(kabsch_candidate.meta or {})
    kabsch_quality = float(kabsch_candidate.pose_quality)
    shape_ratio = float(kabsch_meta.get("shape_ratio", 0.0) or 0.0)
    if kabsch_quality < float(
        config.anchor_rotation_min_quality
    ) or shape_ratio < float(config.min_shape_ratio):
        return backend_candidate

    if (
        backend_candidate is None
        or not bool(backend_candidate.valid)
        or backend_candidate.pose_cam is None
    ):
        return PoseCandidate(
            pose_cam=kabsch_pose.copy(),
            valid=True,
            source="anchor_rotation_kabsch_fallback",
            pose_quality=kabsch_quality,
            meta={
                **kabsch_meta,
                "anchor_rotation_source": "kabsch_fallback",
                "anchor_rotation_delta_deg": None,
            },
        )

    backend_pose = normalize_pose_matrix(backend_candidate.pose_cam)
    if backend_pose is None:
        return backend_candidate
    delta_deg = float(
        np.rad2deg(
            rotation_angle_rad(
                backend_pose[:3, :3],
                kabsch_pose[:3, :3],
            )
        )
    )
    if delta_deg <= float(config.anchor_rotation_max_backend_kabsch_delta_deg):
        return backend_candidate

    fused_pose = backend_pose.copy()
    fused_pose[:3, :3] = kabsch_pose[:3, :3]
    backend_meta = dict(backend_candidate.meta or {})
    return PoseCandidate(
        pose_cam=fused_pose,
        valid=True,
        source=(f"{backend_candidate.source}_anchor_rotation"),
        pose_quality=max(
            float(backend_candidate.pose_quality),
            kabsch_quality,
        ),
        meta={
            **backend_meta,
            "num_correspondences": kabsch_meta.get(
                "num_correspondences",
                backend_meta.get("num_correspondences"),
            ),
            "num_inliers": kabsch_meta.get(
                "num_inliers",
                backend_meta.get("num_inliers"),
            ),
            "residual_mean_m": kabsch_meta.get(
                "residual_mean_m",
                backend_meta.get("residual_mean_m"),
            ),
            "residual_median_m": kabsch_meta.get(
                "residual_median_m",
                backend_meta.get("residual_median_m"),
            ),
            "shape_ratio": shape_ratio,
            "anchor_rotation_source": "kabsch",
            "anchor_rotation_delta_deg": delta_deg,
            "anchor_rotation_backend_source": str(backend_candidate.source or ""),
        },
    )


def select_pose_candidate(
    *,
    kabsch_candidate: Optional[PoseCandidate],
    backend_candidate: Optional[PoseCandidate],
    kabsch_enabled: bool,
) -> Optional[PoseCandidate]:
    priorities = (
        (kabsch_enabled, kabsch_candidate),
        (True, backend_candidate),
    )
    for enabled, candidate in priorities:
        if enabled and candidate is not None and candidate.valid:
            return candidate
    if kabsch_candidate is not None:
        return kabsch_candidate
    return backend_candidate


def _quat_wxyz_from_rotation(
    rotation: np.ndarray,
) -> np.ndarray:
    rotation = np.asarray(
        rotation,
        dtype=float,
    ).reshape(3, 3)
    quaternion_xyzw = Rotation.from_matrix(rotation).as_quat()
    return np.asarray(
        [
            quaternion_xyzw[3],
            quaternion_xyzw[0],
            quaternion_xyzw[1],
            quaternion_xyzw[2],
        ],
        dtype=float,
    )


def pack_pose_record(
    *,
    frame_idx: int,
    pose_cam: np.ndarray,
    cam: Camera,
    has_X_wb: bool,
    R_wb: Optional[np.ndarray],
    t_wb: Optional[np.ndarray],
    pose_valid: bool,
    pose_source: str,
    candidate_source: str,
    pose_quality: float,
    orientation_frozen: bool,
    angle_from_prev_raw_rad: float,
    angle_from_prev_applied_rad: float,
    rejection_reason: Optional[str] = None,
    candidate_meta: Optional[Dict[str, Any]] = None,
    pose_correction_applied: bool = False,
    pose_correction_side: str = "",
) -> Dict[str, Any]:
    pose_cam = np.asarray(
        pose_cam,
        dtype=np.float64,
    ).reshape(4, 4)
    rotation_camera_object = pose_cam[:3, :3]
    translation_camera_object = pose_cam[:3, 3]
    rotation_world_object = cam.E.R_c2w @ rotation_camera_object
    translation_world_object = cam.E.R_c2w @ translation_camera_object + cam.E.t_c2w
    record: Dict[str, Any] = {
        "frame": int(frame_idx),
        "pose_valid": bool(pose_valid),
        "pose_source": str(pose_source),
        "pos_world": translation_world_object.tolist(),
        "R_world": rotation_world_object.tolist(),
        "quat_wxyz": _quat_wxyz_from_rotation(rotation_world_object).tolist(),
        "pos_base": None,
        "R_base": None,
        "quat_wxyz_base": None,
    }
    if has_X_wb and R_wb is not None and t_wb is not None:
        rotation_world_base = np.asarray(
            R_wb,
            dtype=float,
        ).reshape(3, 3)
        translation_world_base = np.asarray(
            t_wb,
            dtype=float,
        ).reshape(3)
        rotation_base_object = rotation_world_base.T @ rotation_world_object
        translation_base_object = rotation_world_base.T @ (
            translation_world_object - translation_world_base
        )
        record["pos_base"] = translation_base_object.tolist()
        record["R_base"] = rotation_base_object.tolist()
        record["quat_wxyz_base"] = _quat_wxyz_from_rotation(
            rotation_base_object
        ).tolist()
    record["candidate_source"] = str(candidate_source or "")
    record["pose_quality"] = float(pose_quality)
    record["orientation_frozen"] = bool(orientation_frozen)
    record["angle_from_prev_raw_rad"] = float(angle_from_prev_raw_rad)
    record["angle_from_prev_applied_rad"] = float(angle_from_prev_applied_rad)
    record["rejection_reason"] = rejection_reason
    record["pose_correction_applied"] = bool(pose_correction_applied)
    if pose_correction_side:
        record["pose_correction_side"] = str(pose_correction_side)
    if isinstance(candidate_meta, dict):
        for key in (
            "num_correspondences",
            "num_inliers",
            "residual_mean_m",
            "residual_median_m",
            "shape_ratio",
            "anchor_rotation_source",
            "anchor_rotation_delta_deg",
            "anchor_rotation_backend_source",
        ):
            if key in candidate_meta:
                record[key] = candidate_meta[key]
        for key in (
            "temporal_guarded",
            "temporal_guard_action",
            "temporal_guard_raw_angle_deg",
            "temporal_guard_applied_angle_deg",
            "temporal_guard_max_angle_deg",
        ):
            if key in candidate_meta:
                record[key] = candidate_meta[key]
    return record


def reuse_cached_pose_payload(
    cached_payload: Optional[Dict[str, Any]],
    *,
    force_recompute: bool,
    expected_input_identity: Mapping[str, Any] | None = None,
) -> Optional[Dict[str, Any]]:
    if force_recompute or cached_payload is None:
        return None
    payload = deepcopy(cached_payload)
    if not isinstance(payload, dict) or not isinstance(
        payload.get("poses", None), list
    ):
        raise ValueError("Invalid cached pose payload")
    meta = payload.setdefault("meta", {})
    if not isinstance(meta, dict):
        raise ValueError("Invalid cached pose payload meta")
    expected = normalize_pose_cache_input_identity(
        expected_input_identity,
        label="expected pose-cache input identity",
    )
    if expected is None:
        raise ValueError("cached pose payload requires expected_input_identity")
    actual = normalize_pose_cache_input_identity(
        meta.get("cache_input_identity", None),
        label="cached pose input identity",
    )
    if actual is None:
        raise ValueError("cached pose payload has no bound input identity")
    if actual != expected:
        raise ValueError("cached pose input identity does not match request")
    return payload


def normalize_pose_cache_input_identity(
    value: Mapping[str, Any] | None,
    *,
    label: str = "pose-cache input identity",
) -> dict[str, Any] | None:
    """Validate the caller-owned digest that binds a reusable pose payload.

    The digest is intentionally caller-owned: the orchestration layer can bind
    file or in-memory inputs without making the pose module aware of bench
    paths.  It must cover the complete effective pose request, including its
    geometry, controller pose, configuration, and any model inputs.
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    normalized = deepcopy(dict(value))
    if normalized.get("format") != POSE_CACHE_INPUT_IDENTITY_SCHEMA:
        raise ValueError(
            f"{label}.format must be {POSE_CACHE_INPUT_IDENTITY_SCHEMA!r}"
        )
    digest = str(normalized.get("input_digest_sha256", "") or "").strip()
    if len(digest) != 64 or any(
        character not in "0123456789abcdef" for character in digest
    ):
        raise ValueError(
            f"{label}.input_digest_sha256 must be 64 lowercase hex characters"
        )
    normalized["input_digest_sha256"] = digest
    try:
        json.dumps(
            normalized,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} must be finite JSON data") from error
    return normalized


def _invoke_candidate_backend(
    backend: Any,
    request: Any,
    *,
    expected_count: int,
) -> PosePrediction:
    infer = getattr(backend, "infer", None)
    if callable(infer):
        prediction = infer(request)
    elif callable(backend):
        prediction = backend(request)
    else:
        raise TypeError("pose backend must be callable or expose infer(request)")
    if not isinstance(prediction, PosePrediction):
        raise TypeError("pose backend must return PosePrediction")
    if not isinstance(prediction.candidates, list):
        raise TypeError("PosePrediction.candidates must be a list")
    if len(prediction.candidates) != int(expected_count):
        raise ValueError("pose backend candidate count must match video frame count")
    if not isinstance(prediction.meta, dict):
        raise TypeError("PosePrediction.meta must be a dict")
    try:
        json.dumps(prediction.meta, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError("PosePrediction.meta must be finite JSON data") from error

    for frame_index, candidate in enumerate(prediction.candidates):
        if candidate is None:
            continue
        if not isinstance(candidate, PoseCandidate):
            raise TypeError(
                f"pose candidate {frame_index} must be PoseCandidate or None"
            )
        if not isinstance(candidate.valid, bool):
            raise TypeError(f"pose candidate {frame_index}.valid must be bool")
        if not str(candidate.source).strip():
            raise ValueError(f"pose candidate {frame_index}.source must be non-empty")
        try:
            pose_quality = float(candidate.pose_quality)
        except (TypeError, ValueError):
            pose_quality = float("nan")
        if (
            isinstance(candidate.pose_quality, bool)
            or not np.isfinite(pose_quality)
            or not 0.0 <= pose_quality <= 1.0
        ):
            raise ValueError(
                f"pose candidate {frame_index}.pose_quality must be "
                "finite and within [0, 1]"
            )
        if candidate.valid:
            if normalize_pose_matrix(candidate.pose_cam) is None:
                raise ValueError(f"pose candidate {frame_index} has an invalid pose")
        elif candidate.pose_cam is not None:
            raise ValueError(
                f"invalid pose candidate {frame_index} must not carry a pose"
            )
        if not isinstance(candidate.meta, dict):
            raise TypeError(f"pose candidate {frame_index}.meta must be a dict")
        try:
            json.dumps(candidate.meta, allow_nan=False)
        except (TypeError, ValueError) as error:
            raise ValueError(
                f"pose candidate {frame_index}.meta must be finite JSON data"
            ) from error
    return prediction


def invoke_pose_backend(
    backend: PoseBackend | PoseBackendCallable,
    request: PoseBackendRequest,
    *,
    validate_requested_identity: bool = True,
) -> PosePrediction:
    prediction = _invoke_candidate_backend(
        backend,
        request,
        expected_count=len(request.video_frames),
    )

    external_backend_id = (
        external_pose_backend_id(request.config.backend)
        if validate_requested_identity
        else None
    )
    if external_backend_id is not None:
        runtime_identity = pose_backend_identity(
            backend,
            source="external pose backend",
        )
        expected_identity = {
            "provider_kind": "external",
            "backend_id": external_backend_id,
            "contract_version": POSE_BACKEND_CONTRACT_VERSION,
        }
        if runtime_identity != expected_identity:
            raise ValueError(
                "external pose backend identity conflicts with requested "
                f"identity: runtime={runtime_identity!r}, "
                f"requested={expected_identity!r}"
            )
        prediction_identity = {
            key: prediction.meta.get(key) for key in expected_identity
        }
        if prediction_identity != expected_identity:
            raise ValueError(
                "PosePrediction.meta identity conflicts with external "
                f"provider: prediction={prediction_identity!r}, "
                f"provider={expected_identity!r}"
            )
    return prediction


def _validate_rigid_pose_request(request: RigidPoseRequest) -> None:
    if not isinstance(request, RigidPoseRequest):
        raise TypeError("rigid pose request must be RigidPoseRequest")
    integer_fields = {
        "num_frames": request.num_frames,
        "min_correspondences": request.min_correspondences,
        "min_inlier_correspondences": request.min_inlier_correspondences,
    }
    for field_name, value in integer_fields.items():
        if isinstance(value, (bool, np.bool_)) or not isinstance(
            value,
            (int, np.integer),
        ):
            raise TypeError(f"rigid pose request.{field_name} must be an integer")
    if int(request.num_frames) < 0:
        raise ValueError("rigid pose request.num_frames must be >= 0")
    if int(request.min_correspondences) < 3:
        raise ValueError("rigid pose request.min_correspondences must be >= 3")
    if int(request.min_inlier_correspondences) < 3:
        raise ValueError("rigid pose request.min_inlier_correspondences must be >= 3")
    if int(request.min_inlier_correspondences) > int(request.min_correspondences):
        raise ValueError(
            "rigid pose request.min_inlier_correspondences cannot exceed "
            "min_correspondences"
        )
    for field_name, value, minimum, inclusive in (
        (
            "inlier_threshold_m",
            request.inlier_threshold_m,
            0.0,
            False,
        ),
        (
            "min_shape_ratio",
            request.min_shape_ratio,
            0.0,
            True,
        ),
    ):
        if isinstance(value, (bool, np.bool_)):
            raise TypeError(f"rigid pose request.{field_name} must be numeric")
        try:
            numeric = float(value)
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"rigid pose request.{field_name} must be numeric"
            ) from error
        if not np.isfinite(numeric) or (
            numeric < minimum if inclusive else numeric <= minimum
        ):
            comparator = ">=" if inclusive else ">"
            raise ValueError(
                f"rigid pose request.{field_name} must be finite and "
                f"{comparator} {minimum:g}"
            )


def invoke_rigid_pose_backend(
    backend: RigidPoseCandidateBackend | RigidPoseBackendCallable,
    request: RigidPoseRequest,
) -> tuple[PosePrediction, dict[str, str]]:
    """Invoke a replaceable rigid solver with complete, truthful identity."""

    _validate_rigid_pose_request(request)
    validate_tracked_point_trajectory(
        request.point_trajectory,
        max_frames=int(request.num_frames),
        source="rigid pose request.point_trajectory",
    )
    declared_identity = rigid_pose_backend_identity(
        backend,
        source="rigid pose backend",
    )
    prediction = _invoke_candidate_backend(
        backend,
        request,
        expected_count=int(request.num_frames),
    )
    prediction_identity = normalize_rigid_pose_backend_identity(
        provider_kind=prediction.meta.get("provider_kind"),
        backend_id=prediction.meta.get("backend_id"),
        algorithm_id=prediction.meta.get("algorithm_id"),
        contract_version=prediction.meta.get("contract_version"),
        source="rigid PosePrediction.meta",
    )
    if prediction_identity != declared_identity:
        raise ValueError(
            "rigid PosePrediction.meta identity conflicts with provider "
            f"declaration: prediction={prediction_identity!r}, "
            f"provider={declared_identity!r}"
        )
    return prediction, declared_identity


def compute_pose_trajectory(
    video_frames: Any,
    depths: np.ndarray,
    *,
    cam: Camera,
    eef_mask: np.ndarray,
    config: PoseConfig,
    backend: Optional[PoseBackend | PoseBackendCallable] = None,
    rigid_backend: Optional[
        RigidPoseCandidateBackend | RigidPoseBackendCallable
    ] = None,
    mesh_path: str = "",
    point_traj: Any = None,
    position_traj: Any = None,
    controller_pose_cam: Optional[np.ndarray] = None,
    R_wb: Optional[np.ndarray] = None,
    t_wb: Optional[np.ndarray] = None,
    pose_correction_payload: Any = None,
    pose_correction_source: str = "",
    cached_payload: Optional[Dict[str, Any]] = None,
    cache_input_identity: Mapping[str, Any] | None = None,
    device: str = "cuda",
) -> Dict[str, Any]:
    del position_traj
    normalized_cache_identity = normalize_pose_cache_input_identity(
        cache_input_identity,
    )
    cached = reuse_cached_pose_payload(
        cached_payload,
        force_recompute=bool(config.force_recompute),
        expected_input_identity=normalized_cache_identity,
    )
    if cached is not None:
        return cached

    if pose_correction_payload is None:
        correction, correction_meta = resolve_pose_correction(
            config,
        )
    else:
        correction, correction_meta = resolve_pose_correction(
            config,
            loaded_payload=pose_correction_payload,
            loaded_source=pose_correction_source,
        )
    correction_side = str(config.pose_correction_side or "right").strip().lower()
    controller_pose = (
        None
        if controller_pose_cam is None
        else np.asarray(
            controller_pose_cam,
            dtype=np.float64,
        ).reshape(4, 4)
    )
    seed_pose = controller_pose if bool(config.use_config_init_pose) else None
    frames = list(video_frames)
    num_frames = len(frames)
    backend_predictions: list[Optional[PoseCandidate]] = [None] * num_frames
    backend_meta: Dict[str, Any] = {}
    rigid_provider: dict[str, str] | None = None
    run_backend = pose_backend_execution_requested(config)
    if run_backend:
        if backend is None:
            raise RuntimeError(
                "pose backend execution was requested but no "
                "backend callable was provided"
            )
        request = PoseBackendRequest(
            video_frames=frames,
            depths=np.asarray(depths, dtype=np.float32),
            cam=cam,
            eef_mask=np.asarray(eef_mask, dtype=bool),
            mesh_path=str(mesh_path or config.mesh_path or ""),
            seed_pose_cam=seed_pose,
            force_register_frame0=bool(config.foundationpose_force_register_frame0),
            config=config,
            device=str(device or "cuda"),
            point_trajectory=point_traj,
            reference_pose_cam=controller_pose,
        )
        prediction = invoke_pose_backend(backend, request)
        backend_predictions = list(prediction.candidates or [])
        backend_meta = dict(prediction.meta or {})

    need_kabsch = bool(config.kabsch_enabled) or bool(
        config.anchor_rotation_constraint_enabled
    )
    if need_kabsch:
        from .backends.tracked_pointcloud_rigid import (
            TrackedPointCloudRigidPoseBackend,
        )

        kabsch_backend = (
            rigid_backend
            if rigid_backend is not None
            else TrackedPointCloudRigidPoseBackend()
        )
        kabsch_request = RigidPoseRequest(
            point_trajectory=point_traj,
            reference_pose_cam=controller_pose,
            num_frames=num_frames,
            min_correspondences=int(config.min_correspondences),
            min_inlier_correspondences=int(config.min_inlier_correspondences),
            inlier_threshold_m=float(config.inlier_threshold_m),
            min_shape_ratio=float(config.min_shape_ratio),
        )
        kabsch_prediction, rigid_provider = invoke_rigid_pose_backend(
            kabsch_backend,
            kabsch_request,
        )
        kabsch_predictions = list(kabsch_prediction.candidates or [])
    else:
        kabsch_predictions = [None] * num_frames

    poses: List[Dict[str, Any]] = []
    previous_pose: Optional[np.ndarray] = None
    has_X_wb = R_wb is not None and t_wb is not None
    for frame_idx in range(num_frames):
        if (
            frame_idx == 0
            and seed_pose is not None
            and bool(config.use_config_init_pose)
            and not bool(config.foundationpose_force_register_frame0)
        ):
            selected = PoseCandidate(
                seed_pose,
                True,
                "config_init",
                1.0,
                {},
            )
        else:
            raw_backend_candidate = (
                backend_predictions[frame_idx]
                if frame_idx < len(backend_predictions)
                else None
            )
            backend_candidate = apply_pose_correction_to_candidate(
                raw_backend_candidate,
                correction,
                side=correction_side,
            )
            kabsch_candidate = (
                kabsch_predictions[frame_idx]
                if frame_idx < len(kabsch_predictions)
                else None
            )
            backend_candidate = apply_anchor_rotation_constraint(
                backend_candidate=backend_candidate,
                kabsch_candidate=kabsch_candidate,
                config=config,
            )
            selected = select_pose_candidate(
                kabsch_candidate=kabsch_candidate,
                backend_candidate=backend_candidate,
                kabsch_enabled=bool(config.kabsch_enabled),
            )

        candidate_source = "none" if selected is None else str(selected.source)
        candidate_meta = {} if selected is None else dict(selected.meta or {})
        pose_cam = (
            None if selected is None else normalize_pose_matrix(selected.pose_cam)
        )
        pose_quality = 0.0 if selected is None else float(selected.pose_quality)
        rejection_reason = candidate_meta.get(
            "reason",
            None,
        )
        pose_valid = bool(
            selected is not None and selected.valid and pose_cam is not None
        )
        orientation_frozen = False
        pose_source = candidate_source
        raw_angle = rotation_angle_rad(
            (None if previous_pose is None else previous_pose[:3, :3]),
            (None if pose_cam is None else pose_cam[:3, :3]),
        )
        applied_angle = raw_angle

        if not pose_valid and previous_pose is not None:
            pose_cam = previous_pose.copy()
            rejection_reason = rejection_reason or "candidate_invalid_carry_forward"
            pose_valid = False
            applied_angle = 0.0
        elif not pose_valid:
            raise RuntimeError(
                f"Pose estimation failed at frame={frame_idx}: "
                f"{rejection_reason or candidate_source}"
            )
        elif (
            bool(config.temporal_guard_enabled)
            and previous_pose is not None
            and pose_cam is not None
            and raw_angle > float(np.deg2rad(config.temporal_guard_max_angle_deg))
        ):
            guard_max_rad = float(np.deg2rad(config.temporal_guard_max_angle_deg))
            guard_mode = str(config.temporal_guard_mode or "clamp").strip().lower()
            pose_cam = (
                np.asarray(
                    pose_cam,
                    dtype=np.float64,
                )
                .reshape(4, 4)
                .copy()
            )
            if guard_mode == "freeze":
                pose_cam[:3, :3] = previous_pose[:3, :3]
                applied_angle = 0.0
                orientation_frozen = True
            else:
                pose_cam[:3, :3] = clamp_rotation_delta(
                    previous_pose[:3, :3],
                    pose_cam[:3, :3],
                    guard_max_rad,
                )
                applied_angle = rotation_angle_rad(
                    previous_pose[:3, :3],
                    pose_cam[:3, :3],
                )
            pose_source = f"{candidate_source}_temporal_guard_{guard_mode}"
            candidate_meta.update(
                {
                    "temporal_guarded": True,
                    "temporal_guard_action": guard_mode,
                    "temporal_guard_raw_angle_deg": float(np.rad2deg(raw_angle)),
                    "temporal_guard_applied_angle_deg": float(
                        np.rad2deg(applied_angle)
                    ),
                    "temporal_guard_max_angle_deg": float(
                        config.temporal_guard_max_angle_deg
                    ),
                }
            )

        record = pack_pose_record(
            frame_idx=frame_idx,
            pose_cam=np.asarray(
                pose_cam,
                dtype=np.float64,
            ).reshape(4, 4),
            cam=cam,
            has_X_wb=has_X_wb,
            R_wb=R_wb,
            t_wb=t_wb,
            pose_valid=pose_valid,
            pose_source=(
                pose_source if pose_valid else "carry_forward_after_invalid_pose"
            ),
            candidate_source=candidate_source,
            pose_quality=pose_quality,
            orientation_frozen=orientation_frozen,
            angle_from_prev_raw_rad=raw_angle,
            angle_from_prev_applied_rad=applied_angle,
            rejection_reason=rejection_reason,
            candidate_meta=candidate_meta,
            pose_correction_applied=bool(correction is not None),
            pose_correction_side=(correction_side if correction is not None else ""),
        )
        poses.append(record)
        previous_pose = np.asarray(
            pose_cam,
            dtype=np.float64,
        ).reshape(4, 4)

    provider_chain: list[dict[str, Any]] = []
    if rigid_provider is not None:
        if run_backend:
            provider_chain.append(
                {
                    "role": "primary_pose",
                    "effective_backend": effective_pose_backend(config),
                }
            )
        provider_chain.append(
            {
                "role": "rigid_pose",
                **rigid_provider,
            }
        )

    result = {
        "meta": {
            # ``backend`` is retained as the requested/configured identity for
            # artifact compatibility.  The added fields state what actually
            # executed, including legacy configs that mislabeled Kabsch as
            # FoundationPose.
            "backend": str(config.backend or POINTCLOUD_KABSCH_BACKEND),
            "requested_backend": str(config.backend or POINTCLOUD_KABSCH_BACKEND),
            "effective_backend": effective_pose_backend(config),
            "model_backend_executed": bool(
                run_backend and effective_pose_backend(config) in MODEL_POSE_BACKENDS
            ),
            "mesh_path": str(mesh_path or config.mesh_path or ""),
            "weights_root": str(config.weights_root or ""),
            "device": str(device or "cuda"),
            "num_frames": int(len(poses)),
            "used_config_init_pose": bool(seed_pose is not None),
            "temporal_reject_enabled": False,
            "temporal_filter_enabled": bool(config.temporal_guard_enabled),
            "temporal_guard_enabled": bool(config.temporal_guard_enabled),
            "temporal_guard_max_angle_deg": float(config.temporal_guard_max_angle_deg),
            "temporal_guard_mode": str(config.temporal_guard_mode),
            **correction_meta,
            "pose_config": pose_config_to_dict(config),
            "backend_meta": backend_meta,
            **(
                {
                    "rigid_provider": rigid_provider,
                    "provider_chain": provider_chain,
                }
                if rigid_provider is not None
                else {}
            ),
        },
        "poses": poses,
    }
    if normalized_cache_identity is not None:
        result["meta"]["cache_input_identity"] = normalized_cache_identity
    return result


class PoseEstimator:
    """Small configuration and dependency-injection facade for pose inference."""

    def __init__(
        self,
        *,
        config: PoseConfig | Dict[str, Any] | None = None,
        backend: str = POINTCLOUD_KABSCH_BACKEND,
        device: str = "cuda",
        weights_root: str = "",
        init_refine_iter: int = 5,
        track_refine_iter: int = 2,
        debug: int = 0,
        force_recompute: bool = False,
        backend_runner: Optional[PoseBackend | PoseBackendCallable] = None,
        rigid_backend_runner: Optional[
            RigidPoseCandidateBackend | RigidPoseBackendCallable
        ] = None,
    ) -> None:
        if isinstance(config, PoseConfig):
            resolved_config = config
        elif isinstance(config, dict):
            resolved_config = load_pose_config(config)
        elif config is None:
            resolved_config = load_pose_config(
                {
                    "backend": backend,
                    "weights_root": weights_root,
                    "init_refine_iter": init_refine_iter,
                    "track_refine_iter": track_refine_iter,
                    "debug": debug,
                    "force_recompute": force_recompute,
                }
            )
        else:
            raise TypeError(f"Unsupported pose config type: {type(config).__name__}")

        self.config = resolved_config
        self.device = str(device or "cuda")
        self.backend_runner = backend_runner
        self.rigid_backend_runner = rigid_backend_runner

    def infer(
        self,
        video_frames: Any,
        depths: np.ndarray,
        *,
        cam: Camera,
        eef_mask: np.ndarray,
        mesh_path: str = "",
        point_traj: Any = None,
        position_traj: Any = None,
        controller_pose_cam: Optional[np.ndarray] = None,
        R_wb: Optional[np.ndarray] = None,
        t_wb: Optional[np.ndarray] = None,
        pose_correction_payload: Any = None,
        pose_correction_source: str = "",
        cached_payload: Optional[Dict[str, Any]] = None,
        cache_input_identity: Mapping[str, Any] | None = None,
        backend_runner: Optional[PoseBackend | PoseBackendCallable] = None,
        rigid_backend_runner: Optional[
            RigidPoseCandidateBackend | RigidPoseBackendCallable
        ] = None,
    ) -> Dict[str, Any]:
        runner = self.backend_runner if backend_runner is None else backend_runner
        rigid_runner = (
            self.rigid_backend_runner
            if rigid_backend_runner is None
            else rigid_backend_runner
        )
        return compute_pose_trajectory(
            video_frames,
            depths,
            cam=cam,
            eef_mask=eef_mask,
            config=self.config,
            backend=runner,
            rigid_backend=rigid_runner,
            mesh_path=mesh_path,
            point_traj=point_traj,
            position_traj=position_traj,
            controller_pose_cam=controller_pose_cam,
            R_wb=R_wb,
            t_wb=t_wb,
            pose_correction_payload=pose_correction_payload,
            pose_correction_source=pose_correction_source,
            cached_payload=cached_payload,
            cache_input_identity=cache_input_identity,
            device=self.device,
        )


def run_pose_estimation(
    *,
    estimator: Optional[PoseEstimator] = None,
    backend_runner: Optional[PoseBackend | PoseBackendCallable] = None,
    rigid_backend_runner: Optional[
        RigidPoseCandidateBackend | RigidPoseBackendCallable
    ] = None,
    **kwargs: Any,
) -> Dict[str, Any]:
    pose_estimator = (
        estimator
        if estimator is not None
        else PoseEstimator(
            backend_runner=backend_runner,
            rigid_backend_runner=rigid_backend_runner,
        )
    )
    return pose_estimator.infer(
        backend_runner=backend_runner,
        rigid_backend_runner=rigid_backend_runner,
        **kwargs,
    )


__all__ = [
    "PoseEstimator",
    "apply_anchor_rotation_constraint",
    "apply_pose_correction",
    "apply_pose_correction_to_candidate",
    "clamp_rotation_delta",
    "coerce_pose_correction_matrix",
    "controller_pose_camera_from_config",
    "compute_kabsch_candidates",
    "compute_pose_trajectory",
    "fit_rigid_transform",
    "invoke_pose_backend",
    "invoke_rigid_pose_backend",
    "pack_pose_record",
    "normalize_pose_cache_input_identity",
    "POSE_CACHE_INPUT_IDENTITY_SCHEMA",
    "pose_backend_execution_requested",
    "resolve_pose_correction",
    "reuse_cached_pose_payload",
    "rotation_angle_rad",
    "run_pose_estimation",
    "select_pose_candidate",
]
