"""Callable, simulator-independent single-EEF video-to-trajectory slice.

This module composes the prepared/full-algorithm path that is already present
in :mod:`dream_exe.video2traj`: region selection, 2D tracking, depth,
3D lifting/visual-center geometry, optional pose, and EEF projection.  It does
not resolve benchmark identities or paths, initialize a simulator, write
artifacts by default, or claim object, gripper, action, execution, or full
legacy-pipeline parity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..geometry.camera import Camera
from ..depth.source import run_depth_source
from ..pose.estimation import (
    PoseEstimator,
    controller_pose_camera_from_config,
    run_pose_estimation,
)
from ..geometry.prepared import (
    assemble_eef_trajectory_from_geometry,
    build_visual_geometry_from_tracks,
)
from ..tracking.core import run_tracking_backend


ArtifactCallback = Callable[[str, Any], None]
LiftDepthTransform = Callable[
    [Mapping[str, Any]],
    Mapping[str, Any],
]


class SingleEEFStageError(RuntimeError):
    """Identify the failed stage while preserving the original exception."""

    def __init__(self, stage: str, cause: Exception) -> None:
        self.stage = str(stage)
        self.cause = cause
        super().__init__(f"single-EEF video2traj stage '{self.stage}' failed: {cause}")


def _recoverable_pose_failure_reason(
    error: SingleEEFStageError,
) -> str | None:
    """Return the current position-only fallback reason, when applicable.

    The maintained pipeline treats a solver-level pose rejection as an
    optional-orientation failure: position geometry remains valid and the
    trajectory continues without a pose sidecar.  Configuration, shape, and
    artifact-publication errors remain strict failures.
    """

    if error.stage != "pose" or not isinstance(error.cause, RuntimeError):
        return None
    reason = str(error.cause)
    if "Pose estimation failed" not in reason and "degenerate_shape" not in reason:
        return None
    return reason


@contextmanager
def _stage(name: str) -> Iterator[None]:
    try:
        yield
    except SingleEEFStageError:
        raise
    except Exception as error:
        raise SingleEEFStageError(name, error) from error


def _emit_artifact(
    callback: ArtifactCallback | None,
    stage: str,
    payload: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(str(stage), payload)
    except Exception as error:
        raise SingleEEFStageError(
            f"{stage}.artifact_callback",
            error,
        ) from error


def _options(
    value: Mapping[str, Any] | None,
    *,
    label: str,
    reserved: set[str],
) -> dict[str, Any]:
    options = dict(value or {})
    overlap = sorted(reserved.intersection(options))
    if overlap:
        raise ValueError(
            f"{label} cannot replace runtime-owned inputs: " + ", ".join(overlap)
        )
    return options


def _normalize_frames(frames: Sequence[Any]) -> list[np.ndarray]:
    try:
        normalized = [np.asarray(frame) for frame in frames]
    except TypeError as error:
        raise ValueError("frames must be a finite RGB frame sequence") from error
    if not normalized:
        raise ValueError("frames must not be empty")
    reference_shape = normalized[0].shape
    if len(reference_shape) != 3 or reference_shape[-1] != 3:
        raise ValueError(
            f"frames must contain RGB arrays [H,W,3], got {reference_shape}"
        )
    for frame_index, frame in enumerate(normalized):
        if frame.shape != reference_shape:
            raise ValueError(
                "all frames must share one shape; "
                f"frame 0={reference_shape}, "
                f"frame {frame_index}={frame.shape}"
            )
    return normalized


def _region_value(
    result: Any,
    name: str,
) -> Any:
    if isinstance(result, Mapping):
        if name not in result:
            raise ValueError(f"region result is missing '{name}'")
        return result[name]
    if not hasattr(result, name):
        raise ValueError(f"region result is missing '{name}'")
    return getattr(result, name)


def _run_region_selector(
    runtime: Any,
    kwargs: Mapping[str, Any],
) -> Any:
    select_target = getattr(
        runtime,
        "select_target",
        None,
    )
    if callable(select_target):
        return select_target(**dict(kwargs))
    if callable(runtime):
        return runtime(**dict(kwargs))
    raise TypeError("region_runtime must be callable or expose select_target(...)")


def _tracking_numpy(
    value: Any,
    *,
    dtype: Any | None = None,
) -> np.ndarray:
    """Detach tensor-like backend output without importing its framework."""

    detached = getattr(value, "detach", None)
    if callable(detached):
        value = detached()
    cpu = getattr(value, "cpu", None)
    if callable(cpu):
        value = cpu()
    to_numpy = getattr(value, "numpy", None)
    if callable(to_numpy):
        value = to_numpy()
    return np.asarray(value, dtype=dtype)


def _validate_tracking(
    tracks: Any,
    visibility: Any,
    *,
    frame_count: int,
) -> tuple[np.ndarray, np.ndarray]:
    tracks_array = _tracking_numpy(
        tracks,
        dtype=np.float32,
    )
    visibility_array = _tracking_numpy(visibility)
    if tracks_array.ndim == 4 and tracks_array.shape[0] == 1:
        tracks_array = tracks_array[0]
    if visibility_array.ndim == 3 and visibility_array.shape[0] == 1:
        visibility_array = visibility_array[0]
    if tracks_array.ndim != 3 or tracks_array.shape[-1] != 2:
        raise ValueError(
            f"tracking backend must return tracks [T,N,2], got {tracks_array.shape}"
        )
    if visibility_array.ndim != 2:
        raise ValueError(
            "tracking backend must return visibility [T,N], "
            f"got {visibility_array.shape}"
        )
    if tracks_array.shape[:2] != visibility_array.shape:
        raise ValueError(
            "track/visibility alignment mismatch: "
            f"tracks={tracks_array.shape}, "
            f"visibility={visibility_array.shape}"
        )
    if tracks_array.shape[0] != int(frame_count):
        raise ValueError(
            "track/frame alignment mismatch: "
            f"tracks={tracks_array.shape[0]}, "
            f"frames={frame_count}"
        )
    if tracks_array.shape[1] <= 0:
        raise ValueError("tracking backend returned no EEF points")
    return tracks_array, visibility_array


def _validate_depths(
    depths: Any,
    *,
    frames: Sequence[np.ndarray],
) -> np.ndarray:
    depth_array = np.asarray(depths, dtype=np.float32)
    expected = (
        len(frames),
        int(frames[0].shape[0]),
        int(frames[0].shape[1]),
    )
    if depth_array.shape != expected:
        raise ValueError(
            "depth/frame alignment mismatch: "
            f"depths={depth_array.shape}, expected={expected}"
        )
    return depth_array


def run_single_eef_video2traj(
    *,
    frames: Sequence[Any],
    camera: Camera,
    simulator_config: Mapping[str, Any],
    region_runtime: Any,
    tracking_backend: Any,
    target_fps: float,
    num_points: int = 50,
    region_options: Mapping[str, Any] | None = None,
    region_output_dir: str | Path = "",
    write_region_artifacts: bool = False,
    tracking_options: Mapping[str, Any] | None = None,
    tracking_output_dir: str | Path = "",
    depth_estimator: Any = None,
    depth_calibration: Any = None,
    depth_options: Mapping[str, Any] | None = None,
    lift_depth_transform: LiftDepthTransform | None = None,
    geometry_options: Mapping[str, Any] | None = None,
    projection_options: Mapping[str, Any] | None = None,
    pose_estimator: PoseEstimator | None = None,
    pose_backend: Any = None,
    rigid_pose_backend: Any = None,
    pose_config: Any = None,
    pose_device: str = "cuda",
    pose_options: Mapping[str, Any] | None = None,
    pose_register_mask: Any = None,
    pose_register_mask_source: str = "",
    metadata: Mapping[str, Any] | None = None,
    artifact_callback: ArtifactCallback | None = None,
) -> dict[str, Any]:
    """Run one explicit EEF stream without resolving simulator/bench state.

    Concrete model adapters are caller-owned.  ``artifact_callback`` is the
    only output hook in this orchestration layer.  Region files are disabled
    unless ``write_region_artifacts=True``; tracking-adapter side effects are
    controlled by the injected backend and its explicit output directory.
    """

    frame_list = _normalize_frames(frames)
    if not isinstance(simulator_config, Mapping):
        raise TypeError("simulator_config must be an explicit mapping")
    if int(num_points) <= 0:
        raise ValueError("num_points must be positive")
    if bool(write_region_artifacts) and not str(region_output_dir).strip():
        raise ValueError(
            "region_output_dir is required when write_region_artifacts=True"
        )
    pose_requested = any(
        value is not None
        for value in (
            pose_estimator,
            pose_backend,
            rigid_pose_backend,
            pose_config,
        )
    )
    if pose_estimator is not None and pose_config is not None:
        raise ValueError(
            "pose_config cannot be supplied with an already-configured pose_estimator"
        )
    if pose_backend is not None and pose_estimator is None and pose_config is None:
        raise ValueError(
            "pose_config is required with a standalone "
            "pose_backend so its solver policy is explicit"
        )
    if (
        rigid_pose_backend is not None
        and pose_estimator is None
        and pose_config is None
    ):
        raise ValueError(
            "pose_config is required with a standalone "
            "rigid_pose_backend so its solver policy is explicit"
        )
    if pose_register_mask is not None and not pose_requested:
        raise ValueError("pose_register_mask requires an enabled explicit pose runtime")
    supplied_pose_mask_source = str(pose_register_mask_source or "").strip()
    if pose_register_mask is not None and not supplied_pose_mask_source:
        raise ValueError(
            "pose_register_mask_source is required with pose_register_mask"
        )
    if pose_register_mask is None and supplied_pose_mask_source:
        raise ValueError("pose_register_mask_source requires pose_register_mask")

    region_kwargs = _options(
        region_options,
        label="region_options",
        reserved={
            "target_name",
            "frame_rgb",
            "output_dir",
            "num_points",
            "environment_config",
            "camera",
            "write_artifacts",
        },
    )
    region_kwargs.update(
        {
            "target_name": "eef",
            "frame_rgb": frame_list[0],
            "output_dir": str(region_output_dir),
            "num_points": int(num_points),
            "environment_config": dict(simulator_config),
            "camera": camera,
            "write_artifacts": bool(write_region_artifacts),
        }
    )
    with _stage("region"):
        region_result = _run_region_selector(
            region_runtime,
            region_kwargs,
        )
        if region_result is None:
            raise RuntimeError("EEF region selector returned no region")
        region_bbox = [
            int(value)
            for value in _region_value(
                region_result,
                "final_bbox_xyxy",
            )
        ]
        if len(region_bbox) != 4:
            raise ValueError("EEF region bbox must contain four values")
        query_points = np.asarray(
            _region_value(
                region_result,
                "sampled_points_xy",
            ),
            dtype=np.float32,
        )
        if (
            query_points.ndim != 2
            or query_points.shape[1] != 2
            or query_points.shape[0] <= 0
        ):
            raise ValueError(
                "EEF region sampled_points_xy must have "
                f"shape [N,2], got {query_points.shape}"
            )
        region_mask = np.asarray(
            _region_value(region_result, "mask"),
            dtype=bool,
        )
        if region_mask.shape != frame_list[0].shape[:2]:
            raise ValueError(
                "EEF region mask/frame mismatch: "
                f"mask={region_mask.shape}, "
                f"frame={frame_list[0].shape[:2]}"
            )
        try:
            sampling_mask = np.asarray(
                _region_value(
                    region_result,
                    "sampling_mask",
                ),
                dtype=bool,
            )
        except ValueError:
            sampling_mask = region_mask
        if sampling_mask.shape != region_mask.shape:
            raise ValueError(
                "EEF sampling mask/frame mismatch: "
                f"mask={sampling_mask.shape}, "
                f"frame={region_mask.shape}"
            )
        if pose_register_mask is None:
            resolved_pose_register_mask = region_mask
            resolved_pose_register_mask_source = "trajectory_region_mask"
        else:
            resolved_pose_register_mask = np.asarray(
                pose_register_mask,
                dtype=bool,
            )
            if resolved_pose_register_mask.shape != region_mask.shape:
                raise ValueError(
                    "pose register mask/frame mismatch: "
                    f"mask={resolved_pose_register_mask.shape}, "
                    f"frame={region_mask.shape}"
                )
            if not np.any(resolved_pose_register_mask):
                raise ValueError("pose_register_mask must contain at least one pixel")
            resolved_pose_register_mask_source = supplied_pose_mask_source
        _emit_artifact(
            artifact_callback,
            "region",
            region_result,
        )

    tracker_kwargs = _options(
        tracking_options,
        label="tracking_options",
        reserved={
            "video_frames",
            "output_dir",
            "region_bbox_xyxy",
            "num_points",
            "query_points_xy",
            "segmentation_mask",
        },
    )
    tracker_kwargs.setdefault("filename", "eef_points")
    tracker_kwargs.setdefault("seed", 42)
    tracker_kwargs.setdefault("query_mode", "points")
    tracker_kwargs.setdefault("grid_size", 0)
    requested_query_mode = (
        str(tracker_kwargs.get("query_mode", "auto") or "auto").strip().lower()
    )
    with _stage("tracking"):
        tracking_output = run_tracking_backend(
            tracking_backend,
            video_frames=frame_list,
            output_dir=str(tracking_output_dir),
            region_bbox_xyxy=region_bbox,
            num_points=int(query_points.shape[0]),
            query_points_xy=(
                None
                if requested_query_mode in {"bbox_center", "mask_grid"}
                else query_points
            ),
            segmentation_mask=sampling_mask,
            **tracker_kwargs,
        )
        tracks_array, visibility_array = _validate_tracking(
            tracking_output.tracks,
            tracking_output.visibility,
            frame_count=len(frame_list),
        )
        tracking_payload = {
            "tracks_uv": tracks_array,
            "visibility": visibility_array,
            "resolved_query_points_xy": (tracking_output.resolved_query_points_xy),
            "effective_query_mode": (tracking_output.effective_query_mode),
            "provider": dict(tracking_output.provider),
        }
        _emit_artifact(
            artifact_callback,
            "tracking",
            tracking_payload,
        )

    depth_kwargs = _options(
        depth_options,
        label="depth_options",
        reserved={
            "frames",
            "target_fps",
            "estimator",
            "calibration",
        },
    )
    depth_kwargs.setdefault("depth_model", "")
    depth_kwargs.setdefault(
        "depth_config_request",
        "",
    )
    depth_kwargs.setdefault("depth_base_cfg", {})
    depth_kwargs.setdefault("selected_video", "")
    depth_kwargs.setdefault("use_depth_cache", False)
    with _stage("depth"):
        (
            depths,
            depth_source,
            depth_info,
            depth_stage_stats,
        ) = run_depth_source(
            frames=frame_list,
            target_fps=float(target_fps),
            estimator=depth_estimator,
            calibration=depth_calibration,
            **depth_kwargs,
        )
        depth_array = _validate_depths(
            depths,
            frames=frame_list,
        )
        canonical_depth_info = dict(depth_info)
        raw_model_depths = canonical_depth_info.pop(
            "raw_model_depths",
            None,
        )
        raw_model_depth_array = (
            None
            if raw_model_depths is None
            else _validate_depths(
                raw_model_depths,
                frames=frame_list,
            )
        )
        depth_payload = {
            "depths": depth_array,
            "raw_model_depths": raw_model_depth_array,
            "source": str(depth_source),
            "info": canonical_depth_info,
            "stage_stats": dict(depth_stage_stats),
        }
        _emit_artifact(
            artifact_callback,
            "depth",
            depth_payload,
        )

    lift_depth_payload = {
        "depths": depth_array,
        "source": "canonical",
        "metadata": {},
    }
    if lift_depth_transform is not None:
        with _stage("lift_depth"):
            transformed = lift_depth_transform(depth_payload)
            if not isinstance(transformed, Mapping):
                raise TypeError("lift_depth_transform must return a mapping")
            transformed_payload = dict(transformed)
            lift_depth_array = _validate_depths(
                transformed_payload.get(
                    "depths",
                    depth_array,
                ),
                frames=frame_list,
            )
            raw_metadata = transformed_payload.get(
                "metadata",
                {},
            )
            if raw_metadata is None:
                raw_metadata = {}
            if not isinstance(raw_metadata, Mapping):
                raise TypeError("lift_depth_transform metadata must be a mapping")
            lift_depth_payload = {
                "depths": lift_depth_array,
                "source": str(
                    transformed_payload.get(
                        "source",
                        "explicit_transform",
                    )
                    or ""
                ),
                "metadata": dict(raw_metadata),
            }
            _emit_artifact(
                artifact_callback,
                "lift_depth",
                lift_depth_payload,
            )

    geometry_kwargs = _options(
        geometry_options,
        label="geometry_options",
        reserved={
            "tracks_uv",
            "visibility",
            "depths",
            "camera",
        },
    )
    with _stage("geometry"):
        geometry = build_visual_geometry_from_tracks(
            tracks_uv=tracks_array,
            visibility=visibility_array,
            depths=lift_depth_payload["depths"],
            camera=camera,
            **geometry_kwargs,
        )
        if len(geometry["points"]) != len(frame_list):
            raise ValueError(
                "geometry/frame alignment mismatch: "
                f"geometry={len(geometry['points'])}, "
                f"frames={len(frame_list)}"
            )
        _emit_artifact(
            artifact_callback,
            "geometry",
            geometry,
        )

    rotation_world_base = geometry_kwargs.get(
        "rotation_world_base",
        None,
    )
    translation_world_base = geometry_kwargs.get(
        "translation_world_base",
        None,
    )
    projection_runtime_options = _options(
        projection_options,
        label="projection_options",
        reserved={
            "geometry",
            "camera",
            "config",
            "rotation_world_base",
            "translation_world_base",
            "pose_records",
            "metadata",
        },
    )
    projection_runtime_options.setdefault(
        "projection_method",
        "translation",
    )
    projection_runtime_options.setdefault(
        "smooth_alpha",
        -1,
    )
    projection_kwargs = {
        "geometry": geometry,
        "camera": camera,
        "config": dict(simulator_config),
        "rotation_world_base": (rotation_world_base),
        "translation_world_base": (translation_world_base),
        "metadata": dict(metadata or {}),
        **projection_runtime_options,
    }
    pose_payload: dict[str, Any] | None = None
    pose_fallback: dict[str, str] | None = None
    if pose_requested:
        with _stage("projection"):
            prepared_positions = assemble_eef_trajectory_from_geometry(
                **projection_kwargs,
            )
        resolved_pose_estimator = (
            pose_estimator
            if pose_estimator is not None
            else PoseEstimator(
                config=pose_config,
                backend_runner=pose_backend,
                rigid_backend_runner=rigid_pose_backend,
                device=str(pose_device or "cuda"),
            )
        )
        pose_kwargs = _options(
            pose_options,
            label="pose_options",
            reserved={
                "video_frames",
                "depths",
                "cam",
                "eef_mask",
                "point_traj",
                "position_traj",
                "controller_pose_cam",
                "R_wb",
                "t_wb",
                "backend_runner",
                "rigid_backend_runner",
                "estimator",
            },
        )
        try:
            with _stage("pose"):
                raw_pose_payload = run_pose_estimation(
                    estimator=resolved_pose_estimator,
                    backend_runner=pose_backend,
                    rigid_backend_runner=rigid_pose_backend,
                    video_frames=frame_list,
                    depths=depth_array,
                    cam=camera,
                    eef_mask=resolved_pose_register_mask,
                    point_traj=geometry["points"],
                    position_traj=prepared_positions["trajectory"]["eef_controller"],
                    controller_pose_cam=(
                        controller_pose_camera_from_config(
                            dict(simulator_config),
                            camera,
                        )
                    ),
                    R_wb=rotation_world_base,
                    t_wb=translation_world_base,
                    **pose_kwargs,
                )
                if not isinstance(
                    raw_pose_payload,
                    Mapping,
                ):
                    raise TypeError("pose stage must return a mapping")
                pose_payload = dict(raw_pose_payload)
                pose_records = list(pose_payload.get("poses", []) or [])
                if len(pose_records) != len(frame_list):
                    raise ValueError(
                        "pose/frame alignment mismatch: "
                        f"pose={len(pose_records)}, "
                        f"frames={len(frame_list)}"
                    )
                _emit_artifact(
                    artifact_callback,
                    "pose",
                    pose_payload,
                )
        except SingleEEFStageError as error:
            fallback_reason = _recoverable_pose_failure_reason(error)
            if fallback_reason is None:
                raise
            pose_payload = None
            pose_fallback = {
                "fallback": "position_only",
                "reason": fallback_reason,
            }
            _emit_artifact(
                artifact_callback,
                "pose.fallback",
                pose_fallback,
            )
            prepared = prepared_positions
        else:
            with _stage("trajectory"):
                prepared = assemble_eef_trajectory_from_geometry(
                    **projection_kwargs,
                    pose_records=pose_records,
                )
    else:
        with _stage("trajectory"):
            prepared = assemble_eef_trajectory_from_geometry(
                **projection_kwargs,
            )

    _emit_artifact(
        artifact_callback,
        "trajectory",
        prepared["trajectory"],
    )
    return {
        "scope": {
            "target": "eef",
            "path": "prepared_single_eef",
            "pose_enabled": bool(pose_requested),
            "full_pipeline_parity": False,
        },
        "region": region_result,
        "tracking": tracking_payload,
        "depth": depth_payload,
        "lift_depth": lift_depth_payload,
        "geometry": prepared["geometry"],
        "pose": pose_payload,
        "pose_fallback": pose_fallback,
        "pose_register_mask_source": (
            resolved_pose_register_mask_source if pose_requested else None
        ),
        "trajectory": prepared["trajectory"],
        "same_tcp_and_controller": prepared["same_tcp_and_controller"],
    }


__all__ = [
    "ArtifactCallback",
    "LiftDepthTransform",
    "SingleEEFStageError",
    "run_single_eef_video2traj",
]
