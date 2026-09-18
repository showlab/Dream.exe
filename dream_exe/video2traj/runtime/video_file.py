"""Raw-video-file wrapper for the prepared single-EEF runtime.

This layer only decodes an explicit video path, adapts an explicit camera
bundle to the decoded dimensions, maps the safe single-EEF portions of an
explicit pipeline configuration, and delegates to
``run_single_eef_video2traj``.  It does not resolve benchmark identities,
discover assets, construct model backends, write outputs by default, or claim
full trajectory-pipeline parity.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...transforms import parse_X_wb
from ..depth.estimator import DEFAULT_DEPTH_PRESET
from ..geometry.camera import make_camera_for_frames
from .config import load_pipeline_config
from ..pose.estimation import (
    PoseEstimator,
    pose_backend_execution_requested,
)
from .single_eef import (
    ArtifactCallback,
    SingleEEFStageError,
    run_single_eef_video2traj,
)
from ..media.video import read_video_frames


VideoReader = Callable[..., tuple[Any, float]]


def _video_read_settings_from_normalized(
    normalized: Mapping[str, Any],
) -> dict[str, Any]:
    input_config = dict(normalized.get("input", {}) or {})
    return {
        "process_length": int(input_config.get("process_length", -1) or -1),
        "target_fps": float(input_config.get("target_fps", -1) or -1),
        "max_res": int(input_config.get("max_res", 1280) or 1280),
    }


def resolve_single_eef_video_read_settings(
    pipeline_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the three decode settings exactly like current extract_traj."""

    normalized = load_pipeline_config(
        dict(pipeline_config),
    )
    return _video_read_settings_from_normalized(normalized)


def _merge_owned_options(
    supplied: Mapping[str, Any] | None,
    owned: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    supplied_options = dict(supplied or {})
    overlap = sorted(set(supplied_options).intersection(owned))
    if overlap:
        raise ValueError(
            f"{label} cannot replace pipeline-config-owned "
            "inputs: " + ", ".join(overlap)
        )
    return {
        **dict(owned),
        **supplied_options,
    }


def _apply_generated_video_depth_defaults(
    depth_config: Mapping[str, Any],
    *,
    selected_video: str,
) -> dict[str, Any]:
    """Apply current implementation generated-video depth defaults without choosing a fine-tune."""

    resolved = dict(depth_config or {})
    if str(selected_video or "").strip().lower() != "gen":
        return resolved

    base_config = dict(resolved.get("base", {}) or {})
    if not bool(base_config.get("enabled", False)):
        base_config["enabled"] = True
        resolved["base"] = base_config

    return resolved


def _depth_options_from_pipeline(
    normalized: Mapping[str, Any],
    *,
    video_settings: Mapping[str, Any],
) -> dict[str, Any]:
    input_config = dict(normalized.get("input", {}) or {})
    selected_video = (
        str(
            input_config.get(
                "selected_video",
                "rollout",
            )
            or "rollout"
        )
        .strip()
        .lower()
    )
    depth_config = _apply_generated_video_depth_defaults(
        dict(normalized.get("depth", {}) or {}),
        selected_video=selected_video,
    )
    depth_model = str(
        depth_config.get(
            "model",
            DEFAULT_DEPTH_PRESET,
        )
        or DEFAULT_DEPTH_PRESET
    )
    depth_config_path = str(depth_config.get("config_path", "") or "")
    use_rollout_gt_depth = bool(
        depth_config.get(
            "use_rollout_gt_depth",
            depth_config.get(
                "use_reference_depth",
                False,
            ),
        )
    )
    if selected_video == "gen":
        use_rollout_gt_depth = False
    target_depth_config = dict(depth_config.get("target_calibrated_lift", {}) or {})
    require_raw_model_depths = (
        bool(target_depth_config.get("enabled", False))
        and str(
            target_depth_config.get(
                "source_stage",
                "raw_model",
            )
            or "raw_model"
        )
        == "raw_model"
    )

    return {
        "depth_model": depth_model,
        "depth_config_request": (depth_config_path or depth_model),
        "depth_base_cfg": dict(depth_config.get("base", {}) or {}),
        "selected_video": selected_video,
        "use_rollout_gt_depth": (use_rollout_gt_depth),
        "rollout_gt_depth_path": str(
            depth_config.get(
                "rollout_gt_depth_path",
                depth_config.get(
                    "reference_depth_path",
                    "",
                ),
            )
            or ""
        ),
        "use_depth_cache": bool(depth_config.get("use_cache", True)),
        "estimated_depth_cache_path": str(
            depth_config.get(
                "estimated_depth_cache_path",
                depth_config.get("cache_path", ""),
            )
            or ""
        ),
        "estimated_depth_cache_meta_path": str(
            depth_config.get(
                "estimated_depth_cache_meta_path",
                depth_config.get("cache_meta_path", ""),
            )
            or ""
        ),
        "force_recompute_depth": bool(
            depth_config.get(
                "force_recompute",
                False,
            )
        ),
        "require_raw_model_depths": require_raw_model_depths,
        "trust_cache_on_signature_mismatch": bool(
            depth_config.get(
                "trust_cache_on_signature_mismatch",
                False,
            )
        ),
        "video_target_fps": float(video_settings["target_fps"]),
        "video_process_length": int(video_settings["process_length"]),
    }


def _geometry_options_from_pipeline(
    normalized: Mapping[str, Any],
    simulator_config: Mapping[str, Any],
) -> dict[str, Any]:
    eef_geometry = dict(
        dict(normalized.get("geometry", {}) or {}).get("targets", {}).get("eef", {})
        or {}
    )
    options: dict[str, Any] = {
        "center_options": {
            "interpolate": bool(
                eef_geometry.get(
                    "interpolate",
                    True,
                )
            ),
            "max_gap": int(eef_geometry.get("max_gap", 10)),
            "fill_ends": bool(
                eef_geometry.get(
                    "fill_ends",
                    False,
                )
            ),
            "end_max": int(eef_geometry.get("end_max", 3)),
            "carry_previous": bool(
                eef_geometry.get(
                    "carry_prev",
                    True,
                )
            ),
            "min_points": int(eef_geometry.get("min_points", 1)),
        }
    }
    transform = (
        dict(simulator_config).get("derived", {}).get("eef", {}).get("X_wb", None)
    )
    if transform is not None:
        rotation, translation = parse_X_wb(transform)
        options.update(
            {
                "rotation_world_base": rotation,
                "translation_world_base": translation,
            }
        )
    return options


def _emit(
    callback: ArtifactCallback | None,
    stage: str,
    payload: Any,
) -> None:
    if callback is None:
        return
    try:
        callback(stage, payload)
    except Exception as error:
        raise SingleEEFStageError(
            f"{stage}.artifact_callback",
            error,
        ) from error


def prepare_explicit_video_file(
    *,
    video_path: str | Path,
    simulator_config: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    video_reader: VideoReader = read_video_frames,
    video_backend: str = "auto",
    video_reader_options: Mapping[str, Any] | None = None,
    artifact_callback: ArtifactCallback | None = None,
) -> dict[str, Any]:
    """Decode one explicit video exactly once and prepare its camera.

    This public preparation boundary is shared by single-EEF and multi-object
    callables.  It performs no benchmark lookup, model construction, or output
    publication.
    """

    if not isinstance(simulator_config, Mapping):
        raise TypeError("simulator_config must be an explicit mapping")
    if not isinstance(pipeline_config, Mapping):
        raise TypeError("pipeline_config must be an explicit mapping")
    normalized = load_pipeline_config(
        dict(pipeline_config),
    )
    video_settings = _video_read_settings_from_normalized(normalized)
    reader_kwargs = _merge_owned_options(
        video_reader_options,
        {
            **video_settings,
            "backend": str(video_backend or "auto"),
        },
        label="video_reader_options",
    )
    frames, decoded_fps = video_reader(
        str(video_path),
        **reader_kwargs,
    )
    first_frame = frames[0]
    frame_height, frame_width = (
        int(first_frame.shape[0]),
        int(first_frame.shape[1]),
    )
    camera = make_camera_for_frames(
        dict(simulator_config),
        frame_width,
        frame_height,
    )
    video_payload = {
        "path": str(video_path),
        "frame_count": int(len(frames)),
        "frame_width": frame_width,
        "frame_height": frame_height,
        "fps": float(decoded_fps),
        "algorithm_fps": int(decoded_fps),
        "read_settings": dict(video_settings),
    }
    _emit(
        artifact_callback,
        "video",
        video_payload,
    )
    _emit(
        artifact_callback,
        "camera",
        camera,
    )
    return {
        "frames": frames,
        "decoded_fps": float(decoded_fps),
        "algorithm_fps": int(decoded_fps),
        "camera": camera,
        "video_file": video_payload,
        "video_settings": dict(video_settings),
        "pipeline_config": normalized,
    }


def run_single_eef_video_file(
    *,
    video_path: str | Path,
    simulator_config: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    region_runtime: Any,
    tracking_backend: Any,
    depth_estimator: Any = None,
    depth_calibration: Any = None,
    pose_estimator: PoseEstimator | None = None,
    pose_backend: Any = None,
    rigid_pose_backend: Any = None,
    region_options: Mapping[str, Any] | None = None,
    tracking_options: Mapping[str, Any] | None = None,
    depth_options: Mapping[str, Any] | None = None,
    geometry_options: Mapping[str, Any] | None = None,
    projection_options: Mapping[str, Any] | None = None,
    pose_options: Mapping[str, Any] | None = None,
    video_reader: VideoReader = read_video_frames,
    video_backend: str = "auto",
    video_reader_options: Mapping[str, Any] | None = None,
    region_output_dir: str | Path = "",
    write_region_artifacts: bool = False,
    tracking_output_dir: str | Path = "",
    metadata: Mapping[str, Any] | None = None,
    artifact_callback: ArtifactCallback | None = None,
) -> dict[str, Any]:
    """Decode one explicit video file and run the single-EEF algorithm slice."""

    prepared_video = prepare_explicit_video_file(
        video_path=video_path,
        simulator_config=simulator_config,
        pipeline_config=pipeline_config,
        video_reader=video_reader,
        video_backend=video_backend,
        video_reader_options=video_reader_options,
        artifact_callback=artifact_callback,
    )
    frames = prepared_video["frames"]
    decoded_fps = float(prepared_video["decoded_fps"])
    camera = prepared_video["camera"]
    video_payload = dict(prepared_video["video_file"])
    video_settings = dict(prepared_video["video_settings"])
    normalized = dict(prepared_video["pipeline_config"])

    region_config = dict(normalized.get("region", {}) or {})
    eef_region_config = dict(
        dict(region_config.get("targets", {}) or {}).get("eef", {}) or {}
    )
    resolved_region_options = _merge_owned_options(
        region_options,
        {
            "target_config": eef_region_config,
        },
        label="region_options",
    )

    tracking_config = dict(normalized.get("tracking", {}) or {})
    eef_tracking_config = dict(
        dict(tracking_config.get("targets", {}) or {}).get("eef", {}) or {}
    )
    num_points = int(
        eef_tracking_config.get(
            "num_points",
            150,
        )
    )

    resolved_depth_options = _merge_owned_options(
        depth_options,
        _depth_options_from_pipeline(
            normalized,
            video_settings=video_settings,
        ),
        label="depth_options",
    )
    resolved_geometry_options = _merge_owned_options(
        geometry_options,
        _geometry_options_from_pipeline(
            normalized,
            simulator_config,
        ),
        label="geometry_options",
    )
    alignment_config = dict(normalized.get("alignment", {}) or {})
    resolved_projection_options = _merge_owned_options(
        projection_options,
        {
            "projection_method": str(
                alignment_config.get(
                    "method",
                    "translation",
                )
                or "translation"
            ),
            "smooth_alpha": -1,
        },
        label="projection_options",
    )

    pose_config = dict(normalized.get("pose", {}) or {})
    pose_enabled = bool(pose_config.get("enabled", False))
    resolved_pose_estimator = None
    resolved_pose_backend = None
    resolved_rigid_pose_backend = None
    resolved_pose_config = None
    if pose_enabled:
        if (
            pose_estimator is None
            and str(
                pose_config.get(
                    "config_path",
                    "",
                )
                or ""
            ).strip()
        ):
            raise ValueError(
                "pipeline pose.config_path is not resolved "
                "by this wrapper; inject a configured "
                "pose_estimator"
            )
        backend_free_config = {
            key: value
            for key, value in pose_config.items()
            if key
            not in {
                "enabled",
                "config_path",
            }
        }
        if pose_estimator is None and pose_backend is None:
            requires_backend = pose_backend_execution_requested(backend_free_config)
            supports_backend_free_pose = bool(
                backend_free_config.get(
                    "kabsch_enabled",
                    False,
                )
                or backend_free_config.get(
                    "anchor_rotation_constraint_enabled",
                    False,
                )
                or backend_free_config.get(
                    "use_config_init_pose",
                    False,
                )
            )
            if requires_backend or not supports_backend_free_pose:
                raise RuntimeError(
                    "pipeline pose.enabled=true requires an "
                    "injected pose_estimator or pose_backend unless its "
                    "explicit policy is backend-free Kabsch/config pose"
                )
        resolved_pose_estimator = pose_estimator
        resolved_pose_backend = pose_backend
        resolved_rigid_pose_backend = rigid_pose_backend
        if pose_estimator is None:
            resolved_pose_config = backend_free_config
    elif rigid_pose_backend is not None:
        raise ValueError("rigid_pose_backend requires pipeline pose.enabled=true")

    runtime_device = str(
        dict(normalized.get("runtime", {}) or {}).get(
            "device",
            "cuda",
        )
        or "cuda"
    )
    result = run_single_eef_video2traj(
        frames=frames,
        camera=camera,
        simulator_config=simulator_config,
        region_runtime=region_runtime,
        tracking_backend=tracking_backend,
        target_fps=int(decoded_fps),
        num_points=num_points,
        region_options=resolved_region_options,
        region_output_dir=region_output_dir,
        write_region_artifacts=(write_region_artifacts),
        tracking_options=tracking_options,
        tracking_output_dir=tracking_output_dir,
        depth_estimator=depth_estimator,
        depth_calibration=depth_calibration,
        depth_options=resolved_depth_options,
        geometry_options=(resolved_geometry_options),
        projection_options=(resolved_projection_options),
        pose_estimator=resolved_pose_estimator,
        pose_backend=resolved_pose_backend,
        rigid_pose_backend=resolved_rigid_pose_backend,
        pose_config=resolved_pose_config,
        pose_device=runtime_device,
        pose_options=pose_options,
        metadata=metadata,
        artifact_callback=artifact_callback,
    )
    result["scope"] = {
        **dict(result["scope"]),
        "input": "explicit_video_file",
    }
    result["video_file"] = video_payload
    result["mapped_pipeline"] = {
        "source": dict(normalized.get("_meta", {}) or {}).get(
            "source", "<explicit mapping>"
        ),
        "selected_video": dict(normalized.get("input", {}) or {}).get(
            "selected_video", "rollout"
        ),
        "video_read": dict(video_settings),
        "eef_num_points": int(num_points),
        "depth": {
            key: resolved_depth_options[key]
            for key in (
                "depth_model",
                "depth_config_request",
                "selected_video",
                "use_rollout_gt_depth",
                "use_depth_cache",
                "force_recompute_depth",
            )
        },
        "pose_enabled": pose_enabled,
        "mapped_sections": [
            "input.decode",
            "runtime.device",
            "region.targets.eef",
            "tracking.targets.eef",
            "depth.canonical_source",
            "geometry.targets.eef",
            "alignment",
            "pose",
        ],
        "full_pipeline_parity": False,
    }
    return result


__all__ = [
    "VideoReader",
    "prepare_explicit_video_file",
    "resolve_single_eef_video_read_settings",
    "run_single_eef_video_file",
]
