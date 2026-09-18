"""Prepared-frame object evidence for the standalone video2traj core.

The current extraction command processes one object stream per unique
manipulated-object identity, even when several semantic stages share that
object.  This module exposes that vertical slice without benchmark, simulator,
VLM, UID/path discovery, or implicit model construction:

``region -> tracking -> prepared depth -> geometry -> dense object evidence``.

Depth is deliberately caller-provided.  It may be the canonical shared depth
stack or an explicitly supplied target-calibrated stack per object.  This keeps
the runtime reusable and prevents it from silently inventing cache, model, or
artifact-path policy.  The optional convenience composition function delegates
EEF work and final payload construction to the existing public runtimes.
"""

from __future__ import annotations

import copy
from collections.abc import Callable, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from dream_exe.transforms import parse_X_wb

from ..geometry.camera import Camera
from .config import load_pipeline_config
from ..geometry.prepared import build_visual_geometry_from_tracks
from .single_eef import (
    _tracking_numpy,
    run_single_eef_video2traj,
)
from ..trajectory.stages import compile_task_runtime, densify_center_records
from ..tracking.core import run_tracking_backend
from ..trajectory.composition import compose_prepared_trajectory_outputs


ObjectArtifactCallback = Callable[[str, Any], None]
InteractionGeometryBuilder = Callable[..., Mapping[str, Any]]


class ObjectEvidenceStageError(RuntimeError):
    """Identify the object and stage that failed without hiding its cause."""

    def __init__(
        self,
        object_id: str,
        stage: str,
        cause: Exception,
    ) -> None:
        self.object_id = str(object_id)
        self.stage = str(stage)
        self.cause = cause
        super().__init__(
            "object video2traj "
            f"object_id='{self.object_id}' stage '{self.stage}' failed: {cause}"
        )


@contextmanager
def _object_stage(
    object_id: str,
    stage: str,
) -> Iterator[None]:
    try:
        yield
    except ObjectEvidenceStageError:
        raise
    except Exception as error:
        raise ObjectEvidenceStageError(
            object_id,
            stage,
            error,
        ) from error


def _emit(
    callback: ObjectArtifactCallback | None,
    *,
    object_id: str,
    stage: str,
    payload: Any,
) -> None:
    if callback is None:
        return
    event = f"object.{object_id}.{stage}"
    try:
        callback(event, payload)
    except Exception as error:
        raise ObjectEvidenceStageError(
            object_id,
            f"{stage}.artifact_callback",
            error,
        ) from error


def _normalize_frames(
    frames: Sequence[Any],
) -> list[np.ndarray]:
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


def _normalize_depths(
    depths: Any,
    *,
    frames: Sequence[np.ndarray],
    label: str,
) -> np.ndarray:
    normalized = np.asarray(depths, dtype=np.float32)
    expected = (
        len(frames),
        int(frames[0].shape[0]),
        int(frames[0].shape[1]),
    )
    if normalized.shape != expected:
        raise ValueError(
            f"{label}/frame alignment mismatch: "
            f"depths={normalized.shape}, expected={expected}"
        )
    return normalized


def _value(
    payload: Any,
    name: str,
    *,
    default: Any = None,
    required: bool = False,
) -> Any:
    if isinstance(payload, Mapping):
        if name in payload:
            return payload[name]
    elif hasattr(payload, name):
        return getattr(payload, name)
    if required:
        raise ValueError(f"region result is missing '{name}'")
    return default


def _select_region(
    runtime: Any,
    kwargs: Mapping[str, Any],
) -> Any:
    select_target = getattr(runtime, "select_target", None)
    if callable(select_target):
        return select_target(**dict(kwargs))
    if callable(runtime):
        return runtime(**dict(kwargs))
    raise TypeError("region_runtime must be callable or expose select_target(...)")


def _options_for(
    options_by_object: Mapping[
        str,
        Mapping[str, Any],
    ]
    | None,
    *,
    object_id: str,
    label: str,
    reserved: set[str],
) -> dict[str, Any]:
    all_options = dict(options_by_object or {})
    raw = all_options.get(str(object_id), {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label}[{object_id!r}] must be a mapping")
    options = dict(raw)
    overlap = sorted(reserved.intersection(options))
    if overlap:
        raise ValueError(
            f"{label}[{object_id!r}] cannot replace "
            "runtime-owned inputs: " + ", ".join(overlap)
        )
    return options


def _validate_per_object_keys(
    value: Mapping[str, Any] | None,
    *,
    known_ids: set[str],
    label: str,
) -> None:
    unexpected = sorted(
        str(key) for key in dict(value or {}) if str(key) not in known_ids
    )
    if unexpected:
        raise ValueError(
            f"{label} contains unknown object ids: " + ", ".join(unexpected)
        )


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
        raise ValueError("tracking backend returned no object points")
    return tracks_array, visibility_array


def _object_center_options(
    stream: Mapping[str, Any],
    overrides: Mapping[str, Any],
) -> dict[str, Any]:
    target = dict(stream.get("geometry_target_cfg", {}) or {})
    configured = {
        "interpolate": bool(target.get("interpolate", False)),
        "max_gap": int(target.get("max_gap", 10)),
        "fill_ends": bool(target.get("fill_ends", False)),
        "end_max": int(target.get("end_max", 3)),
        "carry_previous": bool(target.get("carry_prev", False)),
        "min_points": int(target.get("min_points", 1)),
    }
    explicit = overrides.get("center_options", {})
    if explicit is None:
        explicit = {}
    if not isinstance(explicit, Mapping):
        raise TypeError("geometry center_options must be a mapping")
    return {
        **configured,
        **dict(explicit),
    }


def _region_payload(
    region: Any,
) -> Any:
    to_dict = getattr(region, "to_dict", None)
    if callable(to_dict):
        return to_dict()
    if isinstance(region, Mapping):
        return copy.deepcopy(dict(region))
    return region


def _center_quality(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold_m: float,
    reference_frame: str,
) -> dict[str, Any]:
    position_key = "pos_world" if str(reference_frame) == "world" else "pos_base"
    positions: list[np.ndarray] = []
    for record in list(records or []):
        if not isinstance(record, Mapping):
            continue
        raw = record.get(
            position_key,
            record.get("pos_world", None),
        )
        if raw is None:
            continue
        array = np.asarray(raw, dtype=np.float64).reshape(-1)
        if array.size >= 3 and np.all(np.isfinite(array[:3])):
            positions.append(array[:3].copy())
    if not positions:
        return {
            "valid_frames": 0,
            "total_frames": int(len(records or [])),
            "valid_ratio": 0.0,
            "max_displacement_m": None,
            "motion_detected": False,
        }
    anchor = positions[0]
    max_displacement = max(
        float(np.linalg.norm(position - anchor)) for position in positions
    )
    return {
        "valid_frames": int(len(positions)),
        "total_frames": int(len(records or [])),
        "valid_ratio": float(len(positions) / max(1, len(records or []))),
        "max_displacement_m": float(max_displacement),
        "motion_detected": bool(max_displacement >= float(threshold_m)),
    }


def run_prepared_object_evidence(
    *,
    frames: Sequence[Any],
    camera: Camera,
    simulator_config: Mapping[str, Any],
    object_stream_plan: Sequence[Mapping[str, Any]],
    region_runtime: Any,
    tracking_backend: Any,
    depths: Any,
    object_depths: Mapping[str, Any] | None = None,
    object_depth_metadata: Mapping[
        str,
        Mapping[str, Any],
    ]
    | None = None,
    interaction_geometry_builder: InteractionGeometryBuilder | None = None,
    init_depth: Any = None,
    precomputed_regions: Mapping[str, Any] | None = None,
    region_options_by_object: Mapping[
        str,
        Mapping[str, Any],
    ]
    | None = None,
    tracking_options_by_object: Mapping[
        str,
        Mapping[str, Any],
    ]
    | None = None,
    geometry_options_by_object: Mapping[
        str,
        Mapping[str, Any],
    ]
    | None = None,
    region_output_dirs: Mapping[
        str,
        str | Path,
    ]
    | None = None,
    tracking_output_dirs: Mapping[
        str,
        str | Path,
    ]
    | None = None,
    write_region_artifacts: bool = False,
    release_region_models: bool = True,
    artifact_references_by_object: Mapping[
        str,
        Mapping[str, Any],
    ]
    | None = None,
    motion_threshold_m: float = 0.004,
    motion_reference_frame: str = "world",
    artifact_callback: ObjectArtifactCallback | None = None,
) -> dict[str, Any]:
    """Build current-compatible evidence for prepared object streams.

    ``object_stream_plan`` should normally come from
    :func:`compile_task_runtime`; its unique streams preserve shared-object
    identity across multiple semantic stages.  No directory is created and no
    payload is serialized here.  ``write_region_artifacts`` is opt-in, and any
    tracking/model side effects remain the responsibility of the explicitly
    injected backend.
    """

    frame_list = _normalize_frames(frames)
    if not isinstance(simulator_config, Mapping):
        raise TypeError("simulator_config must be an explicit mapping")
    canonical_depths = _normalize_depths(
        depths,
        frames=frame_list,
        label="depth",
    )
    initial_depth = None
    if init_depth is not None:
        initial_depth = np.asarray(init_depth, dtype=np.float32)
        expected_hw = frame_list[0].shape[:2]
        if initial_depth.shape != expected_hw:
            raise ValueError(
                "init_depth/frame mismatch: "
                f"depth={initial_depth.shape}, "
                f"frame={expected_hw}"
            )

    streams = [
        copy.deepcopy(dict(stream))
        for stream in list(object_stream_plan or [])
        if isinstance(stream, Mapping)
    ]
    if len(streams) != len(list(object_stream_plan or [])):
        raise TypeError("object_stream_plan must contain only mappings")
    object_ids = [str(stream.get("object_id", "") or "") for stream in streams]
    if any(not object_id for object_id in object_ids):
        raise ValueError("every object stream requires a non-empty object_id")
    if len(set(object_ids)) != len(object_ids):
        raise ValueError("object_stream_plan contains duplicate object_id values")
    known_ids = set(object_ids)
    for value, label in (
        (object_depths, "object_depths"),
        (object_depth_metadata, "object_depth_metadata"),
        (precomputed_regions, "precomputed_regions"),
        (region_options_by_object, "region_options_by_object"),
        (tracking_options_by_object, "tracking_options_by_object"),
        (geometry_options_by_object, "geometry_options_by_object"),
        (region_output_dirs, "region_output_dirs"),
        (tracking_output_dirs, "tracking_output_dirs"),
        (
            artifact_references_by_object,
            "artifact_references_by_object",
        ),
    ):
        _validate_per_object_keys(
            value,
            known_ids=known_ids,
            label=label,
        )

    resolved_object_depths = dict(object_depths or {})
    resolved_depth_metadata = dict(object_depth_metadata or {})
    resolved_precomputed = dict(precomputed_regions or {})
    resolved_region_dirs = dict(region_output_dirs or {})
    resolved_tracking_dirs = dict(tracking_output_dirs or {})
    resolved_references = dict(artifact_references_by_object or {})
    default_geometry_transform: dict[str, Any] = {}
    transform = (
        dict(simulator_config).get("derived", {}).get("eef", {}).get("X_wb", None)
    )
    if transform is not None:
        rotation_world_base, translation_world_base = parse_X_wb(transform)
        default_geometry_transform = {
            "rotation_world_base": rotation_world_base,
            "translation_world_base": translation_world_base,
        }

    prepared_states: dict[str, dict[str, Any]] = {}

    # Match the current multi-object stage order: resolve every region before
    # loading/running the tracker.  This also means one shared object stream
    # gets exactly one region query even when several stages reference it.
    for ordinal, stream in enumerate(streams):
        object_id = str(stream["object_id"])
        stage_ids = [
            str(stage_id) for stage_id in list(stream.get("stage_ids", []) or [])
        ]
        manipulated_object = dict(stream.get("manipulated_object", {}) or {})
        region_output_dir = str(resolved_region_dirs.get(object_id, ""))
        if bool(write_region_artifacts) and not region_output_dir.strip():
            raise ValueError(
                "region_output_dirs must provide an explicit "
                f"path for object_id='{object_id}' when "
                "write_region_artifacts=True"
            )

        tracking_config = dict(stream.get("tracking_target_cfg", {}) or {})
        requested_points = int(tracking_config.get("num_points", 50))
        if requested_points <= 0:
            raise ValueError(f"object_id='{object_id}' num_points must be positive")
        region_options = _options_for(
            region_options_by_object,
            object_id=object_id,
            label="region_options_by_object",
            reserved={
                "target_name",
                "frame_rgb",
                "output_dir",
                "num_points",
                "environment_config",
                "target_config",
                "precomputed_region",
                "init_depth",
                "camera",
                "seed",
                "write_artifacts",
            },
        )
        region_kwargs = {
            **region_options,
            "target_name": "obj",
            "frame_rgb": frame_list[0],
            "output_dir": region_output_dir,
            "num_points": requested_points,
            "environment_config": dict(simulator_config),
            "target_config": dict(stream.get("region_target_cfg", {}) or {}),
            "precomputed_region": resolved_precomputed.get(
                object_id,
                None,
            ),
            "init_depth": initial_depth,
            "camera": camera,
            "seed": 7 + ordinal,
            "write_artifacts": bool(write_region_artifacts),
        }
        with _object_stage(object_id, "region"):
            region = _select_region(
                region_runtime,
                region_kwargs,
            )
            if region is None:
                manipulated_name = str(manipulated_object.get("name", "") or "")
                raise RuntimeError(
                    f"object_id={object_id} "
                    f"stages={stage_ids} "
                    f"manipulated_object='{manipulated_name}' "
                    "resolved to no object region."
                )
            bbox = [
                int(value)
                for value in _value(
                    region,
                    "final_bbox_xyxy",
                    required=True,
                )
            ]
            if len(bbox) != 4:
                raise ValueError("object region bbox must contain four values")
            query_points = np.asarray(
                _value(
                    region,
                    "sampled_points_xy",
                    required=True,
                ),
                dtype=np.float32,
            )
            if (
                query_points.ndim != 2
                or query_points.shape[1] != 2
                or query_points.shape[0] <= 0
            ):
                raise ValueError(
                    "object region sampled_points_xy must have "
                    f"shape [N,2], got {query_points.shape}"
                )
            region_mask = np.asarray(
                _value(region, "mask", required=True),
                dtype=bool,
            )
            if region_mask.shape != frame_list[0].shape[:2]:
                raise ValueError(
                    "object region mask/frame mismatch: "
                    f"mask={region_mask.shape}, "
                    f"frame={frame_list[0].shape[:2]}"
                )
            sampling_mask = np.asarray(
                _value(
                    region,
                    "sampling_mask",
                    default=region_mask,
                ),
                dtype=bool,
            )
            if sampling_mask.shape != region_mask.shape:
                raise ValueError(
                    "object sampling mask/frame mismatch: "
                    f"mask={sampling_mask.shape}, "
                    f"frame={region_mask.shape}"
                )
            _emit(
                artifact_callback,
                object_id=object_id,
                stage="region",
                payload=_region_payload(region),
            )

        prepared_states[object_id] = {
            "ordinal": ordinal,
            "stream": stream,
            "stage_ids": stage_ids,
            "manipulated_object": manipulated_object,
            "region": region,
            "bbox": bbox,
            "query_points": query_points,
            "sampling_mask": sampling_mask,
        }

    if bool(release_region_models):
        release_models = getattr(region_runtime, "release_models", None)
        if callable(release_models):
            try:
                release_models()
            except Exception as error:
                raise ObjectEvidenceStageError(
                    "<all>",
                    "region.release_models",
                    error,
                ) from error

    # Current extraction then runs tracking for every resolved object before
    # invoking the shared depth stage.
    for object_id in object_ids:
        state = prepared_states[object_id]
        ordinal = int(state["ordinal"])
        region = state["region"]
        bbox = state["bbox"]
        query_points = state["query_points"]
        sampling_mask = state["sampling_mask"]
        tracking_options = _options_for(
            tracking_options_by_object,
            object_id=object_id,
            label="tracking_options_by_object",
            reserved={
                "video_frames",
                "output_dir",
                "region_bbox_xyxy",
                "num_points",
                "filename",
                "seed",
                "query_points_xy",
                "segmentation_mask",
                "query_mode",
                "grid_size",
            },
        )
        requested_query_mode = str(
            _value(
                region,
                "tracking_input",
                default="points",
            )
            or "points"
        )
        normalized_query_mode = requested_query_mode.strip().lower()
        with _object_stage(object_id, "tracking"):
            tracking_output = run_tracking_backend(
                tracking_backend,
                video_frames=frame_list,
                output_dir=str(resolved_tracking_dirs.get(object_id, "")),
                region_bbox_xyxy=bbox,
                num_points=int(query_points.shape[0]),
                filename=f"{object_id}_points_cloud",
                seed=7 + ordinal,
                query_points_xy=(
                    None
                    if normalized_query_mode in {"bbox_center", "mask_grid"}
                    else query_points
                ),
                segmentation_mask=sampling_mask,
                query_mode=requested_query_mode,
                grid_size=0,
                **tracking_options,
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
            _emit(
                artifact_callback,
                object_id=object_id,
                stage="tracking",
                payload=tracking_payload,
            )
        state["tracking"] = tracking_payload

    # The depth values are already prepared by the caller, but make their
    # per-object selection observable at the same boundary before geometry.
    for object_id in object_ids:
        state = prepared_states[object_id]

        depth_stack = _normalize_depths(
            resolved_object_depths.get(
                object_id,
                canonical_depths,
            ),
            frames=frame_list,
            label=f"object_depths[{object_id!r}]",
        )
        depth_meta_raw = resolved_depth_metadata.get(
            object_id,
            {},
        )
        if depth_meta_raw is None:
            depth_meta_raw = {}
        if not isinstance(depth_meta_raw, Mapping):
            raise TypeError(f"object_depth_metadata[{object_id!r}] must be a mapping")
        depth_meta = copy.deepcopy(dict(depth_meta_raw))
        depth_payload = {
            "depths": depth_stack,
            "source": str(
                depth_meta.get(
                    "depth_source",
                    (
                        "object_override"
                        if object_id in resolved_object_depths
                        else "canonical"
                    ),
                )
                or ""
            ),
            "metadata": depth_meta,
        }
        _emit(
            artifact_callback,
            object_id=object_id,
            stage="depth",
            payload=depth_payload,
        )
        state["depth"] = depth_payload

    runtime_objects: dict[str, dict[str, Any]] = {}
    evidence_by_object: dict[str, dict[str, Any]] = {}
    for object_id in object_ids:
        state = prepared_states[object_id]
        stream = state["stream"]
        stage_ids = state["stage_ids"]
        manipulated_object = state["manipulated_object"]
        region = state["region"]
        bbox = state["bbox"]
        tracking_payload = state["tracking"]
        tracks_array = tracking_payload["tracks_uv"]
        visibility_array = tracking_payload["visibility"]
        depth_payload = state["depth"]
        depth_stack = depth_payload["depths"]
        depth_meta = depth_payload["metadata"]

        geometry_options = _options_for(
            geometry_options_by_object,
            object_id=object_id,
            label="geometry_options_by_object",
            reserved={
                "tracks_uv",
                "visibility",
                "depths",
                "camera",
            },
        )
        transform_keys = {
            "rotation_world_base",
            "translation_world_base",
        }
        supplied_transform_keys = transform_keys.intersection(geometry_options)
        if supplied_transform_keys and (supplied_transform_keys != transform_keys):
            raise ValueError(
                "object geometry transform requires both "
                "rotation_world_base and translation_world_base"
            )
        if not supplied_transform_keys:
            geometry_options.update(copy.deepcopy(default_geometry_transform))
        center_options = _object_center_options(
            stream,
            geometry_options,
        )
        geometry_options.pop("center_options", None)
        geometry_options.setdefault(
            "visibility_threshold",
            0.5,
        )
        with _object_stage(object_id, "geometry"):
            geometry = build_visual_geometry_from_tracks(
                tracks_uv=tracks_array,
                visibility=visibility_array,
                depths=depth_stack,
                camera=camera,
                center_options=center_options,
                **geometry_options,
            )
            if len(geometry["points"]) != len(frame_list):
                raise ValueError(
                    "object geometry/frame alignment mismatch: "
                    f"geometry={len(geometry['points'])}, "
                    f"frames={len(frame_list)}"
                )
            dense_centers = densify_center_records(
                list(geometry["visual_center_records"])
            )
            geometry["dense_visual_center_records"] = dense_centers
            _emit(
                artifact_callback,
                object_id=object_id,
                stage="geometry",
                payload=geometry,
            )

        interaction_geometry: dict[str, Any] = {}
        if interaction_geometry_builder is not None:
            with _object_stage(
                object_id,
                "interaction_geometry",
            ):
                raw_interaction = interaction_geometry_builder(
                    object_id=object_id,
                    stream=copy.deepcopy(stream),
                    region=region,
                    tracking=copy.deepcopy(tracking_payload),
                    geometry=geometry,
                    geometry_options={
                        **copy.deepcopy(geometry_options),
                        "center_options": copy.deepcopy(center_options),
                    },
                )
                if not isinstance(raw_interaction, Mapping):
                    raise TypeError(
                        "interaction_geometry_builder must return a mapping"
                    )
                interaction_geometry = copy.deepcopy(dict(raw_interaction))
                _emit(
                    artifact_callback,
                    object_id=object_id,
                    stage="interaction_geometry",
                    payload=interaction_geometry,
                )

        references_raw = resolved_references.get(
            object_id,
            {},
        )
        if references_raw is None:
            references_raw = {}
        if not isinstance(references_raw, Mapping):
            raise TypeError(
                f"artifact_references_by_object[{object_id!r}] must be a mapping"
            )
        references = copy.deepcopy(dict(references_raw))
        region_json = references.get(
            "region_json",
            _value(region, "output_json", default=None),
        )
        visibility_threshold = float(
            geometry_options.get(
                "visibility_threshold",
                0.5,
            )
        )
        quality = _center_quality(
            dense_centers,
            threshold_m=float(motion_threshold_m),
            reference_frame=str(motion_reference_frame),
        )
        evidence = {
            "object_id": object_id,
            "owner_stage_id": str(stream.get("owner_stage_id", "") or ""),
            "owner_stage_index": int(stream.get("owner_stage_index", 0)),
            "stage_ids": copy.deepcopy(stage_ids),
            "stage_indices": copy.deepcopy(list(stream.get("stage_indices", []) or [])),
            "runtime_object_key": str(stream.get("runtime_object_key", "") or ""),
            "manipulated_object": manipulated_object,
            "obj_region": copy.deepcopy(bbox),
            "obj_region_source": copy.deepcopy(_value(region, "source", default=None)),
            "obj_prompt": copy.deepcopy(_value(region, "prompt", default=None)),
            "obj_num_points": int(tracks_array.shape[1]),
            "vis_threshold": visibility_threshold,
            "region_json": copy.deepcopy(region_json),
            "tracking_input": copy.deepcopy(
                _value(
                    region,
                    "tracking_input",
                    default=None,
                )
            ),
            "tracking_npz": copy.deepcopy(references.get("tracking_npz", None)),
            "tracking_mp4": copy.deepcopy(references.get("tracking_mp4", None)),
            "obj_points_traj_path": copy.deepcopy(
                references.get(
                    "obj_points_traj_path",
                    references.get("points_json", None),
                )
            ),
            "points_flow_npz": copy.deepcopy(references.get("points_flow_npz", None)),
            "quality": quality,
            "obj_visual_center": dense_centers,
            "meta": {
                "object_id": object_id,
                "owner_stage_id": str(stream.get("owner_stage_id", "") or ""),
                "owner_stage_index": int(stream.get("owner_stage_index", 0)),
                "stage_ids": copy.deepcopy(stage_ids),
                "stage_indices": copy.deepcopy(
                    list(stream.get("stage_indices", []) or [])
                ),
                "runtime_object_key": str(stream.get("runtime_object_key", "") or ""),
                "depth_source": depth_payload["source"],
                "depth_metadata": depth_meta,
                "quality": copy.deepcopy(quality),
            },
        }
        if interaction_geometry:
            evidence["interaction_geometry"] = interaction_geometry
            evidence["meta"]["interaction_geometry"] = copy.deepcopy(
                dict(interaction_geometry.get("meta", {}) or {})
            )
        evidence_by_object[object_id] = evidence
        runtime_objects[object_id] = {
            "stream": copy.deepcopy(stream),
            "region": region,
            "tracking": tracking_payload,
            "depth": depth_payload,
            "geometry": geometry,
            "evidence": evidence,
        }
        _emit(
            artifact_callback,
            object_id=object_id,
            stage="evidence",
            payload=evidence,
        )

    return {
        "scope": {
            "target": "objects",
            "path": "prepared_object_evidence",
            "object_count": len(runtime_objects),
            "shared_object_streams": sum(
                1
                for stream in streams
                if len(list(stream.get("stage_ids", []) or [])) > 1
            ),
            "depth_policy": "explicit_prepared",
            "pose_policy": "eef_runtime_only",
            "full_pipeline_parity": False,
        },
        "object_stream_plan": streams,
        "objects": runtime_objects,
        "evidence_by_object": evidence_by_object,
    }


def compose_prepared_multi_object_outputs(
    *,
    uid: str,
    simulator_config: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    single_eef_result: Mapping[str, Any],
    object_runtime_result: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    composition_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Delegate prepared EEF/object evidence to the public composer."""

    trajectory = single_eef_result.get("trajectory", None)
    if not isinstance(trajectory, Mapping):
        raise ValueError("single_eef_result must contain a trajectory mapping")
    evidence = object_runtime_result.get(
        "evidence_by_object",
        None,
    )
    if not isinstance(evidence, Mapping):
        raise ValueError(
            "object_runtime_result must contain an evidence_by_object mapping"
        )
    options = dict(composition_options or {})
    reserved = {
        "uid",
        "cfg",
        "pipeline_config",
        "ee_trajectory",
        "object_evidence",
        "metadata",
    }
    overlap = sorted(reserved.intersection(options))
    if overlap:
        raise ValueError(
            "composition_options cannot replace "
            "runtime-owned inputs: " + ", ".join(overlap)
        )
    return compose_prepared_trajectory_outputs(
        uid=str(uid),
        cfg=dict(simulator_config),
        pipeline_config=dict(pipeline_config),
        ee_trajectory=dict(trajectory),
        object_evidence=dict(evidence),
        metadata=dict(metadata or {}),
        **options,
    )


def run_prepared_multi_object_video2traj(
    *,
    uid: str,
    frames: Sequence[Any],
    camera: Camera,
    simulator_config: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    region_runtime: Any,
    tracking_backend: Any,
    target_fps: float,
    metadata: Mapping[str, Any] | None = None,
    single_eef_options: Mapping[str, Any] | None = None,
    object_runtime_options: Mapping[str, Any] | None = None,
    composition_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Run a bounded prepared-frame EEF + multi-object composition.

    The convenience path first delegates the EEF slice (including shared
    canonical depth and optional EEF pose), then builds all compiled object
    streams from that explicit depth stack, and finally delegates current
    trajectory/gripper/action payload construction to the public composer.

    It intentionally does not claim current target-calibrated object-aware
    depth, union-geometry, artifact, cache, or raw video-file orchestration
    parity.
    """

    if not isinstance(simulator_config, Mapping):
        raise TypeError("simulator_config must be an explicit mapping")
    if not isinstance(pipeline_config, Mapping):
        raise TypeError("pipeline_config must be an explicit mapping")
    normalized_pipeline = load_pipeline_config(dict(pipeline_config))
    metadata_payload = dict(metadata or {})
    task_runtime = compile_task_runtime(
        uid=str(uid),
        metadata=metadata_payload,
        pipeline_config=normalized_pipeline,
    )

    eef_options = dict(single_eef_options or {})
    eef_reserved = {
        "frames",
        "camera",
        "simulator_config",
        "region_runtime",
        "tracking_backend",
        "target_fps",
        "metadata",
    }
    overlap = sorted(eef_reserved.intersection(eef_options))
    if overlap:
        raise ValueError(
            "single_eef_options cannot replace "
            "runtime-owned inputs: " + ", ".join(overlap)
        )
    region_options = dict(eef_options.pop("region_options", {}) or {})
    if "target_config" in region_options:
        raise ValueError(
            "single_eef_options.region_options cannot replace "
            "pipeline region.targets.eef"
        )
    region_options["target_config"] = dict(
        dict(normalized_pipeline.get("region", {}) or {})
        .get("targets", {})
        .get("eef", {})
        or {}
    )
    eef_tracking = dict(
        dict(normalized_pipeline.get("tracking", {}) or {})
        .get("targets", {})
        .get("eef", {})
        or {}
    )
    eef_options.setdefault(
        "num_points",
        int(eef_tracking.get("num_points", 150)),
    )
    eef_options["region_options"] = region_options

    pose_config = dict(normalized_pipeline.get("pose", {}) or {})
    if bool(pose_config.get("enabled", False)):
        has_pose_injection = any(
            eef_options.get(name, None) is not None
            for name in (
                "pose_estimator",
                "pose_backend",
                "pose_config",
            )
        )
        if not has_pose_injection:
            raise RuntimeError(
                "pipeline pose.enabled=true requires an explicit "
                "pose_estimator, pose_backend, or inline pose_config "
                "in single_eef_options"
            )

    single_eef = run_single_eef_video2traj(
        frames=frames,
        camera=camera,
        simulator_config=dict(simulator_config),
        region_runtime=region_runtime,
        tracking_backend=tracking_backend,
        target_fps=float(target_fps),
        metadata=metadata_payload,
        **eef_options,
    )
    depth_payload = single_eef.get("depth", {})
    if not isinstance(depth_payload, Mapping):
        raise ValueError("single EEF runtime did not return a depth mapping")
    shared_depths = depth_payload.get("depths", None)
    if shared_depths is None:
        raise ValueError("single EEF runtime did not return prepared depths")

    object_options = dict(object_runtime_options or {})
    object_reserved = {
        "frames",
        "camera",
        "simulator_config",
        "object_stream_plan",
        "region_runtime",
        "tracking_backend",
        "depths",
    }
    overlap = sorted(object_reserved.intersection(object_options))
    if overlap:
        raise ValueError(
            "object_runtime_options cannot replace "
            "runtime-owned inputs: " + ", ".join(overlap)
        )
    object_runtime = run_prepared_object_evidence(
        frames=frames,
        camera=camera,
        simulator_config=dict(simulator_config),
        object_stream_plan=list(task_runtime.get("object_stream_plan", []) or []),
        region_runtime=region_runtime,
        tracking_backend=tracking_backend,
        depths=shared_depths,
        **object_options,
    )
    outputs = compose_prepared_multi_object_outputs(
        uid=str(uid),
        simulator_config=dict(simulator_config),
        pipeline_config=normalized_pipeline,
        single_eef_result=single_eef,
        object_runtime_result=object_runtime,
        metadata=metadata_payload,
        composition_options=composition_options,
    )
    return {
        "scope": {
            "target": "eef_and_objects",
            "path": "prepared_multi_object",
            "object_count": len(object_runtime["evidence_by_object"]),
            "pose_enabled": bool(
                dict(single_eef.get("scope", {}) or {}).get(
                    "pose_enabled",
                    False,
                )
            ),
            "depth_conditioning": "eef_then_shared_canonical",
            "full_pipeline_parity": False,
        },
        "task_runtime": task_runtime,
        "single_eef": single_eef,
        "object_runtime": object_runtime,
        "outputs": outputs,
    }


__all__ = [
    "InteractionGeometryBuilder",
    "ObjectArtifactCallback",
    "ObjectEvidenceStageError",
    "compose_prepared_multi_object_outputs",
    "run_prepared_multi_object_video2traj",
    "run_prepared_object_evidence",
]
