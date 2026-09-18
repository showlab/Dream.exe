"""Pure task-stage planning helpers for the video-to-trajectory core.

This module deliberately depends only on Python data structures and NumPy.
Callers provide task metadata, pipeline configuration, and prepared trajectory
records explicitly; no benchmark, simulator, or storage policy is inferred.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping, Sequence
from typing import Any, Dict, TypedDict

import numpy as np


__all__ = [
    "ActiveStageFrame",
    "StageTimeline",
    "StageWindow",
    "assign_stage_timelines",
    "compile_runtime_stage_order",
    "compile_task_runtime",
    "compile_stage_plan",
    "compute_motion_onset_frame",
    "compute_motion_settle_frame",
    "densify_center_records",
    "load_task_spec",
    "normalize_gripper_initial_state",
    "resolve_gripper_initial_state",
    "stage_records_by_id",
    "validate_stage_timeline",
]


class StageWindow(TypedDict):
    """Inclusive stage window in both trajectory-index and frame space."""

    start_index: int | None
    end_index: int | None
    start_frame: int | None
    end_frame: int | None


class ActiveStageFrame(TypedDict):
    """One frame's assignment to an ordered task stage."""

    frame: int
    stage_id: str


class StageTimeline(TypedDict):
    """Output contract of the pure stage timeline compiler."""

    active_stage_by_frame: list[ActiveStageFrame]
    search_windows: dict[str, StageWindow]
    active_windows: dict[str, StageWindow]


def _clean_str(value: Any) -> str:
    return str(value or "").strip()


def _normalize_name(value: Any) -> str:
    text = _clean_str(value).lower()
    return re.sub(r"[^a-z0-9]+", "_", text).strip("_")


def _safe_stage_id(value: Any, *, fallback: str) -> str:
    return _normalize_name(value) or str(fallback)


def _task_block_from_payload(payload: Mapping[str, Any] | None) -> dict[str, Any]:
    source = dict(payload or {})
    return copy.deepcopy(dict(source.get("task", {}) or {}))


def normalize_gripper_initial_state(
    value: Any,
    *,
    default: str = "",
) -> str:
    """Normalize the historical open/closed aliases used by trajectory data."""

    state = _clean_str(value).lower()
    if state in {"open", "opened", "opening"}:
        return "open"
    if state in {"close", "closed", "closing", "grasped"}:
        return "closed"
    if state == "unknown":
        return "unknown"
    return _clean_str(default).lower()


def _object_list(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return copy.deepcopy(list(value))
    return []


def _matching_metadata_object(
    base: Mapping[str, Any],
    metadata_objects: Sequence[Any],
    index: int,
) -> dict[str, Any] | None:
    base_instance = _clean_str(base.get("instance_name"))
    base_name = _normalize_name(base.get("name"))
    for candidate in metadata_objects:
        if not isinstance(candidate, Mapping):
            continue
        candidate_instance = _clean_str(candidate.get("instance_name"))
        if base_instance and candidate_instance == base_instance:
            return copy.deepcopy(dict(candidate))
    for candidate in metadata_objects:
        if not isinstance(candidate, Mapping):
            continue
        if base_name and _normalize_name(candidate.get("name")) == base_name:
            return copy.deepcopy(dict(candidate))
    if 0 <= index < len(metadata_objects):
        candidate = metadata_objects[index]
        if isinstance(candidate, Mapping):
            return copy.deepcopy(dict(candidate))
    return None


def _merge_object_lists(
    pipeline_objects: Any,
    metadata_objects: Any,
) -> list[Any]:
    pipeline_list = _object_list(pipeline_objects)
    metadata_list = _object_list(metadata_objects)
    if not pipeline_list:
        return metadata_list
    if not metadata_list:
        return pipeline_list

    merged_objects: list[Any] = []
    for index, pipeline_object in enumerate(pipeline_list):
        if not isinstance(pipeline_object, Mapping):
            merged_objects.append(copy.deepcopy(pipeline_object))
            continue
        base = copy.deepcopy(dict(pipeline_object))
        overlay = _matching_metadata_object(base, metadata_list, index)
        if overlay is None:
            merged_objects.append(base)
            continue

        merged = {**copy.deepcopy(overlay), **base}
        for key in ("name", "instance_name", "selector_mode", "selector_text"):
            if not _clean_str(base.get(key)) and _clean_str(overlay.get(key)):
                merged[key] = copy.deepcopy(overlay[key])
        if "runtime_match" in overlay:
            merged["runtime_match"] = copy.deepcopy(overlay["runtime_match"])
        merged_objects.append(merged)
    return merged_objects


def _merge_stage_specs(
    pipeline_stages: Any,
    metadata_stages: Any,
) -> list[dict[str, Any]]:
    pipeline_list = _object_list(pipeline_stages)
    metadata_list = _object_list(metadata_stages)
    if not pipeline_list:
        return [
            copy.deepcopy(dict(item))
            for item in metadata_list
            if isinstance(item, Mapping)
        ]
    if not metadata_list:
        return [
            copy.deepcopy(dict(item))
            for item in pipeline_list
            if isinstance(item, Mapping)
        ]

    metadata_by_id: dict[str, dict[str, Any]] = {}
    for item in metadata_list:
        if not isinstance(item, Mapping):
            continue
        stage_id = _clean_str(item.get("stage_id"))
        if stage_id:
            metadata_by_id[stage_id] = copy.deepcopy(dict(item))

    result: list[dict[str, Any]] = []
    for item in pipeline_list:
        if not isinstance(item, Mapping):
            continue
        base = copy.deepcopy(dict(item))
        overlay = metadata_by_id.get(_clean_str(base.get("stage_id")))
        if overlay is None:
            result.append(base)
            continue

        merged = copy.deepcopy(base)
        for key in ("stage_id", "task_type", "interaction_mode", "notes"):
            merged[key] = _clean_str(base.get(key)) or _clean_str(overlay.get(key))
        for key in ("gripper_plan", "gripper_inference"):
            selected = base.get(key) or overlay.get(key) or {}
            merged[key] = copy.deepcopy(dict(selected))
        merged["manipulated_objects"] = _merge_object_lists(
            base.get("manipulated_objects"),
            overlay.get("manipulated_objects"),
        )
        merged["target_objects"] = _merge_object_lists(
            base.get("target_objects"),
            overlay.get("target_objects"),
        )
        result.append(merged)
    return result


def _clean_string_list(value: Any) -> list[str]:
    try:
        items = list(value or [])
    except TypeError:
        return []
    return [text for item in items if (text := _clean_str(item))]


def load_task_spec(
    *,
    pipeline_config: dict[str, Any],
    metadata: dict[str, Any],
) -> dict[str, Any]:
    """Return the task fields consumed by trajectory planning."""

    pipeline_task = _task_block_from_payload(pipeline_config)
    metadata_task = _task_block_from_payload(metadata)
    stages = _merge_stage_specs(
        pipeline_task.get("stages"),
        metadata_task.get("stages"),
    )

    task_types = _clean_string_list(
        pipeline_task.get("task_types") or metadata_task.get("task_types")
    )
    interaction_modes = _clean_string_list(
        pipeline_task.get("interaction_modes") or metadata_task.get("interaction_modes")
    )
    if "interaction_count" in pipeline_task:
        interaction_count = int(pipeline_task.get("interaction_count") or len(stages))
    elif "interaction_count" in metadata_task:
        interaction_count = int(metadata_task.get("interaction_count") or len(stages))
    else:
        interaction_count = len(stages)

    if "multi_object_interaction" in pipeline_task:
        multi_object = bool(pipeline_task.get("multi_object_interaction"))
    elif "multi_object_interaction" in metadata_task:
        multi_object = bool(metadata_task.get("multi_object_interaction"))
    else:
        multi_object = len(stages) > 1

    gripper_plan_value = (
        pipeline_task.get("gripper_plan") or metadata_task.get("gripper_plan") or {}
    )
    gripper_init_value = (
        pipeline_task.get("gripper_init")
        if "gripper_init" in pipeline_task
        else metadata_task.get("gripper_init")
    )
    gripper_init = normalize_gripper_initial_state(
        gripper_init_value,
        default="",
    )
    notes = _clean_str(pipeline_task.get("notes")) or _clean_str(
        metadata_task.get("notes")
    )
    return {
        "task_types": task_types,
        "interaction_modes": interaction_modes,
        "interaction_count": interaction_count,
        "multi_object_interaction": multi_object,
        "gripper_init": gripper_init,
        "gripper_plan": copy.deepcopy(dict(gripper_plan_value)),
        "stages": stages,
        "notes": notes,
    }


def resolve_gripper_initial_state(
    *,
    pipeline_config: dict[str, Any],
    metadata: dict[str, Any],
    default: str = "open",
) -> str:
    """Resolve the configured initial state using current precedence rules."""

    task_spec = load_task_spec(
        pipeline_config=pipeline_config,
        metadata=metadata,
    )
    pipeline = dict(pipeline_config or {})
    gripper = dict(pipeline.get("gripper", {}) or {})
    grasp = dict(pipeline.get("grasp", {}) or {})
    if "initial_state" in gripper:
        value = gripper.get("initial_state")
    elif "initial_state" in grasp:
        value = grasp.get("initial_state")
    else:
        value = task_spec.get("gripper_init")
    return normalize_gripper_initial_state(value, default=default)


def _selector_mode_to_region_selector(
    selector_mode: Any,
    configured_selector: Any,
) -> str:
    configured = _clean_str(configured_selector).lower()
    if configured:
        return configured
    mode = _clean_str(selector_mode).lower()
    if mode == "simulation":
        return "simulation"
    return "visual"


def _mapping_list(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, (list, tuple)):
        return []
    return [copy.deepcopy(dict(item)) for item in value if isinstance(item, Mapping)]


def _stage_object_runtime_key(
    manipulated_object: Mapping[str, Any],
) -> tuple[str, str, str]:
    instance_name = _clean_str(manipulated_object.get("instance_name"))
    runtime_match = dict(manipulated_object.get("runtime_match", {}) or {})
    status = _clean_str(runtime_match.get("status"))
    if instance_name:
        return f"instance:{instance_name}", instance_name, status
    return (
        f"name:{_normalize_name(manipulated_object.get('name'))}",
        "",
        status,
    )


def _match_stage_object_cfg(
    candidates: Sequence[Mapping[str, Any]],
    *,
    stage_id: str,
    object_id: str,
    runtime_key: str,
    instance_name: str,
    object_name: str,
) -> dict[str, Any] | None:
    normalized_name = _normalize_name(object_name)
    for candidate_value in candidates:
        candidate = dict(candidate_value)
        matches = (
            (
                bool(_clean_str(candidate.get("runtime_key")))
                and _clean_str(candidate.get("runtime_key")) == runtime_key
            )
            or (
                bool(_clean_str(candidate.get("stage_id")))
                and _clean_str(candidate.get("stage_id")) == stage_id
            )
            or (
                bool(_clean_str(candidate.get("object_id")))
                and _clean_str(candidate.get("object_id")) == object_id
            )
            or (
                bool(instance_name)
                and _clean_str(candidate.get("instance_name")) == instance_name
            )
            or (
                bool(normalized_name)
                and _normalize_name(candidate.get("name")) == normalized_name
            )
        )
        if matches:
            return copy.deepcopy(candidate)
    return None


def _target_configs(
    pipeline_config: Mapping[str, Any],
    section: str,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    section_value = dict(pipeline_config.get(section, {}) or {})
    targets = dict(section_value.get("targets", {}) or {})
    default = copy.deepcopy(dict(targets.get("obj", {}) or {}))
    overrides = _mapping_list(targets.get("objects"))
    return default, overrides


def _compile_target_configs(
    pipeline_config: Mapping[str, Any],
    *,
    stage_id: str,
    object_id: str,
    runtime_key: str,
    instance_name: str,
    manipulated_object: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selected: dict[str, dict[str, Any]] = {}
    object_name = _clean_str(manipulated_object.get("name"))
    for section in ("region", "tracking", "geometry"):
        default, overrides = _target_configs(pipeline_config, section)
        matched = _match_stage_object_cfg(
            overrides,
            stage_id=stage_id,
            object_id=object_id,
            runtime_key=runtime_key,
            instance_name=instance_name,
            object_name=object_name,
        )
        selected[section] = matched if matched is not None else default

    region = selected["region"]
    region["selector"] = _selector_mode_to_region_selector(
        manipulated_object.get("selector_mode"),
        region.get("selector"),
    )
    # Region configuration is the per-sample authority for detector prompts.
    # Task metadata remains a useful fallback for older samples that omit an
    # explicit prompt, but must not silently replace a reviewed region target.
    configured_prompt = _clean_str(region.get("prompt"))
    region["prompt"] = configured_prompt or (
        _clean_str(manipulated_object.get("selector_text")) or object_name
    )
    region.setdefault("bbox_xyxy", None)
    simulation = copy.deepcopy(dict(region.get("simulation", {}) or {}))
    simulation["instance_name"] = instance_name
    region["simulation"] = simulation
    return region, selected["tracking"], selected["geometry"]


def _ambiguous_name(
    stage_entries: Sequence[tuple[int, Mapping[str, Any], Mapping[str, Any]]],
) -> tuple[str, str, str] | None:
    occurrences: dict[str, list[tuple[str, str]]] = {}
    for index, stage, manipulated_object in stage_entries:
        if _clean_str(manipulated_object.get("instance_name")):
            continue
        name = _clean_str(manipulated_object.get("name"))
        normalized = _normalize_name(name)
        stage_id = _clean_str(stage.get("stage_id")) or f"s{index + 1}"
        occurrences.setdefault(normalized, []).append((stage_id, name))
    for values in occurrences.values():
        if len(values) > 1:
            stage_id, name = values[0]
            return stage_id, name, _normalize_name(name)
    return None


def compile_task_runtime(
    *,
    uid: str,
    metadata: dict[str, Any],
    pipeline_config: dict[str, Any],
) -> dict[str, Any]:
    """Compile semantic stages into executable object streams."""

    task_spec = load_task_spec(
        pipeline_config=pipeline_config,
        metadata=metadata,
    )
    executable: list[tuple[int, dict[str, Any], dict[str, Any]]] = []
    stage_index_by_id: dict[str, int] = {}
    for index, raw_stage in enumerate(task_spec.get("stages", []) or []):
        stage = copy.deepcopy(dict(raw_stage or {}))
        stage_id = _clean_str(stage.get("stage_id")) or f"s{index + 1}"
        manipulated = list(stage.get("manipulated_objects", []) or [])
        if not manipulated:
            continue
        previous_index = stage_index_by_id.get(stage_id)
        if previous_index is not None:
            raise ValueError(
                f"[stage_plan] uid={uid} stage_id={stage_id}: duplicate "
                "effective stage_id; first declared at "
                f"stage_index={previous_index}, repeated at "
                f"stage_index={index}."
            )
        stage_index_by_id[stage_id] = index
        if len(manipulated) != 1:
            raise ValueError(
                f"[stage_plan] uid={uid} stage_id={stage_id}: expected exactly "
                f"1 manipulated_object, got {len(manipulated)}."
            )
        if not isinstance(manipulated[0], Mapping):
            raise ValueError(
                f"[stage_plan] uid={uid} stage_id={stage_id}: "
                "manipulated_object must be an object."
            )
        manipulated_object = copy.deepcopy(dict(manipulated[0]))
        if not _clean_str(manipulated_object.get("name")):
            raise ValueError(
                f"[stage_plan] uid={uid} stage_id={stage_id}: "
                "manipulated_object.name is empty."
            )
        executable.append((index, stage, manipulated_object))

    ambiguous = _ambiguous_name(executable)
    if ambiguous is not None:
        stage_id, name, _ = ambiguous
        raise ValueError(
            f"[stage_plan] uid={uid} stage_id={stage_id} object='{name}': "
            "multi-stage manipulated object name is ambiguous without a "
            "unique runtime instance_name."
        )

    stage_plan: list[dict[str, Any]] = []
    streams: list[dict[str, Any]] = []
    stream_by_key: dict[str, dict[str, Any]] = {}
    owner_configs: dict[
        str,
        tuple[dict[str, Any], dict[str, Any], dict[str, Any]],
    ] = {}

    for stage_index, stage, manipulated_object in executable:
        stage_id = _clean_str(stage.get("stage_id")) or f"s{stage_index + 1}"
        safe_stage_id = _safe_stage_id(
            stage_id,
            fallback=f"s{stage_index + 1}",
        )
        runtime_key, instance_name, runtime_status = _stage_object_runtime_key(
            manipulated_object
        )
        stream = stream_by_key.get(runtime_key)
        if stream is None:
            object_id = f"obj_{safe_stage_id}"
            region, tracking, geometry = _compile_target_configs(
                pipeline_config,
                stage_id=stage_id,
                object_id=object_id,
                runtime_key=runtime_key,
                instance_name=instance_name,
                manipulated_object=manipulated_object,
            )
            owner_configs[runtime_key] = (
                copy.deepcopy(region),
                copy.deepcopy(tracking),
                copy.deepcopy(geometry),
            )
            stream = {
                "object_index": len(streams),
                "object_id": object_id,
                "safe_object_id": _safe_stage_id(
                    object_id,
                    fallback=f"obj_s{stage_index + 1}",
                ),
                "owner_stage_id": stage_id,
                "owner_stage_index": stage_index,
                "stage_ids": [],
                "stage_indices": [],
                "runtime_object_key": runtime_key,
                "runtime_match_status": runtime_status,
                "runtime_instance_name": instance_name,
                "manipulated_object": copy.deepcopy(manipulated_object),
                "region_target_cfg": copy.deepcopy(region),
                "tracking_target_cfg": copy.deepcopy(tracking),
                "geometry_target_cfg": copy.deepcopy(geometry),
            }
            stream_by_key[runtime_key] = stream
            streams.append(stream)
        object_id = str(stream["object_id"])
        owner_stage_id = str(stream["owner_stage_id"])
        region, tracking, geometry = owner_configs[runtime_key]
        stream["stage_ids"].append(stage_id)
        stream["stage_indices"].append(stage_index)

        stage_plan.append(
            {
                "stage_index": stage_index,
                "stage_id": stage_id,
                "safe_stage_id": safe_stage_id,
                "object_id": object_id,
                "task_type": _clean_str(stage.get("task_type")) or "unknown",
                "interaction_mode": (
                    _clean_str(stage.get("interaction_mode")) or "unknown"
                ),
                "gripper_plan": copy.deepcopy(
                    dict(stage.get("gripper_plan", {}) or {})
                ),
                "gripper_inference": copy.deepcopy(
                    dict(stage.get("gripper_inference", {}) or {})
                ),
                "manipulated_object": copy.deepcopy(manipulated_object),
                "target_objects": copy.deepcopy(
                    list(stage.get("target_objects", []) or [])
                ),
                "notes": _clean_str(stage.get("notes")),
                "runtime_match_status": runtime_status,
                "runtime_instance_name": instance_name,
                "runtime_object_key": runtime_key,
                "object_stream_key": runtime_key,
                "owner_stage_id": owner_stage_id,
                "region_target_cfg": copy.deepcopy(region),
                "tracking_target_cfg": copy.deepcopy(tracking),
                "geometry_target_cfg": copy.deepcopy(geometry),
            }
        )

    return {
        "task_spec": task_spec,
        "stage_plan": stage_plan,
        "object_stream_plan": copy.deepcopy(streams),
    }


def compile_stage_plan(
    *,
    uid: str,
    metadata: dict[str, Any],
    pipeline_config: dict[str, Any],
) -> list[dict[str, Any]]:
    """Return only the executable stage projection."""

    return compile_task_runtime(
        uid=uid,
        metadata=metadata,
        pipeline_config=pipeline_config,
    )["stage_plan"]


def _is_dense_position(value: Any) -> bool:
    return isinstance(value, list) and len(value) >= 3


def densify_center_records(
    records: Sequence[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Carry the nearest known center fields through sparse frame records."""

    normalized = [
        (
            copy.deepcopy(dict(record))
            if isinstance(record, Mapping)
            else {"frame": index}
        )
        for index, record in enumerate(list(records or []))
    ]
    valid = [
        index
        for index, record in enumerate(normalized)
        if _is_dense_position(record.get("pos_world"))
    ]
    if not valid:
        return normalized

    first = valid[0]
    source_index = first
    valid_set = set(valid)
    for index, record in enumerate(normalized):
        if index in valid_set:
            source_index = index
            continue
        if index < first:
            source_index = first
        source = normalized[source_index]
        for key in ("pos_world", "pos_base", "pos_uv"):
            if key in source:
                record[key] = copy.deepcopy(source[key])
    return normalized


def _motion_samples(
    records: Sequence[dict[str, Any]],
    *,
    reference_frame: str,
    start_frame: int | None,
    end_frame: int | None,
) -> list[tuple[int, np.ndarray | None]]:
    if reference_frame not in {"world", "base"}:
        raise ValueError(f"Unsupported reference_frame='{reference_frame}'")
    primary = "pos_world" if reference_frame == "world" else "pos_base"
    result: list[tuple[int, np.ndarray | None]] = []
    for index, value in enumerate(list(records or [])):
        if not isinstance(value, Mapping):
            frame = index
            if start_frame is not None and frame < int(start_frame):
                continue
            if end_frame is not None and frame > int(end_frame):
                continue
            result.append((frame, None))
            continue
        frame = int(value.get("frame", index))
        if start_frame is not None and frame < int(start_frame):
            continue
        if end_frame is not None and frame > int(end_frame):
            continue
        raw = value.get(primary, value.get("pos_world"))
        if raw is None:
            result.append((frame, None))
            continue
        try:
            array = np.asarray(raw, dtype=np.float64).reshape(-1)
        except (TypeError, ValueError):
            result.append((frame, None))
            continue
        if array.size < 3 or not np.all(np.isfinite(array[:3])):
            result.append((frame, None))
            continue
        result.append((frame, array[:3].copy()))
    return result


def _motion_onset_index(
    positions: Sequence[tuple[int, np.ndarray | None]],
    *,
    threshold_m: float,
    min_run: int,
) -> int | None:
    if len(positions) < 2:
        return None
    anchor_index = next(
        (
            index
            for index, (_, position) in enumerate(positions)
            if position is not None
        ),
        None,
    )
    if anchor_index is None:
        return None
    anchor = positions[anchor_index][1]
    assert anchor is not None
    required = max(1, int(min_run))
    run_start: int | None = None
    run_length = 0
    for index in range(anchor_index + 1, len(positions)):
        position = positions[index][1]
        if position is None:
            run_start = None
            run_length = 0
            continue
        displacement = float(np.linalg.norm(position - anchor))
        if displacement >= float(threshold_m):
            if run_start is None:
                run_start = index
            run_length += 1
            if run_length >= required:
                return run_start
        else:
            run_start = None
            run_length = 0
    return None


def compute_motion_onset_frame(
    records: Sequence[dict[str, Any]],
    *,
    threshold_m: float,
    min_run: int,
    reference_frame: str = "world",
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> int | None:
    positions = _motion_samples(
        records,
        reference_frame=reference_frame,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    onset_index = _motion_onset_index(
        positions,
        threshold_m=threshold_m,
        min_run=min_run,
    )
    return None if onset_index is None else int(positions[onset_index][0])


def compute_motion_settle_frame(
    records: Sequence[dict[str, Any]],
    *,
    threshold_m: float,
    min_run: int,
    reference_frame: str = "world",
    start_frame: int | None = None,
    end_frame: int | None = None,
) -> int | None:
    positions = _motion_samples(
        records,
        reference_frame=reference_frame,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    onset_index = _motion_onset_index(
        positions,
        threshold_m=threshold_m,
        min_run=min_run,
    )
    if onset_index is None:
        return None

    required = max(1, int(min_run))
    step_threshold = float(threshold_m) / float(required)
    saw_moving_step = False
    stable_start: int | None = None
    stable_length = 0
    settle_index: int | None = None
    for index in range(1, len(positions)):
        previous_position = positions[index - 1][1]
        position = positions[index][1]
        if previous_position is None or position is None:
            stable_start = None
            stable_length = 0
            continue
        step = float(np.linalg.norm(position - previous_position))
        if step >= step_threshold:
            saw_moving_step = True
            stable_start = None
            stable_length = 0
            settle_index = None
            continue
        if not saw_moving_step:
            continue
        if stable_start is None:
            stable_start = index
        stable_length += 1
        if stable_length >= required and settle_index is None:
            settle_index = stable_start
    if settle_index is None:
        return None
    return int(positions[settle_index][0])


def _frame_index_at_or_after(frames: Sequence[int], value: Any) -> int | None:
    if value is None:
        return None
    threshold = int(value)
    for index, frame in enumerate(frames):
        if int(frame) >= threshold:
            return index
    return len(frames) - 1 if frames else None


def _stage_event_frames(stage: Mapping[str, Any]) -> list[int]:
    events: list[int] = []
    for key in (
        "motion_settle_frame",
        "eef_depart_frame",
        "release_frame",
    ):
        value = stage.get(key)
        if value is not None:
            events.append(int(value))
    for segment in list(stage.get("segments", []) or []):
        if not isinstance(segment, Mapping):
            continue
        for value in segment.values():
            if isinstance(value, (int, np.integer)):
                events.append(int(value))
    return events


def _window(frames: Sequence[int], start: int, end: int) -> dict[str, Any]:
    return {
        "start_index": int(start),
        "end_index": int(end),
        "start_frame": int(frames[start]),
        "end_frame": int(frames[end]),
    }


def assign_stage_timelines(
    *,
    frames: Sequence[int],
    stage_results: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Assign search and active windows to ordered stage results."""

    frame_list = [int(frame) for frame in list(frames or [])]
    stages = [copy.deepcopy(dict(stage)) for stage in list(stage_results or [])]
    empty = {
        "active_stage_by_frame": [],
        "search_windows": {},
        "active_windows": {},
    }
    if not frame_list or not stages:
        return empty

    stage_ids = [
        _clean_str(stage.get("stage_id")) or f"s{index + 1}"
        for index, stage in enumerate(stages)
    ]
    count = len(stages)
    boundaries = [0]
    for index in range(count - 1):
        current = stages[index]
        following = stages[index + 1]
        segment_opens = [
            int(segment["open"])
            for segment in list(current.get("segments", []) or [])
            if isinstance(segment, Mapping) and segment.get("open") is not None
        ]
        if segment_opens:
            event_index = _frame_index_at_or_after(
                frame_list,
                max(segment_opens),
            )
            assert event_index is not None
            boundary = min(event_index + 1, len(frame_list))
            depart = current.get("eef_depart_frame")
            if depart is not None:
                depart_index = _frame_index_at_or_after(
                    frame_list,
                    depart,
                )
                assert depart_index is not None
                boundary = max(boundary, depart_index)
        elif current.get("motion_settle_frame") is not None:
            boundary = _frame_index_at_or_after(
                frame_list,
                current.get("motion_settle_frame"),
            )
            assert boundary is not None
        elif following.get("motion_onset_frame") is not None:
            boundary = _frame_index_at_or_after(
                frame_list,
                following.get("motion_onset_frame"),
            )
            assert boundary is not None
        else:
            boundary = int(round((len(frame_list) - 1) * (index + 1) / count))
        boundary = max(boundaries[-1], min(boundary, len(frame_list)))
        boundaries.append(boundary)
    boundaries.append(len(frame_list))

    active_by_frame: list[dict[str, Any]] = []
    active_windows: dict[str, dict[str, Any]] = {}
    search_windows: dict[str, dict[str, Any]] = {}
    latest_onset_index = 0
    for index, (stage_id, stage) in enumerate(zip(stage_ids, stages)):
        start = min(boundaries[index], len(frame_list))
        stop_exclusive = min(boundaries[index + 1], len(frame_list))
        if start >= stop_exclusive:
            active_windows[stage_id] = {
                "start_index": None,
                "end_index": None,
                "start_frame": None,
                "end_frame": None,
            }
        else:
            end = stop_exclusive - 1
            active_windows[stage_id] = _window(frame_list, start, end)
            for frame_index in range(start, stop_exclusive):
                active_by_frame.append(
                    {
                        "frame": int(frame_list[frame_index]),
                        "stage_id": stage_id,
                    }
                )

        search_start = latest_onset_index
        next_onset = next(
            (
                future.get("motion_onset_frame")
                for future in stages[index + 1 :]
                if future.get("motion_onset_frame") is not None
            ),
            None,
        )
        if next_onset is not None:
            next_onset_index = _frame_index_at_or_after(
                frame_list,
                next_onset,
            )
            assert next_onset_index is not None
            search_end = max(0, next_onset_index - 1)
        else:
            search_end = len(frame_list) - 1
        search_end = max(search_start, search_end)
        search_windows[stage_id] = _window(
            frame_list,
            search_start,
            search_end,
        )
        current_onset = stage.get("motion_onset_frame")
        if current_onset is not None:
            candidate = _frame_index_at_or_after(frame_list, current_onset)
            if candidate is not None:
                latest_onset_index = candidate

    return {
        "active_stage_by_frame": active_by_frame,
        "search_windows": search_windows,
        "active_windows": active_windows,
    }


def validate_stage_timeline(
    *,
    frames: Sequence[int],
    stage_order: Sequence[str],
    timeline: Mapping[str, Any],
) -> None:
    """Validate the data contract returned by :func:`assign_stage_timelines`.

    Stages are orchestration data, not replaceable model backends.  This check
    keeps their order and frame ownership explicit for downstream gripper and
    action consumers.
    """

    frame_list = [int(frame) for frame in frames]
    order = [_clean_str(stage_id) for stage_id in stage_order]
    if not frame_list or not order:
        expected_empty = {
            "active_stage_by_frame": [],
            "search_windows": {},
            "active_windows": {},
        }
        for key, empty_value in expected_empty.items():
            if timeline.get(key, empty_value) != empty_value:
                raise ValueError(
                    "empty stage timeline must not contain frame assignments"
                )
        return
    if len(frame_list) != len(set(frame_list)):
        raise ValueError("stage timeline frames must be unique")
    if frame_list != sorted(frame_list):
        raise ValueError("stage timeline frames must be ordered")
    if any(not stage_id for stage_id in order):
        raise ValueError("stage timeline stage IDs must be non-empty")
    if len(order) != len(set(order)):
        raise ValueError("stage timeline stage IDs must be unique")

    search_windows = timeline.get("search_windows")
    active_windows = timeline.get("active_windows")
    active_rows = timeline.get("active_stage_by_frame")
    if not isinstance(search_windows, Mapping):
        raise ValueError("stage timeline search_windows must be a mapping")
    if not isinstance(active_windows, Mapping):
        raise ValueError("stage timeline active_windows must be a mapping")
    if not isinstance(active_rows, list):
        raise ValueError("stage timeline active_stage_by_frame must be a list")
    if set(search_windows) != set(order):
        raise ValueError("stage timeline search_windows must cover stage_order exactly")
    if set(active_windows) != set(order):
        raise ValueError("stage timeline active_windows must cover stage_order exactly")

    def validate_window(
        stage_id: str,
        label: str,
        raw_window: Any,
    ) -> None:
        if not isinstance(raw_window, Mapping):
            raise ValueError(f"stage {stage_id!r} {label} window must be a mapping")
        values = [
            raw_window.get("start_index"),
            raw_window.get("end_index"),
            raw_window.get("start_frame"),
            raw_window.get("end_frame"),
        ]
        if all(value is None for value in values):
            if label != "active":
                raise ValueError(f"stage {stage_id!r} search window cannot be empty")
            return
        if any(value is None for value in values):
            raise ValueError(
                f"stage {stage_id!r} {label} window is partially specified"
            )
        start = int(values[0])
        end = int(values[1])
        if start < 0 or end < start or end >= len(frame_list):
            raise ValueError(f"stage {stage_id!r} {label} indices are out of range")
        if int(values[2]) != frame_list[start] or int(values[3]) != frame_list[end]:
            raise ValueError(
                f"stage {stage_id!r} {label} frame/index coordinates disagree"
            )

    for stage_id in order:
        validate_window(stage_id, "search", search_windows[stage_id])
        validate_window(stage_id, "active", active_windows[stage_id])

    assigned_frames: list[int] = []
    assigned_order: list[int] = []
    order_index = {stage_id: index for index, stage_id in enumerate(order)}
    for index, row in enumerate(active_rows):
        if not isinstance(row, Mapping):
            raise ValueError(f"active stage row {index} must be a mapping")
        frame = int(row.get("frame", -1))
        stage_id = _clean_str(row.get("stage_id"))
        if stage_id not in order_index:
            raise ValueError(f"active stage row references unknown stage {stage_id!r}")
        assigned_frames.append(frame)
        assigned_order.append(order_index[stage_id])
    if assigned_frames != frame_list:
        raise ValueError("active stage rows must cover trajectory frames exactly once")
    if assigned_order != sorted(assigned_order):
        raise ValueError("active stage rows must preserve stage_order")


def stage_records_by_id(
    stage_payload: Dict[str, Any],
) -> Dict[str, Dict[str, Any]]:
    """Expand stage aliases with deep-copied object record streams."""

    payload = dict(stage_payload or {})
    objects = dict(payload.get("objects", {}) or {})
    stages = dict(payload.get("stages", {}) or {})
    expanded: dict[str, dict[str, Any]] = {}
    for stage_key, raw_stage in stages.items():
        if not isinstance(raw_stage, Mapping):
            continue
        stage = copy.deepcopy(dict(raw_stage))
        object_id = _clean_str(stage.get("object_id"))
        raw_object = objects.get(object_id)
        if isinstance(raw_object, Mapping):
            object_payload = dict(raw_object)
            stage.setdefault(
                "manipulated_object",
                copy.deepcopy(dict(object_payload.get("manipulated_object", {}) or {})),
            )
            obj_key = _clean_str(stage.get("obj_key")) or "obj_visual_center"
            if obj_key in object_payload:
                stage[obj_key] = copy.deepcopy(object_payload[obj_key])
            if "obj_points_traj_path" in object_payload:
                stage["obj_points_traj_path"] = copy.deepcopy(
                    object_payload["obj_points_traj_path"]
                )
        expanded[str(stage_key)] = stage
    return expanded


def _records_by_frame(
    records: Sequence[Mapping[str, Any]],
    *,
    reference_frame: str,
) -> dict[int, np.ndarray]:
    return {
        frame: position
        for frame, position in _motion_samples(
            records,
            reference_frame=reference_frame,
            start_frame=None,
            end_frame=None,
        )
        if position is not None
    }


def _coupling_episodes(
    *,
    frames: Sequence[int],
    ee_records: Sequence[Mapping[str, Any]],
    object_records: Sequence[Mapping[str, Any]],
    motion_threshold_m: float,
    min_run: int,
    reference_frame: str,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    ee_by_frame = _records_by_frame(
        ee_records,
        reference_frame=reference_frame,
    )
    object_by_frame = _records_by_frame(
        object_records,
        reference_frame=reference_frame,
    )
    common = [
        int(frame)
        for frame in frames
        if int(frame) in ee_by_frame and int(frame) in object_by_frame
    ]
    minimum_duration = max(1, int(min_run))
    min_step = float(motion_threshold_m) / (2.0 * minimum_duration)
    jump_limit = float(motion_threshold_m) * 5.0
    cos_threshold = 0.25
    object_steps: list[float] = []
    coupled: list[tuple[int, float, float]] = []
    previous_coupled_index: int | None = None
    episodes_raw: list[list[tuple[int, float, float]]] = []
    initial_distances = [
        float(np.linalg.norm(ee_by_frame[frame] - object_by_frame[frame]))
        for frame in common
    ]
    distance_gate = max(
        0.18,
        (float(min(initial_distances)) + 0.12 if initial_distances else 0.18),
    )

    for index in range(1, len(common)):
        previous_frame = common[index - 1]
        frame = common[index]
        ee_step = ee_by_frame[frame] - ee_by_frame[previous_frame]
        object_step = object_by_frame[frame] - object_by_frame[previous_frame]
        ee_norm = float(np.linalg.norm(ee_step))
        object_norm = float(np.linalg.norm(object_step))
        object_steps.append(object_norm)
        distance = float(np.linalg.norm(ee_by_frame[frame] - object_by_frame[frame]))
        previous_distance = float(
            np.linalg.norm(
                ee_by_frame[previous_frame] - object_by_frame[previous_frame]
            )
        )
        denominator = ee_norm * object_norm
        cosine = (
            float(np.dot(ee_step, object_step) / denominator)
            if denominator > 0.0
            else -1.0
        )
        is_coupled = (
            ee_norm >= min_step
            and object_norm >= min_step
            and cosine >= cos_threshold
            and distance <= distance_gate
            and abs(distance - previous_distance) <= jump_limit
        )
        if not is_coupled:
            previous_coupled_index = None
            continue
        item = (frame, cosine, distance)
        coupled.append(item)
        if previous_coupled_index is None or previous_coupled_index != index - 1:
            episodes_raw.append([item])
        else:
            episodes_raw[-1].append(item)
        previous_coupled_index = index

    raw_episodes: list[dict[str, Any]] = []
    for items in episodes_raw:
        onset = int(items[0][0])
        release = int(items[-1][0])
        distances = [item[2] for item in items]
        raw_episodes.append(
            {
                "coupling_onset_frame": onset,
                "release_frame": release,
                "duration_frames": int(len(items)),
                "coupling_score": float(np.mean([item[1] for item in items])),
                "mean_distance_m": float(np.median(distances)),
                "min_distance_m": float(np.min(distances)),
                "source": "sustained_eef_object_coupling",
            }
        )
    episodes = [
        copy.deepcopy(episode)
        for episode in raw_episodes
        if int(episode["duration_frames"]) >= minimum_duration
    ]
    median_step = float(np.median(object_steps)) if object_steps else 0.0
    diagnostics = {
        "common_frames": len(common),
        "min_step_m": float(min_step),
        "distance_jump_limit_m": float(jump_limit),
        "distance_gate_m": float(distance_gate),
        "cos_threshold": float(cos_threshold),
        "median_object_step_m": median_step,
        "global_compensated": True,
        "num_episodes": len(episodes),
        "num_raw_episodes": len(raw_episodes),
        "duration_filter_applied": len(episodes) != len(raw_episodes),
        "min_primary_duration_frames": minimum_duration,
        "raw_episodes": raw_episodes,
        "episodes": episodes,
    }
    del coupled
    return episodes, diagnostics


def compile_runtime_stage_order(
    *,
    uid: str,
    frames: Sequence[int],
    ee_records: Sequence[Dict[str, Any]],
    stage_plan: Sequence[Dict[str, Any]],
    object_payloads_by_id: Dict[str, Dict[str, Any]],
    motion_threshold_m: float,
    min_run: int,
    reference_frame: str = "world",
    ambiguity_window_frames: int = 3,
) -> Dict[str, Any]:
    """Order multiple semantic stages from sustained EEF/object coupling."""

    annotated_plan = copy.deepcopy(list(stage_plan or []))
    annotated_order = [_clean_str(stage.get("stage_id")) for stage in annotated_plan]
    if len(annotated_plan) <= 1:
        return {
            "stage_plan": annotated_plan,
            "annotated_stage_order": annotated_order,
            "runtime_stage_order": list(annotated_order),
            "diagnostics": {},
        }

    planned_object_ids = {
        _clean_str(stage.get("object_id")) for stage in annotated_plan
    }
    diagnostics_by_object: dict[str, dict[str, Any]] = {}
    episodes_by_object: dict[str, list[dict[str, Any]]] = {}
    object_order = [
        _clean_str(object_id)
        for object_id in object_payloads_by_id
        if _clean_str(object_id) in planned_object_ids
    ]
    object_order.extend(
        object_id for object_id in planned_object_ids if object_id not in object_order
    )
    for object_id in object_order:
        if object_id in episodes_by_object:
            continue
        payload = dict(object_payloads_by_id.get(object_id, {}) or {})
        object_records = list(payload.get("obj_visual_center", []) or [])
        episodes, diagnostics = _coupling_episodes(
            frames=list(frames or []),
            ee_records=list(ee_records or []),
            object_records=object_records,
            motion_threshold_m=motion_threshold_m,
            min_run=min_run,
            reference_frame=reference_frame,
        )
        episodes_by_object[object_id] = episodes
        diagnostics_by_object[object_id] = diagnostics

    assigned_count: dict[str, int] = {}
    ordered: list[dict[str, Any]] = []
    has_uncoupled_fallback = False
    for stage in annotated_plan:
        stage_id = _clean_str(stage.get("stage_id"))
        object_id = _clean_str(stage.get("object_id"))
        episode_index = assigned_count.get(object_id, 0)
        episodes = episodes_by_object.get(object_id, [])
        if episode_index >= len(episodes):
            gripper_plan = dict(stage.get("gripper_plan", {}) or {})
            requires_coupling = any(
                int(gripper_plan.get(key, 0) or 0) > 0
                for key in (
                    "num_close",
                    "num_open",
                    "close",
                    "open",
                )
            )
            if requires_coupling:
                raise ValueError(
                    f"[runtime_stage_order] uid={uid} stage_id={stage_id} "
                    f"object_id={object_id}: no reliable sustained "
                    "EEF-object coupling episode was detected."
                )
            row = copy.deepcopy(stage)
            row.update(
                {
                    "runtime_order_source": "no_gripper_required",
                    "coupling_onset_frame": None,
                    "release_frame": None,
                    "runtime_coupling_score": 0.0,
                }
            )
            ordered.append(row)
            has_uncoupled_fallback = True
            continue
        assigned_count[object_id] = episode_index + 1
        episode = episodes[episode_index]
        row = copy.deepcopy(stage)
        row.update(
            {
                "runtime_order_source": str(episode["source"]),
                "coupling_onset_frame": int(episode["coupling_onset_frame"]),
                "release_frame": int(episode["release_frame"]),
                "runtime_coupling_score": float(episode["coupling_score"]),
                "_runtime_episode_duration": int(episode["duration_frames"]),
            }
        )
        ordered.append(row)

    overlap_events: list[dict[str, Any]] = []
    if not has_uncoupled_fallback:
        ordered.sort(
            key=lambda stage: (
                int(stage["coupling_onset_frame"]),
                -float(stage["runtime_coupling_score"]),
                int(stage.get("stage_index", 0)),
            )
        )
        ambiguity = int(ambiguity_window_frames)
        for previous, current in zip(ordered, ordered[1:]):
            previous_onset = int(previous["coupling_onset_frame"])
            current_onset = int(current["coupling_onset_frame"])
            if current_onset - previous_onset < ambiguity:
                raise ValueError(
                    f"[runtime_stage_order] uid={uid}: ambiguous runtime "
                    "stage order; coupling onsets are too close "
                    f"({previous_onset}, {current_onset})."
                )

        overlap_conflicts: list[dict[str, Any]] = []
        for previous, current in zip(ordered, ordered[1:]):
            previous_onset = int(previous["coupling_onset_frame"])
            current_onset = int(current["coupling_onset_frame"])
            previous_release = int(previous["release_frame"])
            if current_onset - previous_release >= ambiguity:
                continue
            previous_duration = int(previous["_runtime_episode_duration"])
            current_duration = int(current["_runtime_episode_duration"])
            minimum_longer_duration = int(round(1.5 * previous_duration))
            if current_duration >= minimum_longer_duration:
                overlap_events.append(
                    {
                        "stage_id": _clean_str(previous.get("stage_id")),
                        "reason": (
                            "shorter_previous_episode_blocks_longer_"
                            "overlap_no_later_candidate"
                        ),
                        "blocked_onset": int(previous_onset),
                        "current_onset": current_onset,
                        "current_duration_frames": current_duration,
                        "previous_duration_frames": previous_duration,
                    }
                )
                continue
            overlap_conflicts.append(
                {
                    "stage_id": _clean_str(current.get("stage_id")),
                    "reason": ("overlaps_previous_long_stage_no_later_candidate"),
                    "blocked_onset": current_onset,
                    "occupied_until": previous_release,
                }
            )
        if overlap_conflicts:
            raise ValueError(
                f"[runtime_stage_order] uid={uid}: overlapping coupling "
                "episodes cannot be assigned without interrupting an active "
                f"stage: {overlap_conflicts}"
            )

    for index, stage in enumerate(ordered):
        stage.pop("_runtime_episode_duration", None)
        stage["runtime_order_index"] = index

    runtime_order = [_clean_str(stage.get("stage_id")) for stage in ordered]
    diagnostics = {
        "objects": diagnostics_by_object,
        "overlap_resolution": {
            "applied": False,
            "events": overlap_events,
        },
        "annotated_stage_order": annotated_order,
        "runtime_stage_order": runtime_order,
    }
    return {
        "stage_plan": ordered,
        "annotated_stage_order": annotated_order,
        "runtime_stage_order": runtime_order,
        "diagnostics": diagnostics,
    }
