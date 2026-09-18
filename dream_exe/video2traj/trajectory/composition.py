"""Compose prepared EEF/object evidence into current trajectory outputs.

This is the pure boundary that the current monolithic extraction command keeps
inline after tracking, depth, lifting, and projection.  Inputs are already
prepared trajectories; this module does not load video or models, resolve a
bench sample, initialize a simulator, or write artifacts.

The returned dictionaries preserve the current ``ee_traj.json``,
``obj_trajs.json``, ``gripper.json``, optional ``union_traj.json``, and
``action.json`` payload shapes.  Artifact locations are caller-supplied
references only.
"""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Optional

import numpy as np

from ..action import (
    ActionBuilder,
    ActionPlanner,
    StepBudgetResolver,
    external_action_planner_identity,
    load_action_config,
    validate_action_plan,
)
from ..gripper import GripperInferenceBackend, compute_gripper_payload
from .stages import (
    compile_runtime_stage_order,
    compile_task_runtime,
    compute_motion_onset_frame,
    compute_motion_settle_frame,
    densify_center_records,
    resolve_gripper_initial_state,
)


def _copy_mapping(value: Mapping[str, Any] | None) -> dict[str, Any]:
    return copy.deepcopy(dict(value or {}))


def _resolve_pipeline_resource_path(
    value: Any,
    *,
    dataset_config_path: str,
    pipeline_config_source: str,
) -> str | None:
    """Resolve an existing gripper path with current config precedence."""

    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    candidates = [candidate]
    if not candidate.is_absolute():
        for reference in (
            dataset_config_path,
            pipeline_config_source,
        ):
            reference_text = str(reference or "").strip()
            if not reference_text or reference_text.startswith("<"):
                continue
            reference_path = Path(reference_text).expanduser()
            base = reference_path if reference_path.is_dir() else reference_path.parent
            candidates.append(base / candidate)
        candidates.append(Path(__file__).resolve().parents[3] / candidate)
    for path in candidates:
        if path.exists():
            return path.resolve().as_posix()
    return candidate.as_posix()


def _same_resource_path(left: str, right: str) -> bool:
    return Path(left).expanduser().resolve(strict=False) == Path(
        right
    ).expanduser().resolve(strict=False)


def _select_gripper_resource(
    *,
    name: str,
    configured: str | None,
    supplied: Any,
) -> str | None:
    supplied_text = str(supplied or "").strip() or None
    if (
        configured is not None
        and supplied_text is not None
        and not _same_resource_path(configured, supplied_text)
    ):
        raise ValueError(f"resolved pipeline {name} conflicts with gripper_resources")
    return configured or supplied_text


def _eef_frames(ee_trajectory: Mapping[str, Any]) -> list[int]:
    records = list(ee_trajectory.get("eef_controller", []) or [])
    return [
        int(record.get("frame", index))
        for index, record in enumerate(records)
        if isinstance(record, Mapping)
    ]


def _center_quality(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold_m: float,
    reference_frame: str,
) -> dict[str, Any]:
    """Preserve the current extraction command's object-center summary."""

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

    anchor = np.asarray(positions[0], dtype=np.float64).reshape(3)
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


def _interaction_fields(
    evidence: Mapping[str, Any],
) -> dict[str, Any]:
    interaction = _copy_mapping(
        evidence.get("interaction_geometry", {})
        if isinstance(
            evidence.get("interaction_geometry", {}),
            Mapping,
        )
        else {}
    )
    if interaction:
        meta = _copy_mapping(interaction.get("meta", {}))
        return {
            "gripper_geometry_source": str(meta.get("source", "") or ""),
            "gripper_eef_controller": copy.deepcopy(
                list(interaction.get("eef_controller", []) or [])
            ),
            "gripper_eef_tcp": copy.deepcopy(
                list(interaction.get("eef_tcp", []) or [])
            ),
            "gripper_eef_visual_center": copy.deepcopy(
                list(
                    interaction.get(
                        "eef_visual_center",
                        [],
                    )
                    or []
                )
            ),
            "gripper_obj_visual_center": copy.deepcopy(
                list(
                    interaction.get(
                        "obj_visual_center",
                        [],
                    )
                    or []
                )
            ),
            "gripper_geometry_meta": meta,
        }

    if evidence.get("gripper_obj_visual_center", None):
        return {
            "gripper_geometry_source": str(
                evidence.get(
                    "gripper_geometry_source",
                    "",
                )
                or ""
            ),
            "gripper_eef_controller": copy.deepcopy(
                list(
                    evidence.get(
                        "gripper_eef_controller",
                        [],
                    )
                    or []
                )
            ),
            "gripper_eef_tcp": copy.deepcopy(
                list(
                    evidence.get(
                        "gripper_eef_tcp",
                        [],
                    )
                    or []
                )
            ),
            "gripper_eef_visual_center": copy.deepcopy(
                list(
                    evidence.get(
                        "gripper_eef_visual_center",
                        [],
                    )
                    or []
                )
            ),
            "gripper_obj_visual_center": copy.deepcopy(
                list(
                    evidence.get(
                        "gripper_obj_visual_center",
                        [],
                    )
                    or []
                )
            ),
            "gripper_geometry_meta": _copy_mapping(
                evidence.get("gripper_geometry_meta", {})
            ),
        }
    return {}


def _object_payloads(
    *,
    uid: str,
    object_stream_plan: Sequence[Mapping[str, Any]],
    object_evidence: Mapping[str, Mapping[str, Any]],
    common_metadata: Mapping[str, Any],
    motion_threshold_m: float,
    motion_min_run: int,
    reference_frame: str,
) -> dict[str, dict[str, Any]]:
    planned_ids = [
        str(stream.get("object_id", "") or "")
        for stream in list(object_stream_plan or [])
        if isinstance(stream, Mapping)
    ]
    missing = [
        object_id for object_id in planned_ids if object_id not in object_evidence
    ]
    if missing:
        raise KeyError("missing prepared object evidence for: " + ", ".join(missing))
    unexpected = [
        str(object_id)
        for object_id in object_evidence
        if str(object_id) not in set(planned_ids)
    ]
    if unexpected:
        raise ValueError(
            "prepared object evidence is not present in the "
            "compiled object stream plan: " + ", ".join(unexpected)
        )

    payloads: dict[str, dict[str, Any]] = {}
    for stream in list(object_stream_plan or []):
        if not isinstance(stream, Mapping):
            continue
        object_id = str(stream.get("object_id", "") or "")
        evidence = _copy_mapping(object_evidence[object_id])
        raw_records = list(evidence.get("obj_visual_center", []) or [])
        center_records = densify_center_records(raw_records)
        motion_onset = compute_motion_onset_frame(
            center_records,
            threshold_m=float(motion_threshold_m),
            min_run=int(motion_min_run),
            reference_frame=str(reference_frame),
        )
        motion_settle = compute_motion_settle_frame(
            center_records,
            threshold_m=float(motion_threshold_m),
            min_run=int(motion_min_run),
            reference_frame=str(reference_frame),
        )
        quality = _center_quality(
            center_records,
            threshold_m=float(motion_threshold_m),
            reference_frame=str(reference_frame),
        )
        evidence_quality = _copy_mapping(evidence.get("quality", {}))
        quality = {**evidence_quality, **quality}

        points_path = evidence.get(
            "obj_points_traj_path",
            evidence.get("points_json", None),
        )
        flow_path = evidence.get(
            "points_flow_npz",
            None,
        )
        interaction_fields = _interaction_fields(evidence)
        interaction_meta = _copy_mapping(
            interaction_fields.get(
                "gripper_geometry_meta",
                {},
            )
        )
        object_meta = {
            **_copy_mapping(common_metadata),
            **_copy_mapping(evidence.get("meta", {})),
            "uid": str(uid),
            "object_id": object_id,
            "owner_stage_id": str(stream.get("owner_stage_id", "") or ""),
            "owner_stage_index": int(stream.get("owner_stage_index", 0)),
            "stage_ids": copy.deepcopy(list(stream.get("stage_ids", []) or [])),
            "stage_indices": copy.deepcopy(list(stream.get("stage_indices", []) or [])),
            "runtime_object_key": str(stream.get("runtime_object_key", "") or ""),
            "obj_region": copy.deepcopy(evidence.get("obj_region", None)),
            "obj_region_source": copy.deepcopy(evidence.get("obj_region_source", None)),
            "obj_prompt": copy.deepcopy(evidence.get("obj_prompt", None)),
            "obj_num_points": copy.deepcopy(evidence.get("obj_num_points", None)),
            "vis_threshold": copy.deepcopy(evidence.get("vis_threshold", 0.5)),
            "region_json": copy.deepcopy(evidence.get("region_json", None)),
            "tracking_input": copy.deepcopy(evidence.get("tracking_input", None)),
            "tracking_npz": copy.deepcopy(evidence.get("tracking_npz", None)),
            "tracking_mp4": copy.deepcopy(evidence.get("tracking_mp4", None)),
            "points_json": copy.deepcopy(points_path),
            "points_flow_npz": copy.deepcopy(flow_path),
            "quality": copy.deepcopy(quality),
            "interaction_geometry": interaction_meta,
        }
        object_payload = {
            "object_id": object_id,
            "owner_stage_id": str(stream.get("owner_stage_id", "") or ""),
            "owner_stage_index": int(stream.get("owner_stage_index", 0)),
            "stage_ids": copy.deepcopy(list(stream.get("stage_ids", []) or [])),
            "stage_indices": copy.deepcopy(list(stream.get("stage_indices", []) or [])),
            "runtime_object_key": str(stream.get("runtime_object_key", "") or ""),
            "manipulated_object": _copy_mapping(stream.get("manipulated_object", {})),
            "obj_key": "obj_visual_center",
            "motion_onset_frame": motion_onset,
            "motion_settle_frame": motion_settle,
            "obj_points_traj_path": copy.deepcopy(points_path),
            "points_flow_npz": copy.deepcopy(flow_path),
            "region_json": copy.deepcopy(evidence.get("region_json", None)),
            "tracking_npz": copy.deepcopy(evidence.get("tracking_npz", None)),
            "tracking_mp4": copy.deepcopy(evidence.get("tracking_mp4", None)),
            "quality": copy.deepcopy(quality),
            "obj_visual_center": center_records,
            "meta": object_meta,
        }
        object_payload.update(interaction_fields)
        payloads[object_id] = object_payload
    return payloads


def _runtime_order(
    *,
    uid: str,
    frames: Sequence[int],
    ee_records: Sequence[Mapping[str, Any]],
    stage_plan: Sequence[Mapping[str, Any]],
    object_payloads: Mapping[str, Mapping[str, Any]],
    runtime_order_mode: str,
    motion_threshold_m: float,
    motion_min_run: int,
    reference_frame: str,
) -> dict[str, Any]:
    annotated_order = [
        str(stage.get("stage_id", "") or "")
        for stage in list(stage_plan or [])
        if isinstance(stage, Mapping)
    ]
    if str(runtime_order_mode).strip().lower() == "coupling":
        return compile_runtime_stage_order(
            uid=str(uid),
            frames=list(frames),
            ee_records=list(ee_records),
            stage_plan=list(stage_plan),
            object_payloads_by_id=dict(object_payloads),
            motion_threshold_m=float(motion_threshold_m),
            min_run=int(motion_min_run),
            reference_frame=str(reference_frame),
        )
    return {
        "stage_plan": copy.deepcopy(list(stage_plan)),
        "annotated_stage_order": list(annotated_order),
        "runtime_stage_order": list(annotated_order),
        "diagnostics": {
            "mode": "annotated",
            "reason": (
                "task.runtime_stage_order.mode=annotated preserves semantic stage order"
            ),
        },
    }


def _stage_payloads(
    *,
    stage_plan: Sequence[Mapping[str, Any]],
    object_payloads: Mapping[str, Mapping[str, Any]],
    frames: Sequence[int],
) -> dict[str, dict[str, Any]]:
    full_window = {
        "start_index": 0,
        "end_index": max(0, len(frames) - 1),
        "start_frame": (int(frames[0]) if frames else None),
        "end_frame": (int(frames[-1]) if frames else None),
    }
    payloads: dict[str, dict[str, Any]] = {}
    for stage in list(stage_plan or []):
        if not isinstance(stage, Mapping):
            continue
        object_id = str(stage.get("object_id", "") or "")
        object_payload = _copy_mapping(object_payloads.get(object_id, {}))
        stage_index = int(stage.get("stage_index", 0))
        runtime_index = int(stage.get("runtime_order_index", stage_index))
        stage_payload = {
            "stage_id": str(stage.get("stage_id", "") or ""),
            "stage_index": stage_index,
            "runtime_order_index": runtime_index,
            "runtime_order_source": str(stage.get("runtime_order_source", "") or ""),
            "object_id": object_id,
            "task_type": str(stage.get("task_type", "") or ""),
            "interaction_mode": str(stage.get("interaction_mode", "") or ""),
            "gripper_plan": _copy_mapping(stage.get("gripper_plan", {})),
            "gripper_inference": _copy_mapping(stage.get("gripper_inference", {})),
            "manipulated_object": _copy_mapping(stage.get("manipulated_object", {})),
            "target_objects": copy.deepcopy(
                list(stage.get("target_objects", []) or [])
            ),
            "obj_key": "obj_visual_center",
            "motion_onset_frame": stage.get(
                "coupling_onset_frame",
                None,
            ),
            "motion_settle_frame": stage.get(
                "release_frame",
                None,
            ),
            "motion_onset_frame_raw": object_payload.get(
                "motion_onset_frame",
                None,
            ),
            "motion_settle_frame_raw": object_payload.get(
                "motion_settle_frame",
                None,
            ),
            "coupling_onset_frame": stage.get(
                "coupling_onset_frame",
                None,
            ),
            "release_frame": stage.get(
                "release_frame",
                None,
            ),
            "runtime_coupling_score": stage.get(
                "runtime_coupling_score",
                None,
            ),
            "search_window": copy.deepcopy(full_window),
            "active_window": (
                copy.deepcopy(full_window)
                if runtime_index == 0
                else {
                    "start_index": None,
                    "end_index": None,
                    "start_frame": None,
                    "end_frame": None,
                }
            ),
            "tracking_npz": object_payload.get(
                "tracking_npz",
                None,
            ),
            "tracking_mp4": object_payload.get(
                "tracking_mp4",
                None,
            ),
            "region_json": object_payload.get(
                "region_json",
                None,
            ),
            "obj_points_traj_path": object_payload.get(
                "obj_points_traj_path",
                None,
            ),
            "points_flow_npz": object_payload.get(
                "points_flow_npz",
                None,
            ),
            "quality": _copy_mapping(object_payload.get("quality", {})),
            "meta": _copy_mapping(object_payload.get("meta", {})),
        }
        if object_payload.get(
            "gripper_obj_visual_center",
            None,
        ):
            for key in (
                "gripper_geometry_source",
                "gripper_eef_controller",
                "gripper_eef_tcp",
                "gripper_eef_visual_center",
                "gripper_obj_visual_center",
                "gripper_geometry_meta",
            ):
                stage_payload[key] = copy.deepcopy(object_payload.get(key))
        stage_id = str(stage_payload["stage_id"])
        payloads[stage_id] = stage_payload
    return payloads


def _union_payload(
    *,
    uid: str,
    object_payloads: Mapping[str, Mapping[str, Any]],
    metadata: Mapping[str, Any] | None,
) -> Optional[dict[str, Any]]:
    objects: dict[str, dict[str, Any]] = {}
    stages: dict[str, dict[str, Any]] = {}
    for object_id, raw_payload in object_payloads.items():
        payload = _copy_mapping(raw_payload)
        if not payload.get(
            "gripper_obj_visual_center",
            None,
        ):
            continue
        entry = {
            "object_id": str(object_id),
            "stage_ids": copy.deepcopy(list(payload.get("stage_ids", []) or [])),
            "runtime_object_key": str(payload.get("runtime_object_key", "") or ""),
            "eef_controller": copy.deepcopy(
                list(
                    payload.get(
                        "gripper_eef_controller",
                        [],
                    )
                    or []
                )
            ),
            "eef_tcp": copy.deepcopy(
                list(
                    payload.get(
                        "gripper_eef_tcp",
                        [],
                    )
                    or []
                )
            ),
            "eef_visual_center": copy.deepcopy(
                list(
                    payload.get(
                        "gripper_eef_visual_center",
                        [],
                    )
                    or []
                )
            ),
            "obj_visual_center": copy.deepcopy(
                list(
                    payload.get(
                        "gripper_obj_visual_center",
                        [],
                    )
                    or []
                )
            ),
            "geometry_meta": _copy_mapping(payload.get("gripper_geometry_meta", {})),
        }
        objects[str(object_id)] = entry
        for stage_id in list(payload.get("stage_ids", []) or []):
            stages[str(stage_id)] = {
                "stage_id": str(stage_id),
                "object_id": str(object_id),
                "runtime_object_key": entry["runtime_object_key"],
                "eef_controller": entry["eef_controller"],
                "eef_tcp": entry["eef_tcp"],
                "eef_visual_center": entry["eef_visual_center"],
                "obj_visual_center": entry["obj_visual_center"],
                "geometry_meta": entry["geometry_meta"],
            }
    if not objects:
        return None
    return {
        "meta": {
            "uid": str(uid),
            "source": "gripper_traj_union",
            **_copy_mapping(metadata),
        },
        "objects": objects,
        "stages": stages,
    }


def assemble_prepared_trajectory_bundle(
    *,
    uid: str,
    ee_trajectory: Mapping[str, Any],
    task_runtime: Mapping[str, Any],
    object_evidence: Mapping[
        str,
        Mapping[str, Any],
    ],
    runtime_order_mode: str = "annotated",
    motion_threshold_m: float = 0.004,
    motion_min_run: int = 2,
    reference_frame: str = "world",
    common_metadata: Mapping[str, Any] | None = None,
    union_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build current multi-object trajectory payloads from prepared evidence.

    ``object_evidence`` is keyed by compiled ``object_id``.  Each value must
    provide ``obj_visual_center`` and may provide current artifact references
    (``obj_points_traj_path``, ``points_flow_npz``, ``region_json``,
    ``tracking_npz``, and ``tracking_mp4``) plus optional interaction geometry.
    """

    stage_plan = copy.deepcopy(list(task_runtime.get("stage_plan", []) or []))
    stream_plan = copy.deepcopy(
        list(
            task_runtime.get(
                "object_stream_plan",
                [],
            )
            or []
        )
    )
    if not stage_plan and not stream_plan:
        if object_evidence:
            raise ValueError(
                "prepared object evidence requires a compiled stage/object stream plan"
            )
        return {
            "obj_traj": None,
            "union_traj": None,
        }
    if not stage_plan or not stream_plan:
        raise ValueError(
            "compiled task runtime must contain both stage_plan and object_stream_plan"
        )

    ee_records = list(ee_trajectory.get("eef_controller", []) or [])
    frames = _eef_frames(ee_trajectory)
    object_payloads = _object_payloads(
        uid=str(uid),
        object_stream_plan=stream_plan,
        object_evidence=object_evidence,
        common_metadata=_copy_mapping(common_metadata),
        motion_threshold_m=float(motion_threshold_m),
        motion_min_run=int(motion_min_run),
        reference_frame=str(reference_frame),
    )
    order_result = _runtime_order(
        uid=str(uid),
        frames=frames,
        ee_records=ee_records,
        stage_plan=stage_plan,
        object_payloads=object_payloads,
        runtime_order_mode=str(runtime_order_mode),
        motion_threshold_m=float(motion_threshold_m),
        motion_min_run=int(motion_min_run),
        reference_frame=str(reference_frame),
    )
    runtime_plan = copy.deepcopy(
        list(
            order_result.get(
                "stage_plan",
                stage_plan,
            )
            or stage_plan
        )
    )
    stage_order = [str(stage.get("stage_id", "") or "") for stage in runtime_plan]
    if not stage_order:
        raise ValueError("compiled task runtime contains no executable stages")
    annotated_order = [str(stage.get("stage_id", "") or "") for stage in stage_plan]
    runtime_stage_order = copy.deepcopy(
        list(
            order_result.get(
                "runtime_stage_order",
                stage_order,
            )
            or stage_order
        )
    )
    diagnostics = _copy_mapping(order_result.get("diagnostics", {}))
    diagnostics.setdefault(
        "mode",
        str(runtime_order_mode).strip().lower(),
    )
    stage_payloads = _stage_payloads(
        stage_plan=runtime_plan,
        object_payloads=object_payloads,
        frames=frames,
    )
    legacy_stage_id = stage_order[0]
    legacy_stage = stage_payloads[legacy_stage_id]
    legacy_object_id = str(legacy_stage.get("object_id", "") or "")
    legacy_object = object_payloads[legacy_object_id]
    active_stage_by_frame = [
        {
            "frame": int(frame),
            "stage_id": str(legacy_stage_id),
        }
        for frame in frames
    ]
    obj_traj = {
        "meta": {
            **_copy_mapping(common_metadata),
            "uid": str(uid),
            "annotated_stage_order": annotated_order,
            "runtime_stage_order": runtime_stage_order,
            "stage_order": runtime_stage_order,
            "legacy_alias_stage_id": legacy_stage_id,
            "legacy_alias_object_id": legacy_object_id,
            "object_order": list(object_payloads.keys()),
            "active_stage_by_frame": active_stage_by_frame,
            "runtime_order_diagnostics": diagnostics,
        },
        "objects": object_payloads,
        "stages": stage_payloads,
        "obj_points_traj_path": legacy_object.get(
            "obj_points_traj_path",
            None,
        ),
        "obj_visual_center": copy.deepcopy(
            list(
                legacy_object.get(
                    "obj_visual_center",
                    [],
                )
                or []
            )
        ),
    }
    return {
        "obj_traj": obj_traj,
        "union_traj": _union_payload(
            uid=str(uid),
            object_payloads=object_payloads,
            metadata=union_metadata,
        ),
    }


def _reconcile_stage_motion(
    *,
    obj_traj: dict[str, Any],
    gripper_payload: dict[str, Any],
    motion_threshold_m: float,
    motion_min_run: int,
    reference_frame: str,
) -> tuple[dict[str, Any], dict[str, Any]]:
    updated_obj = copy.deepcopy(obj_traj)
    updated_gripper = copy.deepcopy(gripper_payload)
    active_rows = copy.deepcopy(
        list(
            updated_gripper.get(
                "active_stage_by_frame",
                [],
            )
            or []
        )
    )
    if active_rows:
        updated_obj.setdefault("meta", {})["active_stage_by_frame"] = active_rows

    stage_results_by_id = {
        str(stage_result.get("stage_id", "") or ""): dict(stage_result)
        for stage_result in list(updated_gripper.get("stage_results", []) or [])
        if isinstance(stage_result, Mapping)
    }
    objects = dict(updated_obj.get("objects", {}) or {})
    stages = dict(updated_obj.get("stages", {}) or {})
    for stage_id, stage_payload in stages.items():
        stage_result = _copy_mapping(stage_results_by_id.get(str(stage_id), {}))
        if not stage_result:
            continue
        stage_payload["search_window"] = _copy_mapping(
            stage_result.get(
                "search_window",
                stage_payload.get("search_window", {}),
            )
        )
        stage_payload["active_window"] = _copy_mapping(
            stage_result.get(
                "active_window",
                stage_payload.get("active_window", {}),
            )
        )
        object_id = str(stage_payload.get("object_id", "") or "")
        object_records = list(
            dict(objects.get(object_id, {}) or {}).get(
                "obj_visual_center",
                [],
            )
            or []
        )
        active_window = _copy_mapping(stage_payload.get("active_window", {}))
        search_window = _copy_mapping(stage_payload.get("search_window", {}))
        start_frame = active_window.get(
            "start_frame",
            search_window.get("start_frame", None),
        )
        end_frame = active_window.get(
            "end_frame",
            search_window.get("end_frame", None),
        )
        local_onset = compute_motion_onset_frame(
            object_records,
            threshold_m=float(motion_threshold_m),
            min_run=int(motion_min_run),
            reference_frame=str(reference_frame),
            start_frame=start_frame,
            end_frame=end_frame,
        )
        local_settle = compute_motion_settle_frame(
            object_records,
            threshold_m=float(motion_threshold_m),
            min_run=int(motion_min_run),
            reference_frame=str(reference_frame),
            start_frame=start_frame,
            end_frame=end_frame,
        )
        stage_payload["motion_onset_frame"] = local_onset
        stage_payload["motion_settle_frame"] = local_settle
        stage_payload["motion_window_source"] = "active_window"
        quality = _copy_mapping(stage_payload.get("quality", {}))
        quality["stage_motion_detected"] = local_onset is not None
        quality["stage_motion_window"] = {
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        stage_payload["quality"] = quality
        stage_result["motion_onset_frame"] = local_onset
        stage_result["motion_settle_frame"] = local_settle
        stage_result["active_window"] = _copy_mapping(
            stage_payload.get("active_window", {})
        )
        stage_result["search_window"] = _copy_mapping(
            stage_payload.get("search_window", {})
        )
        stage_result["quality"] = _copy_mapping(stage_payload.get("quality", {}))
        stage_results_by_id[str(stage_id)] = stage_result

    if stage_results_by_id:
        updated_gripper["stage_results"] = [
            copy.deepcopy(
                stage_results_by_id.get(
                    str(
                        stage_result.get(
                            "stage_id",
                            "",
                        )
                        or ""
                    ),
                    stage_result,
                )
            )
            for stage_result in list(
                updated_gripper.get(
                    "stage_results",
                    [],
                )
                or []
            )
            if isinstance(stage_result, Mapping)
        ]
    updated_obj["stages"] = stages
    return updated_obj, updated_gripper


def validate_composed_trajectory_outputs(
    outputs: Mapping[str, Any],
) -> None:
    """Validate the frame/stage/action boundary shared by all implementations."""

    ee_traj = outputs.get("ee_traj")
    if not isinstance(ee_traj, Mapping):
        raise ValueError("composed outputs must contain an ee_traj mapping")
    records = ee_traj.get("eef_controller")
    if not isinstance(records, list) or not records:
        raise ValueError(
            "composed ee_traj must contain a non-empty eef_controller list"
        )
    frames: list[int] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(f"composed EEF record {index} must be a mapping")
        frames.append(int(record.get("frame", index)))
    if frames != sorted(set(frames)):
        raise ValueError("composed EEF frames must be ordered and unique")

    obj_traj = outputs.get("obj_traj")
    gripper = outputs.get("gripper")
    if gripper is not None:
        if not isinstance(gripper, Mapping):
            raise ValueError("composed gripper output must be a mapping")
        actions = gripper.get("actions")
        if not isinstance(actions, list):
            raise ValueError("composed gripper output must contain an actions list")
        action_frames = [
            int(action.get("frame", index))
            for index, action in enumerate(actions)
            if isinstance(action, Mapping)
        ]
        if len(action_frames) != len(actions) or action_frames != frames:
            raise ValueError(
                "composed gripper actions must align exactly with EEF frames"
            )
        active_rows = gripper.get("active_stage_by_frame")
        if not isinstance(active_rows, list):
            raise ValueError(
                "composed gripper output must contain active_stage_by_frame"
            )
        active_frames = [
            int(row.get("frame", index))
            for index, row in enumerate(active_rows)
            if isinstance(row, Mapping)
        ]
        if len(active_frames) != len(active_rows) or active_frames != frames:
            raise ValueError("active_stage_by_frame must align exactly with EEF frames")

        meta = gripper.get("meta", {})
        stage_order = (
            [str(value) for value in list(meta.get("stage_order", []) or [])]
            if isinstance(meta, Mapping)
            else []
        )
        if len(stage_order) != len(set(stage_order)):
            raise ValueError("gripper stage_order must be unique")
        stage_results = gripper.get("stage_results")
        if not isinstance(stage_results, list):
            raise ValueError("composed gripper output must contain stage_results")
        result_order = [
            str(result.get("stage_id", "") or "")
            for result in stage_results
            if isinstance(result, Mapping)
        ]
        if len(result_order) != len(stage_results) or result_order != stage_order:
            raise ValueError("gripper stage_results must preserve stage_order")
        for stage in stage_results:
            stage_id = str(stage.get("stage_id", "") or "")
            for action in list(stage.get("actions", []) or []):
                if not isinstance(action, Mapping):
                    raise ValueError(
                        f"gripper stage {stage_id!r} action must be a mapping"
                    )
                if int(action.get("frame", -1)) not in set(frames):
                    raise ValueError(
                        f"gripper stage {stage_id!r} action frame is invalid"
                    )
            for segment in list(stage.get("segments", []) or []):
                if not isinstance(segment, Mapping):
                    raise ValueError(
                        f"gripper stage {stage_id!r} segment must be a mapping"
                    )
                for value in segment.values():
                    if value is not None and int(value) not in set(frames):
                        raise ValueError(
                            f"gripper stage {stage_id!r} segment frame is invalid"
                        )

        if isinstance(obj_traj, Mapping):
            obj_meta = obj_traj.get("meta", {})
            obj_order = (
                list(obj_meta.get("stage_order", []) or [])
                if isinstance(obj_meta, Mapping)
                else []
            )
            if [str(value) for value in obj_order] != stage_order:
                raise ValueError("object and gripper stage_order must match")

    action = outputs.get("action")
    if action is not None:
        if not isinstance(action, Mapping):
            raise ValueError("composed action output must be a mapping")
        validate_action_plan(
            action,
            eef_frames=frames,
            gripper_payload=(gripper if isinstance(gripper, Mapping) else None),
        )


def compose_prepared_trajectory_outputs(
    *,
    uid: str,
    cfg: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    ee_trajectory: Mapping[str, Any],
    object_evidence: Mapping[
        str,
        Mapping[str, Any],
    ]
    | None = None,
    metadata: Mapping[str, Any] | None = None,
    artifact_references: Mapping[str, Any] | None = None,
    gripper_resources: Mapping[str, Any] | None = None,
    gripper_inference_backend: Optional[GripperInferenceBackend] = None,
    controller_step_budgets: Optional[tuple[float, float]] = None,
    step_budget_resolver: Optional[StepBudgetResolver] = None,
    action_planner: Optional[ActionPlanner] = None,
    union_metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Compose the current pure trajectory, gripper, and action payloads.

    Path resolution and serialization deliberately stay outside this callable.
    ``artifact_references`` only populates the current ``action`` source
    references.  Controller limits must be explicit or caller-resolved.
    """

    pipeline = _copy_mapping(pipeline_config)
    meta_payload = _copy_mapping(metadata)
    environment_config = _copy_mapping(cfg)
    task_runtime = compile_task_runtime(
        uid=str(uid),
        metadata=meta_payload,
        pipeline_config=pipeline,
    )
    action_pipeline_config = _copy_mapping(
        pipeline.get(
            "action",
            pipeline.get("action_plan", {}),
        )
    )
    motion_threshold_m = float(
        action_pipeline_config.get(
            "object_motion_onset_threshold_m",
            0.004,
        )
        or 0.004
    )
    motion_min_run = int(
        action_pipeline_config.get(
            "object_motion_onset_min_run",
            2,
        )
        or 2
    )
    reference_frame = str(
        action_pipeline_config.get(
            "reference_frame",
            "world",
        )
        or "world"
    )
    runtime_order_config = _copy_mapping(
        dict(pipeline.get("task", {}) or {}).get(
            "runtime_stage_order",
            {},
        )
    )
    runtime_order_mode = (
        str(runtime_order_config.get("mode", "annotated") or "annotated")
        .strip()
        .lower()
    )

    ee_payload = _copy_mapping(ee_trajectory)
    ee_meta = _copy_mapping(ee_payload.get("meta", {}))
    common_metadata = {
        key: copy.deepcopy(ee_meta[key])
        for key in (
            "config_path",
            "pipeline_config_path",
            "input_video",
            "camera_name",
        )
        if key in ee_meta
    }
    bundle = assemble_prepared_trajectory_bundle(
        uid=str(uid),
        ee_trajectory=ee_payload,
        task_runtime=task_runtime,
        object_evidence=dict(object_evidence or {}),
        runtime_order_mode=runtime_order_mode,
        motion_threshold_m=motion_threshold_m,
        motion_min_run=motion_min_run,
        reference_frame=reference_frame,
        common_metadata=common_metadata,
        union_metadata=union_metadata,
    )
    obj_payload = bundle["obj_traj"]
    ee_meta["stage_order"] = (
        copy.deepcopy(
            list(
                dict(obj_payload.get("meta", {}) or {}).get(
                    "stage_order",
                    [],
                )
                or []
            )
        )
        if isinstance(obj_payload, Mapping)
        else []
    )
    ee_meta.setdefault("gripper_method", None)
    ee_payload["meta"] = ee_meta

    gripper_payload: Optional[dict[str, Any]] = None
    if isinstance(obj_payload, dict):
        gripper_config = _copy_mapping(
            pipeline.get(
                "gripper",
                pipeline.get("grasp", {}),
            )
        )
        gripper_config["initial_state"] = resolve_gripper_initial_state(
            pipeline_config=pipeline,
            metadata=meta_payload,
            default="open",
        )
        resources = _copy_mapping(gripper_resources)
        dataset_config_path = str(
            resources.get(
                "dataset_config_path",
                ee_meta.get("config_path", ""),
            )
            or ""
        )
        pipeline_source = str(
            dict(pipeline.get("_meta", {}) or {}).get(
                "source",
                ee_meta.get("pipeline_config_path", ""),
            )
            or ""
        )
        numeric_configured = _resolve_pipeline_resource_path(
            gripper_config.get("numeric_config_path"),
            dataset_config_path=dataset_config_path,
            pipeline_config_source=pipeline_source,
        )
        task_prior_config = _copy_mapping(gripper_config.get("task_prior", {}))
        task_prior_configured = _resolve_pipeline_resource_path(
            task_prior_config.get("config_path"),
            dataset_config_path=dataset_config_path,
            pipeline_config_source=pipeline_source,
        )
        legacy_prior_configured = _resolve_pipeline_resource_path(
            task_prior_config.get("prior_config_path"),
            dataset_config_path=dataset_config_path,
            pipeline_config_source=pipeline_source,
        )
        numeric_path = _select_gripper_resource(
            name="gripper.numeric_config_path",
            configured=numeric_configured,
            supplied=resources.get("numeric_config_path"),
        )
        task_prior_params_path = _select_gripper_resource(
            name="gripper.task_prior.config_path",
            configured=task_prior_configured,
            supplied=resources.get(
                "task_prior_params_config_path",
            ),
        )
        prior_path = _select_gripper_resource(
            name="gripper.task_prior prior library",
            configured=(task_prior_configured or legacy_prior_configured),
            supplied=(resources.get("prior_config_path") or task_prior_params_path),
        )
        gripper_payload = compute_gripper_payload(
            uid=str(uid),
            cfg=environment_config,
            gripper_pipeline_cfg=gripper_config,
            ee_traj=ee_payload,
            obj_traj=obj_payload,
            dataset_config_path=dataset_config_path,
            numeric_config_path=numeric_path,
            task_prior_params_config_path=(task_prior_params_path),
            prior_config_path=prior_path,
            inference_backend=gripper_inference_backend,
        )
        obj_payload, gripper_payload = _reconcile_stage_motion(
            obj_traj=obj_payload,
            gripper_payload=gripper_payload,
            motion_threshold_m=motion_threshold_m,
            motion_min_run=motion_min_run,
            reference_frame=reference_frame,
        )
        # The current command serializes ee_traj before recognition and only
        # then mutates this in-memory metadata field.  Its observable artifact
        # therefore keeps the pre-recognition value (normally ``None``).

    references = _copy_mapping(artifact_references)
    action_config = load_action_config(action_pipeline_config)
    action_payload = None
    if bool(action_config.enabled):
        planner_identity = (
            None
            if action_planner is None
            else external_action_planner_identity(action_planner)
        )
        planner: ActionPlanner = (
            action_planner
            if action_planner is not None
            else ActionBuilder(action_config)
        )
        planned = planner.build(
            uid=str(uid),
            cfg=environment_config,
            ee_traj=ee_payload,
            obj_traj=obj_payload,
            gripper_payload=gripper_payload,
            ee_traj_path=str(references.get("ee_traj_path", "") or ""),
            obj_traj_path=(
                None
                if obj_payload is None
                else str(
                    references.get(
                        "obj_traj_path",
                        "",
                    )
                    or ""
                )
            ),
            gripper_path=(
                None
                if gripper_payload is None
                else str(
                    references.get(
                        "gripper_path",
                        "",
                    )
                    or ""
                )
            ),
            controller_step_budgets=controller_step_budgets,
            step_budget_resolver=step_budget_resolver,
        )
        if not isinstance(planned, Mapping):
            raise TypeError("action planner must return a mapping")
        action_payload = copy.deepcopy(dict(planned))
        if planner_identity is not None:
            action_meta = action_payload.get("meta")
            if not isinstance(action_meta, Mapping):
                raise ValueError("action plan must contain a meta mapping")
            action_meta = copy.deepcopy(dict(action_meta))
            planner_meta = action_meta.get("planner")
            if not isinstance(planner_meta, Mapping):
                raise ValueError("action meta.planner must be a mapping")
            planner_meta = copy.deepcopy(dict(planner_meta))
            declared_identity = planner_meta.get("provider")
            if declared_identity is not None and declared_identity != planner_identity:
                raise ValueError(
                    "action planner output provider identity conflicts "
                    "with the injected planner"
                )
            planner_meta["provider"] = planner_identity
            action_meta["planner"] = planner_meta
            action_payload["meta"] = action_meta

    outputs = {
        "ee_traj": ee_payload,
        "obj_traj": obj_payload,
        "union_traj": bundle["union_traj"],
        "gripper": gripper_payload,
        "action": action_payload,
    }
    validate_composed_trajectory_outputs(outputs)
    return outputs


__all__ = [
    "assemble_prepared_trajectory_bundle",
    "compose_prepared_trajectory_outputs",
    "validate_composed_trajectory_outputs",
]
