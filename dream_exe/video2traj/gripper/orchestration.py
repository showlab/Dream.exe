"""Environment-independent gripper orchestration for video-to-trajectory.

This module coordinates the already environment-independent recognizers.  It
does not resolve benchmark paths, initialize a simulator, or write artifacts.
"""

from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Optional, Sequence

from .contract import (
    GripperInferenceBackend,
    GripperInferenceInput,
    GripperResourcePaths,
    GripperStageContext,
    external_gripper_backend_identity,
    normalize_gripper_inference_output,
    validate_gripper_inference_input,
)
from .stages import (
    assign_stage_timelines,
    normalize_gripper_initial_state,
    stage_records_by_id,
)
from .strategy import GripperRecognizer
from ..trajectory.stages import validate_stage_timeline


_EEF_KEY = "eef_controller"
_OBJECT_KEY = "obj_visual_center"


def _clean_text(value: Any) -> str:
    return str(value or "").strip()


def _records(value: Any) -> List[Dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [copy.deepcopy(item) for item in value if isinstance(item, dict)]


def _trajectory_frames(
    ee_traj: Dict[str, Any],
) -> tuple[List[Dict[str, Any]], List[int]]:
    rows = _records(ee_traj.get(_EEF_KEY))
    frames = [int(row.get("frame", index)) for index, row in enumerate(rows)]
    return rows, frames


def _window(
    frames: Sequence[int],
    start_index: int,
    end_index: int,
) -> Dict[str, Optional[int]]:
    if not frames or start_index < 0 or end_index < start_index:
        return {
            "start_index": None,
            "end_index": None,
            "start_frame": None,
            "end_frame": None,
        }
    start = min(int(start_index), len(frames) - 1)
    end = min(int(end_index), len(frames) - 1)
    if end < start:
        return {
            "start_index": None,
            "end_index": None,
            "start_frame": None,
            "end_frame": None,
        }
    return {
        "start_index": start,
        "end_index": end,
        "start_frame": int(frames[start]),
        "end_frame": int(frames[end]),
    }


def _full_window(frames: Sequence[int]) -> Dict[str, Optional[int]]:
    return _window(frames, 0, len(frames) - 1)


def _window_indices(
    window: Dict[str, Any],
    frame_count: int,
) -> tuple[int, int]:
    if frame_count <= 0:
        return 0, -1
    raw_start = window.get("start_index")
    raw_end = window.get("end_index")
    start = 0 if raw_start is None else int(raw_start)
    end = frame_count - 1 if raw_end is None else int(raw_end)
    return max(0, start), min(frame_count - 1, end)


def _first_index_at_or_after(
    frames: Sequence[int],
    target: int,
) -> int:
    for index, frame in enumerate(frames):
        if int(frame) >= int(target):
            return index
    return len(frames)


def _first_index_after(
    frames: Sequence[int],
    target: int,
) -> int:
    for index, frame in enumerate(frames):
        if int(frame) > int(target):
            return index
    return len(frames)


def _first_frame_at_or_after(
    frames: Sequence[int],
    target: int,
) -> Optional[int]:
    index = _first_index_at_or_after(frames, target)
    return int(frames[index]) if index < len(frames) else None


def _first_frame_after(
    frames: Sequence[int],
    target: int,
) -> Optional[int]:
    index = _first_index_after(frames, target)
    return int(frames[index]) if index < len(frames) else None


def _last_frame_before(
    frames: Sequence[int],
    target: int,
) -> Optional[int]:
    previous: Optional[int] = None
    for frame in frames:
        if int(frame) >= int(target):
            break
        previous = int(frame)
    return previous


def _slice_rows(
    rows: Sequence[Dict[str, Any]],
    start: int,
    end: int,
) -> List[Dict[str, Any]]:
    if end < start:
        return []
    return copy.deepcopy(list(rows[start : end + 1]))


def _slice_rows_by_frame(
    rows: Sequence[Dict[str, Any]],
    *,
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> List[Dict[str, Any]]:
    output = []
    for index, row in enumerate(rows):
        frame = int(row.get("frame", index))
        if start_frame is not None and frame < int(start_frame):
            continue
        if end_frame is not None and frame > int(end_frame):
            continue
        output.append(copy.deepcopy(row))
    return output


def _recognition_window_from_frames(
    frames: Sequence[int],
    *,
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> Dict[str, Optional[int]]:
    """Match the current frame-bounded stage-window semantics."""

    if not frames:
        return {
            "start_index": None,
            "end_index": None,
            "start_frame": None,
            "end_frame": None,
        }
    start = 0
    resolved_start = int(frames[0] if start_frame is None else start_frame)
    for index, frame in enumerate(frames):
        if int(frame) >= resolved_start:
            start = index
            resolved_start = int(frame)
            break
    end = len(frames) - 1
    resolved_end = int(frames[-1] if end_frame is None else end_frame)
    for index in range(len(frames) - 1, -1, -1):
        if int(frames[index]) <= resolved_end:
            end = index
            resolved_end = int(frames[index])
            break
    if end < start:
        end = start
        resolved_end = int(frames[end])
    return {
        "start_index": int(start),
        "end_index": int(end),
        "start_frame": int(resolved_start),
        "end_frame": int(resolved_end),
    }


def _segments_from_actions(
    actions: Iterable[Dict[str, Any]],
) -> List[Dict[str, Optional[int]]]:
    """Summarize close/hold/open episodes in frame coordinates."""

    output: List[Dict[str, Optional[int]]] = []
    active: Optional[Dict[str, Optional[int]]] = None
    for index, action in enumerate(actions):
        if not isinstance(action, dict):
            continue
        frame = int(action.get("frame", index))
        event = action.get("event")
        if event == "close" and active is None:
            active = {
                "close": frame,
                "hold_s": frame,
                "hold_e": frame,
                "open": None,
            }
        if event == "open" and active is not None:
            active["open"] = frame
            output.append(active)
            active = None
            continue
        if active is not None and bool(action.get("grasp", False)):
            active["hold_e"] = frame
    if active is not None:
        output.append(active)
    return output


def _last_open_frame(stage_result: Dict[str, Any]) -> Optional[int]:
    openings = [
        int(segment["open"])
        for segment in list(stage_result.get("segments", []) or [])
        if isinstance(segment, dict) and segment.get("open") is not None
    ]
    return max(openings) if openings else None


def _position(
    row: Dict[str, Any],
) -> Optional[tuple[float, float, float]]:
    for key in ("pos_world", "pos", "eef_target_world_6d"):
        value = row.get(key)
        if value is None:
            continue
        try:
            point = tuple(float(component) for component in list(value)[:3])
        except Exception:
            continue
        if len(point) == 3:
            return point
    return None


def _distance(
    left: tuple[float, float, float],
    right: tuple[float, float, float],
) -> float:
    return float(
        sum((float(left[index]) - float(right[index])) ** 2 for index in range(3))
        ** 0.5
    )


def _departure_frame(
    *,
    ee_records: Sequence[Dict[str, Any]],
    object_records: Sequence[Dict[str, Any]],
    open_frame: Optional[int],
) -> Optional[int]:
    """Find the first clear separation relative to the release-frame distance."""

    if open_frame is None:
        return None
    ee_by_frame = {
        int(record.get("frame", index)): dict(record)
        for index, record in enumerate(ee_records)
        if isinstance(record, dict)
    }
    object_by_frame = {
        int(record.get("frame", index)): dict(record)
        for index, record in enumerate(object_records)
        if isinstance(record, dict)
    }
    open_frame = int(open_frame)
    open_ee = _position(ee_by_frame.get(open_frame, {}))
    open_object = _position(object_by_frame.get(open_frame, {}))
    open_distance = (
        _distance(open_ee, open_object)
        if open_ee is not None and open_object is not None
        else 0.0
    )
    threshold = max(0.03, open_distance + 0.02)
    for record in ee_records:
        frame = int(record.get("frame", 0))
        if frame <= int(open_frame):
            continue
        ee_point = _position(dict(record))
        object_point = _position(object_by_frame.get(frame, {}))
        if ee_point is None or object_point is None:
            continue
        if _distance(ee_point, object_point) >= threshold:
            return int(frame)
    return None


def _attach_departure_frames(
    *,
    ee_traj: Dict[str, Any],
    obj_traj: Dict[str, Any],
    stage_results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach handoff evidence from the normal per-object trajectory.

    The current pipeline deliberately computes this only after every stage has
    been recognized.  Gripper-specific bbox/point-cloud rows are suitable for
    recognition, but the stage timeline is driven by the normal object track.
    """

    ee_records = _records(ee_traj.get(_EEF_KEY))
    objects = dict(obj_traj.get("objects", {}) or {})
    attached: List[Dict[str, Any]] = []
    for index, raw_result in enumerate(stage_results):
        result = copy.deepcopy(dict(raw_result))
        if index < len(stage_results) - 1:
            object_id = _clean_text(result.get("object_id"))
            obj_key = _clean_text(result.get("obj_key")) or _OBJECT_KEY
            object_payload = dict(objects.get(object_id, {}) or {})
            object_records = _records(object_payload.get(obj_key))
            departure = _departure_frame(
                ee_records=ee_records,
                object_records=object_records,
                open_frame=_last_open_frame(result),
            )
            if departure is not None:
                result["eef_depart_frame"] = int(departure)
        attached.append(result)
    return attached


def _recognizer_options(
    *,
    uid: str,
    cfg: Dict[str, Any],
    gripper_strategy: str,
    gripper_method: str,
    gripper_close_cmd: float,
    gripper_open_cmd: float,
    gripper_hold_cmd: float,
    invalid_cmd_mode: str,
    dataset_config_path: str,
    numeric_config_path: Optional[str],
    task_prior_params_config_path: Optional[str],
    prior_config_path: Optional[str],
    task_name: str,
    num_close: Optional[int],
    num_open: Optional[int],
    stage_constrained: bool,
    close_timing_profile: Optional[str],
) -> Dict[str, Any]:
    raw_config = cfg.get("raw", {})
    env_name = raw_config.get("env_name")
    effective_task_name = task_name or env_name
    return {
        "strategy": gripper_strategy,
        "method": gripper_method,
        "ee_key": _EEF_KEY,
        "obj_key": _OBJECT_KEY,
        "return_debug": True,
        "gripper_close_cmd": float(gripper_close_cmd),
        "gripper_open_cmd": float(gripper_open_cmd),
        "gripper_hold_cmd": float(gripper_hold_cmd),
        "invalid_cmd_mode": invalid_cmd_mode,
        "task_name": effective_task_name,
        "env_name": env_name,
        "uid": uid,
        "dataset_config_path": dataset_config_path,
        "numeric_params_config_path": numeric_config_path,
        "task_prior_params_config_path": (task_prior_params_config_path),
        "prior_config_path": prior_config_path,
        "num_close": num_close,
        "num_open": num_open,
        "stage_constrained": bool(stage_constrained),
        "close_timing_profile": close_timing_profile,
    }


def _run_single_stage_recognition(
    *,
    uid: str,
    cfg: Dict[str, Any],
    gripper_strategy: str,
    gripper_method: str,
    gripper_close_cmd: float,
    gripper_open_cmd: float,
    gripper_hold_cmd: float,
    invalid_cmd_mode: str,
    gripper_pipeline_cfg: Dict[str, Any],
    ee_traj: Dict[str, Any],
    obj_traj: Dict[str, Any],
    dataset_config_path: str,
    numeric_config_path: Optional[str],
    task_prior_params_config_path: Optional[str],
    prior_config_path: Optional[str],
    task_name: str,
    num_close: Optional[int],
    num_open: Optional[int],
    stage_context: GripperStageContext,
    stage_constrained: bool = False,
    close_timing_profile: Optional[str] = None,
    inference_backend: Optional[GripperInferenceBackend] = None,
) -> Dict[str, Any]:
    """Execute one recognizer on already prepared trajectory streams."""

    options = _recognizer_options(
        uid=uid,
        cfg=cfg,
        gripper_strategy=gripper_strategy,
        gripper_method=gripper_method,
        gripper_close_cmd=gripper_close_cmd,
        gripper_open_cmd=gripper_open_cmd,
        gripper_hold_cmd=gripper_hold_cmd,
        invalid_cmd_mode=invalid_cmd_mode,
        dataset_config_path=dataset_config_path,
        numeric_config_path=numeric_config_path,
        task_prior_params_config_path=(task_prior_params_config_path),
        prior_config_path=prior_config_path,
        task_name=task_name,
        num_close=num_close,
        num_open=num_open,
        stage_constrained=stage_constrained,
        close_timing_profile=close_timing_profile,
    )
    resources = GripperResourcePaths(
        dataset_config_path=str(dataset_config_path or ""),
        numeric_config_path=(str(numeric_config_path) if numeric_config_path else None),
        task_prior_params_config_path=(
            str(task_prior_params_config_path)
            if task_prior_params_config_path
            else None
        ),
        prior_config_path=(str(prior_config_path) if prior_config_path else None),
    )
    # Keep the built-in recognizer on the exact current input shape.  The
    # replaceable backend contract carries additional orchestration metadata in
    # a detached request so plugins receive a complete, validated interface
    # without changing the scientific default backend's inputs.
    request_ee_traj = copy.deepcopy(ee_traj)
    request_obj_traj = copy.deepcopy(obj_traj)
    request_obj_meta = dict(request_obj_traj.get("meta", {}) or {})
    request_obj_meta.update(
        {
            "stage_id": stage_context.stage_id,
            "object_id": stage_context.object_id,
            "annotated_stage_index": (stage_context.annotated_stage_index),
            "runtime_order_index": stage_context.runtime_order_index,
            "runtime_order_mode": stage_context.runtime_order_mode,
            "recognition_window": copy.deepcopy(dict(stage_context.recognition_window)),
        }
    )
    request_obj_traj["meta"] = request_obj_meta
    request = GripperInferenceInput(
        uid=str(uid),
        ee_trajectory=request_ee_traj,
        object_trajectory=request_obj_traj,
        environment_config=copy.deepcopy(cfg),
        resolved_config=copy.deepcopy(gripper_pipeline_cfg),
        recognizer_options=copy.deepcopy(options),
        resources=resources,
        stage=copy.deepcopy(stage_context),
    )
    validate_gripper_inference_input(request)
    if inference_backend is None:
        provider_identity = None
        raw_output = GripperRecognizer(**options).infer(
            ee_traj=ee_traj,
            obj_traj=obj_traj,
        )
    else:
        provider_identity = external_gripper_backend_identity(inference_backend)
        raw_output = inference_backend.infer(request)
    frames = [
        int(record.get("frame", index))
        for index, record in enumerate(list(ee_traj.get(_EEF_KEY, []) or []))
        if isinstance(record, dict)
    ]
    return normalize_gripper_inference_output(
        raw_output,
        expected_frames=frames,
        allow_legacy_segments=inference_backend is None,
        expected_provider_identity=provider_identity,
    )


def _ordered_stage_records(
    obj_traj: Dict[str, Any],
    stage_order: Sequence[str],
) -> List[Dict[str, Any]]:
    available = stage_records_by_id(obj_traj)
    ordered = []
    for index, stage_id in enumerate(stage_order):
        record = copy.deepcopy(available.get(str(stage_id), {}))
        record["stage_id"] = str(stage_id)
        record.setdefault("stage_index", index)
        record.setdefault("object_id", f"obj_{stage_id}")
        record.setdefault("obj_key", _OBJECT_KEY)
        if (
            record.get("motion_onset_frame") is None
            and record.get("motion_onset_frame_raw") is not None
        ):
            record["motion_onset_frame"] = record["motion_onset_frame_raw"]
        ordered.append(record)
    return ordered


def _stage_object_records(
    stage: Dict[str, Any],
) -> List[Dict[str, Any]]:
    preferred = stage.get("gripper_obj_visual_center")
    if isinstance(preferred, list) and preferred:
        return _records(preferred)
    obj_key = _clean_text(stage.get("obj_key")) or _OBJECT_KEY
    return _records(stage.get(obj_key))


def _stage_eef_records(
    stage: Dict[str, Any],
    fallback: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    preferred = stage.get("gripper_eef_controller")
    if isinstance(preferred, list) and preferred:
        return _records(preferred)
    return copy.deepcopy(list(fallback))


def _scene_records(
    *,
    obj_traj: Dict[str, Any],
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> List[Dict[str, Any]]:
    scene = []
    objects = dict(obj_traj.get("objects", {}) or {})
    for object_id, raw_object in objects.items():
        if not isinstance(raw_object, dict):
            continue
        key = _clean_text(raw_object.get("obj_key")) or _OBJECT_KEY
        rows = _records(raw_object.get(key, raw_object.get(_OBJECT_KEY)))
        scene.append(
            {
                "object_id": str(object_id),
                "obj_key": str(key),
                "records": _slice_rows_by_frame(
                    rows,
                    start_frame=start_frame,
                    end_frame=end_frame,
                ),
            }
        )
    return scene


def _stage_inputs(
    *,
    stage: Dict[str, Any],
    stage_context: GripperStageContext,
    ee_traj: Dict[str, Any],
    global_eef: Sequence[Dict[str, Any]],
    obj_traj: Dict[str, Any],
    start_frame: Optional[int],
    end_frame: Optional[int],
) -> tuple[Dict[str, Any], Dict[str, Any]]:
    source = _clean_text(stage.get("gripper_geometry_source"))
    eef_rows = _stage_eef_records(stage, global_eef)
    object_rows = _stage_object_records(stage)

    ee_input: Dict[str, Any] = copy.deepcopy(dict(ee_traj))
    ee_input[_EEF_KEY] = _slice_rows_by_frame(
        eef_rows,
        start_frame=start_frame,
        end_frame=end_frame,
    )
    if source:
        ee_input["meta"] = {
            **dict(ee_input.get("meta", {}) or {}),
            "gripper_geometry_source": source,
        }
    object_input = {
        "meta": {
            **dict(stage.get("meta", {}) or {}),
            "stage_id": _clean_text(stage.get("stage_id")),
            "object_id": _clean_text(stage.get("object_id")),
            "gripper_geometry_source": source,
            "scene_object_records": _scene_records(
                obj_traj=obj_traj,
                start_frame=start_frame,
                end_frame=end_frame,
            ),
        },
        _OBJECT_KEY: _slice_rows_by_frame(
            object_rows,
            start_frame=start_frame,
            end_frame=end_frame,
        ),
    }
    return ee_input, object_input


def _effective_plan(
    stage: Dict[str, Any],
) -> tuple[
    Dict[str, Any],
    Dict[str, Any],
    Dict[str, Any],
    bool,
    str,
]:
    plan = dict(stage.get("gripper_plan", {}) or {})
    inference = dict(stage.get("gripper_inference", {}) or {})
    reason = _clean_text(inference.get("reason"))

    num_close = plan.get("num_close")
    num_open = plan.get("num_open")
    has_events = int(num_close or 0) > 0 or int(num_open or 0) > 0
    enabled = bool(inference.get("enabled")) if "enabled" in inference else has_events
    if not enabled:
        reason = reason or "zero_gripper_plan"
        effective = {"num_close": 0, "num_open": 0}
    else:
        effective = {
            "num_close": num_close,
            "num_open": num_open,
        }
    inference["enabled"] = bool(enabled)
    return plan, inference, effective, enabled, reason


def _skipped_result(reason: str) -> Dict[str, Any]:
    return {
        "actions": [],
        "meta": {
            "gripper_inference": {
                "enabled": False,
                "reason": reason,
            }
        },
        "debug": {
            "skipped": True,
            "skip_reason": reason,
        },
    }


def _multistage_results(
    *,
    uid: str,
    cfg: Dict[str, Any],
    strategy: str,
    method: str,
    close_cmd: float,
    open_cmd: float,
    hold_cmd: float,
    invalid_mode: str,
    gripper_pipeline_cfg: Dict[str, Any],
    ee_traj: Dict[str, Any],
    obj_traj: Dict[str, Any],
    frames: Sequence[int],
    stage_order: Sequence[str],
    dataset_config_path: str,
    numeric_config_path: Optional[str],
    task_prior_params_config_path: Optional[str],
    prior_config_path: Optional[str],
    default_task_name: str,
    close_profile: str,
    handoff_lead: int,
    runtime_order_mode: str,
    inference_backend: Optional[GripperInferenceBackend],
) -> List[Dict[str, Any]]:
    global_eef = _records(ee_traj.get(_EEF_KEY))
    stages = _ordered_stage_records(obj_traj, stage_order)
    mode = "coupling" if runtime_order_mode == "coupling" else "annotated"
    stage_constrained = len(stages) > 1 and mode == "coupling"

    def raw_onset(stage: Dict[str, Any]) -> Optional[int]:
        for key in ("motion_onset_frame", "motion_onset_frame_raw"):
            value = stage.get(key)
            if value is not None:
                return int(value)
        return None

    results: List[Dict[str, Any]] = []
    stage_start_frame: Optional[int] = int(frames[0]) if frames else None
    for order_index, stage in enumerate(stages):
        stage_id = _clean_text(stage.get("stage_id")) or (f"s{order_index + 1}")
        onset = raw_onset(stage)
        if order_index > 0 and onset is not None and stage_start_frame is None:
            stage_start_frame = _first_frame_at_or_after(
                frames,
                onset - handoff_lead,
            )

        stage_end_frame: Optional[int] = None
        if mode == "coupling" and order_index < len(stages) - 1:
            next_onset = raw_onset(stages[order_index + 1])
            if next_onset is not None:
                stage_end_frame = _last_frame_before(
                    frames,
                    next_onset,
                )
        search_window = _recognition_window_from_frames(
            frames,
            start_frame=stage_start_frame,
            end_frame=stage_end_frame,
        )
        annotated_index = int(stage.get("stage_index", order_index))
        runtime_index = order_index
        stage_context = GripperStageContext(
            stage_id=stage_id,
            object_id=_clean_text(stage.get("object_id")),
            annotated_stage_index=annotated_index,
            runtime_order_index=runtime_index,
            runtime_order_mode=mode,
            stage_order=tuple(str(value) for value in stage_order),
            recognition_window=copy.deepcopy(search_window),
        )
        effective_stage = copy.deepcopy(stage)
        if len(stages) == 1:
            task_prior_config = dict(gripper_pipeline_cfg.get("task_prior", {}) or {})
            configured_plan = dict(effective_stage.get("gripper_plan", {}) or {})
            for name in ("num_close", "num_open"):
                if task_prior_config.get(name) is not None:
                    configured_plan[name] = int(task_prior_config.get(name) or 0)
            effective_stage["gripper_plan"] = configured_plan
        plan, inference, effective, enabled, skip_reason = _effective_plan(
            effective_stage
        )
        task_type = _clean_text(stage.get("task_type"))
        explicit_task_name = _clean_text(
            dict(gripper_pipeline_cfg.get("task_prior", {}) or {}).get("task_name")
        )
        recognition_task_name = (
            explicit_task_name
            if len(stages) == 1 and explicit_task_name
            else task_type or default_task_name
        )
        ee_input, object_input = _stage_inputs(
            stage=stage,
            stage_context=stage_context,
            ee_traj=ee_traj,
            global_eef=global_eef,
            obj_traj=obj_traj,
            start_frame=stage_start_frame,
            end_frame=stage_end_frame,
        )

        if enabled:
            recognized = _run_single_stage_recognition(
                uid=uid,
                cfg=cfg,
                gripper_strategy=strategy,
                gripper_method=method,
                gripper_close_cmd=close_cmd,
                gripper_open_cmd=open_cmd,
                gripper_hold_cmd=hold_cmd,
                invalid_cmd_mode=invalid_mode,
                gripper_pipeline_cfg=gripper_pipeline_cfg,
                ee_traj=ee_input,
                obj_traj=object_input,
                dataset_config_path=dataset_config_path,
                numeric_config_path=numeric_config_path,
                task_prior_params_config_path=(task_prior_params_config_path),
                prior_config_path=prior_config_path,
                task_name=recognition_task_name,
                num_close=effective.get("num_close"),
                num_open=effective.get("num_open"),
                stage_constrained=stage_constrained,
                close_timing_profile=(
                    _clean_text(stage.get("close_timing_profile")) or close_profile
                ),
                stage_context=stage_context,
                inference_backend=inference_backend,
            )
        else:
            recognized = _skipped_result(skip_reason)

        actions = _records(recognized.get("actions"))
        stage_result: Dict[str, Any] = {
            "stage_id": stage_id,
            "stage_index": int(stage.get("stage_index", order_index)),
            "runtime_order_index": stage.get("runtime_order_index"),
            "runtime_order_source": _clean_text(stage.get("runtime_order_source")),
            "object_id": _clean_text(stage.get("object_id")),
            "task_type": task_type,
            "gripper_plan": copy.deepcopy(plan),
            "gripper_inference": copy.deepcopy(inference),
            "effective_gripper_plan": copy.deepcopy(effective),
            "motion_onset_frame": stage.get("motion_onset_frame"),
            "motion_settle_frame": stage.get("motion_settle_frame"),
            "motion_onset_frame_raw": stage.get("motion_onset_frame_raw"),
            "motion_settle_frame_raw": stage.get("motion_settle_frame_raw"),
            "coupling_onset_frame": stage.get("coupling_onset_frame"),
            "release_frame": stage.get("release_frame"),
            "runtime_coupling_score": stage.get("runtime_coupling_score"),
            "search_window": copy.deepcopy(search_window),
            "active_window": copy.deepcopy(search_window),
            "obj_key": (_clean_text(stage.get("obj_key")) or _OBJECT_KEY),
            "segments": _segments_from_actions(actions),
            "actions": actions,
            "meta": copy.deepcopy(dict(recognized.get("meta", {}) or {})),
            "debug": copy.deepcopy(dict(recognized.get("debug", {}) or {})),
            "gripper_geometry_source": _clean_text(
                stage.get("gripper_geometry_source")
            ),
            "gripper_geometry_meta": copy.deepcopy(
                dict(stage.get("gripper_geometry_meta", {}) or {})
            ),
        }

        results.append(stage_result)

        next_start: Optional[int] = None
        last_open = _last_open_frame(stage_result)
        if last_open is not None:
            next_start = _first_frame_after(frames, last_open)
        next_onset = (
            raw_onset(stages[order_index + 1])
            if order_index + 1 < len(stages)
            else None
        )
        if next_start is None and next_onset is not None:
            next_start = _first_frame_at_or_after(
                frames,
                next_onset - handoff_lead,
            )
        if next_start is None and stage.get("motion_settle_frame") is not None:
            next_start = _first_frame_at_or_after(
                frames,
                int(stage["motion_settle_frame"]),
            )
        if next_start is not None:
            stage_start_frame = int(next_start)

    results = _attach_departure_frames(
        ee_traj=ee_traj,
        obj_traj=obj_traj,
        stage_results=results,
    )
    active_timeline = assign_stage_timelines(
        frames=frames,
        stage_results=results,
    )
    validate_stage_timeline(
        frames=frames,
        stage_order=stage_order,
        timeline=active_timeline,
    )
    active_windows = dict(active_timeline.get("active_windows", {}) or {})
    timeline_search_windows = dict(active_timeline.get("search_windows", {}) or {})
    for result in results:
        result["active_window"] = copy.deepcopy(
            active_windows.get(
                result["stage_id"],
                result["active_window"],
            )
        )
        result["timeline_search_window"] = copy.deepcopy(
            timeline_search_windows.get(
                result["stage_id"],
                {},
            )
        )
    return results


def _single_stage_result(
    *,
    uid: str,
    cfg: Dict[str, Any],
    strategy: str,
    method: str,
    close_cmd: float,
    open_cmd: float,
    hold_cmd: float,
    invalid_mode: str,
    gripper_pipeline_cfg: Dict[str, Any],
    ee_traj: Dict[str, Any],
    obj_traj: Dict[str, Any],
    frames: Sequence[int],
    dataset_config_path: str,
    numeric_config_path: Optional[str],
    task_prior_params_config_path: Optional[str],
    prior_config_path: Optional[str],
    raw_task_name: str,
    num_close: Optional[int],
    num_open: Optional[int],
    close_profile: str,
    inference_backend: Optional[GripperInferenceBackend],
) -> List[Dict[str, Any]]:
    window = _full_window(frames)
    object_id = (
        _clean_text(dict(obj_traj.get("meta", {}) or {}).get("legacy_alias_object_id"))
        or "obj_s1"
    )
    stage_context = GripperStageContext(
        stage_id="s1",
        object_id=object_id,
        annotated_stage_index=0,
        runtime_order_index=0,
        runtime_order_mode="annotated",
        stage_order=("s1",),
        recognition_window=window,
    )
    object_input = copy.deepcopy(obj_traj)
    object_meta = dict(object_input.get("meta", {}) or {})
    object_meta.update(
        {
            "stage_id": "s1",
            "object_id": object_id,
            "annotated_stage_index": 0,
            "runtime_order_index": 0,
            "runtime_order_mode": "annotated",
            "recognition_window": copy.deepcopy(window),
        }
    )
    object_input["meta"] = object_meta
    recognized = _run_single_stage_recognition(
        uid=uid,
        cfg=cfg,
        gripper_strategy=strategy,
        gripper_method=method,
        gripper_close_cmd=close_cmd,
        gripper_open_cmd=open_cmd,
        gripper_hold_cmd=hold_cmd,
        invalid_cmd_mode=invalid_mode,
        gripper_pipeline_cfg=gripper_pipeline_cfg,
        ee_traj=ee_traj,
        obj_traj=object_input,
        dataset_config_path=dataset_config_path,
        numeric_config_path=numeric_config_path,
        task_prior_params_config_path=task_prior_params_config_path,
        prior_config_path=prior_config_path,
        task_name=raw_task_name,
        num_close=num_close,
        num_open=num_open,
        stage_constrained=False,
        close_timing_profile=close_profile,
        stage_context=stage_context,
        inference_backend=inference_backend,
    )
    actions = _records(recognized.get("actions"))
    return [
        {
            "stage_id": "s1",
            "stage_index": 0,
            "object_id": object_id,
            "task_type": raw_task_name,
            "gripper_plan": {
                "num_close": num_close,
                "num_open": num_open,
            },
            "motion_onset_frame": None,
            "motion_settle_frame": None,
            "search_window": copy.deepcopy(window),
            "active_window": copy.deepcopy(window),
            "obj_key": _OBJECT_KEY,
            "segments": _segments_from_actions(actions),
            "actions": actions,
            "meta": copy.deepcopy(dict(recognized.get("meta", {}) or {})),
            "debug": copy.deepcopy(dict(recognized.get("debug", {}) or {})),
            "timeline_search_window": copy.deepcopy(window),
        }
    ]


def _active_rows(
    *,
    frames: Sequence[int],
    stage_results: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    timeline = assign_stage_timelines(
        frames=frames,
        stage_results=stage_results,
    )
    validate_stage_timeline(
        frames=frames,
        stage_order=[_clean_text(stage.get("stage_id")) for stage in stage_results],
        timeline=timeline,
    )
    return copy.deepcopy(list(timeline.get("active_stage_by_frame", []) or []))


def _merged_actions(
    *,
    frames: Sequence[int],
    active_rows: Sequence[Dict[str, Any]],
    stage_results: Sequence[Dict[str, Any]],
    initial_state: str,
    close_cmd: float,
    open_cmd: float,
) -> List[Dict[str, Any]]:
    stage_lookup = {str(stage.get("stage_id", "")): stage for stage in stage_results}
    action_lookup = {
        stage_id: {
            int(action.get("frame", index)): action
            for index, action in enumerate(_records(stage.get("actions")))
        }
        for stage_id, stage in stage_lookup.items()
    }
    active_lookup = {
        int(item.get("frame", index)): _clean_text(item.get("stage_id"))
        for index, item in enumerate(active_rows)
        if isinstance(item, dict)
    }
    fallback_stage = (
        _clean_text(stage_results[0].get("stage_id")) if stage_results else ""
    )

    closed = initial_state == "closed"
    valid = True
    output = []
    for index, frame in enumerate(frames):
        stage_id = active_lookup.get(int(frame), fallback_stage)
        stage = stage_lookup.get(stage_id, {})
        action = action_lookup.get(stage_id, {}).get(int(frame))
        event = None
        if action is not None:
            event = action.get("event")
            valid = bool(action.get("valid", True))
        if event == "close":
            closed = True
        elif event == "open":
            closed = False
        output.append(
            {
                "frame": int(frame),
                "state": "hold" if closed else "open",
                "event": event,
                "grasp": int(closed),
                "gripper_cmd": float(close_cmd if closed else open_cmd),
                "valid": bool(valid),
                "stage_id": stage_id,
                "object_id": _clean_text(stage.get("object_id")),
                "obj_key": (_clean_text(stage.get("obj_key")) or _OBJECT_KEY),
            }
        )
    return output


def _event_totals(
    stage_results: Sequence[Dict[str, Any]],
) -> tuple[int, int]:
    closes = 0
    opens = 0
    for stage in stage_results:
        plan = dict(
            stage.get(
                "effective_gripper_plan",
                stage.get("gripper_plan", {}),
            )
            or {}
        )
        closes += max(0, int(plan.get("num_close") or 0))
        opens += max(0, int(plan.get("num_open") or 0))
    return closes, opens


def compute_gripper_payload(
    *,
    uid: str,
    cfg: Dict[str, Any],
    gripper_pipeline_cfg: Dict[str, Any],
    ee_traj: Dict[str, Any],
    obj_traj: Dict[str, Any],
    dataset_config_path: str,
    numeric_config_path: Optional[str],
    task_prior_params_config_path: Optional[str],
    prior_config_path: Optional[str],
    inference_backend: Optional[GripperInferenceBackend] = None,
) -> Dict[str, Any]:
    """Compute a serializable gripper payload without artifact I/O."""

    pipeline = dict(gripper_pipeline_cfg or {})
    task_prior = dict(pipeline.get("task_prior", {}) or {})
    strategy = _clean_text(pipeline.get("strategy")) or "task_prior"
    method = _clean_text(pipeline.get("method")) or "3d"
    close_cmd = float(pipeline.get("gripper_close_cmd", 1.0))
    open_cmd = float(pipeline.get("gripper_open_cmd", -1.0))
    hold_cmd = float(pipeline.get("gripper_hold_cmd", 0.0))
    invalid_mode = _clean_text(pipeline.get("invalid_cmd_mode")) or "hold"
    initial_state = normalize_gripper_initial_state(
        pipeline.get("initial_state"),
        default="open",
    )
    raw_task_name = _clean_text(task_prior.get("task_name"))
    num_close = task_prior.get("num_close")
    num_open = task_prior.get("num_open")
    close_profile = (
        _clean_text(task_prior.get("close_timing_profile")) or "default"
    ).lower()
    handoff_lead = max(
        0,
        int(
            task_prior.get(
                "stage_handoff_search_lead_frames",
                45,
            )
            or 45
        ),
    )

    _, frames = _trajectory_frames(ee_traj)
    object_meta = dict(obj_traj.get("meta", {}) or {})
    stage_order = [
        str(value) for value in list(object_meta.get("stage_order", []) or [])
    ]
    has_stage_plan = bool(stage_order and stage_records_by_id(obj_traj))
    runtime_order_diagnostics = dict(
        object_meta.get("runtime_order_diagnostics", {}) or {}
    )
    runtime_order_mode = (
        _clean_text(runtime_order_diagnostics.get("mode")) or "coupling"
    ).lower()

    raw_environment = cfg.get("raw", {})
    env_name = raw_environment.get("env_name")
    default_task_name = raw_task_name or _clean_text(env_name)

    if has_stage_plan:
        stage_results = _multistage_results(
            uid=uid,
            cfg=cfg,
            strategy=strategy,
            method=method,
            close_cmd=close_cmd,
            open_cmd=open_cmd,
            hold_cmd=hold_cmd,
            invalid_mode=invalid_mode,
            gripper_pipeline_cfg=pipeline,
            ee_traj=ee_traj,
            obj_traj=obj_traj,
            frames=frames,
            stage_order=stage_order,
            dataset_config_path=dataset_config_path,
            numeric_config_path=numeric_config_path,
            task_prior_params_config_path=(task_prior_params_config_path),
            prior_config_path=prior_config_path,
            default_task_name=default_task_name,
            close_profile=close_profile,
            handoff_lead=handoff_lead,
            runtime_order_mode=runtime_order_mode,
            inference_backend=inference_backend,
        )
        annotated_order = [
            str(value)
            for value in list(object_meta.get("annotated_stage_order", []) or [])
        ]
        runtime_order = [
            str(value)
            for value in list(
                object_meta.get("runtime_stage_order", stage_order) or stage_order
            )
        ]
        if len(stage_results) > 1:
            task_key = "multi_stage"
            task_label = "multi_stage"
            algorithm = f"{strategy}_multistage"
        else:
            task_key = stage_results[0]["stage_id"]
            task_label = _clean_text(stage_results[0].get("task_type")) or "unknown"
            algorithm = strategy
    else:
        stage_results = _single_stage_result(
            uid=uid,
            cfg=cfg,
            strategy=strategy,
            method=method,
            close_cmd=close_cmd,
            open_cmd=open_cmd,
            hold_cmd=hold_cmd,
            invalid_mode=invalid_mode,
            gripper_pipeline_cfg=pipeline,
            ee_traj=ee_traj,
            obj_traj=obj_traj,
            frames=frames,
            dataset_config_path=dataset_config_path,
            numeric_config_path=numeric_config_path,
            task_prior_params_config_path=(task_prior_params_config_path),
            prior_config_path=prior_config_path,
            raw_task_name=raw_task_name,
            num_close=num_close,
            num_open=num_open,
            close_profile=close_profile,
            inference_backend=inference_backend,
        )
        stage_order = ["s1"]
        annotated_order = []
        runtime_order = ["s1"]
        task_key = "s1"
        task_label = raw_task_name or "unknown"
        algorithm = strategy

    active_rows = _active_rows(
        frames=frames,
        stage_results=stage_results,
    )
    actions = _merged_actions(
        frames=frames,
        active_rows=active_rows,
        stage_results=stage_results,
        initial_state=initial_state,
        close_cmd=close_cmd,
        open_cmd=open_cmd,
    )
    total_close, total_open = _event_totals(stage_results)
    return {
        "meta": {
            "algorithm": algorithm,
            "method": method,
            "T": int(len(frames)),
            "ee_key": _EEF_KEY,
            "obj_key": _OBJECT_KEY,
            "initial_state": initial_state,
            "gripper_map": {
                "close_cmd": close_cmd,
                "open_cmd": open_cmd,
                "hold_cmd": hold_cmd,
                "invalid_cmd_mode": invalid_mode,
            },
            "task_prior": {
                "task_name": task_label,
                "num_close": total_close,
                "num_open": total_open,
                "source": "stage_results",
                "key": task_key,
            },
            "stage_order": list(stage_order),
            "annotated_stage_order": annotated_order,
            "runtime_stage_order": runtime_order,
        },
        "segments": _segments_from_actions(actions),
        "actions": actions,
        "stage_results": stage_results,
        "active_stage_by_frame": active_rows,
    }


__all__ = ["compute_gripper_payload"]
