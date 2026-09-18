"""Environment-independent construction of frame-aligned action streams.

This module deliberately contains only domain-level trajectory and gripper
logic.  Controller-specific step limits are supplied as values or through an
injected resolver at the package boundary.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Mapping, Optional, Protocol, Sequence, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

from .config import ActionConfig, load_action_config


class StepBudgetResolver(Protocol):
    """Resolve controller motion limits without importing a runtime package."""

    def __call__(
        self,
        *,
        cfg: Mapping[str, Any],
        controller_name: str,
        policy_hz: int,
        reference_frame: str,
        want_orientation: bool,
    ) -> Sequence[float]:
        """Return translation-metres and rotation-radians step budgets."""


ConfigInput = Optional[Union[ActionConfig, Mapping[str, Any]]]


@dataclass
class _Pose:
    frame: int
    pos_world: np.ndarray
    pos_ref: np.ndarray
    rotation_world: Optional[Rotation]
    rotation_ref: Optional[Rotation]
    rotation_world_matrix: Optional[np.ndarray]
    rotation_ref_matrix: Optional[np.ndarray]


def _as_vector(
    value: Any,
    *,
    length: int,
    description: str,
) -> np.ndarray:
    try:
        vector = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(description) from exc
    if vector.shape != (length,) or not np.all(np.isfinite(vector)):
        raise ValueError(description)
    return vector


def _as_rotation_matrix(value: Any, *, description: str) -> np.ndarray:
    try:
        matrix = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(description) from exc
    if matrix.shape != (3, 3) or not np.all(np.isfinite(matrix)):
        raise ValueError(description)
    try:
        Rotation.from_matrix(matrix)
        return matrix.copy()
    except ValueError as exc:
        raise ValueError(description) from exc


def _world_rotation(
    record: Mapping[str, Any],
) -> Tuple[Optional[Rotation], Optional[np.ndarray]]:
    if record.get("R") is not None:
        matrix = _as_rotation_matrix(
            record["R"],
            description="invalid EEF world orientation",
        )
        return Rotation.from_matrix(matrix), matrix
    if record.get("R_world") is not None:
        matrix = _as_rotation_matrix(
            record["R_world"],
            description="invalid EEF world orientation",
        )
        return Rotation.from_matrix(matrix), matrix
    if record.get("quat_wxyz") is not None:
        quat = _as_vector(
            record["quat_wxyz"],
            length=4,
            description="invalid EEF world orientation",
        )
        native_rotation = Rotation.from_quat([quat[1], quat[2], quat[3], quat[0]])
        matrix = native_rotation.as_matrix()
        return Rotation.from_matrix(matrix), matrix
    if record.get("quat_xyzw") is not None:
        quat = _as_vector(
            record["quat_xyzw"],
            length=4,
            description="invalid EEF world orientation",
        )
        native_rotation = Rotation.from_quat(quat)
        matrix = native_rotation.as_matrix()
        return Rotation.from_matrix(matrix), matrix
    if record.get("rotvec_world") is not None:
        native_rotation = Rotation.from_rotvec(
            _as_vector(
                record["rotvec_world"],
                length=3,
                description="invalid EEF world orientation",
            )
        )
        matrix = native_rotation.as_matrix()
        return Rotation.from_matrix(matrix), matrix
    return None, None


def _base_transform(
    cfg: Mapping[str, Any],
    *,
    required: bool,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    candidate: Any = cfg
    for key in ("derived", "eef", "X_wb"):
        if not isinstance(candidate, Mapping) or key not in candidate:
            candidate = None
            break
        candidate = candidate[key]
    if not isinstance(candidate, Mapping):
        if required:
            raise ValueError("reference_frame='base' requires cfg.derived.eef.X_wb")
        return None, None
    try:
        rotation = _as_rotation_matrix(
            candidate.get("R"),
            description="cfg.derived.eef.X_wb.R must be a finite 3x3 matrix",
        )
        translation = _as_vector(
            candidate.get("t"),
            length=3,
            description="cfg.derived.eef.X_wb.t must be a finite 3-vector",
        )
    except ValueError:
        if required:
            raise
        return None, None
    return rotation, translation


def _normalize_config(config: ConfigInput) -> ActionConfig:
    if isinstance(config, ActionConfig):
        return config
    if config is None:
        return load_action_config()
    return load_action_config(dict(config))


def _wants_orientation(controller: str) -> bool:
    normalized = str(controller).strip().upper()
    return "POSE" in normalized and "POSITION" not in normalized


def _coerce_budget_pair(
    values: Sequence[float],
    *,
    require_rotation: bool,
) -> Tuple[float, float]:
    try:
        items = tuple(values)
    except TypeError as exc:
        raise ValueError("controller_step_budgets must be a two-item sequence") from exc
    if len(items) != 2:
        raise ValueError("controller_step_budgets must be a two-item sequence")
    try:
        translation = float(items[0])
        rotation = float(items[1])
    except (TypeError, ValueError) as exc:
        raise ValueError("controller_step_budgets must contain numeric values") from exc
    if (
        not math.isfinite(translation)
        or not math.isfinite(rotation)
        or translation <= 0.0
        or (rotation <= 0.0 if require_rotation else rotation < 0.0)
    ):
        requirement = "> 0" if require_rotation else "translation > 0 and rotation >= 0"
        raise ValueError(
            f"controller_step_budgets values must be finite with {requirement}"
        )
    return translation, rotation


def _resolve_step_budgets(
    *,
    cfg: Mapping[str, Any],
    config: ActionConfig,
    want_orientation: bool,
    controller_step_budgets: Optional[Sequence[float]],
    step_budget_resolver: Optional[StepBudgetResolver],
) -> Tuple[float, float]:
    if controller_step_budgets is not None and step_budget_resolver is not None:
        raise ValueError(
            "provide controller_step_budgets or step_budget_resolver, not both"
        )

    supplied: Optional[Tuple[float, float]] = None
    if controller_step_budgets is not None:
        supplied = _coerce_budget_pair(
            controller_step_budgets,
            require_rotation=want_orientation,
        )

    needs_translation = config.translation_step_budget_m is None
    needs_rotation = want_orientation and config.rotation_step_budget_rad <= 0.0
    if (needs_translation or needs_rotation) and supplied is None:
        if step_budget_resolver is None:
            raise ValueError(
                "controller step budget resolution is required; provide "
                "controller_step_budgets or step_budget_resolver"
            )
        supplied = _coerce_budget_pair(
            step_budget_resolver(
                cfg=cfg,
                controller_name=config.controller,
                policy_hz=config.policy_hz,
                reference_frame=config.reference_frame,
                want_orientation=want_orientation,
            ),
            require_rotation=want_orientation,
        )

    inferred_translation = supplied[0] if supplied is not None else 0.0
    inferred_rotation = supplied[1] if supplied is not None else 0.0
    translation = (
        float(config.translation_step_budget_m)
        if config.translation_step_budget_m is not None
        else inferred_translation
    )
    rotation = (
        float(config.rotation_step_budget_rad)
        if config.rotation_step_budget_rad > 0.0
        else inferred_rotation
    )
    if not math.isfinite(translation) or translation <= 0.0:
        raise ValueError("translation step budget must be finite and > 0")
    if want_orientation and (not math.isfinite(rotation) or rotation <= 0.0):
        raise ValueError("rotation step budget must be finite and > 0")
    return translation, rotation


def _eef_poses(
    ee_traj: Mapping[str, Any],
    *,
    eef_key: str,
    reference_frame: str,
    cfg: Mapping[str, Any],
    want_orientation: bool,
) -> Tuple[list[_Pose], Optional[np.ndarray], Optional[np.ndarray]]:
    records = ee_traj.get(eef_key)
    if not isinstance(records, list) or not records:
        raise KeyError(f"ee_traj missing non-empty key '{eef_key}'")
    R_wb, t_wb = _base_transform(
        cfg,
        required=False,
    )
    poses: list[_Pose] = []
    for index, raw in enumerate(records):
        if not isinstance(raw, Mapping):
            raise ValueError(f"invalid EEF record at index {index}")
        frame = int(raw.get("frame", index))
        pos_world = _as_vector(
            raw.get("pos_world"),
            length=3,
            description=(
                "action extraction requires finite world position "
                f"at frame={frame} key={eef_key}"
            ),
        )
        if want_orientation:
            rotation_world, rotation_world_matrix = _world_rotation(raw)
        else:
            rotation_world, rotation_world_matrix = None, None
        if reference_frame == "base":
            if R_wb is None or t_wb is None:
                raise ValueError(
                    "action extraction requires finite base position "
                    f"at frame={frame} key={eef_key}"
                )
            pos_ref = R_wb.T @ (pos_world - t_wb)
            rotation_ref_matrix = (
                R_wb.T @ rotation_world_matrix
                if rotation_world_matrix is not None
                else None
            )
            rotation_ref = (
                Rotation.from_matrix(rotation_ref_matrix)
                if rotation_ref_matrix is not None
                else None
            )
        elif reference_frame == "world":
            pos_ref = pos_world.copy()
            rotation_ref = rotation_world
            rotation_ref_matrix = rotation_world_matrix
        else:
            raise ValueError("action.reference_frame must be 'world' or 'base'")
        poses.append(
            _Pose(
                frame=frame,
                pos_world=pos_world,
                pos_ref=pos_ref,
                rotation_world=rotation_world,
                rotation_ref=rotation_ref,
                rotation_world_matrix=rotation_world_matrix,
                rotation_ref_matrix=rotation_ref_matrix,
            )
        )
    return poses, R_wb, t_wb


def _records_by_frame(payload: Any) -> Dict[int, Mapping[str, Any]]:
    if not isinstance(payload, list):
        return {}
    result: Dict[int, Mapping[str, Any]] = {}
    for index, row in enumerate(payload):
        if isinstance(row, Mapping):
            result[int(row.get("frame", index))] = row
    return result


def _active_stage_by_frame(obj_traj: Optional[Mapping[str, Any]]) -> Dict[int, Any]:
    if not isinstance(obj_traj, Mapping):
        return {}
    meta = obj_traj.get("meta")
    if not isinstance(meta, Mapping):
        return {}
    return {
        int(row["frame"]): row.get("stage_id")
        for row in meta.get("active_stage_by_frame", [])
        if isinstance(row, Mapping) and "frame" in row
    }


def _stage_spec(
    obj_traj: Optional[Mapping[str, Any]],
    stage_id: Any,
) -> Mapping[str, Any]:
    if not isinstance(obj_traj, Mapping) or stage_id is None:
        return {}
    stages = obj_traj.get("stages")
    if not isinstance(stages, Mapping):
        return {}
    stage = stages.get(stage_id)
    return stage if isinstance(stage, Mapping) else {}


def _object_records(
    obj_traj: Optional[Mapping[str, Any]],
    *,
    object_id: Any,
    obj_key: str,
) -> Dict[int, Mapping[str, Any]]:
    if not isinstance(obj_traj, Mapping):
        return {}
    objects = obj_traj.get("objects")
    if object_id is not None and isinstance(objects, Mapping):
        object_payload = objects.get(object_id)
        if isinstance(object_payload, Mapping):
            records = _records_by_frame(object_payload.get(obj_key))
            if records:
                return records
    return _records_by_frame(obj_traj.get(obj_key))


def _object_point(
    record: Optional[Mapping[str, Any]],
    *,
    reference_frame: str,
    R_wb: Optional[np.ndarray],
    t_wb: Optional[np.ndarray],
) -> Tuple[
    Optional[list[float]],
    Optional[list[float]],
    bool,
    Optional[float],
]:
    if record is None or not bool(record.get("valid", True)):
        return None, None, False, None
    pos_world: Optional[np.ndarray] = None
    pos_base: Optional[np.ndarray] = None
    try:
        if record.get("pos_world") is not None:
            pos_world = _as_vector(
                record["pos_world"],
                length=3,
                description="invalid object world position",
            )
        if record.get("pos_base") is not None:
            pos_base = _as_vector(
                record["pos_base"],
                length=3,
                description="invalid object base position",
            )
    except ValueError:
        return None, None, False, None
    if pos_world is None and pos_base is not None and R_wb is not None:
        assert t_wb is not None
        pos_world = R_wb @ pos_base + t_wb
    if pos_base is None and pos_world is not None and R_wb is not None:
        assert t_wb is not None
        pos_base = R_wb.T @ (pos_world - t_wb)
    pos_ref = pos_world if reference_frame == "world" else pos_base
    if pos_world is None or pos_ref is None:
        return None, None, False, None
    vis_raw = record.get("vis")
    vis = None if vis_raw is None else float(vis_raw)
    return pos_world.tolist(), pos_ref.tolist(), True, vis


def _gripper_rows(
    gripper_payload: Optional[Mapping[str, Any]],
) -> Tuple[Dict[int, Dict[str, Any]], Mapping[str, Any]]:
    if not isinstance(gripper_payload, Mapping):
        return {}, {}
    rows: Dict[int, Dict[str, Any]] = {}
    actions = gripper_payload.get("actions")
    if isinstance(actions, list):
        for index, raw in enumerate(actions):
            if isinstance(raw, Mapping):
                row = dict(raw)
                row["frame"] = int(row.get("frame", index))
                rows[row["frame"]] = row
    meta = gripper_payload.get("meta")
    return rows, meta if isinstance(meta, Mapping) else {}


def _position_from_object_record(
    record: Mapping[str, Any],
    *,
    R_wb: Optional[np.ndarray],
    t_wb: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    try:
        if record.get("pos_world") is not None:
            return _as_vector(
                record["pos_world"],
                length=3,
                description="invalid object position",
            )
        if record.get("pos_base") is not None and R_wb is not None:
            assert t_wb is not None
            return (
                R_wb
                @ _as_vector(
                    record["pos_base"],
                    length=3,
                    description="invalid object position",
                )
                + t_wb
            )
    except ValueError:
        return None
    return None


def _motion_xyz(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    try:
        point = np.asarray(value, dtype=np.float64).reshape(-1)
    except (TypeError, ValueError):
        return None
    if point.size < 3 or not np.all(np.isfinite(point[:3])):
        return None
    return point[:3].copy()


def _records_for_stage(
    obj_traj: Optional[Mapping[str, Any]],
    stage_id: str,
    fallback_key: str,
) -> list[Dict[str, Any]]:
    if not isinstance(obj_traj, Mapping):
        return []
    stages = obj_traj.get("stages")
    stage = (
        dict(stages.get(str(stage_id), {}) or {}) if isinstance(stages, Mapping) else {}
    )
    object_id = str(stage.get("object_id", "") or "")
    objects = obj_traj.get("objects")
    object_payload = (
        dict(objects.get(object_id, {}) or {}) if isinstance(objects, Mapping) else {}
    )
    obj_key = str(stage.get("obj_key", fallback_key) or fallback_key)
    raw_records = object_payload.get(
        obj_key,
        object_payload.get(fallback_key, []),
    )
    return [
        dict(record)
        for record in list(raw_records or [])
        if isinstance(record, Mapping)
    ]


def _active_start_frame_for_stage(
    obj_traj: Optional[Mapping[str, Any]],
    stage_id: str,
) -> Optional[int]:
    if not isinstance(obj_traj, Mapping):
        return None
    stages = obj_traj.get("stages")
    stage = (
        dict(stages.get(str(stage_id), {}) or {}) if isinstance(stages, Mapping) else {}
    )
    active_window = dict(stage.get("active_window", {}) or {})
    if active_window.get("start_frame") is not None:
        return int(active_window["start_frame"])
    meta = obj_traj.get("meta")
    rows = (
        list(meta.get("active_stage_by_frame", []) or [])
        if isinstance(meta, Mapping)
        else []
    )
    frames = [
        int(row["frame"])
        for row in rows
        if isinstance(row, Mapping)
        and row.get("frame") is not None
        and str(row.get("stage_id", "") or "") == str(stage_id)
    ]
    return min(frames) if frames else None


def _motion_onset_frame(
    records: Sequence[Mapping[str, Any]],
    *,
    threshold_m: float,
    min_run: int,
    eef_by_frame: Optional[Mapping[int, Mapping[str, Any]]] = None,
    max_eef_object_distance_m: Optional[float] = None,
) -> Optional[int]:
    finite_records = [
        record
        for record in records
        if _motion_xyz(record.get("pos_world", record.get("pos"))) is not None
    ]
    if not finite_records:
        return None
    anchor = _motion_xyz(
        finite_records[0].get(
            "pos_world",
            finite_records[0].get("pos"),
        )
    )
    if anchor is None:
        return None
    moving: list[bool] = []
    for index, record in enumerate(finite_records):
        position = _motion_xyz(record.get("pos_world", record.get("pos")))
        is_moving = bool(
            position is not None
            and float(np.linalg.norm(position - anchor)) > float(threshold_m)
        )
        if (
            is_moving
            and max_eef_object_distance_m is not None
            and eef_by_frame is not None
        ):
            frame = int(record.get("frame", index))
            eef_record = eef_by_frame.get(frame, {})
            eef_position = _motion_xyz(
                eef_record.get("pos_world", eef_record.get("pos"))
            )
            if eef_position is None or position is None:
                is_moving = False
            else:
                is_moving = bool(
                    float(np.linalg.norm(eef_position - position))
                    <= float(max_eef_object_distance_m)
                )
        moving.append(is_moving)
    run = max(1, int(min_run))
    for index in range(0, max(0, len(moving) - run + 1)):
        if all(moving[index : index + run]):
            return int(finite_records[index].get("frame", index))
    return None


def _aligned_gripper_records(
    *,
    frames: Sequence[int],
    rows: Mapping[int, Mapping[str, Any]],
    gripper_meta: Mapping[str, Any],
) -> list[Dict[str, Any]]:
    gripper_map = gripper_meta.get("gripper_map")
    default_cmd = (
        float(gripper_map.get("hold_cmd", 0.0))
        if isinstance(gripper_map, Mapping)
        else 0.0
    )
    last: Dict[str, Any] = {
        "frame": None,
        "state": "open",
        "event": None,
        "grasp": 0,
        "gripper_cmd": default_cmd,
        "valid": False,
        "source": "default",
    }
    aligned: list[Dict[str, Any]] = []
    for frame in frames:
        raw = dict(rows.get(int(frame), {}) or {})
        if raw:
            current = dict(raw)
            current.update(
                {
                    "frame": int(frame),
                    "state": str(raw.get("state", last["state"]) or last["state"]),
                    "event": raw.get("event"),
                    "grasp": int(raw.get("grasp", last["grasp"])),
                    "gripper_cmd": float(raw.get("gripper_cmd", last["gripper_cmd"])),
                    "valid": bool(raw.get("valid", True)),
                    "source": "gripper_payload",
                }
            )
            last = current
        else:
            current = dict(last)
            current["frame"] = int(frame)
            current["event"] = None
            current["source"] = "held_from_previous"
        aligned.append(current)
    return aligned


def _stage_order(
    obj_traj: Optional[Mapping[str, Any]],
    gripper_rows: Mapping[int, Mapping[str, Any]],
) -> list[Any]:
    if isinstance(obj_traj, Mapping):
        meta = obj_traj.get("meta")
        if isinstance(meta, Mapping):
            order = meta.get("stage_order")
            if isinstance(order, list):
                return list(order)
    seen: list[Any] = []
    for row in gripper_rows.values():
        stage_id = row.get("stage_id")
        if stage_id is not None and stage_id not in seen:
            seen.append(stage_id)
    return seen


def _apply_close_constraint(
    *,
    poses: Sequence[_Pose],
    eef_records: Sequence[Mapping[str, Any]],
    rows: Dict[int, Dict[str, Any]],
    gripper_meta: Mapping[str, Any],
    obj_traj: Optional[Mapping[str, Any]],
    config: ActionConfig,
) -> Dict[str, Any]:
    metadata: Dict[str, Any] = {
        "enabled": bool(config.enforce_close_before_object_motion),
        "applied": False,
        "stages": [],
    }
    frame_order = [pose.frame for pose in poses]
    gripper_records = _aligned_gripper_records(
        frames=frame_order,
        rows=rows,
        gripper_meta=gripper_meta,
    )
    for record in gripper_records:
        rows[int(record["frame"])] = dict(record)
    if not config.enforce_close_before_object_motion or not isinstance(
        obj_traj,
        Mapping,
    ):
        return metadata

    frame_index = {frame: index for index, frame in enumerate(frame_order)}
    eef_by_frame = {
        int(record.get("frame", index)): dict(record)
        for index, record in enumerate(eef_records)
        if isinstance(record, Mapping)
    }
    meta = obj_traj.get("meta")
    stage_order = (
        list(meta.get("stage_order", []) or []) if isinstance(meta, Mapping) else []
    )
    stages = obj_traj.get("stages")
    if not stage_order and isinstance(stages, Mapping):
        stage_order = sorted(str(key) for key in stages)

    close_cmd = 1.0
    open_cmd = -1.0
    for record in gripper_records:
        if record.get("event") == "close":
            close_cmd = float(record.get("gripper_cmd", close_cmd))
        if record.get("event") == "open":
            open_cmd = float(record.get("gripper_cmd", open_cmd))

    for raw_stage_id in stage_order:
        stage_id = str(raw_stage_id)
        object_records = _records_for_stage(
            obj_traj,
            stage_id,
            config.obj_key,
        )
        active_start = _active_start_frame_for_stage(
            obj_traj,
            stage_id,
        )
        if active_start is not None:
            object_records = [
                record
                for index, record in enumerate(object_records)
                if int(record.get("frame", index)) >= int(active_start)
            ]
        onset = _motion_onset_frame(
            object_records,
            threshold_m=config.object_motion_onset_threshold_m,
            min_run=config.object_motion_onset_min_run,
            eef_by_frame=eef_by_frame,
            max_eef_object_distance_m=(
                config.max_eef_object_distance_for_motion_close_m
            ),
        )
        close_indices = [
            index
            for index, record in enumerate(gripper_records)
            if record.get("event") == "close"
            and str(record.get("stage_id", "") or "") == stage_id
        ]
        if onset is None or not close_indices:
            metadata["stages"].append(
                {
                    "stage_id": stage_id,
                    "motion_onset_frame": onset,
                    "original_close_frame": (
                        None
                        if not close_indices
                        else int(gripper_records[close_indices[0]]["frame"])
                    ),
                    "shifted_close_to_frame": None,
                    "applied": False,
                }
            )
            continue
        original_index = close_indices[0]
        original_record = dict(gripper_records[original_index])
        requested_target_frame = int(onset) - max(
            0,
            int(config.close_lead_frames_before_object_motion),
        )
        original_frame = int(original_record.get("frame", frame_order[original_index]))
        target_frame = int(requested_target_frame)
        shift_capped = False
        maximum_shift = config.max_close_shift_frames_before_object_motion
        if maximum_shift is not None:
            minimum_allowed = original_frame - max(
                0,
                int(maximum_shift),
            )
            if target_frame < minimum_allowed:
                target_frame = minimum_allowed
                shift_capped = True
        target_candidates = [
            frame for frame in frame_order if int(frame) >= target_frame
        ]
        if not target_candidates:
            target_candidates = [frame_order[-1]]
        target_frame = int(target_candidates[0])
        target_index = frame_index[target_frame]

        stages_map = obj_traj.get("stages")
        stage = (
            dict(stages_map.get(stage_id, {}) or {})
            if isinstance(stages_map, Mapping)
            else {}
        )
        object_id = str(
            original_record.get(
                "object_id",
                stage.get("object_id", ""),
            )
            or ""
        )
        obj_key = str(
            original_record.get(
                "obj_key",
                stage.get("obj_key", config.obj_key),
            )
            or config.obj_key
        )
        if original_index != target_index:
            gripper_records[original_index] = {
                **dict(gripper_records[original_index]),
                "event": None,
                "state": "open",
                "grasp": 0,
                "gripper_cmd": open_cmd,
            }
        gripper_records[target_index] = {
            **dict(gripper_records[target_index]),
            "frame": target_frame,
            "state": "hold",
            "event": "close",
            "grasp": 1,
            "gripper_cmd": close_cmd,
            "valid": True,
            "source": "gripper_constraint",
            "stage_id": stage_id,
            "object_id": object_id,
            "obj_key": obj_key,
        }
        hold_until_index = len(gripper_records)
        hold_stop_reason = "end_of_records"
        for index in range(target_index + 1, len(gripper_records)):
            record = dict(gripper_records[index] or {})
            if record.get("event") in {"close", "open"}:
                hold_until_index = index
                hold_stop_reason = "next_global_gripper_event"
                break
            record_stage_id = str(record.get("stage_id", "") or "")
            if record_stage_id and record_stage_id != stage_id:
                hold_until_index = index
                hold_stop_reason = "stage_change"
                break
        for index in range(target_index + 1, hold_until_index):
            gripper_records[index] = {
                **dict(gripper_records[index]),
                "state": "hold",
                "event": None,
                "grasp": 1,
                "gripper_cmd": close_cmd,
                "valid": True,
                "source": "gripper_constraint_hold",
                "stage_id": stage_id,
                "object_id": object_id,
                "obj_key": obj_key,
            }

        stage_result = {
            "stage_id": stage_id,
            "object_id": object_id,
            "motion_onset_frame": onset,
            "max_eef_object_distance_for_motion_close_m": (
                None
                if config.max_eef_object_distance_for_motion_close_m is None
                else float(config.max_eef_object_distance_for_motion_close_m)
            ),
            "original_close_frame": original_frame,
            "requested_close_frame": requested_target_frame,
            "shifted_close_to_frame": target_frame,
            "max_close_shift_frames": (
                None if maximum_shift is None else int(maximum_shift)
            ),
            "shift_capped": bool(shift_capped),
            "hold_until_frame": (
                None
                if hold_until_index >= len(frame_order)
                else int(frame_order[hold_until_index])
            ),
            "hold_stop_reason": hold_stop_reason,
            "applied": True,
        }
        metadata["stages"].append(stage_result)
        metadata["applied"] = True

    for record in gripper_records:
        rows[int(record["frame"])] = dict(record)
    return metadata


def _gripper_info(
    row: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    if row is None:
        return {
            "state": "open",
            "event": None,
            "grasp": False,
            "cmd": 0.0,
            "valid": False,
            "stage_id": None,
            "object_id": None,
            "obj_key": None,
        }
    state = str(row.get("state") or "open")
    return {
        "state": state,
        "event": row.get("event"),
        "grasp": bool(row.get("grasp", state != "open")),
        "cmd": float(row.get("gripper_cmd", 0.0)),
        "valid": bool(row.get("valid", True)),
        "stage_id": row.get("stage_id"),
        "object_id": row.get("object_id"),
        "obj_key": row.get("obj_key"),
    }


def _suppress_z_bounce(
    *,
    poses: Sequence[_Pose],
    gripper: Sequence[Dict[str, Any]],
    config: ActionConfig,
    R_wb: Optional[np.ndarray],
    t_wb: Optional[np.ndarray],
) -> Dict[str, Any]:
    result: Dict[str, Any] = {
        "enabled": bool(config.suppress_gripper_z_bounce),
        "applied": False,
        "post_close_window": config.post_close_z_bounce_window,
        "post_close_max_drop_m": config.post_close_z_bounce_max_drop_m,
        "pre_open_window": config.pre_open_z_bounce_window,
        "pre_open_min_rise_m": config.pre_open_z_bounce_min_rise_m,
        "pre_open_max_lift_m": config.pre_open_z_bounce_max_lift_m,
        "epsilon_m": config.z_bounce_epsilon_m,
        "events": [],
    }
    if not config.suppress_gripper_z_bounce:
        return result

    epsilon = max(0.0, float(config.z_bounce_epsilon_m))

    def update_pose(index: int, world_z: float) -> None:
        pose = poses[index]
        pose.pos_world[2] = world_z
        if config.reference_frame == "world":
            pose.pos_ref[2] = world_z
        elif R_wb is not None and t_wb is not None:
            pose.pos_ref = R_wb.T @ (pose.pos_world - t_wb)

    post_close_window = max(0, int(config.post_close_z_bounce_window))
    for event_index, info in enumerate(gripper):
        if info["event"] != "close":
            continue
        if post_close_window <= 0:
            continue
        end_index = min(
            len(poses) - 1,
            event_index + post_close_window,
        )
        for probe_index in range(event_index + 1, end_index + 1):
            if gripper[probe_index]["event"] == "open":
                end_index = probe_index - 1
                break
        if end_index <= event_index:
            continue

        z_values = [
            float(poses[index].pos_world[2])
            for index in range(event_index, end_index + 1)
        ]
        if len(z_values) < 2:
            continue
        starting_z = z_values[0]
        minimum_offset = int(np.argmin(np.asarray(z_values)))
        maximum_after_minimum = max(z_values[minimum_offset:])
        maximum_drop = max(0.0, starting_z - min(z_values))
        maximum_allowed = max(
            0.0,
            float(config.post_close_z_bounce_max_drop_m),
        )
        rebounded = maximum_after_minimum >= starting_z - epsilon
        if maximum_allowed > 0.0 and maximum_drop > maximum_allowed and not rebounded:
            result["events"].append(
                {
                    "type": "post_close_downward_preserved",
                    "close_frame": poses[event_index].frame,
                    "start_frame": poses[event_index + 1].frame,
                    "end_frame": poses[end_index].frame,
                    "max_downward_motion_m": maximum_drop,
                    "max_drop_allowed_m": maximum_allowed,
                    "max_after_min_m": maximum_after_minimum,
                    "reason": "sustained_drop_likely_grasp_seating_motion",
                }
            )
            continue

        z_floor = starting_z
        adjusted: list[Tuple[int, float]] = []
        for index in range(event_index + 1, end_index + 1):
            current_z = float(poses[index].pos_world[2])
            if current_z < z_floor - epsilon:
                drop = z_floor - current_z
                adjusted.append((index, drop))
                update_pose(index, z_floor)
            elif current_z >= z_floor:
                z_floor = current_z
        if adjusted:
            result["events"].append(
                {
                    "type": "post_close_downward",
                    "close_frame": poses[event_index].frame,
                    "start_frame": poses[adjusted[0][0]].frame,
                    "end_frame": poses[adjusted[-1][0]].frame,
                    "num_adjusted_frames": len(adjusted),
                    "max_downward_bounce_removed_m": max(
                        value for _, value in adjusted
                    ),
                }
            )

    pre_open_window = max(0, int(config.pre_open_z_bounce_window))
    minimum_rise = max(0.0, float(config.pre_open_z_bounce_min_rise_m))
    maximum_lift = max(0.0, float(config.pre_open_z_bounce_max_lift_m))
    for event_index, info in enumerate(gripper):
        if (
            info["event"] != "open"
            or event_index <= 1
            or pre_open_window <= 0
            or maximum_lift <= 0.0
        ):
            continue
        previous = gripper[event_index - 1]
        previous_state = str(previous.get("state", "") or "").lower()
        if not previous.get("grasp", False) and previous_state not in {
            "hold",
            "closed",
            "close",
        }:
            continue

        start_index = max(0, event_index - pre_open_window)
        for probe_index in range(event_index - 1, start_index - 1, -1):
            probe_event = gripper[probe_index]["event"]
            if probe_event == "close":
                start_index = probe_index
                break
            if probe_event == "open":
                start_index = probe_index + 1
                break
        if event_index - start_index < 3:
            continue

        z_values = [
            (index, float(poses[index].pos_world[2]))
            for index in range(start_index, event_index + 1)
        ]
        placement_index, placement_z = min(
            z_values,
            key=lambda item: item[1],
        )
        if placement_index >= event_index:
            continue
        maximum_after = max(z for index, z in z_values if index >= placement_index)
        rise = maximum_after - placement_z
        if rise < minimum_rise or rise > maximum_lift:
            continue

        adjusted = []
        for index in range(placement_index + 1, event_index + 1):
            lift = float(poses[index].pos_world[2]) - placement_z
            if lift <= epsilon:
                continue
            adjusted.append((index, lift))
            update_pose(index, placement_z)
        if adjusted:
            result["events"].append(
                {
                    "type": "pre_open_lift",
                    "open_frame": poses[event_index].frame,
                    "placement_frame": poses[placement_index].frame,
                    "start_frame": poses[adjusted[0][0]].frame,
                    "end_frame": poses[adjusted[-1][0]].frame,
                    "num_adjusted_frames": len(adjusted),
                    "max_lift_removed_m": max(value for _, value in adjusted),
                }
            )

    result["applied"] = any(
        event.get("type") != "post_close_downward_preserved"
        for event in result["events"]
    )
    return result


def _compress_indices(
    gripper: Sequence[Dict[str, Any]],
    *,
    config: ActionConfig,
) -> Tuple[list[int], Dict[str, Any]]:
    total = len(gripper)
    metadata = {
        "enabled": bool(config.compress_grasped_motion),
        "applied": False,
        "stride": config.grasped_motion_stride,
        "keep_event_neighbors": config.grasped_motion_keep_event_neighbors,
        "num_input_checkpoints": total,
        "num_output_checkpoints": total,
        "num_skipped_checkpoints": 0,
    }
    if not config.compress_grasped_motion or total <= 2:
        return list(range(total)), metadata
    protected: set[int] = {0, total - 1}
    radius = max(0, config.grasped_motion_keep_event_neighbors)
    for index, info in enumerate(gripper):
        if info["event"] is None:
            continue
        protected.update(range(max(0, index - radius), min(total, index + radius + 1)))
    stride = max(1, config.grasped_motion_stride)
    selected: list[int] = []
    for index, info in enumerate(gripper):
        in_grasp = bool(info["grasp"])
        if not in_grasp or index in protected or index % stride == 0:
            selected.append(index)
    metadata["num_output_checkpoints"] = len(selected)
    metadata["num_skipped_checkpoints"] = total - len(selected)
    metadata["applied"] = len(selected) != total
    return selected, metadata


def _absolute_pose_6d(
    position: np.ndarray,
    rotation: Optional[Rotation],
) -> list[float]:
    rotvec = (
        rotation.as_rotvec() if rotation is not None else np.zeros(3, dtype=np.float64)
    )
    return [*position.tolist(), *rotvec.tolist()]


def _rotvec_to_matrix(rotvec: np.ndarray) -> np.ndarray:
    angle = float(np.linalg.norm(rotvec))
    if angle <= 1e-15:
        return np.eye(3, dtype=np.float64)
    axis = rotvec / angle
    skew = np.array(
        [
            [0.0, -axis[2], axis[1]],
            [axis[2], 0.0, -axis[0]],
            [-axis[1], axis[0], 0.0],
        ],
        dtype=np.float64,
    )
    return (
        np.eye(3, dtype=np.float64)
        + math.sin(angle) * skew
        + (1.0 - math.cos(angle)) * (skew @ skew)
    )


def _planner_metadata(
    *,
    config: ActionConfig,
    translation_budget: float,
    rotation_budget: float,
    constraint: Dict[str, Any],
    compression: Dict[str, Any],
    z_suppression: Dict[str, Any],
    close_gate: Dict[str, Any],
    open_gate: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "enabled": config.enabled,
        "eef_key": config.eef_key,
        "obj_key": config.obj_key,
        "reference_frame": config.reference_frame,
        "controller": config.controller,
        "policy_hz": config.policy_hz,
        "translation_step_budget_m": translation_budget,
        "grasped_translation_step_budget_m": (config.grasped_translation_step_budget_m),
        "rotation_step_budget_rad": rotation_budget,
        "zero_motion_epsilon_m": config.zero_motion_epsilon_m,
        "max_motion_steps_per_segment": (config.max_motion_steps_per_segment),
        "gripper_actuation_mode": config.gripper_actuation_mode,
        "emit_initial_noop_step": config.emit_initial_noop_step,
        "embed_gripper_settle_steps": config.embed_gripper_settle_steps,
        "settle_steps_after_close": config.settle_steps_after_close,
        "settle_steps_after_open": config.settle_steps_after_open,
        "insert_close_completion_gate": (config.insert_close_completion_gate),
        "insert_open_completion_gate": config.insert_open_completion_gate,
        "force_gripper_completion_gates": (config.force_gripper_completion_gates),
        "enforce_close_before_object_motion": (
            config.enforce_close_before_object_motion
        ),
        "object_motion_onset_threshold_m": (config.object_motion_onset_threshold_m),
        "object_motion_onset_min_run": (config.object_motion_onset_min_run),
        "close_lead_frames_before_object_motion": (
            config.close_lead_frames_before_object_motion
        ),
        "max_eef_object_distance_for_motion_close_m": (
            config.max_eef_object_distance_for_motion_close_m
        ),
        "max_close_shift_frames_before_object_motion": (
            config.max_close_shift_frames_before_object_motion
        ),
        "compress_grasped_motion": config.compress_grasped_motion,
        "grasped_motion_stride": config.grasped_motion_stride,
        "grasped_motion_keep_event_neighbors": (
            config.grasped_motion_keep_event_neighbors
        ),
        "rotation_delta_guard_enabled": (config.rotation_delta_guard_enabled),
        "rotation_delta_guard_max_rad": (config.rotation_delta_guard_max_rad),
        "grasp_rotation_guard_enabled": (config.grasp_rotation_guard_enabled),
        "grasp_rotation_guard_max_rad": (config.grasp_rotation_guard_max_rad),
        "grasp_rotation_guard_mode": config.grasp_rotation_guard_mode,
        "suppress_gripper_z_bounce": config.suppress_gripper_z_bounce,
        "post_close_z_bounce_window": (config.post_close_z_bounce_window),
        "post_close_z_bounce_max_drop_m": (config.post_close_z_bounce_max_drop_m),
        "pre_open_z_bounce_window": config.pre_open_z_bounce_window,
        "pre_open_z_bounce_min_rise_m": (config.pre_open_z_bounce_min_rise_m),
        "pre_open_z_bounce_max_lift_m": (config.pre_open_z_bounce_max_lift_m),
        "z_bounce_epsilon_m": config.z_bounce_epsilon_m,
        "must_reach_checkpoints": True,
        "gripper_constraint": constraint,
        "checkpoint_compression": compression,
        "z_bounce_suppression": z_suppression,
        "close_completion_gate": close_gate,
        "open_completion_gate": open_gate,
    }


def build_action(
    *,
    uid: str,
    cfg: Mapping[str, Any],
    ee_traj: Mapping[str, Any],
    obj_traj: Optional[Mapping[str, Any]] = None,
    gripper_payload: Optional[Mapping[str, Any]] = None,
    config: ConfigInput = None,
    ee_traj_path: str = "",
    obj_traj_path: Optional[str] = None,
    gripper_path: Optional[str] = None,
    controller_step_budgets: Optional[Sequence[float]] = None,
    step_budget_resolver: Optional[StepBudgetResolver] = None,
) -> Dict[str, Any]:
    """Build an ``action`` payload from aligned domain trajectories."""

    action_config = _normalize_config(config)
    controller_accepts_orientation = _wants_orientation(action_config.controller)
    poses, R_wb, t_wb = _eef_poses(
        ee_traj,
        eef_key=action_config.eef_key,
        reference_frame=action_config.reference_frame,
        cfg=cfg,
        want_orientation=controller_accepts_orientation,
    )
    has_orientation_input = any(pose.rotation_ref is not None for pose in poses)
    translation_budget, rotation_budget = _resolve_step_budgets(
        cfg=cfg,
        config=action_config,
        want_orientation=has_orientation_input,
        controller_step_budgets=controller_step_budgets,
        step_budget_resolver=step_budget_resolver,
    )

    rows, gripper_meta = _gripper_rows(gripper_payload)
    constraint = _apply_close_constraint(
        poses=poses,
        eef_records=list(ee_traj.get(action_config.eef_key, []) or []),
        rows=rows,
        gripper_meta=gripper_meta,
        obj_traj=obj_traj,
        config=action_config,
    )
    active_stage = _active_stage_by_frame(obj_traj)
    gripper_all = [_gripper_info(rows.get(pose.frame)) for pose in poses]
    for pose, info in zip(poses, gripper_all):
        if info["stage_id"] is None:
            info["stage_id"] = active_stage.get(pose.frame)
        stage = _stage_spec(obj_traj, info["stage_id"])
        if info["object_id"] is None:
            info["object_id"] = stage.get("object_id")
        if info["obj_key"] is None:
            info["obj_key"] = stage.get("obj_key")
        if info["obj_key"] is None:
            info["obj_key"] = action_config.obj_key

    z_suppression = _suppress_z_bounce(
        poses=poses,
        gripper=gripper_all,
        config=action_config,
        R_wb=R_wb,
        t_wb=t_wb,
    )
    selected, compression = _compress_indices(gripper_all, config=action_config)
    poses = [poses[index] for index in selected]
    gripper = [gripper_all[index] for index in selected]

    steps: list[Dict[str, Any]] = []
    checkpoints: list[Dict[str, Any]] = []
    close_gate_events: list[Dict[str, Any]] = []
    open_gate_events: list[Dict[str, Any]] = []
    guarded_count = 0
    grasp_guarded_count = 0
    planned_rotation_matrix: Optional[np.ndarray] = None
    planned_rotation_was_guarded = False

    for checkpoint_index, (pose, grip) in enumerate(zip(poses, gripper)):
        previous_pose = poses[checkpoint_index - 1] if checkpoint_index > 0 else None
        previous_grip = gripper[checkpoint_index - 1] if checkpoint_index > 0 else None
        in_grasp_window = bool(
            checkpoint_index > 0
            and (
                grip["grasp"] or (previous_grip is not None and previous_grip["grasp"])
            )
        )
        segment_translation_budget = (
            float(action_config.grasped_translation_step_budget_m)
            if (
                in_grasp_window
                and action_config.grasped_translation_step_budget_m is not None
            )
            else translation_budget
        )

        if previous_pose is None:
            delta_position = np.zeros(3, dtype=np.float64)
            raw_delta_rotation = np.zeros(3, dtype=np.float64)
            applied_delta_rotation = raw_delta_rotation.copy()
            raw_angle = 0.0
            applied_angle = 0.0
            guard_applied = False
            guard_action: Optional[str] = None
            guard_max: Optional[float] = None
            guard_enabled = True
            planned_rotation_matrix = pose.rotation_ref_matrix
        else:
            delta_position = pose.pos_ref - previous_pose.pos_ref
            if (
                pose.rotation_ref_matrix is not None
                and planned_rotation_matrix is not None
            ):
                raw_delta_rotation = Rotation.from_matrix(
                    pose.rotation_ref_matrix
                    @ (
                        np.linalg.inv(planned_rotation_matrix)
                        if planned_rotation_was_guarded
                        else planned_rotation_matrix.T
                    )
                ).as_rotvec()
                raw_angle = float(np.linalg.norm(raw_delta_rotation))
            else:
                raw_delta_rotation = np.zeros(3, dtype=np.float64)
                raw_angle = 0.0
            applied_delta_rotation = raw_delta_rotation.copy()
            guard_applied = False
            guard_action = None
            guard_max = None
            relevant_in_grasp = bool(in_grasp_window and pose.rotation_ref is not None)
            if (
                relevant_in_grasp
                and action_config.grasp_rotation_guard_enabled
                and raw_angle > action_config.grasp_rotation_guard_max_rad
            ):
                guard_applied = True
                guard_action = action_config.grasp_rotation_guard_mode
                guard_max = action_config.grasp_rotation_guard_max_rad
                if guard_action == "freeze":
                    applied_delta_rotation = np.zeros(3, dtype=np.float64)
                else:
                    applied_delta_rotation = raw_delta_rotation * (
                        guard_max / raw_angle
                    )
            elif (
                not relevant_in_grasp
                and pose.rotation_ref is not None
                and action_config.rotation_delta_guard_enabled
                and raw_angle > action_config.rotation_delta_guard_max_rad
            ):
                guard_applied = True
                guard_action = "clamp"
                guard_max = action_config.rotation_delta_guard_max_rad
                applied_delta_rotation = raw_delta_rotation * (guard_max / raw_angle)
            applied_angle = float(np.linalg.norm(applied_delta_rotation))
            guard_enabled = bool(
                action_config.grasp_rotation_guard_enabled
                if relevant_in_grasp
                else action_config.rotation_delta_guard_enabled
            )
            if (
                pose.rotation_ref_matrix is not None
                and planned_rotation_matrix is not None
            ):
                if guard_applied:
                    planned_rotation_matrix = (
                        _rotvec_to_matrix(applied_delta_rotation)
                        @ planned_rotation_matrix
                    )
                else:
                    planned_rotation_matrix = pose.rotation_ref_matrix
            elif pose.rotation_ref_matrix is not None:
                planned_rotation_matrix = pose.rotation_ref_matrix
            planned_rotation_was_guarded = guard_applied

        if guard_applied:
            guarded_count += 1
            if in_grasp_window:
                grasp_guarded_count += 1

        effective_ref_rotation = (
            Rotation.from_matrix(planned_rotation_matrix)
            if planned_rotation_matrix is not None
            else None
        )
        if planned_rotation_matrix is None:
            effective_world_rotation = None
        elif not guard_applied:
            effective_ref_rotation = pose.rotation_ref
            effective_world_rotation = pose.rotation_world
        elif action_config.reference_frame == "base":
            assert R_wb is not None
            effective_world_rotation = Rotation.from_matrix(
                R_wb @ planned_rotation_matrix
            )
        else:
            effective_world_rotation = effective_ref_rotation

        max_abs_position = float(np.max(np.abs(delta_position)))
        position_is_motion = bool(
            max_abs_position > action_config.zero_motion_epsilon_m
        )
        applied_delta_position = (
            delta_position if position_is_motion else np.zeros(3, dtype=np.float64)
        )
        delta_6d = [
            *delta_position.tolist(),
            *applied_delta_rotation.tolist(),
        ]
        step_delta_6d = [
            *applied_delta_position.tolist(),
            *applied_delta_rotation.tolist(),
        ]
        initial = checkpoint_index == 0
        if initial:
            num_motion_steps = 1 if action_config.emit_initial_noop_step else 0
            is_noop = True
            motion_steps_clamped = False
        else:
            translation_steps = (
                int(math.ceil(max_abs_position / segment_translation_budget))
                if position_is_motion
                else 0
            )
            rotation_steps = (
                int(math.ceil(applied_angle / rotation_budget))
                if pose.rotation_ref is not None and rotation_budget > 0.0
                else 0
            )
            requested_steps = max(1, translation_steps, rotation_steps)
            num_motion_steps = min(
                requested_steps,
                max(1, action_config.max_motion_steps_per_segment),
            )
            motion_steps_clamped = num_motion_steps < requested_steps
            is_noop = bool(not position_is_motion and applied_angle <= 1e-12)

        event = grip["event"]
        serial_event = (
            event in {"close", "open"}
            and action_config.gripper_actuation_mode == "serial"
            and previous_grip is not None
        )
        if serial_event:
            motion_gripper_cmd = previous_grip["cmd"]
            motion_gripper_source = "previous_checkpoint_before_gripper_event"
        else:
            motion_gripper_cmd = grip["cmd"]
            motion_gripper_source = "current_checkpoint"

        motion_start = len(steps)
        boundary_step_index: Optional[int] = None
        for local_step in range(num_motion_steps):
            step_index = len(steps)
            if is_noop:
                action = [0.0] * 6
                kind = "noop"
            else:
                action = [value / num_motion_steps for value in step_delta_6d]
                kind = "motion"
            source = "current_checkpoint_initial" if initial else motion_gripper_source
            step: Dict[str, Any] = {
                "step_index": step_index,
                "kind": kind,
                "action_ref_6d": action,
                "gripper_cmd": motion_gripper_cmd,
                "gripper_cmd_source": source,
                "target_checkpoint_index": checkpoint_index,
                "target_frame": pose.frame,
                "source_frame_prev": (
                    previous_pose.frame if previous_pose is not None else None
                ),
                "source_frame_curr": pose.frame,
                "is_checkpoint_boundary": (local_step == num_motion_steps - 1),
            }
            if not initial:
                step["orientation_guard_applied"] = guard_applied
            steps.append(step)
            if step["is_checkpoint_boundary"]:
                boundary_step_index = step_index
        motion_end = len(steps)

        settle_start = len(steps)
        settle_count = 0
        if event in {"close", "open"} and action_config.embed_gripper_settle_steps:
            settle_count = (
                action_config.settle_steps_after_close
                if event == "close"
                else action_config.settle_steps_after_open
            )
        for _ in range(settle_count):
            steps.append(
                {
                    "step_index": len(steps),
                    "kind": "settle",
                    "action_ref_6d": [0.0] * 6,
                    "gripper_cmd": grip["cmd"],
                    "gripper_cmd_source": ("current_checkpoint_settle"),
                    "target_checkpoint_index": checkpoint_index,
                    "target_frame": pose.frame,
                    "source_frame_prev": (
                        previous_pose.frame if previous_pose is not None else None
                    ),
                    "source_frame_curr": pose.frame,
                    "is_checkpoint_boundary": False,
                }
            )
        settle_end = len(steps)

        gate_start = len(steps)
        close_gate_start = gate_start
        close_required = bool(
            event == "close" and action_config.insert_close_completion_gate
        )
        if close_required:
            steps.append(
                {
                    "step_index": len(steps),
                    "kind": "close_only_gate",
                    "action_ref_6d": [0.0] * 6,
                    "gripper_cmd": grip["cmd"],
                    "gripper_cmd_source": ("current_checkpoint_close_gate"),
                    "target_checkpoint_index": checkpoint_index,
                    "target_frame": pose.frame,
                    "source_frame_prev": (
                        previous_pose.frame if previous_pose is not None else None
                    ),
                    "source_frame_curr": pose.frame,
                    "is_checkpoint_boundary": False,
                }
            )
        close_gate_end = close_gate_start + int(close_required)

        open_gate_start = gate_start
        open_required = bool(
            event == "open" and action_config.insert_open_completion_gate
        )
        if open_required:
            steps.append(
                {
                    "step_index": len(steps),
                    "kind": "open_only_gate",
                    "action_ref_6d": [0.0] * 6,
                    "gripper_cmd": grip["cmd"],
                    "gripper_cmd_source": ("current_checkpoint_open_gate"),
                    "target_checkpoint_index": checkpoint_index,
                    "target_frame": pose.frame,
                    "source_frame_prev": (
                        previous_pose.frame if previous_pose is not None else None
                    ),
                    "source_frame_curr": pose.frame,
                    "is_checkpoint_boundary": False,
                }
            )
        open_gate_end = open_gate_start + int(open_required)

        object_rows = _object_records(
            obj_traj,
            object_id=grip["object_id"],
            obj_key=str(grip["obj_key"]),
        )
        obj_world, obj_ref, obj_valid, obj_vis = _object_point(
            object_rows.get(pose.frame),
            reference_frame=action_config.reference_frame,
            R_wb=R_wb,
            t_wb=t_wb,
        )
        has_orientation = pose.rotation_ref is not None
        checkpoint = {
            "checkpoint_index": checkpoint_index,
            "frame": pose.frame,
            "stage_id": grip["stage_id"],
            "object_id": grip["object_id"],
            "obj_key": str(grip["obj_key"]),
            "eef_target_world_6d": _absolute_pose_6d(
                pose.pos_world, effective_world_rotation
            ),
            "eef_target_ref_6d": _absolute_pose_6d(
                pose.pos_ref, effective_ref_rotation
            ),
            "eef_has_orientation": has_orientation,
            "obj_world_3d": obj_world,
            "obj_ref_3d": obj_ref,
            "obj_vis": obj_vis,
            "obj_valid": obj_valid,
            "gripper_state": grip["state"],
            "gripper_event": event,
            "gripper_cmd": grip["cmd"],
            "gripper_valid": grip["valid"],
            "motion_gripper_cmd": motion_gripper_cmd,
            "motion_gripper_cmd_source": motion_gripper_source,
            "segment_from_prev": {
                "from_frame": (
                    previous_pose.frame if previous_pose is not None else None
                ),
                "delta_ref_6d": delta_6d,
                "raw_delta_rot_ref": raw_delta_rotation.tolist(),
                "orientation_guard": {
                    "enabled": guard_enabled,
                    "in_grasp_window": bool(in_grasp_window and has_orientation),
                    "raw_angle_rad": raw_angle,
                    "applied": guard_applied,
                    "action": guard_action,
                    "max_angle_rad": guard_max,
                    "applied_angle_rad": applied_angle,
                },
                "in_grasp_window": in_grasp_window,
                "translation_step_budget_m": (segment_translation_budget),
                "num_motion_steps": num_motion_steps,
                "motion_steps_clamped": motion_steps_clamped,
                "num_settle_steps": settle_count,
            },
            "motion_step_range": {
                "start": motion_start,
                "end": motion_end,
            },
            "settle_step_range": {
                "start": settle_start,
                "end": settle_end,
            },
            "close_gate_required": close_required,
            "close_gate_step_range": {
                "start": close_gate_start,
                "end": close_gate_end,
            },
            "open_gate_required": open_required,
            "open_gate_step_range": {
                "start": open_gate_start,
                "end": open_gate_end,
            },
            "boundary_step_index": boundary_step_index,
        }
        checkpoints.append(checkpoint)

        if close_required:
            close_gate_events.append(
                {
                    "checkpoint_index": checkpoint_index,
                    "frame": pose.frame,
                    "stage_id": grip["stage_id"],
                    "object_id": grip["object_id"],
                    "step_range": {
                        "start": close_gate_start,
                        "end": close_gate_end,
                    },
                }
            )
        if open_required:
            open_gate_events.append(
                {
                    "checkpoint_index": checkpoint_index,
                    "frame": pose.frame,
                    "stage_id": grip["stage_id"],
                    "object_id": grip["object_id"],
                    "step_range": {
                        "start": open_gate_start,
                        "end": open_gate_end,
                    },
                }
            )

    close_gate = {
        "enabled": bool(action_config.insert_close_completion_gate),
        "inserted": bool(close_gate_events),
        "events": close_gate_events,
    }
    open_gate = {
        "enabled": bool(action_config.insert_open_completion_gate),
        "inserted": bool(open_gate_events),
        "events": open_gate_events,
    }
    has_orientation = any(
        checkpoint["eef_has_orientation"] for checkpoint in checkpoints
    )
    summary = {
        "num_checkpoints": len(checkpoints),
        "num_motion_steps": sum(step["kind"] == "motion" for step in steps),
        "num_noop_steps": sum(step["kind"] == "noop" for step in steps),
        "num_settle_steps": sum(step["kind"] == "settle" for step in steps),
        "num_close_gate_steps": sum(
            step["kind"] == "close_only_gate" for step in steps
        ),
        "num_open_gate_steps": sum(step["kind"] == "open_only_gate" for step in steps),
        "num_total_steps": len(steps),
        "num_rotation_guarded": guarded_count,
        "num_grasp_rotation_guarded": grasp_guarded_count,
        "has_orientation": has_orientation,
    }
    return {
        "meta": {
            "format": "action",
            "uid": uid,
            "source": {
                "ee_traj_path": ee_traj_path or "",
                "obj_traj_path": obj_traj_path,
                "gripper_path": gripper_path,
                "ee_key": action_config.eef_key,
                "obj_key": action_config.obj_key,
            },
            "action_space": {
                "type": "delta_6dof_gripper",
                "reference_frame": action_config.reference_frame,
                "position_unit": "meter",
                "rotation_unit": "radian",
                "orientation_mode": "rotvec",
                "has_orientation": has_orientation,
            },
            "planner": _planner_metadata(
                config=action_config,
                translation_budget=translation_budget,
                rotation_budget=rotation_budget,
                constraint=constraint,
                compression=compression,
                z_suppression=z_suppression,
                close_gate=close_gate,
                open_gate=open_gate,
            ),
            "summary": summary,
        },
        "checkpoints": checkpoints,
        "steps": steps,
    }


class ActionBuilder:
    """Reusable facade that binds a normalized action configuration."""

    def __init__(self, config: ConfigInput = None) -> None:
        self.config = _normalize_config(config)

    def build(
        self,
        *,
        uid: str,
        cfg: Mapping[str, Any],
        ee_traj: Mapping[str, Any],
        obj_traj: Optional[Mapping[str, Any]] = None,
        gripper_payload: Optional[Mapping[str, Any]] = None,
        ee_traj_path: str = "",
        obj_traj_path: Optional[str] = None,
        gripper_path: Optional[str] = None,
        controller_step_budgets: Optional[Sequence[float]] = None,
        step_budget_resolver: Optional[StepBudgetResolver] = None,
    ) -> Dict[str, Any]:
        return build_action(
            uid=uid,
            cfg=cfg,
            ee_traj=ee_traj,
            obj_traj=obj_traj,
            gripper_payload=gripper_payload,
            config=self.config,
            ee_traj_path=ee_traj_path,
            obj_traj_path=obj_traj_path,
            gripper_path=gripper_path,
            controller_step_budgets=controller_step_budgets,
            step_budget_resolver=step_budget_resolver,
        )


__all__ = ["ActionBuilder", "StepBudgetResolver", "build_action"]
