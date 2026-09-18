"""Pure task-success policy and current-compatible observation records.

The simulator-facing adapter deliberately lives in :mod:`dream_exe.sim`.
This module only turns an explicit success signal and caller-supplied evidence
into the canonical ``task_check_success`` observation semantics.
"""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

CURRENT_TASK_SUCCESS_SCHEMA = "task_check_success"
CURRENT_ADJUSTMENT_POLICY = "adjusted_articulation_thresholds"
SINGLE_UID_TASK_SUCCESS_VIEWS_SCHEMA = "dream_exe.single-uid-task-success-views"
CURRENT_TASK_SUCCESS_PROTOCOL = "dream_exe.adjusted_task_success"
PAPER_FINAL_TASK_SUCCESS_PROTOCOL = CURRENT_TASK_SUCCESS_PROTOCOL


@dataclass(frozen=True)
class TaskSuccessObservation:
    """One detached success observation.

    ``error`` is reserved for failures which the selected rule cannot absorb.
    The current generic ``env._check_success`` rule converts checker failures
    into a false result with ``fallback_reason`` evidence, so its normal error
    field is ``None``.
    """

    task_check_success: bool | None
    strict_task_check_success: bool | None
    calibrated_task_check_success: bool | None
    meta: dict[str, Any]
    error: str | None = None


@dataclass(frozen=True)
class TaskSuccessCheckResult:
    """Detached result from a simulator-specific success checker."""

    raw_success: bool | None
    error: str | None = None


def summarize_object_trajectories(
    payload: Mapping[str, Any],
    *,
    source: str,
    max_objects: int = 8,
) -> dict[str, Any]:
    """Build the current task-success object-trajectory summary.

    The caller owns JSON/path resolution.  This pure boundary consumes only
    the already-loaded ``obj_trajs.json`` mapping and records the caller's
    explicit source label.
    """

    objects = dict(payload.get("objects", {}) or {})
    output: list[dict[str, Any]] = []
    for object_id, object_info in objects.items():
        if len(output) >= int(max_objects):
            break
        info = dict(object_info or {})
        manipulated = dict(info.get("manipulated_object", {}) or {})
        name = str(manipulated.get("name", "") or "").strip()
        runtime_key = str(info.get("runtime_object_key", "") or "").strip()
        trajectory = list(info.get("obj_visual_center", []) or [])
        if not trajectory:
            output.append(
                {
                    "object_id": str(object_id),
                    "name": name,
                    "runtime_object_key": runtime_key,
                    "error": "missing_obj_visual_center",
                }
            )
            continue

        onset = int(info.get("motion_onset_frame", 0) or 0)
        settle = int(info.get("motion_settle_frame", 0) or 0)
        sorted_frames = sorted(
            trajectory,
            key=lambda item: int((item or {}).get("frame", 0)),
        )
        maximum_frame = int(
            sorted_frames[-1].get(
                "frame",
                len(sorted_frames) - 1,
            )
        )

        def mean_in_window(
            lower: int,
            upper: int,
        ) -> tuple[np.ndarray | None, int, int]:
            strict_points: list[np.ndarray] = []
            any_points: list[np.ndarray] = []
            for row in sorted_frames:
                try:
                    frame = int(row.get("frame", -1))
                except Exception:
                    continue
                if frame < lower or frame > upper:
                    continue
                try:
                    visibility = float(row.get("vis", 0.0))
                except Exception:
                    visibility = 0.0
                position = row.get("pos_world", None)
                if not (isinstance(position, list) and len(position) == 3):
                    continue
                try:
                    point = np.asarray(
                        position,
                        dtype=np.float64,
                    ).reshape(3)
                except Exception:
                    continue
                if visibility > 0.5:
                    strict_points.append(point)
                any_points.append(point)
            selected = strict_points if strict_points else any_points
            if not selected:
                return None, 0, 0
            return (
                np.mean(
                    np.stack(selected, axis=0),
                    axis=0,
                ),
                int(len(selected)),
                int(len(strict_points)),
            )

        pre_upper = min(
            maximum_frame,
            max(0, onset - 1),
        )
        pre_lower = max(0, pre_upper - 10)
        post_lower = min(
            maximum_frame,
            max(settle, maximum_frame - 10),
        )
        post_upper = maximum_frame

        (
            pre_mean,
            pre_used,
            pre_used_strict,
        ) = mean_in_window(pre_lower, pre_upper)
        (
            post_mean,
            post_used,
            post_used_strict,
        ) = mean_in_window(post_lower, post_upper)
        displacement = (
            None
            if pre_mean is None or post_mean is None
            else float(np.linalg.norm(post_mean - pre_mean))
        )
        output.append(
            {
                "object_id": str(object_id),
                "name": name or None,
                "runtime_object_key": (runtime_key or None),
                "motion_onset_frame": onset,
                "motion_settle_frame": settle,
                "pre_window": [
                    int(pre_lower),
                    int(pre_upper),
                ],
                "post_window": [
                    int(post_lower),
                    int(post_upper),
                ],
                "pre_pos_world_mean": (None if pre_mean is None else pre_mean.tolist()),
                "post_pos_world_mean": (
                    None if post_mean is None else post_mean.tolist()
                ),
                "pre_used_frames": int(pre_used),
                "post_used_frames": int(post_used),
                "pre_used_frames_strict": int(pre_used_strict),
                "post_used_frames_strict": int(post_used_strict),
                "displacement_m": displacement,
                "quality": dict(info.get("quality", {}) or {}),
            }
        )
    return {
        "source": str(source),
        "num_objects": int(len(objects)),
        "objects": output,
    }


def summarize_openblenderlid_trajectory(
    payload: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any] | None:
    """Build the legacy OpenBlenderLid trajectory diagnostic evidence.

    This is deliberately a pure adapter over an already-loaded trajectory
    payload.  It does not decide task success and it never reads bench paths.
    """

    raw_objects = dict(payload.get("objects", {}) or {})
    candidates: list[str] = []
    for object_id, raw_info in raw_objects.items():
        info = dict(raw_info or {})
        manipulated = dict(info.get("manipulated_object", {}) or {})
        name = str(manipulated.get("name", "") or "").strip()
        runtime_key = str(info.get("runtime_object_key", "") or "").strip()
        if (
            name == "blender_lid"
            or runtime_key.endswith("blender_lid")
            or "blender_lid" in runtime_key
        ):
            candidates.append(str(object_id))

    if not raw_objects:
        return {
            "source": str(source),
            "error": "missing_objects",
        }
    if not candidates:
        return {
            "source": str(source),
            "error": "no_blender_lid_candidate",
            "available_runtime_keys": [
                str(
                    dict(info or {}).get(
                        "runtime_object_key",
                        "",
                    )
                    or ""
                ).strip()
                for info in list(raw_objects.values())[:50]
            ],
        }

    selected_id = candidates[0]
    selected_raw = dict(raw_objects[selected_id] or {})
    if not list(selected_raw.get("obj_visual_center", []) or []):
        return {
            "source": str(source),
            "object_id": selected_id,
            "error": "missing_obj_visual_center",
        }

    summarized = summarize_object_trajectories(
        payload,
        source=str(source),
        max_objects=max(1, len(raw_objects)),
    )
    selected = next(
        (
            dict(item)
            for item in summarized["objects"]
            if str(item.get("object_id", "")) == selected_id
        ),
        None,
    )
    if selected is None:
        return {
            "source": str(source),
            "object_id": selected_id,
            "error": "missing_obj_visual_center",
        }

    pre_mean = selected.get("pre_pos_world_mean")
    post_mean = selected.get("post_pos_world_mean")
    if int(selected.get("pre_used_frames_strict", 0) or 0) == 0:
        pre_mean = None
    if int(selected.get("post_used_frames_strict", 0) or 0) == 0:
        post_mean = None
    displacement = None
    if pre_mean is not None and post_mean is not None:
        displacement = float(
            np.linalg.norm(
                np.asarray(post_mean, dtype=np.float64)
                - np.asarray(pre_mean, dtype=np.float64)
            )
        )
    threshold_m = 0.05
    return {
        "source": str(source),
        "object_id": selected_id,
        "runtime_object_key": str(selected_raw.get("runtime_object_key", "") or ""),
        "motion_onset_frame": int(selected.get("motion_onset_frame", 0) or 0),
        "motion_settle_frame": int(selected.get("motion_settle_frame", 0) or 0),
        "pre_window": list(selected.get("pre_window", [])),
        "post_window": list(selected.get("post_window", [])),
        "pre_pos_world_mean": pre_mean,
        "post_pos_world_mean": post_mean,
        "displacement_m": displacement,
        "opened_threshold_m": threshold_m,
        "opened_by_displacement": (
            None if displacement is None else bool(displacement >= threshold_m)
        ),
        "quality": dict(selected_raw.get("quality", {}) or {}),
        "candidates_found": candidates,
    }


def _as_float(value: Any) -> float | None:
    try:
        result = float(value)
    except Exception:
        return None
    if result != result:
        return None
    return result


def _metric_value(
    meta: Mapping[str, Any],
    group: str,
    key: str,
) -> float | None:
    metrics = meta.get("metrics", {})
    metrics = metrics if isinstance(metrics, Mapping) else {}
    signals = metrics.get("signals", {})
    signals = signals if isinstance(signals, Mapping) else {}
    values = signals.get(group, {})
    values = values if isinstance(values, Mapping) else {}
    return _as_float(values.get(key, None))


def _point_in_recorded_oriented_xy_bounds(
    point_xy: Any,
    footprint_details: Any,
    *,
    extra_margin_m: float = 0.0,
) -> tuple[bool, dict[str, Any]]:
    details: dict[str, Any] = {"extra_margin_m": float(extra_margin_m)}
    if not isinstance(footprint_details, Mapping):
        details["error"] = "missing_footprint_details"
        return False, details
    try:
        point = np.asarray(
            point_xy,
            dtype=np.float64,
        ).reshape(2)
        center = np.asarray(
            footprint_details.get("center_xy"),
            dtype=np.float64,
        ).reshape(2)
        basis = np.asarray(
            footprint_details.get("basis_xy"),
            dtype=np.float64,
        ).reshape(2, 2)
        lower = np.asarray(
            footprint_details.get("bounds_lo"),
            dtype=np.float64,
        ).reshape(2) - float(extra_margin_m)
        upper = np.asarray(
            footprint_details.get("bounds_hi"),
            dtype=np.float64,
        ).reshape(2) + float(extra_margin_m)
    except Exception as error:
        details["error"] = f"invalid_footprint_details:{type(error).__name__}"
        return False, details
    if not (
        np.all(np.isfinite(point))
        and np.all(np.isfinite(center))
        and np.all(np.isfinite(basis))
    ):
        details["error"] = "nonfinite_point_or_footprint"
        return False, details
    point_projection = (point - center) @ basis
    signed_margin = np.minimum(
        point_projection - lower,
        upper - point_projection,
    )
    inside = bool(np.all(signed_margin >= 0.0))
    details.update(
        {
            "point_xy": point.tolist(),
            "point_proj": point_projection.tolist(),
            "bounds_lo": lower.tolist(),
            "bounds_hi": upper.tolist(),
            "signed_margin_m": signed_margin.tolist(),
            "inside": bool(inside),
        }
    )
    return inside, details


def _adjust_task_success(
    task_name: str,
    *,
    strict_ok: bool,
    raw_success_ok: bool,
    meta: Mapping[str, Any] | None = None,
) -> tuple[bool, dict[str, Any]]:
    """Apply the adjusted public task-success policy.

    Geometry and simulator inspection are intentionally not performed here.
    Task-specific component extraction is an injected simulator concern; this
    function consumes only detached component and trajectory evidence.
    """

    name = str(task_name or "").strip()
    evidence = dict(meta or {})
    details: dict[str, Any] = {"policy": CURRENT_ADJUSTMENT_POLICY}

    if bool(raw_success_ok):
        details["reason"] = "raw_success"
        return True, details

    components = evidence.get("task_success_components", None)
    if not isinstance(components, Mapping):
        components = {}

    if name == "CheesyBread":
        bread_in = components.get("bread_in_receptacle")
        if bread_in is None:
            bread_in = components.get("bread_in_receptacle_contact_and_dist")
        footprint_ok = components.get("cheese_center_in_bread_oriented_footprint")
        z_ok = components.get("cheese_bread_oriented_footprint_z_ok")
        trajectory_footprint_ok = None
        trajectory_footprint_details = None
        trajectory_z_ok = None
        trajectory_point = None
        if not bool(footprint_ok):
            footprint = components.get(
                "cheese_bread_oriented_footprint",
                None,
            )
            footprint_details = (
                footprint.get("oriented_footprint", None)
                if isinstance(footprint, Mapping)
                else None
            )
            trajectory_objects = evidence.get(
                "trajectory_objects",
                {},
            )
            trajectory_objects = (
                trajectory_objects if isinstance(trajectory_objects, Mapping) else {}
            )
            objects = trajectory_objects.get("objects", None)
            if isinstance(objects, list):
                for item in objects:
                    if not isinstance(item, Mapping):
                        continue
                    object_name = str(item.get("name", "") or "").lower()
                    runtime_key = str(item.get("runtime_object_key", "") or "").lower()
                    if object_name != "cheese" and "cheese" not in runtime_key:
                        continue
                    position = item.get(
                        "post_pos_world_mean",
                        None,
                    )
                    if not (isinstance(position, list) and len(position) >= 3):
                        continue
                    trajectory_point = position
                    (
                        trajectory_footprint_ok,
                        trajectory_footprint_details,
                    ) = _point_in_recorded_oriented_xy_bounds(
                        position[:2],
                        footprint_details,
                        extra_margin_m=0.0,
                    )
                    try:
                        post_z = float(position[2])
                        footprint_mapping = (
                            footprint if isinstance(footprint, Mapping) else {}
                        )
                        bread_z_min = _as_float(footprint_mapping.get("bread_z_min_m"))
                        bread_z_max = _as_float(footprint_mapping.get("bread_z_max_m"))
                        trajectory_z_ok = bool(
                            (bread_z_min is None or post_z >= float(bread_z_min) - 0.05)
                            and (
                                bread_z_max is None
                                or post_z <= float(bread_z_max) + 0.25
                            )
                        )
                    except Exception:
                        trajectory_z_ok = None
                    break
        footprint_ok_effective = bool(footprint_ok) or bool(trajectory_footprint_ok)
        if bool(footprint_ok):
            z_ok_effective = z_ok is not False
        elif bool(trajectory_footprint_ok):
            z_ok_effective = trajectory_z_ok is not False
        else:
            z_ok_effective = z_ok is not False and trajectory_z_ok is not False
        success = bool(
            bool(bread_in) and bool(footprint_ok_effective) and bool(z_ok_effective)
        )
        details.update(
            {
                "reason": "cheesybread_oriented_footprint",
                "bread_in_receptacle": (None if bread_in is None else bool(bread_in)),
                "cheese_center_in_bread_oriented_footprint": (
                    None if footprint_ok is None else bool(footprint_ok)
                ),
                "cheese_bread_oriented_footprint_z_ok": (
                    None if z_ok is None else bool(z_ok)
                ),
                "cheese_traj_center_in_bread_oriented_footprint": (
                    None
                    if trajectory_footprint_ok is None
                    else bool(trajectory_footprint_ok)
                ),
                "cheese_traj_center_z_ok": (
                    None if trajectory_z_ok is None else bool(trajectory_z_ok)
                ),
                "cheese_traj_center_world": trajectory_point,
                "cheese_traj_oriented_footprint": (trajectory_footprint_details),
                "requires_release_far": False,
            }
        )
        return success, details

    if name == "OpenBlenderLid":
        trajectory = evidence.get("trajectory_fallback", None)
        if not isinstance(trajectory, Mapping):
            trajectory = {}
        quality = trajectory.get("quality", None)
        if not isinstance(quality, Mapping):
            quality = {}
        distance_to_closed = _as_float(components.get("dist_to_closed_m"))
        maximum_displacement = _as_float(quality.get("max_displacement_m"))
        lid_off = bool(components.get("lid_off_blender"))
        released = bool(components.get("released_ok"))
        post_missing = (
            trajectory.get(
                "post_pos_world_mean",
                "__missing__",
            )
            is None
        )
        motion_detected = bool(quality.get("motion_detected"))
        distance_threshold = 0.25
        displacement_threshold = 0.10
        success = bool(
            lid_off
            and released
            and post_missing
            and motion_detected
            and distance_to_closed is not None
            and distance_to_closed >= distance_threshold
            and maximum_displacement is not None
            and maximum_displacement >= displacement_threshold
        )
        details.update(
            {
                "reason": ("open_blender_lid_lost_post_after_lid_off"),
                "lid_off_blender": lid_off,
                "released_ok": released,
                "post_pos_world_missing": post_missing,
                "motion_detected": motion_detected,
                "dist_to_closed_m": distance_to_closed,
                "dist_to_closed_threshold_m": (distance_threshold),
                "max_displacement_m": maximum_displacement,
                "max_displacement_threshold_m": (displacement_threshold),
            }
        )
        return success, details

    if name in {
        "CloseDrawer",
        "CloseMicrowave",
        "CloseOven",
        "CloseFridgeDrawer",
    }:
        closed_fraction = _metric_value(
            evidence,
            "fraction",
            "closed_fraction_min",
        )
        if closed_fraction is None:
            closed_fraction = _as_float(components.get("closed_fraction_min"))
        threshold = 0.55
        success = bool(closed_fraction is not None and closed_fraction >= threshold)
        details.update(
            {
                "reason": "close_relaxed_threshold",
                "value": closed_fraction,
                "threshold": threshold,
            }
        )
        return success, details

    if name in {"SlideOvenRack", "SlideToasterOvenRack"}:
        progress = _metric_value(
            evidence,
            "fraction",
            "progress_fraction",
        )
        if progress is None:
            progress = _as_float(components.get("progress_fraction"))
        threshold = 0.35 if name == "SlideOvenRack" else 0.60
        success = bool(progress is not None and progress >= threshold)
        details.update(
            {
                "reason": (
                    "slide_oven_rack_relaxed_threshold"
                    if name == "SlideOvenRack"
                    else ("slide_toaster_oven_rack_relaxed_threshold")
                ),
                "value": progress,
                "threshold": threshold,
            }
        )
        return success, details

    if name == "TurnSinkSpout":
        turn_fraction = _metric_value(
            evidence,
            "fraction",
            "turn_fraction_to_target",
        )
        if turn_fraction is None:
            turn_fraction = _as_float(components.get("turn_fraction_to_target"))
        distance_to_target = _metric_value(
            evidence,
            "angle",
            "distance_to_target_rad",
        )
        if distance_to_target is None:
            distance_to_target = _as_float(components.get("distance_to_target_rad"))
        turn_threshold = 0.80
        distance_threshold = 0.17
        success = bool(
            (turn_fraction is not None and turn_fraction >= turn_threshold)
            or (
                distance_to_target is not None
                and distance_to_target <= distance_threshold
            )
        )
        details.update(
            {
                "reason": "turn_sink_spout_relaxed_threshold",
                "turn_fraction": turn_fraction,
                "turn_fraction_threshold": turn_threshold,
                "distance_to_target_rad": distance_to_target,
                "distance_to_target_threshold_rad": (distance_threshold),
            }
        )
        return success, details

    details["reason"] = "strict_false_no_relaxed_threshold"
    return bool(strict_ok), details


def cheesybread_metrics_from_components(
    components: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the current detached CheesyBread progress metrics."""

    values = dict(components)
    subgoals: dict[str, Any] = {}
    quality: dict[str, Any] = {
        "release_ok": None,
        "place_quality": None,
        "articulation_quality": None,
    }
    signals: dict[str, dict[str, Any]] = {
        "dist": {},
        "angle": {},
        "fraction": {},
    }

    def boolean(key: str) -> bool | None:
        value = values.get(key, None)
        return None if value is None else bool(value)

    def number(key: str) -> float | None:
        return _as_float(values.get(key, None))

    def clamp01(value: Any) -> float | None:
        number_value = _as_float(value)
        if number_value is None:
            return None
        return float(np.clip(number_value, 0.0, 1.0))

    def safe_divide(
        numerator: Any,
        denominator: Any,
    ) -> float | None:
        numerator_value = _as_float(numerator)
        denominator_value = _as_float(denominator)
        if (
            numerator_value is None
            or denominator_value is None
            or denominator_value == 0
        ):
            return None
        return float(numerator_value / denominator_value)

    def exponential_close_ratio(
        ratio: Any,
    ) -> float | None:
        ratio_value = _as_float(ratio)
        if ratio_value is None:
            return None
        if ratio_value < 0:
            ratio_value = 0.0
        return float(np.exp(-float(np.log(2.0)) * ratio_value))

    bread_in = (
        boolean("bread_in_receptacle")
        if "bread_in_receptacle" in values
        else boolean("bread_in_receptacle_contact_and_dist")
    )
    cheese_contact = boolean("cheese_bread_contact")
    cheese_footprint = boolean("cheese_center_in_bread_oriented_footprint")
    cheese_z_ok = boolean("cheese_bread_oriented_footprint_z_ok")
    cheese_on_bread = (
        bool(cheese_contact or cheese_footprint)
        if (cheese_contact is not None or cheese_footprint is not None)
        else None
    )

    cheese_xy = number("cheese_to_bread_xy_m")
    cheese_threshold = number("cheese_alignment_threshold_xy_m")
    if cheese_xy is not None:
        signals["dist"]["cheese_to_bread_xy_m"] = float(cheese_xy)
    cheese_z = number("cheese_to_bread_z_m")
    if cheese_z is not None:
        signals["dist"]["cheese_to_bread_z_m"] = float(cheese_z)
    cheese_ratio = (
        safe_divide(cheese_xy, cheese_threshold)
        if (cheese_xy is not None and cheese_threshold is not None)
        else None
    )
    if cheese_ratio is not None:
        signals["fraction"]["cheese_to_bread_xy_ratio"] = float(cheese_ratio)

    bread_xy = number("bread_vs_bread_container_dist_xy_m")
    bread_threshold = number("bread_in_receptacle_threshold_xy_m")
    bread_ratio = (
        safe_divide(bread_xy, bread_threshold)
        if (bread_xy is not None and bread_threshold is not None)
        else None
    )
    if bread_ratio is not None:
        signals["fraction"]["bread_in_receptacle_xy_ratio"] = float(bread_ratio)

    bread_place = 1.0 if bread_in else 0.0 if bread_in is not None else None
    if bread_place is None and bread_ratio is not None:
        bread_place = exponential_close_ratio(bread_ratio)
    cheese_alignment = (
        1.0 if cheese_on_bread else 0.0 if cheese_on_bread is not None else None
    )
    if (
        cheese_alignment is None or cheese_alignment == 0.0
    ) and cheese_ratio is not None:
        cheese_alignment = exponential_close_ratio(cheese_ratio)

    gripper_distance = number("gripper_to_cheese_dist_m")
    if gripper_distance is not None:
        signals["dist"]["eef_to_cheese_m"] = float(gripper_distance)
    release_quality = None
    if gripper_distance is not None:
        excess = float(max(float(gripper_distance) - 0.25, 0.0))
        release_quality = float(1.0 - np.exp(-excess / 0.10))
    quality["release_ok"] = (
        clamp01(release_quality) if release_quality is not None else None
    )
    if bread_place is not None and cheese_alignment is not None:
        quality["place_quality"] = clamp01(
            0.5 * float(bread_place) + 0.5 * float(cheese_alignment)
        )

    subgoals["bread_in_receptacle"] = bread_in
    subgoals["cheese_on_bread"] = cheese_on_bread
    subgoals["cheese_bread_contact"] = (
        bool(cheese_contact) if cheese_contact is not None else None
    )
    subgoals["cheese_center_in_bread_oriented_footprint"] = (
        bool(cheese_footprint) if cheese_footprint is not None else None
    )
    subgoals["cheese_bread_oriented_footprint_z_ok"] = (
        bool(cheese_z_ok) if cheese_z_ok is not None else None
    )
    core = None
    if bread_in is not None and cheese_on_bread is not None:
        core = 0.5 * float(bread_in) + 0.5 * float(cheese_on_bread)
    signals["fraction"]["core_subgoal_fraction"] = core

    primary_progress = None
    if (
        bread_place is not None
        and cheese_alignment is not None
        and quality["release_ok"] is not None
    ):
        primary_progress = clamp01(
            0.35 * float(bread_place)
            + 0.45 * float(cheese_alignment)
            + 0.20 * float(quality["release_ok"])
        )
    return {
        "primary_progress": primary_progress,
        "quality": quality,
        "subgoals": subgoals,
        "signals": signals,
    }


def task_success_metrics_from_components(
    task_name: str,
    components: Mapping[str, Any],
) -> dict[str, Any]:
    """Build the current Table-4 component metrics for one task.

    This function is deliberately simulator-free.  RoboCasa adapters own
    component extraction; this boundary only preserves the current conversion
    from detached components to ``SR-P``, ``Rel``, ``Place``, ``Art``, and
    ``Core`` source fields.
    """

    name = str(task_name or "").strip()
    values = dict(components)
    if name == "CheesyBread":
        return cheesybread_metrics_from_components(values)

    subgoals: dict[str, Any] = {}
    quality: dict[str, Any] = {
        "release_ok": None,
        "place_quality": None,
        "articulation_quality": None,
    }
    signals: dict[str, dict[str, Any]] = {
        "dist": {},
        "angle": {},
        "fraction": {},
    }
    errors: list[str] = []

    def boolean(key: str) -> bool | None:
        value = values.get(key, None)
        return None if value is None else bool(value)

    def number(key: str) -> float | None:
        return _as_float(values.get(key, None))

    def clamp01(value: Any) -> float | None:
        numeric = _as_float(value)
        if numeric is None:
            return None
        return float(np.clip(numeric, 0.0, 1.0))

    def safe_divide(
        numerator: Any,
        denominator: Any,
    ) -> float | None:
        numerator_value = _as_float(numerator)
        denominator_value = _as_float(denominator)
        if (
            numerator_value is None
            or denominator_value is None
            or denominator_value == 0
        ):
            return None
        return float(numerator_value / denominator_value)

    def exponential_close_ratio(value: Any) -> float | None:
        ratio = _as_float(value)
        if ratio is None:
            return None
        return float(np.exp(-float(np.log(2.0)) * max(float(ratio), 0.0)))

    primary_progress: float | None = None

    try:
        if name in {
            "CloseDrawer",
            "CloseMicrowave",
            "CloseOven",
        }:
            open_fraction = number("open_fraction_max")
            closed_fraction = number("closed_fraction_min")
            closed_percent = number("closed_pct_min")
            if open_fraction is not None:
                signals["fraction"]["open_fraction_max"] = float(open_fraction)
                signals["fraction"]["closed_fraction_min"] = float(1.0 - open_fraction)
            if closed_fraction is not None:
                signals["fraction"]["closed_fraction_min"] = float(closed_fraction)
            if closed_percent is not None:
                signals["fraction"]["closed_pct_min"] = float(closed_percent) / 100.0
            primary_progress = clamp01(signals["fraction"].get("closed_fraction_min"))
            quality["articulation_quality"] = primary_progress
            subgoals["is_closed"] = (
                boolean("is_closed") if "is_closed" in values else boolean("all_and")
            )
        elif name == "CloseFridgeDrawer":
            open_fraction = number("drawer_open_fraction")
            if open_fraction is not None:
                signals["fraction"]["open_fraction_max"] = float(open_fraction)
                signals["fraction"]["closed_fraction_min"] = float(1.0 - open_fraction)
            primary_progress = clamp01(signals["fraction"].get("closed_fraction_min"))
            is_closed = (
                boolean("is_closed") if "is_closed" in values else boolean("all_and")
            )
            if primary_progress is None and is_closed is not None:
                primary_progress = clamp01(1.0 if is_closed else 0.0)
            quality["articulation_quality"] = primary_progress
            subgoals["is_closed"] = is_closed
        elif name in {
            "SlideOvenRack",
            "SlideToasterOvenRack",
        }:
            progress = number("progress_fraction")
            if progress is not None:
                signals["fraction"]["progress_fraction"] = float(progress)
                signals["fraction"]["remaining"] = float(1.0 - progress)
            current_position = number("current_pos")
            if current_position is not None:
                signals["fraction"]["current_pos_raw"] = float(current_position)
            primary_progress = clamp01(signals["fraction"].get("progress_fraction"))
            quality["articulation_quality"] = primary_progress
        elif name in {
            "OpenStandMixerHead",
            "CloseStandMixerHead",
        }:
            head = number("head")
            if head is not None:
                head = float(np.clip(head, 0.0, 1.0))
                if name == "OpenStandMixerHead":
                    signals["fraction"]["head_open_fraction"] = head
                    signals["fraction"]["remaining_to_open"] = float(1.0 - head)
                    primary_progress = head
                else:
                    signals["fraction"]["head_closed_fraction"] = float(1.0 - head)
                    signals["fraction"]["remaining_to_close"] = head
                    primary_progress = float(1.0 - head)
            primary_progress = clamp01(primary_progress)
            quality["articulation_quality"] = primary_progress
            subgoals["head_ok"] = boolean("all_and")
        elif name == "OpenBlenderLid":
            distance = number("dist_to_closed_m")
            threshold = number("closed_thresh_m")
            lid_on_counter = boolean("lid_on_any_counter")
            gripper_far = boolean("gripper_lid_far")
            if distance is not None:
                signals["dist"]["dist_to_closed_m"] = float(distance)
            if threshold is not None:
                signals["dist"]["closed_thresh_m"] = float(threshold)
            ratio = (
                safe_divide(distance, threshold)
                if distance is not None and threshold is not None
                else None
            )
            open_progress = None
            if ratio is not None:
                signals["fraction"]["dist_ratio_to_thresh"] = float(ratio)
                open_progress = clamp01(
                    1.0 - float(np.exp(-float(np.log(2.0)) * float(ratio)))
                )
            counter_flag = None
            if lid_on_counter is not None:
                counter_flag = float(1.0 if lid_on_counter else 0.0)
                signals["fraction"]["lid_on_counter_flag"] = counter_flag
            quality["release_ok"] = (
                clamp01(1.0 if gripper_far else 0.0)
                if gripper_far is not None
                else None
            )
            candidates = [
                value for value in (open_progress, counter_flag) if value is not None
            ]
            best_open = max(candidates) if candidates else None
            quality["articulation_quality"] = (
                clamp01(best_open) if best_open is not None else None
            )
            primary_progress = clamp01(best_open) if best_open is not None else None
            if quality["release_ok"] is not None and primary_progress is not None:
                primary_progress = clamp01(
                    float(primary_progress) * float(quality["release_ok"])
                )
            subgoals["lid_off_blender"] = boolean("lid_off_blender")
            subgoals["lid_on_counter"] = lid_on_counter
        elif name == "TurnOnToaster":
            slot_count = number("num_slot_pairs")
            slot_count_integer = int(slot_count) if slot_count is not None else None
            contacts = values.get(
                "contacts_by_slot_pair",
                None,
            )
            contact_fraction = None
            if isinstance(contacts, dict) and slot_count_integer:
                contact_fraction = float(
                    sum(bool(value) for value in contacts.values())
                ) / float(slot_count_integer)
            if contact_fraction is not None:
                signals["fraction"]["toast_slot_contact_fraction"] = float(
                    np.clip(contact_fraction, 0.0, 1.0)
                )
            turned_on = boolean("turned_on_selected_slot")
            if turned_on is not None:
                turned_on_fraction = float(1.0 if turned_on else 0.0)
                signals["fraction"]["turn_on_fraction"] = turned_on_fraction
                signals["fraction"]["toggle_progress"] = turned_on_fraction
            parts = [
                signals["fraction"].get("toast_slot_contact_fraction"),
                signals["fraction"].get("turn_on_fraction"),
            ]
            available_parts = [value for value in parts if value is not None]
            if available_parts:
                primary_progress = float(
                    sum(float(value) for value in available_parts)
                    / len(available_parts)
                )
            primary_progress = clamp01(primary_progress)
        elif name == "TurnSinkSpout":
            handle_state = values.get("handle_state", None)
            spout_joint = (
                _as_float(handle_state.get("spout_joint"))
                if isinstance(handle_state, dict)
                else None
            )
            if spout_joint is not None:
                spout_joint = float(spout_joint) % (2.0 * np.pi)
                signals["angle"]["spout_joint_rad"] = float(spout_joint)
            target = str(values.get("target_spout_ori", "") or "")
            distance_to_target = None
            if spout_joint is not None:
                if target == "left":
                    lower = float(np.pi)
                    upper = float(2.0 * np.pi - np.pi / 6.0)
                elif target == "right":
                    lower = float(np.pi / 6.0)
                    upper = float(np.pi)
                else:
                    lower = None
                    upper = None
                if lower is not None and upper is not None:
                    if lower <= spout_joint <= upper:
                        distance_to_target = 0.0
                    else:
                        distance_to_target = float(
                            min(
                                abs(spout_joint - lower),
                                abs(spout_joint - upper),
                            )
                        )
            if distance_to_target is not None:
                signals["angle"]["distance_to_target_rad"] = float(distance_to_target)
                signals["fraction"]["turn_fraction_to_target"] = clamp01(
                    float(
                        np.exp(
                            -float(np.log(2.0))
                            * float(distance_to_target)
                            / float(np.pi / 6.0)
                        )
                    )
                )
            primary_progress = clamp01(
                signals["fraction"].get("turn_fraction_to_target")
            )
    except Exception as error:
        errors.append(f"{type(error).__name__}: {error}")

    if name == "BreadAndCheese":
        bread = boolean("bread_on_cutting_board")
        cheese = boolean("cheese_on_cutting_board")
        placed = (
            0.5 * float(bread) + 0.5 * float(cheese)
            if bread is not None and cheese is not None
            else None
        )
        subgoals["bread_on_cutting_board"] = bread
        subgoals["cheese_on_cutting_board"] = cheese
        subgoals["placed_fraction_ok"] = None if placed is None else bool(placed >= 1.0)
        signals["fraction"]["placed_fraction"] = placed
        released = boolean("gripper_obj_far")
        quality["release_ok"] = (
            clamp01(1.0 if released else 0.0) if released is not None else None
        )
        quality["place_quality"] = clamp01(placed) if placed is not None else None
        if placed is not None and quality["release_ok"] is not None:
            primary_progress = clamp01(
                0.5 * float(placed) + 0.5 * float(quality["release_ok"])
            )
        signals["fraction"]["core_subgoal_fraction"] = placed
    elif name == "BreadSetupSlicing":
        bread_count = values.get("num_bread", None)
        placements = values.get(
            "bread_on_board_by_obj",
            None,
        )
        placed = None
        if (
            isinstance(bread_count, int)
            and isinstance(placements, dict)
            and bread_count > 0
        ):
            placed = float(sum(bool(value) for value in placements.values())) / float(
                bread_count
            )
        subgoals["bread_on_board_and"] = boolean("bread_on_board_and")
        signals["fraction"]["placed_fraction"] = placed
        released = boolean("gripper_obj_far_obj_0")
        quality["release_ok"] = (
            clamp01(1.0 if released else 0.0) if released is not None else None
        )
        quality["place_quality"] = clamp01(placed) if placed is not None else None
        if placed is not None and quality["release_ok"] is not None:
            primary_progress = clamp01(
                0.7 * float(placed) + 0.3 * float(quality["release_ok"])
            )
        signals["fraction"]["core_subgoal_fraction"] = placed
    elif name == "PackDessert":
        cooked_ok = boolean("cooked_food_in_container_strict")
        dessert_ok = boolean("dessert_in_container")
        subgoals["cooked_food_ok"] = cooked_ok
        subgoals["dessert_ok"] = dessert_ok
        distance = number("dessert_xy_dist_m")
        threshold = number("dessert_xy_th_m")
        if distance is not None:
            signals["dist"]["dessert_to_container_xy_m"] = float(distance)
        ratio = (
            safe_divide(distance, threshold)
            if distance is not None and threshold is not None
            else None
        )
        if ratio is not None:
            signals["fraction"]["dessert_to_container_xy_ratio"] = float(ratio)
            centering = exponential_close_ratio(ratio)
        else:
            centering = 1.0 if dessert_ok else 0.0 if dessert_ok is not None else None
        released = boolean("gripper_far_from_dessert")
        quality["release_ok"] = (
            clamp01(1.0 if released else 0.0) if released is not None else None
        )
        if centering is not None:
            quality["place_quality"] = clamp01(centering)
        cooked_value = float(cooked_ok) if cooked_ok is not None else None
        if (
            cooked_value is not None
            and centering is not None
            and quality["release_ok"] is not None
        ):
            primary_progress = clamp01(
                0.5 * cooked_value
                + 0.4 * float(centering)
                + 0.1 * float(quality["release_ok"])
            )
        signals["fraction"]["core_subgoal_fraction"] = (
            None if dessert_ok is None else float(dessert_ok)
        )
    elif name == "PickPlaceCounterToBlender":
        inside = boolean("obj_inside_blender")
        released = boolean("gripper_obj_far")
        subgoals["obj_inside_blender"] = inside
        quality["release_ok"] = (
            clamp01(1.0 if released else 0.0) if released is not None else None
        )
        quality["place_quality"] = (
            clamp01(1.0 if inside else 0.0) if inside is not None else None
        )
        signals["fraction"]["core_subgoal_fraction"] = (
            None if inside is None else float(inside)
        )
        if quality["place_quality"] is not None and quality["release_ok"] is not None:
            primary_progress = clamp01(
                0.5 * float(quality["place_quality"])
                + 0.5 * float(quality["release_ok"])
            )
    elif name == "PickPlaceCounterToOven":
        on_rack = boolean("on_rack")
        in_tray = boolean("obj_in_oven_tray")
        released = boolean("gripper_obj_far")
        subgoals["tray_on_rack"] = on_rack
        subgoals["obj_in_oven_tray"] = in_tray
        placed = (
            0.5 * float(on_rack) + 0.5 * float(in_tray)
            if on_rack is not None and in_tray is not None
            else None
        )
        signals["fraction"]["placed_subgoals_fraction"] = placed
        signals["fraction"]["core_subgoal_fraction"] = placed
        quality["place_quality"] = clamp01(placed) if placed is not None else None
        quality["release_ok"] = (
            clamp01(1.0 if released else 0.0) if released is not None else None
        )
        if placed is not None and quality["release_ok"] is not None:
            primary_progress = clamp01(
                0.7 * float(placed) + 0.3 * float(quality["release_ok"])
            )
    elif name == "CoffeeServeMug":
        contact = boolean("contact_check")
        released = boolean("gripper_obj_far")
        subgoals["mug_on_counter"] = contact
        quality["place_quality"] = (
            clamp01(1.0 if contact else 0.0) if contact is not None else None
        )
        quality["release_ok"] = (
            clamp01(1.0 if released else 0.0) if released is not None else None
        )
        signals["fraction"]["core_subgoal_fraction"] = (
            None if contact is None else float(contact)
        )
        if quality["place_quality"] is not None and quality["release_ok"] is not None:
            primary_progress = clamp01(
                0.5 * float(quality["place_quality"])
                + 0.5 * float(quality["release_ok"])
            )
    elif name == "MakeIcedCoffee":
        ice_in_cup = boolean("ice_in_cup_or")
        far_from_first = boolean("gripper_far_from_ice_cube1")
        far_from_second = boolean("gripper_far_from_ice_cube2")
        released = (
            bool(far_from_first and far_from_second)
            if (far_from_first is not None and far_from_second is not None)
            else None
        )
        subgoals["ice_in_cup"] = ice_in_cup
        quality["place_quality"] = (
            clamp01(1.0 if ice_in_cup else 0.0) if ice_in_cup is not None else None
        )
        quality["release_ok"] = (
            clamp01(1.0 if released else 0.0) if released is not None else None
        )
        signals["fraction"]["core_subgoal_fraction"] = (
            None if ice_in_cup is None else float(ice_in_cup)
        )
        if quality["place_quality"] is not None and quality["release_ok"] is not None:
            primary_progress = clamp01(
                0.7 * float(quality["place_quality"])
                + 0.3 * float(quality["release_ok"])
            )
    elif name == "PlaceVegetablesEvenly":
        in_pan = values.get("in_pan", None)
        in_pan_fraction = None
        if isinstance(in_pan, dict) and in_pan:
            in_pan_fraction = float(
                sum(bool(value) for value in in_pan.values())
            ) / float(len(in_pan))
        signals["fraction"]["in_pan_fraction"] = in_pan_fraction
        signals["fraction"]["core_subgoal_fraction"] = in_pan_fraction
        z_distance = number("z_distance")
        xy_distance = number("xy_distance")
        z_threshold = number("min_z_distance")
        xy_threshold = number("min_xy_distance")
        if z_distance is not None:
            signals["dist"]["z_distance_m"] = float(z_distance)
        if xy_distance is not None:
            signals["dist"]["xy_distance_m"] = float(xy_distance)
        z_ratio = (
            safe_divide(z_distance, z_threshold)
            if z_distance is not None and z_threshold is not None
            else None
        )
        xy_ratio = (
            safe_divide(xy_distance, xy_threshold)
            if (xy_distance is not None and xy_threshold is not None)
            else None
        )
        if z_ratio is not None:
            signals["fraction"]["z_distance_ratio"] = float(z_ratio)
        if xy_ratio is not None:
            signals["fraction"]["xy_distance_ratio"] = float(xy_ratio)
        z_overlap = exponential_close_ratio(z_ratio) if z_ratio is not None else None
        xy_spread = None
        if xy_ratio is not None:
            xy_spread = clamp01(
                1.0 - float(np.exp(-max(float(xy_ratio) - 1.0, 0.0) / 0.25))
            )
        evenness = None
        if z_overlap is not None and xy_spread is not None:
            evenness = clamp01(0.5 * float(z_overlap) + 0.5 * float(xy_spread))
        quality["place_quality"] = (
            clamp01(in_pan_fraction) if in_pan_fraction is not None else None
        )
        quality["articulation_quality"] = evenness
        released = boolean("gripper_far")
        quality["release_ok"] = (
            clamp01(1.0 if released else 0.0) if released is not None else None
        )
        if (
            in_pan_fraction is not None
            and evenness is not None
            and quality["release_ok"] is not None
        ):
            primary_progress = clamp01(
                0.6 * float(in_pan_fraction)
                + 0.3 * float(evenness)
                + 0.1 * float(quality["release_ok"])
            )

    output: dict[str, Any] = {
        "primary_progress": primary_progress,
        "quality": quality,
        "subgoals": subgoals,
        "signals": signals,
    }
    if errors:
        output["errors"] = errors
    return output


def build_task_success_observation(
    task_name: str,
    *,
    raw_success: Any = False,
    checker_error: str | None = None,
    meta: Mapping[str, Any] | None = None,
    calibration_raw_success: Any | None = None,
    success_impl: str | None = None,
) -> TaskSuccessObservation:
    """Build the current generic success result from detached evidence."""

    name = str(task_name or "").strip()
    strict_ok = bool(raw_success) if checker_error is None else False
    supplied_meta = dict(meta or {})
    output_meta: dict[str, Any] = {"success_impl": "env._check_success"}
    output_meta.update(supplied_meta)
    if checker_error is None:
        output_meta["success_impl"] = (
            "env._check_success" if success_impl is None else str(success_impl)
        )
    else:
        output_meta["success_impl"] = (
            f"{name}.fallback_failed" if success_impl is None else str(success_impl)
        )
        output_meta["fallback_reason"] = str(checker_error)

    if name == "OpenBlenderLid":
        trajectory_fallback = output_meta.get(
            "trajectory_fallback",
            None,
        )
        if (
            isinstance(trajectory_fallback, Mapping)
            and trajectory_fallback
            and not bool(strict_ok)
            and bool(
                trajectory_fallback.get(
                    "opened_by_displacement",
                    None,
                )
            )
        ):
            output_meta["fallback_reason"] = (
                "env._check_success_false_traj_opened_ignored"
                if checker_error is None
                else (f"env._check_success_error_traj_opened_ignored:{checker_error}")
            )
            output_meta["trajectory_fallback_used_for_success"] = False

    components = output_meta.get(
        "task_success_components",
        None,
    )
    if isinstance(components, Mapping) and "metrics" not in output_meta:
        output_meta["metrics"] = task_success_metrics_from_components(
            name,
            components,
        )
    calibration_raw = (
        strict_ok if calibration_raw_success is None else bool(calibration_raw_success)
    )
    adjusted_ok, adjustment = _adjust_task_success(
        name,
        strict_ok=bool(strict_ok),
        raw_success_ok=bool(calibration_raw),
        meta=output_meta,
    )
    output_meta["strict_task_check_success"] = bool(strict_ok)
    # The two raw booleans remain internal compatibility evidence for saved
    # task_check_success artifacts. Public reports expose only SR-B / SR-P.
    output_meta["calibrated_task_check_success"] = bool(adjusted_ok)
    output_meta["success_policy"] = "adjusted"
    output_meta["success_adjustment"] = adjustment
    metrics = output_meta.get("metrics", None)
    if bool(adjusted_ok) and not isinstance(metrics, dict):
        metrics = {}
        output_meta["metrics"] = metrics
    if bool(adjusted_ok) and isinstance(metrics, dict):
        metrics["primary_progress"] = 1.0
    return TaskSuccessObservation(
        task_check_success=bool(adjusted_ok),
        strict_task_check_success=bool(strict_ok),
        calibrated_task_check_success=bool(adjusted_ok),
        meta=output_meta,
        error=None,
    )


def _explicit_boolean_view(
    fields: tuple[tuple[str, Any], ...],
) -> dict[str, Any]:
    present = [
        {"field": field, "value": value} for field, value in fields if value is not None
    ]
    if not present:
        return {
            "status": "missing",
            "value": None,
            "source_fields": [],
        }
    if any(not isinstance(item["value"], bool) for item in present):
        return {
            "status": "invalid",
            "value": None,
            "source_fields": [str(item["field"]) for item in present],
        }
    values = {bool(item["value"]) for item in present}
    if len(values) != 1:
        return {
            "status": "conflict",
            "value": None,
            "source_fields": [str(item["field"]) for item in present],
        }
    return {
        "status": "observed",
        "value": values.pop(),
        "source_fields": [str(item["field"]) for item in present],
    }


def _primary_progress_view(
    final_meta: Mapping[str, Any],
) -> dict[str, Any]:
    metrics = final_meta.get("metrics")
    if not isinstance(metrics, Mapping) or "primary_progress" not in metrics:
        return {"status": "missing", "value": None}
    value = metrics.get("primary_progress")
    if (
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not np.isfinite(float(value))
        or not 0.0 <= float(value) <= 1.0
    ):
        return {"status": "invalid", "value": None}
    return {"status": "observed", "value": float(value)}


def task_success_views_from_payload(
    payload: Mapping[str, Any],
    *,
    source_identity: Mapping[str, Any] | str | None = None,
) -> dict[str, Any]:
    """Derive the single public SR-B / SR-P view.

    Saved artifacts may contain historical checker-evidence fields, but this
    boundary intentionally does not expose them as alternative success rates.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    if source_identity is not None and not isinstance(
        source_identity,
        (Mapping, str),
    ):
        raise TypeError("source_identity must be a mapping or string")

    final_meta_raw = payload.get("final_task_meta")
    final_meta = dict(final_meta_raw) if isinstance(final_meta_raw, Mapping) else {}
    selected = _explicit_boolean_view(
        (
            (
                "final_task_check_success",
                payload.get("final_task_check_success"),
            ),
        )
    )
    progress = _primary_progress_view(final_meta)
    source_schema = payload.get("format")
    schema_valid = source_schema == CURRENT_TASK_SUCCESS_SCHEMA
    success_policy = final_meta.get("success_policy")
    if not schema_valid:
        canonical_status = "invalid_schema"
    elif selected["status"] != "observed":
        canonical_status = str(selected["status"])
    elif success_policy not in {"adjusted", "calibrated"}:
        canonical_status = "invalid_policy"
    elif progress["status"] != "observed":
        canonical_status = str(progress["status"])
    else:
        canonical_status = "observed"
    canonical_final = {
        "protocol": CURRENT_TASK_SUCCESS_PROTOCOL,
        "status": canonical_status,
        "success_policy": "adjusted",
        "SR-B": selected["value"],
        "SR-P": progress["value"],
        "binary_source": "final_task_check_success",
        "partial_source": "final_task_meta.metrics.primary_progress",
    }

    result: dict[str, Any] = {
        "format": SINGLE_UID_TASK_SUCCESS_VIEWS_SCHEMA,
        "source_format": source_schema,
        "canonical_final": canonical_final,
    }
    if source_identity is not None:
        result["source_identity"] = (
            dict(source_identity)
            if isinstance(source_identity, Mapping)
            else source_identity
        )
    return normalize_task_success_payload(result)


def canonical_task_success_view(
    views: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the adjusted public view from the release schema."""

    schema = views.get("format")
    if schema != SINGLE_UID_TASK_SUCCESS_VIEWS_SCHEMA:
        raise ValueError("task-success views schema is unsupported")
    raw = views.get("canonical_final")
    if not isinstance(raw, Mapping):
        raise ValueError("task-success views lack a canonical final view")
    return normalize_task_success_payload(dict(raw))


def normalize_task_success_views(
    views: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one release-schema task-success view."""

    normalized = normalize_task_success_payload(dict(views))
    schema = normalized.get("format")
    if schema != SINGLE_UID_TASK_SUCCESS_VIEWS_SCHEMA:
        raise ValueError("task-success views schema is unsupported")
    canonical_task_success_view(normalized)
    return normalized


def normalize_task_success_payload(
    payload: Mapping[str, Any],
) -> dict[str, Any]:
    """Return one task-success record with JSON-native NumPy values."""

    def to_serializable(value: Any) -> Any:
        if isinstance(value, np.ndarray):
            return value.tolist()
        if isinstance(value, np.bool_):
            return bool(value)
        if isinstance(value, (np.float32, np.float64)):
            return float(value)
        if isinstance(value, (np.int32, np.int64)):
            return int(value)
        if isinstance(value, dict):
            return {key: to_serializable(item) for key, item in value.items()}
        if isinstance(value, list):
            return [to_serializable(item) for item in value]
        return value

    return to_serializable(dict(payload))


def publish_task_success_json(
    payload: Mapping[str, Any],
    output_path: str | os.PathLike[str],
) -> str:
    """Atomically publish one explicit task-success artifact.

    This writer is never selected implicitly by the replay evaluator.
    """

    path = Path(output_path).expanduser().resolve()
    if not str(os.fspath(output_path)).strip():
        raise ValueError("output_path is required")
    if payload.get("format") != CURRENT_TASK_SUCCESS_SCHEMA:
        raise ValueError(
            f"payload format must be {CURRENT_TASK_SUCCESS_SCHEMA!r}"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(
        normalize_task_success_payload(payload),
        indent=4,
        ensure_ascii=False,
    )
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as stream:
            temporary_path = Path(stream.name)
            stream.write(text)
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary_path, 0o644)
        os.replace(temporary_path, path)
        temporary_path = None
    finally:
        if temporary_path is not None:
            temporary_path.unlink(missing_ok=True)
    return path.as_posix()


__all__ = [
    "CURRENT_ADJUSTMENT_POLICY",
    "CURRENT_TASK_SUCCESS_PROTOCOL",
    "CURRENT_TASK_SUCCESS_SCHEMA",
    "PAPER_FINAL_TASK_SUCCESS_PROTOCOL",
    "SINGLE_UID_TASK_SUCCESS_VIEWS_SCHEMA",
    "TaskSuccessCheckResult",
    "TaskSuccessObservation",
    "build_task_success_observation",
    "canonical_task_success_view",
    "cheesybread_metrics_from_components",
    "normalize_task_success_payload",
    "normalize_task_success_views",
    "publish_task_success_json",
    "summarize_openblenderlid_trajectory",
    "summarize_object_trajectories",
    "task_success_metrics_from_components",
]
