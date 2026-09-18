"""Numeric gripper-action recognition.

The recognizer consumes trajectory records, derives simulator-independent
motion features, and emits the stable action payload used by downstream
trajectory and execution code.
"""

from __future__ import annotations

from dataclasses import asdict
from typing import Any, Dict, Literal, Optional

import numpy as np

from .features import (
    Features,
    GraspParams,
    Method,
    _argmax_delta_after,
    _build_features_2d,
    _build_features_3d,
    _clip01,
    _fill_short_false_gaps,
    _first_sustained_true,
    _fuse_features,
    _min_run_filter,
    _segments_from_bool,
    pack_action_trajectory_arrays,
)
from .numeric_config import load_numeric_action_params

__all__ = ["NumericActionRecognizer", "compute_gripper_actions"]


def _detect_attached_segments(
    feat: Features,
    *,
    tau_near: float,
    tau_hold: float,
    tau_vrel_hold: float,
    tau_cos_hold: float,
    params: GraspParams,
) -> tuple[
    list[tuple[int, int]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    """Find inclusive intervals in which the object stays close to the EEF."""
    distance = np.asarray(feat.D_win)
    near_raw = np.asarray(feat.valid, dtype=bool) & (distance <= tau_near)
    near = _fill_short_false_gaps(near_raw, params.bridge_gap_near)
    near = _min_run_filter(
        near,
        min_true=params.min_run_near,
        min_false=1,
    )

    attachment_motion = (np.asarray(feat.VREL_win) < tau_vrel_hold) | (
        np.asarray(feat.COS_win) > tau_cos_hold
    )
    attached_raw = (
        np.asarray(feat.valid, dtype=bool)
        & near
        & (distance < tau_hold)
        & attachment_motion
    )
    attached = _fill_short_false_gaps(
        attached_raw,
        params.bridge_gap_attached,
    )
    attached = _min_run_filter(
        attached,
        min_true=params.min_on,
        min_false=params.min_off,
    )
    return _segments_from_bool(attached), near, attached_raw, attached


def _build_motion_detach(
    feat: Features,
    *,
    tau_vrel_hold: float,
    params: GraspParams,
) -> np.ndarray:
    """Return frames whose relative motion is consistent with detachment."""
    relative_speed = np.asarray(feat.VREL_win)
    motion = relative_speed > (tau_vrel_hold * params.detach_vrel_ratio)
    if params.detach_use_cos:
        motion &= np.asarray(feat.COS_win) < params.tau_cos_detach
    return np.asarray(motion, dtype=bool)


def _build_detach_soft_hard(
    feat: Features,
    *,
    tau_near: float,
    tau_vrel_hold: float,
    params: GraspParams,
    lost_mask: Optional[np.ndarray] = None,
) -> tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Build permissive and distance-only detachment evidence."""
    dist_soft = float(tau_near * params.detach_dist_ratio_soft)
    dist_hi = float(tau_near * params.detach_dist_ratio_hi)
    vrel_thr = float(tau_vrel_hold * params.detach_vrel_ratio)

    distance = np.asarray(feat.D_win)
    soft = distance > dist_soft
    hard = distance > dist_hi
    soft |= _build_motion_detach(
        feat,
        tau_vrel_hold=tau_vrel_hold,
        params=params,
    )
    if lost_mask is not None:
        soft |= np.asarray(lost_mask, dtype=bool)

    thresholds = {
        "dist_soft": dist_soft,
        "dist_hi": dist_hi,
        "vrel_thr": vrel_thr,
        "cos_detach": float(params.tau_cos_detach),
    }
    return (
        np.asarray(soft, dtype=bool),
        np.asarray(hard, dtype=bool),
        thresholds,
    )


def _refine_close_open(
    feat: Features,
    segs: list[tuple[int, int]],
    *,
    tau_near: float,
    tau_hold: float,
    tau_vrel_hold: float,
    tau_cos_hold: float,
    params: GraspParams,
    lost_soft_mask: Optional[np.ndarray] = None,
    feat_2d: Optional[Features] = None,
    feat_3d: Optional[Features] = None,
    tau_near_2d: Optional[float] = None,
    tau_vrel_2d: Optional[float] = None,
    tau_near_3d: Optional[float] = None,
    tau_vrel_3d: Optional[float] = None,
) -> tuple[list[Dict[str, Optional[int]]], Dict[str, Any]]:
    """Convert attached intervals into close, hold, and open boundaries."""
    detach_soft, detach_hard, thresholds = _build_detach_soft_hard(
        feat,
        tau_near=tau_near,
        tau_vrel_hold=tau_vrel_hold,
        params=params,
        lost_mask=lost_soft_mask,
    )

    can_compare_modalities = (
        params.fused_open_need_both
        and feat_2d is not None
        and feat_3d is not None
        and tau_near_2d is not None
        and tau_vrel_2d is not None
        and tau_near_3d is not None
        and tau_vrel_3d is not None
    )
    if can_compare_modalities:
        _, hard_2d, _ = _build_detach_soft_hard(
            feat_2d,
            tau_near=float(tau_near_2d),
            tau_vrel_hold=float(tau_vrel_2d),
            params=params,
        )
        _, hard_3d, _ = _build_detach_soft_hard(
            feat_3d,
            tau_near=float(tau_near_3d),
            tau_vrel_hold=float(tau_vrel_3d),
            params=params,
        )
        detach_hard = hard_2d & hard_3d

    close_distance_valid = np.isfinite(feat.D_win)
    close_speed_valid = np.isfinite(feat.VREL_win)
    close_cosine_valid = np.isfinite(feat.COS_win)

    min_detach_run = int(params.min_detach_run)

    def refine_close(hold_start: int) -> int:
        radius = max(0, int(params.refine_radius))
        score_start = max(0, hold_start - radius)
        score_stop = min(
            len(feat.D_win) - 1,
            hold_start + radius,
        )
        close_scores = []
        for index in range(score_start, score_stop + 1):
            distance_score = 0.0
            if close_distance_valid[index]:
                distance_score = _clip01(
                    1.0 - float(feat.D_win[index]) / max(1e-6, float(tau_hold))
                )
            speed_score = 0.0
            if close_speed_valid[index]:
                speed_score = _clip01(
                    1.0 - float(feat.VREL_win[index]) / max(1e-6, float(tau_vrel_hold))
                )
            cosine_score = 0.0
            if close_cosine_valid[index]:
                cosine_score = _clip01(
                    (float(feat.COS_win[index]) - tau_cos_hold)
                    / max(
                        1e-6,
                        1.0 - float(tau_cos_hold),
                    )
                )
            close_scores.append(
                params.w_dist * distance_score
                + params.w_vrel * speed_score
                + params.w_cos * cosine_score
            )

        score_deltas = np.diff(np.asarray(close_scores, dtype=float))
        if len(score_deltas):
            transition_left = score_start + int(np.argmax(score_deltas))
        else:
            transition_left = hold_start - 1
        backshift = int(params.close_backshift)
        shifted = transition_left + 1 - backshift
        return max(0, min(shifted, hold_start - 1))

    def find_open(hold_end: int) -> Optional[int]:
        search_start = hold_end + 1
        soft_start = _first_sustained_true(
            detach_soft,
            start=search_start,
            min_run=min_detach_run,
        )

        open_index: Optional[int] = None
        if soft_start is not None:
            open_index = _first_sustained_true(
                detach_hard,
                start=soft_start,
                min_run=min_detach_run,
            )
            has_unsustained_hard = bool(np.any(detach_hard[soft_start:]))
            if (
                open_index is None
                and params.enable_max_delta_fallback
                and has_unsustained_hard
            ):
                delta_index = _argmax_delta_after(
                    np.asarray(feat.D_win),
                    start=soft_start,
                    window=int(params.max_delta_window),
                )
                if delta_index is not None:
                    fallback_index = max(
                        search_start,
                        int(delta_index) + max(1, int(params.win_radius)),
                    )
                    if fallback_index < len(detach_hard):
                        open_index = fallback_index
        return open_index

    refined: list[Dict[str, Optional[int]]] = []
    segment_index = 0
    while segment_index < len(segs):
        hold_s, hold_e = (int(value) for value in segs[segment_index])
        close = refine_close(hold_s)
        open_index = find_open(hold_e)

        while segment_index + 1 < len(segs):
            next_start, next_end = (int(value) for value in segs[segment_index + 1])
            if open_index is None:
                merge_gap = 2 * int(params.bridge_gap_attached) + 1
                should_merge = next_start < hold_e + merge_gap
            else:
                should_merge = next_start <= open_index
            if not should_merge:
                break
            segment_index += 1
            hold_e = max(hold_e, next_end)
            open_index = find_open(hold_e)

        refined.append(
            {
                "close": close,
                "hold_s": hold_s,
                "hold_e": hold_e,
                "open": (None if open_index is None else int(open_index)),
            }
        )
        segment_index += 1

    policy: Dict[str, Any] = {
        "dist_soft": thresholds["dist_soft"],
        "dist_hi": thresholds["dist_hi"],
        "min_detach_run": params.min_detach_run,
        "enable_max_delta_fallback": (params.enable_max_delta_fallback),
        "max_delta_window": params.max_delta_window,
        "fused_open_need_both": params.fused_open_need_both,
    }
    return refined, policy


def _finite_list(
    values: np.ndarray,
    *,
    missing: float,
) -> list[Any]:
    array = np.asarray(values)
    return np.where(np.isfinite(array), array, missing).tolist()


def _feature_debug(feat: Features) -> Dict[str, list[Any]]:
    distance = feat.D if feat.D is not None else feat.D_win
    relative_speed = feat.VREL if feat.VREL is not None else feat.VREL_win
    cosine = feat.COS if feat.COS is not None else feat.COS_win
    return {
        "D": _finite_list(distance, missing=-1.0),
        "VREL": _finite_list(relative_speed, missing=-1.0),
        "COS": _finite_list(cosine, missing=-2.0),
        "D_win": _finite_list(feat.D_win, missing=-1.0),
        "VREL_win": _finite_list(feat.VREL_win, missing=-1.0),
        "COS_win": _finite_list(feat.COS_win, missing=-2.0),
        "valid": np.asarray(feat.valid, dtype=bool).tolist(),
    }


def _build_actions(
    frames: np.ndarray,
    valid: np.ndarray,
    segments: list[Dict[str, Optional[int]]],
    *,
    gripper_close_cmd: float,
    gripper_open_cmd: float,
    gripper_hold_cmd: float,
    invalid_cmd_mode: str,
) -> list[Dict[str, Any]]:
    length = len(frames)
    held = np.zeros(length, dtype=bool)
    held_before_close = np.zeros(length, dtype=bool)
    events: list[Optional[str]] = [None] * length

    for segment in segments:
        hold_s = int(segment["hold_s"])
        hold_e = int(segment["hold_e"])
        open_index = segment["open"]
        hold_stop = length if open_index is None else min(length, hold_e + 1)

        close = int(segment["close"])
        if 0 <= close < length:
            held_before_close[close] = bool(held[close])
            events[close] = "close"
        held[hold_s:hold_stop] = True
        if open_index is not None and 0 <= int(open_index) < length:
            events[int(open_index)] = "open"

    actions: list[Dict[str, Any]] = []
    valid_array = np.asarray(valid, dtype=bool)
    command_values = np.empty(length, dtype=np.float32)
    for index, frame in enumerate(frames):
        is_valid = bool(valid_array[index])
        event = events[index]
        is_held = (
            bool(held[index])
            and is_valid
            and (event != "close" or bool(held_before_close[index]))
        )

        if not is_valid:
            command = (
                gripper_hold_cmd if invalid_cmd_mode == "hold" else gripper_open_cmd
            )
        elif event == "close":
            command = gripper_close_cmd
        elif event == "open":
            command = gripper_open_cmd
        elif is_held:
            command = gripper_close_cmd
        else:
            command = gripper_open_cmd
        command_values[index] = float(command)

        actions.append(
            {
                "frame": int(frame),
                "state": "hold" if is_held else "open",
                "event": event,
                "grasp": int(is_held),
                "gripper_cmd": float(command_values[index]),
                "valid": is_valid,
            }
        )
    return actions


def compute_gripper_actions(
    ee_traj: Dict[str, Any],
    obj_traj: Dict[str, Any],
    *,
    method: Method = "3d",
    params: Optional[GraspParams] = None,
    params_config_path: Optional[str] = None,
    ee_key: str = "eef_controller",
    obj_key: str = "obj_visual_center",
    return_debug: bool = True,
    gripper_close_cmd: float = 1.0,
    gripper_open_cmd: float = -1.0,
    gripper_hold_cmd: float = 0.0,
    invalid_cmd_mode: Literal["hold", "open"] = "hold",
) -> Dict[str, Any]:
    """Infer gripper actions from aligned EEF and object trajectories."""
    active_params = (
        params if params is not None else load_numeric_action_params(params_config_path)
    )
    (
        frames,
        ee_uv,
        obj_uv,
        ee_world,
        obj_world,
        obj_vis,
        obj_vis_valid,
    ) = pack_action_trajectory_arrays(
        ee_traj,
        obj_traj,
        ee_key=ee_key,
        obj_key=obj_key,
    )
    if len(frames) == 0:
        raise RuntimeError("Empty trajectory for gripper action extraction.")

    feat_2d = _build_features_2d(ee_uv, obj_uv, active_params)
    feat_3d = _build_features_3d(
        ee_world,
        obj_world,
        active_params,
    )

    obj_vis_used = bool(np.any(obj_vis_valid))
    visibility_lost = np.asarray(
        obj_vis < active_params.obj_vis_th,
        dtype=bool,
    )

    detach_vis: Optional[np.ndarray] = None
    if obj_vis_used and active_params.obj_vis_as_soft_detach:
        detach_vis = _min_run_filter(
            visibility_lost,
            min_true=active_params.obj_vis_min_run,
            min_false=1,
        )

    refine_2d: Optional[Features] = None
    refine_3d: Optional[Features] = None
    tau_near_2d: Optional[float] = None
    tau_vrel_2d: Optional[float] = None
    tau_near_3d: Optional[float] = None
    tau_vrel_3d: Optional[float] = None

    if method == "2d":
        feat = feat_2d
        tau_near = active_params.tau_near_2d
        tau_hold = active_params.tau_hold_2d
        tau_vrel_hold = active_params.tau_vrel_hold_2d
    elif method == "3d":
        feat = feat_3d
        tau_near = active_params.tau_near_3d
        tau_hold = active_params.tau_hold_3d
        tau_vrel_hold = active_params.tau_vrel_hold_3d
    elif method == "fused":
        feat = _fuse_features(feat_3d, feat_2d, active_params)
        if active_params.prefer_3d_in_fused:
            tau_near = active_params.tau_near_3d
            tau_hold = active_params.tau_hold_3d
            tau_vrel_hold = active_params.tau_vrel_hold_3d
        else:
            tau_near = active_params.tau_near_2d
            tau_hold = active_params.tau_hold_2d
            tau_vrel_hold = active_params.tau_vrel_hold_2d
        refine_2d = feat_2d
        refine_3d = feat_3d
        tau_near_2d = active_params.tau_near_2d
        tau_vrel_2d = active_params.tau_vrel_hold_2d
        tau_near_3d = active_params.tau_near_3d
        tau_vrel_3d = active_params.tau_vrel_hold_3d
    else:
        raise ValueError(f"Unknown method: {method}")

    if obj_vis_used:
        feat.valid &= ~visibility_lost

    valid = np.asarray(feat.valid, dtype=bool).copy()
    lost_soft_mask = detach_vis

    segs_raw, near, attached_raw, attached = _detect_attached_segments(
        feat,
        tau_near=tau_near,
        tau_hold=tau_hold,
        tau_vrel_hold=tau_vrel_hold,
        tau_cos_hold=active_params.tau_cos_hold,
        params=active_params,
    )
    segments, open_policy = _refine_close_open(
        feat,
        segs_raw,
        tau_near=tau_near,
        tau_hold=tau_hold,
        tau_vrel_hold=tau_vrel_hold,
        tau_cos_hold=active_params.tau_cos_hold,
        params=active_params,
        lost_soft_mask=lost_soft_mask,
        feat_2d=refine_2d,
        feat_3d=refine_3d,
        tau_near_2d=tau_near_2d,
        tau_vrel_2d=tau_vrel_2d,
        tau_near_3d=tau_near_3d,
        tau_vrel_3d=tau_vrel_3d,
    )

    payload: Dict[str, Any] = {
        "meta": {
            "algorithm": "numeric",
            "method": method,
            "params": asdict(active_params),
            "ee_key": ee_key,
            "obj_key": obj_key,
            "T": len(frames),
            "gripper_map": {
                "close_cmd": float(gripper_close_cmd),
                "open_cmd": float(gripper_open_cmd),
                "hold_cmd": float(gripper_hold_cmd),
                "invalid_cmd_mode": invalid_cmd_mode,
            },
        },
        "segments": segments,
        "actions": _build_actions(
            frames,
            valid,
            segments,
            gripper_close_cmd=gripper_close_cmd,
            gripper_open_cmd=gripper_open_cmd,
            gripper_hold_cmd=gripper_hold_cmd,
            invalid_cmd_mode=invalid_cmd_mode,
        ),
    }

    if return_debug and active_params.debug:
        detach_soft, detach_hard, detach_thresholds = _build_detach_soft_hard(
            feat,
            tau_near=tau_near,
            tau_vrel_hold=tau_vrel_hold,
            params=active_params,
            lost_mask=lost_soft_mask,
        )
        detach_motion = _build_motion_detach(
            feat,
            tau_vrel_hold=tau_vrel_hold,
            params=active_params,
        )
        detach_thresholds.update(
            {
                "use_cos": active_params.detach_use_cos,
                "min_detach_run": active_params.min_detach_run,
                "enable_max_delta_fallback": (active_params.enable_max_delta_fallback),
                "max_delta_window": (active_params.max_delta_window),
            }
        )
        payload["debug"] = {
            "D_win": _finite_list(feat.D_win, missing=-1.0),
            "VREL_win": _finite_list(
                feat.VREL_win,
                missing=-1.0,
            ),
            "COS_win": _finite_list(feat.COS_win, missing=-2.0),
            "valid": valid.tolist(),
            "obj_vis": _finite_list(obj_vis, missing=1.0),
            "obj_vis_used": obj_vis_used,
            "obj_vis_th": active_params.obj_vis_th,
            "detach_vis": (None if detach_vis is None else detach_vis.tolist()),
            "segs_raw": [[int(start), int(end)] for start, end in segs_raw],
            "near": near.tolist(),
            "attached_raw": attached_raw.tolist(),
            "attached": attached.tolist(),
            "detach_soft": detach_soft.tolist(),
            "detach_hard_dist": detach_hard.tolist(),
            "detach_motion": detach_motion.tolist(),
            "detach_thresholds": detach_thresholds,
            "open_policy": open_policy,
            "3d": _feature_debug(feat_3d),
            "2d": _feature_debug(feat_2d),
        }
    return payload


class NumericActionRecognizer:
    """Compatibility facade around :func:`compute_gripper_actions`."""

    def __init__(
        self,
        *,
        method: Method = "3d",
        params: Optional[GraspParams] = None,
        params_config_path: Optional[str] = None,
        ee_key: str = "eef_controller",
        obj_key: str = "obj_visual_center",
        return_debug: bool = True,
        gripper_close_cmd: float = 1.0,
        gripper_open_cmd: float = -1.0,
        gripper_hold_cmd: float = 0.0,
        invalid_cmd_mode: Literal["hold", "open"] = "hold",
    ) -> None:
        self.method = method
        self.params = (
            params
            if params is not None
            else load_numeric_action_params(params_config_path)
        )
        self.ee_key = ee_key
        self.obj_key = obj_key
        self.return_debug = return_debug
        self.gripper_close_cmd = gripper_close_cmd
        self.gripper_open_cmd = gripper_open_cmd
        self.gripper_hold_cmd = gripper_hold_cmd
        self.invalid_cmd_mode = invalid_cmd_mode

    def infer(
        self,
        *,
        ee_traj: Dict[str, Any],
        obj_traj: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Run the recognizer with the constructor's stored options."""
        return compute_gripper_actions(
            ee_traj,
            obj_traj,
            method=self.method,
            params=self.params,
            ee_key=self.ee_key,
            obj_key=self.obj_key,
            return_debug=self.return_debug,
            gripper_close_cmd=self.gripper_close_cmd,
            gripper_open_cmd=self.gripper_open_cmd,
            gripper_hold_cmd=self.gripper_hold_cmd,
            invalid_cmd_mode=self.invalid_cmd_mode,
        )
