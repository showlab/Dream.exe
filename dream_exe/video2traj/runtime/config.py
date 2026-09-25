"""Portable configuration assembly for the video-to-trajectory runtime.

This module owns no benchmark discovery and performs no model loading.  It
turns a caller-provided mapping (or one explicit JSON document) into the
portable runtime configuration consumed by :mod:`dream_exe.video2traj`.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional

from ..action.config import (
    action_config_to_dict,
    default_action_config_dict,
    load_action_config,
)
from ..depth.config import (
    default_depth_base_config_dict,
    default_target_calibrated_lift_config_dict,
    normalize_depth_base_config,
    normalize_target_calibrated_lift_config,
    validate_depth_base_config,
    validate_target_calibrated_lift_config,
)
from ..depth.contract import (
    external_depth_backend_id,
    external_depth_selection,
)
from ..depth.estimator import (
    DEFAULT_DEPTH_PRESET,
    DEPTH_ESTIMATOR_PRESETS,
    SUPPORTED_DEPTH_BACKENDS,
)
from ..pose.config import (
    POINTCLOUD_KABSCH_BACKEND,
    default_pose_config_dict,
    pose_backend_identity_is_known,
)


DEFAULT_PIPELINE_CONFIG_FILENAME = "trajectory.json"
SUPPORTED_DEPTH_PRESETS = DEPTH_ESTIMATOR_PRESETS

_DEFAULT_REGION_TARGET = {
    "selector": "simulation",
    "prompt": "",
    "bbox_xyxy": None,
    "simulation": {"instance_name": ""},
    "sampling": {"method": "mask_3d_fps"},
    "visual": {"bbox_source": "grounding_dino"},
}
_DEFAULT_TRACK_COUNTS = {"eef": 150, "obj": 50}
_DEFAULT_GEOMETRY = {
    "eef": {
        "interpolate": True,
        "max_gap": 10,
        "fill_ends": False,
        "end_max": 3,
        "carry_prev": True,
        "min_points": 1,
    },
    "obj": {
        "interpolate": False,
        "max_gap": 10,
        "fill_ends": False,
        "end_max": 3,
        "carry_prev": False,
        "min_points": 1,
    },
}
_DEFAULT_GRIPPER = {
    "strategy": "task_prior",
    "method": "3d",
    "numeric_config_path": "",
    "gripper_close_cmd": 1.0,
    "gripper_open_cmd": -1.0,
    "gripper_hold_cmd": 0.0,
    "invalid_cmd_mode": "hold",
    "task_prior": {
        "task_name": "",
        "config_path": "",
        "close_timing_profile": "default",
        "num_close": None,
        "num_open": None,
    },
}
_DEPTH_ARTIFACT_DEFAULTS = {
    "save_canonical_npy": True,
    "save_canonical_mp4": True,
    "save_target_depth_npys": False,
    "save_target_depth_mp4s": False,
    "save_masks": False,
}
_DEPTH_FIELDS = frozenset(
    {
        "model",
        "config_path",
        "rollout_gt_depth_path",
        "use_rollout_gt_depth",
        "use_cache",
        "input_size",
        "estimated_depth_cache_path",
        "estimated_depth_cache_meta_path",
        "visualization_video_path",
        "force_recompute",
        "fp32",
        "base",
        "target_calibrated_lift",
        "artifact_policy",
        "save_rollout_gt_visualization",
    }
)


def _overlay(
    foundation: Dict[str, Any],
    changes: Dict[str, Any],
) -> Dict[str, Any]:
    """Return a detached recursive overlay."""

    result = copy.deepcopy(foundation)
    pending: list[tuple[Dict[str, Any], Dict[str, Any]]] = [(result, changes)]
    while pending:
        destination, source = pending.pop()
        for key, incoming in source.items():
            existing = destination.get(key)
            if isinstance(existing, dict) and isinstance(incoming, dict):
                child = copy.deepcopy(existing)
                destination[key] = child
                pending.append((child, incoming))
            else:
                destination[key] = copy.deepcopy(incoming)
    return result


def _deep_merge_dict(
    base: Dict[str, Any],
    override: Dict[str, Any],
) -> Dict[str, Any]:
    """Compatibility spelling for callers that need a detached overlay."""

    return _overlay(base, override)


def _default_pipeline_config() -> Dict[str, Any]:
    pose_defaults = {
        "enabled": True,
        **default_pose_config_dict(),
    }
    eef_region = copy.deepcopy(_DEFAULT_REGION_TARGET)
    eef_region.pop("visual")
    return {
        "input": {
            "selected_video": "rollout",
            "rollout_video_path": "",
            "gen_video_path": "",
        },
        "runtime": {"device": "cuda"},
        "region": {
            "preset": "default",
            "config_path": "",
            "targets": {
                "eef": eef_region,
                "obj": copy.deepcopy(_DEFAULT_REGION_TARGET),
                "objects": [],
            },
        },
        "tracking": {
            "targets": {
                "eef": {"num_points": _DEFAULT_TRACK_COUNTS["eef"]},
                "obj": {"num_points": _DEFAULT_TRACK_COUNTS["obj"]},
                "objects": [],
            }
        },
        "depth": {
            "model": DEFAULT_DEPTH_PRESET,
            "config_path": "",
            "rollout_gt_depth_path": "",
            "use_rollout_gt_depth": False,
            "use_cache": True,
            "input_size": 512,
            "estimated_depth_cache_path": "",
            "estimated_depth_cache_meta_path": "",
            "visualization_video_path": "",
            "force_recompute": False,
            "fp32": False,
            "base": default_depth_base_config_dict(),
            "target_calibrated_lift": (default_target_calibrated_lift_config_dict()),
            "artifact_policy": copy.deepcopy(_DEPTH_ARTIFACT_DEFAULTS),
            "save_rollout_gt_visualization": False,
        },
        "alignment": {"method": "translation"},
        "geometry": {
            "targets": {
                "eef": copy.deepcopy(_DEFAULT_GEOMETRY["eef"]),
                "obj": copy.deepcopy(_DEFAULT_GEOMETRY["obj"]),
                "objects": [],
            }
        },
        "pose": pose_defaults,
        "gripper": copy.deepcopy(_DEFAULT_GRIPPER),
        "action": default_action_config_dict(),
        "task": {"runtime_stage_order": {"mode": "annotated"}},
    }


def default_pipeline_config_dict() -> Dict[str, Any]:
    """Return a new portable default configuration."""

    return _default_pipeline_config()


def _available_depth_presets() -> tuple[str, ...]:
    return tuple(SUPPORTED_DEPTH_PRESETS)


def _text(
    value: Any,
    *,
    fallback: str = "",
    lowercase: bool = False,
) -> str:
    rendered = str(value or fallback).strip()
    return rendered.lower() if lowercase else rendered


def _exact_bool(value: Any, *, path: str) -> bool:
    if type(value) is not bool:
        raise ValueError(f"{path} must be a boolean")
    return value


def _integer(value: Any, *, path: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{path} must be an integer")
    return value


def _object(value: Any) -> Dict[str, Any]:
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _normalize_bbox(value: Any) -> Any:
    if value is None:
        return None
    return [int(coordinate) for coordinate in value]


def _normalize_visual_target(
    entry: Dict[str, Any],
    *,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    target = _overlay(defaults, entry)
    target["selector"] = _text(
        target.get("selector"),
        fallback=str(defaults.get("selector", "simulation")),
    )
    target["prompt"] = _text(target.get("prompt"))
    target["bbox_xyxy"] = _normalize_bbox(target.get("bbox_xyxy"))

    simulation = _overlay(
        _object(defaults.get("simulation")),
        _object(target.get("simulation")),
    )
    simulation["instance_name"] = _text(simulation.get("instance_name"))
    target["simulation"] = simulation

    sampling = _overlay(
        _object(defaults.get("sampling")),
        _object(target.get("sampling")),
    )
    sampling["method"] = _text(
        sampling.get("method"),
        fallback=_text(
            _object(defaults.get("sampling")).get("method"),
            fallback="mask_3d_fps",
        ),
    )
    target["sampling"] = sampling

    visual = _overlay(
        _object(defaults.get("visual")),
        _object(target.get("visual")),
    )
    visual["bbox_source"] = _text(
        visual.get("bbox_source"),
        fallback=_text(
            _object(defaults.get("visual")).get("bbox_source"),
            fallback="grounding_dino",
        ),
    )
    target["visual"] = visual
    return target


def _normalize_region_object_target(
    entry: Dict[str, Any],
) -> Dict[str, Any]:
    identity = {
        "stage_id": _text(entry.get("stage_id")),
        "object_id": _text(entry.get("object_id")),
    }
    payload = _normalize_visual_target(
        entry,
        defaults=_DEFAULT_REGION_TARGET,
    )
    payload.update(identity)
    # Preserve the historical identity-first insertion order.
    return {**identity, **payload}


def _normalize_numeric_object_target(
    entry: Dict[str, Any],
    *,
    default_num_points: int,
) -> Dict[str, Any]:
    payload = copy.deepcopy(entry)
    payload["stage_id"] = _text(payload.get("stage_id"))
    payload["object_id"] = _text(payload.get("object_id"))
    payload["num_points"] = int(payload.get("num_points", default_num_points))
    return payload


def _normalize_geometry_object_target(
    entry: Dict[str, Any],
    *,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    payload = _overlay(defaults, entry)
    payload["stage_id"] = _text(entry.get("stage_id"))
    payload["object_id"] = _text(entry.get("object_id"))
    for field in ("interpolate", "fill_ends", "carry_prev"):
        payload[field] = bool(payload.get(field))
    for field in ("max_gap", "end_max", "min_points"):
        payload[field] = int(payload.get(field))
    identity = {
        "stage_id": payload.pop("stage_id"),
        "object_id": payload.pop("object_id"),
    }
    return {**identity, **payload}


def _normalize_region(root: Dict[str, Any]) -> Dict[str, Any]:
    region = copy.deepcopy(root)
    region["preset"] = _text(
        region.get("preset"),
        fallback="default",
    )
    region["config_path"] = _text(region.get("config_path"))
    targets = _object(region.get("targets"))

    for name in ("eef", "obj"):
        supplied = _object(targets.get(name))
        if name == "obj" and "selector" in supplied and not supplied["selector"]:
            supplied["selector"] = "off"
        targets[name] = _normalize_visual_target(
            supplied,
            defaults=_DEFAULT_REGION_TARGET,
        )

    entries = targets.get("objects")
    if not isinstance(entries, list) or not entries:
        entries = [
            {
                **copy.deepcopy(targets["obj"]),
                "stage_id": "s1",
                "object_id": "obj_s1",
            }
        ]
    targets["objects"] = [
        _normalize_region_object_target(_object(item)) for item in entries
    ]
    region["targets"] = targets
    return region


def _normalize_tracking(root: Dict[str, Any]) -> Dict[str, Any]:
    tracking = copy.deepcopy(root)
    targets = _object(tracking.get("targets"))
    for name in ("eef", "obj"):
        entry = _object(targets.get(name))
        entry["num_points"] = int(entry.get("num_points", _DEFAULT_TRACK_COUNTS[name]))
        targets[name] = entry
    entries = targets.get("objects")
    if not isinstance(entries, list) or not entries:
        entries = [{"stage_id": "s1", "object_id": "obj_s1"}]
    targets["objects"] = [
        _normalize_numeric_object_target(
            _object(item),
            default_num_points=_DEFAULT_TRACK_COUNTS["obj"],
        )
        for item in entries
    ]
    tracking["targets"] = targets
    return tracking


def _normalize_geometry(root: Dict[str, Any]) -> Dict[str, Any]:
    geometry = copy.deepcopy(root)
    targets = _object(geometry.get("targets"))
    for name in ("eef", "obj"):
        entry = _overlay(
            _DEFAULT_GEOMETRY[name],
            _object(targets.get(name)),
        )
        for field in ("interpolate", "fill_ends", "carry_prev"):
            entry[field] = bool(entry.get(field))
        for field in ("max_gap", "end_max", "min_points"):
            entry[field] = int(entry.get(field))
        targets[name] = entry
    entries = targets.get("objects")
    if not isinstance(entries, list) or not entries:
        entries = [{"stage_id": "s1", "object_id": "obj_s1"}]
    targets["objects"] = [
        _normalize_geometry_object_target(
            _object(item),
            defaults=_DEFAULT_GEOMETRY["obj"],
        )
        for item in entries
    ]
    geometry["targets"] = targets
    return geometry


def _normalize_depth(
    root: Dict[str, Any],
    *,
    explicitly_supplied: Any,
) -> Dict[str, Any]:
    if explicitly_supplied is None:
        supplied: Dict[str, Any] = {}
    elif isinstance(explicitly_supplied, dict):
        supplied = copy.deepcopy(explicitly_supplied)
    else:
        raise ValueError("pipeline config depth must be an object")
    unknown = sorted(set(supplied).difference(_DEPTH_FIELDS))
    if unknown:
        raise ValueError("depth contains unknown fields: " + ", ".join(unknown))

    depth = copy.deepcopy(root)
    depth["model"] = _text(
        depth.get("model"),
        fallback=DEFAULT_DEPTH_PRESET,
    )
    external_backend_id = external_depth_backend_id(depth["model"])
    if external_backend_id is not None:
        depth["model"] = external_depth_selection(external_backend_id)
    for name in (
        "config_path",
        "rollout_gt_depth_path",
        "estimated_depth_cache_path",
        "estimated_depth_cache_meta_path",
        "visualization_video_path",
    ):
        depth[name] = _text(depth.get(name))
    for name in (
        "use_rollout_gt_depth",
        "use_cache",
        "force_recompute",
        "fp32",
        "save_rollout_gt_visualization",
    ):
        depth[name] = _exact_bool(
            depth.get(name),
            path=f"depth.{name}",
        )
    depth["input_size"] = _integer(
        depth.get("input_size", 512),
        path="depth.input_size",
    )

    depth["base"] = normalize_depth_base_config(depth.get("base"))
    depth["target_calibrated_lift"] = normalize_target_calibrated_lift_config(
        depth.get("target_calibrated_lift")
    )
    raw_artifact_policy = depth.get("artifact_policy")
    if not isinstance(raw_artifact_policy, dict):
        raise ValueError("depth.artifact_policy must be an object")
    unknown_artifact_fields = sorted(
        set(raw_artifact_policy).difference(_DEPTH_ARTIFACT_DEFAULTS)
    )
    if unknown_artifact_fields:
        raise ValueError(
            "depth.artifact_policy contains unknown fields: "
            + ", ".join(unknown_artifact_fields)
        )
    artifact_policy = _overlay(
        _DEPTH_ARTIFACT_DEFAULTS,
        raw_artifact_policy,
    )
    for name in _DEPTH_ARTIFACT_DEFAULTS:
        artifact_policy[name] = _exact_bool(
            artifact_policy.get(name),
            path=f"depth.artifact_policy.{name}",
        )
    depth["artifact_policy"] = artifact_policy

    return depth


_POSE_BOOLEAN_FIELDS = frozenset(
    {
        "enabled",
        "force_recompute",
        "foundationpose_enabled",
        "foundationpose_fallback",
        "kabsch_enabled",
        "use_config_init_pose",
        "foundationpose_force_register_frame0",
        "anchor_rotation_constraint_enabled",
        "temporal_guard_enabled",
    }
)
_POSE_INTEGER_FIELDS = frozenset(
    {
        "init_refine_iter",
        "track_refine_iter",
        "debug",
        "min_correspondences",
        "min_inlier_correspondences",
    }
)
_POSE_FLOAT_FIELDS = frozenset(
    {
        "anchor_rotation_min_quality",
        "anchor_rotation_max_backend_kabsch_delta_deg",
        "inlier_threshold_m",
        "trim_quantile_scale",
        "min_shape_ratio",
        "max_angle_jump_deg",
        "temporal_guard_max_angle_deg",
        "foundationpose_max_angle_jump_deg",
        "max_step_deg",
        "stationary_position_epsilon_m",
        "stationary_max_angle_deg",
        "small_rotation_epsilon_deg",
        "ema_alpha",
        "min_pose_quality",
    }
)


def _normalize_pose(
    root: Dict[str, Any],
    *,
    explicitly_supplied: Dict[str, Any],
) -> Dict[str, Any]:
    pose = copy.deepcopy(root)
    for name in _POSE_BOOLEAN_FIELDS:
        pose[name] = bool(pose.get(name))
    for name in _POSE_INTEGER_FIELDS:
        pose[name] = int(pose.get(name))
    for name in _POSE_FLOAT_FIELDS:
        pose[name] = float(pose.get(name))

    pose["backend"] = (
        _text(
            pose.get("backend"),
            fallback=POINTCLOUD_KABSCH_BACKEND,
            lowercase=True,
        )
        or POINTCLOUD_KABSCH_BACKEND
    )
    for name in (
        "weights_root",
        "mesh_path",
        "pose_correction_path",
        "config_path",
    ):
        pose[name] = _text(pose.get(name))
    pose["pose_correction_side"] = _text(
        pose.get("pose_correction_side"),
        fallback="right",
        lowercase=True,
    )
    pose["temporal_guard_mode"] = _text(
        pose.get("temporal_guard_mode"),
        fallback="clamp",
        lowercase=True,
    )
    pose["model_kwargs"] = _object(pose.get("model_kwargs"))
    return pose


def _normalize_gripper(root: Dict[str, Any]) -> Dict[str, Any]:
    gripper = copy.deepcopy(root)
    for name in (
        "strategy",
        "method",
        "numeric_config_path",
        "invalid_cmd_mode",
    ):
        gripper[name] = _text(gripper.get(name))
    for name in (
        "gripper_close_cmd",
        "gripper_open_cmd",
        "gripper_hold_cmd",
    ):
        gripper[name] = float(gripper.get(name))

    task_prior = _object(gripper.get("task_prior"))
    task_prior["task_name"] = _text(task_prior.get("task_name"))
    task_prior["config_path"] = _text(task_prior.get("config_path"))
    task_prior["close_timing_profile"] = _text(
        task_prior.get("close_timing_profile"),
        fallback="default",
    )
    for name in ("num_close", "num_open"):
        value = task_prior.get(name)
        task_prior[name] = None if value is None else int(value)
    gripper["task_prior"] = task_prior
    return gripper


def _normalize_task(root: Dict[str, Any]) -> Dict[str, Any]:
    task = copy.deepcopy(root)
    order = task.get("runtime_stage_order")
    if isinstance(order, str):
        mode = _text(order, lowercase=True)
        order = {"mode": "coupling" if mode == "auto" else mode}
    elif isinstance(order, dict):
        order = copy.deepcopy(order)
        order["mode"] = _text(
            order.get("mode"),
            fallback="annotated",
            lowercase=True,
        )
    task["runtime_stage_order"] = order
    return task


def _canonical_source(
    supplied: Dict[str, Any],
    primary: str,
    legacy: str,
) -> Dict[str, Any]:
    if primary in supplied:
        return _object(supplied.get(primary))
    return _object(supplied.get(legacy))


def normalize_pipeline_config(
    cfg: Dict[str, Any],
    *,
    source_cfg: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Normalize a mapping without changing the caller-owned object."""

    supplied = (
        copy.deepcopy(source_cfg)
        if isinstance(source_cfg, dict)
        else copy.deepcopy(cfg)
    )
    normalized = _overlay(_default_pipeline_config(), cfg)

    input_config = _object(normalized.get("input"))
    input_config["selected_video"] = _text(
        input_config.get("selected_video"),
        fallback="rollout",
        lowercase=True,
    )
    normalized["input"] = input_config

    runtime = _object(normalized.get("runtime"))
    runtime["device"] = _text(
        runtime.get("device"),
        fallback="cuda",
        lowercase=True,
    )
    normalized["runtime"] = runtime

    normalized["region"] = _normalize_region(_object(normalized.get("region")))
    normalized["tracking"] = _normalize_tracking(_object(normalized.get("tracking")))
    normalized["depth"] = _normalize_depth(
        _object(normalized.get("depth")),
        explicitly_supplied=supplied.get("depth"),
    )
    alignment = _object(normalized.get("alignment"))
    alignment["method"] = _text(
        alignment.get("method"),
        fallback="translation",
    )
    normalized["alignment"] = alignment
    normalized["geometry"] = _normalize_geometry(_object(normalized.get("geometry")))

    pose_source = _object(supplied.get("pose"))
    normalized["pose"] = _normalize_pose(
        _object(normalized.get("pose")),
        explicitly_supplied=pose_source,
    )

    selected_gripper = _canonical_source(
        supplied,
        "gripper",
        "grasp",
    )
    gripper_root = _overlay(
        _DEFAULT_GRIPPER,
        selected_gripper,
    )
    normalized["gripper"] = _normalize_gripper(gripper_root)
    normalized["grasp"] = copy.deepcopy(normalized["gripper"])

    selected_action = _canonical_source(
        supplied,
        "action",
        "action_plan",
    )
    action_root = _overlay(
        default_action_config_dict(),
        selected_action,
    )
    explicitly_selected_controller = selected_action.get("controller")
    pose_enabled_was_explicit = "enabled" in pose_source
    if "controller" not in selected_action or (
        normalized["pose"]["enabled"]
        and explicitly_selected_controller == "OSC_POSITION"
        and not pose_enabled_was_explicit
    ):
        action_root["controller"] = (
            "OSC_POSE" if normalized["pose"]["enabled"] else "OSC_POSITION"
        )
    normalized["action"] = action_config_to_dict(load_action_config(action_root))
    normalized["action_plan"] = copy.deepcopy(normalized["action"])

    normalized["task"] = _normalize_task(_object(normalized.get("task")))
    return normalized


def _expect_object(
    value: Any,
    *,
    name: str,
    source: str,
) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError(f"pipeline config {name} must be an object: {source}")
    return value


def _expect_list(
    value: Any,
    *,
    name: str,
    source: str,
) -> list[Any]:
    if not isinstance(value, list):
        raise ValueError(f"pipeline config {name} must be a list: {source}")
    return value


def _validate_visual_target(
    target: Dict[str, Any],
    *,
    name: str,
    source: str,
    selectors: frozenset[str],
    require_manual_bbox: bool = True,
) -> None:
    selector = target.get("selector")
    if selector not in selectors:
        raise ValueError(f"Invalid {name}.selector in {source}: {selector}")
    bbox = target.get("bbox_xyxy")
    if bbox is not None and (not isinstance(bbox, (list, tuple)) or len(bbox) != 4):
        raise ValueError(f"Invalid {name}.bbox_xyxy in {source}: {bbox}")

    simulation = _expect_object(
        target.get("simulation"),
        name=f"{name}.simulation",
        source=source,
    )
    del simulation
    sampling = _expect_object(
        target.get("sampling"),
        name=f"{name}.sampling",
        source=source,
    )
    method = sampling.get("method")
    if method not in {
        "mask_2d_fps",
        "mask_3d_fps",
        "bbox_gaussian",
    }:
        raise ValueError(f"Invalid {name}.sampling.method in {source}: {method}")
    visual = _expect_object(
        target.get("visual"),
        name=f"{name}.visual",
        source=source,
    )
    requires_explicit_box = (
        require_manual_bbox
        and selector == "visual"
        and visual.get("bbox_source") == "manual"
    )
    if requires_explicit_box and bbox is None:
        raise ValueError(
            f"{name}.bbox_xyxy is required when visual.bbox_source=manual in {source}"
        )


def _validate_geometry_entry(
    entry: Dict[str, Any],
    *,
    name: str,
    source: str,
) -> None:
    for field in ("max_gap", "end_max"):
        if int(entry.get(field, 0)) < 0:
            raise ValueError(f"{name}.{field} must be >= 0 in {source}")
    if int(entry.get("min_points", 0)) <= 0:
        raise ValueError(f"{name}.min_points must be > 0 in {source}")


def _validate_pose(pose: Dict[str, Any], *, source: str) -> None:
    backend = pose.get("backend")
    if not pose_backend_identity_is_known(backend):
        raise ValueError(f"Unsupported pose.backend in {source}: {backend}")
    if pose.get("init_refine_iter", 0) <= 0:
        raise ValueError(f"pose.init_refine_iter must be > 0 in {source}")
    if pose.get("track_refine_iter", 0) <= 0:
        raise ValueError(f"pose.track_refine_iter must be > 0 in {source}")
    if pose.get("debug", 0) < 0:
        raise ValueError(f"pose.debug must be >= 0 in {source}")

    requirements = (
        ("min_correspondences", 3),
        ("min_inlier_correspondences", 3),
    )
    for field, minimum in requirements:
        if pose.get(field, 0) < minimum:
            raise ValueError(f"pose.{field} must be >= {minimum} in {source}")
    if pose.get("min_inlier_correspondences", 0) > pose.get("min_correspondences", 0):
        raise ValueError(
            "pose.min_inlier_correspondences cannot exceed "
            f"pose.min_correspondences in {source}"
        )
    if pose.get("inlier_threshold_m", 0.0) <= 0:
        raise ValueError(f"pose.inlier_threshold_m must be > 0 in {source}")
    if pose.get("trim_quantile_scale", 0.0) < 1:
        raise ValueError(f"pose.trim_quantile_scale must be >= 1 in {source}")
    if pose.get("min_shape_ratio", -1.0) < 0:
        raise ValueError(f"pose.min_shape_ratio must be >= 0 in {source}")
    for field in ("anchor_rotation_min_quality", "ema_alpha", "min_pose_quality"):
        value = float(pose.get(field, 0.0))
        accepted = 0 <= value <= 1
        if field == "ema_alpha":
            accepted = 0 < value <= 1
        if not accepted:
            interval = "(0, 1]" if field == "ema_alpha" else "[0, 1]"
            raise ValueError(f"pose.{field} must be in {interval} in {source}")


def _validate_gripper(
    gripper: Dict[str, Any],
    *,
    source: str,
) -> None:
    if gripper.get("strategy") not in {"numeric", "task_prior"}:
        raise ValueError(
            f"Unsupported gripper.strategy in {source}: {gripper.get('strategy')}"
        )
    if gripper.get("method") not in {"2d", "3d", "fused"}:
        raise ValueError(
            f"Unsupported gripper.method in {source}: {gripper.get('method')}"
        )
    if gripper.get("invalid_cmd_mode") not in {"hold", "open"}:
        raise ValueError(
            "Unsupported gripper.invalid_cmd_mode in "
            f"{source}: {gripper.get('invalid_cmd_mode')}"
        )
    prior = _expect_object(
        gripper.get("task_prior"),
        name="gripper.task_prior",
        source=source,
    )
    for field in ("num_close", "num_open"):
        value = prior.get(field)
        if value is not None and value < 0:
            raise ValueError(f"gripper.task_prior.{field} must be >= 0 in {source}")
    if prior.get("close_timing_profile") not in {
        "default",
        "contact_safe",
        "early_capture",
    }:
        raise ValueError(
            "gripper.task_prior.close_timing_profile must be one of "
            "'default', 'contact_safe', or 'early_capture' "
            f"in {source}"
        )
    # The benchmark adapter binds the per-task prior from UID/environment
    # metadata at runtime. A portable default therefore does not require a
    # task-specific mode selector or task name in this generic config.


def _validate_action(
    action: Dict[str, Any],
    *,
    pose: Dict[str, Any],
    source: str,
) -> None:
    if action.get("eef_key") != "eef_controller":
        raise ValueError(
            f"Unsupported action.eef_key in {source}: {action.get('eef_key')}"
        )
    if action.get("reference_frame") not in {"base", "world"}:
        raise ValueError(
            f"Unsupported action.reference_frame in {source}: "
            f"{action.get('reference_frame')}"
        )
    if action.get("controller") not in {
        "OSC_POSE",
        "OSC_POSITION",
        "IK_POSE",
    }:
        raise ValueError(
            f"Unsupported action.controller in {source}: {action.get('controller')}"
        )
    if action.get("policy_hz", 0) <= 0:
        raise ValueError(f"action.policy_hz must be > 0 in {source}")
    step = action.get("translation_step_budget_m")
    if step is not None and step <= 0:
        raise ValueError(f"action.translation_step_budget_m must be > 0 in {source}")
    if action.get("rotation_step_budget_rad", -1.0) < 0:
        raise ValueError(f"action.rotation_step_budget_rad must be >= 0 in {source}")
    if action.get("max_motion_steps_per_segment", 0) <= 0:
        raise ValueError(f"action.max_motion_steps_per_segment must be > 0 in {source}")
    if action.get("grasped_motion_stride", 0) <= 0:
        raise ValueError(f"action.grasped_motion_stride must be > 0 in {source}")
    # Keep an explicitly selected position-only controller compatible with
    # paper-era inputs. Pose extraction and orientation control are separate
    # choices: several archived runs intentionally extracted 6-DoF pose while
    # executing translation-only OSC_POSITION actions. The normalizer still
    # selects OSC_POSE by default when pose is enabled; only the explicit
    # legacy combination is preserved here.


def validate_pipeline_config(
    cfg: Dict[str, Any],
    *,
    source: str,
) -> None:
    """Validate the normalized portable configuration."""

    top = _expect_object(cfg, name="", source=source)
    selected_input = _expect_object(
        top.get("input"),
        name="input",
        source=source,
    )
    selected_video = selected_input.get("selected_video")
    if selected_video not in {"rollout", "gen", "custom"}:
        raise ValueError(
            f"Unsupported input.selected_video in {source}: {selected_video}"
        )

    runtime = _expect_object(
        top.get("runtime"),
        name="runtime",
        source=source,
    )
    if runtime.get("device") not in {"cpu", "cuda"}:
        raise ValueError(
            f"Unsupported runtime.device in {source}: {runtime.get('device')}"
        )

    region = _expect_object(
        top.get("region"),
        name="region",
        source=source,
    )
    region_targets = _expect_object(
        region.get("targets"),
        name="region.targets",
        source=source,
    )
    for name, selectors in (
        ("eef", frozenset({"auto", "simulation", "visual"})),
        (
            "obj",
            frozenset({"auto", "simulation", "visual", "off"}),
        ),
    ):
        target = _expect_object(
            region_targets.get(name),
            name=f"region.targets.{name}",
            source=source,
        )
        _validate_visual_target(
            target,
            name=f"region.targets.{name}",
            source=source,
            selectors=selectors,
        )
    region_objects = _expect_list(
        region_targets.get("objects"),
        name="region.targets.objects",
        source=source,
    )
    for index, target_value in enumerate(region_objects):
        name = f"region.targets.objects[{index}]"
        target = _expect_object(
            target_value,
            name=name,
            source=source,
        )
        _validate_visual_target(
            target,
            name=name,
            source=source,
            selectors=frozenset({"auto", "simulation", "visual", "off"}),
            require_manual_bbox=False,
        )

    tracking = _expect_object(
        top.get("tracking"),
        name="tracking",
        source=source,
    )
    tracking_targets = _expect_object(
        tracking.get("targets"),
        name="tracking.targets",
        source=source,
    )
    for name in ("eef", "obj"):
        target = _expect_object(
            tracking_targets.get(name),
            name=f"tracking.targets.{name}",
            source=source,
        )
        if target.get("num_points", 0) <= 0:
            raise ValueError(
                f"tracking.targets.{name}.num_points must be > 0 in {source}"
            )
    track_objects = _expect_list(
        tracking_targets.get("objects"),
        name="tracking.targets.objects",
        source=source,
    )
    for index, target_value in enumerate(track_objects):
        target = _expect_object(
            target_value,
            name=f"tracking.targets.objects[{index}]",
            source=source,
        )
        if target.get("num_points", 0) <= 0:
            raise ValueError(
                f"tracking.targets.objects[{index}].num_points must be > 0 in {source}"
            )

    depth = _expect_object(
        top.get("depth"),
        name="depth",
        source=source,
    )
    artifact_policy = depth.get("artifact_policy")
    if not isinstance(artifact_policy, dict):
        raise ValueError(f"depth.artifact_policy must be an object in {source}")
    model = depth.get("model")
    allowed_depth = set(SUPPORTED_DEPTH_PRESETS)
    if depth.get("config_path"):
        allowed_depth.update(SUPPORTED_DEPTH_BACKENDS)
    if model not in allowed_depth and external_depth_backend_id(model) is None:
        expected = tuple(sorted(allowed_depth))
        raise ValueError(
            f"Unsupported depth.model in {source}: {model}. Expected one of {expected}."
        )
    if depth.get("use_rollout_gt_depth") and not depth.get("rollout_gt_depth_path"):
        raise ValueError(
            "depth.rollout_gt_depth_path is required when "
            f"depth.use_rollout_gt_depth=true in {source}"
        )
    if depth.get("input_size", 0) <= 0:
        raise ValueError(f"depth.input_size must be > 0 in {source}")
    validate_depth_base_config(
        base_config=depth.get("base"),
        source=source,
    )
    validate_target_calibrated_lift_config(
        config=depth.get("target_calibrated_lift"),
        source=source,
    )

    alignment = _expect_object(
        top.get("alignment"),
        name="alignment",
        source=source,
    )
    if alignment.get("method") != "translation":
        raise ValueError(
            f"Unsupported alignment.method in {source}: {alignment.get('method')}"
        )

    geometry = _expect_object(
        top.get("geometry"),
        name="geometry",
        source=source,
    )
    geometry_targets = _expect_object(
        geometry.get("targets"),
        name="geometry.targets",
        source=source,
    )
    for name in ("eef", "obj"):
        target = _expect_object(
            geometry_targets.get(name),
            name=f"geometry.targets.{name}",
            source=source,
        )
        _validate_geometry_entry(
            target,
            name=f"geometry.targets.{name}",
            source=source,
        )
    geometry_objects = _expect_list(
        geometry_targets.get("objects"),
        name="geometry.targets.objects",
        source=source,
    )
    for index, target_value in enumerate(geometry_objects):
        target = _expect_object(
            target_value,
            name=f"geometry.targets.objects[{index}]",
            source=source,
        )
        _validate_geometry_entry(
            target,
            name=f"geometry.targets.objects[{index}]",
            source=source,
        )

    pose = _expect_object(
        top.get("pose"),
        name="pose",
        source=source,
    )
    _validate_pose(pose, source=source)
    gripper = _expect_object(
        top.get("gripper"),
        name="gripper",
        source=source,
    )
    _validate_gripper(gripper, source=source)
    action = _expect_object(
        top.get("action"),
        name="action",
        source=source,
    )
    _validate_action(
        action,
        pose=pose,
        source=source,
    )

    task = _expect_object(
        top.get("task"),
        name="task",
        source=source,
    )
    order = task.get("runtime_stage_order")
    if not isinstance(order, dict):
        raise ValueError(f"task.runtime_stage_order must be an object in {source}")
    mode = order.get("mode")
    if mode not in {"annotated", "coupling"}:
        raise ValueError(
            f"Unsupported task.runtime_stage_order.mode in {source}: {mode}"
        )


def load_pipeline_config(
    config: Optional[Dict[str, Any]] = None,
    *,
    pipeline_config_path: Optional[str] = None,
    source: Optional[str] = None,
) -> Dict[str, Any]:
    """Load one explicit configuration source and return a detached result."""

    if config is not None and pipeline_config_path is not None:
        raise ValueError("provide either config or pipeline_config_path, not both")

    exists_on_disk = False
    if pipeline_config_path is not None:
        path = Path(pipeline_config_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"pipeline config not found: {path}")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        raw = loaded
        origin = source or path.as_posix()
        exists_on_disk = True
    elif config is not None:
        raw = copy.deepcopy(config)
        origin = source or "<explicit mapping>"
    else:
        raw = {}
        origin = source or "<built-in default>"

    normalized = normalize_pipeline_config(raw, source_cfg=raw)
    validate_pipeline_config(normalized, source=origin)
    normalized["_meta"] = {
        "source": origin,
        "exists_on_disk": exists_on_disk,
        "run_key": None,
        "sample_dir": None,
    }
    return normalized


__all__ = [
    "DEFAULT_PIPELINE_CONFIG_FILENAME",
    "SUPPORTED_DEPTH_PRESETS",
    "default_pipeline_config_dict",
    "load_pipeline_config",
    "normalize_pipeline_config",
    "validate_pipeline_config",
]
