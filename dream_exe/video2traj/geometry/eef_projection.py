"""Pure visual-center to TCP/controller trajectory projection.

This module preserves the current translation-only alignment and fixed
world-space TCP-to-controller offset. Simulator site queries and later pose
orientation fusion remain outside this core stage.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from ...transforms import (
    parse_X_wb,
    world_to_base_R,
    world_to_base_point,
)
from .camera import Camera


def _get_nested(
    value: Dict[str, Any],
    path: str,
    default: Any = None,
) -> Any:
    current: Any = value
    for key in path.split("."):
        if not isinstance(current, dict) or key not in current:
            return default
        current = current[key]
    return current


def _rotation_from_wxyz(quaternion_wxyz: Any) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, float).reshape(4)
    quaternion_xyzw = np.array(
        [
            quaternion[1],
            quaternion[2],
            quaternion[3],
            quaternion[0],
        ],
        float,
    )
    return Rotation.from_quat(quaternion_xyzw).as_matrix()


def _quaternion_wxyz_from_rotation(
    rotation: np.ndarray,
) -> np.ndarray:
    quaternion_xyzw = Rotation.from_matrix(
        np.asarray(rotation, dtype=float).reshape(3, 3)
    ).as_quat()
    return np.asarray(
        [
            quaternion_xyzw[3],
            quaternion_xyzw[0],
            quaternion_xyzw[1],
            quaternion_xyzw[2],
        ],
        dtype=float,
    )


def _pose_from_pose_dict(
    pose_dict: Any,
) -> Optional[Tuple[np.ndarray, Optional[np.ndarray]]]:
    if not isinstance(pose_dict, dict):
        return None
    position = pose_dict.get("pos", None)
    if position is None:
        return None
    position_array = np.asarray(position, float).reshape(3)

    rotation = None
    if pose_dict.get("R", None) is not None:
        rotation = np.asarray(pose_dict["R"], float).reshape(3, 3)
    elif pose_dict.get("quat_wxyz", None) is not None:
        rotation = _rotation_from_wxyz(pose_dict["quat_wxyz"])
    return position_array, rotation


def _pose_from_bundle(
    config: Dict[str, Any],
    key: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    if key == "tip":
        key = "tcp"
    if key == "controller":
        key = "controller_ref"

    if key == "tcp":
        candidate = _pose_from_pose_dict(
            _get_nested(config, "derived.eef.tcp_world", None)
        )
        if candidate is None:
            candidate = _pose_from_pose_dict(
                _get_nested(config, "raw.eef.tcp_world", None)
            )
        if candidate is not None:
            return candidate

    if key == "controller_ref":
        candidate = _pose_from_pose_dict(
            _get_nested(
                config,
                "derived.eef.controller_ref_world",
                None,
            )
        )
        if candidate is None:
            candidate = _pose_from_pose_dict(
                _get_nested(
                    config,
                    "raw.eef.controller_ref_world",
                    None,
                )
            )
        if candidate is not None:
            return candidate

    legacy_eef = _get_nested(config, "raw.ee_world", None)
    if isinstance(legacy_eef, dict):
        if key == "tcp" and "tip" in legacy_eef:
            tip = legacy_eef["tip"]
            position = np.asarray(tip["pos"], float).reshape(3)
            rotation = (
                _rotation_from_wxyz(tip["quat_wxyz"])
                if tip.get("quat_wxyz", None) is not None
                else None
            )
            return position, rotation
        if key == "controller_ref" and "controller" in legacy_eef:
            controller = legacy_eef["controller"]
            position = np.asarray(
                controller["pos"],
                float,
            ).reshape(3)
            rotation = (
                _rotation_from_wxyz(controller["quat_wxyz"])
                if controller.get("quat_wxyz", None) is not None
                else None
            )
            return position, rotation

    raise KeyError(f"pose_from_bundle: cannot find pose for key='{key}' in cfg.")


def pose_from_bundle(
    config: Dict[str, Any],
    key: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Read one saved EEF pose without querying a simulator."""

    return _pose_from_bundle(config, key)


def _tcp_site_name(config: Dict[str, Any]) -> Optional[str]:
    value = _get_nested(config, "raw.eef.tcp_site_name", None)
    if value:
        return str(value)
    value = _get_nested(
        config,
        "derived.eef.robot_eef_site_name",
        None,
    )
    return str(value) if value else None


def _controller_site_name(config: Dict[str, Any]) -> Optional[str]:
    value = _get_nested(
        config,
        "raw.eef.controller_ref_site_name",
        None,
    )
    if value:
        return str(value)
    value = _get_nested(
        config,
        "derived.eef.controller_ref_site_name",
        None,
    )
    if value:
        return str(value)
    return _tcp_site_name(config)


def _same_tcp_and_controller(config: Dict[str, Any]) -> bool:
    tcp_name = _tcp_site_name(config) or ""
    controller_name = _controller_site_name(config) or ""
    return tcp_name != "" and tcp_name == controller_name


def _tcp_to_controller_delta_world(
    config: Dict[str, Any],
) -> Optional[np.ndarray]:
    try:
        tcp_position, _ = _pose_from_bundle(config, "tcp")
        controller_position, _ = _pose_from_bundle(
            config,
            "controller_ref",
        )
        return (controller_position - tcp_position).reshape(3)
    except Exception:
        return None


def _tcp_to_controller_rotation(
    config: Dict[str, Any],
) -> Optional[np.ndarray]:
    transform = _get_nested(
        config,
        "derived.eef.X_tcp_to_controller_ref_world",
        None,
    )
    if isinstance(transform, dict) and "R" in transform and "t" in transform:
        return np.asarray(
            transform["R"],
            dtype=float,
        ).reshape(3, 3)

    try:
        tcp_position, tcp_rotation = _pose_from_bundle(
            config,
            "tcp",
        )
        controller_position, controller_rotation = _pose_from_bundle(
            config,
            "controller_ref",
        )
        del tcp_position, controller_position
        if tcp_rotation is None or controller_rotation is None:
            return None
        return np.asarray(tcp_rotation, dtype=float).reshape(3, 3).T @ np.asarray(
            controller_rotation,
            dtype=float,
        ).reshape(3, 3)
    except Exception:
        return None


def _position_in_base(
    config: Dict[str, Any],
    position_world: Optional[np.ndarray],
) -> Optional[np.ndarray]:
    if position_world is None:
        return None
    transform = _get_nested(config, "derived.eef.X_wb", None)
    if transform is None:
        return None
    rotation_world_base, translation_world_base = parse_X_wb(transform)
    return world_to_base_point(
        rotation_world_base,
        translation_world_base,
        position_world,
    )


def _smooth_positions_ema(
    trajectory: List[Dict[str, Any]],
    key: str = "pos_world",
    alpha: float = 0.2,
) -> List[Dict[str, Any]]:
    previous = None
    for frame in trajectory:
        position = frame.get(key, None)
        if position is None:
            continue
        position_array = np.asarray(position, float)
        previous = (
            position_array
            if previous is None
            else (1 - alpha) * previous + alpha * position_array
        )
        frame[key] = previous.tolist()
    return trajectory


def align_visual_centers_by_translation(
    center_trajectory: List[Dict[str, Any]],
    initial_tcp_world: np.ndarray,
    seed_count: int = 5,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Align centers so their initial robust seed equals the initial TCP."""

    aligned = [dict(frame) for frame in center_trajectory]
    tcp_position = np.asarray(initial_tcp_world, float).reshape(3)

    seeds = []
    for frame in aligned:
        center = frame.get("vis_center_world")
        if center and None not in center:
            seeds.append(np.asarray(center, float))
            if len(seeds) >= seed_count:
                break
    if not seeds:
        return aligned, {
            "ok": False,
            "reason": "no valid center",
        }

    seed = np.median(np.stack(seeds, axis=0), axis=0)
    offset = tcp_position - seed
    for frame in aligned:
        center = frame.get("vis_center_world")
        if not center or None in center:
            continue
        position = np.asarray(center, float)
        frame["vis_center_world"] = (position + offset).tolist()

    return aligned, {
        "ok": True,
        "offset_world": offset.tolist(),
        "seed_center_world": seed.tolist(),
        "seed_k": len(seeds),
    }


def visual_centers_to_tcp(
    visual_centers: List[Dict[str, Any]],
    config: Dict[str, Any],
    camera: Camera,
    method: str = "translation",
    smooth_alpha: float = -1,
) -> List[Dict[str, Any]]:
    """Map visual centers to the current TCP position record schema."""

    initial_tcp_world, _ = _pose_from_bundle(config, "tcp")
    if method != "translation":
        method = "translation"

    aligned, alignment = align_visual_centers_by_translation(
        visual_centers,
        initial_tcp_world,
        seed_count=5,
    )
    if not alignment.get("ok", False):
        return [
            {
                "frame": frame["frame"],
                "pos_world": None,
                "pos_uv": None,
                "pos_base": None,
            }
            for frame in visual_centers
        ]

    output: List[Dict[str, Any]] = []
    for frame in aligned:
        center_world = frame.get("vis_center_world")
        if center_world is None or None in center_world:
            output.append(
                {
                    "frame": frame["frame"],
                    "pos_world": None,
                    "pos_uv": None,
                    "pos_base": None,
                }
            )
            continue

        position_world = np.asarray(center_world, float).reshape(3)
        pixel = camera.world_to_pixel(
            position_world,
            clip_inside=False,
            return_float=True,
        )
        position_base = _position_in_base(config, position_world)
        output.append(
            {
                "frame": int(frame["frame"]),
                "pos_world": position_world.tolist(),
                "pos_uv": list(pixel) if pixel is not None else None,
                "pos_base": (
                    position_base.tolist() if position_base is not None else None
                ),
            }
        )

    return (
        output
        if smooth_alpha <= 0
        else _smooth_positions_ema(
            output,
            key="pos_world",
            alpha=smooth_alpha,
        )
    )


def tcp_to_controller(
    tcp_trajectory: List[Dict[str, Any]],
    config: Dict[str, Any],
    camera: Camera,
) -> List[Dict[str, Any]]:
    """Apply the current fixed world-space TCP-to-controller offset."""

    delta_world = _tcp_to_controller_delta_world(config)
    output: List[Dict[str, Any]] = []

    for record in tcp_trajectory:
        position_world = record.get("pos_world", None)
        if position_world is None:
            output.append(
                {
                    "frame": record["frame"],
                    "pos_world": None,
                    "pos_uv": None,
                    "pos_base": None,
                }
            )
            continue

        tcp_position = np.asarray(position_world, float).reshape(3)
        controller_position = (
            None
            if delta_world is None
            else tcp_position + np.asarray(delta_world, float).reshape(3)
        )
        if controller_position is None:
            output.append(
                {
                    "frame": record["frame"],
                    "pos_world": None,
                    "pos_uv": None,
                    "pos_base": None,
                }
            )
            continue

        pixel = camera.world_to_pixel(
            controller_position,
            clip_inside=False,
            return_float=True,
        )
        position_base = _position_in_base(
            config,
            controller_position,
        )
        output.append(
            {
                "frame": int(record["frame"]),
                "pos_world": controller_position.tolist(),
                "pos_uv": list(pixel) if pixel is not None else None,
                "pos_base": (
                    position_base.tolist() if position_base is not None else None
                ),
            }
        )
    return output


def project_eef_trajectories(
    visual_centers: List[Dict[str, Any]],
    config: Dict[str, Any],
    camera: Camera,
    *,
    method: str = "translation",
    smooth_alpha: float = -1,
) -> Dict[str, Any]:
    """Compose the current visual-center, TCP, and controller projection."""

    normalized_method = str(method or "translation")
    normalized_smooth_alpha = float(smooth_alpha)
    tcp_trajectory = visual_centers_to_tcp(
        visual_centers,
        config,
        camera,
        method=normalized_method,
        smooth_alpha=normalized_smooth_alpha,
    )
    same = bool(_same_tcp_and_controller(config))
    controller_trajectory = (
        tcp_trajectory
        if same
        else tcp_to_controller(
            tcp_trajectory,
            config,
            camera,
        )
    )
    return {
        "tcp": tcp_trajectory,
        "controller": controller_trajectory,
        "same_tcp_and_controller": same,
    }


def pack_visual_center_records(
    visual_centers: List[Dict[str, Any]],
    *,
    has_X_wb: bool,
    rotation_world_base: np.ndarray | None = None,
    translation_world_base: np.ndarray | None = None,
    visibility: np.ndarray | None = None,
) -> List[Dict[str, Any]]:
    """Pack the current per-frame visual-center output schema."""

    output: List[Dict[str, Any]] = []
    for index, center in enumerate(visual_centers):
        visibility_value = None if visibility is None else float(visibility[index])
        center_world = center.get("vis_center_world", None)
        center_uv = center.get("vis_center_uv", None)
        if center_world and None not in center_world:
            position_world = np.asarray(
                center_world,
                float,
            ).reshape(3)
            position_base = (
                world_to_base_point(
                    rotation_world_base,
                    translation_world_base,
                    position_world,
                ).tolist()
                if has_X_wb
                else None
            )
            output.append(
                {
                    "frame": int(center["frame"]),
                    "pos_world": position_world.tolist(),
                    "pos_uv": (list(center_uv) if center_uv is not None else None),
                    "pos_base": position_base,
                    "vis": visibility_value,
                }
            )
        else:
            output.append(
                {
                    "frame": int(center["frame"]),
                    "pos_world": None,
                    "pos_uv": None,
                    "pos_base": None,
                    "vis": None,
                }
            )
    return output


def mean_visibility_over_points(
    point_visibility: np.ndarray | None,
) -> np.ndarray | None:
    """Reduce current ``[T,N]`` point visibility to one score per frame."""

    if point_visibility is None:
        return None
    values = np.asarray(
        point_visibility,
        dtype=np.float32,
    )
    if values.ndim != 2:
        return None
    return np.clip(
        np.nanmean(values, axis=1),
        0.0,
        1.0,
    ).astype(np.float32)


def fuse_controller_orientation(
    *,
    position_trajectory: List[Dict[str, Any]],
    pose_records: List[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    """Attach current pose sidecar fields to controller positions by frame."""

    pose_by_frame = {
        int(record["frame"]): record
        for record in pose_records
        if isinstance(record, dict) and record.get("frame", None) is not None
    }
    output: List[Dict[str, Any]] = []
    for entry in position_trajectory:
        frame = int(entry.get("frame", len(output)))
        fused = dict(entry)
        pose_record = pose_by_frame.get(frame, None)
        if pose_record is not None:
            fused["R"] = pose_record.get("R_world", None)
            fused["quat_wxyz"] = pose_record.get(
                "quat_wxyz",
                None,
            )
            fused["R_base"] = pose_record.get("R_base", None)
            fused["quat_wxyz_base"] = pose_record.get(
                "quat_wxyz_base",
                None,
            )
            fused["pose_valid"] = bool(pose_record.get("pose_valid", False))
            fused["pose_source"] = str(
                pose_record.get(
                    "pose_source",
                    "unknown",
                )
            )
            for key in (
                "candidate_source",
                "orientation_frozen",
                "pose_correction_applied",
                "pose_correction_side",
                "position_delta_m",
                "angle_from_prev_raw_rad",
                "angle_from_prev_applied_rad",
                "num_correspondences",
                "num_inliers",
                "residual_mean_m",
                "residual_median_m",
                "shape_ratio",
                "pose_quality",
                "rejection_reason",
            ):
                if key in pose_record:
                    fused[key] = pose_record.get(key)
        output.append(fused)
    return output


def derive_tcp_orientation_from_controller(
    *,
    controller_trajectory: List[Dict[str, Any]],
    config: Dict[str, Any],
    has_X_wb: bool,
    rotation_world_base: np.ndarray | None = None,
) -> List[Dict[str, Any]]:
    """Derive current TCP orientations from controller orientations."""

    rotation_tcp_controller = _tcp_to_controller_rotation(config)
    if rotation_tcp_controller is None:
        raise RuntimeError(
            "EEF controller pose is available, but cfg does not "
            "provide X_tcp_to_controller_ref_world. Cannot derive "
            "eef_tcp orientation from controller orientation."
        )

    output: List[Dict[str, Any]] = []
    for entry in controller_trajectory:
        fused = dict(entry)
        controller_rotation = entry.get("R", None)
        if controller_rotation is None:
            output.append(fused)
            continue
        controller_rotation = np.asarray(
            controller_rotation,
            dtype=np.float64,
        ).reshape(3, 3)
        tcp_rotation = controller_rotation @ rotation_tcp_controller.T
        fused["R"] = tcp_rotation.tolist()
        fused["quat_wxyz"] = _quaternion_wxyz_from_rotation(tcp_rotation).tolist()
        if has_X_wb and rotation_world_base is not None:
            tcp_rotation_base = world_to_base_R(
                rotation_world_base,
                tcp_rotation,
            )
            fused["R_base"] = tcp_rotation_base.tolist()
            fused["quat_wxyz_base"] = _quaternion_wxyz_from_rotation(
                tcp_rotation_base
            ).tolist()
        output.append(fused)
    return output


__all__ = [
    "align_visual_centers_by_translation",
    "derive_tcp_orientation_from_controller",
    "fuse_controller_orientation",
    "mean_visibility_over_points",
    "pack_visual_center_records",
    "pose_from_bundle",
    "project_eef_trajectories",
    "tcp_to_controller",
    "visual_centers_to_tcp",
]
