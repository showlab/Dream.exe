"""Simulator-owned camera pose, mutation, and depth runtime behavior.

This module only adapts an injected environment's ``sim.model`` and
``sim.data`` interfaces.  Camera calibration, projection, and pixel/depth
lifting remain in the simulator-independent ``dream_exe.video2traj`` domain.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation


def _quat_wxyz_to_xyzw(quaternion: Any) -> np.ndarray:
    value = np.asarray(quaternion, float).reshape(4)
    return np.array(
        [value[1], value[2], value[3], value[0]],
        float,
    )


def _quat_xyzw_to_wxyz(quaternion: Any) -> np.ndarray:
    value = np.asarray(quaternion, float).reshape(4)
    return np.array(
        [value[3], value[0], value[1], value[2]],
        float,
    )


def camera_world_pose_from_sim(
    env: Any,
    camera_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Read a camera's resolved world pose, with current model fallback."""

    if hasattr(env.sim.data, "cam_xpos") and hasattr(env.sim.data, "cam_xmat"):
        position_world = np.asarray(
            env.sim.data.cam_xpos[camera_id],
            dtype=float,
        ).reshape(3)
        rotation_camera_to_world = np.asarray(
            env.sim.data.cam_xmat[camera_id],
            dtype=float,
        ).reshape(3, 3)
        quaternion_wxyz = _quat_xyzw_to_wxyz(
            Rotation.from_matrix(rotation_camera_to_world).as_quat()
        )
        return (
            position_world,
            np.asarray(quaternion_wxyz, dtype=float).reshape(4),
        )

    return (
        np.asarray(
            env.sim.model.cam_pos[camera_id],
            dtype=float,
        ).reshape(3),
        np.asarray(
            env.sim.model.cam_quat[camera_id],
            dtype=float,
        ).reshape(4),
    )


def camera_parent_body_name(
    env: Any,
    camera_id: int,
) -> str | None:
    model = env.sim.model
    camera_body_ids = getattr(model, "cam_bodyid", None)
    if camera_body_ids is None:
        return None

    body_id = int(camera_body_ids[camera_id])
    if body_id < 0:
        return None

    if hasattr(model, "body_id2name"):
        body_name = model.body_id2name(body_id)
    else:
        body_names = list(getattr(model, "body_names", []))
        body_name = body_names[body_id] if 0 <= body_id < len(body_names) else None

    if not body_name:
        return None
    body_name = str(body_name)
    if body_name.lower() in {"world", "worldbody"}:
        return None
    return body_name


def build_camera_raw(
    env: Any,
    camera_name: str,
    frame_size: tuple[int, int],
) -> dict[str, Any]:
    """Build the current raw simulator-camera artifact payload."""

    width, height = frame_size
    camera_id = env.sim.model.camera_name2id(camera_name)
    position_world, quaternion_world = camera_world_pose_from_sim(
        env,
        camera_id,
    )
    camera_config: dict[str, Any] = {}
    environment_camera_configs = getattr(env, "_cam_configs", None)
    if isinstance(environment_camera_configs, dict):
        camera_config = dict(environment_camera_configs.get(camera_name, {}) or {})

    parent_body_name = camera_parent_body_name(env, camera_id) or camera_config.get(
        "parent_body"
    )
    local_position = np.asarray(
        env.sim.model.cam_pos[camera_id],
        dtype=float,
    ).reshape(3)
    local_quaternion = np.asarray(
        env.sim.model.cam_quat[camera_id],
        dtype=float,
    ).reshape(4)
    return {
        "name": camera_name,
        "width": int(width),
        "height": int(height),
        "fovy_deg": float(env.sim.model.cam_fovy[camera_id]),
        "pos_w": position_world.astype(float).tolist(),
        "quat_wxyz": quaternion_world.astype(float).tolist(),
        "parent_body_name": parent_body_name,
        "local_pos": local_position.astype(float).tolist(),
        "local_quat_wxyz": local_quaternion.astype(float).tolist(),
        "camera_attribs": dict(camera_config.get("camera_attribs", {}) or {}),
        "pose_kind": "attached" if parent_body_name else "world",
    }


def adjust_camera(
    env: Any,
    camera_name: str,
    translation: Any,
    rotation: Mapping[str, float],
    mode: str = "body",
) -> None:
    """Apply the current local-position and body/world rotation update."""

    camera_id = env.sim.model.camera_name2id(camera_name)
    env.sim.model.cam_pos[camera_id] = env.sim.model.cam_pos[camera_id] + np.asarray(
        translation, float
    )
    default_quaternion_wxyz = env.sim.model.cam_quat[camera_id].copy()
    default_rotation = Rotation.from_quat(_quat_wxyz_to_xyzw(default_quaternion_wxyz))
    yaw = rotation["yaw"]
    pitch = rotation["pitch"]
    roll = rotation["roll"]
    if mode == "body":
        delta_rotation = Rotation.from_euler(
            "zyx",
            [roll, yaw, pitch],
            degrees=True,
        )
        adjusted_rotation = default_rotation * delta_rotation
    else:
        delta_rotation = Rotation.from_euler(
            "ZYX",
            [yaw, pitch, roll],
            degrees=True,
        )
        adjusted_rotation = delta_rotation * default_rotation
    env.sim.model.cam_quat[camera_id] = _quat_xyzw_to_wxyz(adjusted_rotation.as_quat())
    env.sim.forward()


def get_camera_depth(env: Any, camera_name: str) -> Any:
    """Render depth with the current environment camera dimensions."""

    depth = env.sim.render(
        camera_name=camera_name,
        width=env.camera_widths[0],
        height=env.camera_heights[0],
        depth=True,
    )
    if isinstance(depth, tuple):
        depth = depth[1]
    return depth


__all__ = [
    "adjust_camera",
    "build_camera_raw",
    "camera_parent_body_name",
    "camera_world_pose_from_sim",
    "get_camera_depth",
]
