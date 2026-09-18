"""Small rigid-transform primitives shared by algorithm and runtime domains."""

from __future__ import annotations

from typing import Tuple

import numpy as np


Array = np.ndarray


def world_to_base_point(
    R_wb: Array,
    t_wb: Array,
    p_w: Array,
) -> np.ndarray:
    rotation = np.asarray(R_wb, float).reshape(3, 3)
    translation = np.asarray(t_wb, float).reshape(3)
    point = np.asarray(p_w, float).reshape(3)
    return rotation.T @ (point - translation)


def world_to_base_pose(
    R_wb: Array,
    t_wb: Array,
    R_wt: Array,
    p_wt: Array,
) -> Tuple[np.ndarray, np.ndarray]:
    rotation_world_base = np.asarray(R_wb, float).reshape(3, 3)
    translation_world_base = np.asarray(t_wb, float).reshape(3)
    rotation_world_target = np.asarray(R_wt, float).reshape(3, 3)
    point_world_target = np.asarray(p_wt, float).reshape(3)
    rotation_base_target = rotation_world_base.T @ rotation_world_target
    point_base_target = world_to_base_point(
        rotation_world_base,
        translation_world_base,
        point_world_target,
    )
    return rotation_base_target, point_base_target


def world_to_base_R(
    R_wb: np.ndarray,
    R_w: np.ndarray,
) -> np.ndarray:
    return R_wb.T @ R_w


def parse_X_wb(
    X_wb_dict: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    if X_wb_dict is None:
        raise ValueError("X_wb is None")
    rotation_world_base = np.asarray(
        X_wb_dict["R"],
        dtype=np.float64,
    ).reshape(3, 3)
    translation_world_base = np.asarray(
        X_wb_dict["t"],
        dtype=np.float64,
    ).reshape(3)
    return rotation_world_base, translation_world_base


__all__ = [
    "parse_X_wb",
    "world_to_base_R",
    "world_to_base_point",
    "world_to_base_pose",
]
