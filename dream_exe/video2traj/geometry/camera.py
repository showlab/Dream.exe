"""Independent camera calibration and projection primitives."""

# ruff: noqa: RUF046, UP006, UP007, UP035, UP045

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import numpy as np
from scipy.spatial.transform import Rotation

NumberUV = Tuple[Union[int, float], Union[int, float]]


@dataclass(frozen=True)
class CameraIntrinsics:
    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int

    def as_K(self) -> np.ndarray:
        return np.asarray(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=float,
        )


@dataclass(frozen=True)
class CameraExtrinsics:
    R_c2w: np.ndarray
    t_c2w: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "R_c2w",
            np.asarray(self.R_c2w, dtype=float).reshape(3, 3),
        )
        object.__setattr__(
            self,
            "t_c2w",
            np.asarray(self.t_c2w, dtype=float).reshape(3),
        )

    @property
    def R_w2c(self) -> np.ndarray:
        return self.R_c2w.T

    @property
    def t_w2c(self) -> np.ndarray:
        return -self.R_w2c @ self.t_c2w


@dataclass(frozen=True)
class Camera:
    K: CameraIntrinsics
    E: CameraExtrinsics

    @property
    def fx(self) -> float:
        return self.K.fx

    @property
    def fy(self) -> float:
        return self.K.fy

    @property
    def cx(self) -> float:
        return self.K.cx

    @property
    def cy(self) -> float:
        return self.K.cy

    @property
    def width(self) -> int:
        return self.K.width

    @property
    def height(self) -> int:
        return self.K.height

    def cam_to_world(self, p_cam: np.ndarray) -> np.ndarray:
        point = np.asarray(p_cam, dtype=float).reshape(3)
        return self.E.R_c2w @ point + self.E.t_c2w

    def world_to_cam(self, p_world: np.ndarray) -> np.ndarray:
        point = np.asarray(p_world, dtype=float).reshape(3)
        return self.E.R_w2c @ point + self.E.t_w2c

    def cam_to_pixel(
        self,
        p_cam: np.ndarray,
        clip_inside: bool = False,
        return_float: bool = False,
    ) -> Optional[NumberUV]:
        point = np.asarray(p_cam, dtype=float).reshape(3)
        if not np.isfinite(point).all() or point[2] <= 1e-9:
            return None
        u = self.fx * (point[0] / point[2]) + self.cx
        v = -self.fy * (point[1] / point[2]) + self.cy
        if not np.isfinite(u) or not np.isfinite(v):
            return None
        if clip_inside and not (0.0 <= u < self.width and 0.0 <= v < self.height):
            return None
        if return_float:
            return float(u), float(v)
        return int(round(float(u))), int(round(float(v)))

    def world_to_pixel(
        self,
        p_world: np.ndarray,
        clip_inside: bool = False,
        return_float: bool = False,
    ) -> Optional[NumberUV]:
        return self.cam_to_pixel(
            self.world_to_cam(p_world),
            clip_inside=clip_inside,
            return_float=return_float,
        )

    @staticmethod
    def _sample_depth_bilinear(
        depth_map: np.ndarray,
        u: float,
        v: float,
    ) -> float:
        height, width = depth_map.shape
        left = int(np.floor(u))
        top = int(np.floor(v))
        left = int(np.clip(left, 0, width - 2))
        top = int(np.clip(top, 0, height - 2))
        right = left + 1
        bottom = top + 1
        du = float(u) - left
        dv = float(v) - top
        return float(
            (1.0 - du) * (1.0 - dv) * depth_map[top, left]
            + du * (1.0 - dv) * depth_map[top, right]
            + (1.0 - du) * dv * depth_map[bottom, left]
            + du * dv * depth_map[bottom, right]
        )

    def pixel_to_cam(
        self,
        u: float,
        v: float,
        depth_map: np.ndarray,
        bilinear: bool = False,
    ) -> np.ndarray:
        depths = np.asarray(depth_map)
        height, width = depths.shape
        if bilinear:
            depth = self._sample_depth_bilinear(depths, u, v)
        else:
            pixel_u = int(np.clip(round(u), 0, width - 1))
            pixel_v = int(np.clip(round(v), 0, height - 1))
            depth = float(depths[pixel_v, pixel_u])
        return np.asarray(
            [
                (float(u) - self.cx) / self.fx * depth,
                -(float(v) - self.cy) / self.fy * depth,
                depth,
            ],
            dtype=float,
        )

    def pixel_to_world(
        self,
        u: float,
        v: float,
        depth_map: np.ndarray,
        bilinear: bool = False,
    ) -> np.ndarray:
        return self.cam_to_world(self.pixel_to_cam(u, v, depth_map, bilinear=bilinear))


def _rotation_from_quaternion_wxyz(quaternion: Any) -> np.ndarray:
    values = np.asarray(quaternion, dtype=float).reshape(4)
    return Rotation.from_quat([values[1], values[2], values[3], values[0]]).as_matrix()


def camera_from_raw(cam_raw: Dict[str, Any]) -> Camera:
    width = int(cam_raw["width"])
    height = int(cam_raw["height"])
    fovy_deg = float(cam_raw["fovy_deg"])
    focal = 0.5 * height / np.tan(np.deg2rad(fovy_deg) / 2.0)
    intrinsics = CameraIntrinsics(
        fx=float(focal),
        fy=float(focal),
        cx=width / 2.0,
        cy=height / 2.0,
        width=width,
        height=height,
    )
    raw_axes_to_normalized = np.diag([1.0, -1.0, -1.0])
    rotation = (
        _rotation_from_quaternion_wxyz(cam_raw["quat_wxyz"]) @ raw_axes_to_normalized
    )
    translation = np.asarray(cam_raw["pos_w"], dtype=float).reshape(3)
    return Camera(
        intrinsics,
        CameraExtrinsics(rotation, translation),
    )


def compute_camera_derived(
    cam_raw: Dict[str, Any],
    io_vflip_saved: bool = True,
    io_color_space: str = "BGR",
) -> Dict[str, Any]:
    camera = camera_from_raw(cam_raw)
    projection = camera.K.as_K() @ np.column_stack((camera.E.R_w2c, camera.E.t_w2c))
    fovx_deg = float(
        np.rad2deg(
            2.0
            * np.arctan2(
                camera.width,
                2.0 * camera.fx,
            )
        )
    )
    fovy_deg = float(
        np.rad2deg(
            2.0
            * np.arctan2(
                camera.height,
                2.0 * camera.fy,
            )
        )
    )
    return {
        "K": {
            "fx": camera.fx,
            "fy": camera.fy,
            "cx": camera.cx,
            "cy": camera.cy,
            "width": camera.width,
            "height": camera.height,
        },
        "E": {
            "R_c2w": camera.E.R_c2w.tolist(),
            "t_c2w": camera.E.t_c2w.tolist(),
        },
        "P": projection.tolist(),
        "fovx_deg": fovx_deg,
        "fovy_deg": fovy_deg,
        "io": {
            "vflip_saved": bool(io_vflip_saved),
            "color_space": str(io_color_space),
        },
        "conventions": {
            "camera_R_meaning": "camera_to_world",
            "forward_axis": "+Z",
            "quat_format_raw": "wxyz (MuJoCo)",
            "projection_formula": ("p_cam = R^T (p_world - t),  v = -fy*Y/Z + cy"),
            "S_cam_applied_to_vectors": [1.0, -1.0, -1.0],
            "normalized_from": "-Z (MuJoCo default)",
        },
    }


def adapt_camera_for_io(
    cam: Camera,
    vflip_saved: bool,
) -> Camera:
    intrinsics = cam.K
    fy = -intrinsics.fy if vflip_saved else intrinsics.fy
    cy = (intrinsics.height - 1.0) - intrinsics.cy if vflip_saved else intrinsics.cy
    return Camera(
        CameraIntrinsics(
            fx=intrinsics.fx,
            fy=fy,
            cx=intrinsics.cx,
            cy=cy,
            width=intrinsics.width,
            height=intrinsics.height,
        ),
        cam.E,
    )


def load_camera_from_bundle(
    bundle: Dict[str, Any],
) -> Tuple[Camera, bool]:
    derived = bundle.get("derived", {}).get("camera")
    if derived is not None:
        conventions = derived.get("conventions", {})
        forward_axis = conventions.get("forward_axis", "+Z")
        if forward_axis != "+Z":
            raise ValueError(
                "derived.camera.conventions.forward_axis != '+Z'. "
                "Please export derived in CONFIG FILE first."
            )
        values = derived["K"]
        intrinsics = CameraIntrinsics(
            fx=float(values["fx"]),
            fy=float(values["fy"]),
            cx=float(values["cx"]),
            cy=float(values["cy"]),
            width=int(values["width"]),
            height=int(values["height"]),
        )
        extrinsics_values = derived["E"]
        extrinsics = CameraExtrinsics(
            extrinsics_values["R_c2w"],
            extrinsics_values["t_c2w"],
        )
        vflip_saved = bool(derived.get("io", {}).get("vflip_saved", True))
        return Camera(intrinsics, extrinsics), vflip_saved

    return camera_from_raw(bundle.get("raw", {})["camera"]), True


def make_camera_for_frames(
    bundle: Dict[str, Any],
    frame_w: int,
    frame_h: int,
    strict: bool = True,
) -> Camera:
    del strict
    camera, vflip_saved = load_camera_from_bundle(bundle)
    camera = adapt_camera_for_io(camera, vflip_saved)
    width = int(frame_w)
    height = int(frame_h)
    scale_x = width / camera.width
    scale_y = height / camera.height
    fy = camera.fy * scale_y
    cy = camera.cy * scale_y
    intrinsics = CameraIntrinsics(
        fx=camera.fx * scale_x,
        fy=fy,
        cx=camera.cx * scale_x,
        cy=cy,
        width=width,
        height=height,
    )
    return Camera(intrinsics, camera.E)


__all__ = [
    "Camera",
    "CameraExtrinsics",
    "CameraIntrinsics",
    "adapt_camera_for_io",
    "camera_from_raw",
    "compute_camera_derived",
    "load_camera_from_bundle",
    "make_camera_for_frames",
]
