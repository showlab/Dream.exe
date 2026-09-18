"""Deterministic primitives for the :mod:`video2traj.region` package.

The functions in this module operate only on caller-provided arrays and
configuration values.  Model-backed detection, segmentation, simulator masks,
and artifact publication belong to adapters outside this module.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np

from .contract import RegionPointSample


def normalize_region_prompt(prompt: str | None) -> str | None:
    """Normalize a detector prompt using the current punctuation contract."""

    normalized = None if prompt is None else str(prompt).strip()
    if not normalized:
        return None
    if not normalized.endswith("."):
        normalized += "."
    return normalized


def clip_bbox_to_frame(
    bbox_xyxy: Sequence[int | float],
    width: int,
    height: int,
) -> list[int]:
    if bbox_xyxy is None or len(bbox_xyxy) != 4:
        raise ValueError(f"Expected bbox [x0,y0,x1,y1], got {bbox_xyxy}")

    coordinates = [int(round(float(component))) for component in bbox_xyxy]
    left, right = sorted((coordinates[0], coordinates[2]))
    top, bottom = sorted((coordinates[1], coordinates[3]))
    last_x = max(0, int(width) - 1)
    last_y = max(0, int(height) - 1)

    def bounded(value: int, upper: int) -> int:
        return max(0, min(value, upper))

    return [
        bounded(left, last_x),
        bounded(top, last_y),
        bounded(right, last_x),
        bounded(bottom, last_y),
    ]


def bbox_to_mask(
    shape_hw: tuple[int, int],
    bbox_xyxy: Sequence[int | float],
) -> np.ndarray:
    """Convert an inclusive box to a boolean image mask."""

    height, width = shape_hw
    x0, y0, x1, y1 = clip_bbox_to_frame(
        bbox_xyxy,
        width,
        height,
    )
    mask = np.zeros((height, width), dtype=bool)
    mask[y0 : y1 + 1, x0 : x1 + 1] = True
    return mask


def bbox_from_mask(mask: np.ndarray) -> list[int]:
    rows, columns = np.where(mask)
    if columns.size == 0:
        raise RuntimeError("Cannot compute bbox from an empty mask.")
    return [
        int(columns.min()),
        int(rows.min()),
        int(columns.max()),
        int(rows.max()),
    ]


def erode_mask(mask: np.ndarray, kernel_size: int) -> np.ndarray:
    """Apply the current single-pass square OpenCV erosion lazily."""

    normalized = np.asarray(mask, dtype=bool)
    if kernel_size <= 1:
        return normalized.copy()

    # OpenCV is a model/vision optional dependency.  Keeping the import local
    # lets the numerical core and its public package import without it.
    import cv2

    kernel = np.ones(
        (int(kernel_size), int(kernel_size)),
        np.uint8,
    )
    eroded = cv2.erode(
        normalized.astype(np.uint8),
        kernel,
        iterations=1,
    )
    return eroded.astype(bool)


def farthest_point_indices(
    points: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Select indices through the built-in mask-FPS implementation."""

    from .backends.sampling import (
        farthest_point_indices as _backend_farthest_point_indices,
    )

    return _backend_farthest_point_indices(points, count, rng)


def sample_bbox_points(
    *,
    frame_shape_hw: tuple[int, int],
    bbox_xyxy: Sequence[int],
    num_points: int,
    seed: int = 42,
) -> RegionPointSample:
    """Sample bbox queries through the built-in Gaussian backend."""

    from .backends.sampling import BBoxGaussianQuerySampler
    from .contract import RegionSamplingRequest

    return (
        BBoxGaussianQuerySampler()
        .sample(
            RegionSamplingRequest(
                frame_shape_hw=frame_shape_hw,
                bbox_xyxy=bbox_xyxy,
                num_points=num_points,
                seed=seed,
            )
        )
        .sample
    )


def sample_mask_points(
    *,
    mask: np.ndarray,
    num_points: int,
    init_depth: np.ndarray | None = None,
    camera: Any | None = None,
    sampling_mode: str = "3d_fps",
    seed: int = 42,
) -> RegionPointSample:
    """Sample mask queries through the built-in farthest-point backend."""

    from .backends.sampling import MaskFarthestPointQuerySampler
    from .contract import RegionSamplingRequest

    normalized_mask = np.asarray(mask, dtype=bool)
    normalized_mode = str(sampling_mode or "3d_fps").strip().lower()
    if normalized_mode not in {"2d_fps", "3d_fps"}:
        raise ValueError(f"Unsupported mask sampling_mode='{normalized_mode}'.")
    return (
        MaskFarthestPointQuerySampler(
            space=normalized_mode,
        )
        .sample(
            RegionSamplingRequest(
                frame_shape_hw=normalized_mask.shape,
                mask=normalized_mask,
                init_depth=init_depth,
                camera=camera,
                num_points=num_points,
                seed=seed,
            )
        )
        .sample
    )


def resolve_region_prompt(
    *,
    target_name: str,
    config: Mapping[str, Any],
    target_config: Mapping[str, Any],
) -> str:
    """Resolve an explicit or current default prompt for one target."""

    prompt = str(target_config.get("prompt", "") or "").strip()
    if prompt:
        return prompt
    if target_name == "eef":
        return str(infer_default_eef_prompt(config) or "")
    return str(infer_default_object_prompt(config) or "")


def resolve_region_target_plan(
    *,
    target_name: str,
    target_config: Mapping[str, Any],
) -> tuple[str, str, str]:
    """Resolve selector, bbox source, and sampling route."""

    selector_mode = (
        str(target_config.get("selector", "visual") or "visual").strip().lower()
    )
    visual_config = dict(target_config.get("visual", {}))
    visual_bbox_source = (
        str(visual_config.get("bbox_source", "grounding_dino") or "grounding_dino")
        .strip()
        .lower()
    )
    sampling_config = dict(target_config.get("sampling", {}))
    visual_sampling = str(sampling_config.get("method", "") or "").strip().lower()
    if not visual_sampling:
        legacy_sampling = (
            str(visual_config.get("sampling", "sam_mask") or "sam_mask").strip().lower()
        )
        visual_sampling = (
            "bbox_gaussian" if legacy_sampling == "bbox_gaussian" else "mask_3d_fps"
        )

    if selector_mode == "off" and target_name != "obj":
        raise ValueError("selector='off' is only valid for obj.")
    return selector_mode, visual_bbox_source, visual_sampling


def resolve_config_bbox(
    target_config: Mapping[str, Any],
) -> list[int] | None:
    """Resolve the current preferred or legacy manual bbox."""

    bbox_xyxy = target_config.get("bbox_xyxy", None)
    if bbox_xyxy is None:
        bbox_xyxy = target_config.get(
            "legacy_manual_bbox_xyxy",
            None,
        )
    if isinstance(bbox_xyxy, list) and len(bbox_xyxy) == 4:
        return [int(value) for value in bbox_xyxy]
    return None


def resolve_bbox_source(
    *,
    bbox_source: str,
    prompt: str | None,
    manual_bbox: list[int] | None,
) -> str:
    """Resolve the current visual bbox backend choice."""

    normalized_source = str(bbox_source).strip().lower() or "auto"
    if normalized_source not in {
        "auto",
        "manual",
        "grounding_dino",
    }:
        raise ValueError(f"Unsupported bbox_source='{normalized_source}'.")
    if normalized_source == "auto":
        if str(prompt or "").strip():
            return "grounding_dino"
        if manual_bbox is not None:
            return "manual"
    return normalized_source


def resolve_sampling_method(
    *,
    sampling_method: str,
    mask_backend: str,
    bbox_source: str,
    prompt: str | None,
) -> str:
    """Normalize aliases and auto-select current region sampling."""

    del bbox_source
    normalized_method = str(sampling_method).strip().lower() or "auto"
    if normalized_method not in {
        "auto",
        "bbox_center",
        "bbox_gaussian",
        "mask",
        "mask_fps",
        "mask_2d_fps",
        "mask_3d_fps",
    }:
        raise ValueError(f"Unsupported sampling_method='{normalized_method}'.")
    if normalized_method == "bbox_center":
        return "bbox_gaussian"
    if normalized_method == "mask":
        return "mask_3d_fps"
    if normalized_method == "auto":
        explicit_mask_backend = str(mask_backend).strip().lower()
        if explicit_mask_backend not in {"", "auto", "none"}:
            return "mask_3d_fps"
        if bool(str(prompt or "").strip()):
            return "mask_3d_fps"
        return "bbox_gaussian"
    return normalized_method


def resolve_mask_backend(
    *,
    mask_backend: str,
    bbox_source: str,
    sampling_method: str,
) -> str | None:
    """Resolve whether a sampling route needs SAM2 or a supplied mask."""

    del bbox_source
    normalized_backend = str(mask_backend).strip().lower() or "auto"
    if normalized_backend not in {
        "auto",
        "sam2",
        "none",
        "precomputed",
    }:
        raise ValueError(f"Unsupported mask_backend='{normalized_backend}'.")
    if sampling_method not in {
        "mask_fps",
        "mask_2d_fps",
        "mask_3d_fps",
    }:
        return None
    if normalized_backend == "none":
        raise ValueError(
            "mask_backend='none' is incompatible with "
            f"sampling_method='{sampling_method}'."
        )
    if normalized_backend == "auto":
        return "sam2"
    return normalized_backend


def combine_mask_with_manual_bbox(
    *,
    mask: np.ndarray,
    manual_bbox: list[int] | None,
    combine_mode: str,
    shape_hw: tuple[int, int],
) -> tuple[np.ndarray, str]:
    """Apply the current human-bbox constraint and fallback rules."""

    normalized_mask = np.asarray(mask, dtype=bool)
    if manual_bbox is None:
        return normalized_mask, ""

    normalized_mode = str(combine_mode).strip().lower()
    if normalized_mode not in {
        "intersect",
        "union",
        "mask_only",
        "bbox_only",
    }:
        raise ValueError(f"Unsupported combine_mode='{normalized_mode}'.")

    bbox_mask = bbox_to_mask(shape_hw, manual_bbox)
    if normalized_mode == "intersect":
        candidate = normalized_mask & bbox_mask
        if np.any(candidate):
            return candidate, "bbox_intersect"
        return bbox_mask, "bbox_intersect_fallback_bbox"
    if normalized_mode == "union":
        return normalized_mask | bbox_mask, "bbox_union"
    if normalized_mode == "bbox_only":
        return bbox_mask, "bbox_only"
    return normalized_mask.copy(), "mask_only"


def infer_default_object_prompt(
    config: Mapping[str, Any],
) -> str | None:
    """Infer the current object prompt from environment or free joints."""

    root = config.get("raw", {}) if isinstance(config, dict) else {}
    environment_name = str(root.get("env_name", "")).strip()
    environment_prompts = {
        "Lift": "cube",
        "NutAssemblySquare": "square nut",
        "NutAssemblyRound": "round nut",
        "PickPlaceCan": "can",
        "PickPlaceMilk": "milk carton",
        "PickPlaceBread": "bread",
        "PickPlaceCereal": "cereal box",
    }
    if environment_name in environment_prompts:
        return environment_prompts[environment_name]

    free_joints = root.get("free_joints", [])
    names: list[str] = []
    if isinstance(free_joints, list):
        for item in free_joints:
            joint_name = str(item.get("joint_name", "")).strip()
            if not joint_name:
                continue
            names.append(_joint_name_to_prompt(joint_name))
    names = [name for name in names if name]
    names = sorted(set(names))
    if len(names) == 1:
        return names[0]
    return None


def infer_default_eef_prompt(
    config: Mapping[str, Any],
) -> str:
    """Return the current environment-independent EEF prompt."""

    del config
    return "robot gripper"


def _joint_name_to_prompt(name: str) -> str:
    normalized = str(name).strip()
    if not normalized:
        return ""
    normalized = normalized.replace("_joint0", "")
    normalized = normalized.replace("_main", "")
    normalized = normalized.replace("Visual", "")
    normalized = normalized.replace("_", " ")
    normalized = normalized.replace("SquareNut", "square nut")
    normalized = normalized.replace("RoundNut", "round nut")
    normalized = normalized.replace("Can", "can")
    normalized = normalized.replace("Milk", "milk carton")
    normalized = normalized.replace("Bread", "bread")
    normalized = normalized.replace("Cereal", "cereal box")
    return " ".join(normalized.split()).strip().lower()


__all__ = [
    "RegionPointSample",
    "bbox_from_mask",
    "bbox_to_mask",
    "clip_bbox_to_frame",
    "combine_mask_with_manual_bbox",
    "erode_mask",
    "farthest_point_indices",
    "infer_default_eef_prompt",
    "infer_default_object_prompt",
    "normalize_region_prompt",
    "resolve_bbox_source",
    "resolve_config_bbox",
    "resolve_mask_backend",
    "resolve_region_prompt",
    "resolve_region_target_plan",
    "resolve_sampling_method",
    "sample_bbox_points",
    "sample_mask_points",
]
