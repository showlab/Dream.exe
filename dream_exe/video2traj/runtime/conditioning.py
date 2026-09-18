"""Explicit conditioning-image coordinate alignment for video runtimes."""

from __future__ import annotations

import copy
from collections.abc import Mapping
from typing import Any

import numpy as np

from ..region.runtime import PrecomputedRegion
from ..region.selection import bbox_from_mask


CONDITIONING_TRANSFORM_SCHEMA = "dream-exe.conditioning-image-transform"
CONDITIONING_TRANSFORM_ALGORITHM = "scale_to_cover_center_crop"


def _positive_size(value: Any, *, label: str) -> tuple[int, int]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    width = value.get("width")
    height = value.get("height")
    if (
        isinstance(width, bool)
        or not isinstance(width, int)
        or isinstance(height, bool)
        or not isinstance(height, int)
        or width <= 0
        or height <= 0
    ):
        raise ValueError(f"{label} width and height must be positive integers")
    return height, width


def normalize_conditioning_transform(
    value: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one explicit scale-to-cover/center-crop record."""

    payload = copy.deepcopy(dict(value))
    if payload.get("format") != CONDITIONING_TRANSFORM_SCHEMA:
        raise ValueError(
            "conditioning transform format must be "
            f"{CONDITIONING_TRANSFORM_SCHEMA!r}"
        )
    if payload.get("algorithm") != CONDITIONING_TRANSFORM_ALGORITHM:
        raise ValueError(
            "conditioning transform algorithm must be "
            f"{CONDITIONING_TRANSFORM_ALGORITHM!r}"
        )
    input_shape = _positive_size(payload.get("input_size"), label="input_size")
    resized_shape = _positive_size(
        payload.get("resized_size"),
        label="resized_size",
    )
    output_shape = _positive_size(
        payload.get("output_size"),
        label="output_size",
    )
    crop = payload.get("crop_xyxy_in_resized")
    if (
        not isinstance(crop, list)
        or len(crop) != 4
        or any(isinstance(item, bool) or not isinstance(item, int) for item in crop)
    ):
        raise ValueError("crop_xyxy_in_resized must contain four integers")
    left, top, right, bottom = crop
    if (
        left < 0
        or top < 0
        or right > resized_shape[1]
        or bottom > resized_shape[0]
        or right <= left
        or bottom <= top
        or (bottom - top, right - left) != output_shape
    ):
        raise ValueError("conditioning transform crop is inconsistent")
    payload["input_shape_hw"] = list(input_shape)
    payload["resized_shape_hw"] = list(resized_shape)
    payload["output_shape_hw"] = list(output_shape)
    return payload


def _resize_finite_depth(
    array: np.ndarray,
    *,
    width: int,
    height: int,
    cv2_module: Any,
) -> np.ndarray:
    values = np.asarray(array, dtype=np.float32)
    valid = np.isfinite(values)
    weighted = cv2_module.resize(
        np.where(valid, values, 0.0),
        (width, height),
        interpolation=cv2_module.INTER_LINEAR,
    )
    weights = cv2_module.resize(
        valid.astype(np.float32),
        (width, height),
        interpolation=cv2_module.INTER_LINEAR,
    )
    output = np.full((height, width), np.nan, dtype=np.float32)
    np.divide(weighted, weights, out=output, where=weights > 1e-6)
    return output


def warp_conditioning_array(
    array: Any,
    *,
    transform: Mapping[str, Any],
    target_shape_hw: tuple[int, int],
    kind: str,
    cv2_module: Any | None = None,
) -> np.ndarray:
    """Apply the recorded image transform and final video-reader resize."""

    normalized = normalize_conditioning_transform(transform)
    source = np.asarray(array)
    input_shape = tuple(normalized["input_shape_hw"])
    if source.ndim != 2 or tuple(source.shape) != input_shape:
        raise ValueError(
            f"conditioning array/input mismatch: {tuple(source.shape)} != {input_shape}"
        )
    target_height, target_width = (int(item) for item in target_shape_hw)
    if min(target_height, target_width) <= 0:
        raise ValueError("target_shape_hw must be positive")
    if kind not in {"mask", "depth"}:
        raise ValueError("conditioning array kind must be 'mask' or 'depth'")
    if cv2_module is None:
        import cv2 as cv2_module

    resized_height, resized_width = normalized["resized_shape_hw"]
    left, top, right, bottom = normalized["crop_xyxy_in_resized"]
    if kind == "mask":
        resized = cv2_module.resize(
            source.astype(np.uint8),
            (resized_width, resized_height),
            interpolation=cv2_module.INTER_NEAREST,
        )
    else:
        resized = _resize_finite_depth(
            source,
            width=resized_width,
            height=resized_height,
            cv2_module=cv2_module,
        )
    cropped = resized[top:bottom, left:right]
    if tuple(cropped.shape) != tuple(normalized["output_shape_hw"]):
        raise RuntimeError("conditioning transform produced an invalid crop")
    if tuple(cropped.shape) != (target_height, target_width):
        if kind == "mask":
            cropped = cv2_module.resize(
                cropped,
                (target_width, target_height),
                interpolation=cv2_module.INTER_NEAREST,
            )
        else:
            cropped = _resize_finite_depth(
                cropped,
                width=target_width,
                height=target_height,
                cv2_module=cv2_module,
            )
    return (
        np.asarray(cropped, dtype=bool)
        if kind == "mask"
        else np.asarray(cropped, dtype=np.float32)
    )


def _align_region(
    region: Any,
    *,
    transform: Mapping[str, Any],
    target_shape_hw: tuple[int, int],
) -> PrecomputedRegion:
    if not isinstance(region, PrecomputedRegion):
        raise TypeError("precomputed conditioning regions must be PrecomputedRegion")
    mask = warp_conditioning_array(
        region.mask,
        transform=transform,
        target_shape_hw=target_shape_hw,
        kind="mask",
    )
    return PrecomputedRegion(
        mask=mask,
        bbox_xyxy=bbox_from_mask(mask),
        source=f"{region.source}+conditioning_transform",
        matched_names=list(region.matched_names),
        compact_labels=list(region.compact_labels),
    )


def align_runtime_conditioning_inputs(
    *,
    transform: Mapping[str, Any],
    target_shape_hw: tuple[int, int],
    depth_options: Mapping[str, Any] | None,
    target_depth_options: Mapping[str, Any] | None,
    object_runtime_options: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Align every explicit init-depth and saved-region consumer together."""

    normalized = normalize_conditioning_transform(transform)
    aligned_depth = copy.deepcopy(dict(depth_options or {}))
    aligned_target = copy.deepcopy(dict(target_depth_options or {}))
    aligned_objects = copy.deepcopy(dict(object_runtime_options or {}))
    for options in (aligned_depth, aligned_target):
        if options.get("init_ref_depth") is not None:
            options["init_ref_depth"] = warp_conditioning_array(
                options["init_ref_depth"],
                transform=normalized,
                target_shape_hw=target_shape_hw,
                kind="depth",
            )
    if aligned_objects.get("init_depth") is not None:
        aligned_objects["init_depth"] = warp_conditioning_array(
            aligned_objects["init_depth"],
            transform=normalized,
            target_shape_hw=target_shape_hw,
            kind="depth",
        )
    if aligned_objects.get("eef_precomputed_region") is not None:
        aligned_objects["eef_precomputed_region"] = _align_region(
            aligned_objects["eef_precomputed_region"],
            transform=normalized,
            target_shape_hw=target_shape_hw,
        )
    aligned_objects["precomputed_regions"] = {
        str(key): _align_region(
            region,
            transform=normalized,
            target_shape_hw=target_shape_hw,
        )
        for key, region in dict(
            aligned_objects.get("precomputed_regions", {}) or {}
        ).items()
    }
    return {
        "depth_options": aligned_depth,
        "target_depth_options": aligned_target,
        "object_runtime_options": aligned_objects,
        "manifest": {
            "format": CONDITIONING_TRANSFORM_SCHEMA,
            "algorithm": CONDITIONING_TRANSFORM_ALGORITHM,
            "input_shape_hw": list(normalized["input_shape_hw"]),
            "generated_shape_hw": list(normalized["output_shape_hw"]),
            "runtime_shape_hw": [
                int(target_shape_hw[0]),
                int(target_shape_hw[1]),
            ],
        },
    }


def _resize_init_depth_nearest(
    value: Any,
    *,
    target_shape_hw: tuple[int, int],
    label: str,
    cv2_module: Any | None = None,
) -> tuple[np.ndarray, tuple[int, int]]:
    """Resize one legacy benchmark init-depth array with nearest semantics.

    The original video2sim benchmark path resized the simulator's 512x512
    initialization depth to the decoded video frame shape before region
    sampling and depth calibration. Keep this compatibility operation
    separate from :func:`warp_conditioning_array`: conditioning transforms use
    finite-aware linear interpolation and have different provenance.
    """

    source = np.asarray(value, dtype=np.float32)
    if source.ndim != 2 or any(int(size) <= 0 for size in source.shape):
        raise ValueError(
            f"{label} must be a non-empty [H,W] array, got {source.shape}"
        )
    target_height, target_width = (int(size) for size in target_shape_hw)
    if min(target_height, target_width) <= 0:
        raise ValueError("target_shape_hw must contain positive dimensions")
    source_shape = (int(source.shape[0]), int(source.shape[1]))
    if source_shape == (target_height, target_width):
        return source.copy(), source_shape
    if cv2_module is None:
        import cv2 as cv2_module
    resized = cv2_module.resize(
        source,
        (target_width, target_height),
        interpolation=cv2_module.INTER_NEAREST,
    )
    normalized = np.asarray(resized, dtype=np.float32)
    expected = (target_height, target_width)
    if tuple(normalized.shape) != expected:
        raise RuntimeError(
            f"{label} nearest resize produced {normalized.shape}, expected {expected}"
        )
    return normalized, source_shape


def align_runtime_init_depth_inputs(
    *,
    target_shape_hw: tuple[int, int],
    depth_options: Mapping[str, Any] | None,
    target_depth_options: Mapping[str, Any] | None,
    object_runtime_options: Mapping[str, Any] | None,
    cv2_module: Any | None = None,
) -> dict[str, Any]:
    """Restore video2sim's nearest resize for benchmark init-depth inputs.

    This is intentionally only a shape alignment. It does not apply a
    generated-video conditioning transform and therefore must be used only
    when no explicit conditioning transform is present. Every consumer is
    copied before binding so callers' arrays are never mutated.
    """

    aligned_depth = copy.deepcopy(dict(depth_options or {}))
    aligned_target = copy.deepcopy(dict(target_depth_options or {}))
    aligned_objects = copy.deepcopy(dict(object_runtime_options or {}))
    consumers = (
        ("depth_options.init_ref_depth", aligned_depth, "init_ref_depth"),
        (
            "target_depth_options.init_ref_depth",
            aligned_target,
            "init_ref_depth",
        ),
        ("object_runtime_options.init_depth", aligned_objects, "init_depth"),
    )
    alignments: list[dict[str, Any]] = []
    for consumer, options, key in consumers:
        value = options.get(key)
        if value is None:
            continue
        resized, source_shape = _resize_init_depth_nearest(
            value,
            target_shape_hw=target_shape_hw,
            label=consumer,
            cv2_module=cv2_module,
        )
        options[key] = resized
        target_shape = (int(target_shape_hw[0]), int(target_shape_hw[1]))
        if source_shape != target_shape:
            alignments.append(
                {
                    "consumer": consumer,
                    "source_shape_hw": list(source_shape),
                    "target_shape_hw": list(target_shape),
                    "interpolation": "nearest",
                }
            )
    return {
        "depth_options": aligned_depth,
        "target_depth_options": aligned_target,
        "object_runtime_options": aligned_objects,
        "manifest": {
            "format": "dream-exe.init-depth-alignment",
            "algorithm": "resize_nearest",
            "target_shape_hw": [int(target_shape_hw[0]), int(target_shape_hw[1])],
            "consumers": alignments,
        },
    }


__all__ = [
    "CONDITIONING_TRANSFORM_ALGORITHM",
    "CONDITIONING_TRANSFORM_SCHEMA",
    "align_runtime_conditioning_inputs",
    "align_runtime_init_depth_inputs",
    "normalize_conditioning_transform",
    "warp_conditioning_array",
]
