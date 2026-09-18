"""Path-explicit calibrated target-depth media rendering.

The renderer consumes only an in-memory ``[T,H,W]`` depth stack, an FPS value,
and an optional first-frame reference depth.  It does not resolve benchmark
paths, inspect simulator state, publish artifacts, or import video
dependencies until rendering is requested.
"""

from __future__ import annotations

import importlib
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

TARGET_DEPTH_MEDIA_SCHEMA = "dream-exe.calibrated-target-depth-media"
TARGET_DEPTH_MEDIA_FILENAMES = {
    "depth_mp4": "depth.mp4",
    "depth_frame0": "depth_frame0.png",
    "depth_contact": "depth_contact.png",
    "depth_vis_meta": "depth_vis_meta.json",
}
TARGET_DEPTH_MEDIA_LIMITS = {
    # The largest current bench depth stack observed on 2026-07-28 is
    # [805,512,512] float32 (844,103,680 payload bytes).  These limits retain
    # that corpus with headroom while keeping malformed requests bounded.
    "max_frames": 1_024,
    "max_pixels_per_frame": 4_194_304,
    "max_float32_bytes": 1_073_741_824,
    # Current bench MP4s are below 11 MiB.  The encoded limits intentionally
    # leave substantially more headroom without allowing an unbounded writer.
    "max_encoded_file_bytes": 268_435_456,
    "max_encoded_batch_bytes": 2_147_483_648,
}

_VALID_DEPTH_EPSILON = 1e-6
_LOW_PERCENTILE = 1.0
_HIGH_PERCENTILE = 99.0
_MAX_CONTACT_FRAMES = 8
_MAX_CONTACT_COLUMNS = 4
_VIDEO_DEPENDENCY_MESSAGE = (
    "Calibrated target-depth media rendering requires the optional "
    "dream-exe[video] dependencies (OpenCV and ImageIO)."
)


def _shape_tuple(
    value: Any,
    *,
    label: str,
) -> tuple[int, ...] | None:
    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        return None
    try:
        return tuple(int(size) for size in raw_shape)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{label}.shape must contain integer dimensions") from exc


def target_depth_float32_bytes_for_shape(
    shape: tuple[int, ...],
    *,
    label: str = "target_depths",
) -> int:
    if len(shape) != 3:
        raise ValueError(f"{label} must have shape [T,H,W], got {shape}")
    if any(size <= 0 for size in shape):
        raise ValueError(f"{label} must have non-empty [T,H,W] dimensions, got {shape}")
    frame_count, height, width = shape
    pixels_per_frame = int(height) * int(width)
    projected_float32_bytes = int(frame_count) * pixels_per_frame * 4
    if frame_count > int(TARGET_DEPTH_MEDIA_LIMITS["max_frames"]):
        raise ValueError(
            f"{label} frame count {frame_count} exceeds "
            f"{TARGET_DEPTH_MEDIA_LIMITS['max_frames']}"
        )
    if pixels_per_frame > int(TARGET_DEPTH_MEDIA_LIMITS["max_pixels_per_frame"]):
        raise ValueError(
            f"{label} pixels per frame {pixels_per_frame} exceeds "
            f"{TARGET_DEPTH_MEDIA_LIMITS['max_pixels_per_frame']}"
        )
    if projected_float32_bytes > int(TARGET_DEPTH_MEDIA_LIMITS["max_float32_bytes"]):
        raise ValueError(
            f"{label} projected float32 bytes {projected_float32_bytes} exceeds "
            f"{TARGET_DEPTH_MEDIA_LIMITS['max_float32_bytes']}"
        )
    return projected_float32_bytes


def _validate_reference_depth_resource_shape(
    shape: tuple[int, ...],
) -> int:
    label = "init_reference_depth"
    if len(shape) == 2:
        height, width = shape
        supplied_frames = 1
    elif len(shape) == 3:
        supplied_frames, height, width = shape
    else:
        raise ValueError(f"{label} must have shape [H,W] or [T,H,W], got {shape}")
    if any(size <= 0 for size in shape):
        raise ValueError(f"{label} must have non-empty dimensions, got {shape}")
    pixels_per_frame = int(height) * int(width)
    projected_float32_bytes = int(supplied_frames) * pixels_per_frame * 4
    if supplied_frames > int(TARGET_DEPTH_MEDIA_LIMITS["max_frames"]):
        raise ValueError(
            f"{label} frame count {supplied_frames} exceeds "
            f"{TARGET_DEPTH_MEDIA_LIMITS['max_frames']}"
        )
    if pixels_per_frame > int(TARGET_DEPTH_MEDIA_LIMITS["max_pixels_per_frame"]):
        raise ValueError(
            f"{label} pixels per frame {pixels_per_frame} exceeds "
            f"{TARGET_DEPTH_MEDIA_LIMITS['max_pixels_per_frame']}"
        )
    if projected_float32_bytes > int(TARGET_DEPTH_MEDIA_LIMITS["max_float32_bytes"]):
        raise ValueError(
            f"{label} projected float32 bytes "
            f"{projected_float32_bytes} exceeds "
            f"{TARGET_DEPTH_MEDIA_LIMITS['max_float32_bytes']}"
        )
    return projected_float32_bytes


def _load_video_dependencies() -> tuple[Any, Any]:
    try:
        cv2 = importlib.import_module("cv2")
        imageio = importlib.import_module("imageio.v2")
    except (ImportError, ModuleNotFoundError) as exc:
        raise ImportError(_VIDEO_DEPENDENCY_MESSAGE) from exc
    return cv2, imageio


def _as_target_depth_stack(value: Any) -> np.ndarray:
    declared_shape = _shape_tuple(value, label="target_depths")
    if declared_shape is not None:
        target_depth_float32_bytes_for_shape(declared_shape)
    array = np.asarray(value)
    target_depth_float32_bytes_for_shape(tuple(int(size) for size in array.shape))
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError("target_depths must contain real numeric values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _as_reference_depth(value: Any) -> np.ndarray | None:
    if value is None:
        return None
    declared_shape = _shape_tuple(
        value,
        label="init_reference_depth",
    )
    if declared_shape is not None:
        _validate_reference_depth_resource_shape(declared_shape)
    array = np.asarray(value)
    _validate_reference_depth_resource_shape(tuple(int(size) for size in array.shape))
    if array.ndim == 3:
        array = array[0]
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError("init_reference_depth must contain real numeric values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _valid_depth_mask(values: np.ndarray) -> np.ndarray:
    return np.isfinite(values) & (values > _VALID_DEPTH_EPSILON)


def _resize_reference_depth(
    reference: np.ndarray,
    *,
    height: int,
    width: int,
    cv2: Any,
) -> np.ndarray:
    if reference.shape == (height, width):
        return reference
    resizable = np.asarray(reference, dtype=np.float32)
    resized = np.asarray(
        cv2.resize(
            resizable,
            (width, height),
            interpolation=cv2.INTER_NEAREST,
        )
    )
    if resized.shape != (height, width):
        raise RuntimeError(
            "nearest-neighbor reference-depth resize returned "
            f"{resized.shape}, expected {(height, width)}"
        )
    return resized


def _select_depth_range(
    target_depths: np.ndarray,
    *,
    init_reference_depth: np.ndarray | None,
    cv2: Any,
) -> dict[str, Any]:
    height, width = (
        int(target_depths.shape[1]),
        int(target_depths.shape[2]),
    )
    values: np.ndarray
    source = "depth_stack"
    if init_reference_depth is not None:
        resized_reference = _resize_reference_depth(
            init_reference_depth,
            height=height,
            width=width,
            cv2=cv2,
        )
        reference_mask = _valid_depth_mask(resized_reference)
        if np.any(reference_mask):
            values = np.asarray(resized_reference[reference_mask])
            source = "reference_depth"
        else:
            target_mask = _valid_depth_mask(target_depths)
            values = np.asarray(target_depths[target_mask])
    else:
        target_mask = _valid_depth_mask(target_depths)
        values = np.asarray(target_depths[target_mask])

    valid_count = int(values.size)
    percentile = [_LOW_PERCENTILE, _HIGH_PERCENTILE]
    if not valid_count:
        return {
            "vmin": 0.0,
            "vmax": 1.0,
            "percentile": percentile,
            "range_source": source,
            "valid_pixels": 0,
            "expanded": True,
        }

    percentile_low, percentile_high = np.percentile(
        np.asarray(values, dtype=np.float64),
        percentile,
    )
    vmin = float(percentile_low)
    vmax = float(percentile_high)
    if not math.isfinite(vmin) or not math.isfinite(vmax) or vmax <= vmin:
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))
    center = float(0.5 * (vmin + vmax)) if math.isfinite(vmin + vmax) else 0.0
    minimum_span = max(abs(center) * 0.05, 0.05)
    span = float(vmax - vmin)
    expanded = False
    if not math.isfinite(span) or span < minimum_span:
        center = (
            float(0.5 * (vmin + vmax))
            if math.isfinite(vmin + vmax)
            else float(np.nanmedian(values))
        )
        half_span = minimum_span * 0.5
        vmin = center - half_span
        vmax = center + half_span
        expanded = True
    vmin = max(0.0, float(vmin))
    # Preserve the current implementation's positive-only range oracle:
    # clamping vmin may narrow the expanded span, but vmax is extended only
    # when the clamp collapses or reverses the range.
    if vmax <= vmin:
        vmax = vmin + minimum_span
        expanded = True
    return {
        "vmin": float(vmin),
        "vmax": float(vmax),
        "percentile": percentile,
        "range_source": source,
        "valid_pixels": valid_count,
        "expanded": expanded,
    }


def _render_depth_rgb_frame(
    depth: np.ndarray,
    *,
    vmin: float,
    vmax: float,
    cv2: Any,
) -> np.ndarray:
    valid = _valid_depth_mask(depth)
    gray = np.zeros(depth.shape, dtype=np.uint8)
    if np.any(valid):
        normalized = (np.asarray(depth[valid], dtype=np.float64) - float(vmin)) / (
            float(vmax) - float(vmin)
        )
        gray[valid] = np.rint(np.clip(normalized, 0.0, 1.0) * 255.0).astype(np.uint8)
    bgr = np.asarray(cv2.applyColorMap(gray, cv2.COLORMAP_INFERNO))
    if bgr.shape != (*depth.shape, 3):
        raise RuntimeError(
            "OpenCV depth color mapping returned "
            f"{bgr.shape}, expected {(*depth.shape, 3)}"
        )
    rgb = np.asarray(
        cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB),
        dtype=np.uint8,
    ).copy()
    if rgb.shape != (*depth.shape, 3):
        raise RuntimeError(
            "OpenCV BGR-to-RGB conversion returned "
            f"{rgb.shape}, expected {(*depth.shape, 3)}"
        )
    rgb[~valid] = 0
    return rgb


def _contact_frame_indices(frame_count: int) -> np.ndarray:
    count = min(int(frame_count), _MAX_CONTACT_FRAMES)
    indices = np.rint(np.linspace(0, int(frame_count) - 1, num=count)).astype(np.int64)
    return np.unique(indices)


def _build_contact_sheet(
    frames: Mapping[int, np.ndarray],
    *,
    indices: np.ndarray,
    height: int,
    width: int,
) -> tuple[np.ndarray, int, int]:
    columns = min(_MAX_CONTACT_COLUMNS, int(indices.size))
    rows = math.ceil(int(indices.size) / columns)
    sheet = np.zeros(
        (rows * height, columns * width, 3),
        dtype=np.uint8,
    )
    for cell, raw_index in enumerate(indices):
        index = int(raw_index)
        frame = np.asarray(frames[index], dtype=np.uint8)
        if frame.shape != (height, width, 3):
            raise RuntimeError(
                f"contact frame {index} has unexpected shape {frame.shape}"
            )
        row, column = divmod(cell, columns)
        sheet[
            row * height : (row + 1) * height,
            column * width : (column + 1) * width,
        ] = frame
    return sheet, rows, columns


def _metadata_paths(
    final_paths: Mapping[str, str | Path] | None,
) -> dict[str, str]:
    supplied = dict(final_paths or {})
    unknown = sorted(set(supplied).difference(TARGET_DEPTH_MEDIA_FILENAMES))
    if unknown:
        raise ValueError("final_paths has unsupported fields: " + ", ".join(unknown))
    result: dict[str, str] = {}
    for key, filename in TARGET_DEPTH_MEDIA_FILENAMES.items():
        raw_value = supplied.get(key, filename)
        text = os.fspath(raw_value).strip()
        if not text:
            raise ValueError(f"final_paths[{key!r}] must be non-empty")
        if "\x00" in text:
            raise ValueError(f"final_paths[{key!r}] contains a null byte")
        result[key] = Path(text).as_posix()
    return result


def _explicit_output_directory(value: str | Path) -> Path:
    text = os.fspath(value).strip()
    if not text:
        raise ValueError("output_dir must be an explicit non-empty path")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"output_dir must be absolute: {text}")
    path = Path(os.path.abspath(path))
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"output_dir must not traverse a symlink: {current}")
        parent = current.parent
        if parent == current:
            break
        current = parent
    if path.exists() and not path.is_dir():
        raise ValueError(f"output_dir exists but is not a directory: {path}")
    return path


def _partial_path(path: Path) -> Path:
    return path.with_name(f"{path.stem}.partial{path.suffix}")


def _remove_paths(paths: list[Path]) -> None:
    for path in paths:
        try:
            path.unlink(missing_ok=True)
        except OSError:
            pass


def _bounded_regular_file_size(
    path: Path,
    *,
    label: str,
    allow_empty: bool = False,
) -> int:
    try:
        file_stat = path.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} was not created: {path}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError(f"{label} must be a regular file: {path}")
    size = int(file_stat.st_size)
    if not allow_empty and size <= 0:
        raise RuntimeError(f"{label} must be non-empty: {path}")
    if size > int(TARGET_DEPTH_MEDIA_LIMITS["max_encoded_file_bytes"]):
        raise RuntimeError(
            f"{label} size {size} exceeds "
            f"{TARGET_DEPTH_MEDIA_LIMITS['max_encoded_file_bytes']} bytes"
        )
    return size


def _validate_encoded_batch(paths: Mapping[str, Path]) -> dict[str, int]:
    sizes = {
        key: _bounded_regular_file_size(
            path,
            label=f"target-depth {key}",
        )
        for key, path in paths.items()
    }
    total = int(sum(sizes.values()))
    if total > int(TARGET_DEPTH_MEDIA_LIMITS["max_encoded_batch_bytes"]):
        raise RuntimeError(
            f"target-depth encoded batch size {total} exceeds "
            f"{TARGET_DEPTH_MEDIA_LIMITS['max_encoded_batch_bytes']} bytes"
        )
    return sizes


def render_calibrated_target_depth_media(
    output_dir: str | Path,
    *,
    target_depths: Any,
    fps: float,
    init_reference_depth: Any = None,
    final_paths: Mapping[str, str | Path] | None = None,
    stage: str = "",
) -> dict[str, Any]:
    """Render calibrated target-depth video and compact PNG evidence.

    ``output_dir`` is a caller-owned local rendering directory, normally a
    temporary directory whose files are later handed to the shared depth
    publication transaction.  The fixed output filenames are
    ``depth.mp4``, ``depth_frame0.png``, ``depth_contact.png``, and
    ``depth_vis_meta.json``.

    ``final_paths`` changes only the portable paths written into metadata.
    Omitting it records the four filenames, so the payload can be rewritten by
    a publication adapter without leaking the local rendering directory.
    """

    depths = _as_target_depth_stack(target_depths)
    reference = _as_reference_depth(init_reference_depth)
    try:
        source_fps = float(fps)
    except (TypeError, ValueError) as exc:
        raise TypeError("fps must be a finite numeric value") from exc
    if not math.isfinite(source_fps):
        raise ValueError("fps must be finite")
    encoded_fps = max(1, round(source_fps))
    metadata_artifacts = _metadata_paths(final_paths)
    directory = _explicit_output_directory(output_dir)
    frame_count, height, width = (int(value) for value in depths.shape)
    if (height % 2) or (width % 2):
        raise ValueError(
            "target_depths H and W must be even for libx264 yuv420p encoding, "
            f"got {(height, width)}"
        )
    cv2, imageio = _load_video_dependencies()

    depth_range = _select_depth_range(
        depths,
        init_reference_depth=reference,
        cv2=cv2,
    )
    contact_indices = _contact_frame_indices(frame_count)
    contact_set = {int(value) for value in contact_indices}

    output_paths = {
        key: directory / filename
        for key, filename in TARGET_DEPTH_MEDIA_FILENAMES.items()
    }
    partial_paths = {key: _partial_path(path) for key, path in output_paths.items()}
    occupied = [
        path
        for path in [*output_paths.values(), *partial_paths.values()]
        if path.exists() or path.is_symlink()
    ]
    if occupied:
        raise FileExistsError(
            "target-depth render destinations must be unused: "
            + ", ".join(path.as_posix() for path in occupied)
        )

    directory_created = not directory.exists()
    directory.mkdir(parents=True, exist_ok=True)
    cleanup_paths = [*partial_paths.values(), *output_paths.values()]
    selected_frames: dict[int, np.ndarray] = {}
    try:
        writer = imageio.get_writer(
            partial_paths["depth_mp4"].as_posix(),
            fps=encoded_fps,
            macro_block_size=1,
            codec="libx264",
            ffmpeg_params=[
                "-crf",
                "18",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
            ],
        )
        try:
            for index, depth in enumerate(depths):
                rgb = _render_depth_rgb_frame(
                    depth,
                    vmin=float(depth_range["vmin"]),
                    vmax=float(depth_range["vmax"]),
                    cv2=cv2,
                )
                writer.append_data(rgb)
                if index in contact_set:
                    selected_frames[index] = rgb
        finally:
            writer.close()
        encoded_sizes = {
            "depth_mp4": _bounded_regular_file_size(
                partial_paths["depth_mp4"],
                label="target-depth MP4",
            )
        }

        frame0 = selected_frames[0]
        contact, _, _ = _build_contact_sheet(
            selected_frames,
            indices=contact_indices,
            height=height,
            width=width,
        )
        for path, rgb in (
            (partial_paths["depth_frame0"], frame0),
            (partial_paths["depth_contact"], contact),
        ):
            bgr = cv2.cvtColor(
                np.asarray(rgb, dtype=np.uint8),
                cv2.COLOR_RGB2BGR,
            )
            if not cv2.imwrite(path.as_posix(), bgr):
                raise OSError(f"failed to write target-depth PNG: {path}")
        for key in ("depth_frame0", "depth_contact"):
            encoded_sizes[key] = _bounded_regular_file_size(
                partial_paths[key],
                label=f"target-depth {key}",
            )
        metadata = {
            "video_path": metadata_artifacts["depth_mp4"],
            "first_png_path": metadata_artifacts["depth_frame0"],
            "contact_sheet_path": metadata_artifacts["depth_contact"],
            "depth_space": "metric",
            "stage": str(stage or ""),
            "num_frames": frame_count,
            "shape": [height, width],
            "fps": source_fps,
            "colormap": "inferno",
            "invalid_rgb": [0, 0, 0],
            "positive_only": True,
            "range": depth_range,
            "resource_limits": {
                **TARGET_DEPTH_MEDIA_LIMITS,
                "source_float32_bytes": int(depths.size * 4),
                "reference_float32_bytes": (
                    0 if reference is None else int(reference.size * 4)
                ),
            },
            "encoded_media_bytes": int(sum(encoded_sizes.values())),
        }
        partial_paths["depth_vis_meta"].write_text(
            json.dumps(
                metadata,
                ensure_ascii=False,
                allow_nan=False,
                indent=2,
                sort_keys=True,
            )
            + "\n",
            encoding="utf-8",
        )
        _validate_encoded_batch(partial_paths)
        for key in TARGET_DEPTH_MEDIA_FILENAMES:
            partial_paths[key].replace(output_paths[key])
    except BaseException:
        _remove_paths(cleanup_paths)
        if directory_created:
            try:
                directory.rmdir()
            except OSError:
                pass
        raise

    return {
        "paths": {key: path.as_posix() for key, path in output_paths.items()},
        "metadata": metadata,
    }


__all__ = [
    "TARGET_DEPTH_MEDIA_FILENAMES",
    "TARGET_DEPTH_MEDIA_LIMITS",
    "TARGET_DEPTH_MEDIA_SCHEMA",
    "render_calibrated_target_depth_media",
    "target_depth_float32_bytes_for_shape",
]
