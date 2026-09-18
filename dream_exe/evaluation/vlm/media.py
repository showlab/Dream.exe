"""Prepare deterministic image grids from explicit saved-video inputs.

This module preserves the observable preprocessing behavior of the three
paper VLM rubrics without importing a provider, bench registry, or
credential source.  OpenCV is imported only when the default reader or writer
is actually used.
"""

from __future__ import annotations

import os
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from .batch import (
    RUBRIC_PHYSICAL_PLAUSIBILITY,
    RUBRIC_SUBJECT_STABILITY,
    RUBRIC_TASK_ADHERENCE,
    _normalize_rubric,
)
from .scoring import (
    grid_resample_indices,
    list_grid_image_files,
    merge_frame_grid,
    stability_video_frame_indices,
    uniform_video_frame_indices,
)

FrameReader = Callable[[Path, int, str], Sequence[np.ndarray]]
ImageWriter = Callable[[Path, np.ndarray], bool | None]

_PREPARATION_SCHEMA = "dream-exe.vlm-media-preparation"
_SCORE_RUBRICS = {
    RUBRIC_PHYSICAL_PLAUSIBILITY,
    RUBRIC_TASK_ADHERENCE,
}


def _import_cv2() -> Any:
    try:
        import cv2
    except ImportError as error:
        raise RuntimeError(
            "OpenCV is required for default VLM media preparation; "
            "install dream-exe[video] or inject frame_reader and image_writer"
        ) from error
    return cv2


def _opencv_frame_reader(
    video_path: Path,
    extraction_count: int,
    rubric: str,
) -> list[np.ndarray]:
    """Read the same BGR frames selected by the paper VLM scripts."""

    cv2 = _import_cv2()
    capture = cv2.VideoCapture(str(video_path))
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        if rubric == RUBRIC_SUBJECT_STABILITY:
            frame_indices = stability_video_frame_indices(total_frames)
        else:
            frame_indices = uniform_video_frame_indices(
                total_frames,
                extraction_count,
            )
        target_indices = {int(index) for index in frame_indices}
        frames: list[np.ndarray] = []
        current_index = 0
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break
            if current_index in target_indices:
                frames.append(frame)
                if len(frames) >= extraction_count:
                    break
            current_index += 1
        return frames
    finally:
        capture.release()


def _opencv_image_writer(
    output_path: Path,
    grid: np.ndarray,
) -> bool:
    """Write a grid with the current ``cv2.imwrite`` JPEG behavior."""

    cv2 = _import_cv2()
    return bool(cv2.imwrite(str(output_path), grid))


def _discover_video_entries(
    video_path: Path,
) -> tuple[list[str], Path]:
    if video_path.is_dir():
        names = os.listdir(video_path)
        root = video_path
    elif video_path.is_file():
        names = [video_path.name]
        root = video_path.parent
    else:
        raise FileNotFoundError(
            f"Video path not found: {video_path}. "
            "Please provide an existing directory or a video file path."
        )
    names.sort()
    return names, root


def _grid_contract(
    rubric: str,
    num_images: int | None,
) -> tuple[int, int, int, int]:
    if rubric == RUBRIC_SUBJECT_STABILITY:
        selected_count = 2 if num_images is None else num_images
        if selected_count != 2:
            raise ValueError("subject_stability requires num_images=2 for its 1x2 grid")
        return 2, 2, 1, 2

    selected_count = 6 if num_images is None else num_images
    if selected_count != 6:
        raise ValueError(f"{rubric} requires num_images=6 for its 3x2 grid")
    return 16, 6, 3, 2


def _media_records(
    output_dir: Path,
    *,
    rubric: str,
    extraction_count: int,
    num_images: int,
) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "media_path": (output_dir / name).as_posix(),
            "media_sampling": {
                "preparation_schema": _PREPARATION_SCHEMA,
                "rubric": rubric,
                "extraction_count": extraction_count,
                "grid_image_count": num_images,
            },
        }
        for name in list_grid_image_files(output_dir)
    ]


def _error_info(error: Exception) -> dict[str, str]:
    try:
        message = str(error)
    except Exception:
        message = f"<{type(error).__name__}>"
    return {
        "type": type(error).__name__,
        "message": message,
    }


def prepare_vlm_media_grids(
    *,
    video_dir: str | os.PathLike[str],
    output_dir: str | os.PathLike[str],
    rubric: str,
    num_images: int | None = None,
    frame_reader: FrameReader | None = None,
    image_writer: ImageWriter | None = None,
    reuse_existing: bool = True,
    continue_on_error: bool = True,
) -> dict[str, Any]:
    """Prepare current-compatible VLM grid images in an explicit directory.

    ``video_dir`` may also name one video, matching the paper scripts.  Input
    entries are processed in lexical filename order without extension
    filtering.  The output filename is ``<text before first dot>.jpeg``.

    With ``reuse_existing=True``, any non-empty existing output directory is
    reused wholesale, matching the original ``--image_grid_path`` check.
    Supported image files in that directory become ``media_records`` that can
    be passed directly to :func:`run_saved_media_vlm_batch`.

    The default reader keeps OpenCV BGR arrays unchanged: there is no resize or
    color conversion on the grid path.  A false return from an injected/default
    writer is reported as an error instead of being silently ignored.
    """

    clean_rubric = _normalize_rubric(rubric)
    (
        extraction_count,
        selected_count,
        grid_rows,
        grid_cols,
    ) = _grid_contract(clean_rubric, num_images)
    if frame_reader is not None and not callable(frame_reader):
        raise TypeError("frame_reader must be callable")
    if image_writer is not None and not callable(image_writer):
        raise TypeError("image_writer must be callable")

    input_path = Path(video_dir).expanduser().resolve()
    output_path = Path(output_dir).expanduser().resolve()
    reader = frame_reader or _opencv_frame_reader
    writer = image_writer or _opencv_image_writer

    if output_path.exists():
        if not output_path.is_dir():
            raise NotADirectoryError(
                f"VLM grid output path is not a directory: {output_path}"
            )
        if reuse_existing and os.listdir(output_path):
            records = _media_records(
                output_path,
                rubric=clean_rubric,
                extraction_count=extraction_count,
                num_images=selected_count,
            )
            return {
                "format": _PREPARATION_SCHEMA,
                "status": "reused_existing",
                "rubric": clean_rubric,
                "video_dir": input_path.as_posix(),
                "output_dir": output_path.as_posix(),
                "reuse_existing": True,
                "grid": {
                    "extraction_count": extraction_count,
                    "num_images": selected_count,
                    "rows": grid_rows,
                    "cols": grid_cols,
                    "color_order": "BGR",
                    "resize": None,
                },
                "items": [],
                "media_records": records,
                "prepared_count": 0,
                "skipped_count": 0,
                "error_count": 0,
            }

    video_names, video_root = _discover_video_entries(input_path)
    output_path.mkdir(parents=True, exist_ok=True)

    items: list[dict[str, Any]] = []
    for order, video_name in enumerate(video_names):
        source_path = video_root / video_name
        video_id = video_name.split(".")[0]
        grid_path = output_path / f"{video_id}.jpeg"
        item: dict[str, Any] = {
            "order": order,
            "name": video_name,
            "video_id": video_id,
            "video_path": source_path.as_posix(),
            "grid_name": grid_path.name,
            "grid_path": grid_path.as_posix(),
        }
        try:
            frames = list(
                reader(
                    source_path,
                    extraction_count,
                    clean_rubric,
                )
            )
            item["extracted_frame_count"] = len(frames)
            if clean_rubric == RUBRIC_SUBJECT_STABILITY:
                if len(frames) < selected_count:
                    item["status"] = "skipped_short_video"
                    items.append(item)
                    continue
                grid_frames = frames
            elif clean_rubric in _SCORE_RUBRICS:
                indices = grid_resample_indices(
                    len(frames),
                    selected_count,
                )
                item["grid_resample_indices"] = [int(index) for index in indices]
                grid_frames = [frames[int(index)] for index in indices]
            else:  # pragma: no cover - normalized above
                raise AssertionError(f"unhandled rubric: {clean_rubric}")

            grid = merge_frame_grid(
                grid_frames,
                rows=grid_rows,
                cols=grid_cols,
            )
            write_result = writer(grid_path, grid)
            if write_result is False:
                raise OSError(f"image writer returned false for {grid_path}")
            item["status"] = "written"
        except Exception as error:
            if not continue_on_error:
                raise
            item["status"] = "error"
            item["error"] = _error_info(error)
        items.append(item)

    records = _media_records(
        output_path,
        rubric=clean_rubric,
        extraction_count=extraction_count,
        num_images=selected_count,
    )
    prepared_count = sum(item["status"] == "written" for item in items)
    skipped_count = sum(item["status"] == "skipped_short_video" for item in items)
    error_count = sum(item["status"] == "error" for item in items)
    return {
        "format": _PREPARATION_SCHEMA,
        "status": (
            "completed_with_issues" if skipped_count or error_count else "completed"
        ),
        "rubric": clean_rubric,
        "video_dir": input_path.as_posix(),
        "output_dir": output_path.as_posix(),
        "reuse_existing": False,
        "grid": {
            "extraction_count": extraction_count,
            "num_images": selected_count,
            "rows": grid_rows,
            "cols": grid_cols,
            "color_order": "BGR",
            "resize": None,
        },
        "items": items,
        "media_records": records,
        "prepared_count": prepared_count,
        "skipped_count": skipped_count,
        "error_count": error_count,
    }


def prepare_uniform_vlm_media_grid(
    *,
    video_path: str | os.PathLike[str],
    output_path: str | os.PathLike[str],
    frame_count: int = 6,
    rows: int = 3,
    cols: int = 2,
    image_writer: ImageWriter | None = None,
) -> dict[str, Any]:
    """Prepare one direct uniform-frame grid from one explicit saved video.

    This is the provider-free media operation used by the trajectory-aware
    VLM request.  Unlike the paper score-rubric path above, it samples the
    requested number of frames directly from the source video and records
    those exact source indices.
    """

    if (
        isinstance(frame_count, bool)
        or not isinstance(frame_count, int)
        or frame_count < 1
    ):
        raise ValueError("frame_count must be a positive integer")
    if (
        isinstance(rows, bool)
        or not isinstance(rows, int)
        or rows < 1
        or isinstance(cols, bool)
        or not isinstance(cols, int)
        or cols < 1
    ):
        raise ValueError("rows and cols must be positive integers")
    if rows * cols != frame_count:
        raise ValueError("rows * cols must equal frame_count")
    if image_writer is not None and not callable(image_writer):
        raise TypeError("image_writer must be callable")

    source = Path(video_path).expanduser().resolve()
    destination = Path(output_path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Video path not found: {source}")

    cv2 = _import_cv2()
    capture = cv2.VideoCapture(source.as_posix())
    try:
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_indices = [
            int(value)
            for value in uniform_video_frame_indices(
                total_frames,
                frame_count,
            )
        ]
        if len(frame_indices) != frame_count:
            raise ValueError(
                "source video does not contain enough frames for the "
                f"declared {frame_count}-frame grid"
            )
        targets = set(frame_indices)
        frames: list[np.ndarray] = []
        current_index = 0
        while capture.isOpened():
            success, frame = capture.read()
            if not success:
                break
            if current_index in targets:
                frames.append(frame)
            current_index += 1
    finally:
        capture.release()

    if len(frames) != frame_count:
        raise RuntimeError("could not read every declared uniform source-video frame")
    grid = merge_frame_grid(frames, rows=rows, cols=cols)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = image_writer or _opencv_image_writer
    write_result = writer(destination, grid)
    if write_result is False:
        raise OSError(f"image writer returned false for {destination}")
    return {
        "format": _PREPARATION_SCHEMA,
        "status": "completed",
        "video_path": source.as_posix(),
        "media_path": destination.as_posix(),
        "sampled_frame_indices": frame_indices,
        "grid": {
            "frame_count": frame_count,
            "rows": rows,
            "cols": cols,
            "color_order": "BGR",
            "resize": None,
            "sampling": "uniform_source_indices",
        },
    }


__all__ = [
    "FrameReader",
    "ImageWriter",
    "prepare_uniform_vlm_media_grid",
    "prepare_vlm_media_grids",
]
