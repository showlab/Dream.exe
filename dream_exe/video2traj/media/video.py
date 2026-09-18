"""Video decoding helpers at the simulator-independent algorithm boundary.

The current implementation prefers Decord and falls back to OpenCV.  This
module preserves both observable sampling paths while loading either optional
runtime only when the callable is used.  Tests and embedding applications may
inject either backend explicitly.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np


def ensure_even(value: int) -> int:
    """Return ``value`` unchanged when even, otherwise the next integer."""

    return value if value % 2 == 0 else value + 1


def _load_decord_backend() -> tuple[Callable[..., Any], Callable[[int], Any]]:
    from decord import VideoReader, cpu

    return VideoReader, cpu


def _load_cv2_backend() -> Any:
    import cv2

    return cv2


def _read_with_decord(
    video_path: str,
    *,
    process_length: int,
    target_fps: float,
    max_res: int,
    reader_factory: Callable[..., Any],
    cpu_factory: Callable[[int], Any],
) -> tuple[np.ndarray, float]:
    reader = reader_factory(video_path, ctx=cpu_factory(0))
    original_height, original_width = reader.get_batch([0]).shape[1:3]
    height = original_height
    width = original_width
    if max_res > 0 and max(height, width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = ensure_even(round(original_height * scale))
        width = ensure_even(round(original_width * scale))

    reader = reader_factory(
        video_path,
        ctx=cpu_factory(0),
        width=width,
        height=height,
    )
    average_fps = reader.get_avg_fps()
    fps = average_fps if target_fps == -1 else target_fps
    stride = max(round(average_fps / fps), 1)
    frame_indices = list(range(0, len(reader), stride))
    if process_length != -1 and process_length < len(frame_indices):
        frame_indices = frame_indices[:process_length]
    frames = reader.get_batch(frame_indices).asnumpy()
    return frames, fps


def _read_with_cv2(
    video_path: str,
    *,
    process_length: int,
    target_fps: float,
    max_res: int,
    cv2_module: Any,
) -> tuple[np.ndarray, float]:
    capture = cv2_module.VideoCapture(video_path)
    original_fps = capture.get(cv2_module.CAP_PROP_FPS)
    original_height = int(capture.get(cv2_module.CAP_PROP_FRAME_HEIGHT))
    original_width = int(capture.get(cv2_module.CAP_PROP_FRAME_WIDTH))

    if max_res > 0 and max(original_height, original_width) > max_res:
        scale = max_res / max(original_height, original_width)
        height = round(original_height * scale)
        width = round(original_width * scale)

    fps = original_fps if target_fps < 0 else target_fps
    stride = max(round(original_fps / fps), 1)

    frames: list[np.ndarray] = []
    frame_count = 0
    while capture.isOpened():
        ok, frame = capture.read()
        if not ok or (process_length > 0 and frame_count >= process_length):
            break
        if frame_count % stride == 0:
            frame = cv2_module.cvtColor(
                frame,
                cv2_module.COLOR_BGR2RGB,
            )
            if max_res > 0 and max(original_height, original_width) > max_res:
                frame = cv2_module.resize(frame, (width, height))
            frames.append(frame)
        frame_count += 1
    capture.release()
    return np.stack(frames, axis=0), fps


def read_video_frames(
    video_path: str | Path,
    process_length: int,
    target_fps: float = -1,
    max_res: int = -1,
    *,
    backend: str = "auto",
    decord_reader_factory: Callable[..., Any] | None = None,
    decord_cpu_factory: Callable[[int], Any] | None = None,
    cv2_module: Any | None = None,
) -> tuple[np.ndarray, float]:
    """Decode RGB frames with current Decord/OpenCV sampling semantics.

    ``backend`` accepts ``"auto"``, ``"decord"``, or ``"opencv"``.  Auto
    preserves the current preference for Decord.  Supplying backend objects is
    useful for embedding, tests, and environments that manage optional
    dependencies outside this package.
    """

    selected = str(backend or "auto").strip().lower()
    if selected not in {"auto", "decord", "opencv", "cv2"}:
        raise ValueError(f"unsupported video backend: {backend}")
    path = str(video_path)

    if decord_reader_factory is not None:
        if selected in {"opencv", "cv2"}:
            raise ValueError(
                f"decord_reader_factory cannot be used with backend={selected}"
            )
        cpu_factory = decord_cpu_factory or (lambda index: index)
        return _read_with_decord(
            path,
            process_length=int(process_length),
            target_fps=float(target_fps),
            max_res=int(max_res),
            reader_factory=decord_reader_factory,
            cpu_factory=cpu_factory,
        )

    if selected in {"auto", "decord"}:
        try:
            reader_factory, cpu_factory = _load_decord_backend()
        except Exception:
            if selected == "decord":
                raise
        else:
            return _read_with_decord(
                path,
                process_length=int(process_length),
                target_fps=float(target_fps),
                max_res=int(max_res),
                reader_factory=reader_factory,
                cpu_factory=cpu_factory,
            )

    backend_module = cv2_module if cv2_module is not None else _load_cv2_backend()
    return _read_with_cv2(
        path,
        process_length=int(process_length),
        target_fps=float(target_fps),
        max_res=int(max_res),
        cv2_module=backend_module,
    )


__all__ = ["ensure_even", "read_video_frames"]
