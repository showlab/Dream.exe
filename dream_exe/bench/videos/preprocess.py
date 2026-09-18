"""Normalize candidate videos before they enter video2traj."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
import subprocess
from typing import Any

from ...generation.sources import (
    inspect_normalized_video,
    normalize_generated_video,
)


DEFAULT_PREPROCESS_MODE = "resize"


def pipeline_video_preprocess_spec(
    output_contract: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve the benchmark-owned video2traj preprocessing contract."""

    width = output_contract.get("pipeline_width")
    height = output_contract.get("pipeline_height")
    if isinstance(width, bool) or not isinstance(width, int) or width <= 0:
        raise ValueError("generation output_contract.pipeline_width must be positive")
    if isinstance(height, bool) or not isinstance(height, int) or height <= 0:
        raise ValueError("generation output_contract.pipeline_height must be positive")
    if width != height:
        raise ValueError("video2traj currently requires a square pipeline video")
    return {
        "mode": DEFAULT_PREPROCESS_MODE,
        "width": width,
        "height": height,
        "fps": "source",
        "codec": "h264",
        "pixel_format": "yuv420p",
        "audio": "removed",
    }


def preprocess_video_for_video2traj(
    source: str | Path,
    destination: str | Path,
    *,
    output_contract: Mapping[str, Any],
    overwrite: bool = False,
    dry_run: bool = False,
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Create and validate the derived MP4 consumed by video2traj."""

    spec = pipeline_video_preprocess_spec(output_contract)
    result = normalize_generated_video(
        source,
        destination,
        mode=str(spec["mode"]),
        size=int(spec["width"]),
        fps="",
        overwrite=overwrite,
        dry_run=dry_run,
        run=run,
    )
    if result.get("status") == "error":
        message = str(result.get("error", "unknown FFmpeg error"))
        raise RuntimeError(f"video preprocessing failed: {message}")
    if result.get("status") == "skip_exists":
        result = {
            **result,
            "media": inspect_normalized_video(
                destination,
                size=int(spec["width"]),
                fps="",
                run=run,
            ),
        }
    return {**result, "spec": spec}


def inspect_video2traj_input(
    path: str | Path,
    *,
    output_contract: Mapping[str, Any],
    run: Callable[..., Any] = subprocess.run,
) -> dict[str, Any]:
    """Validate a caller-supplied preprocessed video against the same contract."""

    spec = pipeline_video_preprocess_spec(output_contract)
    media = inspect_normalized_video(
        path,
        size=int(spec["width"]),
        fps="",
        run=run,
    )
    return {"status": "validated", "media": media, "spec": spec}


__all__ = [
    "DEFAULT_PREPROCESS_MODE",
    "inspect_video2traj_input",
    "pipeline_video_preprocess_spec",
    "preprocess_video_for_video2traj",
]
