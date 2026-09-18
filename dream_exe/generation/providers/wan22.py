"""Pinned subprocess adapter for the open-source Wan2.2 TI2V-5B model."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import os
from pathlib import Path
import stat
import subprocess
import sys
from typing import Any


WAN22_REPOSITORY = "https://github.com/Wan-Video/Wan2.2.git"
WAN22_REVISION = "42bf4cfaa384bc21833865abc2f9e6c0e67233dc"
WAN22_TASK = "ti2v-5B"
WAN22_MODEL_ID = "Wan-AI/Wan2.2-TI2V-5B"
WAN22_CHECKPOINT_REVISION = "921dbaf3f1674a56f47e83fb80a34bac8a8f203e"
WAN22_DEFAULT_SIZE = "1280*704"
WAN22_DEFAULT_FRAME_NUM = 121
WAN22_DEFAULT_SAMPLE_STEPS = 40
WAN22_DEFAULT_GUIDANCE_SCALE = 5.0


CommandRunner = Callable[..., subprocess.CompletedProcess[str]]
RevisionReader = Callable[[Path], str]
ConditioningTransformReader = Callable[[Path, Path], Mapping[str, Any]]


def _conditioning_transform(
    image_path: Path,
    video_path: Path,
) -> Mapping[str, Any]:
    """Describe the pinned upstream scale-to-cover and center crop."""

    from PIL import Image
    import cv2

    with Image.open(image_path) as image:
        input_width, input_height = (int(value) for value in image.size)
    capture = cv2.VideoCapture(video_path.as_posix())
    try:
        if not capture.isOpened():
            raise RuntimeError(f"cannot open generated Wan2.2 video: {video_path}")
        output_width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        output_height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = float(capture.get(cv2.CAP_PROP_FPS))
    finally:
        capture.release()
    if min(input_width, input_height, output_width, output_height) <= 0:
        raise RuntimeError("Wan2.2 conditioning/video dimensions must be positive")
    scale = max(output_width / input_width, output_height / input_height)
    resized_width = round(input_width * scale)
    resized_height = round(input_height * scale)
    crop_left = (resized_width - output_width) // 2
    crop_top = (resized_height - output_height) // 2
    if crop_left < 0 or crop_top < 0:
        raise RuntimeError("Wan2.2 output is incompatible with scale-to-cover")
    return {
        "format": "dream-exe.conditioning-image-transform",
        "algorithm": "scale_to_cover_center_crop",
        "input_size": {
            "width": input_width,
            "height": input_height,
        },
        "resized_size": {
            "width": resized_width,
            "height": resized_height,
        },
        "crop_xyxy_in_resized": [
            crop_left,
            crop_top,
            crop_left + output_width,
            crop_top + output_height,
        ],
        "output_size": {
            "width": output_width,
            "height": output_height,
        },
        "video": {
            "frame_count": frame_count,
            "fps": fps,
        },
        "source_revision": WAN22_REVISION,
    }


def _require_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} not found: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} must not be a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory")


def _require_file(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} not found: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")


def _require_python_executable(path: Path) -> None:
    """Validate a Python launcher without rejecting a virtualenv symlink.

    Virtual environments commonly expose ``bin/python`` as a symlink.  The
    launcher path itself must be preserved because resolving it to the base
    interpreter bypasses the virtual environment and its installed packages.
    ``stat()`` deliberately follows the final symlink while the command keeps
    using the original absolute path.
    """

    try:
        metadata = path.stat()
    except FileNotFoundError as error:
        raise FileNotFoundError(
            f"Wan2.2 Python executable not found: {path}"
        ) from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError("Wan2.2 Python executable must target a regular file")
    if not os.access(path, os.X_OK):
        raise PermissionError(f"Wan2.2 Python executable is not executable: {path}")


def _default_runner(
    arguments: Sequence[str],
    *,
    cwd: Path,
    environment: Mapping[str, str],
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        tuple(str(item) for item in arguments),
        cwd=cwd.as_posix(),
        env=dict(environment),
        check=False,
        capture_output=True,
        text=True,
    )


def _default_revision_reader(source_root: Path) -> str:
    completed = subprocess.run(
        ("git", "-C", source_root.as_posix(), "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def _integer(
    parameters: Mapping[str, Any],
    key: str,
    default: int,
    *,
    minimum: int,
) -> int:
    value = parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"Wan2.2 parameter {key} must be an integer")
    if value < minimum:
        raise ValueError(f"Wan2.2 parameter {key} must be at least {minimum}")
    return value


def _floating(
    parameters: Mapping[str, Any],
    key: str,
    default: float,
) -> float:
    value = parameters.get(key, default)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"Wan2.2 parameter {key} must be numeric")
    converted = float(value)
    if not (0.0 < converted < float("inf")):
        raise ValueError(f"Wan2.2 parameter {key} must be finite and positive")
    return converted


class Wan22TI2VBackend:
    """Invoke the pinned official Wan2.2 ``generate.py`` entry point.

    Source and checkpoint acquisition remain explicit setup steps.  The
    adapter performs no clone, model download, environment mutation, or bench
    discovery.
    """

    def __init__(
        self,
        *,
        source_root: str | Path,
        checkpoint_root: str | Path,
        python_executable: str | Path = sys.executable,
        runner: CommandRunner | None = None,
        revision_reader: RevisionReader | None = None,
        conditioning_transform_reader: ConditioningTransformReader | None = None,
    ) -> None:
        self.source_root = Path(source_root).expanduser().resolve(strict=False)
        self.checkpoint_root = Path(checkpoint_root).expanduser().resolve(strict=False)
        # Do not resolve this path: a virtualenv's Python launcher is normally
        # a symlink whose spelling is required for Python to discover pyvenv.cfg.
        self.python_executable = Path(python_executable).expanduser().absolute()
        self._runner = runner or _default_runner
        self._revision_reader = revision_reader or _default_revision_reader
        self._conditioning_transform_reader = (
            conditioning_transform_reader or _conditioning_transform
        )

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            "backend": "wan2.2_ti2v_5b_official_cli",
            "model_id": WAN22_MODEL_ID,
            "expected_checkpoint_revision": WAN22_CHECKPOINT_REVISION,
            "repository": WAN22_REPOSITORY,
            "revision": WAN22_REVISION,
            "task": WAN22_TASK,
        }

    def _validate_runtime(self) -> None:
        _require_directory(self.source_root, label="Wan2.2 source root")
        _require_file(
            self.source_root / "generate.py",
            label="Wan2.2 generate.py",
        )
        _require_directory(
            self.checkpoint_root,
            label="Wan2.2 checkpoint root",
        )
        _require_python_executable(self.python_executable)
        revision = str(self._revision_reader(self.source_root) or "").strip()
        if revision != WAN22_REVISION:
            raise RuntimeError(
                "Wan2.2 source revision mismatch: "
                f"expected {WAN22_REVISION}, found {revision or '<empty>'}"
            )

    def generate(
        self,
        *,
        image_path: Path,
        prompt: str,
        output_path: Path,
        seed: int,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        self._validate_runtime()
        unknown = sorted(
            set(parameters).difference(
                {
                    "size",
                    "frame_num",
                    "sample_steps",
                    "guidance_scale",
                }
            )
        )
        if unknown:
            raise ValueError(
                "unsupported Wan2.2 generation parameters: " + ", ".join(unknown)
            )
        size = str(parameters.get("size", WAN22_DEFAULT_SIZE) or "").strip()
        if size not in {"1280*704", "704*1280"}:
            raise ValueError("Wan2.2 TI2V-5B size must be 1280*704 or 704*1280")
        frame_num = _integer(
            parameters,
            "frame_num",
            WAN22_DEFAULT_FRAME_NUM,
            minimum=1,
        )
        if (frame_num - 1) % 4:
            raise ValueError("Wan2.2 frame_num must satisfy 4n+1")
        sample_steps = _integer(
            parameters,
            "sample_steps",
            WAN22_DEFAULT_SAMPLE_STEPS,
            minimum=1,
        )
        guidance_scale = _floating(
            parameters,
            "guidance_scale",
            WAN22_DEFAULT_GUIDANCE_SCALE,
        )

        command = (
            self.python_executable.as_posix(),
            "generate.py",
            "--task",
            WAN22_TASK,
            "--size",
            size,
            "--ckpt_dir",
            self.checkpoint_root.as_posix(),
            "--offload_model",
            "True",
            "--convert_model_dtype",
            "--t5_cpu",
            "--image",
            image_path.as_posix(),
            "--prompt",
            prompt,
            "--save_file",
            output_path.as_posix(),
            "--base_seed",
            str(seed),
            "--frame_num",
            str(frame_num),
            "--sample_steps",
            str(sample_steps),
            "--sample_guide_scale",
            repr(guidance_scale),
        )
        environment = dict(os.environ)
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        completed = self._runner(
            command,
            cwd=self.source_root,
            environment=environment,
        )
        if completed.returncode != 0:
            stderr = str(completed.stderr or "").strip()[-4000:]
            raise RuntimeError(
                "Wan2.2 generation failed with exit code "
                f"{completed.returncode}: {stderr or '<no stderr>'}"
            )
        conditioning_transform = dict(
            self._conditioning_transform_reader(image_path, output_path)
        )
        return {
            "backend": "wan2.2_ti2v_5b_official_cli",
            "frame_num": frame_num,
            "guidance_scale": guidance_scale,
            "sample_steps": sample_steps,
            "size": size,
            "source_revision": WAN22_REVISION,
            "conditioning_transform": conditioning_transform,
        }


__all__ = [
    "WAN22_DEFAULT_FRAME_NUM",
    "WAN22_DEFAULT_GUIDANCE_SCALE",
    "WAN22_DEFAULT_SAMPLE_STEPS",
    "WAN22_DEFAULT_SIZE",
    "WAN22_CHECKPOINT_REVISION",
    "WAN22_MODEL_ID",
    "WAN22_REPOSITORY",
    "WAN22_REVISION",
    "WAN22_TASK",
    "Wan22TI2VBackend",
]
