"""Lazy integration for the supported Video Depth Anything backend."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib
from pathlib import Path
from typing import Any

import numpy as np

from ...runtime.provider_origin import (
    prepend_source_roots,
    require_modules_under_roots,
)


RuntimeLoader = Callable[[], Mapping[str, Any]]


VDA_DEFAULT_PROVIDER_MODULE = "video_depth_anything.video_depth"
_NETWORK_SHAPES: dict[str, dict[str, Any]] = {
    "vits": {
        "encoder": "vits",
        "features": 64,
        "out_channels": [48, 96, 192, 384],
    },
    "vitb": {
        "encoder": "vitb",
        "features": 128,
        "out_channels": [96, 192, 384, 768],
    },
    "vitl": {
        "encoder": "vitl",
        "features": 256,
        "out_channels": [256, 512, 1024, 1024],
    },
}


def stack_video_frames_rgb(frames: Any) -> np.ndarray:
    """Return a contiguous ``[T, H, W, 3]`` RGB array."""

    if isinstance(frames, np.ndarray):
        stacked = frames
    elif hasattr(frames, "detach"):
        stacked = frames.detach().cpu().numpy()
    else:
        stacked = np.stack(frames, axis=0)

    array = np.asarray(stacked)
    if array.ndim != 4 or array.shape[-1] != 3:
        raise ValueError(
            f"Unexpected frames shape: {array.shape}, expected [T,H,W,3] RGB."
        )
    return np.ascontiguousarray(array)


def _make_runtime(
    source_roots: Sequence[str],
    module_name: str,
) -> dict[str, Any]:
    resolved_roots = []
    for root in source_roots:
        resolved = Path(root).expanduser().resolve()
        if not resolved.is_dir():
            raise FileNotFoundError(
                "VDA source directory not found: "
                f"{resolved.as_posix()}. Run python integrations/setup.py vda or "
                "provide an installed compatible provider."
            )
        resolved_roots.append(resolved)
    prepend_source_roots(resolved_roots)

    try:
        provider = importlib.import_module(module_name)
    except ImportError as error:
        raise RuntimeError(
            f"Cannot import VDA provider module {module_name!r}. "
            "Run python integrations/setup.py vda and pass its checkout as "
            "depth.source_root."
        ) from error
    require_modules_under_roots(
        (provider,),
        source_roots=source_roots,
        provider="VDA",
    )
    torch = importlib.import_module("torch")
    return {
        "torch": torch,
        "model_cls": provider.VideoDepthAnything,
    }


class VideoDepthAnythingBackend:
    """Load one explicit VDA checkpoint and expose depth-model predictions."""

    def __init__(
        self,
        *,
        encoder: str = "vitl",
        metric: bool = True,
        device: str = "cuda",
        ckpt_root: str | Path,
        source_roots: Sequence[str | Path] = (),
        module_name: str = VDA_DEFAULT_PROVIDER_MODULE,
        runtime_loader: RuntimeLoader | None = None,
    ) -> None:
        self.encoder = str(encoder)
        self.metric = bool(metric)
        self.device = str(device)
        self.ckpt_root = Path(ckpt_root).expanduser().resolve().as_posix()
        self.source_roots = tuple(
            Path(root).expanduser().resolve().as_posix() for root in source_roots
        )
        self.module_name = str(module_name)
        self._runtime_loader = runtime_loader
        self._runtime: dict[str, Any] | None = None
        self._model: Any = None

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime

        supplied = (
            self._runtime_loader()
            if self._runtime_loader is not None
            else _make_runtime(self.source_roots, self.module_name)
        )
        runtime = dict(supplied)
        missing = sorted({"model_cls", "torch"}.difference(runtime))
        if missing:
            raise RuntimeError("VDA runtime loader is missing: " + ", ".join(missing))
        self._runtime = runtime
        return runtime

    def _load(self) -> None:
        if self._model is not None:
            return

        runtime = self._load_runtime()
        constructor_args = dict(_NETWORK_SHAPES[self.encoder])
        constructor_args["metric"] = self.metric
        model = runtime["model_cls"](**constructor_args)

        prefix = (
            "metric_video_depth_anything" if self.metric else "video_depth_anything"
        )
        checkpoint = Path(self.ckpt_root) / f"{prefix}_{self.encoder}.pth"
        weights = runtime["torch"].load(
            checkpoint.as_posix(),
            map_location="cpu",
        )
        model.load_state_dict(weights, strict=True)
        target_device = runtime["torch"].device(self.device)
        self._model = model.to(target_device).eval()

    def infer(
        self,
        frames: Any,
        target_fps: float,
        *,
        fp32: bool = False,
        input_size: int = 512,
        intrinsics: Any = None,
        extrinsics: Any = None,
    ) -> dict[str, Any]:
        """Infer per-frame VDA depth while retaining current adapter metadata."""

        del intrinsics, extrinsics
        self._load()
        video = stack_video_frames_rgb(frames)
        depths, output_fps = self._model.infer_video_depth(
            video,
            target_fps,
            input_size=input_size,
            device=self.device,
            fp32=fp32,
        )
        depth_array = np.asarray(depths)
        if depth_array.ndim != 3:
            raise ValueError("Unexpected depths format returned by model.")

        return {
            "depths": [depth.astype(np.float32) for depth in depth_array],
            "fps": float(output_fps),
            "fps_source": "target_fps",
            "depth_space": "metric" if self.metric else "affine",
            "meta": {
                "model": "vda",
                "encoder": self.encoder,
                "metric": self.metric,
                "input_size": input_size,
                "input_color_order": "rgb",
            },
            "valid_masks": None,
        }


VDA_BACKEND_REGISTRY = {
    "vda": VideoDepthAnythingBackend,
}


__all__ = [
    "RuntimeLoader",
    "VDA_BACKEND_REGISTRY",
    "VDA_DEFAULT_PROVIDER_MODULE",
    "VideoDepthAnythingBackend",
    "stack_video_frames_rgb",
]
