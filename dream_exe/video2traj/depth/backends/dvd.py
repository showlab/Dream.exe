"""Lazy backend for official and project fine-tuned DVD checkpoints."""

from __future__ import annotations

import copy
import importlib
import json
import math
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from dream_exe.model_assets.dvd_identity import (
    DVD_ASSET_ATTESTATION_FIELD,
    DVD_MODEL_FAMILY,
    normalize_dvd_model_identity,
)

from ...runtime.provider_origin import (
    prepend_source_roots,
    require_modules_under_roots,
)
from ..attestation import attest_dvd_asset_files


RuntimeLoader = Callable[[], Mapping[str, Any]]

_RUNTIME_KEYS = (
    "accelerator_cls",
    "omega_conf",
    "load_file",
    "training_module_cls",
    "torch",
    "functional",
)
_TOKENIZER_FILES = (
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "spiece.model",
)


def stack_video_frames_rgb(frames: Any) -> np.ndarray:
    """Return contiguous video frames with shape ``[T, H, W, 3]``."""

    value = frames
    if hasattr(value, "detach"):
        value = value.detach()
    if hasattr(value, "cpu"):
        value = value.cpu()
    if hasattr(value, "numpy"):
        value = value.numpy()

    if isinstance(value, np.ndarray):
        array = np.asarray(value)
    else:
        try:
            items = list(value)
        except TypeError as error:
            raise TypeError(
                "Video frames must be an array or an iterable of RGB frames."
            ) from error
        if not items:
            raise ValueError("Video frames must contain at least one frame.")
        array = np.stack([np.asarray(item) for item in items], axis=0)

    if array.ndim != 4:
        raise ValueError(
            f"Unexpected frames shape: {tuple(array.shape)}, expected [T,H,W,3] RGB."
        )
    if array.shape[0] <= 0:
        raise ValueError("Video frames must contain at least one frame.")
    if array.shape[-1] != 3:
        raise ValueError(
            f"Unexpected frames shape: {tuple(array.shape)}, expected [T,H,W,3] RGB."
        )
    return np.ascontiguousarray(array)


def compute_scale_and_shift(
    prediction: Any,
    target: Any,
    mask: Any | None = None,
) -> tuple[float, float]:
    """Fit ``target ~= scale * prediction + shift`` by least squares."""

    predicted = np.asarray(prediction, dtype=np.float32)
    reference = np.asarray(target, dtype=np.float32)
    if predicted.shape != reference.shape:
        raise ValueError(
            "Prediction and target must have identical shapes, "
            f"got {predicted.shape} and {reference.shape}."
        )
    if mask is None:
        weights = np.ones_like(predicted, dtype=np.float32)
    else:
        weights = np.asarray(mask, dtype=np.float32)
        if weights.shape != predicted.shape:
            try:
                weights = np.broadcast_to(weights, predicted.shape)
            except ValueError as error:
                raise ValueError(
                    "Alignment mask must be broadcastable to the "
                    f"prediction shape {predicted.shape}."
                ) from error

    a_00 = float(np.sum(weights * predicted * predicted))
    a_01 = float(np.sum(weights * predicted))
    a_11 = float(np.sum(weights))
    b_0 = float(np.sum(weights * predicted * reference))
    b_1 = float(np.sum(weights * reference))
    determinant = a_00 * a_11 - a_01 * a_01
    if abs(determinant) > 1e-12:
        scale = (a_11 * b_0 - a_01 * b_1) / determinant
        shift = (-a_01 * b_0 + a_00 * b_1) / determinant
        return float(scale), float(shift)
    return 1.0, 0.0


def window_indices(
    num_frames: int,
    window_size: int,
    overlap: int,
) -> list[tuple[int, int]]:
    """Partition a video into ordered half-open temporal windows."""

    frame_count = int(num_frames)
    size = int(window_size)
    shared = int(overlap)
    if frame_count <= 0:
        raise ValueError("num_frames must be > 0.")
    if size <= 0:
        raise ValueError("window_size must be > 0.")
    if shared < 0 or shared >= size:
        raise ValueError("overlap must satisfy 0 <= overlap < window_size.")

    windows: list[tuple[int, int]] = []
    start = 0
    while start < frame_count:
        end = min(start + size, frame_count)
        if end == frame_count and end - start < size:
            start = max(0, end - size)
        windows.append((start, end))
        if end >= frame_count:
            break
        start = end - shared
    return windows


def pad_time_mod4(
    video: Any,
    *,
    torch: Any,
) -> tuple[Any, int]:
    """Repeat the final frame until the temporal size has form ``4k + 1``."""

    if getattr(video, "ndim", None) != 5:
        raise ValueError(
            "DVD input video must have shape [B,T,C,H,W], "
            f"got {getattr(video, 'shape', None)}."
        )
    original = int(video.shape[1])
    if original <= 0:
        raise ValueError("DVD input video must contain at least one frame.")
    padding = (4 - ((original - 1) % 4)) % 4
    if padding == 0:
        return video, original
    tail = video[:, -1:, ...].repeat(1, padding, 1, 1, 1)
    return torch.cat((video, tail), dim=1), original


def _multiple_of_16(value: float) -> int:
    return max(16, int(math.ceil(float(value) / 16.0)) * 16)


def resize_for_training_scale(
    video: Any,
    *,
    functional: Any,
    target_h: int,
    target_w: int,
) -> tuple[Any, tuple[int, int]]:
    """Resize ``[B,T,C,H,W]`` spatially while preserving aspect ratio."""

    if getattr(video, "ndim", None) != 5:
        raise ValueError(
            "DVD input video must have shape [B,T,C,H,W], "
            f"got {getattr(video, 'shape', None)}."
        )
    requested_h = int(target_h)
    requested_w = int(target_w)
    if requested_h <= 0 or requested_w <= 0:
        raise ValueError("DVD resize dimensions must be > 0.")

    batch, frames, channels, height, width = map(int, video.shape)
    if height <= 0 or width <= 0:
        raise ValueError("DVD input spatial dimensions must be > 0.")
    scale = max(requested_h / height, requested_w / width)
    resized_h = _multiple_of_16(height * scale)
    resized_w = _multiple_of_16(width * scale)
    if resized_h == height and resized_w == width:
        return video, (height, width)
    flat = video.reshape(batch * frames, channels, height, width)
    resized = functional.interpolate(
        flat,
        size=(resized_h, resized_w),
        mode="bilinear",
        align_corners=False,
    )
    return (
        resized.reshape(
            batch,
            frames,
            channels,
            resized_h,
            resized_w,
        ),
        (height, width),
    )


def resize_depth_back(
    depth: Any,
    original_size: tuple[int, int],
    *,
    torch: Any,
    functional: Any,
) -> np.ndarray:
    """Resize one float32 ``[T,H,W,C]`` depth array to its source size."""

    array = np.asarray(depth, dtype=np.float32)
    if array.ndim != 4:
        raise ValueError(f"DVD depth resize expects [T,H,W,C], got {array.shape}.")
    height, width = (int(original_size[0]), int(original_size[1]))
    if height <= 0 or width <= 0:
        raise ValueError("Original depth dimensions must be > 0.")

    tensor = torch.from_numpy(np.ascontiguousarray(array))
    tensor = tensor.permute(0, 3, 1, 2).float()
    resized = functional.interpolate(
        tensor,
        size=(height, width),
        mode="bilinear",
        align_corners=False,
    )
    output = resized.permute(0, 2, 3, 1).detach().cpu().numpy()
    output = np.ascontiguousarray(output)
    return output


def _default_runtime_loader(source_root: Path) -> Mapping[str, Any]:
    prepend_source_roots((source_root,))
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    accelerator_cls = importlib.import_module("accelerate").Accelerator
    omega_conf = importlib.import_module("omegaconf").OmegaConf
    load_file = importlib.import_module("safetensors.torch").load_file
    training_module = importlib.import_module(
        "examples.wanvideo.model_training.WanTrainingModule"
    )
    require_modules_under_roots(
        (training_module,),
        source_roots=(source_root,),
        provider="DVD",
    )
    training_module_cls = training_module.WanTrainingModule
    return {
        "torch": torch,
        "functional": functional,
        "accelerator_cls": accelerator_cls,
        "omega_conf": omega_conf,
        "load_file": load_file,
        "training_module_cls": training_module_cls,
    }


def _validate_identity(
    model_family: str,
    provenance: Mapping[str, Any] | None,
) -> tuple[str, dict[str, Any]]:
    return normalize_dvd_model_identity(
        model_family,
        provenance,
        source="DVD backend",
    )


def _explicit_path(value: str | Path, *, field: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"DVD {field} must be explicitly provided.")
    return Path(text).expanduser().resolve(strict=False)


def _load_dit_checkpoint(model: Any, state: Mapping[str, Any]) -> None:
    """Load only the canonical DVD DiT namespace, fail-closed."""

    prefix = "pipe.dit."
    dit_state = {
        str(key)[len(prefix) :]: value
        for key, value in state.items()
        if str(key).startswith(prefix)
    }
    model.pipe.dit.load_state_dict(dit_state, strict=True)


class DVDDepthBackend:
    """Lazy official or project-fine-tuned DVD inference backend."""

    def __init__(
        self,
        *,
        ckpt_root: str | Path,
        model_config_path: str | Path,
        source_root: str | Path,
        checkpoints_root: str | Path,
        model_family: str = DVD_MODEL_FAMILY,
        model_provenance: Mapping[str, Any] | None = None,
        device: str = "cuda",
        resize_height: int = 480,
        resize_width: int = 640,
        window_size: int = 81,
        overlap: int = 9,
        scale_only_alignment: bool = False,
        channel_reduce: str = "mean",
        invert_disparity_to_depth: bool = True,
        min_disparity: float = 1e-4,
        allow_download: bool = False,
        runtime_loader: RuntimeLoader | None = None,
        **_unused: Any,
    ) -> None:
        self.device = str(device)
        self.ckpt_root = _explicit_path(ckpt_root, field="ckpt_root")
        self.model_config_path = _explicit_path(
            model_config_path,
            field="model_config_path",
        )
        self.source_root = _explicit_path(source_root, field="source_root")
        self.checkpoints_root = _explicit_path(
            checkpoints_root,
            field="checkpoints_root",
        )
        self.model_family, self.model_provenance = _validate_identity(
            model_family,
            model_provenance,
        )
        self.resize_height = int(resize_height)
        self.resize_width = int(resize_width)
        self.window_size = int(window_size)
        self.overlap = int(overlap)
        self.scale_only_alignment = bool(scale_only_alignment)
        self.channel_reduce = str(channel_reduce).strip().lower()
        self.invert_disparity_to_depth = bool(invert_disparity_to_depth)
        self.min_disparity = float(min_disparity)
        if type(allow_download) is not bool:
            raise ValueError("DVD allow_download must be a boolean.")
        self.allow_download = allow_download
        self._runtime_loader = runtime_loader
        self._runtime: Mapping[str, Any] | None = None
        self._model: Any | None = None
        self._resolved_checkpoint_file: str | None = None
        self._resolved_model_config_path: str | None = None
        self._wan_asset_mode: str | None = None

        if self.resize_height <= 0 or self.resize_width <= 0:
            raise ValueError("DVD resize dimensions must be > 0.")
        if self.window_size <= 0:
            raise ValueError("DVD window_size must be > 0.")
        if self.overlap < 0 or self.overlap >= self.window_size:
            raise ValueError("DVD overlap must satisfy 0 <= overlap < window_size.")
        if self.channel_reduce not in {"first", "mean"}:
            raise ValueError("DVD channel_reduce must be 'first' or 'mean'.")
        if self.min_disparity <= 0.0:
            raise ValueError("DVD min_disparity must be > 0.")

    def _resolve_paths(self) -> tuple[str, str]:
        root = self.ckpt_root
        if not root.is_dir():
            raise FileNotFoundError(
                f"DVD checkpoint directory not found: {root.as_posix()}. "
                "Place DVD weights under ./checkpoints/DVD."
            )
        preferred = root / "model.safetensors"
        if preferred.is_file():
            checkpoint = preferred
        else:
            candidates = sorted(
                path for path in root.glob("*.safetensors") if path.is_file()
            )
            if len(candidates) != 1:
                raise FileNotFoundError(
                    "DVD checkpoint file not found under "
                    f"{root.as_posix()}. Expected model.safetensors "
                    "or exactly one *.safetensors file."
                )
            checkpoint = candidates[0]

        config = self.model_config_path
        if not config.is_file():
            raise FileNotFoundError(
                f"DVD model_config.yaml not found. Looked at {config.as_posix()}."
            )
        self._resolved_checkpoint_file = checkpoint.resolve().as_posix()
        self._resolved_model_config_path = config.resolve().as_posix()
        return (
            self._resolved_checkpoint_file,
            self._resolved_model_config_path,
        )

    def _local_wan_bundle(
        self,
    ) -> tuple[list[str], str] | None:
        root = self.checkpoints_root / "Wan-AI" / "Wan2.1-T2V-1.3B"
        dit_candidates = sorted(
            path
            for path in root.glob("diffusion_pytorch_model*.safetensors")
            if path.is_file()
        )
        vae = root / "Wan2.1_VAE.pth"
        tokenizer = root / "google" / "umt5-xxl"
        required = [vae]
        required.extend(tokenizer / name for name in _TOKENIZER_FILES)
        if not dit_candidates or not all(path.is_file() for path in required):
            return None
        return (
            [
                dit_candidates[0].resolve().as_posix(),
                vae.resolve().as_posix(),
            ],
            tokenizer.resolve().as_posix(),
        )

    def _load_runtime(self) -> Mapping[str, Any]:
        loader = self._runtime_loader
        runtime = (
            loader()
            if loader is not None
            else _default_runtime_loader(self.source_root)
        )
        if not isinstance(runtime, Mapping):
            raise RuntimeError("DVD runtime loader must return a mapping.")
        missing = [name for name in _RUNTIME_KEYS if name not in runtime]
        if missing:
            raise RuntimeError("DVD runtime loader is missing: " + ", ".join(missing))
        self._runtime = dict(runtime)
        return self._runtime

    def _load(self) -> Any:
        if self._model is not None:
            return self._model
        checkpoint, config_path = self._resolve_paths()
        if DVD_ASSET_ATTESTATION_FIELD in self.model_provenance:
            self.model_provenance = attest_dvd_asset_files(
                self.model_provenance,
                checkpoint_file=checkpoint,
                model_config_file=config_path,
            )
        bundle = self._local_wan_bundle()
        if bundle is None and not self.allow_download:
            wan_root = self.checkpoints_root / "Wan-AI" / "Wan2.1-T2V-1.3B"
            raise FileNotFoundError(
                "DVD local Wan/UMT5 asset bundle is incomplete under "
                f"{wan_root.as_posix()}. Network download fallback is "
                "disabled by default; provide the complete local bundle or "
                "set dvd.allow_download=true explicitly."
            )
        runtime = self._load_runtime()
        args = runtime["omega_conf"].load(config_path)
        if bundle is None:
            model_paths = None
            tokenizer_path = None
            skip_download = False
            self._wan_asset_mode = "explicit_download_opt_in"
            origin_paths = getattr(
                args,
                "model_id_with_origin_paths",
                None,
            )
        else:
            paths, tokenizer_path = bundle
            model_paths = json.dumps(paths)
            skip_download = True
            self._wan_asset_mode = "local_bundle"
            origin_paths = None

        accelerator = runtime["accelerator_cls"]()
        model = runtime["training_module_cls"](
            accelerator=accelerator,
            args=args,
            local_model_path=self.checkpoints_root.as_posix(),
            lora_base_model=getattr(args, "lora_base_model", None),
            lora_rank=getattr(args, "lora_rank", None),
            model_id_with_origin_paths=origin_paths,
            model_paths=model_paths,
            skip_download=skip_download,
            tokenizer_path=tokenizer_path,
            trainable_models=None,
            use_gradient_checkpointing=False,
        )
        state = runtime["load_file"](checkpoint, device="cpu")
        _load_dit_checkpoint(model, dict(state))
        model.merge_lora_layer()
        moved = model.to(self.device)
        if moved is not None:
            model = moved
        evaluated = model.eval()
        if evaluated is not None:
            model = evaluated
        self._model = model
        return model

    @staticmethod
    def _pipe_depth(output: Any) -> np.ndarray:
        if isinstance(output, Mapping):
            if "depth" not in output:
                raise TypeError("DVD pipeline output is missing 'depth'.")
            value = output["depth"]
        else:
            value = getattr(output, "depth", None)
            if value is None:
                raise TypeError("DVD pipeline output is missing 'depth'.")
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "cpu"):
            value = value.cpu()
        if hasattr(value, "numpy"):
            value = value.numpy()
        depth = np.asarray(value)
        if depth.ndim == 5:
            if depth.shape[0] != 1:
                raise ValueError(
                    f"DVD output batch size must be 1, got {depth.shape[0]}."
                )
            depth = depth[0]
        if depth.ndim != 4:
            raise ValueError(
                "DVD output must have shape [1,T,H,W,C] or [T,H,W,C], "
                f"got {depth.shape}."
            )
        return depth

    def _reduce_channels(self, depth: np.ndarray) -> np.ndarray:
        channels = int(depth.shape[-1])
        if channels not in {1, 3}:
            raise ValueError(
                f"DVD depth output channel count must be 1 or 3, got {depth.shape}"
            )
        if channels == 1 or self.channel_reduce == "first":
            reduced = depth[..., 0]
        else:
            reduced = np.mean(depth, axis=-1)
        reduced = np.asarray(reduced, dtype=np.float32)
        if self.invert_disparity_to_depth:
            minimum = float(np.nanmin(reduced))
            if not np.isfinite(minimum):
                raise ValueError("DVD depth output contains no finite values.")
            if minimum <= 0.0:
                reduced = reduced - minimum + float(self.min_disparity)
            reduced = 1.0 / np.maximum(reduced, self.min_disparity)
        return np.asarray(reduced, dtype=np.float32)

    def _scale_only(
        self,
        prediction: np.ndarray,
        target: np.ndarray,
    ) -> float:
        current = np.asarray(prediction, dtype=np.float32)
        reference = np.asarray(target, dtype=np.float32)
        denominator = float(np.sum(current * current) + 1e-6)
        return float(np.sum(current * reference) / denominator)

    def infer(
        self,
        frames: Any,
        target_fps: float,
        *,
        fp32: bool = False,
        input_size: int = 512,
        intrinsics: np.ndarray | None = None,
        extrinsics: np.ndarray | None = None,
    ) -> dict[str, Any]:
        """Infer one affine depth map for every input RGB frame."""

        del fp32, input_size, intrinsics, extrinsics
        rgb = stack_video_frames_rgb(frames)
        model = self._load()
        if self._runtime is None:
            raise RuntimeError("DVD runtime is unavailable after model loading.")
        torch = self._runtime["torch"]
        functional = self._runtime["functional"]

        tensor = torch.from_numpy(rgb).permute(0, 3, 1, 2).float() / 255.0
        tensor = tensor.unsqueeze(0)
        tensor, original_size = resize_for_training_scale(
            tensor,
            functional=functional,
            target_h=self.resize_height,
            target_w=self.resize_width,
        )
        tensor = tensor.to(self.device)

        frame_count = int(tensor.shape[1])
        stitched_raw: np.ndarray | None = None
        stitched_end = 0
        context = torch.inference_mode() if hasattr(torch, "inference_mode") else None
        windows = window_indices(
            frame_count,
            self.window_size,
            self.overlap,
        )

        if context is None:
            context_manager = _NullContext()
        else:
            context_manager = context
        with context_manager:
            for start, end in windows:
                window = tensor[:, start:end]
                padded, original_frames = pad_time_mod4(
                    window,
                    torch=torch,
                )
                output = model.pipe(
                    batch_size=int(padded.shape[0]),
                    cfg_scale=1,
                    denoise_step=model.args.denoise_step,
                    extra_image_frame_index=torch.ones(
                        (
                            int(padded.shape[0]),
                            int(padded.shape[1]),
                        ),
                        device=model.pipe.device,
                    ),
                    extra_images=padded,
                    height=int(padded.shape[-2]),
                    input_image=padded[:, 0],
                    input_video=padded,
                    mode=model.args.mode,
                    negative_prompt=[""] * int(padded.shape[0]),
                    num_frames=int(padded.shape[1]),
                    prompt=[""] * int(padded.shape[0]),
                    seed=0,
                    tiled=False,
                    width=int(padded.shape[-1]),
                )
                raw = np.asarray(
                    self._pipe_depth(output)[:original_frames],
                    dtype=np.float32,
                )
                if stitched_raw is None:
                    stitched_raw = raw
                else:
                    shared = max(0, stitched_end - start)
                    if shared:
                        target = stitched_raw[-shared:]
                        prediction = raw[:shared]
                        if self.scale_only_alignment:
                            scale = self._scale_only(prediction, target)
                            shift = 0.0
                        else:
                            scale, shift = compute_scale_and_shift(
                                prediction,
                                target,
                            )
                        scale = float(np.clip(scale, 0.7, 1.5))
                        raw = scale * raw + shift
                        raw[raw < 0] = 0.0
                        blend = np.linspace(
                            0.0,
                            1.0,
                            shared,
                            dtype=np.float32,
                        ).reshape(shared, 1, 1, 1)
                        stitched_raw[-shared:] = (1.0 - blend) * target + blend * raw[
                            :shared
                        ]
                    stitched_raw = np.concatenate(
                        (stitched_raw, raw[shared:]),
                        axis=0,
                    )
                stitched_end = end

        if stitched_raw is None or stitched_raw.shape[0] == 0:
            raise RuntimeError("[Depth] Model returned empty depths.")
        restored = resize_depth_back(
            stitched_raw,
            original_size,
            torch=torch,
            functional=functional,
        )
        raw_output_channels = int(restored.shape[-1])
        combined = self._reduce_channels(restored)
        metadata = {
            "model": "dvd",
            "ckpt_root": self.ckpt_root.as_posix(),
            "checkpoint_file": self._resolved_checkpoint_file,
            "model_config_path": self._resolved_model_config_path,
            "checkpoints_root": self.checkpoints_root.as_posix(),
            "input_color_order": "rgb",
            "resize_height": self.resize_height,
            "resize_width": self.resize_width,
            "window_size": self.window_size,
            "overlap": self.overlap,
            "scale_only_alignment": self.scale_only_alignment,
            "channel_reduce": self.channel_reduce,
            "invert_disparity_to_depth": (self.invert_disparity_to_depth),
            "min_disparity": self.min_disparity,
            "raw_output_channels": raw_output_channels,
            "raw_output_representation": "disparity_like",
            "model_family": self.model_family,
            "fine_tuned": self.model_family == DVD_MODEL_FAMILY,
            "allow_download": self.allow_download,
            "wan_asset_mode": self._wan_asset_mode,
            "model_provenance": copy.deepcopy(self.model_provenance),
        }
        return {
            "depths": [
                np.ascontiguousarray(frame, dtype=np.float32) for frame in combined
            ],
            "fps": float(target_fps),
            "meta": metadata,
            "depth_space": "affine",
            "fps_source": "target_fps",
            "valid_masks": None,
        }


class _NullContext:
    def __enter__(self) -> None:
        return None

    def __exit__(
        self,
        _error_type: type[BaseException] | None,
        _error: BaseException | None,
        _traceback: Any,
    ) -> bool:
        return False


def build_dvd_backend_registry(
    *,
    source_root: str | Path,
    checkpoints_root: str | Path,
    runtime_loader: RuntimeLoader | None = None,
) -> dict[str, Callable[..., DVDDepthBackend]]:
    """Build a lazy DVD registry whose external resources are immutable."""

    owned_source = _explicit_path(source_root, field="source_root")
    owned_checkpoints = _explicit_path(
        checkpoints_root,
        field="checkpoints_root",
    )

    def factory(**kwargs: Any) -> DVDDepthBackend:
        protected = {
            "source_root",
            "checkpoints_root",
            "runtime_loader",
        }
        replaced = sorted(protected.intersection(kwargs))
        if replaced:
            raise ValueError(
                "DVD model kwargs cannot replace registry-owned resources: "
                + ", ".join(replaced)
            )
        return DVDDepthBackend(
            **kwargs,
            source_root=owned_source,
            checkpoints_root=owned_checkpoints,
            runtime_loader=runtime_loader,
        )

    return {"dvd": factory}


DVD_BACKEND_REGISTRY = {
    "dvd": DVDDepthBackend,
}


__all__ = [
    "DVD_BACKEND_REGISTRY",
    "DVDDepthBackend",
    "build_dvd_backend_registry",
    "compute_scale_and_shift",
    "pad_time_mod4",
    "resize_depth_back",
    "resize_for_training_scale",
    "stack_video_frames_rgb",
    "window_indices",
]
