"""DiffSynth adapter for the released Dream.exe Wan2.2 I2V A14B LoRAs."""

from __future__ import annotations

import json
import hashlib
import subprocess
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..video import BaseImageToVideoBackend


MODEL_REPOSITORY = "kaimingyang/VideoModel_as_RoboPolicy_for_Dream.exe"
MODEL_REVISION = "f90d07b69c6c434902617b6305ad83731fe926ff"
BASE_REPOSITORY = "Wan-AI/Wan2.2-I2V-A14B"
BASE_REVISION = "206a9ee1b7bfaaf8f7e4d81335650533490646a3"
DIFFSYNTH_REVISION = "7686e54d41d25c0e8ed5f1318acc23b6bb832654"


def _regular_file(path: Path) -> Path:
    if path.is_symlink() or not path.is_file():
        raise FileNotFoundError(f"required regular model file is missing: {path}")
    return path


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _positive_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (float, int)):
        raise ValueError(f"{name} must be a positive finite number")
    result = float(value)
    if not 0 < result < float("inf"):
        raise ValueError(f"{name} must be a positive finite number")
    return result


class Wan22DreamLoRABackend(BaseImageToVideoBackend):
    """Load one matched high/low noise pair with the pinned Wan I2V base."""

    backend_id = "dream_exe_wan22_i2v_a14b_lora_diffsynth"

    def __init__(
        self,
        *,
        base_model_root: str | Path,
        adapter_root: str | Path,
        source_root: str | Path,
        checkpoint_step: int,
    ) -> None:
        self.base_model_root = Path(base_model_root).expanduser().absolute()
        self.adapter_root = Path(adapter_root).expanduser().absolute()
        self.source_root = Path(source_root).expanduser().absolute()
        self.checkpoint_step = _positive_int(checkpoint_step, "checkpoint_step")
        if self.checkpoint_step not in (2000, 7000):
            raise ValueError("checkpoint_step must be 2000 or 7000")

    @property
    def identity(self) -> Mapping[str, Any]:
        return {
            **super().identity,
            "model_repository": MODEL_REPOSITORY,
            "model_revision": MODEL_REVISION,
            "base_repository": BASE_REPOSITORY,
            "base_revision": BASE_REVISION,
            "implementation_revision": DIFFSYNTH_REVISION,
            "checkpoint_step": self.checkpoint_step,
        }

    def _assets(self) -> tuple[dict[str, Any], Path, Path, list[Path], list[Path]]:
        if self.source_root.is_symlink() or not self.source_root.is_dir():
            raise FileNotFoundError(f"DiffSynth source directory is missing: {self.source_root}")
        revision = subprocess.run(
            ("git", "-C", str(self.source_root), "rev-parse", "HEAD"),
            check=True,
            capture_output=True,
            text=True,
        ).stdout.strip()
        if revision != DIFFSYNTH_REVISION:
            raise RuntimeError(f"DiffSynth revision mismatch: expected {DIFFSYNTH_REVISION}, found {revision}")
        if self.base_model_root.is_symlink() or not self.base_model_root.is_dir():
            raise FileNotFoundError(f"Wan I2V base directory is missing: {self.base_model_root}")
        if self.adapter_root.is_symlink() or not self.adapter_root.is_dir():
            raise FileNotFoundError(f"Dream.exe LoRA directory is missing: {self.adapter_root}")
        config_path = _regular_file(self.adapter_root / "inference_config.json")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        if not isinstance(config, dict):
            raise ValueError("LoRA inference_config.json must be an object")
        pair = self.adapter_root / f"Wan2.2_I2V_A14B_lora_{self.checkpoint_step // 1000}k"
        high_lora = _regular_file(pair / "high_noise_model.safetensors")
        low_lora = _regular_file(pair / "low_noise_model.safetensors")
        manifest_path = _regular_file(self.adapter_root / "weights_manifest.json")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest["base_model"] != {"repo": BASE_REPOSITORY, "revision": BASE_REVISION}:
            raise ValueError("LoRA manifest base-model identity differs from the supported release")
        for expert, path in (("high", high_lora), ("low", low_lora)):
            name = path.relative_to(self.adapter_root).as_posix()
            record = manifest["weights"][name]
            if record["step"] != self.checkpoint_step or record["expert"] != expert:
                raise ValueError(f"LoRA manifest checkpoint identity mismatch: {name}")
            if path.stat().st_size != record["size_bytes"]:
                raise ValueError(f"LoRA checkpoint size mismatch: {name}")
            digest = hashlib.sha256()
            with path.open("rb") as stream:
                for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                    digest.update(chunk)
            if digest.hexdigest() != record["sha256"]:
                raise ValueError(f"LoRA checkpoint SHA256 mismatch: {name}")
        high_base = sorted((self.base_model_root / "high_noise_model").glob("diffusion_pytorch_model-*.safetensors"))
        low_base = sorted((self.base_model_root / "low_noise_model").glob("diffusion_pytorch_model-*.safetensors"))
        if not high_base or not low_base:
            raise FileNotFoundError("Wan I2V base high/low noise diffusion shards are missing")
        for path in (*high_base, *low_base):
            _regular_file(path)
        _regular_file(self.base_model_root / "models_t5_umt5-xxl-enc-bf16.pth")
        _regular_file(self.base_model_root / "Wan2.1_VAE.pth")
        if not (self.base_model_root / "google" / "umt5-xxl").is_dir():
            raise FileNotFoundError("Wan I2V base tokenizer directory is missing")
        return config, high_lora, low_lora, high_base, low_base

    def generate(
        self,
        *,
        image_path: Path,
        prompt: str,
        output_path: Path,
        seed: int,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        config, high_lora, low_lora, high_base, low_base = self._assets()
        unknown = sorted(set(parameters) - {"size", "frame_num", "sample_steps", "guidance_scale"})
        if unknown:
            raise ValueError("unsupported Dream.exe LoRA parameters: " + ", ".join(unknown))
        size = str(parameters.get("size", f'{config["width"]}*{config["height"]}'))
        try:
            width, height = (int(part) for part in size.split("*"))
        except (ValueError, TypeError) as error:
            raise ValueError("size must be WIDTH*HEIGHT") from error
        _positive_int(width, "width")
        _positive_int(height, "height")
        frame_num = _positive_int(parameters.get("frame_num", config["frames_by_nominal_duration_seconds"]["5"]), "frame_num")
        if (frame_num - 1) % 4:
            raise ValueError("frame_num must satisfy 4n+1")
        sample_steps = _positive_int(parameters.get("sample_steps", config["num_inference_steps"]), "sample_steps")
        guidance_scale = _positive_float(parameters.get("guidance_scale", config["cfg_scale"]), "guidance_scale")

        try:
            import torch
            from PIL import Image
            from diffsynth.pipelines.wan_video import ModelConfig, WanVideoPipeline
            from diffsynth.utils.data import save_video
        except ImportError as error:
            raise RuntimeError("Install the pinned DiffSynth-Studio checkout in the generation environment") from error

        base = self.base_model_root
        pipeline = WanVideoPipeline.from_pretrained(
            torch_dtype=torch.bfloat16,
            device="cuda",
            model_configs=[
                ModelConfig(path=[str(path) for path in high_base]),
                ModelConfig(path=[str(path) for path in low_base]),
                ModelConfig(path=str(base / "models_t5_umt5-xxl-enc-bf16.pth")),
                ModelConfig(path=str(base / "Wan2.1_VAE.pth")),
            ],
            tokenizer_config=ModelConfig(path=str(base / "google" / "umt5-xxl")),
        )
        pipeline.load_lora(pipeline.dit, str(high_lora), alpha=config["lora_alpha"])
        pipeline.load_lora(pipeline.dit2, str(low_lora), alpha=config["lora_alpha"])
        with Image.open(image_path) as image, torch.inference_mode():
            frames = pipeline(
                prompt=prompt,
                negative_prompt=config["negative_prompt"],
                input_image=image.convert("RGB"),
                height=height,
                width=width,
                num_frames=frame_num,
                seed=seed,
                tiled=config["tiled"],
                cfg_scale=guidance_scale,
                switch_DiT_boundary=config["switch_DiT_boundary"],
                num_inference_steps=sample_steps,
                sigma_shift=config["sigma_shift"],
            )
        save_video(frames, str(output_path), fps=config["fps"], quality=9)
        return {
            "checkpoint_step": self.checkpoint_step,
            "size": size,
            "frame_num": frame_num,
            "sample_steps": sample_steps,
            "guidance_scale": guidance_scale,
            "fps": config["fps"],
            "model_revision": MODEL_REVISION,
            "base_revision": BASE_REVISION,
        }
