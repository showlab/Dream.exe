"""The released Wan I2V LoRA must load a matched pair and released settings."""

from __future__ import annotations

import hashlib
import json
import sys
import types
from contextlib import nullcontext
from pathlib import Path

import pytest
from PIL import Image

from dream_exe.generation.providers.wan22_lora import (
    BASE_REPOSITORY,
    BASE_REVISION,
    DIFFSYNTH_REVISION,
    Wan22DreamLoRABackend,
)


def test_released_lora_loads_matching_experts_and_config(tmp_path, monkeypatch):
    base = tmp_path / "base"
    adapter = tmp_path / "adapter"
    source = tmp_path / "source"
    source.mkdir()
    for expert in ("high", "low"):
        shard = base / f"{expert}_noise_model" / "diffusion_pytorch_model-00001.safetensors"
        shard.parent.mkdir(parents=True)
        shard.write_bytes(b"base")
    (base / "models_t5_umt5-xxl-enc-bf16.pth").write_bytes(b"text")
    (base / "Wan2.1_VAE.pth").write_bytes(b"vae")
    (base / "google" / "umt5-xxl").mkdir(parents=True)
    pair = adapter / "Wan2.2_I2V_A14B_lora_7k"
    pair.mkdir(parents=True)
    weights = {}
    for expert in ("high", "low"):
        path = pair / f"{expert}_noise_model.safetensors"
        data = expert.encode()
        path.write_bytes(data)
        weights[path.relative_to(adapter).as_posix()] = {
            "step": 7000,
            "expert": expert,
            "size_bytes": len(data),
            "sha256": hashlib.sha256(data).hexdigest(),
        }
    (adapter / "weights_manifest.json").write_text(json.dumps({
        "base_model": {"repo": BASE_REPOSITORY, "revision": BASE_REVISION},
        "weights": weights,
    }))
    (adapter / "inference_config.json").write_text(json.dumps({
        "width": 480,
        "height": 480,
        "frames_by_nominal_duration_seconds": {"5": 81},
        "num_inference_steps": 40,
        "cfg_scale": 3.5,
        "negative_prompt": "bad video",
        "lora_alpha": 1,
        "tiled": True,
        "switch_DiT_boundary": 0.9,
        "sigma_shift": 5.0,
        "fps": 16,
    }))
    monkeypatch.setattr("dream_exe.generation.providers.wan22_lora.subprocess.run", lambda *a, **k: types.SimpleNamespace(stdout=DIFFSYNTH_REVISION))

    calls = {}
    class Pipeline:
        dit = object()
        dit2 = object()

        @classmethod
        def from_pretrained(cls, **kwargs):
            calls["model_configs"] = kwargs["model_configs"]
            return cls()

        def load_lora(self, expert, path, alpha):
            calls.setdefault("loras", []).append((expert, Path(path).name, alpha))

        def __call__(self, **kwargs):
            calls["generation"] = kwargs
            return ["frame"]

    class ModelConfig:
        def __init__(self, *, path):
            self.path = path

    modules = {
        "torch": types.SimpleNamespace(bfloat16="bf16", inference_mode=nullcontext),
        "diffsynth": types.ModuleType("diffsynth"),
        "diffsynth.pipelines": types.ModuleType("diffsynth.pipelines"),
        "diffsynth.pipelines.wan_video": types.SimpleNamespace(ModelConfig=ModelConfig, WanVideoPipeline=Pipeline),
        "diffsynth.utils": types.ModuleType("diffsynth.utils"),
        "diffsynth.utils.data": types.SimpleNamespace(save_video=lambda frames, path, **kw: Path(path).write_bytes(b"mp4")),
    }
    for name, module in modules.items():
        monkeypatch.setitem(sys.modules, name, module)

    image = tmp_path / "first.png"
    Image.new("RGB", (480, 480)).save(image)
    backend = Wan22DreamLoRABackend(base_model_root=base, adapter_root=adapter, source_root=source, checkpoint_step=7000)
    output = tmp_path / "out.mp4"
    result = backend.generate(image_path=image, prompt="close the oven", output_path=output, seed=42, parameters={})
    assert output.read_bytes() == b"mp4"
    assert [item[1] for item in calls["loras"]] == ["high_noise_model.safetensors", "low_noise_model.safetensors"]
    assert calls["loras"][0][0] is not calls["loras"][1][0]
    assert calls["generation"]["negative_prompt"] == "bad video"
    assert calls["generation"]["num_frames"] == 81
    assert calls["generation"]["cfg_scale"] == 3.5
    assert result["checkpoint_step"] == 7000

    (pair / "low_noise_model.safetensors").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="checkpoint size mismatch"):
        backend.generate(image_path=image, prompt="close the oven", output_path=output, seed=42, parameters={})
