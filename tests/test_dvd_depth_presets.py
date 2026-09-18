from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from dream_exe.model_assets.dvd_identity import (
    DVD_MODEL_FAMILY,
    DVD_OFFICIAL_MODEL_FAMILY,
    normalize_dvd_model_identity,
)
from dream_exe.model_assets.models import current_core_model_manifest
from dream_exe.pipeline.stages.video2traj import _bind_benchmark_depth_runtime
from dream_exe.video2traj.depth.backends.dvd import build_dvd_backend_registry
from dream_exe.video2traj.depth.estimator import (
    DEFAULT_DEPTH_PRESET,
    available_depth_estimator_presets,
    default_depth_estimator_config,
    make_depth_estimator,
    preflight_depth_estimator_request,
    resolve_depth_estimator_preset,
    resolve_depth_estimator_runtime,
)


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DVD_ATTESTATION_MANIFEST = (
    REPOSITORY_ROOT / "dream_exe/model_assets/configs/dvd.json"
)


def test_official_and_project_dvd_presets_are_explicit() -> None:
    assert DEFAULT_DEPTH_PRESET == "dvd_lora_specific"
    assert resolve_depth_estimator_preset("dvd_official") == "dvd_official"
    assert resolve_depth_estimator_preset("dvd_base") == "dvd_official"
    assert set(available_depth_estimator_presets()) >= {
        "dvd_official",
        "dvd_lora_shared",
        "dvd_lora_specific",
    }

    official = default_depth_estimator_config("dvd_official")
    assert official["model_family"] == DVD_OFFICIAL_MODEL_FAMILY
    assert official["model_provenance"]["kind"] == "upstream_official"
    assert official["dvd"]["ckpt_root"] == "DVD/official"
    assert official["dvd"]["model_config_path"] == "DVD/official/model_config.yaml"

    expected_roots = {
        "dvd_lora_shared": "DVD/lora/shared",
        "dvd_lora_specific": "DVD/lora/specific",
    }
    for preset in ("dvd_lora_shared", "dvd_lora_specific"):
        project = default_depth_estimator_config(preset)
        assert project["model_family"] == DVD_MODEL_FAMILY
        assert project["model_provenance"]["kind"] == "project_finetuned"
        assert project["dvd"]["ckpt_root"] == expected_roots[preset]
        assert project["dvd"]["model_config_path"] == "DVD/official/model_config.yaml"


def test_plain_dvd_remains_an_explicit_config_backend() -> None:
    with pytest.raises(ValueError, match="backend but not a checkpoint identity"):
        resolve_depth_estimator_preset("dvd")

    official = default_depth_estimator_config("dvd_official")
    preflight = preflight_depth_estimator_request(
        preset="dvd",
        config=official,
    )
    assert preflight["preset"] == "dvd"
    assert preflight["model_name"] == "dvd"


def test_dvd_family_and_provenance_kind_must_agree() -> None:
    official = default_depth_estimator_config("dvd_official")
    family, provenance = normalize_dvd_model_identity(
        official["model_family"],
        official["model_provenance"],
        source="test",
    )
    assert family == DVD_OFFICIAL_MODEL_FAMILY
    assert provenance["kind"] == "upstream_official"

    with pytest.raises(ValueError, match="conflicts with model_family"):
        normalize_dvd_model_identity(
            DVD_MODEL_FAMILY,
            official["model_provenance"],
            source="test",
        )


def test_official_dvd_resolves_and_reaches_the_dvd_backend(tmp_path: Path) -> None:
    runtime = resolve_depth_estimator_runtime(
        device="cuda:0",
        preset="dvd_official",
        weights_root=tmp_path,
    )
    assert runtime["preset"] == "dvd_official"
    assert runtime["model_family"] == DVD_OFFICIAL_MODEL_FAMILY
    assert runtime["model_kwargs"]["ckpt_root"] == (tmp_path / "DVD/official").as_posix()
    assert runtime["model_kwargs"]["model_family"] == DVD_OFFICIAL_MODEL_FAMILY

    real_factory = build_dvd_backend_registry(
        source_root=tmp_path / "DVD-source",
        checkpoints_root=tmp_path,
    )["dvd"]
    real_backend = real_factory(**runtime["model_kwargs"])
    assert real_backend.model_family == DVD_OFFICIAL_MODEL_FAMILY
    assert real_backend.model_provenance["kind"] == "upstream_official"

    captured: dict[str, object] = {}

    class FakeDVD:
        def infer(self, frames: object, target_fps: float, **_kwargs: object) -> dict:
            frame_count, height, width = np.asarray(frames).shape[:3]
            return {
                "depths": np.ones((frame_count, height, width), dtype=np.float32),
                "fps": target_fps,
                "meta": {"model_family": captured["model_family"]},
                "depth_space": "affine",
                "fps_source": "target_fps",
            }

    def factory(**kwargs: object) -> FakeDVD:
        captured.update(kwargs)
        return FakeDVD()

    estimator = make_depth_estimator(
        runtime_config=runtime,
        backend_registry={"dvd": factory},
        validate_assets=False,
    )
    depths, fps, info, _aux = estimator(
        video_frames=np.zeros((2, 4, 5, 3), dtype=np.uint8),
        target_fps=12.0,
    )

    assert depths.shape == (2, 4, 5)
    assert fps == 12.0
    assert captured["model_family"] == DVD_OFFICIAL_MODEL_FAMILY
    assert captured["model_provenance"]["kind"] == "upstream_official"
    assert info["model"]["model_family"] == DVD_OFFICIAL_MODEL_FAMILY


def test_benchmark_router_attests_the_selected_official_dvd() -> None:
    manifest = json.loads(DVD_ATTESTATION_MANIFEST.read_text(encoding="utf-8"))
    expected = manifest["presets"]["dvd_official"]
    packages = {
        item["id"]: item
        for item in current_core_model_manifest()["packages"]
    }
    assert expected == {
        "checkpoint_sha256": packages["dvd-base-checkpoint"]["artifacts"][0][
            "sha256"
        ],
        "model_config_sha256": packages["dvd-base-config"]["artifacts"][0][
            "sha256"
        ],
    }
    runtime = {
        "depth": {
            "backend": "bench_resolved",
            "preset": "bench_resolved",
            "providers": {
                "dvd": {
                    "backend": "dvd",
                    "preset": "bench_resolved",
                    "attestation_manifest_path": (
                        DVD_ATTESTATION_MANIFEST.as_posix()
                    ),
                }
            },
        }
    }

    bound = _bind_benchmark_depth_runtime(
        runtime,
        uid="rc_cheesybread_ep000001",
        pipeline_config={"depth": {"model": "dvd_official"}},
        use_rollout_gt_depth=False,
    )

    assert bound["depth"]["preset"] == "dvd_official"
    config = bound["depth"]["config"]
    assert config["model_family"] == DVD_OFFICIAL_MODEL_FAMILY
    assert config["model_provenance"]["asset_attestation"] == expected
