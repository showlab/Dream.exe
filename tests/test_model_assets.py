from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import ANY

import pytest

from dream_exe.cli.assets import _model_asset_manifest
from dream_exe.model_assets.models import (
    MODEL_ASSET_MANIFEST_SCHEMA,
    current_core_model_manifest,
    plan_model_asset_acquisition,
    publish_model_assets,
    validate_model_asset_manifest,
    verify_model_assets,
)


NONCOMMERCIAL_LICENSES = {
    "cotracker-cc-by-nc-4.0",
    "dvd-cc-by-nc-4.0",
}


def test_core_manifest_is_pinned_and_dry_run_is_no_write(tmp_path: Path) -> None:
    manifest = current_core_model_manifest()

    assert _model_asset_manifest("core") == manifest
    assert {package["id"] for package in manifest["packages"]} == {
        "grounding-dino-swin-t",
        "bert-base-uncased",
        "sam2.1-hiera-large",
        "cotracker3-offline",
        "dvd-base-checkpoint",
        "dvd-base-config",
        "wan2.1-dvd-runtime",
    }
    for package in manifest["packages"]:
        if package["source"]["type"] == "huggingface":
            revision = package["source"]["locator"].rsplit("@", 1)[1]
            assert len(revision) == 40
            assert set(revision) <= set("0123456789abcdef")

    asset_root = tmp_path / "absent-checkpoints"
    blocked = plan_model_asset_acquisition(
        manifest,
        asset_root=asset_root.resolve(),
    )
    assert blocked["status"] == "blocked"
    assert {
        item["detail"]["id"]
        for item in blocked["blocked"]
        if item["reason"] == "license_acceptance_required"
    } == NONCOMMERCIAL_LICENSES

    plan = plan_model_asset_acquisition(
        manifest,
        asset_root=asset_root.resolve(),
        accepted_licenses=NONCOMMERCIAL_LICENSES,
    )
    result = publish_model_assets(
        plan,
        asset_root=asset_root.resolve(),
        dry_run=True,
    )

    assert plan["status"] == "planned"
    assert result["status"] == "dry_run"
    assert not asset_root.exists()


def test_huggingface_source_rejects_a_moving_revision() -> None:
    manifest = current_core_model_manifest()
    changed = copy.deepcopy(manifest)
    package = next(
        item
        for item in changed["packages"]
        if item["source"]["type"] == "huggingface"
    )
    package["source"]["locator"] = package["source"]["locator"].split("@")[0]
    package["source"]["locator"] += "@main"

    with pytest.raises(ValueError, match="40-character-commit"):
        validate_model_asset_manifest(changed)


def test_huggingface_fetch_is_revision_pinned_and_atomically_verified(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    payload = b"pinned checkpoint bytes\n"
    upstream = tmp_path / "upstream.bin"
    upstream.write_bytes(payload)
    calls: list[dict[str, object]] = []

    def fake_hf_hub_download(**kwargs: object) -> str:
        calls.append(kwargs)
        return upstream.as_posix()

    monkeypatch.setitem(
        __import__("sys").modules,
        "huggingface_hub",
        SimpleNamespace(hf_hub_download=fake_hf_hub_download),
    )
    revision = "a" * 40
    manifest = {
        "format": MODEL_ASSET_MANIFEST_SCHEMA,
        "name": "test-hf-asset",
        "description": "test",
        "packages": [
            {
                "id": "checkpoint",
                "source": {
                    "type": "huggingface",
                    "locator": f"owner/repository@{revision}",
                },
                "destination": "model",
                "license": {
                    "id": "test-license",
                    "notice": "test only",
                    "acceptance_required": False,
                },
                "artifacts": [
                    {
                        "path": "checkpoint.bin",
                        "size_bytes": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                ],
            }
        ],
        "unresolved": [],
    }
    asset_root = (tmp_path / "checkpoints").resolve()
    plan = plan_model_asset_acquisition(manifest, asset_root=asset_root)

    result = publish_model_assets(plan, asset_root=asset_root)

    assert result["status"] == "published"
    assert calls == [
        {
            "repo_id": "owner/repository",
            "filename": "checkpoint.bin",
            "revision": revision,
            "local_dir": ANY,
        }
    ]
    assert (asset_root / "model/checkpoint.bin").read_bytes() == payload
    assert verify_model_assets(manifest, asset_root=asset_root)["ready"] is True
