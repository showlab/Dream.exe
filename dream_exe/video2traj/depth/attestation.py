"""Read-only SHA-256 attestation for explicit DVD model assets."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...model_assets.digests import sha256_stable_file
from ...model_assets.dvd_identity import verify_dvd_asset_digests


def sha256_file(path: str | Path, *, label: str) -> str:
    """Hash one required regular file without importing a model runtime."""

    return sha256_stable_file(path, label=label)


def attest_dvd_asset_files(
    provenance: Mapping[str, Any],
    *,
    checkpoint_file: str | Path,
    model_config_file: str | Path,
) -> dict[str, Any]:
    """Verify the complete DVD checkpoint/config pair and return provenance."""

    return verify_dvd_asset_digests(
        provenance,
        checkpoint_sha256=sha256_file(
            checkpoint_file,
            label="DVD checkpoint file",
        ),
        model_config_sha256=sha256_file(
            model_config_file,
            label="DVD model config file",
        ),
    )


__all__ = [
    "attest_dvd_asset_files",
    "sha256_file",
]
