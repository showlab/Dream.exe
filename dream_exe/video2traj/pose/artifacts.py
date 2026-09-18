"""Current-compatible publication for the EEF pose sidecar.

The pose algorithm remains an in-memory, environment-independent callable.
This module is the explicit output adapter: it writes only beneath a caller
provided trajectory output directory and never discovers benchmark paths.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dream_exe.artifacts.io import (
    ensure_traj_asset_dirs,
    update_traj_assets_manifest,
)


DEFAULT_EEF_POSE_FILENAME = "eef_pose.json"


def eef_pose_artifact_path(
    output_dir: str | Path,
) -> str:
    """Return the canonical absolute sidecar path for an explicit output."""

    output_text = str(output_dir or "").strip()
    if not output_text:
        raise ValueError("output_dir must be an explicit non-empty path")
    return (
        Path(output_text).expanduser().resolve() / "pose" / DEFAULT_EEF_POSE_FILENAME
    ).as_posix()


def pose_payload_with_artifact_reference(
    payload: Mapping[str, Any],
    *,
    output_dir: str | Path,
) -> dict[str, Any]:
    """Copy an in-memory pose payload and attach its publication reference."""

    if not isinstance(payload, Mapping):
        raise TypeError("pose payload must be a mapping")
    result = copy.deepcopy(dict(payload))
    meta = result.get("meta", {})
    if not isinstance(meta, Mapping):
        raise TypeError("pose payload meta must be a mapping")
    poses = result.get("poses", None)
    if not isinstance(poses, list):
        raise TypeError("pose payload poses must be a list")
    result["meta"] = copy.deepcopy(dict(meta))
    result["meta"]["pose_json_path"] = eef_pose_artifact_path(output_dir)
    return result


def write_eef_pose_artifact(
    output_dir: str | Path,
    payload: Mapping[str, Any],
    *,
    manifest_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write ``pose/eef_pose.json`` and the current ``assets.json`` section.

    The current estimator adds ``meta.pose_json_path`` only to its returned
    in-memory payload after serializing the sidecar.  Preserve that observable
    distinction: the on-disk JSON omits the self-reference, while trajectory
    metadata and the assets manifest may reference the canonical sidecar path.
    """

    prepared = pose_payload_with_artifact_reference(
        payload,
        output_dir=output_dir,
    )
    if manifest_payload is not None and not isinstance(
        manifest_payload,
        Mapping,
    ):
        raise TypeError("pose manifest_payload must be a mapping")
    manifest_fields = copy.deepcopy(dict(manifest_payload or {}))
    reserved = sorted({"dir", "eef_pose_json"}.intersection(manifest_fields))
    if reserved:
        raise ValueError(
            "pose manifest_payload cannot replace publication-owned fields: "
            + ", ".join(reserved)
        )

    directories = ensure_traj_asset_dirs(str(output_dir))
    pose_path = Path(prepared["meta"]["pose_json_path"])
    pose_path.parent.mkdir(parents=True, exist_ok=True)
    disk_payload = copy.deepcopy(prepared)
    disk_payload["meta"].pop("pose_json_path", None)
    with pose_path.open("w", encoding="utf-8") as handle:
        json.dump(disk_payload, handle, indent=2)

    section = {
        "dir": directories["pose"],
        "eef_pose_json": pose_path.as_posix(),
        **manifest_fields,
    }
    manifest_path = update_traj_assets_manifest(
        str(output_dir),
        "pose",
        section,
    )
    return {
        "path": pose_path.as_posix(),
        "manifest_path": manifest_path,
    }


__all__ = [
    "DEFAULT_EEF_POSE_FILENAME",
    "eef_pose_artifact_path",
    "pose_payload_with_artifact_reference",
    "write_eef_pose_artifact",
]
