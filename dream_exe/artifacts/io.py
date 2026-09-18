"""Current-compatible trajectory and execution artifact bookkeeping.

The active root implementation writes ``assets.json`` inside both ``traj`` and
``exec`` output directories. This module preserves that observed behavior. The
bench path adapter exposes trajectory ``assets.json`` through its compatible
``trajectory_manifest`` role; current implementation does not invent a second manifest file.

Only directory creation, asset-manifest read/update, and the current narrow
trajectory legacy cleanup are included.  Stage artifacts, status, sample
metadata, formal run manifests, metrics, and atomic publication are outside
this slice.
"""

from __future__ import annotations

from collections.abc import Mapping
import copy
import json
import os
from pathlib import Path
from typing import Any, Dict

from .layout import (
    ASSETS_MANIFEST_FILENAME,
    execution_artifact_paths,
    trajectory_artifact_paths,
)

try:
    import numpy as np
except ModuleNotFoundError:  # pragma: no cover - numpy is optional for plain JSON
    np = None


DEFAULT_TRAJ_ASSETS_MANIFEST = ASSETS_MANIFEST_FILENAME
DEFAULT_EXEC_ASSETS_MANIFEST = ASSETS_MANIFEST_FILENAME


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _to_serializable(value: Any) -> Any:
    if np is not None and isinstance(value, np.ndarray):
        return value.tolist()
    if np is not None and isinstance(value, (np.float32, np.float64)):
        return float(value)
    if np is not None and isinstance(value, (np.int32, np.int64)):
        return int(value)
    if isinstance(value, dict):
        return {key: _to_serializable(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_to_serializable(item) for item in value]
    return value


def _save_json(payload: Any, path: str) -> str:
    _ensure_dir(os.path.dirname(path))
    with open(path, "w") as handle:
        json.dump(_to_serializable(payload), handle, indent=4)
    return path


def _load_json(path: str) -> Any:
    with open(path, "r") as handle:
        return json.load(handle)


def _trajectory_role_path(output_dir: str, role: str) -> str:
    paths = trajectory_artifact_paths(output_dir)
    relative = paths[role].relative_to(paths["traj_dir"])
    if not relative.parts:
        return output_dir
    return os.path.join(output_dir, *relative.parts)


def ensure_traj_asset_dirs(output_dir: str) -> Dict[str, str]:
    """Create and return the current trajectory stage-directory mapping."""

    region_dir = _trajectory_role_path(output_dir, "region_dir")
    dirs = {
        "root": output_dir,
        "region": region_dir,
        "regions": region_dir,
        "tracking": _trajectory_role_path(output_dir, "tracking_dir"),
        "depth": _trajectory_role_path(output_dir, "depth_dir"),
        "geometry": _trajectory_role_path(output_dir, "geometry_dir"),
        "pose": _trajectory_role_path(output_dir, "pose_dir"),
        "trajectory": _trajectory_role_path(output_dir, "trajectory_dir"),
        "gripper": _trajectory_role_path(output_dir, "gripper_dir"),
        "action": _trajectory_role_path(output_dir, "action_dir"),
        "visualization": _trajectory_role_path(
            output_dir,
            "visualization_dir",
        ),
    }
    for path in dirs.values():
        _ensure_dir(path)
    return dirs


def load_traj_assets_manifest(
    output_dir: str,
    *,
    manifest_name: str = DEFAULT_TRAJ_ASSETS_MANIFEST,
) -> Dict[str, Any]:
    """Load the trajectory assets map, treating every read failure as empty."""

    manifest_path = (
        _trajectory_role_path(output_dir, "trajectory_manifest")
        if manifest_name == DEFAULT_TRAJ_ASSETS_MANIFEST
        else os.path.join(output_dir, manifest_name)
    )
    if not os.path.exists(manifest_path):
        return {}
    try:
        data = _load_json(manifest_path)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def _relativize_paths(value: Any, *, base_dir: str) -> Any:
    if isinstance(value, dict):
        return {
            key: _relativize_paths(item, base_dir=base_dir)
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [_relativize_paths(item, base_dir=base_dir) for item in value]
    if isinstance(value, str):
        text = str(value or "").strip()
        if not text:
            return text
        if os.path.isabs(text):
            try:
                return os.path.relpath(text, base_dir)
            except Exception:
                return text
    return value


def update_traj_assets_manifest(
    output_dir: str,
    section_name: str,
    section_payload: Dict[str, Any],
    *,
    manifest_name: str = DEFAULT_TRAJ_ASSETS_MANIFEST,
) -> str:
    """Read-update-write one top-level trajectory ``assets.json`` section."""

    manifest_path = (
        _trajectory_role_path(output_dir, "trajectory_manifest")
        if manifest_name == DEFAULT_TRAJ_ASSETS_MANIFEST
        else os.path.join(output_dir, manifest_name)
    )
    manifest = load_traj_assets_manifest(
        output_dir,
        manifest_name=manifest_name,
    )
    manifest[str(section_name)] = _relativize_paths(
        _to_serializable(section_payload),
        base_dir=output_dir,
    )
    _save_json(manifest, manifest_path)
    return manifest_path


def _write_traj_payload_artifact(
    output_dir: str,
    *,
    section_name: str,
    artifact_role: str,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    """Write one current trajectory payload and publish its assets section."""

    directories = ensure_traj_asset_dirs(output_dir)
    artifact_path = _trajectory_role_path(output_dir, artifact_role)
    with open(artifact_path, "w") as handle:
        # The current trajectory pipeline uses indent=2 for gripper/action
        # payloads.  Do not route this through the indent=4 manifest helper.
        json.dump(payload, handle, indent=2)
    manifest_path = update_traj_assets_manifest(
        output_dir,
        section_name,
        {
            "dir": directories[section_name],
            f"{section_name}_json": artifact_path,
        },
    )
    return {
        "path": artifact_path,
        "manifest_path": manifest_path,
    }


def write_trajectory_artifacts(
    output_dir: str,
    ee_payload: Dict[str, Any],
    *,
    object_payload: Dict[str, Any] | None = None,
    union_payload: Dict[str, Any] | None = None,
    eef_pose_path: str | None = None,
    manifest_payload: Mapping[str, Any] | None = None,
) -> Dict[str, str | None]:
    """Publish current trajectory payload files and manifest references."""

    manifest_fields = copy.deepcopy(dict(manifest_payload or {}))
    reserved = {
        "dir",
        "ee_traj_json",
        "obj_trajs_json",
        "union_traj_json",
        "eef_pose_json",
    }.intersection(manifest_fields)
    if reserved:
        raise ValueError(
            "trajectory manifest_payload cannot replace publication-owned "
            "fields: " + ", ".join(sorted(reserved))
        )

    directories = ensure_traj_asset_dirs(output_dir)
    trajectory_dir = directories["trajectory"]
    ee_path = _trajectory_role_path(output_dir, "ee_traj")
    object_path = (
        _trajectory_role_path(output_dir, "obj_trajs")
        if object_payload is not None
        else None
    )
    union_path = (
        _trajectory_role_path(output_dir, "union_traj")
        if union_payload is not None
        else None
    )

    with open(ee_path, "w") as handle:
        json.dump(ee_payload, handle, indent=2)
    if object_path is not None:
        with open(object_path, "w") as handle:
            json.dump(object_payload, handle, indent=2)
    if union_path is not None:
        with open(union_path, "w") as handle:
            json.dump(union_payload, handle, indent=2)

    manifest_path = update_traj_assets_manifest(
        output_dir,
        "trajectory",
        {
            "dir": trajectory_dir,
            "ee_traj_json": ee_path,
            "obj_trajs_json": object_path,
            "union_traj_json": union_path,
            "eef_pose_json": eef_pose_path,
            **manifest_fields,
        },
    )
    return {
        "ee_path": ee_path,
        "object_path": object_path,
        "union_path": union_path,
        "manifest_path": manifest_path,
    }


def write_gripper_artifact(
    output_dir: str,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    """Write current ``gripper/gripper.json`` plus its assets entry."""

    return _write_traj_payload_artifact(
        output_dir,
        section_name="gripper",
        artifact_role="gripper",
        payload=payload,
    )


def write_action_artifact(
    output_dir: str,
    payload: Dict[str, Any],
) -> Dict[str, str]:
    """Write current ``action/action.json`` plus its assets entry."""

    return _write_traj_payload_artifact(
        output_dir,
        section_name="action",
        artifact_role="action",
        payload=payload,
    )


def publish_pose_fallback(
    output_dir: str | Path,
    *,
    pose_config_source: str | None,
    mesh_path: str | None,
    register_mask_source: str | None = None,
    reason: str | None = None,
) -> str:
    """Publish current disabled or position-only pose metadata."""

    directories = ensure_traj_asset_dirs(str(output_dir))
    section: dict[str, Any] = {
        "dir": directories["pose"],
        "eef_pose_json": None,
        "pose_config_source": pose_config_source,
        "foundationpose_debug_dir": None,
        "mesh_path": mesh_path,
    }
    if register_mask_source is not None:
        section["register_mask_source"] = register_mask_source
    if reason is not None:
        section["fallback"] = "position_only"
        section["fallback_reason"] = str(reason)
    return update_traj_assets_manifest(
        str(output_dir),
        "pose",
        section,
    )


def publish_depth_artifacts(
    output_dir: str | Path,
    *,
    source: str,
    rollout_gt_depth_path: str | None,
    estimated_depth_cache_path: str | None,
    depth_npy: str | None,
    depth_mp4: str | None,
    depth_meta_npy: str | None,
    depth_model: str,
    depth_cache_meta_json: str | None = None,
    depth_manifest_json: str | None = None,
    depth_frame0_png: str | None = None,
    depth_contact_png: str | None = None,
    depth_vis_meta_json: str | None = None,
    rollout_gt_depth_identity: Mapping[str, Any] | None = None,
    rollout_gt_depth_diagnostics: Mapping[str, Any] | None = None,
) -> str:
    """Publish current depth artifact references without model I/O."""

    directories = ensure_traj_asset_dirs(str(output_dir))
    payload: dict[str, Any] = {
        "dir": directories["depth"],
        "source": source,
        "rollout_gt_depth_path": rollout_gt_depth_path,
        "estimated_depth_cache_path": estimated_depth_cache_path,
        "depth_npy": depth_npy,
        "depth_mp4": depth_mp4,
        "depth_meta_npy": depth_meta_npy,
        "depth_model": depth_model,
    }
    for name, value in (
        ("depth_cache_meta_json", depth_cache_meta_json),
        ("depth_manifest_json", depth_manifest_json),
        ("depth_frame0_png", depth_frame0_png),
        ("depth_contact_png", depth_contact_png),
        ("depth_vis_meta_json", depth_vis_meta_json),
    ):
        if value is not None:
            payload[name] = value
    if rollout_gt_depth_identity is not None:
        payload["rollout_gt_depth_identity"] = dict(rollout_gt_depth_identity)
    if rollout_gt_depth_diagnostics is not None:
        payload["rollout_gt_depth_diagnostics"] = dict(rollout_gt_depth_diagnostics)
    return update_traj_assets_manifest(
        str(output_dir),
        "depth",
        payload,
    )
def update_exec_assets_manifest(
    output_dir: str,
    block_name: str,
    payload: Dict[str, Any],
) -> str:
    """Read-update-write one execution block using the current exec schema."""

    root = Path(output_dir).expanduser().resolve()
    _ensure_dir(root.as_posix())
    manifest_path = execution_artifact_paths(root)["execution_manifest"]
    if manifest_path.exists():
        manifest = _load_json(manifest_path.as_posix())
        if not isinstance(manifest, dict):
            manifest = {}
    else:
        manifest = {}
    manifest = copy.deepcopy(manifest)
    manifest.setdefault("output_dir", root.as_posix())
    manifest.setdefault("blocks", {})
    manifest["blocks"][str(block_name)] = _to_serializable(payload)
    _save_json(manifest, manifest_path.as_posix())
    return manifest_path.as_posix()


__all__ = [
    "DEFAULT_EXEC_ASSETS_MANIFEST",
    "DEFAULT_TRAJ_ASSETS_MANIFEST",
    "ensure_traj_asset_dirs",
    "load_traj_assets_manifest",
    "publish_depth_artifacts",
    "publish_pose_fallback",
    "update_exec_assets_manifest",
    "update_traj_assets_manifest",
    "write_action_artifact",
    "write_gripper_artifact",
    "write_trajectory_artifacts",
]
