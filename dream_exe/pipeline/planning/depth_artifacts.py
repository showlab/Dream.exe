"""Formal depth-output contract for one benchmark video2traj run.

The video2traj core owns depth computation and transactional publication.  The
benchmark layer only binds the current formal paths and decides which published
files must be retained in resume and single-UID quality evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...artifacts.layout import trajectory_artifact_paths
from .configuration import load_bench_pipeline_config
from ..records.layout import run_key_uses_gt_depth

_CURRENT_SAFE_DEPTH_ROLES = (
    "depth_npy",
    "depth_meta_npy",
    "depth_cache_meta_json",
    "depth_manifest_json",
)
_CANONICAL_MEDIA_ROLES = (
    "depth_mp4",
    "depth_frame0_png",
    "depth_contact_png",
    "depth_vis_meta_json",
)


def build_video2traj_depth_artifact_contract(
    *,
    sample_root: str | Path,
    run_key: str,
    gen_model: str,
    trajectory_config_path: str | Path,
    traj_dir: str | Path,
) -> dict[str, Any]:
    """Resolve the read-only formal depth evidence expected for one run."""

    sample = Path(sample_root).expanduser().resolve()
    trajectory_path = Path(trajectory_config_path).expanduser().resolve()
    trajectory_root = Path(traj_dir).expanduser().resolve()
    if not trajectory_path.is_file():
        raise FileNotFoundError(f"trajectory config not found: {trajectory_path}")
    if not trajectory_root.is_relative_to(sample):
        raise ValueError("formal trajectory root escapes the benchmark sample")

    use_gt_depth = bool(run_key_uses_gt_depth(run_key))
    pipeline = load_bench_pipeline_config(
        sample_dir=sample.as_posix(),
        pipeline_config_path=trajectory_path.as_posix(),
        run_key=run_key,
        gen_model=gen_model,
    )
    raw_depth = pipeline.get("depth", {})
    if not isinstance(raw_depth, Mapping):
        raise TypeError("pipeline depth config must be a mapping")
    depth = dict(raw_depth)
    raw_policy = depth.get("artifact_policy", {})
    if not isinstance(raw_policy, Mapping):
        raise TypeError("depth.artifact_policy must be a mapping")
    policy = dict(raw_policy)
    save_canonical_mp4 = policy.get("save_canonical_mp4", False)
    if not isinstance(save_canonical_mp4, bool):
        raise TypeError("depth.artifact_policy.save_canonical_mp4 must be a bool")
    raw_target_calibration = depth.get("target_calibrated_lift", {})
    if not isinstance(raw_target_calibration, Mapping):
        raise TypeError("depth.target_calibrated_lift must be a mapping")
    target_calibration_enabled = raw_target_calibration.get("enabled", True)
    if not isinstance(target_calibration_enabled, bool):
        raise TypeError("depth.target_calibrated_lift.enabled must be a bool")

    paths: dict[str, str] = {}
    if not use_gt_depth:
        layout = trajectory_artifact_paths(trajectory_root)
        roles = list(_CURRENT_SAFE_DEPTH_ROLES)
        if save_canonical_mp4:
            roles.extend(_CANONICAL_MEDIA_ROLES)
        paths = {role: layout[role].as_posix() for role in roles}
        if target_calibration_enabled:
            paths.update(
                {
                    "depth_runtime_lineage_json": layout[
                        "depth_runtime_lineage_json"
                    ].as_posix(),
                    "eef_consumed_depth_samples_npy": layout[
                        "eef_consumed_depth_samples_npy"
                    ].as_posix(),
                }
            )
    return {
        "mode": (
            "rollout_gt_depth_reference"
            if use_gt_depth
            else "estimated_current_safe_cache"
        ),
        "save_canonical_mp4": save_canonical_mp4,
        "paths": paths,
    }


__all__ = ["build_video2traj_depth_artifact_contract"]
