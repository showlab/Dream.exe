"""Versioned packaged defaults for canonical pipeline composition.

The benchmark owns only experiment choices and per-case differences.  Stable
algorithm and executor defaults remain source-owned and are fingerprinted in
every result request.  Run-owned paths are deliberately absent: they are
bound by the facade after a case and input have been selected.
"""

from __future__ import annotations

import copy
from functools import lru_cache
from typing import Any

from ...sim.execution.config import default_execution_config
from ...video2traj.runtime.config import default_pipeline_config_dict
from .schemas import PIPELINE_STAGES, canonical_sha256


PACKAGED_DEFAULTS_SCHEMA = "dream-exe.packaged-pipeline-defaults"

RUN_OWNED_FIELDS = {
    "video2traj": (
        ("input", "rollout_video_path"),
        ("input", "gen_video_path"),
        ("input", "selected_video"),
        ("depth", "rollout_gt_depth_path"),
        ("depth", "use_rollout_gt_depth"),
        ("pose", "mesh_path"),
    ),
    "execution": (
        ("input", "traj_path"),
        ("input", "action_path"),
    ),
}


def _deep_delete(root: dict[str, Any], path: tuple[str, ...]) -> None:
    current: dict[str, Any] = root
    parents: list[tuple[dict[str, Any], str]] = []
    for name in path[:-1]:
        child = current.get(name)
        if not isinstance(child, dict):
            return
        parents.append((current, name))
        current = child
    current.pop(path[-1], None)
    for parent, name in reversed(parents):
        child = parent.get(name)
        if isinstance(child, dict) and not child:
            parent.pop(name, None)


@lru_cache(maxsize=1)
def _cached_documents() -> dict[str, dict[str, Any]]:
    trajectory = default_pipeline_config_dict()
    action = trajectory.pop("action")
    values: dict[str, dict[str, Any]] = {
        "video2traj": trajectory,
        "action": action,
        "execution": default_execution_config(),
        "evaluation": {},
    }
    for stage, paths in RUN_OWNED_FIELDS.items():
        for path in paths:
            _deep_delete(values[stage], path)
    if set(values) != set(PIPELINE_STAGES):  # pragma: no cover - source invariant
        raise RuntimeError("packaged defaults do not cover the pipeline stages")
    return {
        stage: {
            "format": PACKAGED_DEFAULTS_SCHEMA,
            "stage": stage,
            "values": values[stage],
        }
        for stage in PIPELINE_STAGES
    }


def packaged_default_documents() -> dict[str, dict[str, Any]]:
    """Return detached, versioned documents for all pipeline stages."""

    return copy.deepcopy(_cached_documents())


def packaged_default_digests() -> dict[str, str]:
    """Fingerprint the exact code-owned defaults used by the compiler."""

    return {
        stage: canonical_sha256(document)
        for stage, document in _cached_documents().items()
    }


__all__ = [
    "PACKAGED_DEFAULTS_SCHEMA",
    "RUN_OWNED_FIELDS",
    "packaged_default_digests",
    "packaged_default_documents",
]
