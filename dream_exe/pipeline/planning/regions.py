"""Read-only current-bench init-region adapter for video2traj.

Saved simulator segmentation belongs to the environment/bench boundary.  This
module resolves those explicit artifacts into :class:`PrecomputedRegion`
objects, so the simulator-independent algorithm receives masks and depth as
ordinary injected values rather than discovering a sample layout itself.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path
import re
from typing import Any

import numpy as np

from ...artifacts.layout import sample_artifact_paths
from ...video2traj.region.runtime import PrecomputedRegion
from ...video2traj.trajectory.stages import compile_task_runtime


_ENV_PROMPTS = {
    "Lift": "cube",
    "NutAssemblySquare": "square nut",
    "NutAssemblyRound": "round nut",
    "PickPlaceCan": "can",
    "PickPlaceMilk": "milk carton",
    "PickPlaceBread": "bread",
    "PickPlaceCereal": "cereal box",
}


def _json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _safe_asset_path(
    *,
    sample_root: Path,
    base_dir: Path,
    value: Any,
    label: str,
) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} is empty")
    candidate = Path(text).expanduser()
    resolved = (
        candidate.resolve()
        if candidate.is_absolute()
        else (base_dir / candidate).resolve()
    )
    if not resolved.is_relative_to(sample_root):
        raise ValueError(f"{label} escapes the explicit bench sample: {resolved}")
    return resolved


def resolve_benchmark_init_region_assets(
    sample_dir: str | Path,
) -> dict[str, str]:
    """Resolve current init mask/depth paths without loading or writing them."""

    sample_root = Path(sample_dir).expanduser().resolve()
    if not sample_root.is_dir():
        raise FileNotFoundError(
            f"explicit bench sample directory not found: {sample_root}"
        )
    sample_paths = sample_artifact_paths(sample_root)
    manifest_path = sample_paths["initialization_manifest"]
    instance_path = sample_paths["initialization_instance_segmentation"]
    names_path = sample_paths["initialization_instance_names"]
    mapping_path = sample_paths["initialization_segmentation_mappings"]
    depth_path = sample_paths["initialization_depth"]

    if manifest_path.is_file():
        manifest = _json_object(
            manifest_path,
            label="init assets manifest",
        )
        segmentation = dict(manifest.get("segmentation", {}) or {})
        levels = dict(segmentation.get("levels", {}) or {})
        instance = dict(levels.get("instance", {}) or {})
        if str(instance.get("npy", "") or "").strip():
            instance_path = _safe_asset_path(
                sample_root=sample_root,
                base_dir=manifest_path.parent,
                value=instance["npy"],
                label="init instance segmentation path",
            )
        if str(segmentation.get("instance_names_json", "") or "").strip():
            names_path = _safe_asset_path(
                sample_root=sample_root,
                base_dir=manifest_path.parent,
                value=segmentation["instance_names_json"],
                label="init instance names path",
            )
        if str(segmentation.get("mapping_json", "") or "").strip():
            mapping_path = _safe_asset_path(
                sample_root=sample_root,
                base_dir=manifest_path.parent,
                value=segmentation["mapping_json"],
                label="init segmentation mapping path",
            )
        depth = dict(manifest.get("depth", {}) or {})
        if str(depth.get("metric_npy", "") or "").strip():
            depth_path = _safe_asset_path(
                sample_root=sample_root,
                base_dir=manifest_path.parent,
                value=depth["metric_npy"],
                label="init metric depth path",
            )

    return {
        "sample_dir": sample_root.as_posix(),
        "manifest_path": (manifest_path.as_posix() if manifest_path.is_file() else ""),
        "instance_segmentation_path": instance_path.as_posix(),
        "instance_names_path": names_path.as_posix(),
        "segmentation_mapping_path": mapping_path.as_posix(),
        "init_depth_path": depth_path.as_posix(),
    }


def _load_instance_index(
    *,
    names_path: Path,
    mapping_path: Path,
) -> tuple[list[str], dict[str, int]]:
    if names_path.is_file():
        payload = _json_object(
            names_path,
            label="init instance names",
        )
        names = [str(value) for value in list(payload.get("instance_names", []) or [])]
        label_map = {
            str(key): int(value)
            for key, value in dict(
                payload.get("compact_label_by_instance", {}) or {}
            ).items()
        }
        if names and label_map:
            return names, label_map

    payload = _json_object(
        mapping_path,
        label="init segmentation mappings",
    )
    instances = dict(payload.get("instances_to_ids", {}) or {})
    if not instances:
        raise ValueError("init segmentation mappings contain no instances_to_ids")
    names = [str(name) for name in instances]
    return names, {name: index + 1 for index, name in enumerate(names)}


def load_benchmark_init_region_assets(
    sample_dir: str | Path,
) -> dict[str, Any]:
    """Load detached segmentation/index/depth arrays from one sample."""

    paths = resolve_benchmark_init_region_assets(sample_dir)
    instance_path = Path(paths["instance_segmentation_path"])
    if not instance_path.is_file():
        raise FileNotFoundError(
            f"init instance segmentation not found: {instance_path}"
        )
    segmentation = np.asarray(
        np.load(instance_path, allow_pickle=False),
        dtype=np.int32,
    )
    if segmentation.ndim != 2:
        raise ValueError("init instance segmentation must be a 2D array")
    names, label_map = _load_instance_index(
        names_path=Path(paths["instance_names_path"]),
        mapping_path=Path(paths["segmentation_mapping_path"]),
    )

    depth_path = Path(paths["init_depth_path"])
    init_depth: np.ndarray | None = None
    if depth_path.is_file():
        init_depth = np.asarray(
            np.load(depth_path, allow_pickle=False),
            dtype=np.float32,
        )
        if init_depth.shape != segmentation.shape:
            raise ValueError(
                "init depth and instance segmentation shapes differ: "
                f"{init_depth.shape} != {segmentation.shape}"
            )
    return {
        "paths": paths,
        "segmentation": segmentation,
        "instance_names": names,
        "compact_label_by_instance": label_map,
        "init_depth": init_depth,
    }


def _normalize_name(value: Any) -> str:
    text = str(value or "")
    text = re.sub(r"([a-z0-9])([A-Z])", r"\1 \2", text)
    text = text.replace("_", " ").replace("-", " ").lower()
    text = text.replace("object", " ")
    return " ".join(text.split())


def _match_score(candidate: str, query: str) -> int:
    candidate = _normalize_name(candidate)
    query = _normalize_name(query)
    if not candidate or not query:
        return 0
    score = 0
    if candidate == query:
        score += 100
    if candidate in query or query in candidate:
        score += 60
    score += 15 * len(set(candidate.split()).intersection(query.split()))
    if "visual" in candidate:
        score -= 30
    return score


def _best_matches(
    *,
    instance_names: list[str],
    queries: list[str],
) -> list[str]:
    scored = [
        (
            max(_match_score(name, query) for query in queries),
            name,
        )
        for name in instance_names
    ]
    scored = [(score, name) for score, name in scored if score > 0]
    if not scored:
        return []
    scored.sort(key=lambda item: (-item[0], item[1]))
    best_score = scored[0][0]
    best = [name for score, name in scored if score == best_score]
    non_visual = [name for name in best if "visual" not in _normalize_name(name)]
    return non_visual or best


def _explicit_matches(
    *,
    explicit_name: str,
    instance_names: list[str],
) -> list[str]:
    query = _normalize_name(explicit_name)
    exact = [name for name in instance_names if _normalize_name(name) == query]
    if exact:
        return exact
    return _best_matches(
        instance_names=instance_names,
        queries=[query],
    )


def _joint_prompt(value: Any) -> str:
    name = str(value or "").strip()
    name = name.replace("_joint0", "").replace("_main", "")
    name = name.replace("Visual", "").replace("_", " ")
    replacements = {
        "SquareNut": "square nut",
        "RoundNut": "round nut",
        "Can": "can",
        "Milk": "milk carton",
        "Bread": "bread",
        "Cereal": "cereal box",
    }
    for source, target in replacements.items():
        name = name.replace(source, target)
    return " ".join(name.split()).strip().lower()


def _default_object_prompt(
    simulator_config: Mapping[str, Any],
) -> str:
    raw = dict(simulator_config.get("raw", {}) or {})
    env_name = str(raw.get("env_name", "") or "").strip()
    if env_name in _ENV_PROMPTS:
        return _ENV_PROMPTS[env_name]
    names = sorted(
        {
            _joint_prompt(dict(item or {}).get("joint_name", ""))
            for item in list(raw.get("free_joints", []) or [])
            if isinstance(item, Mapping)
        }
    )
    names = [name for name in names if name]
    return names[0] if len(names) == 1 else ""


def resolve_saved_precomputed_region(
    *,
    target_name: str,
    segmentation: np.ndarray,
    instance_names: list[str],
    compact_label_by_instance: Mapping[str, int],
    simulator_config: Mapping[str, Any],
    target_config: Mapping[str, Any] | None = None,
    prompt: str = "",
) -> PrecomputedRegion | None:
    """Resolve the same saved-mask name matching as the current selector."""

    normalized_target = str(target_name or "").strip().lower()
    if normalized_target not in {"eef", "obj"}:
        raise ValueError(f"unsupported target_name: {target_name}")
    target = dict(target_config or {})
    simulation = dict(target.get("simulation", {}) or {})
    explicit_name = str(simulation.get("instance_name", "") or "").strip()
    names = [str(name) for name in instance_names]

    if explicit_name:
        matched = _explicit_matches(
            explicit_name=explicit_name,
            instance_names=names,
        )
        if not matched:
            raise RuntimeError(
                "[region] simulation.instance_name="
                f"'{explicit_name}' did not match any saved robosuite "
                "instance."
            )
    elif normalized_target == "eef":
        raw = dict(simulator_config.get("raw", {}) or {})
        eef = dict(raw.get("eef", {}) or {})
        arm = str(eef.get("arm", "right") or "right").strip().lower()
        matched = [name for name in names if "gripper" in _normalize_name(name)]
        if arm:
            arm_matches = [name for name in matched if arm in _normalize_name(name)]
            if arm_matches:
                matched = arm_matches
    else:
        primary = [
            _normalize_name(value)
            for value in (
                str(prompt or "").strip(),
                _default_object_prompt(simulator_config),
            )
            if str(value or "").strip()
        ]
        primary = list(dict.fromkeys(primary))
        raw = dict(simulator_config.get("raw", {}) or {})
        fallback = [
            _normalize_name(
                str(dict(item).get("joint_name", "") or "")
                .replace("_joint0", "")
                .replace("_main", "")
            )
            for item in list(raw.get("free_joints", []) or [])
            if isinstance(item, Mapping)
        ]
        fallback = list(dict.fromkeys(value for value in fallback if value))
        queries = primary or fallback
        candidates = [
            name
            for name in names
            if not any(
                token in _normalize_name(name)
                for token in ("gripper", "panda", "mount")
            )
        ]
        matched = (
            _best_matches(
                instance_names=candidates,
                queries=queries,
            )
            if queries
            else []
        )
        if not matched and primary and fallback:
            matched = _best_matches(
                instance_names=candidates,
                queries=fallback,
            )

    labels = [
        int(compact_label_by_instance[name])
        for name in matched
        if name in compact_label_by_instance
    ]
    if not labels:
        return None
    segmentation_array = np.asarray(segmentation, dtype=np.int32)
    mask = np.isin(
        segmentation_array,
        np.asarray(labels, dtype=np.int32),
    )
    if not bool(np.any(mask)):
        return None
    ys, xs = np.nonzero(mask)
    return PrecomputedRegion(
        mask=mask.astype(bool),
        bbox_xyxy=[
            int(xs.min()),
            int(ys.min()),
            int(xs.max()),
            int(ys.max()),
        ],
        source="robosuite_init_mask",
        matched_names=list(matched),
        compact_labels=labels,
    )


def build_benchmark_region_inputs(
    *,
    uid: str,
    sample_dir: str | Path,
    simulator_config: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Build explicit EEF/object region inputs for one trajectory run."""

    clean_uid = str(uid or "").strip()
    metadata_payload = copy_mapping(metadata)
    metadata_uid = str(metadata_payload.get("uid", "") or "").strip()
    if metadata_uid != clean_uid:
        raise ValueError(
            f"benchmark uid/region metadata mismatch: {clean_uid!r} != {metadata_uid!r}"
        )
    assets = load_benchmark_init_region_assets(sample_dir)
    region_targets = dict(
        dict(pipeline_config.get("region", {}) or {}).get(
            "targets",
            {},
        )
        or {}
    )
    eef_target = dict(region_targets.get("eef", {}) or {})
    eef_selector = str(eef_target.get("selector", "simulation") or "simulation").strip()
    eef_region: PrecomputedRegion | None = None
    if eef_selector in {"simulation", "auto"}:
        eef_region = resolve_saved_precomputed_region(
            target_name="eef",
            segmentation=assets["segmentation"],
            instance_names=assets["instance_names"],
            compact_label_by_instance=assets["compact_label_by_instance"],
            simulator_config=simulator_config,
            target_config=eef_target,
        )
        if eef_selector == "simulation" and eef_region is None:
            raise RuntimeError(
                "[region] eef requested simulation selector, but init "
                "segmentation mask is unavailable."
            )

    task_runtime = compile_task_runtime(
        uid=clean_uid,
        metadata=metadata_payload,
        pipeline_config=copy_mapping(pipeline_config),
    )
    object_regions: dict[str, PrecomputedRegion] = {}
    object_manifest: dict[str, Any] = {}
    object_fallbacks: dict[str, Any] = {}
    for stream in list(task_runtime.get("object_stream_plan", []) or []):
        stream_payload = dict(stream)
        object_id = str(stream_payload.get("object_id", "") or "")
        target = dict(stream_payload.get("region_target_cfg", {}) or {})
        selector = str(target.get("selector", "simulation") or "simulation").strip()
        # A visual object selector still needs the saved simulator mask as an
        # explicit fallback candidate.  RegionRuntime preserves the original
        # behavior: it tries the visual detector first and consumes this
        # candidate only when visual selection raises RuntimeError.
        if selector not in {"simulation", "auto", "visual"}:
            continue
        manipulated = dict(stream_payload.get("manipulated_object", {}) or {})
        region = resolve_saved_precomputed_region(
            target_name="obj",
            segmentation=assets["segmentation"],
            instance_names=assets["instance_names"],
            compact_label_by_instance=assets["compact_label_by_instance"],
            simulator_config=simulator_config,
            target_config=target,
            prompt=str(
                target.get(
                    "prompt",
                    manipulated.get("name", ""),
                )
                or ""
            ),
        )
        if selector == "simulation" and region is None:
            visual = dict(target.get("visual", {}) or {})
            object_fallbacks[object_id] = {
                "requested_selector": "simulation",
                "effective_selector": "visual",
                "reason": "init_segmentation_mask_unavailable",
                "bbox_source": str(
                    visual.get("bbox_source", "grounding_dino") or "grounding_dino"
                ),
            }
        if region is not None:
            object_regions[object_id] = region
            object_manifest[object_id] = {
                "matched_names": list(region.matched_names),
                "compact_labels": list(region.compact_labels),
                "bbox_xyxy": list(region.bbox_xyxy or []),
            }

    return {
        "object_runtime_options": {
            "eef_precomputed_region": eef_region,
            "precomputed_regions": object_regions,
            "init_depth": assets["init_depth"],
        },
        "manifest": {
            "paths": dict(assets["paths"]),
            "eef": (
                None
                if eef_region is None
                else {
                    "matched_names": list(eef_region.matched_names),
                    "compact_labels": list(eef_region.compact_labels),
                    "bbox_xyxy": list(eef_region.bbox_xyxy or []),
                }
            ),
            "objects": object_manifest,
            "object_fallbacks": object_fallbacks,
        },
    }


def copy_mapping(value: Mapping[str, Any]) -> dict[str, Any]:
    """Detach a JSON-like mapping without importing bench configuration code."""

    return json.loads(json.dumps(dict(value)))


__all__ = [
    "build_benchmark_region_inputs",
    "load_benchmark_init_region_assets",
    "resolve_benchmark_init_region_assets",
    "resolve_saved_precomputed_region",
]
