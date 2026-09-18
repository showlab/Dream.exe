"""Resolve materialized single-case run identities and artifact paths.

The functions in this module describe run identities, locate existing
configuration inputs, and construct artifact paths.  They intentionally avoid
environment runtimes and never create benchmark files or directories.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

from ...artifacts.layout import (
    EXECUTION_CONFIG_FILENAME,
    GENERATED_DIRECTORY,
    PIPELINE_STATE_FILENAME as ARTIFACT_PIPELINE_STATE_FILENAME,
    RUN_OVERRIDES_FILENAME,
    TRAJECTORY_CONFIG_FILENAME,
    execution_artifact_paths,
    run_artifact_paths,
    run_artifact_paths_for_sample,
    sample_artifact_paths,
    trajectory_artifact_paths,
)
from ..planning.videos import generated_video_candidates


DEFAULT_FORMAL_RUN_KEY = "gt_video/dvd_depth"
FORMAL_RUN_KEYS = (
    "gt_video/dvd_depth",
    "gt_video/gt_depth",
    "gen",
)
RUNS_MANIFEST_FILENAME = "manifest.json"

TRAJECTORY_BASE_FILENAME = TRAJECTORY_CONFIG_FILENAME
EXECUTION_BASE_FILENAME = EXECUTION_CONFIG_FILENAME
LEGACY_TRAJECTORY_BASE_FILENAME = "trajectory.base.json"
LEGACY_EXECUTION_BASE_FILENAME = "execution.base.json"
PIPELINE_STATE_FILENAME = ARTIFACT_PIPELINE_STATE_FILENAME
_MODEL_DEPTH_NAMES = frozenset({"dvd_depth"})
_GT_DEPTH_NAMES = frozenset({"gt_depth"})
_GENERATED_DEPTH_NAMES = _MODEL_DEPTH_NAMES
_CONFIG_KINDS = ("trajectory", "execution")

# The exact key set participates in experiment reproducibility fingerprints.
# Central role helpers may grow, but this projection changes only through an
# explicit schema update.
_FORMAL_TRAJECTORY_ROLE_KEYS = (
    "traj_dir",
    "trajectory_dir",
    "ee_traj",
    "obj_trajs",
    "union_traj",
    "gripper_dir",
    "gripper",
    "action_dir",
    "action",
    "action_array",
    "depth_dir",
    "trajectory_manifest",
)
_FORMAL_EXECUTION_ROLE_KEYS = (
    "exec_dir",
    "exec_summary",
    "checkpoint_trace",
    "dense_tcp_trace",
    "exec_metrics",
    "exec_metrics_per_frame",
    "execution_inputs",
    "action_trace",
    "execution_video",
    "evaluation_result",
    "task_success",
    "execution_manifest",
)


def normalize_run_key(run_key: str) -> str:
    """Return the canonical identity for a supported formal run key."""

    original = run_key
    raw = str(run_key or "").strip()
    if not raw:
        return DEFAULT_FORMAL_RUN_KEY

    pieces = [
        piece.strip() for piece in raw.replace("\\", "/").split("/") if piece.strip()
    ]
    if not pieces or len(pieces) > 2:
        raise ValueError(f"Invalid formal run key: {original}")

    axis = pieces[0].lower()
    if axis == "gt_video":
        if len(pieces) == 1:
            return DEFAULT_FORMAL_RUN_KEY
        slot = pieces[1]
        if slot in _MODEL_DEPTH_NAMES:
            return "gt_video/dvd_depth"
        if slot in _GT_DEPTH_NAMES:
            return "gt_video/gt_depth"
    elif axis == "gen":
        if len(pieces) == 1:
            return "gen"
        if pieces[1] in _GENERATED_DEPTH_NAMES:
            return "gen"

    raise ValueError(f"Unsupported formal run key: {original}")


def split_run_key(run_key: str) -> Tuple[str, str]:
    normalized = normalize_run_key(run_key)
    if normalized == "gen":
        return "gen", ""
    _axis, slot = normalized.split("/", 1)
    # The simulator-independent trajectory core calls a reference video
    # "rollout". Keep that internal enum separate from the public run path.
    return "rollout", slot


def pipeline_axis_for_run_key(run_key: str) -> str:
    return split_run_key(run_key)[0]


def run_key_uses_gt_depth(run_key: str) -> bool:
    return normalize_run_key(run_key) == "gt_video/gt_depth"


def run_key_uses_model_depth(run_key: str) -> bool:
    return not run_key_uses_gt_depth(run_key)


def run_key_has_pose(run_key: str) -> bool:
    # The formal run matrix has no separate pose dimension.
    return False


def pipeline_run_dir_parts(run_key: str) -> Tuple[str, ...]:
    normalized = normalize_run_key(run_key)
    if normalized == "gen":
        return ("gen",)
    axis, slot = normalized.split("/", 1)
    return (axis,) if not slot else (axis, slot)


def artifact_run_dir_parts(
    run_key: str,
    *,
    gen_model: str = "",
) -> Tuple[str, ...]:
    normalized = normalize_run_key(run_key)
    if normalized != "gen":
        return pipeline_run_dir_parts(normalized)

    model = str(gen_model or "").strip()
    if not model:
        raise ValueError(f"gen run requires a model name: {run_key}")
    if (
        model in {".", ".."}
        or "/" in model
        or "\\" in model
        or Path(model).name != model
    ):
        raise ValueError(f"unsafe generated model name: {gen_model!r}")
    return "gen", model


def build_run_id(
    *,
    run_key: str,
    gen_model: str = "",
) -> str:
    return "__".join(artifact_run_dir_parts(run_key, gen_model=gen_model))


def default_run_key_for_video(
    *,
    video_kind: str,
    pose_enabled: bool = False,
    use_gt_depth: bool = False,
) -> str:
    kind = str(video_kind or "").strip().lower()
    if kind in {"", "rollout"}:
        return "gt_video/gt_depth" if use_gt_depth else DEFAULT_FORMAL_RUN_KEY
    if kind == "gen":
        if use_gt_depth:
            raise ValueError("generated-video formal runs do not support GT depth")
        return "gen"
    raise ValueError(f"Unsupported video kind: {video_kind}")


def is_backup_video_name(name: str) -> bool:
    normalized = str(name or "").strip().lower()
    return normalized.endswith((".old.mp4", ".ori.mp4"))


def formal_run_keys() -> List[str]:
    return list(FORMAL_RUN_KEYS)


@dataclass(frozen=True)
class ResolvedBenchmarkRun:
    run_key: str
    run_id: str
    video_kind: str
    slot_name: str
    gen_model: str
    run_dir_parts: Tuple[str, ...]

    def to_manifest_record(self) -> Dict[str, Any]:
        return {
            "run_key": self.run_key,
            "run_id": self.run_id,
            "video_kind": self.video_kind,
            "slot_name": self.slot_name or None,
            "gen_model": self.gen_model or None,
            "uses_gt_depth": run_key_uses_gt_depth(self.run_key),
            "uses_model_depth": run_key_uses_model_depth(self.run_key),
            "pose_enabled": run_key_has_pose(self.run_key),
            "run_dir_parts": list(self.run_dir_parts),
        }


def _ordered_clean_strings(
    values: Sequence[str] | None,
) -> List[str]:
    ordered: List[str] = []
    seen: set[str] = set()
    for value in values or ():
        text = str(value or "").strip()
        if text and text not in seen:
            seen.add(text)
            ordered.append(text)
    return ordered


def expand_available_runs(
    run_keys: Sequence[str],
    *,
    available_gen_models: Sequence[str],
) -> Tuple[List[Dict[str, Any]], List[ResolvedBenchmarkRun]]:
    normalized_keys: List[str] = []
    seen_keys: set[str] = set()
    for value in run_keys or ():
        normalized = normalize_run_key(value)
        if normalized not in seen_keys:
            seen_keys.add(normalized)
            normalized_keys.append(normalized)

    models = _ordered_clean_strings(available_gen_models)
    abstract: List[Dict[str, Any]] = []
    resolved: List[ResolvedBenchmarkRun] = []
    for run_key in normalized_keys:
        video_kind, slot_name = split_run_key(run_key)
        records: List[ResolvedBenchmarkRun] = []
        if video_kind == "gen":
            for model in models:
                records.append(
                    ResolvedBenchmarkRun(
                        run_key=run_key,
                        run_id=build_run_id(
                            run_key=run_key,
                            gen_model=model,
                        ),
                        video_kind=video_kind,
                        slot_name="",
                        gen_model=model,
                        run_dir_parts=artifact_run_dir_parts(
                            run_key,
                            gen_model=model,
                        ),
                    )
                )
        else:
            records.append(
                ResolvedBenchmarkRun(
                    run_key=run_key,
                    run_id=build_run_id(run_key=run_key),
                    video_kind=video_kind,
                    slot_name=slot_name,
                    gen_model="",
                    run_dir_parts=artifact_run_dir_parts(run_key),
                )
            )

        resolved.extend(records)
        abstract.append(
            {
                "run_key": run_key,
                "video_kind": video_kind,
                "slot_name": slot_name or None,
                "status": ("resolved" if records else "skipped_missing_gen_video"),
                "resolved_run_ids": [record.run_id for record in records],
            }
        )
    return abstract, resolved


def _present(path: Path) -> bool:
    return path.exists()


def discover_pipeline_run_keys(
    pipeline_dir: str,
) -> List[str]:
    root = Path(str(pipeline_dir or "")).expanduser().resolve()
    canonical_pair = (
        root / TRAJECTORY_BASE_FILENAME,
        root / EXECUTION_BASE_FILENAME,
    )
    dot_base_pair = (
        root / LEGACY_TRAJECTORY_BASE_FILENAME,
        root / LEGACY_EXECUTION_BASE_FILENAME,
    )
    if all(_present(path) for path in canonical_pair) or all(
        _present(path) for path in dot_base_pair
    ):
        overrides = root / RUN_OVERRIDES_FILENAME
        if _present(overrides):
            _read_json_object(overrides)
        return formal_run_keys()

    scoped_rollout: List[str] = []
    for run_key in FORMAL_RUN_KEYS[:2]:
        parts = pipeline_run_dir_parts(run_key)
        run_root = root.joinpath(*parts)
        if _present(run_root / TRAJECTORY_BASE_FILENAME) and _present(
            run_root / EXECUTION_BASE_FILENAME
        ):
            scoped_rollout.append(run_key)

    inferred: List[str] = []
    if scoped_rollout:
        inferred.extend(scoped_rollout)
    else:
        rollout_root = root / "rollout"
        if _present(rollout_root / TRAJECTORY_BASE_FILENAME) and _present(
            rollout_root / EXECUTION_BASE_FILENAME
        ):
            inferred.extend(FORMAL_RUN_KEYS[:2])
    generated_root = root / GENERATED_DIRECTORY
    if _present(generated_root / TRAJECTORY_BASE_FILENAME) and _present(
        generated_root / EXECUTION_BASE_FILENAME
    ):
        inferred.append("gen")
    return inferred


def _sample_path(sample_dir: str | Path) -> Path:
    return Path(sample_dir).expanduser().resolve()


def pipeline_root(sample_dir: str | Path) -> Path:
    return sample_artifact_paths(sample_dir)["pipeline_root"]


def trajectory_base_path(sample_dir: str | Path) -> Path:
    return pipeline_root(sample_dir) / TRAJECTORY_BASE_FILENAME


def execution_base_path(sample_dir: str | Path) -> Path:
    return pipeline_root(sample_dir) / EXECUTION_BASE_FILENAME


def legacy_trajectory_base_path(
    sample_dir: str | Path,
) -> Path:
    return pipeline_root(sample_dir) / LEGACY_TRAJECTORY_BASE_FILENAME


def legacy_execution_base_path(
    sample_dir: str | Path,
) -> Path:
    return pipeline_root(sample_dir) / LEGACY_EXECUTION_BASE_FILENAME


def run_overrides_path(sample_dir: str | Path) -> Path:
    return pipeline_root(sample_dir) / RUN_OVERRIDES_FILENAME


def is_canonical_pipeline_layout_present(
    sample_dir: str | Path,
) -> bool:
    candidates = (
        trajectory_base_path(sample_dir),
        execution_base_path(sample_dir),
        legacy_trajectory_base_path(sample_dir),
        legacy_execution_base_path(sample_dir),
        run_overrides_path(sample_dir),
    )
    return any(_present(path) for path in candidates)


def infer_virtual_pipeline_kind_and_run_key(
    rel_path: str,
) -> Tuple[str, str]:
    normalized = str(rel_path or "").replace("\\", "/")
    parts = normalized.split("/")
    kind = ""
    run_key = ""
    if len(parts) == 3 and parts[:2] == ["pipeline", "gen"]:
        kind = Path(parts[2]).stem
        run_key = "gen"
    elif (
        len(parts) == 4
        and parts[:2] == ["pipeline", "rollout"]
        and parts[2] in {"depth_model", "gt_depth"}
    ):
        kind = Path(parts[3]).stem
        run_key = f"rollout/{parts[2]}"

    expected_name = f"{kind}.json" if kind in _CONFIG_KINDS else ""
    if not run_key or not expected_name or parts[-1] != expected_name:
        raise ValueError(f"unsupported run-scoped pipeline rel_path: {rel_path}")
    return kind, run_key


def is_virtual_run_pipeline_rel_path(rel_path: str) -> bool:
    try:
        infer_virtual_pipeline_kind_and_run_key(rel_path)
    except ValueError:
        return False
    return True


def legacy_run_config_paths(
    sample_dir: str | Path,
    run_key: str,
) -> Tuple[Path, Path]:
    run_root = pipeline_root(sample_dir).joinpath(*pipeline_run_dir_parts(run_key))
    return (
        run_root / TRAJECTORY_BASE_FILENAME,
        run_root / EXECUTION_BASE_FILENAME,
    )


def formal_artifact_paths(
    sample_root: str | Path,
    run_key: str = DEFAULT_FORMAL_RUN_KEY,
    *,
    gen_model: str = "",
) -> Dict[str, Path]:
    if not str(sample_root or "").strip():
        raise ValueError("sample_root is required")
    sample_paths = sample_artifact_paths(sample_root)
    sample = sample_paths["sample_root"]
    run_paths = run_artifact_paths_for_sample(
        sample,
        artifact_run_dir_parts(
            run_key,
            gen_model=gen_model,
        ),
    )
    trajectory_paths = trajectory_artifact_paths(run_paths["traj_dir"])
    execution_paths = execution_artifact_paths(run_paths["exec_dir"])
    return {
        "sample_root": sample,
        "sample_metadata": sample_paths["sample_metadata"],
        "simulator_config": sample_paths["simulator_config"],
        "scene_override": sample_paths["scene_override"],
        "run_root": run_paths["run_root"],
        **{key: trajectory_paths[key] for key in _FORMAL_TRAJECTORY_ROLE_KEYS},
        **{key: execution_paths[key] for key in _FORMAL_EXECUTION_ROLE_KEYS},
    }


def benchmark_pipeline_state_path(
    sample_root: str | Path,
    run_key: str = DEFAULT_FORMAL_RUN_KEY,
    *,
    gen_model: str = "",
) -> Path:
    """Return the contained current implementation stage-state sidecar for one formal run.

    The helper performs no write. Existing symlinks in the formal run path
    are resolved so a run directory cannot redirect the sidecar outside the
    explicit benchmark sample.
    """

    if sample_root is None or not str(sample_root).strip():
        raise ValueError("sample_root is required for benchmark pipeline state")
    sample = _sample_path(sample_root)
    run_root = formal_artifact_paths(
        sample,
        run_key,
        gen_model=gen_model,
    )["run_root"]
    lexical_run_root = run_root.absolute()
    candidate = sample
    for relative_part in lexical_run_root.relative_to(sample).parts:
        candidate = candidate / relative_part
        if candidate.is_symlink():
            raise ValueError("benchmark pipeline state path cannot traverse a symlink")
    resolved_run_root = run_root.resolve(strict=False)
    if not resolved_run_root.is_relative_to(sample):
        raise ValueError(
            "benchmark pipeline state path escapes the explicit sample root"
        )
    destination = run_artifact_paths(resolved_run_root)["pipeline_state"]
    if destination.is_symlink():
        raise ValueError("benchmark pipeline state path cannot be an existing symlink")
    return destination


def _read_json_object(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    return loaded if isinstance(loaded, dict) else {}


def _normalized_target_spec(value: Any) -> None:
    if not isinstance(value, dict):
        return
    visual = value.get("visual")
    if not isinstance(visual, dict):
        return
    source = visual.get("bbox_source")
    if str(source or "").strip().lower() == "manual_bbox":
        visual["bbox_source"] = "manual"


def cleanup_trajectory_storage_aliases(
    config: Dict[str, Any],
) -> Dict[str, Any]:
    if not isinstance(config, dict):
        return {}
    detached = copy.deepcopy(config)
    region = detached.get("region")
    if not isinstance(region, dict):
        return detached
    targets = region.get("targets")
    if not isinstance(targets, dict):
        return detached

    _normalized_target_spec(targets.get("obj"))
    _normalized_target_spec(targets.get("eef"))
    objects = targets.get("objects")
    if isinstance(objects, list):
        for item in objects:
            _normalized_target_spec(item)
        if objects:
            targets.pop("obj", None)
    return detached


def load_run_overrides(
    sample_dir: str | Path,
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    source = run_overrides_path(sample_dir)
    if not _present(source):
        return {}
    raw = _read_json_object(source)
    normalized: Dict[str, Dict[str, Dict[str, Any]]] = {}
    for run_key in FORMAL_RUN_KEYS:
        raw_entry = raw.get(run_key)
        if not isinstance(raw_entry, dict):
            continue
        entry: Dict[str, Dict[str, Any]] = {}
        trajectory = raw_entry.get("trajectory")
        if isinstance(trajectory, dict) and trajectory:
            cleaned = cleanup_trajectory_storage_aliases(trajectory)
            if cleaned:
                entry["trajectory"] = cleaned
        execution = raw_entry.get("execution")
        if isinstance(execution, dict) and execution:
            entry["execution"] = copy.deepcopy(execution)
        if entry:
            normalized[run_key] = entry
    return normalized


def _merge_objects(
    base: Dict[str, Any],
    overlay: Dict[str, Any],
) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, overlay_value in overlay.items():
        base_value = result.get(key)
        if isinstance(base_value, dict) and isinstance(overlay_value, dict):
            result[key] = _merge_objects(
                base_value,
                overlay_value,
            )
        else:
            result[key] = copy.deepcopy(overlay_value)
    return result


def _generated_video_candidate(
    sample: Path,
    model: str,
) -> str:
    for path in generated_video_candidates(sample, model):
        if _present(path):
            return path.resolve().as_posix()
    return ""


def _with_trajectory_runtime_paths(
    config: Dict[str, Any],
    *,
    sample: Path,
    run_key: str,
    gen_model: str,
) -> Dict[str, Any]:
    result = copy.deepcopy(config)
    video_kind = pipeline_axis_for_run_key(run_key)
    input_config = result.setdefault("input", {})
    input_config["selected_video"] = video_kind
    sample_paths = sample_artifact_paths(sample)
    input_config["rollout_video_path"] = sample_paths["gt_video"].as_posix()

    model = str(gen_model or "").strip()
    if video_kind == "rollout":
        input_config["gen_video_path"] = ""
    elif model:
        input_config["gen_video_path"] = _generated_video_candidate(sample, model)

    depth_config = result.setdefault("depth", {})
    depth_config["rollout_gt_depth_path"] = sample_paths[
        "gt_metric_depth"
    ].as_posix()
    depth_config["use_rollout_gt_depth"] = run_key_uses_gt_depth(run_key)
    return result


def load_resolved_kind_config(
    *,
    sample_dir: str | Path,
    run_key: str,
    kind: str,
    gen_model: str = "",
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    normalized_key = normalize_run_key(run_key)
    if kind not in _CONFIG_KINDS:
        raise ValueError(f"unsupported pipeline config kind: {kind}")

    sample = _sample_path(sample_dir)
    legacy_paths = legacy_run_config_paths(
        sample,
        normalized_key,
    )
    legacy_path = legacy_paths[0] if kind == "trajectory" else legacy_paths[1]
    if kind == "trajectory":
        primary_base = trajectory_base_path(sample)
        dot_base = legacy_trajectory_base_path(sample)
    else:
        primary_base = execution_base_path(sample)
        dot_base = legacy_execution_base_path(sample)

    selected_base: Path | None = None
    if _present(primary_base):
        selected_base = primary_base
    elif _present(dot_base):
        selected_base = dot_base
    overrides_file = run_overrides_path(sample)
    use_canonical = selected_base is not None or _present(overrides_file)

    if not use_canonical:
        payload = _read_json_object(legacy_path) if _present(legacy_path) else {}
        return payload, {
            "mode": "legacy",
            "source": legacy_path.as_posix(),
            "override_source": "",
            "legacy_mirror": legacy_path.as_posix(),
            "run_key": normalized_key,
            "kind": kind,
        }

    payload = _read_json_object(selected_base) if selected_base is not None else {}
    overrides = load_run_overrides(sample)
    if normalized_key == "gen":
        payload = _merge_objects(
            payload,
            overrides.get("gt_video/dvd_depth", {}).get(kind, {}),
        )
    payload = _merge_objects(
        payload,
        overrides.get(normalized_key, {}).get(kind, {}),
    )
    if kind == "trajectory":
        payload = _with_trajectory_runtime_paths(
            payload,
            sample=sample,
            run_key=normalized_key,
            gen_model=gen_model,
        )

    return payload, {
        "mode": "canonical",
        "source": (
            selected_base.as_posix()
            if selected_base is not None
            else "<missing canonical base>"
        ),
        "override_source": (
            overrides_file.as_posix() if _present(overrides_file) else ""
        ),
        "legacy_mirror": legacy_path.as_posix(),
        "run_key": normalized_key,
        "kind": kind,
    }
