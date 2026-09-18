"""Read-only bench adapter for the pure trajectory configuration contract.

This module owns current-layout path interpretation.  It resolves an explicit
bench sample, run identity, canonical overrides, and generated-video namespace
before delegating value normalization and validation to
``dream_exe.video2traj.runtime.config``.

No function here creates directories, writes configuration, materializes a
sample, or imports a simulator.  Callers outside the current bench layout can
skip this adapter and call the pure configuration loader directly.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ...artifacts.layout import (
    pipeline_run_config_path,
    sample_artifact_paths,
    trajectory_artifact_paths,
)
from ..records.layout import (
    DEFAULT_FORMAL_RUN_KEY,
    default_run_key_for_video,
    formal_artifact_paths,
    load_resolved_kind_config,
    normalize_run_key,
    run_key_uses_gt_depth,
)
from .videos import generated_video_candidates
from ...video2traj.runtime.config import (
    DEFAULT_PIPELINE_CONFIG_FILENAME,
    load_pipeline_config as load_core_pipeline_config,
)


_LAYOUT_SENTINEL = Path("/") / "__dream_exe_sample_layout__"


def _sample_role_tail(role: str) -> tuple[str, ...]:
    path = sample_artifact_paths(_LAYOUT_SENTINEL)[role]
    return path.relative_to(_LAYOUT_SENTINEL).parts


def _tail_starts_at(
    tail: list[str],
    index: int,
    prefix: tuple[str, ...],
) -> bool:
    return tuple(tail[index : index + len(prefix)]) == prefix


def _legacy_bench_data_parts(path: str) -> Tuple[Tuple[str, ...], int]:
    """Parse the historical literal ``bench/data`` fallback only."""

    expanded = Path(str(path or "")).expanduser()
    for candidate in (expanded, expanded.resolve()):
        components = candidate.parts
        adjacent = zip(components, components[1:])
        for position, pair in enumerate(adjacent):
            if pair == ("bench", "data") and position + 2 < len(components):
                return components, position
    raise RuntimeError(f"cannot infer bench/data path from path={path}")


def infer_sample_dir_from_bench_path(path: str) -> str:
    """Return the existing ``bench/data/<uid>`` prefix without discovery."""

    parts, index = _legacy_bench_data_parts(path)
    uid_index = index + 2
    uid = str(parts[uid_index]).strip()
    if not uid:
        raise RuntimeError(f"cannot infer uid from bench/data/<uid>/... path={path}")
    return Path(*parts[: uid_index + 1]).resolve().as_posix()


def _sample_relative_tail(
    path: str | Path,
    *,
    sample_dir: str | Path | None,
) -> list[str]:
    if sample_dir is None:
        parts, index = _legacy_bench_data_parts(str(path))
        uid_index = index + 2
        return list(parts[uid_index + 1 :])

    candidate = Path(path).expanduser()
    sample = Path(sample_dir).expanduser()
    pairs = (
        (candidate.absolute(), sample.absolute()),
        (candidate.resolve(), sample.resolve()),
    )
    for candidate_path, sample_path in pairs:
        try:
            return list(candidate_path.relative_to(sample_path).parts)
        except ValueError:
            continue
    raise RuntimeError(
        f"cannot infer sample-relative path from path={path} sample_dir={sample_dir}"
    )


def infer_run_key_from_bench_path(
    path: str | Path,
    *,
    sample_dir: str | Path | None = None,
) -> str:
    """Interpret a run key relative to an explicit sample when available.

    Omitting ``sample_dir`` preserves the compatibility-only parser for paths
    containing the historical literal ``bench/data/<uid>`` prefix.
    """

    tail = _sample_relative_tail(path, sample_dir=sample_dir)

    pipeline_prefix = _sample_role_tail("pipeline_root")
    runs_prefix = _sample_role_tail("runs_root")
    artifacts_prefix = _sample_role_tail("artifacts_root")
    for tail_index in range(len(tail)):
        if not _tail_starts_at(tail, tail_index, pipeline_prefix):
            continue
        semantic_index = tail_index + len(pipeline_prefix)
        if semantic_index < len(tail) and tail[semantic_index] == "gen":
            return "gen"
        if semantic_index < len(tail) and tail[semantic_index] == "rollout":
            if semantic_index + 1 < len(tail) and tail[semantic_index + 1] in {
                "depth_model",
                "depth_model_pose",
                "depth_gt",
                "depth_gt_pose",
                "gt_depth",
                "gt_depth_pose",
            }:
                return normalize_run_key(f"rollout/{tail[semantic_index + 1]}")
            return default_run_key_for_video(video_kind="rollout")
        if (
            semantic_index + 2 < len(tail)
            and tail[semantic_index] == runs_prefix[-1]
        ):
            return normalize_run_key(
                f"{tail[semantic_index + 1]}/{tail[semantic_index + 2]}"
            )

    for tail_index in range(len(tail)):
        if not _tail_starts_at(tail, tail_index, artifacts_prefix):
            continue
        semantic_index = tail_index + len(artifacts_prefix)
        if semantic_index < len(tail) and tail[semantic_index] in {"traj", "exec"}:
            return DEFAULT_FORMAL_RUN_KEY
        if (
            _tail_starts_at(tail, tail_index, runs_prefix)
            and tail_index + len(runs_prefix) < len(tail)
            and tail[tail_index + len(runs_prefix)] == "gen"
        ):
            return "gen"
        if (
            _tail_starts_at(tail, tail_index, runs_prefix)
            and tail_index + len(runs_prefix) < len(tail)
            and tail[tail_index + len(runs_prefix)] == "rollout"
        ):
            run_kind_index = tail_index + len(runs_prefix) + 1
            if run_kind_index < len(tail) and tail[run_kind_index] in {
                "depth_model",
                "depth_model_pose",
                "depth_gt",
                "depth_gt_pose",
                "gt_depth",
                "gt_depth_pose",
            }:
                return normalize_run_key(f"rollout/{tail[run_kind_index]}")
            return DEFAULT_FORMAL_RUN_KEY
    return DEFAULT_FORMAL_RUN_KEY


def _model_leaf(value: Any) -> str:
    leaf = Path(str(value or "").strip()).name
    if leaf.casefold().endswith(".mp4"):
        leaf = leaf[:-4]
    for marker in (".old", ".ori"):
        if leaf.casefold().endswith(marker):
            leaf = leaf[: -len(marker)]
    return leaf.strip()


def normalize_gen_model_name(name: str) -> str:
    return _model_leaf(name)


def infer_gen_model_from_bench_path(
    path: str | Path,
    *,
    sample_dir: str | Path | None = None,
) -> str:
    """Infer a generated-model namespace relative to one sample."""

    tail = _sample_relative_tail(path, sample_dir=sample_dir)
    runs_gen_prefix = (*_sample_role_tail("runs_root"), "gen")
    generated_prefix = _sample_role_tail("generated_root")
    for tail_index in range(len(tail)):
        if (
            _tail_starts_at(tail, tail_index, runs_gen_prefix)
            and tail_index + len(runs_gen_prefix) < len(tail)
        ):
            return str(tail[tail_index + len(runs_gen_prefix)]).strip()
    for tail_index in range(len(tail)):
        if (
            _tail_starts_at(tail, tail_index, generated_prefix)
            and tail_index + len(generated_prefix) < len(tail)
        ):
            model = normalize_gen_model_name(
                tail[tail_index + len(generated_prefix)]
            )
            return "" if model == "rollout" else model
    return ""


def _sample_layout_tail_or_none(
    path: Path,
    *,
    sample_dir: Path | None,
) -> list[str] | None:
    try:
        return _sample_relative_tail(path, sample_dir=sample_dir)
    except RuntimeError:
        return None


def _is_sample_experiment_path(
    path: Path,
    *,
    sample_dir: Path | None,
) -> bool:
    tail = _sample_layout_tail_or_none(path, sample_dir=sample_dir)
    if tail is not None:
        experiment_prefix = _sample_role_tail("sample_experiment_root")
        return any(
            _tail_starts_at(tail, index, experiment_prefix)
            for index in range(len(tail))
        )
    return "experiment" in path.parts


def _is_explicit_run_snapshot(
    path: Path,
    *,
    sample_dir: Path | None,
) -> bool:
    if path.name.endswith(".resolved.json"):
        return True
    tail = _sample_layout_tail_or_none(path, sample_dir=sample_dir)
    if tail is not None:
        runs_prefix = _sample_role_tail("runs_root")
        return any(
            _tail_starts_at(tail, index, runs_prefix)
            for index in range(len(tail))
        )
    return "/artifacts/runs/" in path.as_posix()


def pipeline_config_path_from_dataset_config(
    dataset_config_path: str,
    *,
    filename: str = DEFAULT_PIPELINE_CONFIG_FILENAME,
) -> str:
    """Compatibility-only mapping from an explicit dataset config path.

    This reproduces the current path rule; it does not search for a sample,
    check existence, or create the returned parent directory.
    """

    dataset_path = Path(dataset_config_path).expanduser().resolve()
    environment_tail = _sample_role_tail("environment_dir")
    parent_parts = dataset_path.parent.parts
    if tuple(parent_parts[-len(environment_tail) :]) == environment_tail:
        sample_root = Path(*parent_parts[: -len(environment_tail)])
        return pipeline_run_config_path(
            sample_root,
            ("rollout",),
            filename,
        ).as_posix()
    return (dataset_path.parent / filename).as_posix()


def _load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _project_current_layout_trajectory_config(
    config: Any,
) -> tuple[Any, tuple[str, ...]]:
    """Select the public current implementation depth schema from one detached bench config.

    Bench documents remain immutable experiment records and may contain
    producer-owned depth controls that are not inputs to the public algorithm.
    This outer current-layout adapter selects only fields present in the current implementation
    default depth contract before strict value validation. It neither names nor
    implements any retired depth algorithm, and it never modifies the source
    document. Explicit non-bench configs still go directly to the strict core
    validator and fail on unknown fields.
    """

    if not isinstance(config, dict):
        return copy.deepcopy(config), ()
    projected = copy.deepcopy(config)
    depth = projected.get("depth")
    if not isinstance(depth, dict):
        return projected, ()
    canonical_depth = load_core_pipeline_config()["depth"]
    ignored = {f"depth.{key}" for key in depth if key not in canonical_depth}
    depth = {
        key: copy.deepcopy(value)
        for key, value in depth.items()
        if key in canonical_depth
    }
    for nested_key in ("base", "artifact_policy"):
        nested = depth.get(nested_key)
        canonical_nested = canonical_depth[nested_key]
        if isinstance(nested, dict) and isinstance(
            canonical_nested,
            dict,
        ):
            ignored.update(
                f"depth.{nested_key}.{key}"
                for key in nested
                if key not in canonical_nested
            )
            depth[nested_key] = {
                key: copy.deepcopy(value)
                for key, value in nested.items()
                if key in canonical_nested
            }
    projected["depth"] = depth
    return projected, tuple(sorted(ignored))


def load_bench_pipeline_config(
    *,
    sample_dir: Optional[str] = None,
    dataset_config_path: Optional[str] = None,
    default_config_path: Optional[str] = None,
    pipeline_config_path: Optional[str] = None,
    pipeline_config_filename: str = (DEFAULT_PIPELINE_CONFIG_FILENAME),
    run_key: Optional[str] = None,
    gen_model: str = "",
    _source_config_out: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Resolve current bench config precedence without modifying the sample.

    ``sample_dir`` is the explicit path anchor for callers that already know
    the sample root, including samples below a custom configured bench root.
    Otherwise it is inferred only through the compatibility parser for a
    supplied path containing ``bench/data/<uid>``.  No global config root is
    consulted.
    """

    source_path: Optional[Path] = None
    ignored_nonpublic_depth_fields: tuple[str, ...] = ()
    resolved_sample_dir = (
        Path(sample_dir).expanduser().resolve()
        if str(sample_dir or "").strip()
        else None
    )
    resolved_run_key = str(run_key or "").strip()

    if pipeline_config_path is not None and str(pipeline_config_path).strip():
        source_path = Path(pipeline_config_path).expanduser().resolve()
        if not source_path.exists():
            raise FileNotFoundError(f"pipeline config not found: {source_path}")
        try:
            if (
                not _is_sample_experiment_path(
                    source_path,
                    sample_dir=resolved_sample_dir,
                )
                and not resolved_run_key
            ):
                resolved_run_key = infer_run_key_from_bench_path(
                    source_path.as_posix(),
                    sample_dir=resolved_sample_dir,
                )
            if resolved_sample_dir is None:
                resolved_sample_dir = Path(
                    infer_sample_dir_from_bench_path(source_path.as_posix())
                )
        except Exception:
            if not str(sample_dir or "").strip():
                resolved_sample_dir = None
    elif default_config_path is not None:
        candidate = Path(default_config_path).expanduser().resolve()
        if candidate.exists():
            source_path = candidate
            try:
                if not resolved_run_key:
                    resolved_run_key = infer_run_key_from_bench_path(
                        source_path.as_posix(),
                        sample_dir=resolved_sample_dir,
                    )
                if resolved_sample_dir is None:
                    resolved_sample_dir = Path(
                        infer_sample_dir_from_bench_path(source_path.as_posix())
                    )
            except Exception:
                if not str(sample_dir or "").strip():
                    resolved_sample_dir = None
    elif dataset_config_path is not None:
        candidate = Path(
            pipeline_config_path_from_dataset_config(
                dataset_config_path,
                filename=pipeline_config_filename,
            )
        )
        if candidate.exists():
            source_path = candidate.resolve()
            try:
                if not resolved_run_key:
                    resolved_run_key = infer_run_key_from_bench_path(
                        source_path.as_posix(),
                        sample_dir=resolved_sample_dir,
                    )
                if resolved_sample_dir is None:
                    resolved_sample_dir = Path(
                        infer_sample_dir_from_bench_path(source_path.as_posix())
                    )
            except Exception:
                if not str(sample_dir or "").strip():
                    resolved_sample_dir = None

    if gen_model and not resolved_run_key:
        resolved_run_key = "gen"
    if resolved_sample_dir is not None and source_path is None and not resolved_run_key:
        resolved_run_key = DEFAULT_FORMAL_RUN_KEY

    use_explicit_snapshot = bool(
        source_path is not None
        and _is_explicit_run_snapshot(
            source_path,
            sample_dir=resolved_sample_dir,
        )
    )
    source_config: Any = {}
    source = "<built-in default>"
    source_selected = False
    if use_explicit_snapshot:
        source_config = _load_json(source_path)
        source = source_path.as_posix()
        source_selected = True
    elif resolved_sample_dir is not None and resolved_run_key:
        source_config, canonical_meta = load_resolved_kind_config(
            sample_dir=resolved_sample_dir,
            run_key=resolved_run_key,
            kind="trajectory",
            gen_model=gen_model,
        )
        (
            source_config,
            ignored_nonpublic_depth_fields,
        ) = _project_current_layout_trajectory_config(source_config)
        source = str(canonical_meta.get("source", "") or "<canonical>")
        source_selected = True
    elif source_path is not None:
        source_config = _load_json(source_path)
        source = source_path.as_posix()
        source_selected = True

    if source_selected:
        normalized = load_core_pipeline_config(
            source_config,
            source=source,
        )
    else:
        normalized = load_core_pipeline_config()
    normalized["_meta"] = {
        "source": source,
        "exists_on_disk": source_path is not None,
        "run_key": resolved_run_key or None,
        "sample_dir": (
            resolved_sample_dir.as_posix() if resolved_sample_dir is not None else None
        ),
    }
    if ignored_nonpublic_depth_fields:
        normalized["_meta"]["ignored_nonpublic_depth_fields"] = list(
            ignored_nonpublic_depth_fields
        )
    if _source_config_out is not None:
        _source_config_out.clear()
        _source_config_out["config"] = copy.deepcopy(source_config)
    return normalized


def resolve_default_bench_pipeline_paths(
    *,
    sample_dir: str,
    run_key: str = DEFAULT_FORMAL_RUN_KEY,
    gen_model: str = "",
    output_traj_root: Optional[str] = None,
) -> Dict[str, Any]:
    """Construct current default input and trajectory artifact paths.

    Existence is consulted only to select among generated-video candidates.
    The function never creates any returned directory or file.
    """

    sample_root = Path(sample_dir).expanduser().resolve()
    normalized_run_key = normalize_run_key(run_key)
    normalized_gen_model = normalize_gen_model_name(gen_model)
    formal_paths = formal_artifact_paths(
        sample_root,
        normalized_run_key,
        gen_model=normalized_gen_model,
    )
    trajectory_root = (
        Path(output_traj_root).expanduser().resolve()
        if str(output_traj_root or "").strip()
        else formal_paths["traj_dir"]
    )
    trajectory_paths = trajectory_artifact_paths(trajectory_root)
    sample_paths = sample_artifact_paths(sample_root)
    rollout_video_path = sample_paths["gt_video"]
    rollout_gt_depth_path = sample_paths["gt_metric_depth"]

    generated_video_path = ""
    if normalized_run_key == "gen":
        candidates = generated_video_candidates(
            sample_root,
            normalized_gen_model,
        )
        generated_video_path = next(
            (candidate.as_posix() for candidate in candidates if candidate.exists()),
            "",
        )

    return {
        "sample_dir": sample_root.as_posix(),
        "run_key": normalized_run_key,
        "gen_model": normalized_gen_model,
        "run_root": formal_paths["run_root"].as_posix(),
        "traj_dir": trajectory_root.as_posix(),
        "rollout_video_path": rollout_video_path.as_posix(),
        "gen_video_path": generated_video_path,
        "rollout_gt_depth_path": (rollout_gt_depth_path.as_posix()),
        "use_rollout_gt_depth": bool(run_key_uses_gt_depth(normalized_run_key)),
        "estimated_depth_cache_path": (trajectory_paths["depth_npy"].as_posix()),
        "visualization_video_path": (trajectory_paths["depth_mp4"].as_posix()),
        "depth_meta_path": (trajectory_paths["depth_meta_npy"].as_posix()),
        "canonical_depth_npy_path": (trajectory_paths["depth_npy"].as_posix()),
        "depth_manifest_path": (trajectory_paths["depth_manifest_json"].as_posix()),
    }


__all__ = [
    "infer_gen_model_from_bench_path",
    "infer_run_key_from_bench_path",
    "infer_sample_dir_from_bench_path",
    "load_bench_pipeline_config",
    "normalize_gen_model_name",
    "pipeline_config_path_from_dataset_config",
    "resolve_default_bench_pipeline_paths",
]
