"""Resolve and prepare videos consumed by a materialized single-case run.

The provider-neutral import and normalization code lives in
``dream_exe.generation.sources``.  This module is the narrow bench boundary that
maps a semantic sample/model request onto the existing
``bench/data/<uid>/artifacts`` layout.

Callers pass the resolved bench data root explicitly.  Resolving a configurable
root and deciding which benchmark is active belong to application
configuration, while the per-sample storage mapping remains centralized here.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path
import stat
from typing import Any, Dict

from ...artifacts.layout import bench_sample_path, sample_artifact_paths
from ...generation.sources import (
    import_generated_videos,
    normalize_generated_model_name,
    normalize_generated_video,
)

GENERATED_VIDEO_SOURCE_KINDS = ("normal", "enhanced")
GENERATED_VIDEO_STAGES = ("raw", "processed")


def bench_sample_root(
    bench_data_root: str | Path,
    uid: str,
) -> Path:
    """Compatibility wrapper for the central pure sample resolver."""

    return bench_sample_path(bench_data_root, uid)


def generated_video_stage_dir(
    sample_root: str | Path,
    *,
    stage: str,
    source_kind: str = "normal",
) -> Path:
    """Map a generated-video stage to the active current artifact layout."""

    normalized_stage = str(stage or "").strip().lower()
    if normalized_stage not in GENERATED_VIDEO_STAGES:
        raise ValueError(f"unsupported generated-video stage: {stage}")
    normalized_source = str(source_kind or "").strip().lower()
    if normalized_source not in GENERATED_VIDEO_SOURCE_KINDS:
        raise ValueError(f"unsupported generated-video source kind: {source_kind}")
    paths = sample_artifact_paths(sample_root)
    role = (
        f"generated_{normalized_stage}_dir"
        if normalized_source == "normal"
        else f"generated_enhanced_{normalized_stage}_dir"
    )
    return paths[role]


def generated_video_path(
    sample_root: str | Path,
    model: str,
    *,
    stage: str = "processed",
    source_kind: str = "normal",
) -> Path:
    """Return the current normalized ``<model>.mp4`` artifact path."""

    model_name = normalize_generated_model_name(model)
    if not model_name:
        raise ValueError("generated-video model is required")
    return (
        generated_video_stage_dir(
            sample_root,
            stage=stage,
            source_kind=source_kind,
        )
        / f"{model_name}.mp4"
    )


def generated_video_candidates(
    sample_root: str | Path,
    model: str,
) -> tuple[Path, Path, Path]:
    """Return current processed/raw/legacy candidates for one model.

    ``-enhanced`` is a run-identity suffix, not part of the stored video
    filename.  Existence checks stay with the caller so this layout function
    remains usable by both strict and compatibility readers.
    """

    model_name = normalize_generated_model_name(model)
    if not model_name:
        raise ValueError("generated-video model is required")
    enhanced_suffix = "-enhanced"
    enhanced = model_name.casefold().endswith(enhanced_suffix)
    stored_model = model_name[: -len(enhanced_suffix)] if enhanced else model_name
    if not stored_model:
        raise ValueError("generated-video model is required")
    source_kind = "enhanced" if enhanced else "normal"
    paths = sample_artifact_paths(sample_root)
    legacy_root = paths["generated_enhanced_root" if enhanced else "generated_root"]
    return (
        generated_video_path(
            sample_root,
            stored_model,
            stage="processed",
            source_kind=source_kind,
        ),
        generated_video_path(
            sample_root,
            stored_model,
            stage="raw",
            source_kind=source_kind,
        ),
        legacy_root / f"{stored_model}.mp4",
    )


def import_generated_videos_to_bench(
    *,
    source_root: str | Path,
    bench_data_root: str | Path,
    model: str = "",
    move: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Import current provider outputs into ``artifacts/gen/raw``.

    Existing samples are required.  Copy remains the safe default; moving,
    overwriting, and dry-run behavior are explicit and delegated to the
    provider-neutral importer.
    """

    resolved_data_root = Path(bench_data_root).expanduser().resolve()

    def sample_exists(uid: str) -> bool:
        return bench_sample_root(resolved_data_root, uid).is_dir()

    def destination_for(uid: str, output_filename: str) -> Path:
        return (
            generated_video_stage_dir(
                bench_sample_root(resolved_data_root, uid),
                stage="raw",
            )
            / output_filename
        )

    result = import_generated_videos(
        source_root=source_root,
        destination_for=destination_for,
        sample_exists=sample_exists,
        model=model,
        move=move,
        overwrite=overwrite,
        dry_run=dry_run,
    )
    result["bench_data_root"] = resolved_data_root.as_posix()
    for item in result["skipped"]:
        if item.get("reason") != "missing destination sample":
            continue
        uid = str(item.get("uid", "") or "")
        item["reason"] = (
            f"missing bench sample: "
            f"{bench_sample_root(resolved_data_root, uid).as_posix()}"
        )
    return result


def _real_directory(
    value: str | os.PathLike[str],
    *,
    label: str,
) -> Path:
    raw = Path(value).expanduser()
    if not raw.is_absolute():
        raise ValueError(f"{label} must be absolute")
    current = Path(raw.anchor)
    for part in raw.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except (FileNotFoundError, NotADirectoryError) as error:
            raise FileNotFoundError(f"{label} not found: {raw}") from error
        if stat.S_ISLNK(mode):
            raise ValueError(f"{label} must not traverse symlinks")
    resolved = raw.resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"{label} must be a directory")
    return resolved


def normalize_generated_videos_in_bench(
    *,
    bench_data_root: str | Path,
    uids: Sequence[str] = (),
    models: Sequence[str] = (),
    source: str = "normal",
    mode: str = "resize",
    size: int = 512,
    fps: str = "",
    workers: int = 1,
    overwrite: bool = False,
    dry_run: bool = False,
    run: Callable[..., Any] | None = None,
) -> Dict[str, Any]:
    """Normalize current raw videos into processed videos for explicit UIDs.

    This is the bench-layout adapter around the provider-neutral one-video
    normalizer.  It never discovers a repository or registry and performs no
    write when ``dry_run`` is true.
    """

    data_root = _real_directory(
        bench_data_root,
        label="bench_data_root",
    )
    source_mode = str(source or "").strip().lower()
    if source_mode not in {"normal", "enhanced", "both"}:
        raise ValueError("source must be normal, enhanced, or both")
    if mode not in {"resize", "center_crop"}:
        raise ValueError("mode must be resize or center_crop")
    if int(size) <= 0:
        raise ValueError("size must be positive")
    if int(workers) <= 0:
        raise ValueError("workers must be positive")

    selected_uids: list[str] = []
    seen_uids: set[str] = set()
    raw_uids = list(uids)
    if not raw_uids:
        for path in sorted(data_root.iterdir(), key=lambda item: item.name):
            if path.is_symlink():
                raise ValueError(
                    f"bench_data_root contains a symlink entry: {path.name}"
                )
            if path.is_dir():
                raw_uids.append(path.name)
    for value in raw_uids:
        uid = str(value or "").strip()
        if not uid or uid in {".", ".."} or "/" in uid or "\\" in uid:
            raise ValueError("uid must be a safe non-empty identifier")
        if uid in seen_uids:
            continue
        sample = bench_sample_root(data_root, uid)
        _real_directory(sample, label=f"bench sample {uid}")
        seen_uids.add(uid)
        selected_uids.append(uid)

    requested_models: set[str] = set()
    for value in models:
        model = normalize_generated_model_name(value)
        if model:
            requested_models.add(model)
    source_kinds = ("normal", "enhanced") if source_mode == "both" else (source_mode,)
    jobs: list[dict[str, Any]] = []
    for uid in selected_uids:
        sample = bench_sample_root(data_root, uid)
        for source_kind in source_kinds:
            raw_dir = generated_video_stage_dir(
                sample,
                stage="raw",
                source_kind=source_kind,
            )
            if not raw_dir.exists():
                continue
            raw_dir = _real_directory(
                raw_dir,
                label=f"raw generated-video directory {uid}/{source_kind}",
            )
            processed_dir = generated_video_stage_dir(
                sample,
                stage="processed",
                source_kind=source_kind,
            )
            if processed_dir.exists():
                _real_directory(
                    processed_dir,
                    label=(f"processed generated-video directory {uid}/{source_kind}"),
                )
            for candidate in sorted(raw_dir.glob("*.mp4")):
                if candidate.is_symlink():
                    raise ValueError(
                        f"raw generated video must not be a symlink: {candidate}"
                    )
                if not candidate.is_file():
                    continue
                model = normalize_generated_model_name(candidate.name)
                if requested_models and model not in requested_models:
                    continue
                jobs.append(
                    {
                        "uid": uid,
                        "model": model,
                        "source_kind": source_kind,
                        "source": candidate.resolve(strict=True),
                        "destination": processed_dir / f"{model}.mp4",
                    }
                )

    def execute(job: dict[str, Any]) -> dict[str, Any]:
        options: dict[str, Any] = {}
        if run is not None:
            options["run"] = run
        record = normalize_generated_video(
            job["source"],
            job["destination"],
            mode=mode,
            size=int(size),
            fps=str(fps or ""),
            overwrite=bool(overwrite),
            dry_run=bool(dry_run),
            **options,
        )
        return {
            "uid": job["uid"],
            "model": job["model"],
            "source_kind": job["source_kind"],
            **record,
        }

    if int(workers) == 1:
        records = [execute(job) for job in jobs]
    else:
        with ThreadPoolExecutor(max_workers=int(workers)) as executor:
            records = list(executor.map(execute, jobs))
    counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown") or "unknown")
        counts[status] = counts.get(status, 0) + 1
    return {
        "status": "incomplete" if counts.get("error", 0) else "completed",
        "bench_data_root": data_root.as_posix(),
        "uids": selected_uids,
        "source": source_mode,
        "mode": mode,
        "size": int(size),
        "fps": str(fps or ""),
        "workers": int(workers),
        "overwrite": bool(overwrite),
        "dry_run": bool(dry_run),
        "job_count": len(jobs),
        "counts": counts,
        "records": records,
    }


__all__ = [
    "GENERATED_VIDEO_SOURCE_KINDS",
    "GENERATED_VIDEO_STAGES",
    "bench_sample_root",
    "generated_video_candidates",
    "generated_video_path",
    "generated_video_stage_dir",
    "import_generated_videos_to_bench",
    "normalize_generated_videos_in_bench",
    "normalize_generated_model_name",
]
