"""Convenient, provenance-aware import of externally generated videos."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any

from ..outputs.lifecycle import register_video_output, sha256_file
from ..data.repository import BenchRepository
from ..contracts.schemas import (
    VIDEO_OUTPUT_SCHEMA,
    canonical_sha256,
    validate_document,
)
from ..data.workspace import Workspace, assert_same_device
from .preprocess import (
    inspect_video2traj_input,
    preprocess_video_for_video2traj,
)


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")


def _safe_id(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if not _SAFE_ID.fullmatch(clean):
        raise ValueError(f"{label} must be a safe identifier")
    return clean


def _regular_mp4(path: str | Path, *, label: str) -> Path:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {source}")
    if source.suffix.lower() != ".mp4":
        raise ValueError(f"{label} must be an MP4 file: {source}")
    return source


def _record(path: Path, target: str) -> dict[str, Any]:
    return {
        "path": target,
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def _prompt_digest(
    generation: dict[str, Any],
    prompt_variant: str,
    custom_prompt: str,
) -> str | None:
    if prompt_variant == "custom":
        prompt = custom_prompt.strip()
        return None if not prompt else canonical_sha256({"prompt": prompt})
    prompt = str(generation["prompts"][prompt_variant])
    return canonical_sha256({"prompt": prompt})


def import_external_video(
    *,
    workspace: Workspace,
    uid: str,
    model_id: str,
    prompt_variant: str,
    video_path: str | Path,
    preprocessed_path: str | Path | None = None,
    transfer: str = "copy",
    producer_kind: str = "generator",
    producer_name: str = "",
    producer_revision: str = "",
    seed: int | None = None,
    custom_prompt: str = "",
) -> dict[str, Any]:
    """Register a user-supplied candidate and return a ready run input.

    Copy is the safe default and preserves the caller's video.  ``transfer``
    may be ``move`` to avoid a second full-size copy; that choice removes the
    original only after it has been staged on the same filesystem.  There is
    deliberately no cross-device copy fallback for move mode.
    """

    if transfer not in {"copy", "move"}:
        raise ValueError("transfer must be 'copy' or 'move'")
    uid = _safe_id(uid, "case")
    model_id = _safe_id(model_id, "model")
    repository = BenchRepository(workspace.bench_root)
    repository.load_case(uid)
    generation = repository.load_generation_input(uid)
    video = _regular_mp4(video_path, label="generated video")
    preprocessed = (
        None
        if preprocessed_path is None
        else _regular_mp4(
            preprocessed_path,
            label="preprocessed video2traj input",
        )
    )
    if preprocessed is not None and preprocessed == video:
        raise ValueError(
            "preprocessed_path must be a distinct derived MP4; normally omit it "
            "and let Dream.exe preprocess video_path automatically"
        )

    destination = (
        workspace.outputs_root
        / "videos"
        / uid
        / model_id
        / prompt_variant
    )
    if destination.exists():
        raise FileExistsError(f"video output already exists: {destination}")
    staging = (
        workspace.work_root
        / "video-imports"
        / uid
        / model_id
        / prompt_variant
    )
    if staging.exists():
        raise FileExistsError(f"video import staging already exists: {staging}")
    staging.mkdir(parents=True)
    staged = {"video": staging / "video.mp4"}
    staged["preprocessed"] = staging / "preprocessed.mp4"
    originals = {"video": video}
    if preprocessed is not None:
        originals["preprocessed"] = preprocessed
    transferred: list[str] = []

    try:
        for role, original in originals.items():
            target = staged[role]
            if transfer == "move":
                assert_same_device(original, target.parent)
                os.rename(original, target)
            else:
                shutil.copy2(original, target)
            transferred.append(role)

        if preprocessed is None:
            preprocessing = preprocess_video_for_video2traj(
                staged["video"],
                staged["preprocessed"],
                output_contract=generation["output_contract"],
            )
            preprocessing_kind = "automatic"
            preprocessed_filename = None
        else:
            preprocessing = inspect_video2traj_input(
                staged["preprocessed"],
                output_contract=generation["output_contract"],
            )
            preprocessing_kind = "provided"
            preprocessed_filename = preprocessed.name

        manifest = validate_document(
            {
                "format": VIDEO_OUTPUT_SCHEMA,
                "uid": uid,
                "model_id": model_id,
                "prompt_variant": prompt_variant,
                "producer": {
                    "kind": producer_kind,
                    "name": producer_name.strip() or model_id,
                    "revision": producer_revision,
                    "seed": seed,
                },
                "generation_input_sha256": canonical_sha256(generation),
                "prompt_sha256": _prompt_digest(
                    generation,
                    prompt_variant,
                    custom_prompt,
                ),
                "video": _record(staged["video"], "video.mp4"),
                "preprocessed": _record(
                    staged["preprocessed"],
                    "preprocessed.mp4",
                ),
                "rights": {
                    "status": "unknown",
                    "license": "",
                    "redistributable": None,
                },
                "provenance_status": (
                    "unknown" if producer_kind == "unknown" else "complete"
                ),
                "origin": {
                    "kind": "external_video_import",
                    "video_filename": video.name,
                    "preprocessed_filename": preprocessed_filename,
                    "preprocessing_kind": preprocessing_kind,
                    "preprocessing": preprocessing["spec"],
                    "transfer": transfer,
                },
            },
            expected_schema=VIDEO_OUTPUT_SCHEMA,
        )

        result = register_video_output(
            workspace=workspace,
            manifest=manifest,
            source_paths=staged,
        )
    except Exception:
        if transfer == "move":
            for role in reversed(transferred):
                target = staged[role]
                original = originals[role]
                if target.exists() and not original.exists():
                    original.parent.mkdir(parents=True, exist_ok=True)
                    os.rename(target, original)
        raise

    for parent in (staging, staging.parent, staging.parent.parent):
        try:
            parent.rmdir()
        except OSError:
            break
    return {
        **result,
        "run_input": {
            "kind": "generated",
            "model_id": model_id,
            "prompt_variant": prompt_variant,
            "reference_id": None,
        },
        "source_preserved": transfer == "copy",
    }


__all__ = ["import_external_video"]
