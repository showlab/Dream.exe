"""Atomic publication primitives for videos and experiment results."""

from __future__ import annotations

import hashlib
import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any, Mapping

from ..data.portable import absolute_path_strings
from ..contracts.schemas import (
    RESOLVED_CONFIG_SCHEMA,
    RESULT_REQUEST_SCHEMA,
    RESULT_SCHEMA,
    RUN_SCHEMA,
    VIDEO_OUTPUT_SCHEMA,
    canonical_json_bytes,
    canonical_sha256,
    input_identity_key,
    load_and_validate,
    validate_document,
)
from ..data.workspace import Workspace, assert_same_device


def sha256_file(path: str | Path, *, chunk_size: int = 8 * 1024 * 1024) -> str:
    source = Path(path)
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        for chunk in iter(lambda: stream.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def write_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
    *,
    exclusive: bool = False,
) -> Path:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and destination.exists():
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(canonical_json_bytes(payload))
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, destination)
            temporary.unlink()
        else:
            os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _contained(path: Path, root: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    try:
        resolved.relative_to(root.expanduser().resolve())
    except ValueError as error:
        raise ValueError(f"{label} must be contained by {root}: {resolved}") from error
    return resolved


def _make_read_only(root: Path) -> None:
    for path in sorted(root.rglob("*"), key=lambda item: len(item.parts), reverse=True):
        if path.is_symlink():
            continue
        mode = stat.S_IMODE(path.stat().st_mode)
        path.chmod(mode & ~0o222)
    root.chmod(stat.S_IMODE(root.stat().st_mode) & ~0o222)


def _verify_source(
    source: Path,
    record: Mapping[str, Any],
    *,
    workspace: Workspace,
    label: str,
) -> Path:
    resolved = _contained(source, workspace.work_root, label)
    if resolved.is_symlink() or not resolved.is_file():
        raise ValueError(f"{label} must be a regular non-symlink file: {resolved}")
    if resolved.stat().st_size != record["size"]:
        raise ValueError(f"{label} size mismatch")
    if sha256_file(resolved) != record["sha256"]:
        raise ValueError(f"{label} digest mismatch")
    return resolved


def register_video_output(
    *,
    workspace: Workspace,
    manifest: Mapping[str, Any],
    source_paths: Mapping[str, str | Path],
) -> dict[str, Any]:
    """Move one reviewed candidate video from work into ``outputs/videos``.

    ``video.mp4`` is the exact producer output.  The sibling
    ``preprocessed.mp4`` is the validated derivative consumed by video2traj.
    Legacy manifests may still omit it. Prompt variants are grouped below
    their producing model; no opaque video ID participates in the canonical
    path.
    """

    document = validate_document(manifest, expected_schema=VIDEO_OUTPUT_SCHEMA)
    leaks = list(absolute_path_strings(document))
    if leaks:
        raise ValueError(
            f"video output manifest contains absolute paths: {leaks[:3]}"
        )
    expected_names = {"video"}
    if document["preprocessed"] is not None:
        expected_names.add("preprocessed")
    if set(source_paths) != expected_names:
        raise ValueError(
            f"source_paths must contain exactly {sorted(expected_names)}"
        )
    destination = (
        workspace.outputs_root
        / "videos"
        / document["uid"]
        / document["model_id"]
        / document["prompt_variant"]
    )
    if destination.exists():
        raise FileExistsError(f"video output already exists: {destination}")

    verified = {
        "video": _verify_source(
            Path(source_paths["video"]),
            document["video"],
            workspace=workspace,
            label="generated video",
        )
    }
    if document["preprocessed"] is not None:
        verified["preprocessed"] = _verify_source(
            Path(source_paths["preprocessed"]),
            document["preprocessed"],
            workspace=workspace,
            label="video2traj input video",
        )
    if len({path for path in verified.values()}) != len(verified):
        raise ValueError(
            "one physical file cannot be moved twice; use preprocessed=null "
            "when video.mp4 is already pipeline-compatible"
        )

    promotion_id = "-".join(
        (
            "register-video",
            document["uid"],
            document["model_id"],
            document["prompt_variant"],
        )
    )
    staging = workspace.transaction_root / f".{promotion_id}.staging"
    journal_path = workspace.transaction_root / f"{promotion_id}.journal.jsonl"
    if staging.exists() or journal_path.exists():
        raise FileExistsError(f"video registration state already exists: {promotion_id}")
    staging.mkdir(parents=True)
    moved: list[tuple[Path, Path]] = []

    def journal(payload: Mapping[str, Any]) -> None:
        with journal_path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(payload, sort_keys=True, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        _fsync_directory(journal_path.parent)

    try:
        for name, source in verified.items():
            record = document[name]
            target = staging / str(record["path"])
            target.parent.mkdir(parents=True, exist_ok=True)
            assert_same_device(source, target.parent)
            journal(
                {
                    "event": "rename_planned",
                    "source": source.as_posix(),
                    "destination": target.as_posix(),
                }
            )
            os.rename(source, target)
            moved.append((source, target))
            journal(
                {
                    "event": "rename_completed",
                    "source": source.as_posix(),
                    "destination": target.as_posix(),
                }
            )
        write_json_atomic(staging / "video.json", document, exclusive=True)
        destination.parent.mkdir(parents=True, exist_ok=True)
        assert_same_device(staging, destination.parent)
        journal(
            {
                "event": "publish_planned",
                "source": staging.as_posix(),
                "destination": destination.as_posix(),
            }
        )
        os.rename(staging, destination)
        _fsync_directory(destination.parent)
        journal(
            {
                "event": "publish_completed",
                "source": staging.as_posix(),
                "destination": destination.as_posix(),
            }
        )
        _make_read_only(destination)
    except Exception:
        if destination.exists() and not staging.exists():
            destination.chmod(stat.S_IMODE(destination.stat().st_mode) | stat.S_IWUSR)
            os.rename(destination, staging)
        for source, target in reversed(moved):
            if target.exists() and not source.exists():
                source.parent.mkdir(parents=True, exist_ok=True)
                os.rename(target, source)
        raise
    return {
        "status": "registered",
        "uid": document["uid"],
        "model_id": document["model_id"],
        "prompt_variant": document["prompt_variant"],
        "destination": destination.as_posix(),
        "manifest_sha256": canonical_sha256(document),
        "journal": journal_path.as_posix(),
    }


def result_relative_path(result: Mapping[str, Any]) -> Path:
    input_document = result["input"]
    if input_document["kind"] == "reference":
        suffix = Path("reference", str(input_document["reference_id"]))
    else:
        suffix = Path(
            str(input_document["model_id"]),
            str(input_document["prompt_variant"]),
        )
    return Path(str(result["uid"])) / suffix


def run_relative_path(run: Mapping[str, Any]) -> Path:
    """Return the human-readable location of one published run descriptor.

    A run with one generated input is grouped by model and prompt variant, just
    like its case results.  A reference run is grouped below ``reference``.
    Multi-input runs have no single model owner, so they retain their run ID
    below an explicit ``multi-input`` branch.
    """

    document = validate_document(run, expected_schema=RUN_SCHEMA)
    inputs = document["inputs"]
    if len(inputs) != 1:
        return Path("multi-input", str(document["destination"]["run_id"]))
    input_document = inputs[0]
    if input_document["kind"] == "reference":
        return Path("reference", str(input_document["reference_id"]))
    return Path(
        str(input_document["model_id"]),
        str(input_document["prompt_variant"]),
    )


def discover_run_roots(outputs_root: str | Path) -> dict[str, Path]:
    """Index canonical run descriptors by their stable internal run ID."""

    root = Path(outputs_root).expanduser().resolve()
    runs_root = root / "runs"
    if not runs_root.is_dir():
        return {}
    discovered: dict[str, Path] = {}
    for manifest in sorted(runs_root.rglob("run.json")):
        if manifest.is_symlink() or not manifest.is_file():
            raise ValueError(f"run manifest must be a regular file: {manifest}")
        document = load_and_validate(manifest, expected_schema=RUN_SCHEMA)
        expected = (runs_root / run_relative_path(document)).resolve(strict=False)
        if manifest.parent.resolve() != expected:
            raise ValueError(
                "run manifest is not stored under its canonical model/variant "
                f"path: {manifest}"
            )
        run_id = str(document["destination"]["run_id"])
        if run_id in discovered:
            raise ValueError(f"duplicate published run ID: {run_id}")
        discovered[run_id] = manifest.parent
    return discovered


def finalize_result(
    *,
    workspace: Workspace,
    work_bundle: str | Path,
    result: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate one complete work bundle and atomically publish it."""

    document = validate_document(result, expected_schema=RESULT_SCHEMA)
    source = _contained(Path(work_bundle), workspace.work_root, "work result bundle")
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"work result bundle must be a non-symlink directory: {source}")
    destination = (
        workspace.outputs_root
        / "experiments"
        / result_relative_path(document)
    )
    if destination.exists():
        raise FileExistsError(f"result destination already exists: {destination}")

    resolved_path = source / "resolved_config.json"
    if document["resolved_config_sha256"] is None:
        if resolved_path.exists():
            raise ValueError("result has an unbound resolved_config.json")
    else:
        if resolved_path.is_symlink() or not resolved_path.is_file():
            raise FileNotFoundError(f"resolved_config.json missing: {source}")
        resolved = load_and_validate(
            resolved_path,
            expected_schema=RESOLVED_CONFIG_SCHEMA,
        )
        if canonical_sha256(resolved) != document["resolved_config_sha256"]:
            raise ValueError("result resolved_config_sha256 mismatch")
        if input_identity_key(resolved["input"]) != input_identity_key(document["input"]):
            raise ValueError("resolved config input identity conflicts with result")

    request_path = source / "request.json"
    if request_path.is_symlink() or not request_path.is_file():
        raise FileNotFoundError(f"request.json missing: {source}")
    request = load_and_validate(request_path, expected_schema=RESULT_REQUEST_SCHEMA)
    if canonical_sha256(request) != document["request_sha256"]:
        raise ValueError("result request_sha256 mismatch")
    if request["resolved_config_sha256"] != document["resolved_config_sha256"]:
        raise ValueError("request and result resolved config identities conflict")
    if input_identity_key(request["input"]) != input_identity_key(document["input"]):
        raise ValueError("request input identity conflicts with result")
    if request["input"]["video_sha256"] != document["input"]["video_sha256"]:
        raise ValueError("request video digest conflicts with result")

    for artifact in document["artifacts"]:
        path = _contained(source / artifact["path"], source, "result artifact")
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"result artifact must be a regular file: {path}")
        if path.stat().st_size != artifact["size"] or sha256_file(path) != artifact["sha256"]:
            raise ValueError(f"result artifact integrity mismatch: {artifact['path']}")

    write_json_atomic(source / "result.json", document, exclusive=True)
    destination.parent.mkdir(parents=True, exist_ok=True)
    assert_same_device(source, destination.parent)
    os.rename(source, destination)
    _fsync_directory(destination.parent)
    _make_read_only(destination)
    return {
        "status": "published",
        "destination": destination.as_posix(),
        "result": document,
    }


__all__ = [
    "discover_run_roots",
    "finalize_result",
    "register_video_output",
    "result_relative_path",
    "run_relative_path",
    "sha256_file",
    "write_json_atomic",
]
