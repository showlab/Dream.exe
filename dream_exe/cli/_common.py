"""Shared, behavior-neutral command-line helpers."""

from __future__ import annotations

import json
import os
import stat
import tempfile
from pathlib import Path
from typing import Any


def _json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return value.as_posix()
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)


def _print_json(payload: Any) -> None:
    print(
        json.dumps(
            payload,
            indent=2,
            default=_json_default,
        )
    )


def _strict_json_document(payload: Any) -> str:
    """Serialize one finite JSON document with the established CLI format."""

    return (
        json.dumps(
            payload,
            indent=2,
            default=_json_default,
            allow_nan=False,
        )
        + "\n"
    )


def _explicit_output_path(value: str | Path, *, flag: str) -> Path:
    """Validate one caller-owned absolute output file before expensive work."""

    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{flag} must be a non-empty absolute path")
    selected = Path(text).expanduser()
    if not selected.is_absolute():
        raise ValueError(f"{flag} must be an absolute path")
    destination = selected.absolute()
    parent = destination.parent
    if not parent.exists():
        raise FileNotFoundError(f"{flag} parent directory does not exist: {parent}")
    if parent.is_symlink() or not parent.is_dir():
        raise ValueError(f"{flag} parent must be an existing non-symlink directory")
    try:
        existing = destination.lstat()
    except FileNotFoundError:
        existing = None
    if existing is not None and not stat.S_ISREG(existing.st_mode):
        raise ValueError(f"{flag} destination must be a regular file")
    return destination


def _write_text_atomic(path: Path, content: str, *, label: str) -> None:
    """Atomically publish text without replacing an identical destination."""

    destination = _explicit_output_path(path, flag=label)
    encoded = content.encode("utf-8")
    if destination.exists():
        flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
        flags |= getattr(os, "O_NOFOLLOW", 0)
        descriptor = os.open(destination, flags)
        try:
            before = os.fstat(descriptor)
            if not stat.S_ISREG(before.st_mode):
                raise ValueError(f"{label} destination must be a regular file")
            existing = b""
            if before.st_size == len(encoded):
                chunks = []
                remaining = len(encoded)
                while remaining:
                    block = os.read(descriptor, min(1024 * 1024, remaining))
                    if not block:
                        break
                    chunks.append(block)
                    remaining -= len(block)
                existing = b"".join(chunks)
            after = os.fstat(descriptor)
        finally:
            os.close(descriptor)
        stable = (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) == (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        if stable and existing == encoded:
            return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, destination)
        directory_descriptor = os.open(
            destination.parent,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
        )
        try:
            os.fsync(directory_descriptor)
        finally:
            os.close(directory_descriptor)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise


def _optional(value: str) -> str | None:
    text = str(value or "").strip()
    return text or None


def _runtime_source_root(
    value: str | Path,
    *,
    flag: str,
    relative_to: str | Path | None = None,
) -> Path | None:
    """Resolve one optional code checkout without consulting process CWD."""

    selected = _optional(str(value or ""))
    if selected is None:
        return None
    root = Path(selected).expanduser()
    if not root.is_absolute():
        if relative_to is None:
            raise ValueError(
                f"{flag} must be an absolute path because this command has "
                "no repository/config base"
            )
        base = Path(relative_to).expanduser()
        if not base.is_absolute():
            raise ValueError(f"{flag} resolution base must be absolute")
        root = base / root
    return root.resolve(strict=False)


def _load_json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"{label} not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {source}")
    return payload


def _load_named_roots(
    path: str | Path,
    *,
    label: str = "named roots",
) -> dict[str, Path]:
    """Load an explicit semantic-root catalog without consulting process CWD."""

    source = Path(path).expanduser()
    if not source.is_absolute():
        raise ValueError(f"{label} JSON must be an absolute path")
    payload = _load_json_object(source, label=label)
    roots: dict[str, Path] = {}
    for raw_key, raw_value in payload.items():
        key = str(raw_key or "").strip()
        if not key or key in {".", ".."} or "/" in key or "\\" in key:
            raise ValueError(f"{label} keys must be safe non-empty identifiers")
        if not isinstance(raw_value, str) or not raw_value.strip():
            raise ValueError(f"{label} entry {key!r} must be an absolute path string")
        root = Path(raw_value).expanduser()
        if not root.is_absolute():
            raise ValueError(f"{label} entry {key!r} must be absolute")
        try:
            metadata = root.lstat()
        except OSError as error:
            raise FileNotFoundError(
                f"{label} entry {key!r} is unavailable: {root}"
            ) from error
        if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISDIR(metadata.st_mode):
            raise ValueError(
                f"{label} entry {key!r} must be an existing non-symlink directory"
            )
        roots[key] = root.resolve(strict=True)
    if not roots:
        raise ValueError(f"{label} must not be empty")
    return roots
