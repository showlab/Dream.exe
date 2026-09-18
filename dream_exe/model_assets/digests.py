"""Bounded process-local digest proofs for immutable external model assets.

The cache stores observed bytes, never a caller's verification result.  Every
consumer must still compare the returned digest with its own expected value.
Mutable benchmark artifacts deliberately do not use this module.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import os
from pathlib import Path
import stat
import threading


_MAX_STABLE_FILE_DIGESTS = 512
_StableFileKey = tuple[str, int, int, int, int, int, int, int]
_STABLE_FILE_DIGESTS: OrderedDict[_StableFileKey, str] = OrderedDict()
_STABLE_FILE_DIGESTS_LOCK = threading.RLock()


def _stable_file_key(path: Path, metadata: os.stat_result) -> _StableFileKey:
    return (
        path.as_posix(),
        int(metadata.st_dev),
        int(metadata.st_ino),
        int(metadata.st_mode),
        int(metadata.st_nlink),
        int(metadata.st_size),
        int(metadata.st_mtime_ns),
        int(metadata.st_ctime_ns),
    )


def _require_regular_file(
    path: Path,
    metadata: os.stat_result,
    *,
    label: str,
) -> None:
    if not stat.S_ISREG(metadata.st_mode):
        raise FileNotFoundError(f"{label} not found: {path.as_posix()}")


def _hash_stable_file_once(
    path: Path,
    *,
    expected_key: _StableFileKey,
    label: str,
) -> str:
    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    if not hasattr(os, "O_NOFOLLOW"):
        raise RuntimeError("stable asset hashing requires O_NOFOLLOW support")
    flags |= os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{label} changed before hashing: {path}") from error
    try:
        opened = os.fstat(descriptor)
        _require_regular_file(path, opened, label=label)
        if _stable_file_key(path, opened) != expected_key:
            raise RuntimeError(f"{label} changed before hashing: {path}")
        digest = hashlib.sha256()
        observed_size = 0
        while True:
            chunk = os.read(descriptor, 1024 * 1024)
            if not chunk:
                break
            observed_size += len(chunk)
            if observed_size > int(opened.st_size):
                raise RuntimeError(f"{label} grew while hashing: {path}")
            digest.update(chunk)
        if observed_size != int(opened.st_size):
            raise RuntimeError(f"{label} size changed while hashing: {path}")
        after = os.fstat(descriptor)
        if _stable_file_key(path, after) != expected_key:
            raise RuntimeError(f"{label} changed while hashing: {path}")
    finally:
        os.close(descriptor)
    final = os.stat(path, follow_symlinks=False)
    if _stable_file_key(path, final) != expected_key:
        raise RuntimeError(f"{label} changed after hashing: {path}")
    return digest.hexdigest()


def _confirm_stable_file_key(
    path: Path,
    *,
    expected_key: _StableFileKey,
    label: str,
) -> None:
    """Revalidate a cached proof without rereading immutable asset bytes."""

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0) | os.O_NOFOLLOW
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise RuntimeError(f"{label} changed before verification: {path}") from error
    try:
        opened = os.fstat(descriptor)
        _require_regular_file(path, opened, label=label)
        if _stable_file_key(path, opened) != expected_key:
            raise RuntimeError(f"{label} changed before verification: {path}")
    finally:
        os.close(descriptor)
    final = os.stat(path, follow_symlinks=False)
    if _stable_file_key(path, final) != expected_key:
        raise RuntimeError(f"{label} changed during verification: {path}")


def sha256_stable_file(path: str | Path, *, label: str) -> str:
    """Return a stable SHA-256 proof for one explicit immutable asset file."""

    try:
        source = Path(path).expanduser().resolve(strict=True)
    except OSError as error:
        raise FileNotFoundError(f"{label} not found: {path}") from error
    metadata = os.stat(source, follow_symlinks=False)
    _require_regular_file(source, metadata, label=label)
    key = _stable_file_key(source, metadata)
    with _STABLE_FILE_DIGESTS_LOCK:
        cached = _STABLE_FILE_DIGESTS.get(key)
    if cached is not None:
        _confirm_stable_file_key(
            source,
            expected_key=key,
            label=label,
        )
        with _STABLE_FILE_DIGESTS_LOCK:
            if key in _STABLE_FILE_DIGESTS:
                _STABLE_FILE_DIGESTS.move_to_end(key)
        return cached

    digest = _hash_stable_file_once(
        source,
        expected_key=key,
        label=label,
    )
    with _STABLE_FILE_DIGESTS_LOCK:
        for stale_key in tuple(_STABLE_FILE_DIGESTS):
            if stale_key[0] == key[0]:
                del _STABLE_FILE_DIGESTS[stale_key]
        _STABLE_FILE_DIGESTS[key] = digest
        while len(_STABLE_FILE_DIGESTS) > _MAX_STABLE_FILE_DIGESTS:
            _STABLE_FILE_DIGESTS.popitem(last=False)
    return digest


def reset_stable_file_digest_cache() -> None:
    """Discard all process-local byte proofs (primarily for test isolation)."""

    with _STABLE_FILE_DIGESTS_LOCK:
        _STABLE_FILE_DIGESTS.clear()


__all__ = [
    "reset_stable_file_digest_cache",
    "sha256_stable_file",
]
