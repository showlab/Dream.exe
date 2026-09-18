"""Shared, dependency-light helpers for region model backends."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


RuntimeLoader = Callable[[], Mapping[str, Any]]


def portable_path(value: str | Path) -> str:
    """Return a normalized absolute path without requiring it to exist."""

    text = str(value or "").strip()
    if not text:
        return ""
    return Path(text).expanduser().resolve(strict=False).as_posix()


def require_runtime_keys(
    runtime: Mapping[str, Any],
    *,
    backend: str,
    keys: tuple[str, ...],
) -> dict[str, Any]:
    """Validate a lazily supplied runtime bundle."""

    payload = dict(runtime)
    missing = [key for key in keys if key not in payload]
    if missing:
        raise RuntimeError(
            f"{backend} runtime loader is missing: " + ", ".join(missing)
        )
    return payload


__all__ = ["RuntimeLoader"]
