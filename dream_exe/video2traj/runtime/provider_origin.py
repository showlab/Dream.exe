"""Fail-closed origin checks for explicitly selected provider checkouts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from pathlib import Path
import sys
from types import ModuleType


def prepend_source_roots(
    source_roots: Sequence[str | Path],
) -> tuple[str, ...]:
    """Place explicit roots first while retaining all unrelated import paths."""

    resolved_roots = tuple(Path(root).expanduser().resolve() for root in source_roots)
    for root in reversed(resolved_roots):
        retained = []
        for entry in sys.path:
            try:
                candidate = Path(entry or ".").expanduser().resolve()
            except (OSError, RuntimeError):
                retained.append(entry)
                continue
            if candidate != root:
                retained.append(entry)
        sys.path[:] = [root.as_posix(), *retained]
    return tuple(root.as_posix() for root in resolved_roots)


def require_modules_under_roots(
    modules: Iterable[ModuleType],
    *,
    source_roots: Sequence[str | Path],
    provider: str,
) -> tuple[str, ...]:
    """Require every imported provider module to originate below one root."""

    roots = tuple(Path(root).expanduser().resolve() for root in source_roots)
    if not roots:
        return ()

    origins = []
    for module in modules:
        module_name = str(getattr(module, "__name__", "") or "<unknown>")
        spec = getattr(module, "__spec__", None)
        spec_origin = str(getattr(spec, "origin", "") or "").strip()
        raw_origin = (
            spec_origin
            if spec_origin not in {"", "built-in", "frozen"}
            else str(getattr(module, "__file__", "") or "").strip()
        )
        if not raw_origin:
            raise RuntimeError(
                f"{provider} module {module_name!r} has no filesystem origin"
            )
        origin = Path(raw_origin).expanduser().resolve()
        if not origin.is_file():
            raise RuntimeError(
                f"{provider} module {module_name!r} origin is not a file: "
                f"{origin.as_posix()}"
            )
        if not any(_is_within(origin, root) for root in roots):
            raise RuntimeError(
                f"{provider} module {module_name!r} was imported from "
                f"{origin.as_posix()}, outside configured source roots "
                f"{[root.as_posix() for root in roots]!r}"
            )
        origins.append(origin.as_posix())
    return tuple(origins)


def _is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


__all__ = ["prepend_source_roots", "require_modules_under_roots"]
