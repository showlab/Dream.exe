"""Portable JSON path rewriting and auditing for published bench artifacts."""

from __future__ import annotations

import copy
import os
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from .workspace import Workspace


_WINDOWS_ABSOLUTE = re.compile(r"^[A-Za-z]:[\\/]")


def is_absolute_path_text(value: Any) -> bool:
    if not isinstance(value, str) or not value:
        return False
    return value.startswith(("/", "~/", "file://")) or bool(
        _WINDOWS_ABSOLUTE.match(value)
    )


def absolute_path_strings(value: Any, pointer: str = ""):
    """Yield every absolute filesystem string stored as a JSON value."""

    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            yield from absolute_path_strings(item, f"{pointer}/{key_text}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from absolute_path_strings(item, f"{pointer}/{index}")
    elif is_absolute_path_text(value):
        yield pointer or "/", str(value)


def _filesystem_path(value: str) -> Path:
    if value.startswith("file://"):
        value = value[len("file://") :]
    return Path(value).expanduser().resolve(strict=False)


def _relative_text(target: Path, owner: Path) -> str:
    return Path(os.path.relpath(target, start=owner.parent)).as_posix()


def _prefix_match(path: Path, prefix: Path) -> Path | None:
    try:
        return path.relative_to(prefix)
    except ValueError:
        return None


def _binding_roots(workspace: Workspace) -> list[tuple[Path, str]]:
    output: list[tuple[Path, str]] = []
    for kind, bindings in workspace.bindings.items():
        for name, binding in bindings.items():
            reference = f"workspace.bindings.{kind}.{name}"
            output.append((binding.path.resolve(strict=False), reference))
            if binding.manifest is not None:
                output.append(
                    (
                        binding.manifest.resolve(strict=False),
                        f"{reference}.manifest",
                    )
                )
    return sorted(output, key=lambda item: len(item[0].parts), reverse=True)


def rewrite_absolute_paths(
    document: Any,
    *,
    owner_path: Path,
    published_owner_path: Path,
    path_mappings: Sequence[tuple[Path, Path]],
    workspace: Workspace,
    package_root: Path | None = None,
    allow_non_object_root: bool = False,
) -> Any:
    """Return a deep portable copy of one JSON document.

    Paths to bench/result files become filesystem-relative strings from the
    published JSON file. External source/checkpoint paths become explicit
    workspace binding references whose ``path`` member is relative to that
    binding. Package-source paths receive the same treatment. Unknown absolute
    paths fail publication instead of leaking a machine path or becoming stale.
    """

    normalized_mappings = sorted(
        (
            (source.resolve(strict=False), target.resolve(strict=False))
            for source, target in path_mappings
        ),
        key=lambda item: len(item[0].parts),
        reverse=True,
    )
    bindings = _binding_roots(workspace)
    package = None if package_root is None else package_root.resolve(strict=False)

    def rewrite(value: Any, pointer: str) -> Any:
        if isinstance(value, Mapping):
            return {
                str(key): rewrite(item, f"{pointer}/{key}")
                for key, item in value.items()
            }
        if isinstance(value, list):
            return [rewrite(item, f"{pointer}/{index}") for index, item in enumerate(value)]
        if not is_absolute_path_text(value):
            return copy.deepcopy(value)

        source = _filesystem_path(str(value))
        for old_root, new_root in normalized_mappings:
            relative = _prefix_match(source, old_root)
            if relative is not None:
                return _relative_text(new_root / relative, published_owner_path)
        for binding_root, binding_reference in bindings:
            relative = _prefix_match(source, binding_root)
            if relative is not None:
                return {
                    "root": binding_reference,
                    "path": relative.as_posix() if relative.parts else ".",
                }
        if package is not None:
            relative = _prefix_match(source, package)
            if relative is not None:
                return {
                    "root": "package.dream-exe",
                    "path": relative.as_posix() if relative.parts else ".",
                }
        raise ValueError(
            "published JSON contains an unmapped absolute path at "
            f"{pointer or '/'} in {owner_path}: {value}"
        )

    rewritten = rewrite(document, "")
    if not allow_non_object_root and not isinstance(rewritten, dict):  # pragma: no cover - input contract
        raise TypeError("portable JSON root must remain an object")
    leftovers = list(absolute_path_strings(rewritten))
    if leftovers:  # pragma: no cover - recursive invariant
        raise RuntimeError(f"portable rewrite left absolute paths: {leftovers[:3]}")
    return rewritten


__all__ = [
    "absolute_path_strings",
    "is_absolute_path_text",
    "rewrite_absolute_paths",
]
