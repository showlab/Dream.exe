"""Portable frozen-scene path handling.

Canonical frozen XML files refer to the exact simulator assets vendored below
``bench/sources/simulator-assets`` with relative paths.  A run materializes a
work-local XML whose file attributes are absolute only inside ``work``; the
immutable benchmark never embeds a machine path.
"""

from __future__ import annotations

import gzip
import os
import re
from pathlib import Path


_FILE_ATTRIBUTE = re.compile(r"(?P<prefix>\bfile\s*=\s*[\"'])(?P<path>[^\"']+)(?P<suffix>[\"'])")


def _read_gzip_text(path: Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return stream.read()


def _write_gzip_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # A fixed gzip timestamp makes a canonical rewrite deterministic.
    with path.open("wb") as raw:
        with gzip.GzipFile(filename="", mode="wb", fileobj=raw, mtime=0) as stream:
            stream.write(text.encode("utf-8"))


def canonical_simulator_asset(
    raw_path: str,
    *,
    simulator_assets_root: Path,
) -> Path:
    """Map one historical RoboCasa/robosuite asset path into the bench pack."""

    text = str(raw_path)
    if "/robosuite/models/assets/" in text:
        family = "robosuite"
        relative = text.split("/robosuite/models/assets/", 1)[1]
    elif "/robocasa/models/assets/" in text:
        family = "robocasa"
        relative = text.split("/robocasa/models/assets/", 1)[1]
    else:
        raise ValueError(f"unknown frozen-scene asset path: {raw_path}")
    target = (
        simulator_assets_root
        / family
        / "models"
        / "assets"
        / Path(relative)
    ).resolve(strict=False)
    root = simulator_assets_root.resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError(f"frozen-scene asset escapes simulator pack: {raw_path}") from error
    return target


def canonicalize_frozen_scene_text(
    text: str,
    *,
    canonical_scene_path: Path,
    simulator_assets_root: Path,
) -> tuple[str, tuple[Path, ...]]:
    """Return portable XML and the unique packed files it references."""

    referenced: set[Path] = set()

    def replace(match: re.Match[str]) -> str:
        source = match.group("path")
        if not Path(source).is_absolute():
            target = (canonical_scene_path.parent / source).resolve(strict=False)
            root = simulator_assets_root.resolve(strict=False)
            try:
                target.relative_to(root)
            except ValueError as error:
                raise ValueError(
                    f"relative frozen-scene asset is outside simulator pack: {source}"
                ) from error
        else:
            target = canonical_simulator_asset(
                source,
                simulator_assets_root=simulator_assets_root,
            )
        referenced.add(target)
        relative = Path(
            os.path.relpath(target, start=canonical_scene_path.parent)
        ).as_posix()
        return f"{match.group('prefix')}{relative}{match.group('suffix')}"

    rewritten = _FILE_ATTRIBUTE.sub(replace, text)
    return rewritten, tuple(sorted(referenced))


def canonicalize_frozen_scene_file(
    source: Path,
    destination: Path,
    *,
    simulator_assets_root: Path,
) -> tuple[Path, ...]:
    rewritten, referenced = canonicalize_frozen_scene_text(
        _read_gzip_text(source),
        canonical_scene_path=destination,
        simulator_assets_root=simulator_assets_root,
    )
    _write_gzip_text(destination, rewritten)
    return referenced


def materialize_frozen_scene(
    source: Path,
    destination: Path,
    *,
    bench_root: Path,
) -> tuple[Path, ...]:
    """Resolve canonical relative asset references into one work-local XML."""

    text = _read_gzip_text(source)
    referenced: set[Path] = set()
    canonical_bench = bench_root.resolve(strict=True)

    def replace(match: re.Match[str]) -> str:
        raw = match.group("path")
        if Path(raw).is_absolute():
            raise ValueError(
                f"canonical frozen scene contains an absolute asset path: {source}: {raw}"
            )
        target = (source.parent / raw).resolve(strict=True)
        try:
            target.relative_to(canonical_bench)
        except ValueError as error:
            raise ValueError(
                f"canonical frozen-scene asset escapes bench: {source}: {raw}"
            ) from error
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"frozen-scene asset must be a regular file: {target}")
        referenced.add(target)
        return f"{match.group('prefix')}{target.as_posix()}{match.group('suffix')}"

    materialized = _FILE_ATTRIBUTE.sub(replace, text)
    _write_gzip_text(destination, materialized)
    return tuple(sorted(referenced))


def frozen_scene_references(source: Path, *, bench_root: Path) -> tuple[Path, ...]:
    """Validate and list every canonical relative frozen-scene asset."""

    text = _read_gzip_text(source)
    root = bench_root.resolve(strict=True)
    references: set[Path] = set()
    for match in _FILE_ATTRIBUTE.finditer(text):
        raw = match.group("path")
        if Path(raw).is_absolute():
            raise ValueError(
                f"canonical frozen scene contains an absolute asset path: {source}: {raw}"
            )
        target = (source.parent / raw).resolve(strict=True)
        try:
            target.relative_to(root)
        except ValueError as error:
            raise ValueError(
                f"canonical frozen-scene asset escapes bench: {source}: {raw}"
            ) from error
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"frozen-scene asset must be a regular file: {target}")
        references.add(target)
    return tuple(sorted(references))


__all__ = [
    "canonical_simulator_asset",
    "canonicalize_frozen_scene_file",
    "canonicalize_frozen_scene_text",
    "frozen_scene_references",
    "materialize_frozen_scene",
]
