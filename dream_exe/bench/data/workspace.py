"""Workspace-local roots and external bindings for the canonical benchmark."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

from ..contracts.schemas import WORKSPACE_SCHEMA, load_and_validate


def _resolve_from(anchor: Path, value: str) -> Path:
    path = Path(value).expanduser()
    if not path.is_absolute():
        path = anchor / path
    return path.resolve(strict=False)


def _contains(parent: Path, child: Path) -> bool:
    try:
        child.relative_to(parent)
    except ValueError:
        return False
    return True


def _require_disjoint(roots: Mapping[str, Path]) -> None:
    values = list(roots.items())
    for index, (left_name, left) in enumerate(values):
        for right_name, right in values[index + 1 :]:
            if left == right or _contains(left, right) or _contains(right, left):
                raise ValueError(
                    "workspace roots must be pairwise disjoint: "
                    f"{left_name}={left} and {right_name}={right}"
                )


@dataclass(frozen=True)
class ExternalBinding:
    name: str
    kind: str
    path: Path
    setup_path: Path | None
    manifest: Path | None


@dataclass(frozen=True)
class Workspace:
    config_path: Path
    bench_root: Path
    published_results_root: Path
    outputs_root: Path
    work_root: Path
    archive_root: Path
    external_root: Path
    checkpoint_root: Path
    bindings: Mapping[str, Mapping[str, ExternalBinding]]

    def binding(self, kind: str, name: str) -> ExternalBinding:
        try:
            return self.bindings[kind][name]
        except KeyError as error:
            raise KeyError(f"workspace has no {kind} binding {name!r}") from error

    @property
    def materialized_root(self) -> Path:
        return self.work_root / "materialized"

    @property
    def transaction_root(self) -> Path:
        return self.work_root / "transactions"

    @property
    def run_root(self) -> Path:
        return self.work_root / "runs"

    @property
    def video_output_root(self) -> Path:
        return self.outputs_root / "videos"

    @property
    def published_video_root(self) -> Path:
        return self.published_results_root / "videos"

    @property
    def results_root(self) -> Path:
        return self.outputs_root / "experiments"


def load_workspace(path: str | Path, *, require_existing_bench: bool = True) -> Workspace:
    source = Path(path).expanduser().resolve()
    payload = load_and_validate(source, expected_schema=WORKSPACE_SCHEMA)
    anchor = source.parent
    roots = {
        name: _resolve_from(anchor, str(value))
        for name, value in payload["roots"].items()
    }
    _require_disjoint(roots)
    if require_existing_bench and not roots["bench"].is_dir():
        raise FileNotFoundError(f"benchmark root not found: {roots['bench']}")
    parsed_bindings: dict[str, dict[str, ExternalBinding]] = {}
    for kind, entries in payload["bindings"].items():
        parsed_bindings[kind] = {}
        for name, raw in entries.items():
            binding_path = _resolve_from(anchor, str(raw["path"]))
            manifest_value = raw["manifest"]
            setup_value = raw.get("setup_path")
            manifest = (
                None
                if manifest_value is None
                else _resolve_from(anchor, str(manifest_value))
            )
            parsed_bindings[kind][name] = ExternalBinding(
                name=name,
                kind=kind,
                path=binding_path,
                setup_path=(
                    None
                    if setup_value is None
                    else _resolve_from(anchor, str(setup_value))
                ),
                manifest=manifest,
            )
    return Workspace(
        config_path=source,
        bench_root=roots["bench"],
        published_results_root=roots["published_results"],
        outputs_root=roots["outputs"],
        work_root=roots["work"],
        archive_root=roots["archive"],
        external_root=roots["external"],
        checkpoint_root=roots["checkpoints"],
        bindings=parsed_bindings,
    )


def assert_same_device(source: str | Path, destination_parent: str | Path) -> int:
    """Return the shared device number or fail before an atomic rename."""

    source_path = Path(source).expanduser().resolve()
    destination = Path(destination_parent).expanduser().resolve()
    if not source_path.exists():
        raise FileNotFoundError(source_path)
    probe = destination
    while not probe.exists() and probe != probe.parent:
        probe = probe.parent
    if not probe.exists():
        raise FileNotFoundError(f"no existing destination ancestor: {destination}")
    source_device = os.stat(source_path, follow_symlinks=False).st_dev
    destination_device = os.stat(probe, follow_symlinks=False).st_dev
    if source_device != destination_device:
        raise OSError(
            "cross-device atomic rename is forbidden: "
            f"source dev={source_device}, destination dev={destination_device}"
        )
    return source_device


__all__ = [
    "ExternalBinding",
    "Workspace",
    "assert_same_device",
    "load_workspace",
]
