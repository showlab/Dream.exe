from __future__ import annotations

from pathlib import Path

import pytest

from dream_exe.bench.data.portable import (
    absolute_path_strings,
    rewrite_absolute_paths,
)
from dream_exe.bench.data.workspace import Workspace


def _workspace(tmp_path: Path) -> Workspace:
    return Workspace(
        config_path=tmp_path / "workspace.json",
        bench_root=tmp_path / "bench",
        published_results_root=tmp_path / "published",
        outputs_root=tmp_path / "outputs",
        work_root=tmp_path / "work",
        archive_root=tmp_path / "archive",
        external_root=tmp_path / "external",
        checkpoint_root=tmp_path / "checkpoints",
        bindings={},
    )


def test_portable_rewrite_handles_array_root_when_explicitly_allowed(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "source"
    target_root = tmp_path / "canonical"
    owner = tmp_path / "bundle" / "geometry.json"
    document = [
        {"path": (source_root / "points.npy").as_posix()},
        [(source_root / "mask.png").as_posix(), "finite-value"],
    ]

    rewritten = rewrite_absolute_paths(
        document,
        owner_path=owner,
        published_owner_path=owner,
        path_mappings=[(source_root, target_root)],
        workspace=_workspace(tmp_path),
        allow_non_object_root=True,
    )

    assert rewritten == [
        {"path": "../canonical/points.npy"},
        ["../canonical/mask.png", "finite-value"],
    ]
    assert list(absolute_path_strings(rewritten)) == []


def test_portable_rewrite_rejects_array_root_without_opt_in(tmp_path: Path) -> None:
    with pytest.raises(TypeError, match="portable JSON root must remain an object"):
        rewrite_absolute_paths(
            [],
            owner_path=tmp_path / "owner.json",
            published_owner_path=tmp_path / "owner.json",
            path_mappings=[],
            workspace=_workspace(tmp_path),
        )
