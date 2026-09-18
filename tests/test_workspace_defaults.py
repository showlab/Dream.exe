from __future__ import annotations

from pathlib import Path

from dream_exe.bench.data.workspace import load_workspace


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WORKSPACE = REPOSITORY_ROOT / "configs/workspace.json"
QUICKSTART_WORKSPACE = REPOSITORY_ROOT / "examples/quickstart/workspace.json"


def test_default_workspace_keeps_all_roots_inside_repository() -> None:
    workspace = load_workspace(DEFAULT_WORKSPACE, require_existing_bench=False)

    assert workspace.bench_root == REPOSITORY_ROOT / "data/bench"
    assert workspace.published_results_root == REPOSITORY_ROOT / "data/results"
    for root in (
        workspace.bench_root,
        workspace.published_results_root,
        workspace.outputs_root,
        workspace.work_root,
        workspace.archive_root,
        workspace.external_root,
        workspace.checkpoint_root,
    ):
        assert root.is_relative_to(REPOSITORY_ROOT)


def test_quickstart_uses_its_embedded_data() -> None:
    workspace = load_workspace(QUICKSTART_WORKSPACE)

    assert workspace.bench_root == REPOSITORY_ROOT / "examples/quickstart/data/bench"
    assert workspace.published_results_root == (
        REPOSITORY_ROOT / "examples/quickstart/data/results"
    )
