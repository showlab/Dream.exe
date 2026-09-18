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


def test_quickstart_uses_its_embedded_data_without_configure_step() -> None:
    workspace = load_workspace(QUICKSTART_WORKSPACE)

    assert workspace.bench_root == REPOSITORY_ROOT / "examples/quickstart/data/bench"
    forbidden_command = (
        "dream-exe configure --workspace examples/quickstart/workspace.json"
    )
    assert forbidden_command not in (REPOSITORY_ROOT / "README.md").read_text(
        encoding="utf-8"
    )
    assert forbidden_command not in (
        REPOSITORY_ROOT / "examples/quickstart/README.md"
    ).read_text(encoding="utf-8")
    readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    assert "### 2. Check the bundled case" not in readme
    assert "### 2. Run the bundled case" in readme


def test_root_readme_owns_the_one_case_entry_command() -> None:
    entry = (
        "--workspace examples/quickstart/workspace.json \\\n"
        "  --spec examples/quickstart/run.json"
    )
    root_readme = (REPOSITORY_ROOT / "README.md").read_text(encoding="utf-8")
    assert root_readme.count(entry) == 1

    for relative in (
        "examples/quickstart/README.md",
        "docs/BENCHMARK.md",
    ):
        text = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        assert entry not in text


def test_public_docs_describe_repository_local_benchmark_data() -> None:
    for relative in (
        "README.md",
        "INSTALL.md",
        "configs/README.md",
        "docs/BENCHMARK.md",
        "docs/CONFIGURATION.md",
    ):
        text = (REPOSITORY_ROOT / relative).read_text(encoding="utf-8")
        assert "../data/dream-exe" not in text, relative

    benchmark = (REPOSITORY_ROOT / "docs/BENCHMARK.md").read_text(
        encoding="utf-8"
    )
    assert "--local-dir data" in benchmark
    assert "`roots.bench`" in benchmark
    assert "`roots.published_results`" in benchmark
