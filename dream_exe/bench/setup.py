"""Prepare and validate the configured Dream.exe workspace."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from .data.repository import BenchRepository
from .data.workspace import load_workspace


def _require_within(path: Path, root: Path, *, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} must be below configured root {root}: {path}") from error


def _prepare_directory(path: Path) -> bool:
    """Create one configured local root without following a final symlink."""

    if path.is_symlink():
        raise ValueError(f"configured writable root must not be a symlink: {path}")
    existed = path.is_dir()
    path.mkdir(parents=True, exist_ok=True)
    return not existed


def configure_workspace(
    *,
    workspace_path: str | Path,
    create_directories: bool = True,
) -> dict[str, Any]:
    """Load the path config and create only its empty local root directories."""

    workspace = load_workspace(workspace_path, require_existing_bench=False)
    dataset_ready = (
        (workspace.bench_root / "bench.json").is_file()
        and (workspace.published_results_root / "videos").is_dir()
    )

    for name, binding in workspace.bindings["sources"].items():
        _require_within(binding.path, workspace.external_root, label=f"source {name}")
        if binding.setup_path is None:
            raise ValueError(f"source {name} is missing its setup path")
        _require_within(
            binding.setup_path,
            workspace.external_root,
            label=f"source {name} setup",
        )
    for name, binding in workspace.bindings["checkpoints"].items():
        _require_within(
            binding.path,
            workspace.checkpoint_root,
            label=f"checkpoint {name}",
        )

    created = []
    if create_directories:
        for name, path in (
            ("external", workspace.external_root),
            ("checkpoints", workspace.checkpoint_root),
            ("outputs", workspace.outputs_root),
            ("work", workspace.work_root),
            ("archive", workspace.archive_root),
        ):
            if _prepare_directory(path):
                created.append(name)

    return {
        "format": "dream-exe.workspace-configuration",
        "status": "configured",
        "workspace": workspace.config_path.as_posix(),
        "bench": workspace.bench_root.as_posix(),
        "published_results": workspace.published_results_root.as_posix(),
        "outputs": workspace.outputs_root.as_posix(),
        "external": workspace.external_root.as_posix(),
        "checkpoints": workspace.checkpoint_root.as_posix(),
        "created_roots": created,
        "dataset_ready": dataset_ready,
        "write_mode": "create-directories" if create_directories else "check-only",
    }


def inspect_workspace(
    *,
    workspace_path: str | Path,
    uid: str,
    collection_id: str = "",
    verify_bindings: bool = True,
) -> dict[str, Any]:
    """Validate public bench inputs and, by default, every runtime binding."""

    workspace = load_workspace(workspace_path)
    repository = BenchRepository(workspace.bench_root)
    collection = repository.load_collection(collection_id)
    collection_id = str(collection["collection_id"])
    if uid not in {str(item["uid"]) for item in collection["cases"]}:
        raise ValueError(f"case {uid!r} is not a member of {collection_id!r}")
    case = repository.load_case(uid)
    environment = repository.load_environment(uid, verify_files=True)
    generation = repository.load_generation_input(uid, verify_frame=True)
    reference = repository.load_reference(uid, verify_files=True)
    binding_report = (
        repository.validate_workspace_bindings(workspace)
        if verify_bindings
        else {"status": "skipped", "bindings": []}
    )
    return {
        "format": "dream-exe.workspace-check",
        "status": "pass",
        "workspace": workspace.config_path.as_posix(),
        "collection": collection_id,
        "collection_cases": len(collection["cases"]),
        "case": case["uid"],
        "environment_files": sum(len(paths) for paths in environment.files.values()),
        "generation_first_frame": generation["first_frame"]["path"],
        "reference_video": reference.manifest["video"]["path"],
        "runtime_bindings": binding_report,
    }


__all__ = ["configure_workspace", "inspect_workspace"]
