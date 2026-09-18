"""Canonical benchmark run orchestration."""

from __future__ import annotations

import copy
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...pipeline.records.layout import formal_artifact_paths
from ...models.runtime import public_runtime_config
from ...pipeline.runner.single_case import run_single_uid_workflow
from ...pipeline.runner.workflow import run_benchmark_workflow
from ...pipeline.planning.videos import generated_video_candidates
from ..contracts.action import write_motion_plan_bundle_from_json
from ..contracts.config import (
    compile_case_route_config,
    compile_repository_config,
    config_identity,
)
from ..contracts.schemas import (
    PIPELINE_STAGES,
    RESOLVED_CONFIG_SCHEMA,
    RESULT_REQUEST_SCHEMA,
    RESULT_SCHEMA,
    RUN_SUMMARY_SCHEMA,
    RUN_SCHEMA,
    canonical_sha256,
    input_identity_key,
    load_and_validate,
    validate_document,
)
from ..data.portable import is_absolute_path_text, rewrite_absolute_paths
from ..data.repository import (
    BenchRepository,
    OutputRepository,
    ReferenceSelection,
)
from ..data.workspace import Workspace, load_workspace
from ..outputs.lifecycle import (
    finalize_result,
    run_relative_path,
    sha256_file,
    write_json_atomic,
)
from .init import (
    _copy_file,
    _copy_or_verify_file,
    _file_record,
    _hash_tree,
    _json_object,
    _tree_identity,
)
from .materialization import (
    _fixed_input_files,
    _materialize_run_sample,
    _select_uids,
    _verify_materialized_run_sample,
)


_TRAJECTORY_EXECUTABILITY_FILENAME = "trajectory_executability.json"
_TASK_SUCCESS_FILENAME = "task_success.json"

_IMPLEMENTATION_SOURCE_SUFFIXES = frozenset({".py", ".so"})
_IMPLEMENTATION_IGNORED_DIRECTORIES = frozenset(
    {".git", "__pycache__", "assets", "checkpoints", "debug", "logs", "outputs"}
)
_IMPLEMENTATION_IGNORED_TOP_LEVEL = frozenset({"data", "models"})
_IMPLEMENTATION_DISTRIBUTIONS = (
    "mujoco",
    "numpy",
    "opencv-python",
    "opencv-python-headless",
    "robosuite",
    "sam-2",
    "torch",
    "torchvision",
)


def _source_implementation_identity(root: Path) -> dict[str, Any]:
    """Hash executable provider source without recording its machine path."""

    resolved = root.expanduser().resolve(strict=True)
    if not resolved.is_dir():
        raise ValueError(f"provider source root must be a directory: {resolved}")
    files: list[Path] = []
    for current, directory_names, file_names in os.walk(
        resolved,
        topdown=True,
        followlinks=False,
    ):
        current_path = Path(current)
        is_root = current_path == resolved
        directory_names[:] = sorted(
            name
            for name in directory_names
            if name not in _IMPLEMENTATION_IGNORED_DIRECTORIES
            and not (is_root and name in _IMPLEMENTATION_IGNORED_TOP_LEVEL)
            and not (current_path / name).is_symlink()
        )
        for name in sorted(file_names):
            path = current_path / name
            if (
                path.suffix in _IMPLEMENTATION_SOURCE_SUFFIXES
                and not path.is_symlink()
                and path.is_file()
            ):
                files.append(path)
    if not files:
        raise ValueError(f"provider source root contains no executable source: {resolved}")
    digest = hashlib.sha256()
    total_bytes = 0
    for path in sorted(files, key=lambda item: item.relative_to(resolved).as_posix()):
        relative = path.relative_to(resolved).as_posix()
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
        total_bytes += len(payload)
    return {
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _module_source_root(name: str) -> Path | None:
    spec = importlib.util.find_spec(name)
    if spec is None:
        return None
    if spec.submodule_search_locations:
        locations = [Path(item) for item in spec.submodule_search_locations]
        if len(locations) != 1:
            raise ValueError(f"module {name!r} has ambiguous source roots")
        return locations[0]
    if spec.origin:
        return Path(spec.origin).parent
    return None


def _distribution_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "missing"


def _runtime_implementation_tokens(workspace: Workspace) -> dict[str, str]:
    """Bind exact resume to provider code and installed runtime identities."""

    source_groups = {
        "video2traj": ("grounding_dino", "cotracker", "dvd"),
        "exec": ("robocasa",),
        "task_success": ("robocasa",),
    }
    common = {
        distribution: _distribution_version(distribution)
        for distribution in _IMPLEMENTATION_DISTRIBUTIONS
    }
    sam2_root = _module_source_root("sam2")
    sam2_identity = (
        None if sam2_root is None else _source_implementation_identity(sam2_root)
    )
    tokens: dict[str, str] = {}
    for stage, names in source_groups.items():
        sources = {
            name: _source_implementation_identity(
                workspace.binding("sources", name).path
            )
            for name in names
        }
        payload: dict[str, Any] = {
            "distributions": common,
            "sources": sources,
            "stage": stage,
        }
        if stage == "video2traj":
            payload["sam2"] = sam2_identity
        tokens[stage] = canonical_sha256(payload)
    return tokens


def _package_implementation_identity() -> dict[str, Any]:
    package_root = Path(__file__).resolve().parents[2]
    files = sorted(
        path
        for path in package_root.rglob("*.py")
        if path.is_file() and "__pycache__" not in path.parts
    )
    digest = hashlib.sha256()
    total_bytes = 0
    for path in files:
        relative = path.relative_to(package_root).as_posix()
        payload = path.read_bytes()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(payload)
        digest.update(b"\0")
        total_bytes += len(payload)
    return {
        "files": len(files),
        "bytes": total_bytes,
        "sha256": digest.hexdigest(),
    }


def _materialize_gt_reference_snapshot(
    *,
    uid: str,
    sample: Path,
    run_work_root: Path,
    reference_options: Mapping[str, Any],
    resume_enabled: bool,
) -> Path:
    """Create a stable comparison copy for the GT-depth candidate itself."""

    result = run_benchmark_workflow(
        uid=uid,
        sample_dir=sample,
        run_key=str(reference_options["run_key"]),
        runtime_config=reference_options["runtime_config"],
        source_video_path=reference_options["video_path"],
        trajectory_config_path=reference_options["trajectory_config_path"],
        execution_config_path=reference_options["execution_config_path"],
        only_stages=("video2traj",),
        stage_state_policy="resume" if resume_enabled else "off",
        implementation_tokens=reference_options["implementation_tokens"],
        continue_on_error=False,
    )
    pipeline = result.get("pipeline")
    if not isinstance(pipeline, Mapping) or pipeline.get("status") != "completed":
        raise RuntimeError("GT-depth evaluation reference did not complete")
    formal = formal_artifact_paths(
        sample,
        str(reference_options["run_key"]),
    )
    source = formal["ee_traj"].resolve(strict=True)
    returned = Path(str(dict(result.get("artifacts", {})).get("ee_traj", "")))
    if returned.expanduser().absolute() != source:
        raise RuntimeError("GT-depth reference returned a non-formal trajectory")
    snapshot = (
        run_work_root
        / "references"
        / uid
        / "gt_video-gt_depth"
        / "eef.json"
    )
    _copy_or_verify_file(source, snapshot)
    return snapshot

def _current_model_id(input_spec: Mapping[str, Any]) -> str:
    model = str(input_spec["model_id"])
    variant = str(input_spec["prompt_variant"])
    if variant == "standard":
        return model
    if variant == "enhanced":
        return f"{model}-enhanced"
    return f"{model}--custom"


def _input_label(input_spec: Mapping[str, Any]) -> str:
    if input_spec["kind"] == "reference":
        return f"reference/{input_spec['reference_id']}"
    return f"{input_spec['model_id']}/{input_spec['prompt_variant']}"


def _bind_input(
    *,
    repository: BenchRepository,
    outputs: OutputRepository,
    uid: str,
    input_spec: Mapping[str, Any],
    sample: Path,
    reference: ReferenceSelection,
) -> tuple[dict[str, Any], Mapping[str, Any], Path, str, str]:
    if input_spec["kind"] == "reference":
        identity = {
            "kind": "reference",
            "model_id": None,
            "prompt_variant": None,
            "reference_id": str(input_spec["reference_id"]),
            "video_sha256": reference.manifest["video"]["sha256"],
        }
        return (
            identity,
            reference.manifest,
            sample / "artifacts" / "gt" / "video" / "gt.mp4",
            (
                "gt_video/gt_depth"
                if identity["reference_id"] == "w_gt_depth"
                else "gt_video/dvd_depth"
            ),
            "",
        )
    model_id = str(input_spec["model_id"])
    variant = str(input_spec["prompt_variant"])
    selected = outputs.load_video(uid, model_id, variant)
    generation = repository.load_generation_input(uid, verify_frame=False)
    expected_generation = canonical_sha256(generation)
    declared_generation = selected.manifest["generation_input_sha256"]
    if declared_generation is not None and declared_generation != expected_generation:
        raise ValueError(
            f"video output was generated from another case input: {uid}/{model_id}/{variant}"
        )
    current_model = _current_model_id(input_spec)
    processed, raw, _unused_candidate = generated_video_candidates(sample, current_model)
    _copy_or_verify_file(selected.pipeline_input, processed)
    if selected.video != selected.pipeline_input:
        _copy_or_verify_file(selected.video, raw)
    identity = {
        "kind": "generated",
        "model_id": model_id,
        "prompt_variant": variant,
        "reference_id": None,
        "video_sha256": selected.manifest[
            "video"
            if selected.manifest["preprocessed"] is None
            else "preprocessed"
        ]["sha256"],
    }
    return identity, selected.manifest, processed, "gen", current_model


def _runtime_binding(workspace: Workspace, kind: str, name: str) -> Path:
    binding = workspace.binding(kind, name)
    if not binding.path.exists():
        raise FileNotFoundError(
            f"workspace {kind} binding does not exist: {name}={binding.path}"
        )
    return binding.path


def build_default_runtime_config(
    workspace: Workspace,
    *,
    device: str = "cuda:0",
) -> dict[str, Any]:
    """Resolve provider settings from explicit workspace bindings."""

    grounding_source = _runtime_binding(workspace, "sources", "grounding_dino")
    cotracker_source = _runtime_binding(workspace, "sources", "cotracker")
    dvd_source = _runtime_binding(workspace, "sources", "dvd")
    grounding_checkpoint = _runtime_binding(workspace, "checkpoints", "grounding_dino")
    bert = _runtime_binding(workspace, "checkpoints", "bert_base_uncased")
    sam2 = _runtime_binding(workspace, "checkpoints", "sam2")
    cotracker = _runtime_binding(workspace, "checkpoints", "cotracker")
    dvd_assets = _runtime_binding(workspace, "checkpoints", "dvd_assets")
    dvd_binding = workspace.binding("checkpoints", "dvd_assets")
    dvd_manifest = dvd_binding.manifest
    if dvd_manifest is None or not dvd_manifest.is_file():
        raise FileNotFoundError(
            "workspace checkpoint binding dvd_assets requires its asset manifest"
        )
    dvd_provider = {
        "backend": "dvd",
        "preset": "bench_resolved",
        "attestation_manifest_path": dvd_manifest.as_posix(),
        "weights_root": dvd_assets.as_posix(),
        "source_root": dvd_source.as_posix(),
        "checkpoints_root": dvd_assets.as_posix(),
        "validate_assets": True,
        "input_size": 512,
        "fp32": False,
        "seed": 42,
    }
    return {
        "device": device,
        "region": {
            "backend": "composed",
            "detector": {
                "backend": "grounding_dino",
                "source_root": grounding_source.as_posix(),
                "config_path": (
                    grounding_source
                    / "groundingdino"
                    / "config"
                    / "GroundingDINO_SwinT_OGC.py"
                ).as_posix(),
                "checkpoint_path": grounding_checkpoint.as_posix(),
                "text_encoder_path": bert.as_posix(),
                "box_threshold": 0.4,
                "text_threshold": 0.3,
            },
            "segmenter": {
                "backend": "sam2",
                "config_name": "configs/sam2.1/sam2.1_hiera_l.yaml",
                "checkpoint_path": sam2.as_posix(),
            },
            "query_samplers": {
                "bbox_gaussian": {"backend": "builtin"},
                "mask_fps": {"backend": "builtin"},
            },
        },
        "tracking": {
            "backend": "cotracker",
            "checkpoint_path": cotracker.as_posix(),
            "source_roots": [cotracker_source.as_posix()],
        },
        "depth": {
            "backend": "bench_resolved",
            "preset": "bench_resolved",
            "providers": {"dvd": dvd_provider},
        },
        "depth_calibration": {"backend": "builtin"},
        "pose": {"backend": "pointcloud_kabsch"},
    }


def _bind_runtime_seed(
    runtime_config: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    resolved = copy.deepcopy(dict(runtime_config))
    raw_depth = resolved.get("depth")
    if not isinstance(raw_depth, Mapping):
        return resolved
    depth = copy.deepcopy(dict(raw_depth))
    if str(depth.get("backend", "") or "").strip().lower() == "bench_resolved":
        providers = depth.get("providers")
        if not isinstance(providers, Mapping):
            raise TypeError("bench-resolved depth runtime requires providers")
        depth["providers"] = {
            str(name): {**copy.deepcopy(dict(provider)), "seed": int(seed)}
            for name, provider in providers.items()
        }
    else:
        depth["seed"] = int(seed)
    resolved["depth"] = depth
    return resolved


def _run_values(
    *,
    sample: Path,
    video_path: Path,
    selected_kind: str,
    reference: ReferenceSelection,
    use_gt_depth: bool,
) -> dict[str, Mapping[str, Any]]:
    depth_materialized = (
        sample / "artifacts" / "gt" / "depth" / "gt_metric.npy"
    )
    gt_video = sample / "artifacts" / "gt" / "video" / "gt.mp4"
    return {
        "video2traj": {
            "input": {
                "selected_video": selected_kind,
                "rollout_video_path": gt_video.as_posix(),
                "gen_video_path": "" if selected_kind == "rollout" else video_path.as_posix(),
            },
            "depth": {
                "rollout_gt_depth_path": depth_materialized.as_posix(),
                "use_rollout_gt_depth": use_gt_depth,
            },
            "pose": {
                "mesh_path": (
                    sample
                    / "artifacts"
                    / "init"
                    / "cad"
                    / "eef_mesh_visual_only.obj"
                ).as_posix(),
            },
        },
    }


def _portable_input_mappings(
    *,
    workspace: Workspace,
    repository: BenchRepository,
    outputs: OutputRepository,
    uid: str,
    sample: Path,
    input_identity: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
) -> list[tuple[Path, Path]]:
    """Map work-local runtime inputs back to their canonical owners."""

    case_root = repository.case_dir(uid)
    mappings: list[tuple[Path, Path]] = [
        (
            sample / "artifacts" / "init" / "img" / "init_env.png",
            case_root / "generation" / "first_frame.png",
        ),
        (
            sample / "artifacts" / "init" / "depth",
            case_root / "env" / "init" / "depth",
        ),
        (
            sample / "artifacts" / "init" / "mask",
            case_root / "env" / "init" / "mask",
        ),
        (
            sample / "artifacts" / "init" / "cad" / "eef_mesh_visual_only.obj",
            case_root / "env" / "geometry" / "eef" / "visual.obj",
        ),
        (
            sample
            / "artifacts"
            / "init"
            / "cad"
            / "eef_mesh_visual_only_meta.json",
            case_root / "env" / "geometry" / "eef" / "meta.json",
        ),
        (sample / "artifacts" / "gt", case_root / "references"),
        (sample / "artifacts" / "env" / "scene", case_root / "env" / "scene"),
        (
            sample / "artifacts" / "env" / "state",
            case_root / "env" / "init" / "state",
        ),
    ]
    for name in (
        "environment.json",
        "entities.json",
        "task_runtime.json",
        "runtime.lock.json",
        "origin.json",
    ):
        mappings.append(
            (sample / "artifacts" / "env" / name, case_root / "env" / name)
        )
    mappings.append(
        (
            sample / "artifacts" / "env" / "camera.json",
            case_root / "env" / "environment.json",
        )
    )
    if input_identity["kind"] == "generated":
        model_id = str(input_identity["model_id"])
        variant = str(input_identity["prompt_variant"])
        video_root = outputs.video_dir(uid, model_id, variant)
        current_model = _current_model_id(input_identity)
        processed, raw, _candidate = generated_video_candidates(sample, current_model)
        consumed_name = (
            "video.mp4"
            if input_manifest["preprocessed"] is None
            else "preprocessed.mp4"
        )
        mappings.append((processed, video_root / consumed_name))
        mappings.append((raw, video_root / "video.mp4"))
    return mappings


def _portable_document(
    document: Mapping[str, Any],
    *,
    owner_path: Path,
    published_owner_path: Path,
    path_mappings: Sequence[tuple[Path, Path]],
    workspace: Workspace,
) -> dict[str, Any]:
    return rewrite_absolute_paths(
        document,
        owner_path=owner_path,
        published_owner_path=published_owner_path,
        path_mappings=path_mappings,
        workspace=workspace,
        package_root=Path(__file__).resolve().parents[3],
    )


def _current_trajectory_config(resolved: Mapping[str, Any]) -> dict[str, Any]:
    trajectory = copy.deepcopy(dict(resolved["values"]["video2traj"]))
    if "action" in trajectory:
        raise ValueError("video2traj protocol must not own the action config")
    trajectory["action"] = copy.deepcopy(dict(resolved["values"]["action"]))
    return trajectory


def _ensure_run_metadata(
    workspace: Workspace,
    run: Mapping[str, Any],
) -> None:
    root = workspace.outputs_root / "runs" / run_relative_path(run)
    run_path = root / "run.json"
    if run_path.is_file():
        existing = load_and_validate(run_path, expected_schema=RUN_SCHEMA)
        if canonical_sha256(existing) != canonical_sha256(run):
            raise ValueError("published experiment run spec conflicts")
    else:
        write_json_atomic(run_path, run, exclusive=True)


def _write_run_summary(
    workspace: Workspace,
    run: Mapping[str, Any],
    report: Mapping[str, Any],
) -> dict[str, Any]:
    run_id = str(run["destination"]["run_id"])
    records: list[dict[str, Any]] = []
    for task in report["tasks"]:
        if task["status"] not in {"published", "reused"}:
            continue
        identity = task["input"]
        result_path = _published_result_path(
            workspace,
            run,
            str(task["uid"]),
            identity,
        ) / "result.json"
        if not result_path.is_file():
            raise FileNotFoundError(
                f"run summary result is missing: {result_path}"
            )
        records.append(_file_record(result_path, workspace.outputs_root))
    counts = dict(sorted(Counter(item["status"] for item in report["tasks"]).items()))
    incomplete = any(
        item["status"] not in {"published", "reused"}
        for item in report["tasks"]
    )
    if records and not incomplete:
        status = "completed"
    elif records:
        status = "partial"
    else:
        status = "failed"
    summary = {
        "format": RUN_SUMMARY_SCHEMA,
        "run_id": run_id,
        "run_sha256": canonical_sha256(run),
        "status": status,
        "counts": counts,
        "results": sorted(records, key=lambda item: item["path"]),
    }
    validate_document(summary, expected_schema=RUN_SUMMARY_SCHEMA)
    write_json_atomic(
        workspace.outputs_root / "runs" / run_relative_path(run) / "summary.json",
        summary,
    )
    return summary


def _published_result_path(
    workspace: Workspace,
    run: Mapping[str, Any],
    uid: str,
    input_identity: Mapping[str, Any],
) -> Path:
    root = workspace.outputs_root / "experiments" / uid
    if input_identity["kind"] == "reference":
        return root / "reference" / input_identity["reference_id"]
    return root / input_identity["model_id"] / input_identity["prompt_variant"]


def _result_request(
    *,
    repository: BenchRepository,
    run: Mapping[str, Any],
    uid: str,
    input_identity: Mapping[str, Any],
    input_manifest: Mapping[str, Any],
    resolved: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
    implementation_sha256: str,
    receipt: Mapping[str, Any] | None,
) -> dict[str, Any]:
    case = repository.load_case(uid)
    init = repository.load_environment(uid, verify_files=False)
    generation = repository.load_generation_input(uid, verify_frame=False)
    reference = repository.load_reference(uid, verify_files=False)
    route = (
        "reference_input"
        if input_identity["kind"] == "reference"
        else "candidate"
    )
    config_digests = config_identity(
        repository,
        uid=uid,
        route=route,
    )
    execution_spec = {
        "stages": list(run["stages"]),
        "seed": run["seed"],
        "initialization": copy.deepcopy(run["initialization"]),
        "artifact_level": run["artifact_level"],
    }
    document = {
        "format": RESULT_REQUEST_SCHEMA,
        "execution_spec_sha256": canonical_sha256(execution_spec),
        "case_sha256": canonical_sha256(case),
        "init_sha256": canonical_sha256(init.manifest),
        "generation_input_sha256": (
            None
            if input_identity["kind"] == "reference"
            else canonical_sha256(generation)
        ),
        "reference_sha256": canonical_sha256(reference.manifest),
        "packaged_defaults_sha256": config_digests["packaged_defaults"],
        "protocol_sha256": config_digests["protocol"],
        "case_protocol_sha256": config_digests["case_protocol"],
        "input": dict(input_identity),
        "input_manifest_sha256": canonical_sha256(input_manifest),
        "resolved_config_sha256": canonical_sha256(resolved),
        "runtime_config_sha256": canonical_sha256(runtime_config),
        "implementation_sha256": implementation_sha256,
        "initialization": {
            "mode": run["initialization"]["mode"],
            "receipt_sha256": None if receipt is None else canonical_sha256(receipt),
        },
        "provenance_status": "complete",
    }
    return validate_document(document, expected_schema=RESULT_REQUEST_SCHEMA)


def _move_path(
    source: Path,
    destination: Path,
    *,
    path_mappings: list[tuple[Path, Path]],
) -> None:
    if not source.exists():
        return
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    path_mappings.append(
        (source.resolve(strict=False), destination.resolve(strict=False))
    )
    os.rename(source, destination)


def _move_directory_contents(
    source: Path,
    destination: Path,
    *,
    path_mappings: list[tuple[Path, Path]],
) -> None:
    if not source.is_dir():
        return
    for child in sorted(source.iterdir()):
        _move_path(
            child,
            destination / child.name,
            path_mappings=path_mappings,
        )


def _drop_unpublished_absolute_paths(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _drop_unpublished_absolute_paths(item)
            for key, item in value.items()
            if not is_absolute_path_text(item)
        }
    if isinstance(value, list):
        return [
            _drop_unpublished_absolute_paths(item)
            for item in value
            if not is_absolute_path_text(item)
        ]
    return copy.deepcopy(value)


def _sanitize_core_trajectory_json(path: Path) -> None:
    document = _json_object(path, "core trajectory artifact")
    write_json_atomic(path, _drop_unpublished_absolute_paths(document))


def _separate_workflow_artifacts(
    source_root: Path,
    bundle: Path,
    *,
    artifact_level: str,
) -> list[tuple[Path, Path]]:
    if artifact_level not in {"core", "full"}:
        raise ValueError(f"unsupported artifact level: {artifact_level!r}")
    include_intermediate = artifact_level == "full"
    path_mappings: list[tuple[Path, Path]] = []
    traj = source_root / "traj"
    exact_trajectory = {
        traj / "trajectory" / "ee_traj.json": bundle / "trajectory" / "eef.json",
        traj / "trajectory" / "obj_trajs.json": bundle / "trajectory" / "objects.json",
        traj / "trajectory" / "union_traj.json": bundle / "trajectory" / "combined.json",
        traj / "gripper" / "gripper.json": bundle / "trajectory" / "gripper.json",
    }
    for source, target in exact_trajectory.items():
        _move_path(source, target, path_mappings=path_mappings)
        if target.is_file():
            _sanitize_core_trajectory_json(target)
    producer_action = traj / "action" / "action.json"
    if producer_action.is_file():
        write_motion_plan_bundle_from_json(
            producer_action,
            bundle / "action",
            exclusive=True,
        )
        path_mappings.append(
            (
                producer_action.resolve(strict=False),
                (bundle / "action" / "meta.json").resolve(strict=False),
            )
        )
        if include_intermediate:
            _move_path(
                producer_action,
                bundle / "intermediate" / "action_planner" / "producer_action.json",
                path_mappings=path_mappings,
            )
    if include_intermediate:
        for name in (
            "region",
            "tracking",
            "depth",
            "geometry",
            "pose",
            "visualization",
            "diagnostics",
        ):
            _move_path(
                traj / name,
                bundle / "intermediate" / name,
                path_mappings=path_mappings,
            )
        _move_path(
            traj / "manifest.json",
            bundle / "logs" / "manifests" / "trajectory.json",
            path_mappings=path_mappings,
        )
        if traj.is_dir():
            for path in sorted(traj.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(traj)
                    _move_path(
                        path,
                        bundle / "intermediate" / "trajectory_unclassified" / relative,
                        path_mappings=path_mappings,
                    )

    execution = source_root / "exec"
    path_mappings.append(
        (
            execution.resolve(strict=False),
            (bundle / "execution").resolve(strict=False),
        )
    )
    _move_path(
        execution / "exec.mp4",
        bundle / "execution" / "exec.mp4",
        path_mappings=path_mappings,
    )
    _move_path(
        execution / "exec_summary.json",
        bundle / "execution" / "summary.json",
        path_mappings=path_mappings,
    )
    if include_intermediate:
        for name in (
            "checkpoint_trace.json",
            "dense_tcp_trace.json",
            "action_trace.json",
            "execution_inputs.json",
        ):
            _move_path(
                execution / name,
                bundle / "execution" / "traces" / name,
                path_mappings=path_mappings,
            )
    for path in list(sorted(execution.glob("*"))):
        if not path.exists():
            continue
        name = path.name
        evaluation_target = {
            "exec_metrics.json": _TRAJECTORY_EXECUTABILITY_FILENAME,
            "task_check_success.json": _TASK_SUCCESS_FILENAME,
            "evaluation_result.json": "evaluation_result.json",
        }.get(name)
        if name.startswith("vlm_") or name in {"vlm", "evaluation"}:
            evaluation_target = name
        if evaluation_target is not None:
            _move_path(
                path,
                bundle / "evaluation" / evaluation_target,
                path_mappings=path_mappings,
            )
    if include_intermediate:
        _move_path(
            execution / "manifest.json",
            bundle / "logs" / "manifests" / "execution.json",
            path_mappings=path_mappings,
        )
        if execution.is_dir():
            for path in sorted(execution.rglob("*")):
                if path.is_file():
                    relative = path.relative_to(execution)
                    _move_path(
                        path,
                        bundle / "intermediate" / "execution_unclassified" / relative,
                        path_mappings=path_mappings,
                    )

        _move_directory_contents(
            source_root / "logs",
            bundle / "logs",
            path_mappings=path_mappings,
        )
        _move_path(
            source_root / "pipeline_config",
            bundle / "logs" / "pipeline_config",
            path_mappings=path_mappings,
        )
        for name in ("run.json", "pipeline_state.json"):
            _move_path(
                source_root / name,
                bundle / "logs" / name,
                path_mappings=path_mappings,
            )
        for path in sorted(source_root.rglob("*")):
            if path.is_file():
                relative = path.relative_to(source_root)
                _move_path(
                    path,
                    bundle / "intermediate" / "producer_unclassified" / relative,
                    path_mappings=path_mappings,
                )
    return path_mappings


def _portableize_bundle_json(
    *,
    bundle: Path,
    published_root: Path,
    path_mappings: Sequence[tuple[Path, Path]],
    workspace: Workspace,
) -> None:
    virtual_mappings: list[tuple[Path, Path]] = []
    for source, target in path_mappings:
        try:
            relative = target.resolve(strict=False).relative_to(
                bundle.resolve(strict=False)
            )
        except ValueError:
            virtual_target = target
        else:
            virtual_target = published_root / relative
        virtual_mappings.append((source, virtual_target))
    for path in sorted(bundle.rglob("*.json")):
        if path.is_symlink() or not path.is_file():
            raise ValueError(f"published JSON must be a regular file: {path}")
        # Intermediate producer JSON is not uniformly object-shaped: the
        # interaction-geometry traces are top-level arrays. They still need
        # the same recursive path portability audit as object-shaped files.
        document = json.loads(path.read_text(encoding="utf-8"))
        relative = path.relative_to(bundle)
        portable = rewrite_absolute_paths(
            document,
            owner_path=path,
            published_owner_path=published_root / relative,
            path_mappings=virtual_mappings,
            workspace=workspace,
            package_root=Path(__file__).resolve().parents[3],
            allow_non_object_root=True,
        )
        write_json_atomic(path, portable)


def _result_bundle_from_workflow(
    *,
    workspace: Workspace,
    run: Mapping[str, Any],
    uid: str,
    input_identity: Mapping[str, Any],
    run_key: str,
    current_model: str,
    sample: Path,
    resolved: Mapping[str, Any],
    request: Mapping[str, Any],
    portable_runtime_config: Mapping[str, Any],
    workflow: Mapping[str, Any],
    input_path_mappings: Sequence[tuple[Path, Path]],
    runtime_config_paths: Mapping[str, Path],
) -> dict[str, Any]:
    formal = formal_artifact_paths(
        sample,
        run_key,
        gen_model=current_model,
    )
    source_root = formal["run_root"]
    label = _input_label(input_identity).replace("/", "--")
    bundle = (
        workspace.run_root
        / run["destination"]["run_id"]
        / "publish"
        / uid
        / label
    )
    if bundle.exists():
        raise FileExistsError(bundle)
    bundle.mkdir(parents=True)
    path_mappings = list(input_path_mappings)
    artifact_level = str(run["artifact_level"])
    path_mappings.extend(
        _separate_workflow_artifacts(
            source_root,
            bundle,
            artifact_level=artifact_level,
        )
    )
    reference_trajectory = Path(
        str(workflow.get("reference_trajectory_path", "") or "")
    ).expanduser()
    if reference_trajectory.is_file():
        phases = workflow.get("phases")
        if not isinstance(phases, Mapping):
            raise ValueError("workflow is missing reference phase metadata")
        reference_phase = phases.get("reference")
        if not isinstance(reference_phase, Mapping):
            raise ValueError("workflow is missing its reference phase")
        reference_request = reference_phase.get("request")
        if isinstance(reference_request, Mapping):
            reference_run_key = str(
                reference_request.get("run_key", "") or ""
            ).strip()
            if not reference_run_key:
                raise ValueError("workflow reference request is missing run_key")
            expected_reference_root = formal_artifact_paths(
                sample,
                reference_run_key,
            )["trajectory_dir"].resolve(strict=True)
        elif reference_phase.get("status") == "provided":
            expected_reference_root = (
                workspace.run_root
                / run["destination"]["run_id"]
                / "references"
                / uid
            ).resolve(strict=True)
        else:
            raise ValueError("workflow reference phase is missing its request")
        resolved_reference = reference_trajectory.resolve(strict=True)
        if not resolved_reference.is_relative_to(expected_reference_root):
            raise ValueError("reference trajectory escapes the GT reference run")
        reference_target = bundle / "evaluation" / "reference" / "eef.json"
        _copy_file(resolved_reference, reference_target)
        _sanitize_core_trajectory_json(reference_target)
        path_mappings.append(
            (
                resolved_reference,
                reference_target.resolve(strict=False),
            )
        )
    write_json_atomic(bundle / "resolved_config.json", resolved, exclusive=True)
    write_json_atomic(bundle / "request.json", request, exclusive=True)
    if artifact_level == "full":
        write_json_atomic(
            bundle / "logs" / "runtime" / "providers.json",
            portable_runtime_config,
            exclusive=True,
        )
        for name, source in sorted(runtime_config_paths.items()):
            target = bundle / "logs" / "runtime" / f"{name}.json"
            _copy_file(source, target)
            path_mappings.append(
                (source.resolve(strict=False), target.resolve(strict=False))
            )
        write_json_atomic(bundle / "logs" / "workflow.json", workflow, exclusive=True)
    published_root = _published_result_path(
        workspace,
        run,
        uid,
        input_identity,
    )
    _portableize_bundle_json(
        bundle=bundle,
        published_root=published_root,
        path_mappings=path_mappings,
        workspace=workspace,
    )
    task_path = bundle / "evaluation" / _TASK_SUCCESS_FILENAME
    if task_path.is_file():
        task = _json_object(task_path, "task success")
        outcome = "success" if task.get("final_task_check_success") is True else "failed"
    else:
        outcome = "not_evaluated"
    artifacts = [
        item
        for item in _hash_tree(bundle, relative_base=bundle)
        if item["path"] not in {"request.json", "resolved_config.json"}
    ]
    result = {
        "format": RESULT_SCHEMA,
        "run_id": run["destination"]["run_id"],
        "uid": uid,
        "input": dict(input_identity),
        "status": "completed",
        "completed_stages": list(PIPELINE_STAGES),
        "task_outcome": outcome,
        "provenance_status": "complete",
        "request_sha256": canonical_sha256(request),
        "resolved_config_sha256": canonical_sha256(resolved),
        "artifacts": artifacts,
    }
    return finalize_result(workspace=workspace, work_bundle=bundle, result=result)


def _verify_published_result(
    path: Path,
    *,
    expected_request_sha256: str | None,
) -> dict[str, Any]:
    result = load_and_validate(path / "result.json", expected_schema=RESULT_SCHEMA)
    request = load_and_validate(path / "request.json", expected_schema=RESULT_REQUEST_SCHEMA)
    request_digest = canonical_sha256(request)
    if request_digest != result["request_sha256"]:
        raise ValueError(f"published request digest mismatch: {path}")
    if expected_request_sha256 is not None and request_digest != expected_request_sha256:
        raise ValueError(f"exact resume request mismatch: {path}")
    resolved = load_and_validate(path / "resolved_config.json", expected_schema=RESOLVED_CONFIG_SCHEMA)
    if canonical_sha256(resolved) != result["resolved_config_sha256"]:
        raise ValueError(f"published resolved config digest mismatch: {path}")
    if input_identity_key(result["input"]) != input_identity_key(request["input"]):
        raise ValueError(f"published input identity mismatch: {path}")
    for artifact in result["artifacts"]:
        source = path / artifact["path"]
        if source.stat().st_size != artifact["size"] or sha256_file(source) != artifact["sha256"]:
            raise ValueError(f"published result artifact mismatch: {source}")
    return result



def run_benchmark(
    *,
    workspace_path: str | Path,
    run_spec_path: str | Path,
    single_uid_runner: Callable[..., Mapping[str, Any]] = run_single_uid_workflow,
    runtime_config_builder: Callable[[Workspace], Mapping[str, Any]] = build_default_runtime_config,
    test_only_allow_injected_runner: bool = False,
) -> dict[str, Any]:
    workspace = load_workspace(workspace_path)
    run = load_and_validate(run_spec_path, expected_schema=RUN_SCHEMA)
    repository = BenchRepository(workspace.bench_root)
    outputs = OutputRepository(
        workspace.outputs_root,
        published_root=workspace.published_results_root,
    )
    repository.validate_workspace_bindings(workspace)
    if single_uid_runner is not run_single_uid_workflow and not test_only_allow_injected_runner:
        raise ValueError("single_uid_runner injection is test-only")
    if run["stages"] != list(PIPELINE_STAGES):
        raise ValueError("benchmark run requires the canonical pipeline stage chain")
    run_id = run["destination"]["run_id"]
    runtime_config = _bind_runtime_seed(
        runtime_config_builder(workspace),
        seed=run["seed"],
    )
    implementation_tokens = _runtime_implementation_tokens(workspace)
    implementation_sha256 = canonical_sha256(
        {
            "package": _package_implementation_identity(),
            "providers": implementation_tokens,
        }
    )
    _ensure_run_metadata(workspace, run)
    run_work_root = workspace.run_root / run_id
    run_snapshot = run_work_root / "run.json"
    if run_snapshot.is_file():
        existing = load_and_validate(run_snapshot, expected_schema=RUN_SCHEMA)
        if canonical_sha256(existing) != canonical_sha256(run):
            raise ValueError("experiment work root is bound to another run spec")
    else:
        write_json_atomic(run_snapshot, run, exclusive=True)
    report: dict[str, Any] = {
        "format": "dream-exe.run-report",
        "run_id": run_id,
        "status": "completed",
        "tasks": [],
    }

    for uid in _select_uids(repository, run):
        canonical_case = repository.case_dir(uid)
        protected_before = _tree_identity(canonical_case)
        materialized = run_work_root / "materialized" / uid
        if materialized.exists():
            receipt, reference = _verify_materialized_run_sample(
                workspace=workspace,
                repository=repository,
                run=run,
                uid=uid,
                destination=materialized,
            )
        else:
            receipt, reference = _materialize_run_sample(
                workspace=workspace,
                repository=repository,
                run=run,
                uid=uid,
                destination=materialized,
            )
        _fixed_input_files(materialized)
        reference_identity = {
            "kind": "reference",
            "model_id": None,
            "prompt_variant": None,
            "reference_id": "w_gt_depth",
            "video_sha256": reference.manifest["video"]["sha256"],
        }
        reference_video = materialized / "artifacts" / "gt" / "video" / "gt.mp4"
        reference_resolved = compile_case_route_config(
            repository,
            uid=uid,
            route="evaluation_oracle",
            input_identity=reference_identity,
            run_values=_run_values(
                sample=materialized,
                video_path=reference_video,
                selected_kind="rollout",
                reference=reference,
                use_gt_depth=True,
            ),
        )
        reference_config_root = run_work_root / "configs" / uid / "evaluation_reference"
        reference_traj = write_json_atomic(
            reference_config_root / "trajectory.resolved.json",
            _current_trajectory_config(reference_resolved),
        )
        reference_exec = write_json_atomic(
            reference_config_root / "execution.json",
            reference_resolved["values"]["execution"],
        )
        reference_options = {
            "run_key": "gt_video/gt_depth",
            "runtime_config": runtime_config,
            "implementation_tokens": {
                "video2traj": implementation_tokens["video2traj"],
            },
            "trajectory_config_path": reference_traj,
            "execution_config_path": reference_exec,
            "video_path": reference_video,
        }

        for input_spec in run["inputs"]:
            input_error: Exception | None = None
            try:
                identity, input_manifest, video_path, run_key, current_model = _bind_input(
                    repository=repository,
                    outputs=outputs,
                    uid=uid,
                    input_spec=input_spec,
                    sample=materialized,
                    reference=reference,
                )
            except FileNotFoundError as error:
                failure_status = "input_missing"
                input_error = error
            except ValueError as error:
                failure_status = "input_invalid"
                input_error = error
            else:
                failure_status = ""
            if failure_status:
                label = _input_label(input_spec).replace("/", "--")
                failure = {
                    "uid": uid,
                    "input": dict(input_spec),
                    "status": failure_status,
                    "error": str(input_error),
                }
                report["tasks"].append(failure)
                report["status"] = "partial"
                write_json_atomic(
                    run_work_root / "failures" / uid / f"{label}.json",
                    failure,
                )
                continue
            run_values = _run_values(
                sample=materialized,
                video_path=video_path,
                selected_kind=(
                    "rollout" if identity["kind"] == "reference" else "gen"
                ),
                reference=reference,
                use_gt_depth=(
                    identity["kind"] == "reference"
                    and identity["reference_id"] == "w_gt_depth"
                ),
            )
            if identity["kind"] == "reference":
                resolved = compile_case_route_config(
                    repository,
                    uid=uid,
                    route="reference_input",
                    input_identity=identity,
                    run_values=run_values,
                )
            else:
                resolved = compile_repository_config(
                    repository,
                    uid=uid,
                    input_identity=identity,
                    run_values=run_values,
                )
            published = _published_result_path(workspace, run, uid, identity)
            input_path_mappings = _portable_input_mappings(
                workspace=workspace,
                repository=repository,
                outputs=outputs,
                uid=uid,
                sample=materialized,
                input_identity=identity,
                input_manifest=input_manifest,
            )
            portable_resolved = _portable_document(
                resolved,
                owner_path=run_work_root / "configs" / uid / "resolved_config.json",
                published_owner_path=published / "resolved_config.json",
                path_mappings=input_path_mappings,
                workspace=workspace,
            )
            portable_runtime_config = _portable_document(
                public_runtime_config(runtime_config),
                owner_path=run_work_root / "runtime" / "providers.json",
                published_owner_path=published / "logs" / "runtime" / "providers.json",
                path_mappings=input_path_mappings,
                workspace=workspace,
            )
            request = _result_request(
                repository=repository,
                run=run,
                uid=uid,
                input_identity=identity,
                input_manifest=input_manifest,
                resolved=portable_resolved,
                runtime_config=portable_runtime_config,
                implementation_sha256=implementation_sha256,
                receipt=receipt,
            )
            if published.exists() and run["resume"]["enabled"]:
                result = _verify_published_result(
                    published,
                    expected_request_sha256=(
                        canonical_sha256(request) if run["resume"]["exact"] else None
                    ),
                )
                report["tasks"].append(
                    {"uid": uid, "input": identity, "status": "reused", "result": result}
                )
                continue
            label = _input_label(identity).replace("/", "--")
            config_root = run_work_root / "configs" / uid / label
            candidate_traj = write_json_atomic(
                config_root / "trajectory.resolved.json",
                _current_trajectory_config(resolved),
            )
            candidate_exec = write_json_atomic(
                config_root / "execution.json",
                resolved["values"]["execution"],
            )
            candidate_options = {
                "run_key": run_key,
                "gen_model": current_model,
                "runtime_config": runtime_config,
                "implementation_tokens": implementation_tokens,
                "trajectory_config_path": candidate_traj,
                "execution_config_path": candidate_exec,
                "video_path": video_path,
            }
            reference_trajectory_path: Path | None = None
            effective_reference_options: Mapping[str, Any] | None = reference_options
            if (
                single_uid_runner is run_single_uid_workflow
                and identity["kind"] == "reference"
                and identity["reference_id"] == "w_gt_depth"
            ):
                reference_trajectory_path = _materialize_gt_reference_snapshot(
                    uid=uid,
                    sample=materialized,
                    run_work_root=run_work_root,
                    reference_options=reference_options,
                    resume_enabled=run["resume"]["enabled"],
                )
                effective_reference_options = None
            attempts = 0
            workflow: Mapping[str, Any] | None = None
            completed_status = (
                "technical_completed"
                if single_uid_runner is run_single_uid_workflow
                else "completed"
            )
            while attempts < run["retry"]["max_attempts"]:
                attempts += 1
                workflow = single_uid_runner(
                    uid=uid,
                    sample_dir=materialized,
                    candidate_run=candidate_options,
                    reference_run=effective_reference_options,
                    reference_trajectory_path=reference_trajectory_path,
                    include_task_success=True,
                    stage_state_policy="resume" if run["resume"]["enabled"] else "off",
                    **(
                        {"robocasa_source_root": workspace.bindings["sources"]["robocasa"].path}
                        if "robocasa" in workspace.bindings["sources"]
                        else {}
                    ),
                )
                if workflow.get("status") == completed_status:
                    break
            if workflow is None or workflow.get("status") != completed_status:
                failure = {
                    "uid": uid,
                    "input": identity,
                    "status": "producer_failed",
                    "attempts": attempts,
                    "workflow": dict(workflow or {}),
                }
                report["tasks"].append(failure)
                report["status"] = "partial"
                write_json_atomic(
                    run_work_root / "failures" / uid / f"{label}.json",
                    failure,
                )
                if run["retry"]["failure_policy"] == "stop":
                    break
                continue
            publication = _result_bundle_from_workflow(
                workspace=workspace,
                run=run,
                uid=uid,
                input_identity=identity,
                run_key=run_key,
                current_model=current_model,
                sample=materialized,
                resolved=portable_resolved,
                request=request,
                portable_runtime_config=portable_runtime_config,
                workflow=workflow,
                input_path_mappings=input_path_mappings,
                runtime_config_paths={
                    "candidate-trajectory": candidate_traj,
                    "candidate-execution": candidate_exec,
                    "reference-trajectory": reference_traj,
                    "reference-execution": reference_exec,
                },
            )
            report["tasks"].append(
                {"uid": uid, "input": identity, "status": "published", "publication": publication}
            )
        if _tree_identity(canonical_case) != protected_before:
            raise RuntimeError(f"benchmark no-write guard failed for {uid}")
    report["summary"] = dict(sorted(Counter(item["status"] for item in report["tasks"]).items()))
    write_json_atomic(run_work_root / "run-report.json", report)
    report["published_summary"] = _write_run_summary(
        workspace,
        run,
        report,
    )
    return report


__all__ = ["build_default_runtime_config", "run_benchmark"]
