"""Restore a canonical frozen benchmark environment into disposable work."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
from collections.abc import Callable, Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ..contracts.action import (
    gt_action_payload_from_bundle,
    load_action_bundle,
)
from ..contracts.schemas import (
    INIT_RECEIPT_SCHEMA,
    canonical_sha256,
    validate_document,
)
from ..data.repository import (
    BenchRepository,
    EnvironmentSelection,
    ReferenceSelection,
)
from ..data.scene import materialize_frozen_scene
from ..data.workspace import load_workspace
from ..outputs.lifecycle import sha256_file, write_json_atomic


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _tree_identity(root: Path) -> list[tuple[str, str, int, int, int]]:
    output: list[tuple[str, str, int, int, int]] = []
    for path in sorted(root.rglob("*")):
        status = os.lstat(path)
        kind = "symlink" if path.is_symlink() else ("directory" if path.is_dir() else "file")
        output.append(
            (
                path.relative_to(root).as_posix(),
                kind,
                status.st_size,
                status.st_mtime_ns,
                status.st_ino,
            )
        )
    return output


def _copy_file(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_file():
        raise ValueError(f"fixed input must be a regular non-symlink file: {source}")
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _copy_or_verify_file(source: Path, destination: Path) -> None:
    """Materialize once, then accept only an exact existing work copy."""

    if not destination.exists():
        _copy_file(source, destination)
        return
    if destination.is_symlink() or not destination.is_file():
        raise ValueError(f"materialized input is not a regular file: {destination}")
    if destination.stat().st_size != source.stat().st_size:
        raise ValueError(f"materialized input size changed: {destination}")
    if sha256_file(destination) != sha256_file(source):
        raise ValueError(f"materialized input digest changed: {destination}")


def _copy_tree(source: Path, destination: Path) -> None:
    if source.is_symlink() or not source.is_dir():
        raise ValueError(f"fixed input must be a non-symlink directory: {source}")
    if destination.exists():
        raise FileExistsError(destination)
    shutil.copytree(source, destination, symlinks=False, copy_function=shutil.copy2)


def _file_record(path: Path, root: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError(f"materialized artifact must be a regular file: {path}")
    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": sha256_file(path),
        "size": path.stat().st_size,
    }


def _json_object(path: Path, label: str) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError(f"{label} must be a JSON object: {path}")
    return payload


def _environment_snapshot(init: EnvironmentSelection) -> dict[str, Any]:
    snapshot = _json_object(init.one("environment"), "environment snapshot")
    if not isinstance(snapshot.get("raw"), Mapping):
        raise TypeError("environment snapshot.raw must be an object")
    if not isinstance(snapshot.get("camera"), Mapping):
        raise TypeError("environment snapshot.camera must be an object")
    return snapshot


def _runner_meta(case: Mapping[str, Any]) -> dict[str, Any]:
    task = copy.deepcopy(dict(case["task"]))
    return {
        "uid": case["uid"],
        "language": {"instruction": case["instruction"]},
        "task": task,
        "stages": copy.deepcopy(list(case["stages"])),
    }


def _runner_simulator_config(
    init: EnvironmentSelection,
) -> dict[str, Any]:
    """Bind a direct saved-environment snapshot to current runtime paths.

    The canonical environment contains no scene-override patch. This function
    binds the frozen scene, state, task, and final camera snapshot directly to
    the simulator runtime contract.
    """

    environment = _environment_snapshot(init)
    config = copy.deepcopy(environment)
    raw = config.setdefault("raw", {})
    if not isinstance(raw, dict):
        raise TypeError("environment config.raw must be an object")
    scene_restore = raw.setdefault("scene_restore", {})
    if not isinstance(scene_restore, dict):
        raise TypeError("environment config.raw.scene_restore must be an object")
    scene_restore.update(
        {
            "backend": "robocasa_frozen",
            "paths": {
                "model_xml_gz": "scene/model.xml.gz",
                "ep_meta_json": "task_runtime.json",
                "states_npz": "state/states.npz",
            },
            "path_refs": {},
            "scene_override_ref": {},
        }
    )
    scene_restore.pop("dataset_root_ref", None)
    camera = copy.deepcopy(dict(environment["camera"]))
    active = str(
        camera.get("active_camera_name", camera.get("name", "")) or ""
    ).strip()
    camera_definitions = camera.get(
        "cam_configs",
        camera.get("resolved_cam_configs", {}),
    )
    if isinstance(camera_definitions, Mapping):
        scene_restore["resolved_cam_configs"] = copy.deepcopy(
            dict(camera_definitions)
        )
    if active:
        scene_restore["active_camera_name"] = active
        raw["render_camera_name"] = active
    camera_raw = raw.get("camera")
    if not isinstance(camera_raw, Mapping):
        raise TypeError("environment config.raw.camera must be an object")
    raw.pop("scene_override_json", None)
    scene_restore.pop("scene_override_json", None)
    return config


def _runner_camera(init: EnvironmentSelection) -> dict[str, Any]:
    camera = dict(_environment_snapshot(init)["camera"])
    output = copy.deepcopy(camera)
    calibration = output.setdefault("calibration", {})
    if not isinstance(calibration, dict):
        raise TypeError("environment camera.calibration must be an object")
    io = calibration.setdefault("io", {})
    if not isinstance(io, dict):
        raise TypeError("environment camera.calibration.io must be an object")
    io["first_frame_path"] = "../init/img/init_env.png"
    return output


def _runner_entities(init: EnvironmentSelection) -> dict[str, Any]:
    entities = _json_object(init.one("entities"), "environment entities")
    output = copy.deepcopy(entities)
    assets = output.get("assets")
    if not isinstance(assets, dict):
        raise TypeError("environment entities.assets must be an object")
    expected = {
        "eef_geometry_meta_json": "geometry/eef/meta.json",
        "eef_geometry_obj": "geometry/eef/visual.obj",
        "init_depth_npy": "init/depth/init_depth.npy",
        "init_image_png": "../generation/first_frame.png",
        "segmentation_instance_names_json": "init/mask/instance_names.json",
        "segmentation_instance_npy": "init/mask/init_segmentation_instance.npy",
    }
    if assets != expected:
        raise ValueError("canonical environment entity assets are not exact")
    output["assets"] = {
        "eef_cad_meta_json": "../init/cad/eef_mesh_visual_only_meta.json",
        "eef_cad_obj": "../init/cad/eef_mesh_visual_only.obj",
        "eef_cad_visual_meta_json": "../init/cad/eef_mesh_visual_only_meta.json",
        "eef_cad_visual_obj": "../init/cad/eef_mesh_visual_only.obj",
        "init_assets_json": "../init/assets.json",
        "init_depth_npy": "../init/depth/init_depth.npy",
        "init_image_png": "../init/img/init_env.png",
        "segmentation_instance_names_json": "../init/mask/instance_names.json",
        "segmentation_instance_npy": "../init/mask/init_segmentation_instance.npy",
    }
    return output


def _runner_task_runtime(
    init: EnvironmentSelection,
    *,
    bench_root: Path,
) -> dict[str, Any]:
    source = init.one("task_runtime")
    document = _json_object(source, "environment task runtime")
    simulator_assets = (bench_root / "sources" / "simulator-assets").resolve(
        strict=True
    )

    def materialize(value: Any) -> Any:
        if isinstance(value, Mapping):
            return {str(key): materialize(item) for key, item in value.items()}
        if isinstance(value, list):
            return [materialize(item) for item in value]
        if not isinstance(value, str) or "sources/simulator-assets/" not in value:
            return copy.deepcopy(value)
        if Path(value).is_absolute():
            raise ValueError(
                f"canonical task runtime contains an absolute asset path: {value}"
            )
        target = (source.parent / value).resolve(strict=True)
        try:
            target.relative_to(simulator_assets)
        except ValueError as error:
            raise ValueError(
                f"task-runtime asset escapes the simulator closure: {value}"
            ) from error
        if target.is_symlink() or not target.is_file():
            raise ValueError(f"task-runtime asset must be a regular file: {target}")
        return target.as_posix()

    output = materialize(document)
    if not isinstance(output, dict):  # pragma: no cover - input invariant
        raise TypeError("task runtime root must remain an object")
    return output


def _runner_init_assets(
    init: EnvironmentSelection,
    *,
    first_frame_sha1: str,
) -> dict[str, Any]:
    camera = dict(_environment_snapshot(init)["camera"])
    camera_name = str(camera.get("active_camera_name", "") or "").strip()
    if not camera_name:
        raise ValueError("environment snapshot has no active camera")

    return {
        "format": "dream-exe.init-assets",
        "camera_name": camera_name,
        "image": {
            "png": "img/init_env.png",
            "sha1": first_frame_sha1,
        },
        "depth": {
            "metric_npy": "depth/init_depth.npy",
            "metric_png": None,
            "buffer_npy": None,
            "buffer_png": None,
        },
        "segmentation": {
            "enabled": True,
            "camera_name": camera_name,
            "levels": {
                "instance": {
                    "npy": "mask/init_segmentation_instance.npy",
                    "png": None,
                    "bbox_json": None,
                    "source": "frozen_init",
                }
            },
            "instance_names_json": "mask/instance_names.json",
            "mapping_json": None,
        },
        "eef_cad": {
            "enabled": True,
            "geometry": "visual_only",
            "mesh_obj_path": "cad/eef_mesh_visual_only.obj",
            "meta_json_path": "cad/eef_mesh_visual_only_meta.json",
            "visual_only": {
                "mesh_obj_path": "cad/eef_mesh_visual_only.obj",
                "meta_json_path": "cad/eef_mesh_visual_only_meta.json",
            },
        },
    }


def _materialize_init_sample(
    *,
    repository: BenchRepository,
    uid: str,
    destination: Path,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(destination)
    destination.mkdir(parents=True)
    case = repository.load_case(uid)
    init = repository.load_environment(uid)
    generation = repository.load_generation_input(uid)
    write_json_atomic(destination / "meta.json", _runner_meta(case), exclusive=True)

    env_root = destination / "artifacts" / "env"
    materialize_frozen_scene(
        init.one("scene_model"),
        env_root / "scene" / "model.xml.gz",
        bench_root=repository.root,
    )
    _copy_file(init.one("state"), env_root / "state" / "states.npz")
    for role, filename in (
        ("environment", "environment.json"),
        ("runtime_lock", "runtime.lock.json"),
        ("origin", "origin.json"),
    ):
        _copy_file(init.one(role), env_root / filename)
    write_json_atomic(
        env_root / "task_runtime.json",
        _runner_task_runtime(init, bench_root=repository.root),
        exclusive=True,
    )
    write_json_atomic(
        env_root / "camera.json",
        _runner_camera(init),
        exclusive=True,
    )
    write_json_atomic(
        env_root / "entities.json",
        _runner_entities(init),
        exclusive=True,
    )
    write_json_atomic(
        env_root / "robosuite_config.json",
        _runner_simulator_config(init),
        exclusive=True,
    )

    for role, paths in init.files.items():
        if role in {
            "scene_model",
            "state",
            "environment",
            "entities",
            "task_runtime",
            "runtime_lock",
            "origin",
        }:
            continue
        for source in paths:
            relative = source.relative_to(repository.case_dir(uid) / "env")
            if role == "eef_geometry":
                parts = ("cad", "eef_mesh_visual_only.obj")
            elif role == "eef_geometry_meta":
                parts = ("cad", "eef_mesh_visual_only_meta.json")
            else:
                parts = (
                    relative.parts[1:]
                    if relative.parts[:1] == ("init",)
                    else relative.parts
                )
            _copy_file(source, destination / "artifacts" / "init" / Path(*parts))

    first_frame = repository.case_dir(uid) / "generation" / generation["first_frame"]["path"]
    _copy_file(first_frame, destination / "artifacts" / "init" / "img" / "init_env.png")
    first_frame_sha1 = hashlib.sha1(first_frame.read_bytes()).hexdigest()
    write_json_atomic(
        destination / "artifacts" / "init" / "assets.json",
        _runner_init_assets(init, first_frame_sha1=first_frame_sha1),
        exclusive=True,
    )
    environment = _json_object(init.one("environment"), "init environment")
    initialization = {
        "uid": uid,
        "mode": "saved_environment",
        "seed": int(environment.get("seed", 0) or 0),
        "render": copy.deepcopy(dict(environment.get("render", {}) or {})),
    }
    write_json_atomic(
        destination / "pipeline" / "initialization.json",
        initialization,
        exclusive=True,
    )
    return {"status": "materialized", "uid": uid}


def _add_reference_to_sample(
    *,
    repository: BenchRepository,
    uid: str,
    destination: Path,
) -> tuple[ReferenceSelection, dict[str, Any]]:
    reference = repository.load_reference(uid)
    target = destination / "artifacts" / "gt"
    _copy_file(reference.video, target / "video" / "gt.mp4")
    _copy_file(
        reference.action_array,
        target / "action" / "action.npy",
    )
    runtime_action_meta = _json_object(
        reference.action_meta,
        "canonical GT action metadata",
    )
    runtime_action_meta["array"]["path"] = "action.npy"
    write_json_atomic(
        target / "action" / "meta.json",
        runtime_action_meta,
        exclusive=True,
    )
    runtime_depth_names = {"gt_metric.npy": "gt_metric.npy", "meta.json": "meta.json"}
    for record, source in zip(
        reference.manifest["depth"]["files"],
        reference.depth,
        strict=True,
    ):
        _copy_file(
            source,
            target / "depth" / runtime_depth_names[Path(record["path"]).name],
        )

    action_npy = target / "action" / "action.npy"
    action_meta = target / "action" / "meta.json"
    _bundle_meta, action_array = load_action_bundle(
        action_npy,
        expected_uid=uid,
        expected_kind="ground_truth_demonstration",
    )
    action_json = write_json_atomic(
        target / "action" / "action.json",
        gt_action_payload_from_bundle(action_npy),
        exclusive=True,
    )
    camera = dict(
        _environment_snapshot(repository.load_environment(uid))["camera"]
    )
    camera_name = str(
        camera.get("active_camera_name", camera.get("name", "")) or ""
    ).strip()
    depth_names = {path.name: path for path in reference.depth}
    assets_manifest = {
        "format": "dream-exe.reference-assets",
        "action": {
            "json": "action/action.json",
            "npy": "action/action.npy",
            "meta_json": "action/meta.json",
        },
        "video": {
            "gt_mp4": "video/gt.mp4",
            "camera_name": camera_name,
        },
        "depth": {
            "metric_npy": "depth/gt_metric.npy",
            "meta_json": "depth/meta.json",
            "camera_name": camera_name,
        },
    }
    required_depth = {
        "gt_metric.npy",
        "meta.json",
    }
    if not required_depth.issubset(depth_names):
        raise ValueError(
            "canonical GT depth cannot satisfy the runtime reference manifest: "
            f"missing={sorted(required_depth - set(depth_names))}"
        )
    assets = write_json_atomic(
        target / "assets.json",
        assets_manifest,
        exclusive=True,
    )
    return reference, {
        "gt_action_materialization": {
            "array": _file_record(action_npy, destination),
            "meta": _file_record(action_meta, destination),
            "runtime_json": _file_record(action_json, destination),
            "dtype": action_array.dtype.str,
            "shape": list(action_array.shape),
        },
        "gt_assets_manifest": _file_record(assets, destination),
    }


def initialize_case(
    *,
    workspace_path: str | Path,
    uid: str,
    receipt_id: str,
    initializer: Callable[..., Mapping[str, Any]] | None = None,
    test_only_allow_injected_initializer: bool = False,
) -> dict[str, Any]:
    """Materialize the benchmark-defined saved init under ``work``.

    This command no longer imports the original RoboCasa dataset episode or
    reapplies a camera patch.  The benchmark's final scene, state, task runtime
    and camera snapshot are the initialization input.
    """

    workspace = load_workspace(workspace_path)
    repository = BenchRepository(workspace.bench_root)
    repository.load_case(uid)
    init = repository.load_environment(uid)
    if initializer is not None and not test_only_allow_injected_initializer:
        raise ValueError("initializer injection is test-only")
    receipt_name = str(receipt_id or "").strip()
    if not _SAFE_ID.fullmatch(receipt_name):
        raise ValueError("receipt_id must be a safe identifier")
    receipt_root = workspace.materialized_root / receipt_name
    if receipt_root.exists():
        raise FileExistsError(f"init receipt root already exists: {receipt_root}")
    sample_root = receipt_root / "cases" / uid
    materialized = _materialize_init_sample(
        repository=repository,
        uid=uid,
        destination=sample_root,
    )
    initializer_result = materialized
    if initializer is not None:
        initializer_result = dict(
            initializer(
                workspace=workspace,
                case=repository.load_case(uid),
                init=init.manifest,
                sample_root=sample_root,
            )
        )
    artifacts = _hash_tree(receipt_root, relative_base=receipt_root)
    receipt = {
        "format": INIT_RECEIPT_SCHEMA,
        "receipt_id": receipt_name,
        "uid": uid,
        "created_at": _now(),
        "status": "completed",
        "init_sha256": canonical_sha256(init.manifest),
        "sample_path": sample_root.relative_to(receipt_root).as_posix(),
        "artifacts": artifacts,
    }
    validate_document(receipt, expected_schema=INIT_RECEIPT_SCHEMA)
    receipt_path = write_json_atomic(
        receipt_root / "init-receipt.json",
        receipt,
        exclusive=True,
    )
    return {
        "status": "completed",
        "receipt": receipt,
        "receipt_path": receipt_path.as_posix(),
        "initializer_result": initializer_result,
    }


def _hash_tree(root: Path, *, relative_base: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise ValueError(f"artifact tree must not contain symlinks: {path}")
        if path.is_file():
            output.append(
                {
                    "path": path.relative_to(relative_base).as_posix(),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
    return output


__all__ = ["initialize_case"]
