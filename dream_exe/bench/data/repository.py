"""Read-only repositories for immutable benchmark inputs and output videos."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from ..contracts.action import load_action_bundle
from ..contracts.schemas import (
    BENCH_SCHEMA,
    CASE_PROTOCOL_ROUTES,
    CASE_PROTOCOL_SCHEMA,
    CASE_SCHEMA,
    COLLECTION_SCHEMA,
    ENVIRONMENT_SCHEMA,
    ENVIRONMENT_STATE_SCHEMA,
    GENERATION_INPUT_SCHEMA,
    PROTOCOL_SCHEMA,
    PROTOCOL_STAGES,
    REFERENCE_SCHEMA,
    SOURCE_SCHEMA,
    VIDEO_OUTPUT_SCHEMA,
    canonical_sha256,
    load_and_validate,
    load_json_strict,
)
from .portable import absolute_path_strings
from .scene import frozen_scene_references
from .workspace import Workspace


def load_and_validate_json_object(path: Path, label: str) -> dict[str, Any]:
    """Load opaque case metadata with the shared strict JSON parser."""

    try:
        return load_json_strict(path)
    except Exception as error:
        raise ValueError(f"invalid {label}: {path}") from error


def _safe_child(root: Path, *parts: str) -> Path:
    candidate = root.joinpath(*parts)
    resolved_root = root.resolve(strict=False)
    resolved = candidate.resolve(strict=False)
    try:
        resolved.relative_to(resolved_root)
    except ValueError as error:
        raise ValueError(f"path escapes repository root: {candidate}") from error
    return candidate


def _require_regular_file(path: Path, *, role: str) -> Path:
    if path.is_symlink():
        raise ValueError(f"canonical {role} must not be a symlink: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"canonical {role} not found: {path}")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_file_record(
    owner_root: Path,
    record: Mapping[str, Any],
    *,
    role: str,
    verify_digest: bool,
) -> Path:
    path = _require_regular_file(
        _safe_child(owner_root, str(record["path"])),
        role=role,
    )
    if verify_digest:
        if path.stat().st_size != record["size"]:
            raise ValueError(f"{role} size mismatch: {path}")
        if _sha256_file(path) != record["sha256"]:
            raise ValueError(f"{role} digest mismatch: {path}")
    return path


@dataclass(frozen=True)
class EnvironmentSelection:
    uid: str
    manifest: dict[str, Any]
    files: dict[str, tuple[Path, ...]]

    def one(self, role: str) -> Path:
        candidates = self.files.get(role, ())
        if len(candidates) != 1:
            raise ValueError(
                "environment role "
                f"{role!r} requires exactly one file, got {len(candidates)}"
            )
        return candidates[0]


@dataclass(frozen=True)
class ReferenceSelection:
    uid: str
    manifest: dict[str, Any]
    video: Path
    action_array: Path
    action_meta: Path
    depth: tuple[Path, ...]


@dataclass(frozen=True)
class VideoOutputSelection:
    uid: str
    model_id: str
    prompt_variant: str
    manifest: dict[str, Any]
    video: Path
    pipeline_input: Path


class BenchRepository:
    """Validate and resolve one immutable benchmark tree without writes."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root).expanduser().resolve()
        if not self.root.is_dir():
            raise FileNotFoundError(f"benchmark root not found: {self.root}")
        self.manifest = load_and_validate(
            _require_regular_file(self.root / "bench.json", role="bench manifest"),
            expected_schema=BENCH_SCHEMA,
        )

    def case_dir(self, uid: str) -> Path:
        return _safe_child(self.root, "cases", uid)

    def load_case(self, uid: str) -> dict[str, Any]:
        document = load_and_validate(
            _require_regular_file(self.case_dir(uid) / "case.json", role="case manifest"),
            expected_schema=CASE_SCHEMA,
        )
        if document["uid"] != uid:
            raise ValueError(f"case manifest UID mismatch for {uid!r}")
        return document

    def load_environment(
        self,
        uid: str,
        *,
        verify_files: bool = True,
    ) -> EnvironmentSelection:
        root = self.case_dir(uid) / "env"
        document = load_and_validate(
            _require_regular_file(root / "manifest.json", role="environment manifest"),
            expected_schema=ENVIRONMENT_SCHEMA,
        )
        if document["uid"] != uid:
            raise ValueError(f"environment manifest UID mismatch for {uid!r}")
        grouped: dict[str, list[Path]] = {}
        for record in document["files"]:
            grouped.setdefault(str(record["role"]), []).append(_verify_file_record(
                root,
                record,
                role=f"environment {record['role']}",
                verify_digest=verify_files,
            ))
        expected_paths = {
            "scene_model": "scene/model.xml.gz",
            "state": "init/state/states.npz",
            "environment": "environment.json",
            "entities": "entities.json",
            "task_runtime": "task_runtime.json",
            "runtime_lock": "runtime.lock.json",
            "origin": "origin.json",
            "init_depth": "init/depth/init_depth.npy",
            "instance_segmentation": "init/mask/init_segmentation_instance.npy",
            "instance_names": "init/mask/instance_names.json",
            "eef_geometry": "geometry/eef/visual.obj",
            "eef_geometry_meta": "geometry/eef/meta.json",
        }
        observed_paths = {
            role: paths[0].relative_to(root).as_posix()
            for role, paths in grouped.items()
            if len(paths) == 1
        }
        if observed_paths != expected_paths:
            raise ValueError(
                "environment manifest paths differ from the canonical asset set: "
                f"expected={expected_paths}, observed={observed_paths}"
            )

        environment = load_and_validate(
            grouped["environment"][0],
            expected_schema=ENVIRONMENT_STATE_SCHEMA,
        )
        if environment["uid"] != uid:
            raise ValueError(f"environment state UID mismatch for {uid!r}")
        camera = environment.get("camera")
        if not isinstance(camera, Mapping):
            raise ValueError("environment snapshot must contain camera")
        if not str(camera.get("active_camera_name", "") or "").strip():
            raise ValueError("environment camera must name the active camera")
        calibration = camera.get("calibration", {})
        camera_io = calibration.get("io", {}) if isinstance(calibration, Mapping) else {}
        if not isinstance(camera_io, Mapping) or camera_io.get("first_frame_path") != "../generation/first_frame.png":
            raise ValueError(
                "environment camera first_frame_path must be "
                "../generation/first_frame.png"
            )

        entities = load_and_validate_json_object(
            grouped["entities"][0],
            "environment entities",
        )
        entity_assets = entities.get("assets", {})
        expected_entity_assets = {
            "eef_geometry_meta_json": "geometry/eef/meta.json",
            "eef_geometry_obj": "geometry/eef/visual.obj",
            "init_depth_npy": "init/depth/init_depth.npy",
            "init_image_png": "../generation/first_frame.png",
            "segmentation_instance_names_json": "init/mask/instance_names.json",
            "segmentation_instance_npy": "init/mask/init_segmentation_instance.npy",
        }
        if entity_assets != expected_entity_assets:
            raise ValueError(
                "environment entities.assets differs from the canonical "
                "relative-path contract"
            )

        for json_path in sorted(root.rglob("*.json")):
            payload = load_and_validate_json_object(
                json_path,
                "canonical environment JSON",
            )
            leaks = list(absolute_path_strings(payload))
            if leaks:
                raise ValueError(
                    "canonical environment JSON contains absolute paths: "
                    f"{json_path}: {leaks[:3]}"
                )
        task_runtime = load_and_validate_json_object(
            grouped["task_runtime"][0],
            "environment task runtime",
        )
        simulator_assets = (self.root / "sources" / "simulator-assets").resolve(
            strict=True
        )

        def validate_task_assets(value: Any) -> None:
            if isinstance(value, Mapping):
                for item in value.values():
                    validate_task_assets(item)
                return
            if isinstance(value, list):
                for item in value:
                    validate_task_assets(item)
                return
            if not isinstance(value, str) or "sources/simulator-assets/" not in value:
                return
            if Path(value).is_absolute():
                raise ValueError("canonical task-runtime asset path must be relative")
            target = (grouped["task_runtime"][0].parent / value).resolve(strict=True)
            try:
                target.relative_to(simulator_assets)
            except ValueError as error:
                raise ValueError(
                    f"task-runtime asset escapes simulator closure: {value}"
                ) from error
            _require_regular_file(target, role="task-runtime simulator asset")

        validate_task_assets(task_runtime)
        scene_assets = frozen_scene_references(
            grouped["scene_model"][0],
            bench_root=self.root,
        )
        if not scene_assets:
            raise ValueError("canonical frozen scene contains no simulator assets")

        return EnvironmentSelection(
            uid=uid,
            manifest=document,
            files={role: tuple(paths) for role, paths in grouped.items()},
        )

    def load_generation_input(
        self,
        uid: str,
        *,
        verify_frame: bool = True,
    ) -> dict[str, Any]:
        root = self.case_dir(uid) / "generation"
        document = load_and_validate(
            _require_regular_file(root / "input.json", role="generation input"),
            expected_schema=GENERATION_INPUT_SCHEMA,
        )
        if document["uid"] != uid:
            raise ValueError(f"generation input UID mismatch for {uid!r}")
        _verify_file_record(
            root,
            document["first_frame"],
            role="generation first frame",
            verify_digest=verify_frame,
        )
        return document

    def load_vlm_prompt_metadata(
        self,
        uid: str,
        *,
        prompt_variant: str,
    ) -> dict[str, Any]:
        """Resolve one case's VLM prompt fields from canonical bench inputs."""

        if prompt_variant not in {"standard", "enhanced"}:
            raise ValueError(
                "VLM prompt_variant must be 'standard' or 'enhanced'"
            )
        case = self.load_case(uid)
        generation = self.load_generation_input(uid, verify_frame=False)
        metadata = dict(case["evaluation"]["vlm"])
        protocol = self.load_protocol("evaluation")["values"]["vlm"]
        suffix = str(protocol["task_prompt_suffixes"][prompt_variant]).strip()
        prompt = str(generation["prompts"][prompt_variant]).strip()
        if suffix:
            prompt = f"{prompt} {suffix}"
        return {
            "name": uid,
            "prompt": prompt,
            "view": metadata["view"],
            "robotic manipulator": metadata["robot_subject"],
            "manipulated object": metadata["manipulated_object"],
        }

    def load_vlm_evaluation_config(
        self,
        uid: str,
        *,
        prompt_variant: str,
    ) -> dict[str, Any]:
        """Load portable VLM rubrics, judge declarations, and case metadata."""

        protocol = self.load_protocol("evaluation")["values"]["vlm"]
        protocol_root = self.root / "protocol"

        def prompt_record(relative_path: str) -> dict[str, Any]:
            path = _require_regular_file(
                _safe_child(protocol_root, relative_path),
                role="VLM prompt template",
            )
            text = path.read_text(encoding="utf-8")
            if not text.strip():
                raise ValueError(f"VLM prompt template is empty: {path}")
            return {
                "path": path.relative_to(self.root).as_posix(),
                "sha256": _sha256_file(path),
                "text": text,
            }

        rubrics = dict(protocol["rubrics"])
        subject = dict(rubrics["subject_stability"])
        subject["prompts"] = {
            name: prompt_record(path)
            for name, path in subject.pop("prompt_files").items()
        }
        resolved_rubrics = {"subject_stability": subject}
        for name in ("physical_plausibility", "task_adherence"):
            rubric = dict(rubrics[name])
            rubric["prompt"] = prompt_record(rubric.pop("prompt_file"))
            resolved_rubrics[name] = rubric
        return {
            "prompt_metadata": self.load_vlm_prompt_metadata(
                uid,
                prompt_variant=prompt_variant,
            ),
            "rubrics": resolved_rubrics,
            "judges": [dict(judge) for judge in protocol["judges"]],
            "aggregation": dict(protocol["aggregation"]),
        }

    def load_protocol(self, stage: str) -> dict[str, Any]:
        if stage not in PROTOCOL_STAGES:
            raise ValueError(f"unknown protocol stage: {stage!r}")
        document = load_and_validate(
            _require_regular_file(
                _safe_child(self.root, "protocol", f"{stage}.json"),
                role=f"{stage} protocol",
            ),
            expected_schema=PROTOCOL_SCHEMA,
        )
        if document["stage"] != stage:
            raise ValueError(f"protocol stage mismatch for {stage!r}")
        return document

    def load_case_protocol(self, uid: str) -> dict[str, Any]:
        document = load_and_validate(
            _require_regular_file(
                self.case_dir(uid) / "protocol.json",
                role="case protocol",
            ),
            expected_schema=CASE_PROTOCOL_SCHEMA,
        )
        if document["uid"] != uid:
            raise ValueError(f"case protocol UID mismatch for {uid!r}")
        return document

    def load_case_protocol_values(
        self,
        uid: str,
        stage: str,
        *,
        route: str = "candidate",
    ) -> dict[str, Any]:
        if route not in CASE_PROTOCOL_ROUTES:
            raise ValueError(f"unknown case protocol route: {route!r}")
        if stage not in PROTOCOL_STAGES:
            raise ValueError(f"unknown protocol stage: {stage!r}")
        document = self.load_case_protocol(uid)
        return dict(document["routes"][route].get(stage, {}))

    def load_reference(
        self,
        uid: str,
        *,
        verify_files: bool = True,
    ) -> ReferenceSelection:
        root = self.case_dir(uid) / "references"
        document = load_and_validate(
            _require_regular_file(root / "reference.json", role="GT reference"),
            expected_schema=REFERENCE_SCHEMA,
        )
        if document["uid"] != uid:
            raise ValueError(f"reference UID mismatch for {uid!r}")
        video = _verify_file_record(
            root,
            document["video"],
            role="GT reference video",
            verify_digest=verify_files,
        )
        action_array = _verify_file_record(
            root,
            document["action"]["array"],
            role="GT reference action array",
            verify_digest=verify_files,
        )
        action_meta = _verify_file_record(
            root,
            document["action"]["meta"],
            role="GT reference action metadata",
            verify_digest=verify_files,
        )
        load_action_bundle(
            action_array,
            expected_uid=uid,
            expected_kind="ground_truth_demonstration",
        )
        depth = tuple(
            _verify_file_record(
                root,
                record,
                role="GT reference depth",
                verify_digest=verify_files,
            )
            for record in document["depth"]["files"]
        )
        return ReferenceSelection(
            uid=uid,
            manifest=document,
            video=video,
            action_array=action_array,
            action_meta=action_meta,
            depth=depth,
        )

    def load_source(self, source_id: str) -> dict[str, Any]:
        if source_id not in self.manifest["sources"]:
            raise KeyError(f"source is not declared by bench.json: {source_id!r}")
        document = load_and_validate(
            _require_regular_file(
                _safe_child(self.root, "sources", f"{source_id}.json"),
                role="source manifest",
            ),
            expected_schema=SOURCE_SCHEMA,
        )
        if document["source_id"] != source_id:
            raise ValueError(f"source manifest identity mismatch for {source_id!r}")
        return document

    def validate_workspace_bindings(
        self,
        workspace: Workspace,
        *,
        source_ids: list[str] | None = None,
    ) -> dict[str, Any]:
        selected = list(self.manifest["sources"] if source_ids is None else source_ids)
        bindings: list[dict[str, Any]] = []
        for source_id in selected:
            source = self.load_source(source_id)
            source_bindings: dict[str, tuple[Mapping[str, Any], Any]] = {}
            for requirement in source["bindings"]:
                try:
                    binding = workspace.binding(
                        requirement["kind"], requirement["name"]
                    )
                except KeyError:
                    if requirement["required"]:
                        raise
                    continue
                if not binding.path.exists():
                    if not requirement["required"]:
                        continue
                    raise FileNotFoundError(
                        "required workspace binding is missing: "
                        f"{requirement['kind']}/{requirement['name']}={binding.path}"
                    )
                expected = requirement["sha256"]
                if expected is not None:
                    if binding.manifest is None or not binding.manifest.is_file():
                        raise FileNotFoundError(
                            "digest-bound workspace source requires a manifest file: "
                            f"{requirement['kind']}/{requirement['name']}"
                        )
                    if _sha256_file(binding.manifest) != expected:
                        raise ValueError(
                            "workspace binding manifest digest mismatch: "
                            f"{requirement['kind']}/{requirement['name']}"
                        )
                source_bindings[str(requirement["name"])] = (requirement, binding)

            verified_by_binding: dict[str, dict[str, int]] = {}
            for file_record in source["files"]:
                if file_record["storage"] == "embedded":
                    path = _require_regular_file(
                        _safe_child(
                            self.root / "sources",
                            str(file_record["path"]),
                        ),
                        role=f"embedded source dependency {source_id}",
                    )
                    if path.stat().st_size != file_record["size"]:
                        raise ValueError(
                            "embedded source dependency size mismatch: "
                            f"{source_id}/{file_record['path']}"
                        )
                    if _sha256_file(path) != file_record["sha256"]:
                        raise ValueError(
                            "embedded source dependency digest mismatch: "
                            f"{source_id}/{file_record['path']}"
                        )
                    aggregate = verified_by_binding.setdefault(
                        "embedded", {"files": 0, "bytes": 0}
                    )
                    aggregate["files"] += 1
                    aggregate["bytes"] += int(file_record["size"])
                    continue
                name = str(file_record["binding_name"])
                if name not in source_bindings:
                    raise FileNotFoundError(
                        "source file belongs to an unavailable workspace binding: "
                        f"{file_record['binding_kind']}/{name}/{file_record['path']}"
                    )
                requirement, binding = source_bindings[name]
                if requirement["kind"] != file_record["binding_kind"]:
                    raise ValueError(
                        f"source file binding kind changed for {name!r}"
                    )
                path = _require_regular_file(
                    _safe_child(binding.path, str(file_record["path"])),
                    role=f"source dependency {source_id}/{name}",
                )
                if path.stat().st_size != file_record["size"]:
                    raise ValueError(
                        "workspace source dependency size mismatch: "
                        f"{source_id}/{name}/{file_record['path']}"
                    )
                if _sha256_file(path) != file_record["sha256"]:
                    raise ValueError(
                        "workspace source dependency digest mismatch: "
                        f"{source_id}/{name}/{file_record['path']}"
                    )
                aggregate = verified_by_binding.setdefault(
                    name, {"files": 0, "bytes": 0}
                )
                aggregate["files"] += 1
                aggregate["bytes"] += int(file_record["size"])

            for name, (requirement, binding) in sorted(source_bindings.items()):
                aggregate = verified_by_binding.get(name, {"files": 0, "bytes": 0})
                bindings.append(
                    {
                        "source_id": source_id,
                        "kind": requirement["kind"],
                        "name": name,
                        "binding_ref": f"workspace.bindings.{requirement['kind']}.{name}",
                        "manifest_verified": requirement["sha256"] is not None,
                        "verified_files": aggregate["files"],
                        "verified_bytes": aggregate["bytes"],
                    }
                )
            embedded = verified_by_binding.get("embedded")
            if embedded is not None:
                bindings.append(
                    {
                        "source_id": source_id,
                        "kind": "embedded",
                        "name": "simulator-assets",
                        "binding_ref": "bench.sources",
                        "manifest_verified": True,
                        "verified_files": embedded["files"],
                        "verified_bytes": embedded["bytes"],
                    }
                )
        return {"status": "valid", "bindings": bindings}

    def load_collection(
        self,
        collection_id: str = "",
        *,
        verify_digests: bool = True,
    ) -> dict[str, Any]:
        collection_id = str(collection_id or "").strip()
        if not collection_id:
            declared = [str(value) for value in self.manifest["collections"]]
            if len(declared) != 1:
                raise ValueError(
                    "collection must be specified when the bench declares "
                    f"{len(declared)} collections"
                )
            collection_id = declared[0]
        if collection_id not in self.manifest["collections"]:
            raise KeyError(f"collection is not declared by bench.json: {collection_id!r}")
        document = load_and_validate(
            _require_regular_file(
                _safe_child(self.root, "collections.json"),
                role="collection manifest",
            ),
            expected_schema=COLLECTION_SCHEMA,
        )
        if document["collection_id"] != collection_id:
            raise ValueError(f"collection identity mismatch for {collection_id!r}")
        if verify_digests:
            for member in document["cases"]:
                case = self.load_case(member["uid"])
                if canonical_sha256(case) != member["case_sha256"]:
                    raise ValueError(f"case digest mismatch for {member['uid']!r}")
        return document


class OutputRepository:
    """Read local videos, falling back to downloaded published candidates."""

    def __init__(
        self,
        root: str | Path,
        *,
        published_root: str | Path | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.published_root = (
            None
            if published_root is None
            else Path(published_root).expanduser().resolve()
        )

    def video_dir(self, uid: str, model_id: str, prompt_variant: str) -> Path:
        local = _safe_child(
            self.root,
            "videos",
            uid,
            model_id,
            prompt_variant,
        )
        if local.is_dir() or self.published_root is None:
            return local
        return _safe_child(
            self.published_root,
            "videos",
            uid,
            model_id,
            prompt_variant,
        )

    def load_video(
        self,
        uid: str,
        model_id: str,
        prompt_variant: str,
        *,
        verify_media: bool = True,
    ) -> VideoOutputSelection:
        root = self.video_dir(uid, model_id, prompt_variant)
        document = load_and_validate(
            _require_regular_file(root / "video.json", role="video output manifest"),
            expected_schema=VIDEO_OUTPUT_SCHEMA,
        )
        identity = (document["uid"], document["model_id"], document["prompt_variant"])
        if identity != (uid, model_id, prompt_variant):
            raise ValueError(f"video output identity mismatch for {uid}/{model_id}/{prompt_variant}")
        video = _verify_file_record(
            root,
            document["video"],
            role="generated video",
            verify_digest=verify_media,
        )
        pipeline_record = document["preprocessed"]
        pipeline_input = (
            video
            if pipeline_record is None
            else _verify_file_record(
                root,
                pipeline_record,
                role="video2traj input video",
                verify_digest=verify_media,
            )
        )
        return VideoOutputSelection(
            uid=uid,
            model_id=model_id,
            prompt_variant=prompt_variant,
            manifest=document,
            video=video,
            pipeline_input=pipeline_input,
        )


__all__ = [
    "BenchRepository",
    "EnvironmentSelection",
    "OutputRepository",
    "ReferenceSelection",
    "VideoOutputSelection",
]
