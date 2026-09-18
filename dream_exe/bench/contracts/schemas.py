"""Strict, dependency-free JSON contracts for the canonical benchmark.

Bench documents own immutable inputs.  Generated videos and pipeline results
use output schemas and are never accepted below the bench root.  Structural
objects reject unknown fields; algorithm payloads are deliberately isolated in
explicit ``values`` objects and receive stricter ownership checks when they are
composed by :mod:`dream_exe.bench.contracts.config`.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Callable


WORKSPACE_SCHEMA = "dream-exe.workspace"
BENCH_SCHEMA = "dream-exe.bench"
SOURCE_SCHEMA = "dream-exe.source"
CASE_SCHEMA = "dream-exe.case"
ENVIRONMENT_SCHEMA = "dream-exe.environment"
ENVIRONMENT_STATE_SCHEMA = "dream-exe.environment-state"
INIT_RECEIPT_SCHEMA = "dream-exe.init-receipt"
WORK_MATERIALIZATION_SCHEMA = "dream-exe.work-materialization"
GENERATION_INPUT_SCHEMA = "dream-exe.generation-input"
PROTOCOL_SCHEMA = "dream-exe.protocol"
CASE_PROTOCOL_SCHEMA = "dream-exe.case-protocol"
REFERENCE_SCHEMA = "dream-exe.reference"
VIDEO_OUTPUT_SCHEMA = "dream-exe.video-output"
COLLECTION_SCHEMA = "dream-exe.collection"
RUN_SCHEMA = "dream-exe.run"
RESULT_SCHEMA = "dream-exe.result"
RUN_SUMMARY_SCHEMA = "dream-exe.run-summary"
RESULT_REQUEST_SCHEMA = "dream-exe.result-request"
RESOLVED_CONFIG_SCHEMA = "dream-exe.resolved-config"

MAX_JSON_BYTES = 16 * 1024 * 1024
PROTOCOL_STAGES = (
    "generation",
    "video2traj",
    "action",
    "execution",
    "evaluation",
)
PIPELINE_STAGES = ("video2traj", "action", "execution", "evaluation")
CASE_PROTOCOL_ROUTES = (
    "candidate",
    "reference_input",
    "evaluation_oracle",
)
_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_JSON_POINTER = re.compile(r"(?:/(?:[^~/]|~0|~1)*)+\Z")


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _walk(value: Any):
    if isinstance(value, Mapping):
        for item in value.values():
            yield from _walk(item)
    elif isinstance(value, list):
        for item in value:
            yield from _walk(item)
    else:
        yield value


def load_json_value_strict(
    path: str | Path,
    *,
    max_bytes: int = MAX_JSON_BYTES,
) -> Any:
    """Load one UTF-8 JSON value, rejecting duplicate keys and NaN/Inf."""

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"JSON document not found: {source}")
    encoded = source.read_bytes()
    if len(encoded) > max_bytes:
        raise ValueError(f"JSON document exceeds {max_bytes} bytes: {source}")
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"JSON document must be UTF-8: {source}") from error
    if any(
        isinstance(value, float) and not math.isfinite(value)
        for value in _walk(payload)
    ):
        raise ValueError(f"JSON document contains a non-finite number: {source}")
    return payload


def load_json_strict(
    path: str | Path,
    *,
    max_bytes: int = MAX_JSON_BYTES,
) -> dict[str, Any]:
    """Load one UTF-8 JSON object, rejecting duplicate keys and NaN/Inf."""

    source = Path(path).expanduser().resolve()
    payload = load_json_value_strict(source, max_bytes=max_bytes)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON document must be an object: {source}")
    return payload


def canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def canonical_sha256(value: Any) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def _object(value: Any, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    return value


def _exact(
    value: Any,
    fields: set[str] | frozenset[str],
    label: str,
) -> Mapping[str, Any]:
    obj = _object(value, label)
    missing = sorted(set(fields) - set(obj))
    unknown = sorted(set(obj) - set(fields))
    if missing:
        raise ValueError(f"{label} is missing fields: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"{label} has unknown fields: {', '.join(unknown)}")
    return obj


def _safe_id(value: Any, label: str) -> str:
    if not isinstance(value, str) or not _SAFE_ID.fullmatch(value):
        raise ValueError(f"{label} must match {_SAFE_ID.pattern!r}")
    return value


def _text(value: Any, label: str, *, empty: bool = False) -> str:
    if not isinstance(value, str) or (not empty and not value.strip()):
        qualifier = "a string" if empty else "a non-empty string"
        raise ValueError(f"{label} must be {qualifier}")
    return value


def _bool(value: Any, label: str) -> bool:
    if not isinstance(value, bool):
        raise TypeError(f"{label} must be a boolean")
    return value


def _int(value: Any, label: str, *, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise TypeError(f"{label} must be an integer")
    if minimum is not None and value < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    return value


def _digest(value: Any, label: str, *, nullable: bool = False) -> str | None:
    if value is None and nullable:
        return None
    if not isinstance(value, str) or not _SHA256.fullmatch(value):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return value


def _list(value: Any, label: str, *, nonempty: bool = False) -> list[Any]:
    if not isinstance(value, list):
        raise TypeError(f"{label} must be a list")
    if nonempty and not value:
        raise ValueError(f"{label} must not be empty")
    return value


def _enum(value: Any, choices: set[str], label: str) -> str:
    text = _text(value, label)
    if text not in choices:
        raise ValueError(f"{label} must be one of: {', '.join(sorted(choices))}")
    return text


def _relative_path(value: Any, label: str) -> str:
    text = _text(value, label)
    path = Path(text)
    if path.is_absolute() or ".." in path.parts or path.as_posix() in {".", ""}:
        raise ValueError(f"{label} must be a contained relative path")
    return path.as_posix()


def _file_record(value: Any, label: str) -> Mapping[str, Any]:
    item = _exact(value, {"path", "sha256", "size"}, label)
    _relative_path(item["path"], f"{label}.path")
    _digest(item["sha256"], f"{label}.sha256")
    _int(item["size"], f"{label}.size", minimum=0)
    return item


def _file_list(value: Any, label: str, *, nonempty: bool = False) -> None:
    seen: set[str] = set()
    for index, raw in enumerate(_list(value, label, nonempty=nonempty)):
        item = _file_record(raw, f"{label}[{index}]")
        path = str(item["path"])
        if path in seen:
            raise ValueError(f"{label} contains duplicate path {path!r}")
        seen.add(path)


def _validate_rights(value: Any, label: str) -> None:
    rights = _exact(value, {"status", "license", "redistributable"}, label)
    _enum(rights["status"], {"known", "unknown", "restricted"}, f"{label}.status")
    _text(rights["license"], f"{label}.license", empty=True)
    if rights["redistributable"] is not None:
        _bool(rights["redistributable"], f"{label}.redistributable")
    if rights["status"] != "known" and rights["redistributable"] is True:
        raise ValueError(f"{label} cannot be redistributable unless status is known")


def _validate_binding(value: Any, label: str, *, source: bool) -> None:
    fields = {"path", "manifest", "setup_path"} if source else {"path", "manifest"}
    obj = _exact(value, fields, label)
    _text(obj["path"], f"{label}.path")
    if source:
        _text(obj["setup_path"], f"{label}.setup_path")
    if obj["manifest"] is not None:
        _text(obj["manifest"], f"{label}.manifest")


def _validate_workspace(doc: Mapping[str, Any]) -> None:
    root = _exact(doc, {"format", "roots", "bindings"}, "workspace")
    roots = _exact(
        root["roots"],
        {
            "bench",
            "published_results",
            "outputs",
            "work",
            "archive",
            "external",
            "checkpoints",
        },
        "workspace.roots",
    )
    for name, value in roots.items():
        _text(value, f"workspace.roots.{name}")
    bindings = _exact(
        root["bindings"],
        {"datasets", "sources", "checkpoints"},
        "workspace.bindings",
    )
    for kind, entries in bindings.items():
        mapping = _object(entries, f"workspace.bindings.{kind}")
        for name, value in mapping.items():
            _safe_id(name, f"workspace.bindings.{kind} key")
            _validate_binding(
                value,
                f"workspace.bindings.{kind}.{name}",
                source=kind == "sources",
            )


def _validate_bench(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {
            "format",
            "bench_id",
            "created_at",
            "collections",
            "sources",
            "protocol_stages",
        },
        "bench",
    )
    _safe_id(root["bench_id"], "bench.bench_id")
    _text(root["created_at"], "bench.created_at")
    for label in ("collections", "sources"):
        values = _list(root[label], f"bench.{label}")
        cleaned = [_safe_id(value, f"bench.{label}") for value in values]
        if len(set(cleaned)) != len(cleaned):
            raise ValueError(f"bench.{label} contains duplicate values")
    stages = [_safe_id(value, "bench.protocol_stages") for value in _list(root["protocol_stages"], "bench.protocol_stages")]
    if stages != list(PROTOCOL_STAGES):
        raise ValueError(
            "bench.protocol_stages must list the canonical stages in order: "
            + ", ".join(PROTOCOL_STAGES)
        )


def _validate_source(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {
            "format",
            "source_id",
            "kind",
            "identity",
            "rights",
            "bindings",
            "files",
            "requirements",
            "provenance_status",
        },
        "source",
    )
    _safe_id(root["source_id"], "source.source_id")
    _enum(root["kind"], {"dataset", "model", "runtime", "external", "unknown"}, "source.kind")
    identity = _exact(root["identity"], {"name", "revision", "sha256"}, "source.identity")
    _text(identity["name"], "source.identity.name")
    _text(identity["revision"], "source.identity.revision", empty=True)
    _digest(identity["sha256"], "source.identity.sha256", nullable=True)
    _validate_rights(root["rights"], "source.rights")
    binding_owners: set[tuple[str, str]] = set()
    for index, raw in enumerate(_list(root["bindings"], "source.bindings")):
        item = _exact(raw, {"kind", "name", "required", "sha256"}, f"source.bindings[{index}]")
        kind = _enum(item["kind"], {"datasets", "sources", "checkpoints"}, f"source.bindings[{index}].kind")
        name = _safe_id(item["name"], f"source.bindings[{index}].name")
        owner = (kind, name)
        if owner in binding_owners:
            raise ValueError(f"source.bindings contains duplicate owner {owner}")
        binding_owners.add(owner)
        _bool(item["required"], f"source.bindings[{index}].required")
        _digest(item["sha256"], f"source.bindings[{index}].sha256", nullable=True)
    seen_files: set[tuple[str, str, str, str]] = set()
    for index, raw in enumerate(_list(root["files"], "source.files")):
        item = _exact(
            raw,
            {
                "storage",
                "binding_kind",
                "binding_name",
                "path",
                "sha256",
                "size",
            },
            f"source.files[{index}]",
        )
        storage = _enum(
            item["storage"],
            {"embedded", "binding"},
            f"source.files[{index}].storage",
        )
        if storage == "binding":
            binding_kind = _enum(
                item["binding_kind"],
                {"datasets", "sources", "checkpoints"},
                f"source.files[{index}].binding_kind",
            )
            binding_name = _safe_id(
                item["binding_name"], f"source.files[{index}].binding_name"
            )
            if (binding_kind, binding_name) not in binding_owners:
                raise ValueError(
                    "source.files refers to undeclared binding "
                    f"{binding_kind}/{binding_name}"
                )
        else:
            if item["binding_kind"] is not None or item["binding_name"] is not None:
                raise ValueError(
                    "embedded source.files entries forbid binding_kind/binding_name"
                )
            binding_kind = ""
            binding_name = ""
        path = _relative_path(item["path"], f"source.files[{index}].path")
        identity = (storage, binding_kind, binding_name, path)
        if identity in seen_files:
            raise ValueError(f"source.files contains duplicate file {identity!r}")
        seen_files.add(identity)
        _digest(item["sha256"], f"source.files[{index}].sha256")
        _int(item["size"], f"source.files[{index}].size", minimum=0)
    for index, value in enumerate(_list(root["requirements"], "source.requirements")):
        _text(value, f"source.requirements[{index}]")
    _enum(root["provenance_status"], {"complete", "reconstructed", "unknown"}, "source.provenance_status")


def _validate_case(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {"format", "uid", "instruction", "task", "stages", "evaluation"},
        "case",
    )
    _safe_id(root["uid"], "case.uid")
    _text(root["instruction"], "case.instruction", empty=True)
    _object(root["task"], "case.task")
    for index, stage in enumerate(_list(root["stages"], "case.stages")):
        _object(stage, f"case.stages[{index}]")
    evaluation = _exact(root["evaluation"], {"vlm"}, "case.evaluation")
    vlm = _exact(
        evaluation["vlm"],
        {"view", "robot_subject", "manipulated_object"},
        "case.evaluation.vlm",
    )
    _text(vlm["view"], "case.evaluation.vlm.view")
    _text(vlm["robot_subject"], "case.evaluation.vlm.robot_subject")
    manipulated_object = vlm["manipulated_object"]
    if manipulated_object is not None:
        _text(manipulated_object, "case.evaluation.vlm.manipulated_object")


def _validate_environment(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {"format", "uid", "mode", "files"},
        "environment",
    )
    _safe_id(root["uid"], "environment.uid")
    _enum(root["mode"], {"saved_environment"}, "environment.mode")
    required = {
        "scene_model",
        "state",
        "environment",
        "entities",
        "task_runtime",
        "runtime_lock",
        "origin",
        "init_depth",
        "instance_segmentation",
        "instance_names",
        "eef_geometry",
        "eef_geometry_meta",
    }
    optional: set[str] = set()
    seen_required: set[str] = set()
    seen_paths: set[str] = set()
    for index, raw in enumerate(
        _list(root["files"], "environment.files", nonempty=True)
    ):
        item = _exact(
            raw,
            {"role", "path", "sha256", "size"},
            f"environment.files[{index}]",
        )
        role = _enum(
            item["role"],
            required | optional,
            f"environment.files[{index}].role",
        )
        if role in required:
            if role in seen_required:
                raise ValueError(
                    f"environment.files contains duplicate core role {role!r}"
                )
            seen_required.add(role)
        path = _relative_path(
            item["path"],
            f"environment.files[{index}].path",
        )
        if path in seen_paths:
            raise ValueError(f"environment.files contains duplicate path {path!r}")
        seen_paths.add(path)
        _digest(item["sha256"], f"environment.files[{index}].sha256")
        _int(item["size"], f"environment.files[{index}].size", minimum=0)
    if seen_required != required:
        raise ValueError(
            "environment.files roles differ: "
            f"missing={sorted(required - seen_required)}"
        )




def _validate_environment_state(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {"format", "uid", "raw", "camera"},
        "environment state",
    )
    _safe_id(root["uid"], "environment state.uid")
    _object(root["raw"], "environment state.raw")
    camera = _object(root["camera"], "environment state.camera")
    _text(
        camera.get("active_camera_name"),
        "environment state.camera.active_camera_name",
    )
    _object(camera.get("cam_configs"), "environment state.camera.cam_configs")
    _object(camera.get("calibration"), "environment state.camera.calibration")


def _validate_init_receipt(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {"format", "receipt_id", "uid", "created_at", "status", "init_sha256", "sample_path", "artifacts"},
        "init receipt",
    )
    _safe_id(root["receipt_id"], "init receipt.receipt_id")
    _safe_id(root["uid"], "init receipt.uid")
    _text(root["created_at"], "init receipt.created_at")
    _enum(root["status"], {"completed"}, "init receipt.status")
    _digest(root["init_sha256"], "init receipt.init_sha256")
    _relative_path(root["sample_path"], "init receipt.sample_path")
    _file_list(root["artifacts"], "init receipt.artifacts", nonempty=True)


def _validate_generation_input(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {"format", "uid", "first_frame", "prompts", "variants", "output_contract"},
        "generation input",
    )
    _safe_id(root["uid"], "generation input.uid")
    first_frame = _file_record(root["first_frame"], "generation input.first_frame")
    if first_frame["path"] != "first_frame.png":
        raise ValueError("generation input first frame must be generation/first_frame.png")
    prompts = _exact(
        root["prompts"],
        {"standard", "enhanced", "suffix"},
        "generation input.prompts",
    )
    for name, value in prompts.items():
        _text(value, f"generation input.prompts.{name}", empty=True)
    variants = _object(root["variants"], "generation input.variants")
    if set(variants) != {"standard", "enhanced"}:
        raise ValueError(
            "generation input.variants must contain standard and enhanced"
        )
    for name, raw in variants.items():
        components = _list(raw, f"generation input.variants.{name}", nonempty=True)
        for index, component in enumerate(components):
            _enum(component, set(prompts), f"generation input.variants.{name}[{index}]")
        if len(set(components)) != len(components):
            raise ValueError(f"generation input.variants.{name} contains duplicates")
    _object(root["output_contract"], "generation input.output_contract")


def _validate_work_materialization(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {
            "format",
            "uid",
            "initialization",
            "sources",
            "derivatives",
            "materialized_files",
        },
        "work materialization",
    )
    _safe_id(root["uid"], "work materialization.uid")
    initialization = _exact(
        root["initialization"],
        {"mode", "receipt_sha256"},
        "work materialization.initialization",
    )
    mode = _enum(
        initialization["mode"],
        {"frozen", "receipt"},
        "work materialization.initialization.mode",
    )
    receipt = _digest(
        initialization["receipt_sha256"],
        "work materialization.initialization.receipt_sha256",
        nullable=True,
    )
    if mode == "receipt" and receipt is None:
        raise ValueError("receipt work materialization requires receipt_sha256")
    if mode == "frozen" and receipt is not None:
        raise ValueError("frozen work materialization forbids receipt_sha256")
    _digest_map(
        root["sources"],
        "work materialization.sources",
        {"case", "init", "generation_input", "reference"},
    )
    derivatives = _exact(
        root["derivatives"],
        {"gt_action_materialization", "gt_assets_manifest"},
        "work materialization.derivatives",
    )
    action = _exact(
        derivatives["gt_action_materialization"],
        {"array", "meta", "runtime_json", "dtype", "shape"},
        "work materialization.derivatives.gt_action_materialization",
    )
    action_array = _file_record(
        action["array"],
        "work materialization.derivatives.gt_action_materialization.array",
    )
    if action_array["path"] != "artifacts/gt/action/action.npy":
        raise ValueError("work GT action array must be action/action.npy")
    action_meta = _file_record(
        action["meta"],
        "work materialization.derivatives.gt_action_materialization.meta",
    )
    if action_meta["path"] != "artifacts/gt/action/meta.json":
        raise ValueError("work GT action metadata must be action/meta.json")
    runtime_json = _file_record(
        action["runtime_json"],
        "work materialization.derivatives.gt_action_materialization.runtime_json",
    )
    if runtime_json["path"] != "artifacts/gt/action/action.json":
        raise ValueError("work GT runtime action must be action/action.json")
    if action["dtype"] != "<f4":
        raise ValueError("work GT action array dtype must be little-endian float32")
    shape = _list(
        action["shape"],
        "work materialization.derivatives.gt_action_array.shape",
        nonempty=True,
    )
    if len(shape) != 2:
        raise ValueError("work GT action array shape must have rank 2")
    for index, dimension in enumerate(shape):
        _int(
            dimension,
            f"work materialization.derivatives.gt_action_array.shape[{index}]",
            minimum=1,
        )
    assets = _file_record(
        derivatives["gt_assets_manifest"],
        "work materialization.derivatives.gt_assets_manifest",
    )
    if assets["path"] != "artifacts/gt/assets.json":
        raise ValueError("work GT assets manifest must be artifacts/gt/assets.json")
    _file_list(
        root["materialized_files"],
        "work materialization.materialized_files",
        nonempty=True,
    )


def _validate_protocol(doc: Mapping[str, Any]) -> None:
    root = _exact(doc, {"format", "stage", "values"}, "protocol")
    stage = _enum(root["stage"], set(PROTOCOL_STAGES), "protocol.stage")
    values = _object(root["values"], "protocol.values")
    if stage == "generation":
        generation = _exact(
            values,
            {"first_frame_required", "kind", "prompt_variants"},
            "generation protocol.values",
        )
        _bool(
            generation["first_frame_required"],
            "generation protocol.values.first_frame_required",
        )
        _enum(
            generation["kind"],
            {"image_to_video"},
            "generation protocol.values.kind",
        )
        variants = _list(
            generation["prompt_variants"],
            "generation protocol.values.prompt_variants",
            nonempty=True,
        )
        if variants != ["standard", "enhanced"]:
            raise ValueError(
                "generation protocol prompt_variants must be "
                "['standard', 'enhanced']"
            )
    elif stage == "evaluation":
        evaluation = _exact(
            values,
            {"vlm", "trajectory", "execution"},
            "evaluation protocol.values",
        )
        vlm = _exact(
            evaluation["vlm"],
            {"task_prompt_suffixes", "rubrics", "judges", "aggregation"},
            "evaluation protocol.values.vlm",
        )
        task_prompt_suffixes = _exact(
            vlm["task_prompt_suffixes"],
            {"standard", "enhanced"},
            "evaluation protocol.values.vlm.task_prompt_suffixes",
        )
        for name, suffix in task_prompt_suffixes.items():
            _text(
                suffix,
                f"evaluation protocol.values.vlm.task_prompt_suffixes.{name}",
                empty=True,
            )
        rubrics = _exact(
            vlm["rubrics"],
            {"subject_stability", "physical_plausibility", "task_adherence"},
            "evaluation protocol.values.vlm.rubrics",
        )
        subject = _exact(
            rubrics["subject_stability"],
            {"prompt_files", "sampling", "score_range"},
            "subject_stability rubric",
        )
        subject_prompts = _exact(
            subject["prompt_files"],
            {"robot_subject", "manipulated_object"},
            "subject_stability prompt_files",
        )
        for name, path in subject_prompts.items():
            _relative_path(path, f"subject_stability prompt_files.{name}")
        subject_sampling = _exact(
            subject["sampling"],
            {"frame_fractions", "grid"},
            "subject_stability sampling",
        )
        if subject_sampling["frame_fractions"] != [0.0, 0.75]:
            raise ValueError("subject_stability frame_fractions must be [0.0, 0.75]")
        if subject_sampling["grid"] != [1, 2] or subject["score_range"] != [1, 15]:
            raise ValueError("subject_stability grid or score range is invalid")
        for rubric_name in ("physical_plausibility", "task_adherence"):
            rubric = _exact(
                rubrics[rubric_name],
                {"prompt_file", "sampling", "score_range"},
                f"{rubric_name} rubric",
            )
            _relative_path(rubric["prompt_file"], f"{rubric_name} prompt_file")
            sampling = _exact(
                rubric["sampling"],
                {"uniform_frames", "grid"},
                f"{rubric_name} sampling",
            )
            if sampling["uniform_frames"] != 6 or sampling["grid"] != [3, 2]:
                raise ValueError(f"{rubric_name} sampling must use a 3x2 six-frame grid")
            if rubric["score_range"] != [1, 5]:
                raise ValueError(f"{rubric_name} score_range must be [1, 5]")
        judges = _list(vlm["judges"], "evaluation VLM judges", nonempty=True)
        judge_ids: list[str] = []
        for index, raw_judge in enumerate(judges):
            judge = _exact(
                raw_judge,
                {"id", "provider", "model", "base_url_env", "api_key_env"},
                f"evaluation VLM judges[{index}]",
            )
            judge_ids.append(_safe_id(judge["id"], f"VLM judge {index}.id"))
            _enum(
                judge["provider"],
                {"openai-compatible"},
                f"VLM judge {index}.provider",
            )
            for field in ("model", "base_url_env", "api_key_env"):
                _text(judge[field], f"VLM judge {index}.{field}")
        if len(set(judge_ids)) != len(judge_ids):
            raise ValueError("evaluation VLM judges contain duplicate ids")
        aggregation = _exact(
            vlm["aggregation"],
            {"method", "required_judges"},
            "evaluation VLM aggregation",
        )
        _enum(aggregation["method"], {"mean"}, "evaluation VLM aggregation.method")
        if aggregation["required_judges"] != judge_ids:
            raise ValueError("evaluation VLM aggregation must require every judge in order")
        trajectory = _exact(
            evaluation["trajectory"],
            {"groups", "metrics"},
            "evaluation trajectory protocol",
        )
        if trajectory["groups"] != ["EEF vis", "EEF tcp", "OBJ"]:
            raise ValueError("evaluation trajectory groups do not match the paper")
        if trajectory["metrics"] != ["HSD", "DYN", "NDTW"]:
            raise ValueError("evaluation trajectory metrics do not match the paper")
        execution = _exact(
            evaluation["execution"],
            {"executability_metrics", "task_metrics"},
            "evaluation execution protocol",
        )
        if execution["executability_metrics"] != [
            "E-SR", "nDTW", "Pos95", "Rot95", "Smth"
        ]:
            raise ValueError("evaluation executability metrics do not match the paper")
        if execution["task_metrics"] != [
            "SR-B", "SR-P", "Rel", "Place", "Art", "Core"
        ]:
            raise ValueError("evaluation task metrics do not match the paper")


def _validate_case_protocol(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {"format", "uid", "routes"},
        "case protocol",
    )
    _safe_id(root["uid"], "case protocol.uid")
    routes = _exact(
        root["routes"],
        set(CASE_PROTOCOL_ROUTES),
        "case protocol.routes",
    )
    for route, raw in routes.items():
        stages = _object(raw, f"case protocol.routes.{route}")
        unknown = sorted(set(stages) - set(PIPELINE_STAGES))
        if unknown:
            raise ValueError(
                f"case protocol route {route!r} has unknown stages: {unknown}"
            )
        for stage, values in stages.items():
            _object(values, f"case protocol.routes.{route}.{stage}")


def _validate_reference(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {
            "format",
            "uid",
            "video",
            "action",
            "depth",
            "rights",
            "origin",
        },
        "reference",
    )
    _safe_id(root["uid"], "reference.uid")
    video = _file_record(root["video"], "reference.video")
    if str(video["path"]) != "video/gt.mp4":
        raise ValueError("reference.video must be the fixed video/gt.mp4")
    action = _exact(root["action"], {"array", "meta"}, "reference.action")
    action_array = _file_record(action["array"], "reference.action.array")
    action_meta = _file_record(action["meta"], "reference.action.meta")
    if str(action_array["path"]) != "action/gt.npy":
        raise ValueError("reference.action.array must be action/gt.npy")
    if str(action_meta["path"]) != "action/meta.json":
        raise ValueError("reference.action.meta must be action/meta.json")
    depth = _exact(root["depth"], {"kind", "files"}, "reference.depth")
    _enum(depth["kind"], {"ground_truth"}, "reference.depth.kind")
    depth_files = _list(depth["files"], "reference.depth.files", nonempty=True)
    _file_list(depth_files, "reference.depth.files", nonempty=True)
    expected_depth = {
        "depth/meta.json",
        "depth/gt_metric.npy",
    }
    observed_depth = {str(item["path"]) for item in depth_files}
    if observed_depth != expected_depth:
        raise ValueError(
            "reference.depth.files must contain the complete fixed GT depth "
            f"bundle; missing={sorted(expected_depth - observed_depth)}, "
            f"unknown={sorted(observed_depth - expected_depth)}"
        )
    for index, item in enumerate(depth_files):
        if Path(str(item["path"])).parts[:1] != ("depth",):
            raise ValueError(
                "reference.depth.files"
                f"[{index}] must live below references/depth/"
            )
    _validate_rights(root["rights"], "reference.rights")
    _object(root["origin"], "reference.origin")


def _validate_video_output(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {
            "format",
            "uid",
            "model_id",
            "prompt_variant",
            "producer",
            "generation_input_sha256",
            "prompt_sha256",
            "video",
            "preprocessed",
            "rights",
            "provenance_status",
            "origin",
        },
        "video output",
    )
    _safe_id(root["uid"], "video output.uid")
    _safe_id(root["model_id"], "video output.model_id")
    _enum(
        root["prompt_variant"],
        {"standard", "enhanced", "custom"},
        "video output.prompt_variant",
    )
    producer = _exact(root["producer"], {"kind", "name", "revision", "seed"}, "video output.producer")
    _enum(producer["kind"], {"generator", "policy_rollout", "user", "unknown"}, "video output.producer.kind")
    _text(producer["name"], "video output.producer.name", empty=True)
    _text(producer["revision"], "video output.producer.revision", empty=True)
    if producer["seed"] is not None:
        _int(producer["seed"], "video output.producer.seed", minimum=0)
    _digest(root["generation_input_sha256"], "video output.generation_input_sha256", nullable=True)
    _digest(root["prompt_sha256"], "video output.prompt_sha256", nullable=True)
    video = _file_record(root["video"], "video output.video")
    if video["path"] != "video.mp4":
        raise ValueError("video output must be named video.mp4")
    if root["preprocessed"] is not None:
        pipeline = _file_record(
            root["preprocessed"],
            "video output.preprocessed",
        )
        if pipeline["path"] != "preprocessed.mp4":
            raise ValueError(
                "derived video2traj input must be named preprocessed.mp4"
            )
    _validate_rights(root["rights"], "video output.rights")
    _enum(root["provenance_status"], {"complete", "reconstructed", "unknown"}, "video output.provenance_status")
    _object(root["origin"], "video output.origin")


def _validate_collection(doc: Mapping[str, Any]) -> None:
    root = _exact(doc, {"format", "collection_id", "description", "cases"}, "collection")
    _safe_id(root["collection_id"], "collection.collection_id")
    _text(root["description"], "collection.description", empty=True)
    seen: set[str] = set()
    for index, raw in enumerate(_list(root["cases"], "collection.cases", nonempty=True)):
        item = _exact(raw, {"uid", "case_sha256"}, f"collection.cases[{index}]")
        uid = _safe_id(item["uid"], f"collection.cases[{index}].uid")
        if uid in seen:
            raise ValueError(f"collection has duplicate UID {uid!r}")
        seen.add(uid)
        _digest(item["case_sha256"], f"collection.cases[{index}].case_sha256")


def _validate_input_identity(value: Any, label: str, *, require_digest: bool) -> None:
    fields = {"kind", "model_id", "prompt_variant", "reference_id"}
    if require_digest:
        fields.add("video_sha256")
    item = _exact(value, fields, label)
    kind = _enum(item["kind"], {"reference", "generated"}, f"{label}.kind")
    if kind == "reference":
        if item["model_id"] is not None or item["prompt_variant"] is not None:
            raise ValueError(f"{label} reference identity forbids model fields")
        _enum(
            item["reference_id"],
            {"w_gt_depth", "wo_gt_depth"},
            f"{label}.reference_id",
        )
    else:
        _safe_id(item["model_id"], f"{label}.model_id")
        _enum(
            item["prompt_variant"],
            {"standard", "enhanced", "custom"},
            f"{label}.prompt_variant",
        )
        if item["reference_id"] is not None:
            raise ValueError(f"{label} generated identity forbids reference_id")
    if require_digest:
        _digest(item["video_sha256"], f"{label}.video_sha256")


def input_identity_key(value: Mapping[str, Any]) -> tuple[str, str, str]:
    return (
        str(value["kind"]),
        str(value.get("reference_id") or value.get("model_id") or ""),
        str(value.get("prompt_variant") or ""),
    )


def _validate_run(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {
            "format",
            "selection",
            "inputs",
            "stages",
            "seed",
            "initialization",
            "resume",
            "retry",
            "artifact_level",
            "destination",
        },
        "run",
    )
    selection = _exact(root["selection"], {"collection", "cases"}, "run.selection")
    if selection["collection"] is not None:
        _safe_id(selection["collection"], "run.selection.collection")
    cases = [_safe_id(value, "run.selection.cases") for value in _list(selection["cases"], "run.selection.cases")]
    if len(set(cases)) != len(cases):
        raise ValueError("run.selection.cases contains duplicate UIDs")
    if selection["collection"] is None and not cases:
        raise ValueError("run selection requires a collection or explicit cases")
    seen_inputs: set[tuple[str, str, str]] = set()
    for index, item in enumerate(_list(root["inputs"], "run.inputs", nonempty=True)):
        _validate_input_identity(item, f"run.inputs[{index}]", require_digest=False)
        key = input_identity_key(item)
        if key in seen_inputs:
            raise ValueError(f"run.inputs contains duplicate identity {key}")
        seen_inputs.add(key)
    stages = [_enum(value, set(PIPELINE_STAGES) | {"vlm_evaluation"}, "run.stages") for value in _list(root["stages"], "run.stages", nonempty=True)]
    order = [*PIPELINE_STAGES, "vlm_evaluation"]
    if stages != sorted(set(stages), key=order.index):
        raise ValueError("run.stages must be unique and in canonical order")
    _int(root["seed"], "run.seed", minimum=0)
    initialization = _exact(root["initialization"], {"mode", "receipt"}, "run.initialization")
    mode = _enum(initialization["mode"], {"frozen", "receipt"}, "run.initialization.mode")
    if mode == "receipt":
        _safe_id(initialization["receipt"], "run.initialization.receipt")
    elif initialization["receipt"] is not None:
        raise ValueError("run.initialization.receipt requires mode='receipt'")
    resume = _exact(root["resume"], {"enabled", "exact"}, "run.resume")
    _bool(resume["enabled"], "run.resume.enabled")
    _bool(resume["exact"], "run.resume.exact")
    if resume["exact"] and not resume["enabled"]:
        raise ValueError("run.resume.exact requires enabled=true")
    retry = _exact(root["retry"], {"max_attempts", "failure_policy"}, "run.retry")
    attempts = _int(retry["max_attempts"], "run.retry.max_attempts", minimum=1)
    if attempts > 16:
        raise ValueError("run.retry.max_attempts must be <= 16")
    _enum(retry["failure_policy"], {"stop", "continue"}, "run.retry.failure_policy")
    _enum(root["artifact_level"], {"core", "full"}, "run.artifact_level")
    destination = _exact(root["destination"], {"run_id"}, "run.destination")
    _safe_id(destination["run_id"], "run.destination.run_id")


def _validate_run_summary(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {
            "format",
            "run_id",
            "run_sha256",
            "status",
            "counts",
            "results",
        },
        "run summary",
    )
    _safe_id(root["run_id"], "run summary.run_id")
    _digest(root["run_sha256"], "run summary.run_sha256")
    _enum(root["status"], {"completed", "partial", "failed"}, "run summary.status")
    counts = _object(root["counts"], "run summary.counts")
    for name, value in counts.items():
        _safe_id(name, "run summary.counts key")
        _int(value, f"run summary.counts.{name}", minimum=0)
    _file_list(root["results"], "run summary.results")


def _validate_result(doc: Mapping[str, Any]) -> None:
    root = _exact(
        doc,
        {
            "format",
            "run_id",
            "uid",
            "input",
            "status",
            "completed_stages",
            "task_outcome",
            "provenance_status",
            "request_sha256",
            "resolved_config_sha256",
            "artifacts",
        },
        "result",
    )
    _safe_id(root["run_id"], "result.run_id")
    _safe_id(root["uid"], "result.uid")
    _validate_input_identity(root["input"], "result.input", require_digest=True)
    _enum(root["status"], {"completed", "failed", "partial"}, "result.status")
    completed_stages = [
        _enum(value, set(PIPELINE_STAGES), "result.completed_stages")
        for value in _list(root["completed_stages"], "result.completed_stages")
    ]
    if completed_stages != sorted(set(completed_stages), key=list(PIPELINE_STAGES).index):
        raise ValueError("result.completed_stages must be unique and in canonical order")
    _enum(root["task_outcome"], {"success", "failed", "not_evaluated"}, "result.task_outcome")
    provenance = _enum(root["provenance_status"], {"complete", "reconstructed", "unknown"}, "result.provenance_status")
    _digest(root["request_sha256"], "result.request_sha256")
    resolved = _digest(root["resolved_config_sha256"], "result.resolved_config_sha256", nullable=True)
    if provenance == "complete" and resolved is None:
        raise ValueError("complete result requires resolved_config_sha256")
    _file_list(root["artifacts"], "result.artifacts")


def _digest_map(value: Any, label: str, keys: set[str], *, nullable: bool = False) -> None:
    mapping = _exact(value, keys, label)
    for name, digest in mapping.items():
        _digest(digest, f"{label}.{name}", nullable=nullable)


def _validate_result_request(doc: Mapping[str, Any], *, legacy: bool = False) -> None:
    root = _exact(
        doc,
        {
            "format",
            "execution_spec_sha256",
            "case_sha256",
            "init_sha256",
            "generation_input_sha256",
            "reference_sha256",
            "packaged_defaults_sha256",
            "protocol_sha256",
            "case_protocol_sha256",
            "input",
            "input_manifest_sha256",
            "resolved_config_sha256",
            "runtime_config_sha256",
            "implementation_sha256",
            "initialization",
            "provenance_status",
        } - ({"execution_spec_sha256", "implementation_sha256"} if legacy else set())
        | ({"run_sha256"} if legacy else set()),
        "result request",
    )
    for name in (
        "run_sha256" if legacy else "execution_spec_sha256",
        "case_sha256",
        "init_sha256",
        "reference_sha256",
        "input_manifest_sha256",
    ):
        _digest(root[name], f"result request.{name}")
    _digest(root["generation_input_sha256"], "result request.generation_input_sha256", nullable=True)
    _digest_map(
        root["packaged_defaults_sha256"],
        "result request.packaged_defaults_sha256",
        set(PIPELINE_STAGES),
    )
    _digest_map(root["protocol_sha256"], "result request.protocol_sha256", set(PROTOCOL_STAGES))
    _digest(
        root["case_protocol_sha256"],
        "result request.case_protocol_sha256",
    )
    _validate_input_identity(root["input"], "result request.input", require_digest=True)
    _digest(root["resolved_config_sha256"], "result request.resolved_config_sha256", nullable=True)
    _digest(root["runtime_config_sha256"], "result request.runtime_config_sha256", nullable=True)
    if not legacy:
        _digest(root["implementation_sha256"], "result request.implementation_sha256")
    provenance = _enum(root["provenance_status"], {"complete", "reconstructed", "unknown"}, "result request.provenance_status")
    if legacy and provenance == "complete":
        raise ValueError("legacy result request cannot claim complete implementation provenance")
    if provenance == "complete" and (root["resolved_config_sha256"] is None or root["runtime_config_sha256"] is None):
        raise ValueError("complete result request requires resolved and runtime config digests")
    initialization = _exact(
        root["initialization"],
        {"mode", "receipt_sha256"},
        "result request.initialization",
    )
    mode = _enum(initialization["mode"], {"frozen", "receipt"}, "result request.initialization.mode")
    receipt = _digest(initialization["receipt_sha256"], "result request.initialization.receipt_sha256", nullable=True)
    if mode == "receipt" and receipt is None:
        raise ValueError("receipt initialization requires receipt_sha256")
    if mode == "frozen" and receipt is not None:
        raise ValueError("frozen initialization forbids receipt_sha256")


def validate_legacy_result_request(doc: Mapping[str, Any]) -> None:
    """Validate historical read-only evidence, never a current write schema.

    A historical run digest does not identify the execution implementation.
    Missing execution and implementation identities remain missing.
    """
    if doc.get("format") != RESULT_REQUEST_SCHEMA:
        raise ValueError("unexpected legacy result request format")
    _validate_result_request(doc, legacy=True)


def _validate_resolved_config(doc: Mapping[str, Any]) -> None:
    root = _exact(doc, {"format", "uid", "input", "values", "sources"}, "resolved config")
    _safe_id(root["uid"], "resolved config.uid")
    _validate_input_identity(root["input"], "resolved config.input", require_digest=True)
    values = _exact(root["values"], set(PIPELINE_STAGES), "resolved config.values")
    for stage, payload in values.items():
        _object(payload, f"resolved config.values.{stage}")
    sources = _object(root["sources"], "resolved config.sources")
    for pointer, owner in sources.items():
        if not isinstance(pointer, str) or not _JSON_POINTER.fullmatch(pointer):
            raise ValueError("resolved config source keys must be non-root JSON Pointers")
        _text(owner, f"resolved config.sources[{pointer!r}]")


_VALIDATORS: dict[str, Callable[[Mapping[str, Any]], None]] = {
    WORKSPACE_SCHEMA: _validate_workspace,
    BENCH_SCHEMA: _validate_bench,
    SOURCE_SCHEMA: _validate_source,
    CASE_SCHEMA: _validate_case,
    ENVIRONMENT_SCHEMA: _validate_environment,
    ENVIRONMENT_STATE_SCHEMA: _validate_environment_state,
    INIT_RECEIPT_SCHEMA: _validate_init_receipt,
    WORK_MATERIALIZATION_SCHEMA: _validate_work_materialization,
    GENERATION_INPUT_SCHEMA: _validate_generation_input,
    PROTOCOL_SCHEMA: _validate_protocol,
    CASE_PROTOCOL_SCHEMA: _validate_case_protocol,
    REFERENCE_SCHEMA: _validate_reference,
    VIDEO_OUTPUT_SCHEMA: _validate_video_output,
    COLLECTION_SCHEMA: _validate_collection,
    RUN_SCHEMA: _validate_run,
    RESULT_SCHEMA: _validate_result,
    RUN_SUMMARY_SCHEMA: _validate_run_summary,
    RESULT_REQUEST_SCHEMA: _validate_result_request,
    RESOLVED_CONFIG_SCHEMA: _validate_resolved_config,
}


def validate_document(
    document: Mapping[str, Any],
    *,
    expected_schema: str | None = None,
) -> dict[str, Any]:
    root = _object(document, "document")
    schema = _text(root.get("format"), "document.format")
    if expected_schema is not None and schema != expected_schema:
        raise ValueError(f"expected schema {expected_schema!r}, got {schema!r}")
    validator = _VALIDATORS.get(schema)
    if validator is None:
        raise ValueError(f"unsupported benchmark schema: {schema!r}")
    validator(root)
    return dict(root)


def load_and_validate(
    path: str | Path,
    *,
    expected_schema: str | None = None,
) -> dict[str, Any]:
    return validate_document(load_json_strict(path), expected_schema=expected_schema)


__all__ = [
    "BENCH_SCHEMA",
    "CASE_PROTOCOL_ROUTES",
    "CASE_PROTOCOL_SCHEMA",
    "CASE_SCHEMA",
    "COLLECTION_SCHEMA",
    "GENERATION_INPUT_SCHEMA",
    "INIT_RECEIPT_SCHEMA",
    "ENVIRONMENT_SCHEMA",
    "ENVIRONMENT_STATE_SCHEMA",
    "MAX_JSON_BYTES",
    "PIPELINE_STAGES",
    "PROTOCOL_SCHEMA",
    "PROTOCOL_STAGES",
    "REFERENCE_SCHEMA",
    "RESOLVED_CONFIG_SCHEMA",
    "RESULT_REQUEST_SCHEMA",
    "RESULT_SCHEMA",
    "RUN_SCHEMA",
    "RUN_SUMMARY_SCHEMA",
    "SOURCE_SCHEMA",
    "VIDEO_OUTPUT_SCHEMA",
    "WORKSPACE_SCHEMA",
    "WORK_MATERIALIZATION_SCHEMA",
    "canonical_json_bytes",
    "canonical_sha256",
    "input_identity_key",
    "load_and_validate",
    "load_json_value_strict",
    "load_json_strict",
    "validate_document",
]
