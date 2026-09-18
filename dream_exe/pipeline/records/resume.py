"""Workflow-specific identity, fingerprint, and output-state helpers.

The generic state mechanism lives in :mod:`dream_exe.pipeline.records.state`.
This module binds
that mechanism to the concrete benchmark stages without owning workflow
execution, locking, state publication, or result reconstruction.
"""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from ...artifacts.layout import execution_artifact_paths
from ...model_assets.digests import sha256_stable_file
from ...video2traj.pose.estimation import (
    pose_backend_execution_requested,
)
from ...video2traj.runtime.pose_bridge import (
    prepare_standalone_pose_pipeline,
    runtime_pose_estimator_owns_configuration,
)
from ...sim.robocasa.restore import resolve_root_reference
from ...sim.frozen.restore import resolve_scene_manifest_path
from ...evaluation.contracts import build_evaluation_plan
from ..stages.sim import resolve_bench_execution_request
from ..stages.task_success import resolve_bench_task_success_request
from ..stages.video2traj import resolve_benchmark_video2traj_request
from ..validation.artifacts import read_bounded_json_object
from .state import fingerprint_value, sha256_file

__all__ = [
    "build_stage_descriptor",
    "collect_stage_outputs",
    "stage_implementation_fingerprint",
]


_PATH_FIELD_SUFFIXES = (
    "_cache",
    "_checkpoint",
    "_dir",
    "_file",
    "_path",
    "_root",
)
_SOURCE_FINGERPRINT_SUFFIXES = frozenset({".json", ".py"})


def _location_fingerprint(path: Path) -> str:
    """Return a non-reversible identity for one resolved host location."""

    return hashlib.sha256(path.as_posix().encode("utf-8")).hexdigest()


def _file_identity(
    path: str | Path,
    *,
    role: str,
    verified_sha256: str = "",
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"{role} must be a regular file")
    digest = verified_sha256.strip()
    if digest and (
        len(digest) != 64
        or any(character not in "0123456789abcdef" for character in digest)
    ):
        raise ValueError(f"{role} verified_sha256 must be lowercase SHA-256")
    if digest:
        observed = sha256_stable_file(source, label=role)
        if observed != digest:
            raise ValueError(
                f"{role} changed after runtime asset validation: "
                f"expected {digest}, got {observed}"
            )
    return {
        "role": role,
        "size": int(source.stat().st_size),
        "sha256": digest or sha256_file(source),
    }


def _execution_config_input_identities(
    request: Mapping[str, Any],
    *,
    role_prefix: str,
) -> list[dict[str, Any]]:
    """Hash the base and run-override files that produced one config."""

    meta = dict(request.get("execution_config_meta", {}) or {})
    identities: list[dict[str, Any]] = []
    for field, suffix in (
        ("source", "execution_config"),
        ("override_source", "execution_overrides"),
    ):
        value = str(meta.get(field, "") or "").strip()
        if not value or value.startswith("<"):
            continue
        identities.append(
            _file_identity(
                value,
                role=f"{role_prefix}.{suffix}",
            )
        )
    return identities


def _scene_restore_input_identities(
    simulator_config: Mapping[str, Any],
    scene_restore_options: Mapping[str, Any] | None,
    *,
    role_prefix: str,
) -> list[dict[str, Any]]:
    """Hash every external file read by the RoboCasa manifest restore path."""

    root = (
        dict(simulator_config["raw"])
        if "raw" in simulator_config
        else dict(simulator_config)
    )
    scene_restore = dict(root.get("scene_restore", {}) or {})
    backend = str(scene_restore.get("backend", "") or "").strip()
    if backend != "robocasa_frozen":
        return []

    if scene_restore_options is None:
        options: dict[str, Any] = {}
    elif isinstance(scene_restore_options, Mapping):
        options = dict(scene_restore_options)
    else:
        raise TypeError("scene_restore_options must be a mapping")

    paths = dict(scene_restore.get("paths", {}) or {})
    path_refs = dict(scene_restore.get("path_refs", {}) or {})
    raw_named_roots = options.get("named_roots", {}) or {}
    if not isinstance(raw_named_roots, Mapping):
        raise TypeError("scene_restore_options.named_roots must be a mapping")
    named_roots = dict(raw_named_roots)


    custom_resolver = options.get("path_resolver")
    if custom_resolver is not None and not callable(custom_resolver):
        raise TypeError("scene_restore_options.path_resolver must be callable")
    resolver: Callable[[Mapping[str, Any]], Any] = (
        (
            lambda reference: resolve_root_reference(
                reference,
                named_roots=named_roots,
            )
        )
        if custom_resolver is None
        else custom_resolver
    )

    identities: list[dict[str, Any]] = []
    for key in ("model_xml_gz", "ep_meta_json", "states_npz"):
        resolved = resolve_scene_manifest_path(
            key,
            paths=paths,
            path_refs=path_refs,
            path_resolver=resolver,
        )
        if not resolved:
            raise ValueError(
                "RoboCasa scene restore requires model_xml_gz, "
                "ep_meta_json, and states_npz"
            )
        identities.append(
            _file_identity(
                resolved,
                role=f"{role_prefix}.scene_restore.{key}",
            )
        )

    override_reference = scene_restore.get("scene_override_ref", {}) or {}
    if isinstance(override_reference, Mapping) and override_reference:
        override_path = resolver(override_reference)
        if override_path and Path(override_path).exists():
            identities.append(
                _file_identity(
                    override_path,
                    role=f"{role_prefix}.scene_restore.scene_override_ref",
                )
            )
    return sorted(identities, key=lambda item: str(item["role"]))


def _path_value_identity(
    value: Any,
    *,
    role: str,
    path_base: Path | None = None,
) -> Any:
    if isinstance(value, Path):
        value = value.as_posix()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        candidate = Path(text).expanduser()
        if not candidate.is_absolute() and path_base is not None:
            candidate = path_base / candidate
        if candidate.is_absolute():
            resolved = candidate.resolve(strict=False)
            if resolved.is_file():
                return _file_identity(resolved, role=role)
            if resolved.is_dir():
                return {
                    "role": role,
                    "kind": "external_directory",
                    "leaf": resolved.name,
                    "location_sha256": _location_fingerprint(resolved),
                }
            return {
                "role": role,
                "kind": "missing_path",
                "leaf": resolved.name,
                "location_sha256": _location_fingerprint(resolved),
            }
        return text
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _path_value_identity(
                item,
                role=f"{role}[{index}]",
                path_base=path_base,
            )
            for index, item in enumerate(value)
        ]
    return value


def _portable_semantic_value(
    value: Any,
    *,
    role: str,
    path_base: Path | None = None,
) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, nested in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{role} keys must be strings")
            key = raw_key
            if key == "_meta":
                continue
            lowered = key.lower()
            is_path_field = lowered.endswith(_PATH_FIELD_SUFFIXES) or lowered in {
                "asset_base",
                "asset_root",
                "destination",
                "source_roots",
                "weights",
            }
            output[key] = (
                _path_value_identity(
                    nested,
                    role=f"{role}.{key}",
                    path_base=path_base,
                )
                if is_path_field
                else _portable_semantic_value(
                    nested,
                    role=f"{role}.{key}",
                    path_base=path_base,
                )
            )
        return output
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _portable_semantic_value(
                item,
                role=f"{role}[{index}]",
                path_base=path_base,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, Path):
        return _path_value_identity(
            value.as_posix(),
            role=role,
            path_base=path_base,
        )
    return copy.deepcopy(value)


def _source_scope_fingerprint(stage: str) -> str:
    package_root = Path(__file__).resolve().parents[2]
    pipeline_root = package_root / "pipeline"
    shared = (
        pipeline_root / "runner" / "workflow.py",
        pipeline_root / "records" / "layout.py",
        pipeline_root / "validation" / "artifacts.py",
        pipeline_root / "planning" / "depth_artifacts.py",
        pipeline_root / "records" / "state.py",
        pipeline_root / "records" / "resume.py",
        pipeline_root / "runner" / "sequence.py",
    )
    scopes: dict[str, tuple[Path, ...]] = {
        "video": (
            pipeline_root / "planning" / "configuration.py",
            package_root / "generation" / "sources.py",
        ),
        "video2traj": (
            package_root / "video2traj",
            package_root / "model_assets" / "dvd_identity.py",
            package_root / "transforms.py",
            pipeline_root / "planning" / "configuration.py",
            pipeline_root / "planning" / "depth_routes.py",
            pipeline_root / "stages" / "video2traj.py",
            pipeline_root / "planning" / "regions.py",
            package_root / "generation" / "sources.py",
            package_root / "sim" / "runtime" / "controller.py",
            package_root / "sim" / "execution" / "action_trace.py",
            package_root / "sim" / "execution" / "inputs.py",
            package_root / "artifacts",
        ),
        "exec": (
            package_root / "sim",
            package_root / "transforms.py",
            pipeline_root / "stages" / "sim.py",
            pipeline_root / "records" / "execution_lineage.py",
            package_root / "artifacts" / "io.py",
        ),
        "task_success": (
            package_root / "sim",
            package_root / "transforms.py",
            package_root / "evaluation" / "execution" / "task_success.py",
            pipeline_root / "stages" / "task_success.py",
        ),
        "eval": (
            package_root / "evaluation",
            pipeline_root / "stages" / "evaluation.py",
        ),
    }
    candidates: list[Path] = []
    for scope in (*shared, *scopes[stage]):
        if scope.is_file():
            candidates.append(scope)
        elif scope.is_dir():
            candidates.extend(
                path
                for path in scope.rglob("*")
                if (
                    path.is_file()
                    and path.suffix in _SOURCE_FINGERPRINT_SUFFIXES
                    and "__pycache__" not in path.parts
                )
            )
    if not candidates:
        raise RuntimeError(f"no current implementation source files found for stage fingerprint: {stage}")
    digest = hashlib.sha256()
    for source in sorted(set(candidates)):
        relative = source.relative_to(package_root).as_posix()
        digest.update(relative.encode("utf-8"))
        digest.update(b"\0")
        digest.update(source.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()


def stage_implementation_fingerprint(
    stage: str,
    *,
    token: str = "",
) -> str:
    """Fingerprint the current implementation source scope and optional caller declaration."""

    declaration_digest = (
        hashlib.sha256(token.encode("utf-8")).hexdigest() if token else ""
    )
    return fingerprint_value(
        {
            "stage": stage,
            "source_sha256": _source_scope_fingerprint(stage),
            "caller_declaration_sha256": declaration_digest,
        }
    )


def _component_backend(
    runtime_config: Mapping[str, Any],
    name: str,
    *,
    fallback: str = "",
) -> str:
    value = runtime_config.get(name, {})
    if not isinstance(value, Mapping):
        return fallback
    return str(value.get("backend", fallback) or fallback).strip().lower()


def _configured_path_identity(
    value: Any,
    *,
    role: str,
    sample_root: Path,
) -> Any:
    """Represent configured paths without depending on output existence."""

    if isinstance(value, Path):
        value = value.as_posix()
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return ""
        candidate = Path(text).expanduser()
        if not candidate.is_absolute():
            return text
        resolved = candidate.resolve(strict=False)
        try:
            relative = resolved.relative_to(sample_root)
        except ValueError:
            return {
                "role": role,
                "kind": "external_path",
                "leaf": resolved.name,
                "location_sha256": _location_fingerprint(resolved),
            }
        return {
            "role": role,
            "kind": "sample_path",
            "path": relative.as_posix(),
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _configured_path_identity(
                item,
                role=f"{role}[{index}]",
                sample_root=sample_root,
            )
            for index, item in enumerate(value)
        ]
    return copy.deepcopy(value)


def _portable_pipeline_config(
    value: Any,
    *,
    role: str,
    sample_root: Path,
) -> Any:
    """Keep every normalized semantic field while removing host-local paths."""

    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, nested in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{role} keys must be strings")
            key = raw_key
            if key == "_meta":
                continue
            lowered = key.lower()
            is_path_field = lowered.endswith(_PATH_FIELD_SUFFIXES) or lowered in {
                "asset_base",
                "asset_root",
                "destination",
                "source_roots",
                "weights",
            }
            output[key] = (
                _configured_path_identity(
                    nested,
                    role=f"{role}.{key}",
                    sample_root=sample_root,
                )
                if is_path_field
                else _portable_pipeline_config(
                    nested,
                    role=f"{role}.{key}",
                    sample_root=sample_root,
                )
            )
        return output
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _portable_pipeline_config(
                item,
                role=f"{role}[{index}]",
                sample_root=sample_root,
            )
            for index, item in enumerate(value)
        ]
    if isinstance(value, Path):
        return _configured_path_identity(
            value,
            role=role,
            sample_root=sample_root,
        )
    return copy.deepcopy(value)


def _video2traj_run_options_identity(
    value: Mapping[str, Any],
    *,
    role: str,
    path_base: Path,
    sample_root: Path,
) -> dict[str, Any]:
    """Fingerprint inputs without treating a stage-owned directory as input.

    ``tracking_output_dir`` is an explicit destination created by video2traj
    itself.  Its location affects the request identity, but whether it exists
    must not: a fresh resume run legitimately changes it from missing to a
    directory while publishing tracking diagnostics.
    """

    options = copy.deepcopy(dict(value))
    missing = object()
    tracking_output_dir = options.pop("tracking_output_dir", missing)
    identity = _portable_semantic_value(
        options,
        role=role,
        path_base=path_base,
    )
    if not isinstance(identity, dict):  # pragma: no cover - mapping is owned
        raise TypeError(f"{role} identity must be a mapping")
    if tracking_output_dir is not missing:
        identity["tracking_output_dir"] = _configured_path_identity(
            tracking_output_dir,
            role=f"{role}.tracking_output_dir",
            sample_root=sample_root,
        )
    return identity


def _video2traj_token_reason(
    *,
    runtime_config: Mapping[str, Any],
    use_gt_depth: bool,
    trajectory_runner: Callable[..., Any] | None,
    trajectory_dependencies: Mapping[str, Any] | None,
) -> str:
    if trajectory_runner is not None:
        return "custom_runner_requires_implementation_token"
    if trajectory_dependencies:
        return "injected_dependencies_require_implementation_token"
    if _component_backend(runtime_config, "region") not in {
        "manual",
        "none",
        "precomputed",
    }:
        return "region_provider_requires_implementation_token"
    if _component_backend(runtime_config, "tracking") not in {"none"}:
        return "tracking_provider_requires_implementation_token"
    if _component_backend(runtime_config, "pose", fallback="none") not in {"none"}:
        return "pose_provider_requires_implementation_token"
    if _component_backend(
        runtime_config,
        "depth_calibration",
        fallback="none",
    ) not in {"none"}:
        return "depth_calibration_requires_implementation_token"
    if not use_gt_depth and _component_backend(runtime_config, "depth") not in {"none"}:
        return "depth_provider_requires_implementation_token"
    return ""


def _runtime_asset_base(
    runtime_config: Mapping[str, Any],
    explicit_base: str | Path | None,
) -> Path | None:
    if explicit_base is not None and str(explicit_base).strip():
        base = Path(explicit_base).expanduser()
        if not base.is_absolute():
            raise ValueError("trajectory_runtime_asset_base must be absolute")
        resolved: Path | None = base.resolve(strict=False)
    else:
        meta = runtime_config.get("_meta", {})
        meta_mapping = dict(meta) if isinstance(meta, Mapping) else {}
        base_text = str(meta_mapping.get("base_dir", "") or "").strip()
        resolved = (
            Path(base_text).expanduser().resolve(strict=False) if base_text else None
        )
    asset_root = str(runtime_config.get("asset_root", "") or "").strip()
    if not asset_root:
        return resolved
    root = Path(asset_root).expanduser()
    if not root.is_absolute():
        if resolved is None:
            raise ValueError(
                "relative runtime asset_root requires an explicit asset base"
            )
        root = resolved / root
    return root.resolve(strict=False)


def _resolved_runtime_path(
    value: Any,
    *,
    base: Path | None,
    label: str,
) -> str | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        if base is None:
            raise ValueError(
                f"relative {label} requires an explicit runtime asset base"
            )
        path = base / path
    return path.resolve(strict=False).as_posix()


def _depth_asset_identity(
    *,
    uid: str,
    runtime_config: Mapping[str, Any],
    runtime_asset_base: str | Path | None,
    use_gt_depth: bool,
) -> tuple[dict[str, Any], str]:
    if use_gt_depth:
        return {"mode": "rollout_gt_depth"}, ""
    depth_value = runtime_config.get("depth", {})
    if not isinstance(depth_value, Mapping):
        return {}, "depth_runtime_config_is_not_a_mapping"
    depth = copy.deepcopy(dict(depth_value))
    backend = str(depth.get("backend", "") or "").strip().lower()
    if backend == "none":
        return {"backend": "none"}, ""
    if backend in {"factory", "injected"}:
        declared = depth.get("runtime_config", {})
        model_name = (
            str(dict(declared).get("model_name", "") or "").strip().lower()
            if isinstance(declared, Mapping)
            else ""
        )
        return (
            {"backend": backend, "model_name": model_name},
            f"{backend}_depth_assets_are_not_file_attestable",
        )
    if backend not in {"dvd", "registry", "vda"}:
        return {}, f"unsupported_depth_backend_for_resume:{backend or 'missing'}"

    from ...video2traj.depth.estimator import (
        resolve_depth_estimator_runtime,
        validate_depth_estimator_assets,
    )

    base = _runtime_asset_base(runtime_config, runtime_asset_base)
    config_value = depth.get("config")
    config_path = _resolved_runtime_path(
        depth.get("config_path", ""),
        base=base,
        label="depth.config_path",
    )
    weights_root = _resolved_runtime_path(
        depth.get("weights_root", ""),
        base=base,
        label="depth.weights_root",
    )
    runtime = resolve_depth_estimator_runtime(
        device=str(runtime_config.get("device", "") or ""),
        preset=str(depth.get("preset", "") or ""),
        config=(
            copy.deepcopy(dict(config_value))
            if isinstance(config_value, Mapping)
            else None
        ),
        config_path=config_path,
        fp32=bool(depth.get("fp32", False)),
        weights_root=weights_root,
        runtime_context={
            **dict(depth.get("runtime_context", {}) or {}),
            "uid": uid,
        },
    )
    runtime = validate_depth_estimator_assets(runtime)
    model_name = str(runtime["model_name"])
    model_kwargs = dict(runtime.get("model_kwargs", {}) or {})
    if model_name == "vda":
        checkpoint_root = Path(str(model_kwargs["ckpt_root"])).expanduser().resolve()
        encoder = str(model_kwargs.get("encoder", "vitl"))
        prefix = (
            "metric_video_depth_anything"
            if bool(model_kwargs.get("metric", False))
            else "video_depth_anything"
        )
        checkpoint = checkpoint_root / f"{prefix}_{encoder}.pth"
        return {
            "backend": backend,
            "model_name": "vda",
            "encoder": encoder,
            "metric": bool(model_kwargs.get("metric", False)),
            "checkpoint": _file_identity(
                checkpoint,
                role="depth.vda.checkpoint",
            ),
        }, ""

    provenance = dict(runtime.get("model_provenance", {}) or {})
    validation = dict(provenance.get("validation", {}) or {})
    if (
        validation.get("status") != "verified"
        or validation.get("asset_integrity_gate") != "passed"
        or validation.get("verification_scope") != "caller_declared_asset_bytes"
    ):
        return (
            {
                "backend": backend,
                "model_name": "dvd",
                "validation_status": str(validation.get("status", "declaration_only")),
            },
            "dvd_assets_are_not_runtime_verified",
        )
    checkpoint_root = Path(str(model_kwargs["ckpt_root"])).expanduser().resolve()
    checkpoint = checkpoint_root / "model.safetensors"
    if not checkpoint.is_file():
        candidates = sorted(checkpoint_root.glob("*.safetensors"))
        if len(candidates) != 1:
            return {}, "dvd_checkpoint_is_not_unambiguous"
        checkpoint = candidates[0]
    model_config = Path(str(model_kwargs["model_config_path"])).expanduser().resolve()
    validated_assets = dict(validation.get("assets", {}) or {})
    checkpoint_validation = dict(validated_assets.get("checkpoint", {}) or {})
    model_config_validation = dict(validated_assets.get("model_config", {}) or {})
    return {
        "backend": backend,
        "model_name": "dvd",
        "verification_scope": "caller_declared_asset_bytes",
        "live_checkpoint_finetune_gate": "not_implemented",
        "checkpoint": _file_identity(
            checkpoint,
            role="depth.dvd.checkpoint",
            verified_sha256=str(checkpoint_validation.get("actual_sha256", "") or ""),
        ),
        "model_config": _file_identity(
            model_config,
            role="depth.dvd.model_config",
            verified_sha256=str(model_config_validation.get("actual_sha256", "") or ""),
        ),
    }, ""


def _pose_asset_identity(
    *,
    request: Mapping[str, Any],
    runtime_config: Mapping[str, Any],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    pipeline_config = request.get("pipeline_config", {})
    if not isinstance(pipeline_config, Mapping):
        return {}, "pipeline_config_is_not_a_mapping", {}
    inline_pose = request.get("pipeline_pose_config", {})
    if not isinstance(inline_pose, Mapping):
        return {}, "pipeline_pose_config_is_not_a_mapping", {}
    pipeline_path = Path(str(request["pipeline_config_path"]))
    binding = prepare_standalone_pose_pipeline(
        pipeline_config,
        inline_pose_config=inline_pose,
        pipeline_asset_base=pipeline_path.parent,
        estimator_owns_configuration=(
            runtime_pose_estimator_owns_configuration(runtime_config)
        ),
    )
    manifest = dict(binding["manifest"])
    identity: dict[str, Any] = {
        "enabled": bool(manifest.get("enabled", False)),
        "configuration_authority": str(
            manifest.get("configuration_authority", "") or ""
        ),
        "effective_backend": str(manifest.get("effective_backend", "") or ""),
        "consumed_files": [],
    }
    effective_pipeline = copy.deepcopy(dict(binding["pipeline_config"]))
    if not identity["enabled"]:
        return identity, "", effective_pipeline
    if manifest.get("backend_identity") == "caller_owned":
        identity["backend_identity"] = "caller_owned"
    if bool(manifest.get("config_file_consumed", False)):
        identity["consumed_files"].append(
            _file_identity(
                manifest["config_path_resolved"],
                role="video2traj.pose.config",
            )
        )
    if bool(manifest.get("pose_correction_file_consumed", False)):
        identity["consumed_files"].append(
            _file_identity(
                manifest["pose_correction_path"],
                role="video2traj.pose.correction",
            )
        )

    effective_pose = dict(dict(binding["pipeline_config"]).get("pose", {}) or {})
    backend_config = {
        key: copy.deepcopy(value)
        for key, value in effective_pose.items()
        if key not in {"enabled", "config_path"}
    }
    mesh_path = str(effective_pose.get("mesh_path", "") or "").strip()
    if pose_backend_execution_requested(backend_config) and mesh_path:
        mesh = Path(mesh_path)
        if not mesh.is_file():
            return (
                identity,
                "pose_mesh_is_not_file_attestable",
                effective_pipeline,
            )
        identity["consumed_files"].append(
            _file_identity(
                mesh,
                role="video2traj.pose.mesh",
            )
        )
    return identity, "", effective_pipeline


def _output_identity(
    path: str | Path,
    *,
    sample_root: Path,
    run_root: Path,
) -> dict[str, str]:
    candidate = Path(path).expanduser().resolve(strict=False)
    try:
        relative = candidate.relative_to(run_root)
        root = "run"
    except ValueError:
        try:
            relative = candidate.relative_to(sample_root)
        except ValueError as error:
            raise ValueError(
                "required stage output escapes the benchmark sample"
            ) from error
        root = "sample"
    path_text = relative.as_posix()
    if not path_text or path_text == "." or ".." in relative.parts:
        raise ValueError("required stage output is not a safe file path")
    return {"root": root, "path": path_text}


def _required_output_identities(
    stage: str,
    *,
    context: Mapping[str, Any],
    source_video_path: str | Path | None,
    execution_mode: str,
    video2traj_depth_contract: Mapping[str, Any] | None = None,
) -> list[dict[str, str]]:
    sample_root = Path(context["sample_dir"])
    formal = dict(context["formal"])
    run_root = Path(formal["run_root"])
    if stage == "video":
        if source_video_path is None or not str(source_video_path).strip():
            return []
        paths = [Path(source_video_path)]
    elif stage == "video2traj":
        paths = [
            Path(formal["trajectory_manifest"]),
            Path(formal["ee_traj"]),
            Path(formal["gripper"]),
            Path(formal["action"]),
        ]
        if video2traj_depth_contract is not None:
            raw_depth_paths = video2traj_depth_contract.get("paths", {})
            if not isinstance(raw_depth_paths, Mapping):
                raise TypeError(
                    "video2traj depth artifact contract paths must be a mapping"
                )
            paths.extend(Path(path) for path in raw_depth_paths.values())
    elif stage == "exec":
        paths = [
            Path(formal["exec_summary"]),
            Path(formal["checkpoint_trace"]),
        ]
        if execution_mode == "action":
            paths.append(Path(formal["dense_tcp_trace"]))
        execution_lineage = execution_artifact_paths(formal["exec_dir"])[
            "execution_inputs"
        ]
        if execution_lineage.is_file():
            paths.append(execution_lineage)
    elif stage == "task_success":
        paths = [Path(formal["task_success"])]
    elif stage == "eval":
        paths = [
            execution_artifact_paths(formal["exec_dir"])["exec_metrics"],
            execution_artifact_paths(formal["exec_dir"])["exec_metrics_per_frame"],
            Path(formal["evaluation_result"]),
        ]
    else:  # pragma: no cover - private caller controls stage
        raise ValueError(f"unsupported state stage: {stage}")
    identities = [
        _output_identity(
            path,
            sample_root=sample_root,
            run_root=run_root,
        )
        for path in paths
    ]
    return sorted(identities, key=lambda item: (item["root"], item["path"]))


def _nonreusable_descriptor(
    stage: str,
    *,
    reason: str,
    token: str,
    implementation_fingerprint: str = "",
    input_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    input_fingerprint = (
        fingerprint_value(input_payload)
        if input_payload is not None
        else fingerprint_value(
            {
                "stage": stage,
                "untracked_reason": reason,
            }
        )
    )
    return {
        "reusable": False,
        "reason": reason,
        "input_fingerprint": input_fingerprint,
        "implementation_fingerprint": (
            implementation_fingerprint
            or stage_implementation_fingerprint(
                stage,
                token=token,
            )
        ),
    }


def _evaluation_evidence_identities(
    formal: Mapping[str, Any],
) -> list[dict[str, Any]]:
    exec_root = Path(formal["exec_dir"])
    run_root = Path(formal["run_root"])
    summary_path = Path(formal["exec_summary"])
    summary = read_bounded_json_object(
        summary_path,
        stage="eval",
        reason="unsafe_execution_summary",
    )

    def evidence_path(field: str, fallback: Path) -> Path:
        text = str(summary.get(field, "") or "").strip()
        if not text:
            return fallback
        candidate = Path(text).expanduser()
        return (
            candidate.resolve(strict=False)
            if candidate.is_absolute()
            else (exec_root / candidate).resolve(strict=False)
        )

    candidates = (
        ("summary", summary_path),
        (
            "checkpoint",
            evidence_path(
                "checkpoint_trace_path",
                Path(formal["checkpoint_trace"]),
            ),
        ),
        (
            "dense",
            evidence_path(
                "dense_tcp_trace_path",
                Path(formal["dense_tcp_trace"]),
            ),
        ),
        (
            "action",
            evidence_path(
                "action_path",
                Path(formal["action"]),
            ),
        ),
        (
            "trajectory",
            evidence_path(
                "traj_path",
                Path(formal["ee_traj"]),
            ),
        ),
    )
    identities: list[dict[str, Any]] = []
    for role, path in candidates:
        if path.is_file():
            identities.append(
                _file_identity(
                    path,
                    role=f"eval.execution_evidence.{role}",
                )
            )
        else:
            identities.append(
                {
                    "role": f"eval.execution_evidence.{role}",
                    "kind": "missing",
                    "location": (
                        "run" if path.is_relative_to(run_root) else "external"
                    ),
                }
            )
    return identities


def _contains_callable(value: Any) -> bool:
    if callable(value):
        return True
    if isinstance(value, Mapping):
        return any(_contains_callable(item) for item in value.values())
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return any(_contains_callable(item) for item in value)
    return False


def _declared_callable_value(value: Any) -> Any:
    """Replace injected callables with a token-bound portable marker."""

    if callable(value):
        return {"kind": "injected_callable"}
    if isinstance(value, Mapping):
        return {
            str(key): _declared_callable_value(nested) for key, nested in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_declared_callable_value(item) for item in value]
    return copy.deepcopy(value)


def _manifest_owned_files(
    manifest_path: Path,
    *,
    root: Path,
) -> list[Path]:
    if not manifest_path.is_file():
        return []
    payload = read_bounded_json_object(
        manifest_path,
        stage="collect",
        reason="unsafe_assets_manifest",
    )
    root = root.resolve()
    paths: set[Path] = {manifest_path.absolute()}

    def visit(value: Any) -> None:
        if isinstance(value, Mapping):
            for nested in value.values():
                visit(nested)
            return
        if isinstance(value, Sequence) and not isinstance(
            value,
            (str, bytes, bytearray),
        ):
            for nested in value:
                visit(nested)
            return
        if not isinstance(value, str) or not value.strip():
            return
        candidate = Path(value).expanduser()
        candidate = (
            candidate.absolute()
            if candidate.is_absolute()
            else (root / candidate).absolute()
        )
        resolved = candidate.resolve(strict=False)
        if resolved.is_relative_to(root) and candidate.is_file():
            paths.add(candidate)

    visit(payload)
    return sorted(paths)


def build_stage_descriptor(
    stage: str,
    *,
    settings: Mapping[str, Any],
    token: str,
) -> dict[str, Any]:
    """Build one workflow stage's reusable-state descriptor."""

    context = dict(settings["context"])
    sample_root = Path(context["sample_dir"])
    formal = dict(context["formal"])
    source_video = settings.get("source_video")
    execution_mode = str(settings.get("execution_mode", "") or "")
    frozen_fingerprints = settings.get("implementation_fingerprints", {})
    if not isinstance(frozen_fingerprints, Mapping):
        raise TypeError("implementation_fingerprints must be a mapping")
    implementation_fingerprint = str(frozen_fingerprints.get(stage, "") or "")
    if not implementation_fingerprint:
        implementation_fingerprint = stage_implementation_fingerprint(
            stage,
            token=token,
        )

    def not_reusable(
        reason: str,
        *,
        input_payload: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        try:
            descriptor = _nonreusable_descriptor(
                stage,
                reason=reason,
                token=token,
                implementation_fingerprint=implementation_fingerprint,
                input_payload=input_payload,
            )
        except (TypeError, ValueError):
            descriptor = _nonreusable_descriptor(
                stage,
                reason=reason,
                token=token,
                implementation_fingerprint=implementation_fingerprint,
            )
        try:
            descriptor["required_outputs"] = _required_output_identities(
                stage,
                context=context,
                source_video_path=source_video,
                execution_mode=execution_mode,
                video2traj_depth_contract=settings.get("video2traj_depth_contract"),
            )
        except (KeyError, OSError, TypeError, ValueError):
            descriptor["required_outputs"] = []
        return descriptor

    try:
        if stage == "video":
            if source_video is None or not Path(source_video).is_file():
                return not_reusable("video_is_not_yet_available")
            nonreusable_reason = (
                "custom_acquirer_requires_implementation_token"
                if settings.get("acquire_video") is not None and not token
                else ""
            )
            raw_video_options = settings.get("video_options")
            if callable(raw_video_options):
                return not_reusable(
                    nonreusable_reason
                    or "callable_video_options_are_not_reconstructable"
                )
            input_payload = {
                "run_identity": dict(settings["run_identity"]),
                "video": _file_identity(
                    source_video,
                    role="video.source",
                ),
                "acquisition_options": _portable_semantic_value(
                    dict(raw_video_options or {}),
                    role="video.options",
                ),
            }
            if nonreusable_reason:
                return not_reusable(
                    nonreusable_reason,
                    input_payload=input_payload,
                )
        elif stage == "video2traj":
            if source_video is None or not Path(source_video).is_file():
                return not_reusable("video_input_is_not_available")
            runtime_config = settings.get("runtime_config")
            if not isinstance(runtime_config, Mapping):
                return not_reusable("runtime_config_is_not_a_mapping")
            token_reason = _video2traj_token_reason(
                runtime_config=runtime_config,
                use_gt_depth=bool(settings["use_gt_depth"]),
                trajectory_runner=settings.get("trajectory_runner"),
                trajectory_dependencies=settings.get("trajectory_dependencies"),
            )
            request = resolve_benchmark_video2traj_request(
                uid=str(context["uid"]),
                sample_dir=sample_root,
                run_id=str(context["run_id"]),
                run_key=str(context["run_key"]),
                video_kind=str(context["video_kind"]),
                gen_model=str(context["gen_model"]),
                source_video_path=source_video,
                pipeline_config_path=settings["trajectory_path"],
                output_traj_root=formal["traj_dir"],
                use_rollout_gt_depth=bool(settings["use_gt_depth"]),
                runtime_config=runtime_config,
                simulator_config_path=settings.get("simulator_config_path"),
                runtime_asset_base=settings.get("runtime_asset_base"),
                run_options=settings.get("trajectory_run_options"),
                video_backend=str(settings["trajectory_video_backend"]),
            )
            effective_runtime_config = request.get("runtime_config")
            if not isinstance(effective_runtime_config, Mapping):
                return not_reusable("resolved_runtime_config_is_not_a_mapping")
            depth_identity, depth_reason = _depth_asset_identity(
                uid=str(context["uid"]),
                runtime_config=effective_runtime_config,
                runtime_asset_base=settings.get("runtime_asset_base"),
                use_gt_depth=bool(settings["use_gt_depth"]),
            )
            (
                pose_identity,
                pose_reason,
                effective_pipeline_config,
            ) = _pose_asset_identity(
                request=request,
                runtime_config=effective_runtime_config,
            )
            pipeline_depth = dict(
                dict(request["pipeline_config"]).get("depth", {}) or {}
            )
            gt_depth_identity: dict[str, Any] | None = None
            if bool(settings["use_gt_depth"]):
                gt_depth_identity = _file_identity(
                    pipeline_depth["rollout_gt_depth_path"],
                    role="video2traj.rollout_gt_depth",
                )
            input_payload = {
                "run_identity": dict(settings["run_identity"]),
                "video": _file_identity(
                    source_video,
                    role="video2traj.video",
                ),
                "pipeline_config": _portable_pipeline_config(
                    effective_pipeline_config,
                    role="video2traj.pipeline_config",
                    sample_root=sample_root,
                ),
                "trajectory_config": _file_identity(
                    request["pipeline_config_path"],
                    role="video2traj.trajectory_config",
                ),
                "simulator_config": _file_identity(
                    request["simulator_config_path"],
                    role="video2traj.simulator_config",
                ),
                "runtime_config": _portable_semantic_value(
                    effective_runtime_config,
                    role="video2traj.runtime_config",
                    path_base=_runtime_asset_base(
                        effective_runtime_config,
                        settings.get("runtime_asset_base"),
                    ),
                ),
                "run_options": _video2traj_run_options_identity(
                    dict(settings.get("trajectory_run_options") or {}),
                    role="video2traj.run_options",
                    path_base=Path(settings["trajectory_path"]).parent,
                    sample_root=sample_root,
                ),
                "region_inputs": _portable_semantic_value(
                    request.get("region_inputs_manifest", {}),
                    role="video2traj.region_inputs",
                ),
                "depth_assets": depth_identity,
                "pose_assets": pose_identity,
                "rollout_gt_depth": gt_depth_identity,
                "video_backend": str(settings["trajectory_video_backend"]),
            }
            nonreusable_reason = (
                token_reason
                if token_reason and not token
                else (depth_reason or pose_reason)
            )
            if nonreusable_reason:
                return not_reusable(
                    nonreusable_reason,
                    input_payload=input_payload,
                )
        elif stage == "exec":
            nonreusable_reason = (
                (
                    "custom_simulator_requires_implementation_token"
                    if settings.get("simulator_runner") is not None
                    else "simulator_runtime_requires_implementation_token"
                )
                if not token
                else ""
            )
            request = resolve_bench_execution_request(
                sample_dir=sample_root,
                run_key=str(context["run_key"]),
                gen_model=str(context["gen_model"]),
                simulator_config_path=settings.get("simulator_config_path"),
                execution_config_path=settings["execution_path"],
                action_path=formal["action"],
                output_dir=formal["exec_dir"],
            )
            raw_execution_mode = str(
                dict(request["execution_config"]).get("execution", {}).get("mode", "")
                or ""
            ).strip()
            execution_mode = (
                "frame" if raw_execution_mode == "frame_traj" else raw_execution_mode
            )
            if execution_mode not in {"action", "frame"}:
                return not_reusable("execution_mode_is_not_reconstructable")
            consumed = [
                _file_identity(
                    request["simulator_config_path"],
                    role="exec.simulator_config",
                ),
            ]
            consumed.extend(
                _execution_config_input_identities(
                    request,
                    role_prefix="exec",
                )
            )
            trajectory_path = Path(str(request.get("trajectory_path", "") or ""))
            if execution_mode == "action":
                consumed.append(
                    _file_identity(
                        request["action_path"],
                        role="exec.action",
                    )
                )
            else:
                consumed.append(
                    _file_identity(
                        trajectory_path,
                        role="exec.trajectory",
                    )
                )
            scene_override = str(request.get("scene_override_path", "") or "")
            if scene_override:
                consumed.append(
                    _file_identity(
                        scene_override,
                        role="exec.scene_override",
                    )
                )
            consumed.extend(
                _scene_restore_input_identities(
                    request["simulator_config"],
                    settings.get("scene_restore_options"),
                    role_prefix="exec",
                )
            )
            input_payload = {
                "run_identity": dict(settings["run_identity"]),
                "consumed_files": sorted(
                    consumed,
                    key=lambda item: str(item["role"]),
                ),
                "execution": _portable_semantic_value(
                    request["execution_config"],
                    role="exec.execution_config",
                ),
                "scene_restore_options": _portable_semantic_value(
                    dict(settings.get("scene_restore_options") or {}),
                    role="exec.scene_restore_options",
                ),
            }
            robocasa_source_root = settings.get("robocasa_source_root")
            if robocasa_source_root is not None:
                input_payload["robocasa_runtime_source"] = _path_value_identity(
                    robocasa_source_root,
                    role="exec.robocasa_source_root",
                )
            if nonreusable_reason:
                return not_reusable(
                    nonreusable_reason,
                    input_payload=input_payload,
                )
        elif stage == "task_success":
            raw_options = settings.get("task_success_options")
            if not isinstance(raw_options, Mapping):
                return not_reusable("task_success_options_are_not_a_mapping")
            options = copy.deepcopy(dict(raw_options))
            has_injected_callable = settings.get(
                "task_success_runner"
            ) is not None or _contains_callable(options)
            nonreusable_reason = ""
            if not token:
                nonreusable_reason = (
                    "custom_task_success_requires_implementation_token"
                    if has_injected_callable
                    else (
                        "task_success_simulator_runtime_requires_implementation_token"
                    )
                )
            request = resolve_bench_task_success_request(
                sample_dir=options["sample_dir"],
                run_key=options["run_key"],
                gen_model=options["gen_model"],
                task_name=str(options.get("task_name", "") or ""),
                simulator_config_path=options.get("simulator_config_path"),
                execution_config_path=options.get("execution_config_path"),
                action_path=options.get("action_path"),
                object_trajectories_path=options.get("object_trajectories_path"),
                output_path=options["output_path"],
                scene_override_path=options.get("scene_override_path"),
                execution_request_resolver=options.get(
                    "execution_request_resolver",
                    resolve_bench_execution_request,
                ),
            )
            consumed = [
                _file_identity(
                    request["simulator_config_path"],
                    role="task_success.simulator_config",
                ),
                _file_identity(
                    request["action_path"],
                    role="task_success.action",
                ),
            ]
            consumed.extend(
                _execution_config_input_identities(
                    request,
                    role_prefix="task_success",
                )
            )
            scene_override = str(request.get("scene_override_path", "") or "").strip()
            if scene_override:
                consumed.append(
                    _file_identity(
                        scene_override,
                        role="task_success.scene_override",
                    )
                )
            consumed.extend(
                _scene_restore_input_identities(
                    request["simulator_config"],
                    options.get("scene_restore_options"),
                    role_prefix="task_success",
                )
            )
            object_path = Path(request["object_trajectories_path"])
            object_identity = (
                _file_identity(
                    object_path,
                    role="task_success.object_trajectories",
                )
                if object_path.is_file()
                else _path_value_identity(
                    object_path.as_posix(),
                    role="task_success.object_trajectories",
                )
            )
            behavior_options = {
                key: value
                for key, value in options.items()
                if key
                not in {
                    "action_path",
                    "execution_config_path",
                    "execution_request_resolver",
                    "gen_model",
                    "object_trajectories_path",
                    "output_path",
                    "publish",
                    "run_key",
                    "sample_dir",
                    "scene_override_path",
                    "simulator_config_path",
                }
            }
            input_payload = {
                "run_identity": dict(settings["run_identity"]),
                "task_name": str(request["task_name"]),
                "consumed_files": sorted(
                    consumed,
                    key=lambda item: str(item["role"]),
                ),
                "object_trajectories": object_identity,
                "simulator_config": _portable_semantic_value(
                    request["simulator_config"],
                    role="task_success.simulator_config",
                ),
                "execution_config": _portable_semantic_value(
                    request["execution_config"],
                    role="task_success.execution_config",
                ),
                "behavior_options": _portable_semantic_value(
                    _declared_callable_value(behavior_options),
                    role="task_success.options",
                ),
            }
            if nonreusable_reason:
                return not_reusable(
                    nonreusable_reason,
                    input_payload=input_payload,
                )
        elif stage == "eval":
            custom_metrics = settings.get("metrics_builder") is not None
            custom_similarity = (
                settings.get("trajectory_similarity_evaluator") is not None
            )
            custom_executability = (
                settings.get("trajectory_path_comparison_evaluator") is not None
            )
            custom_task_success_rate = (
                settings.get("task_success_rate_builder") is not None
            )
            explicit_vlm_requests = (
                settings.get("vlm_request_manifest_path") is not None
                or settings.get("vlm_evaluators") is not None
            )
            nonreusable_reason = ""
            injected_evaluator = any(
                (
                    custom_metrics,
                    custom_similarity,
                    custom_executability,
                    custom_task_success_rate,
                    settings.get("vlm_evaluators") is not None,
                )
            )
            if injected_evaluator and not token:
                nonreusable_reason = "custom_evaluator_requires_implementation_token"
            evidence = _evaluation_evidence_identities(formal)
            if not evidence:
                return not_reusable("execution_evidence_is_not_available")
            evaluation_run_identity = {
                "uid": str(context["uid"]),
                "run_id": str(context["run_id"]),
                "run_key": str(context["run_key"]),
                "video_kind": str(context["video_kind"]),
                "gen_model": str(context["gen_model"]),
            }
            evaluation_plan = build_evaluation_plan(
                formal_artifacts=formal,
                run_identity=evaluation_run_identity,
                trajectory_path_comparison_reference_path=settings.get(
                    "trajectory_path_comparison_reference_path"
                ),
                trajectory_similarity_specs=settings.get("trajectory_similarity_specs"),
                task_success_rate_specs=settings.get("task_success_rate_specs"),
                task_success_rate_options=settings.get("task_success_rate_options"),
                vlm_request_manifest_path=settings.get("vlm_request_manifest_path"),
            )
            raw_vlm_evaluators = settings.get("vlm_evaluators")
            vlm_evaluator_modes = (
                []
                if raw_vlm_evaluators is None
                else sorted(str(mode) for mode in raw_vlm_evaluators)
            )
            input_payload = {
                "run_identity": dict(settings["run_identity"]),
                "execution_evidence": evidence,
                "evaluation_plan": evaluation_plan,
                "vlm_evaluator_modes": vlm_evaluator_modes,
                "explicit_vlm_requests": explicit_vlm_requests,
            }
            if nonreusable_reason:
                return not_reusable(
                    nonreusable_reason,
                    input_payload=input_payload,
                )
        else:  # pragma: no cover - private caller controls stage
            raise ValueError(f"unsupported state stage: {stage}")

        required_outputs = _required_output_identities(
            stage,
            context=context,
            source_video_path=source_video,
            execution_mode=execution_mode,
            video2traj_depth_contract=settings.get("video2traj_depth_contract"),
        )
        if not required_outputs:
            return not_reusable("required_output_contract_is_not_available")
        return {
            "reusable": True,
            "reason": "",
            "input_fingerprint": fingerprint_value(input_payload),
            "implementation_fingerprint": implementation_fingerprint,
            "required_outputs": required_outputs,
        }
    except (
        FileNotFoundError,
        KeyError,
        OSError,
        TypeError,
        ValueError,
    ):
        return not_reusable("stage_identity_is_not_safely_reconstructable")


def collect_stage_outputs(
    stage: str,
    *,
    result: Mapping[str, Any],
    settings: Mapping[str, Any],
) -> list[Path]:
    """Collect the bounded files owned by a completed workflow stage."""

    context = dict(settings["context"])
    formal = dict(context["formal"])
    if stage == "video":
        path_text = str(
            result.get(
                "video_path",
                result.get("source_video_path", ""),
            )
            or ""
        ).strip()
        return [Path(path_text)] if path_text else []
    if stage == "video2traj":
        root = Path(formal["traj_dir"])
        manifest = Path(formal["trajectory_manifest"])
        candidates = [
            manifest,
            Path(formal["ee_traj"]),
            Path(formal["gripper"]),
            Path(formal["action"]),
            *_manifest_owned_files(
                manifest,
                root=root,
            ),
        ]
        raw_depth_contract = settings.get("video2traj_depth_contract")
        if raw_depth_contract is not None:
            if not isinstance(raw_depth_contract, Mapping):
                raise TypeError("video2traj_depth_contract must be a mapping")
            raw_depth_paths = raw_depth_contract.get("paths", {})
            if not isinstance(raw_depth_paths, Mapping):
                raise TypeError(
                    "video2traj depth artifact contract paths must be a mapping"
                )
            candidates.extend(Path(path) for path in raw_depth_paths.values())
    elif stage == "exec":
        root = Path(formal["exec_dir"])
        execution_layout = execution_artifact_paths(root)
        candidates = [
            execution_layout["execution_manifest"],
            execution_layout["execution_inputs"],
            Path(formal["exec_summary"]),
            Path(formal["checkpoint_trace"]),
            Path(formal["dense_tcp_trace"]),
            execution_layout["action_trace"],
            execution_layout["execution_video"],
            *_manifest_owned_files(
                execution_layout["execution_manifest"],
                root=root,
            ),
        ]
    elif stage == "task_success":
        return [Path(formal["task_success"])]
    elif stage == "eval":
        root = Path(formal["exec_dir"])
        return [
            execution_artifact_paths(root)["exec_metrics"],
            execution_artifact_paths(root)["exec_metrics_per_frame"],
            Path(formal["evaluation_result"]),
        ]
    else:  # pragma: no cover
        return []
    outputs: list[Path] = []
    for path in sorted(set(candidates)):
        if path.is_symlink():
            raise ValueError(f"{stage} output tree cannot contain symlinks")
        if path.is_file():
            outputs.append(path)
    return outputs
