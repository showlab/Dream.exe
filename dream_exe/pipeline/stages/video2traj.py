"""Run the simulator-independent video-to-trajectory stage for one case.

This module translates one resolved case into explicit simulator, video,
runtime, and output inputs for the simulator-independent algorithm boundary.

All identity and path checks happen before the injected runner is called.
Reading configuration is side-effect free; artifact publication is delegated
to the standalone callable and is enabled only for the explicit trajectory
root supplied by the benchmark orchestrator.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ...artifacts.layout import sample_artifact_paths, trajectory_artifact_paths
from ...generation.sources import normalize_generated_model_name
from ...model_assets.dvd_identity import normalize_dvd_asset_attestation
from ...video2traj.depth.estimator import (
    DEFAULT_DEPTH_PRESET,
    SUPPORTED_DEPTH_BACKENDS,
    default_depth_estimator_config,
    resolve_depth_estimator_preset,
)
from ...video2traj.depth.contract import (
    external_depth_backend_id,
    external_depth_selection,
)
from ...video2traj.runtime.standalone import (
    run_standalone_video2traj,
    summarize_standalone_video2traj_result,
)
from ...video2traj.region.config import (
    resolve_pipeline_region_selector_policy,
)
from ...video2traj.runtime.config import (
    load_pipeline_config as load_core_pipeline_config,
)
from ...video2traj.runtime.conditioning import (
    normalize_conditioning_transform,
)
from ...video2traj.trajectory.stages import compile_task_runtime
from ..planning.configuration import (
    load_bench_pipeline_config,
    resolve_default_bench_pipeline_paths,
)
from ..planning.depth_routes import (
    BENCH_RESOLVED_DEPTH_BACKEND,
    BENCH_RESOLVED_DEPTH_PRESET,
    select_benchmark_depth_provider,
)
from ..records.layout import (
    build_run_id,
    formal_artifact_paths,
    normalize_run_key,
    run_key_uses_gt_depth,
    split_run_key,
)
from ..planning.regions import (
    build_benchmark_region_inputs,
    resolve_benchmark_init_region_assets,
)

StandaloneRunner = Callable[..., dict[str, Any]]

DVD_ASSET_ATTESTATION_MANIFEST_SCHEMA = "dream-exe.dvd-asset-attestation-manifest"


def _load_dvd_asset_attestation(
    path_value: Any,
    *,
    preset: str,
    uid: str,
) -> dict[str, str]:
    path = Path(str(path_value or "")).expanduser().resolve(strict=False)
    if not path.is_file() or path.is_symlink():
        raise FileNotFoundError(
            f"DVD asset attestation manifest is unavailable: {path.as_posix()}"
        )
    encoded = path.read_bytes()
    if len(encoded) > 4 * 1024 * 1024:
        raise ValueError("DVD asset attestation manifest exceeds 4 MiB")
    try:
        payload = json.loads(encoded.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(
            "DVD asset attestation manifest is not valid UTF-8 JSON"
        ) from error
    if not isinstance(payload, Mapping):
        raise TypeError("DVD asset attestation manifest must be an object")
    unknown = sorted(
        set(payload).difference({"format", "presets", "single_task_by_uid"})
    )
    if unknown:
        raise ValueError(
            "DVD asset attestation manifest contains unsupported fields: "
            + ", ".join(unknown)
        )
    if payload.get("format") != DVD_ASSET_ATTESTATION_MANIFEST_SCHEMA:
        raise ValueError("unsupported DVD asset attestation manifest schema")
    table_name = (
        "single_task_by_uid" if preset == "dvd_lora_specific" else "presets"
    )
    raw_table = payload.get(table_name, {})
    if not isinstance(raw_table, Mapping):
        raise TypeError(
            f"DVD asset attestation manifest.{table_name} must be an object"
        )
    key = uid if table_name == "single_task_by_uid" else preset
    raw_attestation = raw_table.get(key)
    if raw_attestation is None:
        raise KeyError(
            f"DVD asset attestation manifest has no entry for {table_name}[{key!r}]"
        )
    return normalize_dvd_asset_attestation(
        raw_attestation,
        source=f"DVD asset attestation manifest.{table_name}[{key!r}]",
    )


def _resolve_current_controller_step_budgets(
    **options: Any,
) -> tuple[float, float]:
    """Bridge simulator controller limits into the algorithm at composition."""

    from ...sim.runtime.controller import resolve_action_step_budgets

    return resolve_action_step_budgets(**options)


def _bind_step_budget_resolver(
    run_options: Mapping[str, Any],
) -> dict[str, Any]:
    """Supply the current simulator resolver without coupling video2traj."""

    resolved = copy.deepcopy(dict(run_options))
    raw_composition = resolved.get(
        "composition_options",
        {},
    )
    if raw_composition is None:
        raw_composition = {}
    if not isinstance(raw_composition, Mapping):
        raise TypeError("run_options.composition_options must be a mapping")
    composition = dict(raw_composition)
    if (
        "controller_step_budgets" not in composition
        and "step_budget_resolver" not in composition
    ):
        composition["step_budget_resolver"] = _resolve_current_controller_step_budgets
    resolved["composition_options"] = composition
    return resolved


def _bind_benchmark_tracking_output_dir(
    run_options: dict[str, Any],
    *,
    trajectory_root: Path,
) -> None:
    """Keep requested tracking media inside the formal trajectory run."""

    if not bool(run_options.get("write_tracking_artifacts", False)):
        return
    canonical = trajectory_artifact_paths(trajectory_root)["tracking_dir"].resolve()
    if "tracking_output_dir" in run_options:
        raw_output_dir = run_options["tracking_output_dir"]
        if not str(raw_output_dir or "").strip():
            raise ValueError(
                "run_options.tracking_output_dir must match the canonical "
                "benchmark trajectory tracking directory when "
                "write_tracking_artifacts=True"
            )
        requested = Path(raw_output_dir).expanduser().resolve()
        if requested != canonical:
            raise ValueError(
                "run_options.tracking_output_dir must match the canonical "
                "benchmark trajectory tracking directory: "
                f"{canonical}"
            )
    run_options["tracking_output_dir"] = canonical.as_posix()


def _load_json_object(path: Path, *, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(f"{label} is not valid JSON: {path}: {error}") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return payload


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _generation_conditioning_record(
    *,
    source_video: Path,
    sample_root: Path,
) -> dict[str, Any] | None:
    sidecar_path = source_video.with_suffix(".generation.json")
    if not sidecar_path.exists():
        return None
    sidecar = _load_json_object(
        sidecar_path,
        label="generated-video sidecar",
    )
    if sidecar.get("status") != "completed":
        raise ValueError("generated-video sidecar status must be 'completed'")
    recorded_video = (
        Path(str(sidecar.get("output_video", "") or "")).expanduser().resolve()
    )
    if recorded_video != source_video:
        raise ValueError("generated-video sidecar output_video mismatch")
    recorded_video_sha = str(sidecar.get("video_sha256", "") or "")
    if recorded_video_sha != _sha256_file(source_video):
        raise ValueError("generated-video sidecar video_sha256 mismatch")
    init_image = (sample_artifact_paths(sample_root)["initialization_image"]).resolve()
    recorded_image = (
        Path(str(sidecar.get("image_path", "") or "")).expanduser().resolve()
    )
    if recorded_image != init_image or not init_image.is_file():
        raise ValueError("generated-video sidecar conditioning image mismatch")
    if str(sidecar.get("image_sha256", "") or "") != _sha256_file(init_image):
        raise ValueError("generated-video sidecar image_sha256 mismatch")
    backend_result = sidecar.get("backend_result")
    if not isinstance(backend_result, Mapping):
        return None
    raw_transform = backend_result.get("conditioning_transform")
    if raw_transform is None:
        return None
    if not isinstance(raw_transform, Mapping):
        raise TypeError("conditioning_transform must be a mapping")
    transform = normalize_conditioning_transform(raw_transform)
    return {
        "transform": transform,
        "provenance": {
            "sidecar_path": sidecar_path.as_posix(),
            "sidecar_sha256": _sha256_file(sidecar_path),
            "video_sha256": recorded_video_sha,
            "image_sha256": str(sidecar["image_sha256"]),
            "backend": str(sidecar.get("backend", "") or ""),
            "backend_identity": copy.deepcopy(
                dict(sidecar.get("backend_identity", {}) or {})
            ),
            "transform": copy.deepcopy(transform),
        },
    }


def _runtime_config(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(
            "runtime_config must be an explicit mapping; benchmark runs "
            "do not discover model checkpoints or runtime assets"
        )
    return copy.deepcopy(dict(value))


def _canonical_depth_selection(value: Any) -> str:
    requested = str(value or DEFAULT_DEPTH_PRESET).strip().lower()
    external_backend_id = external_depth_backend_id(requested)
    if external_backend_id is not None:
        return external_depth_selection(external_backend_id)
    if requested in SUPPORTED_DEPTH_BACKENDS:
        return requested
    return resolve_depth_estimator_preset(requested)


def _bind_benchmark_depth_runtime(
    value: Mapping[str, Any] | None,
    *,
    uid: str,
    pipeline_config: Mapping[str, Any],
    use_rollout_gt_depth: bool,
) -> dict[str, Any]:
    """Bind the caller's assets/provider config to the resolved bench choice."""

    resolved = _runtime_config(value)
    if bool(use_rollout_gt_depth):
        raw_depth = resolved.get("depth")
        if (
            isinstance(raw_depth, Mapping)
            and str(raw_depth.get("backend", "") or "").strip().lower()
            == BENCH_RESOLVED_DEPTH_BACKEND
        ):
            # GT-depth consumes the explicit rollout array and must not build
            # either optional model provider.  Consume the benchmark-only
            # router marker before the core runtime validates its schema.
            resolved["depth"] = {"backend": "none"}
        return resolved
    raw_depth = resolved.get("depth")
    if not isinstance(raw_depth, Mapping):
        raise TypeError("runtime_config.depth must be a mapping for a depth-model run")
    pipeline_depth = dict(pipeline_config.get("depth", {}) or {})
    requested = _canonical_depth_selection(
        pipeline_depth.get("model", DEFAULT_DEPTH_PRESET)
    )
    depth_runtime = select_benchmark_depth_provider(
        raw_depth,
        requested_preset=requested,
    )
    existing = str(depth_runtime.get("preset", "") or "").strip()
    if existing and existing != BENCH_RESOLVED_DEPTH_PRESET:
        existing_canonical = _canonical_depth_selection(existing)
        if existing_canonical != requested:
            raise ValueError(
                "runtime_config.depth.preset conflicts with the resolved "
                "bench pipeline selection: "
                f"runtime={existing_canonical!r}, pipeline={requested!r}"
            )
    depth_runtime["preset"] = requested
    depth_runtime["selection_source"] = "bench_resolved_pipeline_config"
    manifest_path = depth_runtime.pop("attestation_manifest_path", None)
    if manifest_path is not None:
        if (
            depth_runtime.get("config") is not None
            or str(depth_runtime.get("config_path", "") or "").strip()
        ):
            raise ValueError(
                "depth.attestation_manifest_path cannot be combined with an "
                "explicit depth config"
            )
        if requested not in {
            "dvd_official",
            "dvd_lora_shared",
            "dvd_lora_specific",
        }:
            raise ValueError(
                "depth.attestation_manifest_path is valid only for a built-in "
                "DVD preset"
            )
        manifest = Path(str(manifest_path)).expanduser()
        if not manifest.is_absolute():
            raw_meta = resolved.get("_meta", {})
            meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
            base_dir = str(meta.get("base_dir", "") or "").strip()
            if not base_dir:
                raise ValueError(
                    "relative depth.attestation_manifest_path requires "
                    "file-backed runtime metadata"
                )
            manifest = Path(base_dir) / manifest
        estimator_config = default_depth_estimator_config(requested)
        provenance = dict(estimator_config["model_provenance"])
        provenance["asset_attestation"] = _load_dvd_asset_attestation(
            manifest,
            preset=requested,
            uid=uid,
        )
        estimator_config["model_provenance"] = provenance
        depth_runtime["config"] = estimator_config
    if requested == "dvd_lora_specific":
        raw_context = depth_runtime.get("runtime_context", {})
        if raw_context is None:
            raw_context = {}
        if not isinstance(raw_context, Mapping):
            raise TypeError("runtime_config.depth.runtime_context must be a mapping")
        context = copy.deepcopy(dict(raw_context))
        existing_uid = str(context.get("uid", "") or "").strip()
        if existing_uid and existing_uid != uid:
            raise ValueError(
                "runtime_config.depth.runtime_context.uid conflicts with "
                f"the benchmark sample: {existing_uid!r} != {uid!r}"
            )
        context["uid"] = uid
        depth_runtime["runtime_context"] = context
    resolved["depth"] = depth_runtime
    return resolved


def preflight_benchmark_depth_runtime(
    *,
    uid: str,
    sample_dir: str | Path,
    run_key: str,
    gen_model: str,
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    """Resolve one run's depth provider and attestation without writes.

    Suite orchestration calls this before creating output or scratch roots so
    a missing per-UID DVD attestation cannot surface only after expensive
    sample materialization.  The same current bench config precedence and
    provider binding used by the real stage are exercised here.
    """

    clean_uid = str(uid or "").strip()
    if not clean_uid:
        raise ValueError("uid is required")
    sample_root = Path(sample_dir).expanduser().resolve()
    if not sample_root.is_dir() or sample_root.name != clean_uid:
        raise ValueError("benchmark uid/sample_dir mismatch")
    normalized_run_key = normalize_run_key(run_key)
    clean_gen_model = str(gen_model or "").strip()
    video_kind, _slot = split_run_key(normalized_run_key)
    if video_kind == "rollout" and clean_gen_model:
        raise ValueError("rollout benchmark runs cannot name a gen_model")
    paths = sample_artifact_paths(sample_root)
    pipeline_config = load_bench_pipeline_config(
        sample_dir=sample_root.as_posix(),
        pipeline_config_path=paths["trajectory_config"].as_posix(),
        run_key=normalized_run_key,
        gen_model=clean_gen_model,
    )
    return _bind_benchmark_depth_runtime(
        runtime_config,
        uid=clean_uid,
        pipeline_config=pipeline_config,
        use_rollout_gt_depth=run_key_uses_gt_depth(normalized_run_key),
    )


def _needs_saved_regions(
    pipeline_config: Mapping[str, Any],
    *,
    uid: str,
    metadata: Mapping[str, Any],
) -> bool:
    targets = dict(
        dict(pipeline_config.get("region", {}) or {}).get(
            "targets",
            {},
        )
        or {}
    )
    entries = [targets.get("eef", {})]
    task_runtime = compile_task_runtime(
        uid=str(uid),
        metadata=copy.deepcopy(dict(metadata)),
        pipeline_config=copy.deepcopy(dict(pipeline_config)),
    )
    entries.extend(
        dict(stream.get("region_target_cfg", {}) or {})
        for stream in list(task_runtime.get("object_stream_plan", []) or [])
        if isinstance(stream, Mapping)
    )
    return any(
        str(dict(entry or {}).get("selector", "") or "").strip()
        in {"simulation", "auto"}
        for entry in entries
        if isinstance(entry, Mapping)
    )


def _load_benchmark_init_depth(
    sample_root: Path,
) -> np.ndarray | None:
    paths = resolve_benchmark_init_region_assets(sample_root)
    depth_path = Path(paths["init_depth_path"])
    if not depth_path.is_file():
        return None
    with depth_path.open("rb") as stream:
        depth = np.asarray(
            np.load(stream, allow_pickle=False),
            dtype=np.float32,
        )
    if depth.ndim != 2 or any(int(size) <= 0 for size in depth.shape):
        raise ValueError(
            f"bench init depth must be a non-empty [H,W] array: {depth.shape}"
        )
    return depth


def _bind_benchmark_init_depth(
    run_options: dict[str, Any],
    *,
    init_depth: np.ndarray,
) -> None:
    for section_name in ("depth_options", "target_depth_options"):
        raw = run_options.get(section_name, {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError(f"run_options.{section_name} must be a mapping")
        options = copy.deepcopy(dict(raw))
        if "init_ref_depth" in options:
            raise ValueError(
                f"run_options.{section_name}.init_ref_depth is owned by "
                "the benchmark init-depth adapter"
            )
        options["init_ref_depth"] = init_depth.copy()
        run_options[section_name] = options


def summarize_benchmark_video2traj_request(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the bounded path/provenance portion of a resolved request."""

    return {
        key: request.get(key)
        for key in (
            "uid",
            "run_id",
            "run_key",
            "video_kind",
            "gen_model",
            "video_path",
            "simulator_config_path",
            "pipeline_config_path",
            "runtime_asset_base",
            "video_backend",
            "output_dir",
        )
    } | {
        "region_inputs": copy.deepcopy(
            dict(request.get("region_inputs_manifest", {}) or {})
        ),
        "conditioning_transform": copy.deepcopy(request.get("conditioning_transform")),
    }


def resolve_benchmark_video2traj_request(
    *,
    uid: str,
    sample_dir: str | Path,
    run_id: str,
    run_key: str,
    video_kind: str,
    gen_model: str,
    source_video_path: str | Path,
    pipeline_config_path: str | Path,
    output_traj_root: str | Path,
    use_rollout_gt_depth: bool,
    runtime_config: Mapping[str, Any],
    simulator_config_path: str | Path | None = None,
    runtime_asset_base: str | Path | None = None,
    run_options: Mapping[str, Any] | None = None,
    video_backend: str = "auto",
) -> dict[str, Any]:
    """Resolve one matrix trajectory stage without creating output paths."""

    clean_uid = str(uid or "").strip()
    if not clean_uid:
        raise ValueError("uid is required")
    sample_root = Path(sample_dir).expanduser().resolve()
    sample_paths = sample_artifact_paths(sample_root)
    if not sample_root.is_dir():
        raise FileNotFoundError(
            f"explicit bench sample directory not found: {sample_root}"
        )
    if sample_root.name != clean_uid:
        raise ValueError(
            f"benchmark uid/sample_dir mismatch: {clean_uid!r} != {sample_root.name!r}"
        )
    sample_metadata = _load_json_object(
        sample_paths["sample_metadata"],
        label="sample metadata",
    )
    metadata_uid = str(sample_metadata.get("uid", "") or "").strip()
    if metadata_uid != clean_uid:
        raise ValueError(
            f"benchmark uid/sample metadata mismatch: {clean_uid!r} != {metadata_uid!r}"
        )

    normalized_run_key = normalize_run_key(run_key)
    expected_video_kind, _slot = split_run_key(normalized_run_key)
    clean_video_kind = str(video_kind or "").strip()
    if clean_video_kind != expected_video_kind:
        raise ValueError("benchmark video_kind does not match run_key")
    clean_gen_model = str(gen_model or "").strip()
    if expected_video_kind == "rollout" and clean_gen_model:
        raise ValueError("rollout benchmark runs cannot name a gen_model")
    expected_run_id = build_run_id(
        run_key=normalized_run_key,
        gen_model=clean_gen_model,
    )
    if str(run_id or "").strip() != expected_run_id:
        raise ValueError("benchmark run_id does not match run identity")

    expected_gt_depth = bool(run_key_uses_gt_depth(normalized_run_key))
    if bool(use_rollout_gt_depth) != expected_gt_depth:
        raise ValueError("use_rollout_gt_depth does not match the formal run key")

    source_video = Path(source_video_path).expanduser().resolve()
    if not source_video.is_file():
        raise FileNotFoundError(f"explicit source video not found: {source_video}")
    expected_rollout_video = sample_paths["gt_video"].resolve()
    if expected_video_kind == "rollout" and source_video != expected_rollout_video:
        raise ValueError(
            "reference source_video_path must be the current GT video path"
        )
    if expected_video_kind == "gen":
        enhanced_suffix = "-enhanced"
        is_enhanced = clean_gen_model.lower().endswith(enhanced_suffix)
        expected_filename_model = (
            clean_gen_model[: -len(enhanced_suffix)] if is_enhanced else clean_gen_model
        )
        expected_generated_root = sample_paths[
            "generated_enhanced_root" if is_enhanced else "generated_root"
        ].resolve()
        if not source_video.is_relative_to(expected_generated_root):
            raise ValueError(
                "generated source_video_path must remain under the "
                "current sample generated-artifact root"
            )
        if normalize_generated_model_name(source_video.name) != expected_filename_model:
            raise ValueError(
                "generated source_video_path filename does not match gen_model"
            )
    conditioning_record = (
        _generation_conditioning_record(
            source_video=source_video,
            sample_root=sample_root,
        )
        if expected_video_kind == "gen"
        else None
    )

    pipeline_path = Path(pipeline_config_path).expanduser().resolve()
    if not pipeline_path.is_file():
        raise FileNotFoundError(f"pipeline config not found: {pipeline_path}")
    trajectory_root = Path(output_traj_root).expanduser().resolve()
    expected_trajectory_root = formal_artifact_paths(
        sample_root,
        normalized_run_key,
        gen_model=clean_gen_model,
    )["traj_dir"]
    if trajectory_root != expected_trajectory_root:
        raise ValueError("output_traj_root does not match the formal benchmark run")

    simulator_path = (
        Path(simulator_config_path).expanduser().resolve()
        if simulator_config_path is not None and str(simulator_config_path).strip()
        else sample_paths["simulator_config"].resolve()
    )
    simulator_config = _load_json_object(
        simulator_path,
        label="simulator config",
    )

    source_config_capture: dict[str, Any] = {}
    pipeline_config = load_bench_pipeline_config(
        sample_dir=sample_root.as_posix(),
        pipeline_config_path=pipeline_path.as_posix(),
        run_key=normalized_run_key,
        gen_model=clean_gen_model,
        _source_config_out=source_config_capture,
    )
    source_pipeline_config = source_config_capture.get("config", {})
    if not isinstance(source_pipeline_config, Mapping):
        raise TypeError("bench source pipeline config must be a mapping")
    raw_region_value = source_pipeline_config.get("region", {})
    if raw_region_value is None:
        raw_region_value = {}
    if not isinstance(raw_region_value, Mapping):
        raise TypeError("bench source pipeline region must be a mapping")
    source_region_config = copy.deepcopy(dict(raw_region_value))
    region_binding = resolve_pipeline_region_selector_policy(
        source_region_config,
        pipeline_asset_base=pipeline_path.parent,
    )
    pipeline_meta = copy.deepcopy(dict(pipeline_config.get("_meta", {}) or {}))
    pipeline_for_validation = copy.deepcopy(dict(pipeline_config))
    pipeline_for_validation.pop("_meta", None)
    pipeline_for_validation["region"] = copy.deepcopy(
        dict(region_binding["pipeline_region"])
    )
    pipeline_config = load_core_pipeline_config(
        pipeline_for_validation,
        source=str(pipeline_meta.get("source", "") or pipeline_path.as_posix()),
    )
    pipeline_config["_meta"] = pipeline_meta
    source_pose_config = (
        copy.deepcopy(dict(source_pipeline_config.get("pose", {}) or {}))
        if isinstance(source_pipeline_config, Mapping)
        and isinstance(source_pipeline_config.get("pose", {}), Mapping)
        else {}
    )
    path_defaults = resolve_default_bench_pipeline_paths(
        sample_dir=sample_root.as_posix(),
        run_key=normalized_run_key,
        gen_model=clean_gen_model,
        output_traj_root=trajectory_root.as_posix(),
    )
    inputs = pipeline_config.setdefault("input", {})
    inputs["selected_video"] = expected_video_kind
    if expected_video_kind == "rollout":
        inputs["rollout_video_path"] = source_video.as_posix()
        inputs["gen_video_path"] = ""
    else:
        inputs["gen_video_path"] = source_video.as_posix()

    depth = pipeline_config.setdefault("depth", {})
    depth["use_rollout_gt_depth"] = expected_gt_depth
    depth["rollout_gt_depth_path"] = path_defaults["rollout_gt_depth_path"]
    depth["estimated_depth_cache_path"] = path_defaults["estimated_depth_cache_path"]
    depth["estimated_depth_cache_meta_path"] = path_defaults["depth_meta_path"]
    depth["visualization_video_path"] = path_defaults["visualization_video_path"]

    if run_options is None:
        resolved_run_options: dict[str, Any] = {}
    elif isinstance(run_options, Mapping):
        resolved_run_options = copy.deepcopy(dict(run_options))
    else:
        raise TypeError("run_options must be a mapping")
    _bind_benchmark_tracking_output_dir(
        resolved_run_options,
        trajectory_root=trajectory_root,
    )
    if conditioning_record is not None:
        if "conditioning_transform" in resolved_run_options:
            raise ValueError(
                "run_options.conditioning_transform is owned by the "
                "generated-video sidecar"
            )
        resolved_run_options["conditioning_transform"] = copy.deepcopy(
            conditioning_record["transform"]
        )
    if "depth_publication_options" in resolved_run_options:
        raise ValueError(
            "run_options.depth_publication_options is owned by the "
            "benchmark path adapter"
        )
    resolved_run_options["depth_publication_options"] = {
        "cache_path": path_defaults["canonical_depth_npy_path"],
        "meta_path": path_defaults["depth_meta_path"],
        "manifest_path": path_defaults["depth_manifest_path"],
        "transaction_root": path_defaults["sample_dir"],
        "overwrite": bool(depth.get("force_recompute", False)),
        "legacy_metadata_policy": "trusted_pickle_read_only",
        "trusted_legacy_root": path_defaults["sample_dir"],
    }
    region_inputs_manifest: dict[str, Any] = {}
    bench_init_depth: np.ndarray | None = None
    if _needs_saved_regions(
        pipeline_config,
        uid=clean_uid,
        metadata=sample_metadata,
    ):
        region_inputs = build_benchmark_region_inputs(
            uid=clean_uid,
            sample_dir=sample_root,
            simulator_config=simulator_config,
            pipeline_config=pipeline_config,
            metadata=sample_metadata,
        )
        region_options = dict(region_inputs["object_runtime_options"])
        bench_init_depth = region_options.get("init_depth", None)
        existing_object_options = resolved_run_options.get(
            "object_runtime_options",
            {},
        )
        if existing_object_options is None:
            existing_object_options = {}
        if not isinstance(existing_object_options, Mapping):
            raise TypeError("run_options.object_runtime_options must be a mapping")
        existing_object_options = dict(existing_object_options)
        region_owned = {
            "eef_precomputed_region",
            "precomputed_regions",
            "init_depth",
        }
        overlap = sorted(region_owned.intersection(existing_object_options))
        if overlap:
            raise ValueError(
                "run_options.object_runtime_options cannot replace "
                "bench-owned init region inputs: " + ", ".join(overlap)
            )
        existing_object_options.update(region_options)
        resolved_run_options["object_runtime_options"] = existing_object_options
        region_inputs_manifest = copy.deepcopy(dict(region_inputs["manifest"]))
    if not expected_gt_depth:
        if bench_init_depth is None:
            bench_init_depth = _load_benchmark_init_depth(sample_root)
        if bench_init_depth is not None:
            _bind_benchmark_init_depth(
                resolved_run_options,
                init_depth=np.asarray(
                    bench_init_depth,
                    dtype=np.float32,
                ),
            )
    run_metadata = copy.deepcopy(sample_metadata)
    run_metadata.update(
        {
            "uid": clean_uid,
            "run_id": expected_run_id,
            "run_key": normalized_run_key,
            "video_kind": expected_video_kind,
            "gen_model": clean_gen_model or None,
            "source_video_path": source_video.as_posix(),
            "pipeline_config_path": pipeline_path.as_posix(),
            "region_inputs": region_inputs_manifest,
        }
    )
    if conditioning_record is not None:
        run_metadata["conditioning_transform"] = copy.deepcopy(
            conditioning_record["provenance"]
        )
    owned_options = {
        "uid": clean_uid,
        "simulator_config_source": simulator_path.as_posix(),
        "metadata": run_metadata,
    }
    overlap = sorted(set(owned_options).intersection(resolved_run_options))
    if overlap:
        raise ValueError(
            "run_options cannot replace benchmark-owned provenance: "
            + ", ".join(overlap)
        )
    resolved_run_options.update(owned_options)

    asset_base: str | None
    if runtime_asset_base is None or not str(runtime_asset_base).strip():
        asset_base = None
    else:
        asset_path = Path(runtime_asset_base).expanduser()
        if not asset_path.is_absolute():
            raise ValueError("runtime_asset_base must be an absolute path")
        asset_base = asset_path.resolve(strict=False).as_posix()

    return {
        "uid": clean_uid,
        "run_id": expected_run_id,
        "run_key": normalized_run_key,
        "video_kind": expected_video_kind,
        "gen_model": clean_gen_model,
        "video_path": source_video.as_posix(),
        "simulator_config": simulator_config,
        "simulator_config_path": simulator_path.as_posix(),
        "pipeline_config": pipeline_config,
        "pipeline_pose_config": source_pose_config,
        "pipeline_region_config": source_region_config,
        "pipeline_config_path": pipeline_path.as_posix(),
        "runtime_config": _bind_benchmark_depth_runtime(
            runtime_config,
            uid=clean_uid,
            pipeline_config=pipeline_config,
            use_rollout_gt_depth=expected_gt_depth,
        ),
        "runtime_asset_base": asset_base,
        "run_options": resolved_run_options,
        "video_backend": str(video_backend or "auto"),
        "output_dir": trajectory_root.as_posix(),
        "region_inputs_manifest": region_inputs_manifest,
        "conditioning_transform": (
            None
            if conditioning_record is None
            else copy.deepcopy(conditioning_record["provenance"])
        ),
    }


def execute_benchmark_video2traj_stage(
    *,
    runtime_config: Mapping[str, Any],
    dependencies: Mapping[str, Any] | None = None,
    runtime_asset_base: str | Path | None = None,
    run_options: Mapping[str, Any] | None = None,
    video_backend: str = "auto",
    simulator_config_path: str | Path | None = None,
    runner: StandaloneRunner | None = None,
    include_result: bool = False,
    **matrix_options: Any,
) -> dict[str, Any]:
    """Bind matrix trajectory options to the public standalone callable.

    ``runtime_config`` is intentionally not discovered from the bench.  A
    caller binds it, along with optional injected dependencies, before passing
    this function to the formal single-case workflow.
    """

    request = resolve_benchmark_video2traj_request(
        runtime_config=runtime_config,
        runtime_asset_base=runtime_asset_base,
        run_options=run_options,
        video_backend=video_backend,
        simulator_config_path=simulator_config_path,
        **matrix_options,
    )
    run = run_standalone_video2traj if runner is None else runner
    bench_bridge_options = (
        {"_allow_bench_legacy_depth_bridge": True} if runner is None else {}
    )
    result = run(
        video_path=request["video_path"],
        simulator_config=copy.deepcopy(request["simulator_config"]),
        pipeline_config=copy.deepcopy(request["pipeline_config"]),
        pipeline_pose_config=copy.deepcopy(request["pipeline_pose_config"]),
        pipeline_region_config=copy.deepcopy(request["pipeline_region_config"]),
        runtime_config=copy.deepcopy(request["runtime_config"]),
        dependencies=(None if dependencies is None else dict(dependencies)),
        runtime_asset_base=request["runtime_asset_base"],
        pipeline_asset_base=Path(request["pipeline_config_path"]).parent,
        run_options=_bind_step_budget_resolver(request["run_options"]),
        video_backend=request["video_backend"],
        write_artifacts=True,
        output_dir=request["output_dir"],
        **bench_bridge_options,
    )
    response = {
        "ok": True,
        "returncode": 0,
        "uid": request["uid"],
        "run_id": request["run_id"],
        "run_key": request["run_key"],
        "gen_model": request["gen_model"],
        "request": summarize_benchmark_video2traj_request(request),
        "result_summary": summarize_standalone_video2traj_result(
            result,
            portable_roots={
                "output": request["output_dir"],
                "sample": request["run_options"]["depth_publication_options"][
                    "trusted_legacy_root"
                ],
            },
        ),
    }
    if bool(include_result):
        response["result"] = result
    return response


__all__ = [
    "execute_benchmark_video2traj_stage",
    "preflight_benchmark_depth_runtime",
    "resolve_benchmark_video2traj_request",
    "summarize_benchmark_video2traj_request",
]
