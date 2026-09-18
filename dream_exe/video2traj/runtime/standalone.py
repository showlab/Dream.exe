"""Standalone raw-video entry point over the public multi-object runtime."""

from __future__ import annotations

import copy
import hashlib
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..depth.estimator import (
    DEFAULT_DEPTH_PRESET,
    SUPPORTED_DEPTH_BACKENDS,
    default_depth_estimator_config,
    resolve_depth_estimator_preset,
)
from ..depth.config import bind_depth_base_runtime_calibration
from ..depth.contract import (
    external_depth_backend_id,
    external_depth_selection,
)
from ..media.video import read_video_frames
from ..pose.config import (
    effective_pose_backend,
    pose_backend_identity_is_known,
)
from ..pose.contract import validate_external_pose_runtime_identity
from ..region.config import resolve_pipeline_region_selector_policy
from ..trajectory.stages import compile_task_runtime
from .builder import build_video2traj_runtime
from .config import load_pipeline_config
from .multi_object import run_multi_object_video_file
from .pose_bridge import (
    prepare_standalone_pose_pipeline,
    runtime_pose_estimator_owns_configuration,
)

_OWNED_RUN_OPTIONS = {
    "depth_estimator",
    "depth_calibration",
    "output_dir",
    "pipeline_config",
    "pipeline_asset_base",
    "pipeline_pose_config",
    "pipeline_region_config",
    "pose_backend",
    "pose_estimator",
    "rigid_pose_backend",
    "region_runtime",
    "simulator_config",
    "tracking_backend",
    "video_path",
    "write_artifacts",
}

_REGION_PROVIDER_FIELDS = {
    "grounding_dino": {
        "use_local_repo",
        "local_config_path",
        "local_checkpoint_path",
        "local_text_encoder_path",
        "box_threshold",
        "text_threshold",
    },
    "sam2": {
        "config_name",
        "checkpoint_path",
    },
}
_REGION_PROVIDER_FIELD_MAP = {
    "grounding_dino": {
        "local_config_path": "config_path",
        "local_checkpoint_path": "checkpoint_path",
        "local_text_encoder_path": "text_encoder_path",
        "box_threshold": "box_threshold",
        "text_threshold": "text_threshold",
    },
    "sam2": {
        "config_name": "config_name",
        "checkpoint_path": "checkpoint_path",
    },
}
_REGION_PROVIDER_PATH_FIELDS = {
    "grounding_dino": {
        "local_config_path",
        "local_checkpoint_path",
        "local_text_encoder_path",
    },
    "sam2": {"checkpoint_path"},
}


def _run_options(
    value: Mapping[str, Any] | None,
    *,
    allow_bench_legacy_depth_bridge: bool = False,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("run_options must be a mapping")
    options = dict(value)
    overlap = sorted(_OWNED_RUN_OPTIONS.intersection(options))
    if overlap:
        raise ValueError(
            "run_options cannot replace standalone-owned inputs: " + ", ".join(overlap)
        )
    depth_publication = options.get("depth_publication_options", None)
    if isinstance(depth_publication, Mapping):
        legacy_fields = {
            "legacy_metadata_policy",
            "trusted_legacy_root",
        }.intersection(depth_publication)
        if legacy_fields and not allow_bench_legacy_depth_bridge:
            raise ValueError(
                "standalone depth_publication_options cannot enable legacy "
                "metadata trust; legacy object-NPY reads are owned by the "
                "current-bench adapter"
            )
    return options


def _runtime_pose_identity(
    runtime_pose: Mapping[str, Any],
) -> tuple[str, str]:
    """Return a validated provider seam and its effective backend identity."""

    provider = str(runtime_pose.get("backend", "") or "").strip().lower()
    if provider in {"injected", "factory"}:
        identity = validate_external_pose_runtime_identity(
            runtime_pose,
            source="runtime manifest components.pose",
            require_effective_backend=True,
        )
        return provider, str(identity["effective_backend"])
    effective_backend = (
        str(
            runtime_pose.get(
                "effective_backend",
                runtime_pose.get("backend", ""),
            )
            or ""
        )
        .strip()
        .lower()
    )
    return provider, effective_backend


def _region_target_provider_roles(
    pipeline_config: Mapping[str, Any],
    *,
    run_options: Mapping[str, Any],
) -> dict[str, list[str]]:
    region = dict(pipeline_config.get("region", {}) or {})
    targets = dict(region.get("targets", {}) or {})
    object_options = run_options.get("object_runtime_options", {})
    if not isinstance(object_options, Mapping):
        object_options = {}
    object_precomputed = object_options.get("precomputed_regions", {})
    if not isinstance(object_precomputed, Mapping):
        object_precomputed = {}
    entries: list[tuple[str, str, Mapping[str, Any], bool]] = []
    eef_value = targets.get("eef", {})
    if isinstance(eef_value, Mapping):
        entries.append(
            (
                "eef",
                "eef",
                eef_value,
                object_options.get("eef_precomputed_region") is not None,
            )
        )
    raw_metadata = run_options.get("metadata", {})
    if raw_metadata is None:
        raw_metadata = {}
    if not isinstance(raw_metadata, Mapping):
        raise TypeError("run_options.metadata must be a mapping")
    task_runtime = compile_task_runtime(
        uid=str(run_options.get("uid", "") or ""),
        metadata=copy.deepcopy(dict(raw_metadata)),
        pipeline_config=copy.deepcopy(dict(pipeline_config)),
    )
    for index, stream_value in enumerate(
        list(task_runtime.get("object_stream_plan", []) or [])
    ):
        if not isinstance(stream_value, Mapping):
            continue
        stream = dict(stream_value)
        value = stream.get("region_target_cfg", {})
        if not isinstance(value, Mapping):
            continue
        object_id = str(stream.get("object_id", "") or "").strip()
        label = object_id or f"object_stream[{index}]"
        entries.append(
            (
                label,
                "obj",
                value,
                object_precomputed.get(object_id or "obj") is not None,
            )
        )

    required: dict[str, list[str]] = {
        "detector": [],
        "segmenter": [],
    }
    for label, target_name, value, has_precomputed in entries:
        target = dict(value)
        selector = str(target.get("selector", "") or "").strip().lower()
        uses_visual_route = (
            selector == "visual"
            or (selector == "auto" and not has_precomputed)
            or (
                selector == "simulation"
                and target_name == "obj"
                and not has_precomputed
            )
        )
        if not uses_visual_route:
            continue
        visual = dict(target.get("visual", {}) or {})
        sampling = dict(target.get("sampling", {}) or {})
        bbox_source = (
            str(visual.get("bbox_source", "grounding_dino") or "grounding_dino")
            .strip()
            .lower()
        )
        sampling_method = (
            str(sampling.get("method", "mask_3d_fps") or "mask_3d_fps").strip().lower()
        )
        if bbox_source == "grounding_dino":
            required["detector"].append(label)
        if sampling_method != "bbox_gaussian":
            required["segmenter"].append(label)
    return required


def _region_runtime_roles(
    region_config: Mapping[str, Any],
) -> tuple[str, dict[str, dict[str, Any]]]:
    backend = str(region_config.get("backend", "") or "").strip().lower()

    def role(
        name: str,
        config: Mapping[str, Any] | None,
        *,
        expected_builtin: str,
    ) -> dict[str, Any]:
        payload = copy.deepcopy(dict(config or {}))
        selected = str(payload.get("backend", "") or "").strip().lower()
        return {
            "role": name,
            "backend": selected,
            "config": payload,
            "available": selected != "none" and bool(selected),
            "verifiable": True,
            "provider_kind": (
                "builtin"
                if selected == expected_builtin
                else ("external" if selected in {"injected", "factory"} else "unknown")
            ),
        }

    none_detector = role(
        "detector",
        {"backend": "none"},
        expected_builtin="grounding_dino",
    )
    none_segmenter = role(
        "segmenter",
        {"backend": "none"},
        expected_builtin="sam2",
    )
    if backend == "composed":
        detector = region_config.get("detector")
        segmenter = region_config.get("segmenter")
        if not isinstance(detector, Mapping):
            detector = {}
        if not isinstance(segmenter, Mapping):
            segmenter = {}
        return (
            backend,
            {
                "detector": role(
                    "detector",
                    detector,
                    expected_builtin="grounding_dino",
                ),
                "segmenter": role(
                    "segmenter",
                    segmenter,
                    expected_builtin="sam2",
                ),
            },
        )
    if backend in {"injected", "factory"}:
        return (
            backend,
            {
                name: {
                    "role": name,
                    "backend": backend,
                    "config": copy.deepcopy(dict(region_config)),
                    "available": True,
                    "verifiable": False,
                    "provider_kind": "external_full_runtime",
                }
                for name in ("detector", "segmenter")
            },
        )

    detector = none_detector
    segmenter = none_segmenter
    if backend in {"grounding_dino", "grounding_dino_sam2"}:
        raw_detector = region_config.get("grounding_dino")
        detector = role(
            "detector",
            (
                raw_detector
                if isinstance(raw_detector, Mapping)
                else {"backend": "grounding_dino"}
            ),
            expected_builtin="grounding_dino",
        )
        detector["backend"] = "grounding_dino"
        detector["available"] = True
        detector["provider_kind"] = "builtin"
    manual_sam2 = backend == "manual+sam2" or (
        backend == "manual" and region_config.get("sam2") is not None
    )
    if backend in {"sam2", "grounding_dino_sam2"} or manual_sam2:
        raw_segmenter = region_config.get("sam2")
        segmenter = role(
            "segmenter",
            (
                raw_segmenter
                if isinstance(raw_segmenter, Mapping)
                else {"backend": "sam2"}
            ),
            expected_builtin="sam2",
        )
        segmenter["backend"] = "sam2"
        segmenter["available"] = True
        segmenter["provider_kind"] = "builtin"
    return backend, {"detector": detector, "segmenter": segmenter}


def _runtime_region_asset_base(
    runtime_config: Mapping[str, Any],
    *,
    runtime_asset_base: str | Path | None,
) -> Path | None:
    if runtime_asset_base is not None and str(runtime_asset_base).strip():
        base = Path(runtime_asset_base).expanduser()
        if not base.is_absolute():
            raise ValueError("runtime_asset_base must be an absolute path")
        base = base.resolve(strict=False)
    else:
        meta = runtime_config.get("_meta", {})
        meta_mapping = dict(meta) if isinstance(meta, Mapping) else {}
        base_text = str(meta_mapping.get("base_dir", "") or "").strip()
        base = Path(base_text).expanduser().resolve(strict=False) if base_text else None
    asset_root = str(runtime_config.get("asset_root", "") or "").strip()
    if not asset_root:
        return base
    root = Path(asset_root).expanduser()
    if not root.is_absolute():
        if base is None:
            raise ValueError(
                "relative runtime asset_root requires an explicit runtime_asset_base"
            )
        root = base / root
    return root.resolve(strict=False)


def _runtime_provider_path(
    value: Any,
    *,
    runtime_base: Path | None,
) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        if runtime_base is None:
            return None
        path = runtime_base / path
    return path.resolve(strict=False)


def _validate_region_provider_evidence(
    *,
    provider_name: str,
    selector_config: Mapping[str, Any],
    role_binding: Mapping[str, Any],
    active: bool,
    runtime_base: Path | None,
) -> dict[str, Any]:
    role_name = "detector" if provider_name == "grounding_dino" else "segmenter"
    fields = sorted(selector_config)
    if not fields:
        return {
            "activation": "runtime_only" if active else "not_activated",
            "role": role_name,
            "checked_fields": [],
        }
    if not active:
        return {
            "activation": "not_activated",
            "role": role_name,
            "checked_fields": [],
            "evidence_fields": fields,
        }

    unsupported = sorted(
        set(selector_config).difference(_REGION_PROVIDER_FIELDS[provider_name])
    )
    if unsupported:
        raise ValueError(
            f"active pipeline {provider_name} selector config contains "
            "fields that cannot be projected to the standalone runtime: "
            + ", ".join(unsupported)
        )
    if role_binding.get("provider_kind") != "builtin":
        raise ValueError(
            f"active pipeline {provider_name} selector config cannot be "
            f"projected to runtime {role_name} backend "
            f"{role_binding.get('backend')!r}; model/provider choice is "
            "owned by runtime_config"
        )
    expected_backend = "grounding_dino" if provider_name == "grounding_dino" else "sam2"
    if role_binding.get("backend") != expected_backend:
        raise ValueError(
            f"active pipeline {provider_name} selector config conflicts "
            f"with runtime {role_name} backend "
            f"{role_binding.get('backend')!r}"
        )
    runtime_provider = dict(role_binding.get("config", {}) or {})
    if provider_name == "grounding_dino" and "use_local_repo" in selector_config:
        if not bool(selector_config["use_local_repo"]):
            raise ValueError(
                "grounding_dino.use_local_repo=false is unsupported by the "
                "offline selector contract"
            )
        if not str(runtime_provider.get("source_root", "") or "").strip():
            raise ValueError(
                "active grounding_dino.use_local_repo=true requires "
                "runtime_config to declare the provider source_root"
            )

    checked: list[str] = []
    field_map = _REGION_PROVIDER_FIELD_MAP[provider_name]
    for selector_field, runtime_field in field_map.items():
        if selector_field not in selector_config:
            continue
        selector_value = selector_config[selector_field]
        runtime_value = runtime_provider.get(runtime_field)
        if selector_field in _REGION_PROVIDER_PATH_FIELDS[provider_name]:
            selector_path = Path(str(selector_value or "")).expanduser()
            if not selector_path.is_absolute():
                raise ValueError(
                    f"active pipeline {provider_name}.{selector_field} is "
                    "relative and cannot own a provider asset root; declare "
                    "the asset only in runtime_config"
                )
            runtime_path = _runtime_provider_path(
                runtime_value,
                runtime_base=runtime_base,
            )
            if runtime_path is None or (
                selector_path.resolve(strict=False) != runtime_path
            ):
                raise ValueError(
                    f"pipeline {provider_name}.{selector_field} conflicts "
                    f"with runtime {role_name}.{runtime_field}"
                )
        elif selector_field in {"box_threshold", "text_threshold"}:
            if runtime_value is None or float(selector_value) != float(runtime_value):
                raise ValueError(
                    f"pipeline {provider_name}.{selector_field} conflicts "
                    f"with runtime {role_name}.{runtime_field}"
                )
        elif str(selector_value) != str(runtime_value):
            raise ValueError(
                f"pipeline {provider_name}.{selector_field} conflicts "
                f"with runtime {role_name}.{runtime_field}"
            )
        checked.append(selector_field)
    if "use_local_repo" in selector_config:
        checked.append("use_local_repo")
    return {
        "activation": "validated_against_runtime",
        "role": role_name,
        "checked_fields": sorted(set(checked)),
        "evidence_fields": fields,
    }


def _bind_pipeline_region_runtime(
    runtime_config: Mapping[str, Any],
    *,
    pipeline_config: Mapping[str, Any],
    selector_binding: Mapping[str, Any],
    runtime_asset_base: str | Path | None,
    run_options: Mapping[str, Any],
) -> dict[str, Any]:
    runtime = copy.deepcopy(dict(runtime_config))
    raw_region = runtime.get("region")
    if not isinstance(raw_region, Mapping):
        raise TypeError("runtime_config.region must be a mapping")
    region_runtime = copy.deepcopy(dict(raw_region))
    backend, roles = _region_runtime_roles(region_runtime)
    required = _region_target_provider_roles(
        pipeline_config,
        run_options=run_options,
    )
    for role_name, targets in required.items():
        if not targets:
            continue
        role_binding = roles[role_name]
        if not bool(role_binding.get("verifiable", False)):
            # A versioned full-region replacement owns its internal roles.
            # It remains valid when the pipeline has no provider-specific
            # selector fields to project.  Explicit GDINO/SAM2 evidence is
            # rejected below because it cannot be mapped to this opaque seam.
            continue
        if not bool(role_binding.get("available", False)):
            raise ValueError(
                f"pipeline region targets {targets!r} require a "
                f"{role_name}, but runtime_config binds that role to "
                f"{role_binding.get('backend')!r}"
            )

    policy = copy.deepcopy(
        dict(selector_binding.get("runtime_region_policy", {}) or {})
    )
    explicit_policy = copy.deepcopy(
        dict(
            selector_binding.get(
                "explicit_runtime_region_policy",
                {},
            )
            or {}
        )
    )
    policy_binding = "not_requested"
    if policy:
        if backend in {"injected", "factory"}:
            if explicit_policy:
                raise ValueError(
                    "explicit pipeline region selector policy cannot be "
                    "projected into full external region backend "
                    f"{backend!r}"
                )
            policy_binding = "caller_owned_full_region_runtime"
        else:
            raw_runtime_config = region_runtime.get("runtime_config", {})
            if not isinstance(raw_runtime_config, Mapping):
                raise TypeError(
                    "runtime_config.region.runtime_config must be a mapping"
                )
            selector_runtime = copy.deepcopy(dict(raw_runtime_config))
            raw_global = selector_runtime.get("region", {})
            if not isinstance(raw_global, Mapping):
                raise TypeError(
                    "runtime_config.region.runtime_config.region must be a mapping"
                )
            selector_runtime["region"] = {
                **copy.deepcopy(dict(raw_global)),
                **policy,
            }
            region_runtime["runtime_config"] = selector_runtime
            runtime["region"] = region_runtime
            policy_binding = "projected_to_builtin_region_runtime"

    runtime_base = _runtime_region_asset_base(
        runtime,
        runtime_asset_base=runtime_asset_base,
    )
    provider_selector_config = dict(
        selector_binding.get("provider_selector_config", {}) or {}
    )
    provider_activation = {
        "grounding_dino": _validate_region_provider_evidence(
            provider_name="grounding_dino",
            selector_config=dict(
                provider_selector_config.get("grounding_dino", {}) or {}
            ),
            role_binding=roles["detector"],
            active=bool(required["detector"]),
            runtime_base=runtime_base,
        ),
        "sam2": _validate_region_provider_evidence(
            provider_name="sam2",
            selector_config=dict(provider_selector_config.get("sam2", {}) or {}),
            role_binding=roles["segmenter"],
            active=bool(required["segmenter"]),
            runtime_base=runtime_base,
        ),
    }
    selector_manifest = copy.deepcopy(dict(selector_binding.get("manifest", {}) or {}))
    selector_evidence = selector_manifest.get(
        "provider_selector_evidence",
        {},
    )
    if isinstance(selector_evidence, Mapping):
        synchronized_evidence = copy.deepcopy(dict(selector_evidence))
        for provider_name, activation in provider_activation.items():
            provider_entry = synchronized_evidence.get(provider_name, {})
            if isinstance(provider_entry, Mapping):
                synchronized_evidence[provider_name] = {
                    **copy.deepcopy(dict(provider_entry)),
                    "activation": activation["activation"],
                }
        selector_manifest["provider_selector_evidence"] = synchronized_evidence
    return {
        "runtime_config": runtime,
        "manifest": {
            **selector_manifest,
            "required_provider_roles": {
                name: list(values) for name, values in required.items()
            },
            "runtime_region_backend": backend,
            "runtime_policy_binding": policy_binding,
            "provider_activation": provider_activation,
        },
    }


def _preflight_pipeline(
    pipeline_config: Mapping[str, Any],
    *,
    depth_calibration: Any,
    pose_estimator: Any,
    runtime_manifest: Mapping[str, Any],
) -> None:
    normalized = load_pipeline_config(copy.deepcopy(dict(pipeline_config)))
    depth = dict(normalized.get("depth", {}) or {})
    base = dict(depth.get("base", {}) or {})
    selected_video = (
        str(
            dict(normalized.get("input", {}) or {}).get(
                "selected_video",
                "rollout",
            )
            or "rollout"
        )
        .strip()
        .lower()
    )
    use_rollout_gt_depth = bool(
        depth.get(
            "use_rollout_gt_depth",
            depth.get(
                "use_reference_depth",
                False,
            ),
        )
    )
    if selected_video == "gen":
        use_rollout_gt_depth = False
    base_enabled = bool(base.get("enabled", False)) or (selected_video == "gen")
    if base_enabled and not use_rollout_gt_depth and depth_calibration is None:
        raise RuntimeError(
            "pipeline depth.base.enabled=true requires an explicit "
            "depth_calibration runtime; this standalone builder does not "
            "silently omit configured base-depth calibration"
        )

    depth_component = dict(
        dict(runtime_manifest.get("components", {}) or {}).get(
            "depth",
            {},
        )
        or {}
    )
    runtime_backend = str(depth_component.get("backend", "") or "")
    if (
        runtime_backend
        in {
            "dvd",
            "factory",
            "injected",
            "registry",
            "vda",
        }
        and not use_rollout_gt_depth
    ):
        requested_model = str(
            depth.get(
                "model",
                DEFAULT_DEPTH_PRESET,
            )
            or DEFAULT_DEPTH_PRESET
        ).strip()
        external_requested_id = external_depth_backend_id(requested_model)
        if external_requested_id is not None:
            expected_model_name = external_depth_selection(external_requested_id)
            expected_preset = expected_model_name
        elif requested_model in SUPPORTED_DEPTH_BACKENDS:
            expected_model_name = requested_model
            expected_preset = requested_model
        else:
            requested_preset = resolve_depth_estimator_preset(requested_model)
            expected_preset = requested_preset
            expected_model_name = str(
                default_depth_estimator_config(requested_preset).get("model_name", "")
                or ""
            )
        runtime_model_name = str(depth_component.get("model_name", "") or "")
        runtime_preset_raw = str(depth_component.get("preset", "") or "").strip()
        runtime_external_id = external_depth_backend_id(runtime_preset_raw)
        if runtime_external_id is not None:
            runtime_preset = external_depth_selection(runtime_external_id)
        elif runtime_preset_raw in SUPPORTED_DEPTH_BACKENDS:
            runtime_preset = runtime_preset_raw
        else:
            runtime_preset = (
                resolve_depth_estimator_preset(runtime_preset_raw)
                if runtime_preset_raw
                else ""
            )
        if (
            expected_model_name
            and runtime_model_name
            and expected_model_name != runtime_model_name
        ):
            raise ValueError(
                "explicit depth runtime does not match pipeline depth "
                f"selection: pipeline={requested_model!r} "
                f"({expected_model_name}), "
                f"runtime={runtime_model_name!r}"
            )
        if expected_preset and runtime_preset and expected_preset != runtime_preset:
            raise ValueError(
                "explicit depth runtime preset does not match pipeline "
                f"selection: pipeline={expected_preset!r}, "
                f"runtime={runtime_preset!r}"
            )
    pose = dict(normalized.get("pose", {}) or {})
    if bool(pose.get("enabled", False)):
        runtime_pose = dict(
            dict(runtime_manifest.get("components", {}) or {}).get(
                "pose",
                {},
            )
            or {}
        )
        runtime_pose_provider, runtime_pose_backend = _runtime_pose_identity(
            runtime_pose
        )
        pipeline_pose_backend = effective_pose_backend(pose)
        if (
            pose_backend_identity_is_known(runtime_pose_backend)
            and pipeline_pose_backend != runtime_pose_backend
        ):
            raise ValueError(
                "explicit pose runtime does not match pipeline pose "
                f"selection: pipeline={pipeline_pose_backend!r}, "
                f"runtime={runtime_pose_backend!r}"
            )
        if pose_estimator is not None and runtime_pose_provider in {
            "estimator_injected",
            "estimator_factory",
        }:
            return


def run_standalone_video2traj(
    *,
    video_path: str | Path,
    simulator_config: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    pipeline_pose_config: Mapping[str, Any] | None = None,
    pipeline_region_config: Mapping[str, Any] | None = None,
    runtime_config: Mapping[str, Any],
    dependencies: Mapping[str, Any] | None = None,
    runtime_asset_base: str | Path | None = None,
    pipeline_asset_base: str | Path | None = None,
    run_options: Mapping[str, Any] | None = None,
    video_backend: str = "auto",
    write_artifacts: bool = False,
    output_dir: str | Path | None = None,
    _allow_bench_legacy_depth_bridge: bool = False,
) -> dict[str, Any]:
    """Run raw-video multi-object video2traj without bench or sim discovery.

    ``write_artifacts=False`` is strict: an output directory is rejected and
    optional visualization output remains disabled. Final trajectory
    publication is enabled only by the explicit ``write_artifacts`` opt-in.
    """

    if not isinstance(simulator_config, Mapping):
        raise TypeError("simulator_config must be an explicit mapping")
    if not isinstance(pipeline_config, Mapping):
        raise TypeError("pipeline_config must be an explicit mapping")
    if not isinstance(runtime_config, Mapping):
        raise TypeError("runtime_config must be an explicit mapping")
    output_text = str(output_dir or "").strip()
    if bool(write_artifacts) and not output_text:
        raise ValueError("write_artifacts=True requires an explicit output_dir")
    if not bool(write_artifacts) and output_text:
        raise ValueError(
            "output_dir requires write_artifacts=True; no-write is the default"
        )
    options = _run_options(
        run_options,
        allow_bench_legacy_depth_bridge=(_allow_bench_legacy_depth_bridge),
    )
    if bool(options.get("write_tracking_artifacts", False)) and not bool(
        write_artifacts
    ):
        raise ValueError(
            "write_tracking_artifacts requires write_artifacts=True "
            "at the standalone boundary"
        )
    if (
        bool(options.get("write_tracking_artifacts", False))
        and not str(options.get("tracking_output_dir", "") or "").strip()
    ):
        raise ValueError(
            "write_tracking_artifacts requires an explicit "
            "run_options.tracking_output_dir"
        )

    pipeline_input = copy.deepcopy(dict(pipeline_config))
    raw_pipeline_region = (
        pipeline_input.get("region", {})
        if pipeline_region_config is None
        else pipeline_region_config
    )
    if raw_pipeline_region is not None and not isinstance(
        raw_pipeline_region,
        Mapping,
    ):
        label = (
            "pipeline_config.region"
            if pipeline_region_config is None
            else "pipeline_region_config"
        )
        raise TypeError(f"{label} must be a mapping")
    region_selector_binding = resolve_pipeline_region_selector_policy(
        (None if raw_pipeline_region is None else dict(raw_pipeline_region)),
        pipeline_asset_base=pipeline_asset_base,
    )
    region_selector_binding["manifest"]["pipeline_region_input"] = (
        "pipeline_config" if pipeline_region_config is None else "explicit_raw_region"
    )
    pipeline_input["region"] = copy.deepcopy(
        dict(region_selector_binding["pipeline_region"])
    )
    normalized_pipeline = load_pipeline_config(pipeline_input)
    inline_pose = (
        pipeline_config.get("pose", {})
        if pipeline_pose_config is None
        else pipeline_pose_config
    )
    if inline_pose is not None and not isinstance(inline_pose, Mapping):
        raise TypeError("pipeline pose must be a mapping")
    pose_binding = prepare_standalone_pose_pipeline(
        normalized_pipeline,
        inline_pose_config=(
            None if inline_pose is None else copy.deepcopy(dict(inline_pose))
        ),
        pipeline_asset_base=pipeline_asset_base,
        estimator_owns_configuration=(
            runtime_pose_estimator_owns_configuration(runtime_config)
        ),
    )
    effective_pipeline = copy.deepcopy(dict(pose_binding["pipeline_config"]))
    region_runtime_binding = _bind_pipeline_region_runtime(
        runtime_config,
        pipeline_config=effective_pipeline,
        selector_binding=region_selector_binding,
        runtime_asset_base=runtime_asset_base,
        run_options=options,
    )
    active_runtime_config = copy.deepcopy(
        dict(region_runtime_binding["runtime_config"])
    )
    pipeline_pose_enabled = bool(
        dict(effective_pipeline.get("pose", {}) or {}).get(
            "enabled",
            False,
        )
    )
    if not pipeline_pose_enabled:
        # Per-run pipeline overrides are the stage gate.  A runtime profile
        # describes which implementation to use when the stage is enabled;
        # it must never re-enable pose for a sample that explicitly disables
        # the stage.
        active_runtime_config["pose"] = {"backend": "none"}
    runtime = build_video2traj_runtime(
        active_runtime_config,
        dependencies=dependencies,
        asset_base=runtime_asset_base,
    )
    effective_depth = copy.deepcopy(dict(effective_pipeline.get("depth", {}) or {}))
    effective_depth["base"] = bind_depth_base_runtime_calibration(
        dict(effective_depth.get("base", {}) or {}),
        depth_runtime_config=(
            None
            if runtime.get("depth_runtime_config") is None
            else dict(runtime["depth_runtime_config"])
        ),
    )
    effective_pipeline["depth"] = effective_depth
    runtime_pose_component = dict(
        dict(runtime["manifest"].get("components", {}) or {}).get(
            "pose",
            {},
        )
        or {}
    )
    runtime_pose_provider, runtime_pose_backend = _runtime_pose_identity(
        runtime_pose_component
    )
    pipeline_pose_backend = effective_pose_backend(
        dict(effective_pipeline.get("pose", {}) or {})
    )
    pose_manifest = dict(pose_binding["manifest"])
    if (
        pose_backend_identity_is_known(runtime_pose_backend)
        and runtime_pose_backend == pipeline_pose_backend
    ):
        pose_manifest["backend_identity"] = "matched"
        pose_manifest["effective_backend"] = pipeline_pose_backend
    elif runtime_pose_provider in {
        "estimator_injected",
        "estimator_factory",
    }:
        pose_manifest["backend_identity"] = "caller_owned"
    else:
        pose_manifest["effective_backend"] = pipeline_pose_backend
    pose_binding["manifest"] = pose_manifest
    _preflight_pipeline(
        effective_pipeline,
        depth_calibration=runtime["depth_calibration"],
        pose_estimator=runtime["pose_estimator"],
        runtime_manifest=runtime["manifest"],
    )
    depth_options = options.pop("depth_options", None)
    if depth_options is None:
        resolved_depth_options: dict[str, Any] = {}
    elif isinstance(depth_options, Mapping):
        resolved_depth_options = copy.deepcopy(dict(depth_options))
    else:
        raise TypeError("run_options.depth_options must be a mapping")
    depth_runtime_config = runtime.get(
        "depth_runtime_config",
        None,
    )
    if depth_runtime_config is not None:
        if "depth_runtime_config" in resolved_depth_options:
            raise ValueError(
                "run_options.depth_options cannot replace the runtime "
                "builder's depth_runtime_config"
            )
        resolved_depth_options["depth_runtime_config"] = copy.deepcopy(
            dict(depth_runtime_config)
        )
    if resolved_depth_options:
        options["depth_options"] = resolved_depth_options

    pose_options = options.pop("pose_options", None)
    if pose_options is None:
        resolved_pose_options: dict[str, Any] = {}
    elif isinstance(pose_options, Mapping):
        resolved_pose_options = copy.deepcopy(dict(pose_options))
    else:
        raise TypeError("run_options.pose_options must be a mapping")
    correction_payload = pose_binding.get(
        "pose_correction_payload",
        None,
    )
    if correction_payload is not None:
        reserved = {
            "pose_correction_payload",
            "pose_correction_source",
        }.intersection(resolved_pose_options)
        if reserved:
            raise ValueError(
                "run_options.pose_options cannot replace pipeline pose "
                "correction inputs: " + ", ".join(sorted(reserved))
            )
        resolved_pose_options["pose_correction_payload"] = copy.deepcopy(
            correction_payload
        )
        resolved_pose_options["pose_correction_source"] = str(
            pose_binding.get("pose_correction_source", "") or ""
        )
    if resolved_pose_options:
        options["pose_options"] = resolved_pose_options

    dependency_map = dict(dependencies or {})
    video_reader = dependency_map.get(
        "video_reader",
        read_video_frames,
    )
    if not callable(video_reader):
        raise TypeError("dependencies['video_reader'] must be callable")

    result = run_multi_object_video_file(
        video_path=video_path,
        simulator_config=copy.deepcopy(dict(simulator_config)),
        pipeline_config=effective_pipeline,
        region_runtime=runtime["region_runtime"],
        tracking_backend=runtime["tracking_backend"],
        depth_estimator=runtime["depth_estimator"],
        depth_calibration=runtime["depth_calibration"],
        pose_estimator=runtime["pose_estimator"],
        pose_backend=runtime["pose_backend"],
        rigid_pose_backend=runtime.get("rigid_pose_backend"),
        video_reader=video_reader,
        video_backend=str(video_backend or "auto"),
        write_artifacts=bool(write_artifacts),
        output_dir=(output_text or None),
        **options,
    )
    result["standalone_runtime"] = copy.deepcopy(dict(runtime["manifest"]))
    result["standalone_runtime"]["pipeline_pose"] = copy.deepcopy(
        dict(pose_binding["manifest"])
    )
    result["standalone_runtime"]["pipeline_region"] = copy.deepcopy(
        dict(region_runtime_binding["manifest"])
    )
    return result


def _lexical_absolute_path(value: str | Path) -> Path:
    return Path(os.path.abspath(Path(value).expanduser()))


def _portable_depth_path(
    value: Any,
    *,
    roots: Mapping[str, Path],
) -> dict[str, str] | None:
    text = str(value or "").strip()
    if not text:
        return None
    candidate = Path(text).expanduser()
    if candidate.is_absolute():
        candidate = _lexical_absolute_path(candidate)
        for name, root in sorted(
            roots.items(),
            key=lambda item: len(item[1].parts),
            reverse=True,
        ):
            try:
                relative = candidate.relative_to(root)
            except ValueError:
                continue
            reference = {"root": str(name)}
            if relative.parts:
                reference["path"] = relative.as_posix()
            return reference
        return {
            "root": "external",
            "location_sha256": hashlib.sha256(
                candidate.as_posix().encode("utf-8")
            ).hexdigest(),
        }
    if (
        candidate.parts
        and ".." not in candidate.parts
        and not any(character in text for character in ("\x00", "\r", "\n"))
    ):
        return {
            "root": "relative",
            "path": candidate.as_posix(),
        }
    return {
        "root": "external",
        "location_sha256": hashlib.sha256(text.encode("utf-8")).hexdigest(),
    }


def _summarize_depth_publication(
    value: Any,
    *,
    publication: Any,
    portable_roots: Mapping[str, str | Path] | None,
) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("depth_publication must be a mapping")
    payload = dict(value)
    final_publication = dict(publication) if isinstance(publication, Mapping) else {}
    output_text = str(final_publication.get("output_dir", "") or "").strip()
    roots: dict[str, Path] = {}
    for raw_name, raw_path in dict(portable_roots or {}).items():
        name = str(raw_name or "").strip()
        path_text = str(raw_path or "").strip()
        if not name or not path_text or not Path(path_text).expanduser().is_absolute():
            continue
        roots[name] = _lexical_absolute_path(path_text)
    if output_text and Path(output_text).expanduser().is_absolute():
        roots["output"] = _lexical_absolute_path(output_text)
    cache_path = payload.get("cache_path", None)
    meta_path = payload.get("meta_path", None)
    policy = str(payload.get("legacy_metadata_policy", "") or "").strip()
    if policy not in {
        "reject",
        "trusted_pickle_read_only",
    }:
        policy = "unknown"
    status = str(payload.get("status", "") or "").strip()
    if status not in {
        "published",
        "ready",
        "reused",
        "legacy_reused",
        "referenced",
    }:
        status = "unknown"
    trusted_scope_text = str(payload.get("trusted_legacy_scope", "") or "").strip()
    trusted_scope = _portable_depth_path(
        trusted_scope_text,
        roots=roots,
    )
    summary = {
        "status": status,
        "identity_verified": (
            bool(payload["identity_verified"])
            if "identity_verified" in payload
            else None
        ),
        "legacy_metadata_policy": policy,
        "trusted_legacy_scope": trusted_scope,
        "cache": _portable_depth_path(
            cache_path,
            roots=roots,
        ),
        "meta": _portable_depth_path(
            meta_path,
            roots=roots,
        ),
    }
    source_path = payload.get("source_path", None)
    source_identity = payload.get("source_identity", None)
    if source_path is not None:
        summary["source"] = _portable_depth_path(
            source_path,
            roots=roots,
        )
    if isinstance(source_identity, Mapping):
        file_sha256 = str(source_identity.get("file_sha256", "") or "")
        array_fingerprint = str(source_identity.get("array_fingerprint", "") or "")
        summary["source_sha256"] = file_sha256 or None
        summary["array_fingerprint"] = array_fingerprint or None
    return summary


def summarize_standalone_video2traj_result(
    result: Mapping[str, Any],
    *,
    portable_roots: Mapping[str, str | Path] | None = None,
) -> dict[str, Any]:
    """Return a bounded, JSON-ready CLI summary without embedding arrays."""

    payload = dict(result)
    outputs = dict(payload.get("outputs", {}) or {})
    ee_traj = dict(outputs.get("ee_traj", {}) or {})
    obj_traj = dict(outputs.get("obj_traj", {}) or {})
    gripper = dict(outputs.get("gripper", {}) or {})
    action = dict(outputs.get("action", {}) or {})
    video_file = dict(payload.get("video_file", {}) or {})
    publication = payload.get("publication", None)
    depth_publication = _summarize_depth_publication(
        payload.get("depth_publication", None),
        publication=publication,
        portable_roots=portable_roots,
    )
    runtime = dict(payload.get("standalone_runtime", {}) or {})
    return {
        "scope": copy.deepcopy(dict(payload.get("scope", {}) or {})),
        "video": {
            key: video_file.get(key)
            for key in (
                "path",
                "frame_count",
                "frame_width",
                "frame_height",
                "fps",
                "algorithm_fps",
            )
            if key in video_file
        },
        "outputs": {
            "eef_frames": len(list(ee_traj.get("eef_controller", []) or [])),
            "object_frames": len(list(obj_traj.get("obj_visual_center", []) or [])),
            "gripper_actions": len(list(gripper.get("actions", []) or [])),
            "action_checkpoints": len(list(action.get("checkpoints", []) or [])),
        },
        "publication": (
            None if publication is None else copy.deepcopy(dict(publication))
        ),
        "trajectory_diagnostics": (
            None
            if payload.get("trajectory_diagnostics", None) is None
            else copy.deepcopy(dict(payload["trajectory_diagnostics"]))
        ),
        "depth_publication": depth_publication,
        "runtime": runtime,
    }


__all__ = [
    "run_standalone_video2traj",
    "summarize_standalone_video2traj_result",
]
