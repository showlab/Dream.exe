"""Portable configuration for the :mod:`video2traj.region` package.

The normalized configuration matches the current trajectory pipeline except
for the intentional public provider boundary.  Local provider assets remain
portable unless the caller supplies an explicit path resolver; the core never
discovers repository, bench, or simulator roots.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any


REGION_CONFIGS_DIR = Path(__file__).resolve().parent.parent / "configs"
DEFAULT_REGION_CONFIG_PATH = REGION_CONFIGS_DIR / "region.default.json"
RegionPathResolver = Callable[[str], str]

_PIPELINE_REGION_PROVIDER_BLOCKS = (
    "grounding_dino",
    "sam2",
)


def builtin_region_config_path(name: str = "default") -> str:
    """Return one packaged preset path without external root discovery."""

    filename = str(name)
    if filename.endswith(".json"):
        filename = filename[:-5]
    if filename.startswith("region."):
        filename = filename[len("region.") :]
    path = REGION_CONFIGS_DIR / f"region.{filename}.json"
    if not path.exists():
        raise FileNotFoundError(f"Region preset not found: {path}")
    return path.as_posix()


def available_region_configs() -> tuple[str, ...]:
    """List the packaged current-compatible region presets."""

    prefix = "region."
    return tuple(
        sorted(
            path.stem[len(prefix) :]
            for path in REGION_CONFIGS_DIR.glob("region.*.json")
            if path.is_file() and path.stem.startswith(prefix)
        )
    )


def _load_json(path: str | Path) -> Any:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _deep_merge_dict(
    base: dict[str, Any],
    override: dict[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(output.get(key), dict):
            output[key] = _deep_merge_dict(
                output[key],
                value,
            )
        else:
            output[key] = value
    return output


def _normalize_legacy_target_config(
    target_config: dict[str, Any],
    *,
    target_name: str,
) -> dict[str, Any]:
    del target_name
    config = copy.deepcopy(target_config)
    region_source = str(config.get("region_source", "")).strip().lower()
    bbox_source = str(config.get("bbox_source", "")).strip().lower()
    sampling_method = str(config.get("sampling_method", "")).strip().lower()

    selector = "visual"
    if region_source == "robosuite_mask":
        selector = "simulation"
    elif region_source == "off":
        selector = "off"
    elif region_source in {"", "auto"}:
        selector = "auto"

    manual_bbox = config.get("manual_bbox_xyxy")
    visual_bbox_source = bbox_source or (
        "manual" if manual_bbox is not None else "grounding_dino"
    )
    if visual_bbox_source == "auto":
        visual_bbox_source = "grounding_dino"

    visual_sampling = "mask_3d_fps"
    if sampling_method == "bbox_center" or region_source == "bbox":
        visual_sampling = "bbox_gaussian"

    output: dict[str, Any] = {
        "selector": selector,
        "num_points": int(config.get("num_points", 50)),
        "prompt": str(config.get("prompt", "") or ""),
        "sampling": {"method": visual_sampling},
        "visual": {"bbox_source": visual_bbox_source},
    }
    if manual_bbox is not None:
        output["legacy_manual_bbox_xyxy"] = manual_bbox
        output["bbox_xyxy"] = manual_bbox
    return output


def _normalize_target_config(
    target_config: dict[str, Any],
    *,
    target_name: str,
) -> dict[str, Any]:
    config = copy.deepcopy(target_config)
    if "selector" not in config:
        config = _normalize_legacy_target_config(
            config,
            target_name=target_name,
        )

    config["selector"] = str(config.get("selector", "")).strip() or (
        "off" if target_name == "obj" else "visual"
    )
    config["num_points"] = int(config.get("num_points", 50))
    config["prompt"] = str(config.get("prompt", "") or "")
    bbox_xyxy = config.get("bbox_xyxy", None)
    config["bbox_xyxy"] = (
        None if bbox_xyxy is None else [int(value) for value in bbox_xyxy]
    )

    simulation = config.get("simulation", {})
    if not isinstance(simulation, dict):
        simulation = {}
    simulation["instance_name"] = str(simulation.get("instance_name", "") or "")
    config["simulation"] = simulation

    sampling = config.get("sampling", {})
    if not isinstance(sampling, dict):
        sampling = {}
    sampling_method = str(sampling.get("method", "") or "").strip()
    if not sampling_method:
        legacy_visual = config.get("visual", {})
        legacy_sampling = str(
            (legacy_visual or {}).get(
                "sampling",
                "sam_mask",
            )
            or "sam_mask"
        )
        sampling_method = (
            "bbox_gaussian" if legacy_sampling == "bbox_gaussian" else "mask_3d_fps"
        )
    sampling["method"] = sampling_method
    config["sampling"] = sampling

    visual = config.get("visual", {})
    if not isinstance(visual, dict):
        visual = {}
    visual["bbox_source"] = str(
        visual.get("bbox_source", "grounding_dino") or "grounding_dino"
    )
    config["visual"] = visual
    return config


def normalize_region_config(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Normalize a complete region configuration like the current loader."""

    normalized = copy.deepcopy(config)
    normalized.setdefault("targets", {})
    normalized.setdefault("region", {})
    normalized.setdefault("grounding_dino", {})
    normalized.setdefault("sam2", {})

    for target_name in ("eef", "obj"):
        normalized["targets"][target_name] = _normalize_target_config(
            normalized["targets"].get(
                target_name,
                {},
            ),
            target_name=target_name,
        )

    region = normalized["region"]
    region["mask_erosion_kernel_size"] = int(region.get("mask_erosion_kernel_size", 11))
    region["save_debug"] = bool(region.get("save_debug", False))
    region["combine_mode"] = str(region.get("combine_mode", "intersect") or "intersect")
    return normalized


def _normalize_region_override(
    config: Mapping[str, Any],
) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    config_dict = dict(config)

    if "targets" in config_dict and isinstance(
        config_dict["targets"],
        dict,
    ):
        normalized["targets"] = {}
        for target_name, target_config in config_dict["targets"].items():
            if target_name not in {"eef", "obj"} or not isinstance(target_config, dict):
                continue
            normalized["targets"][target_name] = _normalize_target_config(
                target_config,
                target_name=target_name,
            )

    if "region" in config_dict and isinstance(
        config_dict["region"],
        dict,
    ):
        region_override = copy.deepcopy(config_dict["region"])
        if "mask_erosion_kernel_size" in region_override:
            region_override["mask_erosion_kernel_size"] = int(
                region_override["mask_erosion_kernel_size"]
            )
        if "save_debug" in region_override:
            region_override["save_debug"] = bool(region_override["save_debug"])
        if "combine_mode" in region_override:
            region_override["combine_mode"] = str(
                region_override["combine_mode"] or "intersect"
            )
        normalized["region"] = region_override

    for block_name in ("grounding_dino", "sam2"):
        if block_name in config_dict and isinstance(
            config_dict[block_name],
            dict,
        ):
            normalized[block_name] = copy.deepcopy(config_dict[block_name])
    return normalized


def _resolve_region_asset_paths(
    config: Mapping[str, Any],
    *,
    path_resolver: RegionPathResolver | None,
) -> dict[str, Any]:
    output = copy.deepcopy(dict(config))
    if path_resolver is None:
        return output

    grounding_dino = output.get("grounding_dino", {})
    if isinstance(grounding_dino, dict):
        for field in (
            "local_config_path",
            "local_checkpoint_path",
            "local_text_encoder_path",
        ):
            value = str(grounding_dino.get(field, "")).strip()
            if value:
                grounding_dino[field] = path_resolver(value)
        output["grounding_dino"] = grounding_dino

    sam2 = output.get("sam2", {})
    if isinstance(sam2, dict) and str(sam2.get("checkpoint_path", "")).strip():
        sam2["checkpoint_path"] = path_resolver(str(sam2["checkpoint_path"]))
        output["sam2"] = sam2
    return output


def validate_region_config(
    config: Mapping[str, Any],
    *,
    source: str,
) -> None:
    """Validate the current accepted region configuration schema."""

    if not isinstance(config, dict):
        raise ValueError(f"Region config must be a JSON object: {source}")
    config_dict = dict(config)
    for block_name in (
        "targets",
        "region",
        "grounding_dino",
        "sam2",
    ):
        if block_name not in config_dict or not isinstance(
            config_dict[block_name],
            dict,
        ):
            raise ValueError(f"Region config missing object '{block_name}': {source}")

    selector_valid = {"simulation", "visual", "auto"}
    visual_bbox_valid = {"manual", "grounding_dino"}
    sampling_valid = {
        "bbox_gaussian",
        "mask_fps",
        "mask_2d_fps",
        "mask_3d_fps",
    }
    targets = config_dict["targets"]
    for target_name in ("eef", "obj"):
        if target_name not in targets or not isinstance(
            targets[target_name],
            dict,
        ):
            raise ValueError(f"Region config missing targets.{target_name}: {source}")
        target_config = targets[target_name]
        selector = str(target_config.get("selector", "")).strip()
        valid_selectors = set(selector_valid)
        if target_name == "obj":
            valid_selectors.add("off")
        if selector not in valid_selectors:
            raise ValueError(
                f"Invalid targets.{target_name}.selector in {source}: {selector}"
            )

        visual = target_config.get("visual", {})
        if not isinstance(visual, dict):
            raise ValueError(
                f"targets.{target_name}.visual must be an object in {source}"
            )
        simulation = target_config.get(
            "simulation",
            {},
        )
        if not isinstance(simulation, dict):
            raise ValueError(
                f"targets.{target_name}.simulation must be an object in {source}"
            )
        sampling = target_config.get("sampling", {})
        if not isinstance(sampling, dict):
            raise ValueError(
                f"targets.{target_name}.sampling must be an object in {source}"
            )
        bbox_source = str(visual.get("bbox_source", "")).strip()
        if bbox_source not in visual_bbox_valid:
            raise ValueError(
                "Invalid targets."
                f"{target_name}.visual.bbox_source in "
                f"{source}: {bbox_source}"
            )
        sampling_method = str(sampling.get("method", "")).strip()
        if sampling_method not in sampling_valid:
            raise ValueError(
                "Invalid targets."
                f"{target_name}.sampling.method in "
                f"{source}: {sampling_method}"
            )

        for bbox_key in (
            "legacy_manual_bbox_xyxy",
            "bbox_xyxy",
        ):
            bbox = target_config.get(bbox_key, None)
            if bbox is not None and (not isinstance(bbox, list) or len(bbox) != 4):
                raise ValueError(
                    f"Invalid targets.{target_name}.{bbox_key} in {source}: {bbox}"
                )

    region = config_dict["region"]
    if str(region.get("combine_mode", "")).strip() not in {
        "intersect",
        "union",
        "mask_only",
        "bbox_only",
    }:
        raise ValueError(
            f"Invalid region.combine_mode in {source}: {region.get('combine_mode')}"
        )

    grounding_dino = config_dict["grounding_dino"]
    if not bool(grounding_dino.get("use_local_repo", False)):
        raise ValueError(
            f"grounding_dino.use_local_repo must be true (offline-only): {source}"
        )
    if not str(grounding_dino.get("local_config_path", "")).strip():
        raise ValueError(f"grounding_dino.local_config_path is required: {source}")
    if not str(
        grounding_dino.get(
            "local_checkpoint_path",
            "",
        )
    ).strip():
        raise ValueError(f"grounding_dino.local_checkpoint_path is required: {source}")
    sam2 = config_dict["sam2"]
    if not str(sam2.get("config_name", "")).strip():
        raise ValueError(f"sam2.config_name is required: {source}")
    if not str(sam2.get("checkpoint_path", "")).strip():
        raise ValueError(f"sam2.checkpoint_path is required: {source}")


def load_region_config(
    *,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    preset: str = "default",
    runtime_override: Mapping[str, Any] | None = None,
    path_resolver: RegionPathResolver | None = None,
) -> dict[str, Any]:
    """Load, merge, normalize, and validate a region configuration."""

    if config is not None and config_path is not None:
        raise ValueError("Provide either region config or config_path, not both.")
    base_source = builtin_region_config_path("default")
    base = normalize_region_config(_load_json(base_source))

    if config_path is not None:
        source_path = Path(config_path).expanduser().resolve()
        source = source_path.as_posix()
        override = normalize_region_config(_load_json(source_path))
        merged = _deep_merge_dict(base, override)
    elif config is not None:
        source = "<mapping>"
        override = normalize_region_config(config)
        merged = _deep_merge_dict(base, override)
    else:
        source = builtin_region_config_path(preset)
        if str(preset).strip() == "default":
            merged = base
        else:
            override = normalize_region_config(_load_json(source))
            merged = _deep_merge_dict(base, override)

    if runtime_override:
        merged = _deep_merge_dict(
            merged,
            _normalize_region_override(runtime_override),
        )
    merged = _resolve_region_asset_paths(
        merged,
        path_resolver=path_resolver,
    )
    validate_region_config(merged, source=source)
    return merged


def resolve_region_runtime_config(
    *,
    config: Mapping[str, Any] | None = None,
    config_path: str | Path | None = None,
    preset: str = "default",
    runtime_override: Mapping[str, Any] | None = None,
    path_resolver: RegionPathResolver | None = None,
) -> dict[str, Any]:
    """Resolve the region selector payload consumed by :class:`RegionRuntime`."""

    loaded = load_region_config(
        config=config,
        config_path=config_path,
        preset=preset,
        runtime_override=runtime_override,
        path_resolver=path_resolver,
    )
    if config_path is not None:
        config_source = Path(config_path).expanduser().resolve().as_posix()
    elif config is not None:
        config_source = "<mapping>"
    else:
        config_source = builtin_region_config_path(preset)
    return {
        "config": loaded,
        "config_source": config_source,
        "targets": loaded["targets"],
        "region": loaded["region"],
        "grounding_dino": loaded["grounding_dino"],
        "sam2": loaded["sam2"],
    }


def _pipeline_region_base(
    value: str | Path | None,
) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError("pipeline_asset_base must be an absolute path")
    return path.resolve(strict=False)


def _pipeline_region_config_path(
    value: Any,
    *,
    pipeline_asset_base: str | Path | None,
) -> Path | None:
    text = str(value or "").strip()
    if not text:
        return None
    path = Path(text).expanduser()
    if not path.is_absolute():
        base = _pipeline_region_base(pipeline_asset_base)
        if base is None:
            raise ValueError(
                "relative region.config_path requires an explicit "
                "pipeline_asset_base (the pipeline config directory)"
            )
        path = base / path
    return path.resolve(strict=False)


def _pipeline_region_mapping(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return copy.deepcopy(dict(value))


def _pipeline_region_source_document(
    *,
    preset: str,
    config_path: Path | None,
) -> tuple[dict[str, Any], str]:
    if config_path is not None:
        payload = _load_json(config_path)
        source = config_path.as_posix()
    elif preset != "default":
        source = builtin_region_config_path(preset)
        payload = _load_json(source)
    else:
        # The provider blocks in the packaged default preserve the historical
        # loader contract.  They are compatibility defaults, not an implicit
        # request to activate model providers in the standalone runtime.
        return {}, builtin_region_config_path("default")
    if not isinstance(payload, Mapping):
        raise ValueError(f"Region selector config must be a JSON object: {source}")
    return copy.deepcopy(dict(payload)), source


def _pipeline_region_objects_override(
    value: Mapping[str, Any],
    *,
    label: str,
) -> list[dict[str, Any]] | None:
    raw_targets = value.get("targets")
    if raw_targets is None:
        return None
    if not isinstance(raw_targets, Mapping):
        raise TypeError(f"{label}.targets must be a mapping")
    if "objects" not in raw_targets:
        return None
    objects = raw_targets["objects"]
    if not isinstance(objects, list):
        raise TypeError(f"{label}.targets.objects must be a list")
    output: list[dict[str, Any]] = []
    for index, item in enumerate(objects):
        if not isinstance(item, Mapping):
            raise TypeError(f"{label}.targets.objects[{index}] must be a mapping")
        output.append(copy.deepcopy(dict(item)))
    return output


def _pipeline_region_target_sources(
    *,
    source_document: Mapping[str, Any],
    runtime_override: Mapping[str, Any],
    inline_targets: Mapping[str, Any],
) -> dict[str, str]:
    sources: dict[str, str] = {}
    for name in ("eef", "obj", "objects"):
        source = "packaged_default"
        source_targets = source_document.get("targets", {})
        if isinstance(source_targets, Mapping) and name in source_targets:
            source = "selector_config"
        override_targets = runtime_override.get("targets", {})
        if isinstance(override_targets, Mapping) and name in override_targets:
            source = "runtime_override"
        if name in inline_targets:
            source = "pipeline_inline"
        sources[name] = source
    return sources


def _pipeline_region_explicit_policy(
    *,
    source_document: Mapping[str, Any],
    runtime_override: Mapping[str, Any],
    pipeline_region: Mapping[str, Any],
) -> dict[str, Any]:
    policy: dict[str, Any] = {}
    for value, label in (
        (source_document.get("region"), "selector config region"),
        (runtime_override.get("region"), "region.runtime_override.region"),
        (pipeline_region.get("region"), "pipeline region.region"),
    ):
        if value is None:
            continue
        block = _pipeline_region_mapping(value, label=label)
        normalized = _normalize_region_override({"region": block}).get("region", {})
        policy = _deep_merge_dict(policy, normalized)
    return policy


def _pipeline_region_provider_evidence(
    *,
    source_document: Mapping[str, Any],
    runtime_override: Mapping[str, Any],
    pipeline_region: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    providers: dict[str, dict[str, Any]] = {}
    manifest: dict[str, Any] = {}
    for block_name in _PIPELINE_REGION_PROVIDER_BLOCKS:
        block: dict[str, Any] = {}
        field_sources: dict[str, str] = {}
        for document, source_name in (
            (source_document, "selector_config"),
            (runtime_override, "runtime_override"),
            (pipeline_region, "pipeline_inline"),
        ):
            if block_name not in document:
                continue
            incoming = _pipeline_region_mapping(
                document[block_name],
                label=f"{source_name}.{block_name}",
            )
            block = _deep_merge_dict(block, incoming)
            field_sources.update({str(field): source_name for field in incoming})
        providers[block_name] = block
        manifest[block_name] = {
            "explicit": bool(block),
            "fields": sorted(block),
            "field_sources": {
                field: field_sources[field] for field in sorted(field_sources)
            },
            "activation": "pending_runtime_route_check",
        }
    return providers, manifest


def _presence_aware_pipeline_region_config(
    *,
    source_document: Mapping[str, Any],
    runtime_override: Mapping[str, Any],
    source: str,
) -> dict[str, Any]:
    """Resolve production selector layers before injecting schema defaults.

    The historical low-level loader normalizes a partial document before it
    overlays the default.  That behavior is retained by ``load_region_config``
    for compatibility tests, but it is unsuitable at the production pipeline
    boundary because normalization materializes absent fields.  Here raw
    mappings are overlaid first and the complete result is normalized once.
    List-valued ``targets.objects`` follows ordinary replacement semantics.
    """

    base = _load_json(builtin_region_config_path("default"))
    if not isinstance(base, Mapping):  # pragma: no cover - packaged invariant
        raise ValueError("Packaged default region config must be an object")
    merged = _deep_merge_dict(
        copy.deepcopy(dict(base)),
        copy.deepcopy(dict(source_document)),
    )
    if runtime_override:
        merged = _deep_merge_dict(
            merged,
            copy.deepcopy(dict(runtime_override)),
        )
    normalized = normalize_region_config(merged)
    validate_region_config(normalized, source=source)
    return normalized


def resolve_pipeline_region_selector_policy(
    pipeline_region: Mapping[str, Any] | None,
    *,
    pipeline_asset_base: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve pipeline-owned selector controls for production execution.

    The pipeline owns target routing and global selector policy.  Model
    providers and their asset roots remain owned by the explicit standalone
    runtime configuration.  Inline targets have the highest precedence;
    unlike the historical low-level loader, multi-object target entries are
    retained and projected into the effective pipeline.
    """

    region = _pipeline_region_mapping(
        pipeline_region,
        label="pipeline region",
    )
    preset = str(region.get("preset", "default") or "default").strip()
    if not preset:
        preset = "default"
    raw_config_path = str(region.get("config_path", "") or "").strip()
    if raw_config_path and preset != "default":
        raise ValueError(
            "region.preset and region.config_path are ambiguous when both "
            "select a non-default config; use one selector source"
        )
    config_path = _pipeline_region_config_path(
        raw_config_path,
        pipeline_asset_base=pipeline_asset_base,
    )
    runtime_override = _pipeline_region_mapping(
        region.get("runtime_override"),
        label="region.runtime_override",
    )
    source_document, source = _pipeline_region_source_document(
        preset=preset,
        config_path=config_path,
    )
    resolved_config = _presence_aware_pipeline_region_config(
        source_document=source_document,
        runtime_override=runtime_override,
        source=source,
    )

    inline_targets = _pipeline_region_mapping(
        region.get("targets"),
        label="pipeline region.targets",
    )
    selector_controls_active = bool(
        config_path is not None
        or preset != "default"
        or runtime_override
        or region.get("region") is not None
        or any(name in region for name in _PIPELINE_REGION_PROVIDER_BLOCKS)
    )
    effective_region = copy.deepcopy(region)
    if selector_controls_active:
        targets = copy.deepcopy(dict(resolved_config["targets"]))
        # Spell out the list-valued precedence rather than relying on the
        # historical eef/obj normalizer to retain unknown target keys.
        source_objects = _pipeline_region_objects_override(
            source_document,
            label="selector config",
        )
        if source_objects is not None:
            targets["objects"] = source_objects
        runtime_objects = _pipeline_region_objects_override(
            runtime_override,
            label="region.runtime_override",
        )
        if runtime_objects is not None:
            targets["objects"] = runtime_objects
        for name, inline_value in inline_targets.items():
            if name == "objects":
                if not isinstance(inline_value, list):
                    raise TypeError("pipeline region.targets.objects must be a list")
                targets[name] = copy.deepcopy(inline_value)
                continue
            if not isinstance(inline_value, Mapping):
                raise TypeError(f"pipeline region.targets.{name} must be a mapping")
            base_target = targets.get(name, {})
            base_mapping = dict(base_target) if isinstance(base_target, Mapping) else {}
            targets[name] = _deep_merge_dict(
                base_mapping,
                dict(inline_value),
            )
        effective_region["targets"] = targets
        if config_path is not None:
            effective_region["config_path"] = config_path.as_posix()

    explicit_runtime_policy = _pipeline_region_explicit_policy(
        source_document=source_document,
        runtime_override=runtime_override,
        pipeline_region=region,
    )
    runtime_policy = (
        _deep_merge_dict(
            copy.deepcopy(dict(resolved_config["region"])),
            explicit_runtime_policy,
        )
        if selector_controls_active
        else {}
    )
    providers, provider_manifest = _pipeline_region_provider_evidence(
        source_document=source_document,
        runtime_override=runtime_override,
        pipeline_region=region,
    )
    target_sources = _pipeline_region_target_sources(
        source_document=source_document,
        runtime_override=runtime_override,
        inline_targets=inline_targets,
    )
    return {
        "pipeline_region": effective_region,
        "runtime_region_policy": runtime_policy,
        "explicit_runtime_region_policy": explicit_runtime_policy,
        "provider_selector_config": providers,
        "manifest": {
            "contract": "pipeline_region_selector_policy",
            "preset": preset,
            "config_source": source,
            "config_path": ("" if config_path is None else config_path.as_posix()),
            "config_path_anchor": (
                "pipeline_asset_base"
                if raw_config_path and not Path(raw_config_path).is_absolute()
                else "absolute_or_packaged"
            ),
            "selector_controls_consumed": True,
            "projection": (
                "resolved_overlay"
                if selector_controls_active
                else "current_default_noop"
            ),
            "target_precedence": [
                "pipeline_inline",
                "runtime_override",
                "selector_config",
                "packaged_default",
            ],
            "target_sources": target_sources,
            "inline_targets_highest_precedence": True,
            "runtime_policy_fields": sorted(runtime_policy),
            "provider_authority": "standalone_runtime_config",
            "provider_selector_evidence": provider_manifest,
        },
    }


__all__ = [
    "DEFAULT_REGION_CONFIG_PATH",
    "REGION_CONFIGS_DIR",
    "RegionPathResolver",
    "available_region_configs",
    "builtin_region_config_path",
    "load_region_config",
    "normalize_region_config",
    "resolve_pipeline_region_selector_policy",
    "resolve_region_runtime_config",
    "validate_region_config",
]
