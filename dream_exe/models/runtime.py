"""Resolve model selectors into the existing video2traj runtime builder shape."""

from __future__ import annotations

import copy
import hashlib
import importlib
import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .catalog import (
    MODEL_CATALOG_FORMAT,
    ModelCatalog,
    ModelCatalogError,
    ResolvedModel,
    effective_factory_kwargs,
    instantiate_factory_model,
    load_strict_json_object,
    prepare_model_factory,
    resolve_declared_paths,
    resolve_model,
)


RUNTIME_MODEL_FORMAT = "dream-exe.runtime"
_PRIVATE_META_KEY = "_model_runtime_private"
_COMPOSITION_META_KEY = "model_composition"
_RUNTIME_ROOT_FIELDS = frozenset(
    {"format", "category", "device", "region", "tracking", "depth", "pose"}
)


@dataclass(frozen=True)
class RuntimeComposition:
    """Execution config plus its portable fingerprint representation."""

    runtime_config: Mapping[str, Any]
    public_config: Mapping[str, Any]
    selected_models: tuple[Mapping[str, Any], ...]
    runtime_sha256: str | None
    catalog_sha256: str | None


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(output.get(key), Mapping) and isinstance(value, Mapping):
            output[key] = _deep_merge(dict(output[key]), value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value,
        ensure_ascii=True,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _selector(value: Any, *, label: str) -> tuple[str, dict[str, Any]] | None:
    if not isinstance(value, Mapping) or "model" not in value:
        return None
    selected = dict(value)
    unsupported = sorted(set(selected).difference({"model", "options"}))
    if unsupported:
        raise ModelCatalogError(
            f"{label} model selector contains unsupported fields: "
            + ", ".join(unsupported)
        )
    model_id = str(selected.get("model", "") or "").strip()
    if not model_id:
        raise ModelCatalogError(f"{label}.model must be non-empty")
    options = selected.get("options", {})
    if not isinstance(options, Mapping):
        raise ModelCatalogError(f"{label}.options must be an object")
    return model_id, copy.deepcopy(dict(options))


def _factory_identity(model: ResolvedModel) -> dict[str, Any]:
    identity = dict(model.definition["identity"])
    fields = {"provider_kind", "backend_id", "contract_version"}
    if model.kind in {"region_detector", "region_segmenter"}:
        fields.add("algorithm_id")
    return {field: identity[field] for field in fields}


def _factory_component(
    model: ResolvedModel,
    *,
    catalog: ModelCatalog,
    experiment_options: Mapping[str, Any],
    runtime_base: Path,
    path_replacements: dict[str, str],
    factory_replacements: dict[str, str],
    preflight: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    prepared = prepare_model_factory(model, catalog=catalog)
    factory_replacements[prepared.effective_spec] = prepared.original_spec
    kwargs = effective_factory_kwargs(
        model,
        catalog=catalog,
        experiment_options=experiment_options,
        experiment_base=runtime_base,
        replacements=path_replacements,
    )
    if preflight:
        instantiate_factory_model(
            model,
            catalog=catalog,
            experiment_options=experiment_options,
            experiment_base=runtime_base,
        )
    identity = _factory_identity(model)
    component_options = resolve_declared_paths(
        model.definition.get("options", {}),
        base_dir=catalog.base_dir,
        replacements=path_replacements,
        namespace="catalog",
    )
    reserved_component_fields = {
        "backend",
        "backend_id",
        "contract_version",
        "factory",
        "kwargs",
        "provider_kind",
    }
    if model.kind in {"region_detector", "region_segmenter"}:
        reserved_component_fields.add("algorithm_id")
    conflicts = sorted(reserved_component_fields.intersection(component_options))
    if conflicts:
        raise ModelCatalogError(
            f"factory model {model.model_id!r} options may not override: "
            + ", ".join(conflicts)
        )
    if model.kind == "depth":
        external_runtime = {
            "provider_kind": "external",
            "backend_id": identity["backend_id"],
            "contract_version": identity["contract_version"],
            "model_name": f"external:{identity['backend_id']}",
        }
        fp32 = bool(component_options.pop("fp32", False))
        input_size = int(component_options.pop("input_size", 512))
        if component_options:
            unsupported = ", ".join(sorted(component_options))
            raise ModelCatalogError(
                f"factory depth model {model.model_id!r} has unsupported options: "
                + unsupported
            )
        component = {
            "backend": "factory",
            "factory": "dream_exe.models.runtime:ExternalDepthEstimatorAdapter",
            "kwargs": {
                "provider_factory": prepared.effective_spec,
                "provider_kwargs": kwargs,
                "runtime_config": external_runtime,
                "fp32": fp32,
                "input_size": input_size,
            },
            "runtime_config": external_runtime,
            "fp32": fp32,
            "input_size": input_size,
        }
    else:
        component = {
            "backend": "factory",
            "factory": prepared.effective_spec,
            "kwargs": kwargs,
            **identity,
            **component_options,
        }
    manifest = {
        "model_id": model.model_id,
        "kind": model.kind,
        "backend": "factory",
        "identity": identity,
        "implementation": dict(prepared.implementation),
        "effective_options_sha256": _canonical_sha256(
            _public_value(
                {"kwargs": kwargs, "component_options": component_options},
                path_replacements=path_replacements,
                factory_replacements=factory_replacements,
            )
        ),
    }
    return component, manifest


def _builtin_component(
    model: ResolvedModel,
    *,
    current_component: Mapping[str, Any] | None,
    experiment_options: Mapping[str, Any],
    catalog: ModelCatalog | None,
    runtime_base: Path,
    path_replacements: dict[str, str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    catalog_base = runtime_base if catalog is None else catalog.base_dir
    options = resolve_declared_paths(
        model.definition.get("options", {}),
        base_dir=catalog_base,
        replacements=path_replacements,
        namespace="catalog",
    )
    experiment = resolve_declared_paths(
        experiment_options,
        base_dir=runtime_base,
        replacements=path_replacements,
        namespace="runtime",
    )
    for source, values in (("catalog", options), ("runtime", experiment)):
        if "backend" in values:
            raise ModelCatalogError(
                f"{source} options for model {model.model_id!r} may not "
                "override backend"
            )
    current = copy.deepcopy(dict(current_component or {}))
    current_backend = str(current.get("backend", "") or "").strip().lower()
    if current_backend == model.backend:
        base_component = current
    elif model.kind == "depth" and current_backend == "bench_resolved":
        providers = current.get("providers", {})
        if isinstance(providers, Mapping) and isinstance(
            providers.get(model.backend),
            Mapping,
        ):
            base_component = copy.deepcopy(dict(providers[model.backend]))
        else:
            base_component = {}
    else:
        base_component = {}
    component = _deep_merge(base_component, {"backend": model.backend})
    component = _deep_merge(component, options)
    component = _deep_merge(component, experiment)
    if model.kind == "depth" and "preset" not in component:
        component["preset"] = model.backend
    manifest = {
        "model_id": model.model_id,
        "kind": model.kind,
        "backend": model.backend,
        "identity": copy.deepcopy(model.definition.get("identity", {})),
        "implementation": {
            "source_kind": "dream_exe_builtin",
            "adapter": model.backend,
        },
        "effective_options_sha256": _canonical_sha256(
            _public_value(
                component,
                path_replacements=path_replacements,
                factory_replacements={},
            )
        ),
    }
    return component, manifest


def _selected_component(
    model_id: str,
    *,
    expected_kind: str,
    current_component: Mapping[str, Any] | None,
    experiment_options: Mapping[str, Any],
    catalog: ModelCatalog | None,
    runtime_base: Path,
    path_replacements: dict[str, str],
    factory_replacements: dict[str, str],
    preflight: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    model = resolve_model(
        model_id,
        catalog=catalog,
        expected_category="exec",
        expected_kind=expected_kind,
    )
    if model.backend == "factory":
        if catalog is None:
            raise ModelCatalogError("factory model selection requires --models-config")
        return _factory_component(
            model,
            catalog=catalog,
            experiment_options=experiment_options,
            runtime_base=runtime_base,
            path_replacements=path_replacements,
            factory_replacements=factory_replacements,
            preflight=preflight,
        )
    return _builtin_component(
        model,
        current_component=current_component,
        experiment_options=experiment_options,
        catalog=catalog,
        runtime_base=runtime_base,
        path_replacements=path_replacements,
    )


def _runtime_document(path: str | Path) -> tuple[Path, dict[str, Any], str]:
    source, document, digest = load_strict_json_object(
        path,
        label="runtime config",
    )
    return source, document, digest


def _public_value(
    value: Any,
    *,
    path_replacements: Mapping[str, str],
    factory_replacements: Mapping[str, str],
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _public_value(
                item,
                path_replacements=path_replacements,
                factory_replacements=factory_replacements,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        return [
            _public_value(
                item,
                path_replacements=path_replacements,
                factory_replacements=factory_replacements,
            )
            for item in value
        ]
    if isinstance(value, Path):
        value = value.as_posix()
    if isinstance(value, str):
        if value in factory_replacements:
            return factory_replacements[value]
        if value in path_replacements:
            return path_replacements[value]
    return copy.deepcopy(value)


def public_runtime_config(runtime_config: Mapping[str, Any]) -> dict[str, Any]:
    """Remove runtime-only loader paths while retaining all identity digests."""

    document = copy.deepcopy(dict(runtime_config))
    raw_meta = document.get("_meta", {})
    meta = dict(raw_meta) if isinstance(raw_meta, Mapping) else {}
    private = meta.pop(_PRIVATE_META_KEY, {})
    if not isinstance(private, Mapping):
        raise TypeError(f"runtime _meta.{_PRIVATE_META_KEY} must be a mapping")
    path_replacements = private.get("path_replacements", {})
    factory_replacements = private.get("factory_replacements", {})
    if not isinstance(path_replacements, Mapping) or not isinstance(
        factory_replacements, Mapping
    ):
        raise TypeError("runtime private replacement maps must be mappings")
    for field in private.get("remove_meta_fields", []):
        meta.pop(str(field), None)
    if meta:
        document["_meta"] = meta
    else:
        document.pop("_meta", None)
    public = _public_value(
        document,
        path_replacements=path_replacements,
        factory_replacements=factory_replacements,
    )
    if not isinstance(public, dict):  # internal invariant
        raise TypeError("public runtime config must remain a mapping")
    return public


def compose_runtime_config(
    *,
    defaults: Mapping[str, Any] | None,
    catalog: ModelCatalog | None = None,
    runtime_config_path: str | Path | None = None,
    preflight_factories: bool = True,
) -> RuntimeComposition:
    """Compose optional selectors over the existing runtime defaults.

    When neither catalog nor runtime file is supplied, callers should bypass
    this function to preserve byte-for-byte default behavior.  A runtime file
    without ``format`` remains the existing complete backend configuration.
    """

    path_replacements: dict[str, str] = {}
    factory_replacements: dict[str, str] = {}
    selected_manifests: list[dict[str, Any]] = []
    if runtime_config_path is None:
        runtime_source = None
        runtime_document: dict[str, Any] = {}
        runtime_digest = None
        runtime_base = catalog.base_dir if catalog is not None else Path.cwd()
    else:
        runtime_source, runtime_document, runtime_digest = _runtime_document(
            runtime_config_path
        )
        runtime_base = runtime_source.parent

    is_selector_runtime = runtime_document.get("format") == RUNTIME_MODEL_FORMAT
    if "format" in runtime_document and not is_selector_runtime:
        raise ModelCatalogError(
            f"runtime config format must be {RUNTIME_MODEL_FORMAT!r}; "
            "legacy complete runtime configs must omit format"
        )
    if is_selector_runtime:
        unsupported = sorted(set(runtime_document).difference(_RUNTIME_ROOT_FIELDS))
        if unsupported:
            raise ModelCatalogError(
                "runtime config contains unsupported fields: "
                + ", ".join(unsupported)
            )
        category = str(runtime_document.get("category", "exec") or "").strip()
        if category != "exec":
            raise ModelCatalogError(
                "dream-exe.runtime composes only category 'exec' models"
            )
        resolved = copy.deepcopy(dict(defaults or {}))
        if not resolved:
            resolved = {
                "region": {"backend": "composed"},
                "depth_calibration": {"backend": "none"},
            }
        if runtime_document.get("device") is not None:
            device = str(runtime_document["device"] or "").strip()
            if not device:
                raise ModelCatalogError("runtime device must be non-empty")
            resolved["device"] = device

        raw_region = runtime_document.get("region")
        if raw_region is not None:
            if not isinstance(raw_region, Mapping):
                raise ModelCatalogError("runtime region must be an object")
            region = copy.deepcopy(dict(resolved.get("region", {})))
            if str(region.get("backend", "") or "").lower() != "composed":
                region = {"backend": "composed", "query_samplers": {}}
            unsupported_region = sorted(
                set(raw_region).difference({"detector", "segmenter"})
            )
            if unsupported_region:
                raise ModelCatalogError(
                    "runtime region contains unsupported fields: "
                    + ", ".join(unsupported_region)
                )
            for role, kind in (
                ("detector", "region_detector"),
                ("segmenter", "region_segmenter"),
            ):
                if role not in raw_region:
                    continue
                selection = _selector(raw_region[role], label=f"runtime.region.{role}")
                if selection is None:
                    if not isinstance(raw_region[role], Mapping):
                        raise ModelCatalogError(
                            f"runtime.region.{role} must be an object"
                        )
                    region[role] = copy.deepcopy(dict(raw_region[role]))
                    continue
                model_id, options = selection
                component, manifest = _selected_component(
                    model_id,
                    expected_kind=kind,
                    current_component=(
                        region.get(role)
                        if isinstance(region.get(role), Mapping)
                        else None
                    ),
                    experiment_options=options,
                    catalog=catalog,
                    runtime_base=runtime_base,
                    path_replacements=path_replacements,
                    factory_replacements=factory_replacements,
                    preflight=preflight_factories,
                )
                region[role] = component
                selected_manifests.append(manifest)
            resolved["region"] = region

        for stage, kind in (
            ("tracking", "tracking"),
            ("depth", "depth"),
            ("pose", "pose"),
        ):
            if stage not in runtime_document:
                continue
            value = runtime_document[stage]
            selection = _selector(value, label=f"runtime.{stage}")
            if selection is None:
                if not isinstance(value, Mapping):
                    raise ModelCatalogError(f"runtime.{stage} must be an object")
                resolved[stage] = copy.deepcopy(dict(value))
                continue
            model_id, options = selection
            component, manifest = _selected_component(
                model_id,
                expected_kind=kind,
                current_component=(
                    resolved.get(stage)
                    if isinstance(resolved.get(stage), Mapping)
                    else None
                ),
                experiment_options=options,
                catalog=catalog,
                runtime_base=runtime_base,
                path_replacements=path_replacements,
                factory_replacements=factory_replacements,
                preflight=preflight_factories,
            )
            resolved[stage] = component
            selected_manifests.append(manifest)
    elif runtime_config_path is not None:
        resolved = copy.deepcopy(runtime_document)
        if any(
            _selector(resolved.get(stage), label=f"runtime.{stage}") is not None
            for stage in ("tracking", "depth", "pose")
        ):
            raise ModelCatalogError(
                f"model selectors require runtime format {RUNTIME_MODEL_FORMAT!r}"
            )
        existing_meta = resolved.get("_meta", {})
        if existing_meta is None:
            existing_meta = {}
        if not isinstance(existing_meta, Mapping):
            raise ModelCatalogError("legacy runtime _meta must be an object")
        resolved["_meta"] = {
            **dict(existing_meta),
            "source": runtime_source.as_posix(),
            "base_dir": runtime_base.as_posix(),
        }
    else:
        resolved = copy.deepcopy(dict(defaults or {}))

    if not str(resolved.get("device", "") or "").strip():
        raise ModelCatalogError("resolved runtime requires a device")
    for stage in ("region", "tracking", "depth"):
        if not isinstance(resolved.get(stage), Mapping):
            raise ModelCatalogError(f"resolved runtime requires {stage}")
    resolved.setdefault("pose", {"backend": "none"})
    resolved.setdefault("depth_calibration", {"backend": "none"})

    meta = resolved.get("_meta", {})
    if meta is None:
        meta = {}
    if not isinstance(meta, Mapping):
        raise ModelCatalogError("resolved runtime _meta must be an object")
    meta_payload = copy.deepcopy(dict(meta))
    if runtime_config_path is not None:
        meta_payload["base_dir"] = runtime_base.as_posix()
    meta_payload[_COMPOSITION_META_KEY] = {
        "format": RUNTIME_MODEL_FORMAT,
        "catalog_format": MODEL_CATALOG_FORMAT,
        "catalog_sha256": None if catalog is None else catalog.sha256,
        "runtime_sha256": runtime_digest,
        "selected_models": selected_manifests,
    }
    remove_fields: list[str] = []
    if runtime_config_path is not None:
        remove_fields.append("base_dir")
    if runtime_config_path is not None and not is_selector_runtime:
        remove_fields.append("source")
    meta_payload[_PRIVATE_META_KEY] = {
        "path_replacements": path_replacements,
        "factory_replacements": factory_replacements,
        "remove_meta_fields": remove_fields,
    }
    resolved["_meta"] = meta_payload
    public = public_runtime_config(resolved)
    return RuntimeComposition(
        runtime_config=resolved,
        public_config=public,
        selected_models=tuple(selected_manifests),
        runtime_sha256=runtime_digest,
        catalog_sha256=None if catalog is None else catalog.sha256,
    )


class ExternalDepthEstimatorAdapter:
    """Pipeline-facing lazy adapter around ``BaseDepthBackend.infer``."""

    def __init__(
        self,
        *,
        provider_factory: str,
        provider_kwargs: Mapping[str, Any],
        runtime_config: Mapping[str, Any],
        fp32: bool = False,
        input_size: int = 512,
    ) -> None:
        self.provider_factory = str(provider_factory)
        self.provider_kwargs = copy.deepcopy(dict(provider_kwargs))
        self.runtime_config = copy.deepcopy(dict(runtime_config))
        self.fp32 = bool(fp32)
        self.input_size = int(input_size)
        if self.input_size <= 0:
            raise ValueError("external depth input_size must be positive")
        self._provider: Any | None = None

    def _load(self) -> Any:
        if self._provider is not None:
            return self._provider
        module_name, separator, attribute = self.provider_factory.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError("provider_factory must use module:attribute")
        try:
            module = importlib.import_module(module_name)
            factory = getattr(module, attribute)
            provider = factory(**copy.deepcopy(self.provider_kwargs))
        except ModuleNotFoundError as error:
            raise RuntimeError(
                f"external depth provider dependency {error.name!r} is missing"
            ) from error
        if not callable(getattr(provider, "infer", None)):
            raise TypeError("external depth provider must expose infer(...)")
        self._provider = provider
        return provider

    def __call__(
        self,
        *,
        video_frames: Any,
        target_fps: float,
        intrinsics: Any = None,
        extrinsics: Any = None,
    ) -> Any:
        provider = self._load()
        prediction = provider.infer(
            video_frames,
            target_fps,
            fp32=self.fp32,
            input_size=self.input_size,
            intrinsics=intrinsics,
            extrinsics=extrinsics,
        )
        from ..video2traj.depth.estimator import _normalize_backend_prediction

        return _normalize_backend_prediction(
            prediction,
            runtime_config=self.runtime_config,
            target_fps=target_fps,
        )


__all__ = [
    "ExternalDepthEstimatorAdapter",
    "RUNTIME_MODEL_FORMAT",
    "RuntimeComposition",
    "compose_runtime_config",
    "public_runtime_config",
]
