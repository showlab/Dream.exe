"""Static and optional real-inference checks for model catalog entries."""

from __future__ import annotations

import copy
import hashlib
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from .catalog import (
    ModelCatalog,
    ModelCatalogError,
    ModelIntegrationError,
    ResolvedModel,
    check_declared_assets,
    instantiate_factory_model,
    instantiate_video_generation_model,
    instantiate_vlm_model,
    load_strict_json_object,
    prepare_model_factory,
    resolve_model,
)


def _summarize_result(value: Any) -> dict[str, Any]:
    summary: dict[str, Any] = {"python_type": type(value).__name__}
    shape = getattr(value, "shape", None)
    if shape is not None:
        summary["shape"] = [int(item) for item in shape]
    if isinstance(value, str):
        summary["text_length"] = len(value)
        summary["text_sha256"] = hashlib.sha256(value.encode("utf-8")).hexdigest()
    elif isinstance(value, Mapping):
        summary["mapping_keys"] = sorted(str(key) for key in value)
    elif isinstance(value, (list, tuple)):
        summary["sequence_length"] = len(value)
    return summary


def _smoke_factory_payload(
    path: str | Path,
) -> tuple[list[Any], dict[str, Any]]:
    source, document, digest = load_strict_json_object(
        path,
        label="model smoke input",
    )
    if set(document).difference({"factory", "kwargs"}):
        unsupported = sorted(set(document).difference({"factory", "kwargs"}))
        raise ModelCatalogError(
            "model smoke input contains unsupported fields: "
            + ", ".join(unsupported)
        )
    factory_spec = str(document.get("factory", "") or "").strip()
    if not factory_spec:
        raise ModelCatalogError("model smoke input requires factory")
    kwargs = document.get("kwargs", {})
    if not isinstance(kwargs, Mapping):
        raise ModelCatalogError("model smoke input kwargs must be an object")
    fake_catalog = ModelCatalog(
        source=source,
        sha256=digest,
        models={},
        categories={"video_gen": (), "exec": (), "eval": ()},
        layout="smoke-input",
    )
    fake_model = ResolvedModel(
        model_id="smoke_input",
        category="exec",
        kind="pose",
        backend="factory",
        definition={"factory": factory_spec},
        source="catalog",
        catalog_sha256=digest,
    )
    prepared = prepare_model_factory(fake_model, catalog=fake_catalog)
    try:
        payload = prepared.callable(**copy.deepcopy(dict(kwargs)))
    except ModuleNotFoundError as error:
        raise ModelIntegrationError(
            "dependency",
            f"smoke input factory requires missing dependency {error.name!r}",
        ) from error
    if not isinstance(payload, Mapping):
        raise ModelIntegrationError(
            "smoke", "smoke input factory must return {'args': [...], 'kwargs': {...}}"
        )
    if set(payload).difference({"args", "kwargs"}):
        raise ModelIntegrationError(
            "smoke", "smoke input factory result may contain only args and kwargs"
        )
    args = payload.get("args", [])
    call_kwargs = payload.get("kwargs", {})
    if not isinstance(args, (list, tuple)) or not isinstance(call_kwargs, Mapping):
        raise ModelIntegrationError(
            "smoke", "smoke input factory args/kwargs have invalid types"
        )
    return list(args), dict(call_kwargs)


def run_model_smoke(
    instance: Any,
    *,
    model: ResolvedModel,
    smoke_input: str | Path,
) -> dict[str, Any]:
    """Run an explicit caller-built minimal call and summarize the output."""

    args, kwargs = _smoke_factory_payload(smoke_input)
    method_name = {
        "vlm": "infer",
        "video_generation": "generate",
        "region_detector": "detect",
        "region_segmenter": "segment_from_bbox",
        "tracking": (
            "predict" if callable(getattr(instance, "predict", None)) else "track"
        ),
        "depth": "infer",
        "pose": "infer",
    }[model.kind]
    method = getattr(instance, method_name, None)
    if not callable(method):
        raise ModelIntegrationError(
            "smoke", f"model does not expose callable {method_name}(...)"
        )
    result = method(*args, **kwargs)
    return {
        "status": "passed",
        "method": method_name,
        "result": _summarize_result(result),
    }


def check_model(
    *,
    catalog: ModelCatalog,
    model_id: str,
    expected_category: str | None = None,
    expected_kind: str | None = None,
    smoke_input: str | Path | None = None,
) -> dict[str, Any]:
    """Validate config, factory, constructor, protocol, identity, and assets."""

    model = resolve_model(
        model_id,
        catalog=catalog,
        expected_category=expected_category,
        expected_kind=expected_kind,
    )
    report: dict[str, Any] = {
        "format": "dream-exe.model-check",
        "status": "passed",
        "model": model.model_id,
        "category": model.category,
        "kind": model.kind,
        "backend": model.backend,
        "source": model.source,
        "catalog_sha256": catalog.sha256,
        "checks": {
            "catalog": "passed",
            "secrets": "passed",
        },
    }
    instance: Any | None = None
    if model.backend == "factory":
        instance, prepared, actual_identity = instantiate_factory_model(
            model,
            catalog=catalog,
        )
        report["implementation"] = dict(prepared.implementation)
        report["identity"] = actual_identity
        report["checks"].update(
            {
                "source": "passed",
                "dependencies": "passed",
                "constructor": "passed",
                "signature": "passed",
                "identity": "passed",
            }
        )
        assets = check_declared_assets(
            model.definition.get("kwargs", {}),
            base_dir=catalog.base_dir,
        )
    elif model.kind == "vlm":
        instance, details = instantiate_vlm_model(
            model,
            catalog=catalog,
            require_credentials=smoke_input is not None,
        )
        report.update(details)
        report["checks"].update(
            {
                "adapter": "passed",
                "identity": "passed",
                "credential": (
                    "passed"
                    if details.get("credential_configured")
                    else "not-required-for-static-check"
                ),
            }
        )
        assets = []
    elif model.kind == "video_generation":
        instance, details, parameters = instantiate_video_generation_model(
            model,
            catalog=catalog,
        )
        report.update(details)
        report["default_parameters"] = parameters
        report["checks"].update(
            {
                "adapter": "passed",
                "assets": "passed",
                "identity": "passed",
            }
        )
        assets = check_declared_assets(
            model.definition.get("options", {}),
            base_dir=catalog.base_dir,
        )
    else:
        assets = check_declared_assets(
            model.definition.get("options", {}),
            base_dir=catalog.base_dir,
        )
        report["identity"] = copy.deepcopy(model.definition.get("identity", {}))
        report["checks"].update(
            {
                "adapter": "passed",
                "declared_assets": "passed",
                "runtime_construction": "deferred-to-runtime-builder",
            }
        )
    report["declared_assets"] = assets
    if smoke_input is not None:
        if instance is None:
            raise ModelIntegrationError(
                "smoke",
                "built-in pipeline components are smoke-tested through runtime.json; "
                "use a factory model for models check --smoke-input",
            )
        report["smoke"] = run_model_smoke(
            instance,
            model=model,
            smoke_input=smoke_input,
        )
    return report


__all__ = ["check_model", "run_model_smoke"]
