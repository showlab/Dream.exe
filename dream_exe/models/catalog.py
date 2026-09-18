"""Versioned, caller-owned model catalog and explicit factory loading.

Catalog loading is intentionally read-only.  It never downloads weights,
clones repositories, installs packages, or mutates third-party source trees.
Python factories are executable configuration and must therefore come from a
trusted, caller-selected ``models.json`` file.
"""

from __future__ import annotations

import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import os
import re
import stat
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any
from urllib.parse import urlsplit


MODEL_CATALOG_FORMAT = "dream-exe.models"
MODEL_CATEGORIES = (
    "video_gen",
    "exec",
    "eval",
)
MODEL_KINDS_BY_CATEGORY = MappingProxyType(
    {
        "video_gen": ("video_generation",),
        "exec": (
            "region_detector",
            "region_segmenter",
            "tracking",
            "depth",
            "pose",
        ),
        "eval": ("vlm",),
    }
)
MODEL_CATEGORY_BY_KIND = MappingProxyType(
    {
        kind: category
        for category, kinds in MODEL_KINDS_BY_CATEGORY.items()
        for kind in kinds
    }
)
MODEL_KINDS = (
    "video_generation",
    "region_detector",
    "region_segmenter",
    "tracking",
    "depth",
    "pose",
    "vlm",
)
MAX_MODEL_CATALOG_BYTES = 4 * 1024 * 1024
_SAFE_MODEL_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_SAFE_BACKEND = re.compile(r"[a-z0-9][a-z0-9._-]{0,127}\Z")
_SAFE_IDENTITY = re.compile(r"[a-z0-9][a-z0-9._-]{0,191}\Z")
_SAFE_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
_MODULE_NAME = re.compile(
    r"[A-Za-z_]\w*(?:\.[A-Za-z_]\w*)*\Z"
)
_ATTRIBUTE_NAME = re.compile(r"[A-Za-z_]\w*\Z")
_PATH_SUFFIXES = (
    "_checkpoint",
    "_dir",
    "_file",
    "_path",
    "_root",
)


class ModelCatalogError(ValueError):
    """Invalid catalog syntax, identity, or selection."""


class ModelIntegrationError(RuntimeError):
    """Layered factory, dependency, asset, or protocol failure."""

    def __init__(self, category: str, message: str) -> None:
        self.category = str(category)
        super().__init__(f"[{self.category}] {message}")


@dataclass(frozen=True)
class ModelCatalog:
    """One validated file-backed catalog."""

    source: Path
    sha256: str
    models: Mapping[str, Mapping[str, Any]]
    categories: Mapping[str, tuple[str, ...]]
    layout: str

    @property
    def base_dir(self) -> Path:
        return self.source.parent


@dataclass(frozen=True)
class ResolvedModel:
    """One exact catalog or built-in model selection."""

    model_id: str
    category: str
    kind: str
    backend: str
    definition: Mapping[str, Any]
    source: str
    catalog_sha256: str | None


@dataclass(frozen=True)
class PreparedFactory:
    """Imported factory plus portable implementation evidence."""

    callable: Any
    effective_spec: str
    original_spec: str
    implementation: Mapping[str, Any]


def _builtin_models() -> dict[str, dict[str, Any]]:
    from ..video2traj.depth.contract import DEPTH_ESTIMATOR_CONTRACT_VERSION
    from ..video2traj.pose.contract import POSE_BACKEND_CONTRACT_VERSION
    from ..video2traj.region.contract import (
        REGION_DETECTOR_CONTRACT_VERSION,
        REGION_SEGMENTER_CONTRACT_VERSION,
    )
    from ..video2traj.tracking.core import TRACKING_BACKEND_CONTRACT_VERSION

    return {
        "Wan2.2": {
            "kind": "video_generation",
            "backend": "wan2.2",
            "options": {},
        },
        "GroundingDINO": {
            "kind": "region_detector",
            "backend": "grounding_dino",
            "options": {},
            "identity": {
                "provider_kind": "builtin",
                "backend_id": "grounding_dino_box_detector",
                "algorithm_id": "text_conditioned_box_detection",
                "contract_version": REGION_DETECTOR_CONTRACT_VERSION,
            },
        },
        "SAM2": {
            "kind": "region_segmenter",
            "backend": "sam2",
            "options": {},
            "identity": {
                "provider_kind": "builtin",
                "backend_id": "sam2_box_segmenter",
                "algorithm_id": "box_conditioned_mask_segmentation",
                "contract_version": REGION_SEGMENTER_CONTRACT_VERSION,
            },
        },
        "CoTracker": {
            "kind": "tracking",
            "backend": "cotracker",
            "options": {},
            "identity": {
                "provider_kind": "builtin",
                "backend_id": "cotracker",
                "contract_version": TRACKING_BACKEND_CONTRACT_VERSION,
            },
        },
        "DVD": {
            "kind": "depth",
            "backend": "dvd",
            "options": {},
            "identity": {
                "provider_kind": "builtin",
                "backend_id": "dvd",
                "contract_version": DEPTH_ESTIMATOR_CONTRACT_VERSION,
            },
        },
        "VDA": {
            "kind": "depth",
            "backend": "vda",
            "options": {},
            "identity": {
                "provider_kind": "builtin",
                "backend_id": "vda",
                "contract_version": DEPTH_ESTIMATOR_CONTRACT_VERSION,
            },
        },
        "Kabsch": {
            "kind": "pose",
            "backend": "pointcloud_kabsch",
            "options": {},
        },
        "FoundationPose": {
            "kind": "pose",
            "backend": "foundationpose",
            "options": {},
            "identity": {
                "provider_kind": "builtin",
                "backend_id": "foundationpose",
                "contract_version": POSE_BACKEND_CONTRACT_VERSION,
            },
        },
        "FreePose": {
            "kind": "pose",
            "backend": "freepose",
            "options": {},
            "identity": {
                "provider_kind": "builtin",
                "backend_id": "freepose",
                "contract_version": POSE_BACKEND_CONTRACT_VERSION,
            },
        },
        "SinRef-6D": {
            "kind": "pose",
            "backend": "sinref6d",
            "options": {},
            "identity": {
                "provider_kind": "builtin",
                "backend_id": "sinref6d",
                "contract_version": POSE_BACKEND_CONTRACT_VERSION,
            },
        },
    }


def model_category_for_kind(kind: str) -> str:
    """Return the single lifecycle category that owns one backend kind."""

    clean_kind = str(kind or "").strip()
    try:
        return MODEL_CATEGORY_BY_KIND[clean_kind]
    except KeyError as error:
        raise ModelCatalogError(f"unsupported model kind: {clean_kind!r}") from error


BUILTIN_MODELS = MappingProxyType(
    {
        model_id: MappingProxyType(
            {
                **copy.deepcopy(definition),
                "category": model_category_for_kind(str(definition["kind"])),
            }
        )
        for model_id, definition in _builtin_models().items()
    }
)


_BACKENDS_BY_KIND = {
    "vlm": frozenset({"factory", "openai-compatible"}),
    "video_generation": frozenset({"factory", "wan2.2"}),
    "region_detector": frozenset({"factory", "grounding_dino"}),
    "region_segmenter": frozenset({"factory", "sam2"}),
    "tracking": frozenset({"factory", "cotracker"}),
    "depth": frozenset({"factory", "dvd", "vda"}),
    "pose": frozenset(
        {
            "factory",
            "foundationpose",
            "freepose",
            "pointcloud_kabsch",
            "sinref6d",
        }
    ),
}


def _strict_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ModelCatalogError(f"duplicate JSON key: {key}")
        output[key] = value
    return output


def _reject_constant(value: str) -> Any:
    raise ModelCatalogError(f"invalid JSON constant: {value}")


def load_strict_json_object(
    path: str | Path,
    *,
    label: str,
    max_bytes: int = MAX_MODEL_CATALOG_BYTES,
) -> tuple[Path, dict[str, Any], str]:
    """Read a bounded, duplicate-key-free strict JSON object."""

    source = Path(path).expanduser().resolve()
    try:
        metadata = source.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} not found: {source}") from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ModelCatalogError(f"{label} must be a regular non-symlink file")
    if metadata.st_size > max_bytes:
        raise ModelCatalogError(f"{label} exceeds {max_bytes} bytes")
    encoded = source.read_bytes()
    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_strict_object,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ModelCatalogError(f"{label} must be UTF-8") from error
    except json.JSONDecodeError as error:
        raise ModelCatalogError(f"{label} is not valid JSON: {error}") from error
    if not isinstance(payload, dict):
        raise ModelCatalogError(f"{label} must be a JSON object")
    return source, payload, hashlib.sha256(encoded).hexdigest()


def _normalized_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(value).lower()).strip("_")


def _is_sensitive_key(value: Any) -> bool:
    normalized = _normalized_key(value)
    credential_name = (
        normalized[:-4] if normalized.endswith("_env") else normalized
    )
    exact = {
        "api_key",
        "api_token",
        "access_key",
        "access_token",
        "auth_token",
        "bearer_token",
        "client_key",
        "password",
        "private_key",
        "secret",
        "secret_key",
        "token_file",
        "token_path",
        "token_value",
        "token",
        "client_secret",
        "authorization",
        "credential",
        "credentials",
        "headers",
        "extra_headers",
        "default_headers",
    }
    sensitive_stems = (
        "api_key",
        "api_token",
        "access_key",
        "access_token",
        "auth_token",
        "bearer_token",
        "client_key",
        "client_secret",
        "authorization",
        "credential",
        "credentials",
        "headers",
        "password",
        "private_key",
        "secret",
        "secret_key",
    )
    return (
        credential_name in exact
        or any(
            credential_name.startswith(stem + "_")
            for stem in sensitive_stems
        )
        or credential_name.endswith(
            (
                "_api_key",
                "_access_key",
                "_access_token",
                "_auth_token",
                "_token",
                "_password",
                "_secret",
                "_authorization",
            )
        )
    )


def _validate_no_secrets(value: Any, *, pointer: str = "") -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            path = f"{pointer}.{key_text}" if pointer else key_text
            if _is_sensitive_key(key_text):
                if not _normalized_key(key_text).endswith("_env"):
                    raise ModelCatalogError(
                        f"catalog must not store credentials at {path}; "
                        "store only an environment-variable name"
                    )
                if not isinstance(item, str) or not _SAFE_ENV_NAME.fullmatch(item):
                    raise ModelCatalogError(
                        f"{path} must be an environment-variable name"
                    )
            _validate_no_secrets(item, pointer=path)
        return
    if isinstance(value, list):
        for index, item in enumerate(value):
            _validate_no_secrets(item, pointer=f"{pointer}[{index}]")
        return
    if not isinstance(value, str):
        return
    if re.search(r"(?i)\bbearer\s+\S+", value):
        raise ModelCatalogError(f"catalog contains a bearer credential at {pointer}")
    if re.search(r"(?i)\bsk-(?:proj-)?[a-z0-9_-]{16,}\b", value):
        raise ModelCatalogError(f"catalog contains an API credential at {pointer}")
    if value.startswith(("http://", "https://")):
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            raise ModelCatalogError(f"catalog URL contains credentials at {pointer}")


def contract_version_for_kind(kind: str) -> str:
    """Return the repository's current public provider contract version."""

    if kind == "vlm":
        from ..evaluation.vlm.base import VLM_BACKEND_CONTRACT_VERSION

        return VLM_BACKEND_CONTRACT_VERSION
    if kind == "video_generation":
        from ..generation.video import IMAGE_TO_VIDEO_BACKEND_CONTRACT_VERSION

        return IMAGE_TO_VIDEO_BACKEND_CONTRACT_VERSION
    if kind == "region_detector":
        from ..video2traj.region.contract import REGION_DETECTOR_CONTRACT_VERSION

        return REGION_DETECTOR_CONTRACT_VERSION
    if kind == "region_segmenter":
        from ..video2traj.region.contract import REGION_SEGMENTER_CONTRACT_VERSION

        return REGION_SEGMENTER_CONTRACT_VERSION
    if kind == "tracking":
        from ..video2traj.tracking.core import TRACKING_BACKEND_CONTRACT_VERSION

        return TRACKING_BACKEND_CONTRACT_VERSION
    if kind == "depth":
        from ..video2traj.depth.contract import DEPTH_ESTIMATOR_CONTRACT_VERSION

        return DEPTH_ESTIMATOR_CONTRACT_VERSION
    if kind == "pose":
        from ..video2traj.pose.contract import POSE_BACKEND_CONTRACT_VERSION

        return POSE_BACKEND_CONTRACT_VERSION
    raise ModelCatalogError(f"unsupported model kind: {kind!r}")


def _identity(
    value: Any,
    *,
    kind: str,
    source: str,
    required: bool,
) -> dict[str, Any]:
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise ModelCatalogError(f"{source}.identity must be an object")
    identity = copy.deepcopy(dict(value))
    required_fields = {"provider_kind", "backend_id", "contract_version"}
    if kind in {"region_detector", "region_segmenter"}:
        required_fields.add("algorithm_id")
    missing = sorted(field for field in required_fields if field not in identity)
    if missing:
        raise ModelCatalogError(
            f"{source}.identity is missing: {', '.join(missing)}"
        )
    provider_kind = str(identity["provider_kind"] or "").strip().lower()
    if provider_kind not in {"builtin", "external"}:
        raise ModelCatalogError(
            f"{source}.identity.provider_kind must be 'builtin' or 'external'"
        )
    backend_id = str(identity["backend_id"] or "").strip().lower()
    if not _SAFE_IDENTITY.fullmatch(backend_id):
        raise ModelCatalogError(
            f"{source}.identity.backend_id must use [a-z0-9._-]"
        )
    expected_contract = contract_version_for_kind(kind)
    contract = str(identity["contract_version"] or "").strip()
    if contract != expected_contract:
        raise ModelCatalogError(
            f"{source}.identity.contract_version must be {expected_contract!r}"
        )
    identity["provider_kind"] = provider_kind
    identity["backend_id"] = backend_id
    identity["contract_version"] = contract
    if "algorithm_id" in required_fields:
        algorithm_id = str(identity["algorithm_id"] or "").strip().lower()
        if not _SAFE_IDENTITY.fullmatch(algorithm_id):
            raise ModelCatalogError(
                f"{source}.identity.algorithm_id must use [a-z0-9._-]"
            )
        identity["algorithm_id"] = algorithm_id
    return identity


def _validate_entry(
    model_id: str,
    value: Any,
    *,
    expected_category: str | None = None,
    source_prefix: str = "models",
) -> dict[str, Any]:
    source = f"{source_prefix}.{model_id}"
    if not _SAFE_MODEL_ID.fullmatch(model_id):
        raise ModelCatalogError(
            f"model ID {model_id!r} must match [A-Za-z0-9][A-Za-z0-9._-]*"
        )
    if model_id in BUILTIN_MODELS:
        raise ModelCatalogError(
            f"catalog model ID {model_id!r} duplicates a built-in model ID"
        )
    if not isinstance(value, Mapping):
        raise ModelCatalogError(f"{source} must be an object")
    entry = copy.deepcopy(dict(value))
    allowed = {
        "category",
        "kind",
        "backend",
        "factory",
        "kwargs",
        "options",
        "identity",
    }
    unsupported = sorted(set(entry).difference(allowed))
    if unsupported:
        raise ModelCatalogError(
            f"{source} contains unsupported fields: "
            + ", ".join(unsupported)
        )
    kind = str(entry.get("kind", "") or "").strip()
    if kind not in MODEL_KINDS:
        raise ModelCatalogError(
            f"{source}.kind must be one of: {', '.join(MODEL_KINDS)}"
        )
    category = model_category_for_kind(kind)
    declared_category = str(entry.get("category", "") or "").strip()
    if declared_category and declared_category not in MODEL_CATEGORIES:
        raise ModelCatalogError(
            f"{source}.category must be one of: {', '.join(MODEL_CATEGORIES)}"
        )
    if declared_category and declared_category != category:
        raise ModelCatalogError(
            f"{source} kind {kind!r} belongs to category {category!r}, "
            f"not {declared_category!r}"
        )
    if expected_category is not None and category != expected_category:
        raise ModelCatalogError(
            f"{source} kind {kind!r} belongs to category {category!r}, "
            f"so it cannot be listed under category {expected_category!r}"
        )
    backend = str(entry.get("backend", "") or "").strip().lower()
    if not _SAFE_BACKEND.fullmatch(backend):
        raise ModelCatalogError(f"{source}.backend is invalid")
    if backend not in _BACKENDS_BY_KIND[kind]:
        raise ModelCatalogError(
            f"{source} backend {backend!r} is unsupported for {kind}; "
            f"available adapters: {', '.join(sorted(_BACKENDS_BY_KIND[kind]))}"
        )
    kwargs = entry.get("kwargs", {})
    options = entry.get("options", {})
    if not isinstance(kwargs, Mapping):
        raise ModelCatalogError(f"{source}.kwargs must be an object")
    if not isinstance(options, Mapping):
        raise ModelCatalogError(f"{source}.options must be an object")
    if backend == "factory":
        factory = str(entry.get("factory", "") or "").strip()
        if not factory:
            raise ModelCatalogError(f"{source}.factory is required")
        identity = _identity(
            entry.get("identity"),
            kind=kind,
            source=source,
            required=True,
        )
        if identity["provider_kind"] != "external":
            raise ModelCatalogError(
                f"{source}.identity.provider_kind must be 'external' "
                "for factory backends"
            )
    else:
        if "factory" in entry or bool(kwargs):
            raise ModelCatalogError(
                f"{source} may use factory/kwargs only with backend='factory'"
            )
        identity = _identity(
            entry.get("identity"),
            kind=kind,
            source=source,
            required=False,
        )
    normalized = {
        "category": category,
        "kind": kind,
        "backend": backend,
        "options": copy.deepcopy(dict(options)),
    }
    if backend == "factory":
        normalized["factory"] = str(entry["factory"]).strip()
        normalized["kwargs"] = copy.deepcopy(dict(kwargs))
    if identity:
        normalized["identity"] = identity
    _validate_no_secrets(normalized, pointer=source)
    try:
        json.dumps(normalized, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ModelCatalogError(
            f"{source} must contain strict JSON values"
        ) from error
    return normalized


def load_model_catalog(path: str | Path) -> ModelCatalog:
    """Load and fully validate one optional caller-owned catalog."""

    source, payload, digest = load_strict_json_object(
        path,
        label="models config",
    )
    if payload.get("format") != MODEL_CATALOG_FORMAT:
        raise ModelCatalogError(
            f"models config format must be {MODEL_CATALOG_FORMAT!r}"
        )
    fields = set(payload)
    categorized_fields = {"format", "categories"}
    flat_fields = {"format", "models"}
    if fields == categorized_fields:
        layout = "categorized"
        raw_categories = payload["categories"]
        if not isinstance(raw_categories, Mapping):
            raise ModelCatalogError("models config categories must be an object")
        unknown_categories = sorted(set(raw_categories).difference(MODEL_CATEGORIES))
        if unknown_categories:
            raise ModelCatalogError(
                "models config contains unknown categories: "
                + ", ".join(unknown_categories)
            )
        models: dict[str, dict[str, Any]] = {}
        members = {category: [] for category in MODEL_CATEGORIES}
        for category in MODEL_CATEGORIES:
            raw_models = raw_categories.get(category, {})
            if not isinstance(raw_models, Mapping):
                raise ModelCatalogError(
                    f"models config categories.{category} must be an object"
                )
            for raw_model_id, definition in raw_models.items():
                model_id = str(raw_model_id)
                if model_id in models:
                    raise ModelCatalogError(
                        f"catalog model ID {model_id!r} appears in multiple categories"
                    )
                models[model_id] = _validate_entry(
                    model_id,
                    definition,
                    expected_category=category,
                    source_prefix=f"categories.{category}",
                )
                members[category].append(model_id)
    elif fields == flat_fields:
        layout = "flat-legacy"
        raw_models = payload["models"]
        if not isinstance(raw_models, Mapping):
            raise ModelCatalogError("models config models must be an object")
        models = {
            str(model_id): _validate_entry(str(model_id), definition)
            for model_id, definition in raw_models.items()
        }
        members = {
            category: [
                model_id
                for model_id, definition in models.items()
                if definition["category"] == category
            ]
            for category in MODEL_CATEGORIES
        }
    else:
        raise ModelCatalogError(
            "models config must contain exactly format + categories; "
            "legacy format + models remains readable"
        )
    return ModelCatalog(
        source=source,
        sha256=digest,
        models=MappingProxyType(models),
        categories=MappingProxyType(
            {
                category: tuple(members[category])
                for category in MODEL_CATEGORIES
            }
        ),
        layout=layout,
    )


def available_models(
    catalog: ModelCatalog | None = None,
    *,
    category: str | None = None,
    kind: str | None = None,
) -> list[dict[str, Any]]:
    """List built-in and catalog models without importing any factories."""

    if category is not None and category not in MODEL_CATEGORIES:
        raise ModelCatalogError(f"unknown model category: {category!r}")
    if kind is not None and kind not in MODEL_KINDS:
        raise ModelCatalogError(f"unknown model kind: {kind!r}")
    if kind is not None and category is not None:
        kind_category = model_category_for_kind(kind)
        if kind_category != category:
            raise ModelCatalogError(
                f"model kind {kind!r} belongs to category {kind_category!r}, "
                f"not {category!r}"
            )
    rows: list[dict[str, Any]] = []
    for model_id, definition in BUILTIN_MODELS.items():
        if (
            (category is None or definition["category"] == category)
            and (kind is None or definition["kind"] == kind)
        ):
            rows.append(
                {
                    "id": model_id,
                    "category": definition["category"],
                    "kind": definition["kind"],
                    "backend": definition["backend"],
                    "source": "builtin",
                }
            )
    if catalog is not None:
        for model_id, definition in catalog.models.items():
            if (
                (category is None or definition["category"] == category)
                and (kind is None or definition["kind"] == kind)
            ):
                rows.append(
                    {
                        "id": model_id,
                        "category": definition["category"],
                        "kind": definition["kind"],
                        "backend": definition["backend"],
                        "source": "catalog",
                    }
                )
    category_order = {name: index for index, name in enumerate(MODEL_CATEGORIES)}
    return sorted(
        rows,
        key=lambda item: (
            category_order[item["category"]],
            item["kind"],
            item["id"],
        ),
    )


def resolve_model(
    model_id: str,
    *,
    catalog: ModelCatalog | None = None,
    expected_category: str | None = None,
    expected_kind: str | None = None,
) -> ResolvedModel:
    """Resolve an exact ID and fail with same-kind alternatives."""

    if expected_kind is not None:
        kind_category = model_category_for_kind(expected_kind)
        if expected_category is None:
            expected_category = kind_category
        elif expected_category != kind_category:
            raise ModelCatalogError(
                f"expected kind {expected_kind!r} belongs to category "
                f"{kind_category!r}, not {expected_category!r}"
            )
    selected = str(model_id or "").strip()
    if not selected:
        raise ModelCatalogError("model ID must be non-empty")
    if catalog is not None and selected in catalog.models:
        definition = catalog.models[selected]
        source = "catalog"
        digest: str | None = catalog.sha256
    elif selected in BUILTIN_MODELS:
        definition = BUILTIN_MODELS[selected]
        source = "builtin"
        digest = None
    else:
        alternatives = [
            row["id"]
            for row in available_models(
                catalog,
                category=expected_category,
                kind=expected_kind,
            )
        ]
        raise ModelCatalogError(
            f"unknown model ID {selected!r}"
            + (
                f"; available {expected_kind or expected_category or 'all'} models: "
                + (", ".join(alternatives) or "<none>")
            )
        )
    actual_kind = str(definition["kind"])
    actual_category = str(definition["category"])
    if expected_category is not None and actual_category != expected_category:
        alternatives = [
            row["id"]
            for row in available_models(catalog, category=expected_category)
        ]
        raise ModelCatalogError(
            f"model {selected!r} has category {actual_category!r}, expected "
            f"{expected_category!r}; available {expected_category} models: "
            + (", ".join(alternatives) or "<none>")
        )
    if expected_kind is not None and actual_kind != expected_kind:
        alternatives = [
            row["id"]
            for row in available_models(
                catalog,
                category=expected_category,
                kind=expected_kind,
            )
        ]
        raise ModelCatalogError(
            f"model {selected!r} has kind {actual_kind!r}, expected "
            f"{expected_kind!r}; available {expected_kind} models: "
            + (", ".join(alternatives) or "<none>")
        )
    return ResolvedModel(
        model_id=selected,
        category=actual_category,
        kind=actual_kind,
        backend=str(definition["backend"]),
        definition=definition,
        source=source,
        catalog_sha256=digest,
    )


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _factory_parts(spec: str) -> tuple[str, str]:
    module_or_file, separator, attribute = str(spec or "").partition(":")
    if not separator or not module_or_file or not attribute:
        raise ModelIntegrationError(
            "factory",
            f"factory must use 'package.module:Class' or './file.py:Class', got {spec!r}",
        )
    if not _ATTRIBUTE_NAME.fullmatch(attribute):
        raise ModelIntegrationError(
            "factory",
            f"factory attribute must be one Python identifier: {attribute!r}",
        )
    return module_or_file, attribute


def _contained_file(spec_path: str, *, catalog: ModelCatalog) -> Path:
    if not spec_path.startswith("./"):
        raise ModelIntegrationError(
            "factory",
            "single-file factories must start with './' and stay inside the catalog directory",
        )
    candidate = (catalog.base_dir / spec_path).resolve()
    try:
        candidate.relative_to(catalog.base_dir)
    except ValueError as error:
        raise ModelIntegrationError(
            "factory",
            f"factory file escapes the catalog directory: {spec_path!r}",
        ) from error
    try:
        metadata = candidate.lstat()
    except FileNotFoundError as error:
        raise ModelIntegrationError(
            "source",
            f"factory file not found: {spec_path}",
        ) from error
    if stat.S_ISLNK(metadata.st_mode) or not stat.S_ISREG(metadata.st_mode):
        raise ModelIntegrationError(
            "source",
            f"factory file must be a regular non-symlink file: {spec_path}",
        )
    if candidate.suffix != ".py":
        raise ModelIntegrationError("factory", "single-file factories must use .py")
    return candidate


def _module_distribution(module_name: str) -> tuple[str, str]:
    top_level = module_name.split(".", 1)[0]
    distributions = importlib.metadata.packages_distributions().get(top_level, [])
    for distribution in sorted(distributions):
        try:
            return distribution, importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            continue
    return "", ""


def prepare_model_factory(
    model: ResolvedModel,
    *,
    catalog: ModelCatalog | None,
) -> PreparedFactory:
    """Import one explicit factory and record its implementation digest."""

    if model.backend != "factory":
        raise ModelIntegrationError(
            "factory", f"model {model.model_id!r} does not use backend='factory'"
        )
    if catalog is None or model.source != "catalog":
        raise ModelIntegrationError("factory", "factory models require a models config")
    original = str(model.definition["factory"])
    module_or_file, attribute = _factory_parts(original)
    if module_or_file.startswith(".") or module_or_file.endswith(".py"):
        source = _contained_file(module_or_file, catalog=catalog)
        source_sha256 = _sha256_file(source)
        namespace_digest = hashlib.sha256(
            (source.as_posix() + "\0" + source_sha256).encode("utf-8")
        ).hexdigest()[:24]
        module_name = f"dream_exe_user_model_{namespace_digest}"
        if module_name not in sys.modules:
            spec = importlib.util.spec_from_file_location(module_name, source)
            if spec is None or spec.loader is None:
                raise ModelIntegrationError(
                    "factory", f"cannot create a module spec for {module_or_file!r}"
                )
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            try:
                spec.loader.exec_module(module)
            except ModuleNotFoundError as error:
                sys.modules.pop(module_name, None)
                missing = str(error.name or "<unknown>")
                raise ModelIntegrationError(
                    "dependency",
                    f"factory {original!r} requires missing dependency {missing!r}",
                ) from error
            except Exception as error:
                sys.modules.pop(module_name, None)
                raise ModelIntegrationError(
                    "source", f"factory file {module_or_file!r} failed to import: {error}"
                ) from error
        module = sys.modules[module_name]
        implementation = {
            "factory": original,
            "source_kind": "single_file",
            "source_sha256": source_sha256,
            "source_size": source.stat().st_size,
        }
        effective = f"{module_name}:{attribute}"
    else:
        if not _MODULE_NAME.fullmatch(module_or_file):
            raise ModelIntegrationError(
                "factory", f"invalid Python module name: {module_or_file!r}"
            )
        try:
            module = importlib.import_module(module_or_file)
        except ModuleNotFoundError as error:
            missing = str(error.name or module_or_file)
            raise ModelIntegrationError(
                "dependency",
                f"factory {original!r} requires missing dependency {missing!r}",
            ) from error
        except Exception as error:
            raise ModelIntegrationError(
                "source", f"factory module {module_or_file!r} failed to import: {error}"
            ) from error
        distribution, version = _module_distribution(module_or_file)
        implementation = {
            "factory": original,
            "source_kind": "python_module",
            "module": module_or_file,
            "distribution": distribution,
            "distribution_version": version,
        }
        module_file = getattr(module, "__file__", None)
        if module_file:
            source = Path(module_file).resolve(strict=False)
            try:
                metadata = source.lstat()
            except FileNotFoundError:
                metadata = None
            if metadata is not None and stat.S_ISREG(metadata.st_mode):
                implementation["source_sha256"] = _sha256_file(source)
                implementation["source_size"] = metadata.st_size
        effective = original
    try:
        factory = getattr(module, attribute)
    except AttributeError as error:
        raise ModelIntegrationError(
            "factory", f"factory {original!r} has no attribute {attribute!r}"
        ) from error
    if not callable(factory):
        raise ModelIntegrationError("signature", f"factory {original!r} is not callable")
    return PreparedFactory(
        callable=factory,
        effective_spec=effective,
        original_spec=original,
        implementation=MappingProxyType(implementation),
    )


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    output = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(output.get(key), Mapping) and isinstance(value, Mapping):
            output[key] = _deep_merge(dict(output[key]), value)
        else:
            output[key] = copy.deepcopy(value)
    return output


def resolve_declared_paths(
    value: Any,
    *,
    base_dir: Path,
    replacements: dict[str, str] | None = None,
    field_name: str = "",
    namespace: str = "catalog",
) -> Any:
    """Resolve declared path-like fields without probing or acquiring assets."""

    if isinstance(value, Mapping):
        return {
            str(key): resolve_declared_paths(
                item,
                base_dir=base_dir,
                replacements=replacements,
                field_name=str(key),
                namespace=namespace,
            )
            for key, item in value.items()
        }
    if isinstance(value, list):
        child_field = field_name[:-1] if field_name.endswith("s") else field_name
        return [
            resolve_declared_paths(
                item,
                base_dir=base_dir,
                replacements=replacements,
                field_name=child_field,
                namespace=namespace,
            )
            for item in value
        ]
    if not (
        isinstance(value, str)
        and value
        and field_name.endswith(_PATH_SUFFIXES)
    ):
        return copy.deepcopy(value)
    if value.startswith(("http://", "https://")):
        return value
    path = Path(value).expanduser()
    if path.is_absolute():
        return path.resolve(strict=False).as_posix()
    resolved = (base_dir / path).resolve(strict=False).as_posix()
    if replacements is not None:
        replacements[resolved] = f"{namespace}:{Path(value).as_posix()}"
    return resolved


def effective_factory_kwargs(
    model: ResolvedModel,
    *,
    catalog: ModelCatalog,
    experiment_options: Mapping[str, Any] | None = None,
    experiment_base: Path | None = None,
    replacements: dict[str, str] | None = None,
) -> dict[str, Any]:
    """Apply catalog then experiment-level constructor options."""

    catalog_kwargs = resolve_declared_paths(
        model.definition.get("kwargs", {}),
        base_dir=catalog.base_dir,
        replacements=replacements,
        namespace="catalog",
    )
    runtime_kwargs = resolve_declared_paths(
        dict(experiment_options or {}),
        base_dir=experiment_base or catalog.base_dir,
        replacements=replacements,
        namespace="runtime",
    )
    return _deep_merge(catalog_kwargs, runtime_kwargs)


def _method_signature(instance: Any, kind: str) -> tuple[str, set[str]]:
    specifications = {
        "vlm": ("infer", {"prompt", "media_path", "generation_options"}),
        "video_generation": (
            "generate",
            {"image_path", "prompt", "output_path", "seed", "parameters"},
        ),
        "region_detector": ("detect", {"image_rgb", "prompt"}),
        "region_segmenter": (
            "segment_from_bbox",
            {"image_rgb", "bbox_xyxy", "prompt"},
        ),
        "depth": (
            "infer",
            {
                "video_frames",
                "target_fps",
                "fp32",
                "input_size",
                "intrinsics",
                "extrinsics",
            },
        ),
        "pose": ("infer", {"request"}),
    }
    if kind == "tracking":
        method_name = "predict" if callable(getattr(instance, "predict", None)) else "track"
        required = {
            "video_frames",
            "region_bbox_xyxy",
            "num_points",
            "seed",
            "query_points_xy",
            "segmentation_mask",
            "query_mode",
            "grid_size",
        }
    else:
        method_name, required = specifications[kind]
    method = getattr(instance, method_name, None)
    if not callable(method):
        raise ModelIntegrationError(
            "protocol", f"{kind} backend must expose callable {method_name}(...)"
        )
    try:
        signature = inspect.signature(method)
    except (TypeError, ValueError) as error:
        raise ModelIntegrationError(
            "signature", f"cannot inspect {kind} backend {method_name}(...)"
        ) from error
    has_variadic_keywords = any(
        parameter.kind == inspect.Parameter.VAR_KEYWORD
        for parameter in signature.parameters.values()
    )
    missing = sorted(required.difference(signature.parameters))
    if missing and not has_variadic_keywords:
        raise ModelIntegrationError(
            "signature",
            f"{kind} backend {method_name}(...) is missing parameters: "
            + ", ".join(missing),
        )
    return method_name, required


def _actual_identity(instance: Any, *, kind: str) -> dict[str, Any]:
    if kind == "region_detector":
        from ..video2traj.region.contract import region_detector_identity

        return region_detector_identity(instance, source="catalog region detector")
    if kind == "region_segmenter":
        from ..video2traj.region.contract import region_segmenter_identity

        return region_segmenter_identity(instance, source="catalog region segmenter")
    if kind == "tracking":
        from ..video2traj.tracking.core import tracking_backend_identity

        return tracking_backend_identity(instance, source="catalog tracking backend")
    if kind == "pose":
        from ..video2traj.pose.contract import pose_backend_identity

        return pose_backend_identity(instance, source="catalog pose backend")
    identity = {
        "provider_kind": str(getattr(instance, "provider_kind", "") or "").strip(),
        "backend_id": str(getattr(instance, "backend_id", "") or "").strip(),
        "contract_version": str(
            getattr(instance, "contract_version", "") or ""
        ).strip(),
    }
    if kind == "vlm":
        reader = getattr(instance, "inference_identity", None)
        if not callable(reader):
            raise ModelIntegrationError(
                "identity", "VLM backend must expose inference_identity()"
            )
        implementation_identity = reader()
        if not isinstance(implementation_identity, Mapping) or not implementation_identity:
            raise ModelIntegrationError(
                "identity", "VLM inference_identity() must return a non-empty mapping"
            )
        identity["implementation"] = copy.deepcopy(dict(implementation_identity))
    elif kind == "video_generation":
        backend_identity = getattr(instance, "identity", None)
        if not isinstance(backend_identity, Mapping) or not backend_identity:
            raise ModelIntegrationError(
                "identity", "video-generation identity must be a non-empty mapping"
            )
        identity["implementation"] = copy.deepcopy(dict(backend_identity))
        for field in ("provider_kind", "backend_id", "contract_version"):
            if not identity[field] and backend_identity.get(field) is not None:
                identity[field] = str(backend_identity[field])
    elif kind == "depth":
        pass
    try:
        encoded = json.dumps(identity, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ModelIntegrationError(
            "identity", f"{kind} backend identity must be strict JSON data"
        ) from error
    _validate_no_secrets(json.loads(encoded), pointer=f"{kind}.identity")
    return identity


def _validate_identity_match(
    actual: Mapping[str, Any],
    declared: Mapping[str, Any],
    *,
    kind: str,
) -> None:
    fields = ["provider_kind", "backend_id", "contract_version"]
    if kind in {"region_detector", "region_segmenter"}:
        fields.append("algorithm_id")
    mismatches = {
        field: {"declared": declared.get(field), "actual": actual.get(field)}
        for field in fields
        if str(declared.get(field)) != str(actual.get(field))
    }
    if mismatches:
        raise ModelIntegrationError(
            "identity", f"{kind} backend identity conflicts with catalog: {mismatches}"
        )


def instantiate_factory_model(
    model: ResolvedModel,
    *,
    catalog: ModelCatalog,
    experiment_options: Mapping[str, Any] | None = None,
    experiment_base: Path | None = None,
) -> tuple[Any, PreparedFactory, dict[str, Any]]:
    """Construct and validate one external factory before expensive runtime work."""

    prepared = prepare_model_factory(model, catalog=catalog)
    kwargs = effective_factory_kwargs(
        model,
        catalog=catalog,
        experiment_options=experiment_options,
        experiment_base=experiment_base,
    )
    try:
        instance = prepared.callable(**copy.deepcopy(kwargs))
    except ModuleNotFoundError as error:
        missing = str(error.name or "<unknown>")
        raise ModelIntegrationError(
            "dependency",
            f"model {model.model_id!r} requires missing dependency {missing!r}",
        ) from error
    except TypeError as error:
        raise ModelIntegrationError(
            "signature",
            f"model {model.model_id!r} factory rejected configured kwargs: {error}",
        ) from error
    except (FileNotFoundError, PermissionError) as error:
        raise ModelIntegrationError(
            "asset", f"model {model.model_id!r} asset preflight failed: {error}"
        ) from error
    except Exception as error:
        raise ModelIntegrationError(
            "construction", f"model {model.model_id!r} construction failed: {error}"
        ) from error
    if instance is None:
        raise ModelIntegrationError(
            "construction", f"model {model.model_id!r} factory returned None"
        )
    _method_signature(instance, model.kind)
    actual = _actual_identity(instance, kind=model.kind)
    _validate_identity_match(
        actual,
        model.definition["identity"],
        kind=model.kind,
    )
    return instance, prepared, actual


def _declared_asset_paths(value: Any, *, field_name: str = ""):
    if isinstance(value, Mapping):
        for key, item in value.items():
            yield from _declared_asset_paths(item, field_name=str(key))
    elif isinstance(value, list):
        child = field_name[:-1] if field_name.endswith("s") else field_name
        for item in value:
            yield from _declared_asset_paths(item, field_name=child)
    elif (
        isinstance(value, str)
        and value
        and field_name.endswith(_PATH_SUFFIXES)
        and not value.startswith(("http://", "https://"))
    ):
        yield field_name, value


def check_declared_assets(
    value: Mapping[str, Any],
    *,
    base_dir: Path,
) -> list[dict[str, Any]]:
    """Check only assets explicitly named by config; never acquire them."""

    records: list[dict[str, Any]] = []
    for field, raw_path in _declared_asset_paths(value):
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            path = base_dir / path
        path = path.resolve(strict=False)
        if not path.exists():
            raise ModelIntegrationError(
                "asset", f"declared {field} does not exist: {raw_path}"
            )
        records.append(
            {
                "field": field,
                "path_kind": "directory" if path.is_dir() else "file",
            }
        )
    return records


def instantiate_vlm_model(
    model: ResolvedModel,
    *,
    catalog: ModelCatalog,
    experiment_options: Mapping[str, Any] | None = None,
    require_credentials: bool = True,
) -> tuple[Any, dict[str, Any]]:
    """Instantiate one factory or OpenAI-compatible VLM catalog model."""

    if model.category != "eval" or model.kind != "vlm":
        raise ModelCatalogError(f"model {model.model_id!r} is not a VLM")
    if model.backend == "factory":
        instance, prepared, identity = instantiate_factory_model(
            model,
            catalog=catalog,
            experiment_options=experiment_options,
        )
        return instance, {
            "model_id": model.model_id,
            "backend": "factory",
            "implementation": dict(prepared.implementation),
            "identity": identity,
        }
    options = _deep_merge(model.definition.get("options", {}), experiment_options or {})
    allowed = {
        "api_key_env",
        "base_url",
        "max_tokens",
        "media_type",
        "model",
        "seed",
        "token_limit_parameter",
    }
    unsupported = sorted(set(options).difference(allowed))
    if unsupported:
        raise ModelCatalogError(
            f"OpenAI-compatible model {model.model_id!r} has unsupported options: "
            + ", ".join(unsupported)
        )
    upstream_model = str(options.get("model", "") or "").strip()
    base_url = str(options.get("base_url", "") or "").strip()
    api_key_env = str(options.get("api_key_env", "") or "").strip()
    if not upstream_model or not base_url or not api_key_env:
        raise ModelCatalogError(
            f"OpenAI-compatible model {model.model_id!r} requires "
            "options.model, options.base_url, and options.api_key_env"
        )
    if not _SAFE_ENV_NAME.fullmatch(api_key_env):
        raise ModelCatalogError("options.api_key_env must be an environment-variable name")
    api_key = str(os.environ.get(api_key_env, "") or "").strip()
    if require_credentials and not api_key:
        raise ModelIntegrationError(
            "credential", f"environment variable {api_key_env} is not set"
        )
    from ..evaluation.vlm.providers.openai_compatible import (
        CURRENT_MAX_TOKENS,
        CURRENT_MEDIA_TYPE,
        CURRENT_SEED,
        OpenAICompatibleVLMInference,
    )

    adapter = OpenAICompatibleVLMInference(
        api_key=api_key or "dream-exe-static-check",
        model=upstream_model,
        base_url=base_url,
        max_tokens=int(options.get("max_tokens", CURRENT_MAX_TOKENS)),
        seed=options.get("seed", CURRENT_SEED),
        media_type=str(options.get("media_type", CURRENT_MEDIA_TYPE)),
        token_limit_parameter=str(
            options.get("token_limit_parameter", "max_completion_tokens")
        ),
    )
    return adapter, {
        "model_id": model.model_id,
        "backend": model.backend,
        "credential_env": api_key_env,
        "credential_configured": bool(api_key),
        "identity": adapter.inference_identity(),
    }


def instantiate_video_generation_model(
    model: ResolvedModel,
    *,
    catalog: ModelCatalog,
    experiment_options: Mapping[str, Any] | None = None,
) -> tuple[Any, dict[str, Any], dict[str, Any]]:
    """Instantiate one factory or current Wan2.2 generation backend."""

    if model.category != "video_gen" or model.kind != "video_generation":
        raise ModelCatalogError(f"model {model.model_id!r} is not video generation")
    if model.backend == "factory":
        instance, prepared, identity = instantiate_factory_model(
            model,
            catalog=catalog,
            experiment_options=experiment_options,
        )
        details = {
            "model_id": model.model_id,
            "backend": "factory",
            "catalog_sha256": catalog.sha256,
            "implementation": dict(prepared.implementation),
            "identity": identity,
        }
        parameters = model.definition.get("options", {})
        return (
            _CatalogVideoBackend(instance, details),
            details,
            copy.deepcopy(dict(parameters)),
        )
    options = _deep_merge(model.definition.get("options", {}), experiment_options or {})
    allowed = {"checkpoint_root", "parameters", "python_executable", "source_root"}
    unsupported = sorted(set(options).difference(allowed))
    if unsupported:
        raise ModelCatalogError(
            f"Wan2.2 model {model.model_id!r} has unsupported options: "
            + ", ".join(unsupported)
        )
    constructor = resolve_declared_paths(
        {key: value for key, value in options.items() if key != "parameters"},
        base_dir=catalog.base_dir,
    )
    if not constructor.get("source_root") or not constructor.get("checkpoint_root"):
        raise ModelCatalogError(
            f"Wan2.2 model {model.model_id!r} requires source_root and checkpoint_root"
        )
    check_declared_assets(constructor, base_dir=catalog.base_dir)
    from ..generation.providers.wan22 import Wan22TI2VBackend

    backend = Wan22TI2VBackend(**constructor)
    parameters = options.get("parameters", {})
    if not isinstance(parameters, Mapping):
        raise ModelCatalogError("Wan2.2 options.parameters must be an object")
    details = {
        "model_id": model.model_id,
        "backend": model.backend,
        "catalog_sha256": catalog.sha256,
        "identity": dict(backend.identity),
    }
    return (
        _CatalogVideoBackend(backend, details),
        details,
        copy.deepcopy(dict(parameters)),
    )


class _CatalogVideoBackend:
    """Attach catalog implementation evidence without changing model I/O."""

    def __init__(self, backend: Any, details: Mapping[str, Any]) -> None:
        self._backend = backend
        self._details = copy.deepcopy(dict(details))

    @property
    def identity(self) -> Mapping[str, Any]:
        backend_identity = getattr(self._backend, "identity", None)
        if not isinstance(backend_identity, Mapping):
            raise TypeError("video-generation backend identity must be a mapping")
        return {
            "format": "dream-exe.catalog-video-backend",
            "catalog_sha256": self._details["catalog_sha256"],
            "catalog_model_id": self._details["model_id"],
            "backend": self._details["backend"],
            "implementation": copy.deepcopy(
                self._details.get("implementation", {})
            ),
            "backend_identity": copy.deepcopy(dict(backend_identity)),
        }

    def generate(self, **kwargs: Any) -> Mapping[str, Any]:
        return self._backend.generate(**kwargs)


__all__ = [
    "BUILTIN_MODELS",
    "MAX_MODEL_CATALOG_BYTES",
    "MODEL_CATALOG_FORMAT",
    "MODEL_CATEGORIES",
    "MODEL_CATEGORY_BY_KIND",
    "MODEL_KINDS",
    "MODEL_KINDS_BY_CATEGORY",
    "ModelCatalog",
    "ModelCatalogError",
    "ModelIntegrationError",
    "PreparedFactory",
    "ResolvedModel",
    "available_models",
    "check_declared_assets",
    "contract_version_for_kind",
    "effective_factory_kwargs",
    "instantiate_factory_model",
    "instantiate_video_generation_model",
    "instantiate_vlm_model",
    "load_model_catalog",
    "load_strict_json_object",
    "model_category_for_kind",
    "prepare_model_factory",
    "resolve_declared_paths",
    "resolve_model",
]
