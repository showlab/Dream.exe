"""Explicit, lazy runtime assembly for standalone video-to-trajectory runs.

The algorithmic callables in :mod:`dream_exe.video2traj` accept injected
region, tracking, depth, and pose dependencies.  This module turns one
caller-owned configuration object into those dependencies without discovering
benchmark samples, UIDs, repository roots, model roots, or output paths.

Only light-weight Dream.exe adapters are constructed here.  Torch and the
third-party model packages remain lazy and are imported by the adapters on
first inference.  Alternate implementations can be supplied either as
explicitly injected objects or as an explicit ``module:attribute`` factory.
"""

from __future__ import annotations

import copy
import importlib
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ...model_assets.dvd_identity import (
    DVD_ASSET_ATTESTATION_FIELD,
    dvd_provenance_validation,
)
from ..tracking.backends.cotracker import CoTrackerBackend
from ..tracking.core import (
    COTRACKER_BACKEND_ID,
    TRACKING_BACKEND_CONTRACT_VERSION,
    tracking_backend_identity,
    validate_external_tracking_runtime_identity,
)
from ..depth.estimator import (
    SUPPORTED_DEPTH_BACKENDS,
    preflight_depth_estimator_request,
    prepare_depth_estimator,
    resolve_depth_estimator_preset,
    validate_depth_estimator_assets,
    validate_depth_runtime_identity,
)
from ..depth.contract import (
    external_depth_backend_id,
    external_depth_selection,
    validate_external_depth_runtime_identity,
)
from ..depth.backends.dvd import build_dvd_backend_registry
from ..pose.backends.foundationpose import FoundationPoseBackend
from ..pose.backends.freepose import FreePoseBackend
from ..pose.backends.tracked_pointcloud_rigid import (
    TrackedPointCloudRigidPoseBackend,
)
from ..pose.config import (
    POINTCLOUD_KABSCH_BACKEND,
)
from ..pose.contract import (
    normalize_rigid_pose_backend_identity,
    pose_backend_identity,
    rigid_pose_backend_identity,
    validate_external_pose_runtime_identity,
)
from ..region.backends import (
    GroundingDinoDetectorAdapter,
    SAM2SegmenterAdapter,
    builtin_region_query_samplers,
)
from ..region.contract import (
    normalize_region_detector_identity,
    normalize_region_query_sampler_identity,
    normalize_region_segmenter_identity,
    region_detector_identity,
    region_query_sampler_identity,
    region_runtime_identity,
    region_segmenter_identity,
    validate_external_region_runtime_identity,
)
from ..region.runtime import RegionRuntime
from ..pose.backends.sinref6d import SinRef6DBackend
from ..depth.backends.vda import (
    VDA_BACKEND_REGISTRY,
    VDA_DEFAULT_PROVIDER_MODULE,
)
from ..depth.calibration import run_depth_calibration_runtime


_PATH_FIELD_SUFFIXES = (
    "_cache",
    "_checkpoint",
    "_dir",
    "_file",
    "_path",
    "_root",
)

_DEPTH_COMPONENT_ALLOWED_FIELDS = frozenset(
    {
        "backend",
        "backend_registry",
        "attestation_manifest_path",
        "checkpoints_root",
        "config",
        "config_path",
        "factory",
        "fp32",
        "input_size",
        "kwargs",
        "preset",
        "provider_module",
        "runtime_config",
        "runtime_context",
        "seed",
        "selection_source",
        "source_root",
        "validate_assets",
        "weights_root",
    }
)
_EXTERNAL_DEPTH_RUNTIME_ALLOWED_FIELDS = frozenset(
    {
        "backend_id",
        "contract_version",
        "model_name",
        "provider_kind",
    }
)
_REGION_IDENTITY_FIELDS = frozenset(
    {
        "provider_kind",
        "backend_id",
        "algorithm_id",
        "contract_version",
    }
)
_COMPOSED_REGION_ALLOWED_FIELDS = frozenset(
    {
        "backend",
        "runtime_config",
        "detector",
        "segmenter",
        "query_samplers",
    }
)
_REGION_NONE_ROLE_ALLOWED_FIELDS = frozenset({"backend"})
_REGION_DETECTOR_BUILTIN_ALLOWED_FIELDS = frozenset(
    {
        "backend",
        "source_root",
        "config_path",
        "checkpoint_path",
        "text_encoder_path",
        "box_threshold",
        "text_threshold",
        *_REGION_IDENTITY_FIELDS,
    }
)
_REGION_SEGMENTER_BUILTIN_ALLOWED_FIELDS = frozenset(
    {
        "backend",
        "source_root",
        "config_name",
        "checkpoint_path",
        *_REGION_IDENTITY_FIELDS,
    }
)
_REGION_INJECTED_ROLE_ALLOWED_FIELDS = frozenset(
    {
        "backend",
        *_REGION_IDENTITY_FIELDS,
    }
)
_REGION_FACTORY_ROLE_ALLOWED_FIELDS = frozenset(
    {
        "backend",
        "factory",
        "kwargs",
        *_REGION_IDENTITY_FIELDS,
    }
)
_REGION_QUERY_SAMPLER_ALLOWED_FIELDS = frozenset(
    {
        "backend",
        *_REGION_IDENTITY_FIELDS,
    }
)


def load_video2traj_runtime_config(
    path: str | Path,
) -> dict[str, Any]:
    """Load one explicit JSON runtime config and record its explicit base.

    Relative asset paths in a file-backed config are resolved against the
    config file's directory.  A mapping passed directly to
    :func:`build_video2traj_runtime` must instead use absolute asset paths or
    provide ``asset_base`` explicitly.
    """

    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(
            f"video2traj runtime config not found: {source.as_posix()}"
        )
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"video2traj runtime config is not valid JSON: {source.as_posix()}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(
            f"video2traj runtime config must be a JSON object: {source.as_posix()}"
        )
    config = copy.deepcopy(payload)
    existing_meta = config.get("_meta", {})
    if existing_meta is None:
        existing_meta = {}
    if not isinstance(existing_meta, Mapping):
        raise TypeError("video2traj runtime config _meta must be a mapping")
    config["_meta"] = {
        **dict(existing_meta),
        "source": source.as_posix(),
        "base_dir": source.parent.as_posix(),
    }
    return config


def _mapping(
    value: Any,
    *,
    label: str,
    required: bool = False,
) -> dict[str, Any]:
    if value is None and not required:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return copy.deepcopy(dict(value))


def _backend_name(
    config: Mapping[str, Any],
    *,
    label: str,
) -> str:
    value = str(config.get("backend", "") or "").strip().lower()
    if not value:
        raise ValueError(f"{label}.backend must be explicitly provided")
    return value


def _explicit_base(
    config: Mapping[str, Any],
    *,
    asset_base: str | Path | None,
) -> Path | None:
    if asset_base is not None and str(asset_base).strip():
        explicit_base = Path(asset_base).expanduser()
        if not explicit_base.is_absolute():
            raise ValueError("asset_base must be an absolute path")
        base = explicit_base.resolve(strict=False)
    else:
        meta = config.get("_meta", {})
        meta_payload = dict(meta) if isinstance(meta, Mapping) else {}
        base_text = str(meta_payload.get("base_dir", "") or "").strip()
        base = Path(base_text).expanduser().resolve(strict=False) if base_text else None

    asset_root = str(config.get("asset_root", "") or "").strip()
    if not asset_root:
        return base
    root = Path(asset_root).expanduser()
    if not root.is_absolute():
        if base is None:
            raise ValueError(
                "relative runtime asset_root requires an explicit asset_base "
                "or a file-backed runtime config"
            )
        root = base / root
    return root.resolve(strict=False)


def _path(
    value: Any,
    *,
    label: str,
    base: Path | None,
    required: bool = True,
) -> str:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError(f"{label} must be explicitly provided")
        return ""
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        if base is None:
            raise ValueError(
                f"relative {label} requires an explicit asset_base "
                "or a file-backed runtime config"
            )
        candidate = base / candidate
    return candidate.resolve(strict=False).as_posix()


def _paths(
    value: Any,
    *,
    label: str,
    base: Path | None,
    required: bool = False,
) -> tuple[str, ...]:
    if value is None:
        values: Sequence[Any] = ()
    elif isinstance(value, (str, Path)):
        values = (value,)
    elif isinstance(value, Sequence):
        values = value
    else:
        raise TypeError(f"{label} must be a path or a sequence of paths")
    if required and not values:
        raise ValueError(f"{label} must be explicitly provided")
    return tuple(
        _path(
            item,
            label=f"{label}[{index}]",
            base=base,
        )
        for index, item in enumerate(values)
    )


def _normalize_factory_kwargs(
    value: Any,
    *,
    base: Path | None,
    field_name: str = "",
) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _normalize_factory_kwargs(
                child,
                base=base,
                field_name=str(key),
            )
            for key, child in value.items()
        }
    if isinstance(value, list):
        if field_name.endswith("roots"):
            return [
                _path(
                    child,
                    label=field_name,
                    base=base,
                )
                for child in value
            ]
        return [
            _normalize_factory_kwargs(
                child,
                base=base,
                field_name=field_name,
            )
            for child in value
        ]
    if (
        isinstance(value, (str, Path))
        and value
        and field_name.endswith(_PATH_FIELD_SUFFIXES)
    ):
        return _path(
            value,
            label=field_name,
            base=base,
        )
    return copy.deepcopy(value)


def _resolve_import_spec(
    import_spec: str,
) -> Any:
    module_name, separator, attribute_name = str(import_spec or "").partition(":")
    if not separator or not module_name or not attribute_name:
        raise ValueError(
            f"component factory must use 'module:attribute', got {import_spec!r}"
        )
    try:
        module = importlib.import_module(module_name)
    except ModuleNotFoundError as error:
        missing = str(error.name or module_name)
        raise RuntimeError(
            f"component factory {import_spec!r} is unavailable because "
            f"dependency {missing!r} is missing"
        ) from error
    try:
        return getattr(module, attribute_name)
    except AttributeError as error:
        raise RuntimeError(
            f"component factory {import_spec!r} has no attribute {attribute_name!r}"
        ) from error


class _LazyFactoryComponent:
    """Internal proxy that imports and constructs an explicit factory once."""

    def __init__(
        self,
        *,
        import_spec: str,
        kwargs: Mapping[str, Any],
        role: str,
        expected_identity: Mapping[str, str] | None = None,
        identity_reader: Any = None,
    ) -> None:
        self._import_spec = str(import_spec)
        self._kwargs = copy.deepcopy(dict(kwargs))
        self._role = str(role)
        self._component: Any = None
        self._expected_identity = (
            None if expected_identity is None else dict(expected_identity)
        )
        self._identity_reader = identity_reader
        if self._expected_identity is not None and not callable(self._identity_reader):
            raise TypeError(f"{self._role} expected_identity requires identity_reader")
        if self._expected_identity is not None:
            for key, value in self._expected_identity.items():
                setattr(self, key, value)

    def _load(self) -> Any:
        if self._component is not None:
            return self._component
        factory = _resolve_import_spec(self._import_spec)
        if callable(factory):
            try:
                component = factory(**copy.deepcopy(self._kwargs))
            except ModuleNotFoundError as error:
                missing = str(error.name or "<unknown>")
                raise RuntimeError(
                    f"{self._role} factory {self._import_spec!r} "
                    f"is missing dependency {missing!r}"
                ) from error
        else:
            if self._kwargs:
                raise TypeError(
                    f"{self._role} import {self._import_spec!r} is not "
                    "callable but factory kwargs were supplied"
                )
            component = factory
        if component is None:
            raise RuntimeError(
                f"{self._role} factory {self._import_spec!r} returned None"
            )
        if self._expected_identity is not None:
            actual_identity = self._identity_reader(
                component,
                source=(f"{self._role} factory {self._import_spec!r} runtime object"),
            )
            if actual_identity != self._expected_identity:
                raise ValueError(
                    f"{self._role} factory {self._import_spec!r} runtime "
                    "identity conflicts with configured identity: "
                    f"runtime={actual_identity!r}, "
                    f"configured={self._expected_identity!r}"
                )
        self._component = component
        return component

    def __getattr__(self, name: str) -> Any:
        return getattr(self._load(), name)

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        component = self._load()
        if not callable(component):
            raise TypeError(f"{self._role} component is not callable")
        return component(*args, **kwargs)


def _factory_component(
    config: Mapping[str, Any],
    *,
    role: str,
    base: Path | None,
    expected_identity: Mapping[str, str] | None = None,
    identity_reader: Any = None,
) -> _LazyFactoryComponent:
    import_spec = str(config.get("factory", "") or "").strip()
    if not import_spec:
        raise ValueError(f"{role}.factory must be explicitly provided")
    kwargs = _mapping(
        config.get("kwargs", {}),
        label=f"{role}.kwargs",
    )
    return _LazyFactoryComponent(
        import_spec=import_spec,
        kwargs=_normalize_factory_kwargs(
            kwargs,
            base=base,
        ),
        role=role,
        expected_identity=expected_identity,
        identity_reader=identity_reader,
    )


def _injected(
    dependencies: Mapping[str, Any],
    key: str,
) -> Any:
    if key not in dependencies or dependencies[key] is None:
        raise ValueError(f"runtime backend='injected' requires dependencies[{key!r}]")
    return dependencies[key]


def _reject_unsupported_fields(
    config: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    label: str,
) -> None:
    unsupported = sorted(set(config).difference(allowed))
    if unsupported:
        raise ValueError(
            f"{label} contains unsupported fields: {', '.join(unsupported)}"
        )


def _configured_region_role_identity(
    config: Mapping[str, Any],
    *,
    label: str,
    normalizer: Any,
    expected: Mapping[str, str] | None = None,
    external: bool,
) -> dict[str, str]:
    present = {
        field for field in _REGION_IDENTITY_FIELDS if config.get(field) is not None
    }
    if expected is not None and not present:
        return dict(expected)
    if present != _REGION_IDENTITY_FIELDS:
        missing = sorted(_REGION_IDENTITY_FIELDS.difference(present))
        raise ValueError(
            f"{label} identity must declare all fields; missing {missing!r}"
        )
    identity = normalizer(
        provider_kind=config.get("provider_kind"),
        backend_id=config.get("backend_id"),
        algorithm_id=config.get("algorithm_id"),
        contract_version=config.get("contract_version"),
        source=label,
    )
    if external and identity["provider_kind"] != "external":
        raise ValueError(f"{label}.provider_kind must be 'external'")
    if expected is not None and identity != dict(expected):
        raise ValueError(
            f"{label} identity conflicts with the selected built-in "
            f"provider: configured={identity!r}, "
            f"builtin={dict(expected)!r}"
        )
    return identity


def _grounding_dino_binding(
    config: Mapping[str, Any],
    *,
    device: str,
    base: Path | None,
    dependencies: Mapping[str, Any],
    label: str,
    strict_fields: bool = True,
    validate_configured_identity: bool = True,
) -> tuple[
    Any,
    dict[str, str],
    dict[str, Any],
]:
    if strict_fields:
        _reject_unsupported_fields(
            config,
            allowed=_REGION_DETECTOR_BUILTIN_ALLOWED_FIELDS,
            label=label,
        )
    expected_identity = normalize_region_detector_identity(
        provider_kind=GroundingDinoDetectorAdapter.provider_kind,
        backend_id=GroundingDinoDetectorAdapter.backend_id,
        algorithm_id=GroundingDinoDetectorAdapter.algorithm_id,
        contract_version=(GroundingDinoDetectorAdapter.contract_version),
        source="built-in GroundingDINO detector",
    )
    identity = (
        _configured_region_role_identity(
            config,
            label=label,
            normalizer=normalize_region_detector_identity,
            expected=expected_identity,
            external=False,
        )
        if validate_configured_identity
        else expected_identity
    )
    arguments = {
        "source_root": _path(
            config.get("source_root"),
            label=f"{label}.source_root",
            base=base,
            required=False,
        ),
        "config_path": _path(
            config.get("config_path"),
            label=f"{label}.config_path",
            base=base,
        ),
        "checkpoint_path": _path(
            config.get("checkpoint_path"),
            label=f"{label}.checkpoint_path",
            base=base,
        ),
        "text_encoder_path": _path(
            config.get("text_encoder_path"),
            label=f"{label}.text_encoder_path",
            base=base,
        ),
        "device": device,
        "box_threshold": float(config.get("box_threshold", 0.4)),
        "text_threshold": float(config.get("text_threshold", 0.3)),
        "runtime_loader": dependencies.get(
            "grounding_dino_runtime_loader",
            None,
        ),
    }

    def factory() -> GroundingDinoDetectorAdapter:
        return GroundingDinoDetectorAdapter(**arguments)

    public_arguments = {
        key: value for key, value in arguments.items() if key != "runtime_loader"
    }
    return factory, identity, public_arguments


def _sam2_binding(
    config: Mapping[str, Any],
    *,
    device: str,
    base: Path | None,
    dependencies: Mapping[str, Any],
    label: str,
    strict_fields: bool = True,
    validate_configured_identity: bool = True,
) -> tuple[
    Any,
    dict[str, str],
    dict[str, Any],
]:
    if strict_fields:
        _reject_unsupported_fields(
            config,
            allowed=_REGION_SEGMENTER_BUILTIN_ALLOWED_FIELDS,
            label=label,
        )
    expected_identity = normalize_region_segmenter_identity(
        provider_kind=SAM2SegmenterAdapter.provider_kind,
        backend_id=SAM2SegmenterAdapter.backend_id,
        algorithm_id=SAM2SegmenterAdapter.algorithm_id,
        contract_version=SAM2SegmenterAdapter.contract_version,
        source="built-in SAM2 segmenter",
    )
    identity = (
        _configured_region_role_identity(
            config,
            label=label,
            normalizer=normalize_region_segmenter_identity,
            expected=expected_identity,
            external=False,
        )
        if validate_configured_identity
        else expected_identity
    )
    config_name = str(config.get("config_name", "") or "").strip()
    if not config_name:
        raise ValueError(f"{label}.config_name must be explicitly provided")
    arguments = {
        "source_root": _path(
            config.get("source_root"),
            label=f"{label}.source_root",
            base=base,
            required=False,
        ),
        "config_name": config_name,
        "checkpoint_path": _path(
            config.get("checkpoint_path"),
            label=f"{label}.checkpoint_path",
            base=base,
        ),
        "device": device,
        "runtime_loader": dependencies.get(
            "sam2_runtime_loader",
            None,
        ),
    }

    def factory() -> SAM2SegmenterAdapter:
        return SAM2SegmenterAdapter(**arguments)

    public_arguments = {
        key: value for key, value in arguments.items() if key != "runtime_loader"
    }
    return factory, identity, public_arguments


def _external_region_role_identity(
    config: Mapping[str, Any],
    *,
    role: str,
) -> dict[str, str]:
    if role == "detector":
        normalizer = normalize_region_detector_identity
    elif role == "segmenter":
        normalizer = normalize_region_segmenter_identity
    else:  # pragma: no cover - internal invariant
        raise ValueError(f"unknown region role: {role!r}")
    return _configured_region_role_identity(
        config,
        label=f"region.{role}",
        normalizer=normalizer,
        external=True,
    )


def _build_composed_region_role(
    config: Mapping[str, Any],
    *,
    role: str,
    device: str,
    base: Path | None,
    dependencies: Mapping[str, Any],
) -> tuple[Any | None, Any | None, dict[str, str] | None, dict[str, Any]]:
    role_config = _mapping(
        config.get(role),
        label=f"region.{role}",
        required=True,
    )
    backend = _backend_name(
        role_config,
        label=f"region.{role}",
    )
    if backend == "none":
        _reject_unsupported_fields(
            role_config,
            allowed=_REGION_NONE_ROLE_ALLOWED_FIELDS,
            label=f"region.{role} backend 'none'",
        )
        return None, None, None, {"backend": "none"}

    builtin_backend = "grounding_dino" if role == "detector" else "sam2"
    if backend == builtin_backend:
        if role == "detector":
            factory, identity, assets = _grounding_dino_binding(
                role_config,
                device=device,
                base=base,
                dependencies=dependencies,
                label="region.detector",
            )
        else:
            factory, identity, assets = _sam2_binding(
                role_config,
                device=device,
                base=base,
                dependencies=dependencies,
                label="region.segmenter",
            )
        return (
            None,
            factory,
            identity,
            {
                "backend": backend,
                **identity,
                **assets,
                "lazy": True,
            },
        )

    if backend == "injected":
        _reject_unsupported_fields(
            role_config,
            allowed=_REGION_INJECTED_ROLE_ALLOWED_FIELDS,
            label=f"region.{role} backend 'injected'",
        )
        identity = _external_region_role_identity(
            role_config,
            role=role,
        )
        dependency_key = f"region_{role}"
        component = _injected(dependencies, dependency_key)
        identity_reader = (
            region_detector_identity
            if role == "detector"
            else region_segmenter_identity
        )
        actual_identity = identity_reader(
            component,
            source=f"dependencies[{dependency_key!r}]",
        )
        if actual_identity != identity:
            raise ValueError(
                f"dependencies[{dependency_key!r}] identity conflicts "
                f"with configured region.{role} identity: "
                f"runtime={actual_identity!r}, "
                f"configured={identity!r}"
            )
        return (
            component,
            None,
            identity,
            {
                "backend": "injected",
                "dependency": dependency_key,
                **identity,
            },
        )

    if backend == "factory":
        _reject_unsupported_fields(
            role_config,
            allowed=_REGION_FACTORY_ROLE_ALLOWED_FIELDS,
            label=f"region.{role} backend 'factory'",
        )
        identity = _external_region_role_identity(
            role_config,
            role=role,
        )
        identity_reader = (
            region_detector_identity
            if role == "detector"
            else region_segmenter_identity
        )
        import_spec = str(role_config.get("factory", "") or "").strip()
        if not import_spec:
            raise ValueError(f"region.{role}.factory must be explicitly provided")
        factory_kwargs = _mapping(
            role_config.get("kwargs", {}),
            label=f"region.{role}.kwargs",
        )
        normalized_factory_kwargs = _normalize_factory_kwargs(
            factory_kwargs,
            base=base,
        )

        def factory() -> _LazyFactoryComponent:
            return _LazyFactoryComponent(
                import_spec=import_spec,
                kwargs=normalized_factory_kwargs,
                role=f"region.{role}",
                expected_identity=identity,
                identity_reader=identity_reader,
            )

        return (
            None,
            factory,
            identity,
            {
                "backend": "factory",
                "factory": import_spec,
                **identity,
                "lazy": True,
            },
        )
    raise ValueError(f"unsupported region.{role} backend: {backend!r}")


def _composed_region_sampling_backends(
    config: Mapping[str, Any],
    *,
    dependencies: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    query_config = _mapping(
        config.get("query_samplers", {}),
        label="region.query_samplers",
    )
    dependency_value = dependencies.get(
        "region_sampling_backends",
        {},
    )
    if not isinstance(dependency_value, Mapping):
        raise TypeError("dependencies['region_sampling_backends'] must be a mapping")
    dependency_backends = dict(dependency_value)
    builtin_backends = builtin_region_query_samplers()
    noncanonical_methods = [
        method
        for method in query_config
        if (not isinstance(method, str) or method != method.strip().lower())
    ]
    if noncanonical_methods:
        raise ValueError(
            "region.query_samplers method names must be canonical lowercase strings"
        )
    noncanonical_dependency_methods = [
        method
        for method in dependency_backends
        if (not isinstance(method, str) or method != method.strip().lower())
    ]
    if noncanonical_dependency_methods:
        raise ValueError(
            "dependencies['region_sampling_backends'] method names must "
            "be canonical lowercase strings"
        )
    unsupported_methods = sorted(set(query_config).difference(builtin_backends))
    if unsupported_methods:
        raise ValueError(
            "region.query_samplers contains unsupported methods: "
            f"{', '.join(unsupported_methods)}"
        )

    selected: dict[str, Any] = {}
    bindings: dict[str, dict[str, Any]] = {}
    injected_methods: set[str] = set()
    for method, raw_binding in query_config.items():
        normalized_method = str(method).strip().lower()
        binding = _mapping(
            raw_binding,
            label=(f"region.query_samplers[{normalized_method!r}]"),
            required=True,
        )
        _reject_unsupported_fields(
            binding,
            allowed=_REGION_QUERY_SAMPLER_ALLOWED_FIELDS,
            label=(f"region.query_samplers[{normalized_method!r}]"),
        )
        backend = _backend_name(
            binding,
            label=(f"region.query_samplers[{normalized_method!r}]"),
        )
        builtin_identity = region_query_sampler_identity(
            builtin_backends[normalized_method],
            source=(f"built-in region sampler {normalized_method!r}"),
        )
        if backend == "builtin":
            identity = _configured_region_role_identity(
                binding,
                label=(f"region.query_samplers[{normalized_method!r}]"),
                normalizer=normalize_region_query_sampler_identity,
                expected=builtin_identity,
                external=False,
            )
            bindings[normalized_method] = {
                "backend": "builtin",
                **identity,
            }
            continue
        if backend != "injected":
            raise ValueError(
                "region query sampler backend must be 'builtin' or "
                f"'injected', got {backend!r} for "
                f"{normalized_method!r}"
            )
        configured_identity = _configured_region_role_identity(
            binding,
            label=(f"region.query_samplers[{normalized_method!r}]"),
            normalizer=normalize_region_query_sampler_identity,
            external=True,
        )
        if normalized_method not in dependency_backends:
            raise ValueError(
                "runtime backend='injected' requires "
                "dependencies['region_sampling_backends']"
                f"[{normalized_method!r}]"
            )
        sampler = dependency_backends[normalized_method]
        actual_identity = region_query_sampler_identity(
            sampler,
            source=(f"dependencies['region_sampling_backends'][{normalized_method!r}]"),
        )
        if actual_identity != configured_identity:
            raise ValueError(
                "dependencies['region_sampling_backends']"
                f"[{normalized_method!r}] identity conflicts with "
                "configured query sampler identity: "
                f"runtime={actual_identity!r}, "
                f"configured={configured_identity!r}"
            )
        selected[normalized_method] = sampler
        injected_methods.add(normalized_method)
        bindings[normalized_method] = {
            "backend": "injected",
            "dependency": (f"region_sampling_backends[{normalized_method!r}]"),
            **configured_identity,
        }

    unbound_dependencies = sorted(set(dependency_backends).difference(injected_methods))
    if unbound_dependencies:
        raise ValueError(
            "dependencies['region_sampling_backends'] contains "
            "unbound composed-region methods: "
            f"{', '.join(unbound_dependencies)}"
        )
    return selected, bindings


def _build_composed_region(
    config: Mapping[str, Any],
    *,
    device: str,
    base: Path | None,
    dependencies: Mapping[str, Any],
) -> tuple[RegionRuntime, dict[str, Any]]:
    _reject_unsupported_fields(
        config,
        allowed=_COMPOSED_REGION_ALLOWED_FIELDS,
        label="region backend 'composed'",
    )
    runtime_config = _mapping(
        config.get("runtime_config", {}),
        label="region.runtime_config",
    )
    (
        detector,
        detector_factory,
        detector_identity,
        detector_manifest,
    ) = _build_composed_region_role(
        config,
        role="detector",
        device=device,
        base=base,
        dependencies=dependencies,
    )
    (
        segmenter,
        segmenter_factory,
        segmenter_identity,
        segmenter_manifest,
    ) = _build_composed_region_role(
        config,
        role="segmenter",
        device=device,
        base=base,
        dependencies=dependencies,
    )
    sampling_backends, sampler_bindings = _composed_region_sampling_backends(
        config,
        dependencies=dependencies,
    )
    region = RegionRuntime(
        runtime_config=runtime_config,
        device=device,
        detector=detector,
        segmenter=segmenter,
        detector_factory=detector_factory,
        segmenter_factory=segmenter_factory,
        sampling_backends=sampling_backends,
        detector_provider_identity=detector_identity,
        segmenter_provider_identity=segmenter_identity,
    )
    return (
        region,
        {
            "backend": "composed",
            "lazy": True,
            "providers": region.provider_identities(),
            "query_samplers": region.query_sampler_identities(),
            "roles": {
                "detector": detector_manifest,
                "segmenter": segmenter_manifest,
                "query_samplers": sampler_bindings,
            },
        },
    )


def _build_region(
    config: Mapping[str, Any],
    *,
    device: str,
    base: Path | None,
    dependencies: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    backend = _backend_name(config, label="region")
    if backend == "composed":
        return _build_composed_region(
            config,
            device=device,
            base=base,
            dependencies=dependencies,
        )
    sampling_backends_value = dependencies.get(
        "region_sampling_backends",
        {},
    )
    if not isinstance(sampling_backends_value, Mapping):
        raise TypeError("dependencies['region_sampling_backends'] must be a mapping")
    sampling_backends = dict(sampling_backends_value)
    if sampling_backends and backend in {"injected", "factory"}:
        raise ValueError(
            "region_sampling_backends cannot be combined with a "
            f"full region backend {backend!r}"
        )
    runtime_config = _mapping(
        config.get("runtime_config", {}),
        label="region.runtime_config",
    )
    manual_sam2 = backend == "manual+sam2" or (
        backend == "manual" and config.get("sam2") is not None
    )
    if backend in {"manual", "precomputed", "none"} and not manual_sam2:
        region = RegionRuntime(
            runtime_config=runtime_config,
            device=device,
            sampling_backends=sampling_backends,
        )
        return (
            region,
            {
                "backend": backend,
                "model_assets": False,
                "query_samplers": (region.query_sampler_identities()),
            },
        )
    if backend == "injected":
        region_runtime = _injected(
            dependencies,
            "region_runtime",
        )
        runtime_identity = region_runtime_identity(
            region_runtime,
            source="dependencies['region_runtime']",
        )
        configured_identity_fields = {
            key
            for key in (
                "provider_kind",
                "backend_id",
                "algorithm_id",
                "contract_version",
            )
            if config.get(key) is not None
        }
        if configured_identity_fields:
            configured_identity = validate_external_region_runtime_identity(
                config,
                source="region backend 'injected'",
            )
            if runtime_identity != configured_identity:
                raise ValueError(
                    "dependencies['region_runtime'] identity conflicts "
                    "with configured region identity: "
                    f"runtime={runtime_identity!r}, "
                    f"configured={configured_identity!r}"
                )
        if runtime_identity["provider_kind"] == "builtin" and not isinstance(
            region_runtime, RegionRuntime
        ):
            raise ValueError(
                "dependencies['region_runtime'] may claim "
                "provider_kind='builtin' only for RegionRuntime"
            )
        return (
            region_runtime,
            {
                "backend": backend,
                "dependency": "region_runtime",
                **runtime_identity,
            },
        )
    if backend == "factory":
        configured_identity = validate_external_region_runtime_identity(
            config,
            source="region backend 'factory'",
        )
        return (
            _factory_component(
                config,
                role="region",
                base=base,
                expected_identity=configured_identity,
                identity_reader=region_runtime_identity,
            ),
            {
                "backend": backend,
                "factory": str(config.get("factory", "")),
                **configured_identity,
                "lazy": True,
            },
        )
    provider_backends = {
        "grounding_dino",
        "grounding_dino_sam2",
        "manual+sam2",
        "sam2",
    }
    if backend not in provider_backends and not manual_sam2:
        raise ValueError(f"unsupported region backend: {backend!r}")

    use_detector = backend in {
        "grounding_dino",
        "grounding_dino_sam2",
    }
    use_segmenter = backend in {"sam2", "grounding_dino_sam2"} or manual_sam2
    detector_factory: Any | None = None
    detector_identity: dict[str, str] | None = None
    detector_manifest: dict[str, Any] | None = None
    if use_detector:
        detector_config = _mapping(
            config.get("grounding_dino", {}),
            label="region.grounding_dino",
            required=True,
        )
        (
            detector_factory,
            detector_identity,
            detector_manifest,
        ) = _grounding_dino_binding(
            detector_config,
            device=device,
            base=base,
            dependencies=dependencies,
            label="region.grounding_dino",
            strict_fields=False,
            validate_configured_identity=False,
        )

    segmenter_factory: Any | None = None
    segmenter_identity: dict[str, str] | None = None
    segmenter_manifest: dict[str, Any] | None = None
    if use_segmenter:
        segmenter_config = _mapping(
            config.get("sam2", {}),
            label="region.sam2",
            required=True,
        )
        (
            segmenter_factory,
            segmenter_identity,
            segmenter_manifest,
        ) = _sam2_binding(
            segmenter_config,
            device=device,
            base=base,
            dependencies=dependencies,
            label="region.sam2",
            strict_fields=False,
            validate_configured_identity=False,
        )

    region = RegionRuntime(
        runtime_config=runtime_config,
        device=device,
        detector_factory=detector_factory,
        segmenter_factory=segmenter_factory,
        sampling_backends=sampling_backends,
        detector_provider_identity=detector_identity,
        segmenter_provider_identity=segmenter_identity,
    )
    manifest: dict[str, Any] = {
        "backend": backend,
        "lazy": True,
        "providers": region.provider_identities(),
        "query_samplers": region.query_sampler_identities(),
    }
    if detector_manifest is not None:
        manifest["grounding_dino"] = detector_manifest
    if segmenter_manifest is not None:
        manifest["sam2"] = segmenter_manifest
    return (
        region,
        manifest,
    )


def _build_tracking(
    config: Mapping[str, Any],
    *,
    device: str,
    base: Path | None,
    dependencies: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    backend = _backend_name(config, label="tracking")
    if backend == "injected":
        configured_identity = validate_external_tracking_runtime_identity(
            config,
            source="tracking backend 'injected'",
        )
        runtime_backend = _injected(
            dependencies,
            "tracking_backend",
        )
        runtime_identity = tracking_backend_identity(
            runtime_backend,
            source="dependencies['tracking_backend']",
        )
        configured_provider = {
            key: configured_identity[key]
            for key in (
                "provider_kind",
                "backend_id",
                "contract_version",
            )
        }
        if runtime_identity != configured_provider:
            raise ValueError(
                "dependencies['tracking_backend'] identity conflicts with "
                "configured tracking identity: "
                f"runtime={runtime_identity!r}, "
                f"configured={configured_provider!r}"
            )
        return (
            runtime_backend,
            {
                "backend": backend,
                "dependency": "tracking_backend",
                **configured_provider,
            },
        )
    if backend == "factory":
        configured_identity = validate_external_tracking_runtime_identity(
            config,
            source="tracking backend 'factory'",
        )
        configured_provider = {
            key: configured_identity[key]
            for key in (
                "provider_kind",
                "backend_id",
                "contract_version",
            )
        }
        return (
            _factory_component(
                config,
                role="tracking",
                base=base,
                expected_identity=configured_provider,
                identity_reader=tracking_backend_identity,
            ),
            {
                "backend": backend,
                "factory": str(config.get("factory", "")),
                **configured_provider,
                "lazy": True,
            },
        )
    if backend != "cotracker":
        raise ValueError(f"unsupported tracking backend: {backend!r}")
    checkpoint_path = _path(
        config.get("checkpoint_path"),
        label="tracking.checkpoint_path",
        base=base,
    )
    source_roots = _paths(
        config.get("source_roots", ()),
        label="tracking.source_roots",
        base=base,
    )
    tracker = CoTrackerBackend(
        checkpoint_path=checkpoint_path,
        device=device,
        linewidth=int(config.get("linewidth", 1)),
        source_roots=source_roots,
        checkpoint_sha256=config.get("checkpoint_sha256", None),
        runtime_loader=dependencies.get(
            "cotracker_runtime_loader",
            None,
        ),
    )
    return (
        tracker,
        {
            "backend": backend,
            "provider_kind": "builtin",
            "backend_id": COTRACKER_BACKEND_ID,
            "contract_version": TRACKING_BACKEND_CONTRACT_VERSION,
            "lazy": True,
            "checkpoint_path": checkpoint_path,
            "source_roots": list(source_roots),
            "device": device,
            "provider_provenance": tracker.provider_provenance,
        },
    )


def _registry_entries(
    value: Any,
) -> dict[str, str]:
    entries = _mapping(
        value,
        label="depth.backend_registry",
    )
    output: dict[str, str] = {}
    for name, import_spec in entries.items():
        backend_name = str(name).strip().lower()
        if backend_name not in SUPPORTED_DEPTH_BACKENDS:
            raise ValueError(
                f"unsupported depth registry backend: {name!r}. "
                f"Supported backends: {list(SUPPORTED_DEPTH_BACKENDS)!r}."
            )
        if not isinstance(import_spec, str):
            raise TypeError(
                "depth.backend_registry values must be explicit "
                "'module:attribute' strings"
            )
        module_name, separator, attribute = import_spec.partition(":")
        if not separator or not module_name or not attribute:
            raise ValueError(
                "depth backend import spec must be 'module:attribute', "
                f"got {import_spec!r}"
            )
        output[backend_name] = import_spec
    return output


def _merge_registry(
    target: dict[str, Any],
    additions: Mapping[str, Any],
    *,
    label: str,
) -> None:
    unsupported = sorted(
        str(name)
        for name in additions
        if str(name).strip().lower() not in SUPPORTED_DEPTH_BACKENDS
    )
    if unsupported:
        raise ValueError(
            f"{label} contains unsupported depth backends: "
            + ", ".join(unsupported)
            + f". Supported backends: {list(SUPPORTED_DEPTH_BACKENDS)!r}."
        )
    overlap = sorted(set(target).intersection(additions))
    if overlap:
        raise ValueError(
            f"{label} cannot replace already registered depth backends: "
            + ", ".join(overlap)
        )
    target.update(dict(additions))


def _reject_unsupported_depth_backend(
    requested: Any,
    *,
    source: str,
    allowed: set[str],
) -> None:
    normalized = str(requested or "").strip().lower()
    if not normalized or normalized in allowed:
        return
    raise ValueError(
        f"Unsupported depth backend {requested!r} in {source}. "
        f"Supported backends: {sorted(allowed)!r}. "
        "No fallback or remap is performed."
    )


def _prevalidate_raw_depth_scope(value: Any) -> None:
    """Validate the raw depth surface before any runtime path is resolved."""

    if not isinstance(value, Mapping):
        return
    depth = dict(value)
    _reject_unsupported_depth_backend(
        depth.get("backend"),
        source="runtime_config.depth.backend",
        allowed={"dvd", "factory", "injected", "none", "registry", "vda"},
    )
    runtime_identity = depth.get("runtime_config")
    external_runtime = (
        isinstance(runtime_identity, Mapping)
        and str(runtime_identity.get("provider_kind", "") or "").strip().lower()
        == "external"
    )
    requested_preset = str(depth.get("preset", "") or "").strip().lower()
    if (
        requested_preset
        and not external_runtime
        and requested_preset not in SUPPORTED_DEPTH_BACKENDS
    ):
        resolve_depth_estimator_preset(requested_preset)
    for field in ("config", "runtime_config"):
        nested = depth.get(field)
        if (
            isinstance(nested, Mapping)
            and str(nested.get("provider_kind", "") or "").strip().lower() != "external"
        ):
            _reject_unsupported_depth_backend(
                nested.get("model_name"),
                source=f"runtime_config.depth.{field}.model_name",
                allowed=set(SUPPORTED_DEPTH_BACKENDS),
            )
    unknown = sorted(set(depth).difference(_DEPTH_COMPONENT_ALLOWED_FIELDS))
    if unknown:
        raise ValueError(
            "runtime_config.depth contains unknown fields: " + ", ".join(unknown)
        )
    backend = str(depth.get("backend", "") or "").strip().lower()
    if backend in {"injected", "factory"}:
        _declared_depth_runtime(
            depth,
            source=f"depth backend {backend!r}",
        )


def _declared_depth_runtime(
    config: Mapping[str, Any],
    *,
    source: str,
) -> tuple[str, dict[str, Any]]:
    value = config.get("runtime_config", None)
    if not isinstance(value, Mapping):
        raise ValueError(f"{source}.runtime_config must declare a supported model_name")
    if str(value.get("provider_kind", "") or "").strip().lower() == "external":
        unknown = sorted(set(value).difference(_EXTERNAL_DEPTH_RUNTIME_ALLOWED_FIELDS))
        if unknown:
            raise ValueError(
                f"{source}.runtime_config contains unknown fields: "
                + ", ".join(unknown)
            )
        runtime = validate_external_depth_runtime_identity(
            value,
            source=f"{source}.runtime_config",
        )
    else:
        runtime = validate_depth_runtime_identity(
            value,
            source=f"{source}.runtime_config",
        )
    requested_preset = str(config.get("preset", "") or "").strip()
    if requested_preset:
        external_id = external_depth_backend_id(requested_preset)
        if external_id is not None:
            expected_model = external_depth_selection(external_id)
            if str(runtime.get("model_name", "") or "") != expected_model:
                raise ValueError(
                    f"{source}.preset {requested_preset!r} does not match "
                    "the declared external depth backend identity "
                    f"{runtime.get('model_name')!r}"
                )
        runtime["preset"] = requested_preset
    model_name = str(runtime["model_name"])
    if model_name == "dvd" and DVD_ASSET_ATTESTATION_FIELD in runtime.get(
        "model_provenance", {}
    ):
        runtime = validate_depth_estimator_assets(runtime)
    return model_name, runtime


def _depth_identity_manifest(
    runtime_config: Mapping[str, Any],
) -> dict[str, Any]:
    # Every caller reaches this private formatter only after
    # ``_declared_depth_runtime`` or ``prepare_depth_estimator`` has validated
    # identity and, when requested, read and attested both DVD asset files.
    # Re-running the pure identity normalizer here would correctly distrust
    # and downgrade that detached file-backed verification to ``pending``.
    runtime = copy.deepcopy(dict(runtime_config))
    model_name = str(runtime["model_name"])
    output: dict[str, Any] = {
        "model_name": model_name,
    }
    if str(runtime.get("preset", "") or "").strip():
        output["preset"] = str(runtime["preset"])
    if model_name == "dvd":
        output.update(
            {
                "model_family": str(runtime["model_family"]),
                "model_provenance": copy.deepcopy(runtime["model_provenance"]),
                "provenance_validation": dvd_provenance_validation(
                    runtime["model_provenance"]
                ),
            }
        )
    elif str(runtime.get("provider_kind", "") or "") == "external":
        output.update(
            {
                "provider_kind": "external",
                "backend_id": str(runtime["backend_id"]),
                "contract_version": str(runtime["contract_version"]),
            }
        )
    return output


def _depth_selection_manifest(
    config: Mapping[str, Any],
) -> dict[str, str]:
    source = str(config.get("selection_source", "") or "").strip()
    return {} if not source else {"selection_source": source}


def _prevalidate_depth_request(
    config: Mapping[str, Any],
    *,
    base: Path | None,
    dependencies: Mapping[str, Any],
) -> None:
    """Validate depth scope/config before assembling any runtime component."""

    backend = _backend_name(config, label="depth")
    if config.get("attestation_manifest_path") is not None:
        raise ValueError(
            "depth.attestation_manifest_path is benchmark-owned and must be "
            "resolved into an explicit attested config before runtime assembly"
        )
    if backend == "none":
        return
    if backend in {"injected", "factory"}:
        _declared_depth_runtime(
            config,
            source=f"depth backend {backend!r}",
        )
        return
    if backend not in {"dvd", "registry", "vda"}:
        raise ValueError(
            f"unsupported depth backend: {backend!r}. "
            f"Supported concrete backends: {list(SUPPORTED_DEPTH_BACKENDS)!r}."
        )

    preset = str(config.get("preset", "") or "").strip()
    if not preset:
        raise ValueError("depth.preset must be explicitly provided")
    config_mapping = config.get("config", None)
    config_path_value = str(config.get("config_path", "") or "").strip()
    if config_mapping is not None and config_path_value:
        raise ValueError("depth.config and depth.config_path are mutually exclusive")
    if config_mapping is not None and not isinstance(config_mapping, Mapping):
        raise TypeError("depth.config must be a mapping")
    config_path = (
        _path(
            config_path_value,
            label="depth.config_path",
            base=base,
        )
        if config_path_value
        else None
    )
    preflight = preflight_depth_estimator_request(
        preset=preset,
        config=(
            None if config_mapping is None else copy.deepcopy(dict(config_mapping))
        ),
        config_path=config_path,
    )
    model_name = str(preflight["model_name"])
    if backend in SUPPORTED_DEPTH_BACKENDS and backend != model_name:
        raise ValueError(
            f"depth backend {backend!r} does not match the requested "
            f"model family {model_name!r}"
        )

    _registry_entries(config.get("backend_registry", {}))
    injected_registry = dependencies.get("depth_backend_registry", {})
    if not isinstance(injected_registry, Mapping):
        raise TypeError("dependencies['depth_backend_registry'] must be a mapping")
    unsupported_injected = sorted(
        str(name)
        for name in injected_registry
        if str(name).strip().lower() not in SUPPORTED_DEPTH_BACKENDS
    )
    if unsupported_injected:
        raise ValueError(
            "dependencies['depth_backend_registry'] contains unsupported "
            "depth backends: "
            + ", ".join(unsupported_injected)
            + f". Supported backends: {list(SUPPORTED_DEPTH_BACKENDS)!r}."
        )


def _build_depth(
    config: Mapping[str, Any],
    *,
    device: str,
    base: Path | None,
    dependencies: Mapping[str, Any],
) -> tuple[Any, dict[str, Any] | None, dict[str, Any]]:
    backend = _backend_name(config, label="depth")
    if backend == "none":
        return (
            None,
            None,
            {
                "backend": backend,
                "estimator": False,
            },
        )
    if backend == "injected":
        model_name, runtime_config = _declared_depth_runtime(
            config,
            source="depth backend 'injected'",
        )
        return (
            _injected(dependencies, "depth_estimator"),
            runtime_config,
            {
                "backend": backend,
                "dependency": "depth_estimator",
                **_depth_selection_manifest(config),
                **_depth_identity_manifest(runtime_config),
            },
        )
    if backend == "factory":
        model_name, runtime_config = _declared_depth_runtime(
            config,
            source="depth backend 'factory'",
        )
        return (
            _factory_component(
                config,
                role="depth",
                base=base,
            ),
            runtime_config,
            {
                "backend": backend,
                "factory": str(config.get("factory", "")),
                **_depth_selection_manifest(config),
                **_depth_identity_manifest(runtime_config),
                "lazy": True,
            },
        )
    if backend not in {
        "dvd",
        "registry",
        "vda",
    }:
        raise ValueError(f"unsupported depth backend: {backend!r}")

    preset = str(config.get("preset", "") or "").strip()
    if not preset:
        raise ValueError("depth.preset must be explicitly provided")
    config_mapping = config.get("config", None)
    config_path_value = str(config.get("config_path", "") or "").strip()
    if config_mapping is not None and config_path_value:
        raise ValueError("depth.config and depth.config_path are mutually exclusive")
    if config_mapping is not None and not isinstance(
        config_mapping,
        Mapping,
    ):
        raise TypeError("depth.config must be a mapping")
    config_path = (
        _path(
            config_path_value,
            label="depth.config_path",
            base=base,
        )
        if config_path_value
        else None
    )
    weights_root_value = str(config.get("weights_root", "") or "").strip()
    weights_root = (
        _path(
            weights_root_value,
            label="depth.weights_root",
            base=base,
        )
        if weights_root_value
        else None
    )

    registry: dict[str, Any] = {}
    vda_source_root = ""
    vda_provider_module = ""
    if backend == "vda":
        vda_source_root = _path(
            config.get("source_root"),
            label="depth.source_root",
            base=base,
        )
        vda_provider_module = str(
            config.get(
                "provider_module",
                VDA_DEFAULT_PROVIDER_MODULE,
            )
            or ""
        ).strip()
        if not vda_provider_module:
            raise ValueError("depth.provider_module must be explicitly non-empty")
        vda_backend_factory = VDA_BACKEND_REGISTRY["vda"]

        def vda_factory(**kwargs: Any) -> Any:
            return vda_backend_factory(
                **kwargs,
                source_roots=(vda_source_root,),
                module_name=vda_provider_module,
            )

        registry["vda"] = vda_factory
    elif backend == "dvd":
        source_root = _path(
            config.get("source_root"),
            label="depth.source_root",
            base=base,
        )
        checkpoints_root = _path(
            config.get("checkpoints_root"),
            label="depth.checkpoints_root",
            base=base,
        )
        registry.update(
            build_dvd_backend_registry(
                source_root=source_root,
                checkpoints_root=checkpoints_root,
                runtime_loader=dependencies.get(
                    "dvd_runtime_loader",
                    None,
                ),
            )
        )
    _merge_registry(
        registry,
        _registry_entries(
            config.get("backend_registry", {}),
        ),
        label="depth.backend_registry",
    )
    injected_registry = dependencies.get(
        "depth_backend_registry",
        {},
    )
    if not isinstance(injected_registry, Mapping):
        raise TypeError("dependencies['depth_backend_registry'] must be a mapping")
    _merge_registry(
        registry,
        injected_registry,
        label="dependencies['depth_backend_registry']",
    )

    estimator, runtime = prepare_depth_estimator(
        backend_registry=registry,
        device=device,
        preset=preset,
        config=(
            None if config_mapping is None else copy.deepcopy(dict(config_mapping))
        ),
        config_path=config_path,
        fp32=bool(config.get("fp32", False)),
        input_size=int(config.get("input_size", 512)),
        weights_root=weights_root,
        runtime_context=_mapping(
            config.get("runtime_context", {}),
            label="depth.runtime_context",
        ),
        validate_assets=bool(config.get("validate_assets", True)),
        seed=int(config.get("seed", 42)),
        seed_hook=dependencies.get(
            "depth_seed_hook",
            None,
        ),
    )
    model_name = str(runtime.get("model_name", "") or "")
    if model_name not in registry:
        available = ", ".join(sorted(registry)) or "<none>"
        raise ValueError(
            f"depth preset {preset!r} resolves to {model_name!r}, "
            f"but the explicit registry provides: {available}"
        )
    return (
        estimator,
        runtime,
        {
            "backend": backend,
            "requested_preset": preset,
            "preset": str(runtime.get("preset", preset) or preset),
            **_depth_selection_manifest(config),
            **_depth_identity_manifest(runtime),
            "config_source": str(runtime.get("config_source", "") or ""),
            "weights_root": weights_root or "",
            **(
                {
                    "source_root": vda_source_root,
                    "provider_module": vda_provider_module,
                }
                if backend == "vda"
                else {}
            ),
            "registered_backends": sorted(registry),
            "lazy": True,
        },
    )


def _build_depth_calibration(
    config: Mapping[str, Any],
    *,
    base: Path | None,
    dependencies: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    backend = _backend_name(
        config,
        label="depth_calibration",
    )
    if backend == "none":
        return (
            None,
            {
                "backend": backend,
            },
        )
    if backend == "builtin":
        unknown = sorted(set(config).difference({"backend"}))
        if unknown:
            raise ValueError(
                "depth_calibration builtin contains unsupported fields: "
                + ", ".join(unknown)
            )
        return (
            run_depth_calibration_runtime,
            {
                "backend": "builtin",
                "policy": "validated_depth_calibration",
                "provider": False,
            },
        )
    if backend == "injected":
        return (
            _injected(
                dependencies,
                "depth_calibration",
            ),
            {
                "backend": backend,
                "dependency": "depth_calibration",
            },
        )
    if backend == "factory":
        return (
            _factory_component(
                config,
                role="depth_calibration",
                base=base,
            ),
            {
                "backend": backend,
                "factory": str(config.get("factory", "")),
                "lazy": True,
            },
        )
    raise ValueError(
        f"unsupported depth_calibration backend: {backend!r}; "
        "base-depth calibration must be injected explicitly"
    )


def _build_rigid_pose_provider(
    config: Mapping[str, Any],
    *,
    effective_backend: str,
    base: Path | None,
    dependencies: Mapping[str, Any],
) -> tuple[Any, dict[str, Any]]:
    raw_provider = config.get(
        "rigid_provider",
        {"backend": "tracked_pointcloud_rigid"},
    )
    provider = _mapping(
        raw_provider,
        label="pose.rigid_provider",
        required=True,
    )
    provider_backend = _backend_name(
        provider,
        label="pose.rigid_provider",
    )
    declared_effective = (
        str(provider.get("effective_backend", "") or "").strip().lower()
    )
    if (
        declared_effective
        and declared_effective != str(effective_backend).strip().lower()
    ):
        raise ValueError(
            "pose.rigid_provider cannot change pose.effective_backend: "
            f"provider={declared_effective!r}, "
            f"pose={effective_backend!r}"
        )

    common_fields = {
        "algorithm_id",
        "backend",
        "backend_id",
        "contract_version",
        "effective_backend",
        "provider_kind",
    }
    if provider_backend in {
        "builtin",
        "tracked_pointcloud_rigid",
    }:
        unknown = sorted(set(provider).difference(common_fields))
        if unknown:
            raise ValueError(
                "pose.rigid_provider built-in contains unsupported fields: "
                + ", ".join(unknown)
            )
        rigid_backend = TrackedPointCloudRigidPoseBackend()
        identity = rigid_pose_backend_identity(
            rigid_backend,
            source="built-in rigid pose backend",
        )
        configured_identity_fields = {
            field
            for field in (
                "provider_kind",
                "backend_id",
                "algorithm_id",
                "contract_version",
            )
            if provider.get(field) is not None
        }
        if configured_identity_fields:
            if len(configured_identity_fields) != 4:
                raise ValueError(
                    "pose.rigid_provider built-in identity must declare "
                    "provider_kind, backend_id, algorithm_id, and "
                    "contract_version together"
                )
            configured_identity = normalize_rigid_pose_backend_identity(
                provider_kind=provider.get("provider_kind"),
                backend_id=provider.get("backend_id"),
                algorithm_id=provider.get("algorithm_id"),
                contract_version=provider.get("contract_version"),
                source="pose.rigid_provider built-in",
            )
            if configured_identity != identity:
                raise ValueError(
                    "pose.rigid_provider built-in identity conflicts with "
                    "the selected implementation: "
                    f"configured={configured_identity!r}, "
                    f"runtime={identity!r}"
                )
        return (
            rigid_backend,
            {
                "backend": "tracked_pointcloud_rigid",
                **identity,
                "lazy": True,
            },
        )

    external_allowed = set(common_fields)
    if provider_backend == "factory":
        external_allowed.update({"factory", "kwargs"})
    unknown = sorted(set(provider).difference(external_allowed))
    if unknown:
        raise ValueError(
            f"pose.rigid_provider {provider_backend!r} contains unsupported "
            "fields: " + ", ".join(unknown)
        )
    configured_identity = normalize_rigid_pose_backend_identity(
        provider_kind=provider.get("provider_kind"),
        backend_id=provider.get("backend_id"),
        algorithm_id=provider.get("algorithm_id"),
        contract_version=provider.get("contract_version"),
        source=f"pose.rigid_provider {provider_backend!r}",
    )
    if configured_identity["provider_kind"] != "external":
        raise ValueError(
            "an injected or factory pose.rigid_provider must declare "
            "provider_kind='external'"
        )
    if provider_backend == "injected":
        runtime_backend = _injected(
            dependencies,
            "rigid_pose_backend",
        )
        actual_identity = rigid_pose_backend_identity(
            runtime_backend,
            source="dependencies['rigid_pose_backend']",
        )
        if actual_identity != configured_identity:
            raise ValueError(
                "dependencies['rigid_pose_backend'] identity conflicts with "
                "configured rigid provider identity: "
                f"runtime={actual_identity!r}, "
                f"configured={configured_identity!r}"
            )
        return (
            runtime_backend,
            {
                "backend": provider_backend,
                "dependency": "rigid_pose_backend",
                **configured_identity,
            },
        )
    if provider_backend == "factory":
        return (
            _factory_component(
                provider,
                role="rigid_pose_backend",
                base=base,
                expected_identity=configured_identity,
                identity_reader=rigid_pose_backend_identity,
            ),
            {
                "backend": provider_backend,
                "factory": str(provider.get("factory", "")),
                **configured_identity,
                "lazy": True,
            },
        )
    raise ValueError(f"unsupported pose.rigid_provider backend: {provider_backend!r}")


def _build_pose(
    config: Mapping[str, Any],
    *,
    device: str,
    base: Path | None,
    dependencies: Mapping[str, Any],
) -> tuple[Any, Any, Any, dict[str, Any]]:
    backend = _backend_name(config, label="pose")
    if backend == "none":
        if "rigid_provider" in config:
            raise ValueError("pose.rigid_provider requires an enabled pose backend")
        return (
            None,
            None,
            None,
            {
                "backend": backend,
            },
        )
    if backend == "estimator_injected":
        if "rigid_provider" in config:
            raise ValueError(
                "pose.rigid_provider conflicts with estimator_injected; "
                "the injected estimator owns its rigid provider"
            )
        return (
            None,
            _injected(
                dependencies,
                "pose_estimator",
            ),
            None,
            {
                "backend": backend,
                "dependency": "pose_estimator",
            },
        )
    if backend == "estimator_factory":
        if "rigid_provider" in config:
            raise ValueError(
                "pose.rigid_provider conflicts with estimator_factory; "
                "the estimator factory owns its rigid provider"
            )
        return (
            None,
            _factory_component(
                config,
                role="pose_estimator",
                base=base,
            ),
            None,
            {
                "backend": backend,
                "factory": str(config.get("factory", "")),
                "lazy": True,
            },
        )

    def with_rigid_provider(
        model_backend: Any,
        estimator: Any,
        manifest: Mapping[str, Any],
    ) -> tuple[Any, Any, Any, dict[str, Any]]:
        pose_manifest = dict(manifest)
        effective = (
            str(
                pose_manifest.get(
                    "effective_backend",
                    pose_manifest.get("backend", ""),
                )
                or ""
            )
            .strip()
            .lower()
        )
        rigid_backend, rigid_manifest = _build_rigid_pose_provider(
            config,
            effective_backend=effective,
            base=base,
            dependencies=dependencies,
        )
        pose_manifest["rigid_provider"] = rigid_manifest
        return (
            model_backend,
            estimator,
            rigid_backend,
            pose_manifest,
        )

    if backend == "injected":
        runtime_identity = validate_external_pose_runtime_identity(
            config,
            source="pose backend 'injected'",
        )
        configured_provider = {
            key: runtime_identity[key]
            for key in (
                "provider_kind",
                "backend_id",
                "contract_version",
            )
        }
        runtime_backend = _injected(
            dependencies,
            "pose_backend",
        )
        actual_provider = pose_backend_identity(
            runtime_backend,
            source="dependencies['pose_backend']",
        )
        if actual_provider != configured_provider:
            raise ValueError(
                "dependencies['pose_backend'] identity conflicts with "
                "configured pose identity: "
                f"runtime={actual_provider!r}, "
                f"configured={configured_provider!r}"
            )
        return with_rigid_provider(
            runtime_backend,
            None,
            {
                "backend": backend,
                "dependency": "pose_backend",
                **configured_provider,
                "effective_backend": runtime_identity["effective_backend"],
            },
        )
    if backend == "factory":
        runtime_identity = validate_external_pose_runtime_identity(
            config,
            source="pose backend 'factory'",
        )
        configured_provider = {
            key: runtime_identity[key]
            for key in (
                "provider_kind",
                "backend_id",
                "contract_version",
            )
        }
        return with_rigid_provider(
            _factory_component(
                config,
                role="pose",
                base=base,
                expected_identity=configured_provider,
                identity_reader=pose_backend_identity,
            ),
            None,
            {
                "backend": backend,
                "factory": str(config.get("factory", "")),
                **configured_provider,
                "effective_backend": runtime_identity["effective_backend"],
                "lazy": True,
            },
        )
    if backend == POINTCLOUD_KABSCH_BACKEND:
        return with_rigid_provider(
            None,
            None,
            {
                "backend": backend,
                "effective_backend": backend,
                "model_backend": False,
            },
        )
    if backend == "foundationpose":
        source_root = _path(
            config.get("source_root"),
            label="pose.source_root",
            base=base,
        )
        weights_root = _path(
            config.get("weights_root", ""),
            label="pose.weights_root",
            base=base,
            required=False,
        )
        debug_dir = _path(
            config.get("debug_dir", ""),
            label="pose.debug_dir",
            base=base,
            required=False,
        )
        pose_backend = FoundationPoseBackend(
            source_root=source_root,
            weights_root=weights_root,
            device=device,
            debug_dir=debug_dir,
            dependency_loader=dependencies.get(
                "foundationpose_dependency_loader",
                None,
            ),
        )
        return with_rigid_provider(
            pose_backend,
            None,
            {
                "backend": backend,
                "effective_backend": backend,
                "provider_kind": pose_backend.provider_kind,
                "backend_id": pose_backend.backend_id,
                "algorithm_id": pose_backend.algorithm_id,
                "contract_version": pose_backend.contract_version,
                "source_root": source_root,
                "weights_root": weights_root,
                "debug_dir": debug_dir,
                "lazy": True,
            },
        )
    if backend == "freepose":
        source_roots = _paths(
            config.get("source_roots"),
            label="pose.source_roots",
            base=base,
            required=True,
        )
        cache_dir = _path(
            config.get("cache_dir"),
            label="pose.cache_dir",
            base=base,
        )
        torch_home = _path(
            config.get("torch_home"),
            label="pose.torch_home",
            base=base,
        )
        debug_dir = _path(
            config.get("debug_dir", ""),
            label="pose.debug_dir",
            base=base,
            required=False,
        )
        optional = {
            key: config[key]
            for key in (
                "n_coarse_poses",
                "n_fine_poses",
                "bbox_extend",
                "neighborhood",
                "layer",
                "batch_size",
                "mask_scores",
            )
            if key in config
        }
        pose_backend = FreePoseBackend(
            source_roots=source_roots,
            cache_dir=cache_dir,
            torch_home=torch_home,
            device=device,
            debug_dir=debug_dir,
            runtime_loader=dependencies.get(
                "freepose_runtime_loader",
                None,
            ),
            **optional,
        )
        return with_rigid_provider(
            pose_backend,
            None,
            {
                "backend": backend,
                "effective_backend": backend,
                "provider_kind": pose_backend.provider_kind,
                "backend_id": pose_backend.backend_id,
                "algorithm_id": pose_backend.algorithm_id,
                "contract_version": pose_backend.contract_version,
                "source_roots": list(source_roots),
                "cache_dir": cache_dir,
                "torch_home": torch_home,
                "debug_dir": debug_dir,
                "lazy": True,
            },
        )
    if backend == "sinref6d":
        source_root = _path(
            config.get("source_root"),
            label="pose.source_root",
            base=base,
        )
        checkpoint_path = _path(
            config.get("checkpoint_path"),
            label="pose.checkpoint_path",
            base=base,
        )
        runtime_dir = _path(
            config.get("runtime_dir"),
            label="pose.runtime_dir",
            base=base,
        )
        template_dir = _path(
            config.get("template_dir"),
            label="pose.template_dir",
            base=base,
        )
        debug_dir = _path(
            config.get("debug_dir"),
            label="pose.debug_dir",
            base=base,
        )
        optional = {
            key: config[key]
            for key in (
                "n_template_view",
                "template_resolution",
                "det_score_thresh",
                "subprocess_timeout_s",
                "python_executable",
            )
            if key in config
        }
        pose_backend = SinRef6DBackend(
            source_root=source_root,
            checkpoint_path=checkpoint_path,
            runtime_dir=runtime_dir,
            template_dir=template_dir,
            debug_dir=debug_dir,
            device=device,
            runtime_bundle=dependencies.get(
                "sinref6d_runtime_bundle",
                None,
            ),
            runtime_loader=dependencies.get(
                "sinref6d_runtime_loader",
                None,
            ),
            **optional,
        )
        return with_rigid_provider(
            pose_backend,
            None,
            {
                "backend": backend,
                "effective_backend": backend,
                "provider_kind": pose_backend.provider_kind,
                "backend_id": pose_backend.backend_id,
                "algorithm_id": pose_backend.algorithm_id,
                "contract_version": pose_backend.contract_version,
                "source_root": source_root,
                "checkpoint_path": checkpoint_path,
                "runtime_dir": runtime_dir,
                "template_dir": template_dir,
                "debug_dir": debug_dir,
                "lazy": True,
            },
        )
    raise ValueError(f"unsupported pose backend: {backend!r}")


def build_video2traj_runtime(
    runtime_config: Mapping[str, Any],
    *,
    dependencies: Mapping[str, Any] | None = None,
    asset_base: str | Path | None = None,
) -> dict[str, Any]:
    """Assemble one standalone runtime from explicit configuration.

    The returned mapping is intentionally a small callable boundary rather
    than a prescribed application architecture.  ``dependencies`` is the
    explicit injection seam used by tests, alternate model packages, and
    callers that already own initialized resources.
    """

    if not isinstance(runtime_config, Mapping):
        raise TypeError("runtime_config must be an explicit mapping")
    config = copy.deepcopy(dict(runtime_config))
    if dependencies is not None and not isinstance(
        dependencies,
        Mapping,
    ):
        raise TypeError("dependencies must be a mapping")
    dependency_map = dict(dependencies or {})
    _prevalidate_raw_depth_scope(config.get("depth"))
    device = str(config.get("device", "") or "").strip()
    if not device:
        raise ValueError("runtime_config.device must be explicitly provided")
    base = _explicit_base(
        config,
        asset_base=asset_base,
    )

    region_config = _mapping(
        config.get("region", None),
        label="region",
        required=True,
    )
    tracking_config = _mapping(
        config.get("tracking", None),
        label="tracking",
        required=True,
    )
    tracking_backend_name = _backend_name(
        tracking_config,
        label="tracking",
    )
    if tracking_backend_name in {"injected", "factory"}:
        validate_external_tracking_runtime_identity(
            tracking_config,
            source=f"tracking backend {tracking_backend_name!r}",
        )
    depth_config = _mapping(
        config.get("depth", None),
        label="depth",
        required=True,
    )
    _prevalidate_depth_request(
        depth_config,
        base=base,
        dependencies=dependency_map,
    )
    calibration_config = _mapping(
        config.get(
            "depth_calibration",
            {"backend": "none"},
        ),
        label="depth_calibration",
        required=True,
    )
    pose_config = _mapping(
        config.get("pose", {"backend": "none"}),
        label="pose",
        required=True,
    )
    pose_backend_name = _backend_name(pose_config, label="pose")
    if pose_backend_name in {"injected", "factory"}:
        validate_external_pose_runtime_identity(
            pose_config,
            source=f"pose backend {pose_backend_name!r}",
        )
    (
        pose_backend,
        pose_estimator,
        rigid_pose_backend,
        pose_manifest,
    ) = _build_pose(
        pose_config,
        device=device,
        base=base,
        dependencies=dependency_map,
    )

    region_runtime, region_manifest = _build_region(
        region_config,
        device=device,
        base=base,
        dependencies=dependency_map,
    )
    tracking_backend, tracking_manifest = _build_tracking(
        tracking_config,
        device=device,
        base=base,
        dependencies=dependency_map,
    )
    (
        depth_estimator,
        depth_runtime_config,
        depth_manifest,
    ) = _build_depth(
        depth_config,
        device=device,
        base=base,
        dependencies=dependency_map,
    )
    depth_calibration, calibration_manifest = _build_depth_calibration(
        calibration_config,
        base=base,
        dependencies=dependency_map,
    )
    manifest = {
        "contract": "explicit_video2traj_runtime",
        "device": device,
        "asset_base": ("" if base is None else base.as_posix()),
        "components": {
            "region": region_manifest,
            "tracking": tracking_manifest,
            "depth": depth_manifest,
            "depth_calibration": calibration_manifest,
            "pose": pose_manifest,
        },
        "heavy_dependencies_lazy": True,
        "bench_resolution": False,
        "uid_resolution": False,
        "global_root_resolution": False,
    }
    return {
        "region_runtime": region_runtime,
        "tracking_backend": tracking_backend,
        "depth_estimator": depth_estimator,
        "depth_calibration": depth_calibration,
        "depth_runtime_config": depth_runtime_config,
        "pose_backend": pose_backend,
        "pose_estimator": pose_estimator,
        "rigid_pose_backend": rigid_pose_backend,
        "manifest": manifest,
    }


__all__ = [
    "build_video2traj_runtime",
    "load_video2traj_runtime_config",
]
