"""Versioned, path-explicit depth cache and diagnostic publication.

Planning and inspection are read-only.  Publication requires a second,
explicit call and defaults to ``dry_run=True``.  The module does not discover
benchmarks, run keys, model registries, checkpoints, or simulator state.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import os
import shutil
import stat
import uuid
from collections.abc import Callable, Collection, Mapping
from pathlib import Path
from typing import Any

import numpy as np

DEPTH_CACHE_SCHEMA = "dream-exe.depth-cache"
DEPTH_INPUT_IDENTITY_SCHEMA = "dream-exe.depth-input-identity"
DEPTH_PUBLICATION_PLAN_SCHEMA = "dream-exe.depth-publication-plan"
DEPTH_CACHE_METADATA_FILENAME = "depth_cache_meta.json"
DEPTH_COMPAT_METADATA_FILENAME = "depth_meta.npy"
LEGACY_METADATA_POLICY_REJECT = "reject"
LEGACY_METADATA_POLICY_TRUSTED_PICKLE = "trusted_pickle_read_only"
_TRANSACTION_SCHEMA = "dream-exe.depth-publication-transaction"
_PLAN_VERSION = 1
_MAX_METADATA_BYTES = 4 * 1024 * 1024
_MAX_IDENTITY_NODES = 100_000
_MAX_IDENTITY_DEPTH = 32
_CALIBRATION_FIELDS = frozenset({"base"})
_BASE_CALIBRATION_FIELDS = frozenset(
    {
        "H",
        "T",
        "W",
        "applied",
        "b0",
        "b_union",
        "base_frac",
        "cached",
        "calib_mask_equiv",
        "calib_mask_frac",
        "calib_region",
        "calibrated",
        "calibration_solver",
        "depth_space",
        "enabled",
        "init_calibration",
        "final_smooth",
        "init_calibration_mode",
        "invalidate_mode",
        "invalidated",
        "multi_roi",
        "multi_roi_strategy",
        "per_roi",
        "reason",
        "roi0_frac",
        "roi_count",
        "roi_dilate_px",
        "roi_source",
        "roi_used",
        "s0",
        "s_union",
        "sanitize",
        "scheme_b",
        "vm0_frac",
        "warning",
    }
)
_CALIBRATION_STATUS_FIELDS = frozenset(
    {
        "applied",
        "distribution_v1",
        "enabled",
        "mode",
        "reason",
    }
)
_SANITIZE_STATUS_FIELDS = frozenset({"applied", "enabled", "reason"})
_PER_ROI_CALIBRATION_FIELDS = frozenset({"b", "mask_frac", "roi_index", "s"})
_DISTRIBUTION_DIAGNOSTIC_FIELDS = frozenset(
    {
        "applied",
        "blend_px",
        "global_branch",
        "mode",
        "support_branch",
        "support_frame0",
        "support_roi_pixels",
        "task_branch",
        "task_roi_margin_px",
        "task_roi_pixels",
        "vis_threshold",
    }
)
_DISTRIBUTION_GLOBAL_FIELDS = frozenset({"b", "pixel_count", "s"})
_DISTRIBUTION_TASK_FIELDS = frozenset(
    {
        "b",
        "fallback_to_global",
        "pixel_count",
        "s",
        "task_bias_delta_limit_m",
        "task_scale_ratio_limit",
    }
)
_DISTRIBUTION_SUPPORT_FIELDS = frozenset(
    {"bias_m", "pixel_count", "rmse_m", "support_bias_limit_m"}
)
_DISTRIBUTION_SUPPORT_FRAME_FIELDS = frozenset(
    {
        "edge_band_fraction",
        "fallback_reason",
        "pixel_count",
        "support_quantile_max",
        "support_quantile_min",
    }
)
_DISTRIBUTION_COUNT_FIELDS = frozenset({"max", "mean", "min"})
_DEPTH_PARAMETER_FIELDS = frozenset(
    {
        "calibration_enabled",
        "device",
        "fp32",
        "input_size",
        "model_name",
        "preset",
        "seed",
        "target_fps",
        "validate_assets",
    }
)
_DVD_PROVENANCE_FIELDS = frozenset(
    {
        "asset_attestation",
        "checkpoint_identity",
        "config_identity",
        "kind",
        "preset",
        "uid",
        "validation",
    }
)
_VDA_PROVENANCE_FIELDS = frozenset(
    {
        "backend_id",
        "checkpoint_identity",
        "contract_version",
        "encoder",
        "metric",
        "provider_kind",
        "revision",
    }
)
_ASSET_ATTESTATION_FIELDS = frozenset({"checkpoint_sha256", "model_config_sha256"})
_PROVENANCE_VALIDATION_FIELDS = frozenset(
    {
        "algorithm",
        "asset_integrity_gate",
        "assets",
        "live_checkpoint_finetune_gate",
        "status",
        "verification_scope",
    }
)
_PROVENANCE_ASSET_FIELDS = frozenset({"actual_sha256", "expected_sha256"})
_TARGET_CALIBRATION_FIELDS = frozenset(
    {
        "applied",
        "b",
        "calib_region",
        "calibration_solver",
        "depth_mp4",
        "depth_npy",
        "depth_shape",
        "depth_source",
        "enabled",
        "fallback",
        "fallback_reason",
        "kind",
        "mask_frac",
        "mask_pixels",
        "object_id",
        "runtime_object_key",
        "s",
        "safe_object_id",
        "stage_ids",
        "target_name",
    }
)
_CACHE_METADATA_FIELDS = frozenset(
    {
        "cache_path",
        "cache_signature",
        "calibration",
        "debug_media",
        "depth",
        "depth_config_source",
        "depth_model",
        "depth_space",
        "input_fingerprints",
        "input_identity",
        "metadata_format",
        "model_provenance",
        "model_provenance_fingerprint",
        "parameter_fingerprint",
        "parameters",
        "schema",
        "format",
        "source",
        "target_depths",
    }
)
_DEPTH_RECORD_FIELDS = frozenset(
    {"array_fingerprint", "dtype", "file_sha256", "fps", "shape"}
)
_TARGET_DEPTH_RECORD_FIELDS = frozenset(
    {
        "array_fingerprint",
        "calibration",
        "dtype",
        "file_sha256",
        "path",
        "shape",
        "source",
    }
)
_FILE_FINGERPRINT_FIELDS = frozenset({"mtime_ns", "path", "sha256", "size"})
_DEBUG_MEDIA_RECORD_FIELDS = frozenset(
    {"media_type", "path", "sha256", "size", "source"}
)


class DepthCacheValidationError(ValueError):
    """Raised when an explicit depth cache fails validation."""


class DepthPublicationConflictError(RuntimeError):
    """Raised when a publication plan is stale or contains conflicts."""


class DepthPublicationRecoveryRequired(RuntimeError):
    """Raised when automatic rollback could not restore every target."""

    def __init__(self, message: str, *, transaction_dir: Path) -> None:
        super().__init__(message)
        self.transaction_dir = transaction_dir


class _HashWriter:
    def __init__(self) -> None:
        self.digest = hashlib.sha256()
        self.size = 0

    def write(self, value: bytes | bytearray | memoryview) -> int:
        view = memoryview(value)
        self.digest.update(view)
        self.size += int(view.nbytes)
        return int(view.nbytes)

    def tell(self) -> int:
        return int(self.size)

    def flush(self) -> None:
        return None


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(child)
            for key, child in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }
    if isinstance(value, (list, tuple)):
        return [_json_ready(child) for child in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError("provenance and metadata must not contain NaN or infinity")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(
        "provenance and metadata values must be JSON-compatible, "
        f"got {type(value).__name__}"
    )


def _schema_mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    return dict(value)


def _reject_unknown_schema_fields(
    value: Mapping[str, Any],
    *,
    allowed: frozenset[str],
    label: str,
) -> None:
    unknown = sorted(set(value).difference(allowed))
    if unknown:
        raise ValueError(f"{label} contains unknown fields: " + ", ".join(unknown))


def _normalize_distribution_diagnostics(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    diagnostics = _schema_mapping(value, label=label)
    _reject_unknown_schema_fields(
        diagnostics,
        allowed=_DISTRIBUTION_DIAGNOSTIC_FIELDS,
        label=label,
    )
    nested_fields = (
        ("global_branch", _DISTRIBUTION_GLOBAL_FIELDS),
        ("task_branch", _DISTRIBUTION_TASK_FIELDS),
        ("support_branch", _DISTRIBUTION_SUPPORT_FIELDS),
        ("support_frame0", _DISTRIBUTION_SUPPORT_FRAME_FIELDS),
        ("task_roi_pixels", _DISTRIBUTION_COUNT_FIELDS),
        ("support_roi_pixels", _DISTRIBUTION_COUNT_FIELDS),
    )
    for field, allowed in nested_fields:
        if field not in diagnostics:
            continue
        nested = _schema_mapping(
            diagnostics[field],
            label=f"{label}.{field}",
        )
        _reject_unknown_schema_fields(
            nested,
            allowed=allowed,
            label=f"{label}.{field}",
        )
    return _json_ready(diagnostics)


def normalize_depth_calibration_metadata(
    value: Mapping[str, Any] | None,
    *,
    label: str = "calibration",
) -> dict[str, Any]:
    """Normalize the only calibration diagnostics publishable by current implementation."""

    calibration = _schema_mapping(value or {}, label=label)
    _reject_unknown_schema_fields(
        calibration,
        allowed=_CALIBRATION_FIELDS,
        label=label,
    )
    if "base" not in calibration:
        return {}
    base = _schema_mapping(
        calibration["base"],
        label=f"{label}.base",
    )
    _reject_unknown_schema_fields(
        base,
        allowed=_BASE_CALIBRATION_FIELDS,
        label=f"{label}.base",
    )
    if "final_smooth" in base:
        smooth = _schema_mapping(base["final_smooth"], label=f"{label}.base.final_smooth")
        _reject_unknown_schema_fields(
            smooth, allowed=_SANITIZE_STATUS_FIELDS, label=f"{label}.base.final_smooth",
        )
    if "sanitize" in base:
        sanitize = _schema_mapping(
            base["sanitize"],
            label=f"{label}.base.sanitize",
        )
        _reject_unknown_schema_fields(
            sanitize,
            allowed=_SANITIZE_STATUS_FIELDS,
            label=f"{label}.base.sanitize",
        )
    if "init_calibration" in base:
        init = _schema_mapping(
            base["init_calibration"],
            label=f"{label}.base.init_calibration",
        )
        _reject_unknown_schema_fields(
            init,
            allowed=_CALIBRATION_STATUS_FIELDS,
            label=f"{label}.base.init_calibration",
        )
        if "distribution_v1" in init:
            init["distribution_v1"] = _normalize_distribution_diagnostics(
                init["distribution_v1"],
                label=(f"{label}.base.init_calibration.distribution_v1"),
            )
        base["init_calibration"] = init
    if "per_roi" in base:
        records = base["per_roi"]
        if not isinstance(records, list):
            raise TypeError(f"{label}.base.per_roi must be a list")
        normalized_records: list[dict[str, Any]] = []
        for index, record in enumerate(records):
            item = _schema_mapping(
                record,
                label=f"{label}.base.per_roi[{index}]",
            )
            _reject_unknown_schema_fields(
                item,
                allowed=_PER_ROI_CALIBRATION_FIELDS,
                label=f"{label}.base.per_roi[{index}]",
            )
            normalized_records.append(item)
        base["per_roi"] = normalized_records
    calibration["base"] = base
    return _json_ready(calibration)


def normalize_target_depth_calibration_metadata(
    value: Mapping[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    """Normalize one target-lift calibration record."""

    calibration = _schema_mapping(value or {}, label=label)
    _reject_unknown_schema_fields(
        calibration,
        allowed=_TARGET_CALIBRATION_FIELDS,
        label=label,
    )
    return _json_ready(calibration)


def normalize_depth_publication_parameters(
    value: Mapping[str, Any] | None,
    *,
    label: str = "parameters",
) -> dict[str, Any]:
    """Normalize explicit parameters recorded by depth publication."""

    parameters = _schema_mapping(value or {}, label=label)
    _reject_unknown_schema_fields(
        parameters,
        allowed=_DEPTH_PARAMETER_FIELDS,
        label=label,
    )
    for field in ("fp32", "validate_assets", "calibration_enabled"):
        if field in parameters and type(parameters[field]) is not bool:
            raise TypeError(f"{label}.{field} must be a boolean")
    for field in ("seed", "input_size"):
        if field not in parameters:
            continue
        raw = parameters[field]
        if isinstance(raw, bool) or not isinstance(raw, int):
            raise TypeError(f"{label}.{field} must be an integer")
        if field == "input_size" and raw <= 0:
            raise ValueError(f"{label}.input_size must be > 0")
    if "target_fps" in parameters:
        raw_fps = parameters["target_fps"]
        if isinstance(raw_fps, bool) or not isinstance(
            raw_fps,
            (int, float),
        ):
            raise TypeError(f"{label}.target_fps must be numeric")
        if not np.isfinite(float(raw_fps)) or float(raw_fps) <= 0.0:
            raise ValueError(f"{label}.target_fps must be finite and > 0")
    for field in ("device", "model_name", "preset"):
        if field in parameters and not isinstance(
            parameters[field],
            str,
        ):
            raise TypeError(f"{label}.{field} must be a string")
    return _json_ready(parameters)


def _normalize_dvd_validation(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    validation = _schema_mapping(value, label=label)
    _reject_unknown_schema_fields(
        validation,
        allowed=_PROVENANCE_VALIDATION_FIELDS,
        label=label,
    )
    if "assets" in validation:
        assets = _schema_mapping(
            validation["assets"],
            label=f"{label}.assets",
        )
        unknown_assets = sorted(set(assets).difference({"checkpoint", "model_config"}))
        if unknown_assets:
            raise ValueError(
                f"{label}.assets contains unknown fields: " + ", ".join(unknown_assets)
            )
        for name, record in assets.items():
            item = _schema_mapping(
                record,
                label=f"{label}.assets.{name}",
            )
            _reject_unknown_schema_fields(
                item,
                allowed=_PROVENANCE_ASSET_FIELDS,
                label=f"{label}.assets.{name}",
            )
        validation["assets"] = assets
    return _json_ready(validation)


def normalize_depth_model_provenance(
    model_id: str,
    value: Mapping[str, Any] | None,
    *,
    label: str = "model_provenance",
) -> dict[str, Any]:
    """Normalize backend-specific reproducibility provenance."""

    provenance = _schema_mapping(value or {}, label=label)
    if not provenance:
        return {}
    model = str(model_id or "").strip().lower()
    if model == "dvd" or model.startswith("dvd_"):
        allowed = _DVD_PROVENANCE_FIELDS
    elif model == "vda" or model.startswith("vda_"):
        allowed = _VDA_PROVENANCE_FIELDS
    else:
        raise ValueError(f"{label} is unsupported for depth model {model_id!r}")
    _reject_unknown_schema_fields(
        provenance,
        allowed=allowed,
        label=label,
    )
    if model == "dvd" or model.startswith("dvd_"):
        if "uid" in provenance:
            uid = provenance["uid"]
            if not isinstance(uid, str):
                raise TypeError(f"{label}.uid must be a string")
            uid = uid.strip()
            if not uid:
                raise ValueError(f"{label}.uid must be non-empty")
            provenance["uid"] = uid
        if "asset_attestation" in provenance:
            attestation = _schema_mapping(
                provenance["asset_attestation"],
                label=f"{label}.asset_attestation",
            )
            _reject_unknown_schema_fields(
                attestation,
                allowed=_ASSET_ATTESTATION_FIELDS,
                label=f"{label}.asset_attestation",
            )
            provenance["asset_attestation"] = attestation
        if "validation" in provenance:
            provenance["validation"] = _normalize_dvd_validation(
                provenance["validation"],
                label=f"{label}.validation",
            )
    return _json_ready(provenance)


def _canonical_json_bytes(value: Any, *, pretty: bool = False) -> bytes:
    kwargs: dict[str, Any] = {
        "ensure_ascii": False,
        "allow_nan": False,
        "sort_keys": True,
    }
    if pretty:
        kwargs["indent"] = 2
        text = json.dumps(_json_ready(value), **kwargs) + "\n"
    else:
        kwargs["separators"] = (",", ":")
        text = json.dumps(_json_ready(value), **kwargs)
    return text.encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _normalize_legacy_metadata_policy(value: str) -> str:
    policy = str(value or LEGACY_METADATA_POLICY_REJECT).strip()
    if policy not in {
        LEGACY_METADATA_POLICY_REJECT,
        LEGACY_METADATA_POLICY_TRUSTED_PICKLE,
    }:
        raise ValueError(
            "legacy_metadata_policy must be 'reject' or 'trusted_pickle_read_only'"
        )
    return policy


def _open_regular_binary(path: Path, *, label: str) -> Any:
    flags = os.O_RDONLY | getattr(os, "O_NONBLOCK", 0)
    nofollow = getattr(os, "O_NOFOLLOW", 0)
    if not nofollow:
        raise RuntimeError("O_NOFOLLOW is required for depth cache reads")
    flags |= nofollow
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        if path.is_symlink():
            raise ValueError(f"{label} must not be a symlink: {path}") from exc
        raise
    try:
        file_stat = os.fstat(descriptor)
        if not stat.S_ISREG(file_stat.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        return os.fdopen(descriptor, "rb", closefd=True)
    except Exception:
        os.close(descriptor)
        raise


def _sha256_stream(stream: Any) -> str:
    digest = hashlib.sha256()
    while True:
        chunk = stream.read(1024 * 1024)
        if not chunk:
            break
        digest.update(chunk)
    stream.seek(0)
    return digest.hexdigest()


def _read_depth_file_once(path: Path) -> tuple[np.ndarray, str]:
    with _open_regular_binary(path, label="depth cache") as stream:
        file_sha256 = _sha256_stream(stream)
        depth = np.load(stream, allow_pickle=False)
        if not isinstance(depth, np.ndarray):
            raise TypeError("depth cache payload is not an ndarray")
        return np.asarray(depth), file_sha256


def _strict_json_mapping(payload: bytes, *, label: str) -> dict[str, Any]:
    def reject_constant(value: str) -> Any:
        raise ValueError(f"{label} contains non-finite constant {value}")

    def unique_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"{label} contains duplicate key {key!r}")
            result[key] = value
        return result

    parsed = json.loads(
        payload.decode("utf-8"),
        object_pairs_hook=unique_pairs,
        parse_constant=reject_constant,
    )
    if not isinstance(parsed, dict):
        raise TypeError(f"{label} payload is not a mapping")
    return parsed


def load_depth_cache_metadata(
    path: str | Path,
    *,
    legacy_metadata_policy: str = LEGACY_METADATA_POLICY_REJECT,
) -> tuple[dict[str, Any], str]:
    """Read JSON metadata safely, or an explicitly trusted legacy pickle.

    The returned format is either ``json`` or ``trusted_legacy_pickle``.
    Pickle is never attempted unless the caller selects the trusted,
    read-only compatibility policy.
    """

    metadata_path = _absolute_path(path, label="meta_path")
    _reject_symlink_chain(metadata_path, label="meta_path")
    policy = _normalize_legacy_metadata_policy(legacy_metadata_policy)
    with _open_regular_binary(metadata_path, label="depth metadata") as stream:
        payload = stream.read(_MAX_METADATA_BYTES + 1)
    if len(payload) > _MAX_METADATA_BYTES:
        raise ValueError(f"depth metadata exceeds {_MAX_METADATA_BYTES} bytes")
    if payload.lstrip().startswith(b"{"):
        return _strict_json_mapping(payload, label="depth metadata"), "json"
    if policy != LEGACY_METADATA_POLICY_TRUSTED_PICKLE:
        raise ValueError(
            "legacy pickle depth metadata requires the explicit "
            "'trusted_pickle_read_only' policy"
        )
    loaded = np.load(io.BytesIO(payload), allow_pickle=True).item()
    if not isinstance(loaded, Mapping):
        raise TypeError("legacy depth metadata payload is not a mapping")
    return dict(loaded), "trusted_legacy_pickle"


def _absolute_path(value: str | Path, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be an explicit non-empty path")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {text}")
    return Path(os.path.abspath(path))


def _reject_symlink_chain(path: Path, *, label: str) -> None:
    current = path
    while True:
        if current.is_symlink():
            raise ValueError(f"{label} must not traverse a symlink: {current}")
        parent = current.parent
        if parent == current:
            return
        current = parent


def _within_root(path: Path, *, root: Path, label: str) -> None:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must be inside output_root: {path}") from exc


def _path_state(path: Path) -> dict[str, Any]:
    if path.is_symlink():
        raise ValueError(f"publication targets must not be symlinks: {path}")
    if not path.exists():
        return {"exists": False}
    if not path.is_file():
        raise ValueError(f"publication target exists but is not a file: {path}")
    stat = path.stat()
    return {
        "exists": True,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _file_fingerprint(path: Path, *, label: str) -> dict[str, Any]:
    _reject_symlink_chain(path, label=label)
    if not path.exists():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    if not path.is_file():
        raise ValueError(f"{label} is not a file: {path}")
    stat = path.stat()
    return {
        "path": path.as_posix(),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
        "sha256": _sha256_file(path),
    }


def _fingerprint_matches(
    expected: Mapping[str, Any], actual: Mapping[str, Any]
) -> bool:
    return dict(expected) == dict(actual)


def _depth_stack(value: Any, *, label: str) -> np.ndarray:
    stack = np.asarray(value, dtype=np.float32)
    if stack.ndim == 2:
        stack = stack[None, ...]
    if stack.ndim != 3:
        raise ValueError(f"{label} must have shape [T,H,W] or [H,W], got {stack.shape}")
    if any(int(size) <= 0 for size in stack.shape):
        raise ValueError(f"{label} must not contain an empty dimension")
    return np.ascontiguousarray(stack, dtype=np.float32)


def _array_fingerprint(array: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(array)
    digest = hashlib.sha256()
    digest.update(
        _canonical_json_bytes(
            {
                "dtype": str(contiguous.dtype),
                "shape": [int(value) for value in contiguous.shape],
            }
        )
    )
    digest.update(memoryview(contiguous).cast("B"))
    return digest.hexdigest()


def depth_array_fingerprint(value: Any) -> str:
    """Return the canonical dtype+shape+bytes digest used by depth artifacts."""

    return _array_fingerprint(np.asarray(value))


def _png_dimensions(path: Path) -> tuple[int, int]:
    with _open_regular_binary(path, label="GT depth PNG") as stream:
        header = stream.read(24)
    if (
        len(header) != 24
        or header[:8] != b"\x89PNG\r\n\x1a\n"
        or header[12:16] != b"IHDR"
    ):
        raise ValueError(f"invalid PNG header: {path}")
    width = int.from_bytes(header[16:20], byteorder="big")
    height = int.from_bytes(header[20:24], byteorder="big")
    if width <= 0 or height <= 0:
        raise ValueError(f"invalid PNG dimensions: {path}")
    return width, height


def _source_reference(path: Path) -> dict[str, Any]:
    _reject_symlink_chain(
        path,
        label="rollout GT depth diagnostic",
    )
    with _open_regular_binary(
        path,
        label="rollout GT depth diagnostic",
    ) as stream:
        before = os.fstat(stream.fileno())
        file_sha256 = _sha256_stream(stream)
        after = os.fstat(stream.fileno())
    if int(before.st_size) != int(after.st_size) or int(before.st_mtime_ns) != int(
        after.st_mtime_ns
    ):
        raise RuntimeError(
            "rollout GT depth diagnostic changed while its reference was being bound"
        )
    return {
        "path": path.as_posix(),
        "size_bytes": int(after.st_size),
        "file_sha256": file_sha256,
    }


def _rollout_gt_depth_diagnostics(
    source: Path,
    *,
    depth: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    stem = source.stem
    meta_stem = stem[: -len("_metric")] if stem.endswith("_metric") else stem
    candidates = {
        "metric_mp4": source.with_suffix(".mp4"),
        "frame0_png": source.with_name(f"{stem}_frame0.png"),
        "contact_png": source.with_name(f"{stem}_contact.png"),
        "meta_json": source.with_name(f"{meta_stem}_meta.json"),
    }
    diagnostics: dict[str, Any] = {}
    issues: list[dict[str, Any]] = []
    frame_count, height, width = (
        int(depth.shape[0]),
        int(depth.shape[1]),
        int(depth.shape[2]),
    )

    meta_path = candidates["meta_json"]
    if meta_path.is_file():
        try:
            with _open_regular_binary(
                meta_path,
                label="rollout GT depth metadata",
            ) as stream:
                payload = stream.read(_MAX_METADATA_BYTES + 1)
            if len(payload) > _MAX_METADATA_BYTES:
                raise ValueError("rollout GT depth metadata exceeds size limit")
            metadata = _strict_json_mapping(
                payload,
                label="rollout GT depth metadata",
            )
            metadata_shape = list(metadata.get("frame_size", []) or [])
            metadata_frames = int(metadata.get("num_frames", -1))
            if metadata_shape != [width, height]:
                raise ValueError("frame_size does not match the GT depth array")
            if metadata_frames != frame_count:
                raise ValueError("num_frames does not match the GT depth array")
            diagnostics["meta_json"] = {
                **_source_reference(meta_path),
                "camera_name": str(metadata.get("camera_name", "") or ""),
                "fps": float(metadata.get("fps", 0.0)),
                "frame_size": metadata_shape,
                "num_frames": metadata_frames,
            }
        except (OSError, TypeError, ValueError) as exc:
            issues.append(
                {
                    "role": "meta_json",
                    "code": "invalid_metadata",
                    "detail": str(exc),
                }
            )
    else:
        issues.append(
            {
                "role": "meta_json",
                "code": "missing",
            }
        )

    mp4_path = candidates["metric_mp4"]
    if mp4_path.is_file():
        diagnostics["metric_mp4"] = _source_reference(mp4_path)
    else:
        issues.append(
            {
                "role": "metric_mp4",
                "code": "missing",
            }
        )

    expected_png_shapes = {
        "frame0_png": (width, height),
    }
    selected_frames = min(8, frame_count)
    columns = min(4, selected_frames)
    rows = 0 if columns <= 0 else (selected_frames + columns - 1) // columns
    expected_png_shapes["contact_png"] = (
        columns * width,
        rows * height,
    )
    for role in ("frame0_png", "contact_png"):
        png_path = candidates[role]
        if not png_path.is_file():
            issues.append(
                {
                    "role": role,
                    "code": "missing",
                }
            )
            continue
        try:
            actual_shape = _png_dimensions(png_path)
        except (OSError, ValueError) as exc:
            issues.append(
                {
                    "role": role,
                    "code": "invalid_png",
                    "detail": str(exc),
                }
            )
            continue
        expected_shape = expected_png_shapes[role]
        if actual_shape != expected_shape:
            issues.append(
                {
                    "role": role,
                    "code": "shape_mismatch",
                    "expected_wh": list(expected_shape),
                    "actual_wh": list(actual_shape),
                }
            )
            continue
        diagnostics[role] = {
            **_source_reference(png_path),
            "width": actual_shape[0],
            "height": actual_shape[1],
        }
    return diagnostics, issues


def load_rollout_gt_depth_reference(
    path: str | Path,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Load one explicit GT-depth file and bind the used array to its bytes."""

    source = _absolute_path(path, label="rollout_gt_depth_path")
    _reject_symlink_chain(
        source,
        label="rollout_gt_depth_path",
    )
    with _open_regular_binary(
        source,
        label="rollout GT depth",
    ) as stream:
        before = os.fstat(stream.fileno())
        file_sha256 = _sha256_stream(stream)
        loaded = np.load(stream, allow_pickle=False)
        if not isinstance(loaded, np.ndarray):
            raise TypeError("rollout GT depth payload is not an ndarray")
        after = os.fstat(stream.fileno())
    if int(before.st_size) != int(after.st_size) or int(before.st_mtime_ns) != int(
        after.st_mtime_ns
    ):
        raise RuntimeError(
            "rollout GT depth changed while its reference was being bound"
        )
    depth = _depth_stack(
        loaded,
        label="rollout GT depth",
    )
    diagnostics, diagnostic_issues = _rollout_gt_depth_diagnostics(
        source,
        depth=depth,
    )
    return depth, {
        "kind": "file",
        "path": source.as_posix(),
        "size_bytes": int(after.st_size),
        "file_sha256": file_sha256,
        "shape": [int(size) for size in depth.shape],
        "dtype": str(depth.dtype),
        "array_fingerprint": _array_fingerprint(depth),
        "diagnostics": diagnostics,
        "diagnostic_issues": diagnostic_issues,
    }


def _identity_value(
    value: Any,
    *,
    label: str,
    depth: int = 0,
    node_counter: list[int] | None = None,
) -> Any:
    """Return a compact, deterministic semantic fingerprint record."""

    if node_counter is None:
        node_counter = [0]
    node_counter[0] += 1
    if node_counter[0] > _MAX_IDENTITY_NODES:
        raise ValueError(f"{label} exceeds the identity node limit")
    if depth > _MAX_IDENTITY_DEPTH:
        raise ValueError(f"{label} exceeds the identity nesting limit")
    if isinstance(value, Mapping):
        return {
            str(key): _identity_value(
                child,
                label=f"{label}.{key}",
                depth=depth + 1,
                node_counter=node_counter,
            )
            for key, child in sorted(
                value.items(),
                key=lambda item: str(item[0]),
            )
        }
    if isinstance(value, (list, tuple)):
        return [
            _identity_value(
                child,
                label=f"{label}[{index}]",
                depth=depth + 1,
                node_counter=node_counter,
            )
            for index, child in enumerate(value)
        ]
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.dtype.hasobject:
            raise TypeError(f"{label} must not contain object arrays")
        contiguous = np.ascontiguousarray(array)
        return {
            "kind": "ndarray",
            "shape": [int(size) for size in contiguous.shape],
            "dtype": str(contiguous.dtype),
            "sha256": hashlib.sha256(memoryview(contiguous).cast("B")).hexdigest(),
        }
    if isinstance(value, np.generic):
        return _identity_value(
            value.item(),
            label=label,
            depth=depth + 1,
            node_counter=node_counter,
        )
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, bytes):
        return {
            "kind": "bytes",
            "size": len(value),
            "sha256": _sha256_bytes(value),
        }
    if isinstance(value, float):
        if not np.isfinite(value):
            raise ValueError(f"{label} must not contain NaN or infinity")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(
        f"{label} must contain only deterministic identity values, "
        f"got {type(value).__name__}"
    )


def _decoded_frames_identity(frames: Any) -> dict[str, Any]:
    try:
        frame_count = len(frames)
    except TypeError as exc:
        raise ValueError("decoded frames must be a non-empty sequence") from exc
    if int(frame_count) <= 0:
        raise ValueError("decoded frames must be a non-empty sequence")

    digest = hashlib.sha256()
    height: int | None = None
    width: int | None = None
    layouts: list[dict[str, Any]] = []
    for index in range(int(frame_count)):
        frame = np.asarray(frames[index])
        if frame.ndim < 2 or any(int(size) <= 0 for size in frame.shape):
            raise ValueError(f"decoded frame {index} must be a non-empty image array")
        if frame.dtype.hasobject:
            raise TypeError(f"decoded frame {index} must not use object dtype")
        current_height, current_width = int(frame.shape[0]), int(frame.shape[1])
        if height is None:
            height, width = current_height, current_width
        elif (current_height, current_width) != (height, width):
            raise ValueError("all decoded frames must have identical spatial shape")
        contiguous = np.ascontiguousarray(frame)
        layout = {
            "index": index,
            "shape": [int(size) for size in contiguous.shape],
            "dtype": str(contiguous.dtype),
        }
        header = _canonical_json_bytes(layout)
        digest.update(len(header).to_bytes(8, byteorder="big", signed=False))
        digest.update(header)
        digest.update(memoryview(contiguous).cast("B"))
        layouts.append(layout)

    assert height is not None and width is not None
    return {
        "algorithm": "sha256",
        "content_sha256": digest.hexdigest(),
        "shape": [int(frame_count), height, width],
        "frame_layout_fingerprint": _sha256_bytes(_canonical_json_bytes(layouts)),
    }


def build_depth_input_identity(
    *,
    frames: Any,
    decode_settings: Mapping[str, Any] | None = None,
    depth_mode: str,
    gt_depth: Any = None,
    calibration_inputs: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Bind a cache to decoded pixels and all depth-calibration inputs."""

    mode = str(depth_mode or "").strip()
    if mode not in {"estimated_model", "rollout_gt_depth"}:
        raise ValueError("depth_mode must be 'estimated_model' or 'rollout_gt_depth'")
    decoded = _decoded_frames_identity(frames)
    decode_record = _identity_value(
        dict(decode_settings or {}),
        label="decode_settings",
    )
    calibration_record = _identity_value(
        dict(calibration_inputs or {}),
        label="calibration_inputs",
    )
    payload: dict[str, Any] = {
        "schema": DEPTH_INPUT_IDENTITY_SCHEMA,
        "decoded_frames": decoded,
        "decode_settings": {
            "value": decode_record,
            "fingerprint": _sha256_bytes(_canonical_json_bytes(decode_record)),
        },
        "depth_mode": mode,
        "calibration_inputs": {
            "value": calibration_record,
            "fingerprint": _sha256_bytes(_canonical_json_bytes(calibration_record)),
        },
        "gt_depth": None,
    }
    if mode == "rollout_gt_depth":
        if gt_depth is None:
            raise ValueError("rollout_gt_depth identity requires the resolved GT depth")
        depth = _depth_stack(gt_depth, label="rollout GT depth")
        payload["gt_depth"] = {
            "shape": [int(size) for size in depth.shape],
            "dtype": str(depth.dtype),
            "array_fingerprint": _array_fingerprint(depth),
        }
    payload["identity_sha256"] = _sha256_bytes(_canonical_json_bytes(payload))
    return payload


def _normalize_input_identity(
    value: Mapping[str, Any],
    *,
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    identity = _json_ready(dict(value))
    if identity.get("schema") != DEPTH_INPUT_IDENTITY_SCHEMA:
        raise ValueError(f"{label} has an unsupported schema")
    decoded = identity.get("decoded_frames", {})
    if not isinstance(decoded, Mapping):
        raise TypeError(f"{label}.decoded_frames must be a mapping")
    shape = decoded.get("shape", [])
    if (
        not isinstance(shape, list)
        or len(shape) != 3
        or any(
            not isinstance(size, int) or isinstance(size, bool) or int(size) <= 0
            for size in shape
        )
    ):
        raise ValueError(f"{label}.decoded_frames.shape must be positive [T,H,W]")
    content_sha256 = str(decoded.get("content_sha256", "") or "")
    if len(content_sha256) != 64:
        raise ValueError(f"{label}.decoded_frames.content_sha256 must be SHA-256")
    expected_digest = str(identity.pop("identity_sha256", "") or "")
    actual_digest = _sha256_bytes(_canonical_json_bytes(identity))
    identity["identity_sha256"] = expected_digest
    if expected_digest != actual_digest:
        raise ValueError(f"{label}.identity_sha256 does not match its payload")
    return identity


def _normalize_file_fingerprint_record(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    record = _schema_mapping(value, label=label)
    _reject_unknown_schema_fields(
        record,
        allowed=_FILE_FINGERPRINT_FIELDS,
        label=label,
    )
    return _json_ready(record)


def validate_canonical_depth_cache_metadata(
    value: Mapping[str, Any],
    *,
    label: str = "depth cache metadata",
) -> dict[str, Any]:
    """Validate every nested caller-controlled current implementation metadata surface."""

    metadata = _schema_mapping(value, label=label)
    _reject_unknown_schema_fields(
        metadata,
        allowed=_CACHE_METADATA_FIELDS,
        label=label,
    )
    if metadata.get("schema") != DEPTH_CACHE_SCHEMA:
        raise ValueError(f"{label}.schema must be {DEPTH_CACHE_SCHEMA!r}")
    if metadata.get("format") != _PLAN_VERSION:
        raise ValueError(f"{label}.format must be {_PLAN_VERSION}")
    if metadata.get("metadata_format") != "json":
        raise ValueError(f"{label}.metadata_format must be 'json'")
    metadata["input_identity"] = _normalize_input_identity(
        _schema_mapping(
            metadata.get("input_identity"),
            label=f"{label}.input_identity",
        ),
        label=f"{label}.input_identity",
    )
    model_id = str(metadata.get("depth_model", "") or "").strip()
    if not model_id:
        raise ValueError(f"{label}.depth_model must be non-empty")
    metadata["calibration"] = normalize_depth_calibration_metadata(
        metadata.get("calibration"),
        label=f"{label}.calibration",
    )
    metadata["model_provenance"] = normalize_depth_model_provenance(
        model_id,
        metadata.get("model_provenance"),
        label=f"{label}.model_provenance",
    )
    expected_provenance_fingerprint = _sha256_bytes(
        _canonical_json_bytes(metadata["model_provenance"])
    )
    if (
        str(metadata.get("model_provenance_fingerprint", "") or "")
        != expected_provenance_fingerprint
    ):
        raise ValueError(f"{label}.model_provenance_fingerprint does not match")
    metadata["parameters"] = normalize_depth_publication_parameters(
        metadata.get("parameters"),
        label=f"{label}.parameters",
    )
    expected_parameter_fingerprint = _sha256_bytes(
        _canonical_json_bytes(metadata["parameters"])
    )
    if (
        str(metadata.get("parameter_fingerprint", "") or "")
        != expected_parameter_fingerprint
    ):
        raise ValueError(f"{label}.parameter_fingerprint does not match")

    depth = _schema_mapping(
        metadata.get("depth"),
        label=f"{label}.depth",
    )
    _reject_unknown_schema_fields(
        depth,
        allowed=_DEPTH_RECORD_FIELDS,
        label=f"{label}.depth",
    )
    metadata["depth"] = _json_ready(depth)

    input_fingerprints = _schema_mapping(
        metadata.get("input_fingerprints", {}),
        label=f"{label}.input_fingerprints",
    )
    metadata["input_fingerprints"] = {
        str(name): _normalize_file_fingerprint_record(
            record,
            label=f"{label}.input_fingerprints[{name!r}]",
        )
        for name, record in input_fingerprints.items()
    }

    targets = _schema_mapping(
        metadata.get("target_depths", {}),
        label=f"{label}.target_depths",
    )
    normalized_targets: dict[str, Any] = {}
    for name, raw_record in targets.items():
        record = _schema_mapping(
            raw_record,
            label=f"{label}.target_depths[{name!r}]",
        )
        _reject_unknown_schema_fields(
            record,
            allowed=_TARGET_DEPTH_RECORD_FIELDS,
            label=f"{label}.target_depths[{name!r}]",
        )
        record["calibration"] = normalize_target_depth_calibration_metadata(
            record.get("calibration"),
            label=(f"{label}.target_depths[{name!r}].calibration"),
        )
        normalized_targets[str(name)] = _json_ready(record)
    metadata["target_depths"] = normalized_targets

    media = _schema_mapping(
        metadata.get("debug_media", {}),
        label=f"{label}.debug_media",
    )
    normalized_media: dict[str, Any] = {}
    for name, raw_record in media.items():
        record = _schema_mapping(
            raw_record,
            label=f"{label}.debug_media[{name!r}]",
        )
        _reject_unknown_schema_fields(
            record,
            allowed=_DEBUG_MEDIA_RECORD_FIELDS,
            label=f"{label}.debug_media[{name!r}]",
        )
        record["source"] = _normalize_file_fingerprint_record(
            record.get("source"),
            label=f"{label}.debug_media[{name!r}].source",
        )
        normalized_media[str(name)] = _json_ready(record)
    metadata["debug_media"] = normalized_media
    return _json_ready(metadata)


def _npy_payload_info(value: Any, *, allow_pickle: bool) -> dict[str, Any]:
    writer = _HashWriter()
    np.save(writer, value, allow_pickle=allow_pickle)
    return {
        "sha256": writer.digest.hexdigest(),
        "size": int(writer.size),
    }


def _json_payload_info(value: Any) -> dict[str, Any]:
    payload = _canonical_json_bytes(value, pretty=True)
    return {"sha256": _sha256_bytes(payload), "size": len(payload)}


def _normalize_model(
    model_id: str,
    *,
    model_provenance: Mapping[str, Any] | None,
    allowed_model_ids: Collection[str] | None,
) -> dict[str, Any]:
    normalized = str(model_id or "").strip()
    if not normalized:
        raise ValueError("model_id must be non-empty")
    if allowed_model_ids is not None:
        allowed = {str(value).strip() for value in allowed_model_ids}
        if normalized not in allowed:
            raise ValueError(
                f"unknown depth model {normalized!r}; allowed={sorted(allowed)}"
            )
    provenance = normalize_depth_model_provenance(
        normalized,
        model_provenance,
    )
    return {
        "id": normalized,
        "provenance": provenance,
        "provenance_fingerprint": _sha256_bytes(_canonical_json_bytes(provenance)),
    }


def _normalize_input_files(
    input_files: Mapping[str, str | Path] | None,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw_name, raw_path in sorted(dict(input_files or {}).items()):
        name = str(raw_name).strip()
        if not name:
            raise ValueError("input file labels must be non-empty")
        if name in output:
            raise ValueError(f"duplicate input file label: {name}")
        path = _absolute_path(raw_path, label=f"input_files[{name!r}]")
        output[name] = _file_fingerprint(
            path,
            label=f"input_files[{name!r}]",
        )
    return output


def _normalize_target_specs(
    target_depths: Mapping[str, Mapping[str, Any]] | None,
    *,
    canonical_shape: tuple[int, int, int],
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw_name, raw_spec in sorted(
        dict(target_depths or {}).items(),
        key=lambda item: str(item[0]),
    ):
        name = str(raw_name).strip()
        if not name:
            raise ValueError("target depth names must be non-empty")
        if name in output:
            raise ValueError(f"duplicate target depth name: {name}")
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"target_depths[{name!r}] must be a mapping")
        spec = dict(raw_spec)
        unknown = sorted(
            set(spec).difference({"depths", "path", "source", "calibration"})
        )
        if unknown:
            raise ValueError(
                f"target_depths[{name!r}] has unsupported fields: " + ", ".join(unknown)
            )
        if "depths" not in spec:
            raise ValueError(f"target_depths[{name!r}] requires depths")
        target = _depth_stack(
            spec["depths"],
            label=f"target_depths[{name!r}].depths",
        )
        if target.shape != canonical_shape:
            if target.shape[1:] == canonical_shape[1:]:
                raise ValueError(
                    f"target depth frame-count mismatch for {name!r}: "
                    f"target={target.shape[0]} canonical={canonical_shape[0]}"
                )
            raise ValueError(
                f"target depth shape mismatch for {name!r}: "
                f"target={target.shape} canonical={canonical_shape}"
            )
        path = _absolute_path(
            spec.get("path", ""),
            label=f"target_depths[{name!r}].path",
        )
        _within_root(
            path,
            root=output_root,
            label=f"target_depths[{name!r}].path",
        )
        _reject_symlink_chain(
            path,
            label=f"target_depths[{name!r}].path",
        )
        output[name] = {
            "depths": target,
            "path": path,
            "source": str(spec.get("source", "target_calibrated") or ""),
            "calibration": normalize_target_depth_calibration_metadata(
                spec.get("calibration"),
                label=f"target_depths[{name!r}].calibration",
            ),
        }
    return output


def _normalize_debug_media(
    debug_media: Mapping[str, Mapping[str, Any]] | None,
    *,
    output_root: Path,
) -> dict[str, dict[str, Any]]:
    output: dict[str, dict[str, Any]] = {}
    for raw_name, raw_spec in sorted(
        dict(debug_media or {}).items(),
        key=lambda item: str(item[0]),
    ):
        name = str(raw_name).strip()
        if not name:
            raise ValueError("debug media names must be non-empty")
        if name in output:
            raise ValueError(f"duplicate debug media name: {name}")
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"debug_media[{name!r}] must be a mapping")
        spec = dict(raw_spec)
        unknown = sorted(
            set(spec).difference({"source_path", "destination_path", "media_type"})
        )
        if unknown:
            raise ValueError(
                f"debug_media[{name!r}] has unsupported fields: " + ", ".join(unknown)
            )
        source = _absolute_path(
            spec.get("source_path", ""),
            label=f"debug_media[{name!r}].source_path",
        )
        destination = _absolute_path(
            spec.get("destination_path", ""),
            label=f"debug_media[{name!r}].destination_path",
        )
        _within_root(
            destination,
            root=output_root,
            label=f"debug_media[{name!r}].destination_path",
        )
        _reject_symlink_chain(
            destination,
            label=f"debug_media[{name!r}].destination_path",
        )
        if source == destination:
            raise ValueError(
                f"debug_media[{name!r}] source and destination must differ"
            )
        output[name] = {
            "source_path": source,
            "destination_path": destination,
            "media_type": str(spec.get("media_type", "") or "").strip(),
            "source_fingerprint": _file_fingerprint(
                source,
                label=f"debug_media[{name!r}].source_path",
            ),
        }
    return output


def _artifact_action(
    *,
    before: Mapping[str, Any],
    payload_sha256: str,
    overwrite: bool,
) -> tuple[str, str | None]:
    if not bool(before.get("exists", False)):
        return "create", None
    if str(before.get("sha256", "")) == str(payload_sha256):
        return "reuse", None
    if overwrite:
        return "replace", None
    return "conflict", "destination exists with different content"


def _artifact_record(
    *,
    name: str,
    role: str,
    path: Path,
    payload_format: str,
    payload_info: Mapping[str, Any],
    overwrite: bool,
    target_name: str | None = None,
    media_name: str | None = None,
    source_fingerprint: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    before = _path_state(path)
    action, conflict = _artifact_action(
        before=before,
        payload_sha256=str(payload_info["sha256"]),
        overwrite=overwrite,
    )
    record: dict[str, Any] = {
        "name": name,
        "role": role,
        "path": path.as_posix(),
        "format": payload_format,
        "payload_sha256": str(payload_info["sha256"]),
        "payload_size": int(payload_info["size"]),
        "before": before,
        "action": action,
    }
    if conflict is not None:
        record["conflict"] = conflict
    if target_name is not None:
        record["target_name"] = target_name
    if media_name is not None:
        record["media_name"] = media_name
    if source_fingerprint is not None:
        record["source_fingerprint"] = dict(source_fingerprint)
    return record


def _plan_digest(plan: Mapping[str, Any]) -> str:
    payload = copy.deepcopy(dict(plan))
    payload.pop("plan_digest", None)
    return _sha256_bytes(_canonical_json_bytes(payload))


def build_depth_publication_plan(
    *,
    output_root: str | Path,
    depths: Any,
    cache_path: str | Path,
    meta_path: str | Path,
    compat_meta_path: str | Path | None = None,
    model_id: str,
    source: str,
    depth_config_source: str,
    depth_space: str,
    fps: float,
    cache_signature: str,
    input_identity: Mapping[str, Any],
    calibration: Mapping[str, Any] | None = None,
    model_provenance: Mapping[str, Any] | None = None,
    input_files: Mapping[str, str | Path] | None = None,
    parameters: Mapping[str, Any] | None = None,
    target_depths: Mapping[str, Mapping[str, Any]] | None = None,
    debug_media: Mapping[str, Mapping[str, Any]] | None = None,
    manifest_path: str | Path | None = None,
    allowed_model_ids: Collection[str] | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Build a complete zero-write publication plan.

    Existing paths are recorded as compare-and-swap preconditions.  Model
    knowledge remains caller data through ``allowed_model_ids``; this core does
    not embed a model registry.  ``compat_meta_path`` is retained as the public
    path-contract name for callers that also publish ``depth_meta.npy``.  It is
    a filename alias only: current implementation writes the same canonical JSON bytes there as at
    ``meta_path`` and never writes an object-NPY metadata payload.
    """

    root = _absolute_path(output_root, label="output_root")
    _reject_symlink_chain(root, label="output_root")
    canonical = _depth_stack(depths, label="depths")
    normalized_input_identity = _normalize_input_identity(
        input_identity,
        label="input_identity",
    )
    decoded_shape = tuple(
        int(size) for size in normalized_input_identity["decoded_frames"]["shape"]
    )
    if canonical.shape != decoded_shape:
        raise ValueError(
            "canonical depth shape must exactly match decoded [T,H,W]: "
            f"{canonical.shape} != {decoded_shape}"
        )
    cache = _absolute_path(cache_path, label="cache_path")
    metadata = _absolute_path(meta_path, label="meta_path")
    metadata_alias = (
        None
        if compat_meta_path is None
        else _absolute_path(
            compat_meta_path,
            label="compat_meta_path",
        )
    )
    publication_paths = [
        ("cache_path", cache),
        ("meta_path", metadata),
    ]
    if metadata_alias is not None:
        publication_paths.append(("compat_meta_path", metadata_alias))
    for label, path in publication_paths:
        _within_root(path, root=root, label=label)
        _reject_symlink_chain(path, label=label)
    if cache == metadata:
        raise ValueError("cache_path and meta_path must differ")
    if metadata_alias is not None and metadata_alias in {
        cache,
        metadata,
    }:
        raise ValueError("compat_meta_path must differ from cache_path and meta_path")

    manifest = (
        None
        if manifest_path is None
        else _absolute_path(manifest_path, label="manifest_path")
    )
    if manifest is not None:
        _within_root(manifest, root=root, label="manifest_path")
        _reject_symlink_chain(manifest, label="manifest_path")

    model = _normalize_model(
        model_id,
        model_provenance=model_provenance,
        allowed_model_ids=allowed_model_ids,
    )
    inputs = _normalize_input_files(input_files)
    normalized_parameters = normalize_depth_publication_parameters(parameters)
    normalized_calibration = normalize_depth_calibration_metadata(calibration)
    parameter_fingerprint = _sha256_bytes(_canonical_json_bytes(normalized_parameters))
    targets = _normalize_target_specs(
        target_depths,
        canonical_shape=canonical.shape,
        output_root=root,
    )
    media = _normalize_debug_media(debug_media, output_root=root)

    canonical_payload = _npy_payload_info(canonical, allow_pickle=False)
    artifacts: list[dict[str, Any]] = [
        _artifact_record(
            name="canonical_depth",
            role="depth_map",
            path=cache,
            payload_format="npy-float32",
            payload_info=canonical_payload,
            overwrite=overwrite,
        )
    ]

    target_metadata: dict[str, dict[str, Any]] = {}
    for name, spec in targets.items():
        target_payload = _npy_payload_info(
            spec["depths"],
            allow_pickle=False,
        )
        target_record = {
            "path": spec["path"].as_posix(),
            "source": spec["source"],
            "shape": [int(value) for value in spec["depths"].shape],
            "dtype": str(spec["depths"].dtype),
            "array_fingerprint": _array_fingerprint(spec["depths"]),
            "file_sha256": str(target_payload["sha256"]),
            "calibration": spec["calibration"],
        }
        target_metadata[name] = target_record
        artifacts.append(
            _artifact_record(
                name=f"target_depth:{name}",
                role="target_depth_map",
                path=spec["path"],
                payload_format="npy-float32",
                payload_info=target_payload,
                overwrite=overwrite,
                target_name=name,
            )
        )

    media_metadata: dict[str, dict[str, Any]] = {}
    for name, spec in media.items():
        source_fingerprint = spec["source_fingerprint"]
        content_fingerprint = {
            "sha256": source_fingerprint["sha256"],
            "size": source_fingerprint["size"],
        }
        payload_info = {
            "sha256": source_fingerprint["sha256"],
            "size": source_fingerprint["size"],
        }
        media_metadata[name] = {
            "path": spec["destination_path"].as_posix(),
            "media_type": spec["media_type"],
            "sha256": source_fingerprint["sha256"],
            "size": source_fingerprint["size"],
            # Publication sources are transaction-local staging files.  Keep
            # their full path and mtime only in the transient plan artifact
            # record below; persisted cache provenance is content-addressed so
            # an exact rerun remains byte-stable.
            "source": content_fingerprint,
        }
        artifacts.append(
            _artifact_record(
                name=f"debug_media:{name}",
                role="debug_media",
                path=spec["destination_path"],
                payload_format="file-copy",
                payload_info=payload_info,
                overwrite=overwrite,
                media_name=name,
                source_fingerprint=source_fingerprint,
            )
        )

    depth_record = {
        "shape": [int(value) for value in canonical.shape],
        "dtype": str(canonical.dtype),
        "fps": float(fps),
        "array_fingerprint": _array_fingerprint(canonical),
        "file_sha256": str(canonical_payload["sha256"]),
    }
    metadata_payload = {
        "source": str(source or ""),
        "depth_model": model["id"],
        "depth_config_source": str(depth_config_source or ""),
        "depth_space": str(depth_space or "unknown"),
        "calibration": normalized_calibration,
        "cache_path": cache.as_posix(),
        "cache_signature": str(cache_signature or ""),
        "schema": DEPTH_CACHE_SCHEMA,
        "format": _PLAN_VERSION,
        "metadata_format": "json",
        "input_identity": normalized_input_identity,
        "model_provenance": model["provenance"],
        "model_provenance_fingerprint": model["provenance_fingerprint"],
        "input_fingerprints": inputs,
        "parameter_fingerprint": parameter_fingerprint,
        "parameters": normalized_parameters,
        "depth": depth_record,
        "target_depths": target_metadata,
        "debug_media": media_metadata,
    }
    metadata_payload = validate_canonical_depth_cache_metadata(
        metadata_payload,
        label="depth publication metadata",
    )
    metadata_info = _json_payload_info(metadata_payload)
    artifacts.append(
        _artifact_record(
            name="depth_metadata",
            role="cache_metadata",
            path=metadata,
            payload_format="json",
            payload_info=metadata_info,
            overwrite=overwrite,
        )
    )
    if metadata_alias is not None:
        artifacts.append(
            _artifact_record(
                name="depth_metadata_alias",
                role="cache_metadata_alias",
                path=metadata_alias,
                payload_format="json",
                payload_info=metadata_info,
                overwrite=overwrite,
            )
        )

    manifest_payload: dict[str, Any] | None = None
    if manifest is not None:
        manifest_payload = {
            "schema": DEPTH_CACHE_SCHEMA,
            "format": _PLAN_VERSION,
            "model": model,
            "inputs": inputs,
            "parameters": normalized_parameters,
            "parameter_fingerprint": parameter_fingerprint,
            "input_identity": normalized_input_identity,
            "depth": {
                **depth_record,
                "path": cache.as_posix(),
                "meta_path": metadata.as_posix(),
                "compat_meta_path": (
                    None if metadata_alias is None else metadata_alias.as_posix()
                ),
                "source": str(source or ""),
                "depth_config_source": str(depth_config_source or ""),
                "depth_space": str(depth_space or "unknown"),
                "cache_signature": str(cache_signature or ""),
            },
            "target_depths": target_metadata,
            "debug_media": media_metadata,
        }
        manifest_info = _json_payload_info(manifest_payload)
        artifacts.append(
            _artifact_record(
                name="depth_manifest",
                role="manifest",
                path=manifest,
                payload_format="json",
                payload_info=manifest_info,
                overwrite=overwrite,
            )
        )

    paths = [str(record["path"]) for record in artifacts]
    if len(paths) != len(set(paths)):
        raise ValueError("publication artifact paths must be unique")
    conflicts = [
        {
            "name": record["name"],
            "path": record["path"],
            "reason": record["conflict"],
        }
        for record in artifacts
        if record["action"] == "conflict"
    ]
    plan: dict[str, Any] = {
        "schema": DEPTH_PUBLICATION_PLAN_SCHEMA,
        "format": _PLAN_VERSION,
        "output_root": root.as_posix(),
        "model": model,
        "inputs": inputs,
        "parameters": normalized_parameters,
        "parameter_fingerprint": parameter_fingerprint,
        "depth": depth_record,
        "target_depths": target_metadata,
        "debug_media": media_metadata,
        "metadata_payload": metadata_payload,
        "compat_meta_path": (
            None if metadata_alias is None else metadata_alias.as_posix()
        ),
        "manifest_payload": manifest_payload,
        "artifacts": artifacts,
        "overwrite": bool(overwrite),
        "status": "blocked" if conflicts else "ready",
        "conflicts": conflicts,
        "publication_default": "dry_run",
    }
    plan["plan_digest"] = _plan_digest(plan)
    return plan


def _issue(
    issues: list[dict[str, Any]],
    code: str,
    message: str,
    *,
    path: Path | None = None,
) -> None:
    record: dict[str, Any] = {"code": code, "message": message}
    if path is not None:
        record["path"] = path.as_posix()
    issues.append(record)


def _inspect_depth_cache_once(
    *,
    cache_path: str | Path,
    meta_path: str | Path,
    expected_model_id: str | None = None,
    allowed_model_ids: Collection[str] | None = None,
    expected_shape: tuple[int, int, int] | None = None,
    expected_frame_count: int | None = None,
    expected_cache_signature: str | None = None,
    expected_input_identity: Mapping[str, Any] | None = None,
    target_depth_paths: Mapping[str, str | Path] | None = None,
    debug_media_paths: Mapping[str, str | Path] | None = None,
    verify_payload_hash: bool = True,
    legacy_metadata_policy: str = LEGACY_METADATA_POLICY_REJECT,
) -> tuple[dict[str, Any], np.ndarray | None, dict[str, Any]]:
    """Open and validate cache and metadata exactly once."""

    cache = _absolute_path(cache_path, label="cache_path")
    metadata_path = _absolute_path(meta_path, label="meta_path")
    _reject_symlink_chain(cache, label="cache_path")
    _reject_symlink_chain(metadata_path, label="meta_path")
    issues: list[dict[str, Any]] = []
    cache_exists = cache.exists()
    meta_exists = metadata_path.exists()
    if cache_exists and not cache.is_file():
        _issue(issues, "cache_not_file", "depth cache is not a file", path=cache)
    if meta_exists and not metadata_path.is_file():
        _issue(
            issues,
            "metadata_not_file",
            "depth metadata is not a file",
            path=metadata_path,
        )
    if not cache_exists and not meta_exists:
        return (
            {
                "status": "missing",
                "valid": False,
                "cache_path": cache.as_posix(),
                "meta_path": metadata_path.as_posix(),
                "identity_verified": False,
                "issues": [
                    {
                        "code": "cache_missing",
                        "message": "depth cache and metadata are missing",
                    }
                ],
            },
            None,
            {},
        )
    if cache_exists != meta_exists:
        _issue(
            issues,
            "partial_cache",
            "depth cache and metadata must either both exist or both be absent",
        )

    depth: np.ndarray | None = None
    depth_file_sha256 = ""
    metadata: dict[str, Any] = {}
    metadata_format = ""
    if cache_exists and cache.is_file():
        try:
            depth, depth_file_sha256 = _read_depth_file_once(cache)
            if depth.ndim != 3:
                _issue(
                    issues,
                    "depth_shape_invalid",
                    f"depth cache must have shape [T,H,W], got {depth.shape}",
                    path=cache,
                )
        except Exception as exc:
            _issue(
                issues,
                "depth_load_failed",
                f"failed to load depth cache: {exc}",
                path=cache,
            )
    if meta_exists and metadata_path.is_file():
        try:
            metadata, metadata_format = load_depth_cache_metadata(
                metadata_path,
                legacy_metadata_policy=legacy_metadata_policy,
            )
        except Exception as exc:
            _issue(
                issues,
                "metadata_load_failed",
                f"failed to load depth metadata: {exc}",
                path=metadata_path,
            )

    model_id = str(metadata.get("depth_model", "") or "").strip()
    if expected_model_id is not None and model_id != str(expected_model_id).strip():
        _issue(
            issues,
            "model_mismatch",
            f"cached model {model_id!r} != expected {str(expected_model_id).strip()!r}",
        )
    if allowed_model_ids is not None:
        allowed = {str(value).strip() for value in allowed_model_ids}
        if model_id not in allowed:
            _issue(
                issues,
                "unknown_model",
                f"cached model {model_id!r} is not allowed",
            )
    if expected_cache_signature is not None:
        signature = str(metadata.get("cache_signature", "") or "")
        if signature != str(expected_cache_signature):
            _issue(
                issues,
                "cache_signature_mismatch",
                "cached signature does not match the expected signature",
            )

    shape: tuple[int, ...] | None = (
        None if depth is None else tuple(int(value) for value in depth.shape)
    )
    if expected_shape is not None and shape != tuple(expected_shape):
        _issue(
            issues,
            "shape_mismatch",
            f"cached shape {shape} != expected {tuple(expected_shape)}",
        )
    if (
        expected_frame_count is not None
        and shape is not None
        and shape
        and int(shape[0]) != int(expected_frame_count)
    ):
        _issue(
            issues,
            "frame_count_mismatch",
            f"cached frames {shape[0]} != expected {int(expected_frame_count)}",
        )

    cache_schema = str(metadata.get("schema", "") or "")
    format = metadata.get("format", 0)
    is_v1 = (
        metadata_format == "json"
        and cache_schema == DEPTH_CACHE_SCHEMA
        and format == _PLAN_VERSION
        and metadata.get("metadata_format") == "json"
    )
    if (
        metadata_format == "json"
        and cache_schema
        and cache_schema != DEPTH_CACHE_SCHEMA
    ):
        _issue(
            issues,
            "unsupported_schema",
            f"unsupported depth cache schema: {cache_schema!r}",
        )
    elif (
        metadata_format == "json"
        and cache_schema == DEPTH_CACHE_SCHEMA
        and format != _PLAN_VERSION
    ):
        _issue(
            issues,
            "unsupported_format",
            f"unsupported depth cache schema version: {format!r}",
        )
    elif (
        metadata_format == "json"
        and cache_schema == DEPTH_CACHE_SCHEMA
        and metadata.get("metadata_format") != "json"
    ):
        _issue(
            issues,
            "metadata_format_invalid",
            "current implementation JSON metadata must declare metadata_format='json'",
        )

    if is_v1:
        try:
            metadata = validate_canonical_depth_cache_metadata(
                metadata,
                label="metadata",
            )
        except Exception as exc:
            _issue(
                issues,
                "metadata_schema_invalid",
                f"current implementation metadata schema validation failed: {exc}",
            )

    identity_verified = False
    cached_input_identity: dict[str, Any] | None = None
    if is_v1:
        try:
            cached_input_identity = _normalize_input_identity(
                metadata.get("input_identity", {}),
                label="metadata.input_identity",
            )
        except Exception as exc:
            _issue(
                issues,
                "input_identity_invalid",
                f"cached input identity is invalid: {exc}",
            )
        if expected_input_identity is not None:
            try:
                normalized_expected_identity = _normalize_input_identity(
                    expected_input_identity,
                    label="expected_input_identity",
                )
            except Exception as exc:
                _issue(
                    issues,
                    "expected_input_identity_invalid",
                    f"expected input identity is invalid: {exc}",
                )
            else:
                if cached_input_identity == normalized_expected_identity:
                    identity_verified = True
                else:
                    _issue(
                        issues,
                        "input_identity_mismatch",
                        "cached input identity does not match decoded inputs",
                    )
    depth_meta = dict(metadata.get("depth", {}) or {})
    if is_v1 and shape is not None:
        metadata_shape = tuple(int(value) for value in depth_meta.get("shape", []))
        if metadata_shape != shape:
            _issue(
                issues,
                "metadata_shape_mismatch",
                f"metadata shape {metadata_shape} != depth shape {shape}",
            )
        if bool(verify_payload_hash):
            expected_hash = str(depth_meta.get("file_sha256", "") or "")
            if not expected_hash:
                _issue(
                    issues,
                    "depth_fingerprint_missing",
                    "current implementation metadata is missing the depth file fingerprint",
                )
            elif cache_exists and depth_file_sha256 != expected_hash:
                _issue(
                    issues,
                    "depth_fingerprint_mismatch",
                    "depth file fingerprint does not match metadata",
                    path=cache,
                )

    explicit_targets = {
        str(name): _absolute_path(path, label=f"target_depth_paths[{name!r}]")
        for name, path in sorted(dict(target_depth_paths or {}).items())
    }
    target_reports: dict[str, dict[str, Any]] = {}
    metadata_targets = dict(metadata.get("target_depths", {}) or {})
    for name, path in explicit_targets.items():
        _reject_symlink_chain(path, label=f"target_depth_paths[{name!r}]")
        report: dict[str, Any] = {
            "path": path.as_posix(),
            "valid": True,
        }
        if not path.exists():
            report["valid"] = False
            report["issue"] = "missing"
            _issue(
                issues,
                "target_depth_missing",
                f"target depth {name!r} is missing",
                path=path,
            )
        else:
            try:
                target = np.load(
                    path.as_posix(),
                    mmap_mode="r",
                    allow_pickle=False,
                )
                report["shape"] = [int(value) for value in target.shape]
                if shape is not None and tuple(target.shape) != shape:
                    report["valid"] = False
                    report["issue"] = "shape_mismatch"
                    _issue(
                        issues,
                        "target_depth_shape_mismatch",
                        f"target depth {name!r} shape {target.shape} != {shape}",
                        path=path,
                    )
                expected_hash = str(
                    dict(metadata_targets.get(name, {}) or {}).get(
                        "file_sha256",
                        "",
                    )
                    or ""
                )
                if bool(verify_payload_hash) and expected_hash:
                    if _sha256_file(path) != expected_hash:
                        report["valid"] = False
                        report["issue"] = "fingerprint_mismatch"
                        _issue(
                            issues,
                            "target_depth_fingerprint_mismatch",
                            f"target depth {name!r} fingerprint mismatch",
                            path=path,
                        )
            except Exception as exc:
                report["valid"] = False
                report["issue"] = "load_failed"
                _issue(
                    issues,
                    "target_depth_load_failed",
                    f"failed to load target depth {name!r}: {exc}",
                    path=path,
                )
        target_reports[name] = report

    explicit_media = {
        str(name): _absolute_path(path, label=f"debug_media_paths[{name!r}]")
        for name, path in sorted(dict(debug_media_paths or {}).items())
    }
    media_reports: dict[str, dict[str, Any]] = {}
    metadata_media = dict(metadata.get("debug_media", {}) or {})
    for name, path in explicit_media.items():
        _reject_symlink_chain(path, label=f"debug_media_paths[{name!r}]")
        report = {"path": path.as_posix(), "valid": True}
        if not path.exists() or not path.is_file():
            report["valid"] = False
            report["issue"] = "missing"
            _issue(
                issues,
                "debug_media_missing",
                f"debug media {name!r} is missing",
                path=path,
            )
        else:
            expected_hash = str(
                dict(metadata_media.get(name, {}) or {}).get("sha256", "") or ""
            )
            if (
                bool(verify_payload_hash)
                and expected_hash
                and _sha256_file(path) != expected_hash
            ):
                report["valid"] = False
                report["issue"] = "fingerprint_mismatch"
                _issue(
                    issues,
                    "debug_media_fingerprint_mismatch",
                    f"debug media {name!r} fingerprint mismatch",
                    path=path,
                )
        media_reports[name] = report

    if issues:
        status = (
            "partial"
            if any(issue["code"] == "partial_cache" for issue in issues)
            else "invalid"
        )
    else:
        status = "valid" if is_v1 else "legacy_valid"
    report = {
        "status": status,
        "valid": not issues,
        "cache_path": cache.as_posix(),
        "meta_path": metadata_path.as_posix(),
        "metadata_format": metadata_format or None,
        "identity_verified": bool(identity_verified),
        "schema": cache_schema or None,
        "format": format,
        "model_id": model_id,
        "shape": None if shape is None else list(shape),
        "dtype": None if depth is None else str(depth.dtype),
        "metadata": metadata,
        "targets": target_reports,
        "debug_media": media_reports,
        "issues": issues,
    }
    return report, depth, metadata


def inspect_depth_cache(
    *,
    cache_path: str | Path,
    meta_path: str | Path,
    expected_model_id: str | None = None,
    allowed_model_ids: Collection[str] | None = None,
    expected_shape: tuple[int, int, int] | None = None,
    expected_frame_count: int | None = None,
    expected_cache_signature: str | None = None,
    expected_input_identity: Mapping[str, Any] | None = None,
    target_depth_paths: Mapping[str, str | Path] | None = None,
    debug_media_paths: Mapping[str, str | Path] | None = None,
    verify_payload_hash: bool = True,
    legacy_metadata_policy: str = LEGACY_METADATA_POLICY_REJECT,
) -> dict[str, Any]:
    """Inspect explicit cache artifacts without following stored paths."""

    report, _, _ = _inspect_depth_cache_once(
        cache_path=cache_path,
        meta_path=meta_path,
        expected_model_id=expected_model_id,
        allowed_model_ids=allowed_model_ids,
        expected_shape=expected_shape,
        expected_frame_count=expected_frame_count,
        expected_cache_signature=expected_cache_signature,
        expected_input_identity=expected_input_identity,
        target_depth_paths=target_depth_paths,
        debug_media_paths=debug_media_paths,
        verify_payload_hash=verify_payload_hash,
        legacy_metadata_policy=legacy_metadata_policy,
    )
    return report


def read_validated_depth_cache(
    *,
    cache_path: str | Path,
    meta_path: str | Path,
    **inspection_options: Any,
) -> tuple[np.ndarray, dict[str, Any], dict[str, Any]]:
    """Open, validate, and return one immutable cache-file snapshot."""

    report, depth, metadata = _inspect_depth_cache_once(
        cache_path=cache_path,
        meta_path=meta_path,
        **inspection_options,
    )
    if not bool(report["valid"]):
        codes = ", ".join(issue["code"] for issue in report["issues"])
        raise DepthCacheValidationError(
            f"depth cache validation failed: {codes or report['status']}"
        )
    if report["status"] == "valid" and not bool(report["identity_verified"]):
        raise DepthCacheValidationError(
            "depth cache validation failed: input_identity_unverified"
        )
    if depth is None:
        raise DepthCacheValidationError(
            "depth cache validation failed: depth_load_failed"
        )
    return (
        np.asarray(depth, dtype=np.float32),
        metadata,
        report,
    )


def _validate_plan_shape(plan: Mapping[str, Any]) -> dict[str, Any]:
    if not isinstance(plan, Mapping):
        raise TypeError("depth publication plan must be a mapping")
    payload = copy.deepcopy(dict(plan))
    if payload.get("schema") != DEPTH_PUBLICATION_PLAN_SCHEMA:
        raise ValueError("unsupported depth publication plan schema")
    if payload.get("format") != _PLAN_VERSION:
        raise ValueError("unsupported depth publication plan version")
    if str(payload.get("plan_digest", "")) != _plan_digest(payload):
        raise ValueError("depth publication plan digest mismatch")
    if not isinstance(payload.get("artifacts"), list):
        raise TypeError("depth publication plan artifacts must be a list")
    return payload


def _current_input_fingerprint(
    record: Mapping[str, Any], *, label: str
) -> dict[str, Any]:
    path = _absolute_path(str(record.get("path", "")), label=label)
    return _file_fingerprint(path, label=label)


def _write_npy(path: Path, value: Any, *, allow_pickle: bool) -> None:
    with path.open("wb") as stream:
        np.save(stream, value, allow_pickle=allow_pickle)


def _write_json(path: Path, value: Any) -> None:
    path.write_bytes(_canonical_json_bytes(value, pretty=True))


def _normalize_supplied_arrays(
    *,
    depths: Any,
    target_depths: Mapping[str, Mapping[str, Any]] | None,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    canonical = _depth_stack(depths, label="depths")
    targets: dict[str, np.ndarray] = {}
    for raw_name, raw_spec in sorted(
        dict(target_depths or {}).items(),
        key=lambda item: str(item[0]),
    ):
        name = str(raw_name).strip()
        if name in targets:
            raise ValueError(f"duplicate target depth name: {name}")
        if not isinstance(raw_spec, Mapping) or "depths" not in raw_spec:
            raise ValueError(f"target_depths[{name!r}] must provide a depths value")
        targets[name] = _depth_stack(
            raw_spec["depths"],
            label=f"target_depths[{name!r}].depths",
        )
    return canonical, targets


def validate_depth_publication_plan(
    plan: Mapping[str, Any],
    *,
    depths: Any,
    target_depths: Mapping[str, Mapping[str, Any]] | None = None,
    debug_media: Mapping[str, Mapping[str, Any]] | None = None,
    allowed_model_ids: Collection[str] | None = None,
) -> dict[str, Any]:
    """Validate plan integrity, input CAS, output CAS, and payload identity."""

    payload = _validate_plan_shape(plan)
    root = _absolute_path(payload.get("output_root", ""), label="plan.output_root")
    _reject_symlink_chain(root, label="plan.output_root")
    model_id = str(dict(payload.get("model", {}) or {}).get("id", "") or "")
    if allowed_model_ids is not None:
        allowed = {str(value).strip() for value in allowed_model_ids}
        if model_id not in allowed:
            raise ValueError(
                f"unknown depth model {model_id!r}; allowed={sorted(allowed)}"
            )

    for name, record in dict(payload.get("inputs", {}) or {}).items():
        actual = _current_input_fingerprint(
            dict(record),
            label=f"plan.inputs[{name!r}]",
        )
        if not _fingerprint_matches(record, actual):
            raise DepthPublicationConflictError(
                f"input fingerprint changed after planning: {name}"
            )

    canonical, supplied_targets = _normalize_supplied_arrays(
        depths=depths,
        target_depths=target_depths,
    )
    planned_shape = tuple(int(value) for value in payload["depth"]["shape"])
    if canonical.shape != planned_shape:
        raise DepthPublicationConflictError(
            f"canonical depth shape changed: {canonical.shape} != {planned_shape}"
        )
    if _array_fingerprint(canonical) != str(payload["depth"]["array_fingerprint"]):
        raise DepthPublicationConflictError(
            "canonical depth payload changed after planning"
        )
    planned_targets = dict(payload.get("target_depths", {}) or {})
    if set(supplied_targets) != set(planned_targets):
        raise DepthPublicationConflictError(
            "supplied target depth names do not match the plan"
        )
    for name, target in supplied_targets.items():
        if target.shape != planned_shape:
            raise DepthPublicationConflictError(
                f"target depth shape changed for {name!r}"
            )
        if _array_fingerprint(target) != str(
            planned_targets[name]["array_fingerprint"]
        ):
            raise DepthPublicationConflictError(
                f"target depth payload changed for {name!r}"
            )

    supplied_media = dict(debug_media or {})
    planned_media = dict(payload.get("debug_media", {}) or {})
    planned_media_sources = {
        str(record.get("media_name", "") or ""): dict(
            record.get("source_fingerprint", {}) or {}
        )
        for record in list(payload.get("artifacts", []) or [])
        if (
            isinstance(record, Mapping)
            and str(record.get("role", "") or "") == "debug_media"
        )
    }
    supplied_media_names = [str(name).strip() for name in supplied_media]
    if len(supplied_media_names) != len(set(supplied_media_names)):
        raise ValueError("duplicate debug media name")
    if set(supplied_media_names) != set(planned_media):
        raise DepthPublicationConflictError(
            "supplied debug media names do not match the plan"
        )
    media_sources: dict[str, Path] = {}
    for raw_name, raw_spec in supplied_media.items():
        name = str(raw_name).strip()
        if not isinstance(raw_spec, Mapping):
            raise TypeError(f"debug_media[{name!r}] must be a mapping")
        source = _absolute_path(
            raw_spec.get("source_path", ""),
            label=f"debug_media[{name!r}].source_path",
        )
        actual = _file_fingerprint(
            source,
            label=f"debug_media[{name!r}].source_path",
        )
        expected = planned_media_sources.get(name, {})
        if not expected:
            raise DepthPublicationConflictError(
                f"plan is missing debug media source provenance: {name}"
            )
        if not _fingerprint_matches(expected, actual):
            raise DepthPublicationConflictError(
                f"debug media source changed after planning: {name}"
            )
        media_sources[name] = source

    artifact_names: set[str] = set()
    artifact_paths: set[str] = set()
    for raw_record in payload["artifacts"]:
        if not isinstance(raw_record, Mapping):
            raise TypeError("plan artifact records must be mappings")
        record = dict(raw_record)
        name = str(record.get("name", "") or "")
        if not name or name in artifact_names:
            raise ValueError("plan artifact names must be non-empty and unique")
        artifact_names.add(name)
        path = _absolute_path(record.get("path", ""), label=f"artifact {name}")
        _within_root(path, root=root, label=f"artifact {name}")
        _reject_symlink_chain(path, label=f"artifact {name}")
        if path.as_posix() in artifact_paths:
            raise ValueError("plan artifact paths must be unique")
        artifact_paths.add(path.as_posix())
        current = _path_state(path)
        if not _fingerprint_matches(
            dict(record.get("before", {}) or {}),
            current,
        ):
            raise DepthPublicationConflictError(
                f"publication target changed after planning: {path}"
            )

        role = str(record.get("role", "") or "")
        if role == "depth_map":
            info = _npy_payload_info(canonical, allow_pickle=False)
        elif role == "target_depth_map":
            target_name = str(record.get("target_name", "") or "")
            if target_name not in supplied_targets:
                raise ValueError(
                    f"plan references missing target depth {target_name!r}"
                )
            info = _npy_payload_info(
                supplied_targets[target_name],
                allow_pickle=False,
            )
        elif role == "debug_media":
            media_name = str(record.get("media_name", "") or "")
            if media_name not in media_sources:
                raise ValueError(f"plan references missing debug media {media_name!r}")
            source_fingerprint = _file_fingerprint(
                media_sources[media_name],
                label=f"debug_media[{media_name!r}].source_path",
            )
            info = {
                "sha256": source_fingerprint["sha256"],
                "size": source_fingerprint["size"],
            }
        elif role == "cache_metadata":
            info = _json_payload_info(payload["metadata_payload"])
        elif role == "cache_metadata_alias":
            info = _json_payload_info(payload["metadata_payload"])
        elif role == "manifest":
            info = _json_payload_info(payload["manifest_payload"])
        else:
            raise ValueError(f"unsupported plan artifact role: {role!r}")
        if str(info["sha256"]) != str(record.get("payload_sha256", "")) or int(
            info["size"]
        ) != int(record.get("payload_size", -1)):
            raise DepthPublicationConflictError(
                f"artifact payload changed after planning: {name}"
            )

    return {
        "plan": payload,
        "canonical": canonical,
        "targets": supplied_targets,
        "media_sources": media_sources,
    }


def _atomic_json(path: Path, value: Any) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    temporary.write_bytes(_canonical_json_bytes(value, pretty=True))
    os.replace(temporary, path)


def _commit_replace(source: Path, destination: Path) -> None:
    """Indirection used by failure-injection tests."""

    os.replace(source, destination)


def _missing_directories(path: Path) -> list[Path]:
    missing: list[Path] = []
    current = path
    while not current.exists():
        missing.append(current)
        parent = current.parent
        if parent == current:
            break
        current = parent
    return list(reversed(missing))


def _same_device(path_a: Path, path_b: Path) -> bool:
    def nearest_existing(path: Path) -> Path:
        current = path
        while not current.exists() and current.parent != current:
            current = current.parent
        return current

    return (
        nearest_existing(path_a).stat().st_dev == nearest_existing(path_b).stat().st_dev
    )


def _write_stage_artifact(
    *,
    stage_path: Path,
    record: Mapping[str, Any],
    canonical: np.ndarray,
    targets: Mapping[str, np.ndarray],
    media_sources: Mapping[str, Path],
    plan: Mapping[str, Any],
) -> None:
    role = str(record["role"])
    if role == "depth_map":
        _write_npy(stage_path, canonical, allow_pickle=False)
    elif role == "target_depth_map":
        _write_npy(
            stage_path,
            targets[str(record["target_name"])],
            allow_pickle=False,
        )
    elif role == "debug_media":
        shutil.copyfile(
            media_sources[str(record["media_name"])],
            stage_path,
        )
    elif role == "cache_metadata":
        _write_json(stage_path, plan["metadata_payload"])
    elif role == "cache_metadata_alias":
        _write_json(stage_path, plan["metadata_payload"])
    elif role == "manifest":
        _write_json(stage_path, plan["manifest_payload"])
    else:
        raise ValueError(f"unsupported plan artifact role: {role!r}")
    if stage_path.stat().st_size != int(record["payload_size"]) or _sha256_file(
        stage_path
    ) != str(record["payload_sha256"]):
        raise RuntimeError(f"staged artifact fingerprint mismatch: {record['name']}")


def _validate_transaction_record(
    record: Mapping[str, Any],
    *,
    transaction_dir: Path,
    output_root: Path,
) -> tuple[Path, Path, Path]:
    target = _absolute_path(record.get("target", ""), label="journal target")
    stage = _absolute_path(record.get("stage", ""), label="journal stage")
    backup = _absolute_path(record.get("backup", ""), label="journal backup")
    _within_root(target, root=output_root, label="journal target")
    _within_root(stage, root=transaction_dir, label="journal stage")
    _within_root(backup, root=transaction_dir, label="journal backup")
    return target, stage, backup


def _rollback_transaction(
    journal: dict[str, Any],
    *,
    transaction_dir: Path,
) -> list[str]:
    output_root = _absolute_path(
        journal.get("output_root", ""),
        label="journal output_root",
    )
    failures: list[str] = []
    for raw_record in reversed(list(journal.get("records", []))):
        record = dict(raw_record)
        try:
            target, stage, backup = _validate_transaction_record(
                record,
                transaction_dir=transaction_dir,
                output_root=output_root,
            )
            original_existed = bool(record.get("original_existed", False))
            if backup.exists():
                if target.exists():
                    if target.is_dir():
                        raise IsADirectoryError(target)
                    target.unlink()
                os.replace(backup, target)
            elif original_existed:
                if not target.exists():
                    raise FileNotFoundError(
                        f"original target and backup are both missing: {target}"
                    )
            elif target.exists() and not stage.exists():
                if target.is_dir():
                    raise IsADirectoryError(target)
                target.unlink()
        except Exception as exc:
            failures.append(str(exc))

    for raw_path in reversed(list(journal.get("created_directories", []))):
        try:
            path = _absolute_path(raw_path, label="journal created directory")
            _within_root(path, root=output_root, label="journal created directory")
            if path.exists():
                path.rmdir()
        except OSError:
            pass
        except Exception as exc:
            failures.append(str(exc))
    return failures


def recover_depth_publication(
    transaction_dir: str | Path,
    *,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Preview or roll back one explicit interrupted publication."""

    root = _absolute_path(transaction_dir, label="transaction_dir")
    _reject_symlink_chain(root, label="transaction_dir")
    journal_path = root / "journal.json"
    if not journal_path.exists():
        raise FileNotFoundError(f"depth publication journal not found: {journal_path}")
    journal = json.loads(journal_path.read_text(encoding="utf-8"))
    if not isinstance(journal, dict) or journal.get("schema") != _TRANSACTION_SCHEMA:
        raise ValueError("unsupported depth publication transaction journal")
    preview = {
        "transaction_dir": root.as_posix(),
        "status": str(journal.get("status", "") or ""),
        "records": copy.deepcopy(list(journal.get("records", []))),
        "action": "rollback",
        "dry_run": bool(dry_run),
    }
    if dry_run:
        return preview
    failures = _rollback_transaction(journal, transaction_dir=root)
    if failures:
        raise DepthPublicationRecoveryRequired(
            "depth publication rollback failed: " + "; ".join(failures),
            transaction_dir=root,
        )
    shutil.rmtree(root)
    return {
        **preview,
        "status": "rolled_back",
        "dry_run": False,
    }


def _publish_prevalidated_artifacts(
    *,
    artifacts: Collection[Mapping[str, Any]],
    output_root: Path,
    transaction_root: str | Path | None,
    plan_digest: str,
    stage_writer: Callable[[Mapping[str, Any], Path], None],
) -> dict[str, Any]:
    """Commit prevalidated records through the shared publication journal."""

    if transaction_root is None:
        raise ValueError("transaction_root is required when dry_run=False")
    transaction_parent = _absolute_path(
        transaction_root,
        label="transaction_root",
    )
    _reject_symlink_chain(transaction_parent, label="transaction_root")
    if not transaction_parent.exists() or not transaction_parent.is_dir():
        raise ValueError(
            "transaction_root must be an existing directory when publishing"
        )
    records_to_consider = [dict(record) for record in artifacts]
    for record in records_to_consider:
        destination = _absolute_path(
            record["path"],
            label=f"artifact {record['name']}",
        )
        if not _same_device(transaction_parent, destination.parent):
            raise ValueError(
                "transaction_root and all publication targets must share a filesystem"
            )

    changes = [
        record
        for record in records_to_consider
        if record["action"] in {"create", "replace"}
    ]
    if not changes:
        return {
            "status": "reused",
            "dry_run": False,
            "plan_digest": str(plan_digest),
            "published": [],
        }

    transaction_dir = transaction_parent / f".dream-exe-depth-{uuid.uuid4().hex}"
    transaction_dir.mkdir(mode=0o700)
    journal_records: list[dict[str, Any]] = []
    created_directories: list[str] = []
    seen_created: set[str] = set()
    for index, record in enumerate(changes):
        target = _absolute_path(
            record["path"],
            label=f"artifact {record['name']}",
        )
        for directory in _missing_directories(target.parent):
            text = directory.as_posix()
            if text not in seen_created:
                seen_created.add(text)
                created_directories.append(text)
        journal_records.append(
            {
                "name": record["name"],
                "target": target.as_posix(),
                "stage": (transaction_dir / f"stage-{index:04d}.bin").as_posix(),
                "backup": (transaction_dir / f"backup-{index:04d}.bin").as_posix(),
                "original_existed": bool(record["before"]["exists"]),
                "status": "pending",
            }
        )
    journal: dict[str, Any] = {
        "schema": _TRANSACTION_SCHEMA,
        "status": "preparing",
        "plan_digest": str(plan_digest),
        "output_root": output_root.as_posix(),
        "created_directories": created_directories,
        "records": journal_records,
    }
    journal_path = transaction_dir / "journal.json"
    _atomic_json(journal_path, journal)

    try:
        for raw_directory in created_directories:
            Path(raw_directory).mkdir()
        for record, journal_record in zip(
            changes,
            journal_records,
            strict=True,
        ):
            stage_path = Path(journal_record["stage"])
            stage_writer(record, stage_path)
            if stage_path.stat().st_size != int(record["payload_size"]) or _sha256_file(
                stage_path
            ) != str(record["payload_sha256"]):
                raise RuntimeError(
                    f"staged artifact fingerprint mismatch: {record['name']}"
                )
            journal_record["status"] = "staged"
            _atomic_json(journal_path, journal)

        journal["status"] = "committing"
        _atomic_json(journal_path, journal)
        for journal_record in journal_records:
            target = Path(journal_record["target"])
            stage = Path(journal_record["stage"])
            backup = Path(journal_record["backup"])
            if bool(journal_record["original_existed"]):
                _commit_replace(target, backup)
                journal_record["status"] = "backed_up"
                _atomic_json(journal_path, journal)
            _commit_replace(stage, target)
            journal_record["status"] = "committed"
            _atomic_json(journal_path, journal)
        journal["status"] = "committed"
        _atomic_json(journal_path, journal)
    except Exception:
        journal["status"] = "rolling_back"
        try:
            _atomic_json(journal_path, journal)
        except Exception:
            pass
        failures = _rollback_transaction(
            journal,
            transaction_dir=transaction_dir,
        )
        if failures:
            raise DepthPublicationRecoveryRequired(
                "automatic depth publication rollback failed: " + "; ".join(failures),
                transaction_dir=transaction_dir,
            )
        shutil.rmtree(transaction_dir)
        raise

    published = [record["target"] for record in journal_records]
    shutil.rmtree(transaction_dir)
    return {
        "status": "published",
        "dry_run": False,
        "plan_digest": str(plan_digest),
        "published": published,
    }


def publish_artifact_batch(
    *,
    output_root: str | Path,
    debug_media: Mapping[str, Mapping[str, Any]],
    transaction_root: str | Path | None = None,
    overwrite: bool = False,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Plan and transactionally copy an explicit debug-media batch.

    This is the media-only seam of the depth publication machinery.  Sources
    must already exist; destinations must remain inside ``output_root``.
    """

    root = _absolute_path(output_root, label="output_root")
    _reject_symlink_chain(root, label="output_root")
    media = _normalize_debug_media(debug_media, output_root=root)
    artifacts: list[dict[str, Any]] = []
    media_payload: dict[str, dict[str, Any]] = {}
    for name, spec in media.items():
        source_fingerprint = dict(spec["source_fingerprint"])
        payload_info = {
            "sha256": str(source_fingerprint["sha256"]),
            "size": int(source_fingerprint["size"]),
        }
        media_payload[name] = {
            "path": spec["destination_path"].as_posix(),
            "media_type": spec["media_type"],
            "source": source_fingerprint,
        }
        artifacts.append(
            _artifact_record(
                name=f"debug_media:{name}",
                role="debug_media",
                path=spec["destination_path"],
                payload_format="file-copy",
                payload_info=payload_info,
                overwrite=bool(overwrite),
                media_name=name,
                source_fingerprint=source_fingerprint,
            )
        )
    destinations = [str(record["path"]) for record in artifacts]
    if len(destinations) != len(set(destinations)):
        raise ValueError("debug media publication paths must be unique")
    conflicts = [
        {
            "name": record["name"],
            "path": record["path"],
            "reason": record["conflict"],
        }
        for record in artifacts
        if record["action"] == "conflict"
    ]
    plan = {
        "schema": "dream-exe.debug-media-publication-plan",
        "format": _PLAN_VERSION,
        "output_root": root.as_posix(),
        "debug_media": media_payload,
        "artifacts": artifacts,
        "overwrite": bool(overwrite),
        "status": "blocked" if conflicts else "ready",
        "conflicts": conflicts,
    }
    plan_digest = _plan_digest(plan)
    if dry_run:
        return {
            "status": plan["status"],
            "dry_run": True,
            "plan_digest": plan_digest,
            "actions": [
                {
                    "name": record["name"],
                    "path": record["path"],
                    "action": record["action"],
                }
                for record in artifacts
            ],
            "conflicts": copy.deepcopy(conflicts),
        }
    if plan["status"] != "ready":
        raise DepthPublicationConflictError(
            "debug media publication contains unresolved conflicts"
        )

    media_sources: dict[str, Path] = {}
    for name, spec in media.items():
        source = Path(spec["source_path"])
        actual = _file_fingerprint(
            source,
            label=f"debug_media[{name!r}].source_path",
        )
        if not _fingerprint_matches(
            dict(spec["source_fingerprint"]),
            actual,
        ):
            raise DepthPublicationConflictError(
                f"debug media source changed after planning: {name}"
            )
        media_sources[name] = source
    for record in artifacts:
        current = _path_state(Path(record["path"]))
        if not _fingerprint_matches(
            dict(record["before"]),
            current,
        ):
            raise DepthPublicationConflictError(
                f"publication target changed after planning: {record['path']}"
            )

    def stage_writer(
        record: Mapping[str, Any],
        stage_path: Path,
    ) -> None:
        shutil.copyfile(
            media_sources[str(record["media_name"])],
            stage_path,
        )

    return _publish_prevalidated_artifacts(
        artifacts=artifacts,
        output_root=root,
        transaction_root=transaction_root,
        plan_digest=plan_digest,
        stage_writer=stage_writer,
    )


def publish_depth_artifacts(
    plan: Mapping[str, Any],
    *,
    depths: Any,
    target_depths: Mapping[str, Mapping[str, Any]] | None = None,
    debug_media: Mapping[str, Mapping[str, Any]] | None = None,
    allowed_model_ids: Collection[str] | None = None,
    transaction_root: str | Path | None = None,
    dry_run: bool = True,
) -> dict[str, Any]:
    """Publish a validated plan transactionally.

    ``dry_run`` deliberately defaults to true.  A real publication additionally
    requires an explicit, existing transaction root on the output filesystem.
    """

    validated = validate_depth_publication_plan(
        plan,
        depths=depths,
        target_depths=target_depths,
        debug_media=debug_media,
        allowed_model_ids=allowed_model_ids,
    )
    payload = validated["plan"]
    if dry_run:
        return {
            "status": payload["status"],
            "dry_run": True,
            "plan_digest": payload["plan_digest"],
            "actions": [
                {
                    "name": record["name"],
                    "path": record["path"],
                    "action": record["action"],
                }
                for record in payload["artifacts"]
            ],
            "conflicts": copy.deepcopy(payload["conflicts"]),
        }
    if payload["status"] != "ready":
        raise DepthPublicationConflictError(
            "depth publication plan contains unresolved conflicts"
        )
    output_root = _absolute_path(
        payload["output_root"],
        label="plan.output_root",
    )

    def stage_writer(
        record: Mapping[str, Any],
        stage_path: Path,
    ) -> None:
        _write_stage_artifact(
            stage_path=stage_path,
            record=record,
            canonical=validated["canonical"],
            targets=validated["targets"],
            media_sources=validated["media_sources"],
            plan=payload,
        )

    return _publish_prevalidated_artifacts(
        artifacts=payload["artifacts"],
        output_root=output_root,
        transaction_root=transaction_root,
        plan_digest=str(payload["plan_digest"]),
        stage_writer=stage_writer,
    )


__all__ = [
    "DEPTH_CACHE_SCHEMA",
    "DEPTH_PUBLICATION_PLAN_SCHEMA",
    "DepthCacheValidationError",
    "DepthPublicationConflictError",
    "DepthPublicationRecoveryRequired",
    "build_depth_publication_plan",
    "depth_array_fingerprint",
    "inspect_depth_cache",
    "load_rollout_gt_depth_reference",
    "normalize_depth_calibration_metadata",
    "normalize_depth_model_provenance",
    "normalize_depth_publication_parameters",
    "normalize_target_depth_calibration_metadata",
    "publish_artifact_batch",
    "publish_depth_artifacts",
    "read_validated_depth_cache",
    "recover_depth_publication",
    "validate_canonical_depth_cache_metadata",
    "validate_depth_publication_plan",
]
