"""Explicit pose-configuration loading for the standalone runtime boundary."""

from __future__ import annotations

import copy
import json
import math
import os
import stat
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..pose.config import load_pose_config, pose_config_to_dict
from ..pose.estimation import coerce_pose_correction_matrix
from .config import validate_pipeline_config


_MAX_POSE_CONFIG_BYTES = 1_048_576
_MAX_POSE_CORRECTION_BYTES = 262_144
_MAX_JSON_DEPTH = 64
_MAX_JSON_NODES = 20_000
_ESTIMATOR_OWNED_BACKENDS = frozenset({"estimator_injected", "estimator_factory"})


def _absolute_base(value: str | Path | None) -> Path | None:
    if value is None or not str(value).strip():
        return None
    base = Path(value).expanduser()
    if not base.is_absolute():
        raise ValueError("pipeline_asset_base must be an absolute path")
    return Path(os.path.abspath(os.fspath(base)))


def _explicit_path(
    value: Any,
    *,
    label: str,
    base: Path | None,
) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must not be empty")
    candidate = Path(text).expanduser()
    if not candidate.is_absolute():
        if base is None:
            raise ValueError(
                f"relative {label} requires an explicit pipeline_asset_base "
                "or a file-backed pipeline config"
            )
        candidate = base / candidate
    return Path(os.path.abspath(os.fspath(candidate)))


def _reject_constant(value: str) -> Any:
    raise ValueError(f"non-finite JSON number is not allowed: {value}")


def _finite_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"non-finite JSON number is not allowed: {value}")
    return parsed


def _object_without_duplicates(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key is not allowed: {key!r}")
        result[key] = value
    return result


def _validate_json_bounds(payload: Any, *, label: str) -> None:
    pending: list[tuple[Any, int]] = [(payload, 0)]
    nodes = 0
    while pending:
        value, depth = pending.pop()
        nodes += 1
        if nodes > _MAX_JSON_NODES:
            raise ValueError(f"{label} exceeds the JSON node limit")
        if depth > _MAX_JSON_DEPTH:
            raise ValueError(f"{label} exceeds the JSON nesting limit")
        if isinstance(value, float) and not math.isfinite(value):
            raise ValueError(f"{label} contains a non-finite number")
        if isinstance(value, Mapping):
            pending.extend((child, depth + 1) for child in value.values())
        elif isinstance(value, list):
            pending.extend((child, depth + 1) for child in value)


def _read_bounded_json(
    path: Path,
    *,
    label: str,
    max_bytes: int,
) -> Any:
    try:
        before = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} not found: {path}") from error
    if not stat.S_ISREG(before.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")
    if before.st_size > max_bytes:
        raise ValueError(f"{label} exceeds the {max_bytes}-byte limit: {path}")

    flags = os.O_RDONLY | getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as error:
        raise ValueError(f"{label} could not be opened safely: {path}") from error
    try:
        opened = os.fstat(descriptor)
        if not stat.S_ISREG(opened.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ):
            raise ValueError(f"{label} changed before it could be opened: {path}")
        chunks: list[bytes] = []
        total = 0
        while True:
            chunk = os.read(descriptor, min(65_536, max_bytes + 1 - total))
            if not chunk:
                break
            chunks.append(chunk)
            total += len(chunk)
            if total > max_bytes:
                raise ValueError(f"{label} exceeds the {max_bytes}-byte limit: {path}")
        after = os.fstat(descriptor)
        if (
            opened.st_dev,
            opened.st_ino,
            opened.st_size,
            opened.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(f"{label} changed while it was being read: {path}")
    finally:
        os.close(descriptor)

    try:
        text = b"".join(chunks).decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError(f"{label} is not valid UTF-8: {path}") from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_object_without_duplicates,
            parse_constant=_reject_constant,
            parse_float=_finite_float,
        )
    except (json.JSONDecodeError, RecursionError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON: {path}: {error}") from error
    _validate_json_bounds(payload, label=label)
    return payload


def _read_pose_config(path: Path) -> dict[str, Any]:
    payload = _read_bounded_json(
        path,
        label="pose config",
        max_bytes=_MAX_POSE_CONFIG_BYTES,
    )
    if not isinstance(payload, dict):
        raise ValueError(f"pose config must be a JSON object: {path}")
    return payload


def runtime_pose_estimator_owns_configuration(
    runtime_config: Mapping[str, Any],
) -> bool:
    """Return whether the runtime supplies a fully configured estimator."""

    pose = runtime_config.get("pose", {})
    if not isinstance(pose, Mapping):
        return False
    backend = str(pose.get("backend", "") or "").strip().lower()
    return backend in _ESTIMATOR_OWNED_BACKENDS


def _resolve_optional_pose_path(
    value: Any,
    *,
    label: str,
    base: Path | None,
) -> str:
    if not str(value or "").strip():
        return ""
    return _explicit_path(
        value,
        label=label,
        base=base,
    ).as_posix()


def prepare_standalone_pose_pipeline(
    pipeline_config: Mapping[str, Any],
    *,
    inline_pose_config: Mapping[str, Any] | None,
    pipeline_asset_base: str | Path | None,
    estimator_owns_configuration: bool,
) -> dict[str, Any]:
    """Resolve pose file defaults without making pose code perform I/O.

    The returned mapping contains the effective pipeline, an optional
    correction payload for ``pose_options``, and bounded provenance.
    """

    if not isinstance(pipeline_config, Mapping):
        raise TypeError("pipeline_config must be an explicit mapping")
    normalized = copy.deepcopy(dict(pipeline_config))
    pose = normalized.get("pose", {})
    if not isinstance(pose, Mapping):
        raise TypeError("pipeline pose must be a mapping")
    normalized_pose = copy.deepcopy(dict(pose))
    if inline_pose_config is None:
        raw_pose: dict[str, Any] = {}
    elif isinstance(inline_pose_config, Mapping):
        raw_pose = copy.deepcopy(dict(inline_pose_config))
    else:
        raise TypeError("inline_pose_config must be a mapping")
    enabled = bool(normalized_pose.get("enabled", False))
    requested_config_path = str(
        raw_pose.get(
            "config_path",
            normalized_pose.get("config_path", ""),
        )
        or ""
    ).strip()
    pipeline_base = _absolute_base(pipeline_asset_base)
    if not enabled:
        normalized_pose["config_path"] = ""
        normalized["pose"] = normalized_pose
        return {
            "pipeline_config": normalized,
            "pose_correction_payload": None,
            "pose_correction_source": "",
            "manifest": {
                "enabled": False,
                "configuration_authority": "disabled",
                "config_path_requested": requested_config_path,
                "config_file_consumed": False,
                "pose_correction_file_consumed": False,
            },
        }

    if estimator_owns_configuration:
        normalized_pose["config_path"] = ""
        normalized["pose"] = normalized_pose
        validate_pipeline_config(
            normalized,
            source="<standalone estimator-owned pose>",
        )
        return {
            "pipeline_config": normalized,
            "pose_correction_payload": None,
            "pose_correction_source": "",
            "manifest": {
                "enabled": True,
                "configuration_authority": "runtime_pose_estimator",
                "config_path_requested": requested_config_path,
                "config_file_consumed": False,
                "pose_correction_file_consumed": False,
                "backend_identity": "caller_owned",
            },
        }

    config_payload: dict[str, Any] = {}
    config_path: Path | None = None
    pose_path_base = pipeline_base
    if requested_config_path:
        config_path = _explicit_path(
            requested_config_path,
            label="pipeline pose.config_path",
            base=pipeline_base,
        )
        config_payload = _read_pose_config(config_path)
        pose_path_base = config_path.parent

    file_defaults = {
        key: copy.deepcopy(value)
        for key, value in config_payload.items()
        if key not in {"enabled", "config_path"}
    }
    inline_overrides = {
        key: copy.deepcopy(value)
        for key, value in raw_pose.items()
        if key not in {"enabled", "config_path"}
    }
    merged = {
        **file_defaults,
        **inline_overrides,
    }
    for field in ("mesh_path", "pose_correction_path"):
        merged[field] = _resolve_optional_pose_path(
            merged.get(field, ""),
            label=f"pose.{field}",
            base=pose_path_base,
        )

    effective_pose = pose_config_to_dict(load_pose_config(merged))
    resolved_pose = {
        "enabled": True,
        **effective_pose,
        # The outer boundary has consumed the file.  Keeping this populated
        # would make the I/O-free core reject it or falsely imply discovery.
        "config_path": "",
    }
    normalized["pose"] = resolved_pose
    validate_pipeline_config(
        normalized,
        source="<standalone resolved pose>",
    )

    correction_payload: Any = None
    correction_source = ""
    correction_consumed = False
    inline_correction = effective_pose.get("pose_correction_matrix")
    if inline_correction is not None:
        coerce_pose_correction_matrix(inline_correction)
    elif effective_pose.get("pose_correction_path"):
        correction_path = Path(str(effective_pose["pose_correction_path"]))
        correction_payload = _read_bounded_json(
            correction_path,
            label="pose correction",
            max_bytes=_MAX_POSE_CORRECTION_BYTES,
        )
        if coerce_pose_correction_matrix(correction_payload) is None:
            raise ValueError(
                "pose correction does not contain a supported 4x4 matrix: "
                f"{correction_path}"
            )
        correction_source = correction_path.as_posix()
        correction_consumed = True

    return {
        "pipeline_config": normalized,
        "pose_correction_payload": correction_payload,
        "pose_correction_source": correction_source,
        "manifest": {
            "enabled": True,
            "configuration_authority": "pipeline_pose_config",
            "config_path_requested": requested_config_path,
            "config_path_resolved": (
                "" if config_path is None else config_path.as_posix()
            ),
            "config_file_consumed": config_path is not None,
            "effective_backend": str(effective_pose["backend"]),
            "mesh_path": str(effective_pose["mesh_path"]),
            "pose_correction_path": str(effective_pose["pose_correction_path"]),
            "pose_correction_file_consumed": correction_consumed,
        },
    }


__all__ = [
    "prepare_standalone_pose_pipeline",
    "runtime_pose_estimator_owns_configuration",
]
