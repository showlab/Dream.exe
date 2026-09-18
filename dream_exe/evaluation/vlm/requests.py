"""Explicit VLM requests for one benchmark evaluation stage.

This module is intentionally an outer scheduling adapter.  It validates
credential-free saved-artifact requests, dispatches injected mode callables,
and durably records their returned JSON results.  It does not import provider
SDKs, trajectory extraction, or simulator runtimes.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import secrets
import stat
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path, PureWindowsPath
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from ...artifacts.layout import (
    run_artifact_paths,
    trajectory_artifact_paths,
)
from .auxiliary import (
    DEFAULT_MAX_MEDIA_BYTES,
    load_union_trajectory,
)

VLM_REQUEST_MANIFEST_SCHEMA = "dream_exe.vlm-evaluation-requests"
VLM_REQUEST_MANIFEST_SCHEMAS = frozenset({VLM_REQUEST_MANIFEST_SCHEMA})
VLM_RESULT_MANIFEST_SCHEMA = "dream_exe.vlm-evaluation-result"
VLM_RESULT_MANIFEST_SCHEMAS = frozenset({VLM_RESULT_MANIFEST_SCHEMA})
VLM_MODES = frozenset({"video_only", "video_trajectory"})
MAX_VLM_REQUEST_MANIFEST_BYTES = 8 * 1024 * 1024
MAX_VLM_RESULT_MANIFEST_BYTES = 16 * 1024 * 1024

VLMEvaluator = Callable[..., Mapping[str, Any]]
VLMEvaluatorRegistry = Mapping[str, VLMEvaluator]

_REQUEST_ID_PATTERN = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}\Z")
_FINGERPRINT_FIELD_PATTERN = re.compile(r"[a-z][a-z0-9_.-]{0,127}\Z")
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}\Z")
_MANIFEST_FIELDS = frozenset(
    {
        "formal_state_attestation",
        "format",
        "run_identity",
        "preparation_fingerprints",
        "requests",
    }
)
_RUN_IDENTITY_FIELDS = frozenset(
    {"uid", "run_id", "run_key", "video_kind", "gen_model"}
)
_REQUEST_FIELDS = frozenset({"request_id", "mode", "evidence", "behavior", "outputs"})
_FRAMEWORK_RESULT_OUTPUT_FIELD = "result_manifest_path"
_MAX_PREPARATION_FINGERPRINTS = 64
_PREPARATION_FIELDS = frozenset(
    {
        "prompt_template",
        "score_grid",
        "source_video",
        "subject_grid",
        "trajectory_grid",
        "union_trajectory",
    }
)
_FORMAL_UPSTREAM_STAGES = (
    "video2traj",
    "exec",
    "task_success",
)
_PATH_SUFFIXES = (
    "_csv",
    "_dir",
    "_file",
    "_json",
    "_path",
    "_paths",
)
_SENSITIVE_NAMES = frozenset(
    {
        "access_key",
        "access_key_id",
        "access_token",
        "api_key",
        "apikey",
        "auth_token",
        "authorization",
        "bearer_token",
        "client_secret",
        "connection_string",
        "cookie",
        "credential",
        "credentials",
        "dsn",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "token",
    }
)


class _StrictJSONError(ValueError):
    pass


def _strict_json_object(encoded: bytes, *, label: str) -> dict[str, Any]:
    def object_pairs(
        pairs: list[tuple[str, Any]],
    ) -> dict[str, Any]:
        output: dict[str, Any] = {}
        for key, value in pairs:
            if key in output:
                raise _StrictJSONError("duplicate key")
            output[key] = value
        return output

    def reject_constant(_value: str) -> None:
        raise _StrictJSONError("non-finite number")

    try:
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=object_pairs,
            parse_constant=reject_constant,
        )
    except (
        RecursionError,
        UnicodeError,
        json.JSONDecodeError,
        _StrictJSONError,
    ) as error:
        raise ValueError(f"{label} must contain strict JSON") from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must contain a JSON object")
    return payload


def _sensitive_key(value: Any) -> bool:
    raw = str(value).strip().replace("-", "_")
    key = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", "_", raw).lower()
    segments = frozenset(part for part in key.split("_") if part)
    return (
        key in _SENSITIVE_NAMES
        or bool(
            segments.intersection({"credential", "credentials", "password", "secret"})
        )
        or key.endswith(
            (
                "_access_key",
                "_access_key_id",
                "_api_key",
                "_connection_string",
                "_cookie",
                "_dsn",
                "_password",
                "_private_key",
                "_secret",
                "_token",
            )
        )
    )


def _reject_credential_string(value: str, *, label: str) -> None:
    lowered = value.strip().lower()
    if lowered.startswith(("basic ", "bearer ")):
        raise ValueError(f"{label} cannot contain credentials")
    if "-----begin" in lowered and "private key-----" in lowered:
        raise ValueError(f"{label} cannot contain credentials")
    if "://" not in value:
        return
    parsed = urlsplit(value)
    if parsed.username is not None or parsed.password is not None:
        raise ValueError(f"{label} cannot contain URL credentials")
    if any(_sensitive_key(key) for key, _value in parse_qsl(parsed.query)):
        raise ValueError(f"{label} cannot contain URL credentials")


def _credential_free_json(value: Any, *, label: str) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError(f"{label} cannot contain non-finite numbers")
        return value
    if isinstance(value, str):
        if any(character in value for character in ("\x00", "\r", "\n")):
            raise ValueError(f"{label} cannot contain control characters")
        _reject_credential_string(value, label=label)
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, nested in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{label} keys must be strings")
            key = raw_key.strip()
            if not key or key != raw_key:
                raise ValueError(
                    f"{label} keys cannot be empty or have surrounding whitespace"
                )
            if _sensitive_key(key):
                raise ValueError(f"{label} cannot contain credentials")
            output[key] = _credential_free_json(
                nested,
                label=f"{label}.{key}",
            )
        return output
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [
            _credential_free_json(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    raise TypeError(
        f"{label} must contain only JSON values; got {type(value).__name__}"
    )


def _canonical_sha256(value: Any, *, label: str) -> str:
    normalized = _credential_free_json(value, label=label)
    encoded = json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _request_sha256(
    *,
    request_id: str,
    mode: str,
    evidence: Mapping[str, Any],
    behavior: Mapping[str, Any],
    outputs: Mapping[str, Any],
    preparation_sha256: str,
) -> str:
    """Bind every declared behavior, evidence, and output request field."""

    if _SHA256_PATTERN.fullmatch(preparation_sha256) is None:
        raise ValueError("preparation_sha256 must be a lowercase SHA-256 digest")
    payload = {
        "format": VLM_REQUEST_MANIFEST_SCHEMA,
        "preparation_sha256": preparation_sha256,
        "request_id": request_id,
        "mode": mode,
        "evidence": evidence,
        "behavior": behavior,
        "outputs": outputs,
    }
    return _canonical_sha256(
        payload,
        label="normalized VLM request",
    )


def _preparation_sha256(
    *,
    formal_state_attestation: Mapping[str, Any],
    preparation_fingerprints: Sequence[Mapping[str, Any]],
) -> str:
    """Identify the complete normalized preparation contract."""

    return _canonical_sha256(
        {
            "formal_state_attestation": formal_state_attestation,
            "preparation_fingerprints": list(preparation_fingerprints),
        },
        label="normalized VLM preparation contract",
    )


def _normalized_manifest_fingerprint(
    value: Any,
    *,
    label: str,
) -> dict[str, Any]:
    fingerprint = _strict_fields(
        value,
        expected=frozenset({"size", "sha256"}),
        label=label,
    )
    size = fingerprint["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 1:
        raise ValueError(f"{label}.size must be a positive integer")
    sha256 = _clean_text(
        fingerprint["sha256"],
        label=f"{label}.sha256",
    )
    if _SHA256_PATTERN.fullmatch(sha256) is None:
        raise ValueError(f"{label}.sha256 must be a lowercase SHA-256 digest")
    return {"size": size, "sha256": sha256}


def _strict_fields(
    value: Any,
    *,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    payload = dict(value)
    missing = sorted(expected.difference(payload))
    unknown = sorted(set(payload).difference(expected))
    if missing or unknown:
        details = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unknown:
            details.append("unsupported " + ", ".join(unknown))
        raise ValueError(f"{label} fields are invalid: {'; '.join(details)}")
    return payload


def _clean_text(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise TypeError(f"{label} must be a string")
    text = value.strip()
    if text != value:
        raise ValueError(f"{label} cannot have surrounding whitespace")
    if not allow_empty and not text:
        raise ValueError(f"{label} is required")
    if any(character in text for character in ("\x00", "\r", "\n", "\t")):
        raise ValueError(f"{label} contains a control character")
    return text


def _resolve_path(
    value: Any,
    *,
    manifest_dir: Path,
    label: str,
    require_file: bool,
) -> Path:
    text = _clean_text(value, label=label)
    if text.startswith(("~/", r"~\\")):
        raise ValueError(f"{label} cannot use a home-relative path")
    candidate = Path(text)
    if PureWindowsPath(text).is_absolute() and not candidate.is_absolute():
        raise ValueError(f"{label} cannot use a foreign absolute path")
    lexical = Path(
        os.path.abspath(
            candidate if candidate.is_absolute() else manifest_dir / candidate
        )
    )
    if require_file:
        _read_bounded_no_symlink_chain(
            lexical,
            max_bytes=DEFAULT_MAX_MEDIA_BYTES,
            label=label,
        )
    return lexical


def _read_bounded_no_symlink_chain(
    path: Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read a stable regular file without following any symlink component."""

    if not path.is_absolute() or not path.name:
        raise ValueError(f"{label} must resolve to an absolute file path")
    directory_flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current = os.open(path.anchor, directory_flags)
    file_descriptor: int | None = None
    try:
        for part in path.parts[1:-1]:
            try:
                next_descriptor = os.open(
                    part,
                    directory_flags,
                    dir_fd=current,
                )
            except FileNotFoundError as error:
                raise FileNotFoundError(
                    f"{label} does not name an existing saved artifact: {path}"
                ) from error
            except OSError as error:
                raise ValueError(
                    f"{label} path cannot contain symlinks or non-directories"
                ) from error
            os.close(current)
            current = next_descriptor
        try:
            file_descriptor = os.open(
                path.name,
                (
                    os.O_RDONLY
                    | getattr(os, "O_CLOEXEC", 0)
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                ),
                dir_fd=current,
            )
        except FileNotFoundError as error:
            raise FileNotFoundError(
                f"{label} does not name an existing saved artifact: {path}"
            ) from error
        except OSError as error:
            raise ValueError(f"{label} must name a non-symlink regular file") from error
        before = os.fstat(file_descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must name a regular file")
        if before.st_size > int(max_bytes):
            raise ValueError(f"{label} exceeds the bounded size limit")
        chunks: list[bytes] = []
        remaining = int(max_bytes) + 1
        while remaining:
            chunk = os.read(
                file_descriptor,
                min(1024 * 1024, remaining),
            )
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(file_descriptor)
        if len(payload) > int(max_bytes):
            raise ValueError(f"{label} exceeds the bounded size limit")
        if not payload:
            raise ValueError(f"{label} cannot be empty")
        if (
            before.st_dev,
            before.st_ino,
            before.st_size,
            before.st_mtime_ns,
        ) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ) or len(payload) != after.st_size:
            raise RuntimeError(f"{label} changed while it was being read")
        return payload
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(current)


def _open_directory_chain(
    path: Path,
    *,
    create: bool,
    label: str,
) -> int | None:
    """Open an absolute directory chain without following symlinks."""

    if not path.is_absolute():
        raise ValueError(f"{label} must resolve to an absolute path")
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_DIRECTORY", 0)
        | getattr(os, "O_NOFOLLOW", 0)
    )
    current = os.open(path.anchor, flags)
    try:
        for part in path.parts[1:]:
            try:
                next_descriptor = os.open(part, flags, dir_fd=current)
            except FileNotFoundError:
                if not create:
                    os.close(current)
                    return None
                try:
                    os.mkdir(part, mode=0o755, dir_fd=current)
                except FileExistsError:
                    pass
                try:
                    next_descriptor = os.open(part, flags, dir_fd=current)
                except OSError as error:
                    raise ValueError(
                        f"{label} parent path cannot contain symlinks "
                        "or non-directories"
                    ) from error
            except OSError as error:
                raise ValueError(
                    f"{label} parent path cannot contain symlinks or non-directories"
                ) from error
            os.close(current)
            current = next_descriptor
        return current
    except Exception:
        try:
            os.close(current)
        except OSError:
            pass
        raise


def _validate_output_destination(
    path: Path,
    *,
    is_directory: bool,
    label: str,
) -> None:
    target = path if is_directory else path.parent
    descriptor = _open_directory_chain(
        target,
        create=False,
        label=label,
    )
    if descriptor is None:
        return
    try:
        if is_directory:
            return
        try:
            metadata = os.stat(
                path.name,
                dir_fd=descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            return
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(f"{label} existing destination must be a regular file")
    finally:
        os.close(descriptor)


def _is_path_field(key: str) -> bool:
    lowered = key.lower()
    return lowered.endswith(_PATH_SUFFIXES)


def _resolved_path_section(
    value: Any,
    *,
    manifest_dir: Path,
    label: str,
    require_files: bool,
) -> tuple[dict[str, Any], list[tuple[str, Path, bool]]]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    if not value:
        raise ValueError(f"{label} cannot be empty")
    declared = _credential_free_json(value, label=label)
    assert isinstance(declared, dict)
    resolved = copy.deepcopy(declared)
    paths: list[tuple[str, Path, bool]] = []
    for key, raw in declared.items():
        if not _is_path_field(key):
            if label.endswith(".outputs"):
                raise ValueError(
                    f"{label}.{key} must be an explicitly named path field"
                )
            continue
        is_directory = key.lower().endswith(("_dir", "_dirs"))
        if isinstance(raw, str):
            path = _resolve_path(
                raw,
                manifest_dir=manifest_dir,
                label=f"{label}.{key}",
                require_file=require_files,
            )
            resolved[key] = path.as_posix()
            paths.append((f"{label}.{key}", path, is_directory))
            if not require_files:
                _validate_output_destination(
                    path,
                    is_directory=is_directory,
                    label=f"{label}.{key}",
                )
            continue
        if isinstance(raw, list) and raw:
            path_values: list[str] = []
            for index, item in enumerate(raw):
                path = _resolve_path(
                    item,
                    manifest_dir=manifest_dir,
                    label=f"{label}.{key}[{index}]",
                    require_file=require_files,
                )
                path_values.append(path.as_posix())
                paths.append((f"{label}.{key}[{index}]", path, is_directory))
                if not require_files:
                    _validate_output_destination(
                        path,
                        is_directory=is_directory,
                        label=f"{label}.{key}[{index}]",
                    )
            resolved[key] = path_values
            continue
        raise TypeError(f"{label}.{key} must be a path string or non-empty path list")
    if not paths:
        raise ValueError(f"{label} must declare at least one explicit path")
    return resolved, paths


def _paths_collide(
    left: tuple[str, Path, bool],
    right: tuple[str, Path, bool],
) -> bool:
    _left_label, left_path, left_is_dir = left
    _right_label, right_path, right_is_dir = right
    if left_path == right_path:
        return True
    if left_is_dir and right_path.is_relative_to(left_path):
        return True
    return bool(right_is_dir and left_path.is_relative_to(right_path))


def _normalized_protected_paths(
    value: Mapping[str, str | Path] | None,
) -> list[tuple[str, Path, bool]]:
    if value is None:
        return []
    if not isinstance(value, Mapping):
        raise TypeError("protected VLM output paths must be a mapping")
    protected: list[tuple[str, Path, bool]] = []
    for raw_label, raw_path in value.items():
        label = _clean_text(
            raw_label,
            label="protected VLM output path label",
        )
        path_text = _clean_text(
            str(raw_path or ""),
            label=f"protected VLM output path {label}",
        )
        path = Path(path_text).expanduser().absolute()
        protected.append((f"protected.{label}", path, False))
    return protected


def _output_destinations(
    paths: Sequence[tuple[str, Path, bool]],
    *,
    request_label: str,
) -> list[dict[str, Any]]:
    prefix = f"{request_label}.outputs."
    destinations: list[dict[str, Any]] = []
    for label, path, is_directory in paths:
        if is_directory:
            raise ValueError(
                f"{label} cannot be a directory output; durable evaluation "
                "outputs must name bounded files"
            )
        if not label.startswith(prefix):
            raise RuntimeError("internal VLM output label is malformed")
        field = label.removeprefix(prefix)
        owner = "framework" if field == _FRAMEWORK_RESULT_OUTPUT_FIELD else "evaluator"
        destinations.append(
            {
                "field": field,
                "owner": owner,
                "required": True,
                "path": path,
            }
        )
    destinations.sort(key=lambda item: str(item["field"]))
    return destinations


def _evidence_fingerprints(
    paths: Sequence[tuple[str, Path, bool]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    portable: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    for label, path, _is_directory in paths:
        payload = _read_bounded_no_symlink_chain(
            path,
            max_bytes=DEFAULT_MAX_MEDIA_BYTES,
            label=label,
        )
        field = (
            label.split(".evidence.", maxsplit=1)[1] if ".evidence." in label else label
        )
        fingerprint = {
            "field": field,
            "size": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
        }
        portable.append(fingerprint)
        sources.append(
            {
                **fingerprint,
                "path": path,
                "label": label,
            }
        )
    portable.sort(key=lambda item: str(item["field"]))
    sources.sort(key=lambda item: str(item["field"]))
    return portable, sources


def _formal_upstream_projection_sha256(
    encoded: bytes,
    *,
    expected_run_identity: Mapping[str, Any],
    upstream_stages: Sequence[str] = _FORMAL_UPSTREAM_STAGES,
) -> str:
    state = _strict_fields(
        _strict_json_object(
            encoded,
            label="formal pipeline state",
        ),
        expected=frozenset({"format", "run_identity", "stages"}),
        label="formal pipeline state",
    )
    state_identity = _strict_fields(
        state["run_identity"],
        expected=frozenset({"uid", "run_id", "run_key", "gen_model"}),
        label="formal pipeline state run_identity",
    )
    expected_identity = {
        field: expected_run_identity[field]
        for field in ("uid", "run_id", "run_key", "gen_model")
    }
    if state_identity != expected_identity:
        raise ValueError(
            "formal pipeline state run identity does not match the VLM request manifest"
        )
    stages = state["stages"]
    if not isinstance(stages, Mapping):
        raise TypeError("formal pipeline state stages must be a mapping")
    upstream: dict[str, Any] = {}
    for stage in upstream_stages:
        record = stages.get(stage)
        if not isinstance(record, Mapping) or record.get("status") != "completed":
            raise ValueError(f"formal pipeline state stage {stage!r} is not completed")
        upstream[stage] = copy.deepcopy(dict(record))
    return _canonical_sha256(
        {
            "format": state["format"],
            "run_identity": state_identity,
            "stages": upstream,
        },
        label="formal pipeline upstream projection",
    )


def _load_formal_state_attestation(
    value: Any,
    *,
    manifest_dir: Path,
    expected_run_identity: Mapping[str, Any],
    require_observed_file_match: bool,
) -> tuple[dict[str, Any], dict[str, Any]]:
    label = "VLM request manifest formal_state_attestation"
    raw = _strict_fields(
        _credential_free_json(value, label=label),
        expected=frozenset(
            {
                "path",
                "observed_size",
                "observed_sha256",
                "upstream_projection_sha256",
            }
        ),
        label=label,
    )
    path = _resolve_path(
        raw["path"],
        manifest_dir=manifest_dir,
        label=f"{label}.path",
        require_file=True,
    )
    observed_size = raw["observed_size"]
    if (
        isinstance(observed_size, bool)
        or not isinstance(observed_size, int)
        or observed_size < 1
    ):
        raise ValueError(f"{label}.observed_size must be a positive integer")
    observed_sha256 = _clean_text(
        raw["observed_sha256"],
        label=f"{label}.observed_sha256",
    )
    upstream_sha256 = _clean_text(
        raw["upstream_projection_sha256"],
        label=f"{label}.upstream_projection_sha256",
    )
    if (
        _SHA256_PATTERN.fullmatch(observed_sha256) is None
        or _SHA256_PATTERN.fullmatch(upstream_sha256) is None
    ):
        raise ValueError(f"{label} digests must be lowercase SHA-256 values")
    encoded = _read_bounded_no_symlink_chain(
        path,
        max_bytes=MAX_VLM_REQUEST_MANIFEST_BYTES,
        label=f"{label}.path",
    )
    if require_observed_file_match and (
        len(encoded) != observed_size
        or hashlib.sha256(encoded).hexdigest() != observed_sha256
    ):
        raise RuntimeError(
            "formal pipeline state changed before VLM manifest publication"
        )
    matched_upstream_stages: tuple[str, ...] | None = None
    for upstream_stages in (_FORMAL_UPSTREAM_STAGES,):
        try:
            actual_upstream = _formal_upstream_projection_sha256(
                encoded,
                expected_run_identity=expected_run_identity,
                upstream_stages=upstream_stages,
            )
        except ValueError:
            continue
        if actual_upstream == upstream_sha256:
            matched_upstream_stages = tuple(upstream_stages)
            break
    if matched_upstream_stages is None:
        raise RuntimeError(
            "formal pipeline upstream stages do not match the persisted "
            "VLM preparation attestation"
        )
    normalized = {
        "path": path.as_posix(),
        "observed_size": observed_size,
        "observed_sha256": observed_sha256,
        "upstream_projection_sha256": upstream_sha256,
    }
    source = {
        "path": path,
        "label": "formal pipeline state",
        "expected_run_identity": copy.deepcopy(dict(expected_run_identity)),
        "upstream_projection_sha256": upstream_sha256,
        "upstream_stages": matched_upstream_stages,
    }
    return normalized, source


def _verify_formal_state_attestation(
    source: Mapping[str, Any] | None,
) -> None:
    if source is None:
        return
    encoded = _read_bounded_no_symlink_chain(
        Path(source["path"]),
        max_bytes=MAX_VLM_REQUEST_MANIFEST_BYTES,
        label=str(source["label"]),
    )
    actual = _formal_upstream_projection_sha256(
        encoded,
        expected_run_identity=source["expected_run_identity"],
        upstream_stages=source["upstream_stages"],
    )
    if actual != source["upstream_projection_sha256"]:
        raise RuntimeError(
            "formal pipeline upstream stages changed after VLM manifest publication"
        )


def _load_preparation_fingerprints(
    value: Any,
    *,
    manifest_dir: Path,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    normalized = _credential_free_json(
        value,
        label="VLM request manifest preparation_fingerprints",
    )
    if (
        not isinstance(normalized, list)
        or not normalized
        or len(normalized) > _MAX_PREPARATION_FINGERPRINTS
    ):
        raise ValueError(
            "VLM request manifest preparation_fingerprints must be a "
            f"non-empty list of at most {_MAX_PREPARATION_FINGERPRINTS}"
        )
    fingerprints: list[dict[str, Any]] = []
    sources: list[dict[str, Any]] = []
    previous_field = ""
    seen_paths: set[Path] = set()
    for index, raw in enumerate(normalized):
        label = f"VLM request manifest preparation_fingerprints[{index}]"
        record = _strict_fields(
            raw,
            expected=frozenset({"field", "path", "size", "sha256"}),
            label=label,
        )
        field = _clean_text(record["field"], label=f"{label}.field")
        if (
            _FINGERPRINT_FIELD_PATTERN.fullmatch(field) is None
            or field <= previous_field
        ):
            raise ValueError(
                "VLM request manifest preparation fingerprint fields "
                "must be valid, unique, and strictly ordered"
            )
        previous_field = field
        path = _resolve_path(
            record["path"],
            manifest_dir=manifest_dir,
            label=f"{label}.path",
            require_file=True,
        )
        if path in seen_paths:
            raise ValueError(
                "VLM request manifest preparation fingerprint paths must be unique"
            )
        seen_paths.add(path)
        size = record["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError(f"{label}.size must be a positive integer")
        sha256 = _clean_text(
            record["sha256"],
            label=f"{label}.sha256",
        )
        if _SHA256_PATTERN.fullmatch(sha256) is None:
            raise ValueError(f"{label}.sha256 must be a lowercase SHA-256 digest")
        payload = _read_bounded_no_symlink_chain(
            path,
            max_bytes=DEFAULT_MAX_MEDIA_BYTES,
            label=f"{label}.path",
        )
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != sha256:
            raise RuntimeError(
                "VLM preparation artifact changed and does not match its "
                f"persisted fingerprint: {field}"
            )
        fingerprints.append(
            {
                "field": field,
                "path": path.as_posix(),
                "size": size,
                "sha256": sha256,
            }
        )
        sources.append(
            {
                "field": field,
                "path": path,
                "label": f"VLM preparation artifact {field}",
                "size": size,
                "sha256": sha256,
            }
        )
    if {record["field"] for record in fingerprints} != _PREPARATION_FIELDS:
        raise ValueError(
            "VLM request manifest preparation_fingerprints must contain "
            "exactly: " + ", ".join(sorted(_PREPARATION_FIELDS))
        )
    return fingerprints, sources


def _verify_evidence_fingerprints(
    sources: Sequence[Mapping[str, Any]],
) -> None:
    for source in sources:
        payload = _read_bounded_no_symlink_chain(
            Path(source["path"]),
            max_bytes=DEFAULT_MAX_MEDIA_BYTES,
            label=str(source["label"]),
        )
        if (
            len(payload) != int(source["size"])
            or hashlib.sha256(payload).hexdigest() != source["sha256"]
        ):
            raise RuntimeError(
                f"VLM evidence changed after manifest preflight: {source['field']}"
            )


def _verify_preparation_fingerprints(
    sources: Sequence[Mapping[str, Any]],
) -> None:
    for source in sources:
        payload = _read_bounded_no_symlink_chain(
            Path(source["path"]),
            max_bytes=DEFAULT_MAX_MEDIA_BYTES,
            label=str(source["label"]),
        )
        if (
            len(payload) != int(source["size"])
            or hashlib.sha256(payload).hexdigest() != source["sha256"]
        ):
            raise RuntimeError(
                "VLM preparation artifact changed after manifest "
                f"publication: {source['field']}"
            )


def _validate_preparation_request_paths(
    *,
    formal_state_attestation: Mapping[str, Any],
    preparation_fingerprints: Sequence[Mapping[str, Any]],
    requests: Sequence[Mapping[str, Any]],
    expected_formal_run_root: Path,
) -> None:
    by_request = {str(request["request_id"]): request for request in requests}
    expected_modes = {
        "subject-stability": "video_only",
        "physical-plausibility": "video_only",
        "task-adherence": "video_only",
        "video-trajectory": "video_trajectory",
    }
    if set(by_request) != set(expected_modes):
        raise ValueError(
            "VLM request manifest must contain the exact evaluation "
            "single-UID request set"
        )
    if any(
        by_request[request_id]["mode"] != mode
        for request_id, mode in expected_modes.items()
    ):
        raise ValueError(
            "VLM request modes do not match their required request roles"
        )

    def evidence_path(request_id: str, field: str) -> Path:
        value = by_request[request_id]["resolved_evidence"].get(field)
        if not isinstance(value, str):
            raise ValueError(
                f"VLM request {request_id!r} must declare evidence.{field}"
            )
        return Path(value)

    def evidence_fingerprint(
        request_id: str,
        field: str,
    ) -> dict[str, Any]:
        matches = [
            record
            for record in by_request[request_id]["evidence_fingerprints"]
            if record["field"] == field
        ]
        if len(matches) != 1:
            raise ValueError(
                f"VLM request {request_id!r} must fingerprint "
                f"evidence.{field} exactly once"
            )
        return {
            "size": matches[0]["size"],
            "sha256": matches[0]["sha256"],
        }

    source_paths = {
        evidence_path(request_id, "source_video_path") for request_id in expected_modes
    }
    if len(source_paths) != 1:
        raise ValueError("VLM requests must share one source_video_path")
    expected_paths = {
        "prompt_template": evidence_path(
            "video-trajectory",
            "prompt_template_path",
        ),
        "score_grid": evidence_path(
            "physical-plausibility",
            "media_path",
        ),
        "source_video": next(iter(source_paths)),
        "subject_grid": evidence_path(
            "subject-stability",
            "media_path",
        ),
        "trajectory_grid": evidence_path(
            "video-trajectory",
            "media_path",
        ),
        "union_trajectory": evidence_path(
            "video-trajectory",
            "trajectory_path",
        ),
    }
    if evidence_path("task-adherence", "media_path") != expected_paths["score_grid"]:
        raise ValueError(
            "physical-plausibility and task-adherence must share score_grid"
        )
    actual_paths = {
        str(record["field"]): Path(record["path"])
        for record in preparation_fingerprints
    }
    if actual_paths != expected_paths:
        raise ValueError(
            "preparation fingerprint paths do not match request evidence"
        )
    expected_fingerprints = {
        "prompt_template": evidence_fingerprint(
            "video-trajectory",
            "prompt_template_path",
        ),
        "score_grid": evidence_fingerprint(
            "physical-plausibility",
            "media_path",
        ),
        "source_video": evidence_fingerprint(
            "subject-stability",
            "source_video_path",
        ),
        "subject_grid": evidence_fingerprint(
            "subject-stability",
            "media_path",
        ),
        "trajectory_grid": evidence_fingerprint(
            "video-trajectory",
            "media_path",
        ),
        "union_trajectory": evidence_fingerprint(
            "video-trajectory",
            "trajectory_path",
        ),
    }
    for request_id in expected_modes:
        if (
            evidence_fingerprint(
                request_id,
                "source_video_path",
            )
            != expected_fingerprints["source_video"]
        ):
            raise ValueError("VLM source-video fingerprints do not agree")
    if (
        evidence_fingerprint(
            "task-adherence",
            "media_path",
        )
        != expected_fingerprints["score_grid"]
    ):
        raise ValueError("VLM score-grid fingerprints do not agree")
    actual_fingerprints = {
        str(record["field"]): {
            "size": record["size"],
            "sha256": record["sha256"],
        }
        for record in preparation_fingerprints
    }
    if actual_fingerprints != expected_fingerprints:
        raise ValueError("preparation fingerprints do not match request evidence")
    union_path = expected_paths["union_trajectory"]
    state_path = Path(str(formal_state_attestation["path"]))
    run_paths = run_artifact_paths(expected_formal_run_root)
    if state_path != run_paths["pipeline_state"]:
        raise ValueError(
            "formal_state_attestation does not name the selected "
            "formal run pipeline_state.json"
        )
    trajectory_paths = trajectory_artifact_paths(run_paths["traj_dir"])
    if union_path != trajectory_paths["union_traj"]:
        raise ValueError("union trajectory is not in the formal trajectory location")


def _validated_evaluators(
    evaluators: VLMEvaluatorRegistry,
    *,
    requested_modes: set[str],
) -> dict[str, VLMEvaluator]:
    if not isinstance(evaluators, Mapping):
        raise TypeError("vlm_evaluators must be a mode-to-callable mapping")
    normalized: dict[str, VLMEvaluator] = {}
    for raw_mode, evaluator in evaluators.items():
        mode = _clean_text(raw_mode, label="vlm_evaluators mode")
        if mode not in VLM_MODES:
            raise ValueError(f"unsupported VLM evaluator mode: {mode}")
        if not callable(evaluator):
            raise TypeError(f"VLM evaluator for {mode} must be callable")
        normalized[mode] = evaluator
    missing = sorted(requested_modes.difference(normalized))
    if missing:
        raise ValueError(
            "no VLM evaluator is bound for requested mode(s): " + ", ".join(missing)
        )
    return normalized


def load_vlm_request_manifest(
    manifest_path: str | Path,
    *,
    expected_run_identity: Mapping[str, Any],
    protected_output_paths: Mapping[str, str | Path] | None = None,
    durable_run_root: str | Path | None = None,
    expected_formal_run_root: str | Path | None = None,
    require_observed_formal_state: bool = False,
) -> dict[str, Any]:
    """Load and fully preflight one explicit VLM request manifest."""

    path_text = _clean_text(
        str(manifest_path or ""),
        label="vlm_request_manifest_path",
    )
    path = Path(path_text).expanduser().absolute()
    try:
        encoded = _read_bounded_no_symlink_chain(
            path,
            max_bytes=MAX_VLM_REQUEST_MANIFEST_BYTES,
            label="VLM request manifest",
        )
        payload = _strict_json_object(
            encoded,
            label="VLM request manifest",
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "VLM request manifest must be a bounded regular strict-JSON file"
        ) from error
    format = _clean_text(
        payload.get("format"),
        label="VLM request manifest format",
    )
    if format != VLM_REQUEST_MANIFEST_SCHEMA:
        raise ValueError(
            f"unsupported VLM request manifest format: {format!r}"
        )
    manifest = _strict_fields(
        payload,
        expected=_MANIFEST_FIELDS,
        label="VLM request manifest",
    )

    raw_identity = _strict_fields(
        manifest["run_identity"],
        expected=_RUN_IDENTITY_FIELDS,
        label="VLM request manifest run_identity",
    )
    expected_identity = _strict_fields(
        expected_run_identity,
        expected=_RUN_IDENTITY_FIELDS,
        label="expected VLM run identity",
    )
    identity = {
        field: _clean_text(
            raw_identity[field],
            label=f"VLM request manifest run_identity.{field}",
            allow_empty=(field == "gen_model"),
        )
        for field in sorted(_RUN_IDENTITY_FIELDS)
    }
    normalized_expected = {
        field: _clean_text(
            expected_identity[field],
            label=f"expected VLM run identity.{field}",
            allow_empty=(field == "gen_model"),
        )
        for field in sorted(_RUN_IDENTITY_FIELDS)
    }
    if identity != normalized_expected:
        raise ValueError("VLM request manifest run identity does not match the run")

    raw_requests = manifest["requests"]
    if not isinstance(raw_requests, list) or not raw_requests:
        raise ValueError("VLM request manifest requests must be a non-empty list")
    manifest_dir = path.parent
    protected_paths = _normalized_protected_paths(protected_output_paths)
    protected_paths.append(("VLM request manifest", path, False))
    normalized_run_root: Path | None = None
    if durable_run_root is not None:
        run_root_text = _clean_text(
            str(durable_run_root or ""),
            label="durable VLM run root",
        )
        normalized_run_root = Path(run_root_text).expanduser().absolute()
    normalized_formal_run_root: Path | None = None
    if expected_formal_run_root is not None:
        formal_run_root_text = _clean_text(
            str(expected_formal_run_root or ""),
            label="expected formal VLM run root",
        )
        normalized_formal_run_root = (
            Path(formal_run_root_text).expanduser().absolute()
        )
    if normalized_formal_run_root is None:
        raise ValueError("VLM request manifest requires expected_formal_run_root")
    (
        formal_state_attestation,
        formal_state_attestation_source,
    ) = _load_formal_state_attestation(
        manifest["formal_state_attestation"],
        manifest_dir=manifest_dir,
        expected_run_identity=identity,
        require_observed_file_match=require_observed_formal_state,
    )
    (
        preparation_fingerprints,
        preparation_fingerprint_sources,
    ) = _load_preparation_fingerprints(
        manifest["preparation_fingerprints"],
        manifest_dir=manifest_dir,
    )
    preparation_sha256 = _preparation_sha256(
        formal_state_attestation=formal_state_attestation,
        preparation_fingerprints=preparation_fingerprints,
    )
    request_ids: set[str] = set()
    output_paths: list[tuple[str, Path, bool]] = []
    evidence_paths: list[tuple[str, Path, bool]] = []
    requests: list[dict[str, Any]] = []
    for index, raw_request in enumerate(raw_requests):
        label = f"VLM request manifest requests[{index}]"
        request = _strict_fields(
            raw_request,
            expected=_REQUEST_FIELDS,
            label=label,
        )
        request_id = _clean_text(
            request["request_id"],
            label=f"{label}.request_id",
        )
        if _REQUEST_ID_PATTERN.fullmatch(request_id) is None:
            raise ValueError(
                f"{label}.request_id must match [A-Za-z0-9][A-Za-z0-9._-]{{0,127}}"
            )
        if request_id in request_ids:
            raise ValueError(f"duplicate VLM request_id: {request_id}")
        request_ids.add(request_id)

        mode = _clean_text(request["mode"], label=f"{label}.mode")
        if mode not in VLM_MODES:
            raise ValueError(f"{label}.mode is unsupported: {mode}")
        declared_evidence = _credential_free_json(
            request["evidence"],
            label=f"{label}.evidence",
        )
        resolved_evidence, request_evidence_paths = _resolved_path_section(
            declared_evidence,
            manifest_dir=manifest_dir,
            label=f"{label}.evidence",
            require_files=True,
        )
        media_path = resolved_evidence.get("media_path")
        if not isinstance(media_path, str):
            raise ValueError(f"{label}.evidence.media_path is required")
        trajectory_path = resolved_evidence.get("trajectory_path")
        if mode == "video_trajectory" and not isinstance(trajectory_path, str):
            raise ValueError(
                f"{label}.evidence.trajectory_path is required for video_trajectory"
            )
        if mode == "video_trajectory":
            task_metadata = resolved_evidence.get("task_metadata")
            if not isinstance(task_metadata, dict) or not task_metadata:
                raise ValueError(
                    f"{label}.evidence.task_metadata must be a non-empty "
                    "mapping for video_trajectory"
                )
        if mode == "video_only" and "trajectory_path" in resolved_evidence:
            raise ValueError(
                f"{label}.evidence.trajectory_path is not allowed for video_only"
            )

        behavior = _credential_free_json(
            request["behavior"],
            label=f"{label}.behavior",
        )
        if not isinstance(behavior, dict) or not behavior:
            raise ValueError(f"{label}.behavior must be a non-empty mapping")
        behavior_version = behavior.get("behavior_version")
        _clean_text(
            behavior_version,
            label=f"{label}.behavior.behavior_version",
        )
        hidden_behavior_paths = sorted(key for key in behavior if _is_path_field(key))
        if hidden_behavior_paths:
            raise ValueError(
                f"{label}.behavior paths must be declared under evidence or "
                "outputs: " + ", ".join(hidden_behavior_paths)
            )

        declared_outputs = _credential_free_json(
            request["outputs"],
            label=f"{label}.outputs",
        )
        resolved_outputs, request_output_paths = _resolved_path_section(
            declared_outputs,
            manifest_dir=manifest_dir,
            label=f"{label}.outputs",
            require_files=False,
        )
        result_manifest_path = resolved_outputs.get("result_manifest_path")
        if not isinstance(result_manifest_path, str):
            raise ValueError(f"{label}.outputs.result_manifest_path is required")
        destinations = _output_destinations(
            request_output_paths,
            request_label=label,
        )
        (
            evidence_fingerprints,
            evidence_fingerprint_sources,
        ) = _evidence_fingerprints(request_evidence_paths)
        if mode == "video_trajectory":
            assert isinstance(trajectory_path, str)
            load_union_trajectory(
                trajectory_path,
                expected_uid=identity["uid"],
            )
            _verify_evidence_fingerprints(evidence_fingerprint_sources)
        requests.append(
            {
                "request_id": request_id,
                "mode": mode,
                "evidence": copy.deepcopy(declared_evidence),
                "resolved_evidence": resolved_evidence,
                "behavior": behavior,
                "outputs": copy.deepcopy(declared_outputs),
                "resolved_outputs": resolved_outputs,
                "result_manifest_path": result_manifest_path,
                "request_sha256": _request_sha256(
                    request_id=request_id,
                    mode=mode,
                    evidence=declared_evidence,
                    behavior=behavior,
                    outputs=declared_outputs,
                    preparation_sha256=preparation_sha256,
                ),
                "output_destinations": destinations,
                "evidence_fingerprints": evidence_fingerprints,
                "evidence_fingerprint_sources": evidence_fingerprint_sources,
            }
        )
        evidence_paths.extend(request_evidence_paths)
        output_paths.extend(request_output_paths)

    _validate_preparation_request_paths(
        formal_state_attestation=formal_state_attestation,
        preparation_fingerprints=preparation_fingerprints,
        requests=requests,
        expected_formal_run_root=normalized_formal_run_root,
    )

    for index, output_path in enumerate(output_paths):
        for other in output_paths[index + 1 :]:
            if _paths_collide(output_path, other):
                raise ValueError(
                    "VLM requests have colliding outputs: "
                    f"{output_path[0]} and {other[0]}"
                )
        for evidence_path in evidence_paths:
            if _paths_collide(evidence_path, output_path):
                raise ValueError(
                    "VLM output collides with saved evidence: "
                    f"{evidence_path[0]} and {output_path[0]}"
                )
        for protected_path in protected_paths:
            if _paths_collide(output_path, protected_path):
                raise ValueError(
                    "VLM output collides with a protected saved artifact: "
                    f"{output_path[0]} and {protected_path[0]}"
                )
    if normalized_run_root is not None:
        for index, request in enumerate(requests):
            result_manifest = Path(request["result_manifest_path"]).absolute()
            if not result_manifest.is_relative_to(normalized_run_root):
                raise ValueError(
                    "VLM request manifest requests"
                    f"[{index}].outputs.result_manifest_path must be "
                    "inside the explicit durable run root"
                )

    resolved_manifest_path = path.resolve(strict=True)
    manifest_fingerprint = {
        "size": len(encoded),
        "sha256": hashlib.sha256(encoded).hexdigest(),
    }
    _verify_formal_state_attestation(formal_state_attestation_source)
    _verify_preparation_fingerprints(preparation_fingerprint_sources)
    for request in requests:
        _verify_evidence_fingerprints(request["evidence_fingerprint_sources"])
    _verify_request_manifest_fingerprint(
        {"path": resolved_manifest_path},
        expected=manifest_fingerprint,
    )
    return {
        "format": format,
        "path": resolved_manifest_path,
        "manifest_fingerprint": manifest_fingerprint,
        "run_identity": identity,
        "formal_state_attestation": formal_state_attestation,
        "formal_state_attestation_source": (formal_state_attestation_source),
        "preparation_fingerprints": preparation_fingerprints,
        "preparation_fingerprint_sources": (preparation_fingerprint_sources),
        "preparation_sha256": preparation_sha256,
        "requests": requests,
    }


def load_vlm_result_manifest(
    manifest_path: str | Path,
    *,
    expected_run_identity: Mapping[str, Any],
    expected_request_id: str | None = None,
    expected_mode: str | None = None,
    expected_evidence_fingerprints: Sequence[Mapping[str, Any]] | None = None,
    expected_request_sha256: str | None = None,
    expected_request_manifest_fingerprint: Mapping[str, Any] | None = None,
    expected_preparation_sha256: str | None = None,
) -> dict[str, Any]:
    """Load and strictly validate one durable VLM request result."""

    path_text = _clean_text(
        str(manifest_path or ""),
        label="vlm_result_manifest_path",
    )
    path = Path(path_text).expanduser().absolute()
    try:
        encoded = _read_bounded_no_symlink_chain(
            path,
            max_bytes=MAX_VLM_RESULT_MANIFEST_BYTES,
            label="VLM result manifest",
        )
        payload = _strict_json_object(
            encoded,
            label="VLM result manifest",
        )
    except (OSError, TypeError, ValueError) as error:
        raise ValueError(
            "VLM result manifest must be a bounded regular strict-JSON file"
        ) from error

    format = _clean_text(
        payload.get("format"),
        label="VLM result manifest format",
    )
    if format not in VLM_RESULT_MANIFEST_SCHEMAS:
        raise ValueError(
            f"unsupported VLM result manifest format: {format!r}"
        )
    mode = _clean_text(
        payload.get("mode"),
        label="VLM result manifest mode",
    )
    expected_fields = {
        "format",
        "request_manifest_format",
        "request_id",
        "mode",
        "run_identity",
        "evidence_fingerprints",
        "request",
        "result",
        "request_manifest_fingerprint",
        "preparation_sha256",
    }
    if mode == "video_trajectory":
        expected_fields.update({"paper_metric_compatibility", "metric_claims"})
    manifest = _strict_fields(
        payload,
        expected=frozenset(expected_fields),
        label="VLM result manifest",
    )
    request_manifest_format = _clean_text(
        manifest["request_manifest_format"],
        label="VLM result manifest request_manifest_format",
    )
    if request_manifest_format != VLM_REQUEST_MANIFEST_SCHEMA:
        raise ValueError(
            "VLM result manifest schema version does not match its "
            "request manifest schema version"
        )
    if mode not in VLM_MODES:
        raise ValueError(f"unsupported VLM result manifest mode: {mode}")

    request_id = _clean_text(
        manifest["request_id"],
        label="VLM result manifest request_id",
    )
    if _REQUEST_ID_PATTERN.fullmatch(request_id) is None:
        raise ValueError("VLM result manifest request_id is invalid")
    if expected_request_id is not None and request_id != _clean_text(
        expected_request_id,
        label="expected VLM request_id",
    ):
        raise ValueError("VLM result manifest request_id does not match")
    if expected_mode is not None and mode != _clean_text(
        expected_mode,
        label="expected VLM mode",
    ):
        raise ValueError("VLM result manifest mode does not match")

    raw_identity = _strict_fields(
        manifest["run_identity"],
        expected=_RUN_IDENTITY_FIELDS,
        label="VLM result manifest run_identity",
    )
    expected_identity = _strict_fields(
        expected_run_identity,
        expected=_RUN_IDENTITY_FIELDS,
        label="expected VLM run identity",
    )
    identity = {
        field: _clean_text(
            raw_identity[field],
            label=f"VLM result manifest run_identity.{field}",
            allow_empty=(field == "gen_model"),
        )
        for field in sorted(_RUN_IDENTITY_FIELDS)
    }
    normalized_expected = {
        field: _clean_text(
            expected_identity[field],
            label=f"expected VLM run identity.{field}",
            allow_empty=(field == "gen_model"),
        )
        for field in sorted(_RUN_IDENTITY_FIELDS)
    }
    if identity != normalized_expected:
        raise ValueError("VLM result manifest run identity does not match")

    request_manifest_fingerprint = _normalized_manifest_fingerprint(
        manifest["request_manifest_fingerprint"],
        label="VLM result manifest request_manifest_fingerprint",
    )
    preparation_sha256 = _clean_text(
        manifest["preparation_sha256"],
        label="VLM result manifest preparation_sha256",
    )
    if _SHA256_PATTERN.fullmatch(preparation_sha256) is None:
        raise ValueError(
            "VLM result manifest preparation_sha256 must be a "
            "lowercase SHA-256 digest"
        )
    if expected_request_manifest_fingerprint is not None:
        normalized_manifest_fingerprint = _normalized_manifest_fingerprint(
            expected_request_manifest_fingerprint,
            label="expected VLM request manifest fingerprint",
        )
        if request_manifest_fingerprint != normalized_manifest_fingerprint:
            raise ValueError(
                "VLM result manifest request manifest fingerprint does not match"
            )
    if expected_preparation_sha256 is not None:
        normalized_preparation_sha256 = _clean_text(
            expected_preparation_sha256,
            label="expected VLM preparation_sha256",
        )
        if (
            _SHA256_PATTERN.fullmatch(normalized_preparation_sha256) is None
            or preparation_sha256 != normalized_preparation_sha256
        ):
            raise ValueError("VLM result manifest preparation_sha256 does not match")

    raw_fingerprints = manifest["evidence_fingerprints"]
    if not isinstance(raw_fingerprints, list) or not raw_fingerprints:
        raise ValueError("VLM result manifest evidence_fingerprints must be non-empty")
    fingerprints: list[dict[str, Any]] = []
    previous_field = ""
    for index, raw_fingerprint in enumerate(raw_fingerprints):
        fingerprint = _strict_fields(
            raw_fingerprint,
            expected=frozenset({"field", "size", "sha256"}),
            label=(f"VLM result manifest evidence_fingerprints[{index}]"),
        )
        field = _clean_text(
            fingerprint["field"],
            label=(f"VLM result manifest evidence_fingerprints[{index}].field"),
        )
        if field <= previous_field:
            raise ValueError(
                "VLM result manifest evidence fingerprints must be strictly ordered"
            )
        previous_field = field
        size = fingerprint["size"]
        if isinstance(size, bool) or not isinstance(size, int) or size < 1:
            raise ValueError("VLM result manifest evidence fingerprint size is invalid")
        sha256 = _clean_text(
            fingerprint["sha256"],
            label=(f"VLM result manifest evidence_fingerprints[{index}].sha256"),
        )
        if re.fullmatch(r"[0-9a-f]{64}", sha256) is None:
            raise ValueError(
                "VLM result manifest evidence fingerprint digest is invalid"
            )
        fingerprints.append({"field": field, "size": size, "sha256": sha256})
    if expected_evidence_fingerprints is not None:
        expected_fingerprints = _credential_free_json(
            list(expected_evidence_fingerprints),
            label="expected VLM evidence_fingerprints",
        )
        if fingerprints != expected_fingerprints:
            raise ValueError("VLM result manifest evidence fingerprints do not match")

    request = _strict_fields(
        manifest["request"],
        expected=frozenset({"evidence", "behavior", "outputs"}),
        label="VLM result manifest request",
    )
    normalized_request = _credential_free_json(
        request,
        label="VLM result manifest request",
    )
    assert isinstance(normalized_request, dict)
    request_sha256 = _request_sha256(
        request_id=request_id,
        mode=mode,
        evidence=normalized_request["evidence"],
        behavior=normalized_request["behavior"],
        outputs=normalized_request["outputs"],
        preparation_sha256=preparation_sha256,
    )
    if expected_request_sha256 is not None:
        expected_digest = _clean_text(
            expected_request_sha256,
            label="expected normalized VLM request sha256",
        )
        if (
            re.fullmatch(r"[0-9a-f]{64}", expected_digest) is None
            or request_sha256 != expected_digest
        ):
            raise ValueError("VLM result manifest normalized request does not match")
    result = _json_result(
        manifest["result"],
        label="VLM result manifest result",
    )
    normalized: dict[str, Any] = {
        "format": format,
        "request_manifest_format": (request_manifest_format),
        "request_id": request_id,
        "mode": mode,
        "run_identity": identity,
        "evidence_fingerprints": fingerprints,
        "request": normalized_request,
        "request_sha256": request_sha256,
        "result": result,
        "path": path,
    }
    normalized["request_manifest_fingerprint"] = request_manifest_fingerprint
    normalized["preparation_sha256"] = preparation_sha256
    if mode == "video_trajectory":
        if manifest["paper_metric_compatibility"] != "not_claimed":
            raise ValueError(
                "video_trajectory VLM result cannot claim paper compatibility"
            )
        if manifest["metric_claims"] != []:
            raise ValueError("video_trajectory VLM result metric_claims must be empty")
        normalized["paper_metric_compatibility"] = "not_claimed"
        normalized["metric_claims"] = []
    return normalized


def _json_result(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must return a mapping")
    credential_free = _credential_free_json(value, label=label)
    assert isinstance(credential_free, dict)
    try:
        encoded = json.dumps(
            credential_free,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
        )
    except (TypeError, ValueError) as error:
        raise TypeError(f"{label} must return a JSON mapping") from error
    return json.loads(encoded)


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    directory_descriptor = _open_directory_chain(
        path.parent,
        create=True,
        label="VLM result manifest",
    )
    assert directory_descriptor is not None
    temporary_name = f".{path.name}.{os.getpid()}.{secrets.token_hex(8)}.tmp"
    descriptor: int | None = None
    try:
        try:
            existing = os.stat(
                path.name,
                dir_fd=directory_descriptor,
                follow_symlinks=False,
            )
        except FileNotFoundError:
            existing = None
        if existing is not None and not stat.S_ISREG(existing.st_mode):
            raise ValueError("VLM result manifest destination must be a regular file")
        descriptor = os.open(
            temporary_name,
            (
                os.O_WRONLY
                | os.O_CREAT
                | os.O_EXCL
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            ),
            0o600,
            dir_fd=directory_descriptor,
        )
        offset = 0
        while offset < len(encoded):
            offset += os.write(descriptor, encoded[offset:])
        os.fsync(descriptor)
        os.close(descriptor)
        descriptor = None
        os.replace(
            temporary_name,
            path.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    except Exception:
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        raise
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_descriptor)


def parse_strict_json_object(
    encoded: bytes,
    *,
    label: str,
) -> dict[str, Any]:
    """Parse a strict finite JSON object for another bench adapter."""

    return _strict_json_object(encoded, label=label)


def credential_free_json(
    value: Any,
    *,
    label: str,
) -> Any:
    """Return a detached finite JSON value after rejecting credentials."""

    return _credential_free_json(value, label=label)


def read_bounded_saved_artifact(
    path: str | Path,
    *,
    max_bytes: int,
    label: str,
) -> bytes:
    """Read one stable regular saved artifact without following symlinks."""

    return _read_bounded_no_symlink_chain(
        Path(path).absolute(),
        max_bytes=max_bytes,
        label=label,
    )


def write_json_atomic(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    """Atomically replace one JSON file while preserving an older result."""

    _write_json_atomic(Path(path).absolute(), payload)


def _verify_request_manifest_fingerprint(
    manifest: Mapping[str, Any],
    *,
    expected: Mapping[str, Any] | None,
) -> None:
    if expected is None:
        return
    if not isinstance(expected, Mapping):
        raise TypeError("expected VLM request manifest fingerprint must be a mapping")
    if set(expected) != {"size", "sha256"}:
        raise ValueError("expected VLM request manifest fingerprint is invalid")
    expected_size = expected["size"]
    expected_sha256 = expected["sha256"]
    if (
        isinstance(expected_size, bool)
        or not isinstance(expected_size, int)
        or expected_size < 1
        or not isinstance(expected_sha256, str)
        or re.fullmatch(r"[0-9a-f]{64}", expected_sha256) is None
    ):
        raise ValueError("expected VLM request manifest fingerprint is invalid")
    payload = _read_bounded_no_symlink_chain(
        Path(manifest["path"]),
        max_bytes=MAX_VLM_REQUEST_MANIFEST_BYTES,
        label="VLM request manifest",
    )
    if (
        len(payload) != expected_size
        or hashlib.sha256(payload).hexdigest() != expected_sha256
    ):
        raise RuntimeError(
            "VLM request manifest changed after evaluation plan preflight"
        )


def _verify_required_evaluator_outputs(
    request: Mapping[str, Any],
) -> None:
    for destination in request["output_destinations"]:
        if destination["owner"] != "evaluator":
            continue
        _read_bounded_no_symlink_chain(
            Path(destination["path"]),
            max_bytes=MAX_VLM_RESULT_MANIFEST_BYTES,
            label=(f"required VLM evaluator output {destination['field']}"),
        )


def _verify_all_required_outputs(
    request: Mapping[str, Any],
) -> None:
    for destination in request["output_destinations"]:
        _read_bounded_no_symlink_chain(
            Path(destination["path"]),
            max_bytes=MAX_VLM_RESULT_MANIFEST_BYTES,
            label=f"required VLM output {destination['field']}",
        )


def evaluate_vlm_request_manifest(
    manifest_path: str | Path,
    *,
    expected_run_identity: Mapping[str, Any],
    exec_dir: str | Path,
    evaluators: VLMEvaluatorRegistry,
    protected_output_paths: Mapping[str, str | Path] | None = None,
    durable_run_root: str | Path | None = None,
    expected_formal_run_root: str | Path | None = None,
    expected_request_sha256s: Sequence[str] | None = None,
    expected_manifest_fingerprint: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Evaluate preflighted requests in manifest order and publish results."""

    manifest = load_vlm_request_manifest(
        manifest_path,
        expected_run_identity=expected_run_identity,
        protected_output_paths=protected_output_paths,
        durable_run_root=durable_run_root,
        expected_formal_run_root=expected_formal_run_root,
    )
    _verify_formal_state_attestation(manifest["formal_state_attestation_source"])
    _verify_preparation_fingerprints(manifest["preparation_fingerprint_sources"])
    if expected_request_sha256s is not None:
        expected_digests = list(expected_request_sha256s)
        actual_digests = [
            str(request["request_sha256"]) for request in manifest["requests"]
        ]
        if actual_digests != expected_digests:
            raise RuntimeError("VLM requests changed after evaluation plan preflight")
    runtime_manifest_fingerprint = (
        manifest["manifest_fingerprint"]
        if expected_manifest_fingerprint is None
        else expected_manifest_fingerprint
    )
    _verify_request_manifest_fingerprint(
        manifest,
        expected=runtime_manifest_fingerprint,
    )
    requested_modes = {str(request["mode"]) for request in manifest["requests"]}
    registry = _validated_evaluators(
        evaluators,
        requested_modes=requested_modes,
    )
    identity = copy.deepcopy(dict(manifest["run_identity"]))
    result_format = VLM_RESULT_MANIFEST_SCHEMA
    results: list[dict[str, Any]] = []
    for request in manifest["requests"]:
        request_id = str(request["request_id"])
        mode = str(request["mode"])
        _verify_formal_state_attestation(manifest["formal_state_attestation_source"])
        _verify_preparation_fingerprints(manifest["preparation_fingerprint_sources"])
        _verify_evidence_fingerprints(request["evidence_fingerprint_sources"])
        _verify_request_manifest_fingerprint(
            manifest,
            expected=runtime_manifest_fingerprint,
        )
        if mode == "video_trajectory":
            load_union_trajectory(
                request["resolved_evidence"]["trajectory_path"],
                expected_uid=identity["uid"],
            )
            _verify_evidence_fingerprints(request["evidence_fingerprint_sources"])
        raw_result = registry[mode](
            exec_dir=Path(exec_dir).expanduser().resolve().as_posix(),
            **copy.deepcopy(identity),
            request_id=request_id,
            mode=mode,
            evidence=copy.deepcopy(request["resolved_evidence"]),
            evidence_fingerprints=copy.deepcopy(request["evidence_fingerprints"]),
            behavior=copy.deepcopy(request["behavior"]),
            outputs=copy.deepcopy(request["resolved_outputs"]),
        )
        _verify_formal_state_attestation(manifest["formal_state_attestation_source"])
        _verify_preparation_fingerprints(manifest["preparation_fingerprint_sources"])
        _verify_evidence_fingerprints(request["evidence_fingerprint_sources"])
        _verify_request_manifest_fingerprint(
            manifest,
            expected=runtime_manifest_fingerprint,
        )
        result = _json_result(
            raw_result,
            label=f"VLM evaluator {mode!r}",
        )
        _verify_required_evaluator_outputs(request)
        result_manifest_path = Path(request["result_manifest_path"])
        durable = {
            "format": result_format,
            "request_manifest_format": manifest["format"],
            "request_id": request_id,
            "mode": mode,
            "run_identity": copy.deepcopy(identity),
            "evidence_fingerprints": copy.deepcopy(request["evidence_fingerprints"]),
            "request": {
                "evidence": copy.deepcopy(request["evidence"]),
                "behavior": copy.deepcopy(request["behavior"]),
                "outputs": copy.deepcopy(request["outputs"]),
            },
            "result": copy.deepcopy(result),
        }
        durable["request_manifest_fingerprint"] = copy.deepcopy(
            manifest["manifest_fingerprint"]
        )
        durable["preparation_sha256"] = manifest["preparation_sha256"]
        if mode == "video_trajectory":
            durable["paper_metric_compatibility"] = "not_claimed"
            durable["metric_claims"] = []
        _write_json_atomic(result_manifest_path, durable)
        load_vlm_result_manifest(
            result_manifest_path,
            expected_run_identity=identity,
            expected_request_id=request_id,
            expected_mode=mode,
            expected_evidence_fingerprints=request["evidence_fingerprints"],
            expected_request_sha256=request["request_sha256"],
            expected_request_manifest_fingerprint=manifest["manifest_fingerprint"],
            expected_preparation_sha256=manifest["preparation_sha256"],
        )
        _verify_all_required_outputs(request)
        _verify_formal_state_attestation(manifest["formal_state_attestation_source"])
        _verify_preparation_fingerprints(manifest["preparation_fingerprint_sources"])
        _verify_evidence_fingerprints(request["evidence_fingerprint_sources"])
        _verify_request_manifest_fingerprint(
            manifest,
            expected=runtime_manifest_fingerprint,
        )
        item: dict[str, Any] = {
            "request_id": request_id,
            "mode": mode,
            "result_manifest": {
                "format": result_format,
                "path": result_manifest_path.as_posix(),
            },
            "evidence_fingerprints": copy.deepcopy(request["evidence_fingerprints"]),
            "result": result,
        }
        if mode == "video_trajectory":
            item["paper_metric_compatibility"] = "not_claimed"
            item["metric_claims"] = []
        results.append(item)
    return results


__all__ = [
    "MAX_VLM_REQUEST_MANIFEST_BYTES",
    "MAX_VLM_RESULT_MANIFEST_BYTES",
    "VLM_MODES",
    "VLM_REQUEST_MANIFEST_SCHEMA",
    "VLM_REQUEST_MANIFEST_SCHEMAS",
    "VLM_RESULT_MANIFEST_SCHEMA",
    "VLM_RESULT_MANIFEST_SCHEMAS",
    "VLMEvaluator",
    "VLMEvaluatorRegistry",
    "evaluate_vlm_request_manifest",
    "credential_free_json",
    "load_vlm_request_manifest",
    "load_vlm_result_manifest",
    "parse_strict_json_object",
    "read_bounded_saved_artifact",
    "write_json_atomic",
]
