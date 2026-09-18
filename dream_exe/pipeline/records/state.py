"""Verified opt-in stage state for one formal benchmark run.

This module owns no pipeline execution.  It stores only bounded fingerprints
and relative output digests, then produces a pure reuse plan for the existing
single-run workflow.  Bench paths remain caller-owned and no state is read or
written unless a bench-facing caller explicitly asks for it.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from contextlib import contextmanager
import copy
import errno
import hashlib
import json
import os
from pathlib import Path, PurePosixPath, PureWindowsPath
import re
import secrets
import stat
from typing import Any
from urllib.parse import parse_qsl, urlsplit

try:
    import fcntl
except ModuleNotFoundError:  # pragma: no cover - resume targets POSIX hosts
    fcntl = None


PIPELINE_STATE_SCHEMA = "dream_exe.benchmark-stage-state"
PIPELINE_REUSE_PLAN_SCHEMA = "dream_exe.benchmark-stage-reuse-plan"
PIPELINE_STAGE_ORDER = (
    "video",
    "video2traj",
    "exec",
    "task_success",
    "eval",
)

_MAX_FINGERPRINT_INPUT_BYTES = 2 * 1024 * 1024
_MAX_OUTPUTS_PER_STAGE = 512
_MAX_STATE_BYTES = 4 * 1024 * 1024
_HEX_DIGITS = frozenset("0123456789abcdef")
_STATE_KEYS = frozenset({"format", "run_identity", "stages"})
_IDENTITY_KEYS = frozenset({"uid", "run_id", "run_key", "gen_model"})
_STAGE_KEYS = frozenset(
    {
        "status",
        "input_fingerprint",
        "implementation_fingerprint",
        "outputs",
        "failure_type",
    }
)
_OUTPUT_KEYS = frozenset({"root", "path", "size", "sha256"})
_OUTPUT_IDENTITY_KEYS = frozenset({"root", "path"})
_SENSITIVE_KEYS = frozenset(
    {
        "access_token",
        "api_key",
        "apikey",
        "auth_token",
        "authorization",
        "bearer_token",
        "credential",
        "credentials",
        "connection_string",
        "cookie",
        "dsn",
        "password",
        "private_key",
        "refresh_token",
        "secret",
        "token",
    }
)


def _clean_text(value: Any, *, field: str, required: bool = True) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{field} must be a string")
    text = value.strip()
    if text != value:
        raise ValueError(f"{field} cannot contain surrounding whitespace")
    if required and not text:
        raise ValueError(f"{field} is required")
    if any(character in text for character in ("\x00", "\r", "\n", "\t")):
        raise ValueError(f"{field} contains a control character")
    return text


def _is_digest(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    text = value
    return len(text) == 64 and text == text.lower() and set(text).issubset(_HEX_DIGITS)


def _sensitive_key(value: Any) -> bool:
    raw_key = str(value).strip().replace("-", "_")
    key = re.sub(
        r"(?<=[a-z0-9])(?=[A-Z])",
        "_",
        raw_key,
    ).lower()
    segments = frozenset(part for part in key.split("_") if part)
    return (
        key in _SENSITIVE_KEYS
        or bool(
            segments.intersection(
                {
                    "credential",
                    "credentials",
                    "password",
                    "secret",
                }
            )
        )
        or key.endswith("_access_key")
        or key.endswith("_access_key_id")
        or key.endswith("_api_key")
        or key.endswith("_connection_string")
        or key.endswith("_cookie")
        or key.endswith("_dsn")
        or key.endswith("_password")
        or key.endswith("_private_key")
        or key.endswith("_secret")
        or key.endswith("_token")
    )


def _portable_string(value: str, *, field: str) -> str:
    if any(character in value for character in ("\x00", "\r", "\n")):
        raise ValueError(f"{field} contains a control character")
    if (
        Path(value).is_absolute()
        or PureWindowsPath(value).is_absolute()
        or value.startswith(("~/", r"~\\", "file://"))
    ):
        raise ValueError(f"{field} cannot contain an absolute path")
    if "://" in value:
        parsed = urlsplit(value)
        if parsed.username is not None or parsed.password is not None:
            raise ValueError(f"{field} cannot contain URL credentials")
        if any(_sensitive_key(key) for key, _item in parse_qsl(parsed.query)):
            raise ValueError(f"{field} cannot contain URL credentials")
    lowered = value.strip().lower()
    if lowered.startswith(("bearer ", "basic ")):
        raise ValueError(f"{field} cannot contain authorization credentials")
    if "-----begin" in lowered and "private key-----" in lowered:
        raise ValueError(f"{field} cannot contain a private key")
    return value


def _canonical_json_value(value: Any, *, field: str) -> Any:
    if value is None or isinstance(value, (bool, int)):
        return value
    if isinstance(value, str):
        return _portable_string(value, field=field)
    if isinstance(value, float):
        if not (float("-inf") < value < float("inf")):
            raise ValueError(f"{field} contains a non-finite float")
        return value
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for raw_key, nested in value.items():
            if not isinstance(raw_key, str):
                raise TypeError(f"{field} keys must be strings")
            key = _clean_text(raw_key, field=f"{field} key")
            if key != raw_key:
                raise ValueError(f"{field} keys cannot contain surrounding whitespace")
            if _sensitive_key(key):
                raise ValueError(f"{field} cannot contain credentials")
            if key in output:
                raise ValueError(f"{field} contains duplicate normalized key {key!r}")
            output[key] = _canonical_json_value(
                nested,
                field=f"{field}.{key}",
            )
        return output
    if isinstance(value, Sequence) and not isinstance(
        value,
        (str, bytes, bytearray),
    ):
        return [_canonical_json_value(item, field=f"{field}[]") for item in value]
    raise TypeError(
        f"{field} must contain only JSON values; got {type(value).__name__}"
    )


def _strict_json_object(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for key, value in pairs:
        if key in output:
            raise ValueError(f"duplicate JSON object key: {key!r}")
        output[key] = value
    return output


def _reject_json_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON constant is not allowed: {value}")


def fingerprint_value(value: Any) -> str:
    """Hash a credential-free JSON value using canonical serialization."""

    canonical = _canonical_json_value(value, field="fingerprint input")
    encoded = json.dumps(
        canonical,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    if len(encoded) > _MAX_FINGERPRINT_INPUT_BYTES:
        raise ValueError("fingerprint input exceeds the bounded size limit")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    """Return the SHA-256 of one explicit regular file."""

    source = Path(path).expanduser().resolve(strict=True)
    if not source.is_file():
        raise ValueError(f"expected a regular file: {source}")
    digest = hashlib.sha256()
    with source.open("rb") as stream:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _contained_relative(path: Path, root: Path, *, label: str) -> str:
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ValueError(f"{label} escapes its declared root") from error
    text = relative.as_posix()
    pure = PurePosixPath(text)
    if not text or pure.is_absolute() or ".." in pure.parts or "." in pure.parts:
        raise ValueError(f"{label} is not a safe relative path")
    return text


def _digest_relative_regular_file(
    root: Path,
    relative: str,
) -> tuple[int, str]:
    """Hash one contained file without following any path-component symlink."""

    parts = PurePosixPath(relative).parts
    if not parts:
        raise ValueError("output path is empty")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    current_descriptor = os.open(root, directory_flags)
    file_descriptor: int | None = None
    try:
        for part in parts[:-1]:
            next_descriptor = os.open(
                part,
                directory_flags,
                dir_fd=current_descriptor,
            )
            os.close(current_descriptor)
            current_descriptor = next_descriptor
        file_descriptor = os.open(
            parts[-1],
            (os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_NONBLOCK", 0)),
            dir_fd=current_descriptor,
        )
        metadata = os.fstat(file_descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("output path is not a regular file")
        digest = hashlib.sha256()
        with os.fdopen(file_descriptor, "rb") as stream:
            file_descriptor = None
            while True:
                chunk = stream.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
        return int(metadata.st_size), digest.hexdigest()
    finally:
        if file_descriptor is not None:
            os.close(file_descriptor)
        os.close(current_descriptor)


def digest_output_file(
    path: str | Path,
    *,
    sample_root: str | Path,
    run_root: str | Path,
) -> dict[str, Any]:
    """Build a portable output record without storing an absolute path."""

    sample = Path(sample_root).expanduser().resolve(strict=True)
    run = Path(run_root).expanduser().resolve(strict=True)
    try:
        run.relative_to(sample)
    except ValueError as error:
        raise ValueError("run_root escapes sample_root") from error

    source = Path(path).expanduser().absolute()
    try:
        relative = _contained_relative(source, run, label="output path")
        root_name = "run"
        selected_root = run
    except ValueError:
        relative = _contained_relative(source, sample, label="output path")
        root_name = "sample"
        selected_root = sample
    try:
        size, digest = _digest_relative_regular_file(
            selected_root,
            relative,
        )
    except (FileNotFoundError, NotADirectoryError, OSError, ValueError) as error:
        raise ValueError(
            f"expected a contained non-symlink regular output file: {source}"
        ) from error
    return {
        "root": root_name,
        "path": relative,
        "size": size,
        "sha256": digest,
    }


def digest_existing_outputs(
    paths: Sequence[str | Path],
    *,
    sample_root: str | Path,
    run_root: str | Path,
) -> list[dict[str, Any]]:
    """Digest existing explicit outputs in deterministic path order."""

    records = [
        digest_output_file(
            path,
            sample_root=sample_root,
            run_root=run_root,
        )
        for path in paths
    ]
    records.sort(key=lambda item: (item["root"], item["path"]))
    identities = [(item["root"], item["path"]) for item in records]
    if len(identities) != len(set(identities)):
        raise ValueError("duplicate output path in stage state")
    return records


def _validate_output_record(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    raw = dict(value)
    unknown = set(raw).difference(_OUTPUT_KEYS)
    missing = _OUTPUT_KEYS.difference(raw)
    if unknown or missing:
        raise ValueError(f"{field} has invalid fields")
    root = _clean_text(raw["root"], field=f"{field}.root")
    if root not in {"sample", "run"}:
        raise ValueError(f"{field}.root must be sample or run")
    path = _clean_text(raw["path"], field=f"{field}.path")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or path != pure.as_posix()
    ):
        raise ValueError(f"{field}.path must be a safe relative path")
    size = raw["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{field}.size must be a non-negative integer")
    digest = raw["sha256"]
    if not _is_digest(digest):
        raise ValueError(f"{field}.sha256 must be a lowercase SHA-256")
    return {
        "root": root,
        "path": path,
        "size": size,
        "sha256": digest,
    }


def _validate_output_identity(value: Any, *, field: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    raw = dict(value)
    if set(raw) != _OUTPUT_IDENTITY_KEYS:
        raise ValueError(f"{field} has invalid fields")
    root = _clean_text(raw["root"], field=f"{field}.root")
    if root not in {"sample", "run"}:
        raise ValueError(f"{field}.root must be sample or run")
    path = _clean_text(raw["path"], field=f"{field}.path")
    pure = PurePosixPath(path)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or path != pure.as_posix()
    ):
        raise ValueError(f"{field}.path must be a safe relative path")
    return {"root": root, "path": path}


def _validate_stage_record(value: Any, *, field: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field} must be an object")
    raw = dict(value)
    unknown = set(raw).difference(_STAGE_KEYS)
    required = {
        "status",
        "input_fingerprint",
        "implementation_fingerprint",
        "outputs",
    }
    if unknown or required.difference(raw):
        raise ValueError(f"{field} has invalid fields")
    status = _clean_text(raw["status"], field=f"{field}.status")
    if status not in {"completed", "failed"}:
        raise ValueError(f"{field}.status must be completed or failed")
    input_fingerprint = raw["input_fingerprint"]
    implementation_fingerprint = raw["implementation_fingerprint"]
    if not _is_digest(input_fingerprint):
        raise ValueError(f"{field}.input_fingerprint must be a lowercase SHA-256")
    if not _is_digest(implementation_fingerprint):
        raise ValueError(
            f"{field}.implementation_fingerprint must be a lowercase SHA-256"
        )
    raw_outputs = raw["outputs"]
    if not isinstance(raw_outputs, list):
        raise ValueError(f"{field}.outputs must be a list")
    if len(raw_outputs) > _MAX_OUTPUTS_PER_STAGE:
        raise ValueError(f"{field}.outputs exceeds the bounded entry limit")
    outputs = [
        _validate_output_record(item, field=f"{field}.outputs[{index}]")
        for index, item in enumerate(raw_outputs)
    ]
    identities = [(item["root"], item["path"]) for item in outputs]
    if identities != sorted(identities) or len(identities) != len(set(identities)):
        raise ValueError(f"{field}.outputs must be unique and sorted")
    if status == "completed" and not outputs:
        raise ValueError(f"{field}.outputs cannot be empty when completed")
    failure_type = ""
    if "failure_type" in raw:
        failure_type = _clean_text(
            raw["failure_type"],
            field=f"{field}.failure_type",
            required=False,
        )
        if status != "failed" and failure_type:
            raise ValueError(f"{field}.failure_type is allowed only for failed state")
    output = {
        "status": status,
        "input_fingerprint": input_fingerprint,
        "implementation_fingerprint": implementation_fingerprint,
        "outputs": outputs,
    }
    if failure_type:
        output["failure_type"] = failure_type
    return output


def _validate_run_identity(value: Any) -> dict[str, str]:
    if not isinstance(value, Mapping):
        raise ValueError("run_identity must be an object")
    identity_mapping = dict(value)
    if set(identity_mapping) != _IDENTITY_KEYS:
        raise ValueError("run_identity has invalid fields")
    return {
        "uid": _clean_text(identity_mapping["uid"], field="run_identity.uid"),
        "run_id": _clean_text(
            identity_mapping["run_id"],
            field="run_identity.run_id",
        ),
        "run_key": _clean_text(
            identity_mapping["run_key"],
            field="run_identity.run_key",
        ),
        "gen_model": _clean_text(
            identity_mapping["gen_model"],
            field="run_identity.gen_model",
            required=False,
        ),
    }


def validate_benchmark_pipeline_state(
    payload: Any,
) -> dict[str, Any]:
    """Validate and detach one bounded pipeline-state document."""

    if not isinstance(payload, Mapping):
        raise ValueError("benchmark pipeline state must be an object")
    raw = dict(payload)
    if set(raw) != _STATE_KEYS:
        raise ValueError("benchmark pipeline state has invalid fields")
    if raw["format"] != PIPELINE_STATE_SCHEMA:
        raise ValueError("benchmark pipeline state has an unsupported format")
    identity = _validate_run_identity(raw["run_identity"])
    stages_raw = raw["stages"]
    if not isinstance(stages_raw, Mapping):
        raise ValueError("stages must be an object")
    unknown_stages = set(stages_raw).difference(PIPELINE_STAGE_ORDER)
    if unknown_stages:
        raise ValueError(
            "benchmark pipeline state has unsupported stage(s): "
            + ", ".join(sorted(unknown_stages))
        )
    stages = {
        stage: _validate_stage_record(
            stages_raw[stage],
            field=f"stages.{stage}",
        )
        for stage in PIPELINE_STAGE_ORDER
        if stage in stages_raw
    }
    return {
        "format": PIPELINE_STATE_SCHEMA,
        "run_identity": identity,
        "stages": stages,
    }


def load_benchmark_pipeline_state(
    path: str | Path,
) -> dict[str, Any] | None:
    """Load explicit state; a missing file is an ordinary first run."""

    source = Path(path).expanduser().absolute()
    if not source.parent.exists():
        return None
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor = os.open(source.parent, directory_flags)
    descriptor: int | None = None
    try:
        try:
            descriptor = os.open(
                source.name,
                (
                    os.O_RDONLY
                    | getattr(os, "O_NOFOLLOW", 0)
                    | getattr(os, "O_NONBLOCK", 0)
                ),
                dir_fd=directory_descriptor,
            )
        except FileNotFoundError:
            return None
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError(
                    f"benchmark pipeline state cannot be a symlink: {source}"
                ) from error
            raise
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError(
                f"benchmark pipeline state is not a regular file: {source}"
            )
        if metadata.st_size > _MAX_STATE_BYTES:
            raise ValueError(
                f"benchmark pipeline state exceeds the bounded size limit: {source}"
            )
        with os.fdopen(descriptor, "rb") as stream:
            descriptor = None
            encoded = stream.read(_MAX_STATE_BYTES + 1)
        if len(encoded) > _MAX_STATE_BYTES:
            raise ValueError(
                f"benchmark pipeline state exceeds the bounded size limit: {source}"
            )
        payload = json.loads(
            encoded.decode("utf-8"),
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise ValueError(
            f"cannot load benchmark pipeline state: {source}: {error}"
        ) from error
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_descriptor)
    return validate_benchmark_pipeline_state(payload)


def write_benchmark_pipeline_state(
    path: str | Path,
    payload: Mapping[str, Any],
) -> None:
    """Atomically publish validated state at one explicit destination."""

    destination = Path(path).expanduser().absolute()
    if destination.is_symlink():
        raise ValueError(f"benchmark pipeline state cannot be a symlink: {destination}")
    if destination.exists() and not destination.is_file():
        raise ValueError(
            f"benchmark pipeline state destination is not a regular file: {destination}"
        )
    state = validate_benchmark_pipeline_state(payload)
    encoded = (
        json.dumps(
            state,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > _MAX_STATE_BYTES:
        raise ValueError("benchmark pipeline state exceeds the bounded size limit")
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor = os.open(
        destination.parent,
        directory_flags,
    )
    temporary_name = f".{destination.name}.{secrets.token_hex(12)}.tmp"
    descriptor: int | None = None
    try:
        descriptor = os.open(
            temporary_name,
            (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)),
            0o600,
            dir_fd=directory_descriptor,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(
            temporary_name,
            destination.name,
            src_dir_fd=directory_descriptor,
            dst_dir_fd=directory_descriptor,
        )
        os.fsync(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        try:
            os.unlink(temporary_name, dir_fd=directory_descriptor)
        except FileNotFoundError:
            pass
        os.close(directory_descriptor)


@contextmanager
def benchmark_pipeline_state_lock(
    path: str | Path,
    *,
    require_existing: bool = False,
):
    """Hold one cross-process lock for a formal run and its state/artifacts."""

    if fcntl is None:
        raise RuntimeError("verified benchmark resume requires POSIX file locking")
    if not isinstance(require_existing, bool):
        raise TypeError("require_existing must be bool")
    destination = Path(path).expanduser().absolute()
    if require_existing:
        if not destination.parent.is_dir():
            raise FileNotFoundError(
                "verified reuse requires an existing state directory: "
                f"{destination.parent}"
            )
    else:
        destination.parent.mkdir(parents=True, exist_ok=True)
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor = os.open(
        destination.parent,
        directory_flags,
    )
    lock_name = f".{destination.name}.lock"
    descriptor: int | None = None
    try:
        try:
            flags = os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
            if not require_existing:
                flags |= os.O_CREAT
            descriptor = os.open(
                lock_name,
                flags,
                0o600,
                dir_fd=directory_descriptor,
            )
        except FileNotFoundError as error:
            if require_existing:
                raise FileNotFoundError(
                    "verified reuse requires an existing state lock: "
                    f"{destination.parent / lock_name}"
                ) from error
            raise
        except OSError as error:
            if error.errno == errno.ELOOP:
                raise ValueError(
                    "benchmark pipeline state lock cannot be a symlink"
                ) from error
            raise
        metadata = os.fstat(descriptor)
        if not stat.S_ISREG(metadata.st_mode):
            raise ValueError("benchmark pipeline state lock is not a regular file")
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        if descriptor is not None:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
            finally:
                os.close(descriptor)
        os.close(directory_descriptor)


def verify_recorded_outputs(
    records: Sequence[Mapping[str, Any]],
    *,
    sample_root: str | Path,
    run_root: str | Path,
) -> tuple[bool, str]:
    """Verify every recorded output without writing or trusting mtimes."""

    sample = Path(sample_root).expanduser().resolve(strict=True)
    run = Path(run_root).expanduser().resolve(strict=False)
    try:
        run.relative_to(sample)
    except ValueError:
        return False, "run_root_escape"
    for index, raw in enumerate(records):
        try:
            record = _validate_output_record(
                raw,
                field=f"outputs[{index}]",
            )
            root = run if record["root"] == "run" else sample
            size, digest = _digest_relative_regular_file(
                root,
                record["path"],
            )
        except FileNotFoundError:
            return False, f"missing_output:{record['root']}:{record['path']}"
        except (NotADirectoryError, OSError, ValueError):
            return False, "invalid_output_record"
        if size != record["size"]:
            return False, f"output_size_changed:{record['root']}:{record['path']}"
        if digest != record["sha256"]:
            return False, f"output_digest_changed:{record['root']}:{record['path']}"
    return True, "outputs_match"


def _current_stage_descriptor(
    value: Any,
    *,
    stage: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"current_stages.{stage} must be an object")
    raw = dict(value)
    allowed = {
        "input_fingerprint",
        "implementation_fingerprint",
        "required_outputs",
        "reusable",
        "reason",
    }
    if set(raw).difference(allowed):
        raise ValueError(f"current_stages.{stage} has invalid fields")
    reusable_value = raw.get("reusable", True)
    if not isinstance(reusable_value, bool):
        raise ValueError(f"current_stages.{stage}.reusable must be boolean")
    reusable = reusable_value
    reason = _clean_text(
        raw.get("reason", ""),
        field=f"current_stages.{stage}.reason",
        required=False,
    )
    input_fingerprint = raw.get("input_fingerprint", "")
    implementation_fingerprint = raw.get(
        "implementation_fingerprint",
        "",
    )
    if reusable and (
        not _is_digest(input_fingerprint) or not _is_digest(implementation_fingerprint)
    ):
        raise ValueError(
            f"current_stages.{stage} reusable fingerprints must be SHA-256"
        )
    if reusable and "required_outputs" not in raw:
        raise ValueError(
            f"current_stages.{stage}.required_outputs is required for reusable stages"
        )
    raw_required = raw.get("required_outputs", [])
    if not isinstance(raw_required, list):
        raise ValueError(f"current_stages.{stage}.required_outputs must be a list")
    if len(raw_required) > _MAX_OUTPUTS_PER_STAGE:
        raise ValueError(
            f"current_stages.{stage}.required_outputs exceeds the bounded entry limit"
        )
    if reusable and not raw_required:
        raise ValueError(
            f"current_stages.{stage}.required_outputs cannot be empty "
            "for reusable stages"
        )
    required_outputs = [
        _validate_output_identity(
            item,
            field=(f"current_stages.{stage}.required_outputs[{index}]"),
        )
        for index, item in enumerate(raw_required)
    ]
    required_identities = [(item["root"], item["path"]) for item in required_outputs]
    if required_identities != sorted(required_identities) or len(
        required_identities
    ) != len(set(required_identities)):
        raise ValueError(
            f"current_stages.{stage}.required_outputs must be unique and sorted"
        )
    return {
        "reusable": reusable,
        "reason": reason,
        "input_fingerprint": input_fingerprint,
        "implementation_fingerprint": implementation_fingerprint,
        "required_outputs": required_outputs,
    }


def plan_benchmark_stage_reuse(
    *,
    run_identity: Mapping[str, Any],
    current_stages: Mapping[str, Mapping[str, Any]],
    prior_state: Mapping[str, Any] | None,
    selected_stages: Sequence[str],
    sample_root: str | Path,
    run_root: str | Path,
    invalidate_stages: Sequence[str] = (),
) -> dict[str, Any]:
    """Plan safe reuse in canonical order with downstream invalidation."""

    unknown_current = set(current_stages).difference(PIPELINE_STAGE_ORDER)
    if unknown_current:
        raise ValueError(
            "current_stages has unsupported stage(s): "
            + ", ".join(sorted(unknown_current))
        )
    selected = {
        _clean_text(stage, field="selected stage").lower() for stage in selected_stages
    }
    invalidated = {
        _clean_text(stage, field="invalidated stage").lower()
        for stage in invalidate_stages
    }
    unknown = (selected | invalidated).difference(PIPELINE_STAGE_ORDER)
    if unknown:
        raise ValueError("unsupported pipeline stage(s): " + ", ".join(sorted(unknown)))
    expected_identity = _validate_run_identity(run_identity)
    normalized_prior = (
        None if prior_state is None else validate_benchmark_pipeline_state(prior_state)
    )
    identity_mismatch = (
        normalized_prior is not None
        and normalized_prior["run_identity"] != expected_identity
    )
    prior_stages = (
        {}
        if normalized_prior is None or identity_mismatch
        else normalized_prior["stages"]
    )
    stage_plan: dict[str, dict[str, Any]] = {}
    cascade_source = ""

    for stage in PIPELINE_STAGE_ORDER:
        if stage in invalidated and not cascade_source:
            cascade_source = stage
        if stage not in selected:
            stage_plan[stage] = {
                "decision": "not_selected",
                "reason": "not_selected",
            }
            continue
        descriptor = _current_stage_descriptor(
            current_stages.get(
                stage,
                {
                    "reusable": False,
                    "reason": "missing_current_stage_descriptor",
                },
            ),
            stage=stage,
        )
        if cascade_source:
            reason = (
                "manually_invalidated"
                if cascade_source == stage and stage in invalidated
                else f"upstream_invalidated:{cascade_source}"
            )
            stage_plan[stage] = {"decision": "run", "reason": reason}
            continue
        if not descriptor["reusable"]:
            reason = descriptor["reason"] or "untracked_stage_inputs"
            stage_plan[stage] = {"decision": "run", "reason": reason}
            cascade_source = stage
            continue
        if identity_mismatch:
            stage_plan[stage] = {
                "decision": "run",
                "reason": "run_identity_changed",
            }
            cascade_source = stage
            continue
        prior = prior_stages.get(stage)
        if prior is None:
            stage_plan[stage] = {
                "decision": "run",
                "reason": "no_completed_state",
            }
            cascade_source = stage
            continue
        if prior["status"] != "completed":
            stage_plan[stage] = {
                "decision": "run",
                "reason": "prior_stage_not_completed",
            }
            cascade_source = stage
            continue
        if prior["input_fingerprint"] != descriptor["input_fingerprint"]:
            stage_plan[stage] = {
                "decision": "run",
                "reason": "input_fingerprint_changed",
            }
            cascade_source = stage
            continue
        if (
            prior["implementation_fingerprint"]
            != descriptor["implementation_fingerprint"]
        ):
            stage_plan[stage] = {
                "decision": "run",
                "reason": "implementation_fingerprint_changed",
            }
            cascade_source = stage
            continue
        recorded_identities = {
            (record["root"], record["path"]) for record in prior["outputs"]
        }
        missing_required = [
            record
            for record in descriptor["required_outputs"]
            if (record["root"], record["path"]) not in recorded_identities
        ]
        if missing_required:
            first_missing = missing_required[0]
            stage_plan[stage] = {
                "decision": "run",
                "reason": (
                    "missing_required_output_record:"
                    f"{first_missing['root']}:{first_missing['path']}"
                ),
            }
            cascade_source = stage
            continue
        outputs_match, output_reason = verify_recorded_outputs(
            prior["outputs"],
            sample_root=sample_root,
            run_root=run_root,
        )
        if not outputs_match:
            stage_plan[stage] = {
                "decision": "run",
                "reason": output_reason,
            }
            cascade_source = stage
            continue
        stage_plan[stage] = {
            "decision": "reuse",
            "reason": "fingerprints_and_outputs_match",
        }

    return {
        "format": PIPELINE_REUSE_PLAN_SCHEMA,
        "selected_stages": [
            stage for stage in PIPELINE_STAGE_ORDER if stage in selected
        ],
        "reused_stages": [
            stage
            for stage in PIPELINE_STAGE_ORDER
            if stage_plan[stage]["decision"] == "reuse"
        ],
        "run_stages": [
            stage
            for stage in PIPELINE_STAGE_ORDER
            if stage_plan[stage]["decision"] == "run"
        ],
        "stages": copy.deepcopy(stage_plan),
    }


__all__ = [
    "PIPELINE_REUSE_PLAN_SCHEMA",
    "PIPELINE_STAGE_ORDER",
    "PIPELINE_STATE_SCHEMA",
    "benchmark_pipeline_state_lock",
    "digest_existing_outputs",
    "digest_output_file",
    "fingerprint_value",
    "load_benchmark_pipeline_state",
    "plan_benchmark_stage_reuse",
    "sha256_file",
    "validate_benchmark_pipeline_state",
    "verify_recorded_outputs",
    "write_benchmark_pipeline_state",
]
