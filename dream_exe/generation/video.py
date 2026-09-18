"""Provider-neutral execution of one local image-to-video request.

This module deliberately knows nothing about benchmark layouts.  A caller
supplies one image, one prompt, one destination, and an injected local model
backend.  The backend writes to a private staging path; this module validates
and atomically publishes the video plus a credential-free provenance sidecar.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
import time
from typing import Any, Protocol


VIDEO_GENERATION_SCHEMA = "dream-exe.local-video-generation"
VIDEO_GENERATION_IMPLEMENTATION = "dream_exe.generation.video.local-image-to-video"
IMAGE_TO_VIDEO_BACKEND_CONTRACT_VERSION = "image_to_video_backend"
MAX_PROMPT_CHARS = 4096
MAX_GENERATION_SIDECAR_BYTES = 4 * 1024 * 1024
_BUFFER_SIZE = 1024 * 1024
_REDACTED = "[REDACTED]"
_GENERATION_SIDECAR_FIELDS = frozenset(
    {
        "format",
        "implementation",
        "status",
        "backend",
        "image_path",
        "image_sha256",
        "prompt_sha256",
        "output_video",
        "output_sidecar",
        "seed",
        "parameters",
        "backend_identity",
        "backend_result",
        "video_sha256",
        "video_size",
    }
)


class ImageToVideoBackend(Protocol):
    """Replaceable local/open-source image-to-video backend."""

    @property
    def identity(self) -> Mapping[str, Any]: ...

    def generate(
        self,
        *,
        image_path: Path,
        prompt: str,
        output_path: Path,
        seed: int,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, Any]: ...


class BaseImageToVideoBackend(ABC):
    """Minimal template shared by in-process and API video generators."""

    provider_kind = "external"
    backend_id = ""
    contract_version = IMAGE_TO_VIDEO_BACKEND_CONTRACT_VERSION

    @property
    def identity(self) -> Mapping[str, Any]:
        """Return credential-free identity used by generation sidecars."""

        if not str(self.backend_id or "").strip():
            raise ValueError("video-generation backend_id must be non-empty")
        return {
            "provider_kind": str(self.provider_kind),
            "backend_id": str(self.backend_id),
            "contract_version": str(self.contract_version),
        }

    @abstractmethod
    def generate(
        self,
        *,
        image_path: Path,
        prompt: str,
        output_path: Path,
        seed: int,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        """Write one MP4 to ``output_path`` and return safe provenance."""


class PollingImageToVideoBackend(BaseImageToVideoBackend):
    """Bounded submit/poll/download template for asynchronous video APIs.

    Provider adapters implement only the three transport hooks.  This class
    owns finite retries, a monotonic timeout, failure-state handling, and an
    atomic download handoff into the pipeline-owned staging path.
    """

    success_states = frozenset({"completed", "succeeded", "success"})
    failure_states = frozenset({"cancelled", "canceled", "failed", "error"})

    def __init__(
        self,
        *,
        timeout_seconds: float = 900.0,
        poll_interval_seconds: float = 2.0,
        max_transport_attempts: int = 3,
    ) -> None:
        timeout = float(timeout_seconds)
        interval = float(poll_interval_seconds)
        attempts = int(max_transport_attempts)
        if timeout <= 0 or not timeout < float("inf"):
            raise ValueError("timeout_seconds must be finite and positive")
        if interval < 0 or not interval < float("inf"):
            raise ValueError("poll_interval_seconds must be finite and non-negative")
        if attempts < 1:
            raise ValueError("max_transport_attempts must be at least 1")
        self.timeout_seconds = timeout
        self.poll_interval_seconds = interval
        self.max_transport_attempts = attempts

    @abstractmethod
    def submit(
        self,
        *,
        image_path: Path,
        prompt: str,
        seed: int,
        parameters: Mapping[str, Any],
    ) -> Any:
        """Submit a request and return a provider-owned job handle."""

    @abstractmethod
    def poll(self, job: Any) -> Mapping[str, Any]:
        """Return a status mapping containing a string ``status`` field."""

    @abstractmethod
    def download(
        self,
        job: Any,
        status: Mapping[str, Any],
        output_path: Path,
    ) -> Mapping[str, Any] | None:
        """Download a completed job to the supplied temporary MP4 path."""

    def _attempt(self, operation: str, callback: Any) -> Any:
        last_error: Exception | None = None
        for attempt in range(1, self.max_transport_attempts + 1):
            try:
                return callback()
            except Exception as error:  # provider errors are normalized here
                last_error = error
                if attempt == self.max_transport_attempts:
                    break
        assert last_error is not None
        raise RuntimeError(
            f"video API {operation} failed after "
            f"{self.max_transport_attempts} attempts: {last_error}"
        ) from last_error

    def generate(
        self,
        *,
        image_path: Path,
        prompt: str,
        output_path: Path,
        seed: int,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        job = self._attempt(
            "submit",
            lambda: self.submit(
                image_path=image_path,
                prompt=prompt,
                seed=seed,
                parameters=dict(parameters),
            ),
        )
        started = time.monotonic()
        polls = 0
        status: Mapping[str, Any]
        while True:
            if time.monotonic() - started > self.timeout_seconds:
                raise TimeoutError(
                    f"video API job timed out after {self.timeout_seconds:g} seconds"
                )
            status = self._attempt("poll", lambda: self.poll(job))
            if not isinstance(status, Mapping):
                raise TypeError("video API poll(...) must return a mapping")
            state = str(status.get("status", "") or "").strip().lower()
            if not state:
                raise ValueError("video API poll result requires a status field")
            polls += 1
            if state in self.success_states:
                break
            if state in self.failure_states:
                raise RuntimeError(f"video API job entered failure state {state!r}")
            if self.poll_interval_seconds:
                time.sleep(self.poll_interval_seconds)

        descriptor, temporary_name = tempfile.mkstemp(
            prefix=f".{output_path.stem}.",
            suffix=".download.mp4",
            dir=output_path.parent,
        )
        os.close(descriptor)
        temporary = Path(temporary_name)
        temporary.unlink()
        try:
            download_result = self._attempt(
                "download",
                lambda: self.download(job, status, temporary),
            )
            _require_regular_file(temporary, label="downloaded generated video")
            if temporary.stat().st_size <= 0:
                raise ValueError("downloaded generated video is empty")
            temporary.replace(output_path)
        finally:
            if temporary.exists() and not temporary.is_symlink():
                temporary.unlink()
        if download_result is not None and not isinstance(download_result, Mapping):
            raise TypeError("video API download(...) must return a mapping or None")
        return {
            "job_status": str(status.get("status")),
            "poll_count": polls,
            "download": dict(download_result or {}),
        }


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(_BUFFER_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _sha256_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _require_regular_file(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} not found: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} must not be a symlink")
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"{label} must be a regular file")


def _require_real_directory(path: Path, *, label: str) -> None:
    try:
        metadata = path.lstat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"{label} not found: {path}") from error
    if stat.S_ISLNK(metadata.st_mode):
        raise ValueError(f"{label} must not be a symlink")
    if not stat.S_ISDIR(metadata.st_mode):
        raise ValueError(f"{label} must be a directory")


def _safe_provenance(value: Any) -> Any:
    if isinstance(value, Mapping):
        output: dict[str, Any] = {}
        for key, item in value.items():
            name = str(key)
            normalized = re.sub(r"[^a-z0-9]+", "_", name.casefold())
            if any(
                marker in normalized
                for marker in ("api_key", "password", "secret", "token")
            ):
                output[name] = _REDACTED
            else:
                output[name] = _safe_provenance(item)
        return output
    if isinstance(value, (list, tuple)):
        return [_safe_provenance(item) for item in value]
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, str):
        return re.sub(
            r"(?i)\bbearer\s+[^\s,;]+",
            f"Bearer {_REDACTED}",
            value,
        )
    if value is None or isinstance(value, (bool, int, float)):
        return value
    return repr(value)


def _write_staged_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(
            _safe_provenance(payload),
            indent=2,
            ensure_ascii=True,
            sort_keys=True,
            allow_nan=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _strict_json_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_json_constant(value: str) -> Any:
    raise ValueError(f"invalid JSON constant: {value}")


def _load_existing_sidecar(path: Path) -> dict[str, Any]:
    _require_regular_file(path, label="existing generation sidecar")
    with path.open("rb") as stream:
        encoded = stream.read(MAX_GENERATION_SIDECAR_BYTES + 1)
    if len(encoded) > MAX_GENERATION_SIDECAR_BYTES:
        raise ValueError(
            f"existing generation sidecar exceeds {MAX_GENERATION_SIDECAR_BYTES} bytes"
        )
    try:
        text = encoded.decode("utf-8")
    except UnicodeDecodeError as error:
        raise ValueError("existing generation sidecar is not valid UTF-8") from error
    try:
        payload = json.loads(
            text,
            object_pairs_hook=_strict_json_object,
            parse_constant=_reject_json_constant,
        )
    except (json.JSONDecodeError, RecursionError) as error:
        detail = error.msg if isinstance(error, json.JSONDecodeError) else str(error)
        raise ValueError(
            f"existing generation sidecar is invalid JSON: {detail}"
        ) from error
    if not isinstance(payload, Mapping):
        raise ValueError("existing generation sidecar must be a JSON object")
    return dict(payload)


def _strict_json_bytes(value: Any, *, label: str) -> bytes:
    try:
        return json.dumps(
            value,
            ensure_ascii=True,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except (TypeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON data") from error


def _strict_json_equal(left: Any, right: Any) -> bool:
    return _strict_json_bytes(left, label="generation value") == _strict_json_bytes(
        right,
        label="generation value",
    )


def _lexists(path: Path) -> bool:
    try:
        path.lstat()
    except FileNotFoundError:
        return False
    return True


def _backend_identity(backend: ImageToVideoBackend | None) -> dict[str, Any]:
    if backend is None:
        raise RuntimeError(
            "a local image-to-video backend is required to verify "
            "existing generation provenance"
        )
    identity = backend.identity
    if not isinstance(identity, Mapping) or not identity:
        raise TypeError("video-generation backend identity must be a mapping")
    normalized = _safe_provenance(dict(identity))
    if not isinstance(normalized, dict) or not normalized:
        raise TypeError("video-generation backend identity must be a mapping")
    return normalized


def _validate_existing_generation(
    *,
    output_video: Path,
    output_sidecar: Path,
    request: Mapping[str, Any],
    backend: ImageToVideoBackend | None,
) -> dict[str, Any]:
    _require_regular_file(output_video, label="existing generated video")
    payload = _load_existing_sidecar(output_sidecar)
    if set(payload) != _GENERATION_SIDECAR_FIELDS:
        missing = sorted(_GENERATION_SIDECAR_FIELDS.difference(payload))
        unsupported = sorted(set(payload).difference(_GENERATION_SIDECAR_FIELDS))
        details: list[str] = []
        if missing:
            details.append("missing " + ", ".join(missing))
        if unsupported:
            details.append("unsupported " + ", ".join(unsupported))
        raise ValueError(
            "existing generation sidecar fields are invalid: " + "; ".join(details)
        )
    if _safe_provenance(payload) != payload:
        raise ValueError(
            "existing generation sidecar contains unsafe provenance values"
        )
    expected = {**dict(request), "status": "completed"}
    for field, expected_value in expected.items():
        if not _strict_json_equal(payload[field], expected_value):
            raise ValueError(
                "existing generation sidecar does not match current "
                f"request field {field}"
            )

    backend_identity = payload["backend_identity"]
    if not isinstance(backend_identity, Mapping) or not backend_identity:
        raise ValueError(
            "existing generation sidecar backend_identity must be a non-empty object"
        )
    if not _strict_json_equal(backend_identity, _backend_identity(backend)):
        raise ValueError(
            "existing generation sidecar does not match current request "
            "field backend_identity"
        )
    if not isinstance(payload["parameters"], Mapping):
        raise ValueError("existing generation sidecar parameters must be an object")
    if not isinstance(payload["backend_result"], Mapping):
        raise ValueError("existing generation sidecar backend_result must be an object")

    video_size = payload.get("video_size")
    if isinstance(video_size, bool) or not isinstance(video_size, int):
        raise ValueError("existing generation sidecar video_size must be an integer")
    actual_size = output_video.stat().st_size
    if actual_size <= 0:
        raise ValueError("existing generated video is empty")
    if video_size != actual_size:
        raise ValueError("existing generation sidecar video_size does not match video")
    video_sha256 = payload.get("video_sha256")
    if not isinstance(video_sha256, str) or video_sha256 != _sha256_file(output_video):
        raise ValueError(
            "existing generation sidecar video_sha256 does not match video"
        )
    result = dict(payload)
    result["status"] = "skipped_existing"
    return result


def _clean_prompt(prompt: str) -> str:
    if not isinstance(prompt, str):
        raise TypeError("prompt must be a string")
    cleaned = prompt.strip()
    if not cleaned:
        raise ValueError("prompt must be non-empty")
    if len(cleaned) > MAX_PROMPT_CHARS:
        raise ValueError(
            f"prompt is {len(cleaned)} characters, max is {MAX_PROMPT_CHARS}"
        )
    return cleaned


def _clean_seed(seed: int) -> int:
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise TypeError("seed must be an integer")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    return seed


def _publish_pair(
    *,
    staged_video: Path,
    staged_sidecar: Path,
    output_video: Path,
    output_sidecar: Path,
    force: bool,
) -> None:
    backups: dict[Path, Path] = {}
    installed: list[Path] = []
    try:
        for destination in (output_video, output_sidecar):
            if not destination.exists():
                continue
            _require_regular_file(destination, label="existing generation output")
            if not force:
                raise FileExistsError(
                    f"generation output already exists: {destination}"
                )
            descriptor, backup_name = tempfile.mkstemp(
                prefix=f".{destination.name}.",
                suffix=".backup",
                dir=destination.parent,
            )
            os.close(descriptor)
            backup = Path(backup_name)
            backup.unlink()
            destination.replace(backup)
            backups[destination] = backup

        staged_video.replace(output_video)
        installed.append(output_video)
        staged_sidecar.replace(output_sidecar)
        installed.append(output_sidecar)
    except BaseException:
        for destination in reversed(installed):
            if destination.exists() and not destination.is_symlink():
                destination.unlink()
        for destination, backup in backups.items():
            if backup.exists():
                backup.replace(destination)
        raise
    else:
        for backup in backups.values():
            if backup.exists():
                backup.unlink()


def generate_video(
    *,
    image_path: str | Path,
    prompt: str,
    output_video: str | Path,
    output_sidecar: str | Path,
    backend: ImageToVideoBackend | None,
    backend_name: str,
    seed: int = 0,
    parameters: Mapping[str, Any] | None = None,
    force: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate and publish one local video without an intermediate export.

    ``dry_run`` validates the input and reports the exact destination without
    creating a directory or constructing a model backend.
    """

    source = Path(image_path).expanduser().resolve(strict=False)
    destination = Path(output_video).expanduser().resolve(strict=False)
    sidecar = Path(output_sidecar).expanduser().resolve(strict=False)
    _require_regular_file(source, label="generation image")
    clean_prompt = _clean_prompt(prompt)
    clean_seed = _clean_seed(seed)
    clean_backend = str(backend_name or "").strip()
    if not clean_backend:
        raise ValueError("backend_name must be non-empty")
    options = dict(parameters or {})

    if destination == sidecar:
        raise ValueError("output_video and output_sidecar must be different")
    if destination.suffix.casefold() != ".mp4":
        raise ValueError("output_video must use the .mp4 suffix")
    if destination.parent != sidecar.parent:
        raise ValueError("video and sidecar must share one destination directory")

    record: dict[str, Any] = {
        "format": VIDEO_GENERATION_SCHEMA,
        "implementation": VIDEO_GENERATION_IMPLEMENTATION,
        "status": "planned" if dry_run else "pending",
        "backend": clean_backend,
        "image_path": source.as_posix(),
        "image_sha256": _sha256_file(source),
        "prompt_sha256": _sha256_text(clean_prompt),
        "output_video": destination.as_posix(),
        "output_sidecar": sidecar.as_posix(),
        "seed": clean_seed,
        "parameters": _safe_provenance(options),
    }

    if dry_run:
        return record
    existing = [path for path in (destination, sidecar) if _lexists(path)]
    if existing and not force:
        if len(existing) != 2:
            missing = sidecar if destination.exists() else destination
            raise FileExistsError(
                "existing generation output requires a matching video and "
                f"sidecar pair; missing: {missing}"
            )
        return _validate_existing_generation(
            output_video=destination,
            output_sidecar=sidecar,
            request=record,
            backend=backend,
        )
    for path in existing:
        _require_regular_file(path, label="existing generation output")
    if backend is None:
        raise RuntimeError("a local image-to-video backend is required")

    destination.parent.mkdir(parents=True, exist_ok=True)
    _require_real_directory(destination.parent, label="generation output directory")
    video_descriptor, video_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.",
        suffix=".mp4",
        dir=destination.parent,
    )
    os.close(video_descriptor)
    staged_video = Path(video_name)
    staged_video.unlink()
    sidecar_descriptor, sidecar_name = tempfile.mkstemp(
        prefix=f".{sidecar.name}.",
        suffix=".tmp",
        dir=sidecar.parent,
    )
    os.close(sidecar_descriptor)
    staged_sidecar = Path(sidecar_name)

    try:
        result = backend.generate(
            image_path=source,
            prompt=clean_prompt,
            output_path=staged_video,
            seed=clean_seed,
            parameters=options,
        )
        if not isinstance(result, Mapping):
            raise TypeError("video-generation backend must return a mapping")
        _require_regular_file(staged_video, label="staged generated video")
        if staged_video.stat().st_size <= 0:
            raise ValueError("staged generated video is empty")
        final_record = {
            **record,
            "status": "completed",
            "backend_identity": _backend_identity(backend),
            "backend_result": dict(result),
            "video_sha256": _sha256_file(staged_video),
            "video_size": staged_video.stat().st_size,
        }
        _write_staged_json(staged_sidecar, final_record)
        _publish_pair(
            staged_video=staged_video,
            staged_sidecar=staged_sidecar,
            output_video=destination,
            output_sidecar=sidecar,
            force=bool(force),
        )
        return final_record
    finally:
        for staged in (staged_video, staged_sidecar):
            if staged.exists() and not staged.is_symlink():
                staged.unlink()


__all__ = [
    "BaseImageToVideoBackend",
    "IMAGE_TO_VIDEO_BACKEND_CONTRACT_VERSION",
    "ImageToVideoBackend",
    "MAX_GENERATION_SIDECAR_BYTES",
    "MAX_PROMPT_CHARS",
    "PollingImageToVideoBackend",
    "VIDEO_GENERATION_IMPLEMENTATION",
    "VIDEO_GENERATION_SCHEMA",
    "generate_video",
]
