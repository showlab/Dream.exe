"""Load local VLM credentials without placing secrets in saved artifacts."""

from __future__ import annotations

import json
import os
import stat
from pathlib import Path
from typing import Any


_MAX_CREDENTIAL_FILE_BYTES = 64 * 1024
_PLACEHOLDERS = {
    "...",
    "change-me",
    "replace-me",
    "your-api-key",
    "your_api_key",
}


def _text(value: Any, *, label: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{label} must be a string")
    clean = value.strip()
    if not allow_empty and not clean:
        raise ValueError(f"{label} must be non-empty")
    return clean


def _safe_profile(value: Any) -> str:
    profile = _text(value, label="credential profile")
    if profile in {".", ".."} or any(
        character in profile for character in ("/", "\\", "\x00")
    ):
        raise ValueError("credential profile must be a safe identifier")
    return profile


def _read_document(path: str | Path) -> tuple[Path, dict[str, Any]]:
    source = Path(path).expanduser().resolve()
    try:
        metadata = source.stat()
    except FileNotFoundError as error:
        raise FileNotFoundError(f"VLM credentials file not found: {source}") from error
    if not stat.S_ISREG(metadata.st_mode):
        raise ValueError(f"VLM credentials path must be a regular file: {source}")
    if metadata.st_size > _MAX_CREDENTIAL_FILE_BYTES:
        raise ValueError("VLM credentials file exceeds 64 KiB")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError("VLM credentials file is not valid JSON") from error
    if not isinstance(payload, dict) or set(payload) != {"vlm"}:
        raise ValueError("VLM credentials file must contain only the 'vlm' object")
    return source, payload


def load_vlm_credentials(
    path: str | Path,
    *,
    profile: str,
) -> dict[str, str]:
    """Return one strict local provider record without logging its API key."""

    selected = _safe_profile(profile)
    source, payload = _read_document(path)
    profiles = payload["vlm"]
    if not isinstance(profiles, dict) or not profiles:
        raise ValueError("VLM credentials 'vlm' object must not be empty")
    unknown_profiles = [
        name for name in profiles if not isinstance(name, str) or not name.strip()
    ]
    if unknown_profiles:
        raise ValueError("VLM credential profile names must be non-empty strings")
    if selected not in profiles:
        raise KeyError(
            f"VLM credential profile {selected!r} is not declared; available: "
            + ", ".join(sorted(profiles))
        )
    record = profiles[selected]
    if not isinstance(record, dict) or set(record) != {"base_url", "api_key"}:
        raise ValueError(
            f"VLM credential profile {selected!r} must contain only "
            "'base_url' and 'api_key'"
        )
    base_url = _text(
        record["base_url"],
        label=f"VLM credential profile {selected!r} base_url",
        allow_empty=True,
    )
    api_key = _text(
        record["api_key"],
        label=f"VLM credential profile {selected!r} api_key",
        allow_empty=True,
    )
    if api_key.lower() in _PLACEHOLDERS:
        raise ValueError(
            f"VLM credential profile {selected!r} still contains a placeholder API key"
        )
    if api_key and source.stat().st_mode & (stat.S_IRWXG | stat.S_IRWXO):
        raise PermissionError(
            "VLM credentials containing an API key must not be accessible "
            f"to group or other users; run: chmod 600 {source}"
        )
    return {
        "base_url": base_url,
        "api_key": api_key,
    }


def configured_credentials_path(explicit: str | Path | None) -> str | None:
    """Resolve the CLI value, then the process-level file binding."""

    selected = str(explicit or "").strip()
    if selected:
        return selected
    from_environment = str(
        os.environ.get("DREAM_EXE_CREDENTIALS_FILE", "") or ""
    ).strip()
    return from_environment or None


__all__ = ["configured_credentials_path", "load_vlm_credentials"]
