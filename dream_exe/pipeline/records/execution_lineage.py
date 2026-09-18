"""Path-neutral provenance for one resolved benchmark execution request."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ...artifacts.layout import execution_artifact_paths


EXECUTION_INPUT_LINEAGE_FILENAME = execution_artifact_paths(Path("."))[
    "execution_inputs"
].name
EXECUTION_INPUT_LINEAGE_SCHEMA = "dream-exe.execution-input-lineage"

_EXCLUDED_EXECUTION_INPUT_SCOPE = (
    "external_dataset_and_asset_bytes",
    "restored_simulator_state_bytes",
    "simulator_provider_and_runtime_source",
)


def _canonical_json_bytes(value: Any) -> bytes:
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _logical_location(
    path: Path,
    *,
    sample_root: Path,
    role: str,
) -> tuple[str, str]:
    try:
        relative = path.relative_to(sample_root)
    except ValueError:
        return "external", f"<external>/{role}/{path.name}"
    return "sample", relative.as_posix()


def _file_identity(
    value: Any,
    *,
    sample_root: Path,
    role: str,
    required: bool,
) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError(f"execution lineage requires {role}")
        return {"status": "not_supplied"}
    path = Path(text).expanduser().resolve()
    if not path.is_file():
        if required:
            raise FileNotFoundError(
                f"execution lineage {role} is not a regular file: {path}"
            )
        return {"status": "not_available"}
    scope, logical_path = _logical_location(
        path,
        sample_root=sample_root,
        role=role,
    )
    return {
        "status": "verified",
        "scope": scope,
        "logical_path": logical_path,
        "size_bytes": int(path.stat().st_size),
        "sha256": _sha256_file(path),
    }


def _config_source_identities(
    request: Mapping[str, Any],
    *,
    sample_root: Path,
) -> dict[str, dict[str, Any]]:
    meta = dict(request.get("execution_config_meta", {}) or {})
    identities: dict[str, dict[str, Any]] = {}
    for key, role in (
        ("source", "execution_config"),
        ("override_source", "execution_overrides"),
    ):
        text = str(meta.get(key, "") or "").strip()
        if not text or text.startswith("<"):
            continue
        identity = _file_identity(
            text,
            sample_root=sample_root,
            role=role,
            required=False,
        )
        if identity.get("status") == "verified":
            identities[role] = identity
    return identities


def _path_neutral_execution_config(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    config = copy.deepcopy(dict(request.get("execution_config", {}) or {}))
    inputs = config.setdefault("input", {})
    if not isinstance(inputs, dict):
        raise TypeError("resolved execution config input must be a mapping")
    inputs.pop("traj_path", None)
    inputs.pop("action_path", None)
    inputs["trajectory_input_role"] = "trajectory"
    inputs["action_input_role"] = "action"
    _reject_absolute_config_strings(config)
    return config


def _reject_absolute_config_strings(
    value: Any,
    *,
    location: str = "resolved_execution_config",
) -> None:
    if isinstance(value, Mapping):
        for key, item in value.items():
            _reject_absolute_config_strings(
                item,
                location=f"{location}.{key}",
            )
        return
    if isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            _reject_absolute_config_strings(
                item,
                location=f"{location}[{index}]",
            )
        return
    if isinstance(value, str) and Path(value).expanduser().is_absolute():
        raise ValueError(
            "execution lineage cannot publish an unmodeled absolute "
            f"config value at {location}"
        )


def capture_execution_input_lineage(
    request: Mapping[str, Any],
) -> dict[str, Any]:
    """Hash every bench-level file in a resolved current-layout request.

    The returned payload contains no absolute path. Callers should capture it
    immediately before execution, verify it afterward, and publish it only for
    a successful execution. Simulator provider source, external dataset/assets,
    and restored-state bytes remain bound by initialization/provider evidence,
    not by this document.
    """

    sample_text = str(request.get("sample_dir", "") or "").strip()
    if not sample_text:
        raise ValueError("execution lineage requires sample_dir")
    sample_root = Path(sample_text).expanduser().resolve()
    config = _path_neutral_execution_config(request)
    mode = (
        str(dict(config.get("execution", {}) or {}).get("mode", "action") or "action")
        .strip()
        .lower()
    )
    if mode not in {"action", "frame_traj"}:
        raise ValueError(f"unsupported execution lineage mode: {mode}")

    inputs = {
        "simulator_config": _file_identity(
            request.get("simulator_config_path", ""),
            sample_root=sample_root,
            role="simulator_config",
            required=True,
        ),
        "scene_override": _file_identity(
            request.get("scene_override_path", ""),
            sample_root=sample_root,
            role="scene_override",
            required=False,
        ),
        "trajectory": _file_identity(
            request.get("trajectory_path", ""),
            sample_root=sample_root,
            role="trajectory",
            required=True,
        ),
        "action": _file_identity(
            request.get("action_path", ""),
            sample_root=sample_root,
            role="action",
            required=(mode == "action"),
        ),
    }
    inputs.update(
        _config_source_identities(
            request,
            sample_root=sample_root,
        )
    )
    config_sha256 = hashlib.sha256(_canonical_json_bytes(config)).hexdigest()
    return {
        "format": EXECUTION_INPUT_LINEAGE_SCHEMA,
        "uid": str(request.get("uid", "") or ""),
        "run_key": str(request.get("run_key", "") or ""),
        "gen_model": str(request.get("gen_model", "") or ""),
        "execution_mode": mode,
        "scope": {
            "included": [
                "resolved_execution_config_and_sources",
                "simulator_config",
                "scene_override",
                "trajectory",
                "action_when_required",
            ],
            "excluded": list(_EXCLUDED_EXECUTION_INPUT_SCOPE),
        },
        "resolved_execution_config": config,
        "resolved_execution_config_sha256": config_sha256,
        "inputs": dict(sorted(inputs.items())),
    }


def verify_execution_input_lineage(
    captured: Mapping[str, Any],
    request: Mapping[str, Any],
) -> None:
    """Fail if any resolved input or control value changed during execution."""

    observed = capture_execution_input_lineage(request)
    if _canonical_json_bytes(dict(captured)) != _canonical_json_bytes(observed):
        raise RuntimeError("execution inputs changed during execution")


def publish_execution_input_lineage(
    output_dir: str | os.PathLike[str],
    payload: Mapping[str, Any],
) -> Path:
    """Atomically publish one verified path-neutral lineage document."""

    root = Path(output_dir).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    destination = execution_artifact_paths(root)["execution_inputs"]
    content = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    )
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=root,
    )
    temporary_path = Path(temporary)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_path, destination)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return destination


__all__ = [
    "EXECUTION_INPUT_LINEAGE_FILENAME",
    "EXECUTION_INPUT_LINEAGE_SCHEMA",
    "capture_execution_input_lineage",
    "publish_execution_input_lineage",
    "verify_execution_input_lineage",
]
