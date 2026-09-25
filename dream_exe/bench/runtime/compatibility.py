"""Strict provenance contract for historical benchmark compatibility inputs.

Historical trajectories and actions are useful for separating executor changes
from upstream video-to-trajectory changes.  They are never an implicit cache:
callers must opt in with a complete manifest whose benchmark and artifact
digests are verified before any replay is allowed.
"""

from __future__ import annotations

import copy
import hashlib
import json
from collections.abc import Collection, Mapping
from pathlib import Path, PurePosixPath
from typing import Any

from ..contracts.schemas import canonical_sha256
from ..data.repository import BenchRepository


HISTORICAL_COMPATIBILITY_SCHEMA = (
    "dream-exe.historical-compatibility-manifest.v1"
)
HISTORICAL_COMPATIBILITY_RECEIPT_SCHEMA = (
    "dream-exe.historical-compatibility-receipt.v1"
)
_DIGEST_ROLES = ("trajectory", "objects", "action")
_HEX = frozenset("0123456789abcdef")
_ARCHIVED_TASK_FIELDS = (
    "controller",
    "use_ori",
    "pose_correction_mode",
    "pos_tol",
    "ori_tol",
    "max_correction_steps",
)
_TASK_EVALUATOR_DEFAULT_FIELDS = (
    "must_reach_min_correction_steps",
    "arm_pos_gain",
    "arm_ori_gain",
    "warm_start_steps",
    "position_dominate_correction_threshold_m",
    "enable_close_completion_gate",
    "enable_open_completion_gate",
    "close_gate_min_hold_steps",
    "close_gate_max_wait_steps",
    "close_gate_qpos_delta_min",
    "close_gate_qpos_settle_tol",
    "close_gate_settle_window",
    "close_gate_require_non_support_contact",
    "close_gate_contact_settle_steps",
    "close_gate_failure_policy",
)


def _exact(value: object, fields: set[str], label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be an object")
    item = {str(key): child for key, child in value.items()}
    if set(item) != fields:
        raise ValueError(
            f"{label} fields must be exactly {sorted(fields)}; "
            f"got {sorted(item)}"
        )
    return item


def _text(value: object, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value.strip()


def _digest(value: object, label: str) -> str:
    digest = _text(value, label)
    if len(digest) != 64 or any(char not in _HEX for char in digest):
        raise ValueError(f"{label} must be a lowercase SHA256 digest")
    return digest


def _safe_relative_path(value: object, label: str) -> PurePosixPath:
    raw = _text(value, label)
    path = PurePosixPath(raw)
    if path.is_absolute() or not path.parts or any(
        part in {"", ".", ".."} for part in path.parts
    ):
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    if path.as_posix() != raw:
        raise ValueError(f"{label} must be a normalized relative POSIX path")
    return path


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _artifact_record(value: object, label: str) -> dict[str, Any]:
    item = _exact(value, {"path", "size", "sha256"}, label)
    path = _safe_relative_path(item["path"], f"{label}.path")
    size = item["size"]
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"{label}.size must be a non-negative integer")
    return {
        "path": path.as_posix(),
        "size": size,
        "sha256": _digest(item["sha256"], f"{label}.sha256"),
    }


def build_historical_task_evaluator_config(
    *,
    base_config: Mapping[str, Any],
    archived_task_evidence: Mapping[str, Any],
    evaluator_defaults: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply an explicit historical task-replay contract to a base config.

    Table-3 execution parameters and Table-4 replay parameters may differ in
    historical pipelines.  This function requires both the values recorded in
    the archived task artifact and the otherwise-unrecorded evaluator defaults
    to be supplied explicitly; it never guesses them from current execution.
    """

    config = copy.deepcopy(dict(base_config))
    execution = config.get("execution")
    if not isinstance(execution, dict):
        raise TypeError("base_config.execution must be an object")
    evidence = dict(archived_task_evidence)
    missing_evidence = [name for name in _ARCHIVED_TASK_FIELDS if name not in evidence]
    if missing_evidence:
        raise ValueError(
            "archived task evidence is missing fields: "
            + ", ".join(missing_evidence)
        )
    defaults = dict(evaluator_defaults)
    if set(defaults) != set(_TASK_EVALUATOR_DEFAULT_FIELDS):
        raise ValueError(
            "evaluator default fields must be exactly "
            f"{sorted(_TASK_EVALUATOR_DEFAULT_FIELDS)}"
        )
    mode = str(evidence["pose_correction_mode"] or "").strip()
    if mode not in {"coupled", "position_dominate"}:
        raise ValueError("archived pose_correction_mode is invalid")
    controller = str(evidence["controller"] or "").strip()
    if not controller:
        raise ValueError("archived controller must be non-empty")
    if not isinstance(evidence["use_ori"], bool):
        raise TypeError("archived use_ori must be boolean")
    for name in ("pos_tol", "ori_tol"):
        value = evidence[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or value <= 0:
            raise ValueError(f"archived {name} must be positive")
    corrections = evidence["max_correction_steps"]
    if isinstance(corrections, bool) or not isinstance(corrections, int) or corrections < 0:
        raise ValueError("archived max_correction_steps must be non-negative")
    execution.update(defaults)
    execution.update({name: copy.deepcopy(evidence[name]) for name in _ARCHIVED_TASK_FIELDS})
    return config


def validate_historical_compatibility_manifest(
    document: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a complete, outcome-independent compatibility population."""

    root = _exact(
        document,
        {"format", "cohort", "source", "input", "entries"},
        "historical compatibility manifest",
    )
    if root["format"] != HISTORICAL_COMPATIBILITY_SCHEMA:
        raise ValueError("unexpected historical compatibility manifest format")
    cohort = _exact(
        root["cohort"],
        {"collection_id", "expected_count"},
        "historical compatibility manifest.cohort",
    )
    collection_id = _text(cohort["collection_id"], "cohort.collection_id")
    expected_count = cohort["expected_count"]
    if (
        isinstance(expected_count, bool)
        or not isinstance(expected_count, int)
        or expected_count < 1
    ):
        raise ValueError("cohort.expected_count must be a positive integer")
    source = _exact(
        root["source"],
        {"name", "revision", "description"},
        "historical compatibility manifest.source",
    )
    normalized_source = {
        key: _text(source[key], f"source.{key}")
        for key in ("name", "revision", "description")
    }
    input_spec = _exact(
        root["input"],
        {"kind", "reference_id"},
        "historical compatibility manifest.input",
    )
    if input_spec != {"kind": "reference", "reference_id": "wo_gt_depth"}:
        raise ValueError(
            "historical compatibility input must be reference/wo_gt_depth"
        )
    entries = root["entries"]
    if not isinstance(entries, list) or len(entries) != expected_count:
        raise ValueError(
            "manifest entries must contain exactly cohort.expected_count rows"
        )

    normalized_entries: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(entries):
        label = f"historical compatibility manifest.entries[{index}]"
        entry = _exact(
            raw,
            {
                "uid",
                "status",
                "reason",
                "video_sha256",
                "benchmark",
                "artifacts",
            },
            label,
        )
        uid = _text(entry["uid"], f"{label}.uid")
        if "/" in uid or "\\" in uid or uid in seen:
            raise ValueError(f"invalid or duplicate compatibility UID: {uid!r}")
        seen.add(uid)
        status = entry["status"]
        if status not in {"available", "unavailable"}:
            raise ValueError(f"{label}.status must be available or unavailable")
        benchmark = _exact(
            entry["benchmark"],
            {"case_sha256", "environment_sha256", "reference_sha256"},
            f"{label}.benchmark",
        )
        normalized_benchmark = {
            key: _digest(benchmark[key], f"{label}.benchmark.{key}")
            for key in (
                "case_sha256",
                "environment_sha256",
                "reference_sha256",
            )
        }
        reason = entry["reason"]
        artifacts = entry["artifacts"]
        if status == "available":
            if reason is not None:
                raise ValueError(f"{label}.reason must be null when available")
            artifact_map = _exact(
                artifacts,
                set(_DIGEST_ROLES),
                f"{label}.artifacts",
            )
            normalized_artifacts = {
                role: _artifact_record(
                    artifact_map[role], f"{label}.artifacts.{role}"
                )
                for role in _DIGEST_ROLES
            }
        else:
            reason = _text(reason, f"{label}.reason")
            if artifacts is not None:
                raise ValueError(f"{label}.artifacts must be null when unavailable")
            normalized_artifacts = None
        normalized_entries.append(
            {
                "uid": uid,
                "status": status,
                "reason": reason,
                "video_sha256": _digest(
                    entry["video_sha256"], f"{label}.video_sha256"
                ),
                "benchmark": normalized_benchmark,
                "artifacts": normalized_artifacts,
            }
        )

    return {
        "format": HISTORICAL_COMPATIBILITY_SCHEMA,
        "cohort": {
            "collection_id": collection_id,
            "expected_count": expected_count,
        },
        "source": normalized_source,
        "input": {"kind": "reference", "reference_id": "wo_gt_depth"},
        "entries": normalized_entries,
    }


def load_historical_compatibility_manifest(
    path: str | Path,
) -> dict[str, Any]:
    manifest_path = Path(path).expanduser().resolve(strict=True)
    with manifest_path.open(encoding="utf-8") as handle:
        document = json.load(handle)
    return validate_historical_compatibility_manifest(document)


def _resolve_artifact(root: Path, record: Mapping[str, Any], label: str) -> Path:
    candidate = root.joinpath(*PurePosixPath(str(record["path"])).parts)
    resolved = candidate.resolve(strict=True)
    if not resolved.is_relative_to(root):
        raise ValueError(f"{label} escapes the declared artifact root")
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a regular file: {resolved}")
    if resolved.stat().st_size != record["size"]:
        raise ValueError(f"{label} size mismatch: {resolved}")
    if _sha256_file(resolved) != record["sha256"]:
        raise ValueError(f"{label} digest mismatch: {resolved}")
    return resolved


def _verify_entry(
    *,
    entry: Mapping[str, Any],
    root: Path,
    repository: BenchRepository,
) -> dict[str, Any]:
    uid = str(entry["uid"])
    case = repository.load_case(uid)
    environment = repository.load_environment(uid, verify_files=True)
    # The compatibility route consumes the reference video, not the canonical
    # GT depth stack.  Verify that video byte-for-byte without rehashing every
    # large depth shard in the reference package.
    reference = repository.load_reference(uid, verify_files=False)
    actual_benchmark = {
        "case_sha256": canonical_sha256(case),
        "environment_sha256": canonical_sha256(environment.manifest),
        "reference_sha256": canonical_sha256(reference.manifest),
    }
    if actual_benchmark != entry["benchmark"]:
        raise ValueError(f"benchmark identity mismatch for {uid}")
    if reference.manifest["video"]["sha256"] != entry["video_sha256"]:
        raise ValueError(f"reference video identity mismatch for {uid}")
    video_record = reference.manifest["video"]
    if reference.video.stat().st_size != video_record["size"]:
        raise ValueError(f"reference video size mismatch for {uid}")
    if _sha256_file(reference.video) != video_record["sha256"]:
        raise ValueError(f"reference video digest mismatch for {uid}")
    paths = None
    if entry["status"] == "available":
        paths = {
            role: _resolve_artifact(
                root,
                entry["artifacts"][role],
                f"{uid}.{role}",
            ).as_posix()
            for role in _DIGEST_ROLES
        }
    return {
        "uid": uid,
        "status": entry["status"],
        "reason": entry["reason"],
        "artifacts": paths,
    }


def resolve_historical_compatibility_entry(
    *,
    manifest_path: str | Path,
    artifact_root: str | Path,
    repository: BenchRepository,
    uid: str,
) -> dict[str, Any]:
    """Resolve one entry only after checking its benchmark and file digests."""

    manifest = load_historical_compatibility_manifest(manifest_path)
    matches = [entry for entry in manifest["entries"] if entry["uid"] == uid]
    if len(matches) != 1:
        raise KeyError(f"compatibility manifest has no unique entry for {uid!r}")
    root = Path(artifact_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    resolved = _verify_entry(entry=matches[0], root=root, repository=repository)
    return {
        "format": HISTORICAL_COMPATIBILITY_RECEIPT_SCHEMA,
        "manifest_sha256": canonical_sha256(manifest),
        "collection_id": manifest["cohort"]["collection_id"],
        "source": manifest["source"],
        "input": manifest["input"],
        **resolved,
    }


def verify_historical_compatibility_manifest(
    *,
    manifest_path: str | Path,
    artifact_root: str | Path,
    repository: BenchRepository,
    expected_uids: Collection[str] | None = None,
) -> dict[str, Any]:
    """Verify benchmark identities and every declared historical artifact."""

    manifest = load_historical_compatibility_manifest(manifest_path)
    root = Path(artifact_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    entries = manifest["entries"]
    observed_uids = {str(entry["uid"]) for entry in entries}
    if expected_uids is not None:
        expected = {str(uid) for uid in expected_uids}
        if observed_uids != expected:
            raise ValueError(
                "compatibility cohort UID mismatch: "
                f"missing={sorted(expected - observed_uids)}, "
                f"extra={sorted(observed_uids - expected)}"
            )

    verified_entries: list[dict[str, Any]] = []
    for entry in entries:
        verified_entries.append(
            _verify_entry(entry=entry, root=root, repository=repository)
        )

    available = sum(item["status"] == "available" for item in entries)
    return {
        "format": HISTORICAL_COMPATIBILITY_RECEIPT_SCHEMA,
        "manifest_sha256": canonical_sha256(manifest),
        "collection_id": manifest["cohort"]["collection_id"],
        "source": manifest["source"],
        "input": manifest["input"],
        "population": len(entries),
        "available": available,
        "unavailable": len(entries) - available,
        "entries": verified_entries,
    }


__all__ = [
    "HISTORICAL_COMPATIBILITY_RECEIPT_SCHEMA",
    "HISTORICAL_COMPATIBILITY_SCHEMA",
    "build_historical_task_evaluator_config",
    "load_historical_compatibility_manifest",
    "resolve_historical_compatibility_entry",
    "validate_historical_compatibility_manifest",
    "verify_historical_compatibility_manifest",
]
