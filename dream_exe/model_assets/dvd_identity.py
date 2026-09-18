"""DVD model-family identity and immutable model-asset attestation.

These values describe reproducibility claims only.  They import no model
runtime.  DVD declarations remain backward compatible when no asset hashes
are supplied; an explicit, complete hash pair can be promoted to a verified
runtime claim only after both files are read.
"""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any


DVD_MODEL_FAMILY = "dream_exe_project_finetuned_dvd"
DVD_OFFICIAL_MODEL_FAMILY = "upstream_official_dvd"
DVD_MODEL_FAMILIES = (
    DVD_OFFICIAL_MODEL_FAMILY,
    DVD_MODEL_FAMILY,
)
DVD_PROVENANCE_KIND_BY_FAMILY = {
    DVD_OFFICIAL_MODEL_FAMILY: "upstream_official",
    DVD_MODEL_FAMILY: "project_finetuned",
}
DVD_ASSET_ATTESTATION_FIELD = "asset_attestation"
DVD_PROVENANCE_VALIDATION = {
    "status": "declaration_only",
    "live_checkpoint_finetune_gate": "not_implemented",
}
DVD_PROVENANCE_PENDING_VALIDATION = {
    "status": "pending_runtime_verification",
    "asset_integrity_gate": "pending",
    "verification_scope": "caller_declared_asset_bytes",
    "live_checkpoint_finetune_gate": "not_implemented",
}
_DVD_ATTESTATION_KEYS = {
    "checkpoint_sha256",
    "model_config_sha256",
}
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


def normalize_dvd_asset_attestation(
    value: Any,
    *,
    source: str,
) -> dict[str, str]:
    """Normalize one complete caller-supplied DVD SHA-256 declaration."""

    if not isinstance(value, Mapping):
        raise ValueError(f"{source} must be an object")
    payload = dict(value)
    unknown = sorted(set(payload).difference(_DVD_ATTESTATION_KEYS))
    if unknown:
        raise ValueError(f"{source} has unsupported fields: " + ", ".join(unknown))
    missing = sorted(_DVD_ATTESTATION_KEYS.difference(payload))
    if missing:
        raise ValueError(
            f"{source} must provide checkpoint_sha256 and "
            "model_config_sha256 together; missing: " + ", ".join(missing)
        )
    output: dict[str, str] = {}
    for field in sorted(_DVD_ATTESTATION_KEYS):
        digest = str(payload.get(field, "") or "").strip().lower()
        if not _SHA256.fullmatch(digest):
            raise ValueError(f"{source}.{field} must be 64 hexadecimal SHA-256 digits")
        output[field] = digest
    return output


def _verified_validation(
    value: Any,
    *,
    attestation: Mapping[str, str],
) -> dict[str, Any] | None:
    """Return a canonical verified record when every digest agrees."""

    if not isinstance(value, Mapping):
        return None
    if (
        value.get("status") != "verified"
        or value.get("asset_integrity_gate") != "passed"
        or value.get("verification_scope") != "caller_declared_asset_bytes"
        or value.get("live_checkpoint_finetune_gate") != "not_implemented"
        or value.get("algorithm") != "sha256"
    ):
        return None
    raw_assets = value.get("assets")
    if not isinstance(raw_assets, Mapping):
        return None
    expected_by_asset = {
        "checkpoint": attestation["checkpoint_sha256"],
        "model_config": attestation["model_config_sha256"],
    }
    assets: dict[str, dict[str, str]] = {}
    for name, expected in expected_by_asset.items():
        record = raw_assets.get(name)
        if not isinstance(record, Mapping):
            return None
        recorded_expected = str(record.get("expected_sha256", "") or "").strip().lower()
        actual = str(record.get("actual_sha256", "") or "").strip().lower()
        if recorded_expected != expected or actual != expected:
            return None
        assets[name] = {
            "expected_sha256": expected,
            "actual_sha256": actual,
        }
    return {
        "status": "verified",
        "asset_integrity_gate": "passed",
        "verification_scope": "caller_declared_asset_bytes",
        "live_checkpoint_finetune_gate": "not_implemented",
        "algorithm": "sha256",
        "assets": assets,
    }


def normalize_dvd_model_provenance(
    provenance: Mapping[str, Any] | None,
    *,
    source: str,
) -> dict[str, Any]:
    """Validate DVD identity without trusting a caller's verification claim.

    This function performs no filesystem access.  A complete hash declaration
    therefore remains pending even if the input mapping contains a
    self-reported ``verified`` record.  Only :func:`verify_dvd_asset_digests`
    may promote the detached result after comparing observed file bytes.
    """

    if not isinstance(provenance, Mapping):
        raise ValueError(
            f"{source} must identify an official or project fine-tuned DVD checkpoint."
        )
    normalized = copy.deepcopy(dict(provenance))
    kind = str(normalized.get("kind", "") or "").strip()
    allowed_kinds = tuple(DVD_PROVENANCE_KIND_BY_FAMILY.values())
    if kind not in allowed_kinds:
        raise ValueError(
            f"{source}.kind must be one of {allowed_kinds!r}."
        )
    for field in ("checkpoint_identity", "config_identity"):
        if not str(normalized.get(field, "") or "").strip():
            raise ValueError(f"{source}.{field} must be explicitly provided.")

    if DVD_ASSET_ATTESTATION_FIELD not in normalized:
        if "validation" in normalized:
            normalized["validation"] = copy.deepcopy(DVD_PROVENANCE_VALIDATION)
        return normalized

    attestation = normalize_dvd_asset_attestation(
        normalized[DVD_ASSET_ATTESTATION_FIELD],
        source=f"{source}.{DVD_ASSET_ATTESTATION_FIELD}",
    )
    normalized[DVD_ASSET_ATTESTATION_FIELD] = attestation
    normalized["validation"] = copy.deepcopy(DVD_PROVENANCE_PENDING_VALIDATION)
    return normalized


def normalize_dvd_model_identity(
    model_family: Any,
    provenance: Mapping[str, Any] | None,
    *,
    source: str,
) -> tuple[str, dict[str, Any]]:
    """Validate that one DVD family and checkpoint provenance agree."""

    family = str(model_family or "").strip()
    expected_kind = DVD_PROVENANCE_KIND_BY_FAMILY.get(family)
    if expected_kind is None:
        raise ValueError(
            f"{source}.model_family must be one of {DVD_MODEL_FAMILIES!r}."
        )
    normalized = normalize_dvd_model_provenance(
        provenance,
        source=f"{source}.model_provenance",
    )
    actual_kind = str(normalized.get("kind", "") or "").strip()
    if actual_kind != expected_kind:
        raise ValueError(
            f"{source}.model_provenance.kind {actual_kind!r} conflicts with "
            f"model_family {family!r}; expected {expected_kind!r}."
        )
    return family, normalized


def verify_dvd_asset_digests(
    provenance: Mapping[str, Any],
    *,
    checkpoint_sha256: str,
    model_config_sha256: str,
    source: str = "DVD asset attestation",
) -> dict[str, Any]:
    """Fail closed unless both observed asset digests match their declaration."""

    normalized = normalize_dvd_model_provenance(
        provenance,
        source="DVD model_provenance",
    )
    raw_attestation = normalized.get(DVD_ASSET_ATTESTATION_FIELD)
    if not isinstance(raw_attestation, Mapping):
        return normalized
    attestation = dict(raw_attestation)
    actual = {
        "checkpoint": str(checkpoint_sha256 or "").strip().lower(),
        "model_config": str(model_config_sha256 or "").strip().lower(),
    }
    expected = {
        "checkpoint": str(attestation["checkpoint_sha256"]),
        "model_config": str(attestation["model_config_sha256"]),
    }
    mismatches = [
        (f"{name}: expected {expected[name]}, got {actual[name]}")
        for name in ("checkpoint", "model_config")
        if actual[name] != expected[name]
    ]
    if mismatches:
        raise ValueError(f"{source} SHA-256 mismatch; " + "; ".join(mismatches))
    normalized["validation"] = {
        "status": "verified",
        "asset_integrity_gate": "passed",
        "verification_scope": "caller_declared_asset_bytes",
        "live_checkpoint_finetune_gate": "not_implemented",
        "algorithm": "sha256",
        "assets": {
            name: {
                "expected_sha256": expected[name],
                "actual_sha256": actual[name],
            }
            for name in ("checkpoint", "model_config")
        },
    }
    return normalized


def dvd_provenance_validation(
    provenance: Mapping[str, Any],
) -> dict[str, Any]:
    """Summarize an already normalized or file-verified provenance record."""

    if not isinstance(provenance, Mapping):
        return copy.deepcopy(DVD_PROVENANCE_VALIDATION)
    raw_attestation = provenance.get(DVD_ASSET_ATTESTATION_FIELD)
    if not isinstance(raw_attestation, Mapping):
        return copy.deepcopy(DVD_PROVENANCE_VALIDATION)
    attestation = normalize_dvd_asset_attestation(
        raw_attestation,
        source=f"DVD model_provenance.{DVD_ASSET_ATTESTATION_FIELD}",
    )
    verified = _verified_validation(
        provenance.get("validation"),
        attestation=attestation,
    )
    if verified is not None:
        return verified
    return copy.deepcopy(DVD_PROVENANCE_PENDING_VALIDATION)


__all__ = [
    "DVD_ASSET_ATTESTATION_FIELD",
    "DVD_MODEL_FAMILIES",
    "DVD_MODEL_FAMILY",
    "DVD_OFFICIAL_MODEL_FAMILY",
    "DVD_PROVENANCE_KIND_BY_FAMILY",
    "DVD_PROVENANCE_PENDING_VALIDATION",
    "DVD_PROVENANCE_VALIDATION",
    "dvd_provenance_validation",
    "normalize_dvd_asset_attestation",
    "normalize_dvd_model_identity",
    "normalize_dvd_model_provenance",
    "verify_dvd_asset_digests",
]
