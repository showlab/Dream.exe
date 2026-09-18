"""Reproducible acquisition and verification of external model assets.

Importing this module performs no discovery, network access, directory
creation, or framework import.  A caller supplies a pinned manifest and an
explicit asset root, then chooses separately whether to verify, plan, or
publish.  Publication downloads into a temporary tree, validates every
declared byte count and SHA-256 digest, and atomically commits all files under
an advisory root lock.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping
from contextlib import contextmanager
import copy
from datetime import datetime, timezone
import fcntl
import hashlib
from importlib import resources
import json
import os
from pathlib import Path
import re
import shutil
import stat
import sys
import tempfile
from typing import Any


MODEL_ASSET_MANIFEST_SCHEMA = "dream-exe.model-asset-manifest"
MODEL_ASSET_PLAN_SCHEMA = "dream-exe.model-asset-plan"
SUPPORTED_SOURCE_TYPES = {
    "url",
    "huggingface",
    "gdrive_file",
    "gdrive_folder",
    "injected",
}
MISSING_VERSION = "missing"
_HUGGING_FACE_LOCATOR = re.compile(
    r"(?P<repo>[A-Za-z0-9][A-Za-z0-9._-]*/[A-Za-z0-9][A-Za-z0-9._-]*)"
    r"@(?P<revision>[0-9a-f]{40})"
)


_CURRENT_POSE_MODEL_MANIFEST = {
    "format": MODEL_ASSET_MANIFEST_SCHEMA,
    "name": "current-pose-assets",
    "description": (
        "Pinned downloadable assets for the FoundationPose and SinRef-6D "
        "adapters. FreePose assets are not managed by this manifest."
    ),
    "packages": [
        {
            "id": "foundationpose-weights",
            "source": {
                "type": "gdrive_folder",
                "locator": (
                    "https://drive.google.com/drive/folders/"
                    "1DFezOAD0oD1BblsXVxqDsl8fj0qzB82i?usp=sharing"
                ),
            },
            "destination": "foundationpose/weights",
            "license": {
                "id": "foundationpose-nvidia-noncommercial",
                "notice": (
                    "FoundationPose is governed by NVIDIA's source-code "
                    "license and is limited to non-commercial research or "
                    "evaluation use."
                ),
                "acceptance_required": True,
            },
            "artifacts": [
                {
                    "path": "2023-10-28-18-33-37/config.yml",
                    "size_bytes": 708,
                    "sha256": (
                        "28a6ba94a33230ee5fc3c519394862815"
                        "78b0972542bd9e38ca6123e75605686"
                    ),
                },
                {
                    "path": "2023-10-28-18-33-37/model_best.pth",
                    "size_bytes": 68220109,
                    "sha256": (
                        "774700586ddc435d408fc01c9809c43e"
                        "151232936369dfbea0f0f964ba471d60"
                    ),
                },
                {
                    "path": "2024-01-11-20-02-45/config.yml",
                    "size_bytes": 778,
                    "sha256": (
                        "a79db4de3b95885dd5ae86833b37b869"
                        "8a75dad81e87d1086cd50b2fcd8dda3f"
                    ),
                },
                {
                    "path": "2024-01-11-20-02-45/model_best.pth",
                    "size_bytes": 190229389,
                    "sha256": (
                        "81924d384bf5c26c646ee4783104982ae"
                        "3d1e049c181c36641b6a7aeae494c26"
                    ),
                },
            ],
        },
        {
            "id": "sinref6d-pem",
            "source": {
                "type": "gdrive_file",
                "locator": "1joW9IvwsaRJYxoUmGo68dBVg-HcFNyI7",
            },
            "destination": "sinref6d",
            "license": {
                "id": "sinref6d-mit-checkpoint-review",
                "notice": (
                    "SinRef-6D code is MIT licensed; review and accept the "
                    "upstream checkpoint terms before acquisition."
                ),
                "acceptance_required": True,
            },
            "artifacts": [
                {
                    "path": "sam-6d-pem-base.pth",
                    "size_bytes": 1295700437,
                    "sha256": (
                        "19885cc1a9dbb71159830e38ec09c6e88"
                        "04f5b21834fad2a6dba127e93372c59"
                    ),
                }
            ],
        },
        {
            "id": "sinref6d-vmamba",
            "source": {
                "type": "url",
                "locator": (
                    "https://github.com/MzeroMiko/VMamba/releases/download/"
                    "%23v2cls/vssm_small_0229_ckpt_epoch_222.pth"
                ),
            },
            "destination": "sinref6d",
            "license": {
                "id": "vmamba-checkpoint-upstream-review",
                "notice": (
                    "This checkpoint is distributed by the VMamba upstream; "
                    "review its release and model terms before acquisition."
                ),
                "acceptance_required": True,
            },
            "artifacts": [
                {
                    "path": "vssm_small_0229_ckpt_epoch_222.pth",
                    "size_bytes": 200732814,
                    "sha256": (
                        "c540366e366d81acf69251810e7ea7268"
                        "e0ca8bcb6843f3b773c1f0105bd30cb"
                    ),
                }
            ],
        },
    ],
    "unresolved": [],
}


def current_pose_model_manifest() -> dict[str, Any]:
    """Return the pinned current pose-asset manifest."""

    return copy.deepcopy(_CURRENT_POSE_MODEL_MANIFEST)


def current_core_model_manifest() -> dict[str, Any]:
    """Return the pinned public assets required by the core pipeline."""

    resource = resources.files("dream_exe.model_assets").joinpath(
        "configs",
        "core.json",
    )
    with resource.open("r", encoding="utf-8") as stream:
        return validate_model_asset_manifest(json.load(stream))


def _clean(value: Any) -> str:
    return str(value or "").strip()


def _safe_relative(value: Any, *, label: str) -> str:
    text = _clean(value)
    path = Path(text)
    if (
        not text
        or path.is_absolute()
        or "\\" in text
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError(f"{label} must be a safe relative path")
    return path.as_posix()


def _safe_id(value: Any, *, label: str) -> str:
    text = _clean(value)
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]*", text):
        raise ValueError(f"{label} must be a safe identifier")
    return text


def _validate_sha256(value: Any, *, label: str) -> str:
    text = _clean(value).lower()
    if not re.fullmatch(r"[0-9a-f]{64}", text):
        raise ValueError(f"{label} must be a lowercase SHA-256 digest")
    return text


def validate_model_asset_manifest(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate and normalize one model-asset manifest."""

    if not isinstance(manifest, Mapping):
        raise TypeError("model asset manifest must be a mapping")
    if manifest.get("format") != MODEL_ASSET_MANIFEST_SCHEMA:
        raise ValueError("unsupported model asset manifest format")
    name = _safe_id(manifest.get("name"), label="manifest name")
    packages = list(manifest.get("packages", []) or [])
    if not packages:
        raise ValueError("model asset manifest packages must not be empty")
    normalized_packages: list[dict[str, Any]] = []
    package_ids: set[str] = set()
    all_targets: set[str] = set()
    for index, raw_package in enumerate(packages):
        if not isinstance(raw_package, Mapping):
            raise TypeError(f"package {index} must be a mapping")
        package_id = _safe_id(
            raw_package.get("id"),
            label=f"package {index} id",
        )
        if package_id in package_ids:
            raise ValueError(f"duplicate package id: {package_id}")
        package_ids.add(package_id)
        source = dict(raw_package.get("source", {}) or {})
        source_type = _clean(source.get("type")).lower()
        if source_type not in SUPPORTED_SOURCE_TYPES:
            raise ValueError(f"unsupported source type for {package_id}: {source_type}")
        locator = _clean(source.get("locator"))
        if source_type != "injected" and not locator:
            raise ValueError(f"source locator is required for {package_id}")
        if source_type == "huggingface" and not _HUGGING_FACE_LOCATOR.fullmatch(
            locator
        ):
            raise ValueError(
                f"Hugging Face source for {package_id} must use "
                "'namespace/repository@40-character-commit'"
            )
        destination = _safe_relative(
            raw_package.get("destination"),
            label=f"package {package_id} destination",
        )
        license_record = dict(raw_package.get("license", {}) or {})
        license_id = _safe_id(
            license_record.get("id"),
            label=f"package {package_id} license id",
        )
        notice = _clean(license_record.get("notice"))
        if not notice:
            raise ValueError(f"package {package_id} license notice is required")
        artifacts = list(raw_package.get("artifacts", []) or [])
        if not artifacts:
            raise ValueError(f"package {package_id} artifacts must not be empty")
        normalized_artifacts: list[dict[str, Any]] = []
        artifact_paths: set[str] = set()
        for artifact_index, raw_artifact in enumerate(artifacts):
            if not isinstance(raw_artifact, Mapping):
                raise TypeError(
                    f"artifact {artifact_index} in {package_id} must be a mapping"
                )
            relative = _safe_relative(
                raw_artifact.get("path"),
                label=f"artifact path in {package_id}",
            )
            target_key = f"{destination}/{relative}"
            if relative in artifact_paths or target_key in all_targets:
                raise ValueError(f"duplicate model artifact target: {target_key}")
            artifact_paths.add(relative)
            all_targets.add(target_key)
            size = int(raw_artifact.get("size_bytes", 0) or 0)
            if size < 1:
                raise ValueError(f"artifact size must be positive: {target_key}")
            normalized_artifacts.append(
                {
                    "path": relative,
                    "size_bytes": size,
                    "sha256": _validate_sha256(
                        raw_artifact.get("sha256"),
                        label=f"artifact digest for {target_key}",
                    ),
                }
            )
        normalized_packages.append(
            {
                "id": package_id,
                "source": {
                    "type": source_type,
                    "locator": locator,
                },
                "destination": destination,
                "license": {
                    "id": license_id,
                    "notice": notice,
                    "acceptance_required": bool(
                        license_record.get(
                            "acceptance_required",
                            True,
                        )
                    ),
                },
                "artifacts": normalized_artifacts,
            }
        )
    unresolved: list[dict[str, str]] = []
    unresolved_ids: set[str] = set()
    for index, raw in enumerate(list(manifest.get("unresolved", []) or [])):
        if not isinstance(raw, Mapping):
            raise TypeError(f"unresolved entry {index} must be a mapping")
        item_id = _safe_id(
            raw.get("id"),
            label=f"unresolved entry {index} id",
        )
        if item_id in unresolved_ids or item_id in package_ids:
            raise ValueError(f"duplicate unresolved asset id: {item_id}")
        unresolved_ids.add(item_id)
        reason = _clean(raw.get("reason"))
        resolution = _clean(raw.get("required_resolution"))
        if not reason or not resolution:
            raise ValueError(f"unresolved asset {item_id} needs reason and resolution")
        unresolved.append(
            {
                "id": item_id,
                "current_behavior": _clean(raw.get("current_behavior")),
                "reason": reason,
                "required_resolution": resolution,
            }
        )
    return {
        "format": MODEL_ASSET_MANIFEST_SCHEMA,
        "name": name,
        "description": _clean(manifest.get("description")),
        "packages": normalized_packages,
        "unresolved": unresolved,
    }


def load_model_asset_manifest(
    path: str | os.PathLike[str],
) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"model asset manifest not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    return validate_model_asset_manifest(payload)


def model_asset_manifest_digest(
    manifest: Mapping[str, Any],
) -> str:
    normalized = validate_model_asset_manifest(manifest)
    encoded = json.dumps(
        normalized,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _absolute_root(value: str | os.PathLike[str]) -> Path:
    root = Path(value).expanduser()
    if not root.is_absolute():
        raise ValueError("model asset root must be an explicit absolute path")
    return root.resolve(strict=False)


def _target_path(root: Path, destination: str, relative: str) -> Path:
    target = (root / destination / relative).resolve(strict=False)
    try:
        target.relative_to(root)
    except ValueError as error:
        raise ValueError("model artifact target escapes asset root") from error
    return target


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_version(path: Path) -> str:
    if not path.exists() and not path.is_symlink():
        return MISSING_VERSION
    if path.is_symlink():
        return "symlink:" + os.readlink(path)
    if not path.is_file():
        return "non-file"
    return "sha256:" + _sha256_file(path)


def verify_model_assets(
    manifest: Mapping[str, Any],
    *,
    asset_root: str | os.PathLike[str],
) -> dict[str, Any]:
    """Verify declared files without creating or mutating anything."""

    normalized = validate_model_asset_manifest(manifest)
    root = _absolute_root(asset_root)
    package_records: list[dict[str, Any]] = []
    counts = {
        "ready": 0,
        "missing": 0,
        "size_mismatch": 0,
        "digest_mismatch": 0,
        "unsafe": 0,
    }
    for package in normalized["packages"]:
        artifact_records = []
        for artifact in package["artifacts"]:
            target = _target_path(
                root,
                package["destination"],
                artifact["path"],
            )
            if not target.exists():
                status = "missing"
                size = None
                digest = ""
            elif not target.is_file():
                status = "unsafe"
                size = None
                digest = ""
            else:
                size = int(target.stat().st_size)
                if size != int(artifact["size_bytes"]):
                    status = "size_mismatch"
                    digest = ""
                else:
                    digest = _sha256_file(target)
                    status = (
                        "ready" if digest == artifact["sha256"] else "digest_mismatch"
                    )
            counts[status] += 1
            artifact_records.append(
                {
                    "path": target.as_posix(),
                    "relative_path": (f"{package['destination']}/{artifact['path']}"),
                    "status": status,
                    "expected_size_bytes": artifact["size_bytes"],
                    "actual_size_bytes": size,
                    "expected_sha256": artifact["sha256"],
                    "actual_sha256": digest,
                    "version": _file_version(target),
                }
            )
        package_records.append(
            {
                "id": package["id"],
                "status": (
                    "ready"
                    if all(item["status"] == "ready" for item in artifact_records)
                    else "not_ready"
                ),
                "license": copy.deepcopy(package["license"]),
                "artifacts": artifact_records,
            }
        )
    unresolved = copy.deepcopy(normalized["unresolved"])
    ready = (
        all(item["status"] == "ready" for item in package_records) and not unresolved
    )
    return {
        "format": "dream-exe.model-asset-verification",
        "manifest_name": normalized["name"],
        "manifest_sha256": model_asset_manifest_digest(normalized),
        "asset_root": root.as_posix(),
        "status": (
            "ready" if ready else ("incomplete_manifest" if unresolved else "not_ready")
        ),
        "ready": ready,
        "packages": package_records,
        "unresolved": unresolved,
        "counts": counts,
    }


def plan_model_asset_acquisition(
    manifest: Mapping[str, Any],
    *,
    asset_root: str | os.PathLike[str],
    accepted_licenses: Iterable[str] = (),
    offline: bool = False,
    force: bool = False,
) -> dict[str, Any]:
    """Plan missing/corrupt downloads without network or writes."""

    normalized = validate_model_asset_manifest(manifest)
    verification = verify_model_assets(normalized, asset_root=asset_root)
    accepted = {
        _safe_id(value, label="accepted license") for value in accepted_licenses
    }
    verification_by_id = {
        package["id"]: package for package in verification["packages"]
    }
    actions = []
    blocked = []
    for unresolved in normalized["unresolved"]:
        blocked.append(
            {
                "id": unresolved["id"],
                "reason": "unresolved_manifest_entry",
                "detail": unresolved["reason"],
            }
        )
    for package in normalized["packages"]:
        package_verification = verification_by_id[package["id"]]
        if package_verification["status"] == "ready" and not force:
            continue
        non_missing = [
            artifact["status"]
            for artifact in package_verification["artifacts"]
            if artifact["status"] != "missing"
        ]
        if non_missing and not force:
            blocked.append(
                {
                    "id": package["id"],
                    "reason": "force_required_for_existing_invalid_files",
                    "detail": sorted(set(non_missing)),
                }
            )
            continue
        license_record = package["license"]
        if (
            license_record["acceptance_required"]
            and license_record["id"] not in accepted
        ):
            blocked.append(
                {
                    "id": package["id"],
                    "reason": "license_acceptance_required",
                    "detail": copy.deepcopy(license_record),
                }
            )
            continue
        if offline:
            blocked.append(
                {
                    "id": package["id"],
                    "reason": "offline_missing_assets",
                    "detail": [
                        item["relative_path"]
                        for item in package_verification["artifacts"]
                        if item["status"] != "ready"
                    ],
                }
            )
            continue
        actions.append(
            {
                "package": copy.deepcopy(package),
                "base_versions": {
                    item["relative_path"]: item["version"]
                    for item in package_verification["artifacts"]
                },
            }
        )
    root = _absolute_root(asset_root)
    receipt = root / ".dream-exe" / "model-assets" / f"{normalized['name']}.json"
    return {
        "format": MODEL_ASSET_PLAN_SCHEMA,
        "manifest": normalized,
        "manifest_sha256": model_asset_manifest_digest(normalized),
        "asset_root": root.as_posix(),
        "receipt_path": receipt.as_posix(),
        "receipt_base_version": _file_version(receipt),
        "verification": verification,
        "offline": bool(offline),
        "force": bool(force),
        "accepted_licenses": sorted(accepted),
        "actions": actions,
        "blocked": blocked,
        "status": ("blocked" if blocked else ("ready" if not actions else "planned")),
    }


def _reject_symlink_chain(path: Path) -> None:
    current = Path(path.anchor)
    for part in path.parts[1:]:
        current = current / part
        try:
            mode = current.lstat().st_mode
        except (FileNotFoundError, NotADirectoryError):
            break
        if stat.S_ISLNK(mode):
            raise ValueError(
                f"model asset publication must not traverse a symlink: {current}"
            )


@contextmanager
def _locked_root(root: Path):
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(
        root,
        os.O_RDONLY | getattr(os, "O_DIRECTORY", 0),
    )
    try:
        fcntl.flock(descriptor, fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)


def _default_fetch_package(
    package: Mapping[str, Any],
    staging_root: Path,
) -> None:
    source = dict(package["source"])
    source_type = source["type"]
    locator = source["locator"]
    artifacts = list(package["artifacts"])
    if source_type == "url":
        if len(artifacts) != 1:
            raise ValueError("URL packages must declare exactly one artifact")
        import urllib.request

        destination = staging_root / artifacts[0]["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        print(
            f"Downloading pinned model package {package['id']} ...",
            file=sys.stderr,
            flush=True,
        )
        with urllib.request.urlopen(locator, timeout=60) as response:
            with destination.open("wb") as stream:
                shutil.copyfileobj(response, stream, 1024 * 1024)
        return
    if source_type == "huggingface":
        match = _HUGGING_FACE_LOCATOR.fullmatch(locator)
        if match is None:  # guarded by manifest validation
            raise ValueError("invalid pinned Hugging Face source locator")
        try:
            from huggingface_hub import hf_hub_download
        except Exception as error:
            raise RuntimeError(
                "huggingface-hub is required for Hugging Face model assets; "
                "run integrations/setup_dependencies.sh first or install "
                "dream-exe[assets]"
            ) from error
        print(
            f"Downloading pinned model package {package['id']} ...",
            file=sys.stderr,
            flush=True,
        )
        for artifact in artifacts:
            cached = Path(
                hf_hub_download(
                    repo_id=match.group("repo"),
                    filename=artifact["path"],
                    revision=match.group("revision"),
                    local_dir=staging_root,
                )
            )
            destination = staging_root / artifact["path"]
            if cached.resolve(strict=False) != destination.resolve(strict=False):
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(cached, destination)
        return
    if source_type == "gdrive_file":
        if len(artifacts) != 1:
            raise ValueError("Google Drive file packages must declare one artifact")
        try:
            import gdown
        except Exception as error:
            raise RuntimeError(
                "gdown is required for a Google Drive model asset"
            ) from error
        destination = staging_root / artifacts[0]["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        result = gdown.download(
            id=locator,
            output=destination.as_posix(),
            quiet=False,
            fuzzy=False,
            use_cookies=False,
        )
        if not result:
            raise RuntimeError("Google Drive model download failed")
        return
    if source_type == "gdrive_folder":
        try:
            import gdown
        except Exception as error:
            raise RuntimeError(
                "gdown is required for a Google Drive model package"
            ) from error
        staging_root.mkdir(parents=True, exist_ok=True)
        result = gdown.download_folder(
            url=locator,
            output=staging_root.as_posix(),
            quiet=False,
            use_cookies=False,
        )
        if not result:
            raise RuntimeError("Google Drive model package download failed")
        return
    raise RuntimeError("injected model package sources require an explicit fetcher")


def _validate_staged_package(
    package: Mapping[str, Any],
    staging_root: Path,
) -> list[dict[str, Any]]:
    records = []
    for artifact in package["artifacts"]:
        path = (staging_root / artifact["path"]).resolve(strict=False)
        try:
            path.relative_to(staging_root.resolve(strict=False))
        except ValueError as error:
            raise ValueError("staged model artifact escapes package") from error
        if not path.is_file():
            raise RuntimeError(
                f"downloaded model artifact is missing: {artifact['path']}"
            )
        size = int(path.stat().st_size)
        if size != int(artifact["size_bytes"]):
            raise RuntimeError(
                f"downloaded model artifact size mismatch: {artifact['path']}"
            )
        digest = _sha256_file(path)
        if digest != artifact["sha256"]:
            raise RuntimeError(
                f"downloaded model artifact digest mismatch: {artifact['path']}"
            )
        records.append(
            {
                "path": artifact["path"],
                "size_bytes": size,
                "sha256": digest,
            }
        )
    return records


def _validate_plan_versions(
    plan: Mapping[str, Any],
    *,
    root: Path,
) -> None:
    for action in plan["actions"]:
        package = action["package"]
        for artifact in package["artifacts"]:
            relative = f"{package['destination']}/{artifact['path']}"
            target = _target_path(
                root,
                package["destination"],
                artifact["path"],
            )
            expected = _clean(action["base_versions"].get(relative))
            if _file_version(target) != expected:
                raise RuntimeError(f"model artifact changed after planning: {target}")
    receipt = Path(plan["receipt_path"]).resolve(strict=False)
    if _file_version(receipt) != _clean(plan.get("receipt_base_version")):
        raise RuntimeError("model asset receipt changed after planning")


def _commit_files(
    staged_by_final: Mapping[Path, Path],
    *,
    backup_root: Path,
) -> None:
    backup_root.mkdir()
    backups: dict[Path, Path | None] = {}
    committed: list[Path] = []
    try:
        for index, (target, staged) in enumerate(staged_by_final.items()):
            _reject_symlink_chain(target)
            target.parent.mkdir(parents=True, exist_ok=True)
            backup = None
            if target.exists() or target.is_symlink():
                backup = backup_root / f"{index}.bak"
                os.replace(target, backup)
            backups[target] = backup
            os.replace(staged, target)
            committed.append(target)
    except BaseException:
        for target in reversed(committed):
            target.unlink(missing_ok=True)
            backup = backups.get(target)
            if backup is not None and backup.exists():
                os.replace(backup, target)
        for target, backup in backups.items():
            if target in committed:
                continue
            if backup is not None and backup.exists():
                os.replace(backup, target)
        raise


def publish_model_assets(
    plan: Mapping[str, Any],
    *,
    asset_root: str | os.PathLike[str],
    fetch_package: (Callable[[Mapping[str, Any], Path], None] | None) = None,
    dry_run: bool = False,
    completed_at: str | None = None,
) -> dict[str, Any]:
    """Fetch, verify, and transactionally publish a model-asset plan."""

    if plan.get("format") != MODEL_ASSET_PLAN_SCHEMA:
        raise ValueError("unsupported model asset plan format")
    root = _absolute_root(asset_root)
    if root.as_posix() != _absolute_root(plan["asset_root"]).as_posix():
        raise ValueError("model asset plan root does not match publisher root")
    _reject_symlink_chain(root)
    manifest = validate_model_asset_manifest(plan["manifest"])
    if model_asset_manifest_digest(manifest) != _clean(plan.get("manifest_sha256")):
        raise ValueError("model asset plan manifest digest mismatch")
    if list(plan.get("blocked", []) or []):
        raise RuntimeError("model asset acquisition plan is blocked")
    _validate_plan_versions(plan, root=root)
    public = {
        "manifest_name": manifest["name"],
        "manifest_sha256": plan["manifest_sha256"],
        "asset_root": root.as_posix(),
        "receipt_path": plan["receipt_path"],
        "package_ids": [action["package"]["id"] for action in plan["actions"]],
    }
    if dry_run:
        return public | {
            "status": "dry_run",
            "written": [],
        }
    if not plan["actions"]:
        return public | {
            "status": "already_ready",
            "written": [],
        }

    root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(
        tempfile.mkdtemp(
            prefix=f".{manifest['name']}.model-assets-",
            dir=root.parent.as_posix(),
        )
    )
    fetch = fetch_package or _default_fetch_package
    staged_by_final: dict[Path, Path] = {}
    acquired_packages = []
    try:
        for package_index, action in enumerate(plan["actions"]):
            package = action["package"]
            package_stage = stage / f"package-{package_index}"
            package_stage.mkdir()
            fetch(copy.deepcopy(package), package_stage)
            artifact_records = _validate_staged_package(
                package,
                package_stage,
            )
            acquired_packages.append(
                {
                    "id": package["id"],
                    "source": copy.deepcopy(package["source"]),
                    "license": copy.deepcopy(package["license"]),
                    "artifacts": artifact_records,
                }
            )
            for artifact in package["artifacts"]:
                target = _target_path(
                    root,
                    package["destination"],
                    artifact["path"],
                )
                staged_by_final[target] = package_stage / artifact["path"]
        timestamp = (
            _clean(completed_at)
            if completed_at is not None
            else datetime.now(timezone.utc).isoformat()
        )
        if not timestamp:
            raise ValueError("model asset completion timestamp is empty")
        receipt_payload = {
            "format": "dream-exe.model-asset-receipt",
            "manifest_name": manifest["name"],
            "manifest_sha256": plan["manifest_sha256"],
            "completed_at": timestamp,
            "packages": acquired_packages,
        }
        staged_receipt = stage / "receipt.json"
        staged_receipt.write_text(
            json.dumps(
                receipt_payload,
                indent=2,
                ensure_ascii=True,
            )
            + "\n",
            encoding="utf-8",
        )
        receipt = Path(plan["receipt_path"]).resolve(strict=False)
        try:
            receipt.relative_to(root)
        except ValueError as error:
            raise ValueError("model asset receipt escapes asset root") from error
        staged_by_final[receipt] = staged_receipt

        with _locked_root(root):
            _validate_plan_versions(plan, root=root)
            _commit_files(
                staged_by_final,
                backup_root=stage / "backups",
            )
        return public | {
            "status": "published",
            "written": [path.as_posix() for path in staged_by_final],
            "receipt": receipt_payload,
        }
    finally:
        shutil.rmtree(stage, ignore_errors=True)


__all__ = [
    "MODEL_ASSET_MANIFEST_SCHEMA",
    "MODEL_ASSET_PLAN_SCHEMA",
    "SUPPORTED_SOURCE_TYPES",
    "verify_model_assets",
    "current_core_model_manifest",
    "current_pose_model_manifest",
    "load_model_asset_manifest",
    "model_asset_manifest_digest",
    "plan_model_asset_acquisition",
    "publish_model_assets",
    "validate_model_asset_manifest",
]
