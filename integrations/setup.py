"""Prepare pinned external source trees without modifying Dream.exe code.

This script is intentionally separate from package installation. Running
``pip install dream-exe`` never clones a repository, applies a patch, or
downloads a model. Users explicitly select the source providers they need.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
from pathlib import Path
import shutil
import stat
import subprocess
import sys
import tempfile
from typing import Any, Mapping, Sequence


FORMAT = "dream-exe.external-sources"
ROBOCASA_RUNTIME_SHADOW_FORMAT = "dream_exe.robocasa_runtime_shadow"
_ROBOCASA_RUNTIME_SHADOW_RECEIPT = ".dream-exe-robocasa-runtime-shadow.json"
_RUNTIME_SHADOW_IGNORED_NAMES = frozenset(
    {
        ".git",
        ".mypy_cache",
        ".pytest_cache",
        ".ruff_cache",
        "__pycache__",
        _ROBOCASA_RUNTIME_SHADOW_RECEIPT,
    }
)
_SOURCE_FIELDS = frozenset(
    {
        "build",
        "destination",
        "patch_manifest",
        "repository",
        "revision",
        "untracked_data_roots",
    }
)
_SOURCE_BUILDS = frozenset({"grounding_dino_cuda"})
_HEX_DIGITS = frozenset("0123456789abcdef")
_GROUNDING_DINO_VERSION_PATH = "groundingdino/version.py"
_GROUNDING_DINO_VERSION_CONTENT = "__version__ = '0.1.0'\n"
_GROUNDING_DINO_TORCH_VERSION = "2.7.1"
_GROUNDING_DINO_TORCHVISION_VERSION = "0.22.1"
_PROVIDER_GROUPS = {
    "core": ("cotracker", "dvd", "grounding_dino", "robocasa"),
    "generation": ("wan22",),
    "optional": ("vda",),
}

_GROUNDING_DINO_VERIFY = r"""
import importlib
import importlib.metadata
import json
from pathlib import Path
import sys
import sysconfig

checkout = Path(sys.argv[1]).resolve()
sys.path.insert(0, checkout.as_posix())

import torch

torch_version = torch.__version__.split("+", 1)[0]
if torch_version != "2.7.1":
    raise RuntimeError(
        f"GroundingDINO requires torch 2.7.1, found {torch.__version__}"
    )
torchvision_version = importlib.metadata.version("torchvision")
if torchvision_version != "0.22.1":
    raise RuntimeError(
        "GroundingDINO requires torchvision 0.22.1, found "
        f"{torchvision_version}"
    )
if not torch.cuda.is_available():
    raise RuntimeError(
        "GroundingDINO CUDA verification requires a visible CUDA GPU"
    )

extension = importlib.import_module("groundingdino._C")
extension_path = Path(extension.__file__).resolve()
try:
    extension_path.relative_to(checkout)
except ValueError as error:
    raise RuntimeError(
        f"GroundingDINO extension escaped checkout: {extension_path}"
    ) from error

value = torch.arange(
    8,
    device="cuda",
    dtype=torch.float32,
).reshape(1, 4, 1, 2)
spatial_shapes = torch.tensor(
    [[2, 2]],
    device="cuda",
    dtype=torch.int64,
)
level_start_index = torch.tensor(
    [0],
    device="cuda",
    dtype=torch.int64,
)
sampling_locations = torch.full(
    (1, 1, 1, 1, 1, 2),
    0.5,
    device="cuda",
    dtype=torch.float32,
)
attention_weights = torch.ones(
    (1, 1, 1, 1, 1),
    device="cuda",
    dtype=torch.float32,
)
output = extension.ms_deform_attn_forward(
    value,
    spatial_shapes,
    level_start_index,
    sampling_locations,
    attention_weights,
    1,
)
torch.cuda.synchronize()
if tuple(output.shape) != (1, 1, 2):
    raise RuntimeError(
        f"unexpected GroundingDINO smoke shape: {tuple(output.shape)}"
    )
torch.testing.assert_close(
    output.cpu(),
    torch.tensor([[[3.0, 4.0]]], dtype=torch.float32),
)
print(
    json.dumps(
        {
            "cuda_arch": ".".join(
                str(item) for item in torch.cuda.get_device_capability()
            ),
            "extension": extension_path.as_posix(),
            "python_soabi": str(
                sysconfig.get_config_var("SOABI") or "unknown"
            ),
            "torch_cuda": str(torch.version.cuda or "unknown"),
            "torch_version": str(torch.__version__),
            "torchvision_version": torchvision_version,
        },
        sort_keys=True,
    )
)
"""


class ExternalSourceError(RuntimeError):
    """A source declaration or checkout failed closed."""


def _run(
    arguments: Sequence[str],
    *,
    cwd: Path | None = None,
    check: bool = True,
    environment: Mapping[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [str(item) for item in arguments],
        cwd=None if cwd is None else cwd.as_posix(),
        check=check,
        capture_output=True,
        env=None if environment is None else dict(environment),
        text=True,
    )


def _git_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment["GIT_LFS_SKIP_SMUDGE"] = "1"
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _runtime_shadow_ignored(path: Path) -> bool:
    return path.name in _RUNTIME_SHADOW_IGNORED_NAMES


def _logical_runtime_tree(
    root: Path,
    *,
    external_symlink_roots: Sequence[Path] = (),
    reject_symlinks: bool,
) -> tuple[dict[str, Any], set[tuple[int, int]]]:
    """Hash one logical runtime tree, dereferencing only declared symlinks."""

    lexical_root = root.expanduser()
    if lexical_root.is_symlink():
        raise ExternalSourceError(
            f"runtime-shadow root must not be a symlink: {lexical_root}"
        )
    root = lexical_root.resolve(strict=True)
    if not root.is_dir():
        raise ExternalSourceError(
            f"runtime-shadow root must be a real directory: {root}"
        )
    allowed_roots = tuple(
        _relative_path(item, label="runtime-shadow external symlink root")
        for item in external_symlink_roots
    )
    directories: list[str] = ["."]
    files: list[dict[str, Any]] = []
    inodes: set[tuple[int, int]] = set()
    symlink_count = 0
    total_bytes = 0

    def external_symlink_allowed(relative: Path) -> bool:
        return any(
            relative == allowed or relative.is_relative_to(allowed)
            for allowed in allowed_roots
        )

    def visit_directory(
        actual: Path,
        relative: Path,
        ancestors: frozenset[tuple[int, int]],
    ) -> None:
        nonlocal symlink_count, total_bytes
        try:
            directory_stat = actual.stat()
        except OSError as error:
            raise ExternalSourceError(
                f"cannot stat runtime-shadow directory: {actual}"
            ) from error
        identity = (int(directory_stat.st_dev), int(directory_stat.st_ino))
        if identity in ancestors:
            raise ExternalSourceError(
                f"runtime-shadow source contains a symlink cycle: {actual}"
            )
        next_ancestors = ancestors | {identity}
        try:
            entries = sorted(actual.iterdir(), key=lambda item: item.name)
        except OSError as error:
            raise ExternalSourceError(
                f"cannot enumerate runtime-shadow directory: {actual}"
            ) from error
        for entry in entries:
            if _runtime_shadow_ignored(entry):
                continue
            child_relative = (
                Path(entry.name) if relative == Path(".") else relative / entry.name
            )
            try:
                entry_lstat = entry.lstat()
            except OSError as error:
                raise ExternalSourceError(
                    f"cannot inspect runtime-shadow path: {entry}"
                ) from error
            is_symlink = stat.S_ISLNK(entry_lstat.st_mode)
            if is_symlink:
                symlink_count += 1
                if reject_symlinks:
                    raise ExternalSourceError(
                        "runtime shadow must be physically detached and "
                        f"contain no symlink: {child_relative.as_posix()}"
                    )
                try:
                    resolved = entry.resolve(strict=True)
                except OSError as error:
                    raise ExternalSourceError(
                        "runtime-shadow source contains a broken symlink: "
                        f"{child_relative.as_posix()}"
                    ) from error
                try:
                    resolved.relative_to(root)
                except ValueError:
                    if not external_symlink_allowed(child_relative):
                        raise ExternalSourceError(
                            "runtime-shadow source symlink escapes outside a "
                            "declared data root: "
                            f"{child_relative.as_posix()}"
                        )
                entry = resolved
            try:
                entry_stat = entry.stat()
            except OSError as error:
                raise ExternalSourceError(
                    f"cannot stat runtime-shadow path: {entry}"
                ) from error
            if stat.S_ISDIR(entry_stat.st_mode):
                directories.append(child_relative.as_posix())
                visit_directory(entry, child_relative, next_ancestors)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise ExternalSourceError(
                    "runtime-shadow tree contains a non-regular path: "
                    f"{child_relative.as_posix()}"
                )
            size = int(entry_stat.st_size)
            files.append(
                {
                    "path": child_relative.as_posix(),
                    "sha256": _sha256(entry),
                    "size": size,
                }
            )
            total_bytes += size
            inodes.add((int(entry_stat.st_dev), int(entry_stat.st_ino)))

    visit_directory(root, Path("."), frozenset())
    canonical = {
        "directories": sorted(directories),
        "files": sorted(files, key=lambda item: item["path"]),
    }
    encoded = json.dumps(
        canonical,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    return (
        {
            "bytes": total_bytes,
            "directories": len(directories),
            "files": len(files),
            "manifest_sha256": hashlib.sha256(encoded).hexdigest(),
            "symlinks_dereferenced": symlink_count,
        },
        inodes,
    )


def _runtime_shadow_copy_ignore(
    _directory: str,
    names: list[str],
) -> set[str]:
    return {name for name in names if name in _RUNTIME_SHADOW_IGNORED_NAMES}


def _robocasa_shadow_source_identity(
    source_result: Mapping[str, str],
) -> dict[str, str]:
    required = ("repository", "revision", "tracked_tree_sha256")
    identity = {key: str(source_result.get(key, "")).strip() for key in required}
    missing = [key for key, value in identity.items() if not value]
    if missing:
        raise ExternalSourceError(
            "verified RoboCasa source result is missing runtime-shadow "
            f"identity fields: {missing!r}"
        )
    return identity


def materialize_robocasa_runtime_shadow(
    *,
    source_root: str | Path,
    destination: str | Path,
    source_identity: Mapping[str, str],
    external_symlink_roots: Sequence[Path] = (Path("robocasa/models/assets"),),
) -> dict[str, Any]:
    """Create one atomic, dereferenced, physically detached runtime shadow."""

    source = Path(source_root).expanduser().resolve(strict=True)
    target = Path(destination).expanduser().resolve(strict=False)
    if target.exists() or target.is_symlink():
        raise ExternalSourceError(
            f"runtime-shadow destination already exists: {target}"
        )
    try:
        target.relative_to(source)
    except ValueError:
        pass
    else:
        raise ExternalSourceError(
            "runtime-shadow destination must not be inside its source"
        )
    try:
        source.relative_to(target)
    except ValueError:
        pass
    else:
        raise ExternalSourceError(
            "runtime-shadow source must not be inside its destination"
        )
    identity = _robocasa_shadow_source_identity(source_identity)
    source_manifest, source_inodes = _logical_runtime_tree(
        source,
        external_symlink_roots=external_symlink_roots,
        reject_symlinks=False,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with tempfile.TemporaryDirectory(
            prefix=f".{target.name}.shadow-",
            dir=target.parent,
        ) as temporary:
            staging = Path(temporary) / "runtime"
            shutil.copytree(
                source,
                staging,
                copy_function=shutil.copy2,
                ignore=_runtime_shadow_copy_ignore,
                symlinks=False,
            )
            shadow_manifest, shadow_inodes = _logical_runtime_tree(
                staging,
                reject_symlinks=True,
            )
            comparable_source = {
                key: value
                for key, value in source_manifest.items()
                if key != "symlinks_dereferenced"
            }
            comparable_shadow = {
                key: value
                for key, value in shadow_manifest.items()
                if key != "symlinks_dereferenced"
            }
            if comparable_source != comparable_shadow:
                raise ExternalSourceError(
                    "runtime-shadow copy does not match the verified source: "
                    f"source={comparable_source!r}, "
                    f"shadow={comparable_shadow!r}"
                )
            shared_inodes = source_inodes.intersection(shadow_inodes)
            if shared_inodes:
                raise ExternalSourceError(
                    "runtime shadow is not physically detached from source: "
                    f"shared_inodes={len(shared_inodes)}"
                )
            receipt = {
                "format": ROBOCASA_RUNTIME_SHADOW_FORMAT,
                "source_identity": identity,
                "source_manifest": source_manifest,
                "status": "verified",
            }
            receipt_path = staging / _ROBOCASA_RUNTIME_SHADOW_RECEIPT
            receipt_path.write_text(
                json.dumps(receipt, indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )
            staging.replace(target)
    except (OSError, shutil.Error) as error:
        raise ExternalSourceError(
            f"failed to materialize RoboCasa runtime shadow: {target}"
        ) from error
    receipt_path = target / _ROBOCASA_RUNTIME_SHADOW_RECEIPT
    return {
        "bytes": int(source_manifest["bytes"]),
        "destination": target.as_posix(),
        "directories": int(source_manifest["directories"]),
        "files": int(source_manifest["files"]),
        "manifest_sha256": str(source_manifest["manifest_sha256"]),
        "receipt_sha256": _sha256(receipt_path),
        "shared_inodes": 0,
        "source_symlinks_dereferenced": int(source_manifest["symlinks_dereferenced"]),
        "status": "created",
    }


def verify_robocasa_runtime_shadow(
    *,
    source_root: str | Path,
    destination: str | Path,
    source_identity: Mapping[str, str],
    external_symlink_roots: Sequence[Path] = (Path("robocasa/models/assets"),),
) -> dict[str, Any]:
    """Verify a runtime shadow against its current source and receipt."""

    source = Path(source_root).expanduser().resolve(strict=True)
    target = Path(destination).expanduser().resolve(strict=True)
    receipt_path = target / _ROBOCASA_RUNTIME_SHADOW_RECEIPT
    try:
        receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalSourceError(
            f"cannot load RoboCasa runtime-shadow receipt: {receipt_path}"
        ) from error
    if not isinstance(receipt, Mapping) or set(receipt) != {
        "format",
        "source_identity",
        "source_manifest",
        "status",
    }:
        raise ExternalSourceError("invalid RoboCasa runtime-shadow receipt")
    if receipt["format"] != ROBOCASA_RUNTIME_SHADOW_FORMAT:
        raise ExternalSourceError("unsupported RoboCasa runtime-shadow receipt schema")
    identity = _robocasa_shadow_source_identity(source_identity)
    receipt_identity = _mapping(
        receipt["source_identity"],
        label="RoboCasa runtime-shadow source identity",
    )
    if receipt_identity != identity:
        raise ExternalSourceError("RoboCasa runtime-shadow source identity mismatch")
    source_manifest, source_inodes = _logical_runtime_tree(
        source,
        external_symlink_roots=external_symlink_roots,
        reject_symlinks=False,
    )
    receipt_manifest = _mapping(
        receipt["source_manifest"],
        label="RoboCasa runtime-shadow source manifest",
    )
    if receipt_manifest != source_manifest:
        raise ExternalSourceError(
            "RoboCasa runtime-shadow source content changed after creation"
        )
    shadow_manifest, shadow_inodes = _logical_runtime_tree(
        target,
        reject_symlinks=True,
    )
    comparable_source = {
        key: value
        for key, value in source_manifest.items()
        if key != "symlinks_dereferenced"
    }
    comparable_shadow = {
        key: value
        for key, value in shadow_manifest.items()
        if key != "symlinks_dereferenced"
    }
    if comparable_source != comparable_shadow:
        raise ExternalSourceError(
            "RoboCasa runtime-shadow content mismatch: "
            f"source={comparable_source!r}, shadow={comparable_shadow!r}"
        )
    shared_inodes = source_inodes.intersection(shadow_inodes)
    if shared_inodes:
        raise ExternalSourceError(
            "RoboCasa runtime shadow shares file identity with its source: "
            f"shared_inodes={len(shared_inodes)}"
        )
    return {
        "bytes": int(shadow_manifest["bytes"]),
        "destination": target.as_posix(),
        "directories": int(shadow_manifest["directories"]),
        "files": int(shadow_manifest["files"]),
        "manifest_sha256": str(shadow_manifest["manifest_sha256"]),
        "receipt_sha256": _sha256(receipt_path),
        "shared_inodes": 0,
        "status": "verified",
    }


def _mapping(value: Any, *, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ExternalSourceError(f"{label} must be a JSON object")
    return dict(value)


def _relative_path(value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ExternalSourceError(f"{label} must be non-empty")
    path = Path(text)
    if path == Path(".") or path.is_absolute() or ".." in path.parts:
        raise ExternalSourceError(f"{label} must be a contained relative path")
    return path


def _revision(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if len(text) != 40 or any(char not in _HEX_DIGITS for char in text):
        raise ExternalSourceError(
            f"{label} must be a 40-character lowercase Git revision"
        )
    return text


def _sha256_value(value: Any, *, label: str) -> str:
    text = str(value or "").strip()
    if len(text) != 64 or any(char not in _HEX_DIGITS for char in text):
        raise ExternalSourceError(
            f"{label} must be 64 lowercase hexadecimal characters"
        )
    return text


def load_source_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    """Load and strictly validate one external-source declaration."""

    manifest_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalSourceError(
            f"cannot load external-source manifest: {manifest_path}"
        ) from error
    root = _mapping(payload, label="external-source manifest")
    if set(root) != {"format", "sources"}:
        raise ExternalSourceError(
            "external-source manifest must contain exactly format and sources"
        )
    if root["format"] != FORMAT:
        raise ExternalSourceError(
            f"unsupported external-source schema: {root['format']!r}"
        )
    sources = _mapping(root["sources"], label="external-source sources")
    if not sources:
        raise ExternalSourceError(
            "external-source manifest must declare at least one source"
        )

    normalized: dict[str, dict[str, Any]] = {}
    destinations: set[str] = set()
    for raw_name, raw_source in sorted(sources.items()):
        name = str(raw_name or "").strip().lower()
        if not name or name != raw_name:
            raise ExternalSourceError(
                "external-source names must be lowercase non-empty strings"
            )
        source = _mapping(
            raw_source,
            label=f"external source {name!r}",
        )
        unknown = sorted(set(source).difference(_SOURCE_FIELDS))
        required = {"destination", "repository", "revision"}
        missing = sorted(required.difference(source))
        if unknown or missing:
            details = []
            if missing:
                details.append("missing " + ", ".join(missing))
            if unknown:
                details.append("unknown " + ", ".join(unknown))
            raise ExternalSourceError(
                f"external source {name!r}: " + "; ".join(details)
            )
        repository = str(source["repository"] or "").strip()
        if not repository.startswith("https://github.com/"):
            raise ExternalSourceError(
                f"external source {name!r} repository must be HTTPS GitHub"
            )
        destination = _relative_path(
            source["destination"],
            label=f"external source {name!r} destination",
        ).as_posix()
        if destination in destinations:
            raise ExternalSourceError(
                f"duplicate external destination: {destination!r}"
            )
        destinations.add(destination)
        entry = {
            "destination": destination,
            "repository": repository,
            "revision": _revision(
                source["revision"],
                label=f"external source {name!r} revision",
            ),
        }
        if "patch_manifest" in source:
            entry["patch_manifest"] = _relative_path(
                source["patch_manifest"],
                label=f"external source {name!r} patch_manifest",
            ).as_posix()
        if "build" in source:
            build = str(source["build"] or "").strip()
            if build not in _SOURCE_BUILDS:
                raise ExternalSourceError(
                    f"external source {name!r} has unsupported build {build!r}"
                )
            entry["build"] = build
        if "untracked_data_roots" in source:
            roots_raw = source["untracked_data_roots"]
            if not isinstance(roots_raw, list) or not roots_raw:
                raise ExternalSourceError(
                    f"external source {name!r} untracked_data_roots must "
                    "be a non-empty list"
                )
            roots = tuple(
                _relative_path(
                    item,
                    label=(f"external source {name!r} untracked data root"),
                ).as_posix()
                for item in roots_raw
            )
            if len(set(roots)) != len(roots):
                raise ExternalSourceError(
                    f"external source {name!r} has duplicate untracked data roots"
                )
            entry["untracked_data_roots"] = roots
        normalized[name] = entry
    return normalized


def load_workspace_sources(
    path: str | Path,
) -> tuple[Path, Path, dict[str, dict[str, Path]]]:
    """Resolve source setup/runtime paths from the public workspace config."""

    workspace_path = Path(path).expanduser().resolve()
    try:
        payload = json.loads(workspace_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalSourceError(
            f"cannot load workspace configuration: {workspace_path}"
        ) from error
    root = _mapping(payload, label="workspace configuration")
    if root.get("format") != "dream-exe.workspace":
        raise ExternalSourceError(
            f"unsupported workspace configuration: {root.get('format')!r}"
        )
    roots = _mapping(root.get("roots"), label="workspace roots")
    external_value = str(roots.get("external") or "").strip()
    if not external_value:
        raise ExternalSourceError("workspace roots.external must be non-empty")
    anchor = workspace_path.parent
    external_root = (anchor / external_value).resolve(strict=False)

    bindings = _mapping(root.get("bindings"), label="workspace bindings")
    sources = _mapping(bindings.get("sources"), label="workspace source bindings")
    if not sources:
        raise ExternalSourceError("workspace must declare external source bindings")

    manifest_paths: set[Path] = set()
    resolved: dict[str, dict[str, Path]] = {}
    for name, raw_binding in sorted(sources.items()):
        binding = _mapping(raw_binding, label=f"workspace source {name!r}")
        if set(binding) != {"path", "setup_path", "manifest"}:
            raise ExternalSourceError(
                f"workspace source {name!r} must contain path, setup_path, manifest"
            )
        manifest_value = str(binding["manifest"] or "").strip()
        if not manifest_value:
            raise ExternalSourceError(
                f"workspace source {name!r} must declare its source manifest"
            )
        runtime_path = (anchor / str(binding["path"])).resolve(strict=False)
        setup_path = (anchor / str(binding["setup_path"])).resolve(strict=False)
        manifest_path = (anchor / manifest_value).resolve(strict=False)
        for label, candidate in (("runtime", runtime_path), ("setup", setup_path)):
            try:
                candidate.relative_to(external_root)
            except ValueError as error:
                raise ExternalSourceError(
                    f"workspace source {name!r} {label} path escapes external root"
                ) from error
        manifest_paths.add(manifest_path)
        resolved[str(name)] = {
            "runtime_path": runtime_path,
            "setup_path": setup_path,
        }
    if len(manifest_paths) != 1:
        raise ExternalSourceError(
            "workspace external sources must share one provenance manifest"
        )
    return external_root, manifest_paths.pop(), resolved


def _load_patch_manifest(
    path: Path,
    *,
    expected_repository: str,
    expected_revision: str,
) -> tuple[tuple[tuple[Path, str, tuple[str, ...]], ...], str | None]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ExternalSourceError(
            f"cannot load provider patch manifest: {path}"
        ) from error
    root = _mapping(raw, label="provider patch manifest")
    if root.get("format") != "dream_exe.provider_patch":
        raise ExternalSourceError(f"unsupported provider patch schema in {path}")
    if root.get("upstream_repository") != expected_repository:
        raise ExternalSourceError(f"provider patch repository mismatch in {path}")
    if root.get("upstream_commit") != expected_revision:
        raise ExternalSourceError(f"provider patch revision mismatch in {path}")
    raw_patches = root.get("patches")
    if not isinstance(raw_patches, list) or not raw_patches:
        raise ExternalSourceError(f"provider patch manifest has no patches: {path}")
    tree_digest_raw = root.get("post_patch_tracked_tree_sha256")
    tree_digest = (
        _sha256_value(
            tree_digest_raw,
            label="provider patch post_patch_tracked_tree_sha256",
        )
        if tree_digest_raw is not None
        else None
    )
    output = []
    for index, raw_patch in enumerate(raw_patches):
        patch = _mapping(
            raw_patch,
            label=f"provider patch {index}",
        )
        relative = _relative_path(
            patch.get("path"),
            label=f"provider patch {index} path",
        )
        digest = _sha256_value(
            patch.get("sha256"),
            label=f"provider patch {index} sha256",
        )
        targets_raw = patch.get("targets")
        if not isinstance(targets_raw, list) or not targets_raw:
            raise ExternalSourceError(
                f"provider patch {index} targets must be non-empty"
            )
        targets = tuple(
            _relative_path(
                item,
                label=f"provider patch {index} target",
            ).as_posix()
            for item in targets_raw
        )
        output.append((path.parent / relative, digest, targets))
    return tuple(output), tree_digest


def _git_output(checkout: Path, *arguments: str) -> str:
    try:
        completed = _run(("git", "-C", checkout, *arguments))
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExternalSourceError(
            f"Git command failed for {checkout}: {' '.join(arguments)}"
        ) from error
    return completed.stdout.strip()


def _patch_state(checkout: Path, patch: Path) -> str:
    forward = _run(
        ("git", "-C", checkout, "apply", "--check", patch),
        check=False,
    )
    if forward.returncode == 0:
        return "not_applied"
    reverse = _run(
        ("git", "-C", checkout, "apply", "--reverse", "--check", patch),
        check=False,
    )
    if reverse.returncode == 0:
        return "applied"
    raise ExternalSourceError(
        f"patch is neither cleanly applicable nor already applied: {patch}"
    )


def _untracked_files(checkout: Path) -> set[str]:
    return {
        line
        for line in _git_output(
            checkout,
            "ls-files",
            "--others",
        ).splitlines()
        if line
    }


def _generated_python_cache(path: str) -> bool:
    candidate = Path(path)
    return "__pycache__" in candidate.parts and candidate.suffix in {".pyc", ".pyo"}


def _tracked_tree_sha256(checkout: Path) -> str:
    """Hash every tracked working-tree source byte in a stable order.

    The digest binds the index mode, repository-relative path, byte length,
    and working-tree bytes for every stage-zero path. It intentionally omits
    untracked simulator assets and datasets, which have separate contracts.
    """

    completed = _run(("git", "-C", checkout, "ls-files", "--stage", "-z"))
    digest = hashlib.sha256()
    for raw_entry in completed.stdout.split("\0"):
        if not raw_entry:
            continue
        try:
            metadata, raw_path = raw_entry.split("\t", 1)
            mode, _index_blob, stage = metadata.split()
        except ValueError as error:
            raise ExternalSourceError(
                f"invalid tracked source entry in {checkout}"
            ) from error
        if stage != "0":
            raise ExternalSourceError(
                f"external checkout has a non-stage-zero path: {raw_path!r}"
            )
        relative = _relative_path(
            raw_path,
            label="tracked source path",
        )
        path = checkout / relative
        try:
            file_stat = path.lstat()
            if stat.S_ISREG(file_stat.st_mode):
                content = path.read_bytes()
            elif stat.S_ISLNK(file_stat.st_mode):
                content = os.fsencode(os.readlink(path))
            else:
                raise ExternalSourceError(
                    f"tracked source is neither file nor symlink: {path}"
                )
        except OSError as error:
            raise ExternalSourceError(f"cannot hash tracked source: {path}") from error
        for field in (
            mode.encode("ascii"),
            os.fsencode(relative.as_posix()),
            str(len(content)).encode("ascii"),
        ):
            digest.update(field)
            digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return digest.hexdigest()


def _grounding_dino_extension_candidates(checkout: Path) -> tuple[Path, ...]:
    package_root = checkout / "groundingdino"
    return tuple(
        sorted(
            (
                candidate
                for candidate in package_root.glob("_C*.so")
                if candidate.exists() or candidate.is_symlink()
            ),
            key=lambda item: item.as_posix(),
        )
    )


def _validate_grounding_dino_generated_files(
    checkout: Path,
    *,
    require_version: bool,
) -> None:
    version_path = checkout / _GROUNDING_DINO_VERSION_PATH
    if not version_path.exists() and not version_path.is_symlink():
        if require_version:
            raise ExternalSourceError(
                "GroundingDINO build did not create its version metadata"
            )
        return
    if version_path.is_symlink() or not version_path.is_file():
        raise ExternalSourceError(
            f"GroundingDINO generated metadata is not a regular file: {version_path}"
        )
    try:
        content = version_path.read_text(encoding="utf-8")
    except OSError as error:
        raise ExternalSourceError(
            f"cannot read GroundingDINO generated metadata: {version_path}"
        ) from error
    if content != _GROUNDING_DINO_VERSION_CONTENT:
        raise ExternalSourceError(
            "GroundingDINO generated version metadata has unexpected content"
        )


def _grounding_dino_build_environment() -> dict[str, str]:
    try:
        torch = importlib.import_module("torch")
        torchvision_version = importlib.metadata.version("torchvision")
        cpp_extension = importlib.import_module("torch.utils.cpp_extension")
    except (ImportError, importlib.metadata.PackageNotFoundError) as error:
        raise ExternalSourceError(
            "GroundingDINO build requires the region-runtime extra; run "
            'python -m pip install -e ".[region-runtime]" first'
        ) from error

    torch_version = str(torch.__version__).split("+", 1)[0]
    if torch_version != _GROUNDING_DINO_TORCH_VERSION:
        raise ExternalSourceError(
            "GroundingDINO build requires torch "
            f"{_GROUNDING_DINO_TORCH_VERSION}, found {torch.__version__}"
        )
    if torchvision_version != _GROUNDING_DINO_TORCHVISION_VERSION:
        raise ExternalSourceError(
            "GroundingDINO build requires torchvision "
            f"{_GROUNDING_DINO_TORCHVISION_VERSION}, found "
            f"{torchvision_version}"
        )

    cuda_home = getattr(cpp_extension, "CUDA_HOME", None)
    if not cuda_home:
        raise ExternalSourceError(
            "GroundingDINO build requires a visible CUDA toolkit; set "
            "CUDA_HOME to the target toolkit"
        )

    environment = dict(os.environ)
    environment["CUDA_HOME"] = str(cuda_home)
    architecture = environment.get("TORCH_CUDA_ARCH_LIST", "").strip()
    if not architecture:
        if not torch.cuda.is_available():
            raise ExternalSourceError(
                "GroundingDINO build requires either a visible CUDA GPU or "
                "an explicit TORCH_CUDA_ARCH_LIST"
            )
        capability = torch.cuda.get_device_capability()
        architecture = ".".join(str(item) for item in capability)
        environment["TORCH_CUDA_ARCH_LIST"] = architecture
    environment.setdefault("MAX_JOBS", "4")
    environment["GIT_TERMINAL_PROMPT"] = "0"
    return environment


def _verify_grounding_dino_build(checkout: Path) -> dict[str, str]:
    environment = dict(os.environ)
    current_pythonpath = environment.get("PYTHONPATH", "")
    environment["PYTHONPATH"] = (
        checkout.as_posix()
        if not current_pythonpath
        else checkout.as_posix() + os.pathsep + current_pythonpath
    )
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    environment["CUDA_CACHE_DISABLE"] = "1"
    completed = _run(
        (
            sys.executable,
            "-c",
            _GROUNDING_DINO_VERIFY,
            checkout,
        ),
        check=False,
        environment=environment,
    )
    if completed.returncode != 0:
        details = completed.stderr.strip().splitlines()
        summary = details[-1] if details else "verification process failed"
        raise ExternalSourceError(
            f"GroundingDINO CUDA extension verification failed: {summary}"
        )
    try:
        metadata = json.loads(completed.stdout.strip().splitlines()[-1])
    except (IndexError, json.JSONDecodeError) as error:
        raise ExternalSourceError(
            "GroundingDINO CUDA verification returned invalid metadata"
        ) from error
    if not isinstance(metadata, dict) or not all(
        isinstance(key, str) and isinstance(value, str)
        for key, value in metadata.items()
    ):
        raise ExternalSourceError(
            "GroundingDINO CUDA verification metadata must be string-valued"
        )
    extension = Path(metadata.get("extension", "")).resolve(strict=False)
    try:
        extension.relative_to(checkout.resolve())
    except ValueError as error:
        raise ExternalSourceError(
            f"GroundingDINO extension escaped checkout: {extension}"
        ) from error
    if extension.is_symlink() or not extension.is_file():
        raise ExternalSourceError(
            f"GroundingDINO extension is not a regular file: {extension}"
        )
    return dict(metadata)


def _prepare_grounding_dino_build(
    checkout: Path,
    *,
    check_only: bool,
) -> tuple[str, dict[str, str]]:
    _validate_grounding_dino_generated_files(
        checkout,
        require_version=False,
    )
    try:
        metadata = _verify_grounding_dino_build(checkout)
    except ExternalSourceError:
        if check_only:
            raise
    else:
        return "verified", metadata

    stale_extensions = _grounding_dino_extension_candidates(checkout)
    for extension in stale_extensions:
        if extension.is_symlink() or not extension.is_file():
            raise ExternalSourceError(
                "refusing to replace a non-regular GroundingDINO extension: "
                f"{extension}"
            )
        extension.unlink()

    environment = _grounding_dino_build_environment()
    try:
        with tempfile.TemporaryDirectory(
            prefix=".grounding-dino-build-",
            dir=checkout.parent,
        ) as temporary:
            temporary_root = Path(temporary)
            _run(
                (
                    sys.executable,
                    "setup.py",
                    "build_ext",
                    "--inplace",
                    "--force",
                    "--build-temp",
                    temporary_root / "temp",
                    "--build-lib",
                    temporary_root / "lib",
                ),
                cwd=checkout,
                environment=environment,
            )
    except (OSError, subprocess.CalledProcessError) as error:
        raise ExternalSourceError(
            "failed to compile the GroundingDINO CUDA extension; verify "
            "CUDA_HOME, TORCH_CUDA_ARCH_LIST, and the region-runtime extra"
        ) from error

    _validate_grounding_dino_generated_files(
        checkout,
        require_version=True,
    )
    metadata = _verify_grounding_dino_build(checkout)
    return "built", metadata


def prepare_source(
    *,
    name: str,
    declaration: Mapping[str, Any],
    repository_root: Path,
    external_root: Path,
    check_only: bool,
) -> dict[str, str]:
    """Clone/patch or verify one declared external source."""

    repository_root = repository_root.expanduser().resolve()
    external_root = external_root.expanduser().resolve()
    destination = (external_root / declaration["destination"]).resolve(strict=False)
    try:
        destination.relative_to(external_root)
    except ValueError as error:
        raise ExternalSourceError(
            f"external destination escapes root: {destination}"
        ) from error

    created = False
    if not destination.exists() and not destination.is_symlink():
        if check_only:
            raise ExternalSourceError(
                f"external source {name!r} is missing: {destination}"
            )
        destination.parent.mkdir(parents=True, exist_ok=True)
        try:
            with tempfile.TemporaryDirectory(
                prefix=f".{destination.name}.clone-",
                dir=destination.parent,
            ) as temporary:
                checkout = Path(temporary) / "checkout"
                _run(
                    (
                        "git",
                        "clone",
                        "--filter=blob:none",
                        "--no-checkout",
                        declaration["repository"],
                        checkout,
                    ),
                    environment=_git_environment(),
                )
                _run(
                    (
                        "git",
                        "-C",
                        checkout,
                        "checkout",
                        "--detach",
                        declaration["revision"],
                    ),
                    environment=_git_environment(),
                )
                if destination.exists() or destination.is_symlink():
                    raise ExternalSourceError(
                        f"external source destination appeared during clone: "
                        f"{destination}"
                    )
                checkout.replace(destination)
                created = True
        except (OSError, subprocess.CalledProcessError) as error:
            raise ExternalSourceError(
                f"failed to clone external source {name!r}"
            ) from error
    if not (destination / ".git").is_dir():
        raise ExternalSourceError(
            f"external source is not a Git checkout: {destination}"
        )
    if (
        _git_output(destination, "remote", "get-url", "origin")
        != (declaration["repository"])
    ):
        raise ExternalSourceError(f"external source {name!r} has an unexpected origin")
    if _git_output(destination, "rev-parse", "HEAD") != (declaration["revision"]):
        raise ExternalSourceError(
            f"external source {name!r} is at an unexpected revision"
        )

    expected_changes: set[str] = set()
    patch_status = "none"
    patch_manifest = declaration.get("patch_manifest", "")
    if patch_manifest:
        patch_manifest_path = (repository_root / patch_manifest).resolve(strict=False)
        try:
            patch_manifest_path.relative_to(repository_root)
        except ValueError as error:
            raise ExternalSourceError(
                "provider patch manifest escapes repository root"
            ) from error
        patches, expected_tree_sha256 = _load_patch_manifest(
            patch_manifest_path,
            expected_repository=declaration["repository"],
            expected_revision=declaration["revision"],
        )
        states = []
        for patch_path, expected_sha256, targets in patches:
            if not patch_path.is_file():
                raise ExternalSourceError(f"provider patch is missing: {patch_path}")
            if _sha256(patch_path) != expected_sha256:
                raise ExternalSourceError(
                    f"provider patch SHA-256 mismatch: {patch_path}"
                )
            state = _patch_state(destination, patch_path)
            if state == "not_applied":
                if check_only:
                    raise ExternalSourceError(
                        f"provider patch is not applied: {patch_path}"
                    )
                _run(("git", "-C", destination, "apply", patch_path))
                state = "applied"
            states.append(state)
            expected_changes.update(targets)
        patch_status = (
            "applied" if all(item == "applied" for item in states) else "mixed"
        )

    changed = {
        line
        for line in _git_output(
            destination,
            "diff",
            "--name-only",
            "HEAD",
        ).splitlines()
        if line
    }
    build = declaration.get("build", "")
    allowed_untracked = set()
    if build == "grounding_dino_cuda":
        allowed_untracked.add(_GROUNDING_DINO_VERSION_PATH)
        allowed_untracked.update(
            candidate.relative_to(destination).as_posix()
            for candidate in _grounding_dino_extension_candidates(destination)
        )
    untracked = _untracked_files(destination)
    untracked_data_roots = tuple(
        str(item).rstrip("/") for item in declaration.get("untracked_data_roots", ())
    )
    declared_data = {
        path
        for path in untracked
        if any(
            path == root or path.startswith(root + "/") for root in untracked_data_roots
        )
    }
    generated_caches = {path for path in untracked if _generated_python_cache(path)}
    unexpected_untracked = untracked.difference(
        allowed_untracked | declared_data | generated_caches
    )
    if unexpected_untracked:
        raise ExternalSourceError(
            f"external source {name!r} has untracked files: "
            f"{sorted(unexpected_untracked)!r}"
        )
    if changed != expected_changes:
        raise ExternalSourceError(
            f"external source {name!r} has unexpected changes: "
            f"expected={sorted(expected_changes)!r}, actual={sorted(changed)!r}"
        )

    tracked_tree_sha256 = ""
    if patch_manifest:
        tracked_tree_sha256 = _tracked_tree_sha256(destination)
        if (
            expected_tree_sha256 is not None
            and tracked_tree_sha256 != expected_tree_sha256
        ):
            raise ExternalSourceError(
                f"external source {name!r} tracked-tree SHA-256 mismatch: "
                f"expected={expected_tree_sha256!r}, "
                f"actual={tracked_tree_sha256!r}"
            )

    build_status = "none"
    build_metadata: dict[str, str] = {}
    if build == "grounding_dino_cuda":
        build_status, build_metadata = _prepare_grounding_dino_build(
            destination,
            check_only=check_only,
        )
        final_untracked = {
            path
            for path in _untracked_files(destination)
            if not _generated_python_cache(path)
        }
        expected_build_outputs = {_GROUNDING_DINO_VERSION_PATH} | {
            candidate.relative_to(destination).as_posix()
            for candidate in _grounding_dino_extension_candidates(destination)
        }
        if final_untracked != expected_build_outputs:
            raise ExternalSourceError(
                "GroundingDINO build produced unexpected untracked files: "
                f"{sorted(final_untracked)!r}"
            )

    result = {
        "build_status": build_status,
        "destination": destination.as_posix(),
        "patch_status": patch_status,
        "repository": declaration["repository"],
        "revision": declaration["revision"],
        "status": "created" if created else "verified",
    }
    if tracked_tree_sha256:
        result["tracked_tree_sha256"] = tracked_tree_sha256
    if untracked_data_roots:
        result["untracked_data_paths_ignored"] = str(len(declared_data))
    if generated_caches:
        result["generated_python_cache_paths_ignored"] = str(len(generated_caches))
    result.update(
        {f"build_{key}": value for key, value in sorted(build_metadata.items())}
    )
    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare or verify explicit pinned external source checkouts. "
            "No model/checkpoint or simulator data is downloaded."
        )
    )
    parser.add_argument(
        "providers",
        nargs="+",
        help=(
            "provider names declared by configs/workspace.json; groups: "
            "core, generation, optional, all"
        ),
    )
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Dream.exe source root (default: parent of integrations)",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path("configs/workspace.json"),
        help="path configuration (default: configs/workspace.json)",
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="perform a zero-network, zero-write verification only",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    arguments = _parser().parse_args(argv)
    repository_root = arguments.repository_root.expanduser().resolve()
    workspace_path = arguments.workspace.expanduser()
    if not workspace_path.is_absolute():
        workspace_path = repository_root / workspace_path
    external_root, manifest_path, source_paths = load_workspace_sources(
        workspace_path
    )
    declarations = load_source_manifest(manifest_path)
    requested = []
    for raw_name in arguments.providers:
        name = str(raw_name).strip().lower()
        if name == "all":
            for available in sorted(source_paths):
                if available not in requested:
                    requested.append(available)
            continue
        if name in _PROVIDER_GROUPS:
            for available in _PROVIDER_GROUPS[name]:
                if available not in source_paths:
                    raise ExternalSourceError(
                        f"provider group {name!r} requires unavailable source "
                        f"{available!r}"
                    )
                if available not in requested:
                    requested.append(available)
            continue
        if name not in source_paths:
            raise ExternalSourceError(
                f"unknown workspace source {raw_name!r}; "
                f"available={sorted(source_paths)!r}"
            )
        if name not in declarations:
            raise ExternalSourceError(
                f"unknown external source {raw_name!r}; "
                f"available={sorted(declarations)!r}"
            )
        if name not in requested:
            requested.append(name)
    configured_declarations: dict[str, dict[str, Any]] = {}
    for name in requested:
        setup_path = source_paths[name]["setup_path"]
        try:
            destination = setup_path.relative_to(external_root).as_posix()
        except ValueError as error:  # guarded by load_workspace_sources
            raise ExternalSourceError(
                f"workspace source {name!r} setup path escapes external root"
            ) from error
        configured_declarations[name] = {
            **declarations[name],
            "destination": destination,
        }
    results = [
        {
            "provider": name,
            **prepare_source(
                name=name,
                declaration=configured_declarations[name],
                repository_root=repository_root,
                external_root=external_root,
                check_only=bool(arguments.check),
            ),
        }
        for name in requested
    ]
    runtime_shadows = []
    for name, source_result in zip(requested, results):
        shadow_path = source_paths[name]["runtime_path"]
        setup_path = source_paths[name]["setup_path"]
        if shadow_path == setup_path:
            continue
        if name != "robocasa":
            raise ExternalSourceError(
                f"only robocasa supports a detached runtime path, found {name!r}"
            )
        options = {
            "source_root": source_result["destination"],
            "destination": shadow_path,
            "source_identity": _robocasa_shadow_source_identity(source_result),
        }
        runtime_shadow = (
            verify_robocasa_runtime_shadow(**options)
            if arguments.check
            else materialize_robocasa_runtime_shadow(**options)
        )
        runtime_shadow["provider"] = "robocasa"
        runtime_shadows.append(runtime_shadow)
    payload = {
        "check_only": bool(arguments.check),
        "external_root": external_root.as_posix(),
        "results": results,
        "format": FORMAT,
    }
    if runtime_shadows:
        payload["runtime_shadows"] = runtime_shadows
    print(
        json.dumps(
            payload,
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ExternalSourceError as error:
        print(f"external source setup failed: {error}", file=sys.stderr)
        raise SystemExit(2) from error
