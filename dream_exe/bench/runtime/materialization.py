"""Work-local benchmark materialization and integrity verification."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..contracts.action import gt_action_payload_from_bundle, load_action_bundle
from ..contracts.schemas import (
    INIT_RECEIPT_SCHEMA,
    WORK_MATERIALIZATION_SCHEMA,
    canonical_sha256,
    load_and_validate,
    validate_document,
)
from ..data.repository import BenchRepository, ReferenceSelection
from ..data.workspace import Workspace
from ..outputs.lifecycle import sha256_file, write_json_atomic
from .init import (
    _SAFE_ID,
    _add_reference_to_sample,
    _copy_tree,
    _file_record,
    _hash_tree,
    _json_object,
    _materialize_init_sample,
)


_WORK_MATERIALIZATION_FILENAME = "materialization.json"
_RUN_OWNED_TOP_LEVEL = frozenset({"experiment", "logs", "run_state"})
_RUN_OWNED_ARTIFACTS = frozenset({"gen", "gen_enhanced", "runs"})


def _select_uids(repository: BenchRepository, run: Mapping[str, Any]) -> list[str]:
    collection_name = run["selection"]["collection"]
    explicit = list(run["selection"]["cases"])
    if collection_name is None:
        return explicit
    collection = repository.load_collection(collection_name)
    members = [item["uid"] for item in collection["cases"]]
    if not explicit:
        return members
    unknown = sorted(set(explicit) - set(members))
    if unknown:
        raise ValueError(
            f"run selects cases outside collection {collection_name!r}: {unknown}"
        )
    return explicit


def _load_verified_init_receipt(
    *,
    workspace: Workspace,
    receipt_id: str,
    uid: str,
    repository: BenchRepository,
) -> tuple[dict[str, Any], Path]:
    if not _SAFE_ID.fullmatch(receipt_id):
        raise ValueError("init receipt must use a safe receipt ID")
    receipt_root = (workspace.materialized_root / receipt_id).resolve()
    receipt = load_and_validate(
        receipt_root / "init-receipt.json",
        expected_schema=INIT_RECEIPT_SCHEMA,
    )
    if receipt["uid"] != uid:
        raise ValueError("init receipt UID conflicts with run case")
    init = repository.load_environment(uid, verify_files=False)
    if receipt["init_sha256"] != canonical_sha256(init.manifest):
        raise ValueError("init receipt was created from a different saved init")
    sample = (receipt_root / "cases" / uid).resolve()
    declared_sample = (receipt_root / receipt["sample_path"]).resolve()
    if declared_sample != sample or not sample.is_dir():
        raise ValueError("init receipt sample path is not canonical")
    expected = {item["path"]: item for item in receipt["artifacts"]}
    actual = {
        item["path"]: item for item in _hash_tree(sample, relative_base=receipt_root)
    }
    if set(actual) != set(expected):
        raise ValueError("init receipt artifact set changed")
    for path, record in actual.items():
        if record != expected[path]:
            raise ValueError(f"init receipt artifact changed: {path}")
    return receipt, sample


def _materialize_run_sample(
    *,
    workspace: Workspace,
    repository: BenchRepository,
    run: Mapping[str, Any],
    uid: str,
    destination: Path,
) -> tuple[dict[str, Any] | None, ReferenceSelection]:
    receipt: dict[str, Any] | None = None
    if run["initialization"]["mode"] == "frozen":
        _materialize_init_sample(
            repository=repository,
            uid=uid,
            destination=destination,
        )
    else:
        receipt, sample = _load_verified_init_receipt(
            workspace=workspace,
            receipt_id=run["initialization"]["receipt"],
            uid=uid,
            repository=repository,
        )
        _copy_tree(sample, destination)
    reference, derivatives = _add_reference_to_sample(
        repository=repository,
        uid=uid,
        destination=destination,
    )
    sources = _materialization_source_identities(repository, uid=uid)
    materialized_files = _fixed_input_files(destination)
    materialization = {
        "format": WORK_MATERIALIZATION_SCHEMA,
        "uid": uid,
        "initialization": {
            "mode": run["initialization"]["mode"],
            "receipt_sha256": (
                None if receipt is None else canonical_sha256(receipt)
            ),
        },
        "sources": sources,
        "derivatives": derivatives,
        "materialized_files": materialized_files,
    }
    validate_document(
        materialization,
        expected_schema=WORK_MATERIALIZATION_SCHEMA,
    )
    write_json_atomic(
        destination / _WORK_MATERIALIZATION_FILENAME,
        materialization,
        exclusive=True,
    )
    return receipt, reference


def _materialization_source_identities(
    repository: BenchRepository,
    *,
    uid: str,
) -> dict[str, str]:
    return {
        "case": canonical_sha256(repository.load_case(uid)),
        "init": canonical_sha256(repository.load_environment(uid).manifest),
        "generation_input": canonical_sha256(
            repository.load_generation_input(uid)
        ),
        "reference": canonical_sha256(repository.load_reference(uid).manifest),
    }


def _verify_materialized_run_sample(
    *,
    workspace: Workspace,
    repository: BenchRepository,
    run: Mapping[str, Any],
    uid: str,
    destination: Path,
) -> tuple[dict[str, Any] | None, ReferenceSelection]:
    if destination.is_symlink() or not destination.is_dir():
        raise ValueError(f"materialized sample must be a non-symlink directory: {destination}")
    document = load_and_validate(
        destination / _WORK_MATERIALIZATION_FILENAME,
        expected_schema=WORK_MATERIALIZATION_SCHEMA,
    )
    if document["uid"] != uid:
        raise ValueError("work materialization UID conflicts with run case")
    if document["initialization"]["mode"] != run["initialization"]["mode"]:
        raise ValueError("work materialization initialization mode changed")
    expected_sources = _materialization_source_identities(repository, uid=uid)
    if document["sources"] != expected_sources:
        raise ValueError("work materialization was built from different bench inputs")

    receipt: dict[str, Any] | None = None
    if run["initialization"]["mode"] == "receipt":
        receipt, _sample = _load_verified_init_receipt(
            workspace=workspace,
            receipt_id=run["initialization"]["receipt"],
            uid=uid,
            repository=repository,
        )
        if document["initialization"]["receipt_sha256"] != canonical_sha256(
            receipt
        ):
            raise ValueError("work materialization init receipt changed")
    elif document["initialization"]["receipt_sha256"] is not None:
        raise ValueError("frozen work materialization has an init receipt")

    actual_files = [
        record
        for record in _fixed_input_files(destination)
        if record["path"] != _WORK_MATERIALIZATION_FILENAME
    ]
    if actual_files != document["materialized_files"]:
        raise ValueError("work materialization fixed inputs changed")

    action_npy = destination / "artifacts" / "gt" / "action" / "action.npy"
    action_meta = destination / "artifacts" / "gt" / "action" / "meta.json"
    action_json = destination / "artifacts" / "gt" / "action" / "action.json"
    _bundle_meta, action_array = load_action_bundle(
        action_npy,
        expected_uid=uid,
        expected_kind="ground_truth_demonstration",
    )
    expected_runtime = gt_action_payload_from_bundle(action_npy)
    if _json_object(action_json, "work GT runtime action") != expected_runtime:
        raise ValueError("work GT runtime action changed")
    derivatives = document["derivatives"]
    if derivatives["gt_action_materialization"] != {
        "array": _file_record(action_npy, destination),
        "meta": _file_record(action_meta, destination),
        "runtime_json": _file_record(action_json, destination),
        "dtype": action_array.dtype.str,
        "shape": list(action_array.shape),
    }:
        raise ValueError("work GT action materialization receipt changed")
    assets = destination / "artifacts" / "gt" / "assets.json"
    if derivatives["gt_assets_manifest"] != _file_record(assets, destination):
        raise ValueError("work GT reference-assets receipt changed")
    reference = repository.load_reference(uid)
    return receipt, reference


def _fixed_input_files(sample: Path) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for path in sorted(sample.rglob("*")):
        relative = path.relative_to(sample)
        # The formal workflow owns these mutable execution domains.
        # They are run evidence, not benchmark input identity.
        if relative.parts[:1] and relative.parts[0] in _RUN_OWNED_TOP_LEVEL:
            continue
        if (
            len(relative.parts) >= 2
            and relative.parts[0] == "artifacts"
            and relative.parts[1] in _RUN_OWNED_ARTIFACTS
        ):
            continue
        if path.is_symlink():
            raise ValueError(f"materialized fixed input must not be a symlink: {path}")
        if path.is_file():
            output.append(
                {
                    "path": relative.as_posix(),
                    "sha256": sha256_file(path),
                    "size": path.stat().st_size,
                }
            )
    return output
