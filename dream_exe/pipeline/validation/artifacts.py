"""Bounded, read-only validation for resumable benchmark artifacts.

The normal artifact readers intentionally preserve current tolerant behavior.
Verified resume needs a stricter trust decision, so this module validates the
files written by the current current implementation producers without changing those readers or
introducing benchmark concerns into the algorithm and simulator domains.
"""

from __future__ import annotations

import csv
import hashlib
import io
import json
import math
import os
import stat
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

import numpy as np

from ...artifacts.layout import (
    execution_artifact_paths,
    trajectory_artifact_paths,
)
from ...evaluation.contracts import (
    EvaluationResultValidationError,
    build_evaluation_plan,
    load_evaluation_result_bundle,
    validate_evaluation_result_bundle,
)
from ...evaluation.execution.metrics import (
    _CSV_COLUMNS as EXEC_METRICS_CSV_COLUMNS,
)
from ...evaluation.execution.metrics import (
    _METRIC_SCOPE as EXEC_METRICS_SCOPE,
)
from ...evaluation.execution.metrics import (
    _METRIC_VERSION as EXEC_METRICS_VERSION,
)
from ...evaluation.execution import CURRENT_TASK_SUCCESS_SCHEMA
from ...sim.execution.action_trace import validate_action_trace_preflight
from ...sim.execution.inputs import parse_pose_from_entry
from ...video2traj.depth.cache import (
    DEPTH_CACHE_SCHEMA,
    validate_canonical_depth_cache_metadata,
)
from ...video2traj.depth.runtime_lineage import (
    DEPTH_RUNTIME_LINEAGE_SCHEMA,
    MAX_DEPTH_RUNTIME_LINEAGE_NPY_BYTES,
    validate_depth_runtime_lineage,
)
from ..records.state import digest_output_file

__all__ = [
    "BenchmarkArtifactValidationError",
    "read_bounded_json_object",
    "validate_benchmark_stage_artifacts",
]


_MAX_JSON_BYTES = 64 * 1024 * 1024
_MAX_CSV_BYTES = 64 * 1024 * 1024
_MAX_SEQUENCE_ITEMS = 1_000_000


class BenchmarkArtifactValidationError(ValueError):
    """One stable, stage-scoped reason that verified resume can rerun."""

    def __init__(self, stage: str, reason: str) -> None:
        self.stage = str(stage)
        self.reason = str(reason)
        super().__init__(f"{self.stage} artifact validation failed: {self.reason}")


class _StrictJSONError(ValueError):
    pass


def _fail(stage: str, reason: str) -> None:
    raise BenchmarkArtifactValidationError(stage, reason)


def _bounded_regular_bytes(
    path: Path,
    *,
    stage: str,
    reason: str,
    limit: int,
    containment_root: Path | None = None,
) -> bytes:
    flags = (
        os.O_RDONLY
        | getattr(os, "O_CLOEXEC", 0)
        | getattr(os, "O_NOFOLLOW", 0)
        | getattr(os, "O_NONBLOCK", 0)
    )
    try:
        if containment_root is None:
            before = path.lstat()
            descriptor = os.open(path, flags)
        else:
            before = None
            root = Path(containment_root).expanduser().absolute()
            source = Path(path).expanduser().absolute()
            relative = source.relative_to(root)
            if not relative.parts:
                raise ValueError("artifact path must name a file below its root")
            directory_flags = (
                os.O_RDONLY
                | getattr(os, "O_CLOEXEC", 0)
                | getattr(os, "O_DIRECTORY", 0)
                | getattr(os, "O_NOFOLLOW", 0)
            )
            directory_descriptor = os.open(root, directory_flags)
            try:
                for part in relative.parts[:-1]:
                    next_descriptor = os.open(
                        part,
                        directory_flags,
                        dir_fd=directory_descriptor,
                    )
                    os.close(directory_descriptor)
                    directory_descriptor = next_descriptor
                descriptor = os.open(
                    relative.parts[-1],
                    flags,
                    dir_fd=directory_descriptor,
                )
            finally:
                os.close(directory_descriptor)
    except (OSError, ValueError):
        _fail(stage, reason)
    try:
        metadata = os.fstat(descriptor)
        if (
            not stat.S_ISREG(metadata.st_mode)
            or (
                before is not None
                and (
                    not stat.S_ISREG(before.st_mode)
                    or before.st_dev != metadata.st_dev
                    or before.st_ino != metadata.st_ino
                )
            )
            or metadata.st_size > limit
        ):
            _fail(stage, reason)
        chunks: list[bytes] = []
        size = 0
        while True:
            chunk = os.read(
                descriptor,
                min(1024 * 1024, limit + 1 - size),
            )
            if not chunk:
                break
            chunks.append(chunk)
            size += len(chunk)
            if size > limit:
                _fail(stage, reason)
        return b"".join(chunks)
    except OSError:
        _fail(stage, reason)
    finally:
        os.close(descriptor)


def _strict_json_object(
    path: Path,
    *,
    stage: str,
    reason: str,
    containment_root: Path | None = None,
) -> dict[str, Any]:
    encoded = _bounded_regular_bytes(
        path,
        stage=stage,
        reason=reason,
        limit=_MAX_JSON_BYTES,
        containment_root=containment_root,
    )

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
    ):
        _fail(stage, reason)
    if not isinstance(payload, dict):
        _fail(stage, reason)
    return payload


def read_bounded_json_object(
    path: str | Path,
    *,
    stage: str,
    reason: str,
    containment_root: str | Path | None = None,
) -> dict[str, Any]:
    """Read one bounded regular JSON object through the strict resume reader.

    When ``containment_root`` is provided, every existing path component from
    that root through the file is opened without following symlinks.
    """

    return _strict_json_object(
        Path(path),
        stage=str(stage),
        reason=str(reason),
        containment_root=(None if containment_root is None else Path(containment_root)),
    )


def _containment_root_if_beneath(
    path: Path,
    root: Path,
) -> Path | None:
    try:
        path.expanduser().absolute().relative_to(root.expanduser().absolute())
    except ValueError:
        return None
    return root


def _bounded_list(
    value: Any,
    *,
    stage: str,
    reason: str,
    allow_empty: bool,
) -> list[Any]:
    if not isinstance(value, list):
        _fail(stage, reason)
    if not allow_empty and not value:
        _fail(stage, reason)
    if len(value) > _MAX_SEQUENCE_ITEMS:
        _fail(stage, reason)
    return value


def _mapping(
    value: Any,
    *,
    stage: str,
    reason: str,
) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        _fail(stage, reason)
    return value


def _nonnegative_integer(
    value: Any,
    *,
    stage: str,
    reason: str,
) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        _fail(stage, reason)
    return int(value)


def _finite_number_or_none(
    value: Any,
    *,
    stage: str,
    reason: str,
) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        _fail(stage, reason)
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        _fail(stage, reason)
    if not math.isfinite(number):
        _fail(stage, reason)
    return number


def _strict_boolean(
    value: Any,
    *,
    stage: str,
    reason: str,
) -> bool:
    if not isinstance(value, bool):
        _fail(stage, reason)
    return value


def _action_gripper_contract(
    *,
    action_meta: Mapping[str, Any],
    trajectory_frames: Sequence[int],
    gripper_events: Sequence[Any],
    gripper_grasps: Sequence[bool],
    stage: str,
) -> tuple[list[Any], list[bool]]:
    """Return the gripper rows seen by the action checkpoint selector.

    The action builder can move a close event earlier before it selects
    compressed checkpoints.  That adjustment is fully declared in planner
    metadata, so verified resume can reproduce the selector input without
    importing the algorithm implementation into the benchmark domain.
    """

    events = list(gripper_events)
    grasps = list(gripper_grasps)
    planner_raw = action_meta.get("planner")
    if planner_raw is None:
        return events, grasps
    planner = _mapping(
        planner_raw,
        stage=stage,
        reason="invalid_action",
    )
    constraint_raw = planner.get("gripper_constraint")
    if constraint_raw is None:
        return events, grasps
    constraint = _mapping(
        constraint_raw,
        stage=stage,
        reason="invalid_action",
    )
    enabled = _strict_boolean(
        constraint.get("enabled"),
        stage=stage,
        reason="invalid_action",
    )
    applied = _strict_boolean(
        constraint.get("applied"),
        stage=stage,
        reason="invalid_action",
    )
    stages = _bounded_list(
        constraint.get("stages"),
        stage=stage,
        reason="invalid_action",
        allow_empty=True,
    )
    if applied and not enabled:
        _fail(stage, "invalid_action")

    frame_to_index = {frame: index for index, frame in enumerate(trajectory_frames)}
    if len(frame_to_index) != len(trajectory_frames):
        _fail(stage, "trajectory_action_alignment")
    applied_stages = 0
    for raw_constraint_stage in stages:
        constraint_stage = _mapping(
            raw_constraint_stage,
            stage=stage,
            reason="invalid_action",
        )
        stage_applied = _strict_boolean(
            constraint_stage.get("applied"),
            stage=stage,
            reason="invalid_action",
        )
        if not stage_applied:
            continue
        applied_stages += 1
        original_frame = _nonnegative_integer(
            constraint_stage.get("original_close_frame"),
            stage=stage,
            reason="invalid_action",
        )
        shifted_frame = _nonnegative_integer(
            constraint_stage.get("shifted_close_to_frame"),
            stage=stage,
            reason="invalid_action",
        )
        if (
            original_frame not in frame_to_index
            or shifted_frame not in frame_to_index
            or gripper_events[frame_to_index[original_frame]] != "close"
        ):
            _fail(stage, "invalid_action")
        shifted_index = frame_to_index[shifted_frame]
        stop_frame = constraint_stage.get("hold_until_frame")
        if stop_frame is None:
            stop_index = len(trajectory_frames)
        else:
            stop_frame = _nonnegative_integer(
                stop_frame,
                stage=stage,
                reason="invalid_action",
            )
            if stop_frame not in frame_to_index:
                _fail(stage, "invalid_action")
            stop_index = frame_to_index[stop_frame]
        if stop_index <= shifted_index:
            _fail(stage, "invalid_action")
        for index in range(shifted_index, stop_index):
            grasps[index] = True
            events[index] = "close" if index == shifted_index else None
        if original_frame != shifted_frame:
            events[frame_to_index[original_frame]] = None
    if applied != bool(applied_stages):
        _fail(stage, "invalid_action")
    return events, grasps


def _action_checkpoint_source_indices(
    *,
    action_meta: Mapping[str, Any],
    checkpoint_frames: Sequence[int],
    trajectory_frames: Sequence[int],
    gripper_events: Sequence[Any],
    gripper_grasps: Sequence[bool],
    stage: str,
) -> tuple[list[int], list[Any]]:
    """Validate and reproduce the action builder's checkpoint selection."""

    effective_events, effective_grasps = _action_gripper_contract(
        action_meta=action_meta,
        trajectory_frames=trajectory_frames,
        gripper_events=gripper_events,
        gripper_grasps=gripper_grasps,
        stage=stage,
    )
    planner_raw = action_meta.get("planner")
    planner = (
        None
        if planner_raw is None
        else _mapping(
            planner_raw,
            stage=stage,
            reason="invalid_action",
        )
    )
    compression_raw = None if planner is None else planner.get("checkpoint_compression")
    if compression_raw is None:
        if list(checkpoint_frames) != list(trajectory_frames):
            _fail(stage, "trajectory_action_alignment")
        return list(range(len(trajectory_frames))), effective_events

    compression = _mapping(
        compression_raw,
        stage=stage,
        reason="invalid_action",
    )
    enabled = _strict_boolean(
        compression.get("enabled"),
        stage=stage,
        reason="invalid_action",
    )
    applied = _strict_boolean(
        compression.get("applied"),
        stage=stage,
        reason="invalid_action",
    )
    stride = _nonnegative_integer(
        compression.get("stride"),
        stage=stage,
        reason="invalid_action",
    )
    radius = _nonnegative_integer(
        compression.get("keep_event_neighbors"),
        stage=stage,
        reason="invalid_action",
    )
    input_count = _nonnegative_integer(
        compression.get("num_input_checkpoints"),
        stage=stage,
        reason="invalid_action",
    )
    output_count = _nonnegative_integer(
        compression.get("num_output_checkpoints"),
        stage=stage,
        reason="invalid_action",
    )
    skipped_count = _nonnegative_integer(
        compression.get("num_skipped_checkpoints"),
        stage=stage,
        reason="invalid_action",
    )
    total = len(trajectory_frames)
    if (
        stride < 1
        or input_count != total
        or output_count != len(checkpoint_frames)
        or skipped_count != total - len(checkpoint_frames)
        or applied != (len(checkpoint_frames) != total)
        or (applied and not enabled)
    ):
        _fail(stage, "trajectory_action_alignment")

    protected: set[int] = {0, total - 1}
    if enabled:
        for index, event in enumerate(effective_events):
            if event is None:
                continue
            protected.update(
                range(max(0, index - radius), min(total, index + radius + 1))
            )
        selected = [
            index
            for index, grasp in enumerate(effective_grasps)
            if not grasp or index in protected or index % stride == 0
        ]
    else:
        selected = list(range(total))
    expected_frames = [trajectory_frames[index] for index in selected]
    if (
        list(checkpoint_frames) != expected_frames
        or applied != (len(selected) != total)
        or output_count != len(selected)
        or skipped_count != total - len(selected)
    ):
        _fail(stage, "trajectory_action_alignment")
    return selected, effective_events


def _contained_reference(
    value: Any,
    *,
    expected: Path,
    root: Path,
    stage: str,
    reason: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(stage, reason)
    raw = value.strip()
    candidate = Path(raw).expanduser()
    if candidate.is_absolute():
        lexical = candidate.absolute()
    else:
        pure = PurePosixPath(raw)
        if pure.is_absolute() or "." in pure.parts or ".." in pure.parts:
            _fail(stage, reason)
        lexical = (root / candidate).absolute()
    if lexical != expected.absolute():
        _fail(stage, reason)


def _validate_manifest_section(
    manifest: Mapping[str, Any],
    *,
    section_name: str,
    reference_key: str,
    expected_path: Path,
    root: Path,
) -> None:
    stage = "video2traj"
    reason = "invalid_manifest"
    section = _mapping(
        manifest.get(section_name),
        stage=stage,
        reason=reason,
    )
    _contained_reference(
        section.get("dir"),
        expected=expected_path.parent,
        root=root,
        stage=stage,
        reason=reason,
    )
    _contained_reference(
        section.get(reference_key),
        expected=expected_path,
        root=root,
        stage=stage,
        reason=reason,
    )


def _depth_contract(
    settings: Mapping[str, Any],
) -> dict[str, Any] | None:
    raw = settings.get("video2traj_depth_contract")
    if raw is None:
        return None
    if not isinstance(raw, Mapping):
        _fail("video2traj", "invalid_depth_artifact_contract")
    contract = dict(raw)
    if set(contract) != {"mode", "save_canonical_mp4", "paths"}:
        _fail("video2traj", "invalid_depth_artifact_contract")
    mode = contract["mode"]
    if mode not in {
        "estimated_current_safe_cache",
        "rollout_gt_depth_reference",
    }:
        _fail("video2traj", "invalid_depth_artifact_contract")
    if not isinstance(contract["save_canonical_mp4"], bool):
        _fail("video2traj", "invalid_depth_artifact_contract")
    if not isinstance(contract["paths"], Mapping):
        _fail("video2traj", "invalid_depth_artifact_contract")
    return {
        "mode": mode,
        "save_canonical_mp4": contract["save_canonical_mp4"],
        "paths": dict(contract["paths"]),
    }


def _live_identity(
    path: Path,
    *,
    sample_root: Path,
    run_root: Path,
    reason: str,
) -> dict[str, Any]:
    try:
        return digest_output_file(
            path,
            sample_root=sample_root,
            run_root=run_root,
        )
    except (FileNotFoundError, OSError, TypeError, ValueError):
        _fail("video2traj", reason)


def _resolve_sample_reference(
    value: Any,
    *,
    base: Path,
    sample_root: Path,
    reason: str,
) -> Path:
    if not isinstance(value, str) or not value.strip():
        _fail("video2traj", reason)
    raw = value.strip()
    if "\\" in raw:
        _fail("video2traj", reason)
    candidate = Path(raw).expanduser()
    lexical = Path(
        os.path.abspath(candidate if candidate.is_absolute() else base / candidate)
    )
    try:
        lexical.relative_to(sample_root.absolute())
    except ValueError:
        _fail("video2traj", reason)
    return lexical


def _validate_live_declared_identity(
    record: Mapping[str, Any],
    *,
    path: Path,
    sample_root: Path,
    run_root: Path,
    size_field: str,
    digest_field: str,
    reason: str,
) -> dict[str, Any]:
    live = _live_identity(
        path,
        sample_root=sample_root,
        run_root=run_root,
        reason=reason,
    )
    if (
        record.get(size_field) != live["size"]
        or record.get(digest_field) != live["sha256"]
    ):
        _fail("video2traj", reason)
    return live


def _validate_estimated_depth_artifacts(
    *,
    assets: Mapping[str, Any],
    contract: Mapping[str, Any],
    traj_root: Path,
    sample_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    stage = "video2traj"
    reason = "invalid_depth_publication"
    layout = trajectory_artifact_paths(traj_root)
    depth_root = layout["depth_dir"]
    base_roles = {
        "depth_npy",
        "depth_meta_npy",
        "depth_cache_meta_json",
        "depth_manifest_json",
    }
    media_roles = {
        "depth_mp4",
        "depth_frame0_png",
        "depth_contact_png",
        "depth_vis_meta_json",
    }
    lineage_roles = {
        "depth_runtime_lineage_json",
        "eef_consumed_depth_samples_npy",
    }
    expected_roles = set(base_roles)
    if bool(contract["save_canonical_mp4"]):
        expected_roles.update(media_roles)
    raw_paths = dict(contract["paths"])
    declared_lineage = set(raw_paths).intersection(lineage_roles)
    if declared_lineage and declared_lineage != lineage_roles:
        _fail(stage, "invalid_depth_artifact_contract")
    if declared_lineage:
        expected_roles.update(lineage_roles)
    if set(raw_paths) != expected_roles:
        _fail(stage, "invalid_depth_artifact_contract")
    paths = {role: Path(raw_paths[role]).absolute() for role in raw_paths}
    for role in expected_roles:
        if paths[role] != layout[role].absolute():
            _fail(stage, "invalid_depth_artifact_contract")

    depth_assets = _mapping(
        assets.get("depth"),
        stage=stage,
        reason=reason,
    )
    if (
        depth_assets.get("source") == "rollout_gt_depth"
        or not isinstance(depth_assets.get("depth_model"), str)
        or not str(depth_assets.get("depth_model", "")).strip()
    ):
        _fail(stage, reason)
    _contained_reference(
        depth_assets.get("dir"),
        expected=depth_root,
        root=traj_root,
        stage=stage,
        reason=reason,
    )
    for key, role in (
        ("estimated_depth_cache_path", "depth_npy"),
        ("depth_npy", "depth_npy"),
        ("depth_meta_npy", "depth_meta_npy"),
        ("depth_cache_meta_json", "depth_cache_meta_json"),
        ("depth_manifest_json", "depth_manifest_json"),
    ):
        _contained_reference(
            depth_assets.get(key),
            expected=paths[role],
            root=traj_root,
            stage=stage,
            reason=reason,
        )
    if bool(contract["save_canonical_mp4"]):
        for key, role in (
            ("depth_mp4", "depth_mp4"),
            ("depth_frame0_png", "depth_frame0_png"),
            ("depth_contact_png", "depth_contact_png"),
            ("depth_vis_meta_json", "depth_vis_meta_json"),
        ):
            _contained_reference(
                depth_assets.get(key),
                expected=paths[role],
                root=traj_root,
                stage=stage,
                reason=reason,
            )
    elif depth_assets.get("depth_mp4") is not None or any(
        depth_assets.get(key) is not None
        for key in (
            "depth_frame0_png",
            "depth_contact_png",
            "depth_vis_meta_json",
        )
    ):
        _fail(stage, reason)

    metadata = _strict_json_object(
        paths["depth_cache_meta_json"],
        stage=stage,
        reason="invalid_depth_cache_metadata",
        containment_root=sample_root,
    )
    metadata_alias = _strict_json_object(
        paths["depth_meta_npy"],
        stage=stage,
        reason="invalid_depth_metadata_alias",
        containment_root=sample_root,
    )
    authoritative_identity = _live_identity(
        paths["depth_cache_meta_json"],
        sample_root=sample_root,
        run_root=run_root,
        reason="invalid_depth_cache_metadata",
    )
    alias_identity = _live_identity(
        paths["depth_meta_npy"],
        sample_root=sample_root,
        run_root=run_root,
        reason="invalid_depth_metadata_alias",
    )
    if (
        metadata_alias != metadata
        or alias_identity["size"] != authoritative_identity["size"]
        or alias_identity["sha256"] != authoritative_identity["sha256"]
    ):
        _fail(stage, "invalid_depth_metadata_alias")
    try:
        metadata = validate_canonical_depth_cache_metadata(
            metadata,
            label="formal depth cache metadata",
        )
    except (KeyError, TypeError, ValueError):
        _fail(stage, "invalid_depth_cache_metadata")
    if metadata.get("depth_model") != depth_assets.get("depth_model"):
        _fail(stage, "invalid_depth_cache_metadata")
    _expected_path_value(
        metadata.get("cache_path"),
        expected=paths["depth_npy"],
        base=depth_root,
        stage=stage,
        reason="invalid_depth_cache_metadata",
    )
    depth_live = _live_identity(
        paths["depth_npy"],
        sample_root=sample_root,
        run_root=run_root,
        reason="invalid_depth_cache_identity",
    )
    depth_record = _mapping(
        metadata.get("depth"),
        stage=stage,
        reason="invalid_depth_cache_metadata",
    )
    if depth_record.get("file_sha256") != depth_live["sha256"]:
        _fail(stage, "invalid_depth_cache_identity")

    manifest = _strict_json_object(
        paths["depth_manifest_json"],
        stage=stage,
        reason="invalid_depth_manifest",
        containment_root=sample_root,
    )
    if (
        manifest.get("schema") != DEPTH_CACHE_SCHEMA
        or manifest.get("format") != metadata.get("format")
        or manifest.get("input_identity") != metadata.get("input_identity")
        or manifest.get("parameters") != metadata.get("parameters")
        or manifest.get("parameter_fingerprint")
        != metadata.get("parameter_fingerprint")
        or manifest.get("target_depths") != metadata.get("target_depths")
    ):
        _fail(stage, "invalid_depth_manifest")
    model = _mapping(
        manifest.get("model"),
        stage=stage,
        reason="invalid_depth_manifest",
    )
    if (
        model.get("id") != metadata.get("depth_model")
        or model.get("provenance") != metadata.get("model_provenance")
        or model.get("provenance_fingerprint")
        != metadata.get("model_provenance_fingerprint")
    ):
        _fail(stage, "invalid_depth_manifest")
    manifest_depth = _mapping(
        manifest.get("depth"),
        stage=stage,
        reason="invalid_depth_manifest",
    )
    for key, value in depth_record.items():
        if manifest_depth.get(key) != value:
            _fail(stage, "invalid_depth_manifest")
    for key, role in (
        ("path", "depth_npy"),
        ("meta_path", "depth_cache_meta_json"),
        ("compat_meta_path", "depth_meta_npy"),
    ):
        _expected_path_value(
            manifest_depth.get(key),
            expected=paths[role],
            base=depth_root,
            stage=stage,
            reason="invalid_depth_manifest",
        )

    media = _mapping(
        metadata.get("debug_media"),
        stage=stage,
        reason="invalid_depth_cache_metadata",
    )
    expected_media = (
        {
            "depth_mp4",
            "depth_frame0",
            "depth_contact",
            "depth_vis_meta",
        }
        if bool(contract["save_canonical_mp4"])
        else set()
    )
    if set(media) != expected_media or manifest.get("debug_media") != media:
        _fail(stage, "invalid_depth_debug_media")
    media_roles = {
        "depth_mp4": ("depth_mp4", "video/mp4"),
        "depth_frame0": ("depth_frame0_png", "image/png"),
        "depth_contact": ("depth_contact_png", "image/png"),
        "depth_vis_meta": ("depth_vis_meta_json", "application/json"),
    }
    for name in sorted(expected_media):
        record = _mapping(
            media[name],
            stage=stage,
            reason="invalid_depth_debug_media",
        )
        role, media_type = media_roles[name]
        _expected_path_value(
            record.get("path"),
            expected=paths[role],
            base=depth_root,
            stage=stage,
            reason="invalid_depth_debug_media",
        )
        if record.get("media_type") != media_type:
            _fail(stage, "invalid_depth_debug_media")
        live = _validate_live_declared_identity(
            record,
            path=paths[role],
            sample_root=sample_root,
            run_root=run_root,
            size_field="size",
            digest_field="sha256",
            reason="invalid_depth_debug_media",
        )
        source = _mapping(
            record.get("source"),
            stage=stage,
            reason="invalid_depth_debug_media",
        )
        if source.get("size") != live["size"] or source.get("sha256") != live["sha256"]:
            _fail(stage, "invalid_depth_debug_media")
    runtime_lineage = None
    if declared_lineage:
        geometry_assets = _mapping(
            assets.get("geometry"),
            stage=stage,
            reason="invalid_depth_runtime_lineage",
        )
        calibration_assets = _mapping(
            geometry_assets.get("target_depth_calibration"),
            stage=stage,
            reason="invalid_depth_runtime_lineage",
        )
        for key, role in (
            ("runtime_lineage_json", "depth_runtime_lineage_json"),
            (
                "eef_consumed_depth_samples_npy",
                "eef_consumed_depth_samples_npy",
            ),
        ):
            _contained_reference(
                calibration_assets.get(key),
                expected=paths[role],
                root=traj_root,
                stage=stage,
                reason="invalid_depth_runtime_lineage",
            )
        runtime_lineage = _strict_json_object(
            paths["depth_runtime_lineage_json"],
            stage=stage,
            reason="invalid_depth_runtime_lineage",
            containment_root=sample_root,
        )
        if runtime_lineage.get("schema") != DEPTH_RUNTIME_LINEAGE_SCHEMA:
            _fail(stage, "invalid_depth_runtime_lineage")
        sample_bytes = _bounded_regular_bytes(
            paths["eef_consumed_depth_samples_npy"],
            stage=stage,
            reason="invalid_depth_runtime_lineage",
            limit=MAX_DEPTH_RUNTIME_LINEAGE_NPY_BYTES,
            containment_root=sample_root,
        )
        try:
            samples = np.load(
                io.BytesIO(sample_bytes),
                allow_pickle=False,
            )
            validate_depth_runtime_lineage(runtime_lineage, samples)
        except (OSError, TypeError, ValueError):
            _fail(stage, "invalid_depth_runtime_lineage")
        consumed_record = _mapping(
            runtime_lineage.get("consumed_samples"),
            stage=stage,
            reason="invalid_depth_runtime_lineage",
        )
        if (
            consumed_record.get("file_sha256")
            != hashlib.sha256(sample_bytes).hexdigest()
        ):
            _fail(stage, "invalid_depth_runtime_lineage")
        lineage_depth = _mapping(
            runtime_lineage.get("depth"),
            stage=stage,
            reason="invalid_depth_runtime_lineage",
        )
        lineage_arrays = _mapping(
            runtime_lineage.get("arrays"),
            stage=stage,
            reason="invalid_depth_runtime_lineage",
        )
        lineage_canonical = _mapping(
            lineage_arrays.get("canonical_depth"),
            stage=stage,
            reason="invalid_depth_runtime_lineage",
        )
        if (
            not isinstance(lineage_depth.get("runtime_source"), str)
            or not str(lineage_depth.get("runtime_source", "")).strip()
            or lineage_depth.get("canonical_publication_source")
            != metadata.get("source")
            or lineage_depth.get("model") != metadata.get("depth_model")
            or lineage_depth.get("model_provenance_fingerprint")
            != metadata.get("model_provenance_fingerprint")
            or lineage_depth.get("parameter_fingerprint")
            != metadata.get("parameter_fingerprint")
            or lineage_canonical.get("shape") != depth_record.get("shape")
            or lineage_canonical.get("dtype") != depth_record.get("dtype")
            or lineage_canonical.get("array_fingerprint")
            != depth_record.get("array_fingerprint")
        ):
            _fail(stage, "invalid_depth_runtime_lineage")
    return {
        "mode": contract["mode"],
        "assets": dict(depth_assets),
        "cache_metadata": metadata,
        "manifest": manifest,
        "runtime_lineage": runtime_lineage,
    }


def _validate_gt_depth_reference(
    *,
    assets: Mapping[str, Any],
    traj_root: Path,
    sample_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    stage = "video2traj"
    reason = "invalid_gt_depth_reference"
    depth_assets = _mapping(
        assets.get("depth"),
        stage=stage,
        reason=reason,
    )
    if depth_assets.get("source") != "rollout_gt_depth":
        _fail(stage, reason)
    for key in (
        "estimated_depth_cache_path",
        "depth_npy",
        "depth_mp4",
        "depth_meta_npy",
        "depth_cache_meta_json",
        "depth_manifest_json",
        "depth_frame0_png",
        "depth_contact_png",
        "depth_vis_meta_json",
    ):
        if depth_assets.get(key) is not None:
            _fail(stage, reason)
    source_path = _resolve_sample_reference(
        depth_assets.get("rollout_gt_depth_path"),
        base=traj_root,
        sample_root=sample_root,
        reason=reason,
    )
    identity = _mapping(
        depth_assets.get("rollout_gt_depth_identity"),
        stage=stage,
        reason=reason,
    )
    identity_path = _resolve_sample_reference(
        identity.get("path"),
        base=traj_root,
        sample_root=sample_root,
        reason=reason,
    )
    if identity_path != source_path or identity.get("kind") != "file":
        _fail(stage, reason)
    _validate_live_declared_identity(
        identity,
        path=source_path,
        sample_root=sample_root,
        run_root=run_root,
        size_field="size_bytes",
        digest_field="file_sha256",
        reason=reason,
    )
    try:
        array = np.load(
            source_path,
            mmap_mode="r",
            allow_pickle=False,
        )
    except (OSError, TypeError, ValueError):
        _fail(stage, reason)
    if list(array.shape) != identity.get("shape") or str(array.dtype) != identity.get(
        "dtype"
    ):
        _fail(stage, reason)

    diagnostics = _mapping(
        identity.get("diagnostics"),
        stage=stage,
        reason="invalid_gt_depth_diagnostics",
    )
    if depth_assets.get("rollout_gt_depth_diagnostics") != diagnostics:
        _fail(stage, "invalid_gt_depth_diagnostics")
    allowed_roles = {
        "metric_mp4",
        "frame0_png",
        "contact_png",
        "meta_json",
    }
    if not set(diagnostics).issubset(allowed_roles):
        _fail(stage, "invalid_gt_depth_diagnostics")
    for role, raw_record in diagnostics.items():
        record = _mapping(
            raw_record,
            stage=stage,
            reason="invalid_gt_depth_diagnostics",
        )
        path = _resolve_sample_reference(
            record.get("path"),
            base=traj_root,
            sample_root=sample_root,
            reason="invalid_gt_depth_diagnostics",
        )
        _validate_live_declared_identity(
            record,
            path=path,
            sample_root=sample_root,
            run_root=run_root,
            size_field="size_bytes",
            digest_field="file_sha256",
            reason="invalid_gt_depth_diagnostics",
        )
    issues = identity.get("diagnostic_issues")
    if not isinstance(issues, list):
        _fail(stage, "invalid_gt_depth_diagnostics")
    return {
        "mode": "rollout_gt_depth_reference",
        "assets": dict(depth_assets),
        "source_identity": dict(identity),
    }


def _validate_depth_artifacts(
    *,
    settings: Mapping[str, Any],
    assets: Mapping[str, Any],
    traj_root: Path,
) -> dict[str, Any] | None:
    contract = _depth_contract(settings)
    if contract is None:
        return None
    context = _mapping(
        settings.get("context"),
        stage="video2traj",
        reason="invalid_context",
    )
    formal = _mapping(
        context.get("formal"),
        stage="video2traj",
        reason="invalid_context",
    )
    sample_root = Path(formal["sample_root"]).absolute()
    run_root = Path(formal["run_root"]).absolute()
    if contract["mode"] == "rollout_gt_depth_reference":
        if contract["paths"]:
            _fail("video2traj", "invalid_depth_artifact_contract")
        return _validate_gt_depth_reference(
            assets=assets,
            traj_root=traj_root,
            sample_root=sample_root,
            run_root=run_root,
        )
    return _validate_estimated_depth_artifacts(
        assets=assets,
        contract=contract,
        traj_root=traj_root,
        sample_root=sample_root,
        run_root=run_root,
    )


def _validate_video2traj_artifacts(
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    stage = "video2traj"
    context = _mapping(
        settings.get("context"),
        stage=stage,
        reason="invalid_context",
    )
    formal = _mapping(
        context.get("formal"),
        stage=stage,
        reason="invalid_context",
    )
    root = Path(formal["traj_dir"]).absolute()
    sample_root = Path(formal["sample_root"]).absolute()
    manifest_path = Path(formal["trajectory_manifest"]).absolute()
    if manifest_path != trajectory_artifact_paths(root)["trajectory_manifest"]:
        _fail(stage, "invalid_context")
    trajectory_path = Path(formal["ee_traj"]).absolute()
    gripper_path = Path(formal["gripper"]).absolute()
    action_path = Path(formal["action"]).absolute()

    manifest = _strict_json_object(
        manifest_path,
        stage=stage,
        reason="invalid_manifest",
        containment_root=sample_root,
    )
    _validate_manifest_section(
        manifest,
        section_name="trajectory",
        reference_key="ee_traj_json",
        expected_path=trajectory_path,
        root=root,
    )
    _validate_manifest_section(
        manifest,
        section_name="gripper",
        reference_key="gripper_json",
        expected_path=gripper_path,
        root=root,
    )
    _validate_manifest_section(
        manifest,
        section_name="action",
        reference_key="action_json",
        expected_path=action_path,
        root=root,
    )
    depth_artifacts = _validate_depth_artifacts(
        settings=settings,
        assets=manifest,
        traj_root=root,
    )

    trajectory = _strict_json_object(
        trajectory_path,
        stage=stage,
        reason="invalid_trajectory",
        containment_root=sample_root,
    )
    trajectory_rows = _bounded_list(
        trajectory.get("eef_controller"),
        stage=stage,
        reason="invalid_trajectory",
        allow_empty=False,
    )
    trajectory_frames: list[int] = []
    for raw_row in trajectory_rows:
        row = _mapping(
            raw_row,
            stage=stage,
            reason="invalid_trajectory",
        )
        frame = _nonnegative_integer(
            row.get("frame"),
            stage=stage,
            reason="invalid_trajectory",
        )
        if trajectory_frames and frame <= trajectory_frames[-1]:
            _fail(stage, "invalid_trajectory")
        try:
            position, rotation = parse_pose_from_entry(
                dict(row),
                traj_key="eef_controller",
            )
        except (KeyError, TypeError, ValueError):
            _fail(stage, "invalid_trajectory")
        if not np.all(np.isfinite(position)):
            _fail(stage, "invalid_trajectory")
        if rotation is not None and not np.all(np.isfinite(rotation)):
            _fail(stage, "invalid_trajectory")
        trajectory_frames.append(frame)

    gripper = _strict_json_object(
        gripper_path,
        stage=stage,
        reason="invalid_gripper",
        containment_root=sample_root,
    )
    gripper_meta = _mapping(
        gripper.get("meta"),
        stage=stage,
        reason="invalid_gripper",
    )
    if _nonnegative_integer(
        gripper_meta.get("T"),
        stage=stage,
        reason="invalid_gripper",
    ) != len(trajectory_frames):
        _fail(stage, "trajectory_gripper_alignment")
    gripper_rows = _bounded_list(
        gripper.get("actions"),
        stage=stage,
        reason="invalid_gripper",
        allow_empty=False,
    )
    if len(gripper_rows) != len(trajectory_frames):
        _fail(stage, "trajectory_gripper_alignment")
    gripper_frames: list[int] = []
    gripper_events: list[Any] = []
    gripper_grasps: list[bool] = []
    for raw_row in gripper_rows:
        row = _mapping(
            raw_row,
            stage=stage,
            reason="invalid_gripper",
        )
        frame = _nonnegative_integer(
            row.get("frame"),
            stage=stage,
            reason="invalid_gripper",
        )
        if not isinstance(row.get("valid"), bool):
            _fail(stage, "invalid_gripper")
        if bool(row["valid"]):
            _finite_number_or_none(
                row.get("gripper_cmd"),
                stage=stage,
                reason="invalid_gripper",
            )
            if row.get("gripper_cmd") is None:
                _fail(stage, "invalid_gripper")
        event = row.get("event")
        if event is not None and not isinstance(event, str):
            _fail(stage, "invalid_gripper")
        gripper_frames.append(frame)
        gripper_events.append(event)
        gripper_grasps.append(
            bool(
                row.get(
                    "grasp",
                    str(row.get("state") or "open") != "open",
                )
            )
        )
    if gripper_frames != trajectory_frames:
        _fail(stage, "trajectory_gripper_alignment")

    action = _strict_json_object(
        action_path,
        stage=stage,
        reason="invalid_action",
        containment_root=sample_root,
    )
    action_meta = _mapping(
        action.get("meta"),
        stage=stage,
        reason="invalid_action",
    )
    if action_meta.get("format") != "action":
        _fail(stage, "invalid_action")
    try:
        validate_action_trace_preflight(action, None)
    except (TypeError, ValueError):
        _fail(stage, "invalid_action")
    checkpoints = _bounded_list(
        action.get("checkpoints"),
        stage=stage,
        reason="invalid_action",
        allow_empty=False,
    )
    steps = _bounded_list(
        action.get("steps"),
        stage=stage,
        reason="invalid_action",
        allow_empty=False,
    )
    checkpoint_frames: list[int] = []
    checkpoint_events: list[Any] = []
    for raw_checkpoint in checkpoints:
        checkpoint = _mapping(
            raw_checkpoint,
            stage=stage,
            reason="invalid_action",
        )
        frame = _nonnegative_integer(
            checkpoint.get("frame"),
            stage=stage,
            reason="invalid_action",
        )
        checkpoint_frames.append(frame)
        checkpoint_events.append(checkpoint.get("gripper_event"))
    source_indices, effective_gripper_events = _action_checkpoint_source_indices(
        action_meta=action_meta,
        checkpoint_frames=checkpoint_frames,
        trajectory_frames=trajectory_frames,
        gripper_events=gripper_events,
        gripper_grasps=gripper_grasps,
        stage=stage,
    )
    for checkpoint_event, source_index in zip(
        checkpoint_events,
        source_indices,
    ):
        if checkpoint_event != effective_gripper_events[source_index]:
            _fail(stage, "gripper_action_alignment")

    for index, raw_step in enumerate(steps):
        step = _mapping(
            raw_step,
            stage=stage,
            reason="invalid_action",
        )
        if (
            _nonnegative_integer(
                step.get("step_index"),
                stage=stage,
                reason="invalid_action",
            )
            != index
        ):
            _fail(stage, "action_step_alignment")
        checkpoint_index = _nonnegative_integer(
            step.get("target_checkpoint_index"),
            stage=stage,
            reason="invalid_action",
        )
        if checkpoint_index >= len(checkpoints):
            _fail(stage, "action_step_alignment")
        if (
            _nonnegative_integer(
                step.get("target_frame"),
                stage=stage,
                reason="invalid_action",
            )
            != checkpoint_frames[checkpoint_index]
        ):
            _fail(stage, "action_step_alignment")

    summary = _mapping(
        action_meta.get("summary"),
        stage=stage,
        reason="invalid_action",
    )
    if _nonnegative_integer(
        summary.get("num_checkpoints"),
        stage=stage,
        reason="invalid_action",
    ) != len(checkpoints) or _nonnegative_integer(
        summary.get("num_total_steps"),
        stage=stage,
        reason="invalid_action",
    ) != len(steps):
        _fail(stage, "action_summary_alignment")
    result = {
        "assets": manifest,
        "trajectory": trajectory,
        "gripper": gripper,
        "action": action,
    }
    if depth_artifacts is not None:
        result["depth"] = depth_artifacts
    return result


def _expected_path_value(
    value: Any,
    *,
    expected: Path,
    base: Path,
    stage: str,
    reason: str,
) -> None:
    if not isinstance(value, str) or not value.strip():
        _fail(stage, reason)
    candidate = Path(value.strip()).expanduser()
    if not candidate.is_absolute():
        candidate = base / candidate
    if candidate.absolute() != expected.absolute():
        _fail(stage, reason)


def _execution_input_contract(
    settings: Mapping[str, Any],
    *,
    formal: Mapping[str, Any],
    execution_mode: str,
    sample_root: Path,
) -> dict[str, Any]:
    stage = "exec"
    if execution_mode == "action":
        action_path = (
            Path(str(settings.get("execution_action_path", "") or formal["action"]))
            .expanduser()
            .absolute()
        )
        action = _strict_json_object(
            action_path,
            stage=stage,
            reason="invalid_execution_input",
            containment_root=_containment_root_if_beneath(
                action_path,
                sample_root,
            ),
        )
        checkpoints = _bounded_list(
            action.get("checkpoints"),
            stage=stage,
            reason="invalid_execution_input",
            allow_empty=False,
        )
        steps = _bounded_list(
            action.get("steps"),
            stage=stage,
            reason="invalid_execution_input",
            allow_empty=True,
        )
        frames: list[int] = []
        for raw_checkpoint in checkpoints:
            checkpoint = _mapping(
                raw_checkpoint,
                stage=stage,
                reason="invalid_execution_input",
            )
            frames.append(
                _nonnegative_integer(
                    checkpoint.get("frame"),
                    stage=stage,
                    reason="invalid_execution_input",
                )
            )
        return {
            "checkpoint_frames": frames,
            "input_path": action_path,
            "num_action_steps": len(steps),
            "traj_key": None,
        }

    trajectory_path = (
        Path(str(settings.get("execution_trajectory_path", "") or formal["ee_traj"]))
        .expanduser()
        .absolute()
    )
    trajectory = _strict_json_object(
        trajectory_path,
        stage=stage,
        reason="invalid_execution_input",
        containment_root=_containment_root_if_beneath(
            trajectory_path,
            sample_root,
        ),
    )
    trajectory_key = str(
        settings.get("execution_traj_key", "eef_controller") or "eef_controller"
    )
    rows = _bounded_list(
        trajectory.get(trajectory_key),
        stage=stage,
        reason="invalid_execution_input",
        allow_empty=False,
    )
    raw_max_steps = settings.get("execution_max_steps", -1)
    if isinstance(raw_max_steps, bool) or not isinstance(raw_max_steps, int):
        _fail(stage, "invalid_execution_input")
    if raw_max_steps > 0:
        rows = rows[:raw_max_steps]
    frames = []
    for index, raw_row in enumerate(rows):
        row = _mapping(
            raw_row,
            stage=stage,
            reason="invalid_execution_input",
        )
        frame = row.get("frame", index)
        frames.append(
            _nonnegative_integer(
                frame,
                stage=stage,
                reason="invalid_execution_input",
            )
        )
    return {
        "checkpoint_frames": frames,
        "input_path": trajectory_path,
        "num_action_steps": None,
        "traj_key": trajectory_key,
    }


def _validate_execution_artifacts(
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    stage = "exec"
    context = _mapping(
        settings.get("context"),
        stage=stage,
        reason="invalid_context",
    )
    formal = _mapping(
        context.get("formal"),
        stage=stage,
        reason="invalid_context",
    )
    root = Path(formal["exec_dir"]).absolute()
    sample_root = Path(formal["sample_root"]).absolute()
    summary_path = Path(formal["exec_summary"]).absolute()
    checkpoint_path = Path(formal["checkpoint_trace"]).absolute()
    dense_path = Path(formal["dense_tcp_trace"]).absolute()
    execution_mode = str(settings.get("execution_mode", "") or "").strip()
    expected_executor = {
        "action": "action",
        "frame": "frame_traj",
    }.get(execution_mode)
    if expected_executor is None:
        _fail(stage, "invalid_execution_mode")
    input_contract = _execution_input_contract(
        settings,
        formal=formal,
        execution_mode=execution_mode,
        sample_root=sample_root,
    )
    expected_checkpoint_frames = list(input_contract["checkpoint_frames"])

    summary = _strict_json_object(
        summary_path,
        stage=stage,
        reason="invalid_summary",
        containment_root=sample_root,
    )
    if (
        summary.get("uid") != context.get("uid")
        or summary.get("executed") is not True
        or summary.get("executor") != expected_executor
    ):
        _fail(stage, "invalid_summary")
    _expected_path_value(
        summary.get("output_dir"),
        expected=root,
        base=root,
        stage=stage,
        reason="invalid_summary",
    )
    _expected_path_value(
        summary.get("checkpoint_trace_path"),
        expected=checkpoint_path,
        base=root,
        stage=stage,
        reason="invalid_summary",
    )

    planned = _nonnegative_integer(
        summary.get("num_checkpoints"),
        stage=stage,
        reason="invalid_summary",
    )
    evaluated = _nonnegative_integer(
        summary.get("checkpoints_evaluated"),
        stage=stage,
        reason="invalid_summary",
    )
    successes = _nonnegative_integer(
        summary.get("checkpoint_successes"),
        stage=stage,
        reason="invalid_summary",
    )
    failures = _nonnegative_integer(
        summary.get("checkpoint_failures"),
        stage=stage,
        reason="invalid_summary",
    )
    env_steps = _nonnegative_integer(
        summary.get("env_steps"),
        stage=stage,
        reason="invalid_summary",
    )
    terminated_early = summary.get("terminated_early")
    if not isinstance(terminated_early, bool):
        _fail(stage, "invalid_summary")
    if planned != len(expected_checkpoint_frames):
        _fail(stage, "summary_input_alignment")
    if execution_mode == "action":
        if (
            _nonnegative_integer(
                summary.get("num_action_steps"),
                stage=stage,
                reason="invalid_summary",
            )
            != input_contract["num_action_steps"]
        ):
            _fail(stage, "summary_input_alignment")
    else:
        if (
            _nonnegative_integer(
                summary.get("num_frames"),
                stage=stage,
                reason="invalid_summary",
            )
            != planned
            or summary.get("traj_key") != input_contract["traj_key"]
        ):
            _fail(stage, "summary_input_alignment")
    if (
        evaluated > planned
        or successes + failures != evaluated
        or (evaluated < planned and not terminated_early)
    ):
        _fail(stage, "summary_trace_alignment")

    checkpoint = _strict_json_object(
        checkpoint_path,
        stage=stage,
        reason="invalid_checkpoint_trace",
        containment_root=sample_root,
    )
    checkpoint_meta = _mapping(
        checkpoint.get("meta"),
        stage=stage,
        reason="invalid_checkpoint_trace",
    )
    if checkpoint_meta.get("source") != (
        "action_executor" if execution_mode == "action" else "frame_traj_executor"
    ) or checkpoint_meta.get("uid") != context.get("uid"):
        _fail(stage, "checkpoint_input_alignment")
    if execution_mode == "action":
        _expected_path_value(
            checkpoint_meta.get("action_path"),
            expected=input_contract["input_path"],
            base=root,
            stage=stage,
            reason="checkpoint_input_alignment",
        )
    else:
        _expected_path_value(
            checkpoint_meta.get("traj_path"),
            expected=input_contract["input_path"],
            base=root,
            stage=stage,
            reason="checkpoint_input_alignment",
        )
        if checkpoint_meta.get("traj_key") != input_contract["traj_key"]:
            _fail(stage, "checkpoint_input_alignment")
    checkpoint_rows = _bounded_list(
        checkpoint.get("checkpoints"),
        stage=stage,
        reason="invalid_checkpoint_trace",
        allow_empty=True,
    )
    if len(checkpoint_rows) != evaluated:
        _fail(stage, "summary_trace_alignment")
    trace_successes = 0
    trace_terminated = False
    for index, raw_row in enumerate(checkpoint_rows):
        row = _mapping(
            raw_row,
            stage=stage,
            reason="invalid_checkpoint_trace",
        )
        if (
            _nonnegative_integer(
                row.get("checkpoint_index"),
                stage=stage,
                reason="invalid_checkpoint_trace",
            )
            != index
            or not isinstance(row.get("success"), bool)
            or not isinstance(row.get("terminated"), bool)
        ):
            _fail(stage, "invalid_checkpoint_trace")
        trace_frame = _nonnegative_integer(
            row.get("frame"),
            stage=stage,
            reason="invalid_checkpoint_trace",
        )
        if trace_frame != expected_checkpoint_frames[index]:
            _fail(stage, "trace_input_alignment")
        if row["success"] is True:
            trace_successes += 1
        if row["terminated"] is True:
            trace_terminated = True
    trace_failures = len(checkpoint_rows) - trace_successes
    if (
        trace_successes != successes
        or trace_failures != failures
        or (trace_terminated and not terminated_early)
    ):
        _fail(stage, "summary_trace_alignment")

    dense: dict[str, Any] | None = None
    if execution_mode == "action":
        action_path = (
            Path(str(settings.get("execution_action_path", "") or formal["action"]))
            .expanduser()
            .absolute()
        )
        _expected_path_value(
            summary.get("action_path"),
            expected=action_path,
            base=root,
            stage=stage,
            reason="invalid_summary",
        )
        _expected_path_value(
            summary.get("dense_tcp_trace_path"),
            expected=dense_path,
            base=root,
            stage=stage,
            reason="invalid_summary",
        )
        dense = _strict_json_object(
            dense_path,
            stage=stage,
            reason="invalid_dense_trace",
            containment_root=sample_root,
        )
        dense_rows = _bounded_list(
            dense.get("steps"),
            stage=stage,
            reason="invalid_dense_trace",
            allow_empty=True,
        )
        if len(dense_rows) != env_steps:
            _fail(stage, "summary_dense_alignment")
        for index, raw_row in enumerate(dense_rows):
            row = _mapping(
                raw_row,
                stage=stage,
                reason="invalid_dense_trace",
            )
            if (
                _nonnegative_integer(
                    row.get("exec_step_index"),
                    stage=stage,
                    reason="invalid_dense_trace",
                )
                != index
            ):
                _fail(stage, "invalid_dense_trace")
            target_index = _nonnegative_integer(
                row.get("target_checkpoint_index"),
                stage=stage,
                reason="invalid_dense_trace",
            )
            if target_index >= planned:
                _fail(stage, "dense_input_alignment")
            target_frame = _nonnegative_integer(
                row.get("target_frame"),
                stage=stage,
                reason="invalid_dense_trace",
            )
            if target_frame != expected_checkpoint_frames[target_index]:
                _fail(stage, "dense_input_alignment")
    return {
        "summary": summary,
        "checkpoint_trace": checkpoint,
        "dense_tcp_trace": dense,
    }


def _validate_metric_table(metrics: Mapping[str, Any]) -> None:
    stage = "eval"
    rows = _bounded_list(
        metrics.get("metric_table"),
        stage=stage,
        reason="invalid_metrics",
        allow_empty=False,
    )
    names: list[str] = []
    for raw_row in rows:
        row = _mapping(
            raw_row,
            stage=stage,
            reason="invalid_metrics",
        )
        name = row.get("name")
        if not isinstance(name, str) or not name.strip():
            _fail(stage, "invalid_metrics")
        if row.get("status") not in {"computed", "not_computable"}:
            _fail(stage, "invalid_metrics")
        _finite_number_or_none(
            row.get("value"),
            stage=stage,
            reason="invalid_metrics",
        )
        if not isinstance(row.get("direction"), str):
            _fail(stage, "invalid_metrics")
        if not isinstance(row.get("unit"), str):
            _fail(stage, "invalid_metrics")
        if not isinstance(row.get("missing_inputs"), list):
            _fail(stage, "invalid_metrics")
        if not isinstance(row.get("details"), Mapping):
            _fail(stage, "invalid_metrics")
        names.append(name)
    if len(names) != len(set(names)):
        _fail(stage, "invalid_metrics")


def _csv_value(value: Any) -> str:
    return "" if value is None else str(value)


def _validate_metrics_csv(
    path: Path,
    *,
    per_frame: Sequence[Mapping[str, Any]],
    sample_root: Path,
) -> None:
    stage = "eval"
    encoded = _bounded_regular_bytes(
        path,
        stage=stage,
        reason="invalid_metrics_csv",
        limit=_MAX_CSV_BYTES,
        containment_root=sample_root,
    )
    try:
        stream = io.StringIO(encoded.decode("utf-8"), newline="")
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != tuple(EXEC_METRICS_CSV_COLUMNS):
            _fail(stage, "invalid_metrics_csv")
        rows = list(reader)
    except (csv.Error, UnicodeError):
        _fail(stage, "invalid_metrics_csv")
    if len(rows) > _MAX_SEQUENCE_ITEMS or len(rows) != len(per_frame):
        _fail(stage, "metrics_csv_alignment")
    for expected, actual in zip(per_frame, rows):
        if None in actual:
            _fail(stage, "invalid_metrics_csv")
        for column in EXEC_METRICS_CSV_COLUMNS:
            if actual.get(column, "") != _csv_value(expected.get(column)):
                _fail(stage, "metrics_csv_alignment")


def _validate_evaluation_artifacts(
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    stage = "eval"
    context = _mapping(
        settings.get("context"),
        stage=stage,
        reason="invalid_context",
    )
    formal = _mapping(
        context.get("formal"),
        stage=stage,
        reason="invalid_context",
    )
    root = Path(formal["exec_dir"]).absolute()
    sample_root = Path(formal["sample_root"]).absolute()
    execution_layout = execution_artifact_paths(root)
    metrics_path = execution_layout["exec_metrics"]
    csv_path = execution_layout["exec_metrics_per_frame"]
    metrics = _strict_json_object(
        metrics_path,
        stage=stage,
        reason="invalid_metrics",
        containment_root=sample_root,
    )
    if (
        metrics.get("format") != EXEC_METRICS_VERSION
        or metrics.get("metrics_version") != EXEC_METRICS_VERSION
        or metrics.get("metrics_scope") != EXEC_METRICS_SCOPE
        or metrics.get("uid") != context.get("uid")
        or metrics.get("run_key") != context.get("run_key")
        or str(metrics.get("gen_model", "") or "")
        != str(context.get("gen_model", "") or "")
    ):
        _fail(stage, "invalid_metrics")
    _expected_path_value(
        metrics.get("output_dir"),
        expected=root,
        base=root,
        stage=stage,
        reason="invalid_metrics",
    )
    _validate_metric_table(metrics)
    per_frame_raw = _bounded_list(
        metrics.get("per_frame"),
        stage=stage,
        reason="invalid_metrics",
        allow_empty=True,
    )
    per_frame: list[Mapping[str, Any]] = []
    for raw_row in per_frame_raw:
        per_frame.append(
            _mapping(
                raw_row,
                stage=stage,
                reason="invalid_metrics",
            )
        )
    _validate_metrics_csv(
        csv_path,
        per_frame=per_frame,
        sample_root=sample_root,
    )
    run_identity = {
        "uid": str(context["uid"]),
        "run_id": str(context["run_id"]),
        "run_key": str(context["run_key"]),
        "video_kind": str(context["video_kind"]),
        "gen_model": str(context["gen_model"]),
    }
    try:
        plan = build_evaluation_plan(
            formal_artifacts=formal,
            run_identity=run_identity,
            trajectory_path_comparison_reference_path=settings.get(
                "trajectory_path_comparison_reference_path"
            ),
            trajectory_similarity_specs=settings.get("trajectory_similarity_specs"),
            task_success_rate_specs=settings.get("task_success_rate_specs"),
            task_success_rate_options=settings.get("task_success_rate_options"),
            vlm_request_manifest_path=settings.get("vlm_request_manifest_path"),
        )
        evaluation_result = validate_evaluation_result_bundle(
            load_evaluation_result_bundle(
                formal_artifacts=formal,
            ),
            formal_artifacts=formal,
            expected_run_identity=run_identity,
            expected_plan=plan,
        )
    except EvaluationResultValidationError as error:
        _fail(
            stage,
            f"invalid_evaluation_result:{error.reason}",
        )
    return {
        "metrics": metrics,
        "evaluation_result": evaluation_result,
    }


def _validate_task_success_artifacts(
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    stage = "task_success"
    context = _mapping(
        settings.get("context"),
        stage=stage,
        reason="invalid_context",
    )
    formal = _mapping(
        context.get("formal"),
        stage=stage,
        reason="invalid_context",
    )
    sample_root = Path(formal["sample_root"]).absolute()
    raw_options = settings.get("task_success_options")
    if not isinstance(raw_options, Mapping):
        _fail(stage, "invalid_context")
    options = dict(raw_options)
    try:
        from ..stages.sim import resolve_bench_execution_request
        from ..stages.task_success import resolve_bench_task_success_request

        expected_request = resolve_bench_task_success_request(
            sample_dir=options["sample_dir"],
            run_key=options["run_key"],
            gen_model=options["gen_model"],
            task_name=str(options.get("task_name", "") or ""),
            simulator_config_path=options.get("simulator_config_path"),
            execution_config_path=options.get("execution_config_path"),
            action_path=options.get("action_path"),
            object_trajectories_path=options.get("object_trajectories_path"),
            output_path=options["output_path"],
            scene_override_path=options.get("scene_override_path"),
            execution_request_resolver=options.get(
                "execution_request_resolver",
                resolve_bench_execution_request,
            ),
        )
    except Exception:  # noqa: BLE001 - any unresolved binding is untrusted
        _fail(stage, "invalid_task_success_binding")
    if (
        expected_request.get("uid") != context.get("uid")
        or expected_request.get("run_key") != context.get("run_key")
        or str(expected_request.get("gen_model", "") or "")
        != str(context.get("gen_model", "") or "")
    ):
        _fail(stage, "invalid_task_success_binding")
    _expected_path_value(
        expected_request.get("task_success_output_path"),
        expected=Path(formal["task_success"]),
        base=Path(formal["exec_dir"]),
        stage=stage,
        reason="invalid_task_success_binding",
    )
    payload = _strict_json_object(
        Path(formal["task_success"]),
        stage=stage,
        reason="invalid_task_success",
        containment_root=sample_root,
    )
    if (
        payload.get("format") != CURRENT_TASK_SUCCESS_SCHEMA
        or payload.get("uid") != context.get("uid")
        or payload.get("run_key") != context.get("run_key")
        or str(payload.get("gen_model", "") or "")
        != str(context.get("gen_model", "") or "")
    ):
        _fail(stage, "invalid_task_success")
    success = payload.get("final_task_check_success")
    if success is not None and not isinstance(success, bool):
        _fail(stage, "invalid_task_success")
    final_meta = payload.get("final_task_meta")
    if not isinstance(final_meta, Mapping):
        _fail(stage, "invalid_task_success")
    task_name = payload.get("task_name")
    if not isinstance(task_name, str) or task_name != str(
        expected_request.get("task_name", "")
    ):
        _fail(stage, "invalid_task_success")
    frames = _bounded_list(
        payload.get("frames"),
        stage=stage,
        reason="invalid_task_success",
        allow_empty=True,
    )
    for row in frames:
        if not isinstance(row, Mapping):
            _fail(stage, "invalid_task_success")
    for field in (
        "final_strict_task_check_success",
        "final_calibrated_task_check_success",
    ):
        value = payload.get(field)
        if value is not None and not isinstance(value, bool):
            _fail(stage, "invalid_task_success")
    return {"task_success": payload}


def validate_benchmark_stage_artifacts(
    stage: str,
    *,
    settings: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate current producer artifacts without modifying any file."""

    clean_stage = str(stage or "").strip().lower()
    if clean_stage == "video":
        return {}
    if clean_stage == "video2traj":
        return _validate_video2traj_artifacts(settings)
    if clean_stage == "exec":
        return _validate_execution_artifacts(settings)
    if clean_stage == "task_success":
        return _validate_task_success_artifacts(settings)
    if clean_stage == "eval":
        return _validate_evaluation_artifacts(settings)
    raise ValueError(f"unsupported benchmark artifact stage: {stage}")
