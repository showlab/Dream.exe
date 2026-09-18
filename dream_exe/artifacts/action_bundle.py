"""Typed action-array bundles shared by benchmark and execution code.

The canonical artifact is a directory containing ``action.npy`` and
``meta.json``.  Dense numeric values live only in the NumPy file; the JSON
document owns names, units, coordinate/time semantics, provenance, and the
sparse records required to reconstruct the maintained action payload.

Legacy JSON documents remain readable so the storage migration can be
behavior-preserving.  NumPy object arrays and pickle loading are forbidden.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
import math
import os
import re
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np


ACTION_BUNDLE_SCHEMA = "dream-exe.action"
ACTION_ARRAY_FILENAME = "action.npy"
ACTION_META_FILENAME = "meta.json"

_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GT_META_FIELDS = frozenset(
    {
        "T",
        "A",
        "action_names",
        "source",
        "representation",
        "uid",
        "task",
        "date",
        "episode_id",
        "parquet_path",
        "parquet_path_ref",
        "layout",
        "note",
    }
)


def _reject_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON number is forbidden: {value}")


def _unique_object(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key!r}")
        result[key] = value
    return result


def _load_json_object(path: str | Path) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"action JSON document not found: {source}")
    try:
        payload = json.loads(
            source.read_text(encoding="utf-8"),
            object_pairs_hook=_unique_object,
            parse_constant=_reject_constant,
        )
    except UnicodeDecodeError as error:
        raise ValueError(f"action JSON must be UTF-8: {source}") from error
    if not isinstance(payload, dict):
        raise TypeError(f"action JSON must contain an object: {source}")
    return payload


def _canonical_json_bytes(value: Any) -> bytes:
    return (
        json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        )
        + "\n"
    ).encode("utf-8")


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _fsync_directory(path: Path) -> None:
    descriptor = os.open(path, os.O_RDONLY | getattr(os, "O_DIRECTORY", 0))
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _write_bytes_atomic(
    destination: Path,
    payload: bytes,
    *,
    exclusive: bool,
) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if exclusive and (destination.exists() or destination.is_symlink()):
        raise FileExistsError(destination)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        if exclusive:
            os.link(temporary, destination)
            temporary.unlink()
        else:
            os.replace(temporary, destination)
        _fsync_directory(destination.parent)
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return destination


def _finite_number(value: Any, *, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise TypeError(f"{label} must be a finite number")
    converted = float(value)
    if not math.isfinite(converted):
        raise ValueError(f"{label} must be finite")
    return converted


def _typed_array(
    values: Any,
    *,
    dtype: str | np.dtype[Any],
    label: str,
) -> np.ndarray:
    target_dtype = np.dtype(dtype).newbyteorder("<")
    if target_dtype.kind != "f":
        raise ValueError(f"{label} dtype must be a floating-point dtype")
    array = np.ascontiguousarray(np.asarray(values, dtype=target_dtype))
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"{label} must be a non-empty rank-2 array")
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must contain only finite values")
    return array


def action_npy_bytes(array: np.ndarray) -> bytes:
    """Serialize one validated numeric action tensor without pickle."""

    typed = _typed_array(array, dtype=array.dtype, label="action array")
    stream = io.BytesIO()
    np.save(stream, typed, allow_pickle=False)
    return stream.getvalue()


def _array_record_from_bytes(
    payload: bytes,
    array: np.ndarray,
    *,
    filename: str = ACTION_ARRAY_FILENAME,
) -> dict[str, Any]:
    return {
        "path": filename,
        "sha256": _sha256_bytes(payload),
        "size": len(payload),
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "order": "C",
    }


def array_record_from_file(path: str | Path) -> tuple[dict[str, Any], np.ndarray]:
    """Validate an existing pickle-free NPY and return its bundle record."""

    source = Path(path).expanduser().resolve()
    if source.is_symlink() or not source.is_file():
        raise FileNotFoundError(f"action NumPy file not found: {source}")
    try:
        array = np.load(source, allow_pickle=False, mmap_mode="r")
    except ValueError as error:
        raise ValueError(f"invalid pickle-free action NumPy file: {source}") from error
    if array.dtype.kind != "f" or array.dtype.hasobject:
        raise ValueError(f"action NumPy dtype must be floating-point: {source}")
    if array.dtype.byteorder not in {"<", "=", "|"}:
        raise ValueError(f"action NumPy must be little-endian: {source}")
    if array.ndim != 2 or array.shape[0] <= 0 or array.shape[1] <= 0:
        raise ValueError(f"action NumPy must be a non-empty rank-2 array: {source}")
    if not array.flags.c_contiguous:
        raise ValueError(f"action NumPy must be C-contiguous: {source}")
    if not np.isfinite(array).all():
        raise ValueError(f"action NumPy contains non-finite values: {source}")
    record = {
        "path": source.name,
        "sha256": _sha256_file(source),
        "size": source.stat().st_size,
        "dtype": array.dtype.str,
        "shape": list(array.shape),
        "order": "C",
    }
    return record, array


def load_legacy_gt_action_json(
    path: str | Path,
    *,
    expected_uid: str | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Validate the maintained RoboCasa GT action JSON as float32."""

    source = Path(path).expanduser().resolve()
    document = _load_json_object(source)
    if set(document) != {"meta", "actions"}:
        raise ValueError(
            "GT action JSON must contain exactly 'meta' and 'actions': "
            f"{source}"
        )
    meta = document["meta"]
    if not isinstance(meta, Mapping):
        raise TypeError(f"GT action meta must be an object: {source}")
    missing = sorted(_GT_META_FIELDS - set(meta))
    unknown = sorted(set(meta) - _GT_META_FIELDS)
    if missing or unknown:
        raise ValueError(
            "GT action meta fields do not match the raw_12d contract: "
            f"missing={missing}, unknown={unknown}: {source}"
        )
    uid = str(meta["uid"] or "").strip()
    if not uid or not _SAFE_ID.fullmatch(uid):
        raise ValueError(f"GT action meta.uid must be a safe non-empty ID: {source}")
    if expected_uid is not None and uid != expected_uid:
        raise ValueError(
            f"GT action UID mismatch: expected {expected_uid!r}, got {uid!r}"
        )
    if meta["representation"] != "raw_12d":
        raise ValueError(
            "GT action representation must be 'raw_12d': "
            f"{source}"
        )
    if meta["source"] != "robocasa_parquet":
        raise ValueError(
            "GT action source must be 'robocasa_parquet': "
            f"{source}"
        )
    frame_count = meta["T"]
    action_dim = meta["A"]
    if isinstance(frame_count, bool) or not isinstance(frame_count, int) or frame_count <= 0:
        raise ValueError(f"GT action meta.T must be a positive integer: {source}")
    if isinstance(action_dim, bool) or not isinstance(action_dim, int) or action_dim <= 0:
        raise ValueError(f"GT action meta.A must be a positive integer: {source}")
    names = meta["action_names"]
    if (
        not isinstance(names, list)
        or len(names) != action_dim
        or any(not isinstance(name, str) or not name.strip() for name in names)
        or len(set(names)) != len(names)
    ):
        raise ValueError(
            "GT action_names must be unique non-empty names matching meta.A: "
            f"{source}"
        )
    rows = document["actions"]
    if not isinstance(rows, list) or len(rows) != frame_count:
        raise ValueError(f"GT action row count does not match meta.T: {source}")
    normalized: list[list[float]] = []
    for row_index, row in enumerate(rows):
        if not isinstance(row, list) or len(row) != action_dim:
            raise ValueError(
                "GT action row width does not match meta.A: "
                f"row={row_index}: {source}"
            )
        normalized.append(
            [
                _finite_number(
                    value,
                    label=f"GT action actions[{row_index}][{column_index}]",
                )
                for column_index, value in enumerate(row)
            ]
        )
    return document, _typed_array(
        normalized,
        dtype=np.dtype("<f4"),
        label="GT action array",
    )


def _feature_records(
    names: Sequence[str],
    units: Sequence[str | None],
) -> list[dict[str, Any]]:
    if len(names) != len(units):
        raise ValueError("action feature names and units must have equal length")
    return [
        {"index": index, "name": str(name), "unit": unit}
        for index, (name, unit) in enumerate(zip(names, units, strict=True))
    ]


def build_gt_action_meta(
    legacy_json: str | Path,
    action_npy: str | Path,
    *,
    expected_uid: str | None = None,
) -> dict[str, Any]:
    """Build canonical metadata while reusing the exact existing GT NPY."""

    document, expected = load_legacy_gt_action_json(
        legacy_json,
        expected_uid=expected_uid,
    )
    record, stored = array_record_from_file(action_npy)
    if stored.dtype != expected.dtype or stored.shape != expected.shape:
        raise ValueError("GT action JSON and NPY dtype/shape differ")
    if not np.array_equal(stored, expected):
        raise ValueError("GT action JSON and NPY values differ")
    source_meta = copy.deepcopy(document["meta"])
    meta = {
        "format": ACTION_BUNDLE_SCHEMA,
        "uid": source_meta["uid"],
        "kind": "ground_truth_demonstration",
        "representation": source_meta["representation"],
        "array": record,
        "features": _feature_records(
            source_meta["action_names"],
            [None] * int(source_meta["A"]),
        ),
        "timebase": {"kind": "source_frame", "fps": None},
        "coordinate_frame": None,
        "metadata": source_meta,
        "records": None,
    }
    return validate_action_meta(meta)


def _motion_feature_records(action_space: Mapping[str, Any]) -> list[dict[str, Any]]:
    position_unit = str(action_space.get("position_unit", "") or "") or None
    rotation_unit = str(action_space.get("rotation_unit", "") or "") or None
    return _feature_records(
        (
            "delta_position_x",
            "delta_position_y",
            "delta_position_z",
            "delta_rotation_x",
            "delta_rotation_y",
            "delta_rotation_z",
            "gripper_command",
        ),
        (
            position_unit,
            position_unit,
            position_unit,
            rotation_unit,
            rotation_unit,
            rotation_unit,
            None,
        ),
    )


def split_motion_plan_payload(
    payload: Mapping[str, Any],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Split current action into a float64 tensor and sparse metadata."""

    if not isinstance(payload, Mapping):
        raise TypeError("motion plan payload must be an object")
    if set(payload) != {"meta", "checkpoints", "steps"}:
        raise ValueError(
            "motion plan must contain exactly meta, checkpoints, and steps"
        )
    payload_meta = payload["meta"]
    checkpoints = payload["checkpoints"]
    steps = payload["steps"]
    if not isinstance(payload_meta, Mapping):
        raise TypeError("motion plan meta must be an object")
    if payload_meta.get("format") != "action":
        raise ValueError("motion plan meta.format must be action")
    uid = str(payload_meta.get("uid", "") or "").strip()
    if not _SAFE_ID.fullmatch(uid):
        raise ValueError("motion plan meta.uid must be a safe non-empty ID")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError("motion plan checkpoints must be a non-empty list")
    if not isinstance(steps, list) or not steps:
        raise ValueError("motion plan steps must be a non-empty list")
    rows: list[list[float]] = []
    sparse_steps: list[dict[str, Any]] = []
    for row_index, raw_step in enumerate(steps):
        if not isinstance(raw_step, Mapping):
            raise TypeError(f"motion plan step {row_index} must be an object")
        action = raw_step.get("action_ref_6d")
        if not isinstance(action, list) or len(action) != 6:
            raise ValueError(
                f"motion plan step {row_index} action_ref_6d must have width 6"
            )
        row = [
            _finite_number(value, label=f"motion plan step {row_index} action[{index}]")
            for index, value in enumerate(action)
        ]
        row.append(
            _finite_number(
                raw_step.get("gripper_cmd"),
                label=f"motion plan step {row_index} gripper_cmd",
            )
        )
        sparse = copy.deepcopy(dict(raw_step))
        sparse.pop("action_ref_6d", None)
        sparse.pop("gripper_cmd", None)
        if "array_row" in sparse:
            raise ValueError("motion plan source step already owns array_row")
        sparse["array_row"] = row_index
        rows.append(row)
        sparse_steps.append(sparse)
    array = _typed_array(rows, dtype=np.dtype("<f8"), label="motion plan array")
    action_space = payload_meta.get("action_space", {})
    if not isinstance(action_space, Mapping):
        raise TypeError("motion plan meta.action_space must be an object")
    planner = payload_meta.get("planner", {})
    if not isinstance(planner, Mapping):
        raise TypeError("motion plan meta.planner must be an object")
    representation = str(action_space.get("type", "") or "").strip()
    if not representation:
        raise ValueError("motion plan action_space.type must be non-empty")
    coordinate_frame = str(action_space.get("reference_frame", "") or "").strip()
    fps = planner.get("policy_hz")
    if fps is not None:
        fps = _finite_number(fps, label="motion plan planner.policy_hz")
        if fps <= 0:
            raise ValueError("motion plan planner.policy_hz must be positive")
    meta = {
        "format": ACTION_BUNDLE_SCHEMA,
        "uid": uid,
        "kind": "motion_plan",
        "representation": representation,
        "array": {},
        "features": _motion_feature_records(action_space),
        "timebase": {"kind": "execution_step", "fps": fps},
        "coordinate_frame": coordinate_frame or None,
        "metadata": copy.deepcopy(dict(payload_meta)),
        "records": {
            "checkpoints": copy.deepcopy(checkpoints),
            "steps": sparse_steps,
        },
    }
    return array, meta


def validate_action_meta(document: Mapping[str, Any]) -> dict[str, Any]:
    """Strictly validate one canonical ``meta.json`` document."""

    if not isinstance(document, Mapping):
        raise TypeError("action meta must be an object")
    required = {
        "format",
        "uid",
        "kind",
        "representation",
        "array",
        "features",
        "timebase",
        "coordinate_frame",
        "metadata",
        "records",
    }
    missing = sorted(required - set(document))
    unknown = sorted(set(document) - required)
    if missing or unknown:
        raise ValueError(
            f"action meta fields differ: missing={missing}, unknown={unknown}"
        )
    if document["format"] != ACTION_BUNDLE_SCHEMA:
        raise ValueError(f"unsupported action schema: {document['format']!r}")
    uid = document["uid"]
    if not isinstance(uid, str) or not _SAFE_ID.fullmatch(uid):
        raise ValueError("action meta uid must be a safe non-empty ID")
    kind = document["kind"]
    if kind not in {"ground_truth_demonstration", "motion_plan"}:
        raise ValueError("action meta kind is unsupported")
    if not isinstance(document["representation"], str) or not document[
        "representation"
    ].strip():
        raise ValueError("action meta representation must be non-empty")
    array = document["array"]
    if not isinstance(array, Mapping):
        raise TypeError("action meta array must be an object")
    expected_array_fields = {"path", "sha256", "size", "dtype", "shape", "order"}
    if set(array) != expected_array_fields:
        raise ValueError("action meta array fields differ from the canonical contract")
    allowed_array_filenames = (
        {"gt.npy", ACTION_ARRAY_FILENAME}
        if kind == "ground_truth_demonstration"
        else {ACTION_ARRAY_FILENAME}
    )
    if array["path"] not in allowed_array_filenames:
        raise ValueError(
            f"{kind} action meta array.path must be one of "
            f"{sorted(allowed_array_filenames)}"
        )
    if not isinstance(array["sha256"], str) or not _SHA256.fullmatch(array["sha256"]):
        raise ValueError("action meta array.sha256 must be a lowercase SHA-256")
    if isinstance(array["size"], bool) or not isinstance(array["size"], int) or array["size"] <= 0:
        raise ValueError("action meta array.size must be a positive integer")
    try:
        dtype = np.dtype(array["dtype"])
    except TypeError as error:
        raise ValueError("action meta array.dtype is invalid") from error
    if dtype.kind != "f" or dtype.hasobject or dtype.byteorder not in {"<", "=", "|"}:
        raise ValueError("action meta array.dtype must be little-endian floating point")
    shape = array["shape"]
    if (
        not isinstance(shape, list)
        or len(shape) != 2
        or any(isinstance(value, bool) or not isinstance(value, int) or value <= 0 for value in shape)
    ):
        raise ValueError("action meta array.shape must contain two positive integers")
    if array["order"] != "C":
        raise ValueError("action meta array.order must be C")
    features = document["features"]
    if not isinstance(features, list) or len(features) != shape[1]:
        raise ValueError("action meta features must match the action width")
    names: list[str] = []
    for index, feature in enumerate(features):
        if not isinstance(feature, Mapping) or set(feature) != {"index", "name", "unit"}:
            raise ValueError(f"action feature {index} fields are invalid")
        if feature["index"] != index:
            raise ValueError("action feature indices must be contiguous and ordered")
        name = feature["name"]
        if not isinstance(name, str) or not name.strip():
            raise ValueError(f"action feature {index} name must be non-empty")
        names.append(name)
        if feature["unit"] is not None and not isinstance(feature["unit"], str):
            raise TypeError(f"action feature {index} unit must be a string or null")
    if len(set(names)) != len(names):
        raise ValueError("action feature names must be unique")
    timebase = document["timebase"]
    if not isinstance(timebase, Mapping) or set(timebase) != {"kind", "fps"}:
        raise ValueError("action meta timebase fields are invalid")
    if timebase["kind"] not in {"source_frame", "execution_step"}:
        raise ValueError("action meta timebase.kind is unsupported")
    if timebase["fps"] is not None:
        fps = _finite_number(timebase["fps"], label="action meta timebase.fps")
        if fps <= 0:
            raise ValueError("action meta timebase.fps must be positive")
    if document["coordinate_frame"] is not None and (
        not isinstance(document["coordinate_frame"], str)
        or not document["coordinate_frame"].strip()
    ):
        raise ValueError("action meta coordinate_frame must be non-empty or null")
    if not isinstance(document["metadata"], Mapping):
        raise TypeError("action meta metadata must be an object")
    records = document["records"]
    if kind == "ground_truth_demonstration":
        if records is not None:
            raise ValueError("ground-truth action records must be null")
        if document["representation"] != "raw_12d":
            raise ValueError("ground-truth action representation must be raw_12d")
        if document["metadata"].get("uid") != uid:
            raise ValueError("ground-truth action metadata UID mismatch")
    else:
        if not isinstance(records, Mapping) or set(records) != {"checkpoints", "steps"}:
            raise ValueError("motion-plan action records fields are invalid")
        checkpoints = records["checkpoints"]
        steps = records["steps"]
        if not isinstance(checkpoints, list) or not checkpoints:
            raise ValueError("motion-plan checkpoints must be a non-empty list")
        if not isinstance(steps, list) or len(steps) != shape[0]:
            raise ValueError("motion-plan steps must match the action row count")
        for index, step in enumerate(steps):
            if not isinstance(step, Mapping) or step.get("array_row") != index:
                raise ValueError("motion-plan step array rows must be contiguous")
            if "action_ref_6d" in step or "gripper_cmd" in step:
                raise ValueError("motion-plan metadata must not duplicate dense action values")
        if document["metadata"].get("uid") != uid:
            raise ValueError("motion-plan action metadata UID mismatch")
    _canonical_json_bytes(document)
    return copy.deepcopy(dict(document))


def _resolve_bundle_paths(path: str | Path) -> tuple[Path, Path]:
    requested = Path(path).expanduser().resolve()
    if requested.is_dir():
        meta_path = requested / ACTION_META_FILENAME
        if meta_path.is_file():
            meta = _load_json_object(meta_path)
            declared = str(dict(meta.get("array", {}) or {}).get("path", ""))
            if declared in {ACTION_ARRAY_FILENAME, "gt.npy"}:
                return requested / declared, meta_path
        return requested / ACTION_ARRAY_FILENAME, meta_path
    if requested.suffix == ".npy":
        return requested, requested.with_name(ACTION_META_FILENAME)
    if requested.name == ACTION_META_FILENAME:
        meta = _load_json_object(requested)
        declared = str(dict(meta.get("array", {}) or {}).get("path", ""))
        if declared not in {ACTION_ARRAY_FILENAME, "gt.npy"}:
            raise ValueError(f"unsupported action array filename in {requested}: {declared!r}")
        return requested.with_name(declared), requested
    raise ValueError(
        "action bundle path must be a directory, NPY array, or meta.json: "
        f"{requested}"
    )


def load_action_bundle(
    path: str | Path,
    *,
    expected_uid: str | None = None,
    expected_kind: str | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    """Load and cross-check both files in one canonical action bundle."""

    array_path, meta_path = _resolve_bundle_paths(path)
    meta = validate_action_meta(_load_json_object(meta_path))
    if expected_uid is not None and meta["uid"] != expected_uid:
        raise ValueError(
            f"action bundle UID mismatch: expected {expected_uid!r}, got {meta['uid']!r}"
        )
    if expected_kind is not None and meta["kind"] != expected_kind:
        raise ValueError(
            f"action bundle kind mismatch: expected {expected_kind!r}, got {meta['kind']!r}"
        )
    record, array = array_record_from_file(array_path)
    if record != meta["array"]:
        raise ValueError("action.npy does not match meta.json")
    return meta, array


def motion_plan_payload_from_bundle(path: str | Path) -> dict[str, Any]:
    """Reconstruct the maintained action object for simulator consumers."""

    meta, array = load_action_bundle(path, expected_kind="motion_plan")
    records = meta["records"]
    assert isinstance(records, Mapping)
    steps: list[dict[str, Any]] = []
    for sparse in records["steps"]:
        step = copy.deepcopy(dict(sparse))
        row_index = int(step.pop("array_row"))
        row = array[row_index]
        step["action_ref_6d"] = [float(value) for value in row[:6]]
        gripper = float(row[6])
        step["gripper_cmd"] = int(gripper) if gripper.is_integer() else gripper
        steps.append(step)
    return {
        "meta": copy.deepcopy(meta["metadata"]),
        "checkpoints": copy.deepcopy(records["checkpoints"]),
        "steps": steps,
    }


def gt_action_payload_from_bundle(path: str | Path) -> dict[str, Any]:
    meta, array = load_action_bundle(
        path,
        expected_kind="ground_truth_demonstration",
    )
    return {
        "meta": copy.deepcopy(meta["metadata"]),
        "actions": [[float(value) for value in row] for row in array],
    }


def write_action_bundle(
    destination: str | Path,
    array: np.ndarray,
    metadata: Mapping[str, Any],
    *,
    exclusive: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    """Atomically write both files of a fresh canonical action bundle."""

    root = Path(destination)
    typed = _typed_array(array, dtype=array.dtype, label="action array")
    npy_payload = action_npy_bytes(typed)
    meta = copy.deepcopy(dict(metadata))
    array_filename = (
        "gt.npy"
        if meta.get("kind") == "ground_truth_demonstration"
        else ACTION_ARRAY_FILENAME
    )
    meta["array"] = _array_record_from_bytes(
        npy_payload,
        typed,
        filename=array_filename,
    )
    meta = validate_action_meta(meta)
    array_path = _write_bytes_atomic(
        root / array_filename,
        npy_payload,
        exclusive=exclusive,
    )
    try:
        meta_path = _write_bytes_atomic(
            root / ACTION_META_FILENAME,
            _canonical_json_bytes(meta),
            exclusive=exclusive,
        )
    except Exception:
        if exclusive:
            array_path.unlink(missing_ok=True)
        raise
    loaded_meta, loaded_array = load_action_bundle(root)
    if loaded_meta != meta or not np.array_equal(loaded_array, typed):
        raise RuntimeError("written action bundle failed immediate verification")
    return array_path, meta_path, meta


def write_motion_plan_bundle_from_json(
    source_json: str | Path,
    destination: str | Path,
    *,
    exclusive: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    payload = _load_json_object(source_json)
    array, meta = split_motion_plan_payload(payload)
    return write_action_bundle(
        destination,
        array,
        meta,
        exclusive=exclusive,
    )


def write_gt_action_bundle_from_json(
    source_json: str | Path,
    destination: str | Path,
    *,
    exclusive: bool = True,
) -> tuple[Path, Path, dict[str, Any]]:
    document, array = load_legacy_gt_action_json(source_json)
    source_meta = document["meta"]
    meta = {
        "format": ACTION_BUNDLE_SCHEMA,
        "uid": source_meta["uid"],
        "kind": "ground_truth_demonstration",
        "representation": source_meta["representation"],
        "array": {},
        "features": _feature_records(
            source_meta["action_names"],
            [None] * int(source_meta["A"]),
        ),
        "timebase": {"kind": "source_frame", "fps": None},
        "coordinate_frame": None,
        "metadata": copy.deepcopy(source_meta),
        "records": None,
    }
    return write_action_bundle(
        destination,
        array,
        meta,
        exclusive=exclusive,
    )


def write_action_meta(
    path: str | Path,
    document: Mapping[str, Any],
    *,
    exclusive: bool = True,
) -> Path:
    meta = validate_action_meta(document)
    return _write_bytes_atomic(
        Path(path),
        _canonical_json_bytes(meta),
        exclusive=exclusive,
    )


__all__ = [
    "ACTION_ARRAY_FILENAME",
    "ACTION_BUNDLE_SCHEMA",
    "ACTION_META_FILENAME",
    "action_npy_bytes",
    "array_record_from_file",
    "build_gt_action_meta",
    "gt_action_payload_from_bundle",
    "load_action_bundle",
    "load_legacy_gt_action_json",
    "motion_plan_payload_from_bundle",
    "split_motion_plan_payload",
    "validate_action_meta",
    "write_action_bundle",
    "write_action_meta",
    "write_gt_action_bundle_from_json",
    "write_motion_plan_bundle_from_json",
]
