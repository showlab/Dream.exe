"""Pure construction of runtime depth-calibration lineage evidence.

The target-calibration runtime can intentionally consume a depth map that is
different from the static compatibility artifact.  This module records the
arrays and sampled EEF depths that were actually consumed without choosing an
output path or changing any numeric pipeline value.
"""

from __future__ import annotations

import copy
import hashlib
import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .cache import depth_array_fingerprint
from .dynamic_calibration import (
    DYNAMIC_AFFINE_LIFT_SCHEMA,
    DYNAMIC_SHIFT_LIFT_SCHEMA,
)


DEPTH_RUNTIME_LINEAGE_SCHEMA = "dream-exe.depth-runtime-lineage"
DEPTH_RUNTIME_LINEAGE_FILENAME = "runtime_lineage.json"
EEF_CONSUMED_DEPTH_SAMPLES_FILENAME = "eef_consumed_depth_samples.npy"
_MAX_CONSUMED_SAMPLE_BYTES = 256 * 1024 * 1024
_SUMMARY_CHUNK_ELEMENTS = 1024 * 1024
MAX_DEPTH_RUNTIME_LINEAGE_NPY_BYTES = _MAX_CONSUMED_SAMPLE_BYTES + 1024 * 1024


def _numeric_array(
    value: Any,
    *,
    label: str,
    ndim: int | None = None,
    trailing_shape: tuple[int, ...] = (),
) -> np.ndarray:
    declared_shape = getattr(value, "shape", None)
    if declared_shape is not None:
        try:
            shape = tuple(int(size) for size in declared_shape)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label}.shape must contain integer dimensions") from exc
        if any(size <= 0 for size in shape):
            raise ValueError(f"{label} must not contain an empty dimension")
        if ndim is not None and len(shape) != ndim:
            raise ValueError(f"{label} must have {ndim} dimensions, got {shape}")
        if trailing_shape and shape[-len(trailing_shape) :] != trailing_shape:
            raise ValueError(f"{label} must end in {trailing_shape}, got {shape}")
    array = np.asarray(value)
    if (
        not np.issubdtype(array.dtype, np.number)
        and not np.issubdtype(array.dtype, np.bool_)
    ) or np.iscomplexobj(array):
        raise TypeError(f"{label} must contain real numeric values")
    if ndim is not None and array.ndim != ndim:
        raise ValueError(f"{label} must have {ndim} dimensions, got {array.shape}")
    if trailing_shape and array.shape[-len(trailing_shape) :] != trailing_shape:
        raise ValueError(f"{label} must end in {trailing_shape}, got {array.shape}")
    if any(int(size) <= 0 for size in array.shape):
        raise ValueError(f"{label} must not contain an empty dimension")
    return np.ascontiguousarray(array)


def _depth_array(
    value: Any,
    *,
    label: str,
    ndim: int,
) -> np.ndarray:
    array = _numeric_array(value, label=label, ndim=ndim)
    if not np.issubdtype(array.dtype, np.floating):
        raise TypeError(f"{label} must be a floating-point depth array")
    return array


def _optional_depth_array(
    value: Any,
    *,
    label: str,
    ndim: int,
) -> np.ndarray | None:
    if value is None:
        return None
    return _depth_array(value, label=label, ndim=ndim)


def _array_summary(value: Any, *, label: str) -> dict[str, Any]:
    if value is None:
        return {"status": "unavailable"}
    array = _numeric_array(value, label=label)
    flat = array.reshape(-1)
    finite_count = 0
    finite_min: float | None = None
    finite_max: float | None = None
    for start in range(0, int(flat.size), _SUMMARY_CHUNK_ELEMENTS):
        chunk = flat[start : start + _SUMMARY_CHUNK_ELEMENTS]
        finite = np.isfinite(chunk)
        count = int(np.count_nonzero(finite))
        finite_count += count
        if not count:
            continue
        chunk_min = float(np.min(chunk, where=finite, initial=np.inf))
        chunk_max = float(np.max(chunk, where=finite, initial=-np.inf))
        finite_min = chunk_min if finite_min is None else min(finite_min, chunk_min)
        finite_max = chunk_max if finite_max is None else max(finite_max, chunk_max)
    record: dict[str, Any] = {
        "status": "available",
        "shape": [int(size) for size in array.shape],
        "dtype": str(array.dtype),
        "array_fingerprint": depth_array_fingerprint(array),
        "finite_count": finite_count,
        "nonfinite_count": int(array.size - finite_count),
        "finite_min": finite_min,
        "finite_max": finite_max,
    }
    return record


def _npy_file_sha256(array: np.ndarray) -> str:
    class _HashSink:
        def __init__(self) -> None:
            self.digest = hashlib.sha256()

        def write(self, payload: bytes) -> int:
            self.digest.update(payload)
            return len(payload)

        def flush(self) -> None:
            return None

    sink = _HashSink()
    np.save(sink, array, allow_pickle=False)
    return sink.digest.hexdigest()


def _consumed_samples_array(value: Any) -> np.ndarray:
    samples = _numeric_array(
        value,
        label="eef_consumed_depth_samples",
        ndim=2,
    )
    if not np.issubdtype(samples.dtype, np.floating):
        raise TypeError("eef_consumed_depth_samples must be floating-point")
    if int(samples.nbytes) > _MAX_CONSUMED_SAMPLE_BYTES:
        raise ValueError(
            "EEF consumed-depth sample payload exceeds "
            f"{_MAX_CONSUMED_SAMPLE_BYTES} bytes"
        )
    return np.ascontiguousarray(samples)


def _json_ready(value: Any, *, label: str) -> Any:
    if isinstance(value, Mapping):
        return {
            str(key): _json_ready(item, label=f"{label}.{key}")
            for key, item in value.items()
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return [
            _json_ready(item, label=f"{label}[{index}]")
            for index, item in enumerate(value)
        ]
    if isinstance(value, np.generic):
        return _json_ready(value.item(), label=label)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"{label} must not contain NaN or infinity")
        return float(value)
    if isinstance(value, (str, int, bool)) or value is None:
        return value
    raise TypeError(
        f"{label} must contain only JSON-compatible values, got {type(value).__name__}"
    )


def _sha256_text(value: Any, *, label: str, optional: bool = False) -> str | None:
    if value is None and optional:
        return None
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        suffix = " or null" if optional else ""
        raise ValueError(f"{label} must be a lowercase SHA-256 digest{suffix}")
    return value


def _nonempty_text(value: Any, *, label: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be a non-empty string")
    return value


def _input_fingerprint_record(
    *,
    eef_tracks_uv: Any,
    eef_visibility: Any,
    dynamic_lift_stages: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    tracks = _numeric_array(
        eef_tracks_uv,
        label="eef_tracks_uv",
        ndim=3,
        trailing_shape=(2,),
    )
    visibility = _numeric_array(
        eef_visibility,
        label="eef_visibility",
        ndim=2,
    )
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(
            "EEF visibility must align with tracks: "
            f"{visibility.shape} != {tracks.shape[:2]}"
        )
    return {
        "eef_tracks_uv": depth_array_fingerprint(tracks),
        "eef_visibility": depth_array_fingerprint(visibility),
        "stage_tracking_and_masks": _stage_input_fingerprints(dynamic_lift_stages),
    }


def _stage_input_fingerprints(
    stages: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for index, raw_stage in enumerate(stages):
        if not isinstance(raw_stage, Mapping):
            raise TypeError(f"dynamic_lift_stages[{index}] must be a mapping")
        stage = dict(raw_stage)
        tracks = _numeric_array(
            stage.get("tracks_uv"),
            label=f"dynamic_lift_stages[{index}].tracks_uv",
            ndim=3,
            trailing_shape=(2,),
        )
        visibility = _numeric_array(
            stage.get("visibility"),
            label=f"dynamic_lift_stages[{index}].visibility",
            ndim=2,
        )
        mask = np.asarray(stage.get("interaction_mask"), dtype=bool)
        if mask.ndim != 2 or any(int(size) <= 0 for size in mask.shape):
            raise ValueError(
                f"dynamic_lift_stages[{index}].interaction_mask must be non-empty [H,W]"
            )
        if visibility.shape != tracks.shape[:2]:
            raise ValueError(
                "dynamic affine stage visibility must align with tracks: "
                f"{visibility.shape} != {tracks.shape[:2]}"
            )
        records.append(
            {
                "stage_id": str(stage.get("stage_id", "") or f"s{index + 1}"),
                "object_id": str(stage.get("object_id", "") or ""),
                "runtime_object_key": str(stage.get("runtime_object_key", "") or ""),
                "tracks_uv": depth_array_fingerprint(tracks),
                "visibility": depth_array_fingerprint(visibility),
                "interaction_mask": depth_array_fingerprint(mask),
            }
        )
    return records


def build_depth_runtime_lineage(
    *,
    runtime_source: str,
    canonical_publication_source: str,
    depth_model: str,
    model_provenance_fingerprint: str | None,
    parameter_fingerprint: str | None,
    raw_model_aligned: Any,
    canonical_depth: Any,
    init_reference_depth: Any,
    static_eef_depth: Any,
    runtime_eef_depth: Any,
    runtime_metadata: Mapping[str, Any],
    eef_tracks_uv: Any,
    eef_visibility: Any,
    dynamic_lift_stages: Sequence[Mapping[str, Any]],
    positions_camera: Any,
) -> tuple[dict[str, Any], np.ndarray]:
    """Build a path-free receipt and the exact EEF depth samples consumed.

    ``raw_model_aligned`` means the raw model tensor after video-frame
    alignment, which is the tensor target calibration actually receives.  It
    must not be described as a more internal, pre-alignment model output.
    """

    if not isinstance(runtime_metadata, Mapping):
        raise TypeError("runtime_metadata must be a mapping")
    runtime_source = _nonempty_text(
        runtime_source,
        label="runtime_source",
    )
    canonical_publication_source = _nonempty_text(
        canonical_publication_source,
        label="canonical_publication_source",
    )
    depth_model = _nonempty_text(depth_model, label="depth_model")
    model_provenance_fingerprint = _sha256_text(
        model_provenance_fingerprint,
        label="model_provenance_fingerprint",
        optional=True,
    )
    parameter_fingerprint = _sha256_text(
        parameter_fingerprint,
        label="parameter_fingerprint",
        optional=True,
    )
    canonical = _depth_array(
        canonical_depth,
        label="canonical_depth",
        ndim=3,
    )
    runtime = _depth_array(
        runtime_eef_depth,
        label="runtime_eef_depth",
        ndim=3,
    )
    if runtime.shape != canonical.shape:
        raise ValueError(
            "runtime EEF depth must align with canonical depth: "
            f"{runtime.shape} != {canonical.shape}"
        )
    raw = _optional_depth_array(
        raw_model_aligned,
        label="raw_model_aligned",
        ndim=3,
    )
    static = _optional_depth_array(
        static_eef_depth,
        label="static_eef_depth",
        ndim=3,
    )
    reference = _optional_depth_array(
        init_reference_depth,
        label="init_reference_depth",
        ndim=2,
    )
    for label, array in (
        ("raw_model_aligned", raw),
        ("static_eef_depth", static),
    ):
        if array is not None and array.shape != canonical.shape:
            raise ValueError(
                f"{label} must align with canonical depth: "
                f"{array.shape} != {canonical.shape}"
            )
    if reference is not None and reference.shape != canonical.shape[1:]:
        raise ValueError(
            "init_reference_depth must align with canonical depth frames: "
            f"{reference.shape} != {canonical.shape[1:]}"
        )
    positions = _numeric_array(
        positions_camera,
        label="positions_camera",
        ndim=3,
        trailing_shape=(3,),
    )
    tracks = _numeric_array(
        eef_tracks_uv,
        label="eef_tracks_uv",
        ndim=3,
        trailing_shape=(2,),
    )
    visibility = _numeric_array(
        eef_visibility,
        label="eef_visibility",
        ndim=2,
    )
    if positions.shape[:2] != tracks.shape[:2]:
        raise ValueError(
            "positions_camera must align with EEF tracks: "
            f"{positions.shape[:2]} != {tracks.shape[:2]}"
        )
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(
            "EEF visibility must align with tracks: "
            f"{visibility.shape} != {tracks.shape[:2]}"
        )
    if int(positions.shape[0]) != int(canonical.shape[0]):
        raise ValueError(
            "EEF samples must align with canonical depth frames: "
            f"{positions.shape[0]} != {canonical.shape[0]}"
        )
    samples = _consumed_samples_array(positions[..., 2])
    sample_summary = _array_summary(
        samples,
        label="eef_consumed_depth_samples",
    )
    sample_summary.update(
        {
            "path": EEF_CONSUMED_DEPTH_SAMPLES_FILENAME,
            "file_sha256": _npy_file_sha256(samples),
            "valid_count": int(sample_summary["finite_count"]),
            "invalid_count": int(sample_summary["nonfinite_count"]),
            "source": "positions_camera[...,2]",
        }
    )

    metadata = copy.deepcopy(dict(runtime_metadata))
    dynamic_lift = metadata.get("dynamic_lift", None)
    if dynamic_lift is None:
        dynamic_lift = {
            "enabled": False,
            "mode": str(
                dict(metadata.get("eef_traj", {}) or {}).get(
                    "mode",
                    "static_eef",
                )
                or "static_eef"
            ),
            "applied": False,
            "reason": "not_dynamic_lift",
            "stages": [],
        }
    receipt = {
        "schema": DEPTH_RUNTIME_LINEAGE_SCHEMA,
        "depth": {
            "runtime_source": runtime_source,
            "canonical_publication_source": canonical_publication_source,
            "model": depth_model,
            "model_provenance_fingerprint": model_provenance_fingerprint,
            "parameter_fingerprint": parameter_fingerprint,
        },
        "arrays": {
            "raw_model_aligned": _array_summary(
                raw,
                label="raw_model_aligned",
            ),
            "canonical_depth": _array_summary(
                canonical,
                label="canonical_depth",
            ),
            "init_reference_depth": _array_summary(
                reference,
                label="init_reference_depth",
            ),
            "static_eef_depth": _array_summary(
                static,
                label="static_eef_depth",
            ),
            "runtime_eef_depth": _array_summary(
                runtime,
                label="runtime_eef_depth",
            ),
        },
        "dynamic_lift": _json_ready(
            dynamic_lift,
            label="runtime_metadata.dynamic_lift",
        ),
        "input_fingerprints": {
            **_input_fingerprint_record(
                eef_tracks_uv=tracks,
                eef_visibility=visibility,
                dynamic_lift_stages=dynamic_lift_stages,
            )
        },
        "consumed_samples": sample_summary,
    }
    return _json_ready(receipt, label="depth_runtime_lineage"), samples


def validate_depth_runtime_lineage(
    receipt: Mapping[str, Any],
    samples: Any,
    *,
    array_bindings: Mapping[str, Any] | None = None,
    input_bindings: Mapping[str, Any] | None = None,
) -> np.ndarray:
    """Validate the receipt-to-NPY binding before transactional staging."""

    if not isinstance(receipt, Mapping):
        raise TypeError("depth runtime lineage receipt must be a mapping")
    payload = dict(receipt)
    expected_top_level = {
        "schema",
        "depth",
        "arrays",
        "dynamic_lift",
        "input_fingerprints",
        "consumed_samples",
    }
    if set(payload) != expected_top_level:
        raise ValueError(
            "depth runtime lineage receipt has unexpected top-level fields"
        )
    if payload.get("schema") != DEPTH_RUNTIME_LINEAGE_SCHEMA:
        raise ValueError("depth runtime lineage receipt has an unsupported schema")
    depth = payload.get("depth", None)
    if not isinstance(depth, Mapping) or set(depth) != {
        "runtime_source",
        "canonical_publication_source",
        "model",
        "model_provenance_fingerprint",
        "parameter_fingerprint",
    }:
        raise ValueError("depth runtime lineage depth identity is invalid")
    for key in (
        "runtime_source",
        "canonical_publication_source",
        "model",
    ):
        _nonempty_text(depth.get(key), label=f"depth.{key}")
    for key in (
        "model_provenance_fingerprint",
        "parameter_fingerprint",
    ):
        _sha256_text(
            depth.get(key),
            label=f"depth.{key}",
            optional=True,
        )
    arrays = payload.get("arrays", None)
    expected_arrays = {
        "raw_model_aligned",
        "canonical_depth",
        "init_reference_depth",
        "static_eef_depth",
        "runtime_eef_depth",
    }
    if not isinstance(arrays, Mapping) or set(arrays) != expected_arrays:
        raise ValueError("depth runtime lineage array summaries are invalid")

    def validate_array_summary(
        name: str,
        *,
        required: bool,
        ndim: int,
    ) -> dict[str, Any] | None:
        raw = arrays[name]
        if not isinstance(raw, Mapping):
            raise TypeError(f"depth runtime lineage arrays.{name} must be a mapping")
        summary = dict(raw)
        status = summary.get("status")
        if status == "unavailable":
            if required or set(summary) != {"status"}:
                raise ValueError(
                    f"depth runtime lineage arrays.{name} is unexpectedly unavailable"
                )
            return None
        expected_fields = {
            "status",
            "shape",
            "dtype",
            "array_fingerprint",
            "finite_count",
            "nonfinite_count",
            "finite_min",
            "finite_max",
        }
        if status != "available" or set(summary) != expected_fields:
            raise ValueError(
                f"depth runtime lineage arrays.{name} has an invalid summary"
            )
        shape = summary.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != ndim
            or any(type(size) is not int or size <= 0 for size in shape)
        ):
            raise ValueError(f"depth runtime lineage arrays.{name}.shape is invalid")
        dtype_text = summary.get("dtype")
        if not isinstance(dtype_text, str):
            raise TypeError(
                f"depth runtime lineage arrays.{name}.dtype must be a string"
            )
        try:
            dtype = np.dtype(dtype_text)
        except TypeError as exc:
            raise ValueError(
                f"depth runtime lineage arrays.{name}.dtype is invalid"
            ) from exc
        if not np.issubdtype(dtype, np.floating):
            raise ValueError(
                f"depth runtime lineage arrays.{name} must describe "
                "floating-point depth"
            )
        finite_count = summary.get("finite_count")
        nonfinite_count = summary.get("nonfinite_count")
        if (
            type(finite_count) is not int
            or type(nonfinite_count) is not int
            or finite_count < 0
            or nonfinite_count < 0
            or finite_count + nonfinite_count != math.prod(shape)
        ):
            raise ValueError(f"depth runtime lineage arrays.{name} counts are invalid")
        fingerprint = summary.get("array_fingerprint")
        if (
            not isinstance(fingerprint, str)
            or len(fingerprint) != 64
            or any(character not in "0123456789abcdef" for character in fingerprint)
        ):
            raise ValueError(
                f"depth runtime lineage arrays.{name} fingerprint is invalid"
            )
        finite_min = summary.get("finite_min")
        finite_max = summary.get("finite_max")
        if finite_count == 0:
            if finite_min is not None or finite_max is not None:
                raise ValueError(
                    f"depth runtime lineage arrays.{name} finite range is invalid"
                )
        else:
            for key, value in (
                ("finite_min", finite_min),
                ("finite_max", finite_max),
            ):
                if (
                    isinstance(value, bool)
                    or not isinstance(value, (int, float))
                    or not math.isfinite(float(value))
                ):
                    raise ValueError(
                        f"depth runtime lineage arrays.{name}.{key} is invalid"
                    )
            if float(finite_min) > float(finite_max):
                raise ValueError(
                    f"depth runtime lineage arrays.{name} finite range is invalid"
                )
        return summary

    canonical_summary = validate_array_summary(
        "canonical_depth",
        required=True,
        ndim=3,
    )
    runtime_summary = validate_array_summary(
        "runtime_eef_depth",
        required=True,
        ndim=3,
    )
    if canonical_summary is None or runtime_summary is None:
        raise ValueError("canonical and runtime EEF depth must be available")
    if (
        runtime_summary["shape"] != canonical_summary["shape"]
        or runtime_summary["dtype"] != canonical_summary["dtype"]
    ):
        raise ValueError("runtime EEF depth summary must align with canonical depth")
    dynamic = payload.get("dynamic_lift", None)
    if not isinstance(dynamic, Mapping):
        raise TypeError("depth runtime lineage dynamic_lift must be a mapping")
    dynamic_record = dict(dynamic)
    dynamic_enabled = dynamic_record.get("enabled")
    if not isinstance(dynamic_enabled, bool):
        raise TypeError("depth runtime lineage dynamic_lift.enabled must be a bool")
    dynamic_stage_ids: set[str] = set()
    if dynamic_enabled:
        required_dynamic_fields = {
            "enabled",
            "mode",
            "policy",
            "stages",
            "applied",
            "reason",
        }
        if (
            not required_dynamic_fields.issubset(dynamic_record)
            or dynamic_record.get("mode") not in {"dynamic_shift", "dynamic_affine"}
            or not isinstance(dynamic_record.get("applied"), bool)
            or not isinstance(dynamic_record.get("stages"), list)
            or not isinstance(dynamic_record.get("policy"), Mapping)
        ):
            raise ValueError(
                "depth runtime lineage dynamic-lift metadata is incomplete"
            )
        dynamic_mode = str(dynamic_record["mode"])
        if dynamic_mode == "dynamic_affine":
            _nonempty_text(
                dynamic_record.get("calibration_solver"),
                label="dynamic_lift.calibration_solver",
            )
        if not isinstance(dynamic_record.get("reason"), str):
            raise TypeError("dynamic_lift.reason must be text")
        dynamic_applied = bool(dynamic_record["applied"])
        expected_dynamic_schema = (
            DYNAMIC_SHIFT_LIFT_SCHEMA
            if dynamic_mode == "dynamic_shift"
            else DYNAMIC_AFFINE_LIFT_SCHEMA
        )
        if dynamic_applied and dynamic_record.get("schema") != (
            expected_dynamic_schema
        ):
            raise ValueError(
                "applied depth runtime lineage has the wrong dynamic-lift schema"
            )
        if (
            not dynamic_applied
            and dynamic_record.get("schema", expected_dynamic_schema)
            != expected_dynamic_schema
        ):
            raise ValueError("depth runtime lineage dynamic-lift schema is invalid")
        applied_stage_count = 0
        for index, raw_stage in enumerate(dynamic_record["stages"]):
            if not isinstance(raw_stage, Mapping):
                raise TypeError(f"dynamic_lift.stages[{index}] must be a mapping")
            stage = dict(raw_stage)
            required_stage_fields = {
                "stage_id",
                "object_id",
                "runtime_object_key",
                "stage_start",
                "stage_end",
                "switch_strategy",
                "t_switch",
                "valid_region_pixels",
                "ramp_frames",
                "fallback",
                "fallback_reason",
            }
            if not required_stage_fields.issubset(stage):
                raise ValueError(f"dynamic_lift.stages[{index}] metadata is incomplete")
            stage_id = _nonempty_text(
                stage.get("stage_id"),
                label=f"dynamic_lift.stages[{index}].stage_id",
            )
            if stage_id in dynamic_stage_ids:
                raise ValueError("dynamic_lift stage ids must be unique")
            dynamic_stage_ids.add(stage_id)
            if not isinstance(stage.get("object_id"), str) or not isinstance(
                stage.get("runtime_object_key"), str
            ):
                raise TypeError("dynamic_lift stage identity must be text")
            if not (
                type(stage.get("stage_start")) is int
                and type(stage.get("stage_end")) is int
                and 0 <= stage["stage_start"] <= stage["stage_end"]
                and type(stage.get("valid_region_pixels")) is int
                and stage["valid_region_pixels"] >= 0
                and type(stage.get("ramp_frames")) is int
                and stage["ramp_frames"] > 0
                and isinstance(stage.get("fallback"), bool)
                and isinstance(stage.get("fallback_reason"), str)
                and stage.get("switch_strategy") == "proximity_2d"
            ):
                raise ValueError(f"dynamic_lift.stages[{index}] bounds are invalid")
            if not stage["fallback"]:
                applied_stage_count += 1
                t_switch = stage.get("t_switch")
                if (
                    type(t_switch) is not int
                    or not stage["stage_start"] <= t_switch <= stage["stage_end"]
                ):
                    raise ValueError(
                        f"dynamic_lift.stages[{index}].t_switch is invalid"
                    )
                value_fields = (
                    ("delta_ir",)
                    if dynamic_mode == "dynamic_shift"
                    else ("affine_s", "affine_b")
                )
                for key in value_fields:
                    value = stage.get(key)
                    if (
                        isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(float(value))
                    ):
                        raise ValueError(
                            f"dynamic_lift.stages[{index}].{key} is invalid"
                        )
                if dynamic_mode == "dynamic_affine":
                    _nonempty_text(
                        stage.get("calibration_solver"),
                        label=(f"dynamic_lift.stages[{index}].calibration_solver"),
                    )
            elif (
                stage.get("t_switch") is not None
                and type(stage.get("t_switch")) is not int
            ):
                raise TypeError(f"dynamic_lift.stages[{index}].t_switch is invalid")
        if dynamic_applied != bool(applied_stage_count):
            raise ValueError("dynamic_lift.applied does not match its stage metadata")
        raw_summary = validate_array_summary(
            "raw_model_aligned",
            required=(dynamic_applied and dynamic_mode == "dynamic_affine"),
            ndim=3,
        )
        validate_array_summary(
            "init_reference_depth",
            required=dynamic_applied,
            ndim=2,
        )
        static_summary = validate_array_summary(
            "static_eef_depth",
            required=dynamic_applied,
            ndim=3,
        )
    else:
        if (
            dynamic_record.get("applied") is not False
            or not isinstance(dynamic_record.get("mode"), str)
            or not isinstance(dynamic_record.get("reason"), str)
            or dynamic_record.get("stages") != []
        ):
            raise ValueError("disabled depth runtime lineage metadata is incomplete")
        raw_summary = validate_array_summary(
            "raw_model_aligned",
            required=False,
            ndim=3,
        )
        validate_array_summary(
            "init_reference_depth",
            required=False,
            ndim=2,
        )
        static_summary = validate_array_summary(
            "static_eef_depth",
            required=False,
            ndim=3,
        )
    init_summary = arrays["init_reference_depth"]
    if raw_summary is not None and (
        raw_summary["shape"] != canonical_summary["shape"]
        or raw_summary["dtype"] != canonical_summary["dtype"]
    ):
        raise ValueError("raw aligned depth summary must align with canonical depth")
    if static_summary is not None and (
        static_summary["shape"] != canonical_summary["shape"]
        or static_summary["dtype"] != canonical_summary["dtype"]
    ):
        raise ValueError("static EEF depth summary must align with canonical depth")
    if (
        isinstance(init_summary, Mapping)
        and init_summary.get("status") == "available"
        and (
            init_summary.get("shape") != canonical_summary["shape"][1:]
            or init_summary.get("dtype") != canonical_summary["dtype"]
        )
    ):
        raise ValueError(
            "init reference depth summary must align with canonical frames"
        )

    input_fingerprints = payload.get("input_fingerprints", None)
    if not isinstance(input_fingerprints, Mapping) or set(input_fingerprints) != {
        "eef_tracks_uv",
        "eef_visibility",
        "stage_tracking_and_masks",
    }:
        raise ValueError("depth runtime lineage input fingerprints are invalid")
    _sha256_text(
        input_fingerprints.get("eef_tracks_uv"),
        label="input_fingerprints.eef_tracks_uv",
    )
    _sha256_text(
        input_fingerprints.get("eef_visibility"),
        label="input_fingerprints.eef_visibility",
    )
    stage_fingerprints = input_fingerprints.get("stage_tracking_and_masks")
    if not isinstance(stage_fingerprints, list):
        raise TypeError("input_fingerprints.stage_tracking_and_masks must be a list")
    stage_ids: set[str] = set()
    for index, raw_stage in enumerate(stage_fingerprints):
        if not isinstance(raw_stage, Mapping) or set(raw_stage) != {
            "stage_id",
            "object_id",
            "runtime_object_key",
            "tracks_uv",
            "visibility",
            "interaction_mask",
        }:
            raise ValueError("depth runtime lineage stage input fingerprint is invalid")
        stage_id = _nonempty_text(
            raw_stage.get("stage_id"),
            label=f"input_fingerprints.stage_tracking_and_masks[{index}].stage_id",
        )
        if stage_id in stage_ids:
            raise ValueError("depth runtime lineage stage ids must be unique")
        stage_ids.add(stage_id)
        for key in ("object_id", "runtime_object_key"):
            if not isinstance(raw_stage.get(key), str):
                raise TypeError(
                    "depth runtime lineage stage input identity must be text"
                )
        for key in ("tracks_uv", "visibility", "interaction_mask"):
            _sha256_text(
                raw_stage.get(key),
                label=(f"input_fingerprints.stage_tracking_and_masks[{index}].{key}"),
            )
    if dynamic_enabled and dynamic_stage_ids != stage_ids:
        raise ValueError(
            "dynamic-lift metadata and input fingerprint stages do not align"
        )

    if array_bindings is not None:
        if not isinstance(array_bindings, Mapping) or set(array_bindings) != (
            expected_arrays
        ):
            raise ValueError("depth runtime lineage array bindings are incomplete")
        for name in sorted(expected_arrays):
            expected_summary = _array_summary(
                array_bindings[name],
                label=f"array_bindings.{name}",
            )
            if dict(arrays[name]) != expected_summary:
                raise ValueError(
                    f"depth runtime lineage arrays.{name} binding mismatch"
                )
    if input_bindings is not None:
        if not isinstance(input_bindings, Mapping) or set(input_bindings) != {
            "eef_tracks_uv",
            "eef_visibility",
            "dynamic_lift_stages",
        }:
            raise ValueError("depth runtime lineage input bindings are incomplete")
        expected_inputs = _input_fingerprint_record(
            eef_tracks_uv=input_bindings["eef_tracks_uv"],
            eef_visibility=input_bindings["eef_visibility"],
            dynamic_lift_stages=input_bindings["dynamic_lift_stages"],
        )
        if dict(input_fingerprints) != expected_inputs:
            raise ValueError("depth runtime lineage input fingerprint binding mismatch")
    consumed = payload.get("consumed_samples", None)
    if not isinstance(consumed, Mapping):
        raise TypeError("depth runtime lineage consumed_samples must be a mapping")
    record = dict(consumed)
    array = _consumed_samples_array(samples)
    expected = _array_summary(array, label="eef_consumed_depth_samples")
    expected.update(
        {
            "path": EEF_CONSUMED_DEPTH_SAMPLES_FILENAME,
            "file_sha256": _npy_file_sha256(array),
            "valid_count": int(expected["finite_count"]),
            "invalid_count": int(expected["nonfinite_count"]),
            "source": "positions_camera[...,2]",
        }
    )
    if record != expected:
        raise ValueError("depth runtime lineage consumed sample binding mismatch")
    _json_ready(payload, label="depth_runtime_lineage")
    return array


__all__ = [
    "DEPTH_RUNTIME_LINEAGE_FILENAME",
    "DEPTH_RUNTIME_LINEAGE_SCHEMA",
    "EEF_CONSUMED_DEPTH_SAMPLES_FILENAME",
    "MAX_DEPTH_RUNTIME_LINEAGE_NPY_BYTES",
    "build_depth_runtime_lineage",
    "validate_depth_runtime_lineage",
]
