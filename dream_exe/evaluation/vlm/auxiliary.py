"""Generic VLM evaluation over saved media and saved union trajectories.

This consumer is intentionally separate from the paper-compatible video-only
rubrics.  It reads explicit artifacts, builds a bounded deterministic
trajectory context, and runs a caller-identified prompt/parser/judge contract.
It never imports or invokes video2traj, a simulator, bench discovery, or a
provider SDK.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import stat
import tempfile
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

VIDEO_TRAJECTORY_MODE = "video_trajectory"
UNION_TRAJECTORY_INPUT_SCHEMA = "dream-exe.union-trajectory-input"
TRAJECTORY_CONTEXT_SCHEMA = "dream-exe.vlm-trajectory-context"
TRAJECTORY_CONTEXT_BUILDER_ID = "union-traj-context"
VIDEO_TRAJECTORY_PREDICTION_SCHEMA = "dream-exe.vlm-video-trajectory-prediction"
VIDEO_TRAJECTORY_CACHE_SCHEMA = "dream-exe.vlm-video-trajectory-cache"
VIDEO_TRAJECTORY_BATCH_SCHEMA = "dream-exe.saved-media-trajectory-vlm-batch"
GENERIC_JSON_OBJECT_PARSER_ID = "generic-json-object"
VIDEO_TRAJECTORY_FUNCTIONAL_SMOKE_PARSER_ID = (
    "video-trajectory-functional-smoke"
)

DEFAULT_MAX_TRAJECTORY_BYTES = 16 * 1024 * 1024
DEFAULT_MAX_MEDIA_BYTES = 64 * 1024 * 1024
DEFAULT_MAX_CONTEXT_BYTES = 4 * 1024 * 1024
DEFAULT_MAX_OBJECTS = 64
DEFAULT_MAX_STAGES = 128
DEFAULT_MAX_TRACK_POINTS = 10_000
DEFAULT_CONTEXT_POINTS_PER_TRACK = 32

_TRACK_FIELDS = (
    "eef_controller",
    "eef_tcp",
    "eef_visual_center",
    "obj_visual_center",
)
_VISIBILITY_TRACKS = {
    "eef_visual_center",
    "obj_visual_center",
}
_SUPPORTED_MEDIA_EXTENSIONS = {
    ".bmp",
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
}
_SENSITIVE_KEYS = {
    "access_token",
    "api_key",
    "api_token",
    "auth_token",
    "authorization",
    "bearer_token",
    "client_secret",
    "credential",
    "credentials",
    "default_headers",
    "extra_headers",
    "headers",
    "http_client",
    "password",
    "refresh_token",
    "secret",
    "token",
}
_DATA_URL_PATTERN = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[a-zA-Z0-9+/=_-]+")
_NAMED_SECRET_PATTERN = re.compile(
    (
        r"(?i)\b(api[ _-]?key|access[ _-]?token|refresh[ _-]?token|"
        r"authorization|password|client[ _-]?secret|credential)"
        r"(\s*[:=]\s*)([^\s,;]+)"
    )
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")

InferenceCallable = Callable[[str, Path, Mapping[str, Any]], str]
ParserCallable = Callable[[str], Mapping[str, Any]]
JudgeCallable = Callable[
    [Mapping[str, Any], Mapping[str, Any], Mapping[str, Any]],
    Mapping[str, Any],
]


def _duplicate_key_rejector(
    pairs: list[tuple[str, Any]],
) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"duplicate JSON key: {key}")
        result[key] = value
    return result


def _reject_nonfinite_constant(value: str) -> None:
    raise ValueError(f"non-finite JSON value is not allowed: {value}")


def _strict_json_loads(text: str, *, label: str) -> Any:
    try:
        return json.loads(
            text,
            object_pairs_hook=_duplicate_key_rejector,
            parse_constant=_reject_nonfinite_constant,
        )
    except (json.JSONDecodeError, UnicodeError, ValueError) as error:
        raise ValueError(f"{label} is not strict JSON: {error}") from error


def _canonical_bytes(value: Any, *, label: str) -> bytes:
    try:
        text = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be finite JSON-compatible data") from error
    return text.encode("utf-8")


def _json_copy(value: Any, *, label: str) -> Any:
    return _strict_json_loads(
        _canonical_bytes(value, label=label).decode("utf-8"),
        label=label,
    )


def _sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _read_bounded_regular_file(
    path_value: str | os.PathLike[str],
    *,
    max_bytes: int,
    label: str,
) -> tuple[Path, bytes]:
    if isinstance(max_bytes, bool) or int(max_bytes) < 1:
        raise ValueError(f"{label} max_bytes must be a positive integer")
    path = Path(path_value).expanduser()
    try:
        link_stat = path.lstat()
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} not found: {path}") from None
    if stat.S_ISLNK(link_stat.st_mode):
        raise ValueError(f"{label} must not be a symbolic link: {path}")
    if not stat.S_ISREG(link_stat.st_mode):
        raise ValueError(f"{label} must be a regular file: {path}")

    flags = os.O_RDONLY
    flags |= getattr(os, "O_CLOEXEC", 0)
    flags |= getattr(os, "O_NOFOLLOW", 0)
    flags |= getattr(os, "O_NONBLOCK", 0)
    descriptor = os.open(path, flags)
    try:
        before = os.fstat(descriptor)
        if not stat.S_ISREG(before.st_mode):
            raise ValueError(f"{label} must be a regular file: {path}")
        if link_stat.st_dev != before.st_dev or link_stat.st_ino != before.st_ino:
            raise RuntimeError(
                f"{label} changed between path validation and open: {path}"
            )
        if before.st_size > int(max_bytes):
            raise ValueError(
                f"{label} exceeds {int(max_bytes)} bytes: {before.st_size}"
            )
        chunks = []
        remaining = int(max_bytes) + 1
        while remaining:
            chunk = os.read(descriptor, min(1024 * 1024, remaining))
            if not chunk:
                break
            chunks.append(chunk)
            remaining -= len(chunk)
        payload = b"".join(chunks)
        after = os.fstat(descriptor)
    finally:
        os.close(descriptor)

    if len(payload) > int(max_bytes):
        raise ValueError(f"{label} exceeds {int(max_bytes)} bytes")
    if not payload:
        raise ValueError(f"{label} is empty: {path}")
    stable_fields = (
        before.st_dev,
        before.st_ino,
        before.st_size,
        before.st_mtime_ns,
    )
    if (
        stable_fields
        != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        )
        or len(payload) != after.st_size
    ):
        raise RuntimeError(f"{label} changed while it was being read: {path}")
    return path, payload


def _require_mapping(value: Any, *, label: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{label} must be a JSON object")
    return value


def _require_nonempty_text(value: Any, *, label: str) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{label} must be non-empty")
    return clean


def _require_bounded_count(
    value: Any,
    *,
    label: str,
    maximum: int,
) -> Mapping[str, Any]:
    mapping = _require_mapping(value, label=label)
    if not mapping:
        raise ValueError(f"{label} must not be empty")
    if len(mapping) > maximum:
        raise ValueError(f"{label} exceeds the maximum count of {maximum}")
    return mapping


def _finite_number(value: Any, *, label: str) -> int | float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} must be a finite number")
    if not math.isfinite(float(value)):
        raise ValueError(f"{label} must be a finite number")
    return value


def _finite_vector(
    value: Any,
    *,
    dimensions: int,
    label: str,
    allow_none: bool = False,
) -> list[int | float] | None:
    if value is None and allow_none:
        return None
    if (
        not isinstance(value, Sequence)
        or isinstance(value, (str, bytes, bytearray))
        or len(value) != dimensions
    ):
        raise ValueError(f"{label} must contain exactly {dimensions} numbers")
    return [
        _finite_number(item, label=f"{label}[{index}]")
        for index, item in enumerate(value)
    ]


def _validate_track(
    value: Any,
    *,
    label: str,
    require_visibility: bool,
    max_points: int,
) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{label} must be a non-empty JSON array")
    if len(value) > max_points:
        raise ValueError(f"{label} exceeds the maximum point count of {max_points}")
    frames = []
    prior_frame = -1
    for index, raw_row in enumerate(value):
        row = _require_mapping(raw_row, label=f"{label}[{index}]")
        frame = row.get("frame")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0:
            raise ValueError(f"{label}[{index}].frame must be a non-negative integer")
        if frame <= prior_frame:
            raise ValueError(f"{label} frames must be strictly increasing")
        prior_frame = frame
        frames.append(frame)
        for field_name in ("pos_world", "pos_uv", "pos_base"):
            if field_name not in row:
                raise ValueError(
                    f"{label}[{index}].{field_name} must be present or null"
                )
        _finite_vector(
            row.get("pos_world"),
            dimensions=3,
            label=f"{label}[{index}].pos_world",
            allow_none=True,
        )
        _finite_vector(
            row.get("pos_uv"),
            dimensions=2,
            label=f"{label}[{index}].pos_uv",
            allow_none=True,
        )
        _finite_vector(
            row.get("pos_base"),
            dimensions=3,
            label=f"{label}[{index}].pos_base",
            allow_none=True,
        )
        if require_visibility:
            if "vis" not in row:
                raise ValueError(f"{label}[{index}].vis must be present or null")
            if row["vis"] is not None:
                _finite_number(
                    row["vis"],
                    label=f"{label}[{index}].vis",
                )
    return tuple(frames)


def _validate_union_trajectory(
    document: Any,
    *,
    expected_uid: str | None,
    max_objects: int,
    max_stages: int,
    max_track_points: int,
) -> dict[str, str]:
    root = _require_mapping(document, label="union trajectory")
    meta = _require_mapping(root.get("meta"), label="union trajectory.meta")
    uid = _require_nonempty_text(meta.get("uid"), label="union trajectory.meta.uid")
    source = _require_nonempty_text(
        meta.get("source"),
        label="union trajectory.meta.source",
    )
    mode = _require_nonempty_text(
        meta.get("mode"),
        label="union trajectory.meta.mode",
    )
    if source != "gripper_traj_union":
        raise ValueError("union trajectory.meta.source must be 'gripper_traj_union'")
    if expected_uid is not None and uid != str(expected_uid):
        raise ValueError(
            f"union trajectory UID mismatch: expected {expected_uid!r}, got {uid!r}"
        )

    objects = _require_bounded_count(
        root.get("objects"),
        label="union trajectory.objects",
        maximum=max_objects,
    )
    stages = _require_bounded_count(
        root.get("stages"),
        label="union trajectory.stages",
        maximum=max_stages,
    )
    stage_owners: dict[str, str] = {}
    object_frames: dict[str, tuple[int, ...]] = {}
    object_runtime_keys: dict[str, str] = {}
    for object_key, raw_object in objects.items():
        object_id = str(object_key)
        item = _require_mapping(
            raw_object,
            label=f"union trajectory.objects[{object_id!r}]",
        )
        if str(item.get("object_id", "")) != object_id:
            raise ValueError(f"union trajectory object_id mismatch for {object_id!r}")
        stage_ids = item.get("stage_ids")
        if (
            not isinstance(stage_ids, list)
            or not stage_ids
            or any(not str(stage_id).strip() for stage_id in stage_ids)
        ):
            raise ValueError(
                f"union trajectory object {object_id!r} must have stage_ids"
            )
        if len({str(stage_id) for stage_id in stage_ids}) != len(stage_ids):
            raise ValueError(
                f"union trajectory object {object_id!r} has duplicate stage_ids"
            )
        runtime_object_key = item.get("runtime_object_key", "")
        if not isinstance(runtime_object_key, str):
            raise ValueError(
                f"union trajectory object {object_id!r} runtime_object_key "
                "must be a string"
            )
        object_runtime_keys[object_id] = runtime_object_key
        for stage_id_value in stage_ids:
            stage_id = str(stage_id_value)
            if stage_id in stage_owners:
                raise ValueError(
                    f"union trajectory stage {stage_id!r} has multiple owners"
                )
            stage_owners[stage_id] = object_id

        aligned_frames: tuple[int, ...] | None = None
        for track_name in _TRACK_FIELDS:
            frames = _validate_track(
                item.get(track_name),
                label=(f"union trajectory.objects[{object_id!r}].{track_name}"),
                require_visibility=track_name in _VISIBILITY_TRACKS,
                max_points=max_track_points,
            )
            if aligned_frames is None:
                aligned_frames = frames
            elif frames != aligned_frames:
                raise ValueError(
                    f"union trajectory object {object_id!r} track frames "
                    "must be aligned"
                )
        assert aligned_frames is not None
        object_frames[object_id] = aligned_frames

    if set(stage_owners) != {str(key) for key in stages}:
        raise ValueError("union trajectory stages must exactly match object stage_ids")
    for stage_key, raw_stage in stages.items():
        stage_id = str(stage_key)
        item = _require_mapping(
            raw_stage,
            label=f"union trajectory.stages[{stage_id!r}]",
        )
        if str(item.get("stage_id", "")) != stage_id:
            raise ValueError(f"union trajectory stage_id mismatch for {stage_id!r}")
        object_id = str(item.get("object_id", ""))
        if object_id != stage_owners[stage_id]:
            raise ValueError(f"union trajectory stage {stage_id!r} object_id mismatch")
        if item.get("runtime_object_key", "") != object_runtime_keys[object_id]:
            raise ValueError(
                f"union trajectory stage {stage_id!r} runtime_object_key mismatch"
            )
        for track_name in _TRACK_FIELDS:
            raw_track = item.get(track_name)
            if not isinstance(raw_track, list):
                raise ValueError(
                    f"union trajectory.stages[{stage_id!r}].{track_name} "
                    "must be an array"
                )
            stage_frames = tuple(
                row.get("frame") if isinstance(row, Mapping) else None
                for row in raw_track
            )
            if stage_frames != object_frames[object_id]:
                raise ValueError(
                    f"union trajectory stage {stage_id!r} track {track_name!r} "
                    "does not reference the object trajectory frames"
                )
            if raw_track != objects[object_id][track_name]:
                raise ValueError(
                    f"union trajectory stage {stage_id!r} track {track_name!r} "
                    "does not match the object trajectory"
                )
    return {
        "uid": uid,
        "source": source,
        "mode": mode,
    }


def load_union_trajectory(
    path: str | os.PathLike[str],
    *,
    expected_uid: str | None = None,
    max_bytes: int = DEFAULT_MAX_TRAJECTORY_BYTES,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    max_stages: int = DEFAULT_MAX_STAGES,
    max_track_points: int = DEFAULT_MAX_TRACK_POINTS,
) -> dict[str, Any]:
    """Strictly read and validate one explicit saved ``union_traj.json``."""

    source_path, payload = _read_bounded_regular_file(
        path,
        max_bytes=max_bytes,
        label="union trajectory",
    )
    try:
        text = payload.decode("utf-8")
    except UnicodeError as error:
        raise ValueError("union trajectory must be UTF-8 JSON") from error
    document = _strict_json_loads(text, label="union trajectory")
    identity = _validate_union_trajectory(
        document,
        expected_uid=expected_uid,
        max_objects=int(max_objects),
        max_stages=int(max_stages),
        max_track_points=int(max_track_points),
    )
    return {
        "format": UNION_TRAJECTORY_INPUT_SCHEMA,
        "file_name": source_path.name,
        "sha256": _sha256_bytes(payload),
        "byte_size": len(payload),
        "identity": identity,
        "document": document,
    }


def _natural_key(value: Any) -> tuple[tuple[int, Any], ...]:
    parts = re.split(r"(\d+)", str(value))
    return tuple(
        (0, int(part)) if part.isdigit() else (1, part) for part in parts if part
    )


def _sample_indices(length: int, maximum: int) -> list[int]:
    if length <= maximum:
        return list(range(length))
    if maximum == 1:
        return [0]
    denominator = maximum - 1
    indices = [(index * (length - 1)) // denominator for index in range(maximum)]
    indices[-1] = length - 1
    return indices


def _context_point(
    row: Mapping[str, Any],
    *,
    include_visibility: bool,
) -> dict[str, Any]:
    point = {
        "frame": row["frame"],
        "pos_world": (None if row["pos_world"] is None else list(row["pos_world"])),
        "pos_uv": (None if row["pos_uv"] is None else list(row["pos_uv"])),
        "pos_base": (None if row["pos_base"] is None else list(row["pos_base"])),
    }
    if include_visibility:
        point["vis"] = row["vis"]
    return point


def build_union_trajectory_context(
    loaded: Mapping[str, Any],
    *,
    max_points_per_track: int = DEFAULT_CONTEXT_POINTS_PER_TRACK,
    max_context_bytes: int = DEFAULT_MAX_CONTEXT_BYTES,
) -> dict[str, Any]:
    """Build a deterministic bounded context from a validated trajectory."""

    if loaded.get("format") != UNION_TRAJECTORY_INPUT_SCHEMA:
        raise ValueError("loaded trajectory has an unsupported format")
    if (
        isinstance(max_points_per_track, bool)
        or not 1 <= int(max_points_per_track) <= 512
    ):
        raise ValueError("max_points_per_track must be between 1 and 512")
    document = _require_mapping(
        loaded.get("document"),
        label="loaded trajectory.document",
    )
    identity = _require_mapping(
        loaded.get("identity"),
        label="loaded trajectory.identity",
    )
    objects = _require_mapping(
        document.get("objects"),
        label="loaded trajectory.document.objects",
    )
    stages = _require_mapping(
        document.get("stages"),
        label="loaded trajectory.document.stages",
    )
    ordered_stage_ids = sorted((str(key) for key in stages), key=_natural_key)
    stage_rank = {stage_id: index for index, stage_id in enumerate(ordered_stage_ids)}

    def object_order(object_id: str) -> tuple[int, tuple[tuple[int, Any], ...]]:
        stage_ids = list(objects[object_id]["stage_ids"])
        first = min(stage_rank[str(stage_id)] for stage_id in stage_ids)
        return first, _natural_key(object_id)

    context_objects = []
    for object_id in sorted(
        (str(key) for key in objects),
        key=object_order,
    ):
        item = objects[object_id]
        tracks = {}
        for track_name in _TRACK_FIELDS:
            rows = item[track_name]
            indices = _sample_indices(
                len(rows),
                int(max_points_per_track),
            )
            availability_fields = ["pos_world", "pos_uv", "pos_base"]
            if track_name in _VISIBILITY_TRACKS:
                availability_fields.append("vis")
            tracks[track_name] = {
                "original_point_count": len(rows),
                "sampled_point_count": len(indices),
                "sampled_indices": indices,
                "availability": {
                    field_name: {
                        "available_count": sum(
                            row[field_name] is not None for row in rows
                        ),
                        "null_count": sum(row[field_name] is None for row in rows),
                    }
                    for field_name in availability_fields
                },
                "points": [
                    _context_point(
                        rows[index],
                        include_visibility=(track_name in _VISIBILITY_TRACKS),
                    )
                    for index in indices
                ],
            }
        context_objects.append(
            {
                "object_id": object_id,
                "stage_ids": sorted(
                    (str(stage_id) for stage_id in item["stage_ids"]),
                    key=_natural_key,
                ),
                "runtime_object_key": item["runtime_object_key"],
                "tracks": tracks,
            }
        )

    context = {
        "format": TRAJECTORY_CONTEXT_SCHEMA,
        "builder_id": TRAJECTORY_CONTEXT_BUILDER_ID,
        "trajectory": {
            "uid": identity["uid"],
            "source": identity["source"],
            "mode": identity["mode"],
            "object_count": len(objects),
            "stage_count": len(stages),
        },
        "objects": context_objects,
        "stages": [
            {
                "stage_id": stage_id,
                "object_id": str(stages[stage_id]["object_id"]),
                "runtime_object_key": str(
                    stages[stage_id].get("runtime_object_key", "")
                ),
            }
            for stage_id in ordered_stage_ids
        ],
    }
    payload = _canonical_bytes(context, label="trajectory context")
    if len(payload) > int(max_context_bytes):
        raise ValueError(
            f"canonical trajectory context exceeds {int(max_context_bytes)} bytes"
        )
    return {
        "format": TRAJECTORY_CONTEXT_SCHEMA,
        "builder_id": TRAJECTORY_CONTEXT_BUILDER_ID,
        "sha256": _sha256_bytes(payload),
        "byte_size": len(payload),
        "context": context,
    }


def render_video_trajectory_prompt(
    *,
    prompt_template: str,
    task_metadata: Mapping[str, Any],
    trajectory_context: Mapping[str, Any],
    prompt_id: str,
) -> dict[str, Any]:
    """Render a caller-owned prompt plus canonical task and trajectory data."""

    template = _require_nonempty_text(
        prompt_template,
        label="prompt_template",
    )
    prompt_id = _require_nonempty_text(
        prompt_id,
        label="prompt_id",
    )
    if len(template.encode("utf-8")) > 512 * 1024:
        raise ValueError("prompt_template exceeds 524288 bytes")
    task = _json_copy(dict(task_metadata), label="task_metadata")
    context = _json_copy(dict(trajectory_context), label="trajectory_context")
    task_text = _canonical_bytes(task, label="task_metadata").decode("utf-8")
    context_text = _canonical_bytes(
        context,
        label="trajectory_context",
    ).decode("utf-8")
    rendered = (
        template.rstrip()
        + '\n\n<DREAM_EXE_TASK_METADATA>\n'
        + task_text
        + "\n</DREAM_EXE_TASK_METADATA>\n\n"
        + '<DREAM_EXE_TRAJECTORY_CONTEXT schema="'
        + str(context.get("format", ""))
        + '">\n'
        + context_text
        + "\n</DREAM_EXE_TRAJECTORY_CONTEXT>"
    )
    rendered_bytes = rendered.encode("utf-8")
    if len(rendered_bytes) > 5 * 1024 * 1024:
        raise ValueError("rendered VLM prompt exceeds 5242880 bytes")
    return {
        "prompt": rendered,
        "sha256": _sha256_bytes(rendered_bytes),
        "byte_size": len(rendered_bytes),
        "id": prompt_id,
    }


def generic_json_object_parser(raw_response: str) -> Mapping[str, Any]:
    """Parse a strict generic JSON object without assigning metric meaning."""

    parsed = _strict_json_loads(
        str(raw_response).strip(),
        label="VLM response",
    )
    if not isinstance(parsed, Mapping):
        raise ValueError("VLM response must be a JSON object")
    return parsed


def video_trajectory_functional_smoke_parser(
    raw_response: str,
) -> Mapping[str, Any]:
    """Validate a useful qualitative video-plus-trajectory assessment.

    This parser deliberately assigns no paper metric.  It only requires an
    inspectable assessment, bounded confidence, and non-empty evidence.
    """

    parsed = dict(generic_json_object_parser(raw_response))
    assessment = parsed.get("assessment")
    if not isinstance(assessment, str) or not assessment.strip():
        raise ValueError("video_trajectory assessment must be a non-empty string")
    confidence = parsed.get("confidence")
    if (
        isinstance(confidence, bool)
        or not isinstance(confidence, (int, float))
        or not math.isfinite(float(confidence))
        or not 0.0 <= float(confidence) <= 1.0
    ):
        raise ValueError(
            "video_trajectory confidence must be a finite number in [0, 1]"
        )
    evidence = parsed.get("evidence")
    if isinstance(evidence, str):
        normalized_evidence: str | list[str] = evidence.strip()
        if not normalized_evidence:
            raise ValueError("video_trajectory evidence string must be non-empty")
    elif isinstance(evidence, list):
        if not evidence:
            raise ValueError("video_trajectory evidence list must be non-empty")
        normalized_evidence = []
        for index, item in enumerate(evidence):
            if not isinstance(item, str) or not item.strip():
                raise ValueError(
                    "video_trajectory evidence list entries must be "
                    f"non-empty strings; invalid entry at index {index}"
                )
            normalized_evidence.append(item.strip())
    else:
        raise ValueError(
            "video_trajectory evidence must be a non-empty string or "
            "a non-empty list of non-empty strings"
        )
    parsed["assessment"] = assessment.strip()
    parsed["confidence"] = confidence
    parsed["evidence"] = normalized_evidence
    return parsed


def _sensitive_paths(
    value: Any,
    *,
    prefix: str = "",
) -> list[str]:
    paths = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            name = re.sub(
                r"[^a-z0-9]+",
                "_",
                str(key).strip().lower(),
            ).strip("_")
            path = f"{prefix}.{key}" if prefix else str(key)
            if name in _SENSITIVE_KEYS or name.endswith(
                ("_api_key", "_token", "_secret", "_password")
            ):
                paths.append(path)
            paths.extend(_sensitive_paths(item, prefix=path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            paths.extend(_sensitive_paths(item, prefix=f"{prefix}[{index}]"))
    return paths


def _safe_error_text(value: Any) -> str:
    try:
        text = str(value or "")
    except Exception:
        text = f"<{type(value).__name__}>"
    text = _DATA_URL_PATTERN.sub("data:image/[REDACTED]", text)
    text = _NAMED_SECRET_PATTERN.sub(r"\1\2[REDACTED]", text)
    text = _BEARER_PATTERN.sub("Bearer [REDACTED]", text)
    return (text.strip() or "evaluation failed")[:2048]


def _normalize_evidence_record(
    raw: Mapping[str, Any],
    *,
    input_index: int,
) -> dict[str, Any]:
    item = _require_mapping(
        raw,
        label=f"evidence_records[{input_index}]",
    )
    media_text = str(item.get("media_path", "") or "").strip()
    trajectory_text = str(item.get("trajectory_path", "") or "").strip()
    if not media_text:
        raise ValueError(f"evidence_records[{input_index}].media_path is required")
    if not trajectory_text:
        raise ValueError(f"evidence_records[{input_index}].trajectory_path is required")
    media_path = Path(media_text).expanduser()
    trajectory_path = Path(trajectory_text).expanduser()
    name = str(item.get("name", "") or media_path.name).strip()
    logical_id = str(item.get("logical_id", "") or Path(name).stem).strip()
    if not name or not logical_id:
        raise ValueError(
            f"evidence_records[{input_index}] requires name and logical_id"
        )
    task_metadata = _json_copy(
        _require_mapping(
            item.get("task_metadata"),
            label=f"evidence_records[{input_index}].task_metadata",
        ),
        label=f"evidence_records[{input_index}].task_metadata",
    )
    sensitive = _sensitive_paths(task_metadata)
    if sensitive:
        raise ValueError(
            "task_metadata must not contain credentials: " + ", ".join(sensitive)
        )
    return {
        "input_index": input_index,
        "name": name,
        "logical_id": logical_id,
        "media_path": media_path,
        "trajectory_path": trajectory_path,
        "expected_uid": (
            None if item.get("expected_uid") is None else str(item["expected_uid"])
        ),
        "task_metadata": task_metadata,
        "media_sampling": _json_copy(
            item.get("media_sampling", {}),
            label=f"evidence_records[{input_index}].media_sampling",
        ),
        "generation_options": _json_copy(
            item.get("generation_options", {}),
            label=f"evidence_records[{input_index}].generation_options",
        ),
    }


def _read_media_identity(
    path: Path,
    *,
    max_bytes: int,
) -> dict[str, Any]:
    if path.suffix.lower() not in _SUPPORTED_MEDIA_EXTENSIONS:
        raise ValueError(
            "saved VLM media must be one of: "
            + ", ".join(sorted(_SUPPORTED_MEDIA_EXTENSIONS))
        )
    source_path, payload = _read_bounded_regular_file(
        path,
        max_bytes=max_bytes,
        label="saved VLM media",
    )
    return {
        "file_name": source_path.name,
        "sha256": _sha256_bytes(payload),
        "byte_size": len(payload),
    }


def _merged_mapping(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    merged = dict(base or {})
    merged.update(dict(override or {}))
    result = _json_copy(merged, label=label)
    if not isinstance(result, dict):
        raise ValueError(f"{label} must be a mapping")
    sensitive = _sensitive_paths(result)
    if sensitive:
        raise ValueError(
            f"{label} must not contain credentials: " + ", ".join(sensitive)
        )
    return result


def _safe_stem(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", value).strip("._-")
    return (clean or "evidence")[:80]


def _prediction_path(
    root: Path,
    *,
    order: int,
    logical_id: str,
    fingerprint: str,
) -> Path:
    return root / (f"{order:06d}_{_safe_stem(logical_id)}_{fingerprint[:16]}.json")


def _next_append_only_path(path: Path, *, label: str) -> Path:
    if not path.exists():
        return path
    for index in range(1, 10_000):
        candidate = path.with_name(f"{path.stem}.{label}-{index:04d}{path.suffix}")
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate append-only prediction path")


def write_video_trajectory_prediction(
    output_path: str | os.PathLike[str],
    record: Mapping[str, Any],
) -> Path:
    """Write one raw video+trajectory prediction without overwriting evidence."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
    return path


def load_video_trajectory_prediction(
    input_path: str | os.PathLike[str],
) -> dict[str, Any]:
    """Load one offline prediction record without invoking a model."""

    payload = _strict_json_loads(
        Path(input_path).read_text(encoding="utf-8"),
        label="video+trajectory prediction",
    )
    if (
        not isinstance(payload, dict)
        or payload.get("format") != VIDEO_TRAJECTORY_PREDICTION_SCHEMA
    ):
        raise ValueError("unsupported video+trajectory prediction schema")
    return payload


def _cache_reusable(
    record: Mapping[str, Any],
    *,
    fingerprint: str,
) -> bool:
    cache = record.get("cache")
    return (
        isinstance(cache, Mapping)
        and cache.get("format") == VIDEO_TRAJECTORY_CACHE_SCHEMA
        and cache.get("fingerprint") == fingerprint
        and record.get("status") in {"ok", "parse_error"}
    )


def _evaluate_one(
    *,
    item: Mapping[str, Any],
    media_identity: Mapping[str, Any],
    trajectory: Mapping[str, Any],
    context: Mapping[str, Any],
    rendered_prompt: Mapping[str, Any],
    infer: InferenceCallable,
    parser: ParserCallable,
    judge: JudgeCallable | None,
    rubric_id: str,
    parser_id: str,
    judge_id: str,
    backend: str,
    model: str,
    inference_identity: Mapping[str, Any],
    generation_options: Mapping[str, Any],
    media_sampling: Mapping[str, Any],
    max_attempts: int,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "format": VIDEO_TRAJECTORY_PREDICTION_SCHEMA,
        "mode": VIDEO_TRAJECTORY_MODE,
        "name": item["name"],
        "logical_id": item["logical_id"],
        "status": "inference_error",
        "raw_response": None,
        "parsed_response": None,
        "judge_output": None,
        "error": None,
        "failure_counts": {
            "inference": 0,
            "parser": 0,
            "judge": 0,
        },
        "task_metadata": item["task_metadata"],
        "rendered_prompt": {
            "text": rendered_prompt["prompt"],
            "sha256": rendered_prompt["sha256"],
            "byte_size": rendered_prompt["byte_size"],
            "id": rendered_prompt["id"],
        },
        "media": {
            **dict(media_identity),
            "sampling": dict(media_sampling),
        },
        "trajectory": {
            "file_name": trajectory["file_name"],
            "sha256": trajectory["sha256"],
            "byte_size": trajectory["byte_size"],
            "format": trajectory["format"],
            **dict(trajectory["identity"]),
            "context_sha256": context["sha256"],
            "context_builder_id": context["builder_id"],
        },
        "trajectory_context": context["context"],
        "evaluation_contract": {
            "rubric_id": rubric_id,
            "parser_id": parser_id,
            "judge_id": judge_id,
            "paper_metric_compatibility": "not_claimed",
            "metric_claims": [],
        },
        "provenance": {
            "backend": backend,
            "model": model,
            "inference_identity": dict(inference_identity),
            "generation_options": dict(generation_options),
            "max_attempts": max_attempts,
            "attempts": 0,
        },
    }
    failures = []
    for attempt in range(1, max_attempts + 1):
        record["provenance"]["attempts"] = attempt
        try:
            raw_response = infer(
                str(rendered_prompt["prompt"]),
                Path(item["media_path"]),
                generation_options,
            )
            if not isinstance(raw_response, str):
                raise TypeError("inference adapter must return text")
        except Exception as error:
            record["failure_counts"]["inference"] += 1
            failures.append(
                {
                    "attempt": attempt,
                    "type": type(error).__name__,
                    "message": _safe_error_text(error),
                }
            )
            continue
        record["raw_response"] = raw_response
        break
    else:
        record["error"] = {
            "stage": "inference",
            "type": "InferenceError",
            "attempts": failures,
        }
        return record
    if failures:
        record["prior_attempt_failures"] = failures

    try:
        parsed = parser(str(record["raw_response"]))
        parsed_copy = _json_copy(
            _require_mapping(parsed, label="parser output"),
            label="parser output",
        )
    except Exception as error:
        record["failure_counts"]["parser"] = 1
        record["status"] = "parse_error"
        record["error"] = {
            "stage": "parser",
            "type": type(error).__name__,
            "message": _safe_error_text(error),
        }
        return record
    record["parsed_response"] = parsed_copy

    if judge is not None:
        try:
            judged = judge(
                parsed_copy,
                item["task_metadata"],
                context["context"],
            )
            record["judge_output"] = _json_copy(
                _require_mapping(judged, label="judge output"),
                label="judge output",
            )
        except Exception as error:
            record["failure_counts"]["judge"] = 1
            record["status"] = "judge_error"
            record["error"] = {
                "stage": "judge",
                "type": type(error).__name__,
                "message": _safe_error_text(error),
            }
            return record
    record["status"] = "ok"
    return record


def _write_json_atomic(
    output_path: Path,
    payload: Mapping[str, Any],
) -> None:
    encoded = (
        json.dumps(
            payload,
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        link_stat = output_path.lstat()
    except FileNotFoundError:
        link_stat = None
    if link_stat is not None:
        if stat.S_ISLNK(link_stat.st_mode):
            raise ValueError(
                f"JSON destination must not be a symbolic link: {output_path}"
            )
        if not stat.S_ISREG(link_stat.st_mode):
            raise ValueError(f"JSON destination must be a regular file: {output_path}")
        if link_stat.st_size == len(encoded):
            _, existing = _read_bounded_regular_file(
                output_path,
                max_bytes=len(encoded),
                label="existing JSON destination",
            )
            if existing == encoded:
                return
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output_path.name}.",
        suffix=".tmp",
        dir=output_path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, output_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def _durable_batch_report(result: Mapping[str, Any]) -> dict[str, Any]:
    """Return the invocation-independent form of a batch report.

    ``items[*].status == \"cached\"`` describes how the current invocation
    obtained an existing prediction, not the scientific status of that
    prediction.  Persisting that transient value would rewrite an otherwise
    identical report on exact resume.  The callable still returns the cache
    status to its caller; only the durable report normalizes cache hits back to
    the prediction status already preserved in ``records``.
    """

    report = _json_copy(result, label="durable trajectory VLM batch report")
    items = report.get("items")
    records = report.get("records")
    if not isinstance(items, list) or not isinstance(records, list):
        raise ValueError("trajectory VLM batch report items/records are invalid")
    if len(items) != len(records):
        raise ValueError("trajectory VLM batch report items/records are misaligned")

    status_counts: dict[str, int] = {}
    for item, record in zip(items, records, strict=True):
        if not isinstance(item, dict) or not isinstance(record, Mapping):
            raise ValueError("trajectory VLM batch report entry is invalid")
        status = item.get("status")
        if status == "cached":
            status = record.get("status")
            if not isinstance(status, str) or not status:
                raise ValueError("cached trajectory VLM prediction status is invalid")
            item["status"] = status
        if not isinstance(status, str) or not status:
            raise ValueError("trajectory VLM batch item status is invalid")
        status_counts[status] = status_counts.get(status, 0) + 1
    report["item_status_counts"] = status_counts
    return report


def run_saved_media_trajectory_vlm_batch(
    *,
    evidence_records: Sequence[Mapping[str, Any]],
    infer: InferenceCallable,
    prediction_dir: str | os.PathLike[str],
    rubric_id: str,
    prompt_template: str,
    prompt_id: str,
    parser: ParserCallable = generic_json_object_parser,
    parser_id: str = GENERIC_JSON_OBJECT_PARSER_ID,
    judge: JudgeCallable | None = None,
    judge_id: str = "none",
    backend: str,
    model: str,
    inference_identity: Mapping[str, Any] | None = None,
    generation_options: Mapping[str, Any] | None = None,
    media_sampling: Mapping[str, Any] | None = None,
    max_attempts: int = 1,
    use_cache: bool = True,
    force: bool = False,
    preserve_input_order: bool = False,
    output_json: str | os.PathLike[str] | None = None,
    max_trajectory_bytes: int = DEFAULT_MAX_TRAJECTORY_BYTES,
    max_media_bytes: int = DEFAULT_MAX_MEDIA_BYTES,
    max_objects: int = DEFAULT_MAX_OBJECTS,
    max_stages: int = DEFAULT_MAX_STAGES,
    max_track_points: int = DEFAULT_MAX_TRACK_POINTS,
    max_context_points: int = DEFAULT_CONTEXT_POINTS_PER_TRACK,
) -> dict[str, Any]:
    """Evaluate explicit saved media+trajectory pairs with generic semantics."""

    if not callable(infer):
        raise TypeError("infer must be callable")
    if not callable(parser):
        raise TypeError("parser must be callable")
    if judge is not None and not callable(judge):
        raise TypeError("judge must be callable or None")
    if isinstance(max_attempts, bool) or int(max_attempts) < 1:
        raise ValueError("max_attempts must be at least 1")
    rubric_id = _require_nonempty_text(rubric_id, label="rubric_id")
    parser_id = _require_nonempty_text(
        parser_id,
        label="parser_id",
    )
    judge_id = _require_nonempty_text(judge_id, label="judge_id")
    backend = _require_nonempty_text(backend, label="backend")
    model = _require_nonempty_text(model, label="model")
    prediction_text = str(prediction_dir or "").strip()
    if not prediction_text:
        raise ValueError("prediction_dir is required for append-only raw VLM evidence")
    prediction_root = Path(prediction_text).expanduser().resolve()
    global_options = _merged_mapping(
        generation_options,
        None,
        label="generation_options",
    )
    global_sampling = _merged_mapping(
        media_sampling,
        None,
        label="media_sampling",
    )
    normalized_inference_identity = _merged_mapping(
        inference_identity,
        None,
        label="inference_identity",
    )
    normalized = [
        _normalize_evidence_record(record, input_index=index)
        for index, record in enumerate(evidence_records)
    ]
    if not normalized:
        raise ValueError("evidence_records must not be empty")
    if not preserve_input_order:
        normalized.sort(
            key=lambda item: (
                item["logical_id"],
                item["name"],
                item["input_index"],
            )
        )

    prepared = []
    for order, item in enumerate(normalized):
        media_identity = _read_media_identity(
            item["media_path"],
            max_bytes=int(max_media_bytes),
        )
        trajectory = load_union_trajectory(
            item["trajectory_path"],
            expected_uid=item["expected_uid"],
            max_bytes=int(max_trajectory_bytes),
            max_objects=int(max_objects),
            max_stages=int(max_stages),
            max_track_points=int(max_track_points),
        )
        context = build_union_trajectory_context(
            trajectory,
            max_points_per_track=int(max_context_points),
        )
        rendered_prompt = render_video_trajectory_prompt(
            prompt_template=prompt_template,
            task_metadata=item["task_metadata"],
            trajectory_context=context["context"],
            prompt_id=prompt_id,
        )
        per_item_options = _merged_mapping(
            global_options,
            item["generation_options"],
            label="generation_options",
        )
        per_item_sampling = _merged_mapping(
            global_sampling,
            item["media_sampling"],
            label="media_sampling",
        )
        fingerprint_payload = {
            "mode": VIDEO_TRAJECTORY_MODE,
            "name": item["name"],
            "logical_id": item["logical_id"],
            "media": {
                "sha256": media_identity["sha256"],
                "byte_size": media_identity["byte_size"],
                "sampling": per_item_sampling,
            },
            "trajectory": {
                "sha256": trajectory["sha256"],
                "format": trajectory["format"],
                **dict(trajectory["identity"]),
            },
            "context": {
                "sha256": context["sha256"],
                "builder_id": context["builder_id"],
                "max_points_per_track": int(max_context_points),
            },
            "prompt": {
                "sha256": rendered_prompt["sha256"],
                "id": rendered_prompt["id"],
            },
            "rubric": {
                "id": rubric_id,
            },
            "parser_id": parser_id,
            "judge": {
                "id": judge_id,
            },
            "inference": {
                "backend": backend,
                "model": model,
                "identity": normalized_inference_identity,
                "generation_options": per_item_options,
                "max_attempts": int(max_attempts),
            },
        }
        fingerprint = _sha256_bytes(
            _canonical_bytes(
                fingerprint_payload,
                label="cache fingerprint",
            )
        )
        prepared.append(
            {
                "order": order,
                "item": item,
                "media_identity": media_identity,
                "trajectory": trajectory,
                "context": context,
                "rendered_prompt": rendered_prompt,
                "generation_options": per_item_options,
                "media_sampling": per_item_sampling,
                "fingerprint": fingerprint,
            }
        )

    prediction_root.mkdir(parents=True, exist_ok=True)
    items = []
    records = []
    for prepared_item in prepared:
        item = prepared_item["item"]
        canonical_path = _prediction_path(
            prediction_root,
            order=prepared_item["order"],
            logical_id=item["logical_id"],
            fingerprint=prepared_item["fingerprint"],
        )
        cache_issue = None
        if use_cache and not force and canonical_path.exists():
            try:
                cached = load_video_trajectory_prediction(canonical_path)
            except Exception as error:
                cache_issue = {
                    "type": type(error).__name__,
                    "message": _safe_error_text(error),
                }
            else:
                if _cache_reusable(
                    cached,
                    fingerprint=prepared_item["fingerprint"],
                ):
                    records.append(cached)
                    items.append(
                        {
                            "order": prepared_item["order"],
                            "input_index": item["input_index"],
                            "name": item["name"],
                            "logical_id": item["logical_id"],
                            "status": "cached",
                            "prediction_path": canonical_path.relative_to(
                                prediction_root
                            ).as_posix(),
                        }
                    )
                    continue
                cache_issue = {
                    "type": "CacheMismatch",
                    "message": "cached prediction is not reusable",
                }

        record = _evaluate_one(
            item=item,
            media_identity=prepared_item["media_identity"],
            trajectory=prepared_item["trajectory"],
            context=prepared_item["context"],
            rendered_prompt=prepared_item["rendered_prompt"],
            infer=infer,
            parser=parser,
            judge=judge,
            rubric_id=rubric_id,
            parser_id=parser_id,
            judge_id=judge_id,
            backend=backend,
            model=model,
            inference_identity=normalized_inference_identity,
            generation_options=prepared_item["generation_options"],
            media_sampling=prepared_item["media_sampling"],
            max_attempts=int(max_attempts),
        )
        record["cache"] = {
            "format": VIDEO_TRAJECTORY_CACHE_SCHEMA,
            "fingerprint": prepared_item["fingerprint"],
        }
        label = (
            "force" if force else "recovered" if cache_issue is not None else "rerun"
        )
        destination = _next_append_only_path(canonical_path, label=label)
        write_error = None
        try:
            write_video_trajectory_prediction(destination, record)
        except Exception as error:
            write_error = {
                "type": type(error).__name__,
                "message": _safe_error_text(error),
            }
        item_result = {
            "order": prepared_item["order"],
            "input_index": item["input_index"],
            "name": item["name"],
            "logical_id": item["logical_id"],
            "status": record["status"],
            "prediction_path": (
                None
                if write_error is not None
                else destination.relative_to(prediction_root).as_posix()
            ),
        }
        if cache_issue is not None:
            item_result["cache_issue"] = cache_issue
        if write_error is not None:
            item_result["prediction_write_error"] = write_error
        records.append(record)
        items.append(item_result)

    status_counts: dict[str, int] = {}
    for item in items:
        status = str(item["status"])
        status_counts[status] = status_counts.get(status, 0) + 1
    issue_count = sum(
        count
        for status_name, count in status_counts.items()
        if status_name not in {"ok", "cached"}
    ) + sum(
        1 for item in items if "cache_issue" in item or "prediction_write_error" in item
    )
    result: dict[str, Any] = {
        "format": VIDEO_TRAJECTORY_BATCH_SCHEMA,
        "mode": VIDEO_TRAJECTORY_MODE,
        "status": "completed_with_issues" if issue_count else "completed",
        "selected_evidence_count": len(prepared),
        "processed_evidence_count": len(items),
        "item_status_counts": status_counts,
        "items": items,
        "records": records,
        "prediction_dir": prediction_root.name,
        "cache_enabled": bool(use_cache),
        "force": bool(force),
        "evaluation_contract": {
            "rubric_id": rubric_id,
            "prompt_id": prompt_id,
            "parser_id": parser_id,
            "judge_id": judge_id,
            "paper_metric_compatibility": "not_claimed",
            "metric_claims": [],
        },
        "output_json": (
            None
            if output_json is None
            else Path(output_json).name
        ),
    }
    if output_json is not None:
        _write_json_atomic(
            Path(output_json).expanduser().resolve(),
            _durable_batch_report(result),
        )
    return result


__all__ = [
    "DEFAULT_CONTEXT_POINTS_PER_TRACK",
    "DEFAULT_MAX_CONTEXT_BYTES",
    "DEFAULT_MAX_MEDIA_BYTES",
    "DEFAULT_MAX_OBJECTS",
    "DEFAULT_MAX_STAGES",
    "DEFAULT_MAX_TRACK_POINTS",
    "DEFAULT_MAX_TRAJECTORY_BYTES",
    "GENERIC_JSON_OBJECT_PARSER_ID",
    "InferenceCallable",
    "JudgeCallable",
    "ParserCallable",
    "TRAJECTORY_CONTEXT_BUILDER_ID",
    "TRAJECTORY_CONTEXT_SCHEMA",
    "UNION_TRAJECTORY_INPUT_SCHEMA",
    "VIDEO_TRAJECTORY_BATCH_SCHEMA",
    "VIDEO_TRAJECTORY_CACHE_SCHEMA",
    "VIDEO_TRAJECTORY_FUNCTIONAL_SMOKE_PARSER_ID",
    "VIDEO_TRAJECTORY_MODE",
    "VIDEO_TRAJECTORY_PREDICTION_SCHEMA",
    "build_union_trajectory_context",
    "generic_json_object_parser",
    "load_union_trajectory",
    "load_video_trajectory_prediction",
    "render_video_trajectory_prompt",
    "run_saved_media_trajectory_vlm_batch",
    "video_trajectory_functional_smoke_parser",
    "write_video_trajectory_prediction",
]
