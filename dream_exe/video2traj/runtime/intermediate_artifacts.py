"""Explicit current-compatible tracking and point-geometry publication.

The core algorithm keeps intermediate arrays in memory. These helpers are the narrow
serialization boundary for current ``tracking/*.{npz,mp4}`` and
``geometry/points_cloud_traj_*.{json,npz}`` artifacts.  They never discover a
benchmark, simulator, run key, or output root.
"""

from __future__ import annotations

import copy
import json
import tempfile
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dream_exe.artifacts.io import (
    ensure_traj_asset_dirs,
    update_traj_assets_manifest,
)
from ..depth.cache import publish_artifact_batch


def _explicit_root(output_dir: str | Path) -> Path:
    output_text = str(output_dir or "").strip()
    if not output_text:
        raise ValueError("output_dir must be an explicit non-empty path")
    return Path(output_text).expanduser().resolve()


def _stream_rows(
    object_stream_plan: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    rows = []
    seen: set[str] = set()
    seen_safe: dict[str, str] = {}
    for index, raw_stream in enumerate(list(object_stream_plan or [])):
        if not isinstance(raw_stream, Mapping):
            raise TypeError("object_stream_plan must contain only mappings")
        stream = copy.deepcopy(dict(raw_stream))
        object_id = str(stream.get("object_id", "") or "").strip()
        if not object_id:
            raise ValueError("every object stream requires a non-empty object_id")
        if object_id in seen:
            raise ValueError(
                f"object_stream_plan contains duplicate object_id={object_id!r}"
            )
        seen.add(object_id)
        safe_object_id = str(
            stream.get(
                "safe_object_id",
                object_id,
            )
            or object_id
        ).strip()
        if (
            not safe_object_id
            or safe_object_id in {".", ".."}
            or "/" in safe_object_id
            or "\\" in safe_object_id
        ):
            raise ValueError(
                f"unsafe object artifact id for object_id={object_id!r}: "
                f"{safe_object_id!r}"
            )
        previous_object_id = seen_safe.get(safe_object_id)
        if previous_object_id is not None:
            raise ValueError(
                "object_stream_plan contains colliding safe_object_id="
                f"{safe_object_id!r} for object_id={previous_object_id!r} "
                f"and object_id={object_id!r}"
            )
        seen_safe[safe_object_id] = object_id
        rows.append(
            {
                "object_id": object_id,
                "safe_object_id": safe_object_id,
                "stage_ids": [
                    str(stage_id)
                    for stage_id in list(stream.get("stage_ids", []) or [])
                ],
                "ordinal": int(index),
            }
        )
    return rows


def build_tracking_geometry_artifact_plan(
    output_dir: str | Path,
    *,
    object_stream_plan: Sequence[Mapping[str, Any]],
    include_visual_media: bool = False,
) -> dict[str, Any]:
    """Plan canonical intermediate-array paths without creating directories."""

    root = _explicit_root(output_dir)
    tracking_root = root / "tracking"
    geometry_root = root / "geometry"
    objects: dict[str, dict[str, Any]] = {}
    for row in _stream_rows(object_stream_plan):
        object_id = row["object_id"]
        safe_object_id = row["safe_object_id"]
        interaction_prefix = geometry_root / f"interaction_gripper_{safe_object_id}"
        interaction = {
            "eef_points_json": (
                interaction_prefix.with_name(f"{interaction_prefix.name}_eef.json")
            ).as_posix(),
            "eef_points_flow_npz": (
                interaction_prefix.with_name(f"{interaction_prefix.name}_eef_flow.npz")
            ).as_posix(),
            "obj_points_json": (
                interaction_prefix.with_name(f"{interaction_prefix.name}_obj.json")
            ).as_posix(),
            "obj_points_flow_npz": (
                interaction_prefix.with_name(f"{interaction_prefix.name}_obj_flow.npz")
            ).as_posix(),
        }
        objects[object_id] = {
            **row,
            "tracking_npz": (tracking_root / f"{object_id}_tracking.npz").as_posix(),
            "tracking_mp4": (
                (tracking_root / f"{object_id}_points_cloud.mp4").as_posix()
                if bool(include_visual_media)
                else None
            ),
            "points_json": (
                geometry_root / f"points_cloud_traj_{safe_object_id}.json"
            ).as_posix(),
            "points_flow_npz": (
                geometry_root / f"points_cloud_traj_{safe_object_id}_flow.npz"
            ).as_posix(),
            "interaction": interaction,
        }
    return {
        "output_dir": root.as_posix(),
        "include_visual_media": bool(include_visual_media),
        "eef": {
            "tracking_npz": (tracking_root / "eef_tracking.npz").as_posix(),
            "tracking_mp4": (
                (tracking_root / "eef_points_cloud.mp4").as_posix()
                if bool(include_visual_media)
                else None
            ),
            "points_json": (geometry_root / "points_cloud_traj_eef.json").as_posix(),
            "points_flow_npz": (
                geometry_root / "points_cloud_traj_eef_flow.npz"
            ).as_posix(),
        },
        "objects": objects,
        "object_references": {
            object_id: {
                "tracking_npz": row["tracking_npz"],
                "tracking_mp4": row["tracking_mp4"],
                "obj_points_traj_path": row["points_json"],
                "points_flow_npz": row["points_flow_npz"],
                **copy.deepcopy(dict(row["interaction"])),
            }
            for object_id, row in objects.items()
        },
    }


def _validated_plan(
    output_dir: str | Path,
    plan: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(plan, Mapping):
        raise TypeError("diagnostic artifact plan must be a mapping")
    root = _explicit_root(output_dir)
    payload = copy.deepcopy(dict(plan))
    planned_root = Path(str(payload.get("output_dir", "") or "")).expanduser().resolve()
    if planned_root != root:
        raise ValueError(
            "diagnostic artifact plan/output_dir mismatch: "
            f"plan={planned_root} output={root}"
        )
    for record in [
        dict(payload.get("eef", {}) or {}),
        *[
            dict(value or {})
            for value in dict(payload.get("objects", {}) or {}).values()
        ],
    ]:
        for key in (
            "tracking_npz",
            "tracking_mp4",
            "points_json",
            "points_flow_npz",
        ):
            path_text = str(record.get(key, "") or "")
            if key == "tracking_mp4" and not path_text:
                continue
            if not path_text:
                raise ValueError(f"diagnostic artifact plan is missing {key}")
            path = Path(path_text).expanduser().resolve()
            if root != path and root not in path.parents:
                raise ValueError(f"diagnostic artifact path escapes output_dir: {path}")
    for object_id, raw_record in dict(payload.get("objects", {}) or {}).items():
        interaction = dict(dict(raw_record or {}).get("interaction", {}) or {})
        for key in (
            "eef_points_json",
            "eef_points_flow_npz",
            "obj_points_json",
            "obj_points_flow_npz",
        ):
            path_text = str(interaction.get(key, "") or "")
            if not path_text:
                raise ValueError(
                    "diagnostic artifact plan is missing "
                    f"objects[{object_id!r}].interaction.{key}"
                )
            path = Path(path_text).expanduser().resolve()
            if root != path and root not in path.parents:
                raise ValueError(
                    f"diagnostic interaction artifact path escapes output_dir: {path}"
                )
    return root, payload


def _tracking_arrays(
    payload: Mapping[str, Any],
    *,
    label: str,
) -> tuple[np.ndarray, np.ndarray]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} tracking payload must be a mapping")
    tracks = np.asarray(payload.get("tracks_uv", None), dtype=np.float32)
    visibility = np.asarray(
        payload.get("visibility", None),
        dtype=np.float32,
    )
    if tracks.ndim != 3 or tracks.shape[-1] != 2:
        raise ValueError(f"{label} tracks_uv must be [T,N,2], got {tracks.shape}")
    if visibility.shape != tracks.shape[:2]:
        raise ValueError(
            f"{label} visibility mismatch: "
            f"tracks={tracks.shape}, visibility={visibility.shape}"
        )
    return tracks, visibility


def write_tracking_artifacts(
    output_dir: str | Path,
    *,
    plan: Mapping[str, Any],
    eef_tracking: Mapping[str, Any],
    object_tracking: Mapping[str, Mapping[str, Any]],
    video_frames: Any = None,
    write_visual_media: bool = False,
    media_options: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write required tracking arrays; visual previews are not published."""

    root, prepared_plan = _validated_plan(output_dir, plan)
    planned_visual_media = bool(
        prepared_plan.get(
            "include_visual_media",
            False,
        )
    )
    if planned_visual_media or bool(write_visual_media):
        raise ValueError("tracking visual previews are not part of the public runtime")
    objects = dict(prepared_plan.get("objects", {}) or {})
    supplied_objects = dict(object_tracking or {})
    if set(supplied_objects) != set(objects):
        raise ValueError(
            "object tracking ids do not match the diagnostic plan: "
            f"tracking={sorted(supplied_objects)} "
            f"plan={sorted(objects)}"
        )
    arrays = {
        "eef": _tracking_arrays(
            eef_tracking,
            label="eef",
        ),
        **{
            object_id: _tracking_arrays(
                supplied_objects[object_id],
                label=f"object[{object_id}]",
            )
            for object_id in objects
        },
    }
    directories = ensure_traj_asset_dirs(root.as_posix())
    eef_path = Path(prepared_plan["eef"]["tracking_npz"])
    eef_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        eef_path,
        tracks_uv=arrays["eef"][0],
        visibility=arrays["eef"][1],
    )
    written = [eef_path.as_posix()]
    for object_id, record in objects.items():
        path = Path(record["tracking_npz"])
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            tracks_uv=arrays[object_id][0],
            visibility=arrays[object_id][1],
        )
        written.append(path.as_posix())
    media: dict[str, Any] = {}
    first = next(iter(objects.values()), None)
    eef_mp4 = None
    first_mp4 = None
    section = {
        "dir": directories["tracking"],
        "eef_tracks_npz": eef_path.as_posix(),
        "eef_tracking_mp4": eef_mp4,
        "obj_tracks_npz": (None if first is None else first["tracking_npz"]),
        "obj_tracking_mp4": first_mp4,
        "objects_by_id": {
            object_id: {
                "npz": record["tracking_npz"],
                "mp4": None,
                "shape": list(arrays[object_id][0].shape),
                "stage_ids": copy.deepcopy(list(record.get("stage_ids", []) or [])),
            }
            for object_id, record in objects.items()
        },
        "obj_tracks_by_stage": {
            stage_id: {
                "npz": record["tracking_npz"],
                "mp4": None,
                "object_id": object_id,
            }
            for object_id, record in objects.items()
            for stage_id in list(record.get("stage_ids", []) or [])
        },
    }
    manifest_path = update_traj_assets_manifest(
        root.as_posix(),
        "tracking",
        section,
    )
    return {
        "manifest_path": manifest_path,
        "eef_path": eef_path.as_posix(),
        "object_paths": {
            object_id: str(record["tracking_npz"])
            for object_id, record in objects.items()
        },
        "eef_mp4": eef_mp4,
        "object_mp4s": {
            object_id: None
            for object_id, record in objects.items()
        },
        "media": media,
        "written": written,
    }


_FLOW_ARRAY_KEYS = (
    "positions_camera",
    "positions_world",
    "query_tracks_uv",
    "tracker_visibility",
    "tracker_visible_mask",
    "depth_samples_override",
    "depth_valid_mask",
    "valid_mask",
)


def _geometry_payload(
    payload: Mapping[str, Any],
    *,
    label: str,
    target_depth_source: str,
    target_depth_map_path: str,
) -> tuple[list[Any], dict[str, np.ndarray]]:
    if not isinstance(payload, Mapping):
        raise TypeError(f"{label} geometry payload must be a mapping")
    points = payload.get("points", None)
    if not isinstance(points, list):
        raise TypeError(f"{label} geometry points must be a list")
    arrays = {key: np.asarray(payload.get(key, None)) for key in _FLOW_ARRAY_KEYS}
    positions = arrays["positions_camera"]
    if positions.ndim != 3 or positions.shape[-1] != 3:
        raise ValueError(
            f"{label} positions_camera must be [T,N,3], got {positions.shape}"
        )
    expected_tn = positions.shape[:2]
    if arrays["positions_world"].shape != positions.shape:
        raise ValueError(
            f"{label} positions_world mismatch: "
            f"{arrays['positions_world'].shape} != {positions.shape}"
        )
    if arrays["query_tracks_uv"].shape != (*expected_tn, 2):
        raise ValueError(
            f"{label} query_tracks_uv mismatch: {arrays['query_tracks_uv'].shape}"
        )
    for key in (
        "tracker_visibility",
        "tracker_visible_mask",
        "depth_samples_override",
        "depth_valid_mask",
        "valid_mask",
    ):
        if arrays[key].shape != expected_tn:
            raise ValueError(
                f"{label} {key} mismatch: {arrays[key].shape} != {expected_tn}"
            )
    if len(points) != positions.shape[0]:
        raise ValueError(
            f"{label} points/frame mismatch: "
            f"points={len(points)}, arrays={positions.shape[0]}"
        )
    arrays.update(
        {
            "target_depth_source": np.asarray(str(target_depth_source or "canonical")),
            "target_depth_map_path": np.asarray(str(target_depth_map_path or "")),
        }
    )
    return copy.deepcopy(points), arrays


def _write_geometry_pair(
    *,
    json_path: str,
    flow_path: str,
    points: list[Any],
    arrays: Mapping[str, np.ndarray],
) -> list[str]:
    points_path = Path(json_path)
    points_path.parent.mkdir(parents=True, exist_ok=True)
    with points_path.open("w", encoding="utf-8") as handle:
        json.dump(points, handle, indent=4)
    np.savez_compressed(
        flow_path,
        **dict(arrays),
    )
    return [
        points_path.as_posix(),
        Path(flow_path).as_posix(),
    ]


def _nearest_existing_directory(path: Path) -> Path:
    candidate = path.expanduser().resolve()
    while not candidate.exists():
        parent = candidate.parent
        if parent == candidate:
            raise ValueError(f"no existing transaction directory above {path}")
        candidate = parent
    if not candidate.is_dir():
        raise ValueError(
            f"interaction artifact transaction root is not a directory: {candidate}"
        )
    return candidate


def write_interaction_geometry_artifacts(
    output_dir: str | Path,
    *,
    plan: Mapping[str, Any],
    interaction_geometry: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    """Transactionally publish current interaction gripper geometry.

    Every object contributes the already-computed EEF/object geometry pair.
    The four current-compatible files are staged before any destination is
    changed, then committed together through the shared publication journal.
    """

    root, prepared_plan = _validated_plan(output_dir, plan)
    objects = dict(prepared_plan.get("objects", {}) or {})
    supplied_objects = dict(interaction_geometry or {})
    if set(supplied_objects) != set(objects):
        raise ValueError(
            "interaction geometry ids do not match the diagnostic plan: "
            f"geometry={sorted(supplied_objects)} "
            f"plan={sorted(objects)}"
        )

    prepared: dict[
        str,
        tuple[
            tuple[list[Any], dict[str, np.ndarray]],
            tuple[list[Any], dict[str, np.ndarray]],
        ],
    ] = {}
    for object_id in objects:
        raw_payload = supplied_objects[object_id]
        if not isinstance(raw_payload, Mapping):
            raise TypeError(f"interaction_geometry[{object_id!r}] must be a mapping")
        payload = dict(raw_payload)
        depth_reference = payload.get("depth_reference", {})
        if depth_reference is None:
            depth_reference = {}
        if not isinstance(depth_reference, Mapping):
            raise TypeError(
                f"interaction_geometry[{object_id!r}].depth_reference must be a mapping"
            )
        depth_row = dict(depth_reference)
        source = str(depth_row.get("source", "canonical") or "canonical")
        depth_path = str(depth_row.get("path", "") or "")
        prepared[object_id] = (
            _geometry_payload(
                payload.get("eef_geometry", {}),
                label=f"interaction[{object_id}].eef",
                target_depth_source=source,
                target_depth_map_path=depth_path,
            ),
            _geometry_payload(
                payload.get("object_geometry", {}),
                label=f"interaction[{object_id}].object",
                target_depth_source=source,
                target_depth_map_path=depth_path,
            ),
        )

    transaction_root = _nearest_existing_directory(root.parent)
    media: dict[str, dict[str, Any]] = {}
    object_paths: dict[str, dict[str, str]] = {}
    with tempfile.TemporaryDirectory(
        prefix=".dream-exe-interaction-geometry-source-",
        dir=transaction_root,
    ) as temporary:
        source_root = Path(temporary)
        for ordinal, (object_id, record) in enumerate(objects.items()):
            interaction = dict(record["interaction"])
            eef_geometry, object_geometry = prepared[object_id]
            staged = {
                "eef_points_json": source_root / f"{ordinal:04d}-eef.json",
                "eef_points_flow_npz": (source_root / f"{ordinal:04d}-eef-flow.npz"),
                "obj_points_json": source_root / f"{ordinal:04d}-obj.json",
                "obj_points_flow_npz": (source_root / f"{ordinal:04d}-obj-flow.npz"),
            }
            _write_geometry_pair(
                json_path=staged["eef_points_json"].as_posix(),
                flow_path=staged["eef_points_flow_npz"].as_posix(),
                points=eef_geometry[0],
                arrays=eef_geometry[1],
            )
            _write_geometry_pair(
                json_path=staged["obj_points_json"].as_posix(),
                flow_path=staged["obj_points_flow_npz"].as_posix(),
                points=object_geometry[0],
                arrays=object_geometry[1],
            )
            object_paths[object_id] = {key: str(interaction[key]) for key in staged}
            for key, source_path in staged.items():
                media[f"{ordinal:04d}:{key}"] = {
                    "source_path": source_path.as_posix(),
                    "destination_path": str(interaction[key]),
                    "media_type": (
                        "application/json"
                        if key.endswith("_json")
                        else "application/x-npz"
                    ),
                }

        publication = publish_artifact_batch(
            output_root=root,
            debug_media=media,
            transaction_root=transaction_root,
            overwrite=True,
            dry_run=False,
        )

    written = [
        path for object_id in objects for path in object_paths[object_id].values()
    ]
    return {
        **dict(publication),
        "objects": object_paths,
        "written": written,
        "transaction_scope": "interaction_gripper_geometry",
    }


def write_geometry_artifacts(
    output_dir: str | Path,
    *,
    plan: Mapping[str, Any],
    eef_geometry: Mapping[str, Any],
    object_geometry: Mapping[str, Mapping[str, Any]],
    depth_references: Mapping[str, Mapping[str, Any]] | None = None,
    manifest_payload: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write current point JSON/flow NPZ artifacts and geometry manifest."""

    root, prepared_plan = _validated_plan(output_dir, plan)
    objects = dict(prepared_plan.get("objects", {}) or {})
    supplied_objects = dict(object_geometry or {})
    if set(supplied_objects) != set(objects):
        raise ValueError(
            "object geometry ids do not match the diagnostic plan: "
            f"geometry={sorted(supplied_objects)} "
            f"plan={sorted(objects)}"
        )
    depth_rows = dict(depth_references or {})
    unknown_depth_ids = sorted(set(depth_rows).difference({"eef", *objects}))
    if unknown_depth_ids:
        raise ValueError(
            "depth_references contains unknown ids: " + ", ".join(unknown_depth_ids)
        )

    def depth_row(key: str) -> dict[str, Any]:
        raw = depth_rows.get(key, {})
        if raw is None:
            raw = {}
        if not isinstance(raw, Mapping):
            raise TypeError(f"depth_references[{key!r}] must be a mapping")
        return dict(raw)

    eef_depth = depth_row("eef")
    prepared_geometry = {
        "eef": _geometry_payload(
            eef_geometry,
            label="eef",
            target_depth_source=str(
                eef_depth.get("source", "canonical") or "canonical"
            ),
            target_depth_map_path=str(eef_depth.get("path", "") or ""),
        ),
        **{
            object_id: _geometry_payload(
                supplied_objects[object_id],
                label=f"object[{object_id}]",
                target_depth_source=str(
                    depth_row(object_id).get(
                        "source",
                        "canonical",
                    )
                    or "canonical"
                ),
                target_depth_map_path=str(depth_row(object_id).get("path", "") or ""),
            )
            for object_id in objects
        },
    }
    if manifest_payload is not None and not isinstance(
        manifest_payload,
        Mapping,
    ):
        raise TypeError("geometry manifest_payload must be a mapping")
    manifest_fields = copy.deepcopy(dict(manifest_payload or {}))
    reserved = sorted(
        {
            "dir",
            "eef_points_json",
            "eef_flow_npz",
            "obj_points_json",
            "obj_flow_npz",
            "obj_points_by_stage",
        }.intersection(manifest_fields)
    )
    if reserved:
        raise ValueError(
            "geometry manifest_payload cannot replace publication-owned "
            "fields: " + ", ".join(reserved)
        )

    directories = ensure_traj_asset_dirs(root.as_posix())
    written = _write_geometry_pair(
        json_path=prepared_plan["eef"]["points_json"],
        flow_path=prepared_plan["eef"]["points_flow_npz"],
        points=prepared_geometry["eef"][0],
        arrays=prepared_geometry["eef"][1],
    )
    for object_id, record in objects.items():
        written.extend(
            _write_geometry_pair(
                json_path=record["points_json"],
                flow_path=record["points_flow_npz"],
                points=prepared_geometry[object_id][0],
                arrays=prepared_geometry[object_id][1],
            )
        )

    first = next(iter(objects.values()), None)
    section = {
        "dir": directories["geometry"],
        "eef_points_json": prepared_plan["eef"]["points_json"],
        "eef_flow_npz": prepared_plan["eef"]["points_flow_npz"],
        "obj_points_json": (None if first is None else first["points_json"]),
        "obj_flow_npz": (None if first is None else first["points_flow_npz"]),
        "obj_points_by_stage": (
            None
            if not objects
            else {
                stage_id: {
                    "json": record["points_json"],
                    "flow_npz": record["points_flow_npz"],
                }
                for record in objects.values()
                for stage_id in list(record.get("stage_ids", []) or [])
            }
        ),
        **manifest_fields,
    }
    manifest_path = update_traj_assets_manifest(
        root.as_posix(),
        "geometry",
        section,
    )
    return {
        "manifest_path": manifest_path,
        "eef": {
            "json": prepared_plan["eef"]["points_json"],
            "flow_npz": prepared_plan["eef"]["points_flow_npz"],
        },
        "objects": {
            object_id: {
                "json": record["points_json"],
                "flow_npz": record["points_flow_npz"],
            }
            for object_id, record in objects.items()
        },
        "written": written,
    }


__all__ = [
    "build_tracking_geometry_artifact_plan",
    "write_geometry_artifacts",
    "write_interaction_geometry_artifacts",
    "write_tracking_artifacts",
]
