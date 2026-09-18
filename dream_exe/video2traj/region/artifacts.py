"""Explicit current-compatible first-frame region publication.

Region selection remains an in-memory algorithm step.  This module is the
narrow serialization boundary for the current ``region/*.json`` metadata and
static mask PNGs, with opt-in current-compatible first-frame overlays and
summaries.  It never discovers a benchmark, simulator, run key, input video,
or output root, and it does not publish videos, depth diagnostics, or
pose-backend media.
"""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from dream_exe.artifacts.io import (
    ensure_traj_asset_dirs,
    update_traj_assets_manifest,
)


def _explicit_root(output_dir: str | Path) -> Path:
    output_text = str(output_dir or "").strip()
    if not output_text:
        raise ValueError("output_dir must be an explicit non-empty path")
    return Path(output_text).expanduser().resolve()


def _safe_object_id(
    raw_stream: Mapping[str, Any],
    *,
    object_id: str,
) -> str:
    safe_object_id = str(
        raw_stream.get("safe_object_id", object_id) or object_id
    ).strip()
    if (
        not safe_object_id
        or safe_object_id in {".", ".."}
        or "/" in safe_object_id
        or "\\" in safe_object_id
    ):
        raise ValueError(
            f"unsafe object artifact id for object_id={object_id!r}: {safe_object_id!r}"
        )
    return safe_object_id


def build_region_artifact_plan(
    output_dir: str | Path,
    *,
    object_stream_plan: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Plan canonical current region paths without touching the filesystem."""

    root = _explicit_root(output_dir)
    region_root = root / "region"
    objects: dict[str, dict[str, Any]] = {}
    seen: set[str] = set()
    for ordinal, raw_stream in enumerate(list(object_stream_plan or [])):
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
        safe_object_id = _safe_object_id(
            stream,
            object_id=object_id,
        )
        object_root = region_root / "objects" / safe_object_id
        objects[object_id] = {
            "object_id": object_id,
            "safe_object_id": safe_object_id,
            "stage_ids": [
                str(stage_id) for stage_id in list(stream.get("stage_ids", []) or [])
            ],
            "ordinal": int(ordinal),
            "region_json": (object_root / "obj_region.json").as_posix(),
            "mask_png": (object_root / "obj_mask.png").as_posix(),
            "eroded_mask_png": (object_root / "obj_mask_eroded.png").as_posix(),
            "sampling_mask_png": (object_root / "obj_sampling_mask.png").as_posix(),
            "bbox_overlay_png": (object_root / "obj_bbox_overlay.png").as_posix(),
            "mask_overlay_png": (object_root / "obj_mask_overlay.png").as_posix(),
            "eroded_overlay_png": (object_root / "obj_eroded_overlay.png").as_posix(),
            "sampling_overlay_png": (
                object_root / "obj_sampling_overlay.png"
            ).as_posix(),
            "summary_png": (object_root / "obj_summary.png").as_posix(),
        }
    eef = {
        "region_json": (region_root / "eef_region.json").as_posix(),
        "mask_png": (region_root / "eef_mask.png").as_posix(),
        "eroded_mask_png": (region_root / "eef_mask_eroded.png").as_posix(),
        "sampling_mask_png": (region_root / "eef_sampling_mask.png").as_posix(),
        "bbox_overlay_png": (region_root / "eef_bbox_overlay.png").as_posix(),
        "mask_overlay_png": (region_root / "eef_mask_overlay.png").as_posix(),
        "eroded_overlay_png": (region_root / "eef_eroded_overlay.png").as_posix(),
        "sampling_overlay_png": (region_root / "eef_sampling_overlay.png").as_posix(),
        "summary_png": (region_root / "eef_summary.png").as_posix(),
    }
    return {
        "output_dir": root.as_posix(),
        "region_dir": region_root.as_posix(),
        "regions_json": (region_root / "regions.json").as_posix(),
        "eef": eef,
        "objects": objects,
        "object_references": {
            object_id: {
                "region_json": record["region_json"],
            }
            for object_id, record in objects.items()
        },
    }


def _validated_plan(
    output_dir: str | Path,
    plan: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    if not isinstance(plan, Mapping):
        raise TypeError("region artifact plan must be a mapping")
    root = _explicit_root(output_dir)
    payload = copy.deepcopy(dict(plan))
    planned_root = Path(str(payload.get("output_dir", "") or "")).expanduser().resolve()
    if planned_root != root:
        raise ValueError(
            "region artifact plan/output_dir mismatch: "
            f"plan={planned_root} output={root}"
        )
    path_records = [
        {
            "regions_json": payload.get("regions_json", ""),
        },
        dict(payload.get("eef", {}) or {}),
        *[
            dict(value or {})
            for value in dict(payload.get("objects", {}) or {}).values()
        ],
    ]
    for record in path_records:
        for key, value in record.items():
            if key in {
                "object_id",
                "safe_object_id",
                "stage_ids",
                "ordinal",
            }:
                continue
            path_text = str(value or "")
            if not path_text:
                raise ValueError(f"region artifact plan is missing {key}")
            path = Path(path_text).expanduser().resolve()
            if root != path and root not in path.parents:
                raise ValueError(f"region artifact path escapes output_dir: {path}")
    return root, payload


def _value(
    result: Any,
    name: str,
    *,
    default: Any = None,
) -> Any:
    if isinstance(result, Mapping):
        return result.get(name, default)
    return getattr(result, name, default)


def _mask(
    result: Any,
    name: str,
    *,
    fallback: np.ndarray | None = None,
) -> np.ndarray:
    raw = _value(result, name, default=fallback)
    if raw is None:
        raise ValueError(f"region result is missing {name}")
    array = np.asarray(raw, dtype=bool)
    if array.ndim != 2:
        raise ValueError(f"region {name} must be a 2-D mask, got {array.shape}")
    return array


def _base_region_payload(result: Any) -> dict[str, Any]:
    serializer = getattr(result, "to_dict", None)
    if callable(serializer):
        raw = serializer()
        if not isinstance(raw, Mapping):
            raise TypeError("region result to_dict() must return a mapping")
        return copy.deepcopy(dict(raw))
    if isinstance(result, Mapping):
        raw_payload = result.get("payload", None)
        if isinstance(raw_payload, Mapping):
            return copy.deepcopy(dict(raw_payload))

    mask = _mask(result, "mask")
    sampling_mask = _mask(
        result,
        "sampling_mask",
        fallback=mask,
    )
    eroded_mask = _mask(
        result,
        "eroded_mask",
        fallback=sampling_mask,
    )
    sampled_points = np.asarray(
        _value(result, "sampled_points_xy", default=[]),
        dtype=np.float32,
    )
    if sampled_points.ndim != 2 or sampled_points.shape[-1:] != (2,):
        raise ValueError(
            "region sampled_points_xy must have shape [N,2], "
            f"got {sampled_points.shape}"
        )
    sampled_points_xyz_raw = _value(
        result,
        "sampled_points_xyz",
        default=None,
    )
    sampled_points_xyz = (
        None
        if sampled_points_xyz_raw is None
        else np.asarray(sampled_points_xyz_raw).astype(float).tolist()
    )
    source = str(_value(result, "source", default="") or "")
    payload = {
        "target_name": str(_value(result, "target_name", default="") or ""),
        "source": source,
        "bbox_source": str(_value(result, "bbox_source", default=source) or source),
        "sampling_method": str(_value(result, "sampling_method", default="") or ""),
        "mask_backend": _value(
            result,
            "mask_backend",
            default=None,
        ),
        "prompt": _value(result, "prompt", default=None),
        "manual_bbox_xyxy": copy.deepcopy(
            _value(result, "manual_bbox_xyxy", default=None)
        ),
        "detected_bbox_xyxy": copy.deepcopy(
            _value(result, "detected_bbox_xyxy", default=None)
        ),
        "final_bbox_xyxy": [
            int(value)
            for value in list(_value(result, "final_bbox_xyxy", default=[]) or [])
        ],
        "mask_area": int(np.count_nonzero(mask)),
        "eroded_mask_area": int(np.count_nonzero(eroded_mask)),
        "sampling_mask_area": int(np.count_nonzero(sampling_mask)),
        "num_sampled_points": int(sampled_points.shape[0]),
        "used_depth_for_sampling": bool(
            _value(
                result,
                "used_depth_for_sampling",
                default=False,
            )
        ),
        "tracking_input": str(
            _value(result, "tracking_input", default="points") or "points"
        ),
        "sampled_points_xy": sampled_points.astype(float).tolist(),
        "sampled_points_xyz": sampled_points_xyz,
        "artifacts": {},
        "selection_metadata": copy.deepcopy(
            dict(
                _value(
                    result,
                    "selection_metadata",
                    default={},
                )
                or {}
            )
        ),
    }
    sampling_backend = copy.deepcopy(
        dict(
            _value(
                result,
                "sampling_backend",
                default={},
            )
            or {}
        )
    )
    provider_chain = copy.deepcopy(
        dict(
            _value(
                result,
                "provider_chain",
                default={},
            )
            or {}
        )
    )
    if sampling_backend:
        payload["sampling_backend"] = sampling_backend
    if provider_chain:
        payload["provider_chain"] = provider_chain
    return payload


def _prepared_region(
    result: Any,
    *,
    paths: Mapping[str, Any],
    target_name: str,
    write_visual_media: bool,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    mask = _mask(result, "mask")
    sampling_mask = _mask(
        result,
        "sampling_mask",
        fallback=mask,
    )
    eroded_mask = _mask(
        result,
        "eroded_mask",
        fallback=sampling_mask,
    )
    if not (mask.shape == eroded_mask.shape == sampling_mask.shape):
        raise ValueError(
            "region mask shapes must match: "
            f"mask={mask.shape}, eroded={eroded_mask.shape}, "
            f"sampling={sampling_mask.shape}"
        )
    payload = _base_region_payload(result)
    if not str(payload.get("target_name", "") or "").strip():
        payload["target_name"] = str(target_name)
    payload["artifacts"] = {
        "json": str(paths["region_json"]),
        "mask": str(paths["mask_png"]),
        "mask_eroded": str(paths["eroded_mask_png"]),
        "sampling_mask": str(paths["sampling_mask_png"]),
        **(
            {
                "bbox_overlay": str(paths["bbox_overlay_png"]),
                "mask_overlay": str(paths["mask_overlay_png"]),
                "eroded_overlay": str(paths["eroded_overlay_png"]),
                "sampling_overlay": str(paths["sampling_overlay_png"]),
                "summary": str(paths["summary_png"]),
            }
            if bool(write_visual_media)
            else {}
        ),
    }
    return (
        payload,
        {
            "mask": mask,
            "eroded_mask": eroded_mask,
            "sampling_mask": sampling_mask,
        },
    )


def _write_mask_png(path: str, mask: np.ndarray) -> str:
    import cv2

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    success = cv2.imwrite(
        target.as_posix(),
        np.asarray(mask, dtype=np.uint8) * 255,
    )
    if not success:
        raise OSError(f"failed to write region mask PNG: {target}")
    return target.as_posix()


def _draw_region_overlay(
    *,
    frame_rgb: np.ndarray,
    payload: Mapping[str, Any],
    mask_shape: tuple[int, int],
    mask: np.ndarray | None,
    include_points: bool,
    title: str,
) -> np.ndarray:
    import cv2

    overlay = cv2.cvtColor(
        np.asarray(frame_rgb).astype(np.uint8),
        cv2.COLOR_RGB2BGR,
    )
    if mask is not None and np.any(mask):
        mask_bool = np.asarray(mask, dtype=bool)
        overlay[mask_bool] = (
            0.65 * overlay[mask_bool] + 0.35 * np.array([0, 255, 0], dtype=np.float32)
        ).astype(np.uint8)

    manual_bbox = payload.get("manual_bbox_xyxy", None)
    if manual_bbox is not None:
        x0, y0, x1, y1 = [int(value) for value in manual_bbox]
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            (255, 0, 0),
            1,
        )
        cv2.putText(
            overlay,
            "manual",
            (x0, max(14, y0 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )
    detected_bbox = payload.get("detected_bbox_xyxy", None)
    if detected_bbox is not None:
        x0, y0, x1, y1 = [int(value) for value in detected_bbox]
        cv2.rectangle(
            overlay,
            (x0, y0),
            (x1, y1),
            (255, 0, 255),
            1,
        )
        cv2.putText(
            overlay,
            "detected",
            (x0, min(mask_shape[0] - 8, y1 + 14)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    final_bbox = [
        int(value) for value in list(payload.get("final_bbox_xyxy", []) or [])
    ]
    if len(final_bbox) != 4:
        raise ValueError("region final_bbox_xyxy must contain four values")
    x0, y0, x1, y1 = final_bbox
    cv2.rectangle(
        overlay,
        (x0, y0),
        (x1, y1),
        (0, 255, 255),
        2,
    )
    cv2.putText(
        overlay,
        "final",
        (x0, max(28, y0 - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )
    if include_points:
        points = np.asarray(
            payload.get("sampled_points_xy", []),
            dtype=np.float32,
        )
        for x, y in points.astype(int):
            cv2.circle(
                overlay,
                (int(x), int(y)),
                2,
                (0, 0, 255),
                thickness=-1,
            )
    if title:
        cv2.putText(
            overlay,
            title,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            overlay,
            title,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )
    return overlay


def _write_region_visual_media(
    *,
    paths: Mapping[str, Any],
    payload: Mapping[str, Any],
    masks: Mapping[str, np.ndarray],
    frame_rgb: np.ndarray,
) -> list[str]:
    import cv2

    frame = np.asarray(frame_rgb)
    expected_shape = tuple(masks["mask"].shape)
    if frame.ndim != 3 or frame.shape[-1] != 3:
        raise ValueError(f"first_frame_rgb must have shape [H,W,3], got {frame.shape}")
    if tuple(frame.shape[:2]) != expected_shape:
        raise ValueError(
            "region frame/mask mismatch: "
            f"frame={frame.shape[:2]}, mask={expected_shape}"
        )
    target_name = str(payload.get("target_name", "") or "")
    overlays = {
        "bbox_overlay_png": _draw_region_overlay(
            frame_rgb=frame,
            payload=payload,
            mask_shape=expected_shape,
            mask=None,
            include_points=False,
            title=f"{target_name}: bbox",
        ),
        "mask_overlay_png": _draw_region_overlay(
            frame_rgb=frame,
            payload=payload,
            mask_shape=expected_shape,
            mask=masks["mask"],
            include_points=False,
            title=f"{target_name}: mask",
        ),
        "eroded_overlay_png": _draw_region_overlay(
            frame_rgb=frame,
            payload=payload,
            mask_shape=expected_shape,
            mask=masks["eroded_mask"],
            include_points=False,
            title=f"{target_name}: eroded mask",
        ),
        "sampling_overlay_png": _draw_region_overlay(
            frame_rgb=frame,
            payload=payload,
            mask_shape=expected_shape,
            mask=masks["sampling_mask"],
            include_points=True,
            title=f"{target_name}: sampling",
        ),
    }
    summary_top = np.concatenate(
        [
            overlays["bbox_overlay_png"],
            overlays["mask_overlay_png"],
        ],
        axis=1,
    )
    summary_bottom = np.concatenate(
        [
            overlays["eroded_overlay_png"],
            overlays["sampling_overlay_png"],
        ],
        axis=1,
    )
    overlays["summary_png"] = np.concatenate(
        [summary_top, summary_bottom],
        axis=0,
    )
    written = []
    for key, image in overlays.items():
        path = Path(str(paths[key]))
        path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(path.as_posix(), image):
            raise OSError(f"failed to write region visualization PNG: {path}")
        written.append(path.as_posix())
    return written


def _write_region_preview_media(
    output_dir: str | Path,
    *,
    region_payload: Mapping[str, Any],
    mask: Any,
    eroded_mask: Any,
    sampling_mask: Any,
    first_frame_rgb: Any,
    prefix: str,
) -> dict[str, str]:
    """Render the current five first-frame region diagnostic PNGs.

    All behavior-affecting data is caller supplied. The function does not
    resolve benchmark paths, select a region, invoke detector/segmenter
    backends, or update an artifact manifest.
    """

    safe_prefix = str(prefix or "").strip()
    if safe_prefix not in {"eef", "obj"}:
        raise ValueError("region diagnostic prefix must be 'eef' or 'obj'")
    directory = _explicit_root(output_dir)
    filenames = {
        "bbox_overlay_png": f"{safe_prefix}_bbox_overlay.png",
        "mask_overlay_png": f"{safe_prefix}_mask_overlay.png",
        "eroded_overlay_png": f"{safe_prefix}_eroded_overlay.png",
        "sampling_overlay_png": f"{safe_prefix}_sampling_overlay.png",
        "summary_png": f"{safe_prefix}_summary.png",
    }
    paths = {key: (directory / name).as_posix() for key, name in filenames.items()}
    occupied = [
        Path(path)
        for path in paths.values()
        if Path(path).exists() or Path(path).is_symlink()
    ]
    if occupied:
        raise FileExistsError(
            "region diagnostic destinations must be unused: "
            + ", ".join(path.as_posix() for path in occupied)
        )
    masks = {
        "mask": np.asarray(mask, dtype=bool),
        "eroded_mask": np.asarray(eroded_mask, dtype=bool),
        "sampling_mask": np.asarray(sampling_mask, dtype=bool),
    }
    shapes = {tuple(value.shape) for value in masks.values()}
    if len(shapes) != 1 or any(value.ndim != 2 for value in masks.values()):
        raise ValueError(
            "region diagnostic masks must be matching 2-D arrays: "
            + ", ".join(
                f"{key}={value.shape}" for key, value in masks.items()
            )
        )
    written = _write_region_visual_media(
        paths=paths,
        payload=copy.deepcopy(dict(region_payload)),
        masks=masks,
        frame_rgb=np.asarray(first_frame_rgb),
    )
    return {
        key: path
        for key, path in paths.items()
        if path in written
    }


def _write_json(path: str, payload: Mapping[str, Any]) -> str:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    return target.as_posix()


def write_region_artifacts(
    output_dir: str | Path,
    *,
    plan: Mapping[str, Any],
    eef_region: Any,
    object_regions: Mapping[str, Any],
    first_frame_rgb: Any = None,
    write_visual_media: bool = False,
) -> dict[str, Any]:
    """Write current region artifacts and publish manifest references."""

    root, prepared_plan = _validated_plan(output_dir, plan)
    planned_objects = dict(prepared_plan.get("objects", {}) or {})
    supplied_objects = dict(object_regions or {})
    if set(supplied_objects) != set(planned_objects):
        raise ValueError(
            "object region ids do not match the region plan: "
            f"regions={sorted(supplied_objects)} "
            f"plan={sorted(planned_objects)}"
        )

    eef_payload, eef_masks = _prepared_region(
        eef_region,
        paths=dict(prepared_plan["eef"]),
        target_name="eef",
        write_visual_media=bool(write_visual_media),
    )
    prepared_objects = {
        object_id: _prepared_region(
            supplied_objects[object_id],
            paths=record,
            target_name="obj",
            write_visual_media=bool(write_visual_media),
        )
        for object_id, record in planned_objects.items()
    }
    frame = None
    if bool(write_visual_media):
        if first_frame_rgb is None:
            raise ValueError("first_frame_rgb is required when write_visual_media=True")
        frame = np.asarray(first_frame_rgb)
    written = [
        _write_mask_png(
            prepared_plan["eef"]["mask_png"],
            eef_masks["mask"],
        ),
        _write_mask_png(
            prepared_plan["eef"]["eroded_mask_png"],
            eef_masks["eroded_mask"],
        ),
        _write_mask_png(
            prepared_plan["eef"]["sampling_mask_png"],
            eef_masks["sampling_mask"],
        ),
        _write_json(
            prepared_plan["eef"]["region_json"],
            eef_payload,
        ),
    ]
    if frame is not None:
        written.extend(
            _write_region_visual_media(
                paths=dict(prepared_plan["eef"]),
                payload=eef_payload,
                masks=eef_masks,
                frame_rgb=frame,
            )
        )
    for object_id, record in planned_objects.items():
        payload, masks = prepared_objects[object_id]
        written.extend(
            [
                _write_mask_png(
                    record["mask_png"],
                    masks["mask"],
                ),
                _write_mask_png(
                    record["eroded_mask_png"],
                    masks["eroded_mask"],
                ),
                _write_mask_png(
                    record["sampling_mask_png"],
                    masks["sampling_mask"],
                ),
                _write_json(
                    record["region_json"],
                    payload,
                ),
            ]
        )
        if frame is not None:
            written.extend(
                _write_region_visual_media(
                    paths=record,
                    payload=payload,
                    masks=masks,
                    frame_rgb=frame,
                )
            )

    object_payloads = {
        object_id: copy.deepcopy(prepared_objects[object_id][0])
        for object_id in planned_objects
    }
    stage_payloads = {
        stage_id: copy.deepcopy(object_payloads[object_id])
        for object_id, record in planned_objects.items()
        for stage_id in list(record.get("stage_ids", []) or [])
    }
    regions_payload: dict[str, Any] = {
        "eef": copy.deepcopy(eef_payload),
        "objects": object_payloads,
        "stages": stage_payloads,
    }
    first = next(iter(planned_objects), None)
    if first is not None:
        regions_payload["obj"] = copy.deepcopy(object_payloads[first])
    written.append(
        _write_json(
            prepared_plan["regions_json"],
            regions_payload,
        )
    )

    directories = ensure_traj_asset_dirs(root.as_posix())
    first_record = None if first is None else planned_objects[first]
    section = {
        "dir": directories["region"],
        "regions_json": prepared_plan["regions_json"],
        "eef_region_json": prepared_plan["eef"]["region_json"],
        "obj_region_json": (
            None if first_record is None else first_record["region_json"]
        ),
        "obj_regions_by_id": {
            object_id: record["region_json"]
            for object_id, record in planned_objects.items()
        },
        "obj_region_jsons": {
            stage_id: record["region_json"]
            for record in planned_objects.values()
            for stage_id in list(record.get("stage_ids", []) or [])
        },
    }
    manifest_path = update_traj_assets_manifest(
        root.as_posix(),
        "region",
        section,
    )
    return {
        "manifest_path": manifest_path,
        "regions_json": prepared_plan["regions_json"],
        "eef_region_json": prepared_plan["eef"]["region_json"],
        "object_region_jsons": {
            object_id: record["region_json"]
            for object_id, record in planned_objects.items()
        },
        "written": written,
    }


__all__ = [
    "build_region_artifact_plan",
    "write_region_artifacts",
]
