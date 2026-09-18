"""Pure target-specific depth calibration for trajectory lifting.

Artifact publication is deliberately outside this module.  The returned maps
and metadata retain current computational semantics and can be handed to an
explicit writer by a pipeline or bench adapter.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import numpy as np

from .calibration import build_calib_mask, solve_depth_calibration
from .config import normalize_target_calibrated_lift_config
from .dynamic_calibration import (
    apply_dynamic_affine_lift_calibration,
    apply_dynamic_shift_lift_calibration,
)


def _fallback_record(
    *,
    target: dict[str, Any],
    reason: str,
    canonical_depths: np.ndarray,
) -> dict[str, Any]:
    return {
        "target_name": str(target.get("target_name", "")),
        "kind": str(target.get("kind", "")),
        "object_id": str(target.get("object_id", "")),
        "safe_object_id": str(target.get("safe_object_id", "")),
        "stage_ids": list(target.get("stage_ids", []) or []),
        "runtime_object_key": str(target.get("runtime_object_key", "")),
        "enabled": False,
        "applied": False,
        "fallback": True,
        "fallback_reason": str(reason),
        "depth_source": "canonical",
        "depth_shape": [int(value) for value in np.asarray(canonical_depths).shape],
    }


def _calibrate_target_depths(
    *,
    source_depths: np.ndarray | None,
    canonical_depths: np.ndarray,
    reference_depth: np.ndarray | None,
    target: dict[str, Any],
    calibration_region: str,
    calibration_solver: str,
    min_valid_pixels: int,
) -> tuple[np.ndarray | None, dict[str, Any], np.ndarray | None]:
    canonical = np.asarray(canonical_depths, dtype=np.float32)
    if source_depths is None:
        return (
            None,
            _fallback_record(
                target=target,
                reason="source_depth_unavailable",
                canonical_depths=canonical,
            ),
            None,
        )
    if reference_depth is None:
        return (
            None,
            _fallback_record(
                target=target,
                reason="init_ref_depth_unavailable",
                canonical_depths=canonical,
            ),
            None,
        )
    raw_mask = target.get("mask0", None)
    if raw_mask is None:
        return (
            None,
            _fallback_record(
                target=target,
                reason="target_mask_unavailable",
                canonical_depths=canonical,
            ),
            None,
        )
    region_mask = np.asarray(raw_mask, dtype=bool)
    if region_mask.shape != canonical.shape[1:]:
        return (
            None,
            _fallback_record(
                target=target,
                reason="target_mask_shape_mismatch",
                canonical_depths=canonical,
            ),
            None,
        )

    calibration_mask = build_calib_mask(
        source_depths[0],
        reference_depth,
        calib_region=str(calibration_region or "roi∧valid"),
        valid_mask0=None,
        roi_mask0=region_mask,
        tracks_uv0=None,
        custom_calib_masks={},
    )
    pixels = int(np.sum(calibration_mask))
    if pixels < int(min_valid_pixels):
        record = _fallback_record(
            target=target,
            reason=f"insufficient_calib_pixels:{pixels}",
            canonical_depths=canonical,
        )
        record.update(
            {
                "mask_pixels": pixels,
                "mask_frac": float(np.mean(calibration_mask)),
            }
        )
        return None, record, calibration_mask

    try:
        _, parameters = solve_depth_calibration(
            source_depths[0],
            reference_depth,
            valid_mask=calibration_mask,
            calibration_solver=str(calibration_solver),
        )
        scale = float(parameters.get("s", 1.0))
        bias = float(parameters.get("b", 0.0))
        target_depths = (scale * source_depths + bias).astype(np.float32)
        record = {
            "target_name": str(target.get("target_name", "")),
            "kind": str(target.get("kind", "")),
            "object_id": str(target.get("object_id", "")),
            "safe_object_id": str(target.get("safe_object_id", "")),
            "stage_ids": list(target.get("stage_ids", []) or []),
            "runtime_object_key": str(target.get("runtime_object_key", "")),
            "enabled": True,
            "applied": True,
            "fallback": False,
            "fallback_reason": "",
            "depth_source": str(
                target.get(
                    "depth_source",
                    "target_calibrated",
                )
                or "target_calibrated"
            ),
            "s": scale,
            "b": bias,
            "mask_pixels": pixels,
            "mask_frac": float(np.mean(calibration_mask)),
            "calib_region": str(calibration_region),
            "calibration_solver": str(
                parameters.get(
                    "calibration_solver",
                    calibration_solver,
                )
            ),
            "depth_npy": "",
            "depth_mp4": "",
        }
        return target_depths, record, calibration_mask
    except Exception as error:
        record = _fallback_record(
            target=target,
            reason=f"calibration_failed:{error}",
            canonical_depths=canonical,
        )
        record.update(
            {
                "mask_pixels": pixels,
                "mask_frac": float(np.mean(calibration_mask)),
            }
        )
        return None, record, calibration_mask


def _union_masks(
    specifications: Sequence[dict[str, Any]],
    *,
    shape: tuple[int, int],
) -> np.ndarray | None:
    output = np.zeros(shape, dtype=bool)
    seen = False
    for specification in list(specifications or []):
        raw_mask = specification.get("mask0", None)
        if raw_mask is None:
            continue
        mask = np.asarray(raw_mask, dtype=bool)
        if mask.shape != shape:
            continue
        output |= mask
        seen = True
    return output if seen else None


def build_target_calibrated_depth_maps(
    *,
    raw_depths: np.ndarray | None,
    canonical_depths: np.ndarray,
    init_ref_depth: np.ndarray | None,
    target_specs: Sequence[dict[str, Any]],
    config: dict[str, Any],
    eef_tracks_uv: np.ndarray | None = None,
    eef_visibility: np.ndarray | None = None,
    dynamic_lift_stages: Sequence[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Build EEF/object-pair depth maps without performing artifact I/O."""

    normalized = normalize_target_calibrated_lift_config(config)
    canonical = np.asarray(canonical_depths, dtype=np.float32)
    if canonical.ndim != 3:
        raise ValueError(f"canonical_depths must be [T,H,W], got {canonical.shape}")
    metadata: dict[str, Any] = {
        "enabled": bool(normalized.get("enabled", False)),
        "source_stage": str(normalized.get("source_stage", "raw_model")),
        "calibration_solver": str(
            normalized.get(
                "calibration_solver",
                "robust_affine",
            )
        ),
        "min_valid_pixels": int(normalized.get("min_valid_pixels", 16)),
        "eef_traj": dict(normalized.get("eef_traj", {}) or {}),
        "object_traj": dict(normalized.get("object_traj", {}) or {}),
        "gripper_traj": dict(normalized.get("gripper_traj", {}) or {}),
        "targets": {},
    }
    maps: dict[str, np.ndarray] = {}
    masks: dict[str, np.ndarray] = {}
    if not bool(normalized.get("enabled", False)):
        metadata["reason"] = "disabled"
        return {
            "maps": maps,
            "publication_maps": {},
            "masks": masks,
            "meta": metadata,
            "publication_meta": copy.deepcopy(metadata),
            "meta_path": "",
        }

    source_stage = str(normalized.get("source_stage", "raw_model") or "raw_model")
    if source_stage == "canonical":
        source_depths = canonical
    elif raw_depths is not None:
        source_depths = np.asarray(raw_depths, dtype=np.float32)
    else:
        source_depths = None
    if source_depths is not None and source_depths.shape != canonical.shape:
        metadata["source_shape_mismatch"] = {
            "source": [int(value) for value in source_depths.shape],
            "canonical": [int(value) for value in canonical.shape],
        }
        source_depths = None

    reference_depth = (
        None if init_ref_depth is None else np.asarray(init_ref_depth, dtype=np.float32)
    )
    if reference_depth is not None and reference_depth.shape != canonical.shape[1:]:
        metadata["init_ref_shape_mismatch"] = {
            "init_ref": [int(value) for value in reference_depth.shape],
            "depth_hw": [
                int(canonical.shape[1]),
                int(canonical.shape[2]),
            ],
        }
        reference_depth = None

    specifications = [dict(item) for item in list(target_specs or [])]
    eef_specification = next(
        (item for item in specifications if str(item.get("kind", "")) == "eef"),
        None,
    )
    object_specifications = [
        item for item in specifications if str(item.get("kind", "")) == "object"
    ]
    pair_specifications = [
        item for item in specifications if str(item.get("kind", "")) == "pair"
    ]
    object_config = dict(normalized.get("object_traj", {}) or {})
    solver = str(normalized.get("calibration_solver", "robust_affine"))
    min_pixels = int(normalized.get("min_valid_pixels", 16))

    eef_config = dict(normalized.get("eef_traj", {}) or {})
    if bool(eef_config.get("enabled", True)) and eef_specification is not None:
        target = dict(eef_specification)
        target["target_name"] = "eef"
        target["depth_source"] = (
            f"depth_eef_traj:{eef_config.get('mode', 'static_eef')}"
        )
        eef_depths, record, calibration_mask = _calibrate_target_depths(
            source_depths=source_depths,
            canonical_depths=canonical,
            reference_depth=reference_depth,
            target=target,
            calibration_region=str(
                dict(eef_config.get("static_eef", {}) or {}).get(
                    "calib_region",
                    "roi∧valid",
                )
            ),
            calibration_solver=solver,
            min_valid_pixels=min_pixels,
        )
        if eef_depths is not None:
            maps["eef"] = eef_depths
        if calibration_mask is not None:
            masks["eef"] = calibration_mask
        metadata["targets"]["eef"] = record

    gripper_config = dict(normalized.get("gripper_traj", {}) or {})
    if (
        bool(gripper_config.get("enabled", True))
        and eef_specification is not None
        and object_specifications
    ):
        mode = str(gripper_config.get("mode", "pair_roi") or "pair_roi")
        if mode == "all_roi":
            mask = _union_masks(
                [eef_specification, *object_specifications],
                shape=canonical.shape[1:],
            )
            target = {
                "target_name": "gripper_traj_all_roi",
                "kind": "all_roi",
                "object_id": "all",
                "safe_object_id": "all",
                "mask0": mask,
                "depth_source": "depth_gripper_traj:all_roi",
            }
            pair_depths, record, calibration_mask = _calibrate_target_depths(
                source_depths=source_depths,
                canonical_depths=canonical,
                reference_depth=reference_depth,
                target=target,
                calibration_region=str(
                    dict(gripper_config.get("all_roi", {}) or {}).get(
                        "calib_region", "roi∧valid"
                    )
                ),
                calibration_solver=solver,
                min_valid_pixels=min_pixels,
            )
            if pair_depths is not None:
                maps["gripper_traj_all_roi"] = pair_depths
            if calibration_mask is not None:
                masks["gripper_traj_all_roi"] = calibration_mask
            metadata["targets"]["gripper_traj_all_roi"] = record
        else:
            for object_specification in object_specifications:
                pair_key = (
                    "pair_"
                    f"{object_specification.get('object_id', object_specification.get('target_name', 'object'))}"
                )
                existing_pair = next(
                    (
                        item
                        for item in pair_specifications
                        if str(item.get("target_name", "")) == pair_key
                    ),
                    None,
                )
                pair_mask = (
                    np.asarray(existing_pair["mask0"], dtype=bool)
                    if (
                        existing_pair is not None
                        and existing_pair.get("mask0", None) is not None
                    )
                    else _union_masks(
                        [eef_specification, object_specification],
                        shape=canonical.shape[1:],
                    )
                )
                target = {
                    "target_name": pair_key,
                    "kind": "pair",
                    "object_id": str(object_specification.get("object_id", "")),
                    "safe_object_id": str(
                        object_specification.get(
                            "safe_object_id",
                            "",
                        )
                    ),
                    "mask0": pair_mask,
                    "stage_ids": list(
                        object_specification.get(
                            "stage_ids",
                            [],
                        )
                        or []
                    ),
                    "runtime_object_key": str(
                        object_specification.get(
                            "runtime_object_key",
                            "",
                        )
                    ),
                    "depth_source": ("depth_gripper_traj:pair_roi"),
                }
                pair_depths, record, calibration_mask = _calibrate_target_depths(
                    source_depths=source_depths,
                    canonical_depths=canonical,
                    reference_depth=reference_depth,
                    target=target,
                    calibration_region=str(
                        dict(gripper_config.get("pair_roi", {}) or {}).get(
                            "calib_region", "roi∧valid"
                        )
                    ),
                    calibration_solver=solver,
                    min_valid_pixels=min_pixels,
                )
                if pair_depths is not None:
                    maps[pair_key] = pair_depths
                if calibration_mask is not None:
                    masks[pair_key] = calibration_mask
                metadata["targets"][pair_key] = record

    if bool(object_config.get("enabled", False)):
        object_region = str(
            dict(object_config.get("object_roi", {}) or {}).get(
                "calib_region",
                "roi∧valid",
            )
        )
        for object_specification in object_specifications:
            key = str(object_specification.get("object_id", "")) or str(
                object_specification.get(
                    "target_name",
                    "object",
                )
            )
            target = dict(object_specification)
            target["target_name"] = key
            target["depth_source"] = "depth_object_traj:object_roi"
            object_depths, record, calibration_mask = _calibrate_target_depths(
                source_depths=source_depths,
                canonical_depths=canonical,
                reference_depth=reference_depth,
                target=target,
                calibration_region=object_region,
                calibration_solver=solver,
                min_valid_pixels=min_pixels,
            )
            if object_depths is not None:
                maps[key] = object_depths
            if calibration_mask is not None:
                masks[key] = calibration_mask
            metadata["targets"][key] = record

    publication_maps = {key: value.copy() for key, value in maps.items()}
    # The current pipeline publishes the static target-calibration record and
    # then applies the dynamic EEF correction in memory for lifting.  Preserve
    # both views explicitly: publication metadata remains compatible with the
    # saved static calibration artifact, while runtime metadata below records
    # the dynamic map that was actually consumed.
    publication_metadata = copy.deepcopy(metadata)
    dynamic_mode = str(eef_config.get("mode", "static_eef"))
    if dynamic_mode in {"dynamic_shift", "dynamic_affine"}:
        missing = []
        if "eef" not in maps:
            missing.append("static_eef_target")
        if dynamic_mode == "dynamic_affine" and raw_depths is None:
            missing.append("raw_depths")
        if reference_depth is None:
            missing.append("init_ref_depth")
        if eef_tracks_uv is None:
            missing.append("eef_tracks_uv")
        if eef_visibility is None:
            missing.append("eef_visibility")
        if not list(dynamic_lift_stages or []):
            missing.append("dynamic_lift_stages")
        if missing:
            dynamic_meta = {
                "enabled": True,
                "mode": dynamic_mode,
                "applied": False,
                "reason": "missing_inputs:" + ",".join(missing),
                "stages": [],
            }
        elif dynamic_mode == "dynamic_shift":
            dynamic_result = apply_dynamic_shift_lift_calibration(
                static_eef_depths=maps["eef"],
                init_reference_depth=reference_depth,
                eef_tracks_uv=eef_tracks_uv,
                eef_visibility=eef_visibility,
                stages=list(dynamic_lift_stages or []),
                config=dict(eef_config.get("dynamic_shift", {}) or {}),
            )
            maps["eef"] = np.asarray(
                dynamic_result["eef_depths"],
                dtype=np.float32,
            )
            dynamic_meta = dict(dynamic_result["meta"])
        else:
            dynamic_result = apply_dynamic_affine_lift_calibration(
                static_eef_depths=maps["eef"],
                raw_depths=raw_depths,
                init_reference_depth=reference_depth,
                eef_tracks_uv=eef_tracks_uv,
                eef_visibility=eef_visibility,
                stages=list(dynamic_lift_stages or []),
                config=dict(eef_config.get("dynamic_affine", {}) or {}),
                calibration_solver=solver,
            )
            maps["eef"] = np.asarray(
                dynamic_result["eef_depths"],
                dtype=np.float32,
            )
            dynamic_meta = dict(dynamic_result["meta"])
        metadata["dynamic_lift"] = dynamic_meta
        eef_record = dict(metadata["targets"].get("eef", {}) or {})
        eef_record["dynamic_lift"] = dynamic_meta
        eef_record["dynamic_lift_applied"] = bool(dynamic_meta.get("applied", False))
        if eef_record["dynamic_lift_applied"]:
            eef_record["depth_source"] = str(
                dynamic_meta.get("mode", dynamic_mode) or dynamic_mode
            )
            eef_record["dynamic_ir_applied"] = True
        metadata["targets"]["eef"] = eef_record

    return {
        "maps": maps,
        "publication_maps": publication_maps,
        "masks": masks,
        "meta": metadata,
        "publication_meta": publication_metadata,
        "meta_path": "",
    }


__all__ = ["build_target_calibrated_depth_maps"]
