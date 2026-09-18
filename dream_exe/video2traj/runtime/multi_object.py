"""Explicit raw-video-file orchestration for prepared multi-object outputs.

This callable preserves the current high-level phase order while keeping every
environment and storage choice explicit:

``decode once -> camera -> all regions -> release region models ->
all tracking -> shared depth -> target calibration preparation ->
EEF geometry/optional pose -> object and interaction geometry/evidence ->
trajectory/gripper/action composition``.

The region and tracking results are replayed through in-memory adapters when
delegating to the existing public EEF and object runtimes.  Heavy model
inference therefore occurs only in the current phase order and exactly once.
Final artifact publication is disabled by default and writes only to an
explicit output directory when opted in.
"""

from __future__ import annotations

import copy
import errno
import json
import os
import stat
import tempfile
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from dream_exe.artifacts.io import (
    load_traj_assets_manifest,
    write_action_artifact,
    write_gripper_artifact,
    write_trajectory_artifacts,
)
from dream_exe.artifacts.layout import trajectory_artifact_paths
from dream_exe.artifacts.io import (
    publish_depth_artifacts as publish_depth_asset_references,
    publish_pose_fallback as publish_pose_asset_fallback,
)

from ..depth.cache import (
    DEPTH_CACHE_METADATA_FILENAME,
    LEGACY_METADATA_POLICY_REJECT,
    LEGACY_METADATA_POLICY_TRUSTED_PICKLE,
    DepthCacheValidationError,
    DepthPublicationRecoveryRequired,
    build_depth_input_identity,
    build_depth_publication_plan,
    inspect_depth_cache,
    load_rollout_gt_depth_reference,
    publish_artifact_batch,
    read_validated_depth_cache,
)
from ..depth.cache import (
    publish_depth_artifacts as publish_depth_cache_artifacts,
)
from ..depth.media import (
    TARGET_DEPTH_MEDIA_LIMITS,
    render_calibrated_target_depth_media,
    target_depth_float32_bytes_for_shape,
)
from ..depth.runtime_lineage import (
    DEPTH_RUNTIME_LINEAGE_FILENAME,
    EEF_CONSUMED_DEPTH_SAMPLES_FILENAME,
    build_depth_runtime_lineage,
    validate_depth_runtime_lineage,
)
from ..depth.source import build_depth_cache_signature
from ..depth.target_calibration import build_target_calibrated_depth_maps
from .intermediate_artifacts import (
    build_tracking_geometry_artifact_plan,
    write_geometry_artifacts,
    write_interaction_geometry_artifacts,
    write_tracking_artifacts,
)
from ..geometry.prepared import (
    build_eef_trajectory_from_tracks,
    build_visual_geometry_from_tracks,
)
from ..media.video import read_video_frames
from ..pose.artifacts import (
    eef_pose_artifact_path,
    pose_payload_with_artifact_reference,
    write_eef_pose_artifact,
)
from ..pose.estimation import (
    PoseEstimator,
    pose_backend_execution_requested,
)
from ..region.artifacts import (
    build_region_artifact_plan,
    write_region_artifacts,
)
from ..tracking.core import (
    normalize_tracking_backend_identity,
    run_tracking_backend,
    tracking_backend_accepts_write_artifacts,
)
from ..trajectory.stages import (
    compile_task_runtime,
    densify_center_records,
)
from .object import (
    compose_prepared_multi_object_outputs,
    run_prepared_object_evidence,
)
from .single_eef import (
    ArtifactCallback,
    _options,
    _region_value,
    _run_region_selector,
    _validate_tracking,
    run_single_eef_video2traj,
)
from .video_file import (
    VideoReader,
    _depth_options_from_pipeline,
    _geometry_options_from_pipeline,
    _merge_owned_options,
    prepare_explicit_video_file,
)
from .conditioning import (
    align_runtime_conditioning_inputs,
    align_runtime_init_depth_inputs,
)

FinalTrajectoryPublisher = Callable[..., Mapping[str, Any]]


class TargetDepthMediaPublicationError(RuntimeError):
    """A calibrated-target media batch could not be rendered or committed."""


class CanonicalDepthMediaPublicationError(RuntimeError):
    """Canonical depth media could not be rendered or committed atomically."""


class CanonicalDepthMediaPublicationStorageError(CanonicalDepthMediaPublicationError):
    """Canonical depth publication failed because local storage was exhausted."""


_CANONICAL_DEPTH_MEDIA_TYPES = {
    "depth_mp4": "video/mp4",
    "depth_frame0": "image/png",
    "depth_contact": "image/png",
    "depth_vis_meta": "application/json",
}
_MAX_TARGET_PUBLICATION_SOURCE_BYTES = (
    int(TARGET_DEPTH_MEDIA_LIMITS["max_float32_bytes"]) + 1_048_576
)
_MAX_TARGET_PUBLICATION_BATCH_BYTES = 8 * 1_073_741_824


def _bounded_target_depth_array(
    value: Any,
    *,
    label: str,
) -> np.ndarray:
    declared_shape = getattr(value, "shape", None)
    if declared_shape is not None:
        try:
            normalized_shape = tuple(int(size) for size in declared_shape)
        except (TypeError, ValueError) as exc:
            raise TypeError(f"{label}.shape must contain integer dimensions") from exc
        target_depth_float32_bytes_for_shape(
            normalized_shape,
            label=label,
        )
    array = np.asarray(value)
    target_depth_float32_bytes_for_shape(
        tuple(int(size) for size in array.shape),
        label=label,
    )
    if not np.issubdtype(array.dtype, np.number) or np.iscomplexobj(array):
        raise TypeError(f"{label} must contain real numeric values")
    return np.ascontiguousarray(array, dtype=np.float32)


def _absolute_explicit_path(value: Any, *, label: str) -> Path:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be an explicit non-empty path")
    path = Path(text).expanduser()
    if not path.is_absolute():
        raise ValueError(f"{label} must be absolute: {text}")
    return Path(os.path.abspath(path))


def _depth_publication_request(
    value: Mapping[str, Any] | None,
    *,
    write_artifacts: bool,
) -> dict[str, Any] | None:
    """Normalize the narrow, explicit canonical-depth publication seam."""

    if not bool(write_artifacts):
        return None
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError("depth_publication_options must be a mapping")
    options = dict(value)
    supported = {
        "cache_path",
        "meta_path",
        "manifest_path",
        "transaction_root",
        "overwrite",
        "legacy_metadata_policy",
        "trusted_legacy_root",
    }
    unknown = sorted(set(options).difference(supported))
    if unknown:
        raise ValueError(
            "depth_publication_options has unsupported fields: " + ", ".join(unknown)
        )
    overwrite = options.get("overwrite", False)
    if not isinstance(overwrite, bool):
        raise TypeError("depth_publication_options.overwrite must be a bool")
    legacy_metadata_policy = str(
        options.get(
            "legacy_metadata_policy",
            LEGACY_METADATA_POLICY_REJECT,
        )
        or LEGACY_METADATA_POLICY_REJECT
    ).strip()
    if legacy_metadata_policy not in {
        LEGACY_METADATA_POLICY_REJECT,
        LEGACY_METADATA_POLICY_TRUSTED_PICKLE,
    }:
        raise ValueError(
            "depth_publication_options.legacy_metadata_policy must be "
            "'reject' or 'trusted_pickle_read_only'"
        )
    trusted_legacy_root_text = str(options.get("trusted_legacy_root", "") or "").strip()
    if (
        legacy_metadata_policy == LEGACY_METADATA_POLICY_TRUSTED_PICKLE
        and not trusted_legacy_root_text
    ):
        raise ValueError(
            "trusted_pickle_read_only requires an explicit "
            "depth_publication_options.trusted_legacy_root"
        )
    manifest_text = str(options.get("manifest_path", "") or "").strip()
    current_meta_path = _absolute_explicit_path(
        options.get("meta_path", ""),
        label="depth_publication_options.meta_path",
    )
    cache_meta_path = current_meta_path.with_name(DEPTH_CACHE_METADATA_FILENAME)
    if cache_meta_path == current_meta_path:
        raise ValueError(
            "depth_publication_options.meta_path must be the public "
            "current-layout metadata path, not depth_cache_meta.json"
        )
    return {
        "cache_path": _absolute_explicit_path(
            options.get("cache_path", ""),
            label="depth_publication_options.cache_path",
        ),
        "meta_path": current_meta_path,
        "cache_meta_path": cache_meta_path,
        "manifest_path": (
            None
            if not manifest_text
            else _absolute_explicit_path(
                manifest_text,
                label="depth_publication_options.manifest_path",
            )
        ),
        "transaction_root": _absolute_explicit_path(
            options.get("transaction_root", ""),
            label="depth_publication_options.transaction_root",
        ),
        "overwrite": overwrite,
        "legacy_metadata_policy": legacy_metadata_policy,
        "trusted_legacy_root": (
            None
            if not trusted_legacy_root_text
            else _absolute_explicit_path(
                trusted_legacy_root_text,
                label=("depth_publication_options.trusted_legacy_root"),
            )
        ),
    }


def _depth_cache_signature_from_options(
    options: Mapping[str, Any],
) -> str:
    runtime_config = (
        None
        if bool(options.get("use_rollout_gt_depth", False))
        else options.get("depth_runtime_config", None)
    )
    return build_depth_cache_signature(
        depth_model=str(options.get("depth_model", "") or ""),
        depth_config_request=str(options.get("depth_config_request", "") or ""),
        depth_base_cfg=dict(options.get("depth_base_cfg", {}) or {}),
        selected_video=str(options.get("selected_video", "") or ""),
        depth_runtime_config=runtime_config,
    )


def _bind_depth_publication_cache(
    depth_options: Mapping[str, Any],
    *,
    request: Mapping[str, Any] | None,
    frames: Any,
    decode_settings: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Bind and strictly inspect an explicit cache before model execution."""

    resolved = copy.deepcopy(dict(depth_options))
    use_rollout_gt_depth = bool(resolved.get("use_rollout_gt_depth", False))
    rollout_reference: dict[str, Any] | None = None
    referenced_depth: np.ndarray | None = None
    if use_rollout_gt_depth:
        rollout_path_text = str(resolved.get("rollout_gt_depth_path", "") or "").strip()
        if rollout_path_text and Path(rollout_path_text).expanduser().is_file():
            rollout_path = _absolute_explicit_path(
                rollout_path_text,
                label="depth_options.rollout_gt_depth_path",
            )
            referenced_depth, rollout_reference = load_rollout_gt_depth_reference(
                rollout_path
            )
            if resolved.get("rollout_gt_depth", None) is None:
                resolved["rollout_gt_depth"] = referenced_depth
        elif resolved.get("rollout_gt_depth", None) is None:
            if rollout_path_text:
                raise FileNotFoundError(
                    "rollout GT depth file not found: "
                    f"{Path(rollout_path_text).expanduser()}"
                )
            raise ValueError(
                "[Depth] depth.use_rollout_gt_depth=true but "
                "depth.rollout_gt_depth_path is empty."
            )
    input_identity = build_depth_input_identity(
        frames=frames,
        decode_settings=decode_settings,
        depth_mode=("rollout_gt_depth" if use_rollout_gt_depth else "estimated_model"),
        gt_depth=(
            resolved.get("rollout_gt_depth", None) if use_rollout_gt_depth else None
        ),
        calibration_inputs={
            "depth_base_cfg": dict(resolved.get("depth_base_cfg", {}) or {}),
            "calibration_kwargs": dict(resolved.get("calibration_kwargs", {}) or {}),
            "init_ref_depth": resolved.get("init_ref_depth", None),
            "video_target_fps": float(resolved.get("video_target_fps", -1)),
            "video_process_length": int(resolved.get("video_process_length", -1)),
        },
    )
    resolved["expected_cache_input_identity"] = input_identity
    if use_rollout_gt_depth:
        gt_identity = dict(input_identity.get("gt_depth", {}) or {})
        if rollout_reference is not None:
            if str(gt_identity.get("array_fingerprint", "") or "") != str(
                rollout_reference.get(
                    "array_fingerprint",
                    "",
                )
                or ""
            ):
                raise ValueError(
                    "injected rollout GT depth does not match "
                    "depth_options.rollout_gt_depth_path"
                )
            if referenced_depth is not None:
                resolved["rollout_gt_depth"] = referenced_depth
        else:
            rollout_reference = {
                "kind": "in_memory",
                "configured_path": (
                    str(
                        resolved.get(
                            "rollout_gt_depth_path",
                            "",
                        )
                        or ""
                    )
                    or None
                ),
                "shape": list(gt_identity.get("shape", []) or []),
                "dtype": str(gt_identity.get("dtype", "") or ""),
                "array_fingerprint": str(
                    gt_identity.get("array_fingerprint", "") or ""
                ),
            }
        resolved["rollout_gt_depth_reference"] = rollout_reference
        resolved["estimated_depth_cache_path"] = ""
        resolved["estimated_depth_cache_meta_path"] = ""
        resolved.pop("estimated_depth_cache", None)
        resolved.pop("estimated_depth_cache_meta", None)
        return resolved, None
    if request is None:
        return resolved, None
    publication = dict(request)
    publication["input_identity"] = input_identity
    cache_path = Path(publication["cache_path"])
    current_meta_path = Path(publication["meta_path"])
    cache_meta_path = Path(publication["cache_meta_path"])
    trusted_legacy_root = publication.get("trusted_legacy_root", None)
    if publication["legacy_metadata_policy"] == LEGACY_METADATA_POLICY_TRUSTED_PICKLE:
        assert isinstance(trusted_legacy_root, Path)
        for label, path in (
            ("cache_path", cache_path),
            ("meta_path", current_meta_path),
            ("cache_meta_path", cache_meta_path),
        ):
            try:
                path.relative_to(trusted_legacy_root)
            except ValueError as exc:
                raise ValueError(f"{label} must be inside trusted_legacy_root") from exc
    for option_name, expected in (
        ("estimated_depth_cache_path", cache_path),
        (
            "estimated_depth_cache_meta_path",
            current_meta_path,
        ),
    ):
        configured = str(resolved.get(option_name, "") or "").strip()
        if configured:
            actual = _absolute_explicit_path(
                configured,
                label=f"depth_options.{option_name}",
            )
            if actual != expected:
                raise ValueError(
                    f"depth_options.{option_name} must match "
                    f"depth_publication_options: {actual} != {expected}"
                )

    inspection_meta_path = (
        cache_meta_path
        if cache_meta_path.exists() or cache_meta_path.is_symlink()
        else (
            current_meta_path
            if (current_meta_path.exists() or current_meta_path.is_symlink())
            else cache_meta_path
        )
    )
    resolved["estimated_depth_cache_path"] = cache_path.as_posix()
    resolved["estimated_depth_cache_meta_path"] = inspection_meta_path.as_posix()
    publication["inspection_meta_path"] = inspection_meta_path

    expected_signature = _depth_cache_signature_from_options(resolved)
    decoded_shape = tuple(
        int(size) for size in input_identity["decoded_frames"]["shape"]
    )
    inspection = inspect_depth_cache(
        cache_path=cache_path,
        meta_path=inspection_meta_path,
        expected_model_id=str(resolved.get("depth_model", "") or ""),
        expected_shape=decoded_shape,
        expected_frame_count=decoded_shape[0],
        expected_cache_signature=expected_signature,
        expected_input_identity=input_identity,
        legacy_metadata_policy=publication["legacy_metadata_policy"],
    )
    if bool(inspection["valid"]):
        if str(inspection.get("status", "")) == "legacy_valid" and bool(
            publication["overwrite"]
        ):
            resolved["force_recompute_depth"] = True
            publication["inspection"] = inspection
            publication["legacy_upgrade"] = True
            return resolved, publication
        cached_depths, cached_metadata, verified = read_validated_depth_cache(
            cache_path=cache_path,
            meta_path=inspection_meta_path,
            expected_model_id=str(resolved.get("depth_model", "") or ""),
            expected_shape=decoded_shape,
            expected_frame_count=decoded_shape[0],
            expected_cache_signature=expected_signature,
            expected_input_identity=input_identity,
            legacy_metadata_policy=publication["legacy_metadata_policy"],
        )
        resolved["estimated_depth_cache"] = cached_depths
        resolved["estimated_depth_cache_meta"] = cached_metadata
        publication["inspection"] = verified
        publication["legacy_read_only"] = (
            str(verified.get("status", "")) == "legacy_valid"
        )
    elif str(inspection.get("status", "")) == "missing":
        publication["inspection"] = inspection
    elif bool(publication["overwrite"]):
        resolved["force_recompute_depth"] = True
        publication["inspection"] = inspection
    else:
        codes = ", ".join(
            str(issue.get("code", ""))
            for issue in list(inspection.get("issues", []) or [])
            if str(issue.get("code", "") or "")
        )
        raise DepthCacheValidationError(
            "depth cache validation failed before runtime publication: "
            f"{codes or inspection.get('status', 'invalid')}"
        )
    return resolved, publication


def _publish_canonical_depth(
    *,
    output_dir: str | Path,
    depth_payload: Mapping[str, Any],
    depth_options: Mapping[str, Any],
    request: Mapping[str, Any],
    artifact_policy: Mapping[str, Any] | None = None,
    fps: float,
) -> dict[str, Any]:
    """Transactionally publish canonical depth, then advertise it in assets."""

    payload = dict(depth_payload)
    policy = dict(artifact_policy or {})
    save_canonical_mp4 = policy.get("save_canonical_mp4", False)
    if not isinstance(save_canonical_mp4, bool):
        raise TypeError("depth.artifact_policy.save_canonical_mp4 must be a bool")
    info = dict(payload.get("info", {}) or {})
    calibration = info.get("calibration", None)
    if calibration is not None and not isinstance(calibration, Mapping):
        raise TypeError("canonical depth calibration metadata must be a mapping")
    model_diagnostics = info.get("model", None)
    if model_diagnostics is not None and not isinstance(
        model_diagnostics,
        Mapping,
    ):
        raise TypeError("canonical depth model diagnostics must be a mapping")
    model_provenance = info.get("model_provenance", None)
    if model_provenance is None and isinstance(model_diagnostics, Mapping):
        model_provenance = model_diagnostics.get(
            "model_provenance",
            None,
        )
    if model_provenance is not None and not isinstance(
        model_provenance,
        Mapping,
    ):
        raise TypeError("canonical depth model provenance must be a mapping")
    canonical_source = str(info.get("source", payload.get("source", "")) or "")
    if bool(request.get("legacy_read_only", False)):
        if save_canonical_mp4:
            raise ValueError(
                "canonical depth media publication requires current safe "
                "depth metadata; legacy read-only caches are not upgraded"
            )
        assets_path = publish_depth_asset_references(
            output_dir,
            source=canonical_source,
            rollout_gt_depth_path=(
                str(
                    depth_options.get(
                        "rollout_gt_depth_path",
                        "",
                    )
                    or ""
                )
                or None
            ),
            estimated_depth_cache_path=Path(request["cache_path"]).as_posix(),
            depth_npy=Path(request["cache_path"]).as_posix(),
            depth_mp4=None,
            depth_meta_npy=Path(request["meta_path"]).as_posix(),
            depth_model=str(info.get("depth_model", "") or ""),
        )
        return {
            "status": "legacy_reused",
            "dry_run": False,
            "published": [],
            "identity_verified": False,
            "legacy_metadata_policy": request["legacy_metadata_policy"],
            "trusted_legacy_scope": Path(request["trusted_legacy_root"]).as_posix(),
            "cache_path": Path(request["cache_path"]).as_posix(),
            "meta_path": Path(request["meta_path"]).as_posix(),
            "cache_meta_path": None,
            "manifest_path": None,
            "assets_manifest": Path(assets_path).as_posix(),
        }
    output_root = _absolute_explicit_path(
        output_dir,
        label="output_dir",
    )
    output_paths = trajectory_artifact_paths(output_root)
    final_media_paths = {
        "depth_mp4": output_paths["depth_mp4"],
        "depth_frame0": output_paths["depth_frame0_png"],
        "depth_contact": output_paths["depth_contact_png"],
        "depth_vis_meta": output_paths["depth_vis_meta_json"],
    }

    def publish(
        debug_media: Mapping[str, Mapping[str, Any]],
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        plan = build_depth_publication_plan(
            output_root=output_root,
            depths=payload["depths"],
            cache_path=request["cache_path"],
            meta_path=request["cache_meta_path"],
            compat_meta_path=request["meta_path"],
            manifest_path=request.get("manifest_path", None),
            model_id=str(info.get("depth_model", "") or ""),
            source=canonical_source,
            depth_config_source=str(info.get("depth_config_source", "") or ""),
            depth_space=str(info.get("depth_space", "unknown") or "unknown"),
            fps=float(fps),
            cache_signature=str(info.get("cache_signature", "") or ""),
            input_identity=dict(request["input_identity"]),
            calibration=(None if calibration is None else dict(calibration)),
            model_provenance=(
                None if model_provenance is None else dict(model_provenance)
            ),
            debug_media=debug_media,
            overwrite=bool(request["overwrite"]),
        )
        result = publish_depth_cache_artifacts(
            plan,
            depths=payload["depths"],
            debug_media=debug_media,
            transaction_root=request["transaction_root"],
            dry_run=False,
        )
        return dict(result), plan

    if save_canonical_mp4:
        transaction_root = _absolute_explicit_path(
            request["transaction_root"],
            label="depth publication transaction_root",
        )
        try:
            with tempfile.TemporaryDirectory(
                prefix=".dream-exe-canonical-depth-media-source-",
                dir=transaction_root,
            ) as temporary:
                source_root = Path(temporary)
                rendered = render_calibrated_target_depth_media(
                    source_root,
                    target_depths=payload["depths"],
                    fps=float(fps),
                    init_reference_depth=depth_options.get(
                        "init_ref_depth",
                        None,
                    ),
                    final_paths=final_media_paths,
                    stage="base_calibrated",
                )
                rendered_paths = dict(rendered.get("paths", {}) or {})
                if set(rendered_paths) != set(_CANONICAL_DEPTH_MEDIA_TYPES):
                    raise ValueError(
                        "canonical depth media renderer must return exactly "
                        f"{sorted(_CANONICAL_DEPTH_MEDIA_TYPES)!r}"
                    )
                debug_media: dict[str, dict[str, Any]] = {}
                for role, media_type in _CANONICAL_DEPTH_MEDIA_TYPES.items():
                    source_path = _absolute_explicit_path(
                        rendered_paths[role],
                        label=f"canonical depth media {role}",
                    )
                    _bounded_target_staging_source(
                        source_path,
                        source_root=source_root,
                        label=f"canonical depth media {role}",
                    )
                    debug_media[role] = {
                        "source_path": source_path,
                        "destination_path": final_media_paths[role],
                        "media_type": media_type,
                    }
                result, plan = publish(debug_media)
        except DepthPublicationRecoveryRequired:
            raise
        except OSError as exc:
            if exc.errno in {
                errno.ENOSPC,
                getattr(errno, "EDQUOT", None),
            }:
                raise CanonicalDepthMediaPublicationStorageError(
                    "canonical depth media publication storage exhausted "
                    f"(errno={exc.errno})"
                ) from exc
            raise CanonicalDepthMediaPublicationError(
                "canonical depth media publication failed"
            ) from exc
        except Exception as exc:
            raise CanonicalDepthMediaPublicationError(
                "canonical depth media publication failed"
            ) from exc
    else:
        debug_media = {}
        result, plan = publish(debug_media)

    depth_mp4_path = (
        final_media_paths["depth_mp4"].as_posix() if save_canonical_mp4 else None
    )
    assets_path = publish_depth_asset_references(
        output_dir,
        source=canonical_source,
        rollout_gt_depth_path=(
            str(depth_options.get("rollout_gt_depth_path", "") or "") or None
        ),
        estimated_depth_cache_path=Path(request["cache_path"]).as_posix(),
        depth_npy=Path(request["cache_path"]).as_posix(),
        depth_mp4=depth_mp4_path,
        depth_meta_npy=Path(request["meta_path"]).as_posix(),
        depth_model=str(info.get("depth_model", "") or ""),
        depth_cache_meta_json=Path(request["cache_meta_path"]).as_posix(),
        depth_manifest_json=(
            None
            if request.get("manifest_path", None) is None
            else Path(request["manifest_path"]).as_posix()
        ),
        depth_frame0_png=(
            final_media_paths["depth_frame0"].as_posix() if save_canonical_mp4 else None
        ),
        depth_contact_png=(
            final_media_paths["depth_contact"].as_posix()
            if save_canonical_mp4
            else None
        ),
        depth_vis_meta_json=(
            final_media_paths["depth_vis_meta"].as_posix()
            if save_canonical_mp4
            else None
        ),
    )
    return {
        **dict(result),
        "identity_verified": True,
        "legacy_metadata_policy": request["legacy_metadata_policy"],
        "trusted_legacy_scope": (
            None
            if request.get("trusted_legacy_root", None) is None
            else Path(request["trusted_legacy_root"]).as_posix()
        ),
        "cache_path": Path(request["cache_path"]).as_posix(),
        "meta_path": Path(request["meta_path"]).as_posix(),
        "cache_meta_path": Path(request["cache_meta_path"]).as_posix(),
        "manifest_path": (
            None
            if request.get("manifest_path", None) is None
            else Path(request["manifest_path"]).as_posix()
        ),
        "canonical_media": copy.deepcopy(dict(plan.get("debug_media", {}) or {})),
        "canonical_media_paths": (
            {role: path.as_posix() for role, path in final_media_paths.items()}
            if save_canonical_mp4
            else {}
        ),
        "model_provenance_fingerprint": str(
            dict(plan.get("model", {}) or {}).get(
                "provenance_fingerprint",
                "",
            )
            or ""
        ),
        "parameter_fingerprint": str(plan.get("parameter_fingerprint", "") or ""),
        "canonical_publication_source": str(
            dict(plan.get("metadata_payload", {}) or {}).get(
                "source",
                canonical_source,
            )
            or canonical_source
        ),
        "assets_manifest": Path(assets_path).as_posix(),
    }


def _publish_rollout_gt_depth_reference(
    *,
    output_dir: str | Path,
    depth_payload: Mapping[str, Any],
    depth_options: Mapping[str, Any],
    source_identity: Mapping[str, Any],
) -> dict[str, Any]:
    """Publish GT-depth provenance without copying the source array."""

    payload = dict(depth_payload)
    info = dict(payload.get("info", {}) or {})
    canonical_source = str(info.get("source", payload.get("source", "")) or "")
    if canonical_source != "rollout_gt_depth":
        raise ValueError("reference-only depth publication requires rollout_gt_depth")
    if not source_identity:
        raise ValueError(
            "rollout GT depth publication requires a bound source identity"
        )
    source_identity = copy.deepcopy(dict(source_identity))
    raw_diagnostics = source_identity.get("diagnostics", {})
    source_diagnostics = (
        copy.deepcopy(dict(raw_diagnostics))
        if isinstance(raw_diagnostics, Mapping)
        else {}
    )
    source_path = str(
        source_identity.get(
            "path",
            depth_options.get("rollout_gt_depth_path", ""),
        )
        or ""
    )
    assets_path = publish_depth_asset_references(
        output_dir,
        source=canonical_source,
        rollout_gt_depth_path=(source_path or None),
        estimated_depth_cache_path=None,
        depth_npy=None,
        depth_mp4=None,
        depth_meta_npy=None,
        depth_model=str(info.get("depth_model", "") or ""),
        rollout_gt_depth_identity=source_identity,
        rollout_gt_depth_diagnostics=source_diagnostics,
    )
    return {
        "status": "referenced",
        "dry_run": False,
        "published": [],
        "identity_verified": True,
        "legacy_metadata_policy": LEGACY_METADATA_POLICY_REJECT,
        "trusted_legacy_scope": None,
        "source_path": source_path or None,
        "source_identity": source_identity,
        "cache_path": None,
        "meta_path": None,
        "cache_meta_path": None,
        "manifest_path": None,
        "assets_manifest": Path(assets_path).as_posix(),
    }


def _emit(
    callback: ArtifactCallback | None,
    stage: str,
    payload: Any,
) -> None:
    if callback is None:
        return
    callback(str(stage), payload)


def _region_fields(
    result: Any,
    *,
    frame_shape: tuple[int, int],
) -> dict[str, Any]:
    bbox = [
        int(value)
        for value in _region_value(
            result,
            "final_bbox_xyxy",
        )
    ]
    if len(bbox) != 4:
        raise ValueError("region bbox must contain four values")
    query_points = np.asarray(
        _region_value(result, "sampled_points_xy"),
        dtype=np.float32,
    )
    if (
        query_points.ndim != 2
        or query_points.shape[1] != 2
        or query_points.shape[0] <= 0
    ):
        raise ValueError(
            f"region sampled_points_xy must have shape [N,2], got {query_points.shape}"
        )
    mask = np.asarray(
        _region_value(result, "mask"),
        dtype=bool,
    )
    if mask.shape != frame_shape:
        raise ValueError(
            f"region mask/frame mismatch: mask={mask.shape}, frame={frame_shape}"
        )
    try:
        sampling_mask = np.asarray(
            _region_value(result, "sampling_mask"),
            dtype=bool,
        )
    except ValueError:
        sampling_mask = mask
    if sampling_mask.shape != frame_shape:
        raise ValueError(
            "region sampling mask/frame mismatch: "
            f"mask={sampling_mask.shape}, frame={frame_shape}"
        )
    return {
        "result": result,
        "bbox": bbox,
        "query_points": query_points,
        "mask": mask,
        "sampling_mask": sampling_mask,
        "tracking_input": str(
            (
                result.get("tracking_input", "points")
                if isinstance(result, Mapping)
                else getattr(result, "tracking_input", "points")
            )
            or "points"
        ),
    }


def _object_options(
    value: Mapping[str, Any] | None,
    *,
    object_id: str,
    label: str,
    reserved: set[str],
) -> dict[str, Any]:
    options_by_object = dict(value or {})
    raw = options_by_object.get(object_id, {})
    if raw is None:
        raw = {}
    if not isinstance(raw, Mapping):
        raise TypeError(f"{label}[{object_id!r}] must be a mapping")
    options = dict(raw)
    overlap = sorted(reserved.intersection(options))
    if overlap:
        raise ValueError(
            f"{label}[{object_id!r}] cannot replace "
            "runtime-owned inputs: " + ", ".join(overlap)
        )
    return options


def _prepare_region_tracking(
    *,
    frames: Any,
    camera: Any,
    simulator_config: Mapping[str, Any],
    task_runtime: Mapping[str, Any],
    region_runtime: Any,
    tracking_backend: Any,
    eef_num_points: int,
    eef_region_options: Mapping[str, Any],
    eef_tracking_options: Mapping[str, Any] | None,
    object_runtime_options: Mapping[str, Any] | None,
    tracking_output_dir: str | Path,
    write_tracking_artifacts: bool,
    artifact_callback: ArtifactCallback | None,
) -> dict[str, Any]:
    frame_list = list(frames)
    frame_shape = tuple(frame_list[0].shape[:2])
    tracking_write_control: bool | None = None
    if tracking_backend_accepts_write_artifacts(tracking_backend):
        tracking_write_control = bool(write_tracking_artifacts)
    elif bool(write_tracking_artifacts):
        raise TypeError(
            "tracking backend does not accept write_artifacts; "
            "write_tracking_artifacts=True cannot be honored"
        )
    runtime_options = dict(object_runtime_options or {})
    object_streams = [
        copy.deepcopy(dict(stream))
        for stream in list(task_runtime.get("object_stream_plan", []) or [])
    ]
    object_region_options = dict(
        runtime_options.get("region_options_by_object", {}) or {}
    )
    object_tracking_options = dict(
        runtime_options.get("tracking_options_by_object", {}) or {}
    )
    eef_precomputed_region = runtime_options.get(
        "eef_precomputed_region",
        None,
    )
    object_precomputed_regions = dict(
        runtime_options.get("precomputed_regions", {}) or {}
    )
    object_tracking_dirs = dict(runtime_options.get("tracking_output_dirs", {}) or {})
    init_depth = runtime_options.get("init_depth", None)

    eef_region_kwargs = _options(
        eef_region_options,
        label="region_options",
        reserved={
            "target_name",
            "frame_rgb",
            "output_dir",
            "num_points",
            "environment_config",
            "camera",
            "precomputed_region",
            "init_depth",
            "write_artifacts",
        },
    )
    eef_region_kwargs.update(
        {
            "target_name": "eef",
            "frame_rgb": frame_list[0],
            "output_dir": "",
            "num_points": int(eef_num_points),
            "environment_config": dict(simulator_config),
            "camera": camera,
            "precomputed_region": eef_precomputed_region,
            "init_depth": init_depth,
            "write_artifacts": False,
        }
    )
    eef_region = _run_region_selector(
        region_runtime,
        eef_region_kwargs,
    )
    if eef_region is None:
        raise RuntimeError("EEF region selector returned no region")
    eef = _region_fields(
        eef_region,
        frame_shape=frame_shape,
    )
    _emit(artifact_callback, "region.eef", eef_region)

    objects: dict[str, dict[str, Any]] = {}
    for ordinal, stream in enumerate(object_streams):
        object_id = str(stream.get("object_id", "") or "")
        if not object_id:
            raise ValueError("compiled object stream has an empty object_id")
        tracking_config = dict(stream.get("tracking_target_cfg", {}) or {})
        requested_points = int(tracking_config.get("num_points", 50))
        region_options = _object_options(
            object_region_options,
            object_id=object_id,
            label="region_options_by_object",
            reserved={
                "target_name",
                "frame_rgb",
                "output_dir",
                "num_points",
                "environment_config",
                "target_config",
                "precomputed_region",
                "init_depth",
                "camera",
                "seed",
                "write_artifacts",
            },
        )
        region = _run_region_selector(
            region_runtime,
            {
                **region_options,
                "target_name": "obj",
                "frame_rgb": frame_list[0],
                "output_dir": "",
                "num_points": requested_points,
                "environment_config": dict(simulator_config),
                "target_config": dict(stream.get("region_target_cfg", {}) or {}),
                "precomputed_region": object_precomputed_regions.get(
                    object_id,
                    None,
                ),
                "init_depth": init_depth,
                "camera": camera,
                "seed": 7 + ordinal,
                "write_artifacts": False,
            },
        )
        if region is None:
            manipulated = dict(stream.get("manipulated_object", {}) or {})
            raise RuntimeError(
                f"object_id={object_id} "
                f"stages={list(stream.get('stage_ids', []) or [])} "
                "manipulated_object="
                f"'{manipulated.get('name', '') or ''!s}' "
                "resolved to no object region."
            )
        objects[object_id] = {
            "stream": stream,
            **_region_fields(region, frame_shape=frame_shape),
        }
        _emit(
            artifact_callback,
            f"region.object.{object_id}",
            region,
        )

    release_models = getattr(region_runtime, "release_models", None)
    if callable(release_models):
        release_models()

    eef_track_options = _options(
        eef_tracking_options,
        label="tracking_options",
        reserved={
            "video_frames",
            "output_dir",
            "region_bbox_xyxy",
            "num_points",
            "query_points_xy",
            "segmentation_mask",
            "write_artifacts",
        },
    )
    eef_track_options.setdefault(
        "filename",
        "eef_points_cloud",
    )
    eef_track_options.setdefault("seed", 42)
    eef_track_options.setdefault(
        "query_mode",
        eef["tracking_input"],
    )
    eef_track_options.setdefault("grid_size", 0)
    eef_query_mode = (
        str(eef_track_options.get("query_mode", "auto") or "auto").strip().lower()
    )
    eef_tracking_output = run_tracking_backend(
        tracking_backend,
        video_frames=frame_list,
        output_dir=str(tracking_output_dir),
        region_bbox_xyxy=eef["bbox"],
        num_points=int(eef["query_points"].shape[0]),
        query_points_xy=(
            None
            if eef_query_mode in {"bbox_center", "mask_grid"}
            else eef["query_points"]
        ),
        segmentation_mask=eef["sampling_mask"],
        write_artifacts=tracking_write_control,
        **eef_track_options,
    )
    eef_tracks, eef_visibility = _validate_tracking(
        eef_tracking_output.tracks,
        eef_tracking_output.visibility,
        frame_count=len(frame_list),
    )
    eef["tracking"] = {
        "tracks_uv": eef_tracks,
        "visibility": eef_visibility,
        "resolved_query_points_xy": (eef_tracking_output.resolved_query_points_xy),
        "effective_query_mode": (eef_tracking_output.effective_query_mode),
        "provider": dict(eef_tracking_output.provider),
        "query_mode": eef_tracking_output.effective_query_mode,
        "segmentation_mask": eef["sampling_mask"],
    }
    _emit(
        artifact_callback,
        "tracking.eef",
        eef["tracking"],
    )

    for ordinal, stream in enumerate(object_streams):
        object_id = str(stream["object_id"])
        state = objects[object_id]
        options = _object_options(
            object_tracking_options,
            object_id=object_id,
            label="tracking_options_by_object",
            reserved={
                "video_frames",
                "output_dir",
                "region_bbox_xyxy",
                "num_points",
                "filename",
                "seed",
                "query_points_xy",
                "segmentation_mask",
                "query_mode",
                "grid_size",
                "write_artifacts",
            },
        )
        object_query_mode = str(state["tracking_input"] or "auto").strip().lower()
        tracking_output = run_tracking_backend(
            tracking_backend,
            video_frames=frame_list,
            output_dir=str(
                object_tracking_dirs.get(
                    object_id,
                    tracking_output_dir,
                )
            ),
            region_bbox_xyxy=state["bbox"],
            num_points=int(state["query_points"].shape[0]),
            filename=f"{object_id}_points_cloud",
            seed=7 + ordinal,
            query_points_xy=(
                None
                if object_query_mode in {"bbox_center", "mask_grid"}
                else state["query_points"]
            ),
            segmentation_mask=state["sampling_mask"],
            query_mode=state["tracking_input"],
            grid_size=0,
            write_artifacts=tracking_write_control,
            **options,
        )
        tracks, visibility = _validate_tracking(
            tracking_output.tracks,
            tracking_output.visibility,
            frame_count=len(frame_list),
        )
        state["tracking"] = {
            "tracks_uv": tracks,
            "visibility": visibility,
            "resolved_query_points_xy": (tracking_output.resolved_query_points_xy),
            "effective_query_mode": (tracking_output.effective_query_mode),
            "provider": dict(tracking_output.provider),
            "query_mode": tracking_output.effective_query_mode,
            "segmentation_mask": state["sampling_mask"],
        }
        _emit(
            artifact_callback,
            f"tracking.object.{object_id}",
            state["tracking"],
        )

    return {
        "eef": eef,
        "objects": objects,
        "object_order": [str(stream["object_id"]) for stream in object_streams],
    }


class _CachedRegionRuntime:
    def __init__(
        self,
        *,
        eef: Any,
        objects: Mapping[str, Any],
        object_order: list[str],
    ) -> None:
        self.eef = eef
        self.objects = dict(objects)
        self.object_order = list(object_order)
        self.eef_calls = 0
        self.object_calls = 0

    def select_target(self, **kwargs: Any) -> Any:
        target_name = str(kwargs.get("target_name", "") or "")
        if target_name == "eef":
            self.eef_calls += 1
            if self.eef_calls != 1:
                raise RuntimeError("cached EEF region was requested more than once")
            return self.eef
        if target_name != "obj":
            raise ValueError(f"unsupported cached target_name={target_name!r}")
        if self.object_calls >= len(self.object_order):
            raise RuntimeError(
                "cached object region requests exceeded compiled streams"
            )
        object_id = self.object_order[self.object_calls]
        self.object_calls += 1
        return self.objects[object_id]

    def release_models(self) -> None:
        return None


class _CachedTrackingBackend:
    def __init__(
        self,
        *,
        eef: Mapping[str, Any],
        objects: Mapping[str, Mapping[str, Any]],
        object_order: list[str],
    ) -> None:
        self.eef = dict(eef)
        self.objects = {key: dict(value) for key, value in objects.items()}
        self.object_order = list(object_order)
        self.calls = 0
        identity = normalize_tracking_backend_identity(
            **dict(self.eef.get("provider", {}) or {}),
            source="cached EEF tracking provider",
        )
        for object_id, payload in self.objects.items():
            object_identity = normalize_tracking_backend_identity(
                **dict(payload.get("provider", {}) or {}),
                source=f"cached object tracking provider {object_id!r}",
            )
            if object_identity != identity:
                raise ValueError(
                    "cached tracking payloads use conflicting providers: "
                    f"eef={identity!r}, "
                    f"object[{object_id!r}]={object_identity!r}"
                )
        self.provider_kind = identity["provider_kind"]
        self.backend_id = identity["backend_id"]
        self.contract_version = identity["contract_version"]

    def track(self, **kwargs: Any) -> tuple[Any, ...]:
        if self.calls == 0:
            payload = self.eef
        else:
            index = self.calls - 1
            if index >= len(self.object_order):
                raise RuntimeError("cached tracking requests exceeded prepared streams")
            payload = self.objects[self.object_order[index]]
        self.calls += 1
        return (
            payload["tracks_uv"],
            payload["visibility"],
            np.asarray(
                payload["resolved_query_points_xy"],
                dtype=np.float32,
            ),
            str(payload["effective_query_mode"]),
        )


def _depth_roi_mask(
    *,
    normalized_pipeline: Mapping[str, Any],
    eef_mask: np.ndarray,
    object_masks: list[np.ndarray],
) -> np.ndarray | list[np.ndarray]:
    base_config = dict(
        dict(normalized_pipeline.get("depth", {}) or {}).get(
            "base",
            {},
        )
        or {}
    )
    calibration = dict(base_config.get("init_calibration", {}) or {})
    mode = str(calibration.get("calib_roi_mode", "joint") or "joint").strip().lower()
    if mode in {"eef", "eef_only"}:
        return np.asarray(eef_mask, dtype=bool).copy()
    if mode in {"obj", "object", "obj_only", "object_only"}:
        output = np.zeros_like(eef_mask, dtype=bool)
        for mask in object_masks:
            output |= np.asarray(mask, dtype=bool)
        return output
    if mode in {"split", "per_roi", "blend"}:
        return [
            np.asarray(eef_mask, dtype=bool).copy(),
            *[np.asarray(mask, dtype=bool).copy() for mask in object_masks],
        ]
    output = np.asarray(eef_mask, dtype=bool).copy()
    for mask in object_masks:
        output |= np.asarray(mask, dtype=bool)
    return output


def _inject_depth_tracking_context(
    *,
    depth_options: Mapping[str, Any],
    normalized_pipeline: Mapping[str, Any],
    prepared: Mapping[str, Any],
) -> dict[str, Any]:
    options = copy.deepcopy(dict(depth_options))
    eef = dict(prepared["eef"])
    objects = dict(prepared["objects"])
    object_order = list(prepared["object_order"])
    existing = dict(options.get("calibration_kwargs", {}) or {})
    owned = {
        "first_tracks_uv",
        "first_region_masks",
        "eef_tracks_uv",
        "eef_visibility",
        "obj_tracks_uv",
        "obj_visibility",
        "obj_track_groups",
    }
    overlap = sorted(owned.intersection(existing))
    if overlap:
        raise ValueError(
            "depth_options.calibration_kwargs cannot replace "
            "prepared tracking context: " + ", ".join(overlap)
        )
    first_object = dict(objects[object_order[0]]) if object_order else {}
    object_masks = [
        np.asarray(objects[object_id]["sampling_mask"], dtype=bool)
        for object_id in object_order
    ]
    context = {
        "first_tracks_uv": [
            np.asarray(eef["query_points"], dtype=np.float32),
            *[
                np.asarray(
                    objects[object_id]["query_points"],
                    dtype=np.float32,
                )
                for object_id in object_order
            ],
        ],
        "first_region_masks": _depth_roi_mask(
            normalized_pipeline=normalized_pipeline,
            eef_mask=np.asarray(eef["sampling_mask"], dtype=bool),
            object_masks=object_masks,
        ),
        "eef_tracks_uv": eef["tracking"]["tracks_uv"],
        "eef_visibility": eef["tracking"]["visibility"],
        "obj_tracks_uv": (
            first_object.get("tracking", {}).get(
                "tracks_uv",
                None,
            )
        ),
        "obj_visibility": (
            first_object.get("tracking", {}).get(
                "visibility",
                None,
            )
        ),
        "obj_track_groups": [
            {
                "object_id": object_id,
                "stage_ids": list(
                    objects[object_id]["stream"].get(
                        "stage_ids",
                        [],
                    )
                    or []
                ),
                "stage_id": str(
                    objects[object_id]["stream"].get(
                        "owner_stage_id",
                        "",
                    )
                    or ""
                ),
                "tracks_uv": objects[object_id]["tracking"]["tracks_uv"],
                "visibility": objects[object_id]["tracking"]["visibility"],
            }
            for object_id in object_order
        ],
    }
    options["calibration_kwargs"] = {
        **existing,
        **context,
    }
    return options


def _explicit_target_depth_options(
    value: Mapping[str, Any] | None,
) -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise TypeError("target_depth_options must be a mapping")
    options = copy.deepcopy(dict(value))
    supported = {
        "init_ref_depth",
        "canonical_depth_path",
        "interaction_artifact_references_by_object",
    }
    unknown = sorted(set(options).difference(supported))
    if unknown:
        raise ValueError("unsupported target_depth_options: " + ", ".join(unknown))
    references = options.get(
        "interaction_artifact_references_by_object",
        {},
    )
    if references is None:
        references = {}
    if not isinstance(references, Mapping):
        raise TypeError(
            "target_depth_options."
            "interaction_artifact_references_by_object must be a mapping"
        )
    options["interaction_artifact_references_by_object"] = copy.deepcopy(
        dict(references)
    )
    return options


def _target_specifications(
    prepared: Mapping[str, Any],
) -> list[dict[str, Any]]:
    eef = dict(prepared["eef"])
    objects = dict(prepared["objects"])
    object_order = list(prepared["object_order"])
    specifications = [
        {
            "target_name": "eef",
            "kind": "eef",
            "mask0": np.asarray(
                eef["sampling_mask"],
                dtype=bool,
            ),
            "stage_ids": [],
            "runtime_object_key": "",
        }
    ]
    for object_id in object_order:
        state = dict(objects[object_id])
        stream = dict(state["stream"])
        specifications.append(
            {
                "target_name": str(object_id),
                "kind": "object",
                "object_id": str(object_id),
                "safe_object_id": str(
                    stream.get(
                        "safe_object_id",
                        object_id,
                    )
                    or object_id
                ),
                "mask0": np.asarray(
                    state["sampling_mask"],
                    dtype=bool,
                ),
                "stage_ids": list(
                    stream.get(
                        "stage_ids",
                        [],
                    )
                    or []
                ),
                "runtime_object_key": str(
                    stream.get(
                        "runtime_object_key",
                        "",
                    )
                    or ""
                ),
            }
        )
    return specifications


def _dynamic_lift_stage_specifications(
    *,
    prepared: Mapping[str, Any],
    stage_plan: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    objects = dict(prepared["objects"])
    output: list[dict[str, Any]] = []
    for index, raw_stage in enumerate(stage_plan):
        stage = dict(raw_stage)
        object_id = str(stage.get("object_id", "") or "")
        if object_id not in objects:
            raise ValueError(
                "dynamic lift stage references an unprepared object: "
                f"stage={stage.get('stage_id', index)!r}, "
                f"object_id={object_id!r}"
            )
        state = dict(objects[object_id])
        tracking = dict(state["tracking"])
        output.append(
            {
                "stage_id": str(stage.get("stage_id", "") or f"s{index + 1}"),
                "object_id": object_id,
                "runtime_object_key": str(stage.get("runtime_object_key", "") or ""),
                "tracks_uv": tracking["tracks_uv"],
                "visibility": tracking["visibility"],
                "interaction_mask": np.asarray(
                    state["sampling_mask"],
                    dtype=bool,
                ),
            }
        )
    return output


def _prepare_target_lift_depths(
    *,
    depth_payload: Mapping[str, Any],
    normalized_pipeline: Mapping[str, Any],
    prepared: Mapping[str, Any],
    options: Mapping[str, Any],
    publish_target_media_requested: bool,
    artifact_callback: ArtifactCallback | None,
    task_runtime: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    canonical = np.asarray(
        depth_payload["depths"],
        dtype=np.float32,
    )
    depth_source = str(
        depth_payload.get(
            "source",
            "",
        )
        or ""
    )
    target_config = copy.deepcopy(
        dict(
            dict(normalized_pipeline.get("depth", {}) or {}).get(
                "target_calibrated_lift",
                {},
            )
            or {}
        )
    )
    configured = bool(target_config.get("enabled", False))
    state: dict[str, Any] = {
        "configured": configured,
        "attempted": False,
        "applied": False,
        "skip_reason": "",
        "canonical_depths": canonical,
        "canonical_depth_path": str(
            options.get(
                "canonical_depth_path",
                "",
            )
            or ""
        ),
        "maps": {},
        "masks": {},
        "publication_maps": {},
        "publication_masks": {},
        "meta": {},
        "publication_meta": {},
        "meta_path": "",
    }
    if not configured:
        state["skip_reason"] = "disabled"
        return state
    if depth_source == "rollout_gt_depth":
        state["skip_reason"] = "rollout_gt_depth"
        return state

    if "raw_depths" in options:
        raise ValueError(
            "target_depth_options.raw_depths is not supported; raw model "
            "depth must come from the same run_depth_source invocation as "
            "the canonical depth"
        )
    raw_depths = depth_payload.get("raw_model_depths", None)
    if (
        depth_source == "estimated_depth_cache"
        and raw_depths is not None
        and str(target_config.get("source_stage", "raw_model") or "raw_model")
        == "raw_model"
    ):
        raise DepthCacheValidationError(
            "cached canonical depth cannot consume a raw sidecar "
            "without same-transaction lineage in the validated cache "
            "manifest; pathname existence or a caller-supplied array is "
            "not sufficient"
        )
    if (
        bool(publish_target_media_requested)
        and str(target_config.get("source_stage", "raw_model") or "raw_model")
        == "raw_model"
        and raw_depths is None
        and depth_source == "estimated_depth_cache"
    ):
        raise DepthCacheValidationError(
            "target calibration media publication requires raw model depth, "
            "but the canonical depth cache contains no raw source sidecar"
        )
    target_result = build_target_calibrated_depth_maps(
        raw_depths=raw_depths,
        canonical_depths=canonical,
        init_ref_depth=options.get(
            "init_ref_depth",
            None,
        ),
        target_specs=_target_specifications(prepared),
        config=target_config,
        eef_tracks_uv=prepared["eef"]["tracking"]["tracks_uv"],
        eef_visibility=prepared["eef"]["tracking"]["visibility"],
        dynamic_lift_stages=(
            _dynamic_lift_stage_specifications(
                prepared=prepared,
                stage_plan=list(
                    dict(task_runtime or {}).get(
                        "stage_plan",
                        [],
                    )
                    or []
                ),
            )
            if str(
                dict(target_config.get("eef_traj", {}) or {}).get(
                    "mode",
                    "static_eef",
                )
            )
            in {"dynamic_shift", "dynamic_affine"}
            else None
        ),
    )
    state["attempted"] = True
    state["maps"] = {
        str(key): np.asarray(value, dtype=np.float32)
        for key, value in dict(target_result.get("maps", {}) or {}).items()
    }
    raw_publication_maps = dict(target_result.get("publication_maps", {}) or {})
    state["publication_maps"] = {
        str(key): np.asarray(value, dtype=np.float32)
        for key, value in (
            raw_publication_maps.items()
            if raw_publication_maps
            else state["maps"].items()
        )
    }
    state["masks"] = {
        str(key): np.asarray(value, dtype=bool)
        for key, value in dict(target_result.get("masks", {}) or {}).items()
    }
    state["publication_masks"] = {
        key: value.copy() for key, value in state["masks"].items()
    }
    for key, value in state["maps"].items():
        if value.shape != canonical.shape:
            raise ValueError(
                "target-calibrated depth/frame alignment mismatch: "
                f"target={key!r}, depths={value.shape}, "
                f"canonical={canonical.shape}"
            )
    for key, value in state["masks"].items():
        if value.shape != canonical.shape[1:]:
            raise ValueError(
                "target calibration mask/frame alignment mismatch: "
                f"target={key!r}, mask={value.shape}, "
                f"frame={canonical.shape[1:]}"
            )
    state["meta"] = copy.deepcopy(dict(target_result.get("meta", {}) or {}))
    state["publication_meta"] = copy.deepcopy(
        dict(
            target_result.get(
                "publication_meta",
                state["meta"],
            )
            or {}
        )
    )
    state["meta_path"] = str(
        target_result.get(
            "meta_path",
            "",
        )
        or ""
    )
    state["applied"] = any(
        bool(dict(record or {}).get("applied", False))
        for record in dict(state["meta"].get("targets", {}) or {}).values()
    )
    _emit(
        artifact_callback,
        "target_depth",
        {
            "maps": state["maps"],
            "masks": state["masks"],
            "meta": state["meta"],
            "meta_path": state["meta_path"],
        },
    )

    return state


def _interaction_depth_for_object(
    *,
    object_id: str,
    depth_payload: Mapping[str, Any],
    target_state: Mapping[str, Any],
    target_config: Mapping[str, Any],
) -> tuple[np.ndarray, str, str, dict[str, Any]] | None:
    canonical = np.asarray(
        depth_payload["depths"],
        dtype=np.float32,
    )
    depth_source = str(depth_payload.get("source", "") or "")
    canonical_path = str(
        target_state.get(
            "canonical_depth_path",
            "",
        )
        or ""
    )
    gripper_config = dict(
        target_config.get(
            "gripper_traj",
            {},
        )
        or {}
    )
    interaction_enabled = bool(target_config.get("enabled", False)) and bool(
        gripper_config.get("enabled", False)
    )
    if depth_source == "rollout_gt_depth":
        return (
            canonical,
            "rollout_gt_depth",
            canonical_path,
            {
                "enabled": True,
                "source": "rollout_gt_depth",
                "depth_npy": canonical_path,
                "calibration": {
                    "applied": False,
                    "reason": "gt_depth_no_calibration",
                },
            },
        )
    if not interaction_enabled:
        return None

    mode = str(
        gripper_config.get(
            "mode",
            "pair_roi",
        )
        or "pair_roi"
    ).strip()
    maps = dict(target_state.get("maps", {}) or {})
    targets_meta = dict(
        dict(target_state.get("meta", {}) or {}).get(
            "targets",
            {},
        )
        or {}
    )
    if mode == "all_roi":
        pair_key = "gripper_traj_all_roi"
        pair_meta = copy.deepcopy(dict(targets_meta.get(pair_key, {}) or {}))
        pair_depths = maps.get(pair_key, None)
        if pair_depths is not None and not bool(pair_meta.get("fallback", False)):
            return (
                np.asarray(pair_depths, dtype=np.float32),
                "all_roi",
                str(pair_meta.get("depth_npy", "") or ""),
                {
                    "enabled": True,
                    "source": "all_roi",
                    "pair_key": pair_key,
                    "calibration": pair_meta,
                },
            )
        return (
            canonical,
            "canonical_b1_fallback",
            canonical_path,
            {
                "enabled": True,
                "source": "all_roi",
                "pair_key": pair_key,
                "fallback": True,
                "fallback_reason": str(
                    pair_meta.get(
                        "fallback_reason",
                        "all_roi_map_unavailable",
                    )
                    or "all_roi_map_unavailable"
                ),
                "calibration": pair_meta,
            },
        )
    if mode == "pair_roi":
        pair_key = f"pair_{object_id}"
        pair_meta = copy.deepcopy(dict(targets_meta.get(pair_key, {}) or {}))
        pair_depths = maps.get(pair_key, None)
        if pair_depths is not None and not bool(pair_meta.get("fallback", False)):
            return (
                np.asarray(pair_depths, dtype=np.float32),
                "pair_roi",
                str(pair_meta.get("depth_npy", "") or ""),
                {
                    "enabled": True,
                    "source": "pair_roi",
                    "pair_key": pair_key,
                    "calibration": pair_meta,
                },
            )
        return (
            canonical,
            "canonical_b1_fallback",
            canonical_path,
            {
                "enabled": True,
                "source": "pair_roi",
                "pair_key": pair_key,
                "fallback": True,
                "fallback_reason": str(
                    pair_meta.get(
                        "fallback_reason",
                        "pair_map_unavailable",
                    )
                    or "pair_map_unavailable"
                ),
                "calibration": pair_meta,
            },
        )
    return (
        canonical,
        "canonical_b1",
        canonical_path,
        {
            "enabled": True,
            "source": "canonical_b1",
            "depth_npy": canonical_path,
        },
    )


def _trajectory_metadata(
    *,
    uid: str,
    video_path: str | Path,
    simulator_config_source: str,
    simulator_config: Mapping[str, Any],
    normalized_pipeline: Mapping[str, Any],
    prepared: Mapping[str, Any],
    single_eef: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    region_references: Mapping[str, Any] | None = None,
    depth_publication: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    supplied = copy.deepcopy(dict(metadata or {}))
    raw = dict(simulator_config.get("raw", {}) or {})
    raw_eef = dict(raw.get("eef", {}) or {})
    region = prepared["eef"]["result"]
    depth = dict(single_eef.get("depth", {}) or {})
    pose = single_eef.get("pose", None)
    pose_meta = dict(pose.get("meta", {}) or {}) if isinstance(pose, Mapping) else {}
    pipeline_meta = dict(normalized_pipeline.get("_meta", {}) or {})
    pose_config = dict(normalized_pipeline.get("pose", {}) or {})
    region_refs = dict(region_references or {})
    eef_region_ref = dict(region_refs.get("eef", {}) or {})
    publication = dict(depth_publication or {})
    published_depth_meta = str(publication.get("meta_path", "") or "").strip() or None
    base = {
        "uid": str(uid),
        "config_path": str(simulator_config_source or ""),
        "pipeline_config_path": pipeline_meta.get(
            "source",
            "<explicit mapping>",
        ),
        "input_video": str(video_path),
        "align_method": str(
            dict(normalized_pipeline.get("alignment", {}) or {}).get(
                "method", "translation"
            )
            or "translation"
        ),
        "camera_name": str(
            dict(raw.get("camera", {}) or {}).get(
                "name",
                raw.get("render_camera_name", ""),
            )
            or ""
        ),
        "tcp_site_name": str(raw_eef.get("tcp_site_name", "") or ""),
        "controller_ref_site_name": str(
            raw_eef.get("controller_ref_site_name", "") or ""
        ),
        "drive_site_name": str(
            raw_eef.get(
                "drive_site_name",
                raw_eef.get("controller_ref_site_name", ""),
            )
            or ""
        ),
        "eef_region": copy.deepcopy(prepared["eef"]["bbox"]),
        "eef_region_source": copy.deepcopy(
            region.get("source", None)
            if isinstance(region, Mapping)
            else getattr(region, "source", None)
        ),
        "eef_prompt": copy.deepcopy(
            region.get("prompt", None)
            if isinstance(region, Mapping)
            else getattr(region, "prompt", None)
        ),
        "eef_region_json": copy.deepcopy(
            eef_region_ref.get(
                "region_json",
                (
                    region.get("output_json", "")
                    if isinstance(region, Mapping)
                    else getattr(region, "output_json", "")
                ),
            )
        ),
        "eef_tracking_input": prepared["eef"]["tracking_input"],
        "regions_json": region_refs.get("regions_json", None),
        "depth_source": str(depth.get("source", "") or ""),
        "depth_model": str(
            dict(normalized_pipeline.get("depth", {}) or {}).get(
                "model",
                "",
            )
            or ""
        ),
        "depth_config_source": str(
            dict(depth.get("info", {}) or {}).get(
                "depth_config_source",
                "",
            )
            or ""
        ),
        "depth_meta_npy": published_depth_meta,
        "tcp_equals_controller": bool(single_eef.get("same_tcp_and_controller", False)),
        "pose_enabled": bool(pose_config.get("enabled", False)),
        "pose_backend": (
            str(
                pose_meta.get(
                    "backend",
                    pose_config.get("backend", "pointcloud_kabsch"),
                )
                or ""
            )
            if bool(pose_config.get("enabled", False))
            else None
        ),
        "pose_config_source": (
            str(pose_config.get("config_path", "") or "")
            if bool(pose_config.get("enabled", False))
            else None
        ),
        "pose_json_path": (
            str(pose_meta.get("pose_json_path", "") or "") if pose is not None else None
        ),
        "pose_mesh_path": (
            str(pose_config.get("mesh_path", "") or "")
            if bool(pose_config.get("enabled", False))
            else None
        ),
        "pose_weights_root": (
            str(pose_config.get("weights_root", "") or "")
            if bool(pose_config.get("enabled", False))
            else None
        ),
        "gripper_method": None,
    }
    result = {
        **base,
        **supplied,
    }
    result["tcp_equals_controller"] = base["tcp_equals_controller"]
    if published_depth_meta is not None:
        result["depth_meta_npy"] = published_depth_meta
    return result


def _publication_references(
    output_dir: str | Path,
) -> dict[str, str]:
    root = Path(output_dir).expanduser().resolve()
    paths = trajectory_artifact_paths(root)
    return {
        "ee_traj_path": paths["ee_traj"].as_posix(),
        "obj_traj_path": paths["obj_trajs"].as_posix(),
        "gripper_path": paths["gripper"].as_posix(),
    }


def _safe_artifact_name(
    value: Any,
    *,
    fallback: str,
) -> str:
    raw = str(value or "").strip() or str(fallback)
    text = "".join(
        character if character.isalnum() or character in {"_", "-", "."} else "_"
        for character in raw
    )
    normalized = text.strip("_")
    if normalized in {"", ".", ".."}:
        return str(fallback)
    return normalized


def _target_artifact_directory(
    root: Path,
    *,
    key: str,
    record: Mapping[str, Any],
) -> Path:
    kind = str(record.get("kind", "") or "")
    if kind == "eef":
        return root / "depth_eef_traj"
    if kind == "all_roi":
        return root / "depth_gripper_traj" / "all_roi"
    object_name = _safe_artifact_name(
        record.get("object_id", key),
        fallback="object",
    )
    if kind == "pair":
        return root / "depth_gripper_traj" / object_name
    return root / "legacy_object_lift" / object_name


def _strict_target_mapping(
    value: Any,
    *,
    label: str,
    require_records: bool = False,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{label} must be a mapping")
    output: dict[str, Any] = {}
    for key, child in value.items():
        if not isinstance(key, str) or not key.strip():
            raise TypeError(f"{label} keys must be non-empty strings")
        if key in output:
            raise ValueError(f"{label} contains duplicate key {key!r}")
        if require_records and not isinstance(child, Mapping):
            raise TypeError(f"{label}[{key!r}] must be a mapping")
        output[key] = copy.deepcopy(dict(child)) if require_records else child
    return output


def _validate_target_media_destinations(
    *,
    output_root: Path,
    destinations: Mapping[str, str],
) -> None:
    root = _absolute_explicit_path(
        output_root,
        label="target media output_root",
    )
    seen: dict[str, str] = {}
    for owner, path_text in destinations.items():
        path = _absolute_explicit_path(
            path_text,
            label=f"target media destination {owner!r}",
        )
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise ValueError(
                f"target media destinations must remain inside output_root: {path}"
            ) from exc
        current = path
        while True:
            if current.is_symlink():
                raise ValueError(
                    f"target media destinations must not traverse a symlink: {current}"
                )
            if current == current.parent:
                break
            current = current.parent
        if path.exists() and not path.is_file():
            raise ValueError(
                f"target media publication target exists but is not a file: {path}"
            )
        normalized = path.as_posix()
        existing = seen.get(normalized)
        if existing is not None:
            raise ValueError(
                "target calibration media paths must be unique: "
                f"{existing!r} and {owner!r} -> {normalized}"
            )
        seen[normalized] = str(owner)


def _prepare_target_publication_state(
    *,
    output_dir: str | Path,
    target_state: Mapping[str, Any],
    normalized_pipeline: Mapping[str, Any],
    task_runtime: Mapping[str, Any],
    prepared: Mapping[str, Any],
    fps: float = 1.0,
    init_reference_depth: Any = None,
) -> dict[str, Any]:
    state = copy.deepcopy(dict(target_state))
    if not bool(state.get("attempted", False)):
        return state
    output_root = _absolute_explicit_path(
        output_dir,
        label="output_dir",
    )
    root = trajectory_artifact_paths(output_root)["depth_calibration_dir"]
    state["output_root"] = output_root.as_posix()
    eef_meta_path = root / "depth_eef_traj" / "meta.json"
    pair_meta_path = root / "depth_gripper_traj" / "meta.json"
    state["meta_path"] = (root / "target_calibration_meta.json").as_posix()
    meta = copy.deepcopy(dict(state.get("meta", {}) or {}))
    publication_meta = copy.deepcopy(
        dict(
            state.get(
                "publication_meta",
                meta,
            )
            or {}
        )
    )
    meta["depth_eef_traj_meta_json"] = eef_meta_path.as_posix()
    meta["depth_gripper_traj_meta_json"] = pair_meta_path.as_posix()
    publication_meta["depth_eef_traj_meta_json"] = eef_meta_path.as_posix()
    publication_meta["depth_gripper_traj_meta_json"] = pair_meta_path.as_posix()

    artifact_policy = dict(
        dict(normalized_pipeline.get("depth", {}) or {}).get(
            "artifact_policy",
            {},
        )
        or {}
    )
    save_depth_maps = bool(
        artifact_policy.get(
            "save_target_depth_npys",
            False,
        )
    )
    targets = _strict_target_mapping(
        meta.get("targets", {}),
        label="target_depth_state.meta.targets",
        require_records=True,
    )
    publication_targets = _strict_target_mapping(
        publication_meta.get("targets", {}),
        label="target_depth_state.publication_meta.targets",
        require_records=True,
    )
    if set(publication_targets) != set(targets):
        raise ValueError(
            "target depth metadata and publication metadata targets must match"
        )
    publication_maps = _strict_target_mapping(
        state.get(
            "publication_maps",
            state.get("maps", {}),
        )
        or {},
        label="target_depth_state.publication_maps",
    )
    for key, value in publication_maps.items():
        if key not in targets:
            raise ValueError(f"target depth map has no metadata record: {key!r}")
        publication_maps[key] = _bounded_target_depth_array(
            value,
            label=f"target depth map {key!r}",
        )
    state["publication_maps"] = publication_maps
    if save_depth_maps:
        for key, record_raw in targets.items():
            if key not in publication_maps:
                continue
            record = copy.deepcopy(dict(record_raw or {}))
            target_dir = _target_artifact_directory(
                root,
                key=str(key),
                record=record,
            )
            record["depth_npy"] = (target_dir / "depth.npy").as_posix()
            targets[str(key)] = record
            publication_record = copy.deepcopy(
                dict(publication_targets.get(key, {}) or {})
            )
            publication_record["depth_npy"] = record["depth_npy"]
            publication_targets[str(key)] = publication_record

    save_masks = bool(artifact_policy.get("save_masks", False))
    mask_paths: dict[str, str] = {}
    if save_masks:
        publication_masks = _strict_target_mapping(
            state.get(
                "publication_masks",
                state.get("masks", {}),
            )
            or {},
            label="target_depth_state.publication_masks",
        )
        destination_owners: dict[str, str] = {}
        for key in publication_masks:
            if key not in targets:
                raise ValueError(
                    f"target calibration mask has no metadata record: {key!r}"
                )
            record = copy.deepcopy(dict(targets[key] or {}))
            target_dir = _target_artifact_directory(
                root,
                key=str(key),
                record=record,
            )
            mask_path = (target_dir / "calib_mask0.png").as_posix()
            existing_owner = destination_owners.get(mask_path)
            if existing_owner is not None:
                raise ValueError(
                    "target calibration mask paths must be unique: "
                    f"{existing_owner!r} and {key!r} -> {mask_path}"
                )
            destination_owners[mask_path] = str(key)
            mask_paths[str(key)] = mask_path
            record["calib_mask0"] = mask_path
            targets[str(key)] = record
            publication_record = copy.deepcopy(
                dict(publication_targets.get(key, {}) or {})
            )
            publication_record["calib_mask0"] = mask_path
            publication_targets[str(key)] = publication_record
        state["publication_masks"] = publication_masks

    save_target_depth_mp4s = bool(artifact_policy.get("save_target_depth_mp4s", False))
    target_media_paths: dict[str, dict[str, str]] = {}
    if save_target_depth_mp4s:
        for key in publication_maps:
            record = copy.deepcopy(dict(targets[key]))
            if not bool(record.get("applied", False)):
                continue
            target_dir = _target_artifact_directory(
                root,
                key=key,
                record=record,
            )
            paths = {
                "depth_mp4": (target_dir / "depth.mp4").as_posix(),
                "depth_frame0": (target_dir / "depth_frame0.png").as_posix(),
                "depth_contact": (target_dir / "depth_contact.png").as_posix(),
                "depth_vis_meta": (target_dir / "depth_vis_meta.json").as_posix(),
            }
            target_media_paths[key] = paths
            record["depth_mp4"] = paths["depth_mp4"]
            targets[key] = record
            publication_record = copy.deepcopy(dict(publication_targets[key]))
            publication_record["depth_mp4"] = paths["depth_mp4"]
            publication_targets[key] = publication_record
    state["mask_paths"] = mask_paths
    state["target_media_paths"] = target_media_paths
    state["target_media_fps"] = float(fps)
    # Keep the reference lazy here.  The media renderer performs declared-shape
    # resource preflight before any NumPy conversion or optional video import.
    state["target_media_reference_depth"] = init_reference_depth
    media_destinations = {f"mask:{key}": path for key, path in mask_paths.items()}
    for key, paths in target_media_paths.items():
        for role, path in paths.items():
            media_destinations[f"{role}:{key}"] = path
    _validate_target_media_destinations(
        output_root=output_root,
        destinations=media_destinations,
    )
    meta["targets"] = targets
    publication_meta["targets"] = publication_targets
    state["meta"] = meta
    state["publication_meta"] = publication_meta

    streams_by_stage: dict[str, dict[str, Any]] = {}
    objects = dict(prepared.get("objects", {}) or {})
    for object_id in list(prepared.get("object_order", []) or []):
        stream = copy.deepcopy(dict(objects[object_id]["stream"]))
        for stage_id in list(stream.get("stage_ids", []) or []):
            streams_by_stage[str(stage_id)] = stream
    stage_rows = []
    stage_plan = list(task_runtime.get("stage_plan", []) or [])
    for index, raw_stage in enumerate(stage_plan):
        stage = copy.deepcopy(dict(raw_stage or {}))
        stage_id = str(
            stage.get(
                "stage_id",
                "",
            )
            or f"s{index + 1}"
        )
        stream = streams_by_stage.get(stage_id, {})
        stage_rows.append(
            {
                "stage_id": stage_id,
                "stage_index": int(
                    stage.get(
                        "stage_index",
                        index,
                    )
                    or index
                ),
                "object_id": str(
                    stage.get(
                        "object_id",
                        stream.get("object_id", ""),
                    )
                    or ""
                ),
                "runtime_object_key": str(
                    stage.get(
                        "runtime_object_key",
                        stream.get(
                            "runtime_object_key",
                            "",
                        ),
                    )
                    or ""
                ),
                "stage_ids_for_object": list(
                    stream.get(
                        "stage_ids",
                        [],
                    )
                    or []
                ),
                "source": "depth_local_annotated",
                "tracking_npz": "",
            }
        )
    state["stage_info"] = {
        "path": (root / "stage_info" / "depth_stage_info.json").as_posix(),
        "payload": {
            "stage_info_source": "depth_local_annotated",
            "expected_stage_count": len(stage_plan),
            "stages": stage_rows,
        },
    }
    return state


def _attach_target_runtime_lineage(
    *,
    target_state: Mapping[str, Any],
    depth_payload: Mapping[str, Any],
    lift_depth_payload: Mapping[str, Any],
    eef_geometry: Mapping[str, Any],
    prepared: Mapping[str, Any],
    task_runtime: Mapping[str, Any],
    init_reference_depth: Any,
    depth_publication: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Attach additive, path-explicit evidence for the consumed EEF depth."""

    state = dict(target_state)
    if not bool(state.get("attempted", False)):
        return state
    output_root = _absolute_explicit_path(
        state.get("output_root", ""),
        label="target_depth_state.output_root",
    )
    depth_info = dict(depth_payload.get("info", {}) or {})
    publication = dict(depth_publication or {})
    runtime_meta = dict(state.get("meta", {}) or {})
    dynamic_stages = (
        _dynamic_lift_stage_specifications(
            prepared=prepared,
            stage_plan=list(task_runtime.get("stage_plan", []) or []),
        )
        if "dynamic_lift" in runtime_meta
        else []
    )
    array_bindings = {
        "raw_model_aligned": depth_payload.get("raw_model_depths", None),
        "canonical_depth": depth_payload.get("depths", None),
        "init_reference_depth": init_reference_depth,
        "static_eef_depth": dict(state.get("publication_maps", {}) or {}).get(
            "eef", None
        ),
        "runtime_eef_depth": lift_depth_payload.get("depths", None),
    }
    input_bindings = {
        "eef_tracks_uv": dict(prepared["eef"])["tracking"]["tracks_uv"],
        "eef_visibility": dict(prepared["eef"])["tracking"]["visibility"],
        "dynamic_lift_stages": dynamic_stages,
    }
    receipt, samples = build_depth_runtime_lineage(
        runtime_source=str(depth_payload.get("source", "") or ""),
        canonical_publication_source=str(
            publication.get(
                "canonical_publication_source",
                depth_payload.get("source", ""),
            )
            or ""
        ),
        depth_model=str(depth_info.get("depth_model", "") or ""),
        model_provenance_fingerprint=(
            str(publication.get("model_provenance_fingerprint", "") or "") or None
        ),
        parameter_fingerprint=(
            str(publication.get("parameter_fingerprint", "") or "") or None
        ),
        raw_model_aligned=array_bindings["raw_model_aligned"],
        canonical_depth=array_bindings["canonical_depth"],
        init_reference_depth=array_bindings["init_reference_depth"],
        static_eef_depth=array_bindings["static_eef_depth"],
        runtime_eef_depth=array_bindings["runtime_eef_depth"],
        runtime_metadata=runtime_meta,
        eef_tracks_uv=input_bindings["eef_tracks_uv"],
        eef_visibility=input_bindings["eef_visibility"],
        dynamic_lift_stages=input_bindings["dynamic_lift_stages"],
        positions_camera=eef_geometry.get("positions_camera", None),
    )
    root = trajectory_artifact_paths(output_root)["depth_calibration_dir"]
    state["runtime_lineage"] = {
        "path": (root / DEPTH_RUNTIME_LINEAGE_FILENAME).as_posix(),
        "samples_path": (root / EEF_CONSUMED_DEPTH_SAMPLES_FILENAME).as_posix(),
        "payload": receipt,
        "samples": samples,
        "array_bindings": array_bindings,
        "input_bindings": input_bindings,
    }
    return state


def _render_calibration_mask_png(
    path: Path,
    mask: np.ndarray,
) -> None:
    """Render one current-compatible single-channel mask on demand."""

    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "target calibration mask PNG publication requires OpenCV; "
            "install dream-exe[video]"
        ) from exc

    image = np.asarray(mask, dtype=np.uint8) * 255
    if image.ndim != 2 or any(int(size) <= 0 for size in image.shape):
        raise ValueError("target calibration masks must be non-empty [H,W] arrays")
    if not cv2.imwrite(path.as_posix(), image):
        raise OSError(f"failed to render target calibration mask PNG: {path}")


def _nearest_existing_directory(path: Path) -> Path:
    current = path
    while not current.exists() and current.parent != current:
        current = current.parent
    if not current.exists() or not current.is_dir():
        raise ValueError(
            "target media publication requires an existing directory "
            f"above output_root: {path}"
        )
    return current


def _write_target_staging_json(path: Path, payload: Any) -> None:
    path.write_text(
        json.dumps(
            payload,
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def _write_target_staging_npy(
    path: Path,
    payload: Any,
    *,
    preserve_dtype: bool = False,
) -> None:
    array = (
        np.asarray(payload) if preserve_dtype else np.asarray(payload, dtype=np.float32)
    )
    np.save(
        path.as_posix(),
        array,
        allow_pickle=False,
    )


def _bounded_target_staging_source(
    path: Path,
    *,
    source_root: Path,
    label: str,
) -> int:
    source = _absolute_explicit_path(path, label=label)
    try:
        source.relative_to(source_root)
    except ValueError as exc:
        raise ValueError(
            f"{label} must remain inside the target staging root: {source}"
        ) from exc
    try:
        file_stat = source.lstat()
    except FileNotFoundError as exc:
        raise RuntimeError(f"{label} was not generated: {source}") from exc
    if not stat.S_ISREG(file_stat.st_mode):
        raise RuntimeError(f"{label} must be a regular file: {source}")
    size = int(file_stat.st_size)
    if size <= 0:
        raise RuntimeError(f"{label} must be non-empty: {source}")
    if size > _MAX_TARGET_PUBLICATION_SOURCE_BYTES:
        raise RuntimeError(
            f"{label} size {size} exceeds {_MAX_TARGET_PUBLICATION_SOURCE_BYTES} bytes"
        )
    return size


def _publish_target_calibration_media(
    state: Mapping[str, Any],
    *,
    structured_json: list[tuple[str, Path, Any]],
    target_npys: list[tuple[str, Path, np.ndarray]],
) -> dict[str, Any]:
    mask_paths = dict(state.get("mask_paths", {}) or {})
    target_media_paths = dict(state.get("target_media_paths", {}) or {})
    masks = _strict_target_mapping(
        state.get(
            "publication_masks",
            state.get("masks", {}),
        )
        or {},
        label="target_depth_state.publication_masks",
    )
    if mask_paths and set(masks) != set(mask_paths):
        raise ValueError("target calibration mask payloads and destinations must match")
    maps = _strict_target_mapping(
        state.get(
            "publication_maps",
            state.get("maps", {}),
        )
        or {},
        label="target_depth_state.publication_maps",
    )
    if not set(target_media_paths).issubset(maps):
        raise ValueError(
            "target calibration media payloads and destinations must match"
        )
    target_records = _strict_target_mapping(
        dict(
            state.get(
                "publication_meta",
                state.get("meta", {}),
            )
            or {}
        ).get("targets", {}),
        label="target_depth_state.publication_meta.targets",
        require_records=True,
    )
    if not set(target_media_paths).issubset(target_records):
        raise ValueError(
            "target calibration media metadata and destinations must match"
        )
    output_root = _absolute_explicit_path(
        state.get("output_root", ""),
        label="target_depth_state.output_root",
    )
    destinations = {f"mask:{key}": str(path) for key, path in mask_paths.items()}
    for key, raw_paths in target_media_paths.items():
        if not isinstance(raw_paths, Mapping):
            raise TypeError(
                f"target_depth_state.target_media_paths[{key!r}] must be a mapping"
            )
        paths = dict(raw_paths)
        expected_roles = {
            "depth_mp4",
            "depth_frame0",
            "depth_contact",
            "depth_vis_meta",
        }
        if set(paths) != expected_roles:
            raise ValueError(
                f"target media paths for {key!r} must contain exactly "
                f"{sorted(expected_roles)!r}"
            )
        for role, path in paths.items():
            destinations[f"{role}:{key}"] = str(path)
    destinations.update(
        {f"npy:{key}": path.as_posix() for key, path, _array in target_npys}
    )
    destinations.update(
        {f"json:{name}": path.as_posix() for name, path, _payload in structured_json}
    )
    _validate_target_media_destinations(
        output_root=output_root,
        destinations=destinations,
    )
    transaction_root = _nearest_existing_directory(output_root.parent)
    with tempfile.TemporaryDirectory(
        prefix=".dream-exe-target-artifacts-source-",
        dir=transaction_root,
    ) as temporary:
        source_root = Path(temporary)
        media: dict[str, dict[str, Any]] = {}
        staged_total_bytes = 0

        def add_source(
            *,
            name: str,
            source_path: Path,
            destination_path: str,
            media_type: str,
        ) -> None:
            nonlocal staged_total_bytes
            size = _bounded_target_staging_source(
                source_path,
                source_root=source_root,
                label=f"target staging source {name!r}",
            )
            staged_total_bytes += size
            if staged_total_bytes > _MAX_TARGET_PUBLICATION_BATCH_BYTES:
                raise RuntimeError(
                    "target publication staged bytes "
                    f"{staged_total_bytes} exceeds "
                    f"{_MAX_TARGET_PUBLICATION_BATCH_BYTES}"
                )
            media[name] = {
                "source_path": source_path.as_posix(),
                "destination_path": str(destination_path),
                "media_type": str(media_type),
            }

        for index, key in enumerate(sorted(masks)):
            if key not in mask_paths:
                continue
            mask = np.asarray(masks[key], dtype=bool)
            source_path = source_root / f"mask-{index:04d}.png"
            try:
                _render_calibration_mask_png(source_path, mask)
            except Exception as exc:
                raise TargetDepthMediaPublicationError(
                    "target calibration mask rendering failed"
                ) from exc
            add_source(
                name=f"mask:{key}",
                source_path=source_path,
                destination_path=str(mask_paths[key]),
                media_type="image/png",
            )
        for index, key in enumerate(sorted(target_media_paths)):
            target_source_root = source_root / f"target-{index:04d}"
            final_paths = {
                role: str(path) for role, path in dict(target_media_paths[key]).items()
            }
            try:
                rendered = render_calibrated_target_depth_media(
                    target_source_root,
                    target_depths=maps[key],
                    fps=float(state.get("target_media_fps", 1.0)),
                    init_reference_depth=state.get(
                        "target_media_reference_depth",
                        None,
                    ),
                    final_paths=final_paths,
                    stage=(
                        "target_calibrated_lift:"
                        + str(
                            target_records[key].get(
                                "target_name",
                                key,
                            )
                            or key
                        )
                    ),
                )
            except Exception as exc:
                raise TargetDepthMediaPublicationError(
                    f"target calibration depth media rendering failed for {key!r}"
                ) from exc
            rendered_paths = dict(rendered.get("paths", {}) or {})
            for role, media_type in (
                ("depth_mp4", "video/mp4"),
                ("depth_frame0", "image/png"),
                ("depth_contact", "image/png"),
                ("depth_vis_meta", "application/json"),
            ):
                source_path = _absolute_explicit_path(
                    rendered_paths.get(role, ""),
                    label=f"rendered target media {key!r}.{role}",
                )
                try:
                    source_path.relative_to(target_source_root)
                except ValueError as exc:
                    raise ValueError(
                        "target media renderer outputs must remain inside "
                        f"their target staging root: {source_path}"
                    ) from exc
                add_source(
                    name=f"{role}:{key}",
                    source_path=source_path,
                    destination_path=final_paths[role],
                    media_type=media_type,
                )

        def stage_json_artifact(
            index: int,
            name: str,
            destination: Path,
            payload: Any,
        ) -> None:
            source_path = source_root / f"json-{index:04d}.json"
            try:
                _write_target_staging_json(source_path, payload)
            except Exception as exc:
                raise TargetDepthMediaPublicationError(
                    "target calibration structured artifact generation failed"
                ) from exc
            add_source(
                name=f"json:{name}",
                source_path=source_path,
                destination_path=destination.as_posix(),
                media_type="application/json",
            )

        primary_json = [
            artifact for artifact in structured_json if artifact[0] != "stage_info"
        ]
        deferred_json = [
            artifact for artifact in structured_json if artifact[0] == "stage_info"
        ]
        for index, (name, destination, payload) in enumerate(primary_json):
            stage_json_artifact(index, name, destination, payload)
        for index, (key, destination, array) in enumerate(target_npys):
            source_path = source_root / f"npy-{index:04d}.npy"
            try:
                _write_target_staging_npy(
                    source_path,
                    array,
                    preserve_dtype=(key == "eef_consumed_depth_samples"),
                )
            except Exception as exc:
                raise TargetDepthMediaPublicationError(
                    "target calibration structured artifact generation failed"
                ) from exc
            add_source(
                name=f"npy:{key}",
                source_path=source_path,
                destination_path=destination.as_posix(),
                media_type="application/x-npy",
            )
        for index, (name, destination, payload) in enumerate(
            deferred_json,
            start=len(primary_json),
        ):
            stage_json_artifact(index, name, destination, payload)

        try:
            result = publish_artifact_batch(
                output_root=output_root,
                debug_media=media,
                transaction_root=transaction_root,
                overwrite=True,
                dry_run=False,
            )
        except DepthPublicationRecoveryRequired:
            raise
        except Exception as exc:
            raise TargetDepthMediaPublicationError(
                "target calibration media publication failed"
            ) from exc
    return {
        **dict(result),
        "paths": copy.deepcopy(mask_paths),
        "target_media_paths": copy.deepcopy(target_media_paths),
        "artifact_paths": {
            "json": [path.as_posix() for _name, path, _payload in structured_json],
            "target_npys": [path.as_posix() for _key, path, _array in target_npys],
        },
        "staged_bytes": int(staged_total_bytes),
        "resource_limits": {
            "max_source_bytes": _MAX_TARGET_PUBLICATION_SOURCE_BYTES,
            "max_batch_bytes": _MAX_TARGET_PUBLICATION_BATCH_BYTES,
        },
        "transaction_scope": "target_calibration_artifacts",
    }


def _write_target_depth_artifacts(
    state: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not bool(state.get("attempted", False)):
        return None
    meta = copy.deepcopy(
        dict(
            state.get(
                "publication_meta",
                state.get("meta", {}),
            )
            or {}
        )
    )
    meta_path = str(state.get("meta_path", "") or "")
    if not meta_path:
        raise ValueError("target depth publication requires an explicit meta_path")
    eef_meta_path = str(
        meta.get(
            "depth_eef_traj_meta_json",
            "",
        )
        or ""
    )
    pair_meta_path = str(
        meta.get(
            "depth_gripper_traj_meta_json",
            "",
        )
        or ""
    )
    stage_info = dict(state.get("stage_info", {}) or {})
    stage_path_text = str(stage_info.get("path", "") or "")
    targets = _strict_target_mapping(
        meta.get("targets", {}),
        label="target_depth_state.publication_meta.targets",
        require_records=True,
    )
    maps = _strict_target_mapping(
        state.get(
            "publication_maps",
            state.get("maps", {}),
        )
        or {},
        label="target_depth_state.publication_maps",
    )
    target_npys: list[tuple[str, Path, np.ndarray]] = []
    for key, record in targets.items():
        path_text = str(record.get("depth_npy", "") or "")
        if not path_text or key not in maps:
            continue
        target_npys.append(
            (
                key,
                _absolute_explicit_path(
                    path_text,
                    label=f"target depth NPY {key!r}",
                ),
                _bounded_target_depth_array(
                    maps[key],
                    label=f"target depth map {key!r}",
                ),
            )
        )

    runtime_lineage_raw = state.get("runtime_lineage", None)
    runtime_lineage: dict[str, Any] | None = None
    runtime_lineage_payload: dict[str, Any] | None = None
    if runtime_lineage_raw is not None:
        if not isinstance(runtime_lineage_raw, Mapping):
            raise TypeError("target_depth_state.runtime_lineage must be a mapping")
        runtime_lineage = dict(runtime_lineage_raw)
        raw_payload = runtime_lineage.get("payload", None)
        if not isinstance(raw_payload, Mapping):
            raise TypeError(
                "target_depth_state.runtime_lineage.payload must be a mapping"
            )
        runtime_lineage_payload = copy.deepcopy(dict(raw_payload))
        samples = validate_depth_runtime_lineage(
            runtime_lineage_payload,
            runtime_lineage.get("samples", None),
            array_bindings=runtime_lineage.get("array_bindings", None),
            input_bindings=runtime_lineage.get("input_bindings", None),
        )
        target_npys.append(
            (
                "eef_consumed_depth_samples",
                _absolute_explicit_path(
                    runtime_lineage.get("samples_path", ""),
                    label="EEF consumed depth samples NPY",
                ),
                samples,
            )
        )

    structured_json: list[tuple[str, Path, Any]] = []
    if eef_meta_path:
        structured_json.append(
            (
                "eef_meta",
                _absolute_explicit_path(
                    eef_meta_path,
                    label="target depth EEF metadata",
                ),
                {
                    "enabled": bool(
                        dict(meta.get("eef_traj", {}) or {}).get(
                            "enabled",
                            False,
                        )
                    ),
                    "mode": str(
                        dict(meta.get("eef_traj", {}) or {}).get(
                            "mode",
                            "static_eef",
                        )
                    ),
                    "target": copy.deepcopy(dict(targets.get("eef", {}) or {})),
                },
            )
        )
    if pair_meta_path:
        structured_json.append(
            (
                "gripper_meta",
                _absolute_explicit_path(
                    pair_meta_path,
                    label="target depth gripper metadata",
                ),
                {
                    "enabled": bool(
                        dict(meta.get("gripper_traj", {}) or {}).get(
                            "enabled",
                            False,
                        )
                    ),
                    "mode": str(
                        dict(meta.get("gripper_traj", {}) or {}).get(
                            "mode",
                            "pair_roi",
                        )
                    ),
                    "targets": {
                        key: copy.deepcopy(value)
                        for key, value in targets.items()
                        if key.startswith("pair_") or key == "gripper_traj_all_roi"
                    },
                },
            )
        )
    if stage_path_text:
        structured_json.append(
            (
                "stage_info",
                _absolute_explicit_path(
                    stage_path_text,
                    label="target depth stage info",
                ),
                stage_info.get("payload", {}),
            )
        )
    structured_json.append(
        (
            "target_meta",
            _absolute_explicit_path(
                meta_path,
                label="target depth metadata",
            ),
            meta,
        )
    )
    if runtime_lineage is not None and runtime_lineage_payload is not None:
        structured_json.append(
            (
                "runtime_lineage",
                _absolute_explicit_path(
                    runtime_lineage.get("path", ""),
                    label="depth runtime lineage receipt",
                ),
                runtime_lineage_payload,
            )
        )

    output_root = _absolute_explicit_path(
        state.get("output_root", ""),
        label="target_depth_state.output_root",
    )
    destinations = {
        f"mask:{key}": str(path)
        for key, path in dict(state.get("mask_paths", {}) or {}).items()
    }
    for key, raw_paths in dict(state.get("target_media_paths", {}) or {}).items():
        if not isinstance(raw_paths, Mapping):
            raise TypeError(
                f"target_depth_state.target_media_paths[{key!r}] must be a mapping"
            )
        for role, path in dict(raw_paths).items():
            destinations[f"{role}:{key}"] = str(path)
    destinations.update(
        {f"npy:{key}": path.as_posix() for key, path, _array in target_npys}
    )
    destinations.update(
        {f"json:{name}": path.as_posix() for name, path, _payload in structured_json}
    )
    _validate_target_media_destinations(
        output_root=output_root,
        destinations=destinations,
    )

    artifact_publication = _publish_target_calibration_media(
        state,
        structured_json=structured_json,
        target_npys=target_npys,
    )
    primary_json_paths = [
        path.as_posix()
        for name, path, _payload in structured_json
        if name != "stage_info"
    ]
    target_npy_paths = [path.as_posix() for _key, path, _array in target_npys]
    stage_info_paths = [
        path.as_posix()
        for name, path, _payload in structured_json
        if name == "stage_info"
    ]
    written = primary_json_paths + target_npy_paths + stage_info_paths
    has_masks = bool(dict(state.get("mask_paths", {}) or {}))
    has_media = bool(has_masks or dict(state.get("target_media_paths", {}) or {}))
    media_publication = artifact_publication if has_media else None
    return {
        "meta_path": meta_path,
        "eef_meta_path": eef_meta_path,
        "gripper_meta_path": pair_meta_path,
        "stage_info_path": stage_path_text,
        "runtime_lineage_path": (
            None
            if runtime_lineage is None
            else str(runtime_lineage.get("path", "") or "")
        ),
        "eef_consumed_depth_samples_path": (
            None
            if runtime_lineage is None
            else str(runtime_lineage.get("samples_path", "") or "")
        ),
        "mask_publication": media_publication if has_masks else None,
        "media_publication": media_publication,
        "artifact_publication": artifact_publication,
        "written": written,
    }


def publish_composed_trajectory_outputs(
    *,
    output_dir: str | Path,
    outputs: Mapping[str, Any],
    eef_pose_path: str | None = None,
    eef_pose_payload: Mapping[str, Any] | None = None,
    eef_pose_manifest: Mapping[str, Any] | None = None,
    eef_pose_fallback: Mapping[str, Any] | None = None,
    region_state: Mapping[str, Any] | None = None,
    diagnostic_state: Mapping[str, Any] | None = None,
    target_depth_state: Mapping[str, Any] | None = None,
    conditioning_alignment: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write current pose/diagnostic/final payload files and assets map."""

    output_text = str(output_dir or "").strip()
    if not output_text:
        raise ValueError("output_dir must be an explicit non-empty path")
    payloads = dict(outputs)
    ee_payload = payloads.get("ee_traj", None)
    if not isinstance(ee_payload, Mapping):
        raise ValueError("outputs.ee_traj must be a mapping")  # noqa: TRY004
    object_payload = payloads.get("obj_traj", None)
    union_payload = payloads.get("union_traj", None)
    region = None
    if region_state is not None:
        if not isinstance(region_state, Mapping):
            raise TypeError("region_state must be a mapping")
        regions = dict(region_state)
        plan = regions.get("plan", None)
        if not isinstance(plan, Mapping):
            raise TypeError("region_state.plan must be a mapping")
        region = write_region_artifacts(
            output_text,
            plan=plan,
            eef_region=regions.get("eef_region", None),
            object_regions=regions.get("object_regions", {}),
            first_frame_rgb=regions.get("first_frame_rgb", None),
            write_visual_media=bool(regions.get("write_visual_media", False)),
        )
    diagnostics = None
    tracking = None
    geometry = None
    interaction_geometry = None
    if diagnostic_state is not None:
        if not isinstance(diagnostic_state, Mapping):
            raise TypeError("diagnostic_state must be a mapping")
        diagnostics = dict(diagnostic_state)
        plan = diagnostics.get("plan", None)
        if not isinstance(plan, Mapping):
            raise TypeError("diagnostic_state.plan must be a mapping")
        tracking = write_tracking_artifacts(
            output_text,
            plan=plan,
            eef_tracking=diagnostics.get(
                "eef_tracking",
                {},
            ),
            object_tracking=diagnostics.get(
                "object_tracking",
                {},
            ),
            video_frames=diagnostics.get(
                "video_frames",
                None,
            ),
            write_visual_media=bool(
                diagnostics.get(
                    "write_visual_media",
                    False,
                )
            ),
            media_options=diagnostics.get(
                "media_options",
                None,
            ),
        )
    if eef_pose_payload is not None and eef_pose_fallback is not None:
        raise ValueError(
            "eef_pose_payload and eef_pose_fallback are mutually exclusive"
        )
    pose = None
    if eef_pose_payload is not None:
        expected_pose_path = eef_pose_artifact_path(output_text)
        supplied_pose_path = str(eef_pose_path or "").strip()
        if (
            supplied_pose_path
            and Path(supplied_pose_path).expanduser().resolve().as_posix()
            != expected_pose_path
        ):
            raise ValueError(
                "eef_pose_path must use the explicit output_dir pose sidecar "
                "when eef_pose_payload is published"
            )
        pose = write_eef_pose_artifact(
            output_text,
            eef_pose_payload,
            manifest_payload=eef_pose_manifest,
        )
        eef_pose_path = str(pose["path"])
    elif eef_pose_fallback is not None:
        fallback = dict(eef_pose_fallback)
        reason = str(fallback.get("reason", "") or "").strip()
        if not reason:
            raise ValueError("eef_pose_fallback.reason must be non-empty")
        pose = {
            "fallback": "position_only",
            "reason": reason,
            "manifest_path": publish_pose_asset_fallback(
                output_text,
                pose_config_source=(
                    str(fallback.get("pose_config_source", "") or "") or None
                ),
                mesh_path=(str(fallback.get("mesh_path", "") or "") or None),
                register_mask_source=(
                    str(fallback.get("register_mask_source", "") or "") or None
                ),
                reason=reason,
            ),
        }
    target_depth = _write_target_depth_artifacts(dict(target_depth_state or {}))
    if diagnostics is not None:
        raw_interaction_geometry = diagnostics.get(
            "interaction_geometry",
            {},
        )
        if raw_interaction_geometry is None:
            raw_interaction_geometry = {}
        if not isinstance(raw_interaction_geometry, Mapping):
            raise TypeError("diagnostic_state.interaction_geometry must be a mapping")
        if raw_interaction_geometry:
            interaction_geometry = write_interaction_geometry_artifacts(
                output_text,
                plan=diagnostics["plan"],
                interaction_geometry=dict(raw_interaction_geometry),
            )
        geometry = write_geometry_artifacts(
            output_text,
            plan=diagnostics["plan"],
            eef_geometry=diagnostics.get(
                "eef_geometry",
                {},
            ),
            object_geometry=diagnostics.get(
                "object_geometry",
                {},
            ),
            depth_references=diagnostics.get(
                "depth_references",
                {},
            ),
            manifest_payload=diagnostics.get(
                "geometry_manifest",
                {},
            ),
        )
    trajectory = write_trajectory_artifacts(
        output_text,
        copy.deepcopy(dict(ee_payload)),
        object_payload=(
            copy.deepcopy(dict(object_payload))
            if isinstance(object_payload, Mapping)
            else None
        ),
        union_payload=(
            copy.deepcopy(dict(union_payload))
            if isinstance(union_payload, Mapping)
            else None
        ),
        eef_pose_path=eef_pose_path,
        manifest_payload=(
            {"conditioning_alignment": copy.deepcopy(conditioning_alignment)}
            if conditioning_alignment is not None
            else None
        ),
    )
    publication_order = []
    if region is not None:
        publication_order.append("region")
    if tracking is not None:
        publication_order.append("tracking")
    if pose is not None:
        publication_order.append("pose")
    if target_depth is not None:
        publication_order.append("target_depth")
    if geometry is not None:
        publication_order.append("geometry")
    if interaction_geometry is not None:
        publication_order.append("interaction_geometry")
    publication_order.append("trajectory")
    gripper = None
    if isinstance(payloads.get("gripper", None), Mapping):
        gripper = write_gripper_artifact(
            output_text,
            copy.deepcopy(dict(payloads["gripper"])),
        )
        publication_order.append("gripper")
    action = None
    if isinstance(payloads.get("action", None), Mapping):
        action = write_action_artifact(
            output_text,
            copy.deepcopy(dict(payloads["action"])),
        )
        publication_order.append("action")
    return {
        "output_dir": Path(output_text).as_posix(),
        "region": region,
        "tracking": tracking,
        "pose": pose,
        "geometry": geometry,
        "interaction_geometry": interaction_geometry,
        "trajectory": trajectory,
        "target_depth": target_depth,
        "gripper": gripper,
        "action": action,
        "assets_manifest": load_traj_assets_manifest(output_text),
        "publication_order": publication_order,
    }


def run_multi_object_video_file(
    *,
    video_path: str | Path,
    simulator_config: Mapping[str, Any],
    pipeline_config: Mapping[str, Any],
    region_runtime: Any,
    tracking_backend: Any,
    uid: str = "",
    simulator_config_source: str = "",
    depth_estimator: Any = None,
    depth_calibration: Any = None,
    pose_estimator: PoseEstimator | None = None,
    pose_backend: Any = None,
    rigid_pose_backend: Any = None,
    region_options: Mapping[str, Any] | None = None,
    tracking_options: Mapping[str, Any] | None = None,
    depth_options: Mapping[str, Any] | None = None,
    target_depth_options: Mapping[str, Any] | None = None,
    geometry_options: Mapping[str, Any] | None = None,
    projection_options: Mapping[str, Any] | None = None,
    pose_options: Mapping[str, Any] | None = None,
    pose_register_mask: Any = None,
    pose_register_mask_source: str = "",
    conditioning_transform: Mapping[str, Any] | None = None,
    object_runtime_options: Mapping[str, Any] | None = None,
    composition_options: Mapping[str, Any] | None = None,
    video_reader: VideoReader = read_video_frames,
    video_backend: str = "auto",
    video_reader_options: Mapping[str, Any] | None = None,
    tracking_output_dir: str | Path = "",
    write_tracking_artifacts: bool = False,
    write_artifacts: bool = False,
    output_dir: str | Path | None = None,
    depth_publication_options: Mapping[str, Any] | None = None,
    publisher: FinalTrajectoryPublisher | None = None,
    metadata: Mapping[str, Any] | None = None,
    artifact_callback: ArtifactCallback | None = None,
) -> dict[str, Any]:
    """Decode once and produce prepared multi-object final outputs.

    No benchmark identity, run key, model, simulator, or output path is
    discovered. ``write_artifacts`` is the final-publication gate.
    """

    if publisher is not None and not bool(write_artifacts):
        raise ValueError("publisher requires write_artifacts=True")
    if bool(write_artifacts) and not str(output_dir or "").strip():
        raise ValueError("write_artifacts=True requires an explicit output_dir")
    if bool(write_tracking_artifacts):
        raise ValueError("tracking visual previews are not part of the public runtime")
    depth_publication_request = _depth_publication_request(
        depth_publication_options,
        write_artifacts=bool(write_artifacts),
    )
    explicit_pose_mask_source = str(pose_register_mask_source or "").strip()
    if pose_register_mask is not None and not explicit_pose_mask_source:
        raise ValueError(
            "pose_register_mask_source is required with pose_register_mask"
        )
    if pose_register_mask is None and explicit_pose_mask_source:
        raise ValueError("pose_register_mask_source requires pose_register_mask")
    resolved_target_depth_options = _explicit_target_depth_options(target_depth_options)
    prepared_video = prepare_explicit_video_file(
        video_path=video_path,
        simulator_config=simulator_config,
        pipeline_config=pipeline_config,
        video_reader=video_reader,
        video_backend=video_backend,
        video_reader_options=video_reader_options,
        artifact_callback=artifact_callback,
    )
    frames = prepared_video["frames"]
    camera = prepared_video["camera"]
    conditioning_alignment: dict[str, Any] | None = None
    if conditioning_transform is not None:
        aligned_conditioning = align_runtime_conditioning_inputs(
            transform=conditioning_transform,
            target_shape_hw=tuple(frames[0].shape[:2]),
            depth_options=depth_options,
            target_depth_options=resolved_target_depth_options,
            object_runtime_options=object_runtime_options,
        )
        depth_options = aligned_conditioning["depth_options"]
        resolved_target_depth_options = aligned_conditioning["target_depth_options"]
        object_runtime_options = aligned_conditioning["object_runtime_options"]
        conditioning_alignment = dict(aligned_conditioning["manifest"])
    else:
        aligned_init_depth = align_runtime_init_depth_inputs(
            target_shape_hw=tuple(frames[0].shape[:2]),
            depth_options=depth_options,
            target_depth_options=resolved_target_depth_options,
            object_runtime_options=object_runtime_options,
        )
        depth_options = aligned_init_depth["depth_options"]
        resolved_target_depth_options = aligned_init_depth["target_depth_options"]
        object_runtime_options = aligned_init_depth["object_runtime_options"]
        if aligned_init_depth["manifest"]["consumers"]:
            conditioning_alignment = dict(aligned_init_depth["manifest"])
    if pose_register_mask is not None:
        explicit_pose_mask = np.asarray(
            pose_register_mask,
            dtype=bool,
        )
        expected_pose_mask_shape = tuple(frames[0].shape[:2])
        if explicit_pose_mask.shape != expected_pose_mask_shape:
            raise ValueError(
                "pose register mask/frame mismatch: "
                f"mask={explicit_pose_mask.shape}, "
                f"frame={expected_pose_mask_shape}"
            )
        if not np.any(explicit_pose_mask):
            raise ValueError("pose_register_mask must contain at least one pixel")
        pose_register_mask = explicit_pose_mask
    normalized = dict(prepared_video["pipeline_config"])
    artifact_policy = dict(
        dict(normalized.get("depth", {}) or {}).get(
            "artifact_policy",
            {},
        )
        or {}
    )
    publish_masks_requested = bool(
        write_artifacts and artifact_policy.get("save_masks", False)
    )
    publish_target_depth_mp4s_requested = bool(
        write_artifacts and artifact_policy.get("save_target_depth_mp4s", False)
    )
    publish_target_media_requested = bool(
        publish_masks_requested or publish_target_depth_mp4s_requested
    )
    video_settings = dict(prepared_video["video_settings"])
    decoded_fps = float(prepared_video["decoded_fps"])
    metadata_payload = copy.deepcopy(dict(metadata or {}))
    task_runtime = compile_task_runtime(
        uid=str(uid),
        metadata=metadata_payload,
        pipeline_config=normalized,
    )
    region_plan = (
        build_region_artifact_plan(
            output_dir,
            object_stream_plan=list(task_runtime.get("object_stream_plan", []) or []),
        )
        if bool(write_artifacts)
        else None
    )
    diagnostic_plan = (
        build_tracking_geometry_artifact_plan(
            output_dir,
            object_stream_plan=list(task_runtime.get("object_stream_plan", []) or []),
            include_visual_media=False,
        )
        if bool(write_artifacts)
        else None
    )

    region_config = dict(normalized.get("region", {}) or {})
    eef_region_config = dict(
        dict(region_config.get("targets", {}) or {}).get(
            "eef",
            {},
        )
        or {}
    )
    resolved_region_options = _merge_owned_options(
        region_options,
        {"target_config": eef_region_config},
        label="region_options",
    )
    tracking_config = dict(normalized.get("tracking", {}) or {})
    eef_tracking_config = dict(
        dict(tracking_config.get("targets", {}) or {}).get(
            "eef",
            {},
        )
        or {}
    )
    eef_num_points = int(eef_tracking_config.get("num_points", 150))
    resolved_depth_options = _merge_owned_options(
        depth_options,
        _depth_options_from_pipeline(
            normalized,
            video_settings=video_settings,
        ),
        label="depth_options",
    )
    resolved_geometry_options = _merge_owned_options(
        geometry_options,
        _geometry_options_from_pipeline(
            normalized,
            simulator_config,
        ),
        label="geometry_options",
    )
    alignment = dict(normalized.get("alignment", {}) or {})
    resolved_projection_options = _merge_owned_options(
        projection_options,
        {
            "projection_method": str(
                alignment.get("method", "translation") or "translation"
            ),
            "smooth_alpha": -1,
        },
        label="projection_options",
    )

    prepared_region_tracking = _prepare_region_tracking(
        frames=frames,
        camera=camera,
        simulator_config=simulator_config,
        task_runtime=task_runtime,
        region_runtime=region_runtime,
        tracking_backend=tracking_backend,
        eef_num_points=eef_num_points,
        eef_region_options=resolved_region_options,
        eef_tracking_options=tracking_options,
        object_runtime_options=object_runtime_options,
        tracking_output_dir=tracking_output_dir,
        write_tracking_artifacts=write_tracking_artifacts,
        artifact_callback=artifact_callback,
    )
    resolved_depth_options = _inject_depth_tracking_context(
        depth_options=resolved_depth_options,
        normalized_pipeline=normalized,
        prepared=prepared_region_tracking,
    )
    (
        resolved_depth_options,
        depth_publication_request,
    ) = _bind_depth_publication_cache(
        resolved_depth_options,
        request=depth_publication_request,
        frames=frames,
        decode_settings={
            "backend": str(video_backend or "auto"),
            "decoded_fps": float(decoded_fps),
            "read_settings": dict(video_settings),
            "reader_options": dict(video_reader_options or {}),
        },
    )
    rollout_gt_depth_reference = resolved_depth_options.pop(
        "rollout_gt_depth_reference",
        None,
    )
    cached_region = _CachedRegionRuntime(
        eef=prepared_region_tracking["eef"]["result"],
        objects={
            object_id: state["result"]
            for object_id, state in dict(prepared_region_tracking["objects"]).items()
        },
        object_order=list(prepared_region_tracking["object_order"]),
    )
    cached_tracking = _CachedTrackingBackend(
        eef=prepared_region_tracking["eef"]["tracking"],
        objects={
            object_id: state["tracking"]
            for object_id, state in dict(prepared_region_tracking["objects"]).items()
        },
        object_order=list(prepared_region_tracking["object_order"]),
    )

    pose_config = dict(normalized.get("pose", {}) or {})
    pose_enabled = bool(pose_config.get("enabled", False))
    resolved_pose_estimator = None
    resolved_pose_backend = None
    resolved_rigid_pose_backend = None
    resolved_pose_config = None
    if pose_enabled:
        if (
            pose_estimator is None
            and str(pose_config.get("config_path", "") or "").strip()
        ):
            raise ValueError(
                "pipeline pose.config_path is not resolved by this "
                "callable; inject a configured pose_estimator"
            )
        backend_free_config = {
            key: value
            for key, value in pose_config.items()
            if key not in {"enabled", "config_path"}
        }
        if pose_estimator is None and pose_backend is None:
            requires_backend = pose_backend_execution_requested(backend_free_config)
            supports_backend_free_pose = bool(
                backend_free_config.get(
                    "kabsch_enabled",
                    False,
                )
                or backend_free_config.get(
                    "anchor_rotation_constraint_enabled",
                    False,
                )
                or backend_free_config.get(
                    "use_config_init_pose",
                    False,
                )
            )
            if requires_backend or not supports_backend_free_pose:
                raise RuntimeError(
                    "pipeline pose.enabled=true requires an injected "
                    "pose_estimator or pose_backend unless its explicit "
                    "policy is backend-free Kabsch/config pose"
                )
        resolved_pose_estimator = pose_estimator
        resolved_pose_backend = pose_backend
        resolved_rigid_pose_backend = rigid_pose_backend
        if pose_estimator is None:
            resolved_pose_config = backend_free_config
    elif rigid_pose_backend is not None:
        raise ValueError("rigid_pose_backend requires pipeline pose.enabled=true")
    runtime_device = str(
        dict(normalized.get("runtime", {}) or {}).get(
            "device",
            "cuda",
        )
        or "cuda"
    )
    target_config = copy.deepcopy(
        dict(
            dict(normalized.get("depth", {}) or {}).get(
                "target_calibrated_lift",
                {},
            )
            or {}
        )
    )
    target_state_holder: dict[str, Any] = {}

    def lift_depth_transform(
        depth_payload: Mapping[str, Any],
    ) -> dict[str, Any]:
        target_state = _prepare_target_lift_depths(
            depth_payload=depth_payload,
            normalized_pipeline=normalized,
            prepared=prepared_region_tracking,
            task_runtime=task_runtime,
            options=resolved_target_depth_options,
            publish_target_media_requested=(publish_target_media_requested),
            artifact_callback=artifact_callback,
        )
        target_state_holder["value"] = target_state
        eef_meta = copy.deepcopy(
            dict(
                dict(target_state.get("meta", {}) or {})
                .get("targets", {})
                .get("eef", {})
                or {}
            )
        )
        eef_depths = np.asarray(
            dict(target_state.get("maps", {}) or {}).get(
                "eef",
                depth_payload["depths"],
            ),
            dtype=np.float32,
        )
        return {
            "depths": eef_depths,
            "source": str(
                eef_meta.get(
                    "depth_source",
                    "canonical",
                )
                or "canonical"
            ),
            "metadata": eef_meta,
        }

    def single_callback(stage: str, payload: Any) -> None:
        if stage in {"region", "tracking", "trajectory"}:
            return
        mapped = "geometry.eef" if stage == "geometry" else stage
        _emit(artifact_callback, mapped, payload)

    single_eef = run_single_eef_video2traj(
        frames=frames,
        camera=camera,
        simulator_config=simulator_config,
        region_runtime=cached_region,
        tracking_backend=cached_tracking,
        target_fps=int(decoded_fps),
        num_points=eef_num_points,
        region_options=resolved_region_options,
        write_region_artifacts=False,
        tracking_options=tracking_options,
        depth_estimator=depth_estimator,
        depth_calibration=depth_calibration,
        depth_options=resolved_depth_options,
        lift_depth_transform=(
            lift_depth_transform if bool(target_config.get("enabled", False)) else None
        ),
        geometry_options=resolved_geometry_options,
        projection_options=resolved_projection_options,
        pose_estimator=resolved_pose_estimator,
        pose_backend=resolved_pose_backend,
        rigid_pose_backend=resolved_rigid_pose_backend,
        pose_config=resolved_pose_config,
        pose_device=runtime_device,
        pose_options=pose_options,
        pose_register_mask=pose_register_mask,
        pose_register_mask_source=pose_register_mask_source,
        metadata=metadata_payload,
        artifact_callback=single_callback,
    )
    depth_publication = None
    if (
        bool(write_artifacts)
        and str(
            single_eef["depth"].get(
                "source",
                "",
            )
            or ""
        )
        == "rollout_gt_depth"
    ):
        depth_publication = _publish_rollout_gt_depth_reference(
            output_dir=output_dir,
            depth_payload=single_eef["depth"],
            depth_options=resolved_depth_options,
            source_identity=(
                rollout_gt_depth_reference
                if isinstance(
                    rollout_gt_depth_reference,
                    Mapping,
                )
                else {}
            ),
        )
        _emit(
            artifact_callback,
            "depth.publication",
            depth_publication,
        )
    elif depth_publication_request is not None:
        depth_publication = _publish_canonical_depth(
            output_dir=output_dir,
            depth_payload=single_eef["depth"],
            depth_options=resolved_depth_options,
            request=depth_publication_request,
            artifact_policy=artifact_policy,
            fps=decoded_fps,
        )
        _emit(
            artifact_callback,
            "depth.publication",
            depth_publication,
        )
    pose_publication_path = None
    if bool(write_artifacts) and isinstance(
        single_eef.get("pose", None),
        Mapping,
    ):
        single_eef["pose"] = pose_payload_with_artifact_reference(
            single_eef["pose"],
            output_dir=output_dir,
        )
        pose_publication_path = str(single_eef["pose"]["meta"]["pose_json_path"])
    single_eef["trajectory"]["meta"] = _trajectory_metadata(
        uid=str(uid),
        video_path=video_path,
        simulator_config_source=simulator_config_source,
        simulator_config=simulator_config,
        normalized_pipeline=normalized,
        prepared=prepared_region_tracking,
        single_eef=single_eef,
        metadata=metadata_payload,
        region_references=region_plan,
        depth_publication=depth_publication,
    )
    if "value" in target_state_holder:
        target_state = dict(target_state_holder["value"])
    else:
        target_state = _prepare_target_lift_depths(
            depth_payload=single_eef["depth"],
            normalized_pipeline=normalized,
            prepared=prepared_region_tracking,
            task_runtime=task_runtime,
            options=resolved_target_depth_options,
            publish_target_media_requested=(publish_target_media_requested),
            artifact_callback=artifact_callback,
        )
    if bool(write_artifacts):
        target_state = _prepare_target_publication_state(
            output_dir=output_dir,
            target_state=target_state,
            normalized_pipeline=normalized,
            task_runtime=task_runtime,
            prepared=prepared_region_tracking,
            fps=decoded_fps,
            init_reference_depth=resolved_target_depth_options.get(
                "init_ref_depth",
                None,
            ),
        )
        target_state = _attach_target_runtime_lineage(
            target_state=target_state,
            depth_payload=single_eef["depth"],
            lift_depth_payload=single_eef["lift_depth"],
            eef_geometry=single_eef["geometry"],
            prepared=prepared_region_tracking,
            task_runtime=task_runtime,
            init_reference_depth=resolved_target_depth_options.get(
                "init_ref_depth",
                None,
            ),
            depth_publication=depth_publication,
        )
    if publisher is not None and (
        bool(target_state.get("mask_paths", {}))
        or bool(target_state.get("target_media_paths", {}))
    ):
        unsupported = []
        if bool(target_state.get("mask_paths", {})):
            unsupported.append("save_masks")
        if bool(target_state.get("target_media_paths", {})):
            unsupported.append("save_target_depth_mp4s")
        raise ValueError(
            "custom publisher cannot satisfy "
            "depth.artifact_policy."
            + "/".join(unsupported)
            + " because the current custom "
            "publisher contract does not receive target_depth_state; use "
            "the built-in publisher or disable the requested target media"
        )
    gripper_target_config = dict(
        target_config.get(
            "gripper_traj",
            {},
        )
        or {}
    )
    interaction_enabled = bool(
        (
            bool(target_config.get("enabled", False))
            and bool(gripper_target_config.get("enabled", False))
        )
        or str(single_eef["depth"].get("source", "") or "") == "rollout_gt_depth"
    )

    object_options = copy.deepcopy(dict(object_runtime_options or {}))
    if object_options.get("artifact_callback", None) is not None:
        raise ValueError(
            "object_runtime_options.artifact_callback is owned by "
            "run_multi_object_video_file"
        )
    if object_options.get("interaction_geometry_builder", None) is not None:
        raise ValueError(
            "object_runtime_options.interaction_geometry_builder is owned by "
            "run_multi_object_video_file"
        )
    for key in (
        "region_options_by_object",
        "tracking_options_by_object",
        "eef_precomputed_region",
        "precomputed_regions",
        "init_depth",
        "region_output_dirs",
        "tracking_output_dirs",
        "write_region_artifacts",
        "release_region_models",
        "artifact_callback",
    ):
        object_options.pop(key, None)
    if diagnostic_plan is not None or region_plan is not None:
        if "artifact_references_by_object" in object_options:
            raise ValueError(
                "object_runtime_options.artifact_references_by_object "
                "is owned by the explicit output_dir when "
                "write_artifacts=True"
            )
        region_object_refs = dict(
            ({} if region_plan is None else region_plan.get("object_references", {}))
            or {}
        )
        diagnostic_object_refs = dict(
            (
                {}
                if diagnostic_plan is None
                else diagnostic_plan.get("object_references", {})
            )
            or {}
        )
        object_options["artifact_references_by_object"] = {
            object_id: {
                **copy.deepcopy(dict(region_object_refs.get(object_id, {}) or {})),
                **copy.deepcopy(
                    dict(
                        diagnostic_object_refs.get(
                            object_id,
                            {},
                        )
                        or {}
                    )
                ),
            }
            for object_id in dict.fromkeys(
                [
                    *region_object_refs,
                    *diagnostic_object_refs,
                ]
            )
        }
    explicit_object_depths = object_options.pop(
        "object_depths",
        {},
    )
    if explicit_object_depths is None:
        explicit_object_depths = {}
    if not isinstance(explicit_object_depths, Mapping):
        raise TypeError("object_runtime_options.object_depths must be a mapping")
    explicit_object_depth_metadata = object_options.pop(
        "object_depth_metadata",
        {},
    )
    if explicit_object_depth_metadata is None:
        explicit_object_depth_metadata = {}
    if not isinstance(explicit_object_depth_metadata, Mapping):
        raise TypeError(
            "object_runtime_options.object_depth_metadata must be a mapping"
        )
    object_order = list(prepared_region_tracking["object_order"])
    target_maps = dict(target_state.get("maps", {}) or {})
    generated_object_depths = {
        object_id: target_maps[object_id]
        for object_id in object_order
        if object_id in target_maps
    }
    target_records = dict(
        dict(target_state.get("meta", {}) or {}).get(
            "targets",
            {},
        )
        or {}
    )
    generated_object_metadata = {
        object_id: copy.deepcopy(dict(target_records.get(object_id, {}) or {}))
        for object_id in generated_object_depths
    }
    resolved_object_depths = {
        **generated_object_depths,
        **copy.deepcopy(dict(explicit_object_depths)),
    }
    resolved_object_depth_metadata = {
        **generated_object_metadata,
        **copy.deepcopy(dict(explicit_object_depth_metadata)),
    }

    interaction_references = dict(
        resolved_target_depth_options.get(
            "interaction_artifact_references_by_object",
            {},
        )
        or {}
    )
    interaction_geometry_artifacts: dict[str, dict[str, Any]] = {}
    if publisher is None and diagnostic_plan is not None:
        interaction_references = {
            object_id: {
                key: str(dict(reference or {}).get(key, "") or "")
                for key in (
                    "eef_points_json",
                    "eef_points_flow_npz",
                    "obj_points_json",
                    "obj_points_flow_npz",
                )
            }
            for object_id, reference in dict(
                diagnostic_plan.get(
                    "object_references",
                    {},
                )
                or {}
            ).items()
        }

    def interaction_geometry_builder(
        *,
        object_id: str,
        stream: Mapping[str, Any],
        tracking: Mapping[str, Any],
        geometry_options: Mapping[str, Any],
        **_: Any,
    ) -> dict[str, Any]:
        selection = _interaction_depth_for_object(
            object_id=str(object_id),
            depth_payload=single_eef["depth"],
            target_state=target_state,
            target_config=target_config,
        )
        if selection is None:
            return {}
        (
            interaction_depths,
            source_name,
            depth_map_path,
            interaction_meta,
        ) = selection
        eef_tracking = dict(prepared_region_tracking["eef"]["tracking"])
        eef_pair = build_eef_trajectory_from_tracks(
            tracks_uv=eef_tracking["tracks_uv"],
            visibility=eef_tracking["visibility"],
            depths=interaction_depths,
            camera=camera,
            config=simulator_config,
            **copy.deepcopy(resolved_geometry_options),
            **copy.deepcopy(resolved_projection_options),
        )
        object_pair_geometry = build_visual_geometry_from_tracks(
            tracks_uv=tracking["tracks_uv"],
            visibility=tracking["visibility"],
            depths=interaction_depths,
            camera=camera,
            **copy.deepcopy(dict(geometry_options)),
        )
        if publisher is None and diagnostic_plan is not None:
            interaction_geometry_artifacts[str(object_id)] = {
                "eef_geometry": eef_pair["geometry"],
                "object_geometry": object_pair_geometry,
                "depth_reference": {
                    "source": str(source_name or "canonical"),
                    "path": str(depth_map_path or ""),
                },
            }
        references_raw = interaction_references.get(
            object_id,
            {},
        )
        if references_raw is None:
            references_raw = {}
        if not isinstance(references_raw, Mapping):
            raise TypeError(
                "target_depth_options."
                "interaction_artifact_references_by_object"
                f"[{object_id!r}] must be a mapping"
            )
        references = dict(references_raw)
        return {
            "meta": {
                **copy.deepcopy(dict(interaction_meta)),
                "used_by_gripper": True,
                "eef_points_json": str(
                    references.get(
                        "eef_points_json",
                        "",
                    )
                    or ""
                ),
                "eef_points_flow_npz": str(
                    references.get(
                        "eef_points_flow_npz",
                        "",
                    )
                    or ""
                ),
                "obj_points_json": str(
                    references.get(
                        "obj_points_json",
                        "",
                    )
                    or ""
                ),
                "obj_points_flow_npz": str(
                    references.get(
                        "obj_points_flow_npz",
                        "",
                    )
                    or ""
                ),
            },
            "eef_visual_center": copy.deepcopy(
                list(
                    eef_pair["trajectory"].get(
                        "visual_center",
                        [],
                    )
                    or []
                )
            ),
            "eef_controller": copy.deepcopy(
                list(
                    eef_pair["trajectory"].get(
                        "eef_controller",
                        [],
                    )
                    or []
                )
            ),
            "eef_tcp": copy.deepcopy(
                list(
                    eef_pair["trajectory"].get(
                        "eef_tcp",
                        [],
                    )
                    or []
                )
            ),
            "obj_visual_center": densify_center_records(
                list(
                    object_pair_geometry.get(
                        "visual_center_records",
                        [],
                    )
                    or []
                )
            ),
        }

    def object_callback(stage: str, payload: Any) -> None:
        if stage.endswith((".region", ".tracking")):
            return
        if stage.endswith(".depth"):
            return
        _emit(artifact_callback, stage, payload)

    object_runtime = run_prepared_object_evidence(
        frames=frames,
        camera=camera,
        simulator_config=simulator_config,
        object_stream_plan=list(task_runtime.get("object_stream_plan", []) or []),
        region_runtime=cached_region,
        tracking_backend=cached_tracking,
        depths=single_eef["depth"]["depths"],
        object_depths=resolved_object_depths,
        object_depth_metadata=resolved_object_depth_metadata,
        interaction_geometry_builder=(
            interaction_geometry_builder if interaction_enabled else None
        ),
        release_region_models=False,
        artifact_callback=object_callback,
        **object_options,
    )

    resolved_composition_options = copy.deepcopy(dict(composition_options or {}))
    if interaction_enabled:
        if "union_metadata" in resolved_composition_options:
            raise ValueError(
                "composition_options.union_metadata is owned by "
                "target/interaction depth orchestration"
            )
        depth_source_name = str(
            single_eef["depth"].get(
                "source",
                "",
            )
            or ""
        )
        resolved_composition_options["union_metadata"] = {
            "mode": (
                "rollout_gt_depth"
                if depth_source_name == "rollout_gt_depth"
                else str(
                    gripper_target_config.get(
                        "mode",
                        "pair_roi",
                    )
                    or "pair_roi"
                ).strip()
            ),
            "depth_model_only": bool(depth_source_name != "rollout_gt_depth"),
            "target_depth_calibration_meta": (
                str(target_state.get("meta_path", "") or "") or None
            ),
        }
    if bool(write_artifacts):
        if "artifact_references" in resolved_composition_options:
            raise ValueError(
                "composition_options.artifact_references is owned by "
                "the explicit output_dir when write_artifacts=True"
            )
        resolved_composition_options["artifact_references"] = _publication_references(
            output_dir
        )
    outputs = compose_prepared_multi_object_outputs(
        uid=str(uid),
        simulator_config=simulator_config,
        pipeline_config=normalized,
        single_eef_result=single_eef,
        object_runtime_result=object_runtime,
        metadata=metadata_payload,
        composition_options=resolved_composition_options,
    )
    _emit(artifact_callback, "composition", outputs)
    interaction_geometry_applied = any(
        bool(
            dict(evidence or {}).get(
                "interaction_geometry",
                {},
            )
        )
        for evidence in dict(
            object_runtime.get(
                "evidence_by_object",
                {},
            )
            or {}
        ).values()
    )

    publication = None
    if bool(write_artifacts):
        selected_publisher = (
            publisher if publisher is not None else publish_composed_trajectory_outputs
        )
        publisher_options = {
            "output_dir": output_dir,
            "outputs": outputs,
            "eef_pose_path": pose_publication_path,
        }
        if publisher is None:
            publisher_options["conditioning_alignment"] = (
                copy.deepcopy(conditioning_alignment)
            )
            publisher_options["target_depth_state"] = target_state
            if region_plan is not None:
                publisher_options["region_state"] = {
                    "plan": region_plan,
                    "eef_region": prepared_region_tracking["eef"]["result"],
                    "object_regions": {
                        object_id: prepared_region_tracking["objects"][object_id][
                            "result"
                        ]
                        for object_id in object_order
                    },
                    "first_frame_rgb": frames[0],
                    "write_visual_media": False,
                }
            if diagnostic_plan is not None:
                target_publication_meta = dict(
                    target_state.get(
                        "publication_meta",
                        target_state.get("meta", {}),
                    )
                    or {}
                )
                target_root = trajectory_artifact_paths(
                    Path(str(output_dir)).expanduser().resolve()
                )["depth_calibration_dir"]
                publisher_options["diagnostic_state"] = {
                    "plan": diagnostic_plan,
                    "eef_tracking": dict(prepared_region_tracking["eef"]["tracking"]),
                    "object_tracking": {
                        object_id: dict(
                            prepared_region_tracking["objects"][object_id]["tracking"]
                        )
                        for object_id in object_order
                    },
                    "video_frames": frames,
                    "write_visual_media": False,
                    "eef_geometry": dict(single_eef["geometry"]),
                    "object_geometry": {
                        object_id: dict(
                            object_runtime["objects"][object_id]["geometry"]
                        )
                        for object_id in object_order
                    },
                    "interaction_geometry": interaction_geometry_artifacts,
                    "depth_references": {
                        "eef": {
                            "source": str(
                                single_eef["lift_depth"].get(
                                    "source",
                                    "canonical",
                                )
                                or "canonical"
                            ),
                            "path": str(
                                dict(
                                    single_eef["lift_depth"].get(
                                        "metadata",
                                        {},
                                    )
                                    or {}
                                ).get(
                                    "depth_npy",
                                    "",
                                )
                                or ""
                            ),
                        },
                        **{
                            object_id: {
                                "source": str(
                                    object_runtime["objects"][object_id]["depth"].get(
                                        "source",
                                        "canonical",
                                    )
                                    or "canonical"
                                ),
                                "path": str(
                                    dict(
                                        object_runtime["objects"][object_id][
                                            "depth"
                                        ].get(
                                            "metadata",
                                            {},
                                        )
                                        or {}
                                    ).get(
                                        "depth_npy",
                                        "",
                                    )
                                    or ""
                                ),
                            }
                            for object_id in object_order
                        },
                    },
                    "geometry_manifest": {
                        "target_depth_calibration": {
                            "enabled": bool(
                                target_state.get(
                                    "configured",
                                    False,
                                )
                            ),
                            "dir": (
                                target_root.as_posix()
                                if bool(
                                    target_state.get(
                                        "configured",
                                        False,
                                    )
                                )
                                else None
                            ),
                            "meta_json": (
                                str(
                                    target_state.get(
                                        "meta_path",
                                        "",
                                    )
                                    or ""
                                )
                                or None
                            ),
                            "eef_meta_json": (
                                str(
                                    target_publication_meta.get(
                                        "depth_eef_traj_meta_json",
                                        "",
                                    )
                                    or ""
                                )
                                or None
                            ),
                            "gripper_traj_meta_json": (
                                str(
                                    target_publication_meta.get(
                                        "depth_gripper_traj_meta_json",
                                        "",
                                    )
                                    or ""
                                )
                                or None
                            ),
                            "stage_info_json": (
                                str(
                                    dict(
                                        target_state.get(
                                            "stage_info",
                                            {},
                                        )
                                        or {}
                                    ).get(
                                        "path",
                                        "",
                                    )
                                    or ""
                                )
                                or None
                            ),
                            "runtime_lineage_json": (
                                str(
                                    dict(
                                        target_state.get(
                                            "runtime_lineage",
                                            {},
                                        )
                                        or {}
                                    ).get(
                                        "path",
                                        "",
                                    )
                                    or ""
                                )
                                or None
                            ),
                            "eef_consumed_depth_samples_npy": (
                                str(
                                    dict(
                                        target_state.get(
                                            "runtime_lineage",
                                            {},
                                        )
                                        or {}
                                    ).get(
                                        "samples_path",
                                        "",
                                    )
                                    or ""
                                )
                                or None
                            ),
                        },
                        "interaction_geometry": {
                            "enabled": bool(interaction_enabled),
                            "source": (
                                "rollout_gt_depth"
                                if str(
                                    single_eef["depth"].get(
                                        "source",
                                        "",
                                    )
                                    or ""
                                )
                                == "rollout_gt_depth"
                                else str(
                                    gripper_target_config.get(
                                        "mode",
                                        "pair_roi",
                                    )
                                    or "pair_roi"
                                )
                            ),
                            "used_by_gripper": bool(interaction_enabled),
                        },
                    },
                }
            if isinstance(single_eef.get("pose", None), Mapping):
                pose_manifest_config = dict(normalized.get("pose", {}) or {})
                pose_manifest_meta = dict(single_eef["pose"].get("meta", {}) or {})
                publisher_options["eef_pose_payload"] = single_eef["pose"]
                publisher_options["eef_pose_manifest"] = {
                    "pose_config_source": str(
                        pose_manifest_config.get(
                            "config_path",
                            "",
                        )
                        or ""
                    ),
                    "foundationpose_debug_dir": None,
                    "mesh_path": str(
                        pose_manifest_meta.get(
                            "mesh_path",
                            pose_manifest_config.get(
                                "mesh_path",
                                "",
                            ),
                        )
                        or ""
                    ),
                    "register_mask_source": (
                        str(
                            single_eef.get(
                                "pose_register_mask_source",
                                "trajectory_region_mask",
                            )
                            or "trajectory_region_mask"
                        )
                    ),
                }
            elif isinstance(
                single_eef.get("pose_fallback", None),
                Mapping,
            ):
                pose_manifest_config = dict(normalized.get("pose", {}) or {})
                publisher_options["eef_pose_fallback"] = {
                    "pose_config_source": str(
                        pose_manifest_config.get("config_path", "") or ""
                    ),
                    "mesh_path": str(pose_manifest_config.get("mesh_path", "") or ""),
                    "register_mask_source": str(
                        single_eef.get(
                            "pose_register_mask_source",
                            "trajectory_region_mask",
                        )
                        or "trajectory_region_mask"
                    ),
                    "reason": str(
                        dict(single_eef["pose_fallback"]).get(
                            "reason",
                            "",
                        )
                        or ""
                    ),
                }
        publication = dict(selected_publisher(**publisher_options))
        _emit(
            artifact_callback,
            "publication",
            publication,
        )

    return {
        "scope": {
            "input": "explicit_video_file",
            "target": "eef_and_objects",
            "path": "prepared_multi_object_file",
            "decode_count": 1,
            "write_artifacts": bool(write_artifacts),
            "target_calibrated_depth": bool(
                target_state.get(
                    "attempted",
                    False,
                )
            ),
            "target_calibrated_depth_applied": bool(
                target_state.get(
                    "applied",
                    False,
                )
            ),
            "interaction_geometry_configured": bool(interaction_enabled),
            "interaction_geometry": bool(interaction_geometry_applied),
            "full_pipeline_parity": False,
        },
        "video_file": dict(prepared_video["video_file"]),
        "conditioning_alignment": conditioning_alignment,
        "camera": camera,
        "task_runtime": task_runtime,
        "prepared_region_tracking": prepared_region_tracking,
        "single_eef": single_eef,
        "target_depth": target_state,
        "object_runtime": object_runtime,
        "outputs": outputs,
        "depth_publication": depth_publication,
        "publication": publication,
    }


__all__ = [
    "FinalTrajectoryPublisher",
    "publish_composed_trajectory_outputs",
    "run_multi_object_video_file",
]
