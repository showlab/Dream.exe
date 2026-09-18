"""Model-independent contracts for the :mod:`video2traj.tracking` package."""

from __future__ import annotations

import inspect
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any, Protocol, Sequence, runtime_checkable

import numpy as np


TRACKING_BACKEND_CONTRACT_VERSION = "tracking_backend"
COTRACKER_BACKEND_ID = "cotracker"
_TRACKING_PROVIDER_KINDS = frozenset({"builtin", "external"})
_EFFECTIVE_QUERY_MODES = frozenset(
    {
        "points",
        "mask_grid",
        "bbox_center",
    }
)


@dataclass(frozen=True)
class TrackingOutput:
    """Validated output of one model-independent tracking invocation."""

    tracks: Any
    visibility: Any
    resolved_query_points_xy: np.ndarray
    effective_query_mode: str
    provider: dict[str, str]


@runtime_checkable
class TrackingBackend(Protocol):
    """Legacy-compatible tracker contract used by current runtime callers.

    Backends keep the current callable shape so existing CoTracker-compatible
    implementations can be injected without an adapter object or request
    wrapper.  New providers should implement :class:`TrackingPredictionBackend`
    so inference does not own artifact publication.
    """

    provider_kind: str
    backend_id: str
    contract_version: str

    def track(
        self,
        *,
        video_frames: Any,
        output_dir: str,
        region_bbox_xyxy: Any = None,
        num_points: int = 50,
        filename: str = "points_cloud",
        seed: int = 42,
        query_points_xy: Any = None,
        segmentation_mask: Any = None,
        query_mode: str = "auto",
        grid_size: int = 0,
    ) -> tuple[Any, Any, Any, str]:
        """Return tracks, visibility, resolved queries, and query mode."""

        ...


@runtime_checkable
class TrackingPredictionBackend(Protocol):
    """Path-neutral tracking inference without artifact-publication inputs."""

    provider_kind: str
    backend_id: str
    contract_version: str

    def predict(
        self,
        *,
        video_frames: Any,
        region_bbox_xyxy: Any = None,
        num_points: int = 50,
        seed: int = 42,
        query_points_xy: Any = None,
        segmentation_mask: Any = None,
        query_mode: str = "auto",
        grid_size: int = 0,
    ) -> tuple[Any, Any, Any, str]:
        """Return tracks, visibility, resolved queries, and query mode."""

        ...


class BaseTrackingPredictionBackend(ABC):
    """Path-neutral tracker template with a legacy pipeline compatibility shim."""

    provider_kind = "external"
    backend_id = ""
    contract_version = TRACKING_BACKEND_CONTRACT_VERSION

    @abstractmethod
    def predict(
        self,
        *,
        video_frames: Any,
        region_bbox_xyxy: Any = None,
        num_points: int = 50,
        seed: int = 42,
        query_points_xy: Any = None,
        segmentation_mask: Any = None,
        query_mode: str = "auto",
        grid_size: int = 0,
    ) -> tuple[Any, Any, Any, str]:
        """Return tracks, visibility, resolved queries, and query mode."""

    def track(
        self,
        *,
        video_frames: Any,
        output_dir: str,
        region_bbox_xyxy: Any = None,
        num_points: int = 50,
        filename: str = "points_cloud",
        seed: int = 42,
        query_points_xy: Any = None,
        segmentation_mask: Any = None,
        query_mode: str = "auto",
        grid_size: int = 0,
        write_artifacts: bool = False,
    ) -> tuple[Any, Any, Any, str]:
        # The current pipeline publishes normalized tracking artifacts after
        # validation.  The compatibility flag is accepted so a path-neutral
        # provider also works when that caller-owned publication is enabled.
        del output_dir, filename, write_artifacts
        return self.predict(
            video_frames=video_frames,
            region_bbox_xyxy=region_bbox_xyxy,
            num_points=num_points,
            seed=seed,
            query_points_xy=query_points_xy,
            segmentation_mask=segmentation_mask,
            query_mode=query_mode,
            grid_size=grid_size,
        )


def _tracking_backend_id(
    value: Any,
    *,
    source: str,
) -> str:
    backend_id = str(value or "").strip().lower()
    if (
        not backend_id
        or backend_id.startswith(".")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in backend_id
        )
    ):
        raise ValueError(f"{source}.backend_id must use [a-z0-9._-] and be non-empty")
    return backend_id


def normalize_tracking_backend_identity(
    *,
    provider_kind: Any,
    backend_id: Any,
    contract_version: Any,
    source: str,
) -> dict[str, str]:
    """Return one JSON-ready, stable tracking-provider identity."""

    normalized_kind = str(provider_kind or "").strip().lower()
    if normalized_kind not in _TRACKING_PROVIDER_KINDS:
        raise ValueError(
            f"{source}.provider_kind must be one of "
            f"{sorted(_TRACKING_PROVIDER_KINDS)!r}"
        )
    normalized_backend_id = _tracking_backend_id(
        backend_id,
        source=source,
    )
    normalized_contract = str(contract_version or "").strip()
    if normalized_contract != TRACKING_BACKEND_CONTRACT_VERSION:
        raise ValueError(
            f"{source}.contract_version must be {TRACKING_BACKEND_CONTRACT_VERSION!r}"
        )
    if normalized_kind == "builtin" and normalized_backend_id != COTRACKER_BACKEND_ID:
        raise ValueError(
            f"{source} declares unsupported builtin tracking backend "
            f"{normalized_backend_id!r}"
        )
    return {
        "provider_kind": normalized_kind,
        "backend_id": normalized_backend_id,
        "contract_version": normalized_contract,
    }


def validate_external_tracking_runtime_identity(
    runtime_config: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    """Validate and normalize one external tracking declaration."""

    if not isinstance(runtime_config, Mapping):
        raise TypeError(f"{source} must be a mapping")
    runtime = dict(runtime_config)
    if str(runtime.get("provider_kind", "") or "").strip().lower() != "external":
        raise ValueError(f"{source}.provider_kind must be 'external'")
    identity = normalize_tracking_backend_identity(
        provider_kind=runtime.get("provider_kind"),
        backend_id=runtime.get("backend_id"),
        contract_version=runtime.get("contract_version"),
        source=source,
    )
    runtime.update(identity)
    return runtime


def tracking_backend_identity(
    backend: Any,
    *,
    source: str = "tracking backend",
) -> dict[str, str]:
    """Read and validate the identity declared by a runtime backend."""

    return normalize_tracking_backend_identity(
        provider_kind=getattr(backend, "provider_kind", None),
        backend_id=getattr(backend, "backend_id", None),
        contract_version=getattr(backend, "contract_version", None),
        source=source,
    )


def resolve_tracking_query_mode(
    *,
    query_mode: str,
    query_points_xy: Any,
    segmentation_mask: Any,
) -> str:
    """Resolve current point, mask-grid, or bbox-center query semantics."""

    normalized_mode = str(query_mode or "auto").strip().lower()
    if normalized_mode not in {
        "auto",
        "points",
        "mask_grid",
        "bbox_center",
    }:
        raise ValueError(f"Unsupported query_mode='{normalized_mode}'.")
    if normalized_mode == "auto":
        if query_points_xy is not None:
            return "points"
        if segmentation_mask is not None:
            return "mask_grid"
        return "bbox_center"
    if normalized_mode in {"mask_grid", "bbox_center"} and _has_nonempty_query_points(
        query_points_xy
    ):
        raise ValueError(
            f"query_mode={normalized_mode!r} conflicts with non-empty query_points_xy."
        )
    if normalized_mode == "points" and query_points_xy is None:
        raise ValueError("query_mode='points' requires query_points_xy.")
    if normalized_mode == "mask_grid" and segmentation_mask is None:
        raise ValueError("query_mode='mask_grid' requires segmentation_mask.")
    return normalized_mode


def _has_nonempty_query_points(value: Any) -> bool:
    if value is None:
        return False
    shape = getattr(value, "shape", None)
    if shape is not None:
        try:
            return len(shape) == 0 or int(shape[0]) != 0
        except (TypeError, ValueError):
            return True
    try:
        return len(value) != 0
    except (TypeError, ValueError):
        return True


def prepare_tracking_query_points(
    *,
    frame_shape: tuple[int, int],
    query_points_xy: Any = None,
    region_bbox_xyxy: Sequence[int] | None = None,
    num_points: int = 50,
    seed: int = 42,
) -> np.ndarray:
    """Clip explicit points or sample the current Gaussian bbox queries."""

    height, width = frame_shape
    if query_points_xy is not None:
        normalized_points = np.asarray(
            query_points_xy,
            dtype=np.float32,
        )
        if normalized_points.ndim != 2 or normalized_points.shape[1] != 2:
            raise ValueError(
                f"query_points_xy must have shape (N,2), got {normalized_points.shape}"
            )
        if normalized_points.shape[0] == 0:
            raise ValueError("query_points_xy is empty.")
        normalized_points[:, 0] = np.clip(
            normalized_points[:, 0],
            0,
            width - 1,
        )
        normalized_points[:, 1] = np.clip(
            normalized_points[:, 1],
            0,
            height - 1,
        )
        return normalized_points

    if region_bbox_xyxy is None:
        raise ValueError(
            "region_bbox_xyxy is required when query_points_xy is not provided."
        )

    x_min, y_min, x_max, y_max = [int(value) for value in region_bbox_xyxy]
    rng = np.random.default_rng(int(seed))
    center_x = (x_min + x_max) / 2.0
    center_y = (y_min + y_max) / 2.0
    sigma_x = max((x_max - x_min) / 6.0, 1e-6)
    sigma_y = max((y_max - y_min) / 6.0, 1e-6)

    x_coordinates = np.clip(
        rng.normal(center_x, sigma_x, int(num_points)),
        x_min,
        x_max,
    )
    y_coordinates = np.clip(
        rng.normal(center_y, sigma_y, int(num_points)),
        y_min,
        y_max,
    )
    return np.stack(
        (x_coordinates, y_coordinates),
        axis=1,
    ).astype(np.float32)


def infer_tracking_grid_size(num_points: int) -> int:
    """Return the current square grid side for a target point count."""

    return max(
        1,
        int(np.ceil(np.sqrt(max(1, int(num_points))))),
    )


def _tracking_output_shape(
    value: Any,
    *,
    label: str,
) -> tuple[int, ...]:
    """Read a tensor/array shape without moving framework values to CPU."""

    raw_shape = getattr(value, "shape", None)
    if raw_shape is None:
        try:
            raw_shape = np.shape(value)
        except Exception as error:
            raise TypeError(
                f"tracking backend returned unreadable {label} output"
            ) from error
    try:
        return tuple(int(dimension) for dimension in raw_shape)
    except (TypeError, ValueError) as error:
        raise TypeError(
            f"tracking backend returned unreadable {label} shape: {raw_shape!r}"
        ) from error


def _unbatched_tracking_shape(
    shape: tuple[int, ...],
    *,
    label: str,
    unbatched_rank: int,
    expected_suffix: tuple[int, ...] = (),
) -> tuple[int, ...]:
    batched_rank = unbatched_rank + 1
    if len(shape) == batched_rank:
        if shape[0] != 1:
            raise ValueError(
                f"tracking backend {label} batch dimension must be 1, got {shape}"
            )
        shape = shape[1:]
    if len(shape) != unbatched_rank or (
        expected_suffix and shape[-len(expected_suffix) :] != expected_suffix
    ):
        expected = "[T,N,2] or [1,T,N,2]" if label == "tracks" else "[T,N] or [1,T,N]"
        raise ValueError(
            f"tracking backend must return {label} {expected}, got {shape}"
        )
    return shape


def _validate_tracking_output_shapes(
    tracks: Any,
    visibility: Any,
    *,
    frame_count: int | None,
) -> tuple[int, int]:
    tracks_shape = _unbatched_tracking_shape(
        _tracking_output_shape(tracks, label="tracks"),
        label="tracks",
        unbatched_rank=3,
        expected_suffix=(2,),
    )
    visibility_shape = _unbatched_tracking_shape(
        _tracking_output_shape(visibility, label="visibility"),
        label="visibility",
        unbatched_rank=2,
    )
    if tracks_shape[:2] != visibility_shape:
        raise ValueError(
            "track/visibility alignment mismatch: "
            f"tracks={tracks_shape}, visibility={visibility_shape}"
        )
    if frame_count is not None and tracks_shape[0] != int(frame_count):
        raise ValueError(
            "track/frame alignment mismatch: "
            f"tracks={tracks_shape[0]}, frames={frame_count}"
        )
    if tracks_shape[1] <= 0:
        raise ValueError("tracking backend returned no tracked points")
    return tracks_shape[0], tracks_shape[1]


def _validate_tracking_output_finite(
    value: Any,
    *,
    label: str,
) -> None:
    """Validate values without copying an accelerator tensor to host memory."""

    tensor_isfinite = getattr(value, "isfinite", None)
    if callable(tensor_isfinite):
        try:
            finite_values = tensor_isfinite()
            all_finite = finite_values.all()
            item = getattr(all_finite, "item", None)
            scalar = item() if callable(item) else all_finite
            is_finite = bool(scalar)
        except (AttributeError, RuntimeError, TypeError, ValueError) as error:
            raise TypeError(
                f"tracking backend returned unreadable {label} values "
                "for finite validation"
            ) from error
    else:
        try:
            is_finite = bool(np.all(np.isfinite(value)))
        except (TypeError, ValueError) as error:
            raise TypeError(
                f"tracking backend returned unreadable {label} values "
                "for finite validation"
            ) from error
    if not is_finite:
        raise ValueError(f"tracking backend {label} must contain only finite values")


def _tracking_frame_count(video_frames: Any) -> int | None:
    try:
        return int(len(video_frames))
    except (TypeError, ValueError):
        return None


def _tracking_frame_shape(video_frames: Any) -> tuple[int, int]:
    try:
        first_frame = video_frames[0]
    except (IndexError, KeyError, TypeError) as error:
        raise ValueError(
            "cannot reconstruct tracking queries without indexable video_frames"
        ) from error
    shape = _tracking_output_shape(
        first_frame,
        label="video frame",
    )
    if len(shape) < 2 or shape[0] <= 0 or shape[1] <= 0:
        raise ValueError(
            f"cannot reconstruct tracking queries from invalid frame shape {shape}"
        )
    return shape[0], shape[1]


def _normalized_resolved_query_points(
    value: Any,
    *,
    expected_count: int,
) -> np.ndarray:
    candidate = value
    detach = getattr(candidate, "detach", None)
    if callable(detach):
        candidate = detach()
    cpu = getattr(candidate, "cpu", None)
    if callable(cpu):
        candidate = cpu()
    to_numpy = getattr(candidate, "numpy", None)
    if callable(to_numpy):
        candidate = to_numpy()
    try:
        points = np.asarray(candidate, dtype=np.float32)
    except (TypeError, ValueError) as error:
        raise TypeError(
            "tracking backend returned unreadable resolved_query_points_xy"
        ) from error
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError(
            "tracking backend resolved_query_points_xy must have shape "
            f"(N,2), got {points.shape}"
        )
    if points.shape[0] != int(expected_count):
        raise ValueError(
            "tracking query/track count mismatch: "
            f"queries={points.shape[0]}, tracks={expected_count}"
        )
    if not np.all(np.isfinite(points)):
        raise ValueError("tracking backend resolved_query_points_xy must be finite")
    return np.ascontiguousarray(points, dtype=np.float32)


def _resolve_backend_query_mode(
    value: Any,
    *,
    expected_mode: str,
) -> str:
    if value is None:
        return expected_mode
    normalized = str(value).strip().lower()
    if normalized not in _EFFECTIVE_QUERY_MODES:
        raise ValueError(
            "tracking backend effective query mode must be one of "
            f"{sorted(_EFFECTIVE_QUERY_MODES)!r}, got {normalized!r}"
        )
    if normalized != expected_mode:
        raise ValueError(
            "tracking backend query mode conflicts with the resolved request: "
            f"backend={normalized!r}, request={expected_mode!r}"
        )
    return normalized


def _reconstruct_query_points(
    *,
    effective_query_mode: str,
    video_frames: Any,
    query_points_xy: Any,
    region_bbox_xyxy: Any,
    num_points: int,
    seed: int,
) -> np.ndarray:
    if effective_query_mode == "mask_grid":
        raise ValueError(
            "tracking backend returned no resolved_query_points_xy for "
            "query_mode='mask_grid'; exact model-grid queries cannot be "
            "reconstructed from the segmentation mask"
        )
    return prepare_tracking_query_points(
        frame_shape=_tracking_frame_shape(video_frames),
        query_points_xy=(query_points_xy if effective_query_mode == "points" else None),
        region_bbox_xyxy=region_bbox_xyxy,
        num_points=num_points,
        seed=seed,
    )


def tracking_backend_accepts_write_artifacts(backend: Any) -> bool:
    """Return whether ``track`` explicitly accepts the optional current implementation control."""

    track = getattr(backend, "track", None)
    if not callable(track):
        return False
    try:
        parameters = inspect.signature(track).parameters.values()
    except (TypeError, ValueError):
        return False
    return any(
        parameter.name == "write_artifacts"
        or parameter.kind is inspect.Parameter.VAR_KEYWORD
        for parameter in parameters
    )


def _validated_tracking_output(
    result: Any,
    *,
    operation: str,
    provider: dict[str, str],
    frame_count: int | None,
    expected_query_mode: str,
    video_frames: Any,
    query_points_xy: Any,
    region_bbox_xyxy: Any,
    num_points: int,
    seed: int,
) -> TrackingOutput:
    if not isinstance(result, tuple) or len(result) != 4:
        raise TypeError(
            f"tracking backend {operation}(...) must return a 4-item tuple: "
            "(tracks, visibility, resolved_queries, query_mode)"
        )
    (
        predicted_tracks,
        predicted_visibility,
        resolved_query_points,
        backend_query_mode,
    ) = result
    _, tracked_point_count = _validate_tracking_output_shapes(
        predicted_tracks,
        predicted_visibility,
        frame_count=frame_count,
    )
    _validate_tracking_output_finite(
        predicted_tracks,
        label="tracks",
    )
    _validate_tracking_output_finite(
        predicted_visibility,
        label="visibility",
    )
    effective_query_mode = _resolve_backend_query_mode(
        backend_query_mode,
        expected_mode=expected_query_mode,
    )
    if resolved_query_points is None:
        resolved_query_points = _reconstruct_query_points(
            effective_query_mode=effective_query_mode,
            video_frames=video_frames,
            query_points_xy=query_points_xy,
            region_bbox_xyxy=region_bbox_xyxy,
            num_points=num_points,
            seed=seed,
        )
    normalized_queries = _normalized_resolved_query_points(
        resolved_query_points,
        expected_count=tracked_point_count,
    )
    return TrackingOutput(
        tracks=predicted_tracks,
        visibility=predicted_visibility,
        resolved_query_points_xy=normalized_queries,
        effective_query_mode=effective_query_mode,
        provider=provider,
    )


def run_tracking_prediction_backend(
    backend: TrackingPredictionBackend,
    *,
    video_frames: Any,
    region_bbox_xyxy: Any = None,
    num_points: int = 50,
    seed: int = 42,
    query_points_xy: Any = None,
    segmentation_mask: Any = None,
    query_mode: str = "auto",
    grid_size: int = 0,
) -> TrackingOutput:
    """Run path-neutral tracking inference and validate its stage output."""

    predict = getattr(backend, "predict", None)
    if not callable(predict):
        raise TypeError("tracking prediction backend must expose predict(...)")
    provider = tracking_backend_identity(backend)
    frame_count = _tracking_frame_count(video_frames)
    if frame_count == 0:
        raise ValueError("video_frames is empty.")
    expected_query_mode = resolve_tracking_query_mode(
        query_mode=query_mode,
        query_points_xy=query_points_xy,
        segmentation_mask=segmentation_mask,
    )
    result = predict(
        video_frames=video_frames,
        region_bbox_xyxy=region_bbox_xyxy,
        num_points=num_points,
        seed=seed,
        query_points_xy=query_points_xy,
        segmentation_mask=segmentation_mask,
        query_mode=query_mode,
        grid_size=grid_size,
    )
    return _validated_tracking_output(
        result,
        operation="predict",
        provider=provider,
        frame_count=frame_count,
        expected_query_mode=expected_query_mode,
        video_frames=video_frames,
        query_points_xy=query_points_xy,
        region_bbox_xyxy=region_bbox_xyxy,
        num_points=num_points,
        seed=seed,
    )


def run_tracking_backend(
    backend: TrackingBackend,
    *,
    video_frames: Any,
    output_dir: str,
    region_bbox_xyxy: Any = None,
    num_points: int = 50,
    filename: str = "points_cloud",
    seed: int = 42,
    query_points_xy: Any = None,
    segmentation_mask: Any = None,
    query_mode: str = "auto",
    grid_size: int = 0,
    write_artifacts: bool | None = None,
) -> TrackingOutput:
    """Call an identified tracker and validate all observable stage outputs."""

    track = getattr(backend, "track", None)
    if not callable(track):
        raise TypeError("tracking backend must expose callable track(...)")
    provider = tracking_backend_identity(backend)
    frame_count = _tracking_frame_count(video_frames)
    if frame_count == 0:
        raise ValueError("video_frames is empty.")
    expected_query_mode = resolve_tracking_query_mode(
        query_mode=query_mode,
        query_points_xy=query_points_xy,
        segmentation_mask=segmentation_mask,
    )
    supports_write_control = tracking_backend_accepts_write_artifacts(backend)
    if bool(write_artifacts) and not supports_write_control:
        raise TypeError(
            "tracking backend does not accept write_artifacts; "
            "the requested tracking-artifact publication cannot be honored"
        )
    optional_write_control = {}
    if write_artifacts is not None and supports_write_control:
        optional_write_control["write_artifacts"] = bool(write_artifacts)
    result = track(
        video_frames=video_frames,
        output_dir=output_dir,
        region_bbox_xyxy=region_bbox_xyxy,
        num_points=num_points,
        filename=filename,
        seed=seed,
        query_points_xy=query_points_xy,
        segmentation_mask=segmentation_mask,
        query_mode=query_mode,
        grid_size=grid_size,
        **optional_write_control,
    )
    return _validated_tracking_output(
        result,
        operation="track",
        provider=provider,
        frame_count=frame_count,
        expected_query_mode=expected_query_mode,
        video_frames=video_frames,
        query_points_xy=query_points_xy,
        region_bbox_xyxy=region_bbox_xyxy,
        num_points=num_points,
        seed=seed,
    )


__all__ = [
    "BaseTrackingPredictionBackend",
    "COTRACKER_BACKEND_ID",
    "TRACKING_BACKEND_CONTRACT_VERSION",
    "TrackingBackend",
    "TrackingPredictionBackend",
    "TrackingOutput",
    "infer_tracking_grid_size",
    "normalize_tracking_backend_identity",
    "prepare_tracking_query_points",
    "resolve_tracking_query_mode",
    "run_tracking_backend",
    "run_tracking_prediction_backend",
    "tracking_backend_identity",
    "validate_external_tracking_runtime_identity",
]
