"""Typed contracts for replaceable region algorithm providers."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


REGION_QUERY_SAMPLER_CONTRACT_VERSION = "region_query_sampler"
REGION_DETECTOR_CONTRACT_VERSION = "region_detector"
REGION_SEGMENTER_CONTRACT_VERSION = "region_segmenter"
REGION_RUNTIME_CONTRACT_VERSION = "region_runtime"


@dataclass(frozen=True)
class RegionPointSample:
    """Frame-aligned tracking queries selected from one resolved region."""

    points_xy: np.ndarray
    sampling_mask: np.ndarray
    points_xyz: np.ndarray | None = None
    used_depth_for_sampling: bool = False


@dataclass(frozen=True)
class RegionSamplingRequest:
    """All explicit inputs available to one region query sampler."""

    frame_shape_hw: tuple[int, int]
    num_points: int
    seed: int
    bbox_xyxy: Sequence[int] | None = None
    mask: np.ndarray | None = None
    init_depth: np.ndarray | None = None
    camera: Any | None = None


@dataclass(frozen=True)
class RegionSamplingPrediction:
    """Validated sampler output plus implementation identity."""

    sample: RegionPointSample
    backend_id: str
    algorithm_id: str
    coordinate_frame: str
    provider_kind: str = "builtin"
    contract_version: str = REGION_QUERY_SAMPLER_CONTRACT_VERSION
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def identity_dict(self) -> dict[str, Any]:
        """Return a JSON-safe identity without numerical arrays."""

        return {
            "provider_kind": str(self.provider_kind),
            "backend_id": str(self.backend_id),
            "algorithm_id": str(self.algorithm_id),
            "contract_version": str(self.contract_version),
            "coordinate_frame": str(self.coordinate_frame),
            "metadata": dict(self.metadata),
        }


class RegionQuerySampler(Protocol):
    """Structural interface for selecting tracking queries inside a region."""

    provider_kind: str
    backend_id: str
    algorithm_id: str
    contract_version: str

    def sample(
        self,
        request: RegionSamplingRequest,
    ) -> RegionSamplingPrediction:
        """Return deterministic points and truthful sampler provenance."""


def _normalized_identity_token(
    value: Any,
    *,
    label: str,
) -> str:
    token = str(value or "").strip().lower()
    if (
        not token
        or token.startswith(".")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in token
        )
    ):
        raise ValueError(f"{label} must use [a-z0-9._-] and be non-empty")
    return token


def _normalize_region_provider_identity(
    *,
    provider_kind: Any,
    backend_id: Any,
    algorithm_id: Any,
    contract_version: Any,
    expected_contract_version: str,
    source: str,
) -> dict[str, str]:
    normalized_kind = str(provider_kind or "").strip().lower()
    if normalized_kind not in {"builtin", "external"}:
        raise ValueError(f"{source}.provider_kind must be 'builtin' or 'external'")
    normalized_contract = str(contract_version or "").strip()
    if normalized_contract != expected_contract_version:
        raise ValueError(
            f"{source}.contract_version must be {expected_contract_version!r}"
        )
    return {
        "provider_kind": normalized_kind,
        "backend_id": _normalized_identity_token(
            backend_id,
            label=f"{source}.backend_id",
        ),
        "algorithm_id": _normalized_identity_token(
            algorithm_id,
            label=f"{source}.algorithm_id",
        ),
        "contract_version": normalized_contract,
    }


def normalize_region_query_sampler_identity(
    *,
    provider_kind: Any,
    backend_id: Any,
    algorithm_id: Any,
    contract_version: Any,
    source: str,
) -> dict[str, str]:
    """Return one stable identity for a model-free or learned sampler."""

    return _normalize_region_provider_identity(
        provider_kind=provider_kind,
        backend_id=backend_id,
        algorithm_id=algorithm_id,
        contract_version=contract_version,
        expected_contract_version=REGION_QUERY_SAMPLER_CONTRACT_VERSION,
        source=source,
    )


def normalize_region_detector_identity(
    *,
    provider_kind: Any,
    backend_id: Any,
    algorithm_id: Any,
    contract_version: Any,
    source: str,
) -> dict[str, str]:
    """Return one stable identity for a box or proposal detector."""

    return _normalize_region_provider_identity(
        provider_kind=provider_kind,
        backend_id=backend_id,
        algorithm_id=algorithm_id,
        contract_version=contract_version,
        expected_contract_version=REGION_DETECTOR_CONTRACT_VERSION,
        source=source,
    )


def normalize_region_segmenter_identity(
    *,
    provider_kind: Any,
    backend_id: Any,
    algorithm_id: Any,
    contract_version: Any,
    source: str,
) -> dict[str, str]:
    """Return one stable identity for a box-conditioned segmenter."""

    return _normalize_region_provider_identity(
        provider_kind=provider_kind,
        backend_id=backend_id,
        algorithm_id=algorithm_id,
        contract_version=contract_version,
        expected_contract_version=REGION_SEGMENTER_CONTRACT_VERSION,
        source=source,
    )


def normalize_region_runtime_identity(
    *,
    provider_kind: Any,
    backend_id: Any,
    algorithm_id: Any,
    contract_version: Any,
    source: str,
) -> dict[str, str]:
    """Return one stable identity for a complete region runtime."""

    return _normalize_region_provider_identity(
        provider_kind=provider_kind,
        backend_id=backend_id,
        algorithm_id=algorithm_id,
        contract_version=contract_version,
        expected_contract_version=REGION_RUNTIME_CONTRACT_VERSION,
        source=source,
    )


def _role_identity(
    provider: Any,
    *,
    role: str,
    method_name: str,
    normalizer: Any,
    source: str,
) -> dict[str, str]:
    role_reader = getattr(
        provider,
        f"{role}_provider_identity",
        None,
    )
    if callable(role_reader):
        declared = role_reader()
        if not isinstance(declared, Mapping):
            raise TypeError(
                f"{source}.{role}_provider_identity() must return a mapping"
            )
        payload = dict(declared)
    else:
        payload = {
            "provider_kind": getattr(
                provider,
                "provider_kind",
                None,
            ),
            "backend_id": getattr(provider, "backend_id", None),
            "algorithm_id": getattr(
                provider,
                "algorithm_id",
                None,
            ),
            "contract_version": getattr(
                provider,
                "contract_version",
                None,
            ),
        }
    if not callable(getattr(provider, method_name, None)):
        raise TypeError(f"{source} must expose {method_name}(...)")
    return normalizer(
        provider_kind=payload.get("provider_kind"),
        backend_id=payload.get("backend_id"),
        algorithm_id=payload.get("algorithm_id"),
        contract_version=payload.get("contract_version"),
        source=source,
    )


def region_detector_identity(
    detector: Any,
    *,
    source: str = "region detector",
) -> dict[str, str]:
    """Read and validate the identity declared by a detector provider."""

    return _role_identity(
        detector,
        role="detector",
        method_name="detect",
        normalizer=normalize_region_detector_identity,
        source=source,
    )


def region_segmenter_identity(
    segmenter: Any,
    *,
    source: str = "region segmenter",
) -> dict[str, str]:
    """Read and validate the identity declared by a segmenter provider."""

    return _role_identity(
        segmenter,
        role="segmenter",
        method_name="segment_from_bbox",
        normalizer=normalize_region_segmenter_identity,
        source=source,
    )


def region_runtime_identity(
    runtime: Any,
    *,
    source: str = "region runtime",
) -> dict[str, str]:
    """Read and validate a complete region-runtime provider identity."""

    if not (callable(runtime) or callable(getattr(runtime, "select_target", None))):
        raise TypeError(f"{source} must be callable or expose select_target(...)")
    role_reader = getattr(
        runtime,
        "region_runtime_provider_identity",
        None,
    )
    if callable(role_reader):
        declared = role_reader()
        if not isinstance(declared, Mapping):
            raise TypeError(
                f"{source}.region_runtime_provider_identity() must return a mapping"
            )
        payload = dict(declared)
    else:
        payload = {
            key: getattr(runtime, key, None)
            for key in (
                "provider_kind",
                "backend_id",
                "algorithm_id",
                "contract_version",
            )
        }
    return normalize_region_runtime_identity(
        provider_kind=payload.get("provider_kind"),
        backend_id=payload.get("backend_id"),
        algorithm_id=payload.get("algorithm_id"),
        contract_version=payload.get("contract_version"),
        source=source,
    )


def validate_external_region_runtime_identity(
    config: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, str]:
    """Validate configured identity for an external full-runtime provider."""

    identity = normalize_region_runtime_identity(
        provider_kind=config.get("provider_kind"),
        backend_id=config.get("backend_id"),
        algorithm_id=config.get("algorithm_id"),
        contract_version=config.get("contract_version"),
        source=source,
    )
    if identity["provider_kind"] != "external":
        raise ValueError(f"{source}.provider_kind must be 'external'")
    return identity


def region_query_sampler_identity(
    sampler: Any,
    *,
    source: str = "region query sampler",
) -> dict[str, str]:
    """Read and validate the identity declared by a sampler provider."""

    if not callable(getattr(sampler, "sample", None)):
        raise TypeError(f"{source} must expose sample(request)")
    return normalize_region_query_sampler_identity(
        provider_kind=getattr(sampler, "provider_kind", None),
        backend_id=getattr(sampler, "backend_id", None),
        algorithm_id=getattr(sampler, "algorithm_id", None),
        contract_version=getattr(sampler, "contract_version", None),
        source=source,
    )


def invoke_region_query_sampler(
    sampler: RegionQuerySampler,
    request: RegionSamplingRequest,
) -> RegionSamplingPrediction:
    """Invoke and validate one region query sampler without side effects."""

    if len(request.frame_shape_hw) != 2 or any(
        isinstance(value, bool)
        or not isinstance(value, (int, np.integer))
        or int(value) <= 0
        for value in request.frame_shape_hw
    ):
        raise ValueError(
            "region sampling frame_shape_hw must contain two positive integers"
        )
    if (
        isinstance(request.num_points, bool)
        or not isinstance(request.num_points, (int, np.integer))
        or int(request.num_points) <= 0
    ):
        raise ValueError("region sampling num_points must be a positive integer")
    if isinstance(request.seed, bool) or not isinstance(
        request.seed,
        (int, np.integer),
    ):
        raise TypeError("region sampling seed must be an integer")
    declared_identity = region_query_sampler_identity(sampler)
    sample_method = sampler.sample
    prediction = sample_method(request)
    if not isinstance(prediction, RegionSamplingPrediction):
        raise TypeError("region query sampler must return RegionSamplingPrediction")
    prediction_identity = normalize_region_query_sampler_identity(
        provider_kind=prediction.provider_kind,
        backend_id=prediction.backend_id,
        algorithm_id=prediction.algorithm_id,
        contract_version=prediction.contract_version,
        source="region query sampler prediction",
    )
    if prediction_identity != declared_identity:
        raise ValueError(
            "region query sampler prediction identity conflicts with "
            f"provider declaration: prediction={prediction_identity!r}, "
            f"provider={declared_identity!r}"
        )
    if str(prediction.coordinate_frame) not in {"pixel_xy", "world_xyz"}:
        raise ValueError(
            "region query sampler coordinate_frame must be 'pixel_xy' or 'world_xyz'"
        )
    if not isinstance(prediction.metadata, Mapping):
        raise TypeError("region query sampler metadata must be a mapping")
    try:
        json.dumps(dict(prediction.metadata), allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(
            "region query sampler metadata must be finite JSON data"
        ) from error

    sample = prediction.sample
    if not isinstance(sample, RegionPointSample):
        raise TypeError("RegionSamplingPrediction.sample must be RegionPointSample")
    points_xy = np.asarray(sample.points_xy)
    if (
        points_xy.ndim != 2
        or points_xy.shape[1:] != (2,)
        or points_xy.shape[0] == 0
        or not np.all(np.isfinite(points_xy))
    ):
        raise ValueError(
            "region query sampler points_xy must be non-empty and finite "
            "with shape [N,2]"
        )
    expected_shape = tuple(int(value) for value in request.frame_shape_hw)
    sampling_mask = np.asarray(sample.sampling_mask)
    if sampling_mask.shape != expected_shape:
        raise ValueError(
            "region query sampler sampling_mask shape does not match frame"
        )
    if sampling_mask.dtype != np.bool_:
        raise TypeError("region query sampler sampling_mask must have boolean dtype")
    height, width = expected_shape
    if points_xy.shape[0] and (
        np.any(points_xy[:, 0] < 0)
        or np.any(points_xy[:, 0] > max(width - 1, 0))
        or np.any(points_xy[:, 1] < 0)
        or np.any(points_xy[:, 1] > max(height - 1, 0))
    ):
        raise ValueError("region query sampler points_xy must remain inside the frame")
    points_xyz = sample.points_xyz
    if points_xyz is not None:
        normalized_xyz = np.asarray(points_xyz)
        if normalized_xyz.shape != (points_xy.shape[0], 3) or not np.all(
            np.isfinite(normalized_xyz)
        ):
            raise ValueError(
                "region query sampler points_xyz must be finite with "
                "shape [N,3] aligned to points_xy"
            )
    uses_world_coordinates = str(prediction.coordinate_frame) == "world_xyz"
    if bool(sample.used_depth_for_sampling) != (
        points_xyz is not None
    ) or uses_world_coordinates != (points_xyz is not None):
        raise ValueError(
            "region query sampler coordinate_frame, "
            "used_depth_for_sampling, and points_xyz must agree"
        )
    if int(points_xy.shape[0]) > int(request.num_points):
        raise ValueError("region query sampler selected more points than requested")
    pixel_indices = np.rint(points_xy).astype(np.int64)
    if not np.all(
        sampling_mask[
            pixel_indices[:, 1],
            pixel_indices[:, 0],
        ]
    ):
        raise ValueError("region query sampler points_xy must lie inside sampling_mask")
    return prediction


__all__ = [
    "REGION_DETECTOR_CONTRACT_VERSION",
    "REGION_QUERY_SAMPLER_CONTRACT_VERSION",
    "REGION_RUNTIME_CONTRACT_VERSION",
    "REGION_SEGMENTER_CONTRACT_VERSION",
    "RegionPointSample",
    "RegionQuerySampler",
    "RegionSamplingPrediction",
    "RegionSamplingRequest",
    "invoke_region_query_sampler",
    "normalize_region_detector_identity",
    "normalize_region_query_sampler_identity",
    "normalize_region_runtime_identity",
    "normalize_region_segmenter_identity",
    "region_detector_identity",
    "region_query_sampler_identity",
    "region_runtime_identity",
    "region_segmenter_identity",
    "validate_external_region_runtime_identity",
]
