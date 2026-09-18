"""Lazy, environment-independent runtime for first-frame region selection.

Simulation integrations resolve their segmentation assets outside this module
and pass a :class:`PrecomputedRegion`.  GroundingDINO-like detectors and
SAM2-like segmenters are explicit objects or zero-argument factories; no heavy
vision backend is imported or claimed available by the core package.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import copy
import json
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .backends.sampling import builtin_region_query_samplers
from .contract import (
    REGION_DETECTOR_CONTRACT_VERSION,
    REGION_RUNTIME_CONTRACT_VERSION,
    REGION_SEGMENTER_CONTRACT_VERSION,
    RegionQuerySampler,
    RegionSamplingPrediction,
    RegionSamplingRequest,
    invoke_region_query_sampler,
    normalize_region_detector_identity,
    normalize_region_segmenter_identity,
    region_detector_identity,
    region_query_sampler_identity,
    region_segmenter_identity,
)
from .selection import (
    bbox_from_mask,
    bbox_to_mask,
    clip_bbox_to_frame,
    combine_mask_with_manual_bbox,
    erode_mask,
    resolve_bbox_source,
    resolve_config_bbox,
    resolve_mask_backend,
    resolve_region_prompt,
    resolve_region_target_plan,
    resolve_sampling_method,
)


class RegionDetector(Protocol):
    """Injected bounding-box detector contract."""

    provider_kind: str
    backend_id: str
    algorithm_id: str
    contract_version: str

    def detect(
        self,
        image_rgb: np.ndarray,
        prompt: str,
    ) -> RegionProposal | Mapping[str, Any]:
        """Return one typed proposal or a compatible proposal mapping."""


class RegionSegmenter(Protocol):
    """Injected box-conditioned mask segmenter contract."""

    provider_kind: str
    backend_id: str
    algorithm_id: str
    contract_version: str

    def segment_from_bbox(
        self,
        image_rgb: np.ndarray,
        bbox_xyxy: list[int],
        prompt: str | None = None,
    ) -> np.ndarray | RegionSegmentationPrediction:
        """Return one mask prediction aligned with ``image_rgb``."""


class BaseRegionDetector(ABC):
    """Minimal template for an external text-conditioned detector."""

    provider_kind = "external"
    backend_id = ""
    algorithm_id = ""
    contract_version = REGION_DETECTOR_CONTRACT_VERSION

    @abstractmethod
    def detect(
        self,
        image_rgb: np.ndarray,
        prompt: str,
    ) -> RegionProposal | Mapping[str, Any]:
        """Return one proposal in pixel ``xyxy`` coordinates."""


class BaseRegionSegmenter(ABC):
    """Minimal template for an external box-conditioned segmenter."""

    provider_kind = "external"
    backend_id = ""
    algorithm_id = ""
    contract_version = REGION_SEGMENTER_CONTRACT_VERSION

    @abstractmethod
    def segment_from_bbox(
        self,
        image_rgb: np.ndarray,
        bbox_xyxy: list[int],
        prompt: str | None = None,
    ) -> np.ndarray | RegionSegmentationPrediction:
        """Return an ``[H,W]`` mask aligned with ``image_rgb``."""


RegionBackendFactory = Callable[[], Any]


@dataclass
class RegionProposal:
    """Current detector handoff payload."""

    bbox_xyxy: list[int]
    mask: np.ndarray | None = None
    score: float | None = None
    source: str = ""
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provider_kind: str | None = None
    backend_id: str | None = None
    algorithm_id: str | None = None
    contract_version: str | None = None


@dataclass
class RegionSegmentationPrediction:
    """Optional rich mask handoff for external segmenter providers."""

    mask: np.ndarray
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provider_kind: str | None = None
    backend_id: str | None = None
    algorithm_id: str | None = None
    contract_version: str | None = None


@dataclass
class PrecomputedRegion:
    """Environment-resolved mask input for the pure selector runtime."""

    mask: np.ndarray
    bbox_xyxy: list[int] | None = None
    source: str = "precomputed"
    matched_names: list[str] = field(default_factory=list)
    compact_labels: list[int] = field(default_factory=list)


@dataclass
class RegionResult:
    """Current region output plus in-memory masks used by tracking."""

    target_name: str
    source: str
    bbox_source: str
    sampling_method: str
    mask_backend: str | None
    prompt: str | None
    manual_bbox_xyxy: list[int] | None
    detected_bbox_xyxy: list[int] | None
    final_bbox_xyxy: list[int]
    mask: np.ndarray
    eroded_mask: np.ndarray
    sampling_mask: np.ndarray
    sampled_points_xy: np.ndarray
    sampled_points_xyz: np.ndarray | None
    used_depth_for_sampling: bool
    tracking_input: str
    output_json: str
    artifacts: dict[str, str] = field(default_factory=dict)
    selection_metadata: dict[str, Any] = field(default_factory=dict)
    sampling_backend: dict[str, Any] = field(default_factory=dict)
    provider_chain: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize the current JSON contract."""

        return {
            "target_name": self.target_name,
            "source": self.source,
            "bbox_source": self.bbox_source,
            "sampling_method": self.sampling_method,
            "mask_backend": self.mask_backend,
            "prompt": self.prompt,
            "manual_bbox_xyxy": self.manual_bbox_xyxy,
            "detected_bbox_xyxy": self.detected_bbox_xyxy,
            "final_bbox_xyxy": self.final_bbox_xyxy,
            "mask_area": int(np.count_nonzero(self.mask)),
            "eroded_mask_area": int(np.count_nonzero(self.eroded_mask)),
            "sampling_mask_area": int(np.count_nonzero(self.sampling_mask)),
            "num_sampled_points": int(self.sampled_points_xy.shape[0]),
            "used_depth_for_sampling": bool(self.used_depth_for_sampling),
            "tracking_input": self.tracking_input,
            "sampled_points_xy": (self.sampled_points_xy.astype(float).tolist()),
            "sampled_points_xyz": (
                None
                if self.sampled_points_xyz is None
                else self.sampled_points_xyz.astype(float).tolist()
            ),
            "artifacts": {
                "json": self.output_json,
                **dict(self.artifacts),
            },
            "selection_metadata": dict(self.selection_metadata),
            "sampling_backend": copy.deepcopy(self.sampling_backend),
            "provider_chain": copy.deepcopy(self.provider_chain),
        }


def _coerce_region_proposal(payload: Any) -> RegionProposal:
    if isinstance(payload, RegionProposal):
        return payload
    if isinstance(payload, Mapping):
        data = dict(payload)
        bbox = data.get("bbox_xyxy")
        return RegionProposal(
            bbox_xyxy=[int(value) for value in bbox],
            mask=data.get("mask"),
            score=(None if data.get("score") is None else float(data["score"])),
            source=str(data.get("source", "") or ""),
            metadata=dict(data.get("metadata", {}) or {}),
            provider_kind=data.get("provider_kind"),
            backend_id=data.get("backend_id"),
            algorithm_id=data.get("algorithm_id"),
            contract_version=data.get("contract_version"),
        )
    bbox = getattr(payload, "bbox_xyxy")
    score = getattr(payload, "score", None)
    return RegionProposal(
        bbox_xyxy=[int(value) for value in bbox],
        mask=getattr(payload, "mask", None),
        score=None if score is None else float(score),
        source=str(getattr(payload, "source", "") or ""),
        metadata=dict(getattr(payload, "metadata", {}) or {}),
        provider_kind=getattr(payload, "provider_kind", None),
        backend_id=getattr(payload, "backend_id", None),
        algorithm_id=getattr(payload, "algorithm_id", None),
        contract_version=getattr(payload, "contract_version", None),
    )


_PROVIDER_IDENTITY_FIELDS = (
    "provider_kind",
    "backend_id",
    "algorithm_id",
    "contract_version",
)


def _finite_json_mapping(
    value: Any,
    *,
    source: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise TypeError(f"{source} must be a mapping")
    payload = copy.deepcopy(dict(value))
    try:
        json.dumps(payload, allow_nan=False)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{source} must contain finite JSON data") from error
    return payload


def _validate_prediction_identity(
    prediction: Any,
    *,
    provider_identity: Mapping[str, str],
    normalizer: Any,
    source: str,
) -> None:
    declared = {
        key: getattr(prediction, key, None) for key in _PROVIDER_IDENTITY_FIELDS
    }
    populated = {
        key: value
        for key, value in declared.items()
        if value is not None and str(value).strip()
    }
    if not populated:
        return
    if len(populated) != len(_PROVIDER_IDENTITY_FIELDS):
        missing = sorted(set(_PROVIDER_IDENTITY_FIELDS) - set(populated))
        raise ValueError(
            f"{source} provider identity is incomplete; missing {missing!r}"
        )
    prediction_identity = normalizer(
        **declared,
        source=f"{source} provider identity",
    )
    if prediction_identity != dict(provider_identity):
        raise ValueError(
            f"{source} provider identity conflicts with the bound provider: "
            f"prediction={prediction_identity!r}, "
            f"provider={dict(provider_identity)!r}"
        )


def _trusted_builtin_provider(
    provider: Any,
    *,
    role: str,
) -> bool:
    if role == "detector":
        from .backends.grounding_dino import (
            GroundingDinoDetectorAdapter,
        )

        trusted_types: tuple[type[Any], ...] = (GroundingDinoDetectorAdapter,)
    elif role == "segmenter":
        from .backends.sam2 import SAM2SegmenterAdapter

        trusted_types = (SAM2SegmenterAdapter,)
    else:  # pragma: no cover - internal invariant
        raise ValueError(f"unknown region provider role: {role!r}")
    if isinstance(provider, trusted_types):
        return True
    try:
        from .backends.combined import GroundingDinoSAM2Adapter
    except ImportError:  # pragma: no cover - import cycle guard
        return False
    return isinstance(provider, GroundingDinoSAM2Adapter)


def _validate_provider_binding(
    provider: Any,
    *,
    role: str,
    source: str,
    expected_identity: Mapping[str, str] | None,
) -> dict[str, str]:
    identity_reader = (
        region_detector_identity if role == "detector" else region_segmenter_identity
    )
    identity = identity_reader(provider, source=source)
    if identity["provider_kind"] == "builtin" and not _trusted_builtin_provider(
        provider, role=role
    ):
        raise ValueError(
            f"{source} is injected and may not claim provider_kind='builtin'"
        )
    if expected_identity is not None and identity != dict(expected_identity):
        raise ValueError(
            f"{source} identity conflicts with its runtime binding: "
            f"provider={identity!r}, "
            f"binding={dict(expected_identity)!r}"
        )
    return identity


def _detector_prediction(
    detector: RegionDetector,
    *,
    image_rgb: np.ndarray,
    prompt: str,
    expected_identity: Mapping[str, str] | None,
) -> tuple[RegionProposal, dict[str, Any]]:
    provider_identity = _validate_provider_binding(
        detector,
        role="detector",
        source="region detector",
        expected_identity=expected_identity,
    )
    proposal = _coerce_region_proposal(detector.detect(image_rgb, prompt))
    _validate_prediction_identity(
        proposal,
        provider_identity=provider_identity,
        normalizer=normalize_region_detector_identity,
        source="region detector prediction",
    )
    if len(proposal.bbox_xyxy) != 4:
        raise ValueError("region detector bbox_xyxy must contain four coordinates")
    bbox = np.asarray(proposal.bbox_xyxy, dtype=np.float64)
    if bbox.shape != (4,) or not np.all(np.isfinite(bbox)):
        raise ValueError(
            "region detector bbox_xyxy must contain four finite coordinates"
        )
    if proposal.score is not None and not np.isfinite(proposal.score):
        raise ValueError("region detector score must be finite")
    metadata = _finite_json_mapping(
        proposal.metadata,
        source="region detector prediction metadata",
    )
    proposal_mask = proposal.mask
    if proposal_mask is not None:
        normalized_mask = np.asarray(proposal_mask)
        if (
            normalized_mask.shape != np.asarray(image_rgb).shape[:2]
            or normalized_mask.dtype != np.bool_
        ):
            raise ValueError(
                "region detector proposal mask must be boolean and align "
                "with the input frame"
            )
        proposal.mask = normalized_mask.copy()
    provenance: dict[str, Any] = {
        **provider_identity,
        "source": str(proposal.source or ""),
        "score": (None if proposal.score is None else float(proposal.score)),
        "metadata": metadata,
    }
    return proposal, provenance


def _segmenter_prediction(
    segmenter: RegionSegmenter,
    *,
    image_rgb: np.ndarray,
    bbox_xyxy: list[int],
    prompt: str | None,
    expected_identity: Mapping[str, str] | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    provider_identity = _validate_provider_binding(
        segmenter,
        role="segmenter",
        source="region segmenter",
        expected_identity=expected_identity,
    )
    payload = segmenter.segment_from_bbox(
        image_rgb,
        bbox_xyxy,
        prompt=prompt,
    )
    if isinstance(payload, RegionSegmentationPrediction):
        prediction = payload
        mask = np.asarray(prediction.mask)
        metadata = _finite_json_mapping(
            prediction.metadata,
            source="region segmenter prediction metadata",
        )
        _validate_prediction_identity(
            prediction,
            provider_identity=provider_identity,
            normalizer=normalize_region_segmenter_identity,
            source="region segmenter prediction",
        )
    else:
        mask = np.asarray(payload)
        metadata = {}
    if mask.shape != np.asarray(image_rgb).shape[:2]:
        raise ValueError("region segmenter mask shape does not match frame")
    if mask.dtype != np.bool_:
        raise TypeError("region segmenter mask must have boolean dtype")
    return (
        mask.copy(),
        {
            **provider_identity,
            "metadata": metadata,
        },
    )


class RegionRuntime:
    """Reuse injected region backends while loading each one only on demand."""

    provider_kind = "builtin"
    backend_id = "composed_region_runtime"
    algorithm_id = "explicit_region_provider_composition"
    contract_version = REGION_RUNTIME_CONTRACT_VERSION

    def __init__(
        self,
        runtime_config: Mapping[str, Any] | None = None,
        *,
        device: str = "cuda",
        detector: RegionDetector | None = None,
        segmenter: RegionSegmenter | None = None,
        detector_factory: RegionBackendFactory | None = None,
        segmenter_factory: RegionBackendFactory | None = None,
        sampling_backends: Mapping[str, RegionQuerySampler] | None = None,
        detector_provider_identity: Mapping[str, Any] | None = None,
        segmenter_provider_identity: Mapping[str, Any] | None = None,
    ) -> None:
        if detector is not None and detector_factory is not None:
            raise ValueError("Provide either detector or detector_factory, not both.")
        if segmenter is not None and segmenter_factory is not None:
            raise ValueError("Provide either segmenter or segmenter_factory, not both.")
        self.runtime_config = copy.deepcopy(dict(runtime_config or {}))
        self.device = str(device)
        self._detector = detector
        self._segmenter = segmenter
        self._detector_factory = detector_factory
        self._segmenter_factory = segmenter_factory
        self._expected_detector_identity = (
            None
            if detector_provider_identity is None
            else normalize_region_detector_identity(
                **dict(detector_provider_identity),
                source="detector_provider_identity",
            )
        )
        self._expected_segmenter_identity = (
            None
            if segmenter_provider_identity is None
            else normalize_region_segmenter_identity(
                **dict(segmenter_provider_identity),
                source="segmenter_provider_identity",
            )
        )
        self._detector_identity: dict[str, str] | None = None
        self._segmenter_identity: dict[str, str] | None = None
        if self._detector is not None:
            self._detector_identity = _validate_provider_binding(
                self._detector,
                role="detector",
                source="injected region detector",
                expected_identity=self._expected_detector_identity,
            )
        if self._segmenter is not None:
            self._segmenter_identity = _validate_provider_binding(
                self._segmenter,
                role="segmenter",
                source="injected region segmenter",
                expected_identity=self._expected_segmenter_identity,
            )
        self._sampling_backends = builtin_region_query_samplers()
        self._sampling_identities = {
            name: region_query_sampler_identity(
                sampler,
                source=f"built-in region sampler {name!r}",
            )
            for name, sampler in self._sampling_backends.items()
        }
        self._released_sampler_ids: set[int] = set()
        for name, sampler in dict(sampling_backends or {}).items():
            normalized_name = str(name).strip().lower()
            if normalized_name not in self._sampling_backends:
                raise ValueError(
                    "region sampling_backends may replace only current "
                    "sampling methods: "
                    f"{sorted(self._sampling_backends)!r}"
                )
            if not callable(getattr(sampler, "sample", None)):
                raise TypeError(
                    f"region query sampler {normalized_name!r} must expose "
                    "sample(request)"
                )
            identity = region_query_sampler_identity(
                sampler,
                source=(f"region sampling_backends[{normalized_name!r}]"),
            )
            if identity["provider_kind"] == "builtin" and (
                identity != self._sampling_identities[normalized_name]
                or getattr(sampler, "space", None)
                != getattr(
                    self._sampling_backends[normalized_name],
                    "space",
                    None,
                )
            ):
                raise ValueError(
                    "an injected region sampler may declare "
                    "provider_kind='builtin' only when its complete identity "
                    "matches the built-in provider bound to "
                    f"{normalized_name!r}"
                )
            self._sampling_backends[normalized_name] = sampler
            self._sampling_identities[normalized_name] = identity

    def query_sampler_identities(self) -> dict[str, dict[str, str]]:
        """Return complete method-to-provider bindings for manifests."""

        return copy.deepcopy(
            {
                name: self._sampling_identities[name]
                for name in sorted(self._sampling_identities)
            }
        )

    def provider_identities(self) -> dict[str, dict[str, str]]:
        """Return detector and segmenter bindings known without model loads."""

        output: dict[str, dict[str, str]] = {}
        detector_identity = self._detector_identity or self._expected_detector_identity
        segmenter_identity = (
            self._segmenter_identity or self._expected_segmenter_identity
        )
        if detector_identity is not None:
            output["detector"] = copy.deepcopy(detector_identity)
        if segmenter_identity is not None:
            output["segmenter"] = copy.deepcopy(segmenter_identity)
        return output

    def get_detector(self) -> RegionDetector:
        """Instantiate the configured detector only for a detector route."""

        if self._detector is None:
            if self._detector_factory is None:
                raise RuntimeError(
                    "[region] GroundingDINO detector backend is not "
                    "configured; inject detector or detector_factory."
                )
            self._detector = self._detector_factory()
        if not callable(getattr(self._detector, "detect", None)):
            raise TypeError("Injected region detector must define detect().")
        self._detector_identity = _validate_provider_binding(
            self._detector,
            role="detector",
            source="region detector factory result",
            expected_identity=self._expected_detector_identity,
        )
        return self._detector

    def get_segmenter(self) -> RegionSegmenter:
        """Instantiate the configured segmenter only for a mask route."""

        if self._segmenter is None:
            if self._segmenter_factory is None:
                raise RuntimeError(
                    "[region] SAM2 segmenter backend is not "
                    "configured; inject segmenter or "
                    "segmenter_factory."
                )
            self._segmenter = self._segmenter_factory()
        if not callable(
            getattr(
                self._segmenter,
                "segment_from_bbox",
                None,
            )
        ):
            raise TypeError(
                "Injected region segmenter must define segment_from_bbox()."
            )
        self._segmenter_identity = _validate_provider_binding(
            self._segmenter,
            role="segmenter",
            source="region segmenter factory result",
            expected_identity=self._expected_segmenter_identity,
        )
        return self._segmenter

    def get_query_sampler(
        self,
        sampling_method: str,
    ) -> RegionQuerySampler:
        """Return the provider bound to one resolved sampling method."""

        normalized_method = str(sampling_method).strip().lower()
        sampler = self._sampling_backends.get(normalized_method)
        if sampler is None:
            raise ValueError(
                f"No region query sampler is configured for {normalized_method!r}"
            )
        if not callable(getattr(sampler, "sample", None)):
            raise TypeError(
                f"region query sampler {normalized_method!r} must expose "
                "sample(request)"
            )
        return sampler

    def _sample_queries(
        self,
        *,
        sampling_method: str,
        request: RegionSamplingRequest,
    ) -> RegionSamplingPrediction:
        normalized_method = str(sampling_method).strip().lower()
        prediction = invoke_region_query_sampler(
            self.get_query_sampler(sampling_method),
            request,
        )
        expected_identity = self._sampling_identities[normalized_method]
        actual_identity = {
            key: str(getattr(prediction, key)) for key in expected_identity
        }
        if actual_identity != expected_identity:
            raise ValueError(
                "region query sampler prediction identity conflicts with "
                f"its runtime binding: prediction={actual_identity!r}, "
                f"binding={expected_identity!r}"
            )
        return prediction

    def release_models(self) -> None:
        """Release only backends that have actually been instantiated."""

        released: set[int] = set()
        for attribute_name in (
            "_detector",
            "_segmenter",
        ):
            backend = getattr(self, attribute_name, None)
            release = getattr(backend, "release", None)
            if id(backend) not in released and callable(release):
                release()
                released.add(id(backend))
            setattr(self, attribute_name, None)
        for sampler in self._sampling_backends.values():
            sampler_id = id(sampler)
            if sampler_id in self._released_sampler_ids:
                continue
            if sampler_id in released:
                self._released_sampler_ids.add(sampler_id)
                continue
            release = getattr(sampler, "release", None)
            if callable(release):
                release()
                released.add(sampler_id)
                self._released_sampler_ids.add(sampler_id)

    def _resolve_bbox(
        self,
        *,
        frame_rgb: np.ndarray,
        prompt: str | None,
        manual_bbox: list[int] | None,
        bbox_source: str,
    ) -> tuple[
        list[int],
        list[int] | None,
        str,
        RegionProposal | None,
        dict[str, Any] | None,
    ]:
        if bbox_source == "manual":
            if manual_bbox is None:
                raise RuntimeError("manual bbox requested but not provided.")
            return manual_bbox.copy(), None, "manual", None, None

        if bbox_source == "grounding_dino":
            proposal, provenance = _detector_prediction(
                self.get_detector(),
                image_rgb=frame_rgb,
                prompt=str(prompt or ""),
                expected_identity=self._expected_detector_identity,
            )
            clipped_bbox = clip_bbox_to_frame(
                proposal.bbox_xyxy,
                frame_rgb.shape[1],
                frame_rgb.shape[0],
            )
            return (
                clipped_bbox,
                clipped_bbox.copy(),
                proposal.source,
                proposal,
                provenance,
            )

        if bbox_source == "auto":
            if str(prompt or "").strip():
                proposal, provenance = _detector_prediction(
                    self.get_detector(),
                    image_rgb=frame_rgb,
                    prompt=str(prompt or ""),
                    expected_identity=self._expected_detector_identity,
                )
                clipped_bbox = clip_bbox_to_frame(
                    proposal.bbox_xyxy,
                    frame_rgb.shape[1],
                    frame_rgb.shape[0],
                )
                return (
                    clipped_bbox,
                    clipped_bbox.copy(),
                    proposal.source,
                    proposal,
                    provenance,
                )
            if manual_bbox is not None:
                return manual_bbox.copy(), None, "manual", None, None
            raise RuntimeError(
                "No prompt/manual bbox available for auto bbox resolution."
            )
        raise ValueError(f"Unsupported bbox_source='{bbox_source}'.")

    def _resolve_mask_region(
        self,
        *,
        frame_rgb: np.ndarray,
        prompt: str | None,
        manual_bbox: list[int] | None,
        bbox_source: str,
        mask_backend: str | None,
    ) -> tuple[
        list[int],
        list[int] | None,
        np.ndarray,
        str,
        str,
        dict[str, Any],
    ]:
        if mask_backend is None:
            raise ValueError(
                "mask_backend must be resolved before mask region extraction."
            )
        (
            bbox_xyxy,
            detected_bbox,
            resolved_bbox_source,
            proposal,
            detector_provenance,
        ) = self._resolve_bbox(
            frame_rgb=frame_rgb,
            prompt=prompt,
            manual_bbox=manual_bbox,
            bbox_source=bbox_source,
        )
        provider_chain: dict[str, Any] = {}
        if detector_provenance is not None:
            provider_chain["detector"] = detector_provenance
        if proposal is not None and proposal.mask is not None:
            return (
                bbox_xyxy,
                detected_bbox,
                np.asarray(proposal.mask, dtype=bool),
                resolved_bbox_source,
                "proposal_mask",
                provider_chain,
            )
        if mask_backend != "sam2":
            raise ValueError(f"Unsupported mask_backend='{mask_backend}'.")
        mask, segmentation_provenance = _segmenter_prediction(
            self.get_segmenter(),
            image_rgb=frame_rgb,
            bbox_xyxy=bbox_xyxy,
            prompt=prompt,
            expected_identity=self._expected_segmenter_identity,
        )
        provider_chain["segmenter"] = segmentation_provenance
        return (
            bbox_xyxy,
            detected_bbox,
            mask,
            resolved_bbox_source,
            mask_backend,
            provider_chain,
        )

    def select(
        self,
        *,
        target_name: str,
        frame_rgb: np.ndarray,
        output_dir: str | Path,
        num_points: int,
        prompt: str | None = None,
        manual_bbox_xyxy: Sequence[int] | None = None,
        bbox_source: str = "auto",
        sampling_method: str = "auto",
        mask_backend: str = "auto",
        combine_mode: str = "intersect",
        mask_erosion_kernel_size: int = 11,
        init_depth: np.ndarray | None = None,
        camera: Any | None = None,
        precomputed_mask: np.ndarray | None = None,
        precomputed_bbox_xyxy: Sequence[int] | None = None,
        precomputed_source: str | None = None,
        save_debug: bool = True,
        write_artifacts: bool = True,
        seed: int = 42,
        selection_metadata: Mapping[str, Any] | None = None,
    ) -> RegionResult:
        """Run the current bbox/mask/sample pipeline with explicit backends."""

        frame = np.asarray(frame_rgb)
        if frame.ndim != 3 or frame.shape[-1] != 3:
            raise ValueError(f"Expected RGB image [H,W,3], got {frame.shape}")
        height, width = frame.shape[:2]
        output_path = Path(output_dir)
        if bool(write_artifacts):
            output_path.mkdir(parents=True, exist_ok=True)

        manual_bbox = None
        if manual_bbox_xyxy is not None:
            manual_bbox = clip_bbox_to_frame(
                manual_bbox_xyxy,
                width,
                height,
            )
        precomputed_mask_bool = (
            None
            if precomputed_mask is None
            else np.asarray(
                precomputed_mask,
                dtype=bool,
            )
        )
        precomputed_bbox = None
        if precomputed_bbox_xyxy is not None:
            precomputed_bbox = clip_bbox_to_frame(
                precomputed_bbox_xyxy,
                width,
                height,
            )
        if precomputed_mask_bool is not None and precomputed_mask_bool.shape != (
            height,
            width,
        ):
            source_height, source_width = precomputed_mask_bool.shape[:2]
            import cv2

            precomputed_mask_bool = cv2.resize(
                precomputed_mask_bool.astype(np.uint8),
                (width, height),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
            if (
                precomputed_bbox_xyxy is not None
                and source_width > 0
                and source_height > 0
            ):
                scale_x = float(width) / float(source_width)
                scale_y = float(height) / float(source_height)
                x0, y0, x1, y1 = [float(value) for value in precomputed_bbox_xyxy]
                precomputed_bbox = clip_bbox_to_frame(
                    [
                        round(x0 * scale_x),
                        round(y0 * scale_y),
                        round(x1 * scale_x),
                        round(y1 * scale_y),
                    ],
                    width,
                    height,
                )

        if precomputed_mask_bool is not None and not np.any(precomputed_mask_bool):
            raise RuntimeError(f"[{target_name}] precomputed mask is empty.")

        resolved_bbox_source = resolve_bbox_source(
            bbox_source=bbox_source,
            prompt=prompt,
            manual_bbox=manual_bbox,
        )
        use_precomputed_bbox = (
            precomputed_mask_bool is not None and resolved_bbox_source == "auto"
        )
        normalized_precomputed_source = str(precomputed_source or "precomputed")
        if precomputed_mask_bool is not None:
            if use_precomputed_bbox:
                resolved_bbox_source = normalized_precomputed_source
            if str(sampling_method).strip().lower() in {"", "auto"}:
                sampling_method = "mask_3d_fps"
            if str(mask_backend).strip().lower() in {
                "",
                "auto",
            }:
                mask_backend = "precomputed"

        resolved_sampling_method = resolve_sampling_method(
            sampling_method=sampling_method,
            mask_backend=mask_backend,
            bbox_source=resolved_bbox_source,
            prompt=prompt,
        )
        resolved_mask_backend = resolve_mask_backend(
            mask_backend=mask_backend,
            bbox_source=resolved_bbox_source,
            sampling_method=resolved_sampling_method,
        )
        self.get_query_sampler(resolved_sampling_method)

        detected_bbox: list[int] | None = None
        final_bbox: list[int] | None = None
        source = ""
        provider_chain: dict[str, Any] = {}
        resolved_selection_metadata = dict(selection_metadata or {})
        sampling_started = False
        try:
            if resolved_sampling_method == "bbox_gaussian":
                if use_precomputed_bbox:
                    if precomputed_bbox is None:
                        final_bbox = bbox_from_mask(precomputed_mask_bool)
                        sampling_bbox_source = "precomputed_mask"
                    else:
                        final_bbox = precomputed_bbox.copy()
                        sampling_bbox_source = "precomputed_bbox"
                    detected_bbox = final_bbox.copy()
                else:
                    (
                        final_bbox,
                        detected_bbox,
                        resolved_bbox_source,
                        _proposal,
                        detector_provenance,
                    ) = self._resolve_bbox(
                        frame_rgb=frame,
                        prompt=prompt,
                        manual_bbox=manual_bbox,
                        bbox_source=resolved_bbox_source,
                    )
                    if detector_provenance is not None:
                        provider_chain["detector"] = detector_provenance
                    sampling_bbox_source = resolved_bbox_source
                if precomputed_mask_bool is None:
                    base_mask = bbox_to_mask(
                        (height, width),
                        final_bbox,
                    )
                else:
                    base_mask = precomputed_mask_bool.copy()
                    resolved_selection_metadata["precomputed_region"] = {
                        "source": normalized_precomputed_source,
                        "sampling_method": "bbox_gaussian",
                        "sampling_bbox_source": (sampling_bbox_source),
                    }
                eroded_mask = base_mask.copy()
                sampling_started = True
                sampling_prediction = self._sample_queries(
                    sampling_method=resolved_sampling_method,
                    request=RegionSamplingRequest(
                        frame_shape_hw=(height, width),
                        bbox_xyxy=final_bbox,
                        num_points=int(num_points),
                        seed=int(seed),
                    ),
                )
                sampled = sampling_prediction.sample
                sampling_mask = sampled.sampling_mask
                if precomputed_mask_bool is None:
                    source = f"{resolved_bbox_source}/bbox_center"
                else:
                    source = f"{resolved_bbox_source}/bbox_gaussian+precomputed_mask"
            else:
                if (
                    precomputed_mask_bool is not None
                    and resolved_mask_backend == "precomputed"
                ):
                    base_mask = precomputed_mask_bool.copy()
                    final_bbox = precomputed_bbox or bbox_from_mask(base_mask)
                    detected_bbox = precomputed_bbox or bbox_from_mask(base_mask)
                    resolved_bbox_source = str(
                        precomputed_source or resolved_bbox_source or "precomputed"
                    )
                else:
                    (
                        final_bbox,
                        detected_bbox,
                        base_mask,
                        resolved_bbox_source,
                        resolved_mask_backend,
                        region_provider_chain,
                    ) = self._resolve_mask_region(
                        frame_rgb=frame,
                        prompt=prompt,
                        manual_bbox=manual_bbox,
                        bbox_source=resolved_bbox_source,
                        mask_backend=resolved_mask_backend,
                    )
                    provider_chain.update(region_provider_chain)

                (
                    combined_mask,
                    combine_suffix,
                ) = combine_mask_with_manual_bbox(
                    mask=base_mask,
                    manual_bbox=manual_bbox,
                    combine_mode=combine_mode,
                    shape_hw=(height, width),
                )
                if not np.any(combined_mask):
                    raise RuntimeError(f"[{target_name}] combined mask is empty.")
                eroded_mask = erode_mask(
                    combined_mask,
                    kernel_size=int(mask_erosion_kernel_size),
                )
                if not np.any(eroded_mask):
                    eroded_mask = combined_mask.copy()
                    combine_suffix = (
                        f"{combine_suffix}+erosion_fallback"
                        if combine_suffix
                        else "erosion_fallback"
                    )
                sampling_started = True
                sampling_prediction = self._sample_queries(
                    sampling_method=resolved_sampling_method,
                    request=RegionSamplingRequest(
                        frame_shape_hw=(height, width),
                        mask=eroded_mask,
                        num_points=int(num_points),
                        init_depth=init_depth,
                        camera=camera,
                        seed=int(seed),
                    ),
                )
                sampled = sampling_prediction.sample
                sampling_mask = sampled.sampling_mask
                final_bbox = bbox_from_mask(combined_mask)
                base_mask = combined_mask
                mask_source = str(resolved_mask_backend)
                if not mask_source.endswith("_mask"):
                    mask_source = f"{mask_source}_mask"
                source = f"{resolved_bbox_source}/{mask_source}"
                if combine_suffix:
                    source = f"{source}+{combine_suffix}"
        except Exception as error:
            if (
                sampling_started
                or isinstance(error, ValueError)
                or manual_bbox is None
                or resolved_bbox_source != "manual"
            ):
                raise
            final_bbox = manual_bbox.copy()
            detected_bbox = None
            base_mask = bbox_to_mask(
                (height, width),
                final_bbox,
            )
            eroded_mask = base_mask.copy()
            sampling_prediction = self._sample_queries(
                sampling_method="bbox_gaussian",
                request=RegionSamplingRequest(
                    frame_shape_hw=(height, width),
                    bbox_xyxy=final_bbox,
                    num_points=int(num_points),
                    seed=int(seed),
                ),
            )
            sampled = sampling_prediction.sample
            sampling_mask = sampled.sampling_mask
            source = "manual/bbox_center_fallback_after_region_error"
            resolved_bbox_source = "manual"
            resolved_sampling_method = "bbox_gaussian"
            resolved_mask_backend = None

        provider_chain["query_sampler"] = sampling_prediction.identity_dict()

        output_json = (
            (output_path / f"{target_name}_region.json").as_posix()
            if bool(write_artifacts)
            else ""
        )
        result = RegionResult(
            target_name=target_name,
            source=source,
            bbox_source=resolved_bbox_source,
            sampling_method=resolved_sampling_method,
            mask_backend=resolved_mask_backend,
            prompt=((str(prompt).strip() or None) if prompt is not None else None),
            manual_bbox_xyxy=(
                None if manual_bbox is None else [int(value) for value in manual_bbox]
            ),
            detected_bbox_xyxy=(
                None
                if detected_bbox is None
                else [int(value) for value in detected_bbox]
            ),
            final_bbox_xyxy=[int(value) for value in final_bbox],
            mask=base_mask.astype(bool),
            eroded_mask=eroded_mask.astype(bool),
            sampling_mask=sampling_mask.astype(bool),
            sampled_points_xy=(sampled.points_xy.astype(np.float32)),
            sampled_points_xyz=(
                None
                if sampled.points_xyz is None
                else sampled.points_xyz.astype(np.float32)
            ),
            used_depth_for_sampling=bool(sampled.used_depth_for_sampling),
            tracking_input="points",
            output_json=output_json,
            selection_metadata=_finite_json_mapping(
                resolved_selection_metadata,
                source="region selection_metadata",
            ),
            sampling_backend=sampling_prediction.identity_dict(),
            provider_chain=provider_chain,
        )
        if bool(write_artifacts):
            _write_region_files(
                result=result,
                frame_rgb=frame,
                output_dir=output_path,
                save_debug=bool(save_debug),
            )
        return result

    def _select_from_precomputed(
        self,
        *,
        target_name: str,
        frame_rgb: np.ndarray,
        output_dir: str | Path,
        num_points: int,
        sampling_method: str,
        init_depth: np.ndarray | None,
        camera: Any | None,
        precomputed_region: PrecomputedRegion,
        seed: int,
        save_debug: bool,
        write_artifacts: bool,
        selection_metadata: Mapping[str, Any] | None = None,
    ) -> RegionResult:
        region_config = dict(self.runtime_config.get("region", {}))
        durable_selection_metadata = {
            "selector_mode": "simulation",
            "matched_names": list(precomputed_region.matched_names),
            "compact_labels": [
                int(value) for value in precomputed_region.compact_labels
            ],
            "precomputed_mask_source": precomputed_region.source,
            **dict(selection_metadata or {}),
        }
        return self.select(
            target_name=target_name,
            frame_rgb=frame_rgb,
            output_dir=output_dir,
            num_points=num_points,
            prompt=None,
            manual_bbox_xyxy=None,
            bbox_source="auto",
            sampling_method=sampling_method,
            mask_backend="precomputed",
            combine_mode=str(
                region_config.get(
                    "combine_mode",
                    "intersect",
                )
                or "intersect"
            ),
            mask_erosion_kernel_size=int(
                region_config.get(
                    "mask_erosion_kernel_size",
                    11,
                )
            ),
            init_depth=init_depth,
            camera=camera,
            precomputed_mask=precomputed_region.mask,
            precomputed_bbox_xyxy=(precomputed_region.bbox_xyxy),
            precomputed_source=(precomputed_region.source),
            save_debug=save_debug,
            write_artifacts=write_artifacts,
            seed=seed,
            selection_metadata=durable_selection_metadata,
        )

    def select_target(
        self,
        *,
        target_name: str,
        frame_rgb: np.ndarray,
        output_dir: str | Path,
        num_points: int,
        environment_config: Mapping[str, Any],
        target_config: Mapping[str, Any] | None = None,
        precomputed_region: PrecomputedRegion | None = None,
        init_depth: np.ndarray | None = None,
        camera: Any | None = None,
        seed: int = 42,
        write_artifacts: bool = True,
    ) -> RegionResult | None:
        """Run current selector routing with environment assets injected."""

        normalized_target_name = str(target_name).strip().lower()
        if normalized_target_name not in {"eef", "obj"}:
            raise ValueError(f"Unsupported target_name='{normalized_target_name}'.")
        if target_config is None:
            resolved_target_config = dict(
                self.runtime_config.get(
                    "targets",
                    {},
                ).get(normalized_target_name, {})
            )
        else:
            resolved_target_config = dict(target_config)
        region_config = dict(self.runtime_config.get("region", {}))
        prompt = resolve_region_prompt(
            target_name=normalized_target_name,
            config=dict(environment_config),
            target_config=resolved_target_config,
        )
        (
            selector_mode,
            visual_bbox_source,
            visual_sampling,
        ) = resolve_region_target_plan(
            target_name=normalized_target_name,
            target_config=resolved_target_config,
        )
        if selector_mode == "off":
            return None

        if selector_mode == "auto" and precomputed_region is not None:
            return self._select_from_precomputed(
                target_name=normalized_target_name,
                frame_rgb=frame_rgb,
                output_dir=output_dir,
                num_points=int(num_points),
                sampling_method=visual_sampling,
                init_depth=init_depth,
                camera=camera,
                precomputed_region=precomputed_region,
                seed=int(seed),
                save_debug=bool(
                    region_config.get(
                        "save_debug",
                        False,
                    )
                ),
                write_artifacts=bool(write_artifacts),
            )

        if selector_mode == "simulation":
            if precomputed_region is None:
                if normalized_target_name != "obj":
                    raise RuntimeError(
                        "[region] "
                        f"{normalized_target_name} requested "
                        "simulation selector, but init "
                        "segmentation mask is unavailable."
                    )
            else:
                return self._select_from_precomputed(
                    target_name=normalized_target_name,
                    frame_rgb=frame_rgb,
                    output_dir=output_dir,
                    num_points=int(num_points),
                    sampling_method=visual_sampling,
                    init_depth=init_depth,
                    camera=camera,
                    precomputed_region=precomputed_region,
                    seed=int(seed),
                    save_debug=bool(
                        region_config.get(
                            "save_debug",
                            False,
                        )
                    ),
                    write_artifacts=bool(write_artifacts),
                )

        manual_bbox = resolve_config_bbox(resolved_target_config)
        if visual_bbox_source == "manual" and manual_bbox is None:
            raise RuntimeError(
                f"[region] {normalized_target_name} uses "
                "visual.bbox_source=manual, but "
                "region.targets."
                f"{normalized_target_name}.bbox_xyxy is "
                "missing."
            )
        if visual_bbox_source == "grounding_dino" and not str(prompt).strip():
            if normalized_target_name == "obj":
                return None
            raise RuntimeError(
                f"[region] {normalized_target_name} visual "
                "selector requires a prompt when "
                "bbox_source=grounding_dino."
            )

        try:
            visual_selection_metadata: dict[str, Any] = {
                "selector_mode": "visual",
                "visual_bbox_source": visual_bbox_source,
                "visual_sampling": visual_sampling,
            }
            if manual_bbox is not None:
                visual_selection_metadata["manual_bbox_xyxy"] = [
                    int(value) for value in manual_bbox
                ]
            result = self.select(
                target_name=normalized_target_name,
                frame_rgb=frame_rgb,
                output_dir=output_dir,
                num_points=int(num_points),
                prompt=prompt,
                manual_bbox_xyxy=manual_bbox,
                bbox_source=visual_bbox_source,
                sampling_method=visual_sampling,
                mask_backend=("none" if visual_sampling == "bbox_gaussian" else "sam2"),
                combine_mode=str(
                    region_config.get(
                        "combine_mode",
                        "intersect",
                    )
                    or "intersect"
                ),
                mask_erosion_kernel_size=int(
                    region_config.get(
                        "mask_erosion_kernel_size",
                        11,
                    )
                ),
                init_depth=init_depth,
                camera=camera,
                save_debug=bool(
                    region_config.get(
                        "save_debug",
                        False,
                    )
                ),
                write_artifacts=bool(write_artifacts),
                seed=int(seed),
                selection_metadata=visual_selection_metadata,
            )
        except RuntimeError as error:
            if normalized_target_name != "obj" or precomputed_region is None:
                raise
            result = self._select_from_precomputed(
                target_name=normalized_target_name,
                frame_rgb=frame_rgb,
                output_dir=output_dir,
                num_points=int(num_points),
                sampling_method=visual_sampling,
                init_depth=init_depth,
                camera=camera,
                precomputed_region=precomputed_region,
                seed=int(seed),
                save_debug=bool(
                    region_config.get(
                        "save_debug",
                        False,
                    )
                ),
                write_artifacts=bool(write_artifacts),
                selection_metadata={
                    "fallback_from_selector_mode": "visual",
                    "fallback_reason": str(error),
                    "visual_bbox_source": visual_bbox_source,
                    "visual_sampling": visual_sampling,
                },
            )
            return result
        return result


def _draw_debug_overlay(
    *,
    frame_rgb: np.ndarray,
    result: RegionResult,
    mask: np.ndarray | None = None,
    include_points: bool = False,
    title: str = "",
) -> np.ndarray:
    cv2 = __import__("cv2")
    canvas = cv2.cvtColor(
        np.asarray(frame_rgb, dtype=np.uint8),
        cv2.COLOR_RGB2BGR,
    )

    if mask is not None:
        selected_pixels = np.asarray(mask, dtype=bool)
        canvas[selected_pixels] = 0.65 * canvas[selected_pixels] + 0.35 * np.asarray(
            [0, 255, 0]
        )

    manual_box = result.manual_bbox_xyxy
    if manual_box is not None:
        left, top, right, bottom = manual_box
        cv2.rectangle(
            canvas,
            (left, top),
            (right, bottom),
            (255, 0, 0),
            1,
        )
        cv2.putText(
            canvas,
            "manual",
            (left, max(14, top - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 0),
            1,
            cv2.LINE_AA,
        )

    detected_box = result.detected_bbox_xyxy
    if detected_box is not None:
        left, top, right, bottom = detected_box
        cv2.rectangle(
            canvas,
            (left, top),
            (right, bottom),
            (255, 0, 255),
            1,
        )
        cv2.putText(
            canvas,
            "detected",
            (
                left,
                min(canvas.shape[0] - 8, bottom + 14),
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (255, 0, 255),
            1,
            cv2.LINE_AA,
        )

    left, top, right, bottom = result.final_bbox_xyxy
    cv2.rectangle(
        canvas,
        (left, top),
        (right, bottom),
        (0, 255, 255),
        2,
    )
    cv2.putText(
        canvas,
        "final",
        (left, max(28, top - 8)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 255),
        1,
        cv2.LINE_AA,
    )

    if include_points:
        for x_value, y_value in result.sampled_points_xy:
            cv2.circle(
                canvas,
                (int(x_value), int(y_value)),
                2,
                (0, 0, 255),
                thickness=-1,
            )

    if title:
        cv2.putText(
            canvas,
            title,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            canvas,
            title,
            (12, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (0, 0, 0),
            1,
            cv2.LINE_AA,
        )

    return canvas


def _write_region_files(
    *,
    result: RegionResult,
    frame_rgb: np.ndarray,
    output_dir: Path,
    save_debug: bool,
) -> None:
    if save_debug:
        import cv2

        base = output_dir / result.target_name
        artifacts = {
            "mask": str(base.with_name(f"{result.target_name}_mask.png")),
            "mask_eroded": str(base.with_name(f"{result.target_name}_mask_eroded.png")),
            "sampling_mask": str(
                base.with_name(f"{result.target_name}_sampling_mask.png")
            ),
            "bbox_overlay": str(
                base.with_name(f"{result.target_name}_bbox_overlay.png")
            ),
            "mask_overlay": str(
                base.with_name(f"{result.target_name}_mask_overlay.png")
            ),
            "eroded_overlay": str(
                base.with_name(f"{result.target_name}_eroded_overlay.png")
            ),
            "sampling_overlay": str(
                base.with_name(f"{result.target_name}_sampling_overlay.png")
            ),
            "summary": str(base.with_name(f"{result.target_name}_summary.png")),
        }
        cv2.imwrite(
            artifacts["mask"],
            result.mask.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            artifacts["mask_eroded"],
            result.eroded_mask.astype(np.uint8) * 255,
        )
        cv2.imwrite(
            artifacts["sampling_mask"],
            result.sampling_mask.astype(np.uint8) * 255,
        )

        bbox_overlay = _draw_debug_overlay(
            frame_rgb=frame_rgb,
            result=result,
            mask=None,
            include_points=False,
            title=f"{result.target_name}: bbox",
        )
        mask_overlay = _draw_debug_overlay(
            frame_rgb=frame_rgb,
            result=result,
            mask=result.mask,
            include_points=False,
            title=f"{result.target_name}: mask",
        )
        eroded_overlay = _draw_debug_overlay(
            frame_rgb=frame_rgb,
            result=result,
            mask=result.eroded_mask,
            include_points=False,
            title=f"{result.target_name}: eroded mask",
        )
        sampling_overlay = _draw_debug_overlay(
            frame_rgb=frame_rgb,
            result=result,
            mask=result.sampling_mask,
            include_points=True,
            title=f"{result.target_name}: sampling",
        )
        cv2.imwrite(
            artifacts["bbox_overlay"],
            bbox_overlay,
        )
        cv2.imwrite(
            artifacts["mask_overlay"],
            mask_overlay,
        )
        cv2.imwrite(
            artifacts["eroded_overlay"],
            eroded_overlay,
        )
        cv2.imwrite(
            artifacts["sampling_overlay"],
            sampling_overlay,
        )

        summary_top = np.concatenate(
            [bbox_overlay, mask_overlay],
            axis=1,
        )
        summary_bottom = np.concatenate(
            [eroded_overlay, sampling_overlay],
            axis=1,
        )
        summary = np.concatenate(
            [summary_top, summary_bottom],
            axis=0,
        )
        cv2.imwrite(artifacts["summary"], summary)
        result.artifacts.update(artifacts)

    Path(result.output_json).parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    with Path(result.output_json).open(
        "w",
        encoding="utf-8",
    ) as handle:
        json.dump(result.to_dict(), handle, indent=2)


__all__ = [
    "BaseRegionDetector",
    "BaseRegionSegmenter",
    "PrecomputedRegion",
    "RegionBackendFactory",
    "RegionDetector",
    "RegionProposal",
    "RegionResult",
    "RegionRuntime",
    "RegionSegmentationPrediction",
    "RegionSegmenter",
]
