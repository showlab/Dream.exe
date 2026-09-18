"""Public input, output, and identity contracts for depth providers."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping, Sequence
from typing import Any, Protocol, runtime_checkable

import numpy as np


DEPTH_ESTIMATOR_CONTRACT_VERSION = "depth_estimator"
EXTERNAL_DEPTH_SELECTION_PREFIX = "external:"


def external_depth_backend_id(value: Any) -> str | None:
    """Return a validated external backend ID or ``None`` for built-ins."""

    selection = str(value or "").strip().lower()
    if not selection.startswith(EXTERNAL_DEPTH_SELECTION_PREFIX):
        return None
    backend_id = selection[len(EXTERNAL_DEPTH_SELECTION_PREFIX) :].strip()
    if (
        not backend_id
        or backend_id.startswith(".")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in backend_id
        )
    ):
        raise ValueError(
            "external depth selection must be 'external:<backend_id>' using [a-z0-9._-]"
        )
    return backend_id


def external_depth_selection(backend_id: Any) -> str:
    """Build the canonical pipeline identity for an external provider."""

    selection = EXTERNAL_DEPTH_SELECTION_PREFIX + str(backend_id or "").strip().lower()
    parsed = external_depth_backend_id(selection)
    assert parsed is not None
    return f"{EXTERNAL_DEPTH_SELECTION_PREFIX}{parsed}"


@runtime_checkable
class DepthBackend(Protocol):
    """Lower-level model provider used by a lazy estimator adapter."""

    def infer(
        self,
        video_frames: Sequence[Any],
        target_fps: float,
        *,
        fp32: bool,
        input_size: int,
        intrinsics: np.ndarray | None = None,
        extrinsics: np.ndarray | None = None,
    ) -> Any:
        """Return depths plus optional fps, validity, and provenance fields."""


class BaseDepthBackend(ABC):
    """Minimal external depth provider template.

    Outputs may be a mapping/object with ``depths`` or another value accepted
    by the current depth normalizer; the canonical stack is ``[T,H,W]``.
    """

    provider_kind = "external"
    backend_id = ""
    contract_version = DEPTH_ESTIMATOR_CONTRACT_VERSION

    @abstractmethod
    def infer(
        self,
        video_frames: Sequence[Any],
        target_fps: float,
        *,
        fp32: bool,
        input_size: int,
        intrinsics: np.ndarray | None = None,
        extrinsics: np.ndarray | None = None,
    ) -> Any:
        """Return depth predictions aligned to the input frames."""


@runtime_checkable
class DepthEstimator(Protocol):
    """Pipeline-facing callable after provider/configuration binding."""

    def __call__(
        self,
        *,
        video_frames: Sequence[Any],
        target_fps: float,
        intrinsics: np.ndarray | None = None,
        extrinsics: np.ndarray | None = None,
    ) -> Any:
        """Return depths or ``(depths, fps, info[, aux])``."""


def validate_external_depth_runtime_identity(
    runtime_config: Mapping[str, Any],
    *,
    source: str,
) -> dict[str, Any]:
    """Validate truthful identity for an explicitly external estimator."""

    if not isinstance(runtime_config, Mapping):
        raise TypeError(f"{source} must be a mapping")
    runtime = dict(runtime_config)
    if str(runtime.get("provider_kind", "") or "").strip().lower() != "external":
        raise ValueError(f"{source}.provider_kind must be 'external'")
    backend_id = str(runtime.get("backend_id", "") or "").strip()
    canonical = external_depth_selection(backend_id)
    parsed_id = external_depth_backend_id(canonical)
    assert parsed_id is not None
    contract_version = str(runtime.get("contract_version", "") or "").strip()
    if contract_version != DEPTH_ESTIMATOR_CONTRACT_VERSION:
        raise ValueError(
            f"{source}.contract_version must be {DEPTH_ESTIMATOR_CONTRACT_VERSION!r}"
        )
    runtime["provider_kind"] = "external"
    runtime["backend_id"] = parsed_id
    runtime["contract_version"] = contract_version
    runtime["model_name"] = canonical
    return runtime


__all__ = [
    "BaseDepthBackend",
    "DEPTH_ESTIMATOR_CONTRACT_VERSION",
    "EXTERNAL_DEPTH_SELECTION_PREFIX",
    "DepthBackend",
    "DepthEstimator",
    "external_depth_backend_id",
    "external_depth_selection",
    "validate_external_depth_runtime_identity",
]
