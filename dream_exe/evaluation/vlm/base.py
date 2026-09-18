"""Public contracts and templates for replaceable VLM inference backends."""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Protocol, runtime_checkable


VLM_BACKEND_CONTRACT_VERSION = "vlm_backend"


@runtime_checkable
class VLMBackend(Protocol):
    """Structural VLM interface consumed by the saved-artifact evaluators."""

    provider_kind: str
    backend_id: str
    contract_version: str

    def infer(
        self,
        prompt: str,
        media_path: str | Path,
        generation_options: Mapping[str, Any] | None = None,
    ) -> str:
        """Return the provider's raw response text."""

    def inference_identity(self) -> Mapping[str, Any]:
        """Return credential-free behavior and implementation identity."""


class BaseVLMBackend(ABC):
    """Minimal template for an external local or API-backed VLM.

    Subclasses keep credentials in process state and must never return them
    from :meth:`inference_identity`.  The evaluator stores the raw string from
    :meth:`infer` before applying its existing parser and cache contracts.
    """

    __slots__ = ()

    provider_kind = "external"
    backend_id = ""
    contract_version = VLM_BACKEND_CONTRACT_VERSION

    @abstractmethod
    def infer(
        self,
        prompt: str,
        media_path: str | Path,
        generation_options: Mapping[str, Any] | None = None,
    ) -> str:
        """Return raw response text for one prompt and saved media file."""

    @abstractmethod
    def inference_identity(self) -> Mapping[str, Any]:
        """Return a strict-JSON, credential-free identity mapping."""

    def __call__(
        self,
        prompt: str,
        media_path: str | Path,
        generation_options: Mapping[str, Any] | None = None,
    ) -> str:
        result = self.infer(prompt, media_path, generation_options)
        if not isinstance(result, str):
            raise TypeError("VLM backend infer(...) must return raw text")
        return result


__all__ = [
    "BaseVLMBackend",
    "VLMBackend",
    "VLM_BACKEND_CONTRACT_VERSION",
]
