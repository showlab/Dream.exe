"""Minimal factory template for a local or custom-protocol VLM."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from dream_exe.evaluation.vlm import BaseVLMBackend


class ExampleVLMBackend(BaseVLMBackend):
    """Return raw model text; Dream.exe owns parsing, retry, and caching."""

    backend_id = "example_vlm"

    def __init__(self, *, model_name: str) -> None:
        self.model_name = str(model_name)

    def inference_identity(self) -> Mapping[str, Any]:
        return {
            "format": "example.vlm-inference",
            "model": self.model_name,
            "transport": "replace-with-local-or-api-transport",
        }

    def infer(
        self,
        prompt: str,
        media_path: str | Path,
        generation_options: Mapping[str, Any] | None = None,
    ) -> str:
        del prompt, media_path, generation_options
        raise NotImplementedError(
            "Call the model here and return its unparsed response text"
        )


__all__ = ["ExampleVLMBackend"]
