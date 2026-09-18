"""Local and polling-API image-to-video backend templates."""

from __future__ import annotations

from collections.abc import Mapping
import os
from pathlib import Path
from typing import Any

from dream_exe.generation import (
    BaseImageToVideoBackend,
    PollingImageToVideoBackend,
)


class ExampleLocalVideoBackend(BaseImageToVideoBackend):
    """Run an in-process model and write a complete MP4 to output_path."""

    backend_id = "example_local_video"

    def generate(
        self,
        *,
        image_path: Path,
        prompt: str,
        output_path: Path,
        seed: int,
        parameters: Mapping[str, Any],
    ) -> Mapping[str, Any]:
        del image_path, prompt, output_path, seed, parameters
        raise NotImplementedError(
            "Run local inference, write output_path, and return safe provenance"
        )


class ExamplePollingVideoBackend(PollingImageToVideoBackend):
    """Implement provider transport; the base owns timeout/retry/cleanup."""

    backend_id = "example_polling_video"

    def __init__(self, *, api_key_env: str, **polling_options: Any) -> None:
        super().__init__(**polling_options)
        self.api_key_env = str(api_key_env)

    def _api_key(self) -> str:
        value = str(os.environ.get(self.api_key_env, "") or "").strip()
        if not value:
            raise RuntimeError(f"set environment variable {self.api_key_env}")
        return value

    def submit(
        self,
        *,
        image_path: Path,
        prompt: str,
        seed: int,
        parameters: Mapping[str, Any],
    ) -> Any:
        del image_path, prompt, seed, parameters
        self._api_key()
        raise NotImplementedError("Submit one provider job and return its handle")

    def poll(self, job: Any) -> Mapping[str, Any]:
        del job
        self._api_key()
        raise NotImplementedError("Return at least {'status': '<provider-state>'}")

    def download(
        self,
        job: Any,
        status: Mapping[str, Any],
        output_path: Path,
    ) -> Mapping[str, Any] | None:
        del job, status, output_path
        self._api_key()
        raise NotImplementedError("Download the completed MP4 to output_path")


__all__ = ["ExampleLocalVideoBackend", "ExamplePollingVideoBackend"]
