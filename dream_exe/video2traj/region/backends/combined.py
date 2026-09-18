"""Combined GroundingDINO and SAM2 adapter for region selection."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np

from ..contract import (
    region_detector_identity,
    region_segmenter_identity,
)
from ..runtime import RegionProposal
from .grounding_dino import GroundingDinoDetectorAdapter
from .sam2 import SAM2SegmenterAdapter


class GroundingDinoSAM2Adapter:
    """One object implementing both backend protocols for ``RegionRuntime``."""

    def __init__(
        self,
        *,
        grounding_dino: GroundingDinoDetectorAdapter,
        sam2: SAM2SegmenterAdapter,
    ) -> None:
        self.grounding_dino = grounding_dino
        self.sam2 = sam2
        self._active = False

    def detect(
        self,
        image_rgb: np.ndarray,
        prompt: str,
    ) -> RegionProposal:
        self._active = True
        return self.grounding_dino.detect(
            image_rgb,
            prompt,
        )

    def detector_provider_identity(self) -> dict[str, str]:
        """Expose the wrapped detector identity for compatibility only."""

        return region_detector_identity(self.grounding_dino)

    def segment_from_bbox(
        self,
        image_rgb: np.ndarray,
        bbox_xyxy: list[int],
        prompt: str | None = None,
    ) -> np.ndarray:
        self._active = True
        return self.sam2.segment_from_bbox(
            image_rgb,
            bbox_xyxy,
            prompt=prompt,
        )

    def segmenter_provider_identity(self) -> dict[str, str]:
        """Expose the wrapped segmenter identity for compatibility only."""

        return region_segmenter_identity(self.sam2)

    def release(self) -> None:
        if not self._active:
            return
        self.grounding_dino.release()
        self.sam2.release()
        gc.collect()

        torch_modules: dict[int, tuple[Any, bool]] = {}
        for backend in (
            self.grounding_dino,
            self.sam2,
        ):
            runtime = backend._runtime
            if runtime is not None and "torch" in runtime:
                torch = runtime["torch"]
                existing = torch_modules.get(
                    id(torch),
                    (torch, False),
                )
                torch_modules[id(torch)] = (
                    torch,
                    existing[1] or str(backend.device).startswith("cuda"),
                )
        for torch, uses_cuda in torch_modules.values():
            if uses_cuda:
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        torch.cuda.ipc_collect()
                except Exception:
                    pass
        self._active = False


__all__ = ["GroundingDinoSAM2Adapter"]
