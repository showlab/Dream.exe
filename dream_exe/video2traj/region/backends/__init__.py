"""Lazy GroundingDINO and SAM2 region-model backends.

All source roots, checkpoints, and devices are caller-provided. Importing this
package does not import Torch, OpenCV, GroundingDINO, SAM2, or Hydra.
"""

from __future__ import annotations

# Preserve the established monkeypatch seam used to observe release cleanup.
import gc as gc

from .combined import GroundingDinoSAM2Adapter
from .common import RuntimeLoader
from .grounding_dino import GroundingDinoDetectorAdapter
from .sam2 import SAM2SegmenterAdapter
from .sampling import (
    BBoxGaussianQuerySampler,
    MaskFarthestPointQuerySampler,
    builtin_region_query_samplers,
)


# Preserve the historical public class path after the implementation split.
GroundingDinoDetectorAdapter.__module__ = __name__
GroundingDinoSAM2Adapter.__module__ = __name__
SAM2SegmenterAdapter.__module__ = __name__


__all__ = [
    "BBoxGaussianQuerySampler",
    "GroundingDinoDetectorAdapter",
    "GroundingDinoSAM2Adapter",
    "MaskFarthestPointQuerySampler",
    "RuntimeLoader",
    "SAM2SegmenterAdapter",
    "builtin_region_query_samplers",
]
