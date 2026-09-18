"""Lazy box-conditioned SAM2 segmenter backend."""

from __future__ import annotations

import importlib
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np

from ...runtime.provider_origin import (
    prepend_source_roots,
    require_modules_under_roots,
)
from ..contract import REGION_SEGMENTER_CONTRACT_VERSION
from .common import RuntimeLoader, portable_path, require_runtime_keys


_SAM2_RUNTIME_KEYS = (
    "torch",
    "initialize_config_module",
    "global_hydra",
    "build_sam2",
    "predictor_factory",
)

_portable_path = portable_path
_require_runtime_keys = require_runtime_keys


class SAM2SegmenterAdapter:
    """Current box-conditioned SAM2 segmenter with lazy model loading."""

    provider_kind = "builtin"
    backend_id = "sam2_box_segmenter"
    algorithm_id = "box_conditioned_mask_segmentation"
    contract_version = REGION_SEGMENTER_CONTRACT_VERSION

    def __init__(
        self,
        *,
        source_root: str | Path | None = None,
        config_name: str,
        checkpoint_path: str | Path,
        device: str,
        runtime_loader: RuntimeLoader | None = None,
    ) -> None:
        self.source_root = _portable_path(source_root)
        self.config_name = str(config_name or "")
        self.checkpoint_path = _portable_path(checkpoint_path)
        self.device = str(device)
        self._runtime_loader = runtime_loader
        self._runtime: dict[str, Any] | None = None
        self._torch: Any = None
        self._predictor: Any = None

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        if not self.checkpoint_path:
            raise RuntimeError(
                "SAM2 offline-only mode requires a local checkpoint path."
            )
        if not Path(self.checkpoint_path).is_file():
            raise FileNotFoundError(
                f"SAM2 checkpoint not found: {self.checkpoint_path}"
            )
        if self.source_root and not Path(self.source_root).is_dir():
            raise FileNotFoundError(
                f"SAM2 source directory not found: {self.source_root}"
            )

        if self._runtime_loader is not None:
            runtime = dict(self._runtime_loader())
        else:
            if self.source_root:
                prepend_source_roots((self.source_root,))
            try:
                hydra_module = importlib.import_module("hydra")
                global_hydra_module = importlib.import_module("hydra.core.global_hydra")
                build_module = importlib.import_module("sam2.build_sam")
                predictor_module = importlib.import_module("sam2.sam2_image_predictor")
                runtime = {
                    "torch": importlib.import_module("torch"),
                    "initialize_config_module": getattr(
                        hydra_module,
                        "initialize_config_module",
                    ),
                    "global_hydra": getattr(
                        global_hydra_module,
                        "GlobalHydra",
                    ),
                    "build_sam2": getattr(
                        build_module,
                        "build_sam2",
                    ),
                    "predictor_factory": getattr(
                        predictor_module,
                        "SAM2ImagePredictor",
                    ),
                }
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Failed to import SAM2. Install the pinned provider "
                    "requirement or supply an explicit source_root plus its "
                    "runtime dependencies."
                ) from exc
            if self.source_root:
                require_modules_under_roots(
                    (build_module, predictor_module),
                    source_roots=(self.source_root,),
                    provider="SAM2",
                )

        self._runtime = _require_runtime_keys(
            runtime,
            backend="SAM2",
            keys=_SAM2_RUNTIME_KEYS,
        )
        return self._runtime

    def _load(self) -> None:
        if self._predictor is not None:
            return
        runtime = self._load_runtime()
        runtime["global_hydra"].instance().clear()
        runtime["initialize_config_module"](
            "sam2",
            version_base="1.2",
        )
        model = runtime["build_sam2"](
            config_file=self.config_name,
            ckpt_path=self.checkpoint_path,
            device=self.device,
        )
        self._predictor = runtime["predictor_factory"](model)
        self._torch = runtime["torch"]

    def segment_from_bbox(
        self,
        image_rgb: np.ndarray,
        bbox_xyxy: list[int],
        prompt: str | None = None,
    ) -> np.ndarray:
        del prompt
        self._load()

        image = np.asarray(image_rgb, dtype=np.uint8)
        bbox = np.asarray(
            bbox_xyxy,
            dtype=np.float32,
        )
        autocast_context = (
            self._torch.autocast(
                "cuda",
                dtype=self._torch.bfloat16,
            )
            if self.device.startswith("cuda")
            else nullcontext()
        )
        with (
            self._torch.inference_mode(),
            autocast_context,
        ):
            self._predictor.set_image(image)
            masks, _, _ = self._predictor.predict(
                point_coords=None,
                point_labels=None,
                box=bbox,
                multimask_output=False,
            )
        return np.asarray(masks[0], dtype=bool)

    def release(self) -> None:
        predictor = self._predictor
        if predictor is not None:
            model = getattr(predictor, "model", None)
            if model is None:
                model = getattr(
                    predictor,
                    "_model",
                    None,
                )
            if model is not None and hasattr(model, "to"):
                model.to("cpu")
        self._predictor = None


__all__ = ["SAM2SegmenterAdapter"]
