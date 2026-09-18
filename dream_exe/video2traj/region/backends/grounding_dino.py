"""Lazy GroundingDINO detector backend."""

from __future__ import annotations

import importlib
from pathlib import Path
from typing import Any

import numpy as np

from ...runtime.provider_origin import (
    prepend_source_roots,
    require_modules_under_roots,
)
from ..contract import REGION_DETECTOR_CONTRACT_VERSION
from ..runtime import RegionProposal
from ..selection import clip_bbox_to_frame, normalize_region_prompt
from .common import RuntimeLoader, portable_path, require_runtime_keys


_GROUNDING_RUNTIME_KEYS = (
    "torch",
    "pil_image",
    "transforms",
    "build_model",
    "clean_state_dict",
    "config_loader",
)

_portable_path = portable_path
_require_runtime_keys = require_runtime_keys


class GroundingDinoDetectorAdapter:
    """Current offline GroundingDINO detector with lazy model loading."""

    provider_kind = "builtin"
    backend_id = "grounding_dino_box_detector"
    algorithm_id = "text_conditioned_box_detection"
    contract_version = REGION_DETECTOR_CONTRACT_VERSION

    def __init__(
        self,
        *,
        source_root: str | Path | None = None,
        config_path: str | Path,
        checkpoint_path: str | Path,
        text_encoder_path: str | Path,
        device: str,
        box_threshold: float = 0.4,
        text_threshold: float = 0.3,
        runtime_loader: RuntimeLoader | None = None,
    ) -> None:
        self.source_root = _portable_path(source_root)
        self.config_path = _portable_path(config_path)
        self.checkpoint_path = _portable_path(checkpoint_path)
        self.text_encoder_path = _portable_path(text_encoder_path)
        self.device = str(device)
        self.box_threshold = float(box_threshold)
        self.text_threshold = float(text_threshold)
        self._runtime_loader = runtime_loader
        self._runtime: dict[str, Any] | None = None
        self._model: Any = None
        self._torch: Any = None
        self._pil_image: Any = None
        self._transform: Any = None

    @staticmethod
    def _prompt_candidates(prompt: str) -> list[str]:
        text = str(prompt or "").strip().lower()
        text = text.rstrip(".,;:!?")
        if not text:
            return []

        candidates = [text]
        words = [word for word in text.split() if word]
        if len(words) > 1:
            candidates.append(" ".join(words[1:]))
            candidates.append(" ".join(words[-2:]))
            candidates.append(words[-1])

        descriptor_words = {
            "red",
            "blue",
            "green",
            "yellow",
            "orange",
            "white",
            "black",
            "gray",
            "grey",
            "small",
            "large",
            "big",
            "robot",
        }
        if len(words) > 1 and words[0] in descriptor_words:
            candidates.append(" ".join(words[1:]))

        seen: set[str] = set()
        output: list[str] = []
        for candidate in candidates:
            normalized = str(candidate).strip()
            if not normalized or normalized in seen:
                continue
            seen.add(normalized)
            output.append(normalized)
        return output

    @staticmethod
    def _preprocess_caption(caption: str) -> str:
        normalized = str(caption).strip().lower()
        if not normalized.endswith("."):
            normalized = f"{normalized}."
        return normalized

    @staticmethod
    def _cxcywh_to_xyxy(
        box_cxcywh: Any,
        width: int,
        height: int,
    ) -> list[float]:
        center_x, center_y, box_width, box_height = [
            float(value) for value in box_cxcywh
        ]
        return [
            (center_x - box_width / 2.0) * float(width),
            (center_y - box_height / 2.0) * float(height),
            (center_x + box_width / 2.0) * float(width),
            (center_y + box_height / 2.0) * float(height),
        ]

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        if (
            not self.config_path
            or not self.checkpoint_path
            or not self.text_encoder_path
        ):
            raise ValueError(
                "Local GroundingDINO backend requires config, checkpoint, "
                "and text encoder paths."
            )
        if not Path(self.config_path).is_file():
            raise FileNotFoundError(
                f"GroundingDINO config not found: {self.config_path}"
            )
        if not Path(self.checkpoint_path).is_file():
            raise FileNotFoundError(
                f"GroundingDINO checkpoint not found: {self.checkpoint_path}"
            )
        if not Path(self.text_encoder_path).is_dir():
            raise FileNotFoundError(
                "GroundingDINO text encoder directory not found: "
                f"{self.text_encoder_path}"
            )
        if self.source_root and not Path(self.source_root).is_dir():
            raise FileNotFoundError(
                f"GroundingDINO source directory not found: {self.source_root}"
            )

        if self._runtime_loader is not None:
            runtime = dict(self._runtime_loader())
        else:
            if self.source_root:
                prepend_source_roots((self.source_root,))
            try:
                transforms = importlib.import_module(
                    "groundingdino.datasets.transforms"
                )
                model_module = importlib.import_module("groundingdino.models")
                misc_module = importlib.import_module("groundingdino.util.misc")
                config_module = importlib.import_module("groundingdino.util.slconfig")
                runtime = {
                    "torch": importlib.import_module("torch"),
                    "pil_image": importlib.import_module("PIL.Image"),
                    "transforms": transforms,
                    "build_model": getattr(
                        model_module,
                        "build_model",
                    ),
                    "clean_state_dict": getattr(
                        misc_module,
                        "clean_state_dict",
                    ),
                    "config_loader": getattr(
                        config_module,
                        "SLConfig",
                    ).fromfile,
                }
            except Exception as exc:  # pragma: no cover
                raise RuntimeError(
                    "Failed to import GroundingDINO. Install the pinned "
                    "provider requirement or supply an explicit source_root "
                    "plus its runtime dependencies."
                ) from exc
            if self.source_root:
                require_modules_under_roots(
                    (
                        transforms,
                        model_module,
                        misc_module,
                        config_module,
                    ),
                    source_roots=(self.source_root,),
                    provider="GroundingDINO",
                )

        self._runtime = _require_runtime_keys(
            runtime,
            backend="GroundingDINO",
            keys=_GROUNDING_RUNTIME_KEYS,
        )
        return self._runtime

    def _load(self) -> None:
        if self._model is not None:
            return
        runtime = self._load_runtime()
        arguments = runtime["config_loader"](self.config_path)
        arguments.device = self.device
        arguments.text_encoder_type = self.text_encoder_path
        model = runtime["build_model"](arguments)
        checkpoint = runtime["torch"].load(
            self.checkpoint_path,
            map_location="cpu",
        )
        model.load_state_dict(
            runtime["clean_state_dict"](checkpoint["model"]),
            strict=False,
        )
        model.eval()
        self._model = model.to(self.device)
        self._torch = runtime["torch"]
        self._pil_image = runtime["pil_image"]
        transforms = runtime["transforms"]
        self._transform = transforms.Compose(
            [
                transforms.RandomResize(
                    [800],
                    max_size=1333,
                ),
                transforms.ToTensor(),
                transforms.Normalize(
                    [0.485, 0.456, 0.406],
                    [0.229, 0.224, 0.225],
                ),
            ]
        )

    def _detect_candidate(
        self,
        image_rgb: np.ndarray,
        prompt: str,
    ) -> RegionProposal:
        self._load()
        normalized_prompt = normalize_region_prompt(prompt)
        if normalized_prompt is None:
            raise ValueError("GroundingDINO requires a non-empty prompt.")

        image = np.asarray(image_rgb, dtype=np.uint8)
        image_pil = self._pil_image.fromarray(image)
        image_tensor, _ = self._transform(
            image_pil,
            None,
        )
        image_tensor = image_tensor.to(self.device)
        caption = self._preprocess_caption(normalized_prompt)

        with self._torch.no_grad():
            outputs = self._model(
                image_tensor[None],
                captions=[caption],
            )

        prediction_logits = outputs["pred_logits"].sigmoid()[0]
        prediction_boxes = outputs["pred_boxes"][0]
        keep = prediction_logits.max(dim=1).values > float(self.box_threshold)
        if not bool(keep.any()):
            raise RuntimeError(
                f"GroundingDINO did not find any box for prompt '{normalized_prompt}'."
            )

        boxes = prediction_boxes[keep].detach().cpu()
        scores = prediction_logits[keep].max(dim=1).values.detach().cpu().numpy()
        index = int(np.argmax(scores)) if len(scores) > 0 else 0
        height, width = image.shape[:2]
        bbox_xyxy = clip_bbox_to_frame(
            self._cxcywh_to_xyxy(
                boxes[index].tolist(),
                width,
                height,
            ),
            width,
            height,
        )
        score = float(scores[index]) if len(scores) > 0 else None
        return RegionProposal(
            bbox_xyxy=bbox_xyxy,
            score=score,
            source="grounding_dino",
            provider_kind=self.provider_kind,
            backend_id=self.backend_id,
            algorithm_id=self.algorithm_id,
            contract_version=self.contract_version,
        )

    def detect(
        self,
        image_rgb: np.ndarray,
        prompt: str,
    ) -> RegionProposal:
        self._load()
        last_error: Exception | None = None
        best_proposal: RegionProposal | None = None
        for candidate in self._prompt_candidates(prompt):
            try:
                proposal = self._detect_candidate(
                    image_rgb,
                    candidate,
                )
            except Exception as exc:
                last_error = exc
                continue
            if best_proposal is None or (
                (proposal.score or 0.0) > (best_proposal.score or 0.0)
            ):
                best_proposal = proposal

        if best_proposal is not None:
            return best_proposal
        if last_error is not None:
            raise last_error
        raise RuntimeError("GroundingDINO requires a non-empty prompt.")

    def release(self) -> None:
        model = self._model
        if model is not None and hasattr(model, "to"):
            model.to("cpu")
        self._model = None
        self._transform = None


__all__ = ["GroundingDinoDetectorAdapter"]
