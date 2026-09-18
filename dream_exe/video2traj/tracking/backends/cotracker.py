"""Lazy CoTracker adapter for the portable tracking contract."""

from __future__ import annotations

import copy
import hashlib
import importlib
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ...runtime.provider_origin import (
    prepend_source_roots,
    require_modules_under_roots,
)
from ..core import (
    COTRACKER_BACKEND_ID,
    TRACKING_BACKEND_CONTRACT_VERSION,
    infer_tracking_grid_size,
    prepare_tracking_query_points,
    resolve_tracking_query_mode,
)


RuntimeLoader = Callable[[], Mapping[str, Any]]
_PREDICTOR_CACHE: dict[tuple[str, str | None, str, int], Any] = {}
_SHA256_PATTERN = re.compile(r"^[0-9a-f]{64}$")
_OFFICIAL_RUNTIME_NAMESPACE = "cotracker"


def _normalize_checkpoint_sha256(value: Any) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise TypeError(
            "CoTracker checkpoint_sha256 must be 64 lowercase hexadecimal digits."
        )
    digest = value
    if not _SHA256_PATTERN.fullmatch(digest):
        raise ValueError(
            "CoTracker checkpoint_sha256 must be 64 lowercase hexadecimal digits."
        )
    return digest


def _sha256_file(path: Path) -> str:
    if not path.is_file():
        raise FileNotFoundError(
            f"CoTracker checkpoint file not found: {path.as_posix()}"
        )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _checkpoint_provenance(
    *,
    checkpoint_path: Path,
    expected_sha256: str | None,
) -> dict[str, Any]:
    record: dict[str, Any] = {
        "path": checkpoint_path.as_posix(),
        "algorithm": "sha256",
        "expected_sha256": expected_sha256,
        "actual_sha256": None,
        "status": "declaration_only",
    }
    if expected_sha256 is None:
        return record
    actual_sha256 = _sha256_file(checkpoint_path)
    record["actual_sha256"] = actual_sha256
    if actual_sha256 != expected_sha256:
        record["status"] = "mismatch"
        raise ValueError(
            "CoTracker checkpoint SHA-256 mismatch: "
            f"expected {expected_sha256}, got {actual_sha256}"
        )
    record["status"] = "verified"
    return record


def _default_runtime_loader() -> dict[str, Any]:
    torch = importlib.import_module("torch")
    functional = importlib.import_module("torch.nn.functional")
    predictor_module = importlib.import_module("cotracker.predictor")
    model_utils = importlib.import_module("cotracker.models.core.model_utils")
    visualizer_module = importlib.import_module("cotracker.utils.visualizer")
    return {
        "torch": torch,
        "functional": functional,
        "predictor_cls": predictor_module.CoTrackerPredictor,
        "get_points_on_a_grid": model_utils.get_points_on_a_grid,
        "visualizer_cls": visualizer_module.Visualizer,
        "_provider_modules": (
            predictor_module,
            model_utils,
            visualizer_module,
        ),
    }


class CoTrackerBackend:
    """Current-compatible CoTracker execution with lazy dependencies."""

    provider_kind = "builtin"
    backend_id = COTRACKER_BACKEND_ID
    contract_version = TRACKING_BACKEND_CONTRACT_VERSION

    def __init__(
        self,
        *,
        checkpoint_path: str | Path,
        device: str | None = None,
        linewidth: int = 1,
        source_roots: Sequence[str | Path] = (),
        checkpoint_sha256: str | None = None,
        runtime_loader: RuntimeLoader | None = None,
    ) -> None:
        checkpoint_text = str(checkpoint_path or "").strip()
        if not checkpoint_text:
            raise ValueError("CoTracker checkpoint_path is required.")
        checkpoint = Path(checkpoint_text).expanduser().resolve()
        self.checkpoint_path = checkpoint.as_posix()
        self.checkpoint_sha256 = _normalize_checkpoint_sha256(checkpoint_sha256)
        self.device = None if device is None else str(device or "").strip()
        self.linewidth = int(linewidth)
        self.source_roots = tuple(Path(path).expanduser() for path in source_roots)
        self._uses_official_runtime = runtime_loader is None
        resolved_runtime_loader = (
            runtime_loader if runtime_loader is not None else _default_runtime_loader
        )
        runtime_namespace = (
            _OFFICIAL_RUNTIME_NAMESPACE
            if runtime_loader is None
            else str(
                getattr(
                    resolved_runtime_loader,
                    "runtime_namespace",
                    "caller_supplied",
                )
                or "caller_supplied"
            ).strip()
        )
        self._provider_provenance = {
            "runtime_namespace": runtime_namespace,
            "checkpoint": _checkpoint_provenance(
                checkpoint_path=checkpoint,
                expected_sha256=self.checkpoint_sha256,
            ),
        }
        self._runtime_loader = resolved_runtime_loader
        self._runtime: dict[str, Any] | None = None

    @property
    def provider_provenance(self) -> dict[str, Any]:
        """Return detached provider and checkpoint reproducibility metadata."""

        return copy.deepcopy(self._provider_provenance)

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        resolved_roots = []
        for source_root in self.source_roots:
            resolved = source_root.resolve()
            if not resolved.is_dir():
                raise RuntimeError(f"CoTracker source directory not found: {resolved}")
            resolved_roots.append(resolved)
        prepend_source_roots(resolved_roots)
        try:
            runtime = dict(self._runtime_loader())
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "Failed to import CoTracker dependencies; missing "
                f"module '{exc.name or '<unknown>'}'."
            ) from exc
        required = {
            "torch",
            "functional",
            "predictor_cls",
            "get_points_on_a_grid",
            "visualizer_cls",
        }
        missing = sorted(required.difference(runtime))
        if missing:
            raise RuntimeError(
                "CoTracker runtime loader is missing: " + ", ".join(missing)
            )
        if self._uses_official_runtime and self.source_roots:
            provider_modules = runtime.get("_provider_modules", ())
            if not isinstance(provider_modules, tuple):
                raise RuntimeError(
                    "CoTracker official runtime did not expose provider modules"
                )
            require_modules_under_roots(
                provider_modules,
                source_roots=self.source_roots,
                provider="CoTracker",
            )
        self._runtime = runtime
        return runtime

    def _resolve_device(self, runtime: Mapping[str, Any]) -> str:
        if self.device:
            return self.device
        torch = runtime["torch"]
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        return self.device

    def _get_predictor(
        self,
        runtime: Mapping[str, Any],
        *,
        device: str,
    ) -> Any:
        predictor_class = runtime["predictor_cls"]
        key = (
            self.checkpoint_path,
            self.checkpoint_sha256,
            device,
            id(predictor_class),
        )
        predictor = _PREDICTOR_CACHE.get(key)
        if predictor is None:
            predictor = predictor_class(checkpoint=self.checkpoint_path).to(device)
            _PREDICTOR_CACHE[key] = predictor
        return predictor

    @staticmethod
    def _prepare_mask_grid_inputs(
        *,
        runtime: Mapping[str, Any],
        predictor: Any,
        segmentation_mask: Any,
        frame_shape: tuple[int, int],
        num_points: int,
        grid_size: int,
        device: str,
    ) -> tuple[Any, np.ndarray, int]:
        height, width = frame_shape
        mask = np.asarray(segmentation_mask, dtype=bool)
        if mask.shape != (height, width):
            raise ValueError(
                f"segmentation_mask must have shape {(height, width)}, got {mask.shape}"
            )
        if not np.any(mask):
            raise ValueError("segmentation_mask is empty.")

        resolved_grid_size = (
            int(grid_size)
            if int(grid_size) > 0
            else infer_tracking_grid_size(num_points)
        )
        torch = runtime["torch"]
        functional = runtime["functional"]
        segmentation_tensor = torch.from_numpy(mask.astype(np.float32)).to(device)[
            None, None
        ]
        interpolated_mask = functional.interpolate(
            segmentation_tensor,
            tuple(predictor.interp_shape),
            mode="nearest",
        )
        grid_points = runtime["get_points_on_a_grid"](
            resolved_grid_size,
            predictor.interp_shape,
            device=device,
        )
        point_mask = interpolated_mask[0, 0][
            grid_points[0, :, 1].round().long(),
            grid_points[0, :, 0].round().long(),
        ].bool()
        grid_points = grid_points[:, point_mask]
        if grid_points.shape[1] == 0:
            raise ValueError("segmentation_mask selected no CoTracker grid points.")
        query_points = grid_points[0].clone()
        query_points[:, 0] *= (width - 1) / max(
            1,
            predictor.interp_shape[1] - 1,
        )
        query_points[:, 1] *= (height - 1) / max(
            1,
            predictor.interp_shape[0] - 1,
        )
        return (
            segmentation_tensor,
            query_points.detach().cpu().numpy().astype(np.float32),
            resolved_grid_size,
        )

    def _predict_with_visualization_context(
        self,
        *,
        video_frames: Any,
        region_bbox_xyxy: Any = None,
        num_points: int = 50,
        seed: int = 42,
        query_points_xy: Any = None,
        segmentation_mask: Any = None,
        query_mode: str = "auto",
        grid_size: int = 0,
    ) -> tuple[
        tuple[Any, Any, np.ndarray, str],
        Any,
        Any,
    ]:
        frames = list(video_frames)
        if not frames:
            raise ValueError("video_frames is empty.")

        runtime = self._load_runtime()
        torch = runtime["torch"]
        device = self._resolve_device(runtime)
        predictor = self._get_predictor(
            runtime,
            device=device,
        )
        frame_shape = frames[0].shape[:2]
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()
        video_array = np.ascontiguousarray(np.asarray(frames))
        video_tensor = (
            torch.from_numpy(video_array).permute(0, 3, 1, 2)[None].float().to(device)
        )
        resolved_query_mode = resolve_tracking_query_mode(
            query_mode=query_mode,
            query_points_xy=query_points_xy,
            segmentation_mask=segmentation_mask,
        )

        segmentation_for_visualization = None
        if resolved_query_mode == "mask_grid":
            (
                segmentation_tensor,
                resolved_query_points,
                resolved_grid_size,
            ) = self._prepare_mask_grid_inputs(
                runtime=runtime,
                predictor=predictor,
                segmentation_mask=segmentation_mask,
                frame_shape=frame_shape,
                num_points=num_points,
                grid_size=grid_size,
                device=device,
            )
            predicted_tracks, predicted_visibility = predictor(
                video_tensor,
                queries=None,
                segm_mask=segmentation_tensor,
                grid_size=resolved_grid_size,
            )
            segmentation_for_visualization = segmentation_tensor
        else:
            resolved_query_points = prepare_tracking_query_points(
                frame_shape=frame_shape,
                query_points_xy=query_points_xy,
                region_bbox_xyxy=region_bbox_xyxy,
                num_points=num_points,
                seed=seed,
            )
            queries = torch.tensor(
                [
                    [0, float(x_coordinate), float(y_coordinate)]
                    for x_coordinate, y_coordinate in resolved_query_points
                ],
                dtype=torch.float32,
                device=device,
            ).unsqueeze(0)
            predicted_tracks, predicted_visibility = predictor(
                video_tensor,
                queries=queries,
            )

        return (
            (
                predicted_tracks,
                predicted_visibility,
                resolved_query_points,
                resolved_query_mode,
            ),
            video_tensor,
            segmentation_for_visualization,
        )

    def predict(
        self,
        *,
        video_frames: Any,
        region_bbox_xyxy: Any = None,
        num_points: int = 50,
        seed: int = 42,
        query_points_xy: Any = None,
        segmentation_mask: Any = None,
        query_mode: str = "auto",
        grid_size: int = 0,
    ) -> tuple[Any, Any, np.ndarray, str]:
        """Predict tracks without accepting or publishing artifact paths."""

        result, _, _ = self._predict_with_visualization_context(
            video_frames=video_frames,
            region_bbox_xyxy=region_bbox_xyxy,
            num_points=num_points,
            seed=seed,
            query_points_xy=query_points_xy,
            segmentation_mask=segmentation_mask,
            query_mode=query_mode,
            grid_size=grid_size,
        )
        return result

    def track(
        self,
        *,
        video_frames: Any,
        output_dir: str,
        region_bbox_xyxy: Any = None,
        num_points: int = 50,
        filename: str = "points_cloud",
        seed: int = 42,
        query_points_xy: Any = None,
        segmentation_mask: Any = None,
        query_mode: str = "auto",
        grid_size: int = 0,
        write_artifacts: bool = True,
    ) -> tuple[Any, Any, np.ndarray, str]:
        """Preserve the current tracking and visualization call contract."""

        (
            result,
            video_tensor,
            segmentation_for_visualization,
        ) = self._predict_with_visualization_context(
            video_frames=video_frames,
            region_bbox_xyxy=region_bbox_xyxy,
            num_points=num_points,
            seed=seed,
            query_points_xy=query_points_xy,
            segmentation_mask=segmentation_mask,
            query_mode=query_mode,
            grid_size=grid_size,
        )
        if bool(write_artifacts):
            runtime = self._load_runtime()
            visualizer = runtime["visualizer_cls"](
                save_dir=output_dir,
                linewidth=self.linewidth,
            )
            visualizer.visualize(
                video_tensor,
                result[0],
                result[1],
                segm_mask=segmentation_for_visualization,
                filename=str(filename),
            )
        return result


__all__ = ["CoTrackerBackend"]
