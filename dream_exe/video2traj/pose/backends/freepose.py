"""Lazy FreePose adapter for the portable pose backend contract.

The adapter preserves the current per-frame FreePose behavior while keeping
all checkout, cache, and Torch Hub locations explicit.  Importing this module
does not import Torch, trimesh, or the vendored FreePose package.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..contract import (
    POSE_BACKEND_CONTRACT_VERSION,
    PoseBackendRequest,
    PosePrediction,
    invalid_candidate,
    mask_to_bbox,
    pose_candidate_from_cam,
    project_mesh_bbox,
    score_to_unit_interval,
)


RuntimeLoader = Callable[[], Mapping[str, Any]]
_REQUIRED_RUNTIME_KEYS = (
    "torch",
    "trimesh",
    "estimator_factory",
    "renderer_factory",
    "crop_resize_factory",
)


def _explicit_path(value: str | Path, *, label: str) -> str:
    text = str(value or "").strip()
    if not text:
        raise ValueError(f"{label} must be explicitly provided.")
    return Path(text).expanduser().resolve(strict=False).as_posix()


def _explicit_source_roots(
    values: Sequence[str | Path] | str | Path,
) -> tuple[str, ...]:
    if isinstance(values, (str, Path)):
        roots = (values,)
    else:
        roots = tuple(values)
    if not roots:
        raise ValueError("FreePose source_roots must be explicitly provided.")
    return tuple(_explicit_path(value, label="FreePose source root") for value in roots)


class FreePoseBackend:
    """Stateful FreePose runtime adapter with lazy dependency loading."""

    name = "freepose"
    provider_kind = "builtin"
    backend_id = "freepose"
    algorithm_id = "freepose_template_retrieval"
    contract_version = POSE_BACKEND_CONTRACT_VERSION

    def __init__(
        self,
        *,
        source_roots: Sequence[str | Path] | str | Path,
        cache_dir: str | Path,
        torch_home: str | Path,
        device: str = "cuda",
        debug_dir: str | Path = "",
        n_coarse_poses: int = 600,
        n_fine_poses: int = 10000,
        bbox_extend: float = 0.05,
        neighborhood: float = 15.0,
        layer: int = 22,
        batch_size: int = 128,
        mask_scores: bool = False,
        runtime_loader: RuntimeLoader | None = None,
    ) -> None:
        self.source_roots = _explicit_source_roots(source_roots)
        self.cache_dir = _explicit_path(
            cache_dir,
            label="FreePose cache_dir",
        )
        self.torch_home = _explicit_path(
            torch_home,
            label="FreePose torch_home",
        )
        self.device = str(device or "cuda")
        self.debug_dir = str(debug_dir or "")
        self.n_coarse_poses = int(n_coarse_poses)
        self.n_fine_poses = int(n_fine_poses)
        self.bbox_extend = float(bbox_extend)
        self.neighborhood = float(neighborhood)
        self.layer = int(layer)
        self.batch_size = int(batch_size)
        self.mask_scores = bool(mask_scores)
        self._runtime_loader = runtime_loader
        self._runtime: dict[str, Any] | None = None
        self._estimator: Any = None
        self._renderer: Any = None
        self._crop_resize: Any = None
        self._crop_resize_factory: Any = None
        self._mesh: Any = None
        self._mesh_path = ""
        self._mesh_vertices: np.ndarray | None = None
        self._template_dict: dict[str, Any] | None = None
        self._torch: Any = None
        self._trimesh: Any = None

    def _load_runtime(self) -> dict[str, Any]:
        if self._runtime is not None:
            return self._runtime
        if not self.device.startswith("cuda"):
            raise RuntimeError(
                "FreePose currently requires CUDA in this vendored setup."
            )
        os.environ.setdefault("TORCH_HOME", self.torch_home)

        if self._runtime_loader is not None:
            runtime = dict(self._runtime_loader())
        else:
            for source_root in self.source_roots:
                path = Path(source_root)
                if not path.is_dir():
                    raise RuntimeError(
                        f"FreePose source directory not found: {path.as_posix()}"
                    )
            for source_root in self.source_roots:
                if source_root not in sys.path:
                    sys.path.insert(0, source_root)
            try:
                torch = importlib.import_module("torch")
                runtime = {
                    "torch": torch,
                    "trimesh": importlib.import_module("trimesh"),
                    "estimator_factory": getattr(
                        importlib.import_module(
                            "modules.pose.freepose.src.pipeline."
                            "estimators.online_pose_estimator"
                        ),
                        "DinoOnlinePoseEstimator",
                    ),
                    "renderer_factory": getattr(
                        importlib.import_module(
                            "modules.pose.freepose.src.pipeline.retrieval.renderer"
                        ),
                        "MeshRenderer",
                    ),
                    "crop_resize_factory": getattr(
                        importlib.import_module(
                            "modules.pose.freepose.src.utils.bbox_utils"
                        ),
                        "CropResizePad",
                    ),
                }
            except Exception as exc:  # pragma: no cover - runtime specific
                raise RuntimeError(
                    "Failed to import FreePose dependencies from the "
                    "explicit source_roots. Install the vendored FreePose "
                    "runtime and its Torch, trimesh, pyrender, and DINO "
                    "dependencies."
                ) from exc

        missing = [key for key in _REQUIRED_RUNTIME_KEYS if key not in runtime]
        if missing:
            raise RuntimeError(
                "FreePose runtime loader is missing: " + ", ".join(missing)
            )
        torch = runtime["torch"]
        try:
            torch.hub.set_dir(self.torch_home)
        except Exception:
            pass
        self._runtime = runtime
        return runtime

    def _load(self) -> None:
        if self._estimator is not None:
            return
        runtime = self._load_runtime()
        estimator_factory = runtime["estimator_factory"]
        renderer_factory = runtime["renderer_factory"]
        crop_resize_factory = runtime["crop_resize_factory"]
        self._estimator = estimator_factory(
            n_coarse_poses=int(self.n_coarse_poses),
            n_fine_poses=int(self.n_fine_poses),
            cache_dir=self.cache_dir,
        )
        self._renderer = renderer_factory(int(self.n_coarse_poses))
        self._crop_resize = crop_resize_factory(
            target_size=420,
            orig_size=(420, 420),
            bbox_extend=self.bbox_extend,
        )
        self._crop_resize_factory = crop_resize_factory
        self._torch = runtime["torch"]
        self._trimesh = runtime["trimesh"]

    def _ensure_mesh(self, mesh_path: str) -> None:
        resolved_mesh_path = Path(mesh_path).expanduser().resolve().as_posix()
        if (
            self._mesh is not None
            and self._mesh_path == resolved_mesh_path
            and self._template_dict is not None
        ):
            return

        self._load()
        mesh = self._trimesh.load(
            resolved_mesh_path,
            force="mesh",
        )
        if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
            raise RuntimeError(f"Loaded empty EEF mesh: {resolved_mesh_path}")

        renders = self._renderer.render(mesh.copy())
        templates, _poses, _masks = self._renderer.generate_proposals(renders)
        intrinsic = self._torch.tensor(
            [
                [600.0, 0.0, 210.0],
                [0.0, 600.0, 210.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=self._torch.float32,
        )
        depths = [
            self._torch.from_numpy(np.asarray(depth, dtype=np.float32))
            for _, depth, _ in renders
        ]

        self._mesh = mesh
        self._mesh_path = resolved_mesh_path
        self._mesh_vertices = np.asarray(
            mesh.vertices,
            dtype=np.float64,
        ).reshape(-1, 3)
        self._template_dict = {
            "model_name": Path(resolved_mesh_path).stem,
            "templates": templates,
            "depths": depths,
            "intrinsic": intrinsic,
        }

    @staticmethod
    def _bbox_to_tensor(
        bbox: np.ndarray,
        *,
        torch_mod: Any,
    ) -> Any:
        return torch_mod.as_tensor(
            np.asarray(
                bbox,
                dtype=np.float32,
            ).reshape(1, 4)
        )

    def _build_query(
        self,
        *,
        rgb: np.ndarray,
        mask: np.ndarray,
        bbox: np.ndarray,
    ) -> tuple[Any, Any]:
        torch = self._torch
        height, width = rgb.shape[:2]
        processor = self._crop_resize
        if processor is None or processor.h != height or processor.w != width:
            if self._crop_resize_factory is None:
                self._load()
            processor = self._crop_resize_factory(
                target_size=420,
                orig_size=(height, width),
                bbox_extend=self.bbox_extend,
            )
            self._crop_resize = processor

        image = torch.from_numpy(np.asarray(rgb, dtype=np.float32) / 255.0).permute(
            2, 0, 1
        )
        image = image.contiguous()
        mask3 = np.repeat(
            np.asarray(mask, dtype=np.float32)[..., None],
            3,
            axis=2,
        )
        mask_tensor = torch.from_numpy(mask3).permute(2, 0, 1).contiguous()
        box_tensor = self._bbox_to_tensor(
            bbox,
            torch_mod=torch,
        )

        proposal = processor(
            image.unsqueeze(0),
            box_tensor,
        )[0].float()
        proposal_mask = (
            processor(
                mask_tensor.unsqueeze(0),
                box_tensor,
            )[0, 0]
            > 0.5
        )
        proposal = proposal * proposal_mask.unsqueeze(0).float()
        return proposal, proposal_mask

    def infer(
        self,
        request: PoseBackendRequest,
    ) -> PosePrediction:
        self._ensure_mesh(request.mesh_path)
        estimator = self._estimator
        if estimator is None:
            raise RuntimeError("FreePose estimator failed to initialize.")

        camera_matrix = np.asarray(
            request.cam.K.as_K(),
            dtype=np.float64,
        ).reshape(3, 3)
        initial_bbox = mask_to_bbox(request.eef_mask)
        if initial_bbox is None:
            raise RuntimeError("FreePose requires a valid first-frame EEF mask.")

        previous_pose = None if request.force_register_frame0 else request.seed_pose_cam
        previous_bbox = np.asarray(
            initial_bbox,
            dtype=np.int32,
        ).reshape(4)
        candidates = []

        for frame_idx, rgb in enumerate(request.video_frames):
            try:
                if (
                    frame_idx == 0
                    and request.seed_pose_cam is not None
                    and not request.force_register_frame0
                ):
                    candidate = pose_candidate_from_cam(
                        request.seed_pose_cam,
                        source="config_init",
                        pose_quality=1.0,
                    )
                    candidates.append(candidate)
                    previous_pose = np.asarray(
                        request.seed_pose_cam,
                        dtype=np.float64,
                    ).reshape(4, 4)
                    continue

                if previous_pose is not None:
                    projected_bbox = project_mesh_bbox(
                        pose_cam=previous_pose,
                        points_obj=self._mesh_vertices,
                        K=camera_matrix,
                        image_shape=rgb.shape[:2],
                        padding_px=10,
                    )
                    if projected_bbox is not None:
                        previous_bbox = projected_bbox

                frame_mask = np.zeros(
                    rgb.shape[:2],
                    dtype=bool,
                )
                x1, y1, x2, y2 = [int(value) for value in previous_bbox.tolist()]
                frame_mask[y1:y2, x1:x2] = True
                proposal, proposal_mask = self._build_query(
                    rgb=np.asarray(rgb),
                    mask=frame_mask,
                    bbox=previous_bbox,
                )

                output = estimator.forward(
                    proposal=proposal,
                    proposal_mask=proposal_mask,
                    template_dict=self._template_dict,
                    mesh=self._mesh.copy(),
                    K=camera_matrix,
                    bbox=np.asarray(
                        previous_bbox,
                        dtype=np.float32,
                    ),
                    est_scale=1.0,
                    prev_pose=previous_pose,
                    neighborhood=float(self.neighborhood),
                    layer=int(self.layer),
                    batch_size=int(self.batch_size),
                    mask_scores=bool(self.mask_scores),
                )
                pose_cam = np.asarray(
                    output["TCO"][0],
                    dtype=np.float64,
                ).reshape(4, 4)
                score = float(np.asarray(output.get("scores", [0.1]))[0])
                source = (
                    "freepose_register" if previous_pose is None else "freepose_track"
                )
                candidate = pose_candidate_from_cam(
                    pose_cam,
                    source=source,
                    pose_quality=score_to_unit_interval(
                        score,
                        default=0.6,
                    ),
                    raw_score=score,
                )
                candidates.append(candidate)
                if candidate.valid:
                    previous_pose = np.asarray(
                        candidate.pose_cam,
                        dtype=np.float64,
                    ).reshape(4, 4)
            except Exception as exc:
                candidates.append(
                    invalid_candidate(
                        "freepose_error",
                        reason=str(exc),
                        frame_idx=frame_idx,
                    )
                )

        return PosePrediction(
            candidates=candidates,
            meta={
                "model": self.name,
                "cache_dir": str(self.cache_dir),
                "debug_dir": str(self.debug_dir),
                "n_coarse_poses": int(self.n_coarse_poses),
                "n_fine_poses": int(self.n_fine_poses),
                "supports_tracking": True,
                "requires_mesh": True,
            },
        )


__all__ = ["FreePoseBackend", "RuntimeLoader"]
