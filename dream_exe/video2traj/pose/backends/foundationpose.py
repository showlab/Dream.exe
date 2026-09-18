"""Lazy FoundationPose adapter for the portable pose contract.

The adapter owns only model loading and per-frame inference.  Source and
checkpoint roots are explicit constructor inputs so importing
``dream_exe.video2traj`` never assumes a repository checkout or a bench
layout.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..contract import (
    POSE_BACKEND_CONTRACT_VERSION,
    PoseBackendRequest,
    PosePrediction,
    invalid_candidate,
    pose_candidate_from_cam,
)


DependencyLoader = Callable[[], Mapping[str, Any]]


class FoundationPoseBackend:
    """Stateful, lazily loaded FoundationPose runtime adapter."""

    name = "foundationpose"
    provider_kind = "builtin"
    backend_id = "foundationpose"
    algorithm_id = "foundationpose_register_track"
    contract_version = POSE_BACKEND_CONTRACT_VERSION

    def __init__(
        self,
        *,
        source_root: str | Path,
        weights_root: str | Path = "",
        device: str = "cuda",
        debug_dir: str | Path = "",
        dependency_loader: DependencyLoader | None = None,
    ) -> None:
        self.source_root = Path(source_root).expanduser()
        self.weights_root = str(weights_root or "")
        self.device = str(device or "cuda")
        self.debug_dir = str(debug_dir or "")
        self._dependency_loader = dependency_loader
        self._dependencies: dict[str, Any] | None = None
        self._estimator: Any = None
        self._mesh_path = ""

    def _load_dependencies(self) -> dict[str, Any]:
        if self._dependencies is not None:
            return self._dependencies
        if self._dependency_loader is not None:
            dependencies = dict(self._dependency_loader())
        else:
            source_root = self.source_root.resolve()
            if not source_root.exists():
                raise RuntimeError(
                    f"FoundationPose source directory not found: {source_root}"
                )
            source_text = source_root.as_posix()
            if source_text not in sys.path:
                sys.path.insert(0, source_text)
            if self.weights_root:
                os.environ["FOUNDATIONPOSE_WEIGHTS_ROOT"] = self.weights_root
            try:
                dependencies = {
                    "trimesh": importlib.import_module("trimesh"),
                    "dr": importlib.import_module("nvdiffrast.torch"),
                    "torch": importlib.import_module("torch"),
                    "utils_mod": importlib.import_module("Utils"),
                    "estimater_mod": importlib.import_module("estimater"),
                }
            except Exception as exc:  # pragma: no cover - runtime specific
                raise RuntimeError(
                    "Failed to import FoundationPose dependencies. "
                    "Please install trimesh / nvdiffrast / open3d / "
                    "omegaconf and build FoundationPose extensions."
                ) from exc
        required = {
            "trimesh",
            "dr",
            "torch",
            "utils_mod",
            "estimater_mod",
        }
        missing = sorted(required.difference(dependencies))
        if missing:
            raise RuntimeError(
                "FoundationPose dependency loader is missing: " + ", ".join(missing)
            )
        self._dependencies = dependencies
        return dependencies

    def _build_estimator(
        self,
        *,
        mesh_path: str,
        request: PoseBackendRequest,
    ) -> Any:
        dependencies = self._load_dependencies()
        trimesh = dependencies["trimesh"]
        dr = dependencies["dr"]
        utils_mod = dependencies["utils_mod"]
        estimater_mod = dependencies["estimater_mod"]

        if hasattr(utils_mod, "set_seed"):
            utils_mod.set_seed(0)
        warp = getattr(utils_mod, "wp", None)
        if warp is not None and hasattr(warp, "force_load"):
            try:
                warp.force_load(device=str(request.device or self.device))
            except Exception:
                pass

        mesh = trimesh.load(mesh_path, force="mesh")
        if mesh is None or len(getattr(mesh, "vertices", [])) == 0:
            raise RuntimeError(f"Loaded empty EEF mesh: {mesh_path}")

        device = str(request.device or self.device)
        device_arg: Any = 0 if device.startswith("cuda") else device
        context = dr.RasterizeCudaContext(device_arg)
        estimator = estimater_mod.FoundationPose(
            model_pts=np.asarray(mesh.vertices),
            model_normals=np.asarray(mesh.vertex_normals),
            mesh=mesh,
            scorer=None,
            refiner=None,
            glctx=context,
            debug_dir=self.debug_dir,
            debug=int(request.config.debug),
        )
        self._mesh_path = mesh_path
        self._estimator = estimator
        return estimator

    def _get_estimator(self, request: PoseBackendRequest) -> Any:
        mesh_path = Path(request.mesh_path).expanduser().resolve().as_posix()
        if self._estimator is None or self._mesh_path != mesh_path:
            return self._build_estimator(
                mesh_path=mesh_path,
                request=request,
            )
        return self._estimator

    def _prepare_estimator(self, request: PoseBackendRequest) -> Any:
        estimator = self._get_estimator(request)
        if request.seed_pose_cam is None:
            return estimator

        torch = self._load_dependencies()["torch"]
        centered_mesh_transform = (
            estimator.get_tf_to_centered_mesh().data.cpu().numpy().reshape(4, 4)
        )
        estimator.pose_last = torch.as_tensor(
            np.asarray(
                request.seed_pose_cam,
                dtype=np.float64,
            ).reshape(4, 4)
            @ np.linalg.inv(centered_mesh_transform),
            device=str(request.device or self.device),
            dtype=torch.float,
        )
        return estimator

    def infer(self, request: PoseBackendRequest) -> PosePrediction:
        camera_matrix = request.cam.K.as_K()
        mask = np.asarray(request.eef_mask, dtype=bool)
        estimator = self._prepare_estimator(request)

        candidates = []
        for frame_idx, (rgb, depth) in enumerate(
            zip(request.video_frames, request.depths)
        ):
            try:
                if (
                    frame_idx == 0
                    and request.seed_pose_cam is not None
                    and not request.force_register_frame0
                ):
                    candidates.append(
                        pose_candidate_from_cam(
                            request.seed_pose_cam,
                            source="config_init",
                            pose_quality=1.0,
                        )
                    )
                    continue
                if frame_idx == 0:
                    pose_cam = estimator.register(
                        K=camera_matrix,
                        rgb=np.asarray(rgb),
                        depth=np.asarray(depth, dtype=np.float32),
                        ob_mask=mask.astype(np.uint8),
                        iteration=int(request.config.init_refine_iter),
                    )
                    source = "foundationpose_register"
                else:
                    pose_cam = estimator.track_one(
                        rgb=np.asarray(rgb),
                        depth=np.asarray(depth, dtype=np.float32),
                        K=camera_matrix,
                        iteration=int(request.config.track_refine_iter),
                    )
                    source = "foundationpose_track"
                candidates.append(
                    pose_candidate_from_cam(
                        pose_cam,
                        source=source,
                        pose_quality=0.55,
                    )
                )
            except Exception as exc:
                candidates.append(
                    invalid_candidate(
                        "foundationpose_error",
                        reason=str(exc),
                        frame_idx=frame_idx,
                    )
                )

        return PosePrediction(
            candidates=candidates,
            meta={
                "model": self.name,
                "weights_root": self.weights_root,
                "debug_dir": self.debug_dir,
                "init_refine_iter": int(request.config.init_refine_iter),
                "track_refine_iter": int(request.config.track_refine_iter),
                "supports_tracking": True,
            },
        )


__all__ = ["FoundationPoseBackend"]
