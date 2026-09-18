"""Portable subprocess adapter for a separately installed SinRef-6D provider."""

from __future__ import annotations

import importlib
import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any, Callable, Mapping  # noqa: UP035

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

_RUNTIME_FIELDS = ("cv2", "process_runner", "pyrender", "trimesh")
_PROVIDER_FOLDER = "Pose_Estimation_Model"
_ENTRY_FILENAME = "run_inference_custom.py"
_PEM_CHECKPOINT_FILENAME = "sam-6d-pem-base.pth"


def _available_runtime() -> Mapping[str, Any]:
    """Import optional provider dependencies only when inference is requested."""

    try:
        return {
            "cv2": importlib.import_module("cv2"),
            "process_runner": subprocess.run,
            "pyrender": importlib.import_module("pyrender"),
            "trimesh": importlib.import_module("trimesh"),
        }
    except Exception as error:
        raise RuntimeError(
            "Failed to import SinRef-6D runtime dependencies. "
            "Please install cv2, pyrender, and trimesh."
        ) from error


def _write_json(destination: Path, payload: Any) -> None:
    with destination.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream)


def _safe_link(destination: Path, source: Path, *, directory: bool) -> None:
    if destination.exists() or destination.is_symlink():
        return
    destination.symlink_to(
        source.resolve(),
        target_is_directory=directory,
    )


def _super_fibonacci_matrices(count: int) -> np.ndarray:
    """Sample deterministic rotations using the public super-Fibonacci rule."""

    if count <= 0:
        return np.empty((0, 3, 3), dtype=np.float64)
    matrices = []
    for index in range(count):
        sample = float(index) + 0.5
        inner_radius = np.sqrt(sample / float(count))
        outer_radius = np.sqrt(1.0 - sample / float(count))
        alpha = 2.0 * np.pi * sample / np.sqrt(2.0)
        beta = 2.0 * np.pi * sample / 1.5337511687552043
        quaternion = np.asarray(
            [
                inner_radius * np.sin(alpha),
                inner_radius * np.cos(alpha),
                outer_radius * np.sin(beta),
                outer_radius * np.cos(beta),
            ],
            dtype=np.float64,
        )
        x, y, z, w = quaternion
        matrices.append(
            np.asarray(
                [
                    [
                        1.0 - 2.0 * (y * y + z * z),
                        2.0 * (x * y - z * w),
                        2.0 * (x * z + y * w),
                    ],
                    [
                        2.0 * (x * y + z * w),
                        1.0 - 2.0 * (x * x + z * z),
                        2.0 * (y * z - x * w),
                    ],
                    [
                        2.0 * (x * z - y * w),
                        2.0 * (y * z + x * w),
                        1.0 - 2.0 * (x * x + y * y),
                    ],
                ],
                dtype=np.float64,
            )
        )
    return np.stack(matrices, axis=0)


def _xyz_image(depth: np.ndarray) -> np.ndarray:
    height, width = depth.shape
    columns, rows = np.meshgrid(
        np.arange(width, dtype=np.float32),
        np.arange(height, dtype=np.float32),
    )
    depth32 = np.asarray(depth, dtype=np.float32)
    xyz = np.stack(
        (
            (columns - float(width) / 2.0) * depth32 / 600.0,
            (rows - float(height) / 2.0) * depth32 / 600.0,
            depth32,
        ),
        axis=-1,
    )
    xyz[depth32 <= 0.0] = 0.0
    return xyz


def _rectangle_mask(shape: tuple[int, int], box: np.ndarray) -> np.ndarray:
    result = np.zeros(shape, dtype=bool)
    x0, y0, x1, y1 = [int(value) for value in box]
    result[y0:y1, x0:x1] = True
    return result


def _segmentation_record(
    mask: np.ndarray,
    box: np.ndarray,
    cv2: Any,
) -> list[dict[str, Any]]:
    x0, y0, x1, y1 = [int(value) for value in box]
    fallback = [x0, y0, x1, y0, x1, y1, x0, y1]
    contour_result = cv2.findContours(
        np.asarray(mask, dtype=np.uint8) * 255,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    contours = contour_result[-2]
    polygon = fallback
    if contours:
        largest = max(contours, key=cv2.contourArea)
        points = np.asarray(largest).reshape(-1, 2)
        if points.shape[0] >= 3:
            polygon = points.astype(np.int64).reshape(-1).tolist()
    return [
        {
            "scene_id": 0,
            "image_id": 0,
            "category_id": 1,
            "bbox": [x0, y0, x1 - x0, y1 - y0],
            "score": 1.0,
            "segmentation": [polygon],
        }
    ]


def _depth_as_png(depth: Any) -> np.ndarray:
    millimetres = np.asarray(depth, dtype=np.float32) * 1000.0
    bounded = np.nan_to_num(
        millimetres,
        nan=0.0,
        posinf=65535.0,
        neginf=0.0,
    )
    return np.clip(bounded, 0.0, 65535.0).astype(np.uint16)


def _cuda_slot(device: Any) -> str:
    text = str(device or "cuda")
    if not text.lower().startswith("cuda"):
        raise RuntimeError("SinRef-6D currently requires a CUDA device.")
    parts = text.split(":", 1)
    return parts[1] if len(parts) == 2 and parts[1] else "0"


def _pose_from_detection(
    record: Mapping[str, Any],
) -> tuple[np.ndarray, float, float]:
    provider_score = record.get("score")
    raw_score = float(provider_score or 0.0)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = np.asarray(
        record["R"],
        dtype=np.float64,
    ).reshape(3, 3)
    pose[:3, 3] = np.asarray(record["t"], dtype=np.float64).reshape(3) / 1000.0
    return (
        pose,
        raw_score,
        score_to_unit_interval(provider_score),
    )


class SinRef6DBackend:
    """Lazy adapter that confines provider writes to an explicit runtime root."""

    name = "sinref6d"
    provider_kind = "builtin"
    backend_id = "sinref6d"
    algorithm_id = "sinref6d_pose_estimation"
    contract_version = POSE_BACKEND_CONTRACT_VERSION

    def __init__(
        self,
        *,
        source_root: str | Path,
        checkpoint_path: str | Path,
        runtime_dir: str | Path,
        template_dir: str | Path,
        debug_dir: str | Path,
        device: str = "cuda",
        n_template_view: int = 42,
        template_resolution: int = 224,
        det_score_thresh: float = 0.2,
        subprocess_timeout_s: float = 300.0,
        python_executable: str | Path | None = None,
        runtime_bundle: Mapping[str, Any] | None = None,
        runtime_loader: RuntimeLoader | None = None,
    ) -> None:
        self.source_root = Path(source_root).expanduser()
        self.checkpoint_path = Path(checkpoint_path).expanduser()
        self.runtime_dir = Path(runtime_dir).expanduser()
        self.template_dir = Path(template_dir).expanduser()
        self.debug_dir = Path(debug_dir).expanduser()
        self.device = str(device or "cuda")
        self.n_template_view = int(n_template_view)
        self.template_resolution = int(template_resolution)
        self.det_score_thresh = float(det_score_thresh)
        self.subprocess_timeout_s = float(subprocess_timeout_s)
        self.python_executable = str(python_executable or sys.executable)
        self._runtime_loader = runtime_loader
        self._runtime_seed = None if runtime_bundle is None else dict(runtime_bundle)
        self._runtime: dict[str, Any] | None = None
        self._mesh: Any = None
        self._mesh_vertices: np.ndarray | None = None
        self._mesh_identity = ""

    def _runtime_services(self) -> dict[str, Any]:
        if self._runtime is None:
            if self._runtime_seed is not None:
                candidate = dict(self._runtime_seed)
            elif self._runtime_loader is not None:
                candidate = dict(self._runtime_loader())
            else:
                candidate = dict(_available_runtime())
            missing = [name for name in _RUNTIME_FIELDS if name not in candidate]
            if missing:
                raise RuntimeError(
                    "SinRef-6D runtime bundle is missing: " + ", ".join(missing)
                )
            self._runtime = candidate
        return self._runtime

    def _assert_checkpoint(self) -> None:
        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"SinRef-6D checkpoint not found: {self.checkpoint_path.resolve()}"
            )

    def _mesh_for(self, mesh_path: Any, trimesh: Any) -> Any:
        identity = Path(str(mesh_path)).expanduser().resolve().as_posix()
        if self._mesh is None or identity != self._mesh_identity:
            mesh = trimesh.load(identity, force="mesh")
            if mesh is None:
                raise RuntimeError(f"Loaded empty EEF mesh: {identity}")
            vertices = np.asarray(
                getattr(mesh, "vertices", []),
                dtype=np.float64,
            ).reshape(-1, 3)
            if vertices.shape[0] == 0:
                raise RuntimeError(f"Loaded empty EEF mesh: {identity}")
            self._mesh = mesh
            self._mesh_vertices = vertices.copy()
            self._mesh_identity = identity
        return self._mesh

    def _materialize_templates(
        self,
        mesh: Any,
        services: Mapping[str, Any],
    ) -> None:
        already_present = self.template_dir.is_dir()
        self.template_dir.mkdir(parents=True, exist_ok=True)
        if already_present:
            return

        cv2 = services["cv2"]
        pyrender = services["pyrender"]
        resolution = self.template_resolution
        camera = pyrender.IntrinsicsCamera(
            fx=600.0,
            fy=600.0,
            cx=float(resolution) / 2.0,
            cy=float(resolution) / 2.0,
        )
        renderer = pyrender.OffscreenRenderer(
            resolution,
            resolution,
        )
        scene = pyrender.Scene(
            bg_color=np.zeros(4, dtype=np.float64),
            ambient_light=np.full(4, 2.0, dtype=np.float64),
        )
        camera_pose = np.diag([1.0, -1.0, -1.0, 1.0])
        scene.add(camera, pose=camera_pose)
        rendered_mesh = pyrender.Mesh.from_trimesh(mesh)
        mesh_node = scene.add(rendered_mesh)
        rotations = _super_fibonacci_matrices(self.n_template_view)
        for view_index, rotation in enumerate(rotations):
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = rotation
            pose[2, 3] = 1.1
            scene.set_pose(mesh_node, pose=pose)
            rgb, depth = renderer.render(
                scene,
                flags=(pyrender.constants.RenderFlags.SKIP_CULL_FACES),
            )
            rgb_path = self.template_dir / (f"rgb_{view_index}.png")
            mask_path = self.template_dir / (f"mask_{view_index}.png")
            xyz_path = self.template_dir / (f"xyz_{view_index}.npy")
            cv2.imwrite(
                rgb_path.as_posix(),
                cv2.cvtColor(
                    np.asarray(rgb, dtype=np.uint8),
                    cv2.COLOR_RGB2BGR,
                ),
            )
            cv2.imwrite(
                mask_path.as_posix(),
                (np.asarray(depth) > 0.0).astype(np.uint8) * 255,
            )
            np.save(
                xyz_path,
                _xyz_image(np.asarray(depth)),
            )

    def _provider_workspace(self) -> tuple[Path, Path]:
        source_work = self.source_root / _PROVIDER_FOLDER
        source_entry = source_work / _ENTRY_FILENAME
        if not source_entry.is_file():
            raise FileNotFoundError(
                f"SinRef-6D inference entry not found: {source_entry.resolve()}"
            )

        launch_work = self.runtime_dir / "_sinref6d_launcher" / _PROVIDER_FOLDER
        launch_work.mkdir(parents=True, exist_ok=True)
        launch_entry = launch_work / _ENTRY_FILENAME
        _safe_link(
            launch_entry,
            source_entry,
            directory=False,
        )
        source_config = source_work / "config"
        if source_config.is_dir():
            _safe_link(
                launch_work / "config",
                source_config,
                directory=True,
            )
        checkpoint_folder = launch_work / "checkpoints"
        checkpoint_folder.mkdir(parents=True, exist_ok=True)
        _safe_link(
            checkpoint_folder / _PEM_CHECKPOINT_FILENAME,
            self.checkpoint_path,
            directory=False,
        )
        return launch_work, launch_entry

    def _frame_inputs(
        self,
        *,
        frame_index: int,
        rgb: Any,
        depth: Any,
        camera_matrix: np.ndarray,
        mask: np.ndarray,
        box: np.ndarray,
        cv2: Any,
    ) -> dict[str, Path]:
        frame_dir = self.runtime_dir / f"frame_{frame_index:04d}"
        frame_dir.mkdir(parents=True, exist_ok=True)
        _safe_link(
            frame_dir / "templates",
            self.template_dir,
            directory=True,
        )
        paths = {
            "output": frame_dir,
            "rgb": frame_dir / "rgb.png",
            "depth": frame_dir / "depth.png",
            "camera": frame_dir / "cam.json",
            "segmentation": frame_dir / "seg.json",
        }
        cv2.imwrite(
            paths["rgb"].as_posix(),
            cv2.cvtColor(
                np.asarray(rgb, dtype=np.uint8),
                cv2.COLOR_RGB2BGR,
            ),
        )
        cv2.imwrite(
            paths["depth"].as_posix(),
            _depth_as_png(depth),
        )
        _write_json(
            paths["camera"],
            {
                "cam_K": camera_matrix.reshape(-1).tolist(),
                "depth_scale": 1.0,
            },
        )
        _write_json(
            paths["segmentation"],
            _segmentation_record(mask, box, cv2),
        )
        return paths

    def _provider_command(
        self,
        *,
        launch_entry: Path,
        paths: Mapping[str, Path],
        mesh_path: str,
        device: Any,
    ) -> list[str]:
        return [
            self.python_executable,
            launch_entry.as_posix(),
            "--gpus",
            _cuda_slot(device),
            "--config",
            "config/base.yaml",
            "--output_dir",
            paths["output"].resolve().as_posix(),
            "--cad_path",
            mesh_path,
            "--rgb_path",
            paths["rgb"].resolve().as_posix(),
            "--depth_path",
            paths["depth"].resolve().as_posix(),
            "--cam_path",
            paths["camera"].resolve().as_posix(),
            "--seg_path",
            paths["segmentation"].resolve().as_posix(),
            "--det_score_thresh",
            str(self.det_score_thresh),
        ]

    def _execute_provider(
        self,
        *,
        services: Mapping[str, Any],
        launch_work: Path,
        launch_entry: Path,
        paths: Mapping[str, Path],
        mesh_path: str,
        device: Any,
    ) -> tuple[np.ndarray, float, float]:
        environment = dict(os.environ)
        environment["PYOPENGL_PLATFORM"] = "egl"
        environment["SINREF6D_CHECKPOINT_PATH"] = (
            self.checkpoint_path.resolve().as_posix()
        )
        environment["PYTHONDONTWRITEBYTECODE"] = "1"
        command = self._provider_command(
            launch_entry=launch_entry,
            paths=paths,
            mesh_path=mesh_path,
            device=device,
        )
        try:
            completed = services["process_runner"](
                command,
                cwd=launch_work.resolve().as_posix(),
                env=environment,
                check=False,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                timeout=self.subprocess_timeout_s,
            )
        except subprocess.TimeoutExpired as error:
            raise RuntimeError(
                f"subprocess_timeout_s={self.subprocess_timeout_s}: {error}"
            ) from error
        except subprocess.CalledProcessError as error:
            detail = str(error.stderr or error)
            raise RuntimeError(detail) from error

        detection_path = paths["output"] / "sam6d_results" / "detection_pem.json"
        if int(completed.returncode) != 0 and not detection_path.is_file():
            detail = str(completed.stderr or "").strip()
            raise RuntimeError(
                f"SinRef-6D inference failed before writing detection results: {detail}"
            )
        if not detection_path.is_file():
            raise RuntimeError(
                f"SinRef-6D did not write detection results: {detection_path.resolve()}"
            )
        with detection_path.open("r", encoding="utf-8") as stream:
            detections = json.load(stream)
        if not detections:
            raise RuntimeError("SinRef-6D returned no detections.")
        winner = max(
            detections,
            key=lambda item: float(item.get("score", 0.0)),
        )
        return _pose_from_detection(winner)

    def infer(self, request: PoseBackendRequest) -> PosePrediction:
        services = self._runtime_services()
        self._assert_checkpoint()
        mesh = self._mesh_for(
            request.mesh_path,
            services["trimesh"],
        )
        self._materialize_templates(mesh, services)
        launch_work, launch_entry = self._provider_workspace()

        camera_matrix = np.asarray(
            request.cam.K.as_K(),
            dtype=np.float64,
        ).reshape(3, 3)
        initial_mask = np.asarray(request.eef_mask)
        active_box = mask_to_bbox(initial_mask)
        active_mask = initial_mask
        previous_pose = None if request.force_register_frame0 else request.seed_pose_cam
        candidates = []
        paired_frames = zip(
            request.video_frames,
            request.depths,
        )
        for frame_index, (rgb, depth) in enumerate(paired_frames):
            if (
                frame_index == 0
                and request.seed_pose_cam is not None
                and not request.force_register_frame0
            ):
                candidate = pose_candidate_from_cam(
                    request.seed_pose_cam,
                    source="config_init",
                    pose_quality=1.0,
                )
                candidates.append(candidate)
                if candidate.valid:
                    previous_pose = candidate.pose_cam
                continue

            try:
                if previous_pose is not None:
                    projected = project_mesh_bbox(
                        pose_cam=np.asarray(
                            previous_pose,
                            dtype=np.float64,
                        ).reshape(4, 4),
                        points_obj=self._mesh_vertices,
                        K=camera_matrix,
                        image_shape=np.asarray(rgb).shape[:2],
                        padding_px=10,
                    )
                    if projected is not None:
                        active_box = projected
                        active_mask = _rectangle_mask(
                            np.asarray(rgb).shape[:2],
                            active_box,
                        )
                if active_box is None:
                    raise RuntimeError(
                        "SinRef-6D requires a non-empty mask or projected bbox."
                    )
                paths = self._frame_inputs(
                    frame_index=frame_index,
                    rgb=rgb,
                    depth=depth,
                    camera_matrix=camera_matrix,
                    mask=active_mask,
                    box=active_box,
                    cv2=services["cv2"],
                )
                pose, score, quality = self._execute_provider(
                    services=services,
                    launch_work=launch_work,
                    launch_entry=launch_entry,
                    paths=paths,
                    mesh_path=self._mesh_identity,
                    device=request.device or self.device,
                )
                source = "sinref6d_register" if frame_index == 0 else "sinref6d_frame"
                candidate = pose_candidate_from_cam(
                    pose,
                    source=source,
                    pose_quality=quality,
                    raw_score=score,
                )
                candidates.append(candidate)
                if candidate.valid:
                    previous_pose = candidate.pose_cam
            except Exception as error:  # noqa: BLE001
                candidates.append(
                    invalid_candidate(
                        "sinref6d_error",
                        reason=str(error),
                        frame_idx=frame_index,
                    )
                )

        return PosePrediction(
            candidates=candidates,
            meta={
                "model": self.name,
                "weights_root": str(self.checkpoint_path.parent),
                "debug_dir": str(self.debug_dir),
                "n_template_view": self.n_template_view,
                "template_resolution": self.template_resolution,
                "subprocess_timeout_s": self.subprocess_timeout_s,
                "supports_tracking": False,
                "requires_mesh": True,
            },
        )


__all__ = ["RuntimeLoader", "SinRef6DBackend"]
