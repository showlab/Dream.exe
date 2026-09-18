"""Read-only frozen simulator state and scene restoration.

This module preserves the current saved JSON schema and restoration order:
optional RoboCasa scene/XML restore, strict flattened MuJoCo state, legacy
qpos/qvel fallback, compatibility patches, camera restore, then verification.
It never discovers a bench root and never writes an artifact.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
import gzip
import json
from pathlib import Path
from typing import Any

import numpy as np

from ..runtime.camera import build_camera_raw
from ..robocasa.restore import (
    resolve_robocasa_camera_setup_from_manifest,
    resolve_root_reference,
    restore_robocasa_scene,
)


def max_abs_diff(first: Any, second: Any) -> float:
    """Return the current flattened float64 maximum absolute difference."""

    first_array = np.asarray(
        first,
        dtype=np.float64,
    ).reshape(-1)
    second_array = np.asarray(
        second,
        dtype=np.float64,
    ).reshape(-1)
    if first_array.shape != second_array.shape:
        return float("inf")
    if first_array.size == 0:
        return 0.0
    return float(np.max(np.abs(first_array - second_array)))


def restore_camera_from_raw(
    env: Any,
    camera_config: Mapping[str, Any],
) -> None:
    """Restore current local-or-world camera model fields."""

    camera_id = env.sim.model.camera_name2id(camera_config["name"])
    local_position = camera_config.get("local_pos", None)
    local_quaternion = camera_config.get(
        "local_quat_wxyz",
        None,
    )
    if local_position is not None and local_quaternion is not None:
        env.sim.model.cam_pos[camera_id] = np.asarray(
            local_position,
            float,
        )
        env.sim.model.cam_quat[camera_id] = np.asarray(
            local_quaternion,
            float,
        )
    else:
        env.sim.model.cam_pos[camera_id] = np.asarray(
            camera_config["pos_w"],
            float,
        )
        env.sim.model.cam_quat[camera_id] = np.asarray(
            camera_config["quat_wxyz"],
            float,
        )
    env.sim.model.cam_fovy[camera_id] = float(
        camera_config.get(
            "fovy_deg",
            env.sim.model.cam_fovy[camera_id],
        )
    )
    env.sim.forward()


def _load_model_xml(path: str | Path) -> str:
    with gzip.open(path, "rt", encoding="utf-8") as stream:
        return stream.read()


def _load_json_object(path: str | Path) -> Any:
    with Path(path).open("r", encoding="utf-8") as stream:
        return json.load(stream)


def _load_states(path: str | Path) -> np.ndarray:
    loaded = np.load(path)
    if isinstance(loaded, np.ndarray):
        return np.asarray(loaded, dtype=np.float64)
    key = (
        "states"
        if hasattr(loaded, "files") and "states" in loaded.files
        else loaded.files[0]
    )
    return np.asarray(loaded[key], dtype=np.float64)


def _resolve_manifest_path(
    key: str,
    *,
    paths: Mapping[str, Any],
    path_refs: Mapping[str, Any],
    path_resolver: Callable[[Mapping[str, Any]], Any],
) -> Any:
    reference = path_refs.get(key, None)
    if isinstance(reference, dict) and len(reference) > 0:
        resolved = path_resolver(reference)
        if resolved:
            return resolved

    raw_value = str(paths.get(key, "") or "").strip()
    if not raw_value:
        return ""
    path_object = Path(raw_value).expanduser()
    if path_object.is_absolute():
        return path_object.resolve().as_posix()

    return raw_value


def resolve_scene_manifest_path(
    key: str,
    *,
    paths: Mapping[str, Any],
    path_refs: Mapping[str, Any],
    path_resolver: Callable[[Mapping[str, Any]], Any],
) -> Any:
    """Resolve one explicit scene-manifest path with restore precedence."""

    return _resolve_manifest_path(
        key,
        paths=paths,
        path_refs=path_refs,
        path_resolver=path_resolver,
    )


def restore_scene_graph_from_manifest(
    env: Any,
    root: Mapping[str, Any],
    *,
    verify_atol: float,
    named_roots: Mapping[str, Any] | None = None,
    path_resolver: (Callable[[Mapping[str, Any]], Any] | None) = None,
    model_xml_loader: (Callable[[str | Path], str] | None) = None,
    ep_meta_loader: (Callable[[str | Path], Any] | None) = None,
    states_loader: (Callable[[str | Path], Any] | None) = None,
    camera_setup_resolver: Callable[..., Mapping[str, Any]] | None = None,
    scene_restorer: Callable[..., Mapping[str, Any]] | None = None,
    camera_builder: Callable[..., Mapping[str, Any]] | None = None,
    asset_preparer: Callable[[str], list[str]] | None = None,
) -> dict[str, Any]:
    """Restore the current RoboCasa manifest branch from explicit inputs."""

    scene_restore = dict(root.get("scene_restore", {}) or {})
    backend = str(scene_restore.get("backend", "") or "").strip()
    if backend != "robocasa_frozen":
        return {}

    paths = dict(scene_restore.get("paths", {}) or {})
    path_refs = dict(scene_restore.get("path_refs", {}) or {})
    effective_named_roots = dict(named_roots or {})
    resolver = (
        (
            lambda reference: resolve_root_reference(
                reference,
                named_roots=effective_named_roots,
            )
        )
        if path_resolver is None
        else path_resolver
    )

    model_xml_path = resolve_scene_manifest_path(
        "model_xml_gz",
        paths=paths,
        path_refs=path_refs,
        path_resolver=resolver,
    )
    ep_meta_path = resolve_scene_manifest_path(
        "ep_meta_json",
        paths=paths,
        path_refs=path_refs,
        path_resolver=resolver,
    )
    states_path = resolve_scene_manifest_path(
        "states_npz",
        paths=paths,
        path_refs=path_refs,
        path_resolver=resolver,
    )
    if not model_xml_path or not ep_meta_path or not states_path:
        raise RuntimeError(
            "RoboCasa scene restore requires model_xml_gz, "
            "ep_meta_json, and states_npz in "
            "raw.scene_restore.paths."
        )

    load_model_xml = _load_model_xml if model_xml_loader is None else model_xml_loader
    load_ep_meta = _load_json_object if ep_meta_loader is None else ep_meta_loader
    load_states = _load_states if states_loader is None else states_loader
    model_xml = load_model_xml(model_xml_path)
    ep_meta = load_ep_meta(ep_meta_path)
    states = np.asarray(
        load_states(states_path),
        dtype=np.float64,
    )
    if states.ndim != 2 or states.shape[0] == 0:
        raise RuntimeError(f"Invalid RoboCasa states shape: {states.shape}")

    if camera_setup_resolver is None:
        camera_setup = resolve_robocasa_camera_setup_from_manifest(
            scene_restore=scene_restore,
            ep_meta=ep_meta,
            path_resolver=resolver,
            named_roots=effective_named_roots,
        )
    else:
        camera_setup = camera_setup_resolver(
            scene_restore=scene_restore,
            ep_meta=ep_meta,
        )
    resolved_camera_configs = dict(
        camera_setup.get("cam_configs", {})
        or scene_restore.get("resolved_cam_configs", {})
        or {}
    )
    if resolved_camera_configs:
        ep_meta = dict(ep_meta)
        ep_meta["cam_configs"] = resolved_camera_configs

    if scene_restorer is None:
        restore_metadata = restore_robocasa_scene(
            env,
            model_xml=model_xml,
            state0_flat=states[0],
            ep_meta=ep_meta,
            cam_configs=dict(ep_meta.get("cam_configs", {}) or {}),
            verify=True,
            verify_atol=verify_atol,
            asset_preparer=asset_preparer,
        )
    else:
        restore_metadata = scene_restorer(
            env,
            model_xml=model_xml,
            state0_flat=states[0],
            ep_meta=ep_meta,
            cam_configs=dict(ep_meta.get("cam_configs", {}) or {}),
            verify=True,
            verify_atol=verify_atol,
        )

    active_camera_name = str(
        camera_setup.get(
            "active_camera_name",
            scene_restore.get("active_camera_name", ""),
        )
        or ""
    ).strip()
    effective_camera_raw: dict[str, Any] = {}
    if active_camera_name:
        try:
            saved_camera = dict(root.get("camera", {}) or {})
            frame_size = (
                int(saved_camera.get("width", 512)),
                int(saved_camera.get("height", 512)),
            )
            build_camera = (
                build_camera_raw if camera_builder is None else camera_builder
            )
            effective_camera_raw = build_camera(
                env,
                active_camera_name,
                frame_size,
            )
            effective_camera_raw["camera_profile"] = str(
                camera_setup.get(
                    "camera_profile",
                    scene_restore.get("camera_profile", ""),
                )
            )
        except Exception:
            effective_camera_raw = {}

    return {
        "backend": backend,
        "restore_meta": restore_metadata,
        "active_camera_name": active_camera_name,
        "camera_setup": camera_setup,
        "effective_camera_raw": effective_camera_raw,
    }


def restore_env_from_json(
    env: Any,
    config: Mapping[str, Any],
    zero_vel: bool = False,
    *,
    verify: bool = True,
    verify_atol: float = 1e-9,
    apply_redundant_patches_on_flattened: bool = False,
    scene_restore_handler: Callable[..., Mapping[str, Any]] | None = None,
    scene_restore_options: Mapping[str, Any] | None = None,
) -> None:
    """Restore an existing simulator environment from the current JSON schema."""

    root = config["raw"] if "raw" in config else config

    flattened_state = root.get("mujoco_flattened_state", None)
    used_flattened = False
    expected_flattened = None
    qpos_target = None
    qvel_target = None

    restore_scene = (
        restore_scene_graph_from_manifest
        if scene_restore_handler is None
        else scene_restore_handler
    )
    scene_restore_info = restore_scene(
        env,
        root,
        verify_atol=float(verify_atol),
        **dict(scene_restore_options or {}),
    )

    if flattened_state is not None:
        flattened_state = np.asarray(
            flattened_state,
            dtype=np.float64,
        ).reshape(-1)

        current_flattened = env.sim.get_state().flatten()
        if int(flattened_state.shape[0]) != int(current_flattened.shape[0]):
            raise RuntimeError(
                "[RESTORE][FATAL] mujoco_flattened_state dim "
                f"mismatch: cfg={flattened_state.shape[0]} "
                f"env={current_flattened.shape[0]}"
            )

        env.sim.set_state_from_flattened(flattened_state)
        env.sim.forward()
        used_flattened = True

        if zero_vel:
            env.sim.data.qvel[:] = 0
            env.sim.data.qacc[:] = 0
            env.sim.data.qfrc_applied[:] = 0
            env.sim.forward()
            print(
                "[RESTORE][WARN] zero_vel=True modifies flattened "
                "restore; exact flat verification is disabled."
            )
        else:
            expected_flattened = flattened_state.copy()

        print("[✔] Restored from mujoco_flattened_state (STRICT)")

    if not used_flattened:
        qpos_target = np.asarray(
            root["mujoco_state"]["qpos"],
            float,
        ).reshape(-1)
        qvel_target = np.asarray(
            root["mujoco_state"]["qvel"],
            float,
        ).reshape(-1)
        env.sim.data.qpos[: len(qpos_target)] = qpos_target
        env.sim.data.qvel[: len(qvel_target)] = qvel_target

        if zero_vel:
            env.sim.data.qvel[:] = 0
            env.sim.data.qacc[:] = 0
            env.sim.data.qfrc_applied[:] = 0
            qvel_target = np.zeros_like(
                env.sim.data.qvel,
                dtype=np.float64,
            )
        else:
            qvel_target = env.sim.data.qvel.copy()

        env.sim.data.qacc[:] = 0
        env.sim.data.qfrc_applied[:] = 0
        env.sim.step()
        env.sim.forward()
        qpos_target = env.sim.data.qpos.copy()
        qvel_target = env.sim.data.qvel.copy()

        print("[✔] Restored from qpos/qvel fallback (legacy compatible)")

    apply_state_patches = (not used_flattened) or bool(
        apply_redundant_patches_on_flattened
    )

    if apply_state_patches:
        free_joints = root.get("free_joints", None)
        if isinstance(free_joints, list) and len(free_joints) > 0:
            for item in free_joints:
                address = int(item["qpos_adr"])
                env.sim.data.qpos[address : address + 3] = np.asarray(
                    item["pos"], float
                )
                env.sim.data.qpos[address + 3 : address + 7] = np.asarray(
                    item["quat_wxyz"], float
                )
            env.sim.forward()

        if "objects" in root:
            for name, values in root["objects"].items():
                if name.endswith("_main"):
                    joint_name = name.replace(
                        "_main",
                        "_joint0",
                    )
                    if joint_name in env.sim.model.joint_names:
                        joint_id = env.sim.model.joint_name2id(joint_name)
                        address = env.sim.model.jnt_qposadr[joint_id]
                        env.sim.data.qpos[address : address + 3] = np.asarray(
                            values["pos"], float
                        )
                        env.sim.data.qpos[address + 3 : address + 7] = np.asarray(
                            values["quat_wxyz"],
                            float,
                        )
            env.sim.forward()

        if "robot_state" in root:
            robot = env.robots[0]
            joint_positions = np.asarray(
                root["robot_state"]["joint_pos"],
                float,
            )
            if joint_positions.size > 0:
                robot.set_robot_joint_positions(joint_positions)
                env.sim.forward()

    if not used_flattened:
        qpos_target = env.sim.data.qpos.copy()
        qvel_target = env.sim.data.qvel.copy()

    camera = dict(
        (scene_restore_info or {}).get(
            "effective_camera_raw",
            {},
        )
        or {}
    )
    if camera:
        print(
            "[✔] Restored camera(scene-backed) → "
            f"pos={camera['pos_w']}, "
            f"quat_wxyz={camera['quat_wxyz']}"
        )
    else:
        camera = root["camera"]
        restore_camera_from_raw(env, camera)
        print(
            "[✔] Restored camera(raw) → "
            f"pos={camera['pos_w']}, "
            f"quat_wxyz={camera['quat_wxyz']}"
        )

    if verify:
        if (
            used_flattened
            and expected_flattened is not None
            and not apply_state_patches
        ):
            current_flattened = env.sim.get_state().flatten()
            difference = max_abs_diff(
                current_flattened,
                expected_flattened,
            )
            if difference > float(verify_atol):
                raise RuntimeError(
                    "[RESTORE][FATAL] flattened restore "
                    "verification failed: "
                    f"max|cur-target|={difference:.3e} > "
                    f"atol={float(verify_atol):.3e}"
                )
            print(
                "[✔] Verified flattened restore matches saved "
                f"state (max|diff|={difference:.3e})"
            )
        elif not used_flattened and qpos_target is not None and qvel_target is not None:
            qpos_error = max_abs_diff(
                env.sim.data.qpos,
                qpos_target,
            )
            qvel_error = max_abs_diff(
                env.sim.data.qvel,
                qvel_target,
            )
            if max(qpos_error, qvel_error) > float(verify_atol):
                raise RuntimeError(
                    "[RESTORE][FATAL] qpos/qvel restore "
                    "verification failed: "
                    f"qpos_err={qpos_error:.3e}, "
                    f"qvel_err={qvel_error:.3e}, "
                    f"atol={float(verify_atol):.3e}"
                )
            print(
                "[✔] Verified legacy restore matches current "
                "qpos/qvel targets "
                f"(qpos_err={qpos_error:.3e}, "
                f"qvel_err={qvel_error:.3e})"
            )


__all__ = [
    "max_abs_diff",
    "restore_camera_from_raw",
    "restore_env_from_json",
    "restore_scene_graph_from_manifest",
    "resolve_scene_manifest_path",
]
