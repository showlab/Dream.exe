from __future__ import annotations

from dream_exe.sim.runtime.environment import finalize_mujoco_egl_device_id


def test_finalize_keeps_physical_egl_id() -> None:
    env = {
        "MUJOCO_GL": "egl",
        "CUDA_VISIBLE_DEVICES": "5",
        "MUJOCO_EGL_DEVICE_ID": "5",
    }

    finalize_mujoco_egl_device_id(environ=env)

    assert env["MUJOCO_EGL_DEVICE_ID"] == "5"


def test_finalize_repairs_import_mutation_to_first_visible_physical_id() -> None:
    env = {
        "MUJOCO_GL": "egl",
        "CUDA_VISIBLE_DEVICES": "5,7",
        "MUJOCO_EGL_DEVICE_ID": "0",
    }

    finalize_mujoco_egl_device_id(environ=env)

    assert env["MUJOCO_EGL_DEVICE_ID"] == "5"


def test_finalize_preserves_disabled_egl_selection() -> None:
    env = {
        "MUJOCO_GL": "egl",
        "CUDA_VISIBLE_DEVICES": "5",
        "MUJOCO_EGL_DEVICE_ID": "-1",
    }

    finalize_mujoco_egl_device_id(environ=env)

    assert env["MUJOCO_EGL_DEVICE_ID"] == "-1"
