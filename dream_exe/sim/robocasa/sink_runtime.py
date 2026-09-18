"""RoboCasa sink compatibility for restored task-success environments.

The helpers in this module operate only on one caller-owned environment.
They do not resolve datasets, bench samples, configuration files, or output
paths.  Compatibility is installed on the restored sink instance rather than
on RoboCasa or robosuite classes.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

import numpy as np


_HANDLE_JOINT_SUFFIX = "handle_joint"
_SPOUT_JOINT_SUFFIX = "spout_joint"
_TEMPERATURE_JOINT_SUFFIX = "handle_temp_joint"
_SINK_CACHE_ATTRIBUTES = (
    "_handle_joint",
    "_water_site",
    "_high_water_radius",
)


def _sink_context(
    env: Any,
) -> tuple[Any, Any, tuple[Any, ...]] | None:
    sink = getattr(env, "sink", None)
    if sink is None or not hasattr(env, "sim"):
        return None
    simulation = env.sim
    try:
        joint_names = tuple(simulation.model.joint_names)
    except Exception:
        return None
    return sink, simulation, joint_names


def _sink_prefix(sink: Any) -> str:
    try:
        return str(getattr(sink, "naming_prefix", "") or "")
    except Exception:
        return ""


def _read_joint_scalar(simulation: Any, joint_name: str) -> float:
    address = simulation.model.get_joint_qpos_addr(joint_name)
    if isinstance(address, (int, np.int32, np.int64)):
        return float(simulation.data.qpos[int(address)])
    start, end = address
    values = np.asarray(
        simulation.data.qpos[int(start) : int(end)],
        dtype=np.float64,
    ).reshape(-1)
    return float(values[0]) if values.size else float("nan")


def _write_joint_scalar(
    simulation: Any,
    joint_name: str,
    value: float,
) -> None:
    scalar = float(value)
    try:
        simulation.data.set_joint_qpos(joint_name, scalar)
        return
    except Exception:
        pass
    address = simulation.model.get_joint_qpos_addr(joint_name)
    if isinstance(address, (int, np.int32, np.int64)):
        simulation.data.qpos[int(address)] = scalar
        return
    start, end = address
    simulation.data.qpos[int(start) : int(end)] = scalar


def repair_sink_prefix_if_needed(
    env: Any,
) -> dict[str, Any] | None:
    """Align the restored sink fixture name with its live MuJoCo joints."""

    context = _sink_context(env)
    if context is None:
        return None
    sink, _simulation, joint_names = context
    current_prefix = _sink_prefix(sink)
    expected_handle = (
        f"{current_prefix}{_HANDLE_JOINT_SUFFIX}" if current_prefix else ""
    )
    if expected_handle and expected_handle in joint_names:
        return {
            "sink_prefix_repair": "noop",
            "sink_naming_prefix": current_prefix,
        }

    handle_candidates = [
        name
        for name in joint_names
        if (
            isinstance(name, str)
            and name.startswith("sink")
            and name.endswith(_HANDLE_JOINT_SUFFIX)
        )
    ]
    if not handle_candidates:
        return {
            "sink_prefix_repair": "no_candidates",
            "sink_naming_prefix": current_prefix,
        }

    selected_joint = min(handle_candidates, key=len)
    restored_prefix = selected_joint[: -len(_HANDLE_JOINT_SUFFIX)]
    restored_name = restored_prefix.removesuffix("_")
    changed = False
    for attribute in ("_name", "name"):
        if not hasattr(sink, attribute):
            continue
        try:
            setattr(sink, attribute, restored_name)
            changed = True
            break
        except Exception:
            continue

    for attribute in _SINK_CACHE_ATTRIBUTES:
        if not hasattr(sink, attribute):
            continue
        try:
            setattr(sink, attribute, None)
        except Exception:
            pass

    return {
        "sink_prefix_repair": ("applied" if changed else "failed_to_set_name"),
        "sink_joint_candidate": selected_joint,
        "sink_naming_prefix_before": current_prefix,
        "sink_naming_prefix_after": _sink_prefix(sink),
    }


def install_sink_runtime_compat(
    env: Any,
) -> dict[str, Any] | None:
    """Bind restored-model sink state accessors to one sink instance."""

    context = _sink_context(env)
    if context is None:
        return None
    sink, simulation, joint_names = context
    paired_prefixes: list[str] = []
    for name in joint_names:
        if (
            not isinstance(name, str)
            or not name.startswith("sink")
            or not name.endswith(_HANDLE_JOINT_SUFFIX)
        ):
            continue
        prefix = name[: -len(_HANDLE_JOINT_SUFFIX)]
        if f"{prefix}{_SPOUT_JOINT_SUFFIX}" in joint_names:
            paired_prefixes.append(prefix)
    if not paired_prefixes:
        return {"sink_runtime_patch": "no_prefixes"}

    prefix = min(paired_prefixes, key=len)
    handle_joint = f"{prefix}{_HANDLE_JOINT_SUFFIX}"
    spout_joint = f"{prefix}{_SPOUT_JOINT_SUFFIX}"
    temperature_joint = f"{prefix}{_TEMPERATURE_JOINT_SUFFIX}"
    has_temperature = temperature_joint in joint_names

    def set_handle_state(
        _sink: Any,
        env: Any,
        rng: Any,
        mode: str = "on",
    ) -> None:
        del env
        assert mode in ("on", "off", "random")
        selected_mode = rng.choice(["on", "off"]) if mode == "random" else mode
        value = 0.0 if selected_mode == "off" else float(rng.uniform(0.40, 0.50))
        _write_joint_scalar(
            simulation,
            handle_joint,
            value,
        )

    def get_handle_state(
        _sink: Any,
        env: Any,
    ) -> dict[str, Any]:
        handle_position = float(_read_joint_scalar(simulation, handle_joint)) % (
            2 * np.pi
        )
        if handle_position < 0:
            handle_position += 2 * np.pi
        try:
            handle_id = int(env.sim.model.joint_name2id(handle_joint))
            handle_maximum = float(env.sim.model.jnt_range[handle_id][1])
        except Exception:
            handle_maximum = float(2 * np.pi)
        use_ratio = handle_position / handle_maximum if handle_maximum != 0 else 0.0
        water_on = bool(0.40 < handle_position < np.pi)
        if use_ratio > 0.5 and water_on:
            water_pressure = "high"
        elif water_on:
            water_pressure = "low"
        else:
            water_pressure = "zero"

        spout_position = float(_read_joint_scalar(simulation, spout_joint)) % (
            2 * np.pi
        )
        if spout_position < 0:
            spout_position += 2 * np.pi
        if np.pi <= spout_position <= 2 * np.pi - np.pi / 6:
            spout_orientation = "left"
        elif np.pi / 6 <= spout_position <= np.pi:
            spout_orientation = "right"
        else:
            spout_orientation = "center"

        state: dict[str, Any] = {
            "handle_joint": handle_position,
            "water_on": water_on,
            "water_pressure": water_pressure,
            "spout_joint": spout_position,
            "spout_ori": spout_orientation,
        }
        if not has_temperature:
            return state

        temperature_position = float(
            _read_joint_scalar(
                simulation,
                temperature_joint,
            )
        )
        state["temp_joint"] = temperature_position
        try:
            temperature_id = int(env.sim.model.joint_name2id(temperature_joint))
            low, high = [
                float(value) for value in env.sim.model.jnt_range[temperature_id]
            ]
            midpoint = (low + high) / 2
            state["water_temp_state"] = (
                "hot" if temperature_position > midpoint else "cold"
            )
            state["water_temp"] = (
                0.0
                if high == low
                else float((temperature_position - low) / (high - low))
            )
        except Exception:
            state["water_temp_state"] = "unknown"
            state["water_temp"] = None
        return state

    sink.set_handle_state = MethodType(
        set_handle_state,
        sink,
    )
    sink.get_handle_state = MethodType(
        get_handle_state,
        sink,
    )
    return {
        "sink_runtime_patch": "applied",
        "sink_prefix": prefix,
        "handle_joint": handle_joint,
        "spout_joint": spout_joint,
        "temp_joint": (temperature_joint if has_temperature else None),
    }


def prepare_sink_runtime_compat(
    env: Any,
) -> dict[str, Any]:
    """Apply sink compatibility in current task-success preparation order."""

    metadata: dict[str, Any] = {}
    prefix_result = repair_sink_prefix_if_needed(env)
    if isinstance(prefix_result, dict):
        metadata["sink_prefix_repair"] = prefix_result
    runtime_result = install_sink_runtime_compat(env)
    if isinstance(runtime_result, dict):
        metadata["sink_runtime_patch"] = runtime_result
    return metadata


__all__ = [
    "install_sink_runtime_compat",
    "prepare_sink_runtime_compat",
    "repair_sink_prefix_if_needed",
]
