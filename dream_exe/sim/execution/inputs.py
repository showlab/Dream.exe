"""Parse simulator execution inputs without importing simulator runtimes.

This module intentionally contains only artifact-path resolution, JSON
validation, trajectory pose parsing, lightweight trajectory statistics, and
gripper-sidecar alignment.  It stays independent of trajectory-generation and
simulator-specific packages so that execution inputs remain cheap to import.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.spatial.transform import Rotation

from dream_exe.artifacts.layout import (
    ACTION_ARRAY_FILENAME,
    ACTION_FILENAME,
    GRIPPER_FILENAME,
    trajectory_artifact_paths,
)
from dream_exe.artifacts.action_bundle import motion_plan_payload_from_bundle


def _existing_path(candidates: list[Path]) -> str | None:
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve().as_posix()
    return None


def _unique_paths(candidates: list[Path]) -> list[Path]:
    unique: list[Path] = []
    seen: set[str] = set()
    for candidate in candidates:
        text = candidate.as_posix()
        if text in seen:
            continue
        seen.add(text)
        unique.append(candidate)
    return unique


def _tried_lines(candidates: list[Path]) -> str:
    return "\n".join(f"  - {candidate.as_posix()}" for candidate in candidates)


def resolve_action_json_path(
    action_path: str,
    *,
    traj_path: str,
    action_dir: str,
) -> str:
    """Resolve a canonical NPY bundle or an established JSON action artifact.

    The historical function name remains import-compatible for one transition
    cycle.  Explicit paths are never rewritten; inferred paths prefer the
    maintained JSON artifact and then the canonical ``action.npy`` bundle.
    """

    requested = str(action_path or "").strip()
    if requested:
        candidates = [Path(requested).expanduser()]
    else:
        trajectory = Path(str(traj_path or "").strip()).expanduser()
        candidates = []
        configured_dir = str(action_dir or "").strip()
        if configured_dir:
            configured_path = Path(configured_dir).expanduser()
            candidates.append(configured_path / ACTION_FILENAME)
            candidates.append(configured_path / ACTION_ARRAY_FILENAME)
            root_action = configured_path.parent / ACTION_FILENAME
            root_action_array = configured_path.parent / ACTION_ARRAY_FILENAME
        else:
            root_action = Path(ACTION_FILENAME)
            root_action_array = Path(ACTION_ARRAY_FILENAME)
        canonical = trajectory_artifact_paths(trajectory.parent.parent)["action"]
        candidates.extend(
            [
                root_action,
                root_action_array,
                trajectory.with_name(ACTION_FILENAME),
                trajectory.with_name(ACTION_ARRAY_FILENAME),
                trajectory.parent / "action" / ACTION_FILENAME,
                trajectory.parent / "action" / ACTION_ARRAY_FILENAME,
                canonical,
                canonical.with_name(ACTION_ARRAY_FILENAME),
            ]
        )
        candidates = _unique_paths(candidates)

    resolved = _existing_path(candidates)
    if resolved is not None:
        return resolved
    label = requested if requested else "<infer>"
    raise FileNotFoundError(
        "[exec][FATAL] action artifact not found. "
        f"action_path={label}\nTried:\n{_tried_lines(candidates)}"
    )


def load_action_stream(
    action_path: str,
) -> Tuple[
    Dict[str, Any],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
]:
    """Load and validate JSON or canonical ``action.npy + meta.json``."""

    source = Path(action_path).expanduser()
    if source.is_dir() or source.name == ACTION_ARRAY_FILENAME:
        payload = motion_plan_payload_from_bundle(source)
    else:
        with source.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(
            f"Invalid action payload type in {action_path}: {type(payload).__name__}"
        )
    steps = payload.get("steps")
    if not isinstance(steps, list) or not steps:
        raise ValueError(f"action steps missing or empty in {action_path}")
    checkpoints = payload.get("checkpoints")
    if not isinstance(checkpoints, list) or not checkpoints:
        raise ValueError(f"action checkpoints missing or empty in {action_path}")
    return payload, steps, checkpoints


def _manifest_trajectory_candidates(
    trajectory_dir: Path,
) -> list[Path]:
    manifest_path = trajectory_artifact_paths(trajectory_dir)[
        "trajectory_manifest"
    ]
    if not manifest_path.exists():
        return []
    try:
        with open(manifest_path, "r", encoding="utf-8") as handle:
            manifest = json.load(handle)
    except Exception:
        return []
    if not isinstance(manifest, dict):
        return []
    section = manifest.get("trajectory")
    if not isinstance(section, dict):
        return []
    configured = section.get("ee_traj_json")
    if not isinstance(configured, str) or not configured.strip():
        return []
    configured_path = Path(configured.strip()).expanduser()
    return [
        configured_path,
        Path.cwd() / configured_path,
    ]


def resolve_traj_json_path(
    traj_path: str,
    *,
    traj_dir: str,
) -> str:
    """Resolve a trajectory artifact across current and legacy locations."""

    requested_text = str(traj_path or "").strip()
    requested = Path(requested_text).expanduser()
    filename = requested.name
    configured_root = Path(str(traj_dir or "").strip()).expanduser()
    candidates = [requested]

    if filename in {
        "ee_traj.json",
        "obj_traj.json",
        "obj_trajs.json",
    }:
        candidates.append(requested.parent / "trajectory" / filename)
    if requested.parent.name == "trajectory":
        candidates.append(requested.parent.parent / filename)
    candidates.extend(
        [
            trajectory_artifact_paths(configured_root)["trajectory_dir"] / filename,
            configured_root / filename,
        ]
    )
    candidates.extend(_manifest_trajectory_candidates(configured_root))
    candidates = _unique_paths(candidates)

    resolved = _existing_path(candidates)
    if resolved is not None:
        return resolved
    raise FileNotFoundError(
        "[exec][FATAL] trajectory json not found. "
        f"requested={requested_text}\nTried:\n{_tried_lines(candidates)}"
    )


def load_traj_stream(
    traj_path: str,
    traj_key: str,
    max_steps: int,
) -> Tuple[Dict[str, Any], List[Dict[str, Any]], int]:
    """Load one non-empty trajectory list and apply an optional step limit."""

    with open(traj_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if traj_key not in payload:
        raise KeyError(
            f"traj_key={traj_key} not found in traj json keys={list(payload.keys())}"
        )
    trajectory = payload[traj_key]
    if not isinstance(trajectory, list) or not trajectory:
        raise ValueError(f"traj[{traj_key}] is empty or not a list")
    if max_steps > 0:
        trajectory = trajectory[: int(max_steps)]
    return payload, trajectory, len(trajectory)


def _fixed_vector(value: Any, size: int) -> np.ndarray | None:
    if value is None:
        return None
    array = np.asarray(value, dtype=np.float64).reshape(-1)
    return array if array.size == size else None


def _quaternion_xyzw(quaternion: np.ndarray) -> np.ndarray:
    if abs(float(quaternion[0])) > abs(float(quaternion[3])):
        return quaternion[[1, 2, 3, 0]]
    return quaternion


def parse_pose_from_entry(
    entry: Dict[str, Any],
    traj_key: str,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """Return a position and optional rotation from one trajectory record."""

    if traj_key in {"eef_tcp", "eef_controller"}:
        position = np.asarray(
            entry.get("pos_world"),
            dtype=np.float64,
        ).reshape(3)
        rotation_value = entry.get("R")
        rotation = (
            None
            if rotation_value is None
            else np.asarray(
                rotation_value,
                dtype=np.float64,
            ).reshape(3, 3)
        )
        return position, rotation

    if traj_key == "visual_center":
        position = np.asarray(
            entry.get("pos_world"),
            dtype=np.float64,
        ).reshape(3)
        return position, None

    for key in ("pose7", "eef_pose7"):
        pose = _fixed_vector(entry.get(key), 7)
        if pose is not None:
            return (
                pose[:3].copy(),
                Rotation.from_quat(_quaternion_xyzw(pose[3:])).as_matrix(),
            )

    pose6 = _fixed_vector(entry.get("pose6"), 6)
    if pose6 is not None:
        return (
            pose6[:3].copy(),
            Rotation.from_rotvec(pose6[3:]).as_matrix(),
        )

    raise ValueError(
        f"Cannot parse pose from entry keys={list(entry.keys())}, traj_key={traj_key}"
    )


PoseParser = Callable[..., tuple[np.ndarray, np.ndarray | None]]


def traj_motion_stats(
    traj_list,
    traj_key: str,
    T: int,
    parse_pose_fn=parse_pose_from_entry,
):
    """Compute and print the established translation-only motion summary."""

    positions = np.stack(
        [
            parse_pose_fn(
                traj_list[index],
                traj_key=traj_key,
            )[0]
            for index in range(T)
        ],
        axis=0,
    )
    differences = np.diff(positions, axis=0)
    path_length = float(np.sum(np.linalg.norm(differences, axis=1)))
    radius_max = float(np.max(np.linalg.norm(positions - positions[0], axis=1)))
    position_min = np.min(positions, axis=0)
    position_max = np.max(positions, axis=0)
    diagonal = float(np.linalg.norm(position_max - position_min))

    print(f"[traj-stats] T = {T}")
    print(f"[traj-stats] path_length Σ||Δp|| = {path_length}")
    print(f"[traj-stats] max ||p - p0||      = {radius_max}")
    print(f"[traj-stats] bbox diag ||max-min|| = {diagonal}")
    return {
        "L": path_length,
        "Rmax": radius_max,
        "diag": diagonal,
        "pmin": position_min,
        "pmax": position_max,
    }


def estimate_horizon_steps_from_traj(
    traj_list,
    traj_key: str,
    T: int,
    step_max_pos: float,
    parse_pose_fn,
    safety: float = 2.0,
) -> int:
    """Estimate a conservative frame-execution horizon."""

    position_step = max(float(step_max_pos), 1.0e-6)
    previous = np.asarray(
        parse_pose_fn(
            traj_list[0],
            traj_key=traj_key,
        )[0],
        dtype=np.float64,
    ).reshape(3)
    estimated_steps = 20
    for index in range(1, T):
        current = np.asarray(
            parse_pose_fn(
                traj_list[index],
                traj_key=traj_key,
            )[0],
            dtype=np.float64,
        ).reshape(3)
        largest_axis_delta = float(np.max(np.abs(current - previous)))
        estimated_steps += int(math.ceil(largest_axis_delta / position_step)) + 2
        previous = current
    return max(200, int(float(safety) * estimated_steps))


@dataclass(frozen=True)
class GripperScheduleMeta:
    """Describe how an aligned gripper schedule was obtained."""

    source: str
    key: str
    path: str
    invalid_policy: str
    cmd_open: float
    cmd_close: float
    cmd_hold: float


def _normalized_invalid_policy(value: str) -> str:
    normalized = str(value or "").strip().lower()
    if normalized not in {"hold", "open", "close"}:
        return "hold"
    return normalized


def _schedule_metadata(
    *,
    source: str,
    path: str,
    invalid_policy: str,
    cmd_open: float,
    cmd_close: float,
    cmd_hold: float,
) -> GripperScheduleMeta:
    return GripperScheduleMeta(
        source=source,
        key=(
            "actions[].(frame,gripper_cmd,valid)"
            if source == "gripper_json"
            else "--gripper"
        ),
        path=path,
        invalid_policy=invalid_policy,
        cmd_open=cmd_open,
        cmd_close=cmd_close,
        cmd_hold=cmd_hold,
    )


def build_gripper_schedule_from_sidecar(
    *,
    traj_path: str,
    T: int,
    enable: bool = True,
    default_cmd: float = 0.0,
    cmd_open: float = -1.0,
    cmd_close: float = 1.0,
    cmd_hold: float = 0.0,
    invalid_policy: str = "hold",
) -> Tuple[np.ndarray, np.ndarray, GripperScheduleMeta]:
    """Align sparse gripper sidecar events to trajectory indices."""

    default_value = float(default_cmd)
    open_value = float(cmd_open)
    close_value = float(cmd_close)
    hold_value = float(cmd_hold)
    policy = _normalized_invalid_policy(invalid_policy)
    commands = np.full((T,), default_value, dtype=np.float64)
    edge_flags = np.zeros((T,), dtype=np.int8)

    if not enable:
        return (
            commands,
            edge_flags,
            _schedule_metadata(
                source="constant",
                path="N/A",
                invalid_policy=policy,
                cmd_open=open_value,
                cmd_close=close_value,
                cmd_hold=hold_value,
            ),
        )

    trajectory = Path(str(traj_path or "").strip()).expanduser()
    canonical_gripper = trajectory_artifact_paths(trajectory.parent.parent)["gripper"]
    candidates = [
        trajectory.with_name(GRIPPER_FILENAME),
        canonical_gripper,
        trajectory.parent.parent / GRIPPER_FILENAME,
    ]
    gripper_path = candidates[0]
    for candidate in candidates:
        if candidate.exists():
            gripper_path = candidate
            break
    resolved_path = gripper_path.resolve().as_posix()
    if not gripper_path.exists():
        return (
            commands,
            edge_flags,
            _schedule_metadata(
                source="constant_no_gripper_json",
                path=resolved_path,
                invalid_policy=policy,
                cmd_open=open_value,
                cmd_close=close_value,
                cmd_hold=hold_value,
            ),
        )

    with open(gripper_path, "r", encoding="utf-8") as handle:
        payload = json.load(handle)
    actions = payload.get("actions")
    if not isinstance(actions, list):
        raise ValueError(
            f"Invalid gripper sidecar: missing list 'actions' in {resolved_path}"
        )

    commands.fill(hold_value)
    events: dict[int, float] = {}
    for record in actions:
        if not isinstance(record, dict):
            continue
        frame_value = record.get("frame")
        if frame_value is None:
            continue
        frame = int(frame_value)
        if frame < 0 or frame >= T:
            continue

        if not bool(record.get("valid", True)):
            if policy == "hold":
                continue
            events[frame] = open_value if policy == "open" else close_value
            continue

        command = record.get("gripper_cmd")
        if command is None:
            raise ValueError(
                f"gripper sidecar frame {frame} is valid=true but missing gripper_cmd"
            )
        events[frame] = float(command)

    current = hold_value
    for index in range(T):
        if index in events:
            current = events[index]
        commands[index] = current

    if T > 1:
        closed = np.isclose(commands, close_value)
        edge_flags[1:] = np.diff(closed.astype(np.int8))

    return (
        commands,
        edge_flags,
        _schedule_metadata(
            source="gripper_json",
            path=resolved_path,
            invalid_policy=policy,
            cmd_open=open_value,
            cmd_close=close_value,
            cmd_hold=hold_value,
        ),
    )


__all__ = [
    "GripperScheduleMeta",
    "build_gripper_schedule_from_sidecar",
    "estimate_horizon_steps_from_traj",
    "load_action_stream",
    "load_traj_stream",
    "parse_pose_from_entry",
    "resolve_action_json_path",
    "resolve_traj_json_path",
    "traj_motion_stats",
]
