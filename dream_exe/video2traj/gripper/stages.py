"""Compatibility exports for shared task-stage helpers.

The implementations live in :mod:`dream_exe.video2traj.trajectory.stages` so
trajectory assembly and gripper orchestration use one compatible source.
"""

from ..trajectory.stages import (
    assign_stage_timelines,
    normalize_gripper_initial_state,
    stage_records_by_id,
)


__all__ = [
    "assign_stage_timelines",
    "normalize_gripper_initial_state",
    "stage_records_by_id",
]
