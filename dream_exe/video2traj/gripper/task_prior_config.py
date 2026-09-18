"""Load the packaged current task-prior gripper configuration."""

from __future__ import annotations

from dataclasses import fields, replace

from pathlib import Path
from typing import Optional

from .features import GraspParams
from .numeric_config import (
    _load_action_config,
    load_numeric_action_params,
)


DEFAULT_TASK_PRIOR_ACTION_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "gripper_task_prior.default.json"
)


def load_task_prior_action_params(
    config_path: Optional[str] = None,
    *,
    base: Optional[GraspParams] = None,
):
    from dream_exe.video2traj.gripper.task_prior import (
        TaskPriorActionParams,
    )

    document = _load_action_config(
        config_path,
        DEFAULT_TASK_PRIOR_ACTION_CONFIG_PATH,
    )
    selected = dict(document.get("params", document))
    base_changes = selected.pop("base", {})

    base_template = load_numeric_action_params() if base is None else base
    base_field_names = {item.name for item in fields(base_template)}
    merged_base = replace(
        base_template,
        **{
            name: value
            for name, value in base_changes.items()
            if name in base_field_names
        },
    )

    outer_template = TaskPriorActionParams(base=merged_base)
    outer_field_names = {item.name for item in fields(outer_template)}
    return replace(
        outer_template,
        **{
            name: value for name, value in selected.items() if name in outer_field_names
        },
    )


__all__ = [
    "DEFAULT_TASK_PRIOR_ACTION_CONFIG_PATH",
    "load_task_prior_action_params",
]
