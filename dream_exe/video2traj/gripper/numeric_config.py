"""Load the packaged current numeric gripper defaults."""

from __future__ import annotations

import copy
import json
from dataclasses import fields, replace
from pathlib import Path
from typing import Any, Dict, Optional, TypeVar

from .features import GraspParams


DEFAULT_NUMERIC_ACTION_CONFIG_PATH = (
    Path(__file__).resolve().parent.parent / "configs" / "gripper_numeric.default.json"
)

T = TypeVar("T")


def _load_action_config(
    path: Optional[str],
    default_path: Path,
) -> Dict[str, Any]:
    cfg_path = Path(path).expanduser().resolve() if path else default_path
    if not cfg_path.exists():
        raise FileNotFoundError(f"action config not found: {cfg_path}")
    with cfg_path.open("r", encoding="utf-8") as handle:
        raw = json.load(handle)
    if not isinstance(raw, dict):
        raise ValueError(f"action config must be a JSON object: {cfg_path}")
    return raw


def _merge_dataclass(instance: T, override: Dict[str, Any]) -> T:
    data = {
        field.name: copy.deepcopy(getattr(instance, field.name))
        for field in fields(instance)
    }
    for key, value in override.items():
        if key in data:
            data[key] = copy.deepcopy(value)
    return replace(instance, **data)


def load_numeric_action_params(
    config_path: Optional[str] = None,
) -> GraspParams:
    raw = _load_action_config(
        config_path,
        DEFAULT_NUMERIC_ACTION_CONFIG_PATH,
    )
    return _merge_dataclass(GraspParams(), raw)


__all__ = [
    "DEFAULT_NUMERIC_ACTION_CONFIG_PATH",
    "load_numeric_action_params",
]
