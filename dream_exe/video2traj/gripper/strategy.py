"""Unified, simulator-independent gripper strategy dispatch."""

from __future__ import annotations

from typing import Any, Dict, Literal, Optional

from .features import GraspParams, Method
from .numeric import compute_gripper_actions
from .numeric_config import load_numeric_action_params
from .task_prior import (
    TaskPriorActionParams,
    compute_gripper_actions_task_prior,
)


GripperStrategy = Literal["numeric", "task_prior"]


def compute_gripper_actions_by_strategy(
    ee_traj: Dict[str, Any],
    obj_traj: Dict[str, Any],
    *,
    strategy: GripperStrategy = "task_prior",
    method: Method = "3d",
    params: Optional[GraspParams | TaskPriorActionParams] = None,
    ee_key: str = "eef_controller",
    obj_key: str = "obj_visual_center",
    return_debug: bool = True,
    gripper_close_cmd: float = 1.0,
    gripper_open_cmd: float = -1.0,
    gripper_hold_cmd: float = 0.0,
    invalid_cmd_mode: Literal["hold", "open"] = "hold",
    task_name: Optional[str] = None,
    env_name: Optional[str] = None,
    uid: Optional[str] = None,
    dataset_config_path: Optional[str] = None,
    numeric_params_config_path: Optional[str] = None,
    task_prior_params_config_path: Optional[str] = None,
    prior_config_path: Optional[str] = None,
    num_close: Optional[int] = None,
    num_open: Optional[int] = None,
    stage_constrained: bool = False,
    close_timing_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Run one gripper strategy without pipeline, simulator, or artifact I/O."""

    common_kwargs = {
        "method": method,
        "ee_key": ee_key,
        "obj_key": obj_key,
        "return_debug": bool(return_debug),
        "gripper_close_cmd": float(gripper_close_cmd),
        "gripper_open_cmd": float(gripper_open_cmd),
        "gripper_hold_cmd": float(gripper_hold_cmd),
        "invalid_cmd_mode": invalid_cmd_mode,
    }
    if strategy == "numeric":
        numeric_params = params if isinstance(params, GraspParams) else None
        if numeric_params is None:
            numeric_params = load_numeric_action_params(numeric_params_config_path)
        return compute_gripper_actions(
            ee_traj,
            obj_traj,
            params=numeric_params,
            **common_kwargs,
        )
    if strategy == "task_prior":
        prior_params = params if isinstance(params, TaskPriorActionParams) else None
        return compute_gripper_actions_task_prior(
            ee_traj,
            obj_traj,
            task_name=task_name,
            env_name=env_name,
            uid=uid,
            dataset_config_path=dataset_config_path,
            prior_config_path=prior_config_path,
            num_close=num_close,
            num_open=num_open,
            params=prior_params,
            params_config_path=task_prior_params_config_path,
            base_params_config_path=numeric_params_config_path,
            stage_constrained=bool(stage_constrained),
            close_timing_profile=close_timing_profile,
            **common_kwargs,
        )
    raise ValueError(f"Unsupported action strategy: {strategy}")


class GripperRecognizer:
    def __init__(
        self,
        *,
        strategy: GripperStrategy = "task_prior",
        method: Method = "3d",
        params: Optional[GraspParams | TaskPriorActionParams] = None,
        ee_key: str = "eef_controller",
        obj_key: str = "obj_visual_center",
        return_debug: bool = True,
        gripper_close_cmd: float = 1.0,
        gripper_open_cmd: float = -1.0,
        gripper_hold_cmd: float = 0.0,
        invalid_cmd_mode: Literal["hold", "open"] = "hold",
        task_name: Optional[str] = None,
        env_name: Optional[str] = None,
        uid: Optional[str] = None,
        dataset_config_path: Optional[str] = None,
        numeric_params_config_path: Optional[str] = None,
        task_prior_params_config_path: Optional[str] = None,
        prior_config_path: Optional[str] = None,
        num_close: Optional[int] = None,
        num_open: Optional[int] = None,
        stage_constrained: bool = False,
        close_timing_profile: Optional[str] = None,
    ) -> None:
        vars(self).update(
            {
                "strategy": strategy,
                "method": method,
                "params": params,
                "ee_key": ee_key,
                "obj_key": obj_key,
                "return_debug": return_debug,
                "gripper_close_cmd": gripper_close_cmd,
                "gripper_open_cmd": gripper_open_cmd,
                "gripper_hold_cmd": gripper_hold_cmd,
                "invalid_cmd_mode": invalid_cmd_mode,
                "task_name": task_name,
                "env_name": env_name,
                "uid": uid,
                "dataset_config_path": dataset_config_path,
                "numeric_params_config_path": numeric_params_config_path,
                "task_prior_params_config_path": (task_prior_params_config_path),
                "prior_config_path": prior_config_path,
                "num_close": num_close,
                "num_open": num_open,
                "stage_constrained": stage_constrained,
                "close_timing_profile": close_timing_profile,
            }
        )

    def infer(
        self,
        *,
        ee_traj: Dict[str, Any],
        obj_traj: Dict[str, Any],
    ) -> Dict[str, Any]:
        if self.strategy == "numeric":
            numeric_params = (
                self.params
                if isinstance(self.params, GraspParams)
                else load_numeric_action_params(self.numeric_params_config_path)
            )
            return compute_gripper_actions(
                ee_traj,
                obj_traj,
                params=numeric_params,
                method=self.method,
                ee_key=self.ee_key,
                obj_key=self.obj_key,
                return_debug=self.return_debug,
                gripper_close_cmd=self.gripper_close_cmd,
                gripper_open_cmd=self.gripper_open_cmd,
                gripper_hold_cmd=self.gripper_hold_cmd,
                invalid_cmd_mode=self.invalid_cmd_mode,
            )

        if self.strategy == "task_prior":
            prior_params = (
                self.params if isinstance(self.params, TaskPriorActionParams) else None
            )
            return compute_gripper_actions_task_prior(
                ee_traj,
                obj_traj,
                task_name=self.task_name,
                env_name=self.env_name,
                uid=self.uid,
                dataset_config_path=self.dataset_config_path,
                prior_config_path=self.prior_config_path,
                num_close=self.num_close,
                num_open=self.num_open,
                params=prior_params,
                params_config_path=self.task_prior_params_config_path,
                base_params_config_path=self.numeric_params_config_path,
                stage_constrained=self.stage_constrained,
                close_timing_profile=self.close_timing_profile,
                method=self.method,
                ee_key=self.ee_key,
                obj_key=self.obj_key,
                return_debug=self.return_debug,
                gripper_close_cmd=self.gripper_close_cmd,
                gripper_open_cmd=self.gripper_open_cmd,
                gripper_hold_cmd=self.gripper_hold_cmd,
                invalid_cmd_mode=self.invalid_cmd_mode,
            )

        raise ValueError(f"Unsupported action strategy: {self.strategy}")


__all__ = [
    "GripperRecognizer",
    "GripperStrategy",
    "compute_gripper_actions_by_strategy",
]
