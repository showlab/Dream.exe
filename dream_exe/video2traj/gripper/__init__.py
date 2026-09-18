"""Environment-independent gripper inference."""

from .contract import (
    GRIPPER_INFERENCE_CONTRACT_VERSION,
    GripperInferenceBackend,
    GripperInferenceInput,
    GripperInferenceOutput,
    GripperResourcePaths,
    GripperStageContext,
    external_gripper_backend_identity,
    normalize_gripper_inference_output,
    validate_gripper_inference_input,
)
from .features import GraspParams, Method
from .numeric import NumericActionRecognizer, compute_gripper_actions
from .numeric_config import (
    DEFAULT_NUMERIC_ACTION_CONFIG_PATH,
    load_numeric_action_params,
)
from .orchestration import compute_gripper_payload
from .stages import (
    assign_stage_timelines,
    normalize_gripper_initial_state,
    stage_records_by_id,
)
from .strategy import (
    GripperRecognizer,
    GripperStrategy,
    compute_gripper_actions_by_strategy,
)
from .task_prior import (
    TaskPriorActionParams,
    TaskPriorActionRecognizer,
    TaskPriorSpec,
    compute_gripper_actions_task_prior,
    resolve_task_prior_spec,
)
from .task_prior_config import (
    DEFAULT_TASK_PRIOR_ACTION_CONFIG_PATH,
    load_task_prior_action_params,
)


__all__ = [
    "DEFAULT_NUMERIC_ACTION_CONFIG_PATH",
    "DEFAULT_TASK_PRIOR_ACTION_CONFIG_PATH",
    "GRIPPER_INFERENCE_CONTRACT_VERSION",
    "GraspParams",
    "GripperInferenceBackend",
    "GripperInferenceInput",
    "GripperInferenceOutput",
    "GripperRecognizer",
    "GripperResourcePaths",
    "GripperStageContext",
    "GripperStrategy",
    "Method",
    "NumericActionRecognizer",
    "TaskPriorActionParams",
    "TaskPriorActionRecognizer",
    "TaskPriorSpec",
    "assign_stage_timelines",
    "compute_gripper_actions",
    "compute_gripper_actions_by_strategy",
    "compute_gripper_actions_task_prior",
    "compute_gripper_payload",
    "external_gripper_backend_identity",
    "load_numeric_action_params",
    "load_task_prior_action_params",
    "normalize_gripper_initial_state",
    "normalize_gripper_inference_output",
    "resolve_task_prior_spec",
    "stage_records_by_id",
    "validate_gripper_inference_input",
]
