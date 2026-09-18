"""Explicit input/output boundary for environment-independent action plans."""

from __future__ import annotations

from typing import (
    TYPE_CHECKING,
    Any,
    Mapping,
    Optional,
    Protocol,
    Sequence,
    runtime_checkable,
)

from .config import ActionConfig

if TYPE_CHECKING:
    from .builder import StepBudgetResolver


ACTION_PLANNER_CONTRACT_VERSION = "action_planner"


@runtime_checkable
class ActionPlanner(Protocol):
    """Replaceable planner that preserves the current ``action`` boundary.

    The signature intentionally matches :meth:`ActionBuilder.build`; an
    implementation binds its own normalized :class:`ActionConfig` and receives
    only explicit trajectory/configuration inputs at runtime.
    """

    backend_id: str
    contract_version: str

    def build(
        self,
        *,
        uid: str,
        cfg: Mapping[str, Any],
        ee_traj: Mapping[str, Any],
        obj_traj: Optional[Mapping[str, Any]] = None,
        gripper_payload: Optional[Mapping[str, Any]] = None,
        ee_traj_path: str = "",
        obj_traj_path: Optional[str] = None,
        gripper_path: Optional[str] = None,
        controller_step_budgets: Optional[Sequence[float]] = None,
        step_budget_resolver: Optional["StepBudgetResolver"] = None,
    ) -> Mapping[str, Any]:
        """Build an aligned, environment-independent ``action`` payload."""


def _normalized_external_action_planner_identity(
    *,
    backend_id: Any,
    contract_version: Any,
    source: str,
) -> dict[str, str]:
    normalized_backend_id = str(backend_id or "").strip().lower()
    if (
        not normalized_backend_id
        or normalized_backend_id.startswith(".")
        or any(
            character not in "abcdefghijklmnopqrstuvwxyz0123456789._-"
            for character in normalized_backend_id
        )
    ):
        raise ValueError(f"{source}.backend_id must use [a-z0-9._-] and be non-empty")
    normalized_contract = str(contract_version or "").strip()
    if normalized_contract != ACTION_PLANNER_CONTRACT_VERSION:
        raise ValueError(
            f"{source}.contract_version must be {ACTION_PLANNER_CONTRACT_VERSION!r}"
        )
    return {
        "provider_kind": "external",
        "backend_id": normalized_backend_id,
        "contract_version": normalized_contract,
    }


def external_action_planner_identity(
    planner: ActionPlanner,
) -> dict[str, str]:
    """Resolve the mandatory identity of an explicitly injected planner."""

    return _normalized_external_action_planner_identity(
        backend_id=getattr(planner, "backend_id", None),
        contract_version=getattr(planner, "contract_version", None),
        source="action planner",
    )


def _validate_action_meta(meta: Mapping[str, Any]) -> None:
    if "uid" not in meta:
        raise ValueError("action meta.uid is required")

    required_sections = ("source", "action_space", "planner", "summary")
    sections: dict[str, Mapping[str, Any]] = {}
    for name in required_sections:
        value = meta.get(name)
        if not isinstance(value, Mapping):
            raise ValueError(f"action meta.{name} must be a mapping")
        sections[name] = value

    required_source_fields = {
        "ee_traj_path",
        "obj_traj_path",
        "gripper_path",
        "ee_key",
        "obj_key",
    }
    missing_source = sorted(required_source_fields - set(sections["source"]))
    if missing_source:
        raise ValueError("action meta.source is missing: " + ", ".join(missing_source))

    required_action_space_fields = {
        "type",
        "reference_frame",
        "position_unit",
        "rotation_unit",
        "orientation_mode",
        "has_orientation",
    }
    missing_action_space = sorted(
        required_action_space_fields - set(sections["action_space"])
    )
    if missing_action_space:
        raise ValueError(
            "action meta.action_space is missing: " + ", ".join(missing_action_space)
        )
    action_space = sections["action_space"]
    if str(action_space.get("type", "") or "") != "delta_6dof_gripper":
        raise ValueError("action meta.action_space.type must be 'delta_6dof_gripper'")
    if str(action_space.get("position_unit", "") or "") != "meter":
        raise ValueError("action position_unit must be 'meter'")
    if str(action_space.get("rotation_unit", "") or "") != "radian":
        raise ValueError("action rotation_unit must be 'radian'")
    if str(action_space.get("orientation_mode", "") or "") != "rotvec":
        raise ValueError("action orientation_mode must be 'rotvec'")

    provider = sections["planner"].get("provider")
    if provider is not None:
        if not isinstance(provider, Mapping):
            raise ValueError("action meta.planner.provider must be a mapping")
        if str(provider.get("provider_kind", "") or "") != "external":
            raise ValueError(
                "action meta.planner.provider.provider_kind must be 'external'"
            )
        _normalized_external_action_planner_identity(
            backend_id=provider.get("backend_id"),
            contract_version=provider.get("contract_version"),
            source="action meta.planner.provider",
        )


def validate_action_plan(
    payload: Mapping[str, Any],
    *,
    eef_frames: Sequence[int],
    gripper_payload: Optional[Mapping[str, Any]] = None,
) -> None:
    """Reject an injected plan that breaks frame/stage/checkpoint alignment."""

    if not isinstance(payload, Mapping):
        raise TypeError("action planner must return a mapping")
    meta = payload.get("meta")
    if not isinstance(meta, Mapping):
        raise ValueError("action plan must contain a meta mapping")
    if meta.get("format") != "action":
        raise ValueError("action planner must preserve meta.format='action'")
    _validate_action_meta(meta)

    checkpoints = payload.get("checkpoints")
    steps = payload.get("steps")
    if not isinstance(checkpoints, list) or not isinstance(steps, list):
        raise ValueError("action payload must contain checkpoints and steps lists")

    allowed_frames = [int(frame) for frame in eef_frames]
    allowed_set = set(allowed_frames)
    checkpoint_frames: list[int] = []
    gripper_by_frame: dict[int, Mapping[str, Any]] = {}
    if isinstance(gripper_payload, Mapping):
        for index, row in enumerate(list(gripper_payload.get("actions", []) or [])):
            if isinstance(row, Mapping):
                gripper_by_frame[int(row.get("frame", index))] = row

    for index, checkpoint in enumerate(checkpoints):
        if not isinstance(checkpoint, Mapping):
            raise ValueError(f"action checkpoint {index} must be a mapping")
        if int(checkpoint.get("checkpoint_index", -1)) != index:
            raise ValueError("action checkpoint_index must be contiguous and ordered")
        frame = int(checkpoint.get("frame", -1))
        if frame not in allowed_set:
            raise ValueError(
                f"action checkpoint frame {frame} is absent from EEF trajectory"
            )
        checkpoint_frames.append(frame)
        source_gripper = gripper_by_frame.get(frame)
        if source_gripper is not None:
            for field in ("stage_id", "object_id", "obj_key"):
                expected = source_gripper.get(field)
                actual = checkpoint.get(field)
                has_expected = expected is not None and str(expected) != ""
                has_actual = actual is not None and str(actual) != ""
                if has_expected and not has_actual:
                    raise ValueError(
                        f"action checkpoint is missing gripper {field} at frame={frame}"
                    )
                if has_expected and str(actual) != str(expected):
                    raise ValueError(
                        "action checkpoint does not preserve gripper "
                        f"{field} at frame={frame}"
                    )
    expected_order = [
        frame for frame in allowed_frames if frame in set(checkpoint_frames)
    ]
    if checkpoint_frames != expected_order:
        raise ValueError(
            "action checkpoints must be a unique ordered EEF-frame subsequence"
        )

    for index, step in enumerate(steps):
        if not isinstance(step, Mapping):
            raise ValueError(f"action step {index} must be a mapping")
        if int(step.get("step_index", -1)) != index:
            raise ValueError("action step_index must be contiguous and ordered")
        target_index = int(step.get("target_checkpoint_index", -1))
        if target_index < 0 or target_index >= len(checkpoints):
            raise ValueError("action step target_checkpoint_index is out of range")
        target_frame = int(step.get("target_frame", -1))
        if target_frame != checkpoint_frames[target_index]:
            raise ValueError(
                "action step target_frame must match its target checkpoint"
            )

    summary = meta["summary"]
    for field in ("num_checkpoints", "num_total_steps"):
        if field not in summary:
            raise ValueError(f"action summary is missing {field}")
    if int(summary["num_checkpoints"]) != len(checkpoints):
        raise ValueError("action summary.num_checkpoints does not match checkpoints")
    if int(summary["num_total_steps"]) != len(steps):
        raise ValueError("action summary.num_total_steps does not match steps")


__all__ = [
    "ACTION_PLANNER_CONTRACT_VERSION",
    "ActionPlanner",
    "ActionConfig",
    "external_action_planner_identity",
    "validate_action_plan",
]
