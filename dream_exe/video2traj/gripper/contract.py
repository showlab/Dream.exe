"""Explicit, simulator-independent contracts for gripper inference.

The gripper algorithm consumes already lifted trajectories.  Stage ordering
and timing remain orchestration data: a backend receives one bounded stage
request at a time and does not own multi-stage scheduling.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable


GRIPPER_INFERENCE_CONTRACT_VERSION = "gripper_inference"


@dataclass(frozen=True)
class GripperResourcePaths:
    """Resolved per-sample files used by the current gripper strategies."""

    dataset_config_path: str = ""
    numeric_config_path: str | None = None
    task_prior_params_config_path: str | None = None
    prior_config_path: str | None = None


@dataclass(frozen=True)
class GripperStageContext:
    """Identity and recognition window for one ordered task stage.

    ``annotated_stage_index`` retains the dataset's semantic stage index while
    ``runtime_order_index`` identifies this invocation's position after an
    optional coupling-based reorder.  The backend receives no global timeline
    and therefore cannot reinterpret stage ownership.
    """

    stage_id: str
    object_id: str
    annotated_stage_index: int
    runtime_order_index: int
    runtime_order_mode: str
    stage_order: tuple[str, ...]
    recognition_window: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class GripperInferenceInput:
    """All inputs needed to infer gripper state for one trajectory window.

    ``ee_trajectory`` and ``object_trajectory`` are frame-aligned slices.
    ``resolved_config`` is the per-sample gripper block, while
    ``recognizer_options`` records the exact strategy/timing values passed to
    the built-in recognizer.  The backend must not reorder stages or frames.
    """

    uid: str
    ee_trajectory: Mapping[str, Any]
    object_trajectory: Mapping[str, Any]
    environment_config: Mapping[str, Any]
    resolved_config: Mapping[str, Any]
    recognizer_options: Mapping[str, Any]
    resources: GripperResourcePaths
    stage: GripperStageContext


@dataclass(frozen=True)
class GripperInferenceOutput:
    """One stage's frame actions and provenance.

    Actions are the sole backend authority.  The core derives event segments
    from these actions after validation.
    """

    actions: tuple[Mapping[str, Any], ...]
    provenance: Mapping[str, Any] = field(default_factory=dict)
    debug: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, Any],
        *,
        allow_legacy_segments: bool = False,
    ) -> "GripperInferenceOutput":
        """Detach a current recognizer payload without changing its schema."""

        actions = payload.get("actions")
        provenance = payload.get("meta", {})
        debug = payload.get("debug", {})
        if not isinstance(actions, list):
            raise ValueError("gripper backend output must contain an actions list")
        if "segments" in payload and not allow_legacy_segments:
            raise ValueError(
                "gripper backend output must not provide segments; "
                "the core derives segments from actions"
            )
        if not isinstance(provenance, Mapping):
            raise ValueError("gripper backend output meta/provenance must be a mapping")
        if not isinstance(debug, Mapping):
            raise ValueError("gripper backend output debug must be a mapping")
        return cls(
            actions=tuple(copy.deepcopy(actions)),
            provenance=copy.deepcopy(dict(provenance)),
            debug=copy.deepcopy(dict(debug)),
        )

    def to_payload(self) -> dict[str, Any]:
        """Convert to the current recognizer dictionary representation."""

        payload: dict[str, Any] = {
            "meta": copy.deepcopy(dict(self.provenance)),
            "actions": copy.deepcopy(list(self.actions)),
        }
        if self.debug:
            payload["debug"] = copy.deepcopy(dict(self.debug))
        return payload


@runtime_checkable
class GripperInferenceBackend(Protocol):
    """Replaceable single-stage inference seam.

    Multi-stage ordering, timeline windows, and final action merging remain in
    the core orchestrator.  A backend only maps one explicit request to one
    aligned output.
    """

    backend_id: str
    contract_version: str

    def infer(
        self,
        request: GripperInferenceInput,
    ) -> GripperInferenceOutput | Mapping[str, Any]:
        """Infer frame-aligned gripper events for ``request``."""


def _normalized_external_gripper_identity(
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
    if normalized_contract != GRIPPER_INFERENCE_CONTRACT_VERSION:
        raise ValueError(
            f"{source}.contract_version must be {GRIPPER_INFERENCE_CONTRACT_VERSION!r}"
        )
    return {
        "provider_kind": "external",
        "backend_id": normalized_backend_id,
        "contract_version": normalized_contract,
    }


def external_gripper_backend_identity(
    backend: GripperInferenceBackend,
) -> dict[str, str]:
    """Resolve the mandatory identity of an explicitly injected backend."""

    return _normalized_external_gripper_identity(
        backend_id=getattr(backend, "backend_id", None),
        contract_version=getattr(backend, "contract_version", None),
        source="gripper inference backend",
    )


def _trajectory_frames(
    trajectory: Mapping[str, Any],
    *,
    key: str,
) -> list[int]:
    records = trajectory.get(key)
    if not isinstance(records, list):
        raise ValueError(f"gripper inference input must contain a {key!r} list")
    frames: list[int] = []
    for index, record in enumerate(records):
        if not isinstance(record, Mapping):
            raise ValueError(
                f"gripper inference {key!r} record {index} must be a mapping"
            )
        frames.append(int(record.get("frame", index)))
    if len(frames) != len(set(frames)):
        raise ValueError(f"gripper inference {key!r} frames must be unique")
    if frames != sorted(frames):
        raise ValueError(f"gripper inference {key!r} frames must be ordered")
    return frames


def validate_gripper_inference_input(
    request: GripperInferenceInput,
) -> None:
    """Validate the stage slice before invoking a replaceable backend."""

    ee_frames = _trajectory_frames(
        request.ee_trajectory,
        key="eef_controller",
    )
    object_frames = _trajectory_frames(
        request.object_trajectory,
        key="obj_visual_center",
    )
    if ee_frames != object_frames:
        raise ValueError("gripper EEF and object trajectory frames must align exactly")
    stage_id = str(request.stage.stage_id or "")
    if not stage_id:
        raise ValueError("gripper stage_id must be non-empty")
    order = tuple(str(value) for value in request.stage.stage_order)
    if stage_id not in order:
        raise ValueError(f"gripper stage_id {stage_id!r} is absent from stage_order")
    if len(order) != len(set(order)):
        raise ValueError("gripper stage_order must contain unique stage IDs")
    runtime_order_index = int(request.stage.runtime_order_index)
    if runtime_order_index < 0 or runtime_order_index >= len(order):
        raise ValueError("gripper runtime_order_index is outside stage_order")
    if order[runtime_order_index] != stage_id:
        raise ValueError(
            "gripper stage_id disagrees with stage_order/runtime_order_index"
        )
    if int(request.stage.annotated_stage_index) < 0:
        raise ValueError("gripper annotated_stage_index must be non-negative")
    if request.stage.runtime_order_mode not in {"annotated", "coupling"}:
        raise ValueError("gripper runtime_order_mode must be 'annotated' or 'coupling'")

    object_meta = request.object_trajectory.get("meta", {})
    if not isinstance(object_meta, Mapping):
        raise ValueError("gripper object trajectory meta must be a mapping")
    embedded_stage_id = str(object_meta.get("stage_id", "") or "")
    if embedded_stage_id != stage_id:
        raise ValueError("gripper object trajectory stage_id disagrees with context")
    object_id = str(request.stage.object_id or "")
    embedded_object_id = str(object_meta.get("object_id", "") or "")
    if embedded_object_id != object_id:
        raise ValueError("gripper object trajectory object_id disagrees with context")
    for name, expected in (
        ("annotated_stage_index", request.stage.annotated_stage_index),
        ("runtime_order_index", runtime_order_index),
        ("runtime_order_mode", request.stage.runtime_order_mode),
    ):
        if object_meta.get(name) != expected:
            raise ValueError(f"gripper object trajectory {name} disagrees with context")

    window = request.stage.recognition_window
    embedded_window = object_meta.get("recognition_window")
    if not isinstance(window, Mapping) or not isinstance(
        embedded_window,
        Mapping,
    ):
        raise ValueError("gripper recognition_window must be embedded as a mapping")
    if dict(embedded_window) != dict(window):
        raise ValueError("gripper embedded recognition_window disagrees with context")
    required_window_keys = {
        "start_index",
        "end_index",
        "start_frame",
        "end_frame",
    }
    if not required_window_keys.issubset(window):
        raise ValueError("gripper recognition_window is missing required fields")
    if ee_frames:
        start_index = int(window["start_index"])
        end_index = int(window["end_index"])
        if (
            start_index < 0
            or end_index < start_index
            or end_index - start_index + 1 != len(ee_frames)
            or int(window["start_frame"]) != ee_frames[0]
            or int(window["end_frame"]) != ee_frames[-1]
        ):
            raise ValueError("gripper recognition_window disagrees with input frames")
    elif any(window.get(key) is not None for key in required_window_keys):
        raise ValueError("empty gripper input requires an empty recognition_window")


def normalize_gripper_inference_output(
    output: GripperInferenceOutput | Mapping[str, Any],
    *,
    expected_frames: Sequence[int],
    allow_legacy_segments: bool = False,
    expected_provider_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Validate a backend result and return the current payload shape."""

    if isinstance(output, GripperInferenceOutput):
        normalized = output
    elif isinstance(output, Mapping):
        normalized = GripperInferenceOutput.from_payload(
            output,
            allow_legacy_segments=allow_legacy_segments,
        )
    else:
        raise TypeError(
            "gripper backend must return GripperInferenceOutput or a mapping"
        )

    expected = [int(frame) for frame in expected_frames]
    actual: list[int] = []
    for index, action in enumerate(normalized.actions):
        if not isinstance(action, Mapping):
            raise ValueError(f"gripper backend action {index} must be a mapping")
        missing = {
            "frame",
            "state",
            "event",
            "grasp",
            "gripper_cmd",
            "valid",
        }.difference(action)
        if missing:
            raise ValueError(
                "gripper backend action is missing required fields: "
                + ", ".join(sorted(missing))
            )
        actual.append(int(action["frame"]))
        try:
            command = float(action["gripper_cmd"])
        except (TypeError, ValueError) as exc:
            raise ValueError("gripper backend gripper_cmd must be numeric") from exc
        if not math.isfinite(command):
            raise ValueError("gripper backend gripper_cmd must be finite")
        if action["event"] not in {None, "close", "open"}:
            raise ValueError("gripper backend event must be null, 'close', or 'open'")
    if actual != expected:
        raise ValueError(
            "gripper backend actions must align exactly with the requested "
            f"frames: expected={expected}, actual={actual}"
        )
    payload = normalized.to_payload()
    provider = payload["meta"].get("provider")
    if provider is not None:
        if not isinstance(provider, Mapping):
            raise ValueError("gripper meta.provider must be a mapping")
        if str(provider.get("provider_kind", "") or "") != "external":
            raise ValueError("gripper meta.provider.provider_kind must be 'external'")
        provider = _normalized_external_gripper_identity(
            backend_id=provider.get("backend_id"),
            contract_version=provider.get("contract_version"),
            source="gripper meta.provider",
        )
    if expected_provider_identity is not None:
        expected_identity = _normalized_external_gripper_identity(
            backend_id=expected_provider_identity.get("backend_id"),
            contract_version=expected_provider_identity.get("contract_version"),
            source="expected gripper provider",
        )
        if provider is not None and provider != expected_identity:
            raise ValueError(
                "gripper output provider identity conflicts with the injected backend"
            )
        payload["meta"]["provider"] = expected_identity
    return payload


__all__ = [
    "GRIPPER_INFERENCE_CONTRACT_VERSION",
    "GripperInferenceBackend",
    "GripperInferenceInput",
    "GripperInferenceOutput",
    "GripperResourcePaths",
    "GripperStageContext",
    "external_gripper_backend_identity",
    "normalize_gripper_inference_output",
    "validate_gripper_inference_input",
]
