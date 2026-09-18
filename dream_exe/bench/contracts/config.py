"""Compose packaged defaults, shared protocol, and one case protocol."""

from __future__ import annotations

import copy
import re
from collections.abc import Mapping
from typing import Any

from .defaults import (
    RUN_OWNED_FIELDS,
    packaged_default_digests,
    packaged_default_documents,
)
from ..data.repository import BenchRepository
from .schemas import (
    CASE_PROTOCOL_ROUTES,
    CASE_PROTOCOL_SCHEMA,
    PIPELINE_STAGES,
    PROTOCOL_SCHEMA,
    RESOLVED_CONFIG_SCHEMA,
    canonical_sha256,
    validate_document,
)


def _escape_pointer(value: str) -> str:
    return value.replace("~", "~0").replace("/", "~1")


def _leaf_items(value: Any, prefix: str):
    if isinstance(value, Mapping):
        if value:
            for key in sorted(value):
                yield from _leaf_items(
                    value[key],
                    f"{prefix}/{_escape_pointer(str(key))}",
                )
        return
    yield prefix, value


def _contains_machine_path(value: Any, key: str = "") -> bool:
    if isinstance(value, Mapping):
        return any(
            _contains_machine_path(item, str(name))
            for name, item in value.items()
        )
    if isinstance(value, list):
        return any(_contains_machine_path(item, key) for item in value)
    if not isinstance(value, str) or not value:
        return False
    lowered = key.lower()
    if lowered.endswith(("_path", "_root", "_dir")) and value.startswith(("/", "~/")):
        return True
    return value.startswith(("/", "~/", "file://")) or bool(
        re.match(r"^[A-Za-z]:[\\/]", value)
    )


_BENCH_EXTRA_FIELDS: dict[str, frozenset[tuple[str, ...]]] = {
    "video2traj": frozenset(
        {
            ("gripper", "initial_state"),
            ("pose", "config_path"),
            ("region", "targets", "eef", "visual", "bbox_source"),
            ("task", "gripper_init"),
            ("task", "gripper_plan", "num_close"),
            ("task", "gripper_plan", "num_open"),
            ("task", "interaction_count"),
            ("task", "interaction_modes"),
            ("task", "level"),
            ("task", "multi_object_interaction"),
            ("task", "notes"),
            ("task", "stages"),
            ("task", "task_types"),
            ("input", "max_res"),
            ("stage_order_method",),
            ("tracking", "use_cache"),
        }
    ),
    "action": frozenset(),
    "execution": frozenset(),
    "evaluation": frozenset(
        {
            ("vlm", "task_prompt_suffixes", "standard"),
            ("vlm", "task_prompt_suffixes", "enhanced"),
            (
                "vlm",
                "rubrics",
                "subject_stability",
                "prompt_files",
                "robot_subject",
            ),
            (
                "vlm",
                "rubrics",
                "subject_stability",
                "prompt_files",
                "manipulated_object",
            ),
            (
                "vlm",
                "rubrics",
                "subject_stability",
                "sampling",
                "frame_fractions",
            ),
            (
                "vlm",
                "rubrics",
                "subject_stability",
                "sampling",
                "grid",
            ),
            ("vlm", "rubrics", "subject_stability", "score_range"),
            ("vlm", "rubrics", "physical_plausibility", "prompt_file"),
            (
                "vlm",
                "rubrics",
                "physical_plausibility",
                "sampling",
                "uniform_frames",
            ),
            (
                "vlm",
                "rubrics",
                "physical_plausibility",
                "sampling",
                "grid",
            ),
            ("vlm", "rubrics", "physical_plausibility", "score_range"),
            ("vlm", "rubrics", "task_adherence", "prompt_file"),
            (
                "vlm",
                "rubrics",
                "task_adherence",
                "sampling",
                "uniform_frames",
            ),
            ("vlm", "rubrics", "task_adherence", "sampling", "grid"),
            ("vlm", "rubrics", "task_adherence", "score_range"),
            ("vlm", "judges"),
            ("vlm", "aggregation", "method"),
            ("vlm", "aggregation", "required_judges"),
            ("trajectory", "groups"),
            ("trajectory", "metrics"),
            ("execution", "executability_metrics"),
            ("execution", "task_metrics"),
        }
    ),
}


def _structural_leaves(
    value: Any,
    prefix: tuple[str, ...] = (),
):
    if isinstance(value, Mapping):
        if not value:
            yield prefix, True
            return
        for key in sorted(value):
            yield from _structural_leaves(value[key], (*prefix, str(key)))
        return
    yield prefix, False


def _default_leaf_paths(defaults: Mapping[str, Any]) -> set[tuple[str, ...]]:
    return {path for path, _empty in _structural_leaves(defaults) if path}


def _validate_owned_stage_values(
    values: Mapping[str, Any],
    *,
    stage: str,
    owner: str,
    defaults: Mapping[str, Any],
    run_owned: bool,
) -> None:
    if not values:
        return
    allowed = (
        set(RUN_OWNED_FIELDS.get(stage, ()))
        if run_owned
        else _default_leaf_paths(defaults) | set(_BENCH_EXTRA_FIELDS[stage])
    )
    unknown: list[str] = []
    for path, empty_mapping in _structural_leaves(values):
        if path in allowed:
            continue
        if empty_mapping and any(
            len(candidate) > len(path) and candidate[: len(path)] == path
            for candidate in allowed
        ):
            continue
        unknown.append("/" + "/".join(_escape_pointer(item) for item in path))
    if unknown:
        raise ValueError(
            f"{owner} contains unknown {stage} field(s): {sorted(unknown)}"
        )


def _merge_disjoint(
    target: dict[str, Any],
    incoming: Mapping[str, Any],
    *,
    pointer: str,
    current_owner: str,
    incoming_owner: str,
) -> None:
    for key, value in incoming.items():
        clean_key = str(key)
        child_pointer = f"{pointer}/{_escape_pointer(clean_key)}"
        if clean_key not in target:
            target[clean_key] = copy.deepcopy(value)
            continue
        existing = target[clean_key]
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            mutable = dict(existing)
            _merge_disjoint(
                mutable,
                value,
                pointer=child_pointer,
                current_owner=current_owner,
                incoming_owner=incoming_owner,
            )
            target[clean_key] = mutable
            continue
        raise ValueError(
            "configuration leaf has more than one owner: "
            f"{child_pointer} ({current_owner} and {incoming_owner})"
        )


def _record_sources(
    sources: dict[str, str],
    values: Mapping[str, Any],
    *,
    stage: str,
    owner: str,
) -> None:
    for pointer, _value in _leaf_items(values, f"/{_escape_pointer(stage)}"):
        if pointer in sources:
            raise ValueError(
                f"configuration leaf has duplicate source owner: {pointer}"
            )
        sources[pointer] = owner


def _fill_packaged_defaults(
    target: dict[str, Any],
    defaults: Mapping[str, Any],
    *,
    sources: dict[str, str],
    stage: str,
    pointer: str,
) -> None:
    """Fill absent leaves without competing with an explicit bench owner."""

    for key, value in defaults.items():
        clean_key = str(key)
        child_pointer = f"{pointer}/{_escape_pointer(clean_key)}"
        if clean_key not in target:
            target[clean_key] = copy.deepcopy(value)
            for leaf_pointer, _leaf in _leaf_items(value, child_pointer):
                if leaf_pointer in sources:  # pragma: no cover - invariant
                    raise ValueError(
                        "packaged default conflicts with an explicit owner: "
                        f"{leaf_pointer}"
                    )
                sources[leaf_pointer] = f"packaged-default:{stage}"
            continue
        existing = target[clean_key]
        if isinstance(existing, Mapping) and isinstance(value, Mapping):
            mutable = dict(existing)
            _fill_packaged_defaults(
                mutable,
                value,
                sources=sources,
                stage=stage,
                pointer=child_pointer,
            )
            target[clean_key] = mutable


def compile_config(
    *,
    uid: str,
    input_identity: Mapping[str, Any],
    protocols: Mapping[str, Mapping[str, Any]],
    case_protocol: Mapping[str, Any],
    route: str = "candidate",
    run_values: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile one complete pipeline config without profile or patch layers.

    Every final leaf has exactly one owner.  A leaf that varies for even one
    case must therefore be removed from the common protocol and written in the
    corresponding case protocol for every case.  This makes collisions fail
    loudly and keeps ownership understandable without JSON Pointer patches.
    """

    if route not in CASE_PROTOCOL_ROUTES:
        raise ValueError(f"unknown case protocol route: {route!r}")
    case_document = validate_document(
        case_protocol,
        expected_schema=CASE_PROTOCOL_SCHEMA,
    )
    if case_document["uid"] != uid:
        raise ValueError(f"case protocol identity mismatch for {uid!r}")
    route_values = case_document["routes"][route]
    values: dict[str, Any] = {}
    sources: dict[str, str] = {}
    runtime_values = {} if run_values is None else dict(run_values)
    unknown_runtime = sorted(set(runtime_values) - set(PIPELINE_STAGES))
    if unknown_runtime:
        raise ValueError(f"run config has unknown stages: {unknown_runtime}")

    packaged = packaged_default_documents()
    for stage in PIPELINE_STAGES:
        if stage not in protocols:
            raise ValueError(f"missing root protocol document for {stage!r}")
        protocol = validate_document(
            protocols[stage],
            expected_schema=PROTOCOL_SCHEMA,
        )
        if protocol["stage"] != stage:
            raise ValueError(f"protocol identity mismatch for {stage!r}")
        common = protocol["values"]
        if _contains_machine_path(common):
            raise ValueError(f"bench protocol contains a machine path: {stage}")
        stage_defaults = packaged[stage]["values"]
        _validate_owned_stage_values(
            common,
            stage=stage,
            owner=f"protocol:{stage}",
            defaults=stage_defaults,
            run_owned=False,
        )
        merged: dict[str, Any] = copy.deepcopy(dict(common))
        _record_sources(
            sources,
            common,
            stage=stage,
            owner=f"protocol:{stage}",
        )

        case_values = route_values.get(stage, {})
        if case_values:
            if _contains_machine_path(case_values):
                raise ValueError(
                    "bench case protocol contains a machine path: "
                    f"{uid}/{route}/{stage}"
                )
            _validate_owned_stage_values(
                case_values,
                stage=stage,
                owner=f"case-protocol:{uid}:{route}:{stage}",
                defaults=stage_defaults,
                run_owned=False,
            )
            _merge_disjoint(
                merged,
                case_values,
                pointer=f"/{_escape_pointer(stage)}",
                current_owner=f"protocol:{stage}",
                incoming_owner=f"case-protocol:{uid}:{route}:{stage}",
            )
            _record_sources(
                sources,
                case_values,
                stage=stage,
                owner=f"case-protocol:{uid}:{route}:{stage}",
            )

        stage_runtime = runtime_values.get(stage, {})
        if not isinstance(stage_runtime, Mapping):
            raise TypeError(f"run config values for {stage!r} must be an object")
        _validate_owned_stage_values(
            stage_runtime,
            stage=stage,
            owner=f"run:{stage}",
            defaults=stage_defaults,
            run_owned=True,
        )
        _merge_disjoint(
            merged,
            stage_runtime,
            pointer=f"/{_escape_pointer(stage)}",
            current_owner=f"bench:{stage}",
            incoming_owner=f"run:{stage}",
        )
        _record_sources(
            sources,
            stage_runtime,
            stage=stage,
            owner=f"run:{stage}",
        )
        if _contains_machine_path(stage_defaults):  # pragma: no cover - source invariant
            raise RuntimeError(f"packaged defaults contain a machine path: {stage}")
        _fill_packaged_defaults(
            merged,
            stage_defaults,
            sources=sources,
            stage=stage,
            pointer=f"/{_escape_pointer(stage)}",
        )
        values[stage] = merged

    document = {
        "format": RESOLVED_CONFIG_SCHEMA,
        "uid": uid,
        "input": dict(input_identity),
        "values": values,
        "sources": dict(sorted(sources.items())),
    }
    return validate_document(document, expected_schema=RESOLVED_CONFIG_SCHEMA)


def compile_repository_config(
    repository: BenchRepository,
    *,
    uid: str,
    input_identity: Mapping[str, Any],
    route: str = "candidate",
    run_values: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    if route not in CASE_PROTOCOL_ROUTES:
        raise ValueError(f"unknown case protocol route: {route!r}")
    protocols = {
        stage: (
            repository.load_protocol(stage)
            if route == "candidate" or stage == "evaluation"
            else {
                "format": PROTOCOL_SCHEMA,
                "stage": stage,
                "values": {},
            }
        )
        for stage in PIPELINE_STAGES
    }
    return compile_config(
        uid=uid,
        input_identity=input_identity,
        protocols=protocols,
        case_protocol=repository.load_case_protocol(uid),
        route=route,
        run_values=run_values,
    )


def compile_case_route_config(
    repository: BenchRepository,
    *,
    uid: str,
    route: str,
    input_identity: Mapping[str, Any],
    run_values: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Compile a full bench-owned reference route outside GT assets."""

    if route not in {"reference_input", "evaluation_oracle"}:
        raise ValueError(
            "case route config must be reference_input or evaluation_oracle"
        )
    return compile_repository_config(
        repository,
        uid=uid,
        input_identity=input_identity,
        route=route,
        run_values=run_values,
    )


def config_identity(
    repository: BenchRepository,
    *,
    uid: str,
    route: str = "candidate",
) -> dict[str, Any]:
    """Return strict digest maps used by result request receipts."""

    protocol = {
        stage: canonical_sha256(repository.load_protocol(stage))
        for stage in repository.manifest["protocol_stages"]
    }
    if route not in CASE_PROTOCOL_ROUTES:
        raise ValueError(f"unknown case protocol route: {route!r}")
    case_protocol = repository.load_case_protocol(uid)
    return {
        "packaged_defaults": packaged_default_digests(),
        "protocol": protocol,
        "case_protocol": canonical_sha256(case_protocol),
    }


__all__ = [
    "compile_config",
    "compile_case_route_config",
    "compile_repository_config",
    "config_identity",
]
