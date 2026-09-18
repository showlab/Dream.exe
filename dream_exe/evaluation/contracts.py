"""Durable, bench-owned results for one complete evaluation stage.

The core evaluation modules remain path-neutral.  This adapter receives an
explicit ``formal_artifact_paths`` mapping, fingerprints the caller-declared
inputs, and atomically publishes one result that can be strictly reconstructed
by verified workflow resume.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

from ..artifacts.layout import (
    execution_artifact_paths,
    run_artifact_paths,
)
from .vlm.requests import (
    MAX_VLM_RESULT_MANIFEST_BYTES,
    VLM_RESULT_MANIFEST_SCHEMA,
    load_vlm_request_manifest,
    load_vlm_result_manifest,
    parse_strict_json_object,
    read_bounded_saved_artifact,
    write_json_atomic,
)

EVALUATION_PLAN_SCHEMA = "dream_exe.evaluation-plan"
EVALUATION_RESULT_SCHEMA = "dream_exe.evaluation-result"

# The bundle and every independently supplied domain are bounded separately.
MAX_EVALUATION_RESULT_BYTES = 32 * 1024 * 1024
MAX_DETERMINISTIC_DOMAIN_BYTES = 16 * 1024 * 1024
MAX_TRAJECTORY_DOMAIN_BYTES = 16 * 1024 * 1024
MAX_TRAJECTORY_ENTRY_BYTES = 2 * 1024 * 1024
MAX_TASK_SUCCESS_RATE_DOMAIN_BYTES = 8 * 1024 * 1024
MAX_VLM_DOMAIN_BYTES = 16 * 1024 * 1024
MAX_VLM_ENTRY_BYTES = 4 * 1024 * 1024
MAX_INPUT_ARTIFACT_BYTES = 64 * 1024 * 1024
MAX_TRAJECTORY_ENTRIES = 256
MAX_TASK_SUCCESS_RATE_SPECS = 100_000
MAX_VLM_ENTRIES = 256

_HEX64 = re.compile(r"[0-9a-f]{64}\Z")
_RUN_IDENTITY_FIELDS = frozenset(
    {"uid", "run_id", "run_key", "video_kind", "gen_model"}
)
_PLAN_FIELDS = frozenset(
    {
        "format",
        "run_identity",
        "trajectory_similarity",
        "task_success_rate",
        "vlm_request_manifest",
        "vlm",
    }
)
_OPTIONAL_PLAN_FIELDS = frozenset(
    {
        "trajectory_path_comparison",
    }
)
_BUNDLE_FIELDS = frozenset(
    {
        "format",
        "run_identity",
        "plan",
        "plan_sha256",
        "domains",
    }
)
_DOMAIN_FIELDS = frozenset(
    {
        "deterministic",
        "trajectory_similarity",
        "task_success_rate",
        "vlm",
    }
)
_OPTIONAL_DOMAIN_FIELDS = frozenset(
    {
        "trajectory_path_comparison",
    }
)
_ARTIFACT_FIELDS = frozenset({"role", "kind", "location", "size", "sha256"})
_LOCATION_ROOTS = frozenset({"run", "sample", "external"})
_FORMAL_DIRECTORY_FIELDS = frozenset(
    {"sample_root", "run_root", "traj_dir", "exec_dir"}
)


class EvaluationResultValidationError(ValueError):
    """A stable validation failure for one complete evaluation result."""

    def __init__(self, reason: str) -> None:
        self.reason = str(reason)
        super().__init__(f"evaluation result validation failed: {self.reason}")


def _fail(reason: str) -> None:
    raise EvaluationResultValidationError(reason)


def _strict_fields(
    value: Any,
    *,
    expected: frozenset[str],
    label: str,
) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        _fail(f"{label}_must_be_mapping")
    payload = dict(value)
    if set(payload) != set(expected):
        _fail(f"{label}_fields")
    return payload


def _strict_fields_with_optional(
    value: Any,
    *,
    required: frozenset[str],
    optional: frozenset[str],
    label: str,
) -> dict[str, Any]:
    """Validate an additive schema without changing legacy serialized shapes."""

    if not isinstance(value, Mapping):
        _fail(f"{label}_must_be_mapping")
    payload = dict(value)
    fields = set(payload)
    if not set(required).issubset(fields) or fields.difference(
        set(required).union(optional)
    ):
        _fail(f"{label}_fields")
    return payload


def _clean_text(
    value: Any,
    *,
    label: str,
    allow_empty: bool = False,
) -> str:
    if not isinstance(value, str):
        _fail(f"{label}_must_be_string")
    text = value.strip()
    if text != value or (not text and not allow_empty):
        _fail(f"{label}_invalid")
    if any(character in text for character in ("\x00", "\r", "\n", "\t")):
        _fail(f"{label}_invalid")
    return text


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        encoded = json.dumps(
            value,
            ensure_ascii=False,
            allow_nan=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return json.loads(encoded.decode("utf-8"))
    except (
        RecursionError,
        TypeError,
        UnicodeError,
        ValueError,
    ) as error:
        raise EvaluationResultValidationError(f"{label}_must_be_finite_json") from error


def _canonical_bytes(value: Any, *, label: str) -> bytes:
    normalized = _json_copy(value, label=label)
    return json.dumps(
        normalized,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _value_sha256(value: Any, *, label: str) -> str:
    return hashlib.sha256(_canonical_bytes(value, label=label)).hexdigest()


def _enforce_value_size(
    value: Any,
    *,
    limit: int,
    label: str,
) -> None:
    if len(_canonical_bytes(value, label=label)) > int(limit):
        _fail(f"{label}_exceeds_size_limit")


def _normalized_run_identity(value: Any) -> dict[str, str]:
    raw = _strict_fields(
        value,
        expected=_RUN_IDENTITY_FIELDS,
        label="run_identity",
    )
    return {
        field: _clean_text(
            raw[field],
            label=f"run_identity_{field}",
            allow_empty=(field == "gen_model"),
        )
        for field in sorted(_RUN_IDENTITY_FIELDS)
    }


def _formal_roots(
    formal_artifacts: Mapping[str, Any],
) -> tuple[Path, Path, Path]:
    if not isinstance(formal_artifacts, Mapping):
        _fail("formal_artifacts_must_be_mapping")
    try:
        sample_root = Path(formal_artifacts["sample_root"]).absolute()
        run_root = Path(formal_artifacts["run_root"]).absolute()
        result_path = Path(formal_artifacts["evaluation_result"]).absolute()
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationResultValidationError(
            "formal_artifacts_are_incomplete"
        ) from error
    if (
        not sample_root.is_absolute()
        or not run_root.is_absolute()
        or not result_path.is_absolute()
        or not run_root.is_relative_to(sample_root)
        or result_path
        != execution_artifact_paths(run_artifact_paths(run_root)["exec_dir"])[
            "evaluation_result"
        ]
    ):
        _fail("formal_artifacts_are_not_the_explicit_run")
    return sample_root, run_root, result_path


def formal_protected_output_paths(
    formal_artifacts: Mapping[str, Any],
) -> dict[str, Path]:
    """Return every formal file destination, excluding only directory roots."""

    protected: dict[str, Path] = {}
    for raw_role, raw_path in formal_artifacts.items():
        role = str(raw_role)
        if role in _FORMAL_DIRECTORY_FIELDS:
            continue
        try:
            path = Path(raw_path).expanduser().absolute()
        except (TypeError, ValueError) as error:
            raise EvaluationResultValidationError(
                "formal_artifacts_are_incomplete"
            ) from error
        protected[f"formal.{role}"] = path
    try:
        exec_dir = Path(formal_artifacts["exec_dir"]).expanduser().absolute()
        execution_layout = execution_artifact_paths(exec_dir)
        trajectory_manifest = (
            Path(formal_artifacts["trajectory_manifest"]).expanduser().absolute()
        )
    except (KeyError, TypeError, ValueError) as error:
        raise EvaluationResultValidationError(
            "formal_artifacts_are_incomplete"
        ) from error
    protected.update(
        {
            "formal.exec_metrics_per_frame": execution_layout["exec_metrics_per_frame"],
            "formal.exec_assets": execution_layout["execution_manifest"],
            # Keep the historical collision-protection role, but derive it
            # from the same canonical declaration returned to workflow
            # callers so the two names cannot drift.
            "formal.traj_assets": trajectory_manifest,
        }
    )
    return protected


def _read_regular(
    path: Path,
    *,
    limit: int,
    label: str,
) -> bytes:
    try:
        return read_bounded_saved_artifact(
            path,
            max_bytes=limit,
            label=label,
        )
    except FileNotFoundError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise EvaluationResultValidationError(
            f"{label}_path_is_not_bounded_regular_file"
        ) from error


def _location_identity(
    path: str | Path,
    *,
    sample_root: Path,
    run_root: Path,
) -> dict[str, str]:
    lexical = Path(path).expanduser().absolute()
    if lexical.is_relative_to(run_root):
        relative = lexical.relative_to(run_root)
        root = "run"
    elif lexical.is_relative_to(sample_root):
        relative = lexical.relative_to(sample_root)
        root = "sample"
    else:
        return {
            "root": "external",
            "leaf": lexical.name,
            "location_sha256": hashlib.sha256(
                lexical.as_posix().encode("utf-8")
            ).hexdigest(),
        }
    path_text = relative.as_posix()
    pure = PurePosixPath(path_text)
    if (
        not path_text
        or path_text == "."
        or pure.is_absolute()
        or ".." in pure.parts
        or "\\" in path_text
    ):
        _fail("artifact_relative_path")
    return {"root": root, "path": path_text}


def _validate_location(value: Any, *, label: str) -> dict[str, str]:
    if not isinstance(value, Mapping):
        _fail(f"{label}_location")
    raw = dict(value)
    root = _clean_text(raw.get("root"), label=f"{label}_root")
    if root not in _LOCATION_ROOTS:
        _fail(f"{label}_root")
    if root == "external":
        if set(raw) != {"root", "leaf", "location_sha256"}:
            _fail(f"{label}_location_fields")
        leaf = _clean_text(raw["leaf"], label=f"{label}_leaf")
        if Path(leaf).name != leaf or leaf in {".", ".."}:
            _fail(f"{label}_leaf")
        location_sha256 = _clean_text(
            raw["location_sha256"],
            label=f"{label}_location_sha256",
        )
        if _HEX64.fullmatch(location_sha256) is None:
            _fail(f"{label}_location_sha256")
        return {
            "root": root,
            "leaf": leaf,
            "location_sha256": location_sha256,
        }
    if set(raw) != {"root", "path"}:
        _fail(f"{label}_location_fields")
    path_text = _clean_text(raw["path"], label=f"{label}_path")
    pure = PurePosixPath(path_text)
    if (
        pure.is_absolute()
        or ".." in pure.parts
        or "." in pure.parts
        or "\\" in path_text
    ):
        _fail(f"{label}_path")
    return {"root": root, "path": path_text}


def _path_from_location(
    location: Mapping[str, Any],
    *,
    sample_root: Path,
    run_root: Path,
    label: str,
) -> Path:
    normalized = _validate_location(location, label=label)
    root = normalized["root"]
    if root == "external":
        _fail(f"{label}_external_path_is_not_reusable")
    base = run_root if root == "run" else sample_root
    candidate = base / normalized["path"]
    if not candidate.absolute().is_relative_to(base):
        _fail(f"{label}_path_escapes_root")
    return candidate


def _artifact_identity(
    path: str | Path,
    *,
    role: str,
    sample_root: Path,
    run_root: Path,
    limit: int,
    allow_missing: bool,
) -> dict[str, Any]:
    lexical = Path(path).expanduser().absolute()
    location = _location_identity(
        lexical,
        sample_root=sample_root,
        run_root=run_root,
    )
    try:
        payload = _read_regular(
            lexical,
            limit=limit,
            label=role,
        )
    except FileNotFoundError:
        if not allow_missing:
            _fail(f"{role}_is_missing")
        return {
            "role": role,
            "kind": "missing",
            "location": location,
            "size": None,
            "sha256": None,
        }
    return {
        "role": role,
        "kind": "file",
        "location": location,
        "size": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def _validate_artifact_identity(
    value: Any,
    *,
    label: str,
    sample_root: Path,
    run_root: Path,
    limit: int,
    verify_file: bool,
    require_run_root: bool = False,
) -> dict[str, Any]:
    artifact = _strict_fields(
        value,
        expected=_ARTIFACT_FIELDS,
        label=label,
    )
    role = _clean_text(artifact["role"], label=f"{label}_role")
    kind = _clean_text(artifact["kind"], label=f"{label}_kind")
    if kind not in {"file", "missing"}:
        _fail(f"{label}_kind")
    location = _validate_location(
        artifact["location"],
        label=f"{label}_location",
    )
    if require_run_root and location["root"] != "run":
        _fail(f"{label}_must_be_inside_run_root")
    if kind == "missing":
        if artifact["size"] is not None or artifact["sha256"] is not None:
            _fail(f"{label}_missing_identity")
        if verify_file:
            _fail(f"{label}_required_file_is_missing")
        return {
            "role": role,
            "kind": kind,
            "location": location,
            "size": None,
            "sha256": None,
        }
    size = artifact["size"]
    if (
        isinstance(size, bool)
        or not isinstance(size, int)
        or size < 0
        or size > int(limit)
    ):
        _fail(f"{label}_size")
    sha256 = _clean_text(
        artifact["sha256"],
        label=f"{label}_sha256",
    )
    if _HEX64.fullmatch(sha256) is None:
        _fail(f"{label}_sha256")
    normalized = {
        "role": role,
        "kind": kind,
        "location": location,
        "size": size,
        "sha256": sha256,
    }
    if verify_file:
        path = _path_from_location(
            location,
            sample_root=sample_root,
            run_root=run_root,
            label=label,
        )
        payload = _read_regular(
            path,
            limit=limit,
            label=label,
        )
        if len(payload) != size or hashlib.sha256(payload).hexdigest() != sha256:
            _fail(f"{label}_digest_mismatch")
    return normalized


def build_evaluation_plan(
    *,
    formal_artifacts: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    trajectory_path_comparison_reference_path: str | Path | None = None,
    trajectory_similarity_specs: (Sequence[Mapping[str, Any]] | None) = None,
    task_success_rate_specs: (Sequence[Mapping[str, Any]] | None) = None,
    task_success_rate_options: Mapping[str, Any] | None = None,
    vlm_request_manifest_path: str | Path | None = None,
) -> dict[str, Any]:
    """Build the portable, ordered identity of every requested eval domain."""

    sample_root, run_root, _result_path = _formal_roots(formal_artifacts)
    identity = _normalized_run_identity(run_identity)

    def comparison_inputs(
        reference_path: str | Path,
        *,
        role_prefix: str,
    ) -> dict[str, Any]:
        return {
            "trajectory": _artifact_identity(
                formal_artifacts["ee_traj"],
                role=f"{role_prefix}.trajectory",
                sample_root=sample_root,
                run_root=run_root,
                limit=MAX_INPUT_ARTIFACT_BYTES,
                allow_missing=True,
            ),
            "reference": _artifact_identity(
                reference_path,
                role=f"{role_prefix}.reference",
                sample_root=sample_root,
                run_root=run_root,
                limit=MAX_INPUT_ARTIFACT_BYTES,
                allow_missing=True,
            ),
        }

    path_comparison_plan: dict[str, Any] | None = None
    if trajectory_path_comparison_reference_path is not None:
        path_comparison_plan = {
            "inputs": comparison_inputs(
                trajectory_path_comparison_reference_path,
                role_prefix="eval.trajectory_path_comparison",
            )
        }
        _enforce_value_size(
            path_comparison_plan,
            limit=MAX_TRAJECTORY_ENTRY_BYTES,
            label="trajectory_path_comparison_plan",
        )

    trajectory_plan: list[dict[str, Any]] | None = None
    if trajectory_similarity_specs is not None:
        if isinstance(
            trajectory_similarity_specs,
            (str, bytes, bytearray),
        ):
            _fail("trajectory_similarity_specs_type")
        specs = list(trajectory_similarity_specs)
        if len(specs) > MAX_TRAJECTORY_ENTRIES:
            _fail("trajectory_similarity_specs_count")
        trajectory_plan = []
        for index, raw_spec in enumerate(specs):
            if not isinstance(raw_spec, Mapping):
                _fail("trajectory_similarity_spec_type")
            spec = dict(raw_spec)
            try:
                predicted = spec.pop("predicted_path")
                reference = spec.pop("reference_path")
            except KeyError as error:
                raise EvaluationResultValidationError(
                    "trajectory_similarity_spec_path"
                ) from error
            options = _json_copy(
                spec,
                label=f"trajectory_similarity_specs[{index}].options",
            )
            entry = {
                "index": index,
                "inputs": {
                    "predicted": _artifact_identity(
                        predicted,
                        role=(f"eval.trajectory_similarity[{index}].predicted"),
                        sample_root=sample_root,
                        run_root=run_root,
                        limit=MAX_INPUT_ARTIFACT_BYTES,
                        allow_missing=True,
                    ),
                    "reference": _artifact_identity(
                        reference,
                        role=(f"eval.trajectory_similarity[{index}].reference"),
                        sample_root=sample_root,
                        run_root=run_root,
                        limit=MAX_INPUT_ARTIFACT_BYTES,
                        allow_missing=True,
                    ),
                },
                "options": options,
            }
            _enforce_value_size(
                entry,
                limit=MAX_TRAJECTORY_ENTRY_BYTES,
                label="trajectory_plan_entry",
            )
            trajectory_plan.append(entry)

    task_plan: dict[str, Any] | None = None
    if task_success_rate_specs is not None:
        if isinstance(
            task_success_rate_specs,
            (str, bytes, bytearray),
        ):
            _fail("task_success_rate_specs_type")
        raw_specs = list(task_success_rate_specs)
        if len(raw_specs) > MAX_TASK_SUCCESS_RATE_SPECS:
            _fail("task_success_rate_specs_count")
        specs: list[dict[str, Any]] = []
        for index, spec in enumerate(raw_specs):
            if not isinstance(spec, Mapping):
                _fail("task_success_rate_spec_type")
            detached = dict(spec)
            path_value = detached.pop("path", None)
            planned = _json_copy(
                detached,
                label=f"task_success_rate_specs[{index}]",
            )
            assert isinstance(planned, dict)
            if "path" in spec:
                if isinstance(path_value, Path):
                    path_value = path_value.as_posix()
                if isinstance(path_value, str) and path_value.strip():
                    planned["path"] = _artifact_identity(
                        path_value,
                        role=f"eval.task_success_rate[{index}].path",
                        sample_root=sample_root,
                        run_root=run_root,
                        limit=MAX_INPUT_ARTIFACT_BYTES,
                        allow_missing=True,
                    )
                else:
                    planned["path"] = _json_copy(
                        path_value,
                        label=(f"task_success_rate_specs[{index}].path"),
                    )
            specs.append(planned)
        options = _json_copy(
            dict(task_success_rate_options or {}),
            label="task_success_rate_options",
        )
        task_plan = {"specs": specs, "options": options}
        _enforce_value_size(
            task_plan,
            limit=MAX_TASK_SUCCESS_RATE_DOMAIN_BYTES,
            label="task_success_rate_plan",
        )

    vlm_manifest_identity: dict[str, Any] | None = None
    vlm_plan: list[dict[str, Any]] = []
    if vlm_request_manifest_path is not None:
        manifest = load_vlm_request_manifest(
            vlm_request_manifest_path,
            expected_run_identity=identity,
            protected_output_paths=formal_protected_output_paths(formal_artifacts),
            durable_run_root=run_root,
            expected_formal_run_root=run_root,
        )
        manifest_path = Path(manifest["path"])
        vlm_manifest_identity = _artifact_identity(
            manifest_path,
            role="eval.vlm_request_manifest",
            sample_root=sample_root,
            run_root=run_root,
            limit=MAX_INPUT_ARTIFACT_BYTES,
            allow_missing=False,
        )
        if len(manifest["requests"]) > MAX_VLM_ENTRIES:
            _fail("vlm_request_count")
        for index, request in enumerate(manifest["requests"]):
            output_path = Path(request["result_manifest_path"])
            entry = {
                "index": index,
                "request_id": str(request["request_id"]),
                "mode": str(request["mode"]),
                "evidence_fingerprints": copy.deepcopy(
                    request["evidence_fingerprints"]
                ),
                "behavior_sha256": _value_sha256(
                    request["behavior"],
                    label=f"vlm[{index}].behavior",
                ),
                "request_sha256": str(request["request_sha256"]),
                "output": _location_identity(
                    output_path,
                    sample_root=sample_root,
                    run_root=run_root,
                ),
                "outputs": [
                    {
                        "field": str(destination["field"]),
                        "owner": str(destination["owner"]),
                        "required": True,
                        "location": _location_identity(
                            destination["path"],
                            sample_root=sample_root,
                            run_root=run_root,
                        ),
                    }
                    for destination in request["output_destinations"]
                ],
            }
            if manifest["preparation_sha256"] is not None:
                entry["preparation_sha256"] = str(manifest["preparation_sha256"])
            _enforce_value_size(
                entry,
                limit=MAX_VLM_ENTRY_BYTES,
                label="vlm_plan_entry",
            )
            vlm_plan.append(entry)

    plan = {
        "format": EVALUATION_PLAN_SCHEMA,
        "run_identity": identity,
        "trajectory_similarity": trajectory_plan,
        "task_success_rate": task_plan,
        "vlm_request_manifest": vlm_manifest_identity,
        "vlm": vlm_plan,
    }
    if path_comparison_plan is not None:
        plan["trajectory_path_comparison"] = path_comparison_plan
    _validate_plan(
        plan,
        expected_run_identity=identity,
        sample_root=sample_root,
        run_root=run_root,
    )
    return plan


def _validate_plan(
    value: Any,
    *,
    expected_run_identity: Mapping[str, Any],
    sample_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    plan = _strict_fields_with_optional(
        value,
        required=_PLAN_FIELDS,
        optional=_OPTIONAL_PLAN_FIELDS,
        label="plan",
    )
    if plan["format"] != EVALUATION_PLAN_SCHEMA:
        _fail("plan_format")
    identity = _normalized_run_identity(plan["run_identity"])
    if identity != _normalized_run_identity(expected_run_identity):
        _fail("plan_run_identity")

    for plan_field, label in (
        ("trajectory_path_comparison", "trajectory_path_comparison"),
    ):
        comparison = plan.get(plan_field)
        if comparison is None:
            continue
        comparison = _strict_fields(
            comparison,
            expected=frozenset({"inputs"}),
            label=f"plan_{label}",
        )
        inputs = _strict_fields(
            comparison["inputs"],
            expected=frozenset({"trajectory", "reference"}),
            label=f"plan_{label}_inputs",
        )
        for role in ("trajectory", "reference"):
            _validate_artifact_identity(
                inputs[role],
                label=f"plan_{label}_{role}",
                sample_root=sample_root,
                run_root=run_root,
                limit=MAX_INPUT_ARTIFACT_BYTES,
                verify_file=False,
            )
        _enforce_value_size(
            comparison,
            limit=MAX_TRAJECTORY_ENTRY_BYTES,
            label=f"plan_{label}",
        )

    trajectory = plan["trajectory_similarity"]
    if trajectory is not None:
        if not isinstance(trajectory, list) or len(trajectory) > MAX_TRAJECTORY_ENTRIES:
            _fail("plan_trajectory_entries")
        for index, raw_entry in enumerate(trajectory):
            entry = _strict_fields(
                raw_entry,
                expected=frozenset({"index", "inputs", "options"}),
                label="plan_trajectory_entry",
            )
            if entry["index"] != index:
                _fail("plan_trajectory_order")
            inputs = _strict_fields(
                entry["inputs"],
                expected=frozenset({"predicted", "reference"}),
                label="plan_trajectory_inputs",
            )
            for role in ("predicted", "reference"):
                _validate_artifact_identity(
                    inputs[role],
                    label=f"plan_trajectory_{role}",
                    sample_root=sample_root,
                    run_root=run_root,
                    limit=MAX_INPUT_ARTIFACT_BYTES,
                    verify_file=False,
                )
            if not isinstance(entry["options"], Mapping):
                _fail("plan_trajectory_options")
            _enforce_value_size(
                entry,
                limit=MAX_TRAJECTORY_ENTRY_BYTES,
                label="plan_trajectory_entry",
            )

    task = plan["task_success_rate"]
    if task is not None:
        task = _strict_fields(
            task,
            expected=frozenset({"specs", "options"}),
            label="plan_task_success_rate",
        )
        if (
            not isinstance(task["specs"], list)
            or len(task["specs"]) > MAX_TASK_SUCCESS_RATE_SPECS
            or not isinstance(task["options"], Mapping)
        ):
            _fail("plan_task_success_rate")
        _enforce_value_size(
            task,
            limit=MAX_TASK_SUCCESS_RATE_DOMAIN_BYTES,
            label="plan_task_success_rate",
        )

    vlm_manifest = plan["vlm_request_manifest"]
    vlm = plan["vlm"]
    if not isinstance(vlm, list) or len(vlm) > MAX_VLM_ENTRIES:
        _fail("plan_vlm_entries")
    if bool(vlm) != (vlm_manifest is not None):
        _fail("plan_vlm_manifest_alignment")
    if vlm_manifest is not None:
        _validate_artifact_identity(
            vlm_manifest,
            label="plan_vlm_request_manifest",
            sample_root=sample_root,
            run_root=run_root,
            limit=MAX_INPUT_ARTIFACT_BYTES,
            verify_file=False,
        )
    previous_request_id: set[str] = set()
    preparation_binding: str | None = None
    preparation_binding_present: bool | None = None
    for index, raw_entry in enumerate(vlm):
        entry = _strict_fields_with_optional(
            raw_entry,
            required=frozenset(
                {
                    "index",
                    "request_id",
                    "mode",
                    "evidence_fingerprints",
                    "behavior_sha256",
                    "request_sha256",
                    "output",
                    "outputs",
                }
            ),
            optional=frozenset({"preparation_sha256"}),
            label="plan_vlm_entry",
        )
        if entry["index"] != index:
            _fail("plan_vlm_order")
        request_id = _clean_text(
            entry["request_id"],
            label="plan_vlm_request_id",
        )
        if request_id in previous_request_id:
            _fail("plan_vlm_duplicate_request_id")
        previous_request_id.add(request_id)
        if entry["mode"] not in {"video_only", "video_trajectory"}:
            _fail("plan_vlm_mode")
        digest = _clean_text(
            entry["behavior_sha256"],
            label="plan_vlm_behavior_sha256",
        )
        if _HEX64.fullmatch(digest) is None:
            _fail("plan_vlm_behavior_sha256")
        request_digest = _clean_text(
            entry["request_sha256"],
            label="plan_vlm_request_sha256",
        )
        if _HEX64.fullmatch(request_digest) is None:
            _fail("plan_vlm_request_sha256")
        has_preparation_binding = "preparation_sha256" in entry
        if preparation_binding_present is None:
            preparation_binding_present = has_preparation_binding
        elif preparation_binding_present != has_preparation_binding:
            _fail("plan_vlm_preparation_alignment")
        if has_preparation_binding:
            declared_preparation = _clean_text(
                entry["preparation_sha256"],
                label="plan_vlm_preparation_sha256",
            )
            if _HEX64.fullmatch(declared_preparation) is None:
                _fail("plan_vlm_preparation_sha256")
            if preparation_binding is None:
                preparation_binding = declared_preparation
            elif preparation_binding != declared_preparation:
                _fail("plan_vlm_preparation_alignment")
        fingerprints = entry["evidence_fingerprints"]
        if not isinstance(fingerprints, list) or not fingerprints:
            _fail("plan_vlm_evidence_fingerprints")
        previous_field = ""
        for fingerprint in fingerprints:
            raw = _strict_fields(
                fingerprint,
                expected=frozenset({"field", "size", "sha256"}),
                label="plan_vlm_evidence_fingerprint",
            )
            field = _clean_text(
                raw["field"],
                label="plan_vlm_evidence_field",
            )
            if field <= previous_field:
                _fail("plan_vlm_evidence_order")
            previous_field = field
            size = raw["size"]
            if isinstance(size, bool) or not isinstance(size, int) or size < 1:
                _fail("plan_vlm_evidence_size")
            if (
                not isinstance(raw["sha256"], str)
                or _HEX64.fullmatch(raw["sha256"]) is None
            ):
                _fail("plan_vlm_evidence_sha256")
        result_location = _validate_location(
            entry["output"],
            label="plan_vlm_output",
        )
        if result_location["root"] != "run":
            _fail("plan_vlm_result_must_be_inside_run_root")
        outputs = entry["outputs"]
        if not isinstance(outputs, list) or not outputs:
            _fail("plan_vlm_outputs")
        previous_output_field = ""
        framework_results = 0
        for raw_output in outputs:
            output = _strict_fields(
                raw_output,
                expected=frozenset({"field", "owner", "required", "location"}),
                label="plan_vlm_output_entry",
            )
            field = _clean_text(
                output["field"],
                label="plan_vlm_output_field",
            )
            if field <= previous_output_field:
                _fail("plan_vlm_output_order")
            previous_output_field = field
            if output["owner"] not in {"evaluator", "framework"}:
                _fail("plan_vlm_output_owner")
            if output["required"] is not True:
                _fail("plan_vlm_output_must_be_required")
            location = _validate_location(
                output["location"],
                label="plan_vlm_output_location",
            )
            if location["root"] == "external":
                _fail("plan_vlm_output_is_not_reusable")
            if output["owner"] == "framework":
                framework_results += 1
                if field != "result_manifest_path" or location != result_location:
                    _fail("plan_vlm_framework_result_output")
        if framework_results != 1:
            _fail("plan_vlm_framework_result_output")
        _enforce_value_size(
            entry,
            limit=MAX_VLM_ENTRY_BYTES,
            label="plan_vlm_entry",
        )
    return _json_copy(plan, label="plan")


def build_evaluation_result_bundle(
    *,
    formal_artifacts: Mapping[str, Any],
    run_identity: Mapping[str, Any],
    result: Mapping[str, Any],
    plan: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one complete bundle after every requested evaluator succeeded."""

    sample_root, run_root, _result_path = _formal_roots(formal_artifacts)
    identity = _normalized_run_identity(run_identity)
    normalized_plan = _validate_plan(
        plan,
        expected_run_identity=identity,
        sample_root=sample_root,
        run_root=run_root,
    )
    if not isinstance(result, Mapping):
        _fail("evaluation_result_must_be_mapping")
    deterministic = _json_copy(
        result.get("deterministic"),
        label="deterministic_result",
    )
    if not isinstance(deterministic, dict):
        _fail("deterministic_result_must_be_mapping")
    deterministic_artifacts = [
        _artifact_identity(
            formal_artifacts["exec_metrics"],
            role="eval.exec_metrics",
            sample_root=sample_root,
            run_root=run_root,
            limit=MAX_DETERMINISTIC_DOMAIN_BYTES,
            allow_missing=True,
        ),
        _artifact_identity(
            execution_artifact_paths(formal_artifacts["exec_dir"])[
                "exec_metrics_per_frame"
            ],
            role="eval.exec_metrics_per_frame",
            sample_root=sample_root,
            run_root=run_root,
            limit=MAX_DETERMINISTIC_DOMAIN_BYTES,
            allow_missing=True,
        ),
    ]
    metrics_identity = deterministic_artifacts[0]
    if metrics_identity["kind"] == "file":
        metrics_path = Path(formal_artifacts["exec_metrics"])
        try:
            saved_metrics = parse_strict_json_object(
                _read_regular(
                    metrics_path,
                    limit=MAX_DETERMINISTIC_DOMAIN_BYTES,
                    label="deterministic_metrics",
                ),
                label="deterministic_metrics",
            )
        except ValueError as error:
            raise EvaluationResultValidationError(
                "deterministic_metrics_must_be_strict_json"
            ) from error
        if saved_metrics != deterministic:
            _fail("deterministic_result_artifact_mismatch")
    deterministic_domain = {
        "result": deterministic,
        "result_sha256": _value_sha256(
            deterministic,
            label="deterministic_result",
        ),
        "artifacts": deterministic_artifacts,
    }
    _enforce_value_size(
        deterministic_domain,
        limit=MAX_DETERMINISTIC_DOMAIN_BYTES,
        label="deterministic_domain",
    )

    raw_trajectory = result.get("trajectory")
    path_comparison_plan = normalized_plan.get("trajectory_path_comparison")
    path_comparison_result = (
        raw_trajectory.get("path_comparison")
        if isinstance(raw_trajectory, Mapping)
        else None
    )
    path_comparison_domain: dict[str, Any] | None = None
    if path_comparison_plan is None:
        if path_comparison_result is not None:
            _fail("unrequested_trajectory_path_comparison_result")
    else:
        normalized_path_comparison = _json_copy(
            path_comparison_result,
            label="trajectory_path_comparison_result",
        )
        if not isinstance(normalized_path_comparison, dict):
            _fail("trajectory_path_comparison_result_must_be_mapping")
        path_comparison_domain = {
            "plan_sha256": _value_sha256(
                path_comparison_plan,
                label="trajectory_path_comparison_plan",
            ),
            "result": normalized_path_comparison,
            "result_sha256": _value_sha256(
                normalized_path_comparison,
                label="trajectory_path_comparison_result",
            ),
        }
        _enforce_value_size(
            path_comparison_domain,
            limit=MAX_TRAJECTORY_ENTRY_BYTES,
            label="trajectory_path_comparison_domain",
        )

    trajectory_plan = normalized_plan["trajectory_similarity"]
    similarity_result = (
        raw_trajectory.get("similarity")
        if isinstance(raw_trajectory, Mapping)
        else None
    )
    trajectory_domain: list[dict[str, Any]] | None
    if trajectory_plan is None:
        if similarity_result is not None:
            _fail("unrequested_trajectory_result")
        trajectory_domain = None
    else:
        if not isinstance(similarity_result, list) or len(similarity_result) != len(
            trajectory_plan
        ):
            _fail("trajectory_result_alignment")
        trajectory_domain = []
        for index, (plan_entry, raw_result) in enumerate(
            zip(trajectory_plan, similarity_result)
        ):
            normalized_result = _json_copy(
                raw_result,
                label=f"trajectory_result[{index}]",
            )
            if not isinstance(normalized_result, dict):
                _fail("trajectory_result_must_be_mapping")
            entry = {
                "index": index,
                "plan_entry_sha256": _value_sha256(
                    plan_entry,
                    label=f"trajectory_plan[{index}]",
                ),
                "result": normalized_result,
                "result_sha256": _value_sha256(
                    normalized_result,
                    label=f"trajectory_result[{index}]",
                ),
            }
            _enforce_value_size(
                entry,
                limit=MAX_TRAJECTORY_ENTRY_BYTES,
                label="trajectory_result_entry",
            )
            trajectory_domain.append(entry)
        _enforce_value_size(
            trajectory_domain,
            limit=MAX_TRAJECTORY_DOMAIN_BYTES,
            label="trajectory_domain",
        )

    task_plan = normalized_plan["task_success_rate"]
    raw_task_result = result.get("task_success_rate")
    task_domain: dict[str, Any] | None
    if task_plan is None:
        if raw_task_result is not None:
            _fail("unrequested_task_success_rate_result")
        task_domain = None
    else:
        normalized_task = _json_copy(
            raw_task_result,
            label="task_success_rate_result",
        )
        if not isinstance(normalized_task, dict):
            _fail("task_success_rate_result_must_be_mapping")
        task_domain = {
            "plan_sha256": _value_sha256(
                task_plan,
                label="task_success_rate_plan",
            ),
            "result": normalized_task,
            "result_sha256": _value_sha256(
                normalized_task,
                label="task_success_rate_result",
            ),
        }
        _enforce_value_size(
            task_domain,
            limit=MAX_TASK_SUCCESS_RATE_DOMAIN_BYTES,
            label="task_success_rate_domain",
        )

    vlm_plan = normalized_plan["vlm"]
    vlm_request_manifest = normalized_plan["vlm_request_manifest"]
    raw_vlm = result.get("vlm_evaluations", [])
    if not isinstance(raw_vlm, list) or len(raw_vlm) != len(vlm_plan):
        _fail("vlm_result_alignment")
    vlm_domain: list[dict[str, Any]] = []
    for index, (plan_entry, raw_item) in enumerate(zip(vlm_plan, raw_vlm)):
        if not isinstance(raw_item, Mapping):
            _fail("vlm_result_item_must_be_mapping")
        item = dict(raw_item)
        request_id = str(item.get("request_id", ""))
        mode = str(item.get("mode", ""))
        if request_id != plan_entry["request_id"] or mode != plan_entry["mode"]:
            _fail("vlm_result_order")
        result_manifest = item.get("result_manifest")
        if not isinstance(result_manifest, Mapping):
            _fail("vlm_result_manifest_identity")
        result_manifest_path = (
            Path(str(result_manifest.get("path", "") or "")).expanduser().absolute()
        )
        if (
            _location_identity(
                result_manifest_path,
                sample_root=sample_root,
                run_root=run_root,
            )
            != plan_entry["output"]
        ):
            _fail("vlm_result_manifest_identity")
        preparation_sha256 = plan_entry["preparation_sha256"]
        if not isinstance(vlm_request_manifest, Mapping):
            _fail("vlm_result_manifest_identity")
        expected_request_manifest_fingerprint = {
            "size": vlm_request_manifest["size"],
            "sha256": vlm_request_manifest["sha256"],
        }
        try:
            durable = load_vlm_result_manifest(
                result_manifest_path,
                expected_run_identity=identity,
                expected_request_id=request_id,
                expected_mode=mode,
                expected_evidence_fingerprints=(plan_entry["evidence_fingerprints"]),
                expected_request_sha256=plan_entry["request_sha256"],
                expected_request_manifest_fingerprint=(
                    expected_request_manifest_fingerprint
                ),
                expected_preparation_sha256=preparation_sha256,
            )
        except (OSError, RuntimeError, TypeError, ValueError) as error:
            raise EvaluationResultValidationError(
                "vlm_result_manifest_does_not_match_plan"
            ) from error
        if result_manifest.get("format") != durable["format"]:
            _fail("vlm_result_manifest_identity")
        normalized_result = _json_copy(
            item.get("result"),
            label=f"vlm_result[{index}]",
        )
        if (
            not isinstance(normalized_result, dict)
            or durable["result"] != normalized_result
        ):
            _fail("vlm_result_manifest_result")
        manifest_identity = _artifact_identity(
            result_manifest_path,
            role=f"eval.vlm_result[{index}]",
            sample_root=sample_root,
            run_root=run_root,
            limit=MAX_VLM_RESULT_MANIFEST_BYTES,
            allow_missing=False,
        )
        output_artifacts: list[dict[str, Any]] = []
        for output in plan_entry["outputs"]:
            output_path = _path_from_location(
                output["location"],
                sample_root=sample_root,
                run_root=run_root,
                label=f"vlm_output_{index}_{output['field']}",
            )
            output_artifacts.append(
                _artifact_identity(
                    output_path,
                    role=(f"eval.vlm_output[{index}].{output['field']}"),
                    sample_root=sample_root,
                    run_root=run_root,
                    limit=MAX_VLM_RESULT_MANIFEST_BYTES,
                    allow_missing=False,
                )
            )
        entry = {
            "index": index,
            "request_id": request_id,
            "mode": mode,
            "plan_entry_sha256": _value_sha256(
                plan_entry,
                label=f"vlm_plan[{index}]",
            ),
            "result_manifest": manifest_identity,
            "outputs": output_artifacts,
            "result": normalized_result,
            "result_sha256": _value_sha256(
                normalized_result,
                label=f"vlm_result[{index}]",
            ),
            "paper_metric_compatibility": (
                "not_claimed" if mode == "video_trajectory" else None
            ),
            "metric_claims": [],
        }
        entry["result_manifest_format"] = durable["format"]
        _enforce_value_size(
            entry,
            limit=MAX_VLM_ENTRY_BYTES,
            label="vlm_result_entry",
        )
        vlm_domain.append(entry)
    _enforce_value_size(
        vlm_domain,
        limit=MAX_VLM_DOMAIN_BYTES,
        label="vlm_domain",
    )

    domains = {
        "deterministic": deterministic_domain,
        "trajectory_similarity": trajectory_domain,
        "task_success_rate": task_domain,
        "vlm": vlm_domain,
    }
    if path_comparison_plan is not None:
        domains["trajectory_path_comparison"] = path_comparison_domain

    bundle = {
        "format": EVALUATION_RESULT_SCHEMA,
        "run_identity": identity,
        "plan": normalized_plan,
        "plan_sha256": _value_sha256(
            normalized_plan,
            label="evaluation_plan",
        ),
        "domains": domains,
    }
    _enforce_value_size(
        bundle,
        limit=MAX_EVALUATION_RESULT_BYTES,
        label="evaluation_result_bundle",
    )
    return bundle


def publish_evaluation_result_bundle(
    bundle: Mapping[str, Any],
    *,
    formal_artifacts: Mapping[str, Any],
) -> Path:
    """Atomically replace the formal bundle without damaging an older result."""

    _sample_root, _run_root, destination = _formal_roots(formal_artifacts)
    encoded = (
        json.dumps(
            _json_copy(bundle, label="evaluation_result_bundle"),
            ensure_ascii=False,
            indent=2,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    if len(encoded) > MAX_EVALUATION_RESULT_BYTES:
        _fail("evaluation_result_bundle_exceeds_size_limit")
    write_json_atomic(destination, bundle)
    return destination


def load_evaluation_result_bundle(
    *,
    formal_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Read one bounded, non-symlink, strict-JSON formal bundle."""

    _sample_root, _run_root, path = _formal_roots(formal_artifacts)
    try:
        encoded = _read_regular(
            path,
            limit=MAX_EVALUATION_RESULT_BYTES,
            label="evaluation_result_bundle",
        )
    except FileNotFoundError as error:
        raise EvaluationResultValidationError(
            "evaluation_result_bundle_is_missing"
        ) from error
    try:
        return parse_strict_json_object(
            encoded,
            label="evaluation_result_bundle",
        )
    except ValueError as error:
        raise EvaluationResultValidationError(
            "evaluation_result_bundle_must_be_strict_json"
        ) from error


def _bundle_result_projection(
    domains: Mapping[str, Any],
    *,
    plan: Mapping[str, Any],
    sample_root: Path,
    run_root: Path,
) -> dict[str, Any]:
    """Recover only public results; the builder re-derives every identity."""

    deterministic = _strict_fields(
        domains["deterministic"],
        expected=frozenset({"result", "result_sha256", "artifacts"}),
        label="deterministic_domain",
    )
    _enforce_value_size(
        deterministic,
        limit=MAX_DETERMINISTIC_DOMAIN_BYTES,
        label="deterministic_domain",
    )
    deterministic_result = _json_copy(
        deterministic["result"],
        label="deterministic_result",
    )
    if not isinstance(deterministic_result, dict):
        _fail("deterministic_result_must_be_mapping")

    path_comparison_plan = plan.get("trajectory_path_comparison")
    raw_path_comparison = domains.get("trajectory_path_comparison")
    path_comparison_result: dict[str, Any] | None = None
    if path_comparison_plan is None:
        if raw_path_comparison is not None:
            _fail("trajectory_path_comparison_domain_alignment")
    else:
        path_comparison = _strict_fields(
            raw_path_comparison,
            expected=frozenset({"plan_sha256", "result", "result_sha256"}),
            label="trajectory_path_comparison_domain",
        )
        _enforce_value_size(
            path_comparison,
            limit=MAX_TRAJECTORY_ENTRY_BYTES,
            label="trajectory_path_comparison_domain",
        )
        path_comparison_result = _json_copy(
            path_comparison["result"],
            label="trajectory_path_comparison_result",
        )
        if not isinstance(path_comparison_result, dict):
            _fail("trajectory_path_comparison_result_must_be_mapping")

    trajectory_plan = plan["trajectory_similarity"]
    raw_trajectory = domains["trajectory_similarity"]
    trajectory_result: list[dict[str, Any]] | None = None
    if trajectory_plan is None:
        if raw_trajectory is not None:
            _fail("trajectory_domain_alignment")
    else:
        if not isinstance(raw_trajectory, list) or len(raw_trajectory) != len(
            trajectory_plan
        ):
            _fail("trajectory_domain_alignment")
        trajectory_result = []
        for raw_entry in raw_trajectory:
            _enforce_value_size(
                raw_entry,
                limit=MAX_TRAJECTORY_ENTRY_BYTES,
                label="trajectory_result_entry",
            )
            entry = _strict_fields(
                raw_entry,
                expected=frozenset(
                    {
                        "index",
                        "plan_entry_sha256",
                        "result",
                        "result_sha256",
                    }
                ),
                label="trajectory_result_entry",
            )
            result = _json_copy(
                entry["result"],
                label="trajectory_result",
            )
            if not isinstance(result, dict):
                _fail("trajectory_result_must_be_mapping")
            trajectory_result.append(result)
        _enforce_value_size(
            raw_trajectory,
            limit=MAX_TRAJECTORY_DOMAIN_BYTES,
            label="trajectory_domain",
        )

    task_plan = plan["task_success_rate"]
    raw_task = domains["task_success_rate"]
    task_result: dict[str, Any] | None = None
    if task_plan is None:
        if raw_task is not None:
            _fail("task_success_rate_domain_alignment")
    else:
        task = _strict_fields(
            raw_task,
            expected=frozenset({"plan_sha256", "result", "result_sha256"}),
            label="task_success_rate_domain",
        )
        _enforce_value_size(
            task,
            limit=MAX_TASK_SUCCESS_RATE_DOMAIN_BYTES,
            label="task_success_rate_domain",
        )
        task_result = _json_copy(
            task["result"],
            label="task_success_rate_result",
        )
        if not isinstance(task_result, dict):
            _fail("task_success_rate_result_must_be_mapping")

    raw_vlm = domains["vlm"]
    vlm_plan = plan["vlm"]
    if not isinstance(raw_vlm, list) or len(raw_vlm) != len(vlm_plan):
        _fail("vlm_domain_alignment")
    vlm_result: list[dict[str, Any]] = []
    for index, raw_entry in enumerate(raw_vlm):
        _enforce_value_size(
            raw_entry,
            limit=MAX_VLM_ENTRY_BYTES,
            label="vlm_result_entry",
        )
        entry = _strict_fields(
            raw_entry,
            expected=frozenset(
                {
                    "index",
                    "request_id",
                    "mode",
                    "plan_entry_sha256",
                    "result_manifest",
                    "outputs",
                    "result",
                    "result_sha256",
                    "paper_metric_compatibility",
                    "metric_claims",
                    "result_manifest_format",
                }
            ),
            label="vlm_result_entry",
        )
        result_manifest_schema = _clean_text(
            entry["result_manifest_format"],
            label="vlm_result_manifest_format",
        )
        if result_manifest_schema != VLM_RESULT_MANIFEST_SCHEMA:
            _fail("vlm_result_manifest_format")
        manifest = _validate_artifact_identity(
            entry["result_manifest"],
            label=f"vlm_result_manifest_{index}",
            sample_root=sample_root,
            run_root=run_root,
            limit=MAX_VLM_RESULT_MANIFEST_BYTES,
            verify_file=False,
            require_run_root=True,
        )
        manifest_path = _path_from_location(
            manifest["location"],
            sample_root=sample_root,
            run_root=run_root,
            label=f"vlm_result_manifest_{index}",
        )
        output_artifacts = entry["outputs"]
        planned_outputs = vlm_plan[index]["outputs"]
        if not isinstance(output_artifacts, list) or len(output_artifacts) != len(
            planned_outputs
        ):
            _fail("vlm_output_artifact_alignment")
        for output_index, (raw_artifact, planned_output) in enumerate(
            zip(output_artifacts, planned_outputs)
        ):
            artifact = _validate_artifact_identity(
                raw_artifact,
                label=(f"vlm_output_artifact_{index}_{output_index}"),
                sample_root=sample_root,
                run_root=run_root,
                limit=MAX_VLM_RESULT_MANIFEST_BYTES,
                verify_file=False,
            )
            if artifact["location"] != planned_output["location"]:
                _fail("vlm_output_artifact_location")
        result = _json_copy(
            entry["result"],
            label=f"vlm_result[{index}]",
        )
        if not isinstance(result, dict):
            _fail("vlm_result_must_be_mapping")
        vlm_result.append(
            {
                "request_id": entry["request_id"],
                "mode": entry["mode"],
                "result_manifest": {
                    "format": result_manifest_schema,
                    "path": manifest_path.as_posix(),
                },
                "result": result,
            }
        )
    _enforce_value_size(
        raw_vlm,
        limit=MAX_VLM_DOMAIN_BYTES,
        label="vlm_domain",
    )

    trajectory_result_payload: dict[str, Any] = {
        "similarity": trajectory_result,
    }
    if path_comparison_plan is not None:
        trajectory_result_payload["path_comparison"] = path_comparison_result

    return {
        "deterministic": deterministic_result,
        "trajectory": trajectory_result_payload,
        "task_success_rate": task_result,
        "vlm_evaluations": vlm_result,
    }


def validate_evaluation_result_bundle(
    bundle: Mapping[str, Any],
    *,
    formal_artifacts: Mapping[str, Any],
    expected_run_identity: Mapping[str, Any],
    expected_plan: Mapping[str, Any],
    verify_artifacts: bool = True,
) -> dict[str, Any]:
    """Re-derive the canonical bundle and compare every stored byte identity."""

    sample_root, run_root, _result_path = _formal_roots(formal_artifacts)
    identity = _normalized_run_identity(expected_run_identity)
    payload = _strict_fields(
        bundle,
        expected=_BUNDLE_FIELDS,
        label="bundle",
    )
    if payload["format"] != EVALUATION_RESULT_SCHEMA:
        _fail("bundle_format")
    if _normalized_run_identity(payload["run_identity"]) != identity:
        _fail("bundle_run_identity")
    plan = _validate_plan(
        payload["plan"],
        expected_run_identity=identity,
        sample_root=sample_root,
        run_root=run_root,
    )
    normalized_expected_plan = _validate_plan(
        expected_plan,
        expected_run_identity=identity,
        sample_root=sample_root,
        run_root=run_root,
    )
    if plan != normalized_expected_plan:
        _fail("bundle_plan_mismatch")
    if payload["plan_sha256"] != _value_sha256(
        plan,
        label="evaluation_plan",
    ):
        _fail("bundle_plan_digest")
    domains = _strict_fields_with_optional(
        payload["domains"],
        required=_DOMAIN_FIELDS,
        optional=_OPTIONAL_DOMAIN_FIELDS,
        label="domains",
    )
    projection = _bundle_result_projection(
        domains,
        plan=plan,
        sample_root=sample_root,
        run_root=run_root,
    )
    try:
        expected = build_evaluation_result_bundle(
            formal_artifacts=formal_artifacts,
            run_identity=identity,
            result=projection,
            plan=plan,
        )
    except EvaluationResultValidationError:
        raise
    except (OSError, RuntimeError, TypeError, ValueError) as error:
        raise EvaluationResultValidationError("bundle_artifact_validation") from error
    normalized = _json_copy(
        payload,
        label="evaluation_result_bundle",
    )
    _enforce_value_size(
        normalized,
        limit=MAX_EVALUATION_RESULT_BYTES,
        label="evaluation_result_bundle",
    )
    if normalized != expected:
        _fail("bundle_content_or_digest_mismatch")
    # Full validation is intentionally strict: result reconstruction is never
    # allowed without checking live artifacts.
    _ = verify_artifacts
    return expected


def reconstruct_evaluation_result(
    bundle: Mapping[str, Any],
    *,
    formal_artifacts: Mapping[str, Any],
) -> dict[str, Any]:
    """Reconstruct the existing public eval return shape from a valid bundle."""

    sample_root, run_root, _result_path = _formal_roots(formal_artifacts)
    identity = dict(bundle["run_identity"])
    domains = dict(bundle["domains"])
    deterministic = copy.deepcopy(dict(domains["deterministic"]["result"]))
    trajectory_domain = domains["trajectory_similarity"]
    trajectory_similarity = (
        None
        if trajectory_domain is None
        else [copy.deepcopy(dict(entry["result"])) for entry in trajectory_domain]
    )
    path_comparison_domain = domains.get("trajectory_path_comparison")
    path_comparison = (
        None
        if path_comparison_domain is None
        else copy.deepcopy(dict(path_comparison_domain["result"]))
    )
    task_domain = domains["task_success_rate"]
    task_success_rate = (
        None if task_domain is None else copy.deepcopy(dict(task_domain["result"]))
    )
    plan_vlm = list(bundle["plan"]["vlm"])
    vlm_evaluations: list[dict[str, Any]] = []
    for plan_entry, entry in zip(plan_vlm, domains["vlm"]):
        manifest_path = _path_from_location(
            entry["result_manifest"]["location"],
            sample_root=sample_root,
            run_root=run_root,
            label="vlm_result_manifest",
        )
        item: dict[str, Any] = {
            "request_id": entry["request_id"],
            "mode": entry["mode"],
            "result_manifest": {
                "format": entry.get(
                    "result_manifest_format",
                    VLM_RESULT_MANIFEST_SCHEMA,
                ),
                "path": manifest_path.as_posix(),
            },
            "evidence_fingerprints": copy.deepcopy(plan_entry["evidence_fingerprints"]),
            "result": copy.deepcopy(dict(entry["result"])),
        }
        if entry["mode"] == "video_trajectory":
            item["paper_metric_compatibility"] = "not_claimed"
            item["metric_claims"] = []
        vlm_evaluations.append(item)
    trajectory_result = {
        "similarity": trajectory_similarity,
        "executability": copy.deepcopy(deterministic),
    }
    if "trajectory_path_comparison" in bundle["plan"]:
        trajectory_result["path_comparison"] = path_comparison

    return {
        "ok": True,
        "returncode": 0,
        "uid": identity["uid"],
        "run_id": identity["run_id"],
        "run_key": identity["run_key"],
        "video_kind": identity["video_kind"],
        "gen_model": identity["gen_model"],
        "exec_dir": Path(formal_artifacts["exec_dir"]).as_posix(),
        "deterministic": deterministic,
        "vlm_evaluations": vlm_evaluations,
        "trajectory": trajectory_result,
        "task_success_rate": task_success_rate,
    }


__all__ = [
    "EVALUATION_PLAN_SCHEMA",
    "EVALUATION_RESULT_SCHEMA",
    "EvaluationResultValidationError",
    "MAX_DETERMINISTIC_DOMAIN_BYTES",
    "MAX_EVALUATION_RESULT_BYTES",
    "MAX_TASK_SUCCESS_RATE_DOMAIN_BYTES",
    "MAX_TRAJECTORY_DOMAIN_BYTES",
    "MAX_TRAJECTORY_ENTRIES",
    "MAX_TRAJECTORY_ENTRY_BYTES",
    "MAX_VLM_DOMAIN_BYTES",
    "MAX_VLM_ENTRIES",
    "MAX_VLM_ENTRY_BYTES",
    "build_evaluation_plan",
    "build_evaluation_result_bundle",
    "formal_protected_output_paths",
    "load_evaluation_result_bundle",
    "publish_evaluation_result_bundle",
    "reconstruct_evaluation_result",
    "validate_evaluation_result_bundle",
]
