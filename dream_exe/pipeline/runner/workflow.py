"""Concrete single-run current-bench workflow over public current implementation callables."""

from __future__ import annotations

import copy
import json
import os
import re
import secrets
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from pathlib import Path
from typing import Any

from ...artifacts.layout import (
    run_artifact_paths,
    sample_artifact_paths,
)
from ...generation.sources import normalize_generated_model_name
from .sequence import (
    EventSink,
    StageOptions,
    StageRunner,
    run_pipeline,
)
from ..planning.configuration import (
    resolve_default_bench_pipeline_paths,
)
from ...evaluation.contracts import reconstruct_evaluation_result
from ..records.layout import (
    benchmark_pipeline_state_path,
    build_run_id,
    formal_artifact_paths,
    normalize_run_key,
    split_run_key,
)
from ..validation.artifacts import (
    BenchmarkArtifactValidationError,
    validate_benchmark_stage_artifacts,
)
from ..planning.depth_artifacts import (
    build_video2traj_depth_artifact_contract,
)
from ..records.state import (
    PIPELINE_STAGE_ORDER as STATE_STAGE_ORDER,
)
from ..records.state import (
    PIPELINE_STATE_SCHEMA,
    benchmark_pipeline_state_lock,
    digest_existing_outputs,
    load_benchmark_pipeline_state,
    plan_benchmark_stage_reuse,
    write_benchmark_pipeline_state,
)
from ..records.resume import (
    build_stage_descriptor,
    collect_stage_outputs,
    stage_implementation_fingerprint,
)
from ..stages.evaluation import (
    MetricsBuilder,
    TaskSuccessRateBuilder,
    TrajectoryPathComparisonEvaluator,
    TrajectorySimilarityEvaluator,
    VLMEvaluatorRegistry,
    evaluate_benchmark_domains_stage,
)
from ..stages.sim import (
    execute_benchmark_stage,
    resolve_bench_execution_request,
)
from ..stages.task_success import (
    evaluate_bench_task_success,
    resolve_bench_task_success_request,
)
from ..stages.video2traj import (
    execute_benchmark_video2traj_stage,
)

_BENCH_OWNED_VIDEO2TRAJ_STAGE = execute_benchmark_video2traj_stage

VideoAcquirer = Callable[..., Mapping[str, Any]]
CancelCheck = Callable[[], bool]

_WORKFLOW_STAGE_ORDER = ("init", *STATE_STAGE_ORDER)
_TASK_SUCCESS_PROTECTED_OPTIONS = frozenset(
    {
        "artifact_sink",
        "gen_model",
        "output_path",
        "publish",
        "robocasa_source_root",
        "run_key",
        "sample_dir",
    }
)
_VIDEO2TRAJ_RECOVERY_SCHEMA = "dream_exe.video2traj-recovery-attempt"


class BenchmarkWorkflowCancelled(RuntimeError):
    """Signal cancellation before a selected workflow stage is invoked."""


def _reject_symlinked_contained_path(
    path: Path,
    *,
    sample_root: Path,
    label: str,
) -> Path:
    """Return one lexical sample-contained path without following aliases."""

    sample = sample_root.expanduser().resolve(strict=True)
    candidate = path.expanduser().absolute()
    try:
        relative = candidate.relative_to(sample)
    except ValueError as error:
        raise ValueError(f"{label} escapes the benchmark sample") from error
    current = sample
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(f"{label} cannot traverse a symlink")
    if not candidate.resolve(strict=False).is_relative_to(sample):
        raise ValueError(f"{label} escapes the benchmark sample")
    return candidate


def _write_recovery_manifest(
    destination: Path,
    payload: Mapping[str, Any],
) -> None:
    """Durably create the immutable manifest for one recovery attempt."""

    encoded = (
        json.dumps(
            dict(payload),
            allow_nan=False,
            ensure_ascii=False,
            indent=2,
            sort_keys=True,
        )
        + "\n"
    ).encode("utf-8")
    directory_flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    directory_flags |= getattr(os, "O_NOFOLLOW", 0)
    directory_descriptor = os.open(destination.parent, directory_flags)
    descriptor: int | None = None
    try:
        descriptor = os.open(
            destination.name,
            (os.O_WRONLY | os.O_CREAT | os.O_EXCL | getattr(os, "O_NOFOLLOW", 0)),
            0o600,
            dir_fd=directory_descriptor,
        )
        with os.fdopen(descriptor, "wb") as stream:
            descriptor = None
            stream.write(encoded)
            stream.flush()
            os.fsync(stream.fileno())
        os.fsync(directory_descriptor)
    finally:
        if descriptor is not None:
            os.close(descriptor)
        os.close(directory_descriptor)


def _quarantine_video2traj_tree(
    *,
    sample_root: Path,
    run_root: Path,
    traj_dir: Path,
    run_identity: Mapping[str, Any],
    reuse_reason: str,
    prior_stage: Mapping[str, Any] | None,
    current_stage: Mapping[str, Any],
) -> dict[str, Any] | None:
    """Atomically isolate a stale or partial formal trajectory tree.

    A resumed ``video2traj`` attempt must never mix files from an earlier
    attempt with newly computed artifacts.  The whole formal ``traj`` tree is
    therefore moved, under the already-held pipeline-state lock, into a
    sample-local audit area before the stage runner is called.
    """

    sample = sample_root.expanduser().resolve(strict=True)
    run = _reject_symlinked_contained_path(
        run_root,
        sample_root=sample,
        label="formal run root",
    )
    source = traj_dir.expanduser().absolute()
    run_paths = run_artifact_paths(run)
    if source != run_paths["traj_dir"]:
        raise ValueError(
            "formal video2traj recovery source must be the run traj directory"
        )
    source = _reject_symlinked_contained_path(
        source,
        sample_root=sample,
        label="formal trajectory tree",
    )
    if not source.exists():
        return None
    if not source.is_dir():
        raise ValueError("formal trajectory tree must be a directory")
    try:
        next(source.iterdir())
    except StopIteration:
        return None

    recovery_root = _reject_symlinked_contained_path(
        sample_artifact_paths(sample)["sample_recovery_root"],
        sample_root=sample,
        label="video2traj recovery root",
    )
    recovery_root.mkdir(parents=True, exist_ok=True)
    recovery_root = _reject_symlinked_contained_path(
        recovery_root,
        sample_root=sample,
        label="video2traj recovery root",
    )
    if not recovery_root.is_dir():
        raise ValueError("video2traj recovery root must be a directory")

    attempt_id = f"video2traj-{secrets.token_hex(12)}"
    attempt_root = recovery_root / attempt_id
    attempt_root.mkdir(mode=0o700)
    archived_tree = attempt_root / "traj"
    manifest_path = attempt_root / "manifest.json"

    prior_summary: dict[str, Any] | None = None
    if prior_stage is not None:
        prior_summary = {
            key: copy.deepcopy(prior_stage[key])
            for key in (
                "status",
                "input_fingerprint",
                "implementation_fingerprint",
                "failure_type",
            )
            if key in prior_stage
        }
    manifest = {
        "format": _VIDEO2TRAJ_RECOVERY_SCHEMA,
        "operation": "quarantine_formal_traj_before_rerun",
        "attempt_id": attempt_id,
        "run_identity": copy.deepcopy(dict(run_identity)),
        "reuse_reason": str(reuse_reason),
        "source": source.relative_to(sample).as_posix(),
        "archive": archived_tree.relative_to(sample).as_posix(),
        "prior_stage": prior_summary,
        "current_stage": {
            "input_fingerprint": current_stage["input_fingerprint"],
            "implementation_fingerprint": current_stage["implementation_fingerprint"],
        },
    }
    _write_recovery_manifest(manifest_path, manifest)
    try:
        os.replace(source, archived_tree)
    except OSError as error:
        raise RuntimeError(
            "failed to quarantine the formal trajectory tree before rerun"
        ) from error

    for directory in (attempt_root, run):
        descriptor = os.open(
            directory,
            os.O_RDONLY | getattr(os, "O_DIRECTORY", 0) | getattr(os, "O_NOFOLLOW", 0),
        )
        try:
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
    return {
        "attempt_id": attempt_id,
        "manifest_path": manifest_path.as_posix(),
        "archived_tree": archived_tree.as_posix(),
        "reuse_reason": str(reuse_reason),
    }


def _configured_task_success_options(
    options: Mapping[str, Any] | None,
    *,
    context: Mapping[str, Any],
    simulator_config_path: str | Path | None,
    execution_config_path: str | Path | None,
    scene_restore_options: Mapping[str, Any] | None,
    robocasa_source_root: str | Path | None,
) -> dict[str, Any]:
    """Bind one producer invocation to the formal run before any effect."""

    if options is None:
        normalized: dict[str, Any] = {}
    elif isinstance(options, Mapping):
        normalized = copy.deepcopy(dict(options))
    else:
        raise TypeError("task_success_options must be a mapping")
    conflicts = sorted(set(normalized).intersection(_TASK_SUCCESS_PROTECTED_OPTIONS))
    if conflicts:
        raise ValueError(
            "task_success_options cannot override canonical workflow fields: "
            + ", ".join(conflicts)
        )
    formal = dict(context["formal"])
    normalized.update(
        {
            "sample_dir": Path(context["sample_dir"]).as_posix(),
            "run_key": str(context["run_key"]),
            "gen_model": str(context["gen_model"]),
            "output_path": Path(formal["task_success"]).as_posix(),
            "publish": True,
        }
    )
    normalized.setdefault(
        "simulator_config_path",
        simulator_config_path,
    )
    if execution_config_path is not None:
        normalized.setdefault(
            "execution_config_path",
            execution_config_path,
        )
    normalized.setdefault(
        "scene_restore_options",
        scene_restore_options,
    )
    if robocasa_source_root is not None:
        normalized["robocasa_source_root"] = robocasa_source_root
    return normalized


def _validated_formal_task_success_path(
    value: str | Path,
    *,
    sample_root: Path,
) -> Path:
    """Reject aliases or symlink redirection of the formal producer output."""

    sample = sample_root.expanduser().resolve(strict=True)
    candidate = Path(value).expanduser()
    if not candidate.is_absolute():
        raise ValueError("formal task-success output path must be absolute")
    candidate = candidate.absolute()
    try:
        relative = candidate.relative_to(sample)
    except ValueError as error:
        raise ValueError(
            "formal task-success output path escapes the benchmark sample"
        ) from error
    current = sample
    for part in relative.parts:
        current = current / part
        if current.is_symlink():
            raise ValueError(
                "formal task-success output path cannot traverse a symlink"
            )
    if candidate.exists() and not candidate.is_file():
        raise ValueError("formal task-success output path is not a regular file")
    return candidate


def _cancel_guarded_runner(
    stage: str,
    runner: StageRunner,
    *,
    cancel_check: CancelCheck | None,
) -> StageRunner:
    if cancel_check is None:
        return runner

    def guarded(**options: Any) -> Mapping[str, Any]:
        cancelled = cancel_check()
        if not isinstance(cancelled, bool):
            raise TypeError("cancel_check must return bool")
        if cancelled:
            raise BenchmarkWorkflowCancelled(
                f"benchmark workflow cancelled before {stage} stage invocation"
            )
        return runner(**options)

    return guarded


def _bind_init_robocasa_source_root(
    options: StageOptions | None,
    *,
    robocasa_source_root: str | Path | None,
) -> StageOptions | None:
    if robocasa_source_root is None:
        return options
    selected_root = Path(robocasa_source_root).expanduser().resolve(strict=False)

    def bind(value: Mapping[str, Any] | None) -> dict[str, Any]:
        if value is None:
            normalized: dict[str, Any] = {}
        elif isinstance(value, Mapping):
            normalized = copy.deepcopy(dict(value))
        else:
            raise TypeError("init_options must resolve to a mapping")
        supplied = normalized.get("robocasa_source_root")
        if (
            supplied is not None
            and Path(supplied).expanduser().resolve(strict=False) != selected_root
        ):
            raise ValueError(
                "init_options.robocasa_source_root conflicts with the "
                "top-level robocasa_source_root"
            )
        normalized["robocasa_source_root"] = selected_root
        return normalized

    if callable(options):

        def bound(
            results: Mapping[str, Mapping[str, Any]],
        ) -> Mapping[str, Any]:
            return bind(options(results))

        return bound
    return bind(options)


def _reject_strict_descriptor_resolvers(
    *,
    scene_restore_options: Mapping[str, Any] | None,
    task_success_options: Mapping[str, Any],
) -> None:
    """Reject Python callbacks that descriptor planning would execute."""

    candidates = (
        (
            "scene_restore_options.path_resolver",
            dict(scene_restore_options or {}).get("path_resolver"),
        ),
        (
            "task_success_options.execution_request_resolver",
            task_success_options.get("execution_request_resolver"),
        ),
        (
            "task_success_options.scene_restore_options.path_resolver",
            dict(task_success_options.get("scene_restore_options") or {}).get(
                "path_resolver"
            ),
        ),
    )
    for label, value in candidates:
        if callable(value):
            raise ValueError(
                f"require_all_reused rejects descriptor-time callable: {label}"
            )


def _stage_state_request(
    *,
    policy: str,
    selected_stages: Sequence[str],
    invalidate_stages: Sequence[str],
    implementation_tokens: Mapping[str, str] | None,
    completed_results: Mapping[str, Mapping[str, Any]] | None,
) -> dict[str, Any]:
    clean_policy = str(policy or "off").strip().lower()
    if clean_policy not in {"off", "resume"}:
        raise ValueError("stage_state_policy must be 'off' or 'resume'")
    selected_set = {
        str(stage or "").strip().lower()
        for stage in selected_stages
        if str(stage or "").strip()
    }
    unknown_selected = selected_set.difference(_WORKFLOW_STAGE_ORDER)
    if unknown_selected:
        raise ValueError(
            "unsupported pipeline stage(s): " + ", ".join(sorted(unknown_selected))
        )
    invalidated = {
        str(stage or "").strip().lower()
        for stage in invalidate_stages
        if str(stage or "").strip()
    }
    unknown_invalidated = invalidated.difference(STATE_STAGE_ORDER)
    if unknown_invalidated:
        raise ValueError(
            "unsupported invalidated stage(s): "
            + ", ".join(sorted(unknown_invalidated))
        )
    raw_tokens = dict(implementation_tokens or {})
    for raw_stage in raw_tokens:
        if not isinstance(raw_stage, str):
            raise TypeError("implementation_tokens stage names must be strings")
        if raw_stage != raw_stage.strip() or not raw_stage:
            raise ValueError(
                "implementation_tokens stage names must be canonical non-empty strings"
            )
    unknown_tokens = set(raw_tokens).difference(STATE_STAGE_ORDER)
    if unknown_tokens:
        raise ValueError(
            "implementation_tokens has unsupported stage(s): "
            + ", ".join(sorted(unknown_tokens))
        )
    tokens: dict[str, str] = {}
    for stage, raw_token in raw_tokens.items():
        if not isinstance(raw_token, str):
            raise TypeError(f"implementation_tokens[{stage!r}] must be a string")
        token = raw_token.strip()
        if (
            not token
            or token != raw_token
            or len(token) > 256
            or any(character in token for character in ("\x00", "\r", "\n"))
            or re.fullmatch(
                r"[A-Za-z0-9][A-Za-z0-9._:@+-]{0,255}",
                token,
            )
            is None
        ):
            raise ValueError(
                f"implementation_tokens[{stage!r}] must be one stable, "
                "non-secret label of at most 256 characters"
            )
        tokens[stage] = token

    if clean_policy == "off":
        if invalidated:
            raise ValueError("invalidate_stages requires stage_state_policy='resume'")
        if tokens:
            raise ValueError(
                "implementation_tokens requires stage_state_policy='resume'"
            )
    else:
        if "init" in selected_set:
            raise ValueError(
                "stage_state_policy='resume' does not yet track init; "
                "run init separately, then resume the formal bench stages"
            )
        if completed_results:
            raise ValueError(
                "stage_state_policy='resume' cannot trust caller-supplied "
                "completed_results"
            )
    return {
        "policy": clean_policy,
        "selected": tuple(
            stage for stage in _WORKFLOW_STAGE_ORDER if stage in selected_set
        ),
        "invalidated": tuple(
            stage for stage in STATE_STAGE_ORDER if stage in invalidated
        ),
        "tokens": tokens,
    }


def _reconstructed_result(
    stage: str,
    *,
    settings: Mapping[str, Any],
    validated_artifacts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    context = dict(settings["context"])
    formal = dict(context["formal"])
    if stage == "video":
        return resolve_benchmark_video_handoff(
            uid=str(context["uid"]),
            sample_dir=context["sample_dir"],
            run_key=str(context["run_key"]),
            gen_model=str(context["gen_model"]),
            video_path=settings.get("source_video"),
        )
    payload: dict[str, Any] = {
        "ok": True,
        "returncode": 0,
        "uid": str(context["uid"]),
        "run_id": str(context["run_id"]),
        "run_key": str(context["run_key"]),
        "gen_model": str(context["gen_model"]),
    }
    validated = dict(
        validate_benchmark_stage_artifacts(
            stage,
            settings=settings,
        )
        if validated_artifacts is None
        else validated_artifacts
    )
    if stage == "video2traj":
        for label in ("assets", "trajectory", "gripper", "action"):
            if not isinstance(validated.get(label), Mapping):
                raise TypeError(f"verified {label} artifact must be a JSON object")
        payload["output_dir"] = Path(formal["traj_dir"]).as_posix()
    elif stage == "exec":
        summary = validated.get("summary")
        if not isinstance(summary, Mapping):
            raise ValueError("verified execution summary must be a JSON object")
        payload["output_dir"] = Path(formal["exec_dir"]).as_posix()
        payload["exec_summary"] = copy.deepcopy(dict(summary))
    elif stage == "task_success":
        task_success = validated.get("task_success")
        if not isinstance(task_success, Mapping):
            raise ValueError("verified task-success artifact must be a JSON object")
        task_options = dict(settings["task_success_options"])
        request = resolve_bench_task_success_request(
            sample_dir=task_options["sample_dir"],
            run_key=task_options["run_key"],
            gen_model=task_options["gen_model"],
            task_name=str(task_options.get("task_name", "") or ""),
            simulator_config_path=task_options.get("simulator_config_path"),
            execution_config_path=task_options.get("execution_config_path"),
            action_path=task_options.get("action_path"),
            object_trajectories_path=task_options.get("object_trajectories_path"),
            output_path=task_options["output_path"],
            scene_override_path=task_options.get("scene_override_path"),
            execution_request_resolver=task_options.get(
                "execution_request_resolver",
                resolve_bench_execution_request,
            ),
        )
        payload.update(
            {
                "task_name": str(request["task_name"]),
                "artifact_path": Path(formal["task_success"]).as_posix(),
                "request": request,
                "result": copy.deepcopy(dict(task_success)),
            }
        )
    elif stage == "eval":
        evaluation_result = validated.get("evaluation_result")
        if not isinstance(evaluation_result, Mapping):
            raise ValueError("verified evaluation result must be a JSON object")
        payload = reconstruct_evaluation_result(
            evaluation_result,
            formal_artifacts=formal,
        )
    return payload


def _identity(
    *,
    uid: str,
    sample_dir: str | Path,
    run_key: str,
    gen_model: str,
) -> dict[str, Any]:
    clean_uid = str(uid or "").strip()
    if not clean_uid:
        raise ValueError("uid is required")
    sample_root = Path(sample_dir).expanduser().resolve()
    if sample_root.name != clean_uid:
        raise ValueError(
            f"benchmark uid/sample_dir mismatch: {clean_uid!r} != {sample_root.name!r}"
        )
    normalized_run_key = normalize_run_key(run_key)
    video_kind, _slot = split_run_key(normalized_run_key)
    clean_gen_model = str(gen_model or "").strip()
    if video_kind == "rollout" and clean_gen_model:
        raise ValueError("rollout benchmark runs cannot name a gen_model")
    run_id = build_run_id(
        run_key=normalized_run_key,
        gen_model=clean_gen_model,
    )
    formal = formal_artifact_paths(
        sample_root,
        normalized_run_key,
        gen_model=clean_gen_model,
    )
    return {
        "uid": clean_uid,
        "sample_dir": sample_root,
        "run_key": normalized_run_key,
        "run_id": run_id,
        "video_kind": video_kind,
        "gen_model": clean_gen_model,
        "formal": formal,
    }


def resolve_benchmark_video_handoff(
    *,
    uid: str,
    sample_dir: str | Path,
    run_key: str,
    gen_model: str = "",
    video_path: str | Path | None = None,
) -> dict[str, Any]:
    """Validate one existing rollout or generated video without writing."""

    context = _identity(
        uid=uid,
        sample_dir=sample_dir,
        run_key=run_key,
        gen_model=gen_model,
    )
    sample_root = context["sample_dir"]
    if not sample_root.is_dir():
        raise FileNotFoundError(
            f"explicit bench sample directory not found: {sample_root}"
        )
    defaults = resolve_default_bench_pipeline_paths(
        sample_dir=sample_root.as_posix(),
        run_key=context["run_key"],
        gen_model=context["gen_model"],
    )
    explicit_text = str(video_path or "").strip()
    selected = (
        Path(
            explicit_text
            or (
                defaults["rollout_video_path"]
                if context["video_kind"] == "rollout"
                else defaults["gen_video_path"]
            )
        )
        .expanduser()
        .resolve()
    )
    if not selected.is_file():
        raise FileNotFoundError(f"benchmark video not found: {selected}")

    if context["video_kind"] == "rollout":
        expected = sample_artifact_paths(sample_root)["gt_video"]
        if selected != expected:
            raise ValueError("reference video must use the current GT video path")
    else:
        model = context["gen_model"]
        enhanced_suffix = "-enhanced"
        enhanced = model.lower().endswith(enhanced_suffix)
        filename_model = model[: -len(enhanced_suffix)] if enhanced else model
        sample_paths = sample_artifact_paths(sample_root)
        expected_root = sample_paths[
            "generated_enhanced_root" if enhanced else "generated_root"
        ]
        if not selected.is_relative_to(expected_root):
            raise ValueError(
                "generated video must remain under the current sample "
                "generated-artifact root"
            )
        if normalize_generated_model_name(selected.name) != filename_model:
            raise ValueError("generated video filename does not match gen_model")

    return {
        "uid": context["uid"],
        "run_id": context["run_id"],
        "run_key": context["run_key"],
        "video_kind": context["video_kind"],
        "gen_model": context["gen_model"],
        "video_path": selected.as_posix(),
        "source_video_path": selected.as_posix(),
    }


def run_benchmark_workflow(
    *,
    uid: str,
    sample_dir: str | Path,
    run_key: str,
    runtime_config: Mapping[str, Any] | None = None,
    gen_model: str = "",
    source_video_path: str | Path | None = None,
    initialize: StageRunner | None = None,
    init_options: StageOptions | None = None,
    acquire_video: VideoAcquirer | None = None,
    video_options: StageOptions | None = None,
    trajectory_dependencies: Mapping[str, Any] | None = None,
    trajectory_runtime_asset_base: str | Path | None = None,
    trajectory_run_options: Mapping[str, Any] | None = None,
    trajectory_video_backend: str = "auto",
    trajectory_runner: Callable[..., dict[str, Any]] | None = None,
    simulator_runner: Callable[..., dict[str, Any]] | None = None,
    scene_restore_options: Mapping[str, Any] | None = None,
    robocasa_source_root: str | Path | None = None,
    metrics_builder: MetricsBuilder | None = None,
    vlm_request_manifest_path: str | Path | None = None,
    vlm_evaluators: VLMEvaluatorRegistry | None = None,
    trajectory_path_comparison_reference_path: str | Path | None = None,
    trajectory_path_comparison_evaluator: (
        TrajectoryPathComparisonEvaluator | None
    ) = None,
    trajectory_similarity_specs: (Sequence[Mapping[str, Any]] | None) = None,
    trajectory_similarity_evaluator: (TrajectorySimilarityEvaluator | None) = None,
    task_success_rate_specs: (Sequence[Mapping[str, Any]] | None) = None,
    task_success_rate_builder: TaskSuccessRateBuilder | None = None,
    task_success_rate_options: Mapping[str, Any] | None = None,
    task_success_options: Mapping[str, Any] | None = None,
    task_success_runner: StageRunner | None = None,
    simulator_config_path: str | Path | None = None,
    trajectory_config_path: str | Path | None = None,
    execution_config_path: str | Path | None = None,
    only_stages: Sequence[str] | None = None,
    completed_results: (Mapping[str, Mapping[str, Any]] | None) = None,
    stage_state_policy: str = "off",
    invalidate_stages: Sequence[str] = (),
    implementation_tokens: Mapping[str, str] | None = None,
    require_all_reused: bool = False,
    continue_on_error: bool = False,
    event_sink: EventSink | None = None,
    cancel_check: CancelCheck | None = None,
) -> dict[str, Any]:
    """Run one formal benchmark run through the canonical stage sequence.

    The default is ``video → video2traj → exec → eval`` for an already
    initialized current-layout sample. Supplying ``task_success_options`` (an
    empty mapping is sufficient), a ``task_success_runner``, or selecting the
    stage explicitly adds the simulator-backed post-exec producer:
    ``video → video2traj → exec → task_success → eval``. Supplying
    ``initialize`` adds the init stage. Supplying ``acquire_video`` lets an
    external generator/importer produce the video; its returned ``video_path``
    is validated before the algorithm stage. The additive eval result always
    contains deterministic trajectory executability and may consume an
    explicit Table 3 reference trajectory, Table 2 trajectory-reference,
    task-success cohort, and VLM inputs. When the
    producer is enabled and no explicit task-success cohort is supplied, eval
    receives the produced artifact as one explicit run-local aggregation
    record. The ``run-bench`` CLI currently binds only its existing
    deterministic defaults; richer inputs use this Python callable until a
    versioned CLI config surface is added. ``require_all_reused`` is a strict
    evidence-consumer gate: under ``resume`` it permits reconstruction only
    when every selected stage's descriptor, recorded bytes, and semantic
    artifact contract verify. It fails before recovery, state publication,
    or any stage runner is invoked when one stage would execute.
    """

    if cancel_check is not None and not callable(cancel_check):
        raise TypeError("cancel_check must be callable")
    if not isinstance(require_all_reused, bool):
        raise TypeError("require_all_reused must be bool")

    context = _identity(
        uid=uid,
        sample_dir=sample_dir,
        run_key=run_key,
        gen_model=gen_model,
    )
    sample_root: Path = context["sample_dir"]
    configured_init_options = _bind_init_robocasa_source_root(
        init_options,
        robocasa_source_root=robocasa_source_root,
    )
    task_success_explicit = (
        task_success_options is not None or task_success_runner is not None
    )
    selected_stages = (
        tuple(only_stages)
        if only_stages is not None
        else (
            (
                "init",
                "video",
                "video2traj",
                "exec",
                *(("task_success",) if task_success_explicit else ()),
                "eval",
            )
            if initialize is not None
            else (
                "video",
                "video2traj",
                "exec",
                *(("task_success",) if task_success_explicit else ()),
                "eval",
            )
        )
    )
    state_request = _stage_state_request(
        policy=stage_state_policy,
        selected_stages=selected_stages,
        invalidate_stages=invalidate_stages,
        implementation_tokens=implementation_tokens,
        completed_results=completed_results,
    )
    if require_all_reused and state_request["policy"] != "resume":
        raise ValueError("require_all_reused requires stage_state_policy='resume'")
    if (
        "init" not in selected_stages
        and "init" not in dict(completed_results or {})
        and not sample_root.is_dir()
    ):
        raise FileNotFoundError(
            f"explicit bench sample directory not found: {sample_root}"
        )
    if "video2traj" in selected_stages and not isinstance(
        runtime_config,
        Mapping,
    ):
        raise TypeError(
            "runtime_config must be an explicit mapping when video2traj is selected"
        )

    trajectory_path = (
        Path(trajectory_config_path).expanduser().resolve()
        if trajectory_config_path is not None and str(trajectory_config_path).strip()
        else sample_artifact_paths(sample_root)["trajectory_config"]
    )
    explicit_execution_path = (
        Path(execution_config_path).expanduser().resolve()
        if execution_config_path is not None and str(execution_config_path).strip()
        else None
    )
    formal = context["formal"]
    task_success_selected = "task_success" in state_request["selected"]
    if task_success_runner is not None and not callable(task_success_runner):
        raise TypeError("task_success_runner must be callable")
    configured_task_success_options = (
        _configured_task_success_options(
            task_success_options,
            context=context,
            simulator_config_path=simulator_config_path,
            execution_config_path=explicit_execution_path,
            scene_restore_options=scene_restore_options,
            robocasa_source_root=robocasa_source_root,
        )
        if task_success_selected
        else {}
    )
    if require_all_reused:
        _reject_strict_descriptor_resolvers(
            scene_restore_options=scene_restore_options,
            task_success_options=configured_task_success_options,
        )
    bind_produced_task_success = (
        task_success_selected
        and "eval" in state_request["selected"]
        and task_success_rate_specs is None
    )
    effective_task_success_rate_specs = (
        [
            {
                "uid": context["uid"],
                "path": Path(formal["task_success"]).as_posix(),
            }
        ]
        if bind_produced_task_success
        else task_success_rate_specs
    )

    def default_video_runner() -> dict[str, Any]:
        return resolve_benchmark_video_handoff(
            uid=context["uid"],
            sample_dir=sample_root,
            run_key=context["run_key"],
            gen_model=context["gen_model"],
            video_path=source_video_path,
        )

    def acquired_video_runner(**options: Any) -> dict[str, Any]:
        assert acquire_video is not None
        raw = acquire_video(**options)
        if not isinstance(raw, Mapping):
            raise TypeError("video acquirer must return a mapping")
        payload = copy.deepcopy(dict(raw))
        acquired_path = str(
            payload.get(
                "video_path",
                payload.get("source_video_path", ""),
            )
            or ""
        ).strip()
        if not acquired_path:
            raise ValueError("video acquirer result must contain video_path")
        handoff = resolve_benchmark_video_handoff(
            uid=context["uid"],
            sample_dir=sample_root,
            run_key=context["run_key"],
            gen_model=context["gen_model"],
            video_path=acquired_path,
        )
        payload.update(handoff)
        return payload

    def trajectory_options(
        results: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        video_result = dict(results.get("video", {}) or {})
        selected_video = str(
            video_result.get(
                "video_path",
                video_result.get(
                    "source_video_path",
                    source_video_path or "",
                ),
            )
            or ""
        ).strip()
        handoff = resolve_benchmark_video_handoff(
            uid=context["uid"],
            sample_dir=sample_root,
            run_key=context["run_key"],
            gen_model=context["gen_model"],
            video_path=(selected_video or None),
        )
        return {
            "uid": context["uid"],
            "sample_dir": sample_root.as_posix(),
            "run_id": context["run_id"],
            "run_key": context["run_key"],
            "video_kind": context["video_kind"],
            "gen_model": context["gen_model"],
            "source_video_path": handoff["video_path"],
            "pipeline_config_path": trajectory_path.as_posix(),
            "output_traj_root": formal["traj_dir"].as_posix(),
            "use_rollout_gt_depth": (context["run_key"] == "gt_video/gt_depth"),
        }

    execution_options = {
        "uid": context["uid"],
        "sample_dir": sample_root.as_posix(),
        "run_id": context["run_id"],
        "run_key": context["run_key"],
        "video_kind": context["video_kind"],
        "gen_model": context["gen_model"],
        "execution_config_path": None,
        "trajectory_root": formal["traj_dir"].as_posix(),
        "action_json_path": formal["action"].as_posix(),
        "output_exec_root": formal["exec_dir"].as_posix(),
    }
    if explicit_execution_path is not None:
        execution_options["execution_config_path"] = explicit_execution_path.as_posix()
    if robocasa_source_root is not None:
        execution_options["robocasa_source_root"] = robocasa_source_root

    def evaluation_options(
        results: Mapping[str, Mapping[str, Any]],
    ) -> dict[str, Any]:
        if bind_produced_task_success:
            producer = results.get("task_success")
            if not isinstance(producer, Mapping):
                raise RuntimeError(
                    "eval requires the selected task_success producer result"
                )
            produced_path = str(producer.get("artifact_path", "") or "").strip()
            expected_path = Path(formal["task_success"]).as_posix()
            if (
                not produced_path
                or Path(produced_path).expanduser().resolve()
                != Path(expected_path).resolve()
            ):
                raise ValueError(
                    "task_success producer did not bind the formal artifact"
                )
        return {
            "uid": context["uid"],
            "sample_dir": sample_root.as_posix(),
            "run_id": context["run_id"],
            "run_key": context["run_key"],
            "video_kind": context["video_kind"],
            "gen_model": context["gen_model"],
            "output_exec_root": formal["exec_dir"].as_posix(),
            "task_success_rate_specs": effective_task_success_rate_specs,
            "task_success_rate_options": task_success_rate_options,
        }

    trajectory_stage = partial(
        execute_benchmark_video2traj_stage,
        runtime_config=runtime_config,
        dependencies=trajectory_dependencies,
        runtime_asset_base=trajectory_runtime_asset_base,
        run_options=trajectory_run_options,
        video_backend=trajectory_video_backend,
        simulator_config_path=simulator_config_path,
        runner=trajectory_runner,
    )
    execution_stage = partial(
        execute_benchmark_stage,
        simulator_config_path=simulator_config_path,
        scene_restore_options=scene_restore_options,
        runner=simulator_runner,
    )
    evaluation_stage = partial(
        evaluate_benchmark_domains_stage,
        metrics_builder=metrics_builder,
        vlm_request_manifest_path=vlm_request_manifest_path,
        vlm_evaluators=vlm_evaluators,
        trajectory_path_comparison_reference_path=(
            trajectory_path_comparison_reference_path
        ),
        trajectory_path_comparison_evaluator=(trajectory_path_comparison_evaluator),
        trajectory_similarity_specs=trajectory_similarity_specs,
        trajectory_similarity_evaluator=(trajectory_similarity_evaluator),
        task_success_rate_builder=task_success_rate_builder,
    )
    selected_task_success_runner = (
        evaluate_bench_task_success
        if task_success_runner is None
        else task_success_runner
    )

    def task_success_stage(**options: Any) -> dict[str, Any]:
        expected_path = _validated_formal_task_success_path(
            formal["task_success"],
            sample_root=sample_root,
        )
        selected_output = _validated_formal_task_success_path(
            options.get("output_path", ""),
            sample_root=sample_root,
        )
        if selected_output != expected_path:
            raise ValueError(
                "task_success stage output_path does not match the formal artifact"
            )
        raw_result = selected_task_success_runner(**options)
        if not isinstance(raw_result, Mapping):
            raise TypeError("task_success_runner must return a mapping")
        result = copy.deepcopy(dict(raw_result))
        artifact_text = str(result.get("artifact_path", "") or "").strip()
        if not artifact_text:
            raise ValueError("task_success_runner result must bind artifact_path")
        artifact_path = _validated_formal_task_success_path(
            artifact_text,
            sample_root=sample_root,
        )
        expected_path = _validated_formal_task_success_path(
            formal["task_success"],
            sample_root=sample_root,
        )
        if artifact_path != expected_path:
            raise ValueError("task_success_runner did not bind the formal artifact")
        validate_benchmark_stage_artifacts(
            "task_success",
            settings={
                "context": context,
                "task_success_options": copy.deepcopy(dict(options)),
            },
        )
        result["artifact_path"] = expected_path.as_posix()
        return result

    video_stage: StageRunner = (
        default_video_runner if acquire_video is None else acquired_video_runner
    )
    guarded_initialize = (
        None
        if initialize is None
        else _cancel_guarded_runner(
            "init",
            initialize,
            cancel_check=cancel_check,
        )
    )

    def invoke_pipeline(
        *,
        stage_runners: Mapping[str, StageRunner],
        verified_completed: (Mapping[str, Mapping[str, Any]] | None),
    ) -> dict[str, Any]:
        return run_pipeline(
            initialize=guarded_initialize,
            acquire_video=stage_runners["video"],
            video2traj=stage_runners["video2traj"],
            execute=stage_runners["exec"],
            task_success=stage_runners["task_success"],
            evaluate=stage_runners["eval"],
            init_options=configured_init_options,
            video_options=video_options,
            video2traj_options=trajectory_options,
            exec_options=execution_options,
            task_success_options=configured_task_success_options,
            eval_options=evaluation_options,
            only_stages=selected_stages,
            completed_results=verified_completed,
            continue_on_error=continue_on_error,
            event_sink=event_sink,
        )

    default_runners: dict[str, StageRunner] = {
        stage: _cancel_guarded_runner(
            stage,
            runner,
            cancel_check=cancel_check,
        )
        for stage, runner in {
            "video": video_stage,
            "video2traj": trajectory_stage,
            "exec": execution_stage,
            "task_success": task_success_stage,
            "eval": evaluation_stage,
        }.items()
    }
    state_summary: dict[str, Any] | None = None
    if state_request["policy"] == "off":
        result = invoke_pipeline(
            stage_runners=default_runners,
            verified_completed=completed_results,
        )
    else:
        run_identity = {
            "uid": context["uid"],
            "run_id": context["run_id"],
            "run_key": context["run_key"],
            "gen_model": context["gen_model"],
        }
        try:
            handoff = resolve_benchmark_video_handoff(
                uid=context["uid"],
                sample_dir=sample_root,
                run_key=context["run_key"],
                gen_model=context["gen_model"],
                video_path=source_video_path,
            )
            state_source_video: Path | None = Path(handoff["video_path"])
        except (FileNotFoundError, OSError, ValueError):
            state_source_video = (
                Path(source_video_path).expanduser().resolve(strict=False)
                if source_video_path is not None and str(source_video_path).strip()
                else None
            )
        execution_mode = ""
        execution_trajectory_path = formal["ee_traj"].as_posix()
        execution_action_path = formal["action"].as_posix()
        execution_traj_key = "eef_controller"
        execution_max_steps = -1
        try:
            execution_request = resolve_bench_execution_request(
                sample_dir=sample_root,
                run_key=context["run_key"],
                gen_model=context["gen_model"],
                simulator_config_path=simulator_config_path,
                execution_config_path=explicit_execution_path,
                action_path=formal["action"],
                output_dir=formal["exec_dir"],
            )
            execution_config = dict(execution_request["execution_config"])
            raw_execution_mode = str(
                dict(execution_config.get("execution", {})).get("mode", "") or ""
            ).strip()
            execution_mode = (
                "frame" if raw_execution_mode == "frame_traj" else raw_execution_mode
            )
            execution_inputs = dict(execution_config.get("input", {}))
            execution_runtime = dict(execution_config.get("runtime", {}))
            execution_trajectory_path = str(
                execution_request.get("trajectory_path", "")
                or formal["ee_traj"].as_posix()
            )
            execution_action_path = str(
                execution_request.get("action_path", "") or formal["action"].as_posix()
            )
            execution_traj_key = str(
                execution_inputs.get("traj_key", "eef_controller") or "eef_controller"
            )
            execution_max_steps = int(execution_runtime.get("max_steps", -1))
        except (FileNotFoundError, OSError, TypeError, ValueError):
            pass
        state_settings: dict[str, Any] = {
            "context": context,
            "run_identity": run_identity,
            "source_video": state_source_video,
            "execution_mode": execution_mode,
            "execution_trajectory_path": execution_trajectory_path,
            "execution_action_path": execution_action_path,
            "execution_traj_key": execution_traj_key,
            "execution_max_steps": execution_max_steps,
            "runtime_config": runtime_config,
            "runtime_asset_base": trajectory_runtime_asset_base,
            "trajectory_path": trajectory_path,
            # ``None`` means the current-layout adapter owns resolution of the
            # canonical base plus the run-specific override.  Only a caller-
            # supplied path is explicit and therefore allowed to bypass that
            # merge.
            "execution_path": explicit_execution_path,
            "simulator_config_path": simulator_config_path,
            "trajectory_run_options": trajectory_run_options,
            "trajectory_video_backend": trajectory_video_backend,
            "trajectory_runner": trajectory_runner,
            "trajectory_dependencies": trajectory_dependencies,
            "simulator_runner": simulator_runner,
            "scene_restore_options": scene_restore_options,
            "robocasa_source_root": robocasa_source_root,
            "metrics_builder": metrics_builder,
            "vlm_request_manifest_path": vlm_request_manifest_path,
            "vlm_evaluators": vlm_evaluators,
            "trajectory_path_comparison_reference_path": (
                trajectory_path_comparison_reference_path
            ),
            "trajectory_path_comparison_evaluator": (
                trajectory_path_comparison_evaluator
            ),
            "trajectory_similarity_specs": trajectory_similarity_specs,
            "trajectory_similarity_evaluator": (trajectory_similarity_evaluator),
            "task_success_rate_specs": effective_task_success_rate_specs,
            "task_success_rate_builder": task_success_rate_builder,
            "task_success_rate_options": task_success_rate_options,
            "task_success_options": configured_task_success_options,
            "task_success_runner": task_success_runner,
            "acquire_video": acquire_video,
            "video_options": video_options,
            "use_gt_depth": (context["run_key"] == "gt_video/gt_depth"),
        }
        if (
            "video2traj" in state_request["selected"]
            and trajectory_runner is None
            and execute_benchmark_video2traj_stage is _BENCH_OWNED_VIDEO2TRAJ_STAGE
        ):
            state_settings["video2traj_depth_contract"] = (
                build_video2traj_depth_artifact_contract(
                    sample_root=sample_root,
                    run_key=context["run_key"],
                    gen_model=context["gen_model"],
                    trajectory_config_path=trajectory_path,
                    traj_dir=formal["traj_dir"],
                )
            )
        state_path = benchmark_pipeline_state_path(
            sample_root,
            context["run_key"],
            gen_model=context["gen_model"],
        )
        if require_all_reused and not state_path.exists():
            raise RuntimeError(
                "strict verified-reuse gate refused formal stage execution: "
                "prior_state_missing"
            )
        with benchmark_pipeline_state_lock(
            state_path,
            require_existing=require_all_reused,
        ):
            prior_state = load_benchmark_pipeline_state(state_path)
            selected_state_stages = [
                stage
                for stage in STATE_STAGE_ORDER
                if stage in state_request["selected"]
            ]
            implementation_fingerprints = {
                stage: stage_implementation_fingerprint(
                    stage,
                    token=state_request["tokens"].get(stage, ""),
                )
                for stage in selected_state_stages
            }
            state_settings["implementation_fingerprints"] = implementation_fingerprints

            def assert_implementation_snapshot() -> None:
                for stage, expected in implementation_fingerprints.items():
                    actual = stage_implementation_fingerprint(
                        stage,
                        token=state_request["tokens"].get(stage, ""),
                    )
                    if actual != expected:
                        raise RuntimeError(
                            "current implementation source changed while planning or running "
                            f"the {stage} stage; retry the workflow"
                        )

            current_descriptors = {
                stage: build_stage_descriptor(
                    stage,
                    settings=state_settings,
                    token=state_request["tokens"].get(stage, ""),
                )
                for stage in selected_state_stages
            }
            assert_implementation_snapshot()
            reuse_plan = plan_benchmark_stage_reuse(
                run_identity=run_identity,
                current_stages=current_descriptors,
                prior_state=prior_state,
                selected_stages=selected_state_stages,
                sample_root=sample_root,
                run_root=formal["run_root"],
                invalidate_stages=state_request["invalidated"],
            )
            validated_reuse_artifacts: dict[str, Mapping[str, Any]] = {}
            for stage in tuple(reuse_plan["reused_stages"]):
                try:
                    validated_reuse_artifacts[stage] = (
                        validate_benchmark_stage_artifacts(
                            stage,
                            settings=state_settings,
                        )
                    )
                except BenchmarkArtifactValidationError as error:
                    invalid_descriptor = copy.deepcopy(current_descriptors[stage])
                    invalid_descriptor["reusable"] = False
                    invalid_descriptor["reason"] = (
                        f"invalid_stage_artifacts:{error.reason}"
                    )
                    current_descriptors[stage] = invalid_descriptor
                    reuse_plan = plan_benchmark_stage_reuse(
                        run_identity=run_identity,
                        current_stages=current_descriptors,
                        prior_state=prior_state,
                        selected_stages=selected_state_stages,
                        sample_root=sample_root,
                        run_root=formal["run_root"],
                        invalidate_stages=state_request["invalidated"],
                    )
                    break
            if require_all_reused and reuse_plan["run_stages"]:
                reasons = ", ".join(
                    f"{stage}={reuse_plan['stages'][stage]['reason']}"
                    for stage in reuse_plan["run_stages"]
                )
                raise RuntimeError(
                    "strict verified-reuse gate refused formal stage "
                    f"execution: {reasons}"
                )
            verified_completed = {
                stage: _reconstructed_result(
                    stage,
                    settings=state_settings,
                    validated_artifacts=validated_reuse_artifacts.get(stage),
                )
                for stage in reuse_plan["reused_stages"]
            }
            state_document: dict[str, Any] = {
                "format": PIPELINE_STATE_SCHEMA,
                "run_identity": copy.deepcopy(run_identity),
                "stages": (
                    copy.deepcopy(prior_state["stages"])
                    if prior_state is not None
                    and prior_state["run_identity"] == run_identity
                    else {}
                ),
            }
            prior_video2traj_stage = (
                copy.deepcopy(prior_state["stages"].get("video2traj"))
                if prior_state is not None
                and prior_state["run_identity"] == run_identity
                else None
            )
            recovery_records: list[dict[str, Any]] = []

            def invalidate_state_from(stage: str) -> None:
                start = STATE_STAGE_ORDER.index(stage)
                for downstream in STATE_STAGE_ORDER[start:]:
                    state_document["stages"].pop(downstream, None)
                write_benchmark_pipeline_state(
                    state_path,
                    state_document,
                )

            if state_request["invalidated"]:
                invalidate_state_from(state_request["invalidated"][0])

            def stateful_runner(
                stage: str,
                runner: StageRunner,
            ) -> StageRunner:
                def run_and_publish(**options: Any) -> Mapping[str, Any]:
                    invalidate_state_from(stage)
                    descriptor_before = build_stage_descriptor(
                        stage,
                        settings=state_settings,
                        token=state_request["tokens"].get(
                            stage,
                            "",
                        ),
                    )
                    try:
                        if stage == "video2traj":
                            recovery = _quarantine_video2traj_tree(
                                sample_root=sample_root,
                                run_root=Path(formal["run_root"]),
                                traj_dir=Path(formal["traj_dir"]),
                                run_identity=run_identity,
                                reuse_reason=str(
                                    reuse_plan["stages"]["video2traj"]["reason"]
                                ),
                                prior_stage=prior_video2traj_stage,
                                current_stage=descriptor_before,
                            )
                            if recovery is not None:
                                recovery_records.append(recovery)
                        raw_result = runner(**options)
                        if not isinstance(raw_result, Mapping):
                            raise TypeError(
                                f"{stage} stage returned a non-mapping result"
                            )
                        stage_result = copy.deepcopy(dict(raw_result))
                        if stage == "video":
                            video_text = str(
                                stage_result.get(
                                    "video_path",
                                    stage_result.get(
                                        "source_video_path",
                                        "",
                                    ),
                                )
                                or ""
                            ).strip()
                            state_settings["source_video"] = (
                                Path(video_text).expanduser().resolve()
                                if video_text
                                else None
                            )
                        output_paths = collect_stage_outputs(
                            stage,
                            result=stage_result,
                            settings=state_settings,
                        )
                        output_records = digest_existing_outputs(
                            output_paths,
                            sample_root=sample_root,
                            run_root=formal["run_root"],
                        )
                        descriptor = build_stage_descriptor(
                            stage,
                            settings=state_settings,
                            token=state_request["tokens"].get(
                                stage,
                                "",
                            ),
                        )
                        required_outputs = descriptor.get(
                            "required_outputs",
                            [],
                        )
                        if not required_outputs:
                            raise RuntimeError(
                                f"{stage} has no verifiable required-output contract"
                            )
                        required = {
                            (item["root"], item["path"]) for item in required_outputs
                        }
                        recorded = {
                            (item["root"], item["path"]) for item in output_records
                        }
                        missing = sorted(required.difference(recorded))
                        if missing:
                            root_name, relative = missing[0]
                            raise RuntimeError(
                                f"{stage} did not publish required output "
                                f"{root_name}:{relative}"
                            )
                        expected_implementation = implementation_fingerprints[stage]
                        actual_implementation = stage_implementation_fingerprint(
                            stage,
                            token=state_request["tokens"].get(stage, ""),
                        )
                        implementation_changed = (
                            descriptor["implementation_fingerprint"]
                            != expected_implementation
                            or actual_implementation != expected_implementation
                        )
                        inputs_changed = stage != "video" and (
                            descriptor_before["input_fingerprint"]
                            != descriptor["input_fingerprint"]
                            or descriptor_before["reusable"] != descriptor["reusable"]
                        )
                        if implementation_changed or inputs_changed:
                            raise RuntimeError(
                                f"{stage} inputs or implementation changed "
                                "during execution"
                            )
                        validated_artifacts = validate_benchmark_stage_artifacts(
                            stage,
                            settings=state_settings,
                        )
                        if output_records:
                            state_document["stages"][stage] = {
                                "status": "completed",
                                "input_fingerprint": descriptor["input_fingerprint"],
                                "implementation_fingerprint": descriptor[
                                    "implementation_fingerprint"
                                ],
                                "outputs": output_records,
                            }
                        write_benchmark_pipeline_state(
                            state_path,
                            state_document,
                        )
                        if descriptor["reusable"]:
                            return _reconstructed_result(
                                stage,
                                settings=state_settings,
                                validated_artifacts=validated_artifacts,
                            )
                        return stage_result
                    except Exception as error:
                        try:
                            descriptor = build_stage_descriptor(
                                stage,
                                settings=state_settings,
                                token=state_request["tokens"].get(
                                    stage,
                                    "",
                                ),
                            )
                        except Exception:  # noqa: BLE001 - best-effort failure state
                            descriptor = descriptor_before
                        state_document["stages"][stage] = {
                            "status": "failed",
                            "input_fingerprint": descriptor["input_fingerprint"],
                            "implementation_fingerprint": descriptor[
                                "implementation_fingerprint"
                            ],
                            "outputs": [],
                            "failure_type": type(error).__name__,
                        }
                        try:
                            write_benchmark_pipeline_state(
                                state_path,
                                state_document,
                            )
                        except Exception as state_error:
                            if hasattr(error, "add_note"):
                                error.add_note(
                                    "benchmark failed-state publication also "
                                    f"failed with {type(state_error).__name__}"
                                )
                            raise error from state_error
                        raise

                return run_and_publish

            wrapped_runners = {
                stage: (
                    default_runners[stage]
                    if stage in reuse_plan["reused_stages"]
                    else stateful_runner(
                        stage,
                        default_runners[stage],
                    )
                )
                for stage in STATE_STAGE_ORDER
            }
            result = invoke_pipeline(
                stage_runners=wrapped_runners,
                verified_completed=verified_completed,
            )
            assert_implementation_snapshot()
            state_summary = {
                "policy": "resume",
                "state_path": state_path.as_posix(),
                "reuse_plan": reuse_plan,
                "recoveries": recovery_records,
            }

    response = {
        "format": "dream_exe.benchmark_workflow",
        "uid": context["uid"],
        "run_id": context["run_id"],
        "run_key": context["run_key"],
        "video_kind": context["video_kind"],
        "gen_model": context["gen_model"],
        "sample_dir": sample_root.as_posix(),
        "artifacts": {
            key: path.as_posix() for key, path in formal.items() if key != "sample_root"
        },
        "pipeline": result,
    }
    if state_summary is not None:
        response["stage_state"] = state_summary
    return response


__all__ = [
    "BenchmarkWorkflowCancelled",
    "CancelCheck",
    "TrajectoryPathComparisonEvaluator",
    "VideoAcquirer",
    "resolve_benchmark_video_handoff",
    "run_benchmark_workflow",
]
