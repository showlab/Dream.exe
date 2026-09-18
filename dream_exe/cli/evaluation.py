"""Trajectory, task success-rate, execution, and optional VLM CLI commands."""

from __future__ import annotations

import argparse
import copy
import json
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..bench.data.repository import BenchRepository
from ..bench.data.workspace import load_workspace
from ..bench.outputs.lifecycle import write_json_atomic
from ..evaluation.suite import (
    EVALUATION_FAMILIES,
    EvaluationSelection,
    evaluate_executability_selection,
    evaluate_task_selection,
    evaluate_trajectory_selection,
    portable_error_message,
    portable_workspace_path,
    resolve_evaluation_selection,
    selection_summary,
)
from ..evaluation.execution import (
    aggregate_task_success_paths,
    build_metrics_from_exec_dir,
)
from ..evaluation.vlm.media import prepare_vlm_media_grids
from ..evaluation.vlm.aggregate import aggregate_vlm_judges
from ..evaluation.vlm.credentials import (
    configured_credentials_path,
    load_vlm_credentials,
)
from ..evaluation.vlm.providers import (
    CURRENT_MAX_TOKENS,
    CURRENT_SEED,
    TOKEN_LIMIT_PARAMETERS,
    OpenAICompatibleVLMInference,
)
from ..evaluation.vlm.batch import (
    SAVED_MEDIA_RUBRICS,
    run_saved_media_vlm_batch,
)
from ..evaluation.vlm.prompts import render_bench_vlm_prompts
from ..evaluation.trajectory import (
    DEFAULT_TRAJECTORY_SIMILARITY_PROTOCOL,
    TRAJECTORY_SIMILARITY_GROUPS,
    TRAJECTORY_SIMILARITY_PROTOCOLS,
    evaluate_trajectory_executability_dir,
    evaluate_trajectory_path_comparison_paths,
    evaluate_trajectory_similarity_files,
)
from ..evaluation.vlm.auxiliary import (
    DEFAULT_CONTEXT_POINTS_PER_TRACK,
    GENERIC_JSON_OBJECT_PARSER_ID,
    run_saved_media_trajectory_vlm_batch,
)
from ..evaluation.vlm.scoring import list_grid_image_files
from ._common import _optional, _print_json

OPENAI_COMPATIBLE_API_KEY_ENV = "OPENAI_API_KEY"
PAPER_VISUAL_RUBRICS = (
    "subject_stability",
    "physical_plausibility",
    "task_adherence",
)


def _suite_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--scope", choices=("one", "all"), required=True)
    parser.add_argument("--case", default="")
    parser.add_argument("--candidate-model", required=True)
    parser.add_argument(
        "--prompt-variant",
        choices=("standard", "enhanced"),
        default="standard",
    )


def _suite_visual_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--judge",
        default="both",
        help=(
            "One judge ID declared by the benchmark, or 'both' for the "
            "protocol-declared mean."
        ),
    )
    parser.add_argument(
        "--rubric",
        choices=("all", *PAPER_VISUAL_RUBRICS),
        default="all",
    )
    parser.add_argument("--credentials", default="")
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--fresh-grids", action="store_true")


def register_evaluation_suite(parser: argparse.ArgumentParser) -> None:
    families = parser.add_subparsers(dest="evaluation_family", required=True)
    for family in EVALUATION_FAMILIES:
        child = families.add_parser(family)
        _suite_common_arguments(child)
        if family in {"visual", "all"}:
            _suite_visual_arguments(child)
        if family in {"trajectory", "all"}:
            child.add_argument(
                "--trajectory-protocol",
                choices=TRAJECTORY_SIMILARITY_PROTOCOLS,
                default=DEFAULT_TRAJECTORY_SIMILARITY_PROTOCOL,
            )
        child.set_defaults(_handler=_run_evaluation_suite)


def _add_vlm_evaluation_arguments(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--mode",
        choices=("video_only", "video_trajectory"),
        default="video_only",
        help=(
            "video_only runs the benchmark visual rubrics; "
            "video_trajectory consumes explicit saved media+union_traj pairs."
        ),
    )
    media_group = parser.add_mutually_exclusive_group()
    media_group.add_argument(
        "--media-dir",
        help="Explicit directory containing already prepared VLM grid images.",
    )
    media_group.add_argument(
        "--video-dir",
        help="Explicit video directory or file to prepare and evaluate.",
    )
    parser.add_argument(
        "--grid-output-dir",
        default="",
        help="Required with --video-dir; receives current-compatible grid images.",
    )
    parser.add_argument(
        "--fresh-grids",
        action="store_true",
        help="Rebuild grids instead of reusing a non-empty grid output directory.",
    )
    parser.add_argument("--prompts-json", default="")
    parser.add_argument(
        "--workspace",
        default="",
        help="Workspace JSON used to resolve benchmark-owned VLM inputs.",
    )
    parser.add_argument("--case", default="", help="Benchmark case UID.")
    parser.add_argument(
        "--candidate-model",
        default="",
        help="Generated-video model directory under workspace video results.",
    )
    parser.add_argument(
        "--prompt-variant",
        choices=("standard", "enhanced"),
        default="standard",
    )
    parser.add_argument(
        "--judge",
        default="",
        help="Judge ID declared by bench/protocol/evaluation.json.",
    )
    parser.add_argument(
        "--rubric",
        choices=SAVED_MEDIA_RUBRICS,
        default=None,
    )
    parser.add_argument("--output-csv", default="")
    parser.add_argument(
        "--trajectory-inputs-json",
        default="",
        help=(
            "video_trajectory only: explicit JSON array of media_path, "
            "trajectory_path, task_metadata, and optional logical identity."
        ),
    )
    parser.add_argument(
        "--trajectory-prompt-file",
        default="",
        help="video_trajectory only: caller-owned rubric prompt text.",
    )
    parser.add_argument(
        "--trajectory-prompt-id",
        dest="trajectory_prompt_id",
        default="",
    )
    parser.add_argument("--trajectory-rubric-id", default="")
    parser.add_argument(
        "--trajectory-parser-id",
        dest="trajectory_parser_id",
        default=GENERIC_JSON_OBJECT_PARSER_ID,
    )
    parser.add_argument(
        "--trajectory-max-context-points",
        type=int,
        default=DEFAULT_CONTEXT_POINTS_PER_TRACK,
    )
    parser.add_argument(
        "--output-json",
        default="",
        help="video_trajectory only: deterministic batch report path.",
    )
    parser.add_argument("--prediction-dir", default="")
    parser.add_argument("--model", default="")
    parser.add_argument(
        "--models-config",
        default="",
        help=(
            "Optional trusted dream-exe.models catalog. With this flag, "
            "--model is an exact eval/vlm catalog ID and direct "
            "endpoint/credential flags are not allowed."
        ),
    )
    parser.add_argument("--base-url", default="")
    parser.add_argument(
        "--api-key-env",
        default="",
    )
    parser.add_argument(
        "--credentials",
        default="",
        help=(
            "Local JSON credential file. The selected bench judge is the "
            "default profile; DREAM_EXE_CREDENTIALS_FILE provides the same binding."
        ),
    )
    parser.add_argument(
        "--credential-profile",
        default="",
        help="Credential profile for an explicit non-bench VLM request.",
    )
    parser.add_argument(
        "--backend-label",
        default="",
    )
    parser.add_argument(
        "--prompt-id",
        dest="prompt_id",
        default="paper",
    )
    parser.add_argument(
        "--token-limit-parameter",
        choices=TOKEN_LIMIT_PARAMETERS,
        default="max_completion_tokens",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=CURRENT_MAX_TOKENS,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=CURRENT_SEED,
    )
    parser.add_argument("--max-attempts", type=int, default=1)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-cache", action="store_true")


def _load_json_list(path: str | Path, *, label: str) -> list[Any]:
    source = Path(path).expanduser().resolve()
    if not source.exists():
        raise FileNotFoundError(f"{label} not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{label} must be a JSON array: {source}")
    return payload


def _read_bounded_text(
    path: str | Path,
    *,
    label: str,
    max_bytes: int = 1024 * 1024,
) -> str:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} not found: {source}")
    if source.stat().st_size > max_bytes:
        raise ValueError(f"{label} exceeds {max_bytes} bytes: {source}")
    text = source.read_text(encoding="utf-8")
    if not text.strip():
        raise ValueError(f"{label} must be non-empty: {source}")
    return text


def _runtime_api_key(
    *,
    env_name: str,
    configured_key: str,
    label: str,
) -> str:
    clean_env_name = str(env_name or "").strip()
    if clean_env_name:
        value = str(os.environ.get(clean_env_name, "") or "").strip()
        if value:
            return value
    clean_configured_key = str(configured_key or "").strip()
    if clean_configured_key:
        return clean_configured_key
    raise RuntimeError(
        f"{label} API key is required; set "
        f"{clean_env_name or OPENAI_COMPATIBLE_API_KEY_ENV} "
        "or pass --credentials"
    )


def _apply_credentials(
    args: argparse.Namespace,
    *,
    default_profile: str | None,
) -> None:
    """Apply one local profile after explicit CLI and environment settings."""

    args._credential_api_key = ""
    selected_path = configured_credentials_path(args.credentials)
    if selected_path is None:
        return
    profile = str(args.credential_profile or default_profile or "").strip()
    if not profile:
        raise ValueError(
            "--credential-profile is required when --credentials is used "
            "without a benchmark judge"
        )
    record = load_vlm_credentials(selected_path, profile=profile)
    args.credential_profile = profile
    if not args.base_url and record["base_url"]:
        args.base_url = record["base_url"]
    args._credential_api_key = record["api_key"]


def _require_cli_value(
    value: Any,
    *,
    flag: str,
    mode: str,
) -> str:
    clean = str(value or "").strip()
    if not clean:
        raise ValueError(f"{flag} is required for {mode}")
    return clean


def _reject_cli_values(
    args: argparse.Namespace,
    *,
    flags: tuple[tuple[str, str], ...],
    mode: str,
) -> None:
    invalid = [
        flag
        for attribute, flag in flags
        if (
            bool(getattr(args, attribute))
            if isinstance(getattr(args, attribute), bool)
            else _optional(getattr(args, attribute)) is not None
        )
    ]
    if invalid:
        raise ValueError(", ".join(invalid) + f" are not valid for {mode}")


def _vlm_adapter(args: argparse.Namespace) -> OpenAICompatibleVLMInference:
    catalog_adapter = getattr(args, "_catalog_vlm_adapter", None)
    if catalog_adapter is not None:
        return catalog_adapter
    return OpenAICompatibleVLMInference(
        api_key=_runtime_api_key(
            env_name=args.api_key_env or OPENAI_COMPATIBLE_API_KEY_ENV,
            configured_key=getattr(args, "_credential_api_key", ""),
            label="OpenAI-compatible VLM",
        ),
        model=args.model,
        base_url=args.base_url,
        max_tokens=args.max_tokens,
        seed=args.seed,
        token_limit_parameter=args.token_limit_parameter,
    )


def _adapter_inference_identity(
    adapter: Any,
    args: argparse.Namespace,
) -> Mapping[str, Any]:
    identity_builder = getattr(adapter, "inference_identity", None)
    if callable(identity_builder):
        backend_identity = identity_builder()
        catalog_details = getattr(args, "_catalog_vlm_details", None)
        if isinstance(catalog_details, Mapping):
            return {
                "format": "dream-exe.catalog-vlm-inference",
                "catalog_sha256": catalog_details["catalog_sha256"],
                "catalog_model_id": catalog_details["model_id"],
                "backend": catalog_details["backend"],
                "implementation": copy.deepcopy(
                    catalog_details.get("implementation", {})
                ),
                "backend_identity": backend_identity,
            }
        return backend_identity
    return {
        "format": "dream-exe.cli-vlm-inference",
        "protocol": "injected-test-adapter",
        "model": args.model,
        "request": {
            "token_limit_parameter": args.token_limit_parameter,
            "max_tokens": args.max_tokens,
            "seed": args.seed,
        },
    }


def _prepare_catalog_vlm(args: argparse.Namespace) -> None:
    models_config = str(getattr(args, "models_config", "") or "").strip()
    if not models_config:
        return
    if getattr(args, "_catalog_vlm_adapter", None) is not None:
        return
    conflicts = [
        flag
        for attribute, flag in (
            ("base_url", "--base-url"),
            ("api_key_env", "--api-key-env"),
            ("credentials", "--credentials"),
            ("credential_profile", "--credential-profile"),
            ("backend_label", "--backend-label"),
        )
        if str(getattr(args, attribute, "") or "").strip()
    ]
    if conflicts:
        raise ValueError(
            "catalog VLM mode cannot be mixed with direct connection flags: "
            + ", ".join(conflicts)
        )
    model_id = _require_cli_value(
        args.model,
        flag="--model",
        mode="catalog VLM",
    )
    from ..models import (
        instantiate_vlm_model,
        load_model_catalog,
        resolve_model,
    )

    catalog = load_model_catalog(models_config)
    selected = resolve_model(
        model_id,
        catalog=catalog,
        expected_category="eval",
        expected_kind="vlm",
    )
    experiment_options = None
    if selected.backend == "openai-compatible":
        experiment_options = {
            "max_tokens": args.max_tokens,
            "seed": args.seed,
            "token_limit_parameter": args.token_limit_parameter,
        }
    adapter, details = instantiate_vlm_model(
        selected,
        catalog=catalog,
        experiment_options=experiment_options,
        require_credentials=True,
    )
    args._catalog_vlm_adapter = adapter
    args._catalog_vlm_details = {
        **details,
        "catalog_sha256": catalog.sha256,
    }
    declared_identity = selected.definition.get("identity", {})
    args.backend_label = str(
        declared_identity.get("backend_id", selected.backend)
    )


def _resolve_bench_video_only_inputs(
    args: argparse.Namespace,
) -> tuple[list[dict[str, Any]], Path, Path, Path]:
    """Resolve one candidate and its VLM evaluation contract from a workspace."""

    workspace_path = _require_cli_value(
        args.workspace,
        flag="--workspace",
        mode="bench-driven video_only",
    )
    uid = _require_cli_value(
        args.case,
        flag="--case",
        mode="bench-driven video_only",
    )
    candidate_model = _require_cli_value(
        args.candidate_model,
        flag="--candidate-model",
        mode="bench-driven video_only",
    )
    judge_id = _require_cli_value(
        args.judge,
        flag="--judge",
        mode="bench-driven video_only",
    )
    rubric = _require_cli_value(
        args.rubric,
        flag="--rubric",
        mode="bench-driven video_only",
    )
    workspace = load_workspace(workspace_path)
    repository = BenchRepository(workspace.bench_root)
    config = repository.load_vlm_evaluation_config(
        uid,
        prompt_variant=args.prompt_variant,
    )
    judges = {
        str(judge["id"]): dict(judge)
        for judge in config["judges"]
    }
    if judge_id not in judges:
        raise ValueError(
            f"unknown VLM judge {judge_id!r}; expected one of: "
            + ", ".join(sorted(judges))
        )
    judge = judges[judge_id]
    catalog_mode = bool(str(getattr(args, "models_config", "") or "").strip())
    if catalog_mode:
        _require_cli_value(args.model, flag="--model", mode="catalog VLM")
        _prepare_catalog_vlm(args)
    else:
        if args.model and args.model != judge["model"]:
            raise ValueError("--model conflicts with the selected bench judge")
        args.model = str(judge["model"])
        args.backend_label = str(judge["provider"])
        args.api_key_env = str(judge["api_key_env"])
        if not args.base_url:
            args.base_url = str(
                os.environ.get(str(judge["base_url_env"]), "") or ""
            )
        _apply_credentials(args, default_profile=judge_id)
        if not args.base_url:
            raise RuntimeError(
                f"set {judge['base_url_env']} or configure profile {judge_id!r} "
                "in the VLM credentials file"
            )

    candidates = [
        workspace.video_output_root
        / uid
        / candidate_model
        / args.prompt_variant
        / "video.mp4",
        workspace.published_video_root
        / uid
        / candidate_model
        / args.prompt_variant
        / "video.mp4",
    ]
    explicit_video = _optional(args.video_dir)
    if explicit_video is not None:
        video_path = Path(explicit_video).expanduser().resolve()
    else:
        available = [path for path in candidates if path.is_file()]
        if not available:
            raise FileNotFoundError(
                "candidate video not found in workspace outputs or published results: "
                f"{uid}/{candidate_model}/{args.prompt_variant}/video.mp4"
            )
        video_path = available[0]

    root = (
        workspace.work_root
        / "evaluation"
        / "vlm"
        / uid
        / candidate_model
        / args.prompt_variant
        / (f"{judge_id}--{args.model}" if catalog_mode else judge_id)
        / rubric
    )
    media_root = root / "media"
    output_csv = root / "results.csv"
    root.mkdir(parents=True, exist_ok=True)
    prompt = render_bench_vlm_prompts(config)
    prompt["case_uid"] = uid
    prompt["candidate_model"] = candidate_model
    prompt["prompt_variant"] = args.prompt_variant
    # Prepared media uses the canonical source filename, so its prompt lookup
    # key is "video" rather than the case UID.
    prompt["name"] = video_path.stem
    return [prompt], video_path, media_root, output_csv


def _run_video_only_vlm_evaluation(args: argparse.Namespace) -> int:
    _reject_cli_values(
        args,
        flags=(
            ("trajectory_inputs_json", "--trajectory-inputs-json"),
            ("trajectory_prompt_file", "--trajectory-prompt-file"),
            ("trajectory_prompt_id", "--trajectory-prompt-id"),
            ("trajectory_rubric_id", "--trajectory-rubric-id"),
            ("output_json", "--output-json"),
        ),
        mode="video_only",
    )
    bench_driven = _optional(args.workspace) is not None
    if bench_driven:
        prompts, inferred_video, inferred_grid, inferred_output = (
            _resolve_bench_video_only_inputs(args)
        )
        args.video_dir = inferred_video.as_posix()
        args.grid_output_dir = inferred_grid.as_posix()
        args.output_csv = inferred_output.as_posix()
        if not args.prediction_dir:
            args.prediction_dir = (inferred_output.parent / "predictions").as_posix()
    else:
        if str(args.models_config or "").strip():
            _prepare_catalog_vlm(args)
        else:
            _apply_credentials(args, default_profile=None)
        _require_cli_value(
            args.prompts_json,
            flag="--prompts-json",
            mode="video_only",
        )
        _require_cli_value(
            args.output_csv,
            flag="--output-csv",
            mode="video_only",
        )
        prompts = _load_json_list(
            args.prompts_json,
            label="VLM prompts",
        )
        _require_cli_value(args.model, flag="--model", mode="video_only")
        if not str(args.models_config or "").strip():
            _require_cli_value(args.base_url, flag="--base-url", mode="video_only")
        if not args.backend_label:
            args.backend_label = "openai-compatible"
    _require_cli_value(args.rubric, flag="--rubric", mode="video_only")
    if _optional(args.media_dir) is None and _optional(args.video_dir) is None:
        raise ValueError("--media-dir or --video-dir is required for video_only")
    preparation = None
    video_source = _optional(args.video_dir)
    if video_source is not None:
        grid_output = _optional(args.grid_output_dir)
        if grid_output is None:
            raise ValueError("--grid-output-dir is required with --video-dir")
        preparation = prepare_vlm_media_grids(
            video_dir=video_source,
            output_dir=grid_output,
            rubric=args.rubric,
            reuse_existing=not bool(args.fresh_grids),
        )
        media_records = list(preparation["media_records"])
        media_root = Path(grid_output).expanduser().resolve()
    else:
        if _optional(args.grid_output_dir) is not None:
            raise ValueError("--grid-output-dir is only valid with --video-dir")
        if bool(args.fresh_grids):
            raise ValueError("--fresh-grids is only valid with --video-dir")
        media_root = Path(args.media_dir).expanduser().resolve()
        if not media_root.is_dir():
            raise FileNotFoundError(
                f"saved VLM media directory not found: {media_root}"
            )
        media_records = [
            media_root / name for name in list_grid_image_files(media_root)
        ]
    if not media_records:
        raise RuntimeError(f"no supported saved VLM images found: {media_root}")
    infer = _vlm_adapter(args)
    result = run_saved_media_vlm_batch(
        media_records=media_records,
        prompts=prompts,
        rubric=args.rubric,
        infer=infer,
        backend=args.backend_label,
        model=args.model,
        prompt_id=args.prompt_id,
        output_csv=args.output_csv,
        prediction_dir=_optional(args.prediction_dir),
        max_attempts=args.max_attempts,
        use_cache=not bool(args.no_cache),
        force=args.force,
        inference_identity=_adapter_inference_identity(infer, args),
    )
    summary = {
        key: result.get(key)
        for key in (
            "format",
            "status",
            "rubric",
            "selected_media_count",
            "processed_media_count",
            "report_record_count",
            "item_status_counts",
            "record_status_counts",
            "output_csv",
            "prediction_dir",
            "cache_enabled",
            "cache_disabled_reason",
        )
    }
    if preparation is not None:
        summary["media_preparation"] = {
            key: preparation.get(key)
            for key in (
                "format",
                "status",
                "video_dir",
                "output_dir",
                "prepared_count",
                "skipped_count",
                "error_count",
            )
        }
    _print_json(summary)
    preparation_failed = bool(
        preparation is not None and int(preparation.get("error_count", 0))
    )
    return 0 if result.get("status") == "completed" and not preparation_failed else 1


def _run_video_trajectory_vlm_evaluation(
    args: argparse.Namespace,
) -> int:
    _reject_cli_values(
        args,
        flags=(
            ("media_dir", "--media-dir"),
            ("video_dir", "--video-dir"),
            ("grid_output_dir", "--grid-output-dir"),
            ("fresh_grids", "--fresh-grids"),
            ("prompts_json", "--prompts-json"),
            ("workspace", "--workspace"),
            ("case", "--case"),
            ("candidate_model", "--candidate-model"),
            ("judge", "--judge"),
            ("rubric", "--rubric"),
            ("output_csv", "--output-csv"),
        ),
        mode="video_trajectory",
    )
    inputs_path = _require_cli_value(
        args.trajectory_inputs_json,
        flag="--trajectory-inputs-json",
        mode="video_trajectory",
    )
    prompt_path = _require_cli_value(
        args.trajectory_prompt_file,
        flag="--trajectory-prompt-file",
        mode="video_trajectory",
    )
    prompt_id = _require_cli_value(
        args.trajectory_prompt_id,
        flag="--trajectory-prompt-id",
        mode="video_trajectory",
    )
    rubric_id = _require_cli_value(
        args.trajectory_rubric_id,
        flag="--trajectory-rubric-id",
        mode="video_trajectory",
    )
    output_json = _require_cli_value(
        args.output_json,
        flag="--output-json",
        mode="video_trajectory",
    )
    prediction_dir = _require_cli_value(
        args.prediction_dir,
        flag="--prediction-dir",
        mode="video_trajectory",
    )
    evidence_records = _load_json_list(
        inputs_path,
        label="video+trajectory VLM inputs",
    )
    prompt_template = _read_bounded_text(
        prompt_path,
        label="video+trajectory VLM prompt",
    )
    if str(args.models_config or "").strip():
        _prepare_catalog_vlm(args)
    else:
        _apply_credentials(args, default_profile=None)
    _require_cli_value(args.model, flag="--model", mode="video_trajectory")
    if not str(args.models_config or "").strip():
        _require_cli_value(args.base_url, flag="--base-url", mode="video_trajectory")
    if not args.backend_label:
        args.backend_label = "openai-compatible"
    infer = _vlm_adapter(args)
    result = run_saved_media_trajectory_vlm_batch(
        evidence_records=evidence_records,
        infer=infer,
        prediction_dir=prediction_dir,
        rubric_id=rubric_id,
        prompt_template=prompt_template,
        prompt_id=prompt_id,
        parser_id=args.trajectory_parser_id,
        judge_id="none",
        backend=args.backend_label,
        model=args.model,
        inference_identity=_adapter_inference_identity(infer, args),
        max_attempts=args.max_attempts,
        use_cache=not bool(args.no_cache),
        force=bool(args.force),
        max_context_points=args.trajectory_max_context_points,
        output_json=output_json,
    )
    _print_json(
        {
            key: result.get(key)
            for key in (
                "format",
                "mode",
                "status",
                "selected_evidence_count",
                "processed_evidence_count",
                "item_status_counts",
                "prediction_dir",
                "cache_enabled",
                "output_json",
                "evaluation_contract",
            )
        }
    )
    return 0 if result.get("status") == "completed" else 1


def _run_vlm_evaluation(args: argparse.Namespace) -> int:
    if str(args.mode) == "video_trajectory":
        return _run_video_trajectory_vlm_evaluation(args)
    return _run_video_only_vlm_evaluation(args)


def _run_exec_evaluation(args: argparse.Namespace) -> int:
    metrics = build_metrics_from_exec_dir(Path(args.exec_dir))
    _print_json(metrics)
    return 0 if metrics else 1


def _run_trajectory_evaluation(args: argparse.Namespace) -> int:
    kind = str(args.kind)
    exec_dir = _optional(args.exec_dir)
    predicted_path = _optional(args.predicted_path)
    reference_path = _optional(args.reference_path)
    group = _optional(args.group)
    if kind == "path-comparison":
        if exec_dir is not None:
            raise ValueError("--exec-dir is not valid for path-comparison")
        if group is not None:
            raise ValueError("--group is only valid for similarity")
        if bool(args.publish):
            raise ValueError("--publish is only valid for executability")
        missing = [
            flag
            for flag, value in (
                ("--predicted-path", predicted_path),
                ("--reference-path", reference_path),
            )
            if value is None
        ]
        if missing:
            raise ValueError("path-comparison requires explicit " + ", ".join(missing))
        report = evaluate_trajectory_path_comparison_paths(
            predicted_path,
            reference_path,
        )
    elif kind == "executability":
        if exec_dir is None:
            raise ValueError("--exec-dir is required for executability")
        if group is not None:
            raise ValueError("--group is only valid for similarity")
        if any(value is not None for value in (predicted_path, reference_path)):
            raise ValueError(
                "--predicted-path and --reference-path are only valid for "
                "similarity or path-comparison"
            )
        report = evaluate_trajectory_executability_dir(
            exec_dir,
            publish=bool(args.publish),
        )
    else:
        if exec_dir is not None:
            raise ValueError("--exec-dir is only valid for executability")
        if bool(args.publish):
            raise ValueError("--publish is only valid for executability")
        missing = [
            flag
            for flag, value in (
                ("--predicted-path", predicted_path),
                ("--reference-path", reference_path),
                ("--group", group),
            )
            if value is None
        ]
        if missing:
            raise ValueError("similarity requires explicit " + ", ".join(missing))
        normalization_overrides = {
            name: value
            for name, value in (
                ("HSD", args.hsd_max),
                ("DYN", args.dyn_max),
                ("NDTW", args.ndtw_max),
            )
            if value is not None
        }
        report = evaluate_trajectory_similarity_files(
            predicted_path,
            reference_path,
            group=group,
            protocol=args.protocol,
            visibility_threshold=args.visibility_threshold,
            fps=args.fps,
            normalization_overrides=normalization_overrides,
        )
    _print_json(report)
    return 0 if report else 1


def _run_task_success_rate_evaluation(args: argparse.Namespace) -> int:
    specs = _load_json_list(
        args.inputs_json,
        label="task success-rate input specs",
    )
    report = aggregate_task_success_paths(
        specs,
        by_level=bool(args.by_level),
        include_records=bool(args.include_records),
    )
    _print_json(report)
    return 0 if int(report.get("rows_total", 0)) > 0 else 1


def _visual_case_namespace(
    args: argparse.Namespace,
    selection: EvaluationSelection,
    *,
    uid: str,
    judge: str,
    rubric: str,
) -> argparse.Namespace:
    return argparse.Namespace(
        workspace=selection.workspace.config_path.as_posix(),
        case=uid,
        candidate_model=selection.candidate_model,
        prompt_variant=selection.prompt_variant,
        judge=judge,
        rubric=rubric,
        model="",
        models_config="",
        backend_label="",
        api_key_env="",
        base_url="",
        credentials=args.credentials,
        credential_profile="",
        video_dir="",
        media_dir="",
        grid_output_dir="",
        fresh_grids=bool(args.fresh_grids),
        output_csv="",
        prediction_dir="",
        prompt_id="paper",
        token_limit_parameter="max_completion_tokens",
        max_tokens=CURRENT_MAX_TOKENS,
        seed=CURRENT_SEED,
        max_attempts=int(args.max_attempts),
        force=bool(args.force),
        no_cache=bool(args.no_cache),
    )


def _run_visual_case(
    args: argparse.Namespace,
    selection: EvaluationSelection,
    *,
    uid: str,
    judge: str,
    rubric: str,
) -> dict[str, Any]:
    local = _visual_case_namespace(
        args,
        selection,
        uid=uid,
        judge=judge,
        rubric=rubric,
    )
    prompts, video_path, media_root, output_csv = _resolve_bench_video_only_inputs(local)
    local.video_dir = video_path.as_posix()
    local.grid_output_dir = media_root.as_posix()
    local.output_csv = output_csv.as_posix()
    local.prediction_dir = (output_csv.parent / "predictions").as_posix()
    preparation = prepare_vlm_media_grids(
        video_dir=video_path,
        output_dir=media_root,
        rubric=rubric,
        reuse_existing=not bool(local.fresh_grids),
    )
    media_records = list(preparation["media_records"])
    if not media_records:
        raise RuntimeError("VLM media preparation produced no supported images")
    infer = _vlm_adapter(local)
    report = run_saved_media_vlm_batch(
        media_records=media_records,
        prompts=prompts,
        rubric=rubric,
        infer=infer,
        backend=local.backend_label,
        model=local.model,
        prompt_id=local.prompt_id,
        output_csv=output_csv,
        prediction_dir=local.prediction_dir,
        max_attempts=local.max_attempts,
        use_cache=not bool(local.no_cache),
        force=bool(local.force),
        inference_identity=_adapter_inference_identity(infer, local),
    )
    report["uid"] = uid
    report["judge"] = judge
    report["candidate"] = {
        "model": selection.candidate_model,
        "prompt_variant": selection.prompt_variant,
    }
    report["media_preparation"] = {
        "status": preparation.get("status"),
        "prepared_count": preparation.get("prepared_count"),
        "skipped_count": preparation.get("skipped_count"),
        "error_count": preparation.get("error_count"),
    }
    write_json_atomic(
        selection.output_root
        / "cases"
        / uid
        / "visual"
        / judge
        / rubric
        / "report.json",
        report,
    )
    return report


def _combined_judge_report(
    reports: Mapping[str, Mapping[str, Any]],
    *,
    rubric: str,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for uid, report in sorted(reports.items()):
        raw_records = report.get("records")
        if not isinstance(raw_records, list):
            continue
        for raw in raw_records:
            if not isinstance(raw, Mapping):
                continue
            record = copy.deepcopy(dict(raw))
            record["name"] = f"{uid}/{str(record.get('name', 'video'))}"
            records.append(record)
    return {"rubric": rubric, "records": records}


def _visual_mean_report(
    judge_reports: Mapping[str, Mapping[str, Any]],
    *,
    required_judges: list[str],
    expected_items: int,
) -> dict[str, Any]:
    aggregate = aggregate_vlm_judges(
        judge_reports,
        required_judges=required_judges,
    )
    values = [
        float(item["mean_score"])
        for item in aggregate["items"]
        if item.get("mean_score") is not None
    ]
    aggregate["expected_items"] = expected_items
    aggregate["evaluated_items"] = len(values)
    aggregate["coverage"] = len(values) / expected_items if expected_items else 0.0
    aggregate["mean_score"] = sum(values) / len(values) if values else None
    aggregate["status"] = (
        "completed" if len(values) == expected_items else "not_evaluated"
    )
    return aggregate


def _run_visual_selection(
    args: argparse.Namespace,
    selection: EvaluationSelection,
) -> dict[str, Any]:
    protocol = selection.repository.load_protocol("evaluation")["values"]["vlm"]
    declared_judges = [str(item["id"]) for item in protocol["judges"]]
    requested_judge = str(args.judge)
    if requested_judge == "both":
        judges = [str(item) for item in protocol["aggregation"]["required_judges"]]
    elif requested_judge in declared_judges:
        judges = [requested_judge]
    else:
        raise ValueError(
            f"unknown judge {requested_judge!r}; expected one of "
            + ", ".join((*declared_judges, "both"))
        )
    rubrics = (
        list(PAPER_VISUAL_RUBRICS)
        if str(args.rubric) == "all"
        else [str(args.rubric)]
    )
    reports: dict[str, dict[str, dict[str, Any]]] = {
        judge: {rubric: {} for rubric in rubrics} for judge in judges
    }
    case_status: list[dict[str, Any]] = []
    for uid in selection.uids:
        status = "completed"
        for rubric in rubrics:
            by_judge: dict[str, Mapping[str, Any]] = {}
            for judge in judges:
                try:
                    report = _run_visual_case(
                        args,
                        selection,
                        uid=uid,
                        judge=judge,
                        rubric=rubric,
                    )
                    reports[judge][rubric][uid] = report
                    by_judge[judge] = report
                except Exception as error:
                    status = "not_evaluated"
                    failure = {
                        "format": "dream-exe.evaluation-case",
                        "family": "visual",
                        "status": "not_evaluated",
                        "uid": uid,
                        "judge": judge,
                        "rubric": rubric,
                        "error": {
                            "type": type(error).__name__,
                            "message": portable_error_message(error, selection.workspace),
                        },
                    }
                    write_json_atomic(
                        selection.output_root
                        / "cases"
                        / uid
                        / "visual"
                        / judge
                        / rubric
                        / "report.json",
                        failure,
                    )
            if len(by_judge) == len(judges):
                try:
                    case_aggregate = _visual_mean_report(
                        by_judge,
                        required_judges=judges,
                        expected_items=1,
                    )
                except Exception as error:
                    status = "not_evaluated"
                    case_aggregate = {
                        "format": "dream-exe.vlm-judge-aggregate",
                        "status": "not_evaluated",
                        "rubric": rubric,
                        "error": {
                            "type": type(error).__name__,
                            "message": portable_error_message(error, selection.workspace),
                        },
                    }
                if case_aggregate.get("status") != "completed":
                    status = "not_evaluated"
                write_json_atomic(
                    selection.output_root
                    / "cases"
                    / uid
                    / "visual"
                    / f"{rubric}.json",
                    case_aggregate,
                )
            else:
                write_json_atomic(
                    selection.output_root
                    / "cases"
                    / uid
                    / "visual"
                    / f"{rubric}.json",
                    {
                        "format": "dream-exe.vlm-judge-aggregate",
                        "status": "not_evaluated",
                        "rubric": rubric,
                        "required_judges": judges,
                        "expected_items": 1,
                        "evaluated_items": 0,
                        "coverage": 0.0,
                        "mean_score": None,
                    },
                )
        case_status.append({"uid": uid, "status": status})

    rubric_aggregates: dict[str, Any] = {}
    for rubric in rubrics:
        combined = {
            judge: _combined_judge_report(reports[judge][rubric], rubric=rubric)
            for judge in judges
        }
        try:
            aggregate = _visual_mean_report(
                combined,
                required_judges=judges,
                expected_items=len(selection.uids),
            )
        except Exception as error:
            aggregate = {
                "format": "dream-exe.vlm-judge-aggregate",
                "status": "not_evaluated",
                "rubric": rubric,
                "required_judges": judges,
                "expected_items": len(selection.uids),
                "evaluated_items": 0,
                "coverage": 0.0,
                "mean_score": None,
                "error": {
                    "type": type(error).__name__,
                    "message": portable_error_message(error, selection.workspace),
                },
            }
        write_json_atomic(
            selection.output_root / "aggregate" / "visual" / f"{rubric}.json",
            aggregate,
        )
        rubric_aggregates[rubric] = aggregate
    summary = {
        "format": "dream-exe.evaluation-summary",
        "family": "visual",
        "status": (
            "completed"
            if all(item.get("status") == "completed" for item in rubric_aggregates.values())
            else "not_evaluated"
        ),
        "candidate": {
            "model": selection.candidate_model,
            "prompt_variant": selection.prompt_variant,
        },
        "judges": judges,
        "judge_aggregation": str(protocol["aggregation"]["method"]),
        "expected_cases": len(selection.uids),
        "rubrics": rubric_aggregates,
        "case_status": case_status,
    }
    summary_path = write_json_atomic(
        selection.output_root / "aggregate" / "visual" / "summary.json",
        summary,
    )
    return {"aggregate": summary, "aggregate_path": summary_path, "cases": case_status}


def _run_evaluation_suite(args: argparse.Namespace) -> int:
    selection = resolve_evaluation_selection(
        workspace_path=args.workspace,
        scope=args.scope,
        case=args.case,
        candidate_model=args.candidate_model,
        prompt_variant=args.prompt_variant,
    )
    family = str(args.evaluation_family)
    results: dict[str, Any] = {}
    if family in {"trajectory", "all"}:
        results["trajectory"] = evaluate_trajectory_selection(
            selection,
            protocol=args.trajectory_protocol,
        )
    if family in {"executability", "all"}:
        results["executability"] = evaluate_executability_selection(selection)
    if family in {"task", "all"}:
        results["task"] = evaluate_task_selection(selection)
    if family in {"visual", "all"}:
        results["visual"] = _run_visual_selection(args, selection)
    output = {
        "aggregate": {
            name: value["aggregate"] for name, value in results.items()
        },
        "selection": selection_summary(selection),
        "aggregate_paths": {
            name: portable_workspace_path(
                Path(value["aggregate_path"]), selection.workspace
            )
            for name, value in results.items()
        },
    }
    _print_json(output)
    return 0 if all(
        all(item.get("status") == "completed" for item in value["cases"])
        for value in results.values()
    ) else 1


def register_exec_evaluation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--exec-dir", required=True)
    parser.set_defaults(_handler=_run_exec_evaluation)


def register_trajectory_evaluation(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--kind",
        required=True,
        choices=("similarity", "executability", "path-comparison"),
    )
    parser.add_argument("--exec-dir", default="")
    parser.add_argument("--predicted-path", default="")
    parser.add_argument("--reference-path", default="")
    parser.add_argument(
        "--group",
        default="",
        choices=("", *TRAJECTORY_SIMILARITY_GROUPS),
    )
    parser.add_argument(
        "--protocol",
        choices=TRAJECTORY_SIMILARITY_PROTOCOLS,
        default=DEFAULT_TRAJECTORY_SIMILARITY_PROTOCOL,
    )
    parser.add_argument("--visibility-threshold", type=float, default=0.1)
    parser.add_argument("--fps", type=float, default=1.0)
    parser.add_argument("--hsd-max", type=float, default=None)
    parser.add_argument("--dyn-max", type=float, default=None)
    parser.add_argument("--ndtw-max", type=float, default=None)
    parser.add_argument(
        "--publish",
        action="store_true",
        help=(
            "For executability only, explicitly refresh current-compatible "
            "exec metric artifacts instead of evaluating read-only."
        ),
    )
    parser.set_defaults(_handler=_run_trajectory_evaluation)


def register_task_success_rate_evaluation(
    parser: argparse.ArgumentParser,
) -> None:
    parser.add_argument(
        "--inputs-json",
        required=True,
        help=(
            "Explicit JSON array of {uid, path, level?, metadata?} specs. "
            "No bench paths are inferred."
        ),
    )
    parser.add_argument("--by-level", action="store_true")
    parser.add_argument("--include-records", action="store_true")
    parser.set_defaults(_handler=_run_task_success_rate_evaluation)


def register_vlm_evaluation(parser: argparse.ArgumentParser) -> None:
    _add_vlm_evaluation_arguments(parser)
    parser.set_defaults(_handler=_run_vlm_evaluation)
