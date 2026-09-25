"""Canonical benchmark public CLI commands."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
from pathlib import Path

from ..bench.contracts.schemas import (
    RUN_SCHEMA,
    VIDEO_OUTPUT_SCHEMA,
    canonical_sha256,
    load_and_validate,
    validate_document,
)
from ..bench.data.repository import BenchRepository
from ..bench.data.workspace import Workspace, load_workspace
from ..bench.outputs.lifecycle import (
    discover_run_roots,
    register_video_output,
    write_json_atomic,
)
from ..bench.outputs.results import ResultRepository
from ..bench.runtime import (
    build_default_runtime_config,
    initialize_case,
    run_benchmark,
    verify_historical_compatibility_manifest,
)
from ..bench.setup import configure_workspace, inspect_workspace
from ..bench.videos.generation import generate_video_output
from ..bench.videos.import_video import import_external_video
from ._common import _print_json


def _auto_receipt_id(uid: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{uid}-{stamp}"


def _benchmark_exit_code(report: object) -> int:
    """Return success only for a fully completed benchmark report."""

    if isinstance(report, dict) and report.get("status") == "completed":
        return 0
    return 1


def run_init(args: argparse.Namespace) -> int:
    receipt_id = str(args.receipt_id or "").strip() or _auto_receipt_id(args.case)
    result = initialize_case(
        workspace_path=args.workspace,
        uid=args.case,
        receipt_id=receipt_id,
    )
    _print_json(result)
    return 0


def _run(args: argparse.Namespace) -> int:
    if not args.models_config and not args.runtime_config:
        runtime_builder = build_default_runtime_config
    else:
        from ..models import compose_runtime_config, load_model_catalog

        catalog = (
            load_model_catalog(args.models_config)
            if str(args.models_config or "").strip()
            else None
        )

        def runtime_builder(workspace: object):
            composition = compose_runtime_config(
                defaults=build_default_runtime_config(workspace),
                catalog=catalog,
                runtime_config_path=args.runtime_config or None,
                preflight_factories=True,
            )
            return composition.runtime_config

    report = run_benchmark(
        workspace_path=args.workspace,
        run_spec_path=args.spec,
        runtime_config_builder=runtime_builder,
    )
    _print_json(report)
    return _benchmark_exit_code(report)


def _configure(args: argparse.Namespace) -> int:
    _print_json(
        configure_workspace(
            workspace_path=args.workspace,
            create_directories=not args.check,
        )
    )
    return 0


def _doctor(args: argparse.Namespace) -> int:
    case = args.case
    if not case:
        workspace = load_workspace(args.workspace)
        collection = BenchRepository(workspace.bench_root).load_collection()
        cases = list(collection.get("cases", []))
        if not cases:
            raise ValueError("configured benchmark collection has no cases")
        case = str(cases[0]["uid"])
    _print_json(
        inspect_workspace(
            workspace_path=args.workspace,
            uid=case,
            verify_bindings=not args.bench_only,
        )
    )
    return 0


def register_configure(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--workspace",
        default="configs/workspace.json",
        help=(
            "Existing path configuration "
            "(default: configs/workspace.json)."
        ),
    )
    parser.add_argument(
        "--check",
        action="store_true",
        help="Validate configured paths without creating local root directories.",
    )
    parser.set_defaults(_handler=_configure)


def register_doctor(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--case",
        default=None,
        help="Case to inspect (default: first case in the workspace collection).",
    )
    parser.add_argument(
        "--bench-only",
        action="store_true",
        help="Validate benchmark assets without checking external runtimes.",
    )
    parser.set_defaults(_handler=_doctor)


def _generate(args: argparse.Namespace) -> int:
    workspace = load_workspace(args.workspace)
    backend = None
    backend_name = "wan2.2_ti2v_5b_official_cli"
    catalog_parameters: dict[str, object] = {}
    catalog_mode = bool(str(args.models_config or "").strip())
    if catalog_mode:
        if args.source_binding or args.checkpoint_binding:
            raise ValueError(
                "--models-config cannot be mixed with --source-binding or "
                "--checkpoint-binding"
            )
        from ..models import (
            instantiate_video_generation_model,
            load_model_catalog,
            resolve_model,
        )

        catalog = load_model_catalog(args.models_config)
        selected = resolve_model(
            args.model,
            catalog=catalog,
            expected_category="video_gen",
            expected_kind="video_generation",
        )
        backend_name = (
            str(selected.definition.get("identity", {}).get("backend_id", ""))
            or selected.backend
        )
        if args.dry_run:
            if selected.backend == "factory":
                catalog_parameters = dict(selected.definition.get("options", {}))
            else:
                catalog_parameters = dict(
                    selected.definition.get("options", {}).get("parameters", {})
                )
        else:
            backend, _details, catalog_parameters = (
                instantiate_video_generation_model(
                    selected,
                    catalog=catalog,
                )
            )
    elif not args.dry_run:
        from ..generation.providers.wan22 import Wan22TI2VBackend

        source = workspace.binding("sources", args.source_binding or "wan22").path
        checkpoint = workspace.binding(
            "checkpoints",
            args.checkpoint_binding or "wan22",
        ).path
        backend = Wan22TI2VBackend(
            source_root=source,
            checkpoint_root=checkpoint,
        )
    cli_parameters = {
        key: value
        for key, value in {
            "size": args.size,
            "frame_num": args.frame_num,
            "sample_steps": args.sample_steps,
            "guidance_scale": args.guidance_scale,
        }.items()
        if value is not None
    }
    parameters = {**catalog_parameters, **cli_parameters}
    if not catalog_mode:
        parameters = {
            "size": args.size or "1280*704",
            "frame_num": 121 if args.frame_num is None else args.frame_num,
            "sample_steps": 40 if args.sample_steps is None else args.sample_steps,
            "guidance_scale": (
                5.0 if args.guidance_scale is None else args.guidance_scale
            ),
        }
    generated = generate_video_output(
        workspace=workspace,
        uid=args.case,
        model_id=args.model,
        prompt_variant=args.variant,
        backend=backend,
        backend_name=backend_name,
        producer_revision=args.producer_revision,
        seed=args.seed,
        parameters=parameters,
        force_work=args.force_work,
        dry_run=args.dry_run,
    )
    if args.dry_run:
        _print_json(generated)
        return 0
    run_id = args.run_id or f"{args.case}-{args.model}-{args.variant}"
    document = _run_document(
        cases=[args.case],
        collection=None,
        model=args.model,
        variant=args.variant,
        run_id=run_id,
        artifact_level=args.artifact_level,
    )
    spec = _write_import_spec(workspace, document, args.run_spec)
    result = {**generated, "run_spec": spec.as_posix()}
    if args.run:
        result["run"] = run_benchmark(
            workspace_path=args.workspace,
            run_spec_path=spec,
        )
    _print_json(result)
    return _benchmark_exit_code(result["run"]) if args.run else 0


def register_generate(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True, help="Strict workspace JSON.")
    parser.add_argument("--case", required=True)
    parser.add_argument("--model", default="Wan2.2")
    parser.add_argument(
        "--models-config",
        default="",
        help=(
            "Optional trusted dream-exe.models catalog. With this flag, "
            "--model is an exact video_gen/video_generation model ID."
        ),
    )
    parser.add_argument(
        "--variant",
        required=True,
        choices=("standard", "enhanced"),
    )
    parser.add_argument("--source-binding", default="")
    parser.add_argument("--checkpoint-binding", default="")
    parser.add_argument("--producer-revision", default="")
    parser.add_argument("--size", default=None)
    parser.add_argument("--frame-num", type=int, default=None)
    parser.add_argument("--sample-steps", type=int, default=None)
    parser.add_argument("--guidance-scale", type=float, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--force-work",
        action="store_true",
        help="Replace only a matching work-local generation pair; results stay immutable.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--run-id", default="")
    parser.add_argument("--run-spec", default="")
    parser.add_argument(
        "--artifact-level",
        choices=("core", "full"),
        default="core",
    )
    parser.add_argument(
        "--run",
        action="store_true",
        help="Run the full evaluation pipeline after video generation.",
    )
    parser.set_defaults(_handler=_generate)


def register_run(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True, help="Strict workspace JSON.")
    parser.add_argument("--spec", required=True, help="Strict run JSON.")
    parser.add_argument(
        "--models-config",
        default="",
        help="Optional trusted Dream.exe model catalog.",
    )
    parser.add_argument(
        "--runtime-config",
        default="",
        help=(
            "Optional Dream.exe model composition or existing complete "
            "video2traj runtime JSON."
        ),
    )
    parser.set_defaults(_handler=_run)


def _reproduce(args: argparse.Namespace) -> int:
    workspace = load_workspace(args.workspace)
    discovered = discover_run_roots(workspace.published_results_root)
    selected = list(args.run_id) if args.run_id else sorted(discovered)
    missing = sorted(set(selected) - set(discovered))
    if missing:
        raise KeyError(f"unknown published run IDs: {', '.join(missing)}")
    if args.dry_run:
        _print_json(
            {
                "format": "dream-exe.reproduction-plan",
                "status": "ready",
                "runs": [
                    {
                        "run_id": run_id,
                        "spec": (discovered[run_id] / "run.json").as_posix(),
                    }
                    for run_id in selected
                ],
            }
        )
        return 0
    reports = []
    for run_id in selected:
        report = run_benchmark(
            workspace_path=args.workspace,
            run_spec_path=discovered[run_id] / "run.json",
        )
        reports.append(
            {
                "run_id": run_id,
                "status": report["status"],
                "summary": report["summary"],
            }
        )
        if report["status"] != "completed" and args.stop_on_failure:
            break
    result = {
        "format": "dream-exe.reproduction-report",
        "status": (
            "completed"
            if len(reports) == len(selected)
            and all(item["status"] == "completed" for item in reports)
            else "partial"
        ),
        "runs": reports,
    }
    _print_json(result)
    return _benchmark_exit_code(result)


def register_reproduce(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True)
    parser.add_argument(
        "--run-id",
        action="append",
        default=[],
        help="Run only this published experiment ID; repeat as needed.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--stop-on-failure", action="store_true")
    parser.set_defaults(_handler=_reproduce)


def _parse_sources(values: list[str]) -> dict[str, Path]:
    output: dict[str, Path] = {}
    for value in values:
        if "=" not in value:
            raise ValueError(
                "--source must use video=/absolute/path or "
                "preprocessed=/absolute/path"
            )
        name, raw_path = value.split("=", 1)
        name = name.strip()
        if not name or name in output:
            raise ValueError(f"duplicate or empty video source role: {name!r}")
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            raise ValueError("--source paths must be absolute")
        output[name] = path.resolve()
    return output


def _register_video(args: argparse.Namespace) -> int:
    workspace = load_workspace(args.workspace)
    manifest = load_and_validate(args.manifest, expected_schema=VIDEO_OUTPUT_SCHEMA)
    _print_json(
        register_video_output(
            workspace=workspace,
            manifest=manifest,
            source_paths=_parse_sources(args.source),
        )
    )
    return 0


def _run_document(
    *,
    cases: list[str],
    collection: str | None,
    model: str,
    variant: str,
    run_id: str,
    artifact_level: str,
) -> dict:
    return validate_document(
        {
            "format": RUN_SCHEMA,
            "selection": {"collection": collection, "cases": cases},
            "inputs": [
                {
                    "kind": "generated",
                    "model_id": model,
                    "prompt_variant": variant,
                    "reference_id": None,
                }
            ],
            "stages": ["video2traj", "action", "execution", "evaluation"],
            "seed": 42,
            "initialization": {"mode": "frozen", "receipt": None},
            "resume": {"enabled": True, "exact": True},
            "retry": {"max_attempts": 1, "failure_policy": "continue"},
            "artifact_level": artifact_level,
            "destination": {"run_id": run_id},
        },
        expected_schema=RUN_SCHEMA,
    )


def _write_import_spec(
    workspace: Workspace,
    document: dict,
    requested: str,
) -> Path:
    if requested:
        destination = Path(requested).expanduser().resolve()
    else:
        run_id = str(document["destination"]["run_id"])
        destination = workspace.work_root / "imports" / run_id / "run.json"
    if destination.is_file():
        existing = load_and_validate(destination, expected_schema=RUN_SCHEMA)
        if canonical_sha256(existing) != canonical_sha256(document):
            raise FileExistsError(f"run spec already exists with different content: {destination}")
    else:
        write_json_atomic(destination, document, exclusive=True)
    return destination


def _import_video(args: argparse.Namespace) -> int:
    workspace = load_workspace(args.workspace)
    run_id = args.run_id or f"{args.case}-{args.model}-{args.variant}"
    document = _run_document(
        cases=[args.case],
        collection=None,
        model=args.model,
        variant=args.variant,
        run_id=run_id,
        artifact_level=args.artifact_level,
    )
    imported = import_external_video(
        workspace=workspace,
        uid=args.case,
        model_id=args.model,
        prompt_variant=args.variant,
        video_path=args.video,
        preprocessed_path=args.preprocessed or None,
        transfer="move" if args.move else "copy",
        producer_kind=args.producer_kind,
        producer_name=args.producer_name,
        producer_revision=args.producer_revision,
        seed=args.seed,
        custom_prompt=args.prompt,
    )
    spec = _write_import_spec(workspace, document, args.run_spec)
    result = {**imported, "run_spec": spec.as_posix()}
    if args.run:
        result["run"] = run_benchmark(
            workspace_path=args.workspace,
            run_spec_path=spec,
        )
    _print_json(result)
    return _benchmark_exit_code(result["run"]) if args.run else 0


def _import_videos(args: argparse.Namespace) -> int:
    workspace = load_workspace(args.workspace)
    repository = BenchRepository(workspace.bench_root)
    collection = repository.load_collection()
    collection_id = str(collection["collection_id"])
    uids = [str(item["uid"]) for item in collection["cases"]]
    video_root = Path(args.video_dir).expanduser().resolve()
    if "{uid}" not in args.pattern:
        raise ValueError("--pattern must contain {uid}")
    run_id = args.run_id or f"{collection_id}-{args.model}-{args.variant}"
    document = _run_document(
        cases=[],
        collection=collection_id,
        model=args.model,
        variant=args.variant,
        run_id=run_id,
        artifact_level=args.artifact_level,
    )
    sources: list[tuple[str, Path]] = []
    for uid in uids:
        source = (video_root / args.pattern.format(uid=uid)).resolve()
        try:
            source.relative_to(video_root)
        except ValueError as error:
            raise ValueError("--pattern must stay inside --video-dir") from error
        if source.is_symlink() or not source.is_file() or source.suffix.lower() != ".mp4":
            raise FileNotFoundError(f"generated MP4 not found for {uid}: {source}")
        destination = workspace.video_output_root / uid / args.model / args.variant
        if destination.exists():
            raise FileExistsError(f"video output already exists: {destination}")
        sources.append((uid, source))

    imported = []
    for uid, source in sources:
        imported.append(
            import_external_video(
                workspace=workspace,
                uid=uid,
                model_id=args.model,
                prompt_variant=args.variant,
                video_path=source,
                transfer="move" if args.move else "copy",
                producer_kind=args.producer_kind,
                producer_name=args.producer_name,
                producer_revision=args.producer_revision,
                seed=args.seed,
            )
        )
    spec = _write_import_spec(workspace, document, args.run_spec)
    result = {
        "format": "dream-exe.video-batch-import",
        "status": "imported",
        "collection": collection_id,
        "videos": len(imported),
        "run_spec": spec.as_posix(),
    }
    if args.run:
        result["run"] = run_benchmark(
            workspace_path=args.workspace,
            run_spec_path=spec,
        )
    _print_json(result)
    return _benchmark_exit_code(result["run"]) if args.run else 0


def _aggregate_results(args: argparse.Namespace) -> int:
    workspace = load_workspace(args.workspace)
    report = ResultRepository(workspace.outputs_root).aggregate_run(
        run_id=args.run_id,
        expected_uids=None,
        require_vlm=args.require_vlm,
        verify_artifacts=not args.skip_artifact_hashes,
    )
    _print_json(report)
    return 0


def _verify_compatibility_input(args: argparse.Namespace) -> int:
    workspace = load_workspace(args.workspace)
    repository = BenchRepository(workspace.bench_root)
    collection = repository.load_collection(args.collection)
    receipt = verify_historical_compatibility_manifest(
        manifest_path=args.manifest,
        artifact_root=args.artifact_root,
        repository=repository,
        expected_uids=[str(item["uid"]) for item in collection["cases"]],
    )
    if receipt["collection_id"] != collection["collection_id"]:
        raise ValueError(
            "compatibility manifest collection differs from the selected "
            "benchmark collection"
        )
    _print_json(receipt)
    return 0


def register_bench(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="bench_command", required=True)
    promote = commands.add_parser(
        "register-video",
        help="Move a validated work video into outputs/videos by model and prompt variant.",
    )
    promote.add_argument("--workspace", required=True)
    promote.add_argument("--manifest", required=True)
    promote.add_argument(
        "--source",
        action="append",
        default=[],
        required=True,
        help=(
            "Use video=/absolute/work/path and optional "
            "preprocessed=/absolute/work/path."
        ),
    )
    promote.set_defaults(_handler=_register_video)

    import_video = commands.add_parser(
        "import-video",
        help=(
            "Import an external model video and emit the exact run input; "
            "preprocessing is automatic and the original is preserved unless "
            "--move is explicit."
        ),
    )
    import_video.add_argument("--workspace", required=True)
    import_video.add_argument("--case", required=True)
    import_video.add_argument("--model", required=True)
    import_video.add_argument(
        "--variant",
        required=True,
        choices=("standard", "enhanced", "custom"),
    )
    import_video.add_argument(
        "--video",
        required=True,
        help="External generated MP4.",
    )
    import_video.add_argument(
        "--preprocessed",
        default="",
        help=(
            "Advanced override: a distinct MP4 already matching the benchmark "
            "video2traj contract. Normally omit; preprocessing is automatic."
        ),
    )
    import_video.add_argument(
        "--move",
        action="store_true",
        help=(
            "Use same-device rename and remove the caller's original after "
            "successful registration; the default copies safely."
        ),
    )
    import_video.add_argument(
        "--producer-kind",
        choices=("generator", "policy_rollout", "user", "unknown"),
        default="generator",
    )
    import_video.add_argument("--producer-name", default="")
    import_video.add_argument("--producer-revision", default="")
    import_video.add_argument("--seed", type=int, default=None)
    import_video.add_argument(
        "--prompt",
        default="",
        help="Prompt text for variant=custom; only its digest is stored.",
    )
    import_video.add_argument("--run-id", default="")
    import_video.add_argument("--run-spec", default="")
    import_video.add_argument(
        "--artifact-level",
        choices=("core", "full"),
        default="core",
    )
    import_video.add_argument(
        "--run",
        action="store_true",
        help="Run video2traj, action, execution, and evaluation after import.",
    )
    import_video.set_defaults(_handler=_import_video)

    import_videos = commands.add_parser(
        "import-videos",
        help=(
            "Import one <uid>.mp4 for every collection case, emit a full-run "
            "spec, and optionally run it."
        ),
    )
    import_videos.add_argument("--workspace", required=True)
    import_videos.add_argument("--model", required=True)
    import_videos.add_argument(
        "--variant",
        required=True,
        choices=("standard", "enhanced"),
    )
    import_videos.add_argument("--video-dir", required=True)
    import_videos.add_argument(
        "--pattern",
        default="{uid}.mp4",
        help="Relative filename pattern below --video-dir; {uid} is required.",
    )
    import_videos.add_argument("--move", action="store_true")
    import_videos.add_argument(
        "--producer-kind",
        choices=("generator", "policy_rollout", "user", "unknown"),
        default="generator",
    )
    import_videos.add_argument("--producer-name", default="")
    import_videos.add_argument("--producer-revision", default="")
    import_videos.add_argument("--seed", type=int, default=None)
    import_videos.add_argument("--run-id", default="")
    import_videos.add_argument("--run-spec", default="")
    import_videos.add_argument(
        "--artifact-level",
        choices=("core", "full"),
        default="core",
    )
    import_videos.add_argument("--run", action="store_true")
    import_videos.set_defaults(_handler=_import_videos)

    aggregate = commands.add_parser(
        "aggregate-results",
        help="Re-read immutable result bundles and aggregate a closed cohort.",
    )
    aggregate.add_argument("--workspace", required=True)
    aggregate.add_argument("--run-id", required=True)
    aggregate.add_argument("--require-vlm", action="store_true")
    aggregate.add_argument("--skip-artifact-hashes", action="store_true")
    aggregate.set_defaults(_handler=_aggregate_results)

    compatibility = commands.add_parser(
        "verify-compatibility-input",
        help=(
            "Verify an explicit historical trajectory/action population "
            "against immutable benchmark and artifact digests."
        ),
    )
    compatibility.add_argument("--workspace", required=True)
    compatibility.add_argument("--manifest", required=True)
    compatibility.add_argument("--artifact-root", required=True)
    compatibility.add_argument(
        "--collection",
        default="",
        help="Benchmark collection ID (default: the workspace's only collection).",
    )
    compatibility.set_defaults(_handler=_verify_compatibility_input)


__all__ = [
    "register_bench",
    "register_configure",
    "register_doctor",
    "register_generate",
    "register_reproduce",
    "register_run",
    "run_init",
]
