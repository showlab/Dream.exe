"""Root CLI parser with stable command order and help text."""

from __future__ import annotations

import argparse


def build_parser() -> argparse.ArgumentParser:
    from . import (
        assets,
        bench,
        evaluation,
        models,
        sim,
    )

    parser = argparse.ArgumentParser(
        prog="dream-exe",
        description=(
            "Dream.exe callable pipeline commands for explicit "
            "initialization, video workflows, simulator execution, and "
            "trajectory, task success-rate, or optional VLM evaluation."
        ),
    )
    subparsers = parser.add_subparsers(
        dest="command",
        required=True,
    )
    preflight_robocasa_parser = subparsers.add_parser(
        "preflight-robocasa",
        help=(
            "Validate RoboSuite and RoboCasa imports/registration without "
            "creating a simulator environment."
        ),
    )
    sim.register_preflight_robocasa(preflight_robocasa_parser)
    configure_parser = subparsers.add_parser(
        "configure",
        help="Bind a downloaded benchmark and local providers to one workspace file.",
    )
    bench.register_configure(configure_parser)
    doctor_parser = subparsers.add_parser(
        "doctor",
        help="Validate benchmark assets, paths, and external runtime bindings.",
    )
    bench.register_doctor(doctor_parser)
    init_parser = subparsers.add_parser(
        "init",
        help=(
            "Materialize one canonical benchmark case environment receipt."
        ),
    )
    sim.register_init(init_parser)
    run_parser = subparsers.add_parser(
        "run",
        help="Run one strict benchmark spec using workspace-owned roots.",
    )
    bench.register_run(run_parser)
    reproduce_parser = subparsers.add_parser(
        "reproduce",
        help="Run every published experiment spec, or a selected subset.",
    )
    bench.register_reproduce(reproduce_parser)
    generate_parser = subparsers.add_parser(
        "generate",
        help=(
            "Generate one standard or enhanced case video from the canonical "
            "first frame, then register it under results/videos."
        ),
    )
    bench.register_generate(generate_parser)
    bench_parser = subparsers.add_parser(
        "bench",
        help="Import/register candidate videos or aggregate saved results.",
    )
    bench.register_bench(bench_parser)
    evaluation_parser = subparsers.add_parser(
        "eval-exec",
        help=(
            "Build current-compatible deterministic metrics from one "
            "explicit execution directory."
        ),
    )
    evaluation.register_exec_evaluation(evaluation_parser)
    trajectory_evaluation_parser = subparsers.add_parser(
        "eval-trajectory",
        help=(
            "Evaluate explicit trajectory similarity, raw executability, "
            "or policy-free path-length comparison."
        ),
    )
    evaluation.register_trajectory_evaluation(trajectory_evaluation_parser)
    task_success_rate_parser = subparsers.add_parser(
        "eval-task-success-rate",
        help=(
            "Aggregate explicit saved task-success artifacts with "
            "paper-aligned metrics."
        ),
    )
    evaluation.register_task_success_rate_evaluation(task_success_rate_parser)
    evaluation_suite_parser = subparsers.add_parser(
        "evaluate",
        help=(
            "Evaluate visual quality, trajectory similarity, executability, "
            "task completion, or all four for one case or a collection."
        ),
    )
    evaluation.register_evaluation_suite(evaluation_suite_parser)
    vlm_parser = subparsers.add_parser(
        "eval-vlm",
        help=(
            "Evaluate explicit saved evidence in video-only or "
            "video+trajectory VLM mode."
        ),
    )
    evaluation.register_vlm_evaluation(vlm_parser)
    models_parser = subparsers.add_parser(
        "models",
        help=(
            "List or preflight trusted caller-owned backends by "
            "video_gen, exec, or eval category."
        ),
    )
    models.register_models(models_parser)
    model_verification_parser = subparsers.add_parser(
        "verify-model-assets",
        help=(
            "Verify an explicit model asset root against pinned sizes, "
            "digests, and license metadata without writes."
        ),
    )
    assets.register_model_asset_verification(model_verification_parser)
    model_acquire_parser = subparsers.add_parser(
        "acquire-model-assets",
        help=("Plan, verify, and atomically publish explicitly licensed model assets."),
    )
    assets.register_model_asset_acquisition(model_acquire_parser)
    return parser
