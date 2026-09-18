"""Canonical benchmark initialization and RoboCasa preflight commands."""

from __future__ import annotations

import argparse

from ..sim.runtime.environment import preflight_robocasa_runtime
from ._common import _print_json, _runtime_source_root


ROBOCASA_RUNTIME_PREFLIGHT_SCHEMA = "dream-exe.robocasa-runtime-preflight"


def _run_init(args: argparse.Namespace) -> int:
    from .bench import run_init

    return run_init(args)


def register_init(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--workspace", required=True)
    parser.add_argument("--case", required=True)
    parser.add_argument(
        "--receipt-id",
        default="",
        help="Optional stable work receipt ID for benchmark initialization.",
    )
    parser.set_defaults(_handler=_run_init)


def _add_robocasa_preflight_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--robocasa-source-root",
        default="",
        help=(
            "Optional absolute RoboCasa source root. Omit it to validate the "
            "installed package."
        ),
    )


def _run_robocasa_preflight(args: argparse.Namespace) -> int:
    source_root = _runtime_source_root(
        args.robocasa_source_root,
        flag="--robocasa-source-root",
    )
    preflight_robocasa_runtime(robocasa_source_root=source_root)
    payload = {
        "format": ROBOCASA_RUNTIME_PREFLIGHT_SCHEMA,
        "status": "pass",
        "runtime": "robocasa",
        "source_mode": (
            "installed_package" if source_root is None else "explicit_source_root"
        ),
        "environment_created": False,
        "frozen_assets": "bench_owned",
        "episode_dataset_required": False,
    }
    _print_json(payload)
    return 0


def register_preflight_robocasa(parser: argparse.ArgumentParser) -> None:
    _add_robocasa_preflight_arguments(parser)
    parser.set_defaults(_handler=_run_robocasa_preflight)


__all__ = ["register_init", "register_preflight_robocasa"]
