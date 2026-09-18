"""Model-asset verification and acquisition CLI commands."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from ..model_assets.models import (
    verify_model_assets,
    current_core_model_manifest,
    current_pose_model_manifest,
    load_model_asset_manifest,
    plan_model_asset_acquisition,
    publish_model_assets,
)
from ._common import _print_json


def _add_model_asset_arguments(
    parser: argparse.ArgumentParser,
    *,
    acquire: bool,
) -> None:
    parser.add_argument(
        "--manifest",
        default="current-pose",
        help=(
            "Explicit manifest JSON path, 'core' for the pinned public core "
            "assets, or 'current-pose' for the optional pose assets."
        ),
    )
    parser.add_argument(
        "--asset-root",
        required=True,
        help="Explicit model asset destination root.",
    )
    if acquire:
        parser.add_argument(
            "--accept-license",
            action="append",
            default=[],
            help=("Explicit accepted license id; repeat for each required package."),
        )
        parser.add_argument("--offline", action="store_true")
        parser.add_argument("--force", action="store_true")
        parser.add_argument("--dry-run", action="store_true")


def _model_asset_manifest(argument: str) -> dict[str, Any]:
    name = str(argument or "").strip()
    if name == "core":
        return current_core_model_manifest()
    if name == "current-pose":
        return current_pose_model_manifest()
    return load_model_asset_manifest(argument)


def _run_model_asset_verification(args: argparse.Namespace) -> int:
    result = verify_model_assets(
        _model_asset_manifest(args.manifest),
        asset_root=Path(args.asset_root).expanduser().resolve(),
    )
    _print_json(result)
    return 0 if result["ready"] else 1


def _run_model_asset_acquisition(args: argparse.Namespace) -> int:
    root = Path(args.asset_root).expanduser().resolve()
    plan = plan_model_asset_acquisition(
        _model_asset_manifest(args.manifest),
        asset_root=root,
        accepted_licenses=args.accept_license,
        offline=bool(args.offline),
        force=bool(args.force),
    )
    if plan["status"] == "blocked":
        _print_json(plan)
        return 1
    result = publish_model_assets(
        plan,
        asset_root=root,
        dry_run=bool(args.dry_run),
    )
    _print_json(result)
    return 0


def register_model_asset_verification(parser: argparse.ArgumentParser) -> None:
    _add_model_asset_arguments(parser, acquire=False)
    parser.set_defaults(_handler=_run_model_asset_verification)


def register_model_asset_acquisition(parser: argparse.ArgumentParser) -> None:
    _add_model_asset_arguments(parser, acquire=True)
    parser.set_defaults(_handler=_run_model_asset_acquisition)
