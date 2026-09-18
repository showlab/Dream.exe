"""Public model catalog inspection and preflight commands."""

from __future__ import annotations

import argparse

from ..models import (
    MODEL_CATEGORIES,
    MODEL_KINDS,
    available_models,
    check_model,
    load_model_catalog,
)
from ._common import _print_json


def _list(args: argparse.Namespace) -> int:
    catalog = (
        load_model_catalog(args.models_config)
        if str(args.models_config or "").strip()
        else None
    )
    _print_json(
        {
            "format": "dream-exe.models-list",
            "models_config": bool(catalog),
            "catalog_sha256": None if catalog is None else catalog.sha256,
            "catalog_layout": None if catalog is None else catalog.layout,
            "category": args.category or None,
            "kind": args.kind or None,
            "models": available_models(
                catalog,
                category=args.category or None,
                kind=args.kind or None,
            ),
        }
    )
    return 0


def _check(args: argparse.Namespace) -> int:
    catalog = load_model_catalog(args.models_config)
    _print_json(
        check_model(
            catalog=catalog,
            model_id=args.model,
            expected_category=args.category or None,
            expected_kind=args.kind or None,
            smoke_input=args.smoke_input or None,
        )
    )
    return 0


def register_models(parser: argparse.ArgumentParser) -> None:
    commands = parser.add_subparsers(dest="models_command", required=True)
    listing = commands.add_parser(
        "list",
        help=(
            "List built-in and caller-owned models by lifecycle category "
            "and backend kind."
        ),
    )
    listing.add_argument("--models-config", default="")
    listing.add_argument("--category", choices=MODEL_CATEGORIES, default="")
    listing.add_argument("--kind", choices=MODEL_KINDS, default="")
    listing.set_defaults(_handler=_list)

    checking = commands.add_parser(
        "check",
        help=(
            "Validate one catalog model before generation, execution, "
            "evaluation, or expensive model startup."
        ),
    )
    checking.add_argument("--models-config", required=True)
    checking.add_argument("--model", required=True)
    checking.add_argument("--category", choices=MODEL_CATEGORIES, default="")
    checking.add_argument("--kind", choices=MODEL_KINDS, default="")
    checking.add_argument(
        "--smoke-input",
        default="",
        help=(
            "Optional trusted JSON factory that builds args/kwargs for one real "
            "minimal inference call."
        ),
    )
    checking.set_defaults(_handler=_check)


__all__ = ["register_models"]
