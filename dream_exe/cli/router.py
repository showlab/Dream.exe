"""Root CLI dispatch."""

from __future__ import annotations

from collections.abc import Sequence

from .parser import build_parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(None if argv is None else list(argv))
    handler = getattr(args, "_handler", None)
    if not callable(handler):
        parser.error(f"unsupported command: {args.command}")
        return 2
    return int(handler(args))


__all__ = ["main"]
