"""Dream.exe command-line entry points."""

from .parser import build_parser
from .router import main


__all__ = ["build_parser", "main"]
