"""Thin CLI for the explicit standalone video2traj callable."""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

from ..models import compose_runtime_config, load_model_catalog
from .runtime.standalone import (
    run_standalone_video2traj,
    summarize_standalone_video2traj_result,
)


def _json_object(path: str | Path, *, label: str) -> dict[str, Any]:
    source = Path(path).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"{label} not found: {source.as_posix()}")
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise ValueError(
            f"{label} is not valid JSON: {source.as_posix()}: {error}"
        ) from error
    if not isinstance(payload, dict):
        raise ValueError(f"{label} must be a JSON object: {source.as_posix()}")
    return payload


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m dream_exe.video2traj",
        description=(
            "Run simulator-independent raw-video multi-object video2traj "
            "from explicit files and replaceable model backends."
        ),
    )
    parser.add_argument("--video", required=True)
    parser.add_argument("--simulator-config", required=True)
    parser.add_argument("--pipeline-config", required=True)
    parser.add_argument("--runtime-config", required=True)
    parser.add_argument(
        "--models-config",
        default="",
        help="Optional trusted dream-exe.models catalog for exec components.",
    )
    parser.add_argument(
        "--run-options",
        default="",
        help=(
            "Optional explicit JSON object containing advanced public "
            "run_multi_object_video_file options."
        ),
    )
    parser.add_argument(
        "--video-backend",
        default="auto",
        choices=("auto", "decord", "cv2"),
    )
    parser.add_argument(
        "--write-artifacts",
        action="store_true",
        help=(
            "Publish current-compatible final trajectory artifacts to the "
            "explicit --output-dir."
        ),
    )
    parser.add_argument("--output-dir", default="")
    parser.add_argument(
        "--summary-json",
        default="",
        help="Optional explicit path for the bounded command summary.",
    )
    return parser


def _write_summary(path: str | Path, summary: dict[str, Any]) -> None:
    target = Path(path).expanduser().resolve()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if bool(args.write_artifacts) != bool(str(args.output_dir or "").strip()):
        parser.error("--write-artifacts and --output-dir must be supplied together")
    try:
        simulator_config = _json_object(
            args.simulator_config,
            label="simulator config",
        )
        pipeline_config = _json_object(
            args.pipeline_config,
            label="pipeline config",
        )
        catalog = (
            load_model_catalog(args.models_config)
            if str(args.models_config or "").strip()
            else None
        )
        composition = compose_runtime_config(
            defaults=None,
            catalog=catalog,
            runtime_config_path=args.runtime_config,
            preflight_factories=True,
        )
        run_options = (
            _json_object(args.run_options, label="run options")
            if str(args.run_options or "").strip()
            else {}
        )
        result = run_standalone_video2traj(
            video_path=Path(args.video).expanduser().resolve(),
            simulator_config=simulator_config,
            pipeline_config=pipeline_config,
            runtime_config=composition.runtime_config,
            pipeline_asset_base=(
                Path(args.pipeline_config).expanduser().resolve().parent
            ),
            runtime_asset_base=(
                Path(args.runtime_config).expanduser().resolve().parent
            ),
            run_options=run_options,
            video_backend=args.video_backend,
            write_artifacts=bool(args.write_artifacts),
            output_dir=(args.output_dir or None),
        )
        summary = summarize_standalone_video2traj_result(result)
        summary["model_composition"] = copy_model_composition(
            composition.public_config
        )
        if str(args.summary_json or "").strip():
            _write_summary(args.summary_json, summary)
        print(json.dumps(summary, indent=2))
        return 0
    except (KeyboardInterrupt, SystemExit):
        raise
    except Exception as error:
        print(f"video2traj failed: {error}", file=sys.stderr)
        return 1


def copy_model_composition(runtime_config: dict[str, Any] | Any) -> dict[str, Any]:
    if not isinstance(runtime_config, dict):
        runtime_config = dict(runtime_config)
    meta = runtime_config.get("_meta", {})
    if not isinstance(meta, dict):
        return {}
    value = meta.get("model_composition", {})
    return dict(value) if isinstance(value, dict) else {}


__all__ = ["main"]
