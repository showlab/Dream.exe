"""Run simulator execution for one materialized frozen benchmark case."""

from __future__ import annotations

import copy
import json
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

from ...artifacts.layout import (
    execution_artifact_paths,
    sample_artifact_paths,
    trajectory_artifact_paths,
)
from ..records.layout import (
    DEFAULT_FORMAL_RUN_KEY,
    build_run_id,
    formal_artifact_paths,
    load_resolved_kind_config,
    normalize_run_key,
    split_run_key,
)
from ...sim.execution.config import normalize_execution_config
from ...sim.execution.runtime import execute_simulation_explicit
from ..records.execution_lineage import (
    capture_execution_input_lineage,
    publish_execution_input_lineage,
    verify_execution_input_lineage,
)


def _load_json_object(
    path: Path,
    *,
    label: str,
) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"{label} not found: {path}")
    with path.open(encoding="utf-8") as stream:
        value = json.load(stream)
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return value


def _resolve_configured_path(
    value: Any,
    *,
    source_dir: Path,
) -> str:
    text = str(value or "").strip()
    if not text:
        return ""
    path = Path(text).expanduser()
    if path.is_absolute():
        return path.resolve().as_posix()
    return (source_dir / path).resolve().as_posix()


def _resolve_frozen_scene_paths(
    config: Mapping[str, Any],
    *,
    manifest_dir: Path,
) -> dict[str, Any]:
    """Resolve portable frozen paths in memory without changing the manifest."""

    resolved = copy.deepcopy(dict(config))
    raw = resolved.get("raw", resolved)
    if not isinstance(raw, dict):
        raise TypeError("simulator config raw must be an object")
    scene_restore = raw.get("scene_restore", {})
    if not isinstance(scene_restore, dict):
        raise TypeError("simulator config raw.scene_restore must be an object")
    paths = scene_restore.get("paths", {})
    if not isinstance(paths, dict):
        raise TypeError("simulator config raw.scene_restore.paths must be an object")
    scene_restore["paths"] = {
        str(role): _resolve_configured_path(path, source_dir=manifest_dir)
        for role, path in paths.items()
    }
    return resolved


def resolve_bench_execution_request(
    *,
    sample_dir: str | Path,
    run_key: str = DEFAULT_FORMAL_RUN_KEY,
    gen_model: str = "",
    simulator_config_path: str | Path | None = None,
    execution_config_path: str | Path | None = None,
    trajectory_path: str | Path | None = None,
    action_path: str | Path | None = None,
    output_dir: str | Path | None = None,
    scene_override_path: str | Path | None = None,
) -> dict[str, Any]:
    """Resolve one bench execution request without creating or writing paths."""

    sample_root = Path(sample_dir).expanduser().resolve()
    sample_paths = sample_artifact_paths(sample_root)
    normalized_run_key = normalize_run_key(run_key)
    formal_paths = formal_artifact_paths(
        sample_root,
        normalized_run_key,
        gen_model=gen_model,
    )
    simulator_path = (
        Path(simulator_config_path).expanduser().resolve()
        if simulator_config_path is not None and str(simulator_config_path).strip()
        else formal_paths["simulator_config"]
    )
    simulator_config = _resolve_frozen_scene_paths(
        _load_json_object(
            simulator_path,
            label="simulator config",
        ),
        manifest_dir=simulator_path.parent,
    )

    if execution_config_path is not None and str(execution_config_path).strip():
        execution_path = Path(execution_config_path).expanduser().resolve()
        source_execution = _load_json_object(
            execution_path,
            label="execution config",
        )
        execution_meta = {
            "mode": "explicit",
            "source": execution_path.as_posix(),
            "run_key": normalized_run_key,
            "kind": "execution",
        }
        execution_source_dir = execution_path.parent
    else:
        source_execution, execution_meta = load_resolved_kind_config(
            sample_dir=sample_root,
            run_key=normalized_run_key,
            kind="execution",
            gen_model=gen_model,
        )
        source_text = str(execution_meta.get("source", "") or "")
        execution_source_dir = (
            Path(source_text).expanduser().resolve().parent
            if source_text and source_text != "<missing canonical base>"
            else sample_paths["pipeline_root"]
        )

    execution_config = normalize_execution_config(source_execution)
    configured_inputs = execution_config["input"]
    configured_trajectory = _resolve_configured_path(
        configured_inputs.get("traj_path", ""),
        source_dir=execution_source_dir,
    )
    configured_action = _resolve_configured_path(
        configured_inputs.get("action_path", ""),
        source_dir=execution_source_dir,
    )
    resolved_trajectory = (
        Path(trajectory_path).expanduser().resolve().as_posix()
        if trajectory_path is not None and str(trajectory_path).strip()
        else configured_trajectory or formal_paths["ee_traj"].as_posix()
    )
    resolved_action = (
        Path(action_path).expanduser().resolve().as_posix()
        if action_path is not None and str(action_path).strip()
        else configured_action or formal_paths["action"].as_posix()
    )
    resolved_output = (
        Path(output_dir).expanduser().resolve().as_posix()
        if output_dir is not None and str(output_dir).strip()
        else formal_paths["exec_dir"].as_posix()
    )

    raw = dict(simulator_config.get("raw", {}) or {})
    scene_restore = dict(raw.get("scene_restore", {}) or {})
    configured_scene_override = _resolve_configured_path(
        scene_restore.get("scene_override_json", ""),
        source_dir=simulator_path.parent,
    )
    default_scene_override = formal_paths["scene_override"]
    resolved_scene_override = (
        Path(scene_override_path).expanduser().resolve()
        if scene_override_path is not None and str(scene_override_path).strip()
        else (
            Path(configured_scene_override)
            if configured_scene_override and Path(configured_scene_override).exists()
            else default_scene_override
        )
    )
    scene_override: dict[str, Any] = {}
    if resolved_scene_override.exists():
        try:
            scene_override = _load_json_object(
                resolved_scene_override,
                label="scene override",
            )
        except Exception:
            scene_override = {}
    active_camera_name = str(
        scene_override.get(
            "active_camera_name",
            scene_restore.get("active_camera_name", ""),
        )
        or ""
    ).strip()

    return {
        "sample_dir": sample_root.as_posix(),
        "uid": sample_root.name,
        "run_key": normalized_run_key,
        "gen_model": str(gen_model or ""),
        "simulator_config": simulator_config,
        "simulator_config_path": simulator_path.as_posix(),
        "execution_config": execution_config,
        "execution_config_meta": copy.deepcopy(execution_meta),
        "trajectory_path": resolved_trajectory,
        "action_path": resolved_action,
        "output_dir": resolved_output,
        "scene_override_path": (
            resolved_scene_override.as_posix()
            if resolved_scene_override.exists()
            else ""
        ),
        "active_camera_name": active_camera_name,
    }


def execute_bench_simulation(
    *,
    runner: Callable[..., dict[str, Any]] | None = None,
    scene_restore_options: Mapping[str, Any] | None = None,
    robocasa_source_root: str | Path | None = None,
    **request_options: Any,
) -> dict[str, Any]:
    """Resolve current bench paths, then call the bench-free sim runtime."""

    request = resolve_bench_execution_request(**request_options)
    captured_lineage = (
        capture_execution_input_lineage(request) if runner is None else None
    )
    execute = execute_simulation_explicit if runner is None else runner
    runtime_options: dict[str, Any] = {
        "simulator_config": request["simulator_config"],
        "execution_config": request["execution_config"],
        "output_dir": request["output_dir"],
        "trajectory_path": request["trajectory_path"],
        "action_path": request["action_path"],
        "uid": request["uid"],
        "run_key": request["run_key"],
        "gen_model": (request["gen_model"] or None),
        "active_camera_name": (request["active_camera_name"] or None),
        "scene_restore_options": scene_restore_options,
    }
    if robocasa_source_root is not None:
        runtime_options["robocasa_source_root"] = robocasa_source_root
    result = execute(
        **runtime_options,
    )
    response = {
        "request": request,
        "result": result,
    }
    if captured_lineage is not None:
        verify_execution_input_lineage(captured_lineage, request)
        output_root = Path(request["output_dir"])
        summary_path = execution_artifact_paths(output_root)["exec_summary"]
        if not summary_path.is_file():
            raise RuntimeError(
                "successful simulator execution did not publish "
                f"exec_summary.json: {summary_path}"
            )
        lineage_path = publish_execution_input_lineage(
            output_root,
            captured_lineage,
        )
        response["execution_input_lineage_path"] = lineage_path.as_posix()
    return response


def execute_benchmark_stage(
    *,
    uid: str,
    sample_dir: str | Path,
    run_id: str,
    run_key: str,
    video_kind: str,
    gen_model: str,
    execution_config_path: str | Path,
    trajectory_root: str | Path,
    action_json_path: str | Path,
    output_exec_root: str | Path,
    simulator_config_path: str | Path | None = None,
    scene_override_path: str | Path | None = None,
    scene_restore_options: Mapping[str, Any] | None = None,
    robocasa_source_root: str | Path | None = None,
    runner: Callable[..., dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Bind one benchmark-matrix execution stage to the current implementation sim callable.

    The signature intentionally accepts the exact options emitted by the
    formal single-case workflow. Identity checks happen before any simulator
    construction or output write.
    """

    sample_root = Path(sample_dir).expanduser().resolve()
    clean_uid = str(uid or "").strip()
    if not clean_uid:
        raise ValueError("uid is required")
    if sample_root.name != clean_uid:
        raise ValueError(
            f"benchmark uid/sample_dir mismatch: {clean_uid!r} != {sample_root.name!r}"
        )
    normalized_run_key = normalize_run_key(run_key)
    expected_video_kind, _slot = split_run_key(normalized_run_key)
    if str(video_kind or "").strip() != expected_video_kind:
        raise ValueError("benchmark video_kind does not match run_key")
    expected_run_id = build_run_id(
        run_key=normalized_run_key,
        gen_model=str(gen_model or ""),
    )
    if str(run_id or "").strip() != expected_run_id:
        raise ValueError("benchmark run_id does not match run identity")
    trajectory_root_path = Path(trajectory_root).expanduser().resolve()
    action_path = Path(action_json_path).expanduser().resolve()
    expected_action = trajectory_artifact_paths(trajectory_root_path)["action"]
    if action_path != expected_action:
        raise ValueError(
            "benchmark action_json_path does not match "
            "trajectory_root/action/action.json"
        )

    execution = execute_bench_simulation(
        sample_dir=sample_root,
        run_key=normalized_run_key,
        gen_model=str(gen_model or ""),
        simulator_config_path=simulator_config_path,
        execution_config_path=execution_config_path,
        action_path=action_path,
        output_dir=output_exec_root,
        scene_override_path=scene_override_path,
        scene_restore_options=scene_restore_options,
        robocasa_source_root=robocasa_source_root,
        runner=runner,
    )
    request = dict(execution.get("request", {}) or {})
    if request.get("uid") != clean_uid:
        raise RuntimeError("resolved execution request changed benchmark uid")
    response = {
        "ok": True,
        "returncode": 0,
        "uid": clean_uid,
        "run_id": expected_run_id,
        "run_key": normalized_run_key,
        "gen_model": str(gen_model or ""),
        "request": request,
        "result": execution.get("result"),
    }
    if "execution_input_lineage_path" in execution:
        response["execution_input_lineage_path"] = execution[
            "execution_input_lineage_path"
        ]
    return response


__all__ = [
    "execute_benchmark_stage",
    "execute_bench_simulation",
    "resolve_bench_execution_request",
]
