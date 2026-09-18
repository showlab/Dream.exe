"""Pure path roles for the current Dream.exe benchmark artifact layout.

This module is the single storage-layout vocabulary shared by bench producers
and saved-artifact consumers. It accepts already configured roots and only
returns ``Path`` objects: importing or calling it never creates, moves, reads,
or writes data.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Sequence


PIPELINE_DIRECTORY = "pipeline"
ARTIFACTS_DIRECTORY = "artifacts"
ENVIRONMENT_DIRECTORY = "env"
INITIALIZATION_DIRECTORY = "init"
RUNS_DIRECTORY = "runs"
GENERATED_DIRECTORY = "gen"
GENERATED_ENHANCED_DIRECTORY = "gen_enhanced"
GROUND_TRUTH_DIRECTORY = "gt"
TRAJECTORY_DIRECTORY = "trajectory"
GRIPPER_DIRECTORY = "gripper"
ACTION_DIRECTORY = "action"
REGION_DIRECTORY = "region"
TRACKING_DIRECTORY = "tracking"
GEOMETRY_DIRECTORY = "geometry"
POSE_DIRECTORY = "pose"
VISUALIZATION_DIRECTORY = "visualization"
EXECUTION_DIRECTORY = "exec"
VIDEO_DIRECTORY = "video"
DEPTH_DIRECTORY = "depth"
VLM_DIRECTORY = "vlm"
VLM_PREDICTIONS_DIRECTORY = "predictions"
MASK_DIRECTORY = "mask"
CAD_DIRECTORY = "cad"
CAMERA_DIRECTORY = "cam"
RAW_DIRECTORY = "raw"
PROCESSED_DIRECTORY = "processed"
EXPERIMENT_DIRECTORY = "experiment"
LOGS_DIRECTORY = "logs"
RUN_STATE_DIRECTORY = "run_state"
RECOVERY_DIRECTORY = "recovery"
BENCH_DATA_DIRECTORY = "data"
BENCH_REGISTRY_DIRECTORY = "registry"
BENCH_METRICS_SUMMARIES_DIRECTORY = "metrics_summaries"
BENCH_COSMOS_DATA_DIRECTORY = "cosmos_data"
BENCHMARK_SUITE_DIRECTORY = "benchmark_suite"
BENCHMARK_SUITE_SUMMARY_FILENAME = "summary.json"
BENCHMARK_SUITE_MATERIALIZATION_FILENAME = "materialization.json"
BENCHMARK_SUITE_TRANSITIONS_FILENAME = "transitions.jsonl"

INITIALIZATION_CONFIG_FILENAME = "initialization.json"
TRAJECTORY_CONFIG_FILENAME = "trajectory.json"
EXECUTION_CONFIG_FILENAME = "execution.json"
RUN_OVERRIDES_FILENAME = "run_overrides.json"
EVALUATION_RESULT_FILENAME = "evaluation_result.json"
SAMPLE_METADATA_FILENAME = "meta.json"
SIMULATOR_CONFIG_FILENAME = "robosuite_config.json"
SCENE_OVERRIDE_FILENAME = "scene_override.json"
ASSETS_MANIFEST_FILENAME = "assets.json"
EEF_TRAJECTORY_FILENAME = "ee_traj.json"
OBJECT_TRAJECTORIES_FILENAME = "obj_trajs.json"
UNION_TRAJECTORY_FILENAME = "union_traj.json"
GRIPPER_FILENAME = "gripper.json"
ACTION_FILENAME = "action.json"
ACTION_ARRAY_FILENAME = "action.npy"
EXECUTION_SUMMARY_FILENAME = "exec_summary.json"
CHECKPOINT_TRACE_FILENAME = "checkpoint_trace.json"
DENSE_TCP_TRACE_FILENAME = "dense_tcp_trace.json"
EXECUTION_METRICS_FILENAME = "exec_metrics.json"
EXECUTION_METRICS_PER_FRAME_FILENAME = "exec_metrics_per_frame.csv"
EXECUTION_INPUTS_FILENAME = "execution_inputs.json"
ACTION_TRACE_FILENAME = "action_trace.json"
EXECUTION_VIDEO_FILENAME = "exec.mp4"
TASK_SUCCESS_FILENAME = "task_check_success.json"
PIPELINE_STATE_FILENAME = "pipeline_state.json"
INIT_RESUME_STATE_FILENAME = "init_resume_state.json"
RUNS_MANIFEST_FILENAME = "manifest.json"
GT_VIDEO_FILENAME = "gt.mp4"
GT_VIDEO_PROBE_FILENAME = "gt_probe.png"
GT_METRIC_DEPTH_FILENAME = "gt_metric.npy"
INITIALIZATION_IMAGE_FILENAME = "init_env.png"
INITIALIZATION_DEPTH_BUFFER_FILENAME = "init_depth_buffer.npy"
INITIALIZATION_DEPTH_FILENAME = "init_depth.npy"
INITIALIZATION_DEPTH_IMAGE_FILENAME = "init_depth.png"
INSTANCE_NAMES_FILENAME = "instance_names.json"
INITIALIZATION_INSTANCE_SEGMENTATION_FILENAME = "init_segmentation_instance.npy"
INITIALIZATION_SEGMENTATION_MAPPINGS_FILENAME = "init_segmentation_mappings.json"
GT_METRIC_DEPTH_VIDEO_FILENAME = "gt_metric.mp4"
GT_METRIC_DEPTH_FRAME0_FILENAME = "gt_metric_frame0.png"
GT_METRIC_DEPTH_CONTACT_FILENAME = "gt_metric_contact.png"
GT_DEPTH_METADATA_FILENAME = "meta.json"
GT_DEPTH_BUFFER_VIDEO_FILENAME = "gt_depth_buffer.mp4"
GT_DEPTH_BUFFER_FILENAME = "gt_depth_buffer.npy"
DEPTH_NPY_FILENAME = "depth.npy"
DEPTH_VIDEO_FILENAME = "depth.mp4"
DEPTH_META_NPY_FILENAME = "depth_meta.npy"
DEPTH_MANIFEST_FILENAME = "depth_manifest.json"
DEPTH_CACHE_METADATA_FILENAME = "depth_cache_meta.json"
DEPTH_FRAME0_FILENAME = "depth_frame0.png"
DEPTH_CONTACT_FILENAME = "depth_contact.png"
DEPTH_VIS_METADATA_FILENAME = "depth_vis_meta.json"
DEPTH_CALIBRATION_DIRECTORY = "depth_calibration"
DEPTH_RUNTIME_LINEAGE_FILENAME = "runtime_lineage.json"
EEF_CONSUMED_DEPTH_SAMPLES_FILENAME = "eef_consumed_depth_samples.npy"


def _root(path: str | Path) -> Path:
    expanded = Path(path).expanduser()
    return expanded if expanded.is_absolute() else expanded.absolute()


def bench_layout_paths(bench_root: str | Path) -> Dict[str, Path]:
    """Resolve the code-owned current bench layout from one configured root."""

    root = _root(bench_root)
    return {
        "bench_root": root,
        "bench_data_root": root / BENCH_DATA_DIRECTORY,
        "bench_registry_root": root / BENCH_REGISTRY_DIRECTORY,
        "bench_metrics_summaries_root": (root / BENCH_METRICS_SUMMARIES_DIRECTORY),
        "bench_cosmos_data_root": root / BENCH_COSMOS_DATA_DIRECTORY,
    }


def benchmark_suite_artifact_paths(
    bench_root: str | Path,
    suite_id: str,
) -> Dict[str, Path]:
    """Resolve owner-suite outputs below one isolated bench-shaped root."""

    clean_suite_id = _safe_parts([suite_id], label="benchmark suite")[0]
    metrics_root = bench_layout_paths(bench_root)["bench_metrics_summaries_root"]
    suite_root = metrics_root / BENCHMARK_SUITE_DIRECTORY / clean_suite_id
    return {
        "benchmark_suite_root": suite_root,
        "benchmark_suite_summary": suite_root / BENCHMARK_SUITE_SUMMARY_FILENAME,
        "benchmark_suite_materialization": (
            suite_root / BENCHMARK_SUITE_MATERIALIZATION_FILENAME
        ),
        "benchmark_suite_transitions": (
            suite_root / BENCHMARK_SUITE_TRANSITIONS_FILENAME
        ),
    }


def initialization_artifact_paths(
    initialization_root: str | Path,
) -> Dict[str, Path]:
    """Resolve roles below one explicit initialization output directory."""

    root = _root(initialization_root)
    depth = root / DEPTH_DIRECTORY
    image = root / "img"
    return {
        "initialization_dir": root,
        "initialization_camera_dir": root / CAMERA_DIRECTORY,
        "initialization_cad_dir": root / CAD_DIRECTORY,
        "initialization_manifest": root / ASSETS_MANIFEST_FILENAME,
        "initialization_image_dir": image,
        "initialization_image": image / INITIALIZATION_IMAGE_FILENAME,
        "initialization_depth_dir": depth,
        "initialization_depth_buffer": (depth / INITIALIZATION_DEPTH_BUFFER_FILENAME),
        "initialization_depth": depth / INITIALIZATION_DEPTH_FILENAME,
        "initialization_depth_image": (depth / INITIALIZATION_DEPTH_IMAGE_FILENAME),
        "initialization_mask_dir": root / MASK_DIRECTORY,
        "initialization_instance_names": (
            root / MASK_DIRECTORY / INSTANCE_NAMES_FILENAME
        ),
        "initialization_instance_segmentation": (
            root / MASK_DIRECTORY / INITIALIZATION_INSTANCE_SEGMENTATION_FILENAME
        ),
        "initialization_segmentation_mappings": (
            root / MASK_DIRECTORY / INITIALIZATION_SEGMENTATION_MAPPINGS_FILENAME
        ),
    }


def ground_truth_artifact_paths(
    ground_truth_root: str | Path,
) -> Dict[str, Path]:
    """Resolve roles below one explicit ground-truth output directory."""

    root = _root(ground_truth_root)
    action = root / ACTION_DIRECTORY
    video = root / VIDEO_DIRECTORY
    depth = root / DEPTH_DIRECTORY
    return {
        "ground_truth_root": root,
        "ground_truth_action_dir": action,
        "ground_truth_action": action / ACTION_FILENAME,
        "ground_truth_action_array": action / ACTION_ARRAY_FILENAME,
        "ground_truth_manifest": root / ASSETS_MANIFEST_FILENAME,
        "ground_truth_video_dir": video,
        "ground_truth_depth_dir": depth,
        "gt_video": video / GT_VIDEO_FILENAME,
        "gt_video_probe": video / GT_VIDEO_PROBE_FILENAME,
        "gt_metric_depth": depth / GT_METRIC_DEPTH_FILENAME,
        "gt_depth_buffer_video": (
            depth / GT_DEPTH_BUFFER_VIDEO_FILENAME
        ),
        "gt_depth_buffer": depth / GT_DEPTH_BUFFER_FILENAME,
        "gt_metric_depth_video": depth / GT_METRIC_DEPTH_VIDEO_FILENAME,
        "gt_metric_depth_frame0": depth / GT_METRIC_DEPTH_FRAME0_FILENAME,
        "gt_metric_depth_contact": depth / GT_METRIC_DEPTH_CONTACT_FILENAME,
        "gt_depth_metadata": depth / GT_DEPTH_METADATA_FILENAME,
    }


def generated_artifact_paths(generated_root: str | Path) -> Dict[str, Path]:
    """Resolve roles below one explicit generated-video output directory."""

    root = _root(generated_root)
    return {
        "generated_root": root,
        "generated_raw_dir": root / RAW_DIRECTORY,
        "generated_processed_dir": root / PROCESSED_DIRECTORY,
    }


def sample_artifact_paths(sample_root: str | Path) -> Dict[str, Path]:
    """Resolve current sample-level storage roles without touching storage."""

    sample = _root(sample_root)
    pipeline = sample / PIPELINE_DIRECTORY
    artifacts = sample / ARTIFACTS_DIRECTORY
    environment = artifacts / ENVIRONMENT_DIRECTORY
    initialization = artifacts / INITIALIZATION_DIRECTORY
    runs = artifacts / RUNS_DIRECTORY
    ground_truth = artifacts / GROUND_TRUTH_DIRECTORY
    generated = artifacts / GENERATED_DIRECTORY
    generated_enhanced = artifacts / GENERATED_ENHANCED_DIRECTORY
    sample_run_state = sample / RUN_STATE_DIRECTORY
    return {
        "sample_root": sample,
        "sample_metadata": sample / SAMPLE_METADATA_FILENAME,
        "pipeline_root": pipeline,
        "initialization_config": pipeline / INITIALIZATION_CONFIG_FILENAME,
        "trajectory_config": pipeline / TRAJECTORY_CONFIG_FILENAME,
        "execution_config": pipeline / EXECUTION_CONFIG_FILENAME,
        "run_overrides": pipeline / RUN_OVERRIDES_FILENAME,
        "artifacts_root": artifacts,
        "environment_dir": environment,
        "simulator_config": environment / SIMULATOR_CONFIG_FILENAME,
        "scene_override": environment / SCENE_OVERRIDE_FILENAME,
        **initialization_artifact_paths(initialization),
        "runs_root": runs,
        "runs_manifest": runs / RUNS_MANIFEST_FILENAME,
        "generated_runs_root": runs / GENERATED_DIRECTORY,
        **generated_artifact_paths(generated),
        "generated_enhanced_root": generated_enhanced,
        "generated_enhanced_raw_dir": (generated_enhanced / RAW_DIRECTORY),
        "generated_enhanced_processed_dir": (generated_enhanced / PROCESSED_DIRECTORY),
        "sample_experiment_root": sample / EXPERIMENT_DIRECTORY,
        "sample_logs_root": sample / LOGS_DIRECTORY,
        "sample_run_state_root": sample_run_state,
        "init_resume_state": sample_run_state / INIT_RESUME_STATE_FILENAME,
        "sample_recovery_root": sample_run_state / RECOVERY_DIRECTORY,
        **ground_truth_artifact_paths(ground_truth),
    }


def bench_initialization_path_context(
    sample_root: str | Path,
    initialization_config_path: str | Path,
) -> Dict[str, Path]:
    """Resolve current bench/init path anchors from one explicit sample.

    This is the centralized inverse of :func:`bench_sample_path` for the
    current ``<bench_root>/data/<uid>`` contract.  It performs no I/O.  An
    explicitly supplied non-canonical initialization document retains the
    established compatibility anchor two levels above that document; the
    canonical pipeline document anchors relative source paths at the sample.
    """

    sample = _root(sample_root)
    data_root = sample.parent
    layout = bench_layout_paths(data_root.parent)
    if layout["bench_data_root"] != data_root:
        raise ValueError(
            "sample_root must use the current bench layout "
            "<bench_root>/data/<uid>"
        )
    initialization_config = _root(initialization_config_path)
    canonical_config = sample_artifact_paths(sample)["initialization_config"]
    relative_source_root = (
        sample
        if initialization_config == canonical_config
        else initialization_config.parent.parent
    )
    return {
        **layout,
        "sample_root": sample,
        "initialization_config": initialization_config,
        "config_root": initialization_config.parent,
        "relative_source_root": relative_source_root,
    }


def sample_scaffold_directories(sample_root: str | Path) -> tuple[Path, ...]:
    """Return the directories created for a new current-layout sample."""

    paths = sample_artifact_paths(sample_root)
    generated = paths["generated_root"]
    return (
        paths["pipeline_root"],
        paths["environment_dir"],
        paths["ground_truth_action_dir"],
        paths["ground_truth_depth_dir"],
        paths["ground_truth_video_dir"],
        paths["initialization_cad_dir"],
        paths["initialization_camera_dir"],
        paths["initialization_depth_dir"],
        paths["initialization_image_dir"],
        paths["initialization_mask_dir"],
        generated,
        generated / RAW_DIRECTORY,
        generated / PROCESSED_DIRECTORY,
        paths["runs_root"],
        paths["runs_root"] / "gt_video" / "dvd_depth",
        paths["runs_root"] / "gt_video" / "gt_depth",
        paths["sample_experiment_root"],
        paths["sample_logs_root"],
    )


def sample_scaffold_relative_directories() -> tuple[str, ...]:
    """Return portable relative names for the current sample scaffold."""

    sentinel = Path("/") / "__dream_exe_sample_layout__"
    return tuple(
        path.relative_to(sentinel).as_posix()
        for path in sample_scaffold_directories(sentinel)
    )


def pipeline_run_config_path(
    sample_root: str | Path,
    run_parts: Sequence[str],
    filename: str,
) -> Path:
    """Resolve one current run-scoped pipeline config without I/O."""

    safe_filename = str(filename or "").strip()
    if (
        not safe_filename
        or safe_filename in {".", ".."}
        or Path(safe_filename).name != safe_filename
        or "/" in safe_filename
        or "\\" in safe_filename
    ):
        raise ValueError(f"unsafe pipeline config filename: {filename!r}")
    clean_parts = _safe_parts(run_parts, label="pipeline run")
    return sample_artifact_paths(sample_root)["pipeline_root"].joinpath(
        *clean_parts,
        safe_filename,
    )


def _safe_parts(parts: Sequence[str], *, label: str) -> list[str]:
    clean_parts: list[str] = []
    for raw_part in parts:
        part = str(raw_part or "").strip()
        if (
            not part
            or part in {".", ".."}
            or "/" in part
            or "\\" in part
            or Path(part).name != part
        ):
            raise ValueError(f"unsafe {label} path component: {raw_part!r}")
        clean_parts.append(part)
    if not clean_parts:
        raise ValueError(f"{label} parts must contain at least one component")
    return clean_parts


def bench_sample_path(
    bench_data_root: str | Path,
    uid: str,
) -> Path:
    """Resolve one sample below an already configured bench data root.

    This is a pure lexical mapping.  It deliberately performs no filesystem
    discovery or creation; callers that enforce physical containment or
    symlink policy remain responsible for those checks at their I/O boundary.
    """

    clean_uid = _safe_parts([uid], label="bench sample uid")[0]
    return _root(bench_data_root) / clean_uid


def run_artifact_paths(run_root: str | Path) -> Dict[str, Path]:
    """Resolve current roles below one already selected formal run root."""

    root = _root(run_root)
    logs = root / LOGS_DIRECTORY
    return {
        "run_root": root,
        "traj_dir": root / "traj",
        "exec_dir": root / EXECUTION_DIRECTORY,
        "logs_dir": logs,
        "init_log": logs / "init.log",
        "traj_log": logs / "traj.log",
        "exec_log": logs / "exec.log",
        "eval_log": logs / "eval.log",
        "run_json": root / "run.json",
        "pipeline_state": root / PIPELINE_STATE_FILENAME,
    }


def run_artifact_paths_for_sample(
    sample_root: str | Path,
    run_parts: Sequence[str],
) -> Dict[str, Path]:
    """Resolve one formal run from safe semantic identity components."""

    clean_parts = _safe_parts(run_parts, label="run")
    runs_root = sample_artifact_paths(sample_root)["runs_root"]
    return run_artifact_paths(runs_root.joinpath(*clean_parts))


def trajectory_artifact_paths(trajectory_root: str | Path) -> Dict[str, Path]:
    """Resolve artifact roles relative to one explicit ``traj`` directory."""

    root = _root(trajectory_root)
    trajectory = root / TRAJECTORY_DIRECTORY
    gripper = root / GRIPPER_DIRECTORY
    action = root / ACTION_DIRECTORY
    depth = root / DEPTH_DIRECTORY
    depth_calibration = root / GEOMETRY_DIRECTORY / DEPTH_CALIBRATION_DIRECTORY
    return {
        "traj_dir": root,
        "region_dir": root / REGION_DIRECTORY,
        "tracking_dir": root / TRACKING_DIRECTORY,
        "geometry_dir": root / GEOMETRY_DIRECTORY,
        "pose_dir": root / POSE_DIRECTORY,
        "trajectory_dir": trajectory,
        "ee_traj": trajectory / EEF_TRAJECTORY_FILENAME,
        "obj_trajs": trajectory / OBJECT_TRAJECTORIES_FILENAME,
        "union_traj": trajectory / UNION_TRAJECTORY_FILENAME,
        "gripper_dir": gripper,
        "gripper": gripper / GRIPPER_FILENAME,
        "action_dir": action,
        "action": action / ACTION_FILENAME,
        "action_array": action / ACTION_ARRAY_FILENAME,
        "depth_dir": depth,
        "depth_npy": depth / DEPTH_NPY_FILENAME,
        "depth_mp4": depth / DEPTH_VIDEO_FILENAME,
        "depth_meta_npy": depth / DEPTH_META_NPY_FILENAME,
        "depth_cache_meta_json": depth / DEPTH_CACHE_METADATA_FILENAME,
        "depth_manifest_json": depth / DEPTH_MANIFEST_FILENAME,
        "depth_frame0_png": depth / DEPTH_FRAME0_FILENAME,
        "depth_contact_png": depth / DEPTH_CONTACT_FILENAME,
        "depth_vis_meta_json": depth / DEPTH_VIS_METADATA_FILENAME,
        "depth_calibration_dir": depth_calibration,
        "depth_runtime_lineage_json": (
            depth_calibration / DEPTH_RUNTIME_LINEAGE_FILENAME
        ),
        "eef_consumed_depth_samples_npy": (
            depth_calibration / EEF_CONSUMED_DEPTH_SAMPLES_FILENAME
        ),
        "visualization_dir": root / VISUALIZATION_DIRECTORY,
        "trajectory_manifest": root / ASSETS_MANIFEST_FILENAME,
    }


def execution_artifact_paths(execution_root: str | Path) -> Dict[str, Path]:
    """Resolve artifact roles relative to one explicit ``exec`` directory."""

    root = _root(execution_root)
    vlm = root / VLM_DIRECTORY
    return {
        "exec_dir": root,
        "exec_summary": root / EXECUTION_SUMMARY_FILENAME,
        "checkpoint_trace": root / CHECKPOINT_TRACE_FILENAME,
        "dense_tcp_trace": root / DENSE_TCP_TRACE_FILENAME,
        "exec_metrics": root / EXECUTION_METRICS_FILENAME,
        "exec_metrics_per_frame": (root / EXECUTION_METRICS_PER_FRAME_FILENAME),
        "execution_inputs": root / EXECUTION_INPUTS_FILENAME,
        "action_trace": root / ACTION_TRACE_FILENAME,
        "execution_video": root / EXECUTION_VIDEO_FILENAME,
        "evaluation_result": root / EVALUATION_RESULT_FILENAME,
        "task_success": root / TASK_SUCCESS_FILENAME,
        "execution_manifest": root / ASSETS_MANIFEST_FILENAME,
        "vlm_root": vlm,
        "vlm_predictions_root": vlm / VLM_PREDICTIONS_DIRECTORY,
    }


__all__ = [
    "ACTION_ARRAY_FILENAME",
    "ACTION_TRACE_FILENAME",
    "ACTION_FILENAME",
    "ASSETS_MANIFEST_FILENAME",
    "CHECKPOINT_TRACE_FILENAME",
    "DENSE_TCP_TRACE_FILENAME",
    "DEPTH_MANIFEST_FILENAME",
    "DEPTH_META_NPY_FILENAME",
    "DEPTH_NPY_FILENAME",
    "DEPTH_VIDEO_FILENAME",
    "EVALUATION_RESULT_FILENAME",
    "EXECUTION_METRICS_FILENAME",
    "EXECUTION_METRICS_PER_FRAME_FILENAME",
    "EXECUTION_INPUTS_FILENAME",
    "EXECUTION_VIDEO_FILENAME",
    "EXECUTION_SUMMARY_FILENAME",
    "EXECUTION_CONFIG_FILENAME",
    "GRIPPER_FILENAME",
    "INITIALIZATION_CONFIG_FILENAME",
    "INITIALIZATION_DEPTH_BUFFER_FILENAME",
    "INITIALIZATION_DEPTH_FILENAME",
    "INITIALIZATION_DEPTH_IMAGE_FILENAME",
    "INITIALIZATION_INSTANCE_SEGMENTATION_FILENAME",
    "INITIALIZATION_SEGMENTATION_MAPPINGS_FILENAME",
    "INSTANCE_NAMES_FILENAME",
    "OBJECT_TRAJECTORIES_FILENAME",
    "PIPELINE_STATE_FILENAME",
    "SAMPLE_METADATA_FILENAME",
    "SCENE_OVERRIDE_FILENAME",
    "SIMULATOR_CONFIG_FILENAME",
    "TASK_SUCCESS_FILENAME",
    "TRAJECTORY_CONFIG_FILENAME",
    "UNION_TRAJECTORY_FILENAME",
    "RUN_OVERRIDES_FILENAME",
    "GT_DEPTH_METADATA_FILENAME",
    "GT_METRIC_DEPTH_CONTACT_FILENAME",
    "GT_METRIC_DEPTH_FRAME0_FILENAME",
    "GT_METRIC_DEPTH_VIDEO_FILENAME",
    "GT_VIDEO_FILENAME",
    "execution_artifact_paths",
    "benchmark_suite_artifact_paths",
    "bench_initialization_path_context",
    "bench_layout_paths",
    "bench_sample_path",
    "generated_artifact_paths",
    "ground_truth_artifact_paths",
    "initialization_artifact_paths",
    "pipeline_run_config_path",
    "run_artifact_paths",
    "run_artifact_paths_for_sample",
    "sample_artifact_paths",
    "sample_scaffold_directories",
    "sample_scaffold_relative_directories",
    "trajectory_artifact_paths",
]
