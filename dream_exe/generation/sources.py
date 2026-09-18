"""Provider-neutral import and normalization of generated video artifacts.

This module knows source folders and explicit destination callbacks, but not a
bench root.  Bench-compatible storage remains the responsibility of the
central I/O boundary.
"""

from __future__ import annotations

from fractions import Fraction
import json
import os
import shutil
import subprocess
import tempfile
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any, Dict, Tuple


MODEL_OUTPUT_FILENAMES: Dict[str, str] = {
    "kling_v3": "Kling3.0.mp4",
    "kling3": "Kling3.0.mp4",
    "kling3.0": "Kling3.0.mp4",
    "kling_3_0": "Kling3.0.mp4",
    "seedance": "Seedance2.0.mp4",
    "seedance2": "Seedance2.0.mp4",
    "seedance2.0": "Seedance2.0.mp4",
    "seedance_2_0": "Seedance2.0.mp4",
    "veo31": "Veo3.1.mp4",
    "veo3.1": "Veo3.1.mp4",
    "veo_3_1": "Veo3.1.mp4",
    "wan": "Wan2.7.mp4",
    "wan27": "Wan2.7.mp4",
    "wan2.7": "Wan2.7.mp4",
    "wan_2_7": "Wan2.7.mp4",
}


def clean_model_key(value: str) -> str:
    """Normalize the aliases exercised by the current local importer."""

    return str(value or "").strip().lower().replace("-", "_").replace(" ", "_")


def _model_leaf(value: Any) -> str:
    leaf = Path(str(value or "").strip()).name
    if leaf.casefold().endswith(".mp4"):
        leaf = leaf[:-4]
    for marker in (".old", ".ori"):
        if leaf.casefold().endswith(marker):
            leaf = leaf[: -len(marker)]
    return leaf.strip()


def normalize_generated_model_name(name: str) -> str:
    return _model_leaf(name)


def infer_generated_model_from_video_path(path: str | Path) -> str:
    """Infer the current model identity encoded by a generated-video name."""

    if not str(path or "").strip():
        return ""
    model = normalize_generated_model_name(str(path))
    return model if model != "rollout" else ""


def resolve_optional_runtime_path(
    path: str | Path,
    *,
    dataset_config_path: str | Path,
    reference_paths: Iterable[str | Path] = (),
    runtime_root: str | Path | None = None,
) -> str:
    """Resolve a configured path with the current search precedence.

    ``runtime_root`` replaces the current implementation's implicit repository
    root.  Supplying the repository root reproduces current behavior while
    keeping this module installable and independent of a source checkout.
    """

    path_text = str(path or "").strip()
    if not path_text:
        return ""

    candidate = Path(path_text)
    candidates = [candidate]
    if not candidate.is_absolute():
        reference_dirs = [Path(dataset_config_path).resolve().parent]
        for reference_path in reference_paths:
            reference_text = str(reference_path or "").strip()
            if not reference_text:
                continue
            reference = Path(reference_text).expanduser()
            reference_dirs.append(
                (reference if reference.is_dir() else reference.parent).resolve()
            )
        seen: set[str] = set()
        for reference_dir in reference_dirs:
            reference_key = reference_dir.as_posix()
            if reference_key in seen:
                continue
            seen.add(reference_key)
            candidates.append(reference_dir / candidate)
        if runtime_root is not None:
            candidates.append(Path(runtime_root).expanduser().resolve() / candidate)

    for item in candidates:
        if item.exists():
            return item.resolve().as_posix()
    return candidates[0].as_posix()


def generated_video_roots(
    gen_dir: str | Path,
    *,
    include_enhanced: bool = True,
) -> list[tuple[Path, bool]]:
    """Return current normal/enhanced generated-video search roots."""

    root = Path(gen_dir).expanduser().resolve()
    roots: list[Path] = [root]
    if root.name == "processed":
        roots.append(root.parent / "raw")
        roots.append(root.parent)
    output = [(item, False) for item in roots]
    if include_enhanced:
        artifacts_dir = (
            root.parent.parent if root.name in {"processed", "raw"} else root.parent
        )
        enhanced_processed = artifacts_dir / "gen_enhanced" / "processed"
        output.extend(
            [
                (enhanced_processed, True),
                (enhanced_processed.parent / "raw", True),
                (enhanced_processed.parent, True),
            ]
        )
    return output


def list_available_generated_models(gen_dir: str | Path) -> list[str]:
    """List current model identities, including ``-enhanced`` variants."""

    models: set[str] = set()
    for candidate_root, enhanced in generated_video_roots(
        gen_dir,
        include_enhanced=True,
    ):
        if not candidate_root.exists():
            continue
        for path in candidate_root.glob("*.mp4"):
            if not path.is_file():
                continue
            model = normalize_generated_model_name(path.name)
            if enhanced:
                model = f"{model}-enhanced"
            models.add(model)
    return sorted(models)


def select_generated_video(
    *,
    gen_dir: str | Path,
    dataset_config_path: str | Path,
    pipeline_config: dict[str, Any],
    explicit_video_path: str | Path = "",
    requested_model: str = "",
    runtime_root: str | Path | None = None,
) -> tuple[str, str]:
    """Resolve one generated video using the current selection contract.

    The caller supplies the semantic generated-video directory.  No bench root
    or repository layout is discovered here.
    """

    requested = normalize_generated_model_name(requested_model)
    pipeline_source = pipeline_config.get("_meta", {}).get("source", "")
    explicit = resolve_optional_runtime_path(
        explicit_video_path,
        dataset_config_path=dataset_config_path,
        reference_paths=[pipeline_source],
        runtime_root=runtime_root,
    )
    if explicit:
        if not Path(explicit).exists():
            raise FileNotFoundError(
                f"[traj][FATAL] explicit generated video not found: {explicit}"
            )
        inferred = requested or infer_generated_model_from_video_path(explicit)
        if not inferred:
            raise ValueError(
                "[traj][FATAL] could not infer gen_model from generated "
                f"video path: {explicit}"
            )
        return Path(explicit).resolve().as_posix(), inferred

    configured = resolve_optional_runtime_path(
        pipeline_config.get("input", {}).get("gen_video_path", ""),
        dataset_config_path=dataset_config_path,
        reference_paths=[pipeline_source],
        runtime_root=runtime_root,
    )
    if configured and Path(configured).exists():
        configured_model = infer_generated_model_from_video_path(configured)
        if requested and configured_model and requested == configured_model:
            return Path(configured).resolve().as_posix(), requested
        if configured_model and not requested:
            return Path(configured).resolve().as_posix(), configured_model

    wants_enhanced = requested.lower().endswith("-enhanced")
    candidates: list[tuple[Path, bool]] = []
    for candidate_root, enhanced in generated_video_roots(
        gen_dir,
        include_enhanced=True,
    ):
        if requested and bool(enhanced) != wants_enhanced:
            continue
        if candidate_root.exists():
            for path in sorted(candidate_root.glob("*.mp4")):
                if path.is_file():
                    candidates.append((path, enhanced))

    if requested:
        base_model = (
            requested[:-9] if requested.lower().endswith("-enhanced") else requested
        )
        for candidate, enhanced in candidates:
            candidate_model = normalize_generated_model_name(candidate.name)
            if enhanced:
                candidate_model = f"{candidate_model}-enhanced"
            if (
                candidate_model == requested
                or normalize_generated_model_name(candidate.name) == base_model
            ):
                return candidate.resolve().as_posix(), requested
        available = ", ".join(list_available_generated_models(gen_dir)) or "<none>"
        raise FileNotFoundError(
            "[traj][FATAL] generated video for "
            f"gen_model={requested} not found under "
            f"{Path(gen_dir).expanduser().resolve()}. "
            f"Available models: {available}"
        )

    if len(candidates) == 1:
        candidate, enhanced = candidates[0]
        model = normalize_generated_model_name(candidate.name)
        if enhanced:
            model = f"{model}-enhanced"
        return candidate.resolve().as_posix(), model

    available = ", ".join(list_available_generated_models(gen_dir)) or "<none>"
    raise ValueError(
        "[traj][FATAL] multiple generated videos found under "
        f"{Path(gen_dir).expanduser().resolve()}. "
        "Please pass --gen_model or --gen_video_path. "
        f"Available models: {available}"
    )


def resolve_input_video(
    *,
    gen_dir: str | Path,
    gt_dir: str | Path,
    env_dir: str | Path,
    dataset_config_path: str | Path,
    pipeline_config: dict[str, Any],
    default_input_video_path: str | Path,
    runtime_root: str | Path | None = None,
) -> str:
    """Resolve GT-reference, generated, or custom video with current precedence."""

    pipeline_source = str(
        pipeline_config.get("_meta", {}).get("source", "") or ""
    ).strip()
    input_config = dict(pipeline_config.get("input", {}))
    selected_video = (
        str(input_config.get("selected_video", "rollout") or "rollout").strip().lower()
    )
    configured = resolve_optional_runtime_path(
        input_config.get("video_path", ""),
        dataset_config_path=dataset_config_path,
        reference_paths=[pipeline_source],
        runtime_root=runtime_root,
    )
    if configured and Path(configured).exists():
        return configured

    rollout_video_path = resolve_optional_runtime_path(
        input_config.get("rollout_video_path", ""),
        dataset_config_path=dataset_config_path,
        reference_paths=[pipeline_source],
        runtime_root=runtime_root,
    )
    generated_video_path = resolve_optional_runtime_path(
        input_config.get("gen_video_path", ""),
        dataset_config_path=dataset_config_path,
        reference_paths=[pipeline_source],
        runtime_root=runtime_root,
    )

    default_video = Path(default_input_video_path).expanduser().resolve()
    rollout_candidates = [
        Path(rollout_video_path) if rollout_video_path else None,
        default_video,
        default_video.parent / "video" / default_video.name,
        default_video.parent / "video" / "gt.mp4",
        Path(gt_dir).expanduser().resolve() / "video" / "gt.mp4",
        Path(env_dir).expanduser().resolve() / "video" / "gt.mp4",
    ]
    generated_dir = Path(gen_dir).expanduser().resolve()
    generated_candidates = [
        Path(generated_video_path) if generated_video_path else None,
        generated_dir / "video" / "generated.mp4",
        generated_dir / "generated.mp4",
        generated_dir / "video" / "gen.mp4",
        generated_dir / "gen.mp4",
    ]

    if selected_video == "gen":
        candidates = [path for path in generated_candidates if path is not None]
    elif selected_video == "custom":
        candidates = []
    else:
        candidates = [path for path in rollout_candidates if path is not None]

    for candidate in candidates:
        if candidate.exists():
            return candidate.as_posix()

    if selected_video == "gen":
        mp4_candidates = sorted(generated_dir.glob("*.mp4"))
        mp4_candidates += sorted((generated_dir / "video").glob("*.mp4"))
    else:
        mp4_candidates = sorted(default_video.parent.glob("*.mp4"))
        if len(mp4_candidates) == 1:
            return mp4_candidates[0].resolve().as_posix()

        nested_candidates = sorted(default_video.parent.glob("video/*.mp4"))
        if len(nested_candidates) == 1:
            return nested_candidates[0].resolve().as_posix()

    if len(mp4_candidates) == 1:
        return mp4_candidates[0].resolve().as_posix()

    raise FileNotFoundError(
        "[traj][FATAL] input video not found for "
        f"uid={default_video.parent.name}. Expected pipeline "
        f"input.video_path / input.{selected_video}_video_path "
        f"or default {default_video.as_posix()}"
    )


def iter_generated_video_files(
    source_root: str | Path,
    *,
    model: str = "",
) -> Iterable[Tuple[Path, str]]:
    """Discover current flat or per-model MP4 source layouts."""

    root = Path(source_root)
    if model:
        model_dir = root / model
        if model_dir.is_dir():
            for path in sorted(model_dir.glob("*.mp4")):
                if path.is_file():
                    yield path, model
            return
        for path in sorted(root.glob("*.mp4")):
            if path.is_file():
                yield path, model
        return

    for path in sorted(root.glob("*/*.mp4")):
        if path.is_file():
            yield path, path.parent.name


def uid_from_video_path(path: str | Path) -> str:
    return Path(path).stem.strip()


def import_generated_videos(
    *,
    source_root: str | Path,
    destination_for: Callable[[str, str], str | Path],
    sample_exists: Callable[[str], bool] | None = None,
    model: str = "",
    model_output_filenames: Dict[str, str] | None = None,
    move: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Import generated videos through explicit semantic destination mapping.

    ``destination_for(uid, output_filename)`` is supplied by the bench or user
    workspace adapter.  Copy is the default so provider outputs remain intact.
    """

    root = Path(source_root).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(f"source root not found: {root}")
    output_names = dict(
        MODEL_OUTPUT_FILENAMES
        if model_output_filenames is None
        else model_output_filenames
    )
    exists = sample_exists or (lambda _uid: True)

    imported = []
    skipped = []
    errors = []
    for source, raw_model in iter_generated_video_files(root, model=model):
        model_key = clean_model_key(raw_model)
        output_filename = output_names.get(model_key, "")
        uid = uid_from_video_path(source)
        if not output_filename:
            skipped.append(
                {
                    "source": source.as_posix(),
                    "reason": f"unknown model folder/name: {raw_model}",
                }
            )
            continue
        if not uid:
            skipped.append(
                {
                    "source": source.as_posix(),
                    "reason": "empty uid filename",
                }
            )
            continue
        if not exists(uid):
            skipped.append(
                {
                    "source": source.as_posix(),
                    "uid": uid,
                    "reason": "missing destination sample",
                }
            )
            continue

        destination = Path(destination_for(uid, output_filename)).expanduser()
        if destination.exists() and not overwrite:
            skipped.append(
                {
                    "source": source.as_posix(),
                    "dest": destination.as_posix(),
                    "uid": uid,
                    "reason": "destination exists",
                }
            )
            continue

        imported.append(
            {
                "source": source.as_posix(),
                "dest": destination.as_posix(),
                "uid": uid,
                "model": output_filename,
            }
        )
        if dry_run:
            continue
        try:
            destination.parent.mkdir(parents=True, exist_ok=True)
            if move:
                shutil.move(source.as_posix(), destination.as_posix())
            else:
                shutil.copy2(source.as_posix(), destination.as_posix())
        except Exception as exc:
            errors.append(
                {
                    "source": source.as_posix(),
                    "dest": destination.as_posix(),
                    "uid": uid,
                    "error": str(exc),
                }
            )

    return {
        "source_root": root.as_posix(),
        "mode": "move" if move else "copy",
        "overwrite": overwrite,
        "dry_run": dry_run,
        "imported_count": len(imported) - len(errors),
        "skipped_count": len(skipped),
        "error_count": len(errors),
        "imported": imported,
        "skipped": skipped,
        "errors": errors,
    }


_SQUARE_VIDEO_FILTERS = {
    "resize": "scale={side}:{side}",
    "center_crop": "crop=min(iw\\,ih):min(iw\\,ih),scale={side}:{side}",
}


def video_filter_expression(*, mode: str, size: int) -> str:
    """Render one supported square-normalization filter for ffmpeg."""

    template = _SQUARE_VIDEO_FILTERS.get(mode) if isinstance(mode, str) else None
    if template is None:
        raise ValueError(f"unsupported mode: {mode}")
    return template.format(side=size)


def build_video_normalization_command(
    source: str | Path,
    destination: str | Path,
    *,
    mode: str,
    size: int,
    fps: str = "",
    overwrite: bool = False,
) -> list[str]:
    """Build the current ffmpeg command without running it."""

    source_path = Path(source)
    destination_path = Path(destination)
    command = [
        "ffmpeg",
        "-hide_banner",
        "-loglevel",
        "error",
        "-y" if overwrite else "-n",
        "-i",
        source_path.as_posix(),
        "-vf",
        video_filter_expression(mode=mode, size=size),
    ]
    if str(fps or "").strip():
        command.extend(["-r", str(fps).strip()])
    command.extend(
        [
            "-an",
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            destination_path.as_posix(),
        ]
    )
    return command


def build_video_probe_command(path: str | Path) -> list[str]:
    """Build the bounded ffprobe command used before publishing an MP4."""

    return [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        (
            "stream=codec_name,width,height,pix_fmt,avg_frame_rate,"
            "duration,nb_frames:format=duration"
        ),
        "-of",
        "json",
        Path(path).as_posix(),
    ]


def _process_error(process: Any) -> str:
    return str(process.stderr or process.stdout or "").strip()


def _positive_float(value: Any, *, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"normalized video has invalid {label}") from error
    if not number > 0.0:
        raise ValueError(f"normalized video must have positive {label}")
    return number


def _video_rate(value: Any) -> Fraction:
    try:
        rate = Fraction(str(value or ""))
    except (ValueError, ZeroDivisionError) as error:
        raise ValueError("normalized video has invalid frame rate") from error
    if rate <= 0:
        raise ValueError("normalized video must have positive frame rate")
    return rate


def inspect_normalized_video(
    path: str | Path,
    *,
    size: int,
    fps: str = "",
    run: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    """Validate one staged normalized MP4 before it becomes visible."""

    candidate = Path(path)
    process = run(
        build_video_probe_command(candidate),
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if process.returncode != 0:
        raise ValueError(f"ffprobe failed: {_process_error(process)}")
    try:
        document = json.loads(str(process.stdout or ""))
    except json.JSONDecodeError as error:
        raise ValueError("ffprobe returned invalid JSON") from error
    streams = document.get("streams") if isinstance(document, dict) else None
    if not isinstance(streams, list) or len(streams) != 1:
        raise ValueError("normalized video must contain exactly one selected stream")
    stream = streams[0]
    if not isinstance(stream, dict):
        raise ValueError("normalized video stream metadata is invalid")
    width = int(stream.get("width", 0) or 0)
    height = int(stream.get("height", 0) or 0)
    if (width, height) != (int(size), int(size)):
        raise ValueError(
            "normalized video dimensions do not match the requested square size"
        )
    codec = str(stream.get("codec_name", "") or "").strip().lower()
    if codec != "h264":
        raise ValueError("normalized video codec must be h264")
    pixel_format = str(stream.get("pix_fmt", "") or "").strip().lower()
    if pixel_format != "yuv420p":
        raise ValueError("normalized video pixel format must be yuv420p")
    rate = _video_rate(stream.get("avg_frame_rate"))
    requested_fps = str(fps or "").strip()
    if requested_fps:
        expected_rate = _video_rate(requested_fps)
        if abs(float(rate - expected_rate)) > 1e-6:
            raise ValueError("normalized video frame rate does not match requested fps")
    format_meta = document.get("format")
    format_duration = (
        format_meta.get("duration") if isinstance(format_meta, dict) else None
    )
    duration = _positive_float(
        stream.get("duration") or format_duration,
        label="duration",
    )
    frames_value = stream.get("nb_frames")
    frames = None
    if str(frames_value or "").strip() not in {"", "N/A"}:
        frames = int(frames_value)
        if frames <= 0:
            raise ValueError("normalized video must contain at least one frame")
    version = run(
        ["ffmpeg", "-version"],
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    if version.returncode != 0:
        raise ValueError(f"ffmpeg version query failed: {_process_error(version)}")
    version_line = str(version.stdout or version.stderr or "").splitlines()
    if not version_line:
        raise ValueError("ffmpeg version query returned no identity")
    return {
        "codec": codec,
        "pixel_format": pixel_format,
        "width": width,
        "height": height,
        "average_frame_rate": str(rate),
        "duration_seconds": duration,
        "frame_count": frames,
        "ffmpeg_version": version_line[0].strip(),
    }


def _sync_file(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _sync_directory(path: Path) -> None:
    flags = os.O_RDONLY | getattr(os, "O_DIRECTORY", 0)
    descriptor = os.open(path, flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _publish_normalized_video(
    staged: Path,
    destination: Path,
    *,
    overwrite: bool,
) -> None:
    _sync_file(staged)
    if overwrite:
        os.replace(staged, destination)
    else:
        try:
            os.link(staged, destination, follow_symlinks=False)
        except FileExistsError as error:
            raise FileExistsError(
                "destination appeared while normalized video was staged"
            ) from error
        staged.unlink()
    _sync_directory(destination.parent)


def normalize_generated_video(
    source: str | Path,
    destination: str | Path,
    *,
    mode: str,
    size: int,
    fps: str = "",
    overwrite: bool = False,
    dry_run: bool = False,
    run: Callable[..., Any] = subprocess.run,
) -> Dict[str, Any]:
    """Normalize, validate, and atomically publish one generated video."""

    source_path = Path(source)
    destination_path = Path(destination)
    if destination_path.exists() and not overwrite:
        return {
            "source": source_path.as_posix(),
            "dest": destination_path.as_posix(),
            "status": "skip_exists",
        }
    visible_command = build_video_normalization_command(
        source_path,
        destination_path,
        mode=mode,
        size=size,
        fps=fps,
        overwrite=overwrite,
    )
    if dry_run:
        return {
            "source": source_path.as_posix(),
            "dest": destination_path.as_posix(),
            "status": "dry_run",
            "cmd": " ".join(visible_command),
        }
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, staged_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".tmp.mp4",
        dir=destination_path.parent,
    )
    os.close(descriptor)
    staged = Path(staged_name)
    staged.unlink()
    try:
        command = build_video_normalization_command(
            source_path,
            staged,
            mode=mode,
            size=size,
            fps=fps,
            overwrite=True,
        )
        process = run(
            command,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        if process.returncode != 0:
            return {
                "source": source_path.as_posix(),
                "dest": destination_path.as_posix(),
                "status": "error",
                "error": _process_error(process),
            }
        if staged.is_symlink() or not staged.is_file() or staged.stat().st_size <= 0:
            raise ValueError("ffmpeg did not produce a non-empty ordinary video")
        media = inspect_normalized_video(
            staged,
            size=int(size),
            fps=fps,
            run=run,
        )
        _publish_normalized_video(
            staged,
            destination_path,
            overwrite=overwrite,
        )
    except Exception as error:
        return {
            "source": source_path.as_posix(),
            "dest": destination_path.as_posix(),
            "status": "error",
            "error_type": type(error).__name__,
            "error": str(error),
        }
    finally:
        if staged.exists() or staged.is_symlink():
            staged.unlink()
    return {
        "source": source_path.as_posix(),
        "dest": destination_path.as_posix(),
        "status": "processed",
        "media": media,
        "publication": "atomic_replace" if overwrite else "atomic_create",
    }


__all__ = [
    "MODEL_OUTPUT_FILENAMES",
    "build_video_normalization_command",
    "build_video_probe_command",
    "clean_model_key",
    "generated_video_roots",
    "import_generated_videos",
    "infer_generated_model_from_video_path",
    "iter_generated_video_files",
    "list_available_generated_models",
    "normalize_generated_model_name",
    "normalize_generated_video",
    "inspect_normalized_video",
    "resolve_input_video",
    "resolve_optional_runtime_path",
    "select_generated_video",
    "uid_from_video_path",
    "video_filter_expression",
]
