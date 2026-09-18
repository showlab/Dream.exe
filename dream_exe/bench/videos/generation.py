"""Generate one benchmark video into work and publish it as an output."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from ...generation.video import ImageToVideoBackend, generate_video
from ..outputs.lifecycle import register_video_output, sha256_file
from ..data.repository import BenchRepository
from ..contracts.schemas import (
    VIDEO_OUTPUT_SCHEMA,
    canonical_sha256,
    validate_document,
)
from ..data.workspace import Workspace
from .preprocess import preprocess_video_for_video2traj


_SAFE_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,191}\Z")


def _safe_id(value: str, label: str) -> str:
    clean = str(value or "").strip()
    if not _SAFE_ID.fullmatch(clean):
        raise ValueError(f"{label} must be a safe identifier")
    return clean


def _prompt(generation: Mapping[str, Any], variant: str) -> str:
    if variant not in {"standard", "enhanced"}:
        raise ValueError("generated benchmark videos use standard or enhanced")
    prompts = generation["prompts"]
    components = generation["variants"][variant]
    parts = [str(prompts[name]).strip() for name in components]
    prompt = "\n".join(part for part in parts if part)
    if not prompt:
        raise ValueError(f"generation prompt is empty for variant {variant}")
    return prompt


def resolve_video_generation_request(
    *,
    workspace: Workspace,
    uid: str,
    model_id: str,
    prompt_variant: str,
    seed: int,
    parameters: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Resolve the immutable first frame and prompt into one work request."""

    clean_uid = _safe_id(uid, "case")
    clean_model = _safe_id(model_id, "model")
    if isinstance(seed, bool) or not isinstance(seed, int) or seed < 0:
        raise ValueError("seed must be a non-negative integer")
    repository = BenchRepository(workspace.bench_root)
    repository.load_case(clean_uid)
    generation = repository.load_generation_input(clean_uid)
    variant = str(prompt_variant)
    prompt = _prompt(generation, variant)
    generation_root = repository.case_dir(clean_uid) / "generation"
    first_frame = generation_root / generation["first_frame"]["path"]
    work_root = (
        workspace.work_root
        / "video-generation"
        / clean_uid
        / clean_model
        / variant
    )
    return {
        "uid": clean_uid,
        "model_id": clean_model,
        "prompt_variant": variant,
        "first_frame": first_frame,
        "prompt": prompt,
        "prompt_sha256": canonical_sha256({"prompt": prompt}),
        "generation_input_sha256": canonical_sha256(generation),
        "output_contract": dict(generation["output_contract"]),
        "seed": seed,
        "parameters": dict(parameters or {}),
        "work_video": work_root / "video.mp4",
        "work_preprocessed": work_root / "preprocessed.mp4",
        "work_sidecar": work_root / "generation.json",
    }


def generate_video_output(
    *,
    workspace: Workspace,
    uid: str,
    model_id: str,
    prompt_variant: str,
    backend: ImageToVideoBackend | None,
    backend_name: str,
    seed: int = 0,
    parameters: Mapping[str, Any] | None = None,
    producer_revision: str = "",
    force_work: bool = False,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Generate in ``work`` and atomically register under ``results/videos``.

    The producer never writes into the immutable benchmark.  The generated
    MP4 is first validated in the mutable work root, then the normal video
    lifecycle moves it to the case/model/prompt-variant output slot.  The
    provider sidecar remains in work because it may contain machine-local
    diagnostic paths; ``video.json`` contains only portable provenance.
    """

    request = resolve_video_generation_request(
        workspace=workspace,
        uid=uid,
        model_id=model_id,
        prompt_variant=prompt_variant,
        seed=seed,
        parameters=parameters,
    )
    destination = (
        workspace.outputs_root
        / "videos"
        / request["uid"]
        / request["model_id"]
        / request["prompt_variant"]
    )
    if destination.exists():
        raise FileExistsError(f"video output already exists: {destination}")

    generated = generate_video(
        image_path=request["first_frame"],
        prompt=request["prompt"],
        output_video=request["work_video"],
        output_sidecar=request["work_sidecar"],
        backend=backend,
        backend_name=backend_name,
        seed=request["seed"],
        parameters=request["parameters"],
        force=force_work,
        dry_run=dry_run,
    )
    run_input = {
        "kind": "generated",
        "model_id": request["model_id"],
        "prompt_variant": request["prompt_variant"],
        "reference_id": None,
    }
    if dry_run:
        preprocessing = preprocess_video_for_video2traj(
            request["work_video"],
            request["work_preprocessed"],
            output_contract=request["output_contract"],
            dry_run=True,
        )
        return {
            "status": "planned",
            "case": request["uid"],
            "model_id": request["model_id"],
            "prompt_variant": request["prompt_variant"],
            "generation_input_sha256": request["generation_input_sha256"],
            "prompt_sha256": request["prompt_sha256"],
            "first_frame_sha256": generated["image_sha256"],
            "work_video": request["work_video"].as_posix(),
            "work_preprocessed": request["work_preprocessed"].as_posix(),
            "preprocessing": preprocessing,
            "destination": destination.as_posix(),
            "run_input": run_input,
        }

    work_video = request["work_video"]
    work_preprocessed = request["work_preprocessed"]
    work_sidecar = request["work_sidecar"]
    preprocessing = preprocess_video_for_video2traj(
        work_video,
        work_preprocessed,
        output_contract=request["output_contract"],
        overwrite=True,
    )
    identity = generated["backend_identity"]
    revision = str(producer_revision or identity.get("revision", "")).strip()
    manifest = validate_document(
        {
            "format": VIDEO_OUTPUT_SCHEMA,
            "uid": request["uid"],
            "model_id": request["model_id"],
            "prompt_variant": request["prompt_variant"],
            "producer": {
                "kind": "generator",
                "name": str(backend_name).strip(),
                "revision": revision,
                "seed": request["seed"],
            },
            "generation_input_sha256": request["generation_input_sha256"],
            "prompt_sha256": request["prompt_sha256"],
            "video": {
                "path": "video.mp4",
                "sha256": sha256_file(work_video),
                "size": work_video.stat().st_size,
            },
            "preprocessed": {
                "path": "preprocessed.mp4",
                "sha256": sha256_file(work_preprocessed),
                "size": work_preprocessed.stat().st_size,
            },
            "rights": {
                "status": "unknown",
                "license": "",
                "redistributable": None,
            },
            "provenance_status": "complete",
            "origin": {
                "kind": "local_image_to_video",
                "backend": str(backend_name).strip(),
                "backend_identity_sha256": canonical_sha256(identity),
                "generation_sidecar_sha256": sha256_file(work_sidecar),
                "output_contract": request["output_contract"],
                "preprocessing": preprocessing["spec"],
            },
        },
        expected_schema=VIDEO_OUTPUT_SCHEMA,
    )
    published = register_video_output(
        workspace=workspace,
        manifest=manifest,
        source_paths={
            "video": work_video,
            "preprocessed": work_preprocessed,
        },
    )
    return {
        **published,
        "generation_sidecar": work_sidecar.as_posix(),
        "run_input": run_input,
    }


__all__ = [
    "generate_video_output",
    "resolve_video_generation_request",
]
