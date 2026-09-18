"""Provider-independent VLM evaluation over explicit saved image artifacts.

The paper evaluation scripts sort grid-image names, resolve prompt metadata by
exact/longest-prefix match, evaluate each item independently, and use ordered
``Pool.imap`` results when writing CSV rows.  This module preserves those
observable rules while accepting explicit media records and an injected
inference callable.  It performs no bench, registry, provider, or credential
discovery.

Raw prediction records are optional append-only evidence.  Cache reuse is
fingerprinted by media bytes and behavior-affecting inputs.  Forced runs never
overwrite prior raw evidence.
"""

from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import warnings
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any

from .scoring import (
    build_prompt_by_name_map,
    create_physical_plausibility_prompt,
    create_subject_stability_prompts,
    create_task_adherence_prompt,
    evaluate_score_media,
    evaluate_subject_stability_media,
    grid_image_to_vid_id,
    load_prediction_record,
    resolve_prompt_info,
    save_results_to_csv,
    save_stability_results_to_csv,
    write_prediction_record,
)

RUBRIC_SUBJECT_STABILITY = "subject_stability"
RUBRIC_PHYSICAL_PLAUSIBILITY = "physical_plausibility"
RUBRIC_TASK_ADHERENCE = "task_adherence"
LEGACY_RUBRIC_TASK_ADHERENCE = "task_adherence_consistency"
SAVED_MEDIA_RUBRICS = (
    RUBRIC_SUBJECT_STABILITY,
    RUBRIC_PHYSICAL_PLAUSIBILITY,
    RUBRIC_TASK_ADHERENCE,
    LEGACY_RUBRIC_TASK_ADHERENCE,
)

_GRID_IMAGE_EXTENSIONS = (
    ".jpeg",
    ".jpg",
    ".png",
    ".webp",
    ".bmp",
)
_CACHE_SCHEMA = "dream-exe.vlm-cache"
_BATCH_SCHEMA = "dream-exe.saved-media-vlm-batch"
_PREDICTION_SCHEMA = "dream-exe.vlm-prediction"
_CSV_SUFFIXES = {
    RUBRIC_SUBJECT_STABILITY: "1_robot_subject_stability",
    RUBRIC_PHYSICAL_PLAUSIBILITY: "2_physical_plausibility",
    RUBRIC_TASK_ADHERENCE: "3_task_adherence_consistency",
}
_SENSITIVE_OPTION_NAMES = {
    "api_key",
    "api_token",
    "access_token",
    "auth_token",
    "bearer_token",
    "token",
    "password",
    "secret",
    "client_secret",
    "authorization",
    "credential",
    "credentials",
    "headers",
    "extra_headers",
    "default_headers",
    "client",
    "http_client",
    "base_url",
}
_DATA_URL_PATTERN = re.compile(r"data:image/[a-zA-Z0-9.+-]+;base64,[a-zA-Z0-9+/=_-]+")
_NAMED_CREDENTIAL_PATTERN = re.compile(
    (
        r"(?i)\b(api[ _-]?key|access[ _-]?token|"
        r"refresh[ _-]?token|authorization|password|"
        r"client[ _-]?secret|credential)"
        r"(\s*[:=]\s*)([^\s,;]+)"
    )
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")

InferenceCallable = Callable[
    [str, Path, Mapping[str, Any]],
    str,
]
ExecutorFactory = Callable[..., Any]


def _normalize_rubric(rubric: str) -> str:
    clean = str(rubric or "").strip().lower().replace("-", "_")
    aliases = {
        "robot_subject_stability": RUBRIC_SUBJECT_STABILITY,
        LEGACY_RUBRIC_TASK_ADHERENCE: RUBRIC_TASK_ADHERENCE,
    }
    clean = aliases.get(clean, clean)
    if clean not in SAVED_MEDIA_RUBRICS:
        raise ValueError(f"unsupported saved-media VLM rubric: {rubric}")
    return clean


def paper_vlm_csv_filename(
    *,
    video_path: str | os.PathLike[str],
    rubric: str,
    timestamp: str,
) -> str:
    """Return the timestamped paper VLM CSV basename."""

    clean_rubric = _normalize_rubric(rubric)
    clean_timestamp = str(timestamp or "").strip()
    if not clean_timestamp:
        raise ValueError("timestamp is required")
    video_dir_tag = os.path.basename(os.path.normpath(str(video_path))) or "videos"
    video_dir_tag = video_dir_tag.strip().replace("/", "_").replace(" ", "_")
    return f"{video_dir_tag}_{_CSV_SUFFIXES[clean_rubric]}_{clean_timestamp}.csv"


def _normalize_option_name(value: Any) -> str:
    return re.sub(
        r"[^a-z0-9]+",
        "_",
        str(value or "").strip().lower(),
    ).strip("_")


def _sensitive_option_paths(
    value: Any,
    *,
    prefix: str = "",
    seen: set[int] | None = None,
) -> list[str]:
    visited = seen if seen is not None else set()
    if isinstance(value, Mapping):
        identity = id(value)
        if identity in visited:
            return []
        visited.add(identity)
        paths = []
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            normalized = _normalize_option_name(key)
            collapsed = normalized.replace("_", "")
            if (
                normalized in _SENSITIVE_OPTION_NAMES
                or collapsed
                in {
                    "apikey",
                    "apitoken",
                    "accesstoken",
                    "authtoken",
                    "bearertoken",
                    "clientsecret",
                    "authorization",
                    "extraheaders",
                    "defaultheaders",
                    "httpclient",
                    "baseurl",
                }
                or normalized.endswith(
                    (
                        "_api_key",
                        "_access_token",
                        "_auth_token",
                        "_secret",
                        "_password",
                        "_headers",
                    )
                )
            ):
                paths.append(path)
            paths.extend(
                _sensitive_option_paths(
                    item,
                    prefix=path,
                    seen=visited,
                )
            )
        return paths
    if isinstance(value, (list, tuple)):
        identity = id(value)
        if identity in visited:
            return []
        visited.add(identity)
        paths = []
        for index, item in enumerate(value):
            paths.extend(
                _sensitive_option_paths(
                    item,
                    prefix=f"{prefix}[{index}]",
                    seen=visited,
                )
            )
        return paths
    return []


def _safe_error_text(value: Any) -> str:
    try:
        text = str(value or "")
    except Exception:
        text = f"<{type(value).__name__}>"
    text = _DATA_URL_PATTERN.sub(
        "data:image/[REDACTED]",
        text,
    )
    text = _NAMED_CREDENTIAL_PATTERN.sub(
        r"\1\2[REDACTED]",
        text,
    )
    text = _BEARER_PATTERN.sub(
        "Bearer [REDACTED]",
        text,
    )
    return (text.strip() or "evaluation failed")[:2048]


def _json_copy(value: Any, *, label: str) -> Any:
    failed = False
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
        )
        result = json.loads(serialized)
    except (TypeError, ValueError, OverflowError):
        failed = True
        result = None
    if failed:
        raise ValueError(f"{label} must be JSON-compatible")
    return result


def _media_record(
    item: Mapping[str, Any] | str | os.PathLike[str],
    *,
    input_index: int,
) -> dict[str, Any]:
    if isinstance(item, Mapping):
        raw = dict(item)
        path_value = raw.get(
            "media_path",
            raw.get("path"),
        )
        name = str(raw.get("name", "") or Path(str(path_value or "")).name).strip()
        sampling = _json_copy(
            raw.get("media_sampling", {}),
            label="media_sampling",
        )
        generation_options = _json_copy(
            raw.get("generation_options", {}),
            label="generation_options",
        )
    else:
        raw = {}
        path_value = item
        name = Path(str(item)).name
        sampling = {}
        generation_options = {}
    path_text = str(path_value or "").strip()
    if not path_text:
        raise ValueError(f"media record {input_index} is missing media_path")
    if not name:
        raise ValueError(f"media record {input_index} is missing name")
    return {
        "input_index": input_index,
        "name": name,
        "media_path": Path(path_text).expanduser().resolve(),
        "media_sampling": sampling,
        "generation_options": generation_options,
    }


def _select_media_records(
    media_records: Sequence[Mapping[str, Any] | str | os.PathLike[str]],
    *,
    preserve_input_order: bool,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    selected = []
    filtered = []
    for input_index, item in enumerate(media_records):
        record = _media_record(
            item,
            input_index=input_index,
        )
        name = record["name"]
        lower = name.lower()
        reason = ""
        if name.startswith("."):
            reason = "hidden_media"
        elif not any(lower.endswith(extension) for extension in _GRID_IMAGE_EXTENSIONS):
            reason = "unsupported_media_extension"
        if reason:
            filtered.append(
                {
                    "input_index": input_index,
                    "name": name,
                    "media_path": record["media_path"].name,
                    "status": "filtered",
                    "reason": reason,
                }
            )
            continue
        selected.append(record)
    if not preserve_input_order:
        selected.sort(
            key=lambda record: (
                record["name"],
                record["input_index"],
            )
        )
    for order, record in enumerate(selected):
        record["order"] = order
    return selected, filtered


def _merged_json_mapping(
    base: Mapping[str, Any] | None,
    override: Mapping[str, Any] | None,
    *,
    label: str,
) -> dict[str, Any]:
    merged = dict(base or {})
    merged.update(dict(override or {}))
    result = _json_copy(merged, label=label)
    if not isinstance(result, dict):
        raise ValueError(f"{label} must be a mapping")
    return result


def _media_digest(path: Path) -> str:
    try:
        payload = path.read_bytes()
    except Exception as error:
        return f"unreadable:{type(error).__name__}"
    return hashlib.sha256(payload).hexdigest()


def _rendered_prompt_payload(
    *,
    rubric: str,
    prompt_info: Mapping[str, Any] | None,
) -> dict[str, str | None] | None:
    """Build the exact rubric prompt text when prompt metadata is complete."""

    if prompt_info is None:
        return None
    try:
        task_prompt = str(prompt_info["prompt"])
        explicit = prompt_info.get("evaluation_prompts")
        if isinstance(explicit, Mapping):
            rendered = explicit.get(rubric)
            if rubric == RUBRIC_SUBJECT_STABILITY and isinstance(rendered, Mapping):
                return {
                    "task_prompt": task_prompt,
                    "q1": str(rendered["q1"]),
                    "q2": (
                        None if rendered.get("q2") is None else str(rendered["q2"])
                    ),
                }
            if rubric != RUBRIC_SUBJECT_STABILITY and isinstance(rendered, str):
                return {
                    "task_prompt": task_prompt,
                    "evaluation_prompt": rendered,
                }
        if rubric == RUBRIC_SUBJECT_STABILITY:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                q1, q2 = create_subject_stability_prompts(
                    str(prompt_info["robotic manipulator"]),
                    prompt_info.get("manipulated object"),
                )
            return {
                "task_prompt": task_prompt,
                "q1": q1,
                "q2": q2,
            }
        view = str(prompt_info["view"])
        evaluation_prompt = (
            create_physical_plausibility_prompt(view, task_prompt)
            if rubric == RUBRIC_PHYSICAL_PLAUSIBILITY
            else create_task_adherence_prompt(view, task_prompt)
        )
        return {
            "task_prompt": task_prompt,
            "evaluation_prompt": evaluation_prompt,
        }
    except (KeyError, TypeError, ValueError):
        return None


def _json_digest(value: Any) -> str:
    serialized = json.dumps(
        value,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _fingerprint(
    *,
    rubric: str,
    media: Mapping[str, Any],
    vid_id: str,
    prompt_info: Mapping[str, Any] | None,
    backend: str,
    model: str,
    prompt_id: str,
    parser_id: str,
    generation_options: Mapping[str, Any],
    media_sampling: Mapping[str, Any],
    max_attempts: int,
    inference_identity: Mapping[str, Any],
    rendered_prompt_payload: Mapping[str, Any] | None,
) -> str:
    payload = {
        "fingerprint_contract": "portable-rendered-prompt",
        "rubric": rubric,
        "media_name": media["name"],
        "media_sha256": _media_digest(media["media_path"]),
        "vid_id": vid_id,
        "prompt_info": (
            None
            if prompt_info is None
            else _json_copy(
                dict(prompt_info),
                label="prompt metadata",
            )
        ),
        "backend": str(backend),
        "model": str(model),
        "prompt_id": str(prompt_id),
        "parser_id": str(parser_id),
        "generation_options": dict(generation_options),
        "media_sampling": dict(media_sampling),
        "max_attempts": int(max_attempts),
        "inference_identity": dict(inference_identity),
        "rendered_prompt_sha256": (
            None
            if rendered_prompt_payload is None
            else _json_digest(rendered_prompt_payload)
        ),
    }
    return _json_digest(payload)


def _safe_cache_stem(name: str) -> str:
    stem = Path(str(name)).stem
    clean = re.sub(r"[^A-Za-z0-9._-]+", "_", stem).strip("._-")
    return (clean or "media")[:80]


def _canonical_prediction_path(
    prediction_dir: Path,
    *,
    order: int,
    name: str,
    fingerprint: str,
) -> Path:
    return prediction_dir / (
        f"{order:06d}_{_safe_cache_stem(name)}_{fingerprint[:16]}.json"
    )


def _next_evidence_path(
    canonical: Path,
    *,
    label: str,
) -> Path:
    if not canonical.exists():
        return canonical
    for attempt in range(1, 10000):
        candidate = canonical.with_name(
            f"{canonical.stem}.{label}-{attempt:04d}{canonical.suffix}"
        )
        if not candidate.exists():
            return candidate
    raise RuntimeError("could not allocate append-only prediction path")


def _cache_record_is_reusable(
    record: Mapping[str, Any],
    *,
    fingerprint: str,
) -> bool:
    cache = record.get("cache")
    status = str(record.get("status", "") or "")
    return (
        isinstance(cache, Mapping)
        and cache.get("format") == _CACHE_SCHEMA
        and cache.get("fingerprint") == fingerprint
        and status
        in {
            "ok",
            "parse_error",
            "missing_prompt",
        }
    )


def _sanitize_failure_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _sanitize_failure_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_failure_value(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_failure_value(item) for item in value]
    if isinstance(value, str):
        return _safe_error_text(value)
    return copy.deepcopy(value)


def _sanitize_prediction_failures(
    record: Mapping[str, Any],
) -> dict[str, Any]:
    output = copy.deepcopy(dict(record))
    for key in ("error", "prior_attempt_failures"):
        if key in output:
            output[key] = _sanitize_failure_value(output[key])
    if output.get("status") == "inference_error":
        response = output.get("response")
        if isinstance(response, Mapping):
            response = dict(response)
            if isinstance(response.get("reason"), str):
                response["reason"] = _safe_error_text(response["reason"])
            output["response"] = response
    return output


def _batch_error_record(
    *,
    name: str,
    media_path: Path,
    error: Exception,
) -> dict[str, Any]:
    return {
        "format": _PREDICTION_SCHEMA,
        "name": name,
        "prompt": "",
        "raw_response": None,
        "status": "batch_error",
        "response": {
            "score": -1,
            "reason": _safe_error_text(error),
        },
        "error": {
            "type": type(error).__name__,
            "message": _safe_error_text(error),
        },
        "media": {
            "path": media_path.name,
            "sampling": {},
        },
    }


def _missing_task_prompt_record(
    *,
    vid_id: str,
    media_path: Path,
) -> dict[str, Any]:
    reason = f"Prompt not found for video {vid_id!r}"
    return {
        "format": _PREDICTION_SCHEMA,
        "name": vid_id,
        "prompt": "",
        "raw_response": None,
        "status": "missing_prompt",
        "response": {
            "score": -1,
            "reason": reason,
        },
        "error": {
            "type": "PromptNotFound",
            "message": reason,
        },
        "media": {
            "path": media_path.name,
            "sampling": {},
        },
    }


def _evaluate_one(
    *,
    rubric: str,
    media: Mapping[str, Any],
    prompt_info: Mapping[str, Any],
    infer: InferenceCallable,
    backend: str,
    model: str,
    prompt_id: str,
    parser_id: str,
    generation_options: Mapping[str, Any],
    media_sampling: Mapping[str, Any],
    max_attempts: int,
) -> dict[str, Any]:
    task_prompt = str(prompt_info["prompt"])
    explicit = prompt_info.get("evaluation_prompts")
    if rubric == RUBRIC_SUBJECT_STABILITY:
        return evaluate_subject_stability_media(
            name=str(media["name"]),
            media_path=media["media_path"],
            task_prompt=task_prompt,
            manipulator_phrase=str(prompt_info["robotic manipulator"]),
            object_phrase=prompt_info.get("manipulated object"),
            infer=infer,
            backend=backend,
            model=model,
            prompt_id=prompt_id,
            generation_options=generation_options,
            media_sampling=media_sampling,
            parser_id=parser_id,
            max_attempts=max_attempts,
            evaluation_prompts=(
                explicit.get(rubric)
                if isinstance(explicit, Mapping)
                and isinstance(explicit.get(rubric), Mapping)
                else None
            ),
        )
    view = str(prompt_info["view"])
    explicit_prompt = explicit.get(rubric) if isinstance(explicit, Mapping) else None
    evaluation_prompt = str(explicit_prompt) if isinstance(explicit_prompt, str) else (
        create_physical_plausibility_prompt(view, task_prompt)
        if rubric == RUBRIC_PHYSICAL_PLAUSIBILITY
        else create_task_adherence_prompt(view, task_prompt)
    )
    return evaluate_score_media(
        name=grid_image_to_vid_id(str(media["name"])),
        media_path=media["media_path"],
        task_prompt=task_prompt,
        evaluation_prompt=evaluation_prompt,
        infer=infer,
        backend=backend,
        model=model,
        prompt_id=prompt_id,
        generation_options=generation_options,
        media_sampling=media_sampling,
        parser_id=parser_id,
        max_attempts=max_attempts,
    )


def _is_report_record(
    record: Mapping[str, Any],
) -> bool:
    return str(record.get("status", "") or "") not in {
        "inference_error",
        "batch_error",
    }


def run_saved_media_vlm_batch(
    *,
    media_records: Sequence[Mapping[str, Any] | str | os.PathLike[str]],
    prompts: Sequence[Mapping[str, Any]],
    rubric: str,
    infer: InferenceCallable,
    backend: str,
    model: str,
    prompt_id: str,
    output_csv: str | os.PathLike[str],
    prediction_dir: str | os.PathLike[str] | None = None,
    generation_options: Mapping[str, Any] | None = None,
    media_sampling: Mapping[str, Any] | None = None,
    parser_id: str | None = None,
    max_attempts: int = 1,
    use_cache: bool = True,
    force: bool = False,
    continue_on_error: bool = True,
    preserve_input_order: bool = False,
    parallelism: int = 1,
    executor_factory: ExecutorFactory | None = None,
    inference_identity: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate an explicit saved-image batch and write a current CSV.

    The default media ordering matches current sorted grid filenames.
    ``parallelism > 1`` is available only through an injected executor factory;
    outcomes are sorted back to their deterministic batch order before CSV
    publication.  No default process or thread runtime is selected here.
    """

    clean_rubric = _normalize_rubric(rubric)
    if not callable(infer):
        raise TypeError("infer must be callable")
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    if isinstance(parallelism, bool) or int(parallelism) < 1:
        raise ValueError("parallelism must be a positive integer")
    workers = int(parallelism)
    if workers > 64:
        raise ValueError("parallelism must not exceed 64")
    if workers > 1 and executor_factory is None:
        raise ValueError("parallelism > 1 requires an explicit executor_factory")
    if workers > 1 and not continue_on_error:
        raise ValueError("parallel execution requires continue_on_error=true")
    output_path_text = str(output_csv or "").strip()
    if not output_path_text:
        raise ValueError("output_csv is required")
    output_path = Path(output_path_text).expanduser().resolve()
    prediction_root = (
        None
        if not str(prediction_dir or "").strip()
        else Path(prediction_dir).expanduser().resolve()
    )
    global_generation_options = _merged_json_mapping(
        generation_options,
        None,
        label="generation_options",
    )
    sensitive_paths = _sensitive_option_paths(global_generation_options)
    if sensitive_paths:
        raise ValueError(
            "generation_options must not contain credentials "
            "or client configuration: " + ", ".join(sensitive_paths)
        )
    global_media_sampling = _merged_json_mapping(
        media_sampling,
        None,
        label="media_sampling",
    )
    normalized_inference_identity = _merged_json_mapping(
        inference_identity,
        None,
        label="inference_identity",
    )
    identity_sensitive_paths = _sensitive_option_paths(normalized_inference_identity)
    if identity_sensitive_paths:
        raise ValueError(
            "inference_identity must not contain credentials or raw client "
            "configuration: " + ", ".join(identity_sensitive_paths)
        )
    normalized_prompts = [
        _json_copy(
            dict(prompt),
            label="prompt metadata",
        )
        for prompt in prompts
    ]
    prompt_map = build_prompt_by_name_map(normalized_prompts)
    selected, filtered = _select_media_records(
        media_records,
        preserve_input_order=preserve_input_order,
    )
    selected_parser_id = str(
        parser_id
        or (
            "stability-options"
            if clean_rubric == RUBRIC_SUBJECT_STABILITY
            else "score-json"
        )
    )
    if prediction_root is not None:
        prediction_root.mkdir(
            parents=True,
            exist_ok=True,
        )

    def run_task(
        task: tuple[int, Mapping[str, Any]],
    ) -> tuple[int, dict[str, Any]]:
        order, media = task
        name = str(media["name"])
        media_path = Path(media["media_path"])
        vid_id = grid_image_to_vid_id(name)
        prompt_info = resolve_prompt_info(
            vid_id,
            prompt_map,
            normalized_prompts,
        )
        rendered_prompt_payload = _rendered_prompt_payload(
            rubric=clean_rubric,
            prompt_info=prompt_info,
        )
        item: dict[str, Any] = {
            "order": order,
            "input_index": media["input_index"],
            "name": name,
            "vid_id": vid_id,
            "media_path": media_path.name,
            "status": "pending",
            "record": None,
            "prediction_path": None,
        }
        if prompt_info is None and (clean_rubric != RUBRIC_TASK_ADHERENCE):
            item["status"] = "skipped_missing_prompt"
            return order, item

        per_item_options = _merged_json_mapping(
            global_generation_options,
            media["generation_options"],
            label="generation_options",
        )
        per_item_sensitive_paths = _sensitive_option_paths(per_item_options)
        if per_item_sensitive_paths:
            error = ValueError(
                "generation options contain sensitive "
                "configuration at: " + ", ".join(per_item_sensitive_paths)
            )
            item["status"] = "batch_error"
            item["record"] = _batch_error_record(
                name=name,
                media_path=media_path,
                error=error,
            )
            return order, item
        per_item_sampling = _merged_json_mapping(
            global_media_sampling,
            media["media_sampling"],
            label="media_sampling",
        )

        fingerprint = _fingerprint(
            rubric=clean_rubric,
            media=media,
            vid_id=vid_id,
            prompt_info=prompt_info,
            backend=backend,
            model=model,
            prompt_id=prompt_id,
            parser_id=selected_parser_id,
            generation_options=per_item_options,
            media_sampling=per_item_sampling,
            max_attempts=max_attempts,
            inference_identity=normalized_inference_identity,
            rendered_prompt_payload=rendered_prompt_payload,
        )
        canonical_path = (
            None
            if prediction_root is None
            else _canonical_prediction_path(
                prediction_root,
                order=order,
                name=name,
                fingerprint=fingerprint,
            )
        )
        cache_issue = None
        if (
            canonical_path is not None
            and use_cache
            and not force
            and canonical_path.exists()
        ):
            try:
                cached = load_prediction_record(canonical_path)
            except Exception as error:
                cache_issue = {
                    "type": type(error).__name__,
                    "message": _safe_error_text(error),
                }
            else:
                if _cache_record_is_reusable(
                    cached,
                    fingerprint=fingerprint,
                ):
                    item["status"] = "cached"
                    item["record"] = cached
                    item["prediction_path"] = canonical_path.relative_to(
                        prediction_root
                    ).as_posix()
                    return order, item
                cache_issue = {
                    "type": "CacheMismatch",
                    "message": ("cached prediction is not reusable"),
                }

        try:
            if prompt_info is None:
                record = _missing_task_prompt_record(
                    vid_id=vid_id,
                    media_path=media_path,
                )
            else:
                record = _evaluate_one(
                    rubric=clean_rubric,
                    media=media,
                    prompt_info=prompt_info,
                    infer=infer,
                    backend=str(backend),
                    model=str(model),
                    prompt_id=str(prompt_id),
                    parser_id=(selected_parser_id),
                    generation_options=(per_item_options),
                    media_sampling=per_item_sampling,
                    max_attempts=max_attempts,
                )
            record = _sanitize_prediction_failures(record)
        except Exception as error:
            record = _batch_error_record(
                name=name,
                media_path=media_path,
                error=error,
            )
        record["cache"] = {
            "format": _CACHE_SCHEMA,
            "fingerprint_contract": "portable-rendered-prompt",
            "fingerprint": fingerprint,
            "rendered_prompt_sha256": (
                None
                if rendered_prompt_payload is None
                else _json_digest(rendered_prompt_payload)
            ),
        }
        provenance = record.get("provenance")
        if isinstance(provenance, Mapping):
            provenance_copy = dict(provenance)
            provenance_copy["inference_identity"] = copy.deepcopy(
                normalized_inference_identity
            )
            record["provenance"] = provenance_copy
        item["record"] = record
        record_status = str(record.get("status", "") or "")
        item["status"] = (
            "evaluated"
            if record_status not in {"batch_error", "inference_error"}
            else record_status
        )
        if cache_issue is not None:
            item["cache_issue"] = cache_issue

        if canonical_path is not None:
            label = (
                "force"
                if force
                else "recovered"
                if cache_issue is not None
                else "rerun"
            )
            destination = _next_evidence_path(
                canonical_path,
                label=label,
            )
            try:
                write_prediction_record(
                    destination,
                    record,
                )
            except Exception as error:
                item["prediction_write_error"] = {
                    "type": type(error).__name__,
                    "message": _safe_error_text(error),
                }
            else:
                item["prediction_path"] = destination.relative_to(
                    prediction_root
                ).as_posix()
        return order, item

    tasks = [(index, media) for index, media in enumerate(selected)]
    outcomes: list[tuple[int, dict[str, Any]]] = []
    if workers == 1:
        for task in tasks:
            outcome = run_task(task)
            outcomes.append(outcome)
            item = outcome[1]
            record = item.get("record")
            record_status = (
                str(record.get("status", "")) if isinstance(record, Mapping) else ""
            )
            if not continue_on_error and (
                item["status"]
                in {
                    "batch_error",
                    "inference_error",
                }
                or record_status
                in {
                    "batch_error",
                    "inference_error",
                }
            ):
                break
    else:
        assert executor_factory is not None
        with executor_factory(max_workers=workers) as executor:
            outcomes = list(executor.map(run_task, tasks))
    outcomes.sort(key=lambda result: result[0])
    items = [outcome[1] for outcome in outcomes]
    records = [
        item["record"] for item in items if isinstance(item.get("record"), Mapping)
    ]
    report_records = [record for record in records if _is_report_record(record)]

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    if clean_rubric == RUBRIC_SUBJECT_STABILITY:
        save_stability_results_to_csv(
            report_records,
            output_path,
        )
    else:
        save_results_to_csv(
            report_records,
            output_path,
        )

    record_status_counts: dict[str, int] = {}
    for record in records:
        status = str(record.get("status", "unknown") or "unknown")
        record_status_counts[status] = record_status_counts.get(status, 0) + 1
    item_status_counts: dict[str, int] = {}
    for item in items:
        status = str(item["status"])
        item_status_counts[status] = item_status_counts.get(status, 0) + 1
    issue_count = sum(
        record_status_counts.get(status, 0)
        for status in (
            "batch_error",
            "inference_error",
            "parse_error",
            "missing_prompt",
        )
    ) + sum(
        1
        for item in items
        if ("cache_issue" in item or "prediction_write_error" in item)
    )
    return {
        "format": _BATCH_SCHEMA,
        "status": ("completed_with_issues" if issue_count else "completed"),
        "rubric": clean_rubric,
        "selected_media_count": len(selected),
        "processed_media_count": len(items),
        "filtered_media": filtered,
        "items": items,
        "records": records,
        "report_record_count": len(report_records),
        "item_status_counts": item_status_counts,
        "record_status_counts": (record_status_counts),
        "output_csv": output_path.name,
        "prediction_dir": (
            None if prediction_root is None else prediction_root.name
        ),
        "parallelism": workers,
        "force": bool(force),
        "cache_enabled": bool(use_cache and prediction_root is not None),
        "cache_disabled_reason": (
            None
            if use_cache and prediction_root is not None
            else (
                "disabled_by_caller"
                if not use_cache
                else "prediction_dir_not_configured"
            )
        ),
    }


__all__ = [
    "ExecutorFactory",
    "InferenceCallable",
    "LEGACY_RUBRIC_TASK_ADHERENCE",
    "RUBRIC_PHYSICAL_PLAUSIBILITY",
    "RUBRIC_SUBJECT_STABILITY",
    "RUBRIC_TASK_ADHERENCE",
    "SAVED_MEDIA_RUBRICS",
    "paper_vlm_csv_filename",
    "run_saved_media_vlm_batch",
]
