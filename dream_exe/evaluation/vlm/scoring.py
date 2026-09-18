"""Reproducible VLM evaluation over already materialized media artifacts.

This module deliberately has no model-client, simulator, or bench dependency.
Callers provide an inference function and explicit media paths.  The returned
record keeps the current score-result shape while adding enough provenance to
audit or regenerate reports without another model call.
"""

from __future__ import annotations

import csv
import io
import json
import os
import re
import stat
import tempfile
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

VLM_PAPER_METRICS = (
    {
        "field": "subject_stability",
        "paper_label": "Stab.",
        "paper_name": "robot-subject stability",
        "direction": "higher_is_better",
        "scale": [1, 15],
    },
    {
        "field": "physical_plausibility",
        "paper_label": "Phys.",
        "paper_name": "physical plausibility",
        "direction": "higher_is_better",
        "scale": [1, 5],
    },
    {
        "field": "task_adherence",
        "paper_label": "Task Adh.",
        "paper_name": "task adherence",
        "direction": "higher_is_better",
        "scale": [1, 5],
    },
)


_GRID_IMAGE_EXTS = (".jpeg", ".jpg", ".png", ".webp", ".bmp")
_PREDICTION_SCHEMA = "dream-exe.vlm-prediction"
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
    "credentials",
}


def extract_json(string: str) -> Any:
    """Extract and decode the outermost JSON object, matching current scripts."""

    start = string.find("{")
    end = string.rfind("}") + 1
    json_part = string[start:end]
    return json.loads(json_part)


def list_grid_image_files(image_grid_path: str | os.PathLike[str]) -> list[str]:
    """List supported raster inputs using the current script ordering."""

    names = []
    for filename in os.listdir(image_grid_path):
        if filename.startswith("."):
            continue
        low = filename.lower()
        if any(low.endswith(ext) for ext in _GRID_IMAGE_EXTS):
            names.append(filename)
    return sorted(names)


def grid_image_to_vid_id(grid_image_name: str) -> str:
    """Map a generated grid filename back to its current video identifier."""

    stem = os.path.splitext(str(grid_image_name))[0]
    return re.sub(r"_(?:6|2)frame$", "", stem)


def uniform_video_frame_indices(
    total_frames: int,
    num_frames: int = 16,
) -> np.ndarray:
    """Return the uniform indices used by the current score rubrics."""

    if total_frames <= num_frames:
        return np.arange(total_frames)
    return np.linspace(
        0,
        total_frames - 1,
        num_frames,
        dtype=int,
    )


def stability_video_frame_indices(total_frames: int) -> np.ndarray:
    """Return current reference/future indices for stability scoring."""

    if total_frames == 0:
        return np.asarray([], dtype=int)
    if total_frames > 1:
        indices = [0, int(0.75 * (total_frames - 1))]
        return np.asarray(sorted(set(indices)), dtype=int)
    return np.asarray([0], dtype=int)


def grid_resample_indices(
    extracted_frame_count: int,
    num_images: int = 6,
) -> np.ndarray:
    """Select current evenly spaced frames from an extracted sequence."""

    return np.linspace(
        0,
        extracted_frame_count - 1,
        num_images,
        dtype=int,
    )


def merge_frame_grid(
    image_list: Sequence[np.ndarray],
    rows: int = 3,
    cols: int = 2,
) -> np.ndarray:
    """Merge equal-shaped BGR frames using the current row-major layout."""

    assert len(image_list) == rows * cols, (
        f"需要 {rows * cols} 张图片，但传入 {len(image_list)} 张"
    )
    row_images = []
    for row_index in range(rows):
        row = np.concatenate(
            image_list[row_index * cols : (row_index + 1) * cols],
            axis=1,
        )
        row_images.append(row)
    return np.concatenate(row_images, axis=0)


def build_prompt_by_name_map(
    prompts: Sequence[Mapping[str, Any]],
) -> dict[str, Mapping[str, Any]]:
    """Build the exact-name prompt lookup used by the current evaluators."""

    return {
        str(prompt.get("name", "")).strip(): prompt
        for prompt in prompts
        if str(prompt.get("name", "")).strip()
    }


def resolve_prompt_info(
    vid_id: str,
    prompt_map: Mapping[str, Mapping[str, Any]],
    prompts: Sequence[Mapping[str, Any]],
) -> Mapping[str, Any] | None:
    """Resolve exact or longest prefix-compatible prompt metadata."""

    if vid_id in prompt_map:
        return prompt_map[vid_id]
    candidates = []
    for prompt in prompts:
        name = str(prompt.get("name", "")).strip()
        if not name:
            continue
        if vid_id.startswith(name) or name.startswith(vid_id):
            candidates.append((len(name), prompt))
    if candidates:
        candidates.sort(key=lambda item: item[0], reverse=True)
        return candidates[0][1]
    return None


def analyze_phrase(phrase_1: str) -> str:
    """Describe the current stability evaluator's terminal inconsistency."""

    phrase_1_lower = phrase_1.lower()

    has_gripper = re.search(r"\bgrippers?\b", phrase_1_lower)
    has_arm = re.search(r"\barms?\b", phrase_1_lower)
    has_hand = re.search(r"\bhands?\b", phrase_1_lower)
    has_robot = re.search(r"\brobots?\b", phrase_1_lower)

    if has_gripper or has_arm:
        return (
            f"In the right frame, {phrase_1} is replaced by a 'robotic hand' "
            "or 'human hand' when interacting with the object, compared with "
            "the left frame."
        )
    if has_hand:
        return (
            f"In the right frame, {phrase_1} is replaced by a 'human hand' "
            "when interacting with the object, compared with the left frame."
        )
    if has_robot:
        return (
            f"{phrase_1} is missing in the right frame while it exists in the "
            "left image."
        )
    warnings.warn(
        "Unexpected phrase_1 value: Expected to contain 'gripper(s)', "
        f"'hand(s)', or 'robot(s)', but got '{phrase_1}'.",
        stacklevel=2,
    )
    return ""


def score_mapping(opt_str: str) -> int | str:
    """Map current stability options to their legacy 1–15 score."""

    choices = opt_str.split(",")
    parsed = [re.fullmatch(r"([A-E])([12])", item) for item in choices]
    if any(item is None for item in parsed):
        return "bad reply"

    decoded = [(item.group(1), item.group(2)) for item in parsed if item]
    rank = {letter: index for index, letter in enumerate("ABCDE")}
    if len(decoded) == 1:
        letter, view = decoded[0]
        if view != "1":
            return "bad reply"
        return (15, 11, 7, 4, 1)[rank[letter]]

    if len(decoded) != 2 or {view for _, view in decoded} != {"1", "2"}:
        return "bad reply"

    row, column = sorted(rank[letter] for letter, _ in decoded)
    upper_triangle = (
        (15, 14, 12, 10, 8),
        (13, 11, 9, 7),
        (6, 5, 3),
        (4, 2),
        (1,),
    )
    return upper_triangle[row][column - row]


def create_physical_plausibility_prompt(view: str, description: str) -> str:
    """Build the current physical-plausibility rubric verbatim."""

    return f"""
    The provided image presents sequential frames, arranged in a grid, from a {view} perspective AI-generated task video about {description}
    Does this video comply with common-sense expectations for human-level interactions? 
 
    # A. Anomaly Checks:
    1) Physical grounding violation: any part of the robot appears floating, or intersecting/penetrating other geometry.
    2) Spontaneous object appearance: any object or robotic agent that suddenly appears between frames without a plausible cause. Pay special attention to the top two frames, where such anomalies often occur early in the sequence.
    3) Non-contact attachment / false grasp: If the video involves a grasping interaction, carefully check whether the object remains attached to or moves with the robotic gripper/hand without clear physical contact, closure.
    If either anomaly is present, treat it as a major violation and assign a low score (1–2). 
 
    # B. Human-Level Common-Sense Consistency:
    Evaluate whether this video complies with common-sense expectations for human-level interactions?
    Rate the video on a scale from 1 to 5, where 5 means fully consistent with human common sense and 1 means major violations. 
 
    ### Special Notes:
    1) Be very cautious when assigning scores of 4 or 5: these scores should only be used when clearly correct, and free of errors. Do not give 4 or 5 lightly.
    2) Use step-by-step reasoning internally to make your selection.
    3) Your output must be a valid JSON object with two fields:
    - "reason": a breif justification for the given score
    - "score": an integer between 1 and 5
    """


def create_task_adherence_prompt(view: str, description: str) -> str:
    """Build the current task-adherence rubric verbatim."""

    return f"""    
    The provided image presents sequential frames, arranged in a grid, from a {view} perspective AI-generated task video.
    In this AI-generated video, does the robot successfully perform the task: "{description}"?
    Please rate the video on a scale from 1 to 5, where 5 indicates a perfect match and 1 indicates no relevance.
    
    ### Special Notes:
    1) Be very cautious when assigning scores of 4 or 5: these scores should only be used when clearly correct, and free of errors. Do not give 4 or 5 lightly.
    2) Use step-by-step reasoning internally to make your selection.
    3) Your output must be a valid JSON object with two fields:
    - "reason": a breif justification for the given score
    - "score": an integer between 1 and 5
    """


def create_subject_stability_prompts(
    manipulator_phrase: str,
    object_phrase: str | None,
) -> tuple[str, str | None]:
    """Build the current two-question subject-stability rubric."""

    terminal_option = analyze_phrase(manipulator_phrase)
    phrase_lower = manipulator_phrase.lower()
    has_gripper = re.search(r"\bgrippers?\b", phrase_lower)
    has_hand = re.search(r"\bhands?\b", phrase_lower)

    if has_gripper or has_hand:
        manipulator_prompt = f""" 
The provided image shows two sequential frames from an AI-generated video about robot doing a task. 
The left frame is the correct reference image, while the right frame is the AI-generated video frame. 
Focuse on how '{manipulator_phrase}' appears in both frames, and evaluate the consistency of '{manipulator_phrase}' between the reference and the generated frame.

Note: 
1) Pay special attention to distinguishing between robotic gripper and robotic hand (if visible). Robotic gripper usually has a small number of rigid gripping jaws or prongs, while a robotic hand has multiple articulated fingers and more complex structures.
2) Changes in orientation or position are acceptable and should not affect the consistency rating.
3) Important: Do NOT assign option A or B lightly. 

Question:
A: '{manipulator_phrase}' in the right frame is clear and consistent with the left image.  
B: '{manipulator_phrase}' in the right frame is mostly consistent with the left image, with minor visual issues.  
C: '{manipulator_phrase}' in the right frame shows noticeable inconsistencies compared with the left image, such as changes in shape, structure.  
D: '{manipulator_phrase}' in the right frame is highly inconsistent with the left image, transforms into another type of '{manipulator_phrase}'.
E: {terminal_option}
The options A to E represent increasing levels of inconsistency, select the most suitable option.
Put the option in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., C).
"""
    else:
        manipulator_prompt = f""" 
The provided image shows two sequential frames from an AI-generated video about robot doing a task. 
The left frame is the correct reference image, while the right frame is the AI-generated video frame. 
Focuse on how '{manipulator_phrase}' appears in both frames, and evaluate the consistency of '{manipulator_phrase}' between the reference and the generated frame.
Note: 
1) If the subject has a robotic gripper/hand, pay special attention to distinguishing between robotic gripper and robotic hand. Robotic gripper usually has a small number of rigid gripping jaws or prongs, while a robotic hand has multiple articulated fingers and more complex structures.
2) Changes in orientation or position are acceptable and should not affect the consistency rating.
3) Important: Do NOT assign option A or B lightly. 

Question:
A: '{manipulator_phrase}' in the right frame is clear and consistent with the left image.  
B: '{manipulator_phrase}' in the right frame is mostly consistent with the left image, with minor visual issues.  
C: '{manipulator_phrase}' in the right frame shows noticeable inconsistencies compared with the left image, such as changes in shape, structure.  
D: '{manipulator_phrase}' in the right frame is highly inconsistent with the left image, transforms into another type of '{manipulator_phrase}'.
E: {terminal_option}
The options A to E represent increasing levels of inconsistency, select the most suitable option.
Put the option in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., C).
"""

    normalized_object = (
        None
        if object_phrase is None or str(object_phrase).strip().lower() == "none"
        else str(object_phrase)
    )
    if normalized_object is None:
        return manipulator_prompt, None

    object_prompt = f"""
The provided image shows two sequential frames from an AI-generated video about robot doing a task. 
The left frame is the correct reference image, while the right frame is the AI-generated video frame. 
Focuse on how '{normalized_object}' appears in both frames, and evaluate the consistency of '{normalized_object}' between the reference and the generated frame.

Note: 
1) Changes in orientation or position are acceptable and should not affect the consistency rating.
2) Important: Do NOT assign option A or B lightly. 

Question: 
A: '{normalized_object}' in the right frame is clear and consistent with the left image.
B: '{normalized_object}' in the right frame is mostly consistent with the left image, with minor visual issues. 
C: '{normalized_object}' in the right frame shows noticeable inconsistencies compared with the left image.  
D: '{normalized_object}' in the right frame undergoes a major transformation, appears as an AI-generated artifact or is duplicated compared with the left image.  
E: '{normalized_object}' is missing in the right frame while it exists in the left image.

The options A to E represent increasing levels of inconsistency, select the most suitable option.
Put the option in JSON format with the following keys: option (e.g., A), explanation (explaining the option made within 50 words), adjust (adjusted option after explanation, e.g., C).
"""
    return manipulator_prompt, object_prompt


def parse_subject_stability_responses(
    manipulator_raw_response: str,
    object_raw_response: str | None = None,
) -> dict[str, Any]:
    """Parse current stability options and preserve its bad-reply fallback."""

    try:
        manipulator_response = extract_json(manipulator_raw_response)
        if not isinstance(manipulator_response, dict):
            raise ValueError("manipulator response must be a JSON object")
        manipulator_adjust = manipulator_response.get("adjust")
        if manipulator_adjust not in set("ABCDE"):
            raise ValueError("manipulator adjust must be one of A, B, C, D, E")
        if not isinstance(manipulator_response.get("explanation"), str):
            raise ValueError("manipulator explanation must be text")
        if object_raw_response is None:
            option = f"{manipulator_adjust}1"
            object_explanation = ""
        else:
            object_response = extract_json(object_raw_response)
            if not isinstance(object_response, dict):
                raise ValueError("object response must be a JSON object")
            object_adjust = object_response.get("adjust")
            if object_adjust not in set("ABCDE"):
                raise ValueError("object adjust must be one of A, B, C, D, E")
            object_explanation = object_response.get("explanation")
            if not isinstance(object_explanation, str):
                raise ValueError("object explanation must be text")
            option = f"{manipulator_adjust}1,{object_adjust}2"
        score = score_mapping(option)
    except Exception:
        option = "bad reply"
        score = "bad reply"
        manipulator_response = {"explanation": ""}
        object_explanation = ""

    return {
        "option": option,
        "score": score,
        "explanation_q1": manipulator_response.get("explanation", ""),
        "explanation_q2": object_explanation,
    }


def parse_score_response(raw_response: str) -> tuple[dict[str, Any], str]:
    """Parse one paper rubric response with a strict integer score range."""

    try:
        parsed = extract_json(raw_response)
        if not isinstance(parsed, dict) or set(parsed) != {"score", "reason"}:
            raise ValueError("JSON response is missing score or reason")
        score = parsed["score"]
        reason = parsed["reason"]
        if isinstance(score, bool) or not isinstance(score, int):
            raise ValueError("score must be an integer")
        if not 1 <= score <= 5:
            raise ValueError("score must be between 1 and 5")
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("reason must be non-empty text")
    except Exception:
        return {"score": -1, "reason": raw_response}, "parse_error"
    return parsed, "ok"


def _sensitive_generation_option_paths(
    value: Any,
    prefix: str = "",
) -> list[str]:
    paths = []
    if isinstance(value, Mapping):
        for key, item in value.items():
            key_text = str(key)
            path = f"{prefix}.{key_text}" if prefix else key_text
            normalized = key_text.strip().lower().replace("-", "_")
            if normalized in _SENSITIVE_OPTION_NAMES:
                paths.append(path)
            paths.extend(_sensitive_generation_option_paths(item, path))
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            path = f"{prefix}[{index}]"
            paths.extend(_sensitive_generation_option_paths(item, path))
    return paths


def evaluate_score_media(
    *,
    name: str,
    media_path: str | os.PathLike[str],
    task_prompt: str,
    evaluation_prompt: str,
    infer: Callable[[str, Path, Mapping[str, Any]], str],
    backend: str,
    model: str,
    prompt_id: str,
    generation_options: Mapping[str, Any] | None = None,
    media_sampling: Mapping[str, Any] | None = None,
    parser_id: str = "score-json",
    max_attempts: int = 1,
) -> dict[str, Any]:
    """Evaluate one saved media item through an injected model adapter.

    ``infer`` receives ``(evaluation_prompt, media_path, generation_options)``.
    Authentication and client construction remain outside this package and are
    intentionally absent from the persisted generation options.
    """

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")

    path = Path(media_path)
    options = dict(generation_options or {})
    sensitive_paths = _sensitive_generation_option_paths(options)
    if sensitive_paths:
        joined = ", ".join(sensitive_paths)
        raise ValueError(
            "generation_options must not contain credentials; inject them "
            f"through the backend adapter instead: {joined}"
        )
    record: dict[str, Any] = {
        "format": _PREDICTION_SCHEMA,
        "name": str(name),
        "prompt": str(task_prompt),
        "response": {"score": -1, "reason": ""},
        "raw_response": None,
        "status": "inference_error",
        "error": None,
        "evaluation_prompt": str(evaluation_prompt),
        "media": {
            "path": path.name,
            "sampling": dict(media_sampling or {}),
        },
        "provenance": {
            "backend": str(backend),
            "model": str(model),
            "prompt_id": str(prompt_id),
            "parser_id": str(parser_id),
            "generation_options": options,
            "attempts": 0,
        },
    }

    failures = []
    for attempt in range(1, max_attempts + 1):
        record["provenance"]["attempts"] = attempt
        try:
            raw_response = infer(evaluation_prompt, path, options)
            if not isinstance(raw_response, str):
                raise TypeError("inference adapter must return text")
        except Exception as exc:
            failures.append(
                {
                    "attempt": attempt,
                    "type": type(exc).__name__,
                    "message": str(exc),
                }
            )
            continue

        response, status = parse_score_response(raw_response)
        record["raw_response"] = raw_response
        record["response"] = response
        record["status"] = status
        record["error"] = (
            None
            if status == "ok"
            else {
                "type": "ResponseParseError",
                "message": (
                    "response must contain only an integer score from 1 to 5 "
                    "and a non-empty reason"
                ),
            }
        )
        if failures:
            record["prior_attempt_failures"] = failures
        return record

    record["response"] = {
        "score": -1,
        "reason": failures[-1]["message"] if failures else "inference failed",
    }
    record["error"] = {
        "type": "InferenceError",
        "attempts": failures,
    }
    return record


def evaluate_subject_stability_media(
    *,
    name: str,
    media_path: str | os.PathLike[str],
    task_prompt: str,
    manipulator_phrase: str,
    object_phrase: str | None,
    infer: Callable[[str, Path, Mapping[str, Any]], str],
    backend: str,
    model: str,
    prompt_id: str,
    generation_options: Mapping[str, Any] | None = None,
    media_sampling: Mapping[str, Any] | None = None,
    parser_id: str = "stability-options",
    max_attempts: int = 1,
    evaluation_prompts: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Evaluate the current one- or two-subject stability rubric."""

    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    path = Path(media_path)
    options = dict(generation_options or {})
    sensitive_paths = _sensitive_generation_option_paths(options)
    if sensitive_paths:
        joined = ", ".join(sensitive_paths)
        raise ValueError(
            "generation_options must not contain credentials; inject them "
            f"through the backend adapter instead: {joined}"
        )

    if evaluation_prompts is None:
        q1, q2 = create_subject_stability_prompts(
            manipulator_phrase,
            object_phrase,
        )
    else:
        q1 = str(evaluation_prompts.get("q1", "") or "").strip()
        q2_value = evaluation_prompts.get("q2")
        q2 = None if q2_value is None else str(q2_value).strip()
        if not q1 or (object_phrase is not None and not q2):
            raise ValueError("subject_stability evaluation prompts are incomplete")
    normalized_object = (
        None
        if object_phrase is None or str(object_phrase).strip().lower() == "none"
        else str(object_phrase)
    )
    record: dict[str, Any] = {
        "format": _PREDICTION_SCHEMA,
        "name": str(name),
        "prompt": str(task_prompt),
        "robotic_phrase": str(manipulator_phrase),
        "object_phrase": normalized_object,
        "evaluation_prompts": {"q1": q1, "q2": q2},
        "raw_response": {"q1": None, "q2": None},
        "option": "bad reply",
        "score": "bad reply",
        "explanation_q1": "",
        "explanation_q2": "",
        "status": "inference_error",
        "error": None,
        "media": {
            "path": path.name,
            "sampling": dict(media_sampling or {}),
        },
        "provenance": {
            "backend": str(backend),
            "model": str(model),
            "prompt_id": str(prompt_id),
            "parser_id": str(parser_id),
            "generation_options": options,
            "attempts": {"q1": 0, "q2": 0},
        },
    }

    def _infer_question(label: str, prompt: str) -> tuple[str | None, list]:
        failures = []
        for attempt in range(1, max_attempts + 1):
            record["provenance"]["attempts"][label] = attempt
            try:
                raw = infer(prompt, path, options)
                if not isinstance(raw, str):
                    raise TypeError("inference adapter must return text")
                return raw, failures
            except Exception as exc:
                failures.append(
                    {
                        "attempt": attempt,
                        "type": type(exc).__name__,
                        "message": str(exc),
                    }
                )
        return None, failures

    q1_raw, q1_failures = _infer_question("q1", q1)
    if q1_raw is None:
        record["error"] = {
            "type": "InferenceError",
            "question": "q1",
            "attempts": q1_failures,
        }
        return record
    record["raw_response"]["q1"] = q1_raw

    q2_raw = None
    if q2 is not None:
        q2_raw, q2_failures = _infer_question("q2", q2)
        if q2_raw is None:
            record["error"] = {
                "type": "InferenceError",
                "question": "q2",
                "attempts": q2_failures,
            }
            return record
        record["raw_response"]["q2"] = q2_raw

    parsed = parse_subject_stability_responses(q1_raw, q2_raw)
    record.update(parsed)
    if parsed["score"] == "bad reply":
        record["status"] = "parse_error"
        record["error"] = {
            "type": "ResponseParseError",
            "message": "response did not contain adjusted stability options",
        }
    else:
        record["status"] = "ok"
    prior_failures = {}
    if q1_failures:
        prior_failures["q1"] = q1_failures
    if q2 is not None and q2_failures:
        prior_failures["q2"] = q2_failures
    if prior_failures:
        record["prior_attempt_failures"] = prior_failures
    return record


def write_prediction_record(
    output_path: str | os.PathLike[str],
    record: Mapping[str, Any],
) -> Path:
    """Persist one raw prediction record without overwriting prior evidence."""

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(record, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    return path


def load_prediction_record(
    input_path: str | os.PathLike[str],
) -> dict[str, Any]:
    record = json.loads(Path(input_path).read_text(encoding="utf-8"))
    if not isinstance(record, dict):
        raise ValueError("prediction record must be a JSON object")
    return record


def score_report_rows(
    records: Iterable[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Regenerate the legacy score-table rows from saved prediction records."""

    rows = []
    for record in records:
        response = record.get("response") or {}
        rows.append(
            {
                "name": record.get("name", ""),
                "score": response.get("score", ""),
                "prompt": record.get("prompt", ""),
                "reason": response.get("reason", ""),
            }
        )
    return rows


def _write_report_bytes_if_changed(path: Path, payload: bytes) -> None:
    """Atomically update one report while preserving identical cached bytes."""

    try:
        link_stat = path.lstat()
    except FileNotFoundError:
        link_stat = None
    if link_stat is not None:
        if stat.S_ISLNK(link_stat.st_mode):
            raise ValueError(f"report destination must not be a symlink: {path}")
        if not stat.S_ISREG(link_stat.st_mode):
            raise ValueError(f"report destination must be a regular file: {path}")
        if link_stat.st_size == len(payload):
            flags = os.O_RDONLY
            flags |= getattr(os, "O_CLOEXEC", 0)
            flags |= getattr(os, "O_NOFOLLOW", 0)
            descriptor = os.open(path, flags)
            try:
                before = os.fstat(descriptor)
                if not stat.S_ISREG(before.st_mode):
                    raise ValueError(
                        f"report destination must be a regular file: {path}"
                    )
                if (
                    link_stat.st_dev != before.st_dev
                    or link_stat.st_ino != before.st_ino
                ):
                    raise RuntimeError(
                        f"report destination changed while opening: {path}"
                    )
                existing = bytearray()
                while len(existing) <= len(payload):
                    chunk = os.read(
                        descriptor,
                        min(1024 * 1024, len(payload) + 1 - len(existing)),
                    )
                    if not chunk:
                        break
                    existing.extend(chunk)
                after = os.fstat(descriptor)
            finally:
                os.close(descriptor)
            stable = (
                before.st_dev,
                before.st_ino,
                before.st_size,
                before.st_mtime_ns,
            ) == (
                after.st_dev,
                after.st_ino,
                after.st_size,
                after.st_mtime_ns,
            )
            if not stable:
                raise RuntimeError(f"report destination changed while reading: {path}")
            if bytes(existing) == payload:
                return

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def save_results_to_csv(
    results: Iterable[Mapping[str, Any] | None],
    output_csv: str | os.PathLike[str],
) -> Path:
    """Write the current four-column score report from prediction records."""

    path = Path(output_csv)
    csvfile = io.StringIO(newline="")
    fieldnames = ["name", "score", "prompt", "reason"]
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        if result is None:
            continue
        response = result.get("response") or {}
        writer.writerow(
            {
                "name": result.get("name", ""),
                "score": response.get("score", ""),
                "prompt": result.get("prompt", ""),
                "reason": response.get("reason", ""),
            }
        )
    _write_report_bytes_if_changed(path, csvfile.getvalue().encode("utf-8"))
    return path


def save_stability_results_to_csv(
    results: Iterable[Mapping[str, Any] | None],
    output_csv: str | os.PathLike[str],
) -> Path:
    """Regenerate the current stability CSV from saved prediction records."""

    path = Path(output_csv)
    fieldnames = [
        "name",
        "prompt",
        "robotic_phrase",
        "object_phrase",
        "option",
        "score",
        "explanation_q1",
        "explanation_q2",
    ]
    csvfile = io.StringIO(newline="")
    writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
    writer.writeheader()
    for result in results:
        if not result:
            continue
        writer.writerow({field: result.get(field, "") for field in fieldnames})
    _write_report_bytes_if_changed(path, csvfile.getvalue().encode("utf-8"))
    return path
