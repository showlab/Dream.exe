"""Render benchmark-owned VLM prompt templates without invoking a model."""

from __future__ import annotations

import json
from collections.abc import Mapping
from typing import Any

from .scoring import analyze_phrase

_SCORE_JSON_EXAMPLE = json.dumps(
    {"reason": "<brief justification>", "score": 3},
    ensure_ascii=False,
)
_OPTION_JSON_EXAMPLE = json.dumps(
    {"option": "A", "explanation": "<brief explanation>", "adjust": "A"},
    ensure_ascii=False,
)


def render_bench_vlm_prompts(config: Mapping[str, Any]) -> dict[str, Any]:
    """Attach rendered paper-rubric prompts to canonical case metadata."""

    metadata = dict(config["prompt_metadata"])
    rubrics = config["rubrics"]
    view = str(metadata["view"])
    description = str(metadata["prompt"])
    robot_subject = str(metadata["robotic manipulator"])
    manipulated_object = metadata.get("manipulated object")
    subject_prompts = rubrics["subject_stability"]["prompts"]
    robot_prompt = subject_prompts["robot_subject"]["text"].format(
        robot_subject=robot_subject,
        terminal_option=analyze_phrase(robot_subject),
        json_example=_OPTION_JSON_EXAMPLE,
    )
    object_prompt = None
    if manipulated_object is not None:
        object_prompt = subject_prompts["manipulated_object"]["text"].format(
            manipulated_object=str(manipulated_object),
            json_example=_OPTION_JSON_EXAMPLE,
        )
    metadata["evaluation_prompts"] = {
        "subject_stability": {"q1": robot_prompt, "q2": object_prompt},
        "physical_plausibility": rubrics["physical_plausibility"]["prompt"][
            "text"
        ].format(
            view=view,
            description=description,
            json_example=_SCORE_JSON_EXAMPLE,
        ),
        "task_adherence": rubrics["task_adherence"]["prompt"]["text"].format(
            view=view,
            description=description,
            json_example=_SCORE_JSON_EXAMPLE,
        ),
    }
    metadata["prompt_evidence"] = {
        name: (
            {
                key: {
                    "path": record["path"],
                    "sha256": record["sha256"],
                }
                for key, record in rubric["prompts"].items()
            }
            if name == "subject_stability"
            else {
                "path": rubric["prompt"]["path"],
                "sha256": rubric["prompt"]["sha256"],
            }
        )
        for name, rubric in rubrics.items()
    }
    return metadata


__all__ = ["render_bench_vlm_prompts"]
