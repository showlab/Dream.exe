"""Aggregate VLM scores across explicitly declared judges."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from .batch import RUBRIC_SUBJECT_STABILITY, SAVED_MEDIA_RUBRICS, _normalize_rubric

VLM_JUDGE_AGGREGATE_SCHEMA = "dream-exe.vlm-judge-aggregate"


def _record_score(record: Mapping[str, Any], *, rubric: str) -> int | None:
    if record.get("status") != "ok":
        return None
    value: Any
    if rubric == RUBRIC_SUBJECT_STABILITY:
        value = record.get("score")
        maximum = 15
    else:
        response = record.get("response")
        value = response.get("score") if isinstance(response, Mapping) else None
        maximum = 5
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value if 1 <= value <= maximum else None


def aggregate_vlm_judges(
    judge_reports: Mapping[str, Mapping[str, Any]],
    *,
    required_judges: Sequence[str],
) -> dict[str, Any]:
    """Mean matching item scores only when every required judge is valid."""

    required = [str(value).strip() for value in required_judges]
    if not required or any(not value for value in required):
        raise ValueError("required_judges must contain non-empty judge IDs")
    if len(set(required)) != len(required):
        raise ValueError("required_judges contains duplicates")
    missing = [judge for judge in required if judge not in judge_reports]
    if missing:
        raise ValueError("missing required VLM judge reports: " + ", ".join(missing))

    rubric: str | None = None
    by_judge: dict[str, dict[str, int | None]] = {}
    identities: set[str] | None = None
    for judge in required:
        report = judge_reports[judge]
        current_rubric = _normalize_rubric(str(report.get("rubric", "")))
        if current_rubric not in SAVED_MEDIA_RUBRICS:
            raise ValueError(f"unsupported rubric for judge {judge!r}")
        if rubric is None:
            rubric = current_rubric
        elif current_rubric != rubric:
            raise ValueError("VLM judge reports use different rubrics")
        records = report.get("records")
        if not isinstance(records, list):
            raise ValueError(f"VLM judge report {judge!r} has no records list")
        scores: dict[str, int | None] = {}
        for index, record in enumerate(records):
            if not isinstance(record, Mapping):
                raise ValueError(f"VLM judge report {judge!r} record {index} is invalid")
            name = str(record.get("name", "") or "").strip()
            if not name or name in scores:
                raise ValueError(f"VLM judge report {judge!r} has invalid item identity")
            scores[name] = _record_score(record, rubric=current_rubric)
        current_identities = set(scores)
        if identities is None:
            identities = current_identities
        elif current_identities != identities:
            raise ValueError("VLM judge reports cover different items")
        by_judge[judge] = scores

    assert rubric is not None and identities is not None
    rows = []
    evaluated = 0
    for name in sorted(identities):
        scores = {judge: by_judge[judge][name] for judge in required}
        values = list(scores.values())
        complete = all(value is not None for value in values)
        mean = (
            sum(int(value) for value in values if value is not None) / len(values)
            if complete
            else None
        )
        evaluated += int(complete)
        rows.append(
            {
                "name": name,
                "status": "evaluated" if complete else "not_evaluated",
                "judge_scores": scores,
                "mean_score": mean,
            }
        )
    return {
        "format": VLM_JUDGE_AGGREGATE_SCHEMA,
        "rubric": rubric,
        "aggregation": "mean",
        "required_judges": required,
        "expected_items": len(rows),
        "evaluated_items": evaluated,
        "coverage": evaluated / len(rows) if rows else 0.0,
        "items": rows,
    }


__all__ = ["VLM_JUDGE_AGGREGATE_SCHEMA", "aggregate_vlm_judges"]
