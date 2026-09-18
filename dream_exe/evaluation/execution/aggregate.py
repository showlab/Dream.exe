"""Pure aggregation for saved task-success observations.

This module does not discover bench samples or simulator outputs.  Callers
provide either already-loaded payload specs, flattened records, or explicit
paths. The public contract exposes adjusted binary success ``SR-B``, partial
progress ``SR-P``, and the paper sub-goal metrics ``Rel``, ``Place``, ``Art``,
and ``Core``. Historical checker variants remain raw artifact evidence.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from .task_success import CURRENT_TASK_SUCCESS_PROTOCOL

CURRENT_TASK_SUCCESS_RATE_SCHEMA = "task_success_rate_summary"
CURRENT_TASK_SUCCESS_RATE_PROTOCOL = CURRENT_TASK_SUCCESS_PROTOCOL

CANONICAL_TASK_SUCCESS_COLUMNS = (
    "SR-B",
    "SR-P",
    "Rel",
    "Place",
    "Art",
    "Core",
)
PAPER_TASK_SUCCESS_COLUMNS = CANONICAL_TASK_SUCCESS_COLUMNS
TASK_SUCCESS_RATE_COLUMNS = CANONICAL_TASK_SUCCESS_COLUMNS

_LEGACY_ALIASES = {
    "SR": "SR-B",
    "P": "SR-P",
}
_BINARY_COLUMNS = {"SR-B"}
_PAYLOAD_PATHS: dict[str, tuple[tuple[str, ...], ...]] = {
    "SR-B": (("final_task_check_success",),),
    "SR-P": (("final_task_meta", "metrics", "primary_progress"),),
    "Rel": (("final_task_meta", "metrics", "quality", "release_ok"),),
    "Place": (("final_task_meta", "metrics", "quality", "place_quality"),),
    "Art": (("final_task_meta", "metrics", "quality", "articulation_quality"),),
    "Core": (
        (
            "final_task_meta",
            "metrics",
            "signals",
            "fraction",
            "core_subgoal_fraction",
        ),
    ),
}


def task_success_rate_metadata() -> dict[str, Any]:
    """Return detached metadata for the public task-success metric contract."""

    metrics: dict[str, dict[str, Any]] = {}
    for name in TASK_SUCCESS_RATE_COLUMNS:
        source_name = _LEGACY_ALIASES.get(name, name)
        paths = _PAYLOAD_PATHS[source_name]
        role = "diagnostic"
        if name in CANONICAL_TASK_SUCCESS_COLUMNS:
            role = "canonical"
        elif name in _LEGACY_ALIASES:
            role = "legacy_alias"
        metrics[name] = {
            "role": role,
            "source_paths": [".".join(path) for path in paths],
            "binary": name in _BINARY_COLUMNS,
        }
        if name in _LEGACY_ALIASES:
            metrics[name]["alias_of"] = _LEGACY_ALIASES[name]

    return {
        "format": CURRENT_TASK_SUCCESS_RATE_SCHEMA,
        "metric_family": "task_success",
        "protocol": {
            "id": CURRENT_TASK_SUCCESS_RATE_PROTOCOL,
            "binary_metric": "SR-B",
            "binary_source": "final_task_check_success",
            "partial_metric": "SR-P",
            "partial_source": ("final_task_meta.metrics.primary_progress"),
            "subgoal_metrics": ["Rel", "Place", "Art", "Core"],
            "success_policy": "adjusted",
            "observation_timing": "final",
            "success_once": {
                "aggregated": False,
                "reason": ("not inferred from final-state task-success artifacts"),
            },
        },
        "canonical_columns": list(CANONICAL_TASK_SUCCESS_COLUMNS),
        "columns": list(TASK_SUCCESS_RATE_COLUMNS),
        "metrics": metrics,
        "statistics": {
            "denominator": "valid values independently per column",
            "standard_deviation": "population",
        },
        "value_validation": {
            "finite_required": False,
            "canonical_range_enforced": False,
            "compatibility_note": (
                "NaN is treated as missing. Existing finite and infinite "
                "numeric values are retained for saved-artifact compatibility; "
                "the public metric names remain SR-B, SR-P, Rel, Place, Art, and Core."
            ),
        },
    }


def _nested_value(
    payload: Mapping[str, Any],
    path: tuple[str, ...],
) -> Any:
    value: Any = payload
    for key in path:
        if not isinstance(value, Mapping) or key not in value:
            return None
        value = value[key]
    return value


def _first_payload_value(
    payload: Mapping[str, Any],
    paths: tuple[tuple[str, ...], ...],
) -> Any:
    for path in paths:
        value = _nested_value(payload, path)
        if value is not None:
            return value
    return None


def _as_metric_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    if number != number:
        return None
    return number


def task_success_record_from_payload(
    payload: Mapping[str, Any],
    *,
    uid: str,
    level: str | None = None,
    source_path: str | Path | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Flatten one explicit task-success payload into an aggregate record.

    ``uid``, ``level`` and ``source_path`` are caller-owned labels.  They are
    never inferred from a bench root or directory shape.
    """

    if not isinstance(payload, Mapping):
        raise TypeError("payload must be a mapping")
    normalized_uid = str(uid or "").strip()
    if not normalized_uid:
        raise ValueError("uid is required")
    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping when provided")

    row: dict[str, Any] = {
        "uid": normalized_uid,
        "task_name": payload.get("task_name"),
    }
    if level is not None:
        row["level"] = str(level or "unknown")
    if source_path is not None:
        source = Path(source_path).expanduser().as_posix()
        row["source_path"] = source
        row["out_path"] = source
    if metadata is not None:
        row["metadata"] = dict(metadata)

    sr_b = _as_metric_float(_first_payload_value(payload, _PAYLOAD_PATHS["SR-B"]))
    sr_p = _as_metric_float(_first_payload_value(payload, _PAYLOAD_PATHS["SR-P"]))
    row["SR-B"] = sr_b
    row["SR-P"] = sr_p

    for name in TASK_SUCCESS_RATE_COLUMNS:
        if name in {"SR-B", "SR-P", "SR", "P"}:
            continue
        row[name] = _as_metric_float(
            _first_payload_value(payload, _PAYLOAD_PATHS[name])
        )
    return row


def _error_record(
    *,
    index: int,
    error: str,
    uid: Any = None,
    path: Any = None,
) -> dict[str, Any]:
    item: dict[str, Any] = {
        "index": int(index),
        "uid": None if uid is None else str(uid),
        "error": str(error),
    }
    if path is not None:
        source = Path(str(path)).expanduser().as_posix()
        item["source_path"] = source
        item["out_path"] = source
    return item


def task_success_records_from_payloads(
    payload_specs: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Build records from explicit ``{uid, payload, ...}`` specifications."""

    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, spec in enumerate(payload_specs):
        if not isinstance(spec, Mapping):
            errors.append(
                _error_record(
                    index=index,
                    error="invalid_payload_spec: expected_mapping",
                )
            )
            continue
        uid = spec.get("uid")
        payload = spec.get("payload")
        if not str(uid or "").strip():
            errors.append(
                _error_record(
                    index=index,
                    uid=uid,
                    path=spec.get("path"),
                    error="invalid_payload_spec: missing_uid",
                )
            )
            continue
        if not isinstance(payload, Mapping):
            errors.append(
                _error_record(
                    index=index,
                    uid=uid,
                    path=spec.get("path"),
                    error="invalid_payload_spec: payload_not_object",
                )
            )
            continue
        try:
            records.append(
                task_success_record_from_payload(
                    payload,
                    uid=str(uid),
                    level=spec.get("level"),
                    source_path=spec.get("path"),
                    metadata=spec.get("metadata"),
                )
            )
        except (TypeError, ValueError) as error:
            errors.append(
                _error_record(
                    index=index,
                    uid=uid,
                    path=spec.get("path"),
                    error=(f"invalid_payload_spec: {type(error).__name__}: {error}"),
                )
            )
    return records, errors


def _load_json_mapping(path: Path) -> Mapping[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, Mapping):
        raise TypeError("json_not_object")
    return payload


def task_success_records_from_paths(
    path_specs: Iterable[Mapping[str, Any]],
    *,
    loader: Callable[[Path], Mapping[str, Any]] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Read records from explicit ``{uid, path, ...}`` specifications.

    No run key, bench root, generated-model namespace, or registry is
    discovered.  The function is read-only.
    """

    load = _load_json_mapping if loader is None else loader
    records: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for index, spec in enumerate(path_specs):
        if not isinstance(spec, Mapping):
            errors.append(
                _error_record(
                    index=index,
                    error="invalid_path_spec: expected_mapping",
                )
            )
            continue
        uid = spec.get("uid")
        raw_path = spec.get("path")
        if not str(uid or "").strip():
            errors.append(
                _error_record(
                    index=index,
                    uid=uid,
                    path=raw_path,
                    error="invalid_path_spec: missing_uid",
                )
            )
            continue
        if raw_path is None or not str(raw_path).strip():
            errors.append(
                _error_record(
                    index=index,
                    uid=uid,
                    error="invalid_path_spec: missing_path",
                )
            )
            continue
        try:
            path = Path(raw_path).expanduser()
        except (TypeError, ValueError):
            errors.append(
                _error_record(
                    index=index,
                    uid=uid,
                    error=(
                        "invalid_path_spec: invalid_path_type: "
                        f"{type(raw_path).__name__}"
                    ),
                )
            )
            continue
        if not path.exists():
            errors.append(
                _error_record(
                    index=index,
                    uid=uid,
                    path=path,
                    error="missing_output_json",
                )
            )
            continue
        try:
            payload = load(path)
        except TypeError as error:
            message = str(error)
            if message == "json_not_object":
                reason = message
            else:
                reason = f"read_json_failed: {type(error).__name__}: {error}"
            errors.append(
                _error_record(
                    index=index,
                    uid=uid,
                    path=path,
                    error=reason,
                )
            )
            continue
        except Exception as error:
            errors.append(
                _error_record(
                    index=index,
                    uid=uid,
                    path=path,
                    error=(f"read_json_failed: {type(error).__name__}: {error}"),
                )
            )
            continue
        if not isinstance(payload, Mapping):
            errors.append(
                _error_record(
                    index=index,
                    uid=uid,
                    path=path,
                    error="json_not_object",
                )
            )
            continue
        try:
            records.append(
                task_success_record_from_payload(
                    payload,
                    uid=str(uid),
                    level=spec.get("level"),
                    source_path=path,
                    metadata=spec.get("metadata"),
                )
            )
        except (TypeError, ValueError) as error:
            errors.append(
                _error_record(
                    index=index,
                    uid=uid,
                    path=path,
                    error=(f"invalid_path_spec: {type(error).__name__}: {error}"),
                )
            )
    return records, errors


def _canonical_alias_value(
    record: Mapping[str, Any],
    *,
    canonical: str,
    legacy: str,
) -> float | None:
    canonical_value = _as_metric_float(record.get(canonical))
    legacy_value = _as_metric_float(record.get(legacy))
    if canonical_value is None:
        return legacy_value
    if legacy_value is not None and canonical_value != legacy_value:
        raise ValueError(
            f"alias_conflict: {canonical}={canonical_value!r} "
            f"does not match {legacy}={legacy_value!r}"
        )
    return canonical_value


def _normalized_record(record: Mapping[str, Any]) -> dict[str, Any]:
    uid = str(record.get("uid", "") or "").strip()
    if not uid:
        raise ValueError("record uid is required")

    row: dict[str, Any] = {
        "uid": uid,
        "task_name": record.get("task_name"),
    }
    for name in ("level", "source_path", "out_path", "metadata"):
        if name in record:
            row[name] = record[name]

    sr_b = _canonical_alias_value(
        record,
        canonical="SR-B",
        legacy="SR",
    )
    sr_p = _canonical_alias_value(
        record,
        canonical="SR-P",
        legacy="P",
    )
    row["SR-B"] = sr_b
    row["SR-P"] = sr_p
    for name in TASK_SUCCESS_RATE_COLUMNS:
        if name in {"SR-B", "SR-P", "SR", "P"}:
            continue
        row[name] = _as_metric_float(record.get(name))
    return row


def _column_stats(
    rows: list[dict[str, Any]],
) -> dict[str, dict[str, int | float | None]]:
    output: dict[str, dict[str, int | float | None]] = {}
    attempted = len(rows)
    for name in TASK_SUCCESS_RATE_COLUMNS:
        values = [
            float(row[name]) for row in rows if isinstance(row.get(name), (int, float))
        ]
        if values:
            count = len(values)
            mean = float(sum(values) / count)
            variance = float(sum((value - mean) ** 2 for value in values) / count)
            std: float | None = float(variance**0.5)
        else:
            count = 0
            mean = None
            std = None
        output[name] = {
            "mean": mean,
            "std": std,
            "count": count,
            "valid_count": count,
            "missing_count": attempted - count,
        }
    return output


def aggregate_task_success_rate(
    records: Iterable[Mapping[str, Any]],
    *,
    by_level: bool = False,
    include_records: bool = False,
    errors: Iterable[Mapping[str, Any]] = (),
    input_count: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate explicit flattened records with per-column denominators.

    Record and error order follows the caller's input order.  Level groups are
    emitted in lexical order, and metric columns follow the public contract.
    """

    if metadata is not None and not isinstance(metadata, Mapping):
        raise TypeError("metadata must be a mapping when provided")

    rows: list[dict[str, Any]] = []
    supplied_errors = list(errors)
    collected_errors = [dict(error) for error in supplied_errors]
    supplied_records = list(records)
    seen_uids: set[str] = set()
    for index, record in enumerate(supplied_records):
        if not isinstance(record, Mapping):
            collected_errors.append(
                _error_record(
                    index=index,
                    error="invalid_record: expected_mapping",
                )
            )
            continue
        try:
            normalized = _normalized_record(record)
        except (TypeError, ValueError) as error:
            collected_errors.append(
                _error_record(
                    index=index,
                    uid=record.get("uid"),
                    path=record.get(
                        "source_path",
                        record.get("out_path"),
                    ),
                    error=(f"invalid_record: {type(error).__name__}: {error}"),
                )
            )
            continue
        uid = str(normalized["uid"])
        if uid in seen_uids:
            collected_errors.append(
                _error_record(
                    index=index,
                    uid=uid,
                    path=normalized.get(
                        "source_path",
                        normalized.get("out_path"),
                    ),
                    error=f"invalid_record: duplicate_uid: {uid}",
                )
            )
            continue
        seen_uids.add(uid)
        rows.append(normalized)

    supplied_total = len(supplied_records) + len(supplied_errors)
    if input_count is None:
        total = supplied_total
    else:
        if (
            isinstance(input_count, bool)
            or not isinstance(input_count, int)
            or input_count < 0
            or input_count != supplied_total
        ):
            raise ValueError(
                "input_count must equal the number of supplied records and errors"
            )
        total = input_count
    attempted_uids = {
        str(item.get("uid") or "").strip()
        for item in (*supplied_records, *supplied_errors)
        if isinstance(item, Mapping) and str(item.get("uid") or "").strip()
    }
    contract = task_success_rate_metadata()
    summary: dict[str, Any] = {
        "format": CURRENT_TASK_SUCCESS_RATE_SCHEMA,
        "metric_family": "task_success",
        "protocol": contract["protocol"],
        "canonical_columns": contract["canonical_columns"],
        "columns": contract["columns"],
        "metric_metadata": contract["metrics"],
        "statistics": contract["statistics"],
        "value_validation": contract["value_validation"],
        "ordering": {
            "rows": "input",
            "errors": "input",
            "columns": "contract",
            "by_level": "lexical",
        },
        "records_total": total,
        "inputs_total": total,
        "uids_total": len(attempted_uids),
        "rows_total": len(rows),
        "error_total": len(collected_errors),
        "column_stats": _column_stats(rows),
        "errors": collected_errors,
    }
    if metadata is not None:
        summary["metadata"] = dict(metadata)

    if by_level:
        grouped: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            level = str(row.get("level", "unknown") or "unknown")
            grouped.setdefault(level, []).append(row)
        summary["by_level"] = {
            level: {
                "uids": len(level_rows),
                "column_stats": _column_stats(level_rows),
            }
            for level, level_rows in sorted(grouped.items())
        }
    if include_records:
        summary["rows"] = rows
    return summary


def aggregate_task_success_payloads(
    payload_specs: Iterable[Mapping[str, Any]],
    *,
    by_level: bool = False,
    include_records: bool = False,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Aggregate explicit ``{uid, payload, ...}`` specifications."""

    specs = list(payload_specs)
    records, errors = task_success_records_from_payloads(specs)
    return aggregate_task_success_rate(
        records,
        by_level=by_level,
        include_records=include_records,
        errors=errors,
        input_count=len(specs),
        metadata=metadata,
    )


def aggregate_task_success_paths(
    path_specs: Iterable[Mapping[str, Any]],
    *,
    by_level: bool = False,
    include_records: bool = False,
    metadata: Mapping[str, Any] | None = None,
    loader: Callable[[Path], Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Aggregate explicit ``{uid, path, ...}`` specifications read-only."""

    specs = list(path_specs)
    records, errors = task_success_records_from_paths(
        specs,
        loader=loader,
    )
    return aggregate_task_success_rate(
        records,
        by_level=by_level,
        include_records=include_records,
        errors=errors,
        input_count=len(specs),
        metadata=metadata,
    )


__all__ = [
    "CURRENT_TASK_SUCCESS_RATE_PROTOCOL",
    "CURRENT_TASK_SUCCESS_RATE_SCHEMA",
    "CANONICAL_TASK_SUCCESS_COLUMNS",
    "PAPER_TASK_SUCCESS_COLUMNS",
    "TASK_SUCCESS_RATE_COLUMNS",
    "aggregate_task_success_paths",
    "aggregate_task_success_payloads",
    "aggregate_task_success_rate",
    "task_success_rate_metadata",
    "task_success_record_from_payload",
    "task_success_records_from_paths",
    "task_success_records_from_payloads",
]
