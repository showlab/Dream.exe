"""Deterministic supervision for explicit caller-owned task records.

The current GT and generated-video batch scripts share a small observable
orchestration contract: preserve selected task order, split tasks round-robin,
skip already-completed identities, retry bounded attempts, optionally stop
after a final failure, and append enough status evidence to recover progress.

This module owns only that common contract.  It does not discover a bench,
registry, sample, materialization rule, migration, subprocess, GPU, credential,
or pipeline implementation.  Callers supply stable JSON-compatible task
records, a one-task callable, optional coordination and status effects, and an
optional bounded executor.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, wait
import copy
from datetime import datetime, timezone
import json
import re
import threading
from typing import Any


BATCH_SCHEMA = "dream-exe.batch-supervisor"
TRANSITION_SCHEMA = "dream-exe.batch-transition"

RunOne = Callable[..., Mapping[str, Any]]
StatusSink = Callable[[Mapping[str, Any]], Any]
ClaimTask = Callable[..., bool]
ExecutorFactory = Callable[..., Any]
CancelCheck = Callable[[], bool]
Clock = Callable[[], Any]

_NAMED_CREDENTIAL_PATTERN = re.compile(
    (
        r"(?i)\b(api[ _-]?key|access[ _-]?token|"
        r"refresh[ _-]?token|authorization|password|"
        r"client[ _-]?secret|credential)"
        r"(\s*[:=]\s*)([^\s,;]+)"
    )
)
_BEARER_PATTERN = re.compile(r"(?i)\bbearer\s+[^\s,;]+")
_URL_PATTERN = re.compile(r"(?i)\bhttps?://[^\s,;]+")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_error_text(value: Any) -> str:
    """Return bounded error text with common credential forms redacted."""

    try:
        text = str(value or "")
    except Exception:
        text = f"<{type(value).__name__}>"
    # Remove the credential marker as well as its value.  Transition writers
    # deliberately reject even redacted strings that still contain names such
    # as ``api_key`` or ``Authorization``.
    text = _URL_PATTERN.sub("[REDACTED_URL]", text)
    text = _BEARER_PATTERN.sub("[REDACTED_CREDENTIAL]", text)
    text = _NAMED_CREDENTIAL_PATTERN.sub("[REDACTED_CREDENTIAL]", text)
    return (text.strip() or "task failed")[:4096]


def _json_copy(value: Any, *, label: str) -> Any:
    try:
        serialized = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            allow_nan=False,
        )
        return json.loads(serialized)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{label} must be JSON-compatible") from error


def _positive_int(value: Any, *, label: str, maximum: int | None = None) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{label} must be a positive integer")
    if isinstance(value, int):
        normalized = value
    elif isinstance(value, str) and re.fullmatch(
        r"[+-]?\d+",
        value.strip(),
    ):
        normalized = int(value)
    else:
        raise ValueError(f"{label} must be a positive integer")
    if normalized < 1:
        raise ValueError(f"{label} must be a positive integer")
    if maximum is not None and normalized > maximum:
        raise ValueError(f"{label} must not exceed {maximum}")
    return normalized


def _task_id(value: Any, *, label: str = "task_id") -> str:
    normalized = str(value or "").strip()
    if not normalized:
        raise ValueError(f"{label} is required")
    return normalized


def _normalized_tasks(
    tasks: Iterable[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    unique: list[dict[str, Any]] = []
    duplicate_inputs: list[dict[str, Any]] = []
    by_id: dict[str, tuple[int, str]] = {}

    for input_index, raw_task in enumerate(tasks):
        if not isinstance(raw_task, Mapping):
            raise TypeError(f"task {input_index} must be a mapping")
        task = _json_copy(
            dict(raw_task),
            label=f"task {input_index}",
        )
        task_id = _task_id(task.get("task_id"))
        task["task_id"] = task_id
        fingerprint = json.dumps(
            task,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        )
        prior = by_id.get(task_id)
        if prior is not None:
            first_order, first_fingerprint = prior
            if fingerprint != first_fingerprint:
                raise ValueError(f"conflicting task records share task_id {task_id!r}")
            duplicate_inputs.append(
                {
                    "task_id": task_id,
                    "input_index": input_index,
                    "first_order": first_order,
                }
            )
            continue
        order = len(unique)
        by_id[task_id] = (order, fingerprint)
        unique.append(
            {
                "task_id": task_id,
                "order": order,
                "input_index": input_index,
                "task": task,
            }
        )
    return unique, duplicate_inputs


def select_task_shard(
    tasks: Iterable[Mapping[str, Any]],
    *,
    shard_count: int = 1,
    shard_index: int = 0,
) -> dict[str, Any]:
    """Normalize, deduplicate, and select one round-robin task shard.

    ``shard_index`` is zero-based.  First-seen unique input order is the stable
    order used by the current batch builders before ``order % shard_count``.
    Exact duplicate records are reported and run once.  Conflicting records
    with the same identity are rejected before any caller-owned effect.
    """

    count = _positive_int(
        shard_count,
        label="shard_count",
    )
    if isinstance(shard_index, bool):
        raise ValueError("shard_index must be a zero-based integer")
    if isinstance(shard_index, int):
        index = shard_index
    elif isinstance(shard_index, str) and re.fullmatch(
        r"[+-]?\d+",
        shard_index.strip(),
    ):
        index = int(shard_index)
    else:
        raise ValueError("shard_index must be a zero-based integer")
    if index < 0 or index >= count:
        raise ValueError(f"shard_index must be in [0, {count})")

    unique, duplicate_inputs = _normalized_tasks(tasks)
    selected = [
        copy.deepcopy(record)
        for record in unique
        if int(record["order"]) % count == index
    ]
    skipped = [
        {
            "task_id": record["task_id"],
            "order": record["order"],
        }
        for record in unique
        if int(record["order"]) % count != index
    ]
    return {
        "input_task_count": len(unique) + len(duplicate_inputs),
        "unique_task_count": len(unique),
        "shard_count": count,
        "shard_index": index,
        "selected": selected,
        "shard_skipped": skipped,
        "duplicate_inputs": copy.deepcopy(duplicate_inputs),
    }


def _completed_ids(values: Iterable[str]) -> set[str]:
    completed: set[str] = set()
    for index, value in enumerate(values):
        completed.add(
            _task_id(
                value,
                label=f"completed_task_ids[{index}]",
            )
        )
    return completed


def replay_batch_transitions(
    transitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Replay one transition stream into recoverable task state.

    The stream must retain its append order and strictly increasing sequence.
    This detects truncated/reordered/concatenated evidence instead of silently
    inventing a latest state.
    """

    last_sequence = 0
    task_states: dict[str, dict[str, Any]] = {}
    batch_events: list[dict[str, Any]] = []

    terminal_status = {
        "task_completed": "completed",
        "task_resumed": "completed",
        "task_failed": "failed",
        "task_claim_failed": "failed",
        "task_claim_rejected": "claim_rejected",
        "task_cancelled": "cancelled",
        "task_blocked": "blocked_fail_fast",
        "task_planned": "dry_run",
    }
    for input_index, raw_transition in enumerate(transitions):
        if not isinstance(raw_transition, Mapping):
            raise TypeError(f"transition {input_index} must be a mapping")
        transition = _json_copy(
            dict(raw_transition),
            label=f"transition {input_index}",
        )
        if transition.get("format") != TRANSITION_SCHEMA:
            raise ValueError(f"transition {input_index} has an unsupported schema")
        sequence = transition.get("sequence")
        if (
            isinstance(sequence, bool)
            or not isinstance(sequence, int)
            or sequence <= last_sequence
        ):
            raise ValueError("transition sequence must be strictly increasing")
        last_sequence = sequence
        event = str(transition.get("event", "") or "").strip()
        if not event:
            raise ValueError(f"transition {input_index} is missing event")
        task_id_value = transition.get("task_id")
        if task_id_value is None:
            batch_events.append(transition)
            continue
        task_id = _task_id(task_id_value)
        state = task_states.setdefault(
            task_id,
            {
                "task_id": task_id,
                "status": "pending",
                "attempts": 0,
                "last_sequence": 0,
                "last_event": None,
            },
        )
        attempt = transition.get("attempt")
        if isinstance(attempt, int) and not isinstance(attempt, bool):
            state["attempts"] = max(
                int(state["attempts"]),
                attempt,
            )
        state["last_sequence"] = sequence
        state["last_event"] = event
        if event in terminal_status:
            state["status"] = terminal_status[event]
        elif event == "task_claimed":
            state["status"] = "claimed"
        elif event == "attempt_started":
            state["status"] = "running"
        elif event == "attempt_failed":
            state["status"] = "retry_pending"
        elif event == "attempt_succeeded":
            state["status"] = "finishing"

    completed = sorted(
        task_id
        for task_id, state in task_states.items()
        if state["status"] == "completed"
    )
    return {
        "format": BATCH_SCHEMA,
        "last_sequence": last_sequence,
        "completed_task_ids": completed,
        "task_states": copy.deepcopy(task_states),
        "batch_events": copy.deepcopy(batch_events),
    }


def run_batch_supervisor(
    *,
    tasks: Iterable[Mapping[str, Any]],
    run_one: RunOne,
    completed_task_ids: Iterable[str] = (),
    prior_transitions: Sequence[Mapping[str, Any]] = (),
    max_attempts: int = 1,
    shard_count: int = 1,
    shard_index: int = 0,
    max_workers: int = 1,
    executor_factory: ExecutorFactory | None = None,
    claim_task: ClaimTask | None = None,
    status_sink: StatusSink | None = None,
    clock: Clock | None = None,
    is_cancelled: CancelCheck | None = None,
    dry_run: bool = False,
    continue_on_error: bool = True,
    revalidate_completed_tasks: bool = False,
) -> dict[str, Any]:
    """Run explicit task records with bounded retries and recoverable status.

    ``run_one`` is called as
    ``run_one(task=<detached mapping>, attempt=<one-based int>,
    is_cancelled=<callable>)`` and must return a JSON-compatible mapping with
    a real boolean ``ok`` field.

    ``max_workers > 1`` requires an injected executor factory supporting the
    context-manager, ``submit``, and standard Future contracts.  The supervisor
    never selects a process/thread implementation.

    Transition creation and ``status_sink`` calls are serialized even when
    tasks run concurrently.  The sink may still be called from a worker thread
    and therefore must not depend on main-thread affinity.

    ``claim_task`` is called immediately before submission as
    ``claim_task(task_id=..., task=<detached mapping>)``.  It must atomically
    return ``True`` only to one supervisor when cross-supervisor deduplication
    is needed.  Within one call, first-seen task-ID deduplication guarantees at
    most one submission.  A retry retains the original claim.  A durable claim
    implementation must use a recoverable lease or release failed/cancelled
    claims when it consumes the corresponding status transitions.

    Dry-run returns the exact selected plan and in-memory transitions but does
    not invoke ``claim_task``, ``run_one``, ``executor_factory``, or the external
    ``status_sink``.  Supplying a previously persisted transition stream both
    resumes its completed task IDs and continues its sequence numbers, so the
    old and new append-only records remain replayable as one stream.

    ``revalidate_completed_tasks=True`` still validates explicit completed
    task IDs and replays prior transitions to continue their sequence, but it
    submits those historical completed identities again instead of treating
    them as a supervisor-level skip set.  The delegated runner is then
    responsible for verified reuse or rerun decisions.
    """

    if not callable(run_one):
        raise TypeError("run_one must be callable")
    attempts_limit = _positive_int(
        max_attempts,
        label="max_attempts",
    )
    workers = _positive_int(
        max_workers,
        label="max_workers",
        maximum=64,
    )
    if workers > 1 and executor_factory is None:
        raise ValueError("max_workers > 1 requires an explicit executor_factory")
    if executor_factory is not None and not callable(executor_factory):
        raise TypeError("executor_factory must be callable")
    if claim_task is not None and not callable(claim_task):
        raise TypeError("claim_task must be callable")
    if status_sink is not None and not callable(status_sink):
        raise TypeError("status_sink must be callable")
    if clock is not None and not callable(clock):
        raise TypeError("clock must be callable")
    if is_cancelled is not None and not callable(is_cancelled):
        raise TypeError("is_cancelled must be callable")

    selection = select_task_shard(
        tasks,
        shard_count=shard_count,
        shard_index=shard_index,
    )
    prior = [
        _json_copy(
            dict(transition),
            label=f"prior transition {index}",
        )
        for index, transition in enumerate(prior_transitions)
    ]
    prior_state = (
        replay_batch_transitions(prior)
        if prior
        else {
            "last_sequence": 0,
            "completed_task_ids": [],
        }
    )
    completed = _completed_ids(completed_task_ids)
    completed.update(prior_state["completed_task_ids"])
    if bool(revalidate_completed_tasks):
        completed.clear()
    now = clock if clock is not None else _utc_now
    external_cancel = is_cancelled if is_cancelled is not None else (lambda: False)
    fail_fast = threading.Event()
    emit_lock = threading.Lock()
    clock_lock = threading.Lock()
    transitions: list[dict[str, Any]] = []
    sequence = int(prior_state["last_sequence"])

    def cancellation_reason() -> str | None:
        if bool(external_cancel()):
            return "cancellation_requested"
        if fail_fast.is_set():
            return "earlier_task_failed"
        return None

    def cancelled() -> bool:
        return cancellation_reason() is not None

    def clock_value() -> Any:
        with clock_lock:
            timestamp = now()
            if isinstance(timestamp, datetime):
                timestamp = timestamp.isoformat()
            return _json_copy(
                timestamp,
                label="clock result",
            )

    def emit(
        event: str,
        *,
        publish: bool = True,
        **fields: Any,
    ) -> dict[str, Any]:
        nonlocal sequence
        with emit_lock:
            sequence += 1
            transition = {
                "format": TRANSITION_SCHEMA,
                "sequence": sequence,
                "at": clock_value(),
                "event": str(event),
                **_json_copy(
                    fields,
                    label=f"{event} transition",
                ),
            }
            transitions.append(transition)
            if publish and not bool(dry_run) and status_sink is not None:
                status_sink(copy.deepcopy(transition))
            return copy.deepcopy(transition)

    emit(
        "batch_started",
        input_task_count=selection["input_task_count"],
        unique_task_count=selection["unique_task_count"],
        selected_task_count=len(selection["selected"]),
        shard_count=selection["shard_count"],
        shard_index=selection["shard_index"],
        max_attempts=attempts_limit,
        max_workers=workers,
        dry_run=bool(dry_run),
        continue_on_error=bool(continue_on_error),
        revalidate_completed_tasks=bool(revalidate_completed_tasks),
        prior_transition_count=len(prior),
        prior_last_sequence=int(prior_state["last_sequence"]),
    )

    items_by_order: dict[int, dict[str, Any]] = {}
    runnable: list[dict[str, Any]] = []
    for prepared in selection["selected"]:
        task_id = prepared["task_id"]
        base = {
            "task_id": task_id,
            "order": prepared["order"],
            "input_index": prepared["input_index"],
            "task": copy.deepcopy(prepared["task"]),
            "attempts": [],
        }
        if task_id in completed:
            base["status"] = "skipped_completed"
            items_by_order[prepared["order"]] = base
            emit(
                "task_resumed",
                task_id=task_id,
                order=prepared["order"],
            )
        elif bool(dry_run):
            base["status"] = "dry_run"
            items_by_order[prepared["order"]] = base
            emit(
                "task_planned",
                publish=False,
                task_id=task_id,
                order=prepared["order"],
            )
        else:
            runnable.append(prepared)

    def failure(
        *,
        task_id: str,
        order: int,
        error: BaseException,
    ) -> dict[str, Any]:
        return {
            "type": type(error).__name__,
            "message": sanitize_error_text(error),
            "task_id": task_id,
            "order": order,
        }

    def claimed_or_terminal(
        prepared: Mapping[str, Any],
    ) -> dict[str, Any] | None:
        task_id = str(prepared["task_id"])
        order = int(prepared["order"])
        base = {
            "task_id": task_id,
            "order": order,
            "input_index": prepared["input_index"],
            "task": copy.deepcopy(prepared["task"]),
            "attempts": [],
        }
        reason = cancellation_reason()
        if reason is not None:
            status = (
                "blocked_fail_fast" if reason == "earlier_task_failed" else "cancelled"
            )
            base["status"] = status
            emit(
                ("task_blocked" if status == "blocked_fail_fast" else "task_cancelled"),
                task_id=task_id,
                order=order,
                reason=reason,
            )
            return base
        if claim_task is not None:
            try:
                claimed = claim_task(
                    task_id=task_id,
                    task=copy.deepcopy(prepared["task"]),
                )
                if not isinstance(claimed, bool):
                    raise TypeError("claim_task must return bool")
            except Exception as error:
                base["status"] = "failed"
                base["error"] = failure(
                    task_id=task_id,
                    order=order,
                    error=error,
                )
                emit(
                    "task_claim_failed",
                    task_id=task_id,
                    order=order,
                    error=base["error"],
                )
                if not bool(continue_on_error):
                    fail_fast.set()
                return base
            if not claimed:
                base["status"] = "claim_rejected"
                emit(
                    "task_claim_rejected",
                    task_id=task_id,
                    order=order,
                )
                return base
        emit(
            "task_claimed",
            task_id=task_id,
            order=order,
        )
        return None

    def execute_claimed(
        prepared: Mapping[str, Any],
    ) -> dict[str, Any]:
        task_id = str(prepared["task_id"])
        order = int(prepared["order"])
        item: dict[str, Any] = {
            "task_id": task_id,
            "order": order,
            "input_index": prepared["input_index"],
            "task": copy.deepcopy(prepared["task"]),
            "status": "pending",
            "attempts": [],
        }
        for attempt in range(1, attempts_limit + 1):
            reason = cancellation_reason()
            if reason is not None:
                item["status"] = (
                    "blocked_fail_fast"
                    if reason == "earlier_task_failed"
                    else "cancelled"
                )
                emit(
                    (
                        "task_blocked"
                        if item["status"] == "blocked_fail_fast"
                        else "task_cancelled"
                    ),
                    task_id=task_id,
                    order=order,
                    attempt=attempt,
                    reason=reason,
                )
                return item

            started_at = clock_value()
            emit(
                "attempt_started",
                task_id=task_id,
                order=order,
                attempt=attempt,
            )
            attempt_record: dict[str, Any] = {
                "attempt": attempt,
                "started_at": started_at,
                "status": "running",
            }
            try:
                outcome_raw = run_one(
                    task=copy.deepcopy(prepared["task"]),
                    attempt=attempt,
                    is_cancelled=cancelled,
                )
                if not isinstance(outcome_raw, Mapping):
                    raise TypeError("run_one must return a mapping")
                outcome = _json_copy(
                    dict(outcome_raw),
                    label="run_one result",
                )
                if not isinstance(outcome.get("ok"), bool):
                    raise TypeError("run_one result requires a boolean 'ok'")
            except Exception as error:
                attempt_record["status"] = "failed"
                attempt_record["error"] = failure(
                    task_id=task_id,
                    order=order,
                    error=error,
                )
                event_fields = {
                    "task_id": task_id,
                    "order": order,
                    "attempt": attempt,
                    "error": attempt_record["error"],
                }
            else:
                attempt_record["outcome"] = outcome
                if outcome["ok"]:
                    attempt_record["status"] = "completed"
                    attempt_record["finished_at"] = clock_value()
                    item["attempts"].append(attempt_record)
                    item["status"] = "completed"
                    item["result"] = outcome
                    emit(
                        "attempt_succeeded",
                        task_id=task_id,
                        order=order,
                        attempt=attempt,
                    )
                    emit(
                        "task_completed",
                        task_id=task_id,
                        order=order,
                        attempt=attempt,
                    )
                    return item
                attempt_record["status"] = "failed"
                event_fields = {
                    "task_id": task_id,
                    "order": order,
                    "attempt": attempt,
                    "outcome": outcome,
                }

            attempt_record["finished_at"] = clock_value()
            item["attempts"].append(attempt_record)
            emit(
                "attempt_failed",
                **event_fields,
            )
            if attempt < attempts_limit and not cancelled():
                continue
            reason = cancellation_reason()
            if reason is not None:
                item["status"] = (
                    "blocked_fail_fast"
                    if reason == "earlier_task_failed"
                    else "cancelled"
                )
                emit(
                    (
                        "task_blocked"
                        if item["status"] == "blocked_fail_fast"
                        else "task_cancelled"
                    ),
                    task_id=task_id,
                    order=order,
                    attempt=attempt,
                    reason=reason,
                )
                return item
            item["status"] = "failed"
            if "error" in attempt_record:
                item["error"] = copy.deepcopy(attempt_record["error"])
            else:
                item["result"] = copy.deepcopy(attempt_record["outcome"])
            emit(
                "task_failed",
                task_id=task_id,
                order=order,
                attempt=attempt,
            )
            if not bool(continue_on_error):
                fail_fast.set()
            return item
        raise AssertionError("unreachable attempt loop")

    def claim_and_run(
        prepared: Mapping[str, Any],
    ) -> dict[str, Any]:
        terminal = claimed_or_terminal(prepared)
        if terminal is not None:
            return terminal
        return execute_claimed(prepared)

    if not bool(dry_run):
        if workers == 1:
            for prepared in runnable:
                item = claim_and_run(prepared)
                items_by_order[int(prepared["order"])] = item
        elif runnable:
            assert executor_factory is not None
            next_index = 0
            active: dict[Any, Mapping[str, Any]] = {}
            with executor_factory(max_workers=workers) as executor:
                if not hasattr(executor, "submit"):
                    raise TypeError("executor must provide submit")

                def fill_slots() -> None:
                    nonlocal next_index
                    while (
                        next_index < len(runnable)
                        and len(active) < workers
                        and not cancelled()
                    ):
                        prepared = runnable[next_index]
                        next_index += 1
                        terminal = claimed_or_terminal(prepared)
                        if terminal is not None:
                            items_by_order[int(prepared["order"])] = terminal
                            continue
                        future = executor.submit(
                            execute_claimed,
                            prepared,
                        )
                        active[future] = prepared

                fill_slots()
                while active:
                    done, _ = wait(
                        tuple(active),
                        timeout=0.1,
                        return_when=FIRST_COMPLETED,
                    )
                    if not done:
                        continue
                    for future in sorted(
                        done,
                        key=lambda item: int(active[item]["order"]),
                    ):
                        prepared = active.pop(future)
                        item = future.result()
                        items_by_order[int(prepared["order"])] = item
                    fill_slots()

            remaining = runnable[next_index:]
            for prepared in remaining:
                item = claimed_or_terminal(prepared)
                if item is None:
                    raise AssertionError("unscheduled task was unexpectedly claimable")
                items_by_order[int(prepared["order"])] = item

    items = [
        items_by_order[int(prepared["order"])] for prepared in selection["selected"]
    ]
    counts: dict[str, int] = {}
    for item in items:
        status = str(item.get("status", "") or "unknown")
        counts[status] = counts.get(status, 0) + 1

    failure_count = int(counts.get("failed", 0))
    cancelled_count = int(counts.get("cancelled", 0))
    blocked_count = int(counts.get("blocked_fail_fast", 0))
    partial_count = int(counts.get("claim_rejected", 0))
    if bool(dry_run):
        status = "dry_run"
    elif failure_count:
        status = "failed"
    elif cancelled_count:
        status = "cancelled"
    elif blocked_count or partial_count:
        status = "partial"
    else:
        status = "ok"

    emit(
        ("batch_cancelled" if status == "cancelled" else "batch_completed"),
        status=status,
        status_counts=counts,
    )
    return {
        "format": BATCH_SCHEMA,
        "status": status,
        "ok": status == "ok",
        "exit_code": (1 if status in {"failed", "cancelled"} else 0),
        "input_task_count": selection["input_task_count"],
        "unique_task_count": selection["unique_task_count"],
        "selected_task_count": len(selection["selected"]),
        "shard_count": selection["shard_count"],
        "shard_index": selection["shard_index"],
        "max_attempts": attempts_limit,
        "max_workers": workers,
        "dry_run": bool(dry_run),
        "continue_on_error": bool(continue_on_error),
        "prior_transition_count": len(prior),
        "status_counts": counts,
        "failure_count": failure_count,
        "cancelled_count": cancelled_count,
        "blocked_count": blocked_count,
        "attempt_count": sum(len(item.get("attempts", ())) for item in items),
        "items": copy.deepcopy(items),
        "duplicate_inputs": copy.deepcopy(selection["duplicate_inputs"]),
        "shard_skipped": copy.deepcopy(selection["shard_skipped"]),
        "transitions": copy.deepcopy(transitions),
    }


__all__ = [
    "BATCH_SCHEMA",
    "TRANSITION_SCHEMA",
    "CancelCheck",
    "ClaimTask",
    "Clock",
    "ExecutorFactory",
    "RunOne",
    "StatusSink",
    "replay_batch_transitions",
    "run_batch_supervisor",
    "sanitize_error_text",
    "select_task_shard",
]
