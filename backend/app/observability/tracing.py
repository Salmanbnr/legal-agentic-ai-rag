"""Structured tracing for the agent workflow.

Only identifiers, counts, timings, and status codes are recorded. Document
text, prompts, completions, and credentials never reach a log record or a
trace event.
"""

from __future__ import annotations

import functools
import logging
import time
from typing import Any, Callable, Mapping

from pydantic import BaseModel, ConfigDict, Field

logger = logging.getLogger(__name__)

SAFE_DETAIL_KEYS = (
    "span_count",
    "document_count",
    "subquestion_count",
    "plan_count",
    "gap_count",
    "claim_count",
    "coverage_ok",
    "task_type",
    "limit",
    "graph_version",
    "attempt",
)


class TraceEvent(BaseModel):
    """One node execution, recorded for the trace summary."""

    model_config = ConfigDict(extra="forbid")

    node: str
    status: str = "ok"
    duration_ms: int = 0
    tokens_used: int = 0
    reason_codes: list[str] = Field(default_factory=list)
    details: dict[str, Any] = Field(default_factory=dict)


def safe_details(details: Mapping[str, Any] | None) -> dict[str, Any]:
    """Keep only the detail keys that cannot carry document content."""
    if not details:
        return {}
    return {
        key: value
        for key, value in details.items()
        if key in SAFE_DETAIL_KEYS
    }


def log_node_event(event: TraceEvent, *, graph_version: str | None = None) -> None:
    """Write one structured log line for a node execution."""
    logger.info(
        "agent node finished",
        extra={
            "node": event.node,
            "node_status": event.status,
            "duration_ms": event.duration_ms,
            "tokens_used": event.tokens_used,
            "reason_codes": list(event.reason_codes),
            "details": safe_details(event.details),
            "graph_version": graph_version,
        },
    )


def trace_node(
    node_name: str, *, graph_version: str | None = None
) -> Callable[[Callable[..., dict[str, Any]]], Callable[..., dict[str, Any]]]:
    """Record a node execution into the state trace and the log.

    The wrapped function returns the state update for a graph node. A budget
    failure is turned into a state update carrying the reason code, so the
    graph can finish safely instead of raising.
    """

    def decorator(
        func: Callable[..., dict[str, Any]],
    ) -> Callable[..., dict[str, Any]]:
        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> dict[str, Any]:
            from app.agents.budget import BudgetExceeded

            state = args[0] if args else None
            started = time.perf_counter()
            try:
                update = func(*args, **kwargs) or {}
            except BudgetExceeded as exc:
                duration_ms = int((time.perf_counter() - started) * 1000)
                event = TraceEvent(
                    node=node_name,
                    status="budget_exceeded",
                    duration_ms=duration_ms,
                    tokens_used=int(getattr(state, "tokens_used", 0) or 0),
                    reason_codes=[exc.reason_code],
                    details={"limit": exc.limit},
                )
                log_node_event(event, graph_version=graph_version)
                existing = list(getattr(state, "reason_codes", []) or [])
                return {
                    "reason_codes": [*existing, exc.reason_code],
                    "trace": [event],
                }
            except Exception as exc:
                duration_ms = int((time.perf_counter() - started) * 1000)
                event = TraceEvent(
                    node=node_name,
                    status="error",
                    duration_ms=duration_ms,
                    tokens_used=int(getattr(state, "tokens_used", 0) or 0),
                    reason_codes=[type(exc).__name__],
                )
                log_node_event(event, graph_version=graph_version)
                raise

            duration_ms = int((time.perf_counter() - started) * 1000)
            update = dict(update)
            details = safe_details(update.get("details"))
            update.pop("details", None)
            event = TraceEvent(
                node=node_name,
                status=str(update.get("status", "ok")),
                duration_ms=duration_ms,
                tokens_used=int(update.get("tokens_used", 0) or 0),
                reason_codes=list(update.get("reason_codes", []) or []),
                details=details,
            )
            log_node_event(event, graph_version=graph_version)
            update["trace"] = [*list(update.get("trace", []) or []), event]
            update["steps_used"] = int(
                getattr(state, "steps_used", 0) or 0
            ) + 1
            return update

        return wrapper

    return decorator
