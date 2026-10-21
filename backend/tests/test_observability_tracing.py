"""Tests for the agent node tracing decorator."""

from __future__ import annotations

import logging
from typing import Any
from uuid import uuid4

import pytest

from app.agents.budget import BudgetExceeded
from app.agents.state import AgenticRAGState
from app.observability.tracing import (
    SAFE_DETAIL_KEYS,
    TraceEvent,
    log_node_event,
    safe_details,
    trace_node,
)


def _state() -> AgenticRAGState:
    return AgenticRAGState(
        user_id=uuid4(),
        case_id=uuid4(),
        question="Did the tenant pay the rent?",
    )


def test_the_decorator_records_the_node_and_duration() -> None:
    @trace_node("retrieve")
    def node(state: AgenticRAGState) -> dict[str, Any]:
        return {"tokens_used": 42, "details": {"span_count": 3}}

    update = node(_state())

    event = update["trace"][0]
    assert isinstance(event, TraceEvent)
    assert event.node == "retrieve"
    assert event.status == "ok"
    assert event.duration_ms >= 0
    assert event.tokens_used == 42
    assert event.details == {"span_count": 3}
    assert update["steps_used"] == 1


def test_the_decorator_keeps_only_safe_details() -> None:
    @trace_node("retrieve")
    def node(state: AgenticRAGState) -> dict[str, Any]:
        return {
            "details": {
                "span_count": 2,
                "document_text": "The tenant failed to pay the rent.",
            }
        }

    update = node(_state())

    assert "document_text" not in update["trace"][0].details
    assert update["trace"][0].details == {"span_count": 2}


def test_safe_details_only_keeps_known_keys() -> None:
    details = safe_details({"span_count": 1, "secret": "value"})

    assert details == {"span_count": 1}
    assert "span_count" in SAFE_DETAIL_KEYS


def test_a_budget_failure_becomes_a_safe_state_update() -> None:
    @trace_node("retrieve")
    def node(state: AgenticRAGState) -> dict[str, Any]:
        raise BudgetExceeded("steps")

    update = node(_state())

    assert "BUDGET_EXCEEDED" in update["reason_codes"]
    assert update["trace"][0].status == "budget_exceeded"
    assert update["trace"][0].details == {"limit": "steps"}


def test_an_unexpected_error_is_recorded_and_raised() -> None:
    @trace_node("retrieve")
    def node(state: AgenticRAGState) -> dict[str, Any]:
        raise RuntimeError("internal detail")

    with pytest.raises(RuntimeError):
        node(_state())


def test_the_log_record_never_contains_document_text(
    caplog: pytest.LogCaptureFixture,
) -> None:
    event = TraceEvent(
        node="retrieve",
        status="ok",
        duration_ms=3,
        tokens_used=10,
        reason_codes=["WEAK_COVERAGE"],
        details={"span_count": 1},
    )

    with caplog.at_level(logging.INFO):
        log_node_event(event, graph_version="agentic-rag-1")

    record = caplog.records[-1]
    assert record.node == "retrieve"
    assert record.duration_ms == 3
    assert record.graph_version == "agentic-rag-1"
    assert "document_text" not in str(record.__dict__)
