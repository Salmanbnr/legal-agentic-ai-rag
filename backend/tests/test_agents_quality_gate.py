"""Tests for the deterministic quality gate."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.agents.quality_gate import (
    GATE_ABSTAIN,
    GATE_ACCEPT,
    GATE_ASK_CLARIFICATION,
    GATE_RETRY_RETRIEVAL,
    coverage_summary,
    decide,
)
from app.agents.state import (
    REASON_AUTHORIZATION_FAILED,
    REASON_BUDGET_EXCEEDED,
    AgenticRAGState,
    RetrievedEvidence,
    Subquestion,
)
from app.core.config import Settings


def _state(
    *,
    spans: int = 0,
    documents: int = 0,
    retries_used: int = 0,
    reason_codes: list[str] | None = None,
    question: str = "Did the tenant pay the rent?",
    plan: list[Subquestion] | None = None,
) -> AgenticRAGState:
    """Build a state with the given retrieval results."""
    document_ids = [uuid4() for _ in range(max(1, documents))]
    results = [
        RetrievedEvidence(
            subquestion_id="sq-1",
            evidence_id=uuid4(),
            document_id=document_ids[index % len(document_ids)],
            page_start=1,
            page_end=1,
        )
        for index in range(spans)
    ]
    return AgenticRAGState(
        user_id=uuid4(),
        case_id=uuid4(),
        question=question,
        plan=plan
        if plan is not None
        else [Subquestion(id="sq-1", text="question", purpose="purpose")],
        retrieval_results=results,
        retries_used=retries_used,
        reason_codes=list(reason_codes or []),
    )


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None).model_copy(update=overrides)


def test_enough_evidence_is_accepted() -> None:
    state = _state(spans=2, documents=1)

    assert decide(state) == GATE_ACCEPT


def test_weak_coverage_with_retries_left_retries() -> None:
    state = _state(spans=0, retries_used=0)

    assert decide(state) == GATE_RETRY_RETRIEVAL


def test_weak_coverage_without_retries_abstains() -> None:
    state = _state(spans=1, retries_used=2)

    assert decide(state) == GATE_ABSTAIN


def test_an_authorization_failure_abstains() -> None:
    state = _state(spans=5, documents=2, reason_codes=[REASON_AUTHORIZATION_FAILED])

    assert decide(state) == GATE_ABSTAIN


def test_a_budget_failure_abstains() -> None:
    state = _state(spans=5, documents=2, reason_codes=[REASON_BUDGET_EXCEEDED])

    assert decide(state) == GATE_ABSTAIN


def test_a_vague_question_asks_for_clarification() -> None:
    state = _state(spans=3, documents=1, question="rent?")

    assert decide(state) == GATE_ASK_CLARIFICATION


def test_an_empty_plan_asks_for_clarification() -> None:
    state = _state(spans=3, documents=1, plan=[])

    assert decide(state) == GATE_ASK_CLARIFICATION


def test_the_coverage_summary_counts_spans_and_documents() -> None:
    state = _state(spans=3, documents=2)

    summary = coverage_summary(state)

    assert summary["span_count"] == 3
    assert summary["document_count"] == 2
    assert summary["coverage_ok"] is True


def test_the_gate_uses_the_configured_retry_limit() -> None:
    state = _state(spans=0, retries_used=1)

    assert decide(state, settings=_settings(agent_max_retrieval_retries=1)) == (
        GATE_ABSTAIN
    )
    assert decide(
        state, settings=_settings(agent_max_retrieval_retries=2)
    ) == GATE_RETRY_RETRIEVAL
