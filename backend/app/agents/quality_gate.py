"""The deterministic quality gate of the agentic workflow.

The gate decides whether the workflow has enough evidence to continue. It
never calls a model: the decision depends only on the retrieved evidence, the
coverage it gives, the remaining retries, and the budget.
"""

from __future__ import annotations

from typing import Any

from app.agents.state import (
    REASON_AUTHORIZATION_FAILED,
    REASON_BUDGET_EXCEEDED,
    REASON_NO_EVIDENCE,
    REASON_WEAK_COVERAGE,
)
from app.core.config import Settings, get_settings

GATE_ACCEPT = "accept"
GATE_RETRY_RETRIEVAL = "retry_retrieval"
GATE_ABSTAIN = "abstain"
GATE_ASK_CLARIFICATION = "ask_clarification"

MIN_EVIDENCE_SPANS = 2
MIN_DOCUMENTS_COVERED = 1


def coverage_summary(state: Any) -> dict[str, int | bool]:
    """Return the coverage numbers of the retrieved evidence."""
    results = list(getattr(state, "retrieval_results", []) or [])
    documents = {str(item.document_id) for item in results}
    span_count = len(results)
    documents_covered = len(documents)
    coverage_ok = (
        span_count >= MIN_EVIDENCE_SPANS
        and documents_covered >= MIN_DOCUMENTS_COVERED
    )
    return {
        "span_count": span_count,
        "document_count": documents_covered,
        "coverage_ok": coverage_ok,
    }


def decide(
    state: Any, *, settings: Settings | None = None
) -> str:
    """Decide how the workflow continues.

    Args:
        state: The current agent state.
        settings: Optional validated settings for the retry limit.

    Returns:
        One of ``accept``, ``retry_retrieval``, ``abstain``, or
        ``ask_clarification``.
    """
    resolved = settings or get_settings()
    reason_codes = list(getattr(state, "reason_codes", []) or [])

    if REASON_AUTHORIZATION_FAILED in reason_codes:
        return GATE_ABSTAIN
    if REASON_BUDGET_EXCEEDED in reason_codes:
        return GATE_ABSTAIN

    question = (getattr(state, "question", "") or "").strip()
    if len(question.split()) < 2:
        return GATE_ASK_CLARIFICATION

    plan = list(getattr(state, "plan", []) or [])
    if not plan:
        return GATE_ASK_CLARIFICATION

    coverage = coverage_summary(state)
    if coverage["coverage_ok"]:
        return GATE_ACCEPT

    retries_used = int(getattr(state, "retries_used", 0) or 0)
    if retries_used < resolved.agent_max_retrieval_retries:
        return GATE_RETRY_RETRIEVAL
    return GATE_ABSTAIN


def reason_codes_for(state: Any) -> list[str]:
    """Return the reason codes that describe the current coverage state."""
    reason_codes = list(getattr(state, "reason_codes", []) or [])
    coverage = coverage_summary(state)
    if not coverage["coverage_ok"]:
        if not coverage["span_count"]:
            reason_codes.append(REASON_NO_EVIDENCE)
        else:
            reason_codes.append(REASON_WEAK_COVERAGE)
    return reason_codes
