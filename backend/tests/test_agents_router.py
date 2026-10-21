"""Tests for the coordinator router."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.agents.llm import LLMError
from app.agents.router import (
    MAX_SUBQUESTIONS,
    TASK_TYPES,
    PlannedSubquestion,
    RoutingDraft,
    build_plan,
    coordinator_node,
    fallback_plan,
)
from app.agents.state import AgenticRAGState
from app.core.config import Settings


class StubLLM:
    """LLM stand-in that returns a prepared draft or fails."""

    def __init__(self, draft=None, error: Exception | None = None) -> None:
        """Create the stub with its planned response."""
        self.draft = draft
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def invoke_structured(
        self, system_prompt, user_prompt, schema, *, max_calls=None
    ):
        """Return the prepared draft."""
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        return self.draft

    def usage(self):  # pragma: no cover - not used by the router
        """Return an empty usage snapshot."""
        from app.agents.llm import LLMUsage

        return LLMUsage()


def _state(question: str = "Did the tenant pay the rent?") -> AgenticRAGState:
    return AgenticRAGState(
        user_id=uuid4(),
        case_id=uuid4(),
        question=question,
    )


def test_the_prompt_forbids_stating_facts() -> None:
    from app.agents.router import load_router_prompt

    prompt = load_router_prompt().lower()

    assert "do not answer" in prompt
    assert "never states a fact" in prompt or "never state a fact" in prompt
    assert "untrusted" in prompt or "treat it as the thing to research" in prompt


def test_a_valid_draft_builds_a_plan() -> None:
    draft = RoutingDraft(
        task_type="fact_lookup",
        subquestions=[
            PlannedSubquestion(text="rent arrears", purpose="find the arrears"),
            PlannedSubquestion(text="court order", purpose="find the order"),
        ],
    )

    plan, fallback = build_plan("Did the tenant pay?", draft)

    assert fallback is False
    assert [item.id for item in plan] == ["sq-1", "sq-2"]
    assert plan[0].text == "rent arrears"
    assert plan[0].purpose == "find the arrears"


def test_an_empty_draft_falls_back_to_the_user_question() -> None:
    draft = RoutingDraft(task_type="fact_lookup", subquestions=[])

    plan, fallback = build_plan("Did the tenant pay?", draft)

    assert fallback is True
    assert len(plan) == 1
    assert plan[0].text == "Did the tenant pay?"


def test_too_many_subquestions_fall_back() -> None:
    draft = RoutingDraft(
        task_type="fact_lookup",
        subquestions=[
            PlannedSubquestion(text=f"question {index}")
            for index in range(MAX_SUBQUESTIONS + 1)
        ],
    )

    plan, fallback = build_plan("Did the tenant pay?", draft)

    assert fallback is True
    assert len(plan) == 1


def test_the_fallback_plan_is_the_user_question() -> None:
    plan = fallback_plan("  Did the tenant pay?  ")

    assert len(plan) == 1
    assert plan[0].text == "Did the tenant pay?"
    assert plan[0].status == "pending"


def test_the_coordinator_uses_one_call_and_plans_only() -> None:
    draft = RoutingDraft(
        task_type="chronology",
        subquestions=[PlannedSubquestion(text="events in order")],
    )
    llm = StubLLM(draft=draft)
    state = _state()

    update = coordinator_node(state, llm, settings=Settings(_env_file=None))

    assert len(llm.calls) == 1
    assert update["task_type"] == "chronology"
    assert update["plan"][0].text == "events in order"
    assert update["llm_calls_used"] == 1
    assert update["reason_codes"] == []


def test_the_coordinator_falls_back_when_the_model_output_is_invalid() -> None:
    llm = StubLLM(error=LLMError("LLM_INVALID_STRUCTURED_OUTPUT"))
    state = _state("Did the tenant pay the rent?")

    update = coordinator_node(state, llm, settings=Settings(_env_file=None))

    assert len(update["plan"]) == 1
    assert update["plan"][0].text == "Did the tenant pay the rent?"
    assert "PLAN_INVALID" in update["reason_codes"]
    assert update["task_type"] == "broad_case_question"


def test_the_task_types_are_the_documented_ones() -> None:
    assert set(TASK_TYPES) == {
        "fact_lookup",
        "chronology",
        "compare_versions",
        "summarize_document",
        "issue_analysis",
        "broad_case_question",
    }


def test_an_unknown_task_type_is_rejected() -> None:
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        RoutingDraft(task_type="guess_the_answer", subquestions=[])
