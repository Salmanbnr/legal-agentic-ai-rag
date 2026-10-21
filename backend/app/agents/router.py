"""The coordinator node: classify the question and plan the subquestions.

The coordinator makes exactly one model call and it only plans. It never
states a fact, and its output is only used to build retrieval queries. When
the model returns something unusable the router falls back to a single
subquestion that is the user's question.
"""

from __future__ import annotations

import logging
from typing import Any

from pydantic import BaseModel, Field

from app.agents.llm import LLMClient, LLMError, load_prompt
from app.agents.state import Subquestion, TaskType
from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

ROUTER_PROMPT_NAME = "route_question.txt"
MAX_SUBQUESTIONS = 4
MIN_SUBQUESTIONS = 1

TASK_TYPES: tuple[TaskType, ...] = (
    "fact_lookup",
    "chronology",
    "compare_versions",
    "summarize_document",
    "issue_analysis",
    "broad_case_question",
)


class PlannedSubquestion(BaseModel):
    """One subquestion as the planner described it."""

    text: str = Field(min_length=1)
    purpose: str = ""


class RoutingDraft(BaseModel):
    """The planner's output before the backend assigns ids and statuses."""

    task_type: TaskType = "broad_case_question"
    subquestions: list[PlannedSubquestion] = Field(default_factory=list)
    rationale: str = ""


def load_router_prompt() -> str:
    """Load the planner system prompt."""
    return load_prompt(ROUTER_PROMPT_NAME)


def fallback_plan(question: str) -> list[Subquestion]:
    """Return the safe fallback plan: one subquestion, the user's question."""
    text = (question or "").strip() or "What does the case file say?"
    return [Subquestion(id="sq-1", text=text, purpose="Answer the question.")]


def build_plan(
    question: str, draft: RoutingDraft
) -> tuple[list[Subquestion], bool]:
    """Turn a planner draft into subquestions.

    A draft is usable when it names between one and four subquestions with
    non-empty text. Anything else falls back to a single subquestion that is
    the user's question.

    Returns:
        The plan and whether the fallback was used.
    """
    planned = [
        item for item in draft.subquestions if item.text and item.text.strip()
    ]
    if not MIN_SUBQUESTIONS <= len(planned) <= MAX_SUBQUESTIONS:
        return fallback_plan(question), True
    return (
        [
            Subquestion(
                id=f"sq-{index}",
                text=item.text.strip(),
                purpose=item.purpose.strip(),
            )
            for index, item in enumerate(planned, start=1)
        ],
        False,
    )


def coordinator_node(
    state: Any,
    llm: LLMClient,
    *,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Classify the question and plan the subquestions.

    Args:
        state: The current agent state.
        llm: The LLM client used for the single planning call.
        settings: Optional validated settings.

    Returns:
        The state update with the task type and the plan.
    """
    resolved = settings or get_settings()
    question = state.question
    try:
        draft = llm.invoke_structured(
            load_router_prompt(),
            f"QUESTION: {question}",
            RoutingDraft,
            max_calls=resolved.agent_max_llm_calls,
        )
    except LLMError as exc:
        logger.info(
            "planner output was unusable, using the fallback plan",
            extra={"error_code": exc.error_code},
        )
        draft = None

    if draft is None:
        return {
            "task_type": "broad_case_question",
            "plan": fallback_plan(question),
            "reason_codes": ["PLAN_INVALID"],
            "llm_calls_used": int(state.llm_calls_used) + 1,
        }

    plan, fallback_used = build_plan(question, draft)
    return {
        "task_type": draft.task_type,
        "plan": plan,
        "reason_codes": ["PLAN_INVALID"] if fallback_used else [],
        "llm_calls_used": int(state.llm_calls_used) + 1,
        "details": {
            "plan_count": len(plan),
            "task_type": draft.task_type,
        },
    }
