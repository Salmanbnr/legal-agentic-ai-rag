"""The budget that bounds one agentic run.

Every node checks the budget before it does any work. When a limit is
exceeded the run stops and finishes with the reason code
``BUDGET_EXCEEDED`` instead of continuing to spend.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Callable

from app.core.config import Settings, get_settings

REASON_BUDGET_EXCEEDED = "BUDGET_EXCEEDED"

_LIMIT_STEPS = "steps"
_LIMIT_RETRIES = "retrieval_retries"
_LIMIT_LLM_CALLS = "llm_calls"
_LIMIT_TOKENS = "tokens"
_LIMIT_DEADLINE = "deadline"


class BudgetExceeded(RuntimeError):
    """Raised when an agentic run exceeds one of its limits."""

    def __init__(self, limit: str) -> None:
        """Record which limit was reached, without any document content."""
        super().__init__(REASON_BUDGET_EXCEEDED)
        self.reason_code = REASON_BUDGET_EXCEEDED
        self.limit = limit


@dataclass(frozen=True, slots=True)
class Budget:
    """Limits for one agentic run, including a wall clock deadline."""

    max_steps: int
    max_retrieval_retries: int
    max_llm_calls: int
    max_tokens: int
    deadline: float | None = None
    now: Callable[[], float] = time.monotonic

    @classmethod
    def from_settings(
        cls,
        settings: Settings | None = None,
        *,
        deadline: float | None = None,
        now: Callable[[], float] = time.monotonic,
    ) -> "Budget":
        """Build a budget from the configured agent limits."""
        resolved = settings or get_settings()
        if deadline is None:
            deadline = now() + resolved.agent_timeout_seconds
        return cls(
            max_steps=resolved.agent_max_steps,
            max_retrieval_retries=resolved.agent_max_retrieval_retries,
            max_llm_calls=resolved.agent_max_llm_calls,
            max_tokens=resolved.agent_max_tokens,
            deadline=deadline,
            now=now,
        )

    def remaining_seconds(self) -> float:
        """Return the seconds left before the deadline."""
        if self.deadline is None:
            return float("inf")
        return max(0.0, self.deadline - self.now())

    def check(
        self,
        *,
        steps: int = 0,
        llm_calls: int = 0,
        tokens: int = 0,
        retries: int = 0,
    ) -> None:
        """Raise when the run has used more than one of its limits.

        Args:
            steps: Nodes executed so far.
            llm_calls: Model calls used so far.
            tokens: Tokens used so far.
            retries: Retrieval retries used so far.

        Raises:
            BudgetExceeded: When a limit has been reached.
        """
        if steps > self.max_steps:
            raise BudgetExceeded(_LIMIT_STEPS)
        if retries > self.max_retrieval_retries:
            raise BudgetExceeded(_LIMIT_RETRIES)
        if llm_calls > self.max_llm_calls:
            raise BudgetExceeded(_LIMIT_LLM_CALLS)
        if tokens > self.max_tokens:
            raise BudgetExceeded(_LIMIT_TOKENS)
        if self.deadline is not None and self.now() > self.deadline:
            raise BudgetExceeded(_LIMIT_DEADLINE)

    def snapshot(self) -> dict[str, int | float | None]:
        """Return the limits and the remaining time for tracing."""
        return {
            "max_steps": self.max_steps,
            "max_retrieval_retries": self.max_retrieval_retries,
            "max_llm_calls": self.max_llm_calls,
            "max_tokens": self.max_tokens,
            "remaining_seconds": round(self.remaining_seconds(), 3),
        }
