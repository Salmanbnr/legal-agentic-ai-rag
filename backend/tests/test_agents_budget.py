"""Tests for the agent budget."""

from __future__ import annotations

import pytest

from app.agents.budget import REASON_BUDGET_EXCEEDED, Budget, BudgetExceeded
from app.core.config import Settings


def _budget(**overrides) -> Budget:
    """Return a budget with deterministic limits."""
    settings = Settings(_env_file=None).model_copy(update=overrides)
    return Budget.from_settings(settings, deadline=100.0, now=lambda: 50.0)


def test_the_budget_reads_the_configured_limits() -> None:
    budget = _budget(
        agent_max_steps=5,
        agent_max_retrieval_retries=1,
        agent_max_llm_calls=2,
        agent_max_tokens=1000,
    )

    assert budget.max_steps == 5
    assert budget.max_retrieval_retries == 1
    assert budget.max_llm_calls == 2
    assert budget.max_tokens == 1000
    assert budget.remaining_seconds() == pytest.approx(50.0)


def test_a_run_inside_the_limits_passes_the_check() -> None:
    budget = _budget(agent_max_steps=5, agent_max_llm_calls=2)

    budget.check(steps=4, llm_calls=1, tokens=10, retries=0)


def test_the_step_limit_is_enforced() -> None:
    budget = _budget(agent_max_steps=2)

    with pytest.raises(BudgetExceeded) as error:
        budget.check(steps=3)

    assert error.value.reason_code == REASON_BUDGET_EXCEEDED
    assert error.value.limit == "steps"


def test_the_retry_limit_is_enforced() -> None:
    budget = _budget(agent_max_retrieval_retries=1)

    with pytest.raises(BudgetExceeded) as error:
        budget.check(retries=2)

    assert error.value.limit == "retrieval_retries"


def test_the_llm_call_limit_is_enforced() -> None:
    budget = _budget(agent_max_llm_calls=1)

    with pytest.raises(BudgetExceeded) as error:
        budget.check(llm_calls=2)

    assert error.value.limit == "llm_calls"


def test_the_token_limit_is_enforced() -> None:
    budget = _budget(agent_max_tokens=100)

    with pytest.raises(BudgetExceeded) as error:
        budget.check(tokens=101)

    assert error.value.limit == "tokens"


def test_the_deadline_is_enforced() -> None:
    budget = Budget(
        max_steps=5,
        max_retrieval_retries=1,
        max_llm_calls=1,
        max_tokens=100,
        deadline=10.0,
        now=lambda: 11.0,
    )

    with pytest.raises(BudgetExceeded) as error:
        budget.check()

    assert error.value.limit == "deadline"


def test_the_default_deadline_comes_from_the_settings() -> None:
    settings = Settings(_env_file=None).model_copy(
        update={"agent_timeout_seconds": 30}
    )

    budget = Budget.from_settings(settings)

    assert budget.deadline is not None
    assert budget.remaining_seconds() <= 30.0
    assert budget.remaining_seconds() > 0


def test_the_snapshot_only_exposes_limits_and_time() -> None:
    budget = _budget()

    snapshot = budget.snapshot()

    assert set(snapshot) == {
        "max_steps",
        "max_retrieval_retries",
        "max_llm_calls",
        "max_tokens",
        "remaining_seconds",
    }
