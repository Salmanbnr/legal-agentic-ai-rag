"""Tests for the research agent."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from app.agents import research_agent as research_module
from app.agents.research_agent import (
    QueryRewrite,
    detect_cross_reference,
    research,
)
from app.agents.state import Subquestion
from app.auth.dependencies import CurrentUser
from app.core.config import Settings
from app.retrieval.schemas import EvidenceSpan

TIMESTAMP = "2026-01-01T00:00:00+00:00"


class StubLLM:
    """LLM stand-in that returns a prepared rewrite."""

    def __init__(self, rewrite: QueryRewrite | None = None) -> None:
        """Create the stub with its rewrite."""
        self.rewrite = rewrite
        self.calls: list[tuple[str, str]] = []

    def invoke_structured(
        self, system_prompt, user_prompt, schema, *, max_calls=None
    ):
        """Return the prepared rewrite."""
        self.calls.append((system_prompt, user_prompt))
        if self.rewrite is None:
            from app.agents.llm import LLMError

            raise LLMError("LLM_INVALID_STRUCTURED_OUTPUT")
        return self.rewrite

    @property
    def usage(self):
        """Return the usage counters."""
        from app.agents.llm import LLMUsage

        return LLMUsage(calls=len(self.calls))


def _span(
    document_id,
    index: int = 0,
    text: str = "The tenant failed to pay the rent.",
    quality_flags: list[Any] | None = None,
) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=uuid4(),
        case_id=uuid4(),
        document_id=document_id,
        document_version_id=None,
        title="lease.pdf",
        page_start=index + 1,
        page_end=index + 1,
        section_path=[],
        quote_or_span=text,
        extraction_method="native",
        quality_status="ready",
        ocr_confidence=None,
        source_anchor=f"doc:1|p:{index + 1}-{index + 1}|c:{index}",
        retrieval_method="hybrid",
        retrieval_rank=1,
        quality_flags=list(quality_flags or []),
        retrieval_score=0.5,
    )


@pytest.fixture()
def scope(monkeypatch: pytest.MonkeyPatch):
    """Record every search the research agent performs."""
    case_id = uuid4()
    document_id = uuid4()
    calls: list[dict[str, Any]] = []

    def fake_search(
        client, target_case_id, user, query, *, document_ids=None, **kwargs
    ):
        calls.append(
            {"query": query, "document_ids": list(document_ids or [])}
        )
        return []

    monkeypatch.setattr(research_module, "hybrid_search", fake_search)
    return {
        "case_id": case_id,
        "document_id": document_id,
        "calls": calls,
        "user": CurrentUser(id=uuid4(), email="lawyer@example.com"),
    }


def _subquestions(*texts: str) -> list[Subquestion]:
    return [
        Subquestion(
            id=f"sq-{index}",
            text=text,
            purpose="Answer the question.",
        )
        for index, text in enumerate(texts, start=1)
    ]


def test_the_search_never_leaves_the_authorized_documents(scope) -> None:
    output = research(
        object(),
        scope["case_id"],
        scope["user"],
        "Did the tenant pay the rent?",
        _subquestions("rent arrears"),
        [scope["document_id"]],
        settings=Settings(_env_file=None),
        llm=StubLLM(),
    )

    assert scope["calls"]
    for call in scope["calls"]:
        assert call["document_ids"] == [str(scope["document_id"])]


def test_a_rewrite_cannot_widen_the_scope(scope) -> None:
    rewrite = QueryRewrite(
        rewritten_query="secret-agreement.pdf clause 12 arrears",
        strategy="section name",
        terms_added=["secret-agreement.pdf"],
    )

    research(
        object(),
        scope["case_id"],
        scope["user"],
        "Did the tenant pay the rent?",
        _subquestions("rent arrears"),
        [scope["document_id"]],
        settings=Settings(_env_file=None),
        llm=StubLLM(rewrite),
    )

    for call in scope["calls"]:
        assert call["document_ids"] == [str(scope["document_id"])]


def test_the_research_agent_stops_after_the_retry_limit(
    scope, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(_env_file=None).model_copy(
        update={"agent_max_retrieval_retries": 1}
    )
    budget = _budget(settings)

    output = research(
        object(),
        scope["case_id"],
        scope["user"],
        "Did the tenant pay the rent?",
        _subquestions("rent arrears"),
        [scope["document_id"]],
        settings=settings,
        llm=StubLLM(),
        budget=budget,
    )

    assert len(scope["calls"]) == 1
    assert output.gaps
    assert output.recommended_next_search == "rent arrears"


def test_one_rewrite_call_is_made_when_the_coverage_is_weak(
    scope, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id = scope["document_id"]
    calls = scope["calls"]

    def fake_search(
        client, target_case_id, user, query, *, document_ids=None, **kwargs
    ):
        calls.append({"query": query, "document_ids": list(document_ids or [])})
        if len(calls) == 1:
            return [_span(document_id, 0)]
        return [_span(document_id, 1)]

    monkeypatch.setattr(research_module, "hybrid_search", fake_search)
    llm = StubLLM(
        QueryRewrite(
            rewritten_query="arrears notice fourteen days",
            strategy="synonym",
            terms_added=["arrears"],
        )
    )

    output = research(
        object(),
        scope["case_id"],
        scope["user"],
        "Did the tenant pay the rent?",
        _subquestions("rent arrears"),
        [document_id],
        settings=Settings(_env_file=None),
        llm=llm,
        retries_remaining=1,
    )

    assert len(llm.calls) == 1
    strategies = [attempt.strategy for attempt in output.queries_attempted]
    assert "rewrite" in strategies
    rewrite_attempt = next(
        attempt
        for attempt in output.queries_attempted
        if attempt.strategy == "rewrite"
    )
    assert rewrite_attempt.original_question == "Did the tenant pay the rent?"
    assert rewrite_attempt.rewrite_strategy == "synonym"


def test_no_rewrite_is_made_when_no_retries_remain(
    scope, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id = scope["document_id"]
    calls = scope["calls"]

    def fake_search(
        client, target_case_id, user, query, *, document_ids=None, **kwargs
    ):
        calls.append({"query": query, "document_ids": list(document_ids or [])})
        return [_span(document_id, 0)]

    monkeypatch.setattr(research_module, "hybrid_search", fake_search)
    llm = StubLLM(
        QueryRewrite(
            rewritten_query="arrears notice fourteen days",
            strategy="synonym",
        )
    )

    output = research(
        object(),
        scope["case_id"],
        scope["user"],
        "Did the tenant pay the rent?",
        _subquestions("rent arrears"),
        [document_id],
        settings=Settings(_env_file=None),
        llm=llm,
        retries_remaining=0,
    )

    assert llm.calls == []
    assert all(
        attempt.strategy != "rewrite" for attempt in output.queries_attempted
    )


def test_the_evidence_is_merged_and_deduplicated(
    scope, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id = scope["document_id"]
    shared = _span(document_id, 0)
    calls = scope["calls"]

    def fake_search(
        client, target_case_id, user, query, *, document_ids=None, **kwargs
    ):
        calls.append({"query": query, "document_ids": list(document_ids or [])})
        return [shared, _span(document_id, 1)]

    monkeypatch.setattr(research_module, "hybrid_search", fake_search)

    output = research(
        object(),
        scope["case_id"],
        scope["user"],
        "Did the tenant pay the rent?",
        _subquestions("rent arrears", "court order"),
        [document_id],
        settings=Settings(_env_file=None),
        llm=StubLLM(),
    )

    ids = [str(span.evidence_id) for span in output.evidence]
    assert len(ids) == len(set(ids))
    assert str(shared.evidence_id) in ids


def test_a_cross_reference_triggers_one_multi_hop_search(
    scope, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id = scope["document_id"]
    calls = scope["calls"]

    def fake_search(
        client, target_case_id, user, query, *, document_ids=None, **kwargs
    ):
        calls.append({"query": query, "document_ids": list(document_ids or [])})
        if len(calls) == 1:
            return [
                _span(
                    document_id,
                    0,
                    "The claim is subject to clause 12 of the agreement.",
                )
            ]
        return [_span(document_id, 1, "Clause 12 sets out the notice period.")]

    monkeypatch.setattr(research_module, "hybrid_search", fake_search)

    output = research(
        object(),
        scope["case_id"],
        scope["user"],
        "What does clause 12 say?",
        _subquestions("clause 12"),
        [document_id],
        settings=Settings(_env_file=None),
        llm=StubLLM(),
    )

    strategies = [attempt.strategy for attempt in output.queries_attempted]
    assert "multi_hop" in strategies
    hop = next(
        attempt
        for attempt in output.queries_attempted
        if attempt.strategy == "multi_hop"
    )
    assert "clause 12" in hop.query.lower()


def test_the_multi_hop_search_stays_inside_the_scope(
    scope, monkeypatch: pytest.MonkeyPatch
) -> None:
    document_id = scope["document_id"]
    calls = scope["calls"]

    def fake_search(
        client, target_case_id, user, query, *, document_ids=None, **kwargs
    ):
        calls.append({"query": query, "document_ids": list(document_ids or [])})
        return [
            _span(
                document_id,
                0,
                "The agreement is dated 1 April 2021 and clause 9 applies.",
            )
        ]

    monkeypatch.setattr(research_module, "hybrid_search", fake_search)

    research(
        object(),
        scope["case_id"],
        scope["user"],
        "When was the agreement dated?",
        _subquestions("agreement date"),
        [document_id],
        settings=Settings(_env_file=None),
        llm=StubLLM(),
    )

    for call in calls:
        assert call["document_ids"] == [str(document_id)]


def test_a_cross_reference_is_detected_in_clauses_and_dates() -> None:
    assert (
        detect_cross_reference("This is subject to clause 12 of the lease.")
        == "subject to clause 12"
    )
    assert (
        detect_cross_reference("The deed dated 4 May 2019 is in the file.")
        == "deed dated 4 May 2019"
    )
    assert detect_cross_reference("Nothing special here.") is None


def test_the_research_agent_reports_a_gap_when_nothing_is_found(scope) -> None:
    output = research(
        object(),
        scope["case_id"],
        scope["user"],
        "What did the arbitration settle for?",
        _subquestions("arbitration settlement"),
        [scope["document_id"]],
        settings=Settings(_env_file=None),
        llm=StubLLM(),
    )

    assert output.evidence == []
    assert output.gaps
    assert "NO_EVIDENCE_RETRIEVED" in output.reason_codes
    assert output.coverage[0].coverage_ok is False


def _budget(settings: Settings):
    """Return a budget with a fixed deadline."""
    from app.agents.budget import Budget

    return Budget.from_settings(settings, deadline=1000.0, now=lambda: 1.0)
