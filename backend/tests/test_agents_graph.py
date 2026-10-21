"""Tests for the agentic graph orchestration."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from app.agents import graph as graph_module
from app.agents import research_agent as research_module
from app.agents.graph import GRAPH_VERSION, broaden_query, run_agentic_rag
from app.agents.llm import LLMUsage
from app.agents.router import PlannedSubquestion, RoutingDraft
from app.agents.state import REASON_AUTHORIZATION_FAILED
from app.auth.dependencies import CurrentUser
from app.core.config import Settings
from app.db.repos import Case, CaseAccessDenied, Document
from app.retrieval.answer_schema import AnswerResponse
from app.retrieval.schemas import EvidenceSpan

TIMESTAMP = "2026-01-01T00:00:00+00:00"


class StubLLM:
    """LLM stand-in that returns a prepared draft."""

    def __init__(self, draft: Any = None, error: Exception | None = None) -> None:
        """Create the stub with its planned response."""
        self.draft = draft
        self.error = error
        self.calls: list[tuple[str, str]] = []

    def invoke_structured(
        self, system_prompt, user_prompt, schema, *, max_calls=None
    ):
        """Return the prepared draft for the requested schema."""
        self.calls.append((system_prompt, user_prompt))
        if self.error is not None:
            raise self.error
        if "rewritten_query" in getattr(schema, "model_fields", {}):
            from app.agents.research_agent import QueryRewrite

            return QueryRewrite(
                rewritten_query="", strategy="", terms_added=[]
            )
        return self.draft

    @property
    def usage(self) -> LLMUsage:
        """Return the usage counters."""
        return LLMUsage(calls=len(self.calls))


def _case(case_id, owner_id) -> Case:
    return Case.model_validate(
        {
            "id": str(case_id),
            "owner_user_id": str(owner_id),
            "title": "Test case",
            "description": None,
            "status": "active",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    )


def _document(case_id, owner_id) -> Document:
    return Document.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "owner_user_id": str(owner_id),
            "storage_path": f"{owner_id}/{case_id}/{uuid4()}",
            "original_filename": "lease.pdf",
            "detected_mime": "application/pdf",
            "sha256": "a" * 64,
            "size_bytes": 1024,
            "page_count": 1,
            "document_type": None,
            "version_group_id": None,
            "duplicate_of": None,
            "quality_status": "ready",
            "quality_report": {},
            "processing_status": "ready",
            "parser_version": "1.0.0",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    )


def _span(document_id, index: int = 0) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=uuid4(),
        case_id=uuid4(),
        document_id=document_id,
        document_version_id=None,
        title="lease.pdf",
        page_start=index + 1,
        page_end=index + 1,
        section_path=[],
        quote_or_span="The tenant failed to pay the rent.",
        extraction_method="native",
        quality_status="ready",
        ocr_confidence=None,
        source_anchor="doc:1|p:1-1|c:0",
        retrieval_method="hybrid",
        retrieval_rank=1,
        quality_flags=[],
        retrieval_score=0.5,
    )


@pytest.fixture()
def agent_environment(monkeypatch: pytest.MonkeyPatch):
    """Wire the graph to in-memory data and record the calls."""
    owner_id = uuid4()
    case_id = uuid4()
    case = _case(case_id, owner_id)
    document = _document(case_id, owner_id)
    calls: dict[str, list] = {
        "retrieve": [],
        "documents": [],
        "answer": [],
    }

    def fake_get_case(client, user_id, target_case_id):
        if str(target_case_id) != str(case_id):
            raise CaseAccessDenied("Case not found")
        return case

    def fake_list_documents(client, user_id, target_case_id, **kwargs):
        calls["documents"].append(target_case_id)
        return [document]

    def fake_hybrid_search(
        client, target_case_id, user, query, *, document_ids=None, **kwargs
    ):
        calls["retrieve"].append(
            {"query": query, "document_ids": list(document_ids or [])}
        )
        return []

    def fake_answer_question(
        client, target_case_id, user, question, **kwargs
    ):
        calls["answer"].append(question)
        from app.retrieval.baseline_rag import AnswerResult

        return AnswerResult(
            answer=AnswerResponse(
                answer_text="The retrieved evidence does not answer this question.",
                claims=[],
                final_status="insufficient_evidence",
                reason_codes=["NO_SUPPORTED_CLAIMS"],
            ),
            spans=[],
            usage=LLMUsage(calls=1),
            llm_called=True,
        )

    monkeypatch.setattr(graph_module, "get_case_for_user", fake_get_case)
    monkeypatch.setattr(graph_module, "list_documents", fake_list_documents)
    monkeypatch.setattr(
        research_module, "hybrid_search", fake_hybrid_search
    )
    monkeypatch.setattr(graph_module, "answer_question", fake_answer_question)

    return {
        "owner_id": owner_id,
        "case_id": case_id,
        "document": document,
        "calls": calls,
        "user": CurrentUser(id=owner_id, email="lawyer@example.com"),
    }


def _run(environment, *, settings=None, llm=None, monkeypatch=None):
    """Run the agentic workflow against the in-memory environment."""
    return run_agentic_rag(
        object(),
        environment["case_id"],
        environment["user"],
        "Did the tenant pay the rent?",
        settings=settings or Settings(_env_file=None),
        llm=llm
        or StubLLM(
            draft=RoutingDraft(
                task_type="fact_lookup",
                subquestions=[
                    PlannedSubquestion(text="rent arrears", purpose="find it")
                ],
            )
        ),
    )


def test_the_graph_ends_within_budget_when_retrieval_returns_nothing(
    agent_environment,
) -> None:
    result = _run(agent_environment)

    assert result.state.final_status == "insufficient_evidence"
    assert result.answer.final_status == "insufficient_evidence"
    assert result.answer.claims == []
    assert result.graph_version == GRAPH_VERSION


def test_the_retry_loop_never_exceeds_the_limit(agent_environment) -> None:
    settings = Settings(_env_file=None).model_copy(
        update={"agent_max_retrieval_retries": 2}
    )

    result = _run(agent_environment, settings=settings)

    retrieve_calls = len(agent_environment["calls"]["retrieve"])
    assert retrieve_calls == settings.agent_max_retrieval_retries + 1
    assert result.state.retries_used == settings.agent_max_retrieval_retries


def test_the_node_path_shows_the_workflow(agent_environment) -> None:
    result = _run(agent_environment)

    path = [event.node for event in result.state.trace]
    assert path[0] == "authorize"
    assert "route" in path
    assert "build_plan" in path
    assert "retrieve" in path
    assert "assess_coverage" in path
    assert path[-1] == "final_gate"
    assert "synthesize" in path


def test_authorization_failure_stops_everything(
    agent_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deny(client, user_id, target_case_id):
        raise CaseAccessDenied("Case not found")

    monkeypatch.setattr(graph_module, "get_case_for_user", deny)

    result = _run(agent_environment)

    path = [event.node for event in result.state.trace]
    assert path == ["authorize"]
    assert REASON_AUTHORIZATION_FAILED in result.state.reason_codes
    assert result.state.final_status == "insufficient_evidence"
    assert agent_environment["calls"]["retrieve"] == []
    assert agent_environment["calls"]["answer"] == []


def test_a_failed_authorization_still_ends_with_a_safe_answer(
    agent_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    def deny(client, user_id, target_case_id):
        raise CaseAccessDenied("Case not found")

    monkeypatch.setattr(graph_module, "get_case_for_user", deny)

    result = _run(agent_environment)

    assert result.answer.final_status == "insufficient_evidence"
    assert result.answer.claims == []


def test_the_router_falls_back_safely_on_bad_llm_output(
    agent_environment,
) -> None:
    from app.agents.llm import LLMError

    llm = StubLLM(error=LLMError("LLM_INVALID_STRUCTURED_OUTPUT"))

    result = _run(agent_environment, llm=llm)

    assert result.state.plan[0].text == "Did the tenant pay the rent?"
    assert "PLAN_INVALID" in result.state.reason_codes
    assert result.state.final_status == "insufficient_evidence"


def test_the_budget_stops_a_run_that_exceeds_its_steps(
    agent_environment,
) -> None:
    settings = Settings(_env_file=None).model_copy(
        update={"agent_max_steps": 1, "agent_max_retrieval_retries": 5}
    )

    result = _run(agent_environment, settings=settings)

    assert "BUDGET_EXCEEDED" in result.state.reason_codes
    assert result.state.final_status == "insufficient_evidence"
    assert result.answer.claims == []


def test_every_retrieval_is_filtered_to_the_authorized_documents(
    agent_environment, monkeypatch: pytest.MonkeyPatch
) -> None:
    spans = [_span(agent_environment["document"].id)]
    monkeypatch.setattr(
        research_module,
        "hybrid_search",
        lambda client, case_id, user, query, *, document_ids=None, **kwargs: spans,
    )

    result = _run(agent_environment)

    assert result.state.authorized_document_ids == [
        agent_environment["document"].id
    ]
    assert result.state.retrieval_results
    assert result.state.retrieval_results[0].document_id == (
        agent_environment["document"].id
    )
    # The state keeps references, not the document text.
    assert not hasattr(result.state.retrieval_results[0], "quote_or_span")


def test_the_trace_summary_is_safe_and_short(agent_environment) -> None:
    result = _run(agent_environment)

    summary = result.trace_summary()

    assert summary.graph_version == GRAPH_VERSION
    assert summary.node_path[0] == "authorize"
    assert summary.steps == len(result.state.trace)
    assert summary.final_status == "insufficient_evidence"
    assert "lease.pdf" not in str(summary.model_dump())


def test_broadening_a_query_drops_the_question_words() -> None:
    broadened = broaden_query("What did the court say about the rent?")

    assert "what" not in broadened.lower()
    assert "court" in broadened.lower()
    assert "rent" in broadened.lower()


def test_broadening_keeps_a_query_that_is_only_question_words() -> None:
    assert broaden_query("What is it?") == "What is it?"
