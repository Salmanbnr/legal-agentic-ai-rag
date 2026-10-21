"""API tests for the case question answering endpoint."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.api import answers as answers_api
from app.auth.dependencies import CurrentUser, get_current_user
from app.core.config import get_settings
from app.db.client import get_db_client
from app.db.repos import Answer
from app.main import create_app
from app.retrieval.answer_schema import AnswerClaim, AnswerResponse
from app.retrieval.baseline_rag import AnswerResult
from app.retrieval.schemas import EvidenceSpan

TIMESTAMP = "2026-01-01T00:00:00+00:00"
REQUEST_ID = "req-answer-test"


@pytest.fixture()
def app():
    get_settings.cache_clear()
    return create_app()


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def _wire(app, user: CurrentUser) -> None:
    app.dependency_overrides[get_current_user] = lambda: user
    app.dependency_overrides[get_db_client] = lambda: MagicMock()


def _span() -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=uuid4(),
        case_id=uuid4(),
        document_id=uuid4(),
        document_version_id=None,
        title="lease.pdf",
        page_start=3,
        page_end=4,
        section_path=[],
        quote_or_span="The tenant failed to pay the rent.",
        extraction_method="native",
        quality_status="ready",
        ocr_confidence=91.0,
        source_anchor="doc:1|p:3-4|c:0",
        retrieval_method="hybrid",
        retrieval_rank=1,
        quality_flags=[],
        retrieval_score=0.03,
    )


def _answer(span: EvidenceSpan) -> AnswerResponse:
    return AnswerResponse(
        answer_text="The tenant failed to pay the rent.",
        claims=[
            AnswerClaim(
                text="The tenant failed to pay the rent.",
                evidence_ids=[span.evidence_id],
                type="direct",
            )
        ],
        final_status="answered",
        reason_codes=[],
        evidence=[span],
    )


def _answer_row(case_id: UUID) -> Answer:
    return Answer.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "user_id": str(uuid4()),
            "question": "Did the tenant pay the rent?",
            "answer_payload": {},
            "evidence_refs": [],
            "graph_version": "baseline-rag-1",
            "final_status": "answered",
            "created_at": TIMESTAMP,
        }
    )


def test_ask_returns_a_cited_answer(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    span = _span()
    captured: dict[str, Any] = {}

    def fake_answer_question(
        db, target_case_id, current_user, question, **kwargs
    ) -> AnswerResult:
        captured["case_id"] = target_case_id
        captured["user"] = current_user
        captured["question"] = question
        return AnswerResult(answer=_answer(span), spans=[span])

    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)
    monkeypatch.setattr(answers_api, "answer_question", fake_answer_question)
    monkeypatch.setattr(
        answers_api,
        "insert_answer",
        lambda db, owner_id, target_case_id, data: _answer_row(target_case_id),
    )
    monkeypatch.setattr(
        answers_api, "write_audit_event", lambda *args, **kwargs: None
    )

    response = client.post(
        f"/api/v1/cases/{case_id}/ask",
        headers={"X-Request-ID": REQUEST_ID},
        json={"question": "Did the tenant pay the rent?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["case_id"] == str(case_id)
    assert payload["question"] == "Did the tenant pay the rent?"
    assert payload["request_id"] == REQUEST_ID
    assert payload["answer"]["final_status"] == "answered"
    assert payload["answer"]["claims"][0]["evidence_ids"] == [
        str(span.evidence_id)
    ]
    assert payload["answer"]["evidence"][0]["page_start"] == 3
    assert captured["case_id"] == case_id
    assert captured["user"].id == user_id


def test_ask_stores_the_answer_with_an_evidence_manifest(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    span = _span()
    stored: dict[str, Any] = {}

    def fake_insert_answer(db, owner_id, target_case_id, data):
        stored["owner_id"] = owner_id
        stored["case_id"] = target_case_id
        stored["data"] = data
        return _answer_row(target_case_id)

    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)
    monkeypatch.setattr(
        answers_api,
        "answer_question",
        lambda *a, **k: AnswerResult(answer=_answer(span), spans=[span]),
    )
    monkeypatch.setattr(answers_api, "insert_answer", fake_insert_answer)
    monkeypatch.setattr(
        answers_api, "write_audit_event", lambda *args, **kwargs: None
    )

    response = client.post(
        f"/api/v1/cases/{case_id}/ask",
        json={"question": "Did the tenant pay the rent?"},
    )

    assert response.status_code == 200
    assert stored["owner_id"] == user_id
    assert stored["case_id"] == case_id
    data = stored["data"]
    assert data.final_status == "answered"
    assert data.graph_version == "baseline-rag-1"
    manifest = data.evidence_refs
    assert manifest[0]["evidence_id"] == str(span.evidence_id)
    assert manifest[0]["page_start"] == 3
    assert manifest[0]["page_end"] == 4
    # The manifest never carries document text.
    assert "The tenant failed to pay the rent." not in str(manifest)


def test_agentic_mode_runs_the_graph_and_returns_a_trace(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    span = _span()
    captured: dict[str, Any] = {}

    class FakeState:
        llm_calls_used = 2
        final_status = "answered"

    class FakeResult:
        def __init__(self) -> None:
            self.answer = _answer(span)
            self.state = FakeState()

        @staticmethod
        def trace_summary():
            from app.agents.state import AgentTraceSummary

            return AgentTraceSummary(
                graph_version="agentic-rag-1",
                node_path=["authorize", "route", "final_gate"],
                steps=3,
                llm_calls=2,
                tokens_used=120,
                retries_used=0,
                final_status="answered",
                reason_codes=[],
            )

    def fake_run_agentic_rag(
        db, target_case_id, current_user, question, **kwargs
    ):
        captured["case_id"] = target_case_id
        captured["question"] = question
        captured["request_id"] = kwargs.get("request_id")
        return FakeResult()

    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)
    monkeypatch.setattr(answers_api, "run_agentic_rag", fake_run_agentic_rag)
    monkeypatch.setattr(
        answers_api,
        "insert_answer",
        lambda db, owner_id, target_case_id, data: _answer_row(target_case_id),
    )
    monkeypatch.setattr(
        answers_api, "write_audit_event", lambda *args, **kwargs: None
    )

    response = client.post(
        f"/api/v1/cases/{case_id}/ask",
        headers={"X-Request-ID": REQUEST_ID},
        json={"question": "Did the tenant pay the rent?", "mode": "agentic"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "agentic"
    assert payload["answer"]["final_status"] == "answered"
    assert payload["trace"]["graph_version"] == "agentic-rag-1"
    assert payload["trace"]["node_path"][0] == "authorize"
    assert payload["trace"]["steps"] == 3
    assert captured["case_id"] == case_id
    assert captured["request_id"] == REQUEST_ID


def test_ask_requires_authentication(app, client: TestClient) -> None:
    response = client.post(
        f"/api/v1/cases/{uuid4()}/ask",
        json={"question": "Did the tenant pay?"},
    )

    assert response.status_code == 401


def test_ask_refuses_a_case_the_user_does_not_own(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException, status

    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)

    def deny(*args, **kwargs):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Case not found"
        )

    monkeypatch.setattr(answers_api, "require_case_access", deny)
    monkeypatch.setattr(
        answers_api,
        "answer_question",
        lambda *a, **k: pytest.fail("the baseline must not run"),
    )

    response = client.post(
        f"/api/v1/cases/{uuid4()}/ask",
        json={"question": "Did the tenant pay?"},
    )

    assert response.status_code == 404


def test_ask_rejects_an_empty_question(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)

    response = client.post(
        f"/api/v1/cases/{uuid4()}/ask",
        json={"question": "   "},
    )

    assert response.status_code == 422


def test_ask_rejects_a_question_over_the_budget(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)

    response = client.post(
        f"/api/v1/cases/{uuid4()}/ask",
        json={"question": "x" * 5000},
    )

    assert response.status_code == 422


def test_ask_rejects_an_unknown_mode(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)

    response = client.post(
        f"/api/v1/cases/{uuid4()}/ask",
        json={"question": "Did the tenant pay?", "mode": "deep"},
    )

    assert response.status_code == 422
