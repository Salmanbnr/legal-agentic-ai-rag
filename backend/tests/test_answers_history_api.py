"""API tests for the stored answer history and the citation evidence endpoint."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.agents.state import AgentTraceSummary, TracePlanStep
from app.api import answers as answers_api
from app.auth.dependencies import CurrentUser, get_current_user
from app.core.config import get_settings
from app.db.client import get_db_client
from app.db.repos import Answer, Document, DocumentChunk, DocumentPage
from app.main import create_app
from app.retrieval.answer_schema import AnswerClaim, AnswerResponse
from app.retrieval.schemas import EvidenceSpan

TIMESTAMP = "2026-01-01T00:00:00+00:00"


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


def _span(evidence_id: UUID | None = None) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=evidence_id or uuid4(),
        case_id=uuid4(),
        document_id=uuid4(),
        document_version_id=None,
        title="Lease agreement.pdf",
        page_start=7,
        page_end=7,
        section_path=["Clause 12"],
        quote_or_span="Commencing 14 March 2024 the monthly rent is GBP 1,250.",
        extraction_method="ocr",
        quality_status="ready_with_warnings",
        ocr_confidence=58.0,
        source_anchor="doc:1|p:7|c:0",
        retrieval_method="hybrid",
        retrieval_rank=1,
        quality_flags=["low_ocr_confidence"],
        retrieval_score=0.04,
    )


def _answer(span: EvidenceSpan) -> AnswerResponse:
    return AnswerResponse(
        answer_text="The rent is GBP 1,250 per month.",
        claims=[
            AnswerClaim(
                text="The rent is GBP 1,250 per month.",
                evidence_ids=[span.evidence_id],
                type="direct",
            )
        ],
        final_status="answered",
        evidence=[span],
    )


def _trace() -> AgentTraceSummary:
    return AgentTraceSummary(
        graph_version="agentic-rag-3",
        node_path=["authorize", "route", "build_plan", "retrieve", "final_gate"],
        plan=[
            TracePlanStep(
                text="What rent does clause 12 set?",
                purpose="Find the rent clause.",
                status="retrieved",
            )
        ],
        queries=["What rent does clause 12 set?"],
        steps=5,
        llm_calls=3,
        tokens_used=900,
        retries_used=1,
        final_status="answered",
    )


def _row(
    case_id: UUID,
    *,
    graph_version: str = "agentic-rag-3",
    final_status: str = "answered",
    payload: dict[str, Any] | None = None,
) -> Answer:
    return Answer.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "user_id": str(uuid4()),
            "question": "What is the monthly rent?",
            "answer_payload": payload or {},
            "evidence_refs": [],
            "graph_version": graph_version,
            "final_status": final_status,
            "created_at": TIMESTAMP,
        }
    )


def _document(document_id: UUID, case_id: UUID) -> Document:
    return Document.model_validate(
        {
            "id": str(document_id),
            "case_id": str(case_id),
            "owner_user_id": str(uuid4()),
            "storage_path": f"{uuid4()}/{case_id}/{uuid4()}",
            "original_filename": "Lease agreement.pdf",
            "detected_mime": "application/pdf",
            "sha256": "a" * 64,
            "size_bytes": 2048,
            "page_count": 8,
            "quality_status": "ready_with_warnings",
            "quality_report": {},
            "processing_status": "ready",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    )


def _chunk(document_id: UUID, case_id: UUID) -> DocumentChunk:
    return DocumentChunk.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "document_id": str(document_id),
            "document_version_id": None,
            "chunk_index": 0,
            "page_start": 7,
            "page_end": 7,
            "section_path": ["Clause 12"],
            "text_content": "The monthly rent is GBP 1,250.",
            "content_hash": "b" * 64,
            "source_anchor": "doc:1|p:7|c:0",
            "embedding_model": None,
            "index_schema_version": 1,
            "index_status": "indexed",
            "quality_flags": [],
            "created_at": TIMESTAMP,
        }
    )


def _page(document_id: UUID, case_id: UUID) -> DocumentPage:
    return DocumentPage.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "document_id": str(document_id),
            "page_number": 7,
            "extraction_method": "ocr",
            "text_density": 0.6,
            "ocr_confidence": 58.0,
            "raw_text": None,
            "clean_text": None,
            "quality_flags": ["low_ocr_confidence"],
            "transform_log": [],
            "created_at": TIMESTAMP,
        }
    )


def test_list_answers_returns_sections_without_evidence_text(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_id = uuid4()
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    span = _span()
    payload = answers_api.stored_payload(_answer(span), _trace())
    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)
    monkeypatch.setattr(
        answers_api, "list_answers", lambda *a, **k: [_row(case_id, payload=payload)]
    )

    response = client.get(f"/api/v1/cases/{case_id}/answers")

    assert response.status_code == 200
    body = response.json()
    assert body["has_more"] is False
    assert body["limit"] == 20
    item = body["items"][0]
    assert item["question"] == "What is the monthly rent?"
    assert item["mode"] == "agentic"
    assert item["final_status"] == "answered"
    assert item["answer"]["final_status"] == "answered"
    assert item["answer"]["claims"][0]["type"] == "direct"
    citation = item["answer"]["evidence"][0]
    assert citation["title"] == "Lease agreement.pdf"
    assert citation["page_start"] == 7
    assert citation["ocr_confidence"] == 58.0
    # The listing never carries document text.
    assert citation["quote_or_span"] == ""
    assert "Commencing 14 March 2024" not in response.text


def test_list_answers_returns_the_plan_and_queries(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_id = uuid4()
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    span = _span()
    payload = answers_api.stored_payload(_answer(span), _trace())
    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)
    monkeypatch.setattr(
        answers_api, "list_answers", lambda *a, **k: [_row(case_id, payload=payload)]
    )

    response = client.get(f"/api/v1/cases/{case_id}/answers")

    trace = response.json()["items"][0]["trace"]
    assert trace["plan"][0]["text"] == "What rent does clause 12 set?"
    assert trace["plan"][0]["status"] == "retrieved"
    assert trace["queries"] == ["What rent does clause 12 set?"]
    assert trace["retries_used"] == 1
    assert trace["node_path"][0] == "authorize"


def test_list_answers_marks_a_baseline_answer(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_id = uuid4()
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    payload = _answer(_span()).model_dump(mode="json")
    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)
    monkeypatch.setattr(
        answers_api,
        "list_answers",
        lambda *a, **k: [
            _row(case_id, graph_version="baseline-rag-1", payload=payload)
        ],
    )

    response = client.get(f"/api/v1/cases/{case_id}/answers")

    item = response.json()["items"][0]
    assert item["mode"] == "baseline"
    assert item["trace"] is None


def test_list_answers_uses_one_extra_row_to_report_more(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_id = uuid4()
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    captured: dict[str, Any] = {}
    rows = [_row(case_id) for _ in range(3)]

    def fake_list(db, owner_id, target_case_id, **kwargs):
        captured.update(kwargs)
        return rows

    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)
    monkeypatch.setattr(answers_api, "list_answers", fake_list)

    response = client.get(
        f"/api/v1/cases/{case_id}/answers?limit=2&offset=4"
    )

    body = response.json()
    assert captured == {"limit": 3, "offset": 4}
    assert len(body["items"]) == 2
    assert body["has_more"] is True


def test_list_answers_rejects_a_page_size_over_the_limit(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)

    response = client.get(f"/api/v1/cases/{uuid4()}/answers?limit=500")

    assert response.status_code == 422


def test_list_answers_requires_authentication(app, client: TestClient) -> None:
    response = client.get(f"/api/v1/cases/{uuid4()}/answers")

    assert response.status_code == 401


def test_list_answers_hides_a_foreign_case(
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
        "list_answers",
        lambda *a, **k: pytest.fail("a foreign case must not be listed"),
    )

    response = client.get(f"/api/v1/cases/{uuid4()}/answers")

    assert response.status_code == 404


def test_evidence_returns_the_text_and_the_page_warning(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_id = uuid4()
    document_id = uuid4()
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    chunk = _chunk(document_id, case_id)
    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)
    monkeypatch.setattr(
        answers_api, "get_chunks_by_ids", lambda *a, **k: [chunk]
    )
    monkeypatch.setattr(
        answers_api,
        "get_document_for_user",
        lambda *a, **k: _document(document_id, case_id),
    )
    monkeypatch.setattr(
        answers_api,
        "get_pages_for_documents",
        lambda *a, **k: [_page(document_id, case_id)],
    )

    response = client.get(
        f"/api/v1/cases/{case_id}/evidence/{chunk.id}"
    )

    assert response.status_code == 200
    body = response.json()
    assert body["text"] == "The monthly rent is GBP 1,250."
    assert body["page_start"] == 7
    assert body["page_end"] == 7
    assert body["extraction_method"] == "ocr"
    assert body["ocr_confidence"] == 58.0
    assert body["document_title"] == "Lease agreement.pdf"
    assert body["document_mime"] == "application/pdf"
    assert any("Low OCR confidence" in item for item in body["quality_warnings"])
    assert body["source_anchor"] == "doc:1|p:7|c:0"


def test_evidence_reports_a_missing_citation(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    case_id = uuid4()
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)
    monkeypatch.setattr(answers_api, "get_chunks_by_ids", lambda *a, **k: [])

    response = client.get(f"/api/v1/cases/{case_id}/evidence/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Evidence not found"


def test_evidence_hides_a_document_the_user_no_longer_owns(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from app.db.repos import DocumentAccessDenied

    case_id = uuid4()
    document_id = uuid4()
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    chunk = _chunk(document_id, case_id)
    monkeypatch.setattr(answers_api, "require_case_access", lambda *a, **k: None)
    monkeypatch.setattr(
        answers_api, "get_chunks_by_ids", lambda *a, **k: [chunk]
    )

    def deny(*args, **kwargs):
        raise DocumentAccessDenied("not yours")

    monkeypatch.setattr(answers_api, "get_document_for_user", deny)

    response = client.get(f"/api/v1/cases/{case_id}/evidence/{chunk.id}")

    assert response.status_code == 404
    assert "GBP" not in response.text


def test_evidence_requires_authentication(app, client: TestClient) -> None:
    response = client.get(
        f"/api/v1/cases/{uuid4()}/evidence/{uuid4()}"
    )

    assert response.status_code == 401
