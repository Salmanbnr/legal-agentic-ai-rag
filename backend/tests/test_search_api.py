"""API tests for the authenticated case search endpoint."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api import search as search_api
from app.auth.dependencies import CurrentUser, get_current_user
from app.core.config import get_settings
from app.db.client import get_db_client
from app.main import create_app
from app.retrieval.dense import RetrievalError
from app.retrieval.schemas import EvidenceSpan

TIMESTAMP = "2026-01-01T00:00:00+00:00"


def _span(case_id, document_id) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=uuid4(),
        case_id=case_id,
        document_id=document_id,
        document_version_id=None,
        title="pleading.pdf",
        page_start=3,
        page_end=4,
        section_path=["Article 1"],
        quote_or_span="The tenant failed to pay the rent.",
        extraction_method="native",
        quality_status="ready",
        ocr_confidence=91.0,
        source_anchor=f"doc:{document_id}|p:3-4|c:0",
        retrieval_method="hybrid",
        retrieval_rank=1,
        quality_flags=[],
        retrieval_score=0.03,
    )


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


def test_search_returns_evidence_spans(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    captured: dict[str, Any] = {}

    def fake_hybrid_search(
        client, target_case_id, current_user, query, **kwargs
    ):
        captured["case_id"] = target_case_id
        captured["user"] = current_user
        captured["query"] = query
        captured.update(kwargs)
        return [_span(case_id, document_id)]

    monkeypatch.setattr(search_api, "require_case_access", lambda *a, **k: None)
    monkeypatch.setattr(search_api, "hybrid_search", fake_hybrid_search)

    response = client.post(
        f"/api/v1/cases/{case_id}/search",
        json={"query": "who failed to pay the rent", "top_k": 5},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["case_id"] == str(case_id)
    assert payload["query"] == "who failed to pay the rent"
    assert len(payload["spans"]) == 1
    assert payload["spans"][0]["page_start"] == 3
    assert payload["spans"][0]["page_end"] == 4
    assert payload["spans"][0]["source_anchor"] == (
        f"doc:{document_id}|p:3-4|c:0"
    )
    assert payload["coverage"]["span_count"] == 1
    assert payload["coverage"]["documents_covered"] == 1

    assert captured["case_id"] == case_id
    assert captured["user"].id == user_id
    assert captured["top_k"] == 5


def test_search_passes_the_requested_document_ids(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    captured: dict[str, Any] = {}

    def fake_hybrid_search(
        client, target_case_id, current_user, query, **kwargs
    ):
        captured.update(kwargs)
        return []

    monkeypatch.setattr(search_api, "require_case_access", lambda *a, **k: None)
    monkeypatch.setattr(search_api, "hybrid_search", fake_hybrid_search)

    response = client.post(
        f"/api/v1/cases/{case_id}/search",
        json={"query": "damages", "document_ids": [str(document_id)]},
    )

    assert response.status_code == 200
    assert captured["document_ids"] == [document_id]
    assert captured["top_k"] == 8


def test_search_requires_authentication(app, client: TestClient) -> None:
    response = client.post(
        f"/api/v1/cases/{uuid4()}/search",
        json={"query": "damages"},
    )

    assert response.status_code == 401


def test_search_refuses_a_case_the_user_does_not_own(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException, status

    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)

    def deny(*args, **kwargs):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Case not found"
        )

    monkeypatch.setattr(search_api, "require_case_access", deny)
    monkeypatch.setattr(
        search_api,
        "hybrid_search",
        lambda *args, **kwargs: pytest.fail("search must not run"),
    )

    response = client.post(
        f"/api/v1/cases/{uuid4()}/search",
        json={"query": "damages"},
    )

    assert response.status_code == 404


def test_search_rejects_an_empty_query(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    monkeypatch.setattr(search_api, "require_case_access", lambda *a, **k: None)

    response = client.post(
        f"/api/v1/cases/{uuid4()}/search",
        json={"query": ""},
    )

    assert response.status_code == 422


def test_search_rejects_an_unknown_field(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    monkeypatch.setattr(search_api, "require_case_access", lambda *a, **k: None)

    response = client.post(
        f"/api/v1/cases/{uuid4()}/search",
        json={"query": "damages", "case_id": str(uuid4())},
    )

    assert response.status_code == 422


def test_search_reports_an_unavailable_backend(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)

    def failing(*args, **kwargs):
        raise RetrievalError("DENSE_SEARCH_FAILED")

    monkeypatch.setattr(search_api, "require_case_access", lambda *a, **k: None)
    monkeypatch.setattr(search_api, "hybrid_search", failing)

    response = client.post(
        f"/api/v1/cases/{uuid4()}/search",
        json={"query": "damages"},
    )

    assert response.status_code == 503
    assert "DENSE_SEARCH_FAILED" not in response.text
