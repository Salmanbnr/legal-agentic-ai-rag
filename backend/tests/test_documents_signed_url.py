"""Tests for reading a short-lived URL to one original document."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

from app.api import documents as documents_api
from app.auth.dependencies import CurrentUser, get_current_user
from app.core.config import get_settings
from app.db.client import get_db_client
from app.db.repos import Document
from app.main import create_app
from app.storage.supabase_storage import StorageError

TIMESTAMP = "2026-01-01T00:00:00+00:00"
SIGNED_URL = "https://example.supabase.co/storage/v1/object/sign/case-documents/x?token=abc"


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


def _document(deletion_requested: bool = False) -> Document:
    return Document.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(uuid4()),
            "owner_user_id": str(uuid4()),
            "storage_path": f"{uuid4()}/{uuid4()}/{uuid4()}",
            "original_filename": "Lease agreement.pdf",
            "detected_mime": "application/pdf",
            "sha256": "a" * 64,
            "size_bytes": 2048,
            "page_count": 8,
            "quality_status": "ready",
            "quality_report": (
                {"deletion_requested": True} if deletion_requested else {}
            ),
            "processing_status": "ready",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    )


def test_signed_url_is_short_lived_and_uses_the_stored_path(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    document = _document()
    captured: dict[str, object] = {}

    def fake_create_signed_url(storage_path, seconds, *, client=None):
        captured["storage_path"] = storage_path
        captured["seconds"] = seconds
        return SIGNED_URL

    monkeypatch.setattr(
        documents_api,
        "require_document_access",
        lambda *a, **k: document,
    )
    monkeypatch.setattr(
        documents_api, "create_signed_url", fake_create_signed_url
    )

    response = client.get(f"/api/v1/documents/{document.id}/signed-url")

    assert response.status_code == 200
    body = response.json()
    assert body["url"] == SIGNED_URL
    assert body["expires_in_seconds"] == 60
    assert body["filename"] == "Lease agreement.pdf"
    assert body["mime"] == "application/pdf"
    assert captured["storage_path"] == document.storage_path
    assert captured["seconds"] == 60


def test_signed_url_refuses_a_document_being_deleted(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    document = _document(deletion_requested=True)
    monkeypatch.setattr(
        documents_api,
        "require_document_access",
        lambda *a, **k: document,
    )
    monkeypatch.setattr(
        documents_api,
        "create_signed_url",
        lambda *a, **k: pytest.fail("a deleted document must not be signed"),
    )

    response = client.get(f"/api/v1/documents/{document.id}/signed-url")

    assert response.status_code == 409


def test_signed_url_reports_storage_failure_without_details(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)
    document = _document()
    monkeypatch.setattr(
        documents_api,
        "require_document_access",
        lambda *a, **k: document,
    )

    def fail(*args, **kwargs):
        raise StorageError("bucket exploded: secret-token")

    monkeypatch.setattr(documents_api, "create_signed_url", fail)

    response = client.get(f"/api/v1/documents/{document.id}/signed-url")

    assert response.status_code == 503
    assert response.json()["detail"] == "The original file is temporarily unavailable"
    assert "secret-token" not in response.text


def test_signed_url_hides_a_foreign_document(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fastapi import HTTPException, status

    user = CurrentUser(id=uuid4(), email="lawyer@example.com")
    _wire(app, user)

    def deny(*args, **kwargs):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND, detail="Document not found"
        )

    monkeypatch.setattr(documents_api, "require_document_access", deny)
    monkeypatch.setattr(
        documents_api,
        "create_signed_url",
        lambda *a, **k: pytest.fail("a foreign document must not be signed"),
    )

    response = client.get(f"/api/v1/documents/{uuid4()}/signed-url")

    assert response.status_code == 404


def test_signed_url_requires_authentication(app, client: TestClient) -> None:
    response = client.get(f"/api/v1/documents/{uuid4()}/signed-url")

    assert response.status_code == 401
