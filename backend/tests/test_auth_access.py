"""Tests for ownership-safe case and document access checks."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi import HTTPException

from app.auth import access
from app.auth.dependencies import CurrentUser
from app.db.repos import CaseAccessDenied, DocumentAccessDenied

TIMESTAMP = "2026-01-01T00:00:00+00:00"


def _user() -> CurrentUser:
    return CurrentUser(id=uuid4(), email="lawyer@example.com")


def _case_row(case_id: UUID, owner_id: UUID) -> dict[str, object]:
    return {
        "id": str(case_id),
        "owner_user_id": str(owner_id),
        "title": "Test case",
        "description": None,
        "status": "active",
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    }


def _document_row(
    document_id: UUID, case_id: UUID, owner_id: UUID
) -> dict[str, object]:
    return {
        "id": str(document_id),
        "case_id": str(case_id),
        "owner_user_id": str(owner_id),
        "storage_path": f"{owner_id}/{case_id}/{document_id}.pdf",
        "original_filename": "filing.pdf",
        "detected_mime": "application/pdf",
        "sha256": None,
        "size_bytes": 1,
        "page_count": 1,
        "document_type": None,
        "version_group_id": None,
        "duplicate_of": None,
        "quality_status": "pending",
        "quality_report": {},
        "processing_status": "uploaded",
        "parser_version": None,
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    }


def _single_row_client(
    table_name: str, rows: list[dict[str, object]]
) -> MagicMock:
    client = MagicMock()
    table = MagicMock()
    table.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = (
        rows
    )
    client.table.side_effect = lambda name: table if name == table_name else MagicMock()
    return client


def test_foreign_case_returns_same_404_as_missing_case() -> None:
    user = _user()
    case_id = uuid4()
    foreign_client = _single_row_client(
        "cases", [_case_row(case_id, uuid4())]
    )
    missing_client = _single_row_client("cases", [])

    with pytest.raises(HTTPException) as foreign:
        access.require_case_access(case_id, user, client=foreign_client)
    with pytest.raises(HTTPException) as missing:
        access.require_case_access(uuid4(), user, client=missing_client)

    assert foreign.value.status_code == 404
    assert missing.value.status_code == 404
    assert foreign.value.detail == missing.value.detail == "Case not found"


def test_foreign_document_returns_same_404_as_missing_document() -> None:
    user = _user()
    case_id = uuid4()
    document_id = uuid4()
    foreign_client = _single_row_client(
        "documents",
        [_document_row(document_id, case_id, uuid4())],
    )
    missing_client = _single_row_client("documents", [])

    with pytest.raises(HTTPException) as foreign:
        access.require_document_access(document_id, user, client=foreign_client)
    with pytest.raises(HTTPException) as missing:
        access.require_document_access(uuid4(), user, client=missing_client)

    assert foreign.value.status_code == 404
    assert missing.value.status_code == 404
    assert foreign.value.detail == missing.value.detail == "Document not found"


def test_case_access_maps_domain_denial_to_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny(*args: object, **kwargs: object) -> None:
        raise CaseAccessDenied("denied")

    monkeypatch.setattr(access, "get_case_for_user", deny)

    with pytest.raises(HTTPException) as error:
        access.require_case_access(uuid4(), _user(), client=MagicMock())

    assert error.value.status_code == 404
    assert error.value.detail == "Case not found"


def test_document_access_maps_domain_denial_to_404(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def deny(*args: object, **kwargs: object) -> None:
        raise DocumentAccessDenied("denied")

    monkeypatch.setattr(access, "get_document_for_user", deny)

    with pytest.raises(HTTPException) as error:
        access.require_document_access(uuid4(), _user(), client=MagicMock())

    assert error.value.status_code == 404
    assert error.value.detail == "Document not found"
