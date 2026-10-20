"""Ownership-focused tests for the typed repository layer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from app.db.repos import (
    CaseAccessDenied,
    ChunkCreate,
    DocumentCreate,
    DocumentUpdate,
    JobUpdate,
    PageCreate,
    get_case_for_user,
    create_case,
    create_document,
    insert_chunks,
    insert_pages,
    list_cases,
    update_document,
    update_job,
    write_audit_event,
)

TIMESTAMP = "2026-01-01T00:00:00+00:00"


def _case_row(case_id: UUID, owner_id: UUID, **overrides: Any) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": str(case_id),
        "owner_user_id": str(owner_id),
        "title": "Test case",
        "description": None,
        "status": "active",
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    }
    row.update(overrides)
    return row


def _document_row(
    document_id: UUID, case_id: UUID, owner_id: UUID, **overrides: Any
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "id": str(document_id),
        "case_id": str(case_id),
        "owner_user_id": str(owner_id),
        "storage_path": f"{owner_id}/{case_id}/{document_id}.pdf",
        "original_filename": "filing.pdf",
        "detected_mime": "application/pdf",
        "sha256": "a" * 64,
        "size_bytes": 1024,
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
    row.update(overrides)
    return row


def _page_row(document_id: UUID, case_id: UUID) -> dict[str, Any]:
    return {
        "id": str(uuid4()),
        "case_id": str(case_id),
        "document_id": str(document_id),
        "page_number": 1,
        "extraction_method": "native",
        "text_density": 0.9,
        "ocr_confidence": None,
        "raw_text": None,
        "clean_text": "Extracted page text.",
        "quality_flags": [],
        "transform_log": [],
        "created_at": TIMESTAMP,
    }


def _chunk_row(document_id: UUID, case_id: UUID) -> dict[str, Any]:
    return {
        "id": str(uuid4()),
        "case_id": str(case_id),
        "document_id": str(document_id),
        "document_version_id": None,
        "chunk_index": 0,
        "page_start": 1,
        "page_end": 1,
        "section_path": [],
        "text_content": "Evidence excerpt.",
        "content_hash": "b" * 64,
        "source_anchor": None,
        "embedding_model": None,
        "index_schema_version": 1,
        "index_status": "pending",
        "quality_flags": [],
        "created_at": TIMESTAMP,
    }


def _job_row(job_id: UUID, case_id: UUID, document_id: UUID) -> dict[str, Any]:
    return {
        "id": str(job_id),
        "case_id": str(case_id),
        "document_id": str(document_id),
        "job_type": "ingest",
        "idempotency_key": "ingest-1",
        "status": "queued",
        "attempts": 0,
        "error_code": None,
        "error_summary": None,
        "pipeline_version": "v1",
        "started_at": None,
        "finished_at": None,
        "created_at": TIMESTAMP,
    }


def _table_mock(
    *,
    select: list[dict[str, Any]] | None = None,
    insert: list[dict[str, Any]] | None = None,
    update: list[dict[str, Any]] | None = None,
) -> MagicMock:
    table = MagicMock()
    first = table.select.return_value.eq.return_value
    first.limit.return_value.execute.return_value.data = select
    first.order.return_value.range.return_value.execute.return_value.data = select
    first.eq.return_value.order.return_value.range.return_value.execute.return_value.data = (
        select
    )
    table.insert.return_value.execute.return_value.data = insert
    table.update.return_value.eq.return_value.execute.return_value.data = update
    table.update.return_value.eq.return_value.eq.return_value.execute.return_value.data = (
        update
    )
    return table


def _client(tables: dict[str, MagicMock]) -> MagicMock:
    client = MagicMock()
    client.table.side_effect = lambda name: tables[name]
    return client


def test_get_case_for_user_rejects_another_users_case() -> None:
    user_id = uuid4()
    other_user_id = uuid4()
    case_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, other_user_id)])
    client = _client({"cases": cases})

    with pytest.raises(CaseAccessDenied):
        get_case_for_user(client, user_id, case_id)


def test_get_case_for_user_returns_owned_case() -> None:
    user_id = uuid4()
    case_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    client = _client({"cases": cases})

    result = get_case_for_user(client, user_id, case_id)

    assert result.id == case_id
    assert result.owner_user_id == user_id


def test_list_cases_filters_by_authenticated_owner() -> None:
    user_id = uuid4()
    case_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    client = _client({"cases": cases})

    result = list_cases(client, user_id)

    assert [case.id for case in result] == [case_id]
    cases.select.return_value.eq.assert_called_once_with(
        "owner_user_id", str(user_id)
    )


def test_create_case_sets_owner_from_authenticated_user() -> None:
    user_id = uuid4()
    case_id = uuid4()
    cases = _table_mock(insert=[_case_row(case_id, user_id)])
    client = _client({"cases": cases})

    result = create_case(client, user_id, title="  Test case  ")

    assert result.owner_user_id == user_id
    payload = cases.insert.call_args.args[0]
    assert payload["owner_user_id"] == str(user_id)
    assert payload["title"] == "Test case"


def test_create_document_rejects_unowned_case() -> None:
    user_id = uuid4()
    other_user_id = uuid4()
    case_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, other_user_id)])
    documents = _table_mock()
    client = _client({"cases": cases, "documents": documents})

    with pytest.raises(CaseAccessDenied):
        create_document(
            client,
            user_id,
            case_id,
            DocumentCreate(
                storage_path=f"{user_id}/{case_id}/file.pdf",
                original_filename="file.pdf",
            ),
        )

    documents.insert.assert_not_called()


def test_update_document_rejects_case_owned_by_another_user() -> None:
    user_id = uuid4()
    other_user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, other_user_id)])
    documents = _table_mock(
        select=[_document_row(document_id, case_id, user_id)]
    )
    client = _client({"cases": cases, "documents": documents})

    with pytest.raises(CaseAccessDenied):
        update_document(
            client,
            user_id,
            document_id,
            DocumentUpdate(processing_status="ready"),
        )

    documents.update.assert_not_called()


def test_insert_pages_rejects_unowned_case() -> None:
    user_id = uuid4()
    other_user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, other_user_id)])
    documents = _table_mock(select=[_document_row(document_id, case_id, user_id)])
    pages = _table_mock()
    client = _client(
        {"cases": cases, "documents": documents, "document_pages": pages}
    )

    with pytest.raises(CaseAccessDenied):
        insert_pages(
            client,
            user_id,
            case_id,
            document_id,
            [PageCreate(page_number=1, extraction_method="native")],
        )

    pages.insert.assert_not_called()


def test_insert_chunks_rejects_unowned_case() -> None:
    user_id = uuid4()
    other_user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, other_user_id)])
    documents = _table_mock(select=[_document_row(document_id, case_id, user_id)])
    chunks = _table_mock()
    client = _client(
        {"cases": cases, "documents": documents, "document_chunks": chunks}
    )

    with pytest.raises(CaseAccessDenied):
        insert_chunks(
            client,
            user_id,
            case_id,
            document_id,
            [
                ChunkCreate(
                    chunk_index=0,
                    text_content="Evidence excerpt.",
                    content_hash="b" * 64,
                )
            ],
        )

    chunks.insert.assert_not_called()


def test_update_job_rejects_unowned_case() -> None:
    user_id = uuid4()
    other_user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    jobs = _table_mock(select=[_job_row(job_id, case_id, document_id)])
    cases = _table_mock(select=[_case_row(case_id, other_user_id)])
    client = _client({"cases": cases, "processing_jobs": jobs})

    with pytest.raises(CaseAccessDenied):
        update_job(client, user_id, job_id, JobUpdate(status="running"))

    jobs.update.assert_not_called()


def test_write_audit_event_rejects_unowned_case() -> None:
    user_id = uuid4()
    other_user_id = uuid4()
    case_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, other_user_id)])
    audit = _table_mock()
    client = _client({"cases": cases, "audit_events": audit})

    with pytest.raises(CaseAccessDenied):
        write_audit_event(
            client, user_id, event_type="case.opened", case_id=case_id
        )

    audit.insert.assert_not_called()


def test_insert_pages_stores_case_and_document_ids() -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    documents = _table_mock(select=[_document_row(document_id, case_id, user_id)])
    pages = _table_mock(insert=[_page_row(document_id, case_id)])
    client = _client(
        {"cases": cases, "documents": documents, "document_pages": pages}
    )

    result = insert_pages(
        client,
        user_id,
        case_id,
        document_id,
        [PageCreate(page_number=1, extraction_method="native")],
    )

    assert len(result) == 1
    payload = pages.insert.call_args.args[0][0]
    assert payload["case_id"] == str(case_id)
    assert payload["document_id"] == str(document_id)
