"""Ownership-focused tests for the typed repository layer."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from app.db.repos import (
    AnswerCreate,
    CaseAccessDenied,
    CaseUpdate,
    ChunkCreate,
    DocumentAccessDenied,
    DocumentCreate,
    DocumentUpdate,
    JobCreate,
    JobUpdate,
    PageCreate,
    RecordNotFound,
    create_case,
    create_document,
    create_job,
    find_document_by_sha256,
    find_job_by_idempotency_key,
    get_case_for_user,
    get_chunks_by_ids,
    get_document_for_user,
    get_job_for_user,
    get_neighbor_chunks,
    get_pages_for_documents,
    insert_chunks,
    insert_answer,
    insert_pages,
    list_answers,
    list_cases,
    update_case,
    update_document,
    update_job,
    update_page,
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


def _job_row(
    job_id: UUID, case_id: UUID, document_id: UUID, **overrides: Any
) -> dict[str, Any]:
    row: dict[str, Any] = {
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
    row.update(overrides)
    return row


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
    first.eq.return_value.eq.return_value.order.return_value.limit.return_value.execute.return_value.data = (
        select
    )
    first.eq.return_value.eq.return_value.order.return_value.range.return_value.execute.return_value.data = (
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


def test_update_case_updates_owned_case() -> None:
    user_id = uuid4()
    case_id = uuid4()
    original = _case_row(case_id, user_id)
    updated = _case_row(case_id, user_id, title="Updated case")
    cases = _table_mock(select=[original], update=[updated])
    client = _client({"cases": cases})

    result = update_case(
        client, user_id, case_id, CaseUpdate(title="Updated case")
    )

    assert result.title == "Updated case"
    assert cases.update.call_args.args[0] == {"title": "Updated case"}


def test_get_document_for_user_returns_owned_document() -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    documents = _table_mock(
        select=[_document_row(document_id, case_id, user_id)]
    )
    client = _client({"cases": cases, "documents": documents})

    result = get_document_for_user(client, user_id, document_id)

    assert result.id == document_id
    assert result.case_id == case_id


def test_find_document_by_sha256_returns_active_document() -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    documents = _table_mock(
        select=[_document_row(document_id, case_id, user_id)]
    )
    client = _client({"cases": cases, "documents": documents})

    result = find_document_by_sha256(client, user_id, case_id, "a" * 64)

    assert result is not None
    assert result.id == document_id


def test_find_document_by_sha256_skips_deletion_requested() -> None:
    user_id = uuid4()
    case_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    documents = _table_mock(
        select=[
            _document_row(
                uuid4(),
                case_id,
                user_id,
                quality_report={"deletion_requested": True},
            )
        ]
    )
    client = _client({"cases": cases, "documents": documents})

    assert find_document_by_sha256(client, user_id, case_id, "a" * 64) is None


def test_get_job_for_user_returns_owned_job() -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    jobs = _table_mock(select=[_job_row(job_id, case_id, document_id)])
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    client = _client({"processing_jobs": jobs, "cases": cases})

    result = get_job_for_user(client, user_id, job_id)

    assert result.id == job_id
    assert result.document_id == document_id


def test_find_job_by_idempotency_key_returns_owned_job() -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    jobs = _table_mock(
        select=[
            _job_row(
                job_id,
                case_id,
                document_id,
                idempotency_key="key-1",
            )
        ]
    )
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    client = _client({"processing_jobs": jobs, "cases": cases})

    result = find_job_by_idempotency_key(client, user_id, "key-1")

    assert result is not None
    assert result.id == job_id


def test_find_job_by_idempotency_key_returns_none() -> None:
    user_id = uuid4()
    jobs = _table_mock(select=[])
    client = _client({"processing_jobs": jobs})

    assert find_job_by_idempotency_key(client, user_id, "missing") is None


def test_create_document_stores_password_required_metadata() -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    documents = _table_mock(
        insert=[
            _document_row(
                document_id,
                case_id,
                user_id,
                quality_status="password_required",
                quality_report={"error_code": "PASSWORD_REQUIRED"},
            )
        ]
    )
    client = _client({"cases": cases, "documents": documents})

    result = create_document(
        client,
        user_id,
        case_id,
        DocumentCreate(
            storage_path=f"{user_id}/{case_id}/{document_id}",
            original_filename="encrypted.pdf",
            detected_mime="application/pdf",
            sha256="a" * 64,
            size_bytes=100,
            quality_status="password_required",
            quality_report={"error_code": "PASSWORD_REQUIRED"},
        ),
    )

    assert result.quality_status == "password_required"
    payload = documents.insert.call_args.args[0]
    assert payload["quality_status"] == "password_required"
    assert payload["quality_report"] == {"error_code": "PASSWORD_REQUIRED"}


def test_create_job_stores_idempotency_and_pipeline_version() -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    documents = _table_mock(select=[_document_row(document_id, case_id, user_id)])
    jobs = _table_mock(
        insert=[
            _job_row(
                job_id,
                case_id,
                document_id,
                idempotency_key=f"{document_id}:v1",
            )
        ]
    )
    client = _client(
        {"cases": cases, "documents": documents, "processing_jobs": jobs}
    )

    result = create_job(
        client,
        user_id,
        case_id,
        document_id,
        JobCreate(
            job_type="ingest",
            idempotency_key=f"{document_id}:v1",
            pipeline_version="v1",
        ),
    )

    assert result.id == job_id
    payload = jobs.insert.call_args.args[0]
    assert payload["idempotency_key"] == f"{document_id}:v1"
    assert payload["pipeline_version"] == "v1"
    assert payload["job_type"] == "ingest"


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


def test_update_page_rewrites_ocr_fields_without_page_number() -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    page_row = _page_row(document_id, case_id)
    page_row.update(
        {
            "extraction_method": "ocr",
            "ocr_confidence": 88.5,
            "raw_text": "OCR text.",
            "quality_flags": ["possible_table"],
        }
    )
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    documents = _table_mock(select=[_document_row(document_id, case_id, user_id)])
    pages = _table_mock(update=[page_row])
    client = _client(
        {"cases": cases, "documents": documents, "document_pages": pages}
    )

    result = update_page(
        client,
        user_id,
        case_id,
        document_id,
        PageCreate(
            page_number=1,
            extraction_method="ocr",
            ocr_confidence=88.5,
            raw_text="OCR text.",
            quality_flags=["possible_table"],
            transform_log=[{"stage": "ocr", "step": "result"}],
        ),
    )

    assert result.extraction_method == "ocr"
    assert result.ocr_confidence == 88.5
    payload = pages.update.call_args.args[0]
    assert "page_number" not in payload
    assert payload["raw_text"] == "OCR text."
    assert payload["quality_flags"] == ["possible_table"]


class _FakeResponse:
    def __init__(self, data: list[dict[str, Any]]) -> None:
        self.data = data


class _FakeQuery:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows
        self.filters: list[tuple[str, Any]] = []
        self.lower: tuple[str, Any] | None = None
        self.upper: tuple[str, Any] | None = None
        self.order_by: tuple[str, bool] | None = None
        self.limit_value: int | None = None

    def select(self, *args: Any, **kwargs: Any) -> "_FakeQuery":
        return self

    def eq(self, key: str, value: Any) -> "_FakeQuery":
        self.filters.append((key, value))
        return self

    def in_(self, key: str, values: list[Any]) -> "_FakeQuery":
        self.filters.append((key, list(values)))
        return self

    def gte(self, key: str, value: Any) -> "_FakeQuery":
        self.lower = (key, value)
        return self

    def lte(self, key: str, value: Any) -> "_FakeQuery":
        self.upper = (key, value)
        return self

    def order(self, key: str, desc: bool = False) -> "_FakeQuery":
        self.order_by = (key, desc)
        return self

    def limit(self, value: int) -> "_FakeQuery":
        self.limit_value = value
        return self

    def execute(self) -> _FakeResponse:
        rows = list(self.rows)
        for key, value in self.filters:
            if isinstance(value, list):
                allowed = {str(item) for item in value}
                rows = [row for row in rows if str(row.get(key)) in allowed]
            else:
                rows = [row for row in rows if row.get(key) == value]
        if self.lower is not None:
            key, value = self.lower
            rows = [row for row in rows if row.get(key) >= value]
        if self.upper is not None:
            key, value = self.upper
            rows = [row for row in rows if row.get(key) <= value]
        if self.order_by is not None:
            key, desc = self.order_by
            rows.sort(key=lambda row: row[key], reverse=desc)
        if self.limit_value is not None:
            rows = rows[: self.limit_value]
        return _FakeResponse(rows)


class _FakeTable:
    def __init__(self, rows: list[dict[str, Any]]) -> None:
        self.rows = rows

    def select(self, *args: Any, **kwargs: Any) -> _FakeQuery:
        return _FakeQuery(self.rows)


class _FakeClient:
    def __init__(self, tables: dict[str, list[dict[str, Any]]]) -> None:
        self.tables = tables

    def table(self, name: str) -> _FakeTable:
        return _FakeTable(self.tables[name])


def _chunk_rows(document_id: UUID, case_id: UUID) -> list[dict[str, Any]]:
    rows = []
    for index in range(5):
        row = _chunk_row(document_id, case_id)
        row.update(
            {
                "chunk_index": index,
                "text_content": f"Chunk {index}.",
                "source_anchor": f"doc:{document_id}|p:1-1|c:{index}",
            }
        )
        rows.append(row)
    return rows


def test_get_neighbor_chunks_returns_ordered_window_around_target() -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    chunks = _chunk_rows(document_id, case_id)
    client = _FakeClient(
        {
            "document_chunks": chunks,
            "documents": [_document_row(document_id, case_id, user_id)],
            "cases": [_case_row(case_id, user_id)],
        }
    )

    result = get_neighbor_chunks(
        client, user_id, chunks[2]["id"], before=1, after=1
    )

    assert [chunk.chunk_index for chunk in result] == [1, 2, 3]


def test_get_neighbor_chunks_checks_document_ownership() -> None:
    user_id = uuid4()
    other_user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    chunks = _chunk_rows(document_id, case_id)
    client = _FakeClient(
        {
            "document_chunks": chunks,
            "documents": [_document_row(document_id, case_id, other_user_id)],
            "cases": [_case_row(case_id, other_user_id)],
        }
    )

    with pytest.raises(DocumentAccessDenied):
        get_neighbor_chunks(client, user_id, chunks[2]["id"])


def test_get_neighbor_chunks_rejects_missing_target() -> None:
    client = _FakeClient({"document_chunks": []})

    with pytest.raises(RecordNotFound):
        get_neighbor_chunks(client, uuid4(), uuid4())


def test_get_neighbor_chunks_rejects_negative_offsets() -> None:
    client = _FakeClient({"document_chunks": []})

    with pytest.raises(ValueError):
        get_neighbor_chunks(client, uuid4(), uuid4(), before=-1)


def test_get_chunks_by_ids_returns_only_requested_chunks() -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    chunks = _chunk_rows(document_id, case_id)
    client = _FakeClient(
        {
            "document_chunks": chunks,
            "documents": [_document_row(document_id, case_id, user_id)],
            "cases": [_case_row(case_id, user_id)],
        }
    )

    result = get_chunks_by_ids(
        client, user_id, case_id, [chunks[0]["id"], chunks[2]["id"]]
    )

    assert [chunk.chunk_index for chunk in result] == [0, 2]


def test_get_chunks_by_ids_rejects_a_case_owned_by_another_user() -> None:
    user_id = uuid4()
    other_user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    chunks = _chunk_rows(document_id, case_id)
    client = _FakeClient(
        {
            "document_chunks": chunks,
            "documents": [_document_row(document_id, case_id, other_user_id)],
            "cases": [_case_row(case_id, other_user_id)],
        }
    )

    with pytest.raises(CaseAccessDenied):
        get_chunks_by_ids(client, user_id, case_id, [chunks[0]["id"]])


def test_get_chunks_by_ids_without_ids_returns_nothing() -> None:
    user_id = uuid4()
    case_id = uuid4()
    client = _FakeClient(
        {
            "document_chunks": [],
            "documents": [],
            "cases": [_case_row(case_id, user_id)],
        }
    )

    assert get_chunks_by_ids(client, user_id, case_id, []) == []


def test_get_pages_for_documents_returns_only_requested_documents() -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    other_document_id = uuid4()
    pages = [
        _page_row(document_id, case_id),
        _page_row(other_document_id, case_id),
    ]
    client = _FakeClient(
        {
            "document_pages": pages,
            "documents": [_document_row(document_id, case_id, user_id)],
            "cases": [_case_row(case_id, user_id)],
        }
    )

    result = get_pages_for_documents(
        client, user_id, case_id, [document_id]
    )

    assert [page.document_id for page in result] == [document_id]


def test_get_pages_for_documents_rejects_a_foreign_case() -> None:
    user_id = uuid4()
    other_user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    client = _FakeClient(
        {
            "document_pages": [_page_row(document_id, case_id)],
            "documents": [_document_row(document_id, case_id, other_user_id)],
            "cases": [_case_row(case_id, other_user_id)],
        }
    )

    with pytest.raises(CaseAccessDenied):
        get_pages_for_documents(client, user_id, case_id, [document_id])


def test_insert_answer_stores_the_payload_and_manifest() -> None:
    user_id = uuid4()
    case_id = uuid4()
    answer_id = uuid4()
    row = {
        "id": str(answer_id),
        "case_id": str(case_id),
        "user_id": str(user_id),
        "question": "Did the tenant pay the rent?",
        "answer_payload": {"final_status": "answered"},
        "evidence_refs": [{"evidence_id": str(uuid4()), "page_start": 3}],
        "graph_version": "baseline-rag-1",
        "final_status": "answered",
        "created_at": TIMESTAMP,
    }
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    answers = _table_mock(insert=[row])
    client = _client({"cases": cases, "answers": answers})

    result = insert_answer(
        client,
        user_id,
        case_id,
        AnswerCreate(
            question="Did the tenant pay the rent?",
            answer_payload={"final_status": "answered"},
            evidence_refs=[{"evidence_id": str(uuid4()), "page_start": 3}],
            graph_version="baseline-rag-1",
            final_status="answered",
        ),
    )

    assert result.id == answer_id
    assert result.final_status == "answered"
    payload = answers.insert.call_args.args[0]
    assert payload["case_id"] == str(case_id)
    assert payload["user_id"] == str(user_id)
    assert payload["evidence_refs"][0]["page_start"] == 3


def test_list_answers_scopes_to_the_owner_and_paginates() -> None:
    user_id = uuid4()
    case_id = uuid4()
    row = {
        "id": str(uuid4()),
        "case_id": str(case_id),
        "user_id": str(user_id),
        "question": "What is the monthly rent?",
        "answer_payload": {"final_status": "answered"},
        "evidence_refs": [],
        "graph_version": "baseline-rag-1",
        "final_status": "answered",
        "created_at": TIMESTAMP,
    }
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    answers = _table_mock(select=[row])
    client = _client({"cases": cases, "answers": answers})

    result = list_answers(client, user_id, case_id, limit=5, offset=10)

    assert [str(answer.id) for answer in result] == [row["id"]]
    answers.select.return_value.eq.assert_called_once_with(
        "case_id", str(case_id)
    )
    answers.select.return_value.eq.return_value.eq.assert_called_once_with(
        "user_id", str(user_id)
    )
    answers.select.return_value.eq.return_value.eq.return_value.order.return_value.range.assert_called_once_with(
        10, 14
    )


def test_list_answers_rejects_a_case_owned_by_another_user() -> None:
    user_id = uuid4()
    case_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, uuid4())])
    answers = _table_mock(select=[])
    client = _client({"cases": cases, "answers": answers})

    with pytest.raises(CaseAccessDenied):
        list_answers(client, user_id, case_id)


def test_insert_answer_rejects_a_case_owned_by_another_user() -> None:
    user_id = uuid4()
    other_user_id = uuid4()
    case_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, other_user_id)])
    answers = _table_mock(insert=[])
    client = _client({"cases": cases, "answers": answers})

    with pytest.raises(CaseAccessDenied):
        insert_answer(
            client,
            user_id,
            case_id,
            AnswerCreate(
                question="question",
                answer_payload={},
                final_status="answered",
            ),
        )
