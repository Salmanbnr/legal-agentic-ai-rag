"""Tests for worker-scoped job claiming and persistence helpers."""

from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock
from uuid import uuid4

from app.db.repos import Document, DocumentUpdate
from app.jobs import store

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
TIMESTAMP = NOW.isoformat()


def _job_row(**overrides):
    row = {
        "id": str(uuid4()),
        "case_id": str(uuid4()),
        "document_id": str(uuid4()),
        "job_type": "ingest",
        "idempotency_key": str(uuid4()),
        "status": "queued",
        "attempts": 0,
        "error_code": None,
        "error_summary": None,
        "pipeline_version": "1.0.0",
        "started_at": None,
        "finished_at": None,
        "next_attempt_at": TIMESTAMP,
        "created_at": TIMESTAMP,
    }
    row.update(overrides)
    return row


def _select_client(rows):
    table = MagicMock()
    query = MagicMock()
    table.select.return_value = query
    query.eq.return_value = query
    query.lt.return_value = query
    query.lte.return_value = query
    query.order.return_value = query
    query.limit.return_value = query
    query.execute.return_value.data = rows
    client = MagicMock()
    client.table.return_value = table
    return client, table, query


def _update_client(updated_rows):
    table = MagicMock()
    update = MagicMock()
    table.update.return_value = update
    update.eq.return_value = update
    update.execute.return_value.data = updated_rows
    client = MagicMock()
    client.table.return_value = table
    return client, table, update


def test_claim_next_job_uses_conditional_update() -> None:
    row = _job_row(attempts=0)
    claimed = _job_row(id=row["id"], status="running", attempts=1)
    table = MagicMock()
    query = MagicMock()
    table.select.return_value = query
    query.eq.return_value = query
    query.lt.return_value = query
    query.lte.return_value = query
    query.order.return_value = query
    query.limit.return_value = query
    query.execute.return_value.data = [row]
    update = MagicMock()
    table.update.return_value = update
    update.eq.return_value = update
    update.execute.return_value.data = [claimed]
    client = MagicMock()
    client.table.return_value = table

    result = store.claim_next_job(client, now=NOW, max_attempts=3)

    assert result is not None
    assert str(result.id) == row["id"]
    assert result.status == "running"
    payload = table.update.call_args.args[0]
    assert payload["status"] == "running"
    assert payload["attempts"] == 1
    assert payload["started_at"] == TIMESTAMP
    assert payload["error_code"] is None
    assert update.eq.call_args_list[-1].args == ("status", "queued")


def test_claim_next_job_returns_none_when_queue_empty() -> None:
    client, _, query = _select_client([])

    assert store.claim_next_job(client, now=NOW, max_attempts=3) is None
    assert query.lt.call_args.args == ("attempts", 3)
    assert query.lte.call_args.args == ("next_attempt_at", TIMESTAMP)


def test_mark_job_succeeded_sets_pipeline_and_finish() -> None:
    job_id = uuid4()
    row = _job_row(
        id=str(job_id), status="succeeded", attempts=1, finished_at=TIMESTAMP
    )
    client, table, _ = _update_client([row])

    result = store.mark_job_succeeded(
        client, job_id, pipeline_version="1.0.0", now=NOW
    )

    assert result.status == "succeeded"
    payload = table.update.call_args.args[0]
    assert payload["status"] == "succeeded"
    assert payload["pipeline_version"] == "1.0.0"
    assert payload["finished_at"] == TIMESTAMP


def test_schedule_job_retry_persists_backoff_time() -> None:
    job_id = uuid4()
    row = _job_row(
        id=str(job_id),
        status="queued",
        attempts=1,
        error_code="STORAGE_ERROR",
        next_attempt_at=TIMESTAMP,
    )
    client, table, _ = _update_client([row])

    result = store.schedule_job_retry(
        client,
        job_id,
        attempts=1,
        next_attempt_at=NOW,
        error_code="STORAGE_ERROR",
        now=NOW,
    )

    assert result.status == "queued"
    payload = table.update.call_args.args[0]
    assert payload["next_attempt_at"] == TIMESTAMP
    assert payload["error_code"] == "STORAGE_ERROR"
    assert payload["error_summary"] == "Processing attempt failed"


def test_mark_job_failed_stores_safe_code() -> None:
    job_id = uuid4()
    row = _job_row(
        id=str(job_id), status="failed", attempts=3, error_code="PROCESSING_FAILED"
    )
    client, table, _ = _update_client([row])

    result = store.mark_job_failed(
        client, job_id, error_code="PROCESSING_FAILED", now=NOW
    )

    assert result.status == "failed"
    payload = table.update.call_args.args[0]
    assert payload["status"] == "failed"
    assert payload["error_code"] == "PROCESSING_FAILED"
    assert "document" not in str(payload).lower()


def test_delete_derived_rows_targets_document() -> None:
    table = MagicMock()
    delete = MagicMock()
    table.delete.return_value = delete
    delete.eq.return_value = delete
    delete.execute.return_value.data = []
    client = MagicMock()
    client.table.return_value = table
    document_id = uuid4()

    store.delete_pages_for_document(client, document_id)
    store.delete_chunks_for_document(client, document_id)

    assert delete.eq.call_count == 2
    assert delete.eq.call_args_list[0].args == ("document_id", str(document_id))
    assert table.delete.call_count == 2


def test_mark_document_failed_preserves_safe_quality_report(monkeypatch) -> None:
    document = Document.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(uuid4()),
            "owner_user_id": str(uuid4()),
            "storage_path": f"{uuid4()}/{uuid4()}/{uuid4()}",
            "original_filename": "clean.pdf",
            "detected_mime": "application/pdf",
            "sha256": "a" * 64,
            "size_bytes": 100,
            "page_count": 1,
            "document_type": None,
            "version_group_id": None,
            "duplicate_of": None,
            "quality_status": "pending",
            "quality_report": {"previous": True},
            "processing_status": "extracting",
            "parser_version": "1.0.0",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    )
    captured: dict[str, DocumentUpdate] = {}

    monkeypatch.setattr(
        store, "get_document_for_worker", lambda *args, **kwargs: document
    )

    def fake_update(client, user_id, document_id, update):
        captured["update"] = update
        row = document.model_dump()
        row.update(update.model_dump(mode="json", exclude_unset=True))
        return Document.model_validate(row)

    monkeypatch.setattr(store, "update_document", fake_update)

    result = store.mark_document_failed(
        MagicMock(), document.id, error_code="PROCESSING_FAILED"
    )

    assert result.processing_status == "failed"
    assert result.quality_status == "failed"
    assert result.quality_report == {
        "previous": True,
        "error_code": "PROCESSING_FAILED",
    }


def test_get_pages_orders_by_page_number() -> None:
    document_id = uuid4()
    row = {
        "id": str(uuid4()),
        "case_id": str(uuid4()),
        "document_id": str(document_id),
        "page_number": 1,
        "extraction_method": "native",
        "text_density": 0.01,
        "ocr_confidence": None,
        "raw_text": "text",
        "clean_text": None,
        "quality_flags": [],
        "transform_log": [],
        "created_at": TIMESTAMP,
    }
    client, _, query = _select_client([row])

    pages = store.get_pages_for_document(client, document_id)

    assert len(pages) == 1
    assert query.order.call_args.args == ("page_number",)
