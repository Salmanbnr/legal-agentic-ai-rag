"""Tests for comparing Postgres chunk ids with Pinecone vector ids."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest
from fakes import FakePineconeIndex

from app.core.config import Settings
from app.db.repos import Document, ProcessingJob
from app.indexing import reconcile as reconcile_module
from app.indexing.reconcile import fix_reconcile, reconcile_case

TIMESTAMP = "2026-01-01T00:00:00+00:00"


@dataclass
class ReconcileHarness:
    """In-memory chunks and documents used by reconciliation tests."""

    settings: Settings
    chunks: dict[str, list[Any]] = field(default_factory=dict)
    documents: dict[str, Document] = field(default_factory=dict)
    created_jobs: list[Any] = field(default_factory=list)
    existing_jobs: dict[str, ProcessingJob] = field(default_factory=dict)


def _document(case_id: UUID, owner_id: UUID) -> Document:
    return Document.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "owner_user_id": str(owner_id),
            "storage_path": f"{owner_id}/{case_id}/{uuid4()}",
            "original_filename": "clean.pdf",
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


def _chunk(case_id: UUID, document_id: UUID, index: int, status: str):
    from app.db.repos import DocumentChunk

    return DocumentChunk.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "document_id": str(document_id),
            "document_version_id": None,
            "chunk_index": index,
            "page_start": 1,
            "page_end": 1,
            "section_path": [],
            "text_content": f"Chunk {index}",
            "content_hash": f"{index:064d}",
            "source_anchor": None,
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "index_schema_version": 1,
            "index_status": status,
            "quality_flags": [],
            "created_at": TIMESTAMP,
        }
    )


@pytest.fixture()
def harness(monkeypatch: pytest.MonkeyPatch) -> ReconcileHarness:
    """Provide a reconciliation harness wired to in-memory storage."""
    state = ReconcileHarness(settings=Settings(_env_file=None))

    monkeypatch.setattr(
        reconcile_module,
        "get_chunks_for_case",
        lambda client, case_id: list(state.chunks.get(str(case_id), [])),
    )
    monkeypatch.setattr(
        reconcile_module,
        "get_document_for_worker",
        lambda client, document_id: state.documents[str(document_id)],
    )
    monkeypatch.setattr(
        reconcile_module,
        "find_job_by_idempotency_key",
        lambda client, user_id, key: state.existing_jobs.get(key),
    )

    def fake_create_job(
        client, owner_id, case_id, document_id, data
    ) -> ProcessingJob:
        state.created_jobs.append(data)
        job = ProcessingJob.model_validate(
            {
                "id": str(uuid4()),
                "case_id": str(case_id),
                "document_id": str(document_id),
                "job_type": data.job_type,
                "idempotency_key": data.idempotency_key,
                "status": "queued",
                "attempts": 0,
                "error_code": None,
                "error_summary": None,
                "pipeline_version": data.pipeline_version,
                "started_at": None,
                "finished_at": None,
                "created_at": TIMESTAMP,
            }
        )
        state.existing_jobs[data.idempotency_key] = job
        return job

    monkeypatch.setattr(reconcile_module, "create_job", fake_create_job)
    return state


def test_reconcile_reports_orphaned_and_missing_vectors(
    harness: ReconcileHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = _document(case_id, owner_id)
    harness.documents[str(document.id)] = document
    indexed = _chunk(case_id, document.id, 0, "indexed")
    pending = _chunk(case_id, document.id, 1, "pending")
    harness.chunks[str(case_id)] = [indexed, pending]

    index = FakePineconeIndex()
    index.upsert(
        vectors=[
            (str(indexed.id), [1.0] * 384, {}),
            ("orphan-id", [1.0] * 384, {}),
        ],
        namespace=str(case_id),
    )

    report = reconcile_case(
        object(), case_id, settings=harness.settings, index=index
    )

    assert report.orphaned_ids == ["orphan-id"]
    assert report.missing_ids == []
    assert report.postgres_chunk_count == 2
    assert report.pinecone_vector_count == 2
    assert report.is_clean is False


def test_reconcile_reports_indexed_chunks_without_vectors(
    harness: ReconcileHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = _document(case_id, owner_id)
    harness.documents[str(document.id)] = document
    indexed = _chunk(case_id, document.id, 0, "indexed")
    harness.chunks[str(case_id)] = [indexed]

    report = reconcile_case(
        object(),
        case_id,
        settings=harness.settings,
        index=FakePineconeIndex(),
    )

    assert report.missing_ids == [str(indexed.id)]
    assert report.orphaned_ids == []


def test_fix_deletes_orphans_and_queues_a_reindex(
    harness: ReconcileHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = _document(case_id, owner_id)
    harness.documents[str(document.id)] = document
    indexed = _chunk(case_id, document.id, 0, "indexed")
    harness.chunks[str(case_id)] = [indexed]

    index = FakePineconeIndex()
    index.upsert(
        vectors=[("orphan-id", [1.0] * 384, {})],
        namespace=str(case_id),
    )

    result = fix_reconcile(
        object(), case_id, settings=harness.settings, index=index
    )

    assert result.deleted_orphan_count == 1
    assert index.vectors[str(case_id)] == {}
    assert len(harness.created_jobs) == 1
    assert harness.created_jobs[0].job_type == "reindex"
    assert str(document.id) in harness.created_jobs[0].idempotency_key


def test_fix_is_a_no_op_for_a_clean_case(
    harness: ReconcileHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = _document(case_id, owner_id)
    harness.documents[str(document.id)] = document
    indexed = _chunk(case_id, document.id, 0, "indexed")
    harness.chunks[str(case_id)] = [indexed]

    index = FakePineconeIndex()
    index.upsert(
        vectors=[(str(indexed.id), [1.0] * 384, {})],
        namespace=str(case_id),
    )

    result = fix_reconcile(
        object(), case_id, settings=harness.settings, index=index
    )

    assert result.report.is_clean is True
    assert result.deleted_orphan_count == 0
    assert harness.created_jobs == []
