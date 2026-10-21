"""Tests for chunk embedding, Pinecone upserts, verification, and deletion."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest
from fakes import FakeEmbedder, FakePineconeIndex

from app.core.config import Settings
from app.db.repos import Document, DocumentChunk, DocumentUpdate
from app.indexing import indexer as indexer_module
from app.indexing.indexer import (
    index_document_chunks,
    run_delete_document,
    run_reindex_document,
)

TIMESTAMP = "2026-01-01T00:00:00+00:00"

_ALLOWED_METADATA_KEYS = {
    "case_id",
    "document_id",
    "document_version_id",
    "page_start",
    "page_end",
    "chunk_index",
    "quality_flags",
    "embedding_model",
    "index_schema_version",
}


def _document(
    case_id: UUID, owner_id: UUID, document_id: UUID | None = None
) -> Document:
    resolved_id = document_id or uuid4()
    return Document.model_validate(
        {
            "id": str(resolved_id),
            "case_id": str(case_id),
            "owner_user_id": str(owner_id),
            "storage_path": f"{owner_id}/{case_id}/{resolved_id}",
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
            "processing_status": "uploaded",
            "parser_version": "1.0.0",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    )


def _chunk(
    case_id: UUID,
    document_id: UUID,
    index: int,
    *,
    index_status: str = "pending",
    embedding_model: str | None = None,
    index_schema_version: int = 1,
) -> DocumentChunk:
    return DocumentChunk.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "document_id": str(document_id),
            "document_version_id": None,
            "chunk_index": index,
            "page_start": 1,
            "page_end": 1,
            "section_path": ["Article 1"],
            "text_content": f"Chunk number {index} with legal text.",
            "content_hash": f"{index:064d}",
            "source_anchor": f"doc:{document_id}|p:1-1|c:{index}",
            "embedding_model": embedding_model,
            "index_schema_version": index_schema_version,
            "index_status": index_status,
            "quality_flags": [],
            "created_at": TIMESTAMP,
        }
    )


@dataclass
class IndexerHarness:
    """In-memory stand-ins for the worker's document and chunk storage."""

    settings: Settings
    documents: dict[str, Document]
    chunks: dict[str, list[DocumentChunk]] = field(default_factory=dict)
    document_updates: list[DocumentUpdate] = field(default_factory=list)
    deleted_documents: list[str] = field(default_factory=list)
    deleted_originals: list[str] = field(default_factory=list)

    def add_document(self, document: Document) -> Document:
        """Register a document with its chunk list."""
        self.documents[str(document.id)] = document
        self.chunks.setdefault(str(document.id), [])
        return document

    def add_chunks(
        self, document: Document, chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """Register chunks for a document."""
        self.chunks[str(document.id)] = list(chunks)
        return chunks


@pytest.fixture()
def settings() -> Settings:
    """Provide validated test settings."""
    return Settings(_env_file=None)


@pytest.fixture()
def harness(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> IndexerHarness:
    """Provide an indexer harness wired to in-memory storage."""
    state = IndexerHarness(settings=settings, documents={})

    def fake_get_document(client, document_id):
        document = state.documents.get(str(document_id))
        if document is None:
            raise AssertionError("unknown document")
        return document

    def fake_update_document(client, user_id, document_id, update):
        existing = state.documents[str(document_id)]
        row = existing.model_dump()
        row.update(update.model_dump(mode="json", exclude_unset=True))
        updated = Document.model_validate(row)
        state.documents[str(document_id)] = updated
        state.document_updates.append(update)
        return updated

    def fake_get_chunks(client, document_id, **kwargs):
        return list(state.chunks.get(str(document_id), []))

    def fake_mark_indexed(client, chunk_ids, *, embedding_model):
        marked = 0
        for chunks in state.chunks.values():
            for chunk in chunks:
                if str(chunk.id) in {str(value) for value in chunk_ids}:
                    chunk.index_status = "indexed"
                    chunk.embedding_model = embedding_model
                    marked += 1
        return marked

    def fake_mark_stale(
        client, document_id, *, embedding_model, index_schema_version
    ):
        marked = 0
        for chunk in state.chunks.get(str(document_id), []):
            if chunk.index_status == "indexed" and (
                chunk.embedding_model != embedding_model
                or chunk.index_schema_version != index_schema_version
            ):
                chunk.index_status = "stale"
                marked += 1
        return marked

    def fake_delete_document_row(client, document_id):
        state.deleted_documents.append(str(document_id))
        state.documents.pop(str(document_id), None)

    def fake_delete_original(storage_path, *, client=None):
        state.deleted_originals.append(storage_path)

    monkeypatch.setattr(
        indexer_module, "get_document_for_worker", fake_get_document
    )
    monkeypatch.setattr(indexer_module, "update_document", fake_update_document)
    monkeypatch.setattr(
        indexer_module, "get_chunks_for_document", fake_get_chunks
    )
    monkeypatch.setattr(indexer_module, "mark_chunks_indexed", fake_mark_indexed)
    monkeypatch.setattr(indexer_module, "mark_chunks_stale", fake_mark_stale)
    monkeypatch.setattr(
        indexer_module, "delete_document_row", fake_delete_document_row
    )
    monkeypatch.setattr(indexer_module, "delete_original", fake_delete_original)
    return state


def _job(document: Document, job_type: str = "reindex") -> Any:
    class Job:
        def __init__(self) -> None:
            self.id = uuid4()
            self.case_id = document.case_id
            self.document_id = document.id
            self.job_type = job_type
            self.attempts = 1

    return Job()


def test_index_document_chunks_upserts_in_batches_of_one_hundred(
    harness: IndexerHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(_document(case_id, owner_id))
    harness.add_chunks(
        document,
        [_chunk(case_id, document.id, index) for index in range(250)],
    )
    index = FakePineconeIndex()
    embedder = FakeEmbedder()

    result = index_document_chunks(
        object(),
        document.id,
        settings=harness.settings,
        embedder=embedder,
        index=index,
    )

    assert [len(call[0]) for call in index.upsert_calls] == [100, 100, 50]
    assert result.status == "ready"
    assert result.chunk_count == 250
    assert result.indexed_count == 250
    assert len(index.vectors[str(case_id)]) == 250
    assert all(
        chunk.index_status == "indexed" for chunk in harness.chunks[str(document.id)]
    )
    assert harness.documents[str(document.id)].processing_status == "ready"


def test_embedding_batches_respect_the_configured_size(
    harness: IndexerHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(_document(case_id, owner_id))
    harness.add_chunks(
        document,
        [_chunk(case_id, document.id, index) for index in range(5)],
    )
    embedder = FakeEmbedder()

    index_document_chunks(
        object(),
        document.id,
        settings=harness.settings,
        embedder=embedder,
        index=FakePineconeIndex(),
    )

    assert [len(batch) for batch in embedder.batches] == [5]


def test_rerunning_the_indexer_does_not_duplicate_vectors(
    harness: IndexerHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(_document(case_id, owner_id))
    harness.add_chunks(
        document,
        [_chunk(case_id, document.id, index) for index in range(3)],
    )
    index = FakePineconeIndex()
    embedder = FakeEmbedder()

    first = index_document_chunks(
        object(),
        document.id,
        settings=harness.settings,
        embedder=embedder,
        index=index,
    )
    second = index_document_chunks(
        object(),
        document.id,
        settings=harness.settings,
        embedder=embedder,
        index=index,
    )

    assert first.status == "ready"
    assert second.status == "ready"
    assert len(index.upsert_calls) == 1
    assert len(index.vectors[str(case_id)]) == 3


def test_count_mismatch_marks_the_document_needs_review(
    harness: IndexerHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(_document(case_id, owner_id))
    chunks = harness.add_chunks(
        document,
        [_chunk(case_id, document.id, index) for index in range(3)],
    )
    index = FakePineconeIndex()
    index.drop_upsert_ids = {str(chunks[1].id)}

    result = index_document_chunks(
        object(),
        document.id,
        settings=harness.settings,
        embedder=FakeEmbedder(),
        index=index,
    )

    assert result.status == "needs_review"
    assert result.indexed_count == 2
    assert result.chunk_count == 3
    stored = harness.documents[str(document.id)]
    assert stored.processing_status == "needs_review"
    assert stored.quality_status == "needs_review"
    assert stored.quality_report["indexing"]["error_code"] == (
        "INDEX_COUNT_MISMATCH"
    )


def test_vector_metadata_excludes_document_text(
    harness: IndexerHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(_document(case_id, owner_id))
    harness.add_chunks(document, [_chunk(case_id, document.id, 0)])
    index = FakePineconeIndex()

    index_document_chunks(
        object(),
        document.id,
        settings=harness.settings,
        embedder=FakeEmbedder(),
        index=index,
    )

    vectors = index.upsert_calls[0][0]
    vector_id, _values, metadata = vectors[0]
    assert set(metadata) <= _ALLOWED_METADATA_KEYS
    assert metadata["case_id"] == str(case_id)
    assert metadata["document_id"] == str(document.id)
    assert metadata["chunk_index"] == 0
    assert metadata["page_start"] == 1
    assert metadata["page_end"] == 1
    assert metadata["embedding_model"] == harness.settings.embedding_model
    assert metadata["index_schema_version"] == (
        harness.settings.index_schema_version
    )
    assert "Chunk number 0" not in str(metadata)
    assert vector_id == str(harness.chunks[str(document.id)][0].id)


def test_document_without_chunks_is_not_marked_ready(
    harness: IndexerHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(_document(case_id, owner_id))

    result = index_document_chunks(
        object(),
        document.id,
        settings=harness.settings,
        embedder=FakeEmbedder(),
        index=FakePineconeIndex(),
    )

    assert result.status == "needs_review"
    assert result.chunk_count == 0
    stored = harness.documents[str(document.id)]
    assert stored.processing_status == "needs_review"
    assert stored.quality_report["indexing"]["error_code"] == "NO_CHUNKS"


def test_reindex_marks_stale_chunks_and_reembeds_them(
    harness: IndexerHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(_document(case_id, owner_id))
    harness.add_chunks(
        document,
        [
            _chunk(
                case_id,
                document.id,
                0,
                index_status="indexed",
                embedding_model="old-model",
                index_schema_version=1,
            )
        ],
    )
    index = FakePineconeIndex()
    embedder = FakeEmbedder()

    result = run_reindex_document(
        object(),
        _job(document),
        settings=harness.settings,
        embedder=embedder,
        index=index,
    )

    assert result.status == "ready"
    assert len(index.upsert_calls) == 1
    assert harness.chunks[str(document.id)][0].index_status == "indexed"
    assert harness.chunks[str(document.id)][0].embedding_model == (
        harness.settings.embedding_model
    )


def test_delete_job_removes_vectors_originals_and_rows(
    harness: IndexerHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(_document(case_id, owner_id))
    chunks = harness.add_chunks(
        document,
        [_chunk(case_id, document.id, index) for index in range(2)],
    )
    index = FakePineconeIndex()
    for chunk in chunks:
        index.upsert(
            vectors=[(str(chunk.id), [1.0] * 384, {})],
            namespace=str(case_id),
        )

    result = run_delete_document(
        object(),
        _job(document, job_type="delete"),
        settings=harness.settings,
        index=index,
    )

    assert result.status == "deleted"
    assert result.deleted_vector_count == 2
    assert index.vectors[str(case_id)] == {}
    assert harness.deleted_originals == [document.storage_path]
    assert harness.deleted_documents == [str(document.id)]
    assert str(document.id) not in harness.documents
