"""Embed pending chunks and keep Pinecone in sync with Postgres.

Postgres remains the source of truth. Indexing embeds chunk text, upserts
vectors whose ids are chunk ids, marks those rows indexed, and only then marks
the document ready after confirming every chunk id exists in the namespace.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence
from uuid import UUID

from supabase import Client

from app.core.config import Settings, get_settings
from app.db.repos import Document, DocumentChunk, DocumentUpdate
from app.embeddings.model import TextEmbedder, get_embedder
from app.indexing import pinecone_store
from app.jobs.store import (
    delete_document_row,
    get_chunks_for_document,
    get_document_for_worker,
    mark_chunks_indexed,
    mark_chunks_stale,
    update_document,
)
from app.storage.supabase_storage import StorageError, delete_original

logger = logging.getLogger(__name__)

_REINDEXABLE_STATUSES = ("pending", "stale", "failed")


class IndexingError(RuntimeError):
    """A safe indexing failure with a stable, persistable error code."""

    def __init__(self, error_code: str, *, retryable: bool = True) -> None:
        """Create an indexing error without including text or vectors."""
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class IndexResult:
    """Outcome of an indexing run for one document."""

    status: str
    chunk_count: int
    indexed_count: int
    document: Document


@dataclass(frozen=True, slots=True)
class DeleteResult:
    """Outcome of a document deletion job."""

    status: str
    deleted_vector_count: int


def _needs_indexing(chunk: DocumentChunk, settings: Settings) -> bool:
    """Return whether a chunk must be embedded for the current index schema."""
    if chunk.index_status in _REINDEXABLE_STATUSES:
        return True
    if chunk.index_status == "indexed":
        return (
            chunk.embedding_model != settings.embedding_model
            or chunk.index_schema_version != settings.index_schema_version
        )
    return False


def _safe_flags(values: Sequence[Any]) -> list[str]:
    """Keep Pinecone metadata to its supported scalar-list shape."""
    return [str(value) for value in values]


def chunk_vector_metadata(
    chunk: DocumentChunk, settings: Settings
) -> dict[str, Any]:
    """Build the allowed Pinecone metadata for one chunk.

    Document text is deliberately excluded. Null values are omitted because
    Pinecone metadata does not support null.
    """
    metadata: dict[str, Any] = {
        "case_id": str(chunk.case_id),
        "document_id": str(chunk.document_id),
        "page_start": chunk.page_start,
        "page_end": chunk.page_end,
        "chunk_index": chunk.chunk_index,
        "quality_flags": _safe_flags(chunk.quality_flags),
        "embedding_model": settings.embedding_model,
        "index_schema_version": settings.index_schema_version,
    }
    if chunk.document_version_id is not None:
        metadata["document_version_id"] = str(chunk.document_version_id)
    return {key: value for key, value in metadata.items() if value is not None}


def _build_vectors(
    chunks: Sequence[DocumentChunk],
    embedder: TextEmbedder,
    settings: Settings,
) -> list[tuple[str, list[float], dict[str, Any]]]:
    """Embed a batch and pair each vector with its chunk id."""
    vectors = embedder.embed_texts([chunk.text_content for chunk in chunks])
    if len(vectors) != len(chunks):
        raise IndexingError("EMBEDDING_FAILED")
    return [
        (
            str(chunk.id),
            vector,
            chunk_vector_metadata(chunk, settings),
        )
        for chunk, vector in zip(chunks, vectors)
    ]


def _index_report(
    document: Document,
    *,
    chunk_count: int,
    indexed_count: int,
    settings: Settings,
    error_code: str | None = None,
) -> dict[str, Any]:
    """Merge safe indexing counts into the document quality report."""
    report = dict(document.quality_report or {})
    indexing: dict[str, Any] = {
        "chunk_count": chunk_count,
        "indexed_vector_count": indexed_count,
        "embedding_model": settings.embedding_model,
        "index_schema_version": settings.index_schema_version,
    }
    if error_code is not None:
        indexing["error_code"] = error_code
    report["indexing"] = indexing
    return report


def index_document_chunks(
    client: Client,
    document_id: UUID | str,
    *,
    settings: Settings | None = None,
    embedder: TextEmbedder | None = None,
    index: Any | None = None,
) -> IndexResult:
    """Embed, upsert, and verify every outstanding chunk for a document.

    Upserts are idempotent because the vector id is the chunk id. A count
    mismatch leaves the document in ``needs_review`` with a safe note instead
    of claiming the index is complete.
    """
    resolved = settings or get_settings()
    document = get_document_for_worker(client, document_id)
    resolved_index = index or pinecone_store.get_index(resolved)

    document = update_document(
        client,
        document.owner_user_id,
        document.id,
        DocumentUpdate(processing_status="indexing"),
    )

    chunks = get_chunks_for_document(client, document.id)
    if not chunks:
        status = "failed" if document.quality_status == "failed" else "needs_review"
        quality_status = document.quality_status
        if quality_status not in ("failed", "needs_review"):
            quality_status = "needs_review"
        updated = update_document(
            client,
            document.owner_user_id,
            document.id,
            DocumentUpdate(
                processing_status=status,
                quality_status=quality_status,
                quality_report=_index_report(
                    document,
                    chunk_count=0,
                    indexed_count=0,
                    settings=resolved,
                    error_code="NO_CHUNKS",
                ),
            ),
        )
        logger.warning(
            "indexing skipped because the document has no chunks",
            extra={"document_id": str(document.id), "status": status},
        )
        return IndexResult(
            status=status,
            chunk_count=0,
            indexed_count=0,
            document=updated,
        )

    pending = [chunk for chunk in chunks if _needs_indexing(chunk, resolved)]
    batch_size = resolved.pinecone_upsert_batch_size
    embedded_count = 0
    namespace = pinecone_store.namespace_for_case(document.case_id)
    active_embedder = embedder or get_embedder(resolved)

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        vectors = _build_vectors(batch, active_embedder, resolved)
        pinecone_store.upsert_with_retry(
            resolved_index,
            namespace,
            vectors,
            settings=resolved,
        )
        mark_chunks_indexed(
            client,
            [chunk.id for chunk in batch],
            embedding_model=resolved.embedding_model,
        )
        embedded_count += len(batch)

    all_ids = [str(chunk.id) for chunk in chunks]
    present = pinecone_store.fetch_existing_ids(
        resolved_index, namespace, all_ids
    )
    indexed_count = len(set(all_ids) & present)

    if indexed_count != len(chunks):
        updated = update_document(
            client,
            document.owner_user_id,
            document.id,
            DocumentUpdate(
                processing_status="needs_review",
                quality_status="needs_review",
                quality_report=_index_report(
                    document,
                    chunk_count=len(chunks),
                    indexed_count=indexed_count,
                    settings=resolved,
                    error_code="INDEX_COUNT_MISMATCH",
                ),
            ),
        )
        logger.error(
            "indexed vector count does not match chunk count",
            extra={
                "document_id": str(document.id),
                "chunk_count": len(chunks),
                "indexed_vector_count": indexed_count,
            },
        )
        return IndexResult(
            status="needs_review",
            chunk_count=len(chunks),
            indexed_count=indexed_count,
            document=updated,
        )

    quality_status = document.quality_status
    if quality_status in ("pending", "failed", "password_required"):
        quality_status = "needs_review"
    updated = update_document(
        client,
        document.owner_user_id,
        document.id,
        DocumentUpdate(
            processing_status="ready",
            quality_status=quality_status,
            quality_report=_index_report(
                document,
                chunk_count=len(chunks),
                indexed_count=indexed_count,
                settings=resolved,
            ),
        ),
    )
    logger.info(
        "indexing stage finished",
        extra={
            "document_id": str(document.id),
            "chunk_count": len(chunks),
            "embedded_count": embedded_count,
            "indexed_vector_count": indexed_count,
            "namespace": namespace,
        },
    )
    return IndexResult(
        status="ready",
        chunk_count=len(chunks),
        indexed_count=indexed_count,
        document=updated,
    )


def run_reindex_document(
    client: Client,
    job: Any,
    *,
    settings: Settings | None = None,
    embedder: TextEmbedder | None = None,
    index: Any | None = None,
) -> IndexResult:
    """Mark stale chunks and reindex one document for a queued reindex job."""
    if job.job_type != "reindex":
        raise IndexingError("UNSUPPORTED_JOB_TYPE", retryable=False)
    resolved = settings or get_settings()
    document = get_document_for_worker(client, job.document_id)
    mark_chunks_stale(
        client,
        document.id,
        embedding_model=resolved.embedding_model,
        index_schema_version=resolved.index_schema_version,
    )
    return index_document_chunks(
        client,
        document.id,
        settings=resolved,
        embedder=embedder,
        index=index,
    )


def run_delete_document(
    client: Client,
    job: Any,
    *,
    settings: Settings | None = None,
    index: Any | None = None,
) -> DeleteResult:
    """Delete vectors, the original object, and cascading database rows.

    The document row is deleted last. Its foreign keys cascade to pages,
    chunks, and the running job row, so the worker must not mark the job
    succeeded after this function returns.
    """
    if job.job_type != "delete":
        raise IndexingError("UNSUPPORTED_JOB_TYPE", retryable=False)
    resolved = settings or get_settings()
    document = get_document_for_worker(client, job.document_id)
    chunks = get_chunks_for_document(client, document.id)
    chunk_ids = [str(chunk.id) for chunk in chunks]

    deleted_vectors = pinecone_store.delete_document_vectors(
        document.case_id,
        document.id,
        chunk_ids=chunk_ids,
        index=index,
        settings=resolved,
    )

    try:
        delete_original(document.storage_path, client=client)
    except StorageError as exc:
        logger.warning(
            "original storage deletion failed",
            extra={
                "document_id": str(document.id),
                "error_class": type(exc).__name__,
            },
        )

    delete_document_row(client, document.id)
    logger.info(
        "delete job finished",
        extra={
            "document_id": str(document.id),
            "deleted_vector_count": deleted_vectors,
            "namespace": pinecone_store.namespace_for_case(document.case_id),
        },
    )
    return DeleteResult(
        status="deleted",
        deleted_vector_count=deleted_vectors,
    )
