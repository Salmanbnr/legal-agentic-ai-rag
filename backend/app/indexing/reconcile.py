"""Compare Postgres chunk ids with Pinecone vector ids for one case."""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from uuid import UUID

from supabase import Client

from app.core.config import Settings, get_settings
from app.db.repos import Document, JobCreate, create_job, find_job_by_idempotency_key
from app.indexing import pinecone_store
from app.indexing.keys import reindex_idempotency_key
from app.jobs.store import get_chunks_for_case, get_document_for_worker

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class ReconcileReport:
    """Orphaned and missing vector ids for one case."""

    case_id: UUID
    postgres_chunk_count: int
    pinecone_vector_count: int
    orphaned_ids: list[str] = field(default_factory=list)
    missing_ids: list[str] = field(default_factory=list)

    @property
    def is_clean(self) -> bool:
        """Return whether the case index matches Postgres."""
        return not self.orphaned_ids and not self.missing_ids


@dataclass(frozen=True, slots=True)
class ReconcileFixResult:
    """Actions taken while repairing one case."""

    report: ReconcileReport
    deleted_orphan_count: int
    requeued_document_ids: list[str] = field(default_factory=list)


def reconcile_case(
    client: Client,
    case_id: UUID | str,
    *,
    settings: Settings | None = None,
    index: object | None = None,
) -> ReconcileReport:
    """Report vectors that have no chunk and indexed chunks with no vector."""
    resolved = settings or get_settings()
    normalized_case = pinecone_store.namespace_for_case(case_id)
    chunks = get_chunks_for_case(client, normalized_case)
    postgres_ids = {str(chunk.id) for chunk in chunks}
    indexed_ids = {
        str(chunk.id)
        for chunk in chunks
        if chunk.index_status == "indexed"
    }

    resolved_index = index or pinecone_store.get_index(resolved)
    pinecone_ids = pinecone_store.list_vector_ids(
        resolved_index, normalized_case
    )

    orphaned = sorted(pinecone_ids - postgres_ids)
    missing = sorted(indexed_ids - pinecone_ids)
    logger.info(
        "case index reconciliation finished",
        extra={
            "case_id": normalized_case,
            "postgres_chunk_count": len(postgres_ids),
            "pinecone_vector_count": len(pinecone_ids),
            "orphaned_count": len(orphaned),
            "missing_count": len(missing),
        },
    )
    return ReconcileReport(
        case_id=UUID(normalized_case),
        postgres_chunk_count=len(postgres_ids),
        pinecone_vector_count=len(pinecone_ids),
        orphaned_ids=orphaned,
        missing_ids=missing,
    )


def _ensure_reindex_job(
    client: Client, document: Document, settings: Settings
) -> str:
    """Create or return the current reindex job for a document."""
    key = reindex_idempotency_key(document.id, settings)
    existing = find_job_by_idempotency_key(client, document.owner_user_id, key)
    if existing is not None:
        return str(existing.id)
    job = create_job(
        client,
        document.owner_user_id,
        document.case_id,
        document.id,
        JobCreate(
            job_type="reindex",
            idempotency_key=key,
            pipeline_version=settings.ingest_pipeline_version,
        ),
    )
    return str(job.id)


def fix_reconcile(
    client: Client,
    case_id: UUID | str,
    *,
    settings: Settings | None = None,
    index: object | None = None,
) -> ReconcileFixResult:
    """Delete orphaned vectors and queue a reindex for missing chunks."""
    resolved = settings or get_settings()
    report = reconcile_case(
        client, case_id, settings=resolved, index=index
    )
    resolved_index = index or pinecone_store.get_index(resolved)
    namespace = pinecone_store.namespace_for_case(case_id)

    deleted = 0
    if report.orphaned_ids:
        deleted = pinecone_store.delete_vectors(
            resolved_index,
            namespace,
            report.orphaned_ids,
            settings=resolved,
        )

    requeued: list[str] = []
    if report.missing_ids:
        missing = set(report.missing_ids)
        chunks = get_chunks_for_case(client, namespace)
        documents: dict[str, UUID] = {
            str(chunk.document_id): chunk.document_id
            for chunk in chunks
            if str(chunk.id) in missing
        }
        for document_id in documents.values():
            document = get_document_for_worker(client, document_id)
            _ensure_reindex_job(client, document, resolved)
            requeued.append(str(document.id))

    logger.info(
        "case index reconciliation fixed",
        extra={
            "case_id": namespace,
            "deleted_orphan_count": deleted,
            "requeued_document_count": len(requeued),
        },
    )
    return ReconcileFixResult(
        report=report,
        deleted_orphan_count=deleted,
        requeued_document_ids=sorted(requeued),
    )
