"""Worker-scoped persistence helpers for durable processing jobs.

The API uses ownership-checked repositories. The worker is a trusted
service-role component that only operates on jobs and documents already
selected from Postgres, so these helpers deliberately do not accept a user id.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Sequence
from uuid import UUID

from supabase import Client

from app.db.repos import (
    Document,
    DocumentChunk,
    DocumentPage,
    DocumentUpdate,
    IndexStatus,
    ProcessingJob,
    RecordNotFound,
    RepositoryError,
    update_document,
)


def _rows(response: Any) -> list[dict[str, Any]]:
    """Normalize a Supabase response payload into a list of rows."""
    data = getattr(response, "data", None)
    if data is None:
        return []
    if isinstance(data, dict):
        return [data]
    return [row for row in data if isinstance(row, dict)]


def _first_row(response: Any) -> dict[str, Any] | None:
    """Return the first row from a Supabase response, if any."""
    rows = _rows(response)
    return rows[0] if rows else None


def _as_text(value: UUID | str) -> str:
    """Canonicalize a UUID-like value for PostgREST filters."""
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid UUID value") from exc


def _utc_now(value: datetime | None = None) -> datetime:
    """Return an aware UTC timestamp for scheduling decisions."""
    if value is None:
        return datetime.now(timezone.utc)
    if value.tzinfo is None:
        return value.replace(tzinfo=timezone.utc)
    return value.astimezone(timezone.utc)


def claim_next_job(
    client: Client,
    *,
    now: datetime | None = None,
    max_attempts: int = 3,
) -> ProcessingJob | None:
    """Claim one due job by flipping it from queued to running.

    The conditional update is the claim: if another worker already changed the
    row, the update returns no row and this worker tries the next candidate.
    Ingest, reindex, and delete jobs all share the durable queue.
    """
    if max_attempts < 1:
        raise ValueError("max_attempts must be at least 1")
    current = _utc_now(now)
    current_text = current.isoformat()

    response = (
        client.table("processing_jobs")
        .select("*")
        .eq("status", "queued")
        .lt("attempts", max_attempts)
        .lte("next_attempt_at", current_text)
        .order("next_attempt_at")
        .order("created_at")
        .limit(10)
        .execute()
    )

    for row in _rows(response):
        attempts = int(row.get("attempts") or 0) + 1
        claimed_response = (
            client.table("processing_jobs")
            .update(
                {
                    "status": "running",
                    "attempts": attempts,
                    "started_at": current_text,
                    "finished_at": None,
                    "error_code": None,
                    "error_summary": None,
                }
            )
            .eq("id", _as_text(row["id"]))
            .eq("status", "queued")
            .execute()
        )
        claimed = _first_row(claimed_response)
        if claimed is not None:
            return ProcessingJob.model_validate(claimed)
    return None


def mark_job_succeeded(
    client: Client,
    job_id: UUID | str,
    *,
    pipeline_version: str,
    now: datetime | None = None,
) -> ProcessingJob:
    """Mark a claimed job succeeded with the pipeline version that ran."""
    current_text = _utc_now(now).isoformat()
    response = (
        client.table("processing_jobs")
        .update(
            {
                "status": "succeeded",
                "pipeline_version": pipeline_version,
                "error_code": None,
                "error_summary": None,
                "finished_at": current_text,
            }
        )
        .eq("id", _as_text(job_id))
        .execute()
    )
    return _updated_job(response)


def schedule_job_retry(
    client: Client,
    job_id: UUID | str,
    *,
    attempts: int,
    next_attempt_at: datetime,
    error_code: str,
    now: datetime | None = None,
) -> ProcessingJob:
    """Return a failed attempt to the queue after a durable backoff."""
    if attempts < 1:
        raise ValueError("attempts must be at least 1")
    current_text = _utc_now(now).isoformat()
    response = (
        client.table("processing_jobs")
        .update(
            {
                "status": "queued",
                "attempts": attempts,
                "next_attempt_at": _utc_now(next_attempt_at).isoformat(),
                "error_code": error_code,
                "error_summary": "Processing attempt failed",
                "finished_at": None,
            }
        )
        .eq("id", _as_text(job_id))
        .execute()
    )
    return _updated_job(response)


def mark_job_failed(
    client: Client,
    job_id: UUID | str,
    *,
    error_code: str,
    now: datetime | None = None,
) -> ProcessingJob:
    """Mark a job failed after its final allowed attempt."""
    current_text = _utc_now(now).isoformat()
    response = (
        client.table("processing_jobs")
        .update(
            {
                "status": "failed",
                "error_code": error_code,
                "error_summary": "Processing failed",
                "finished_at": current_text,
            }
        )
        .eq("id", _as_text(job_id))
        .execute()
    )
    return _updated_job(response)


def mark_document_failed(
    client: Client,
    document_id: UUID | str,
    *,
    error_code: str,
) -> Document:
    """Mark a document failed without exposing internal exception details."""
    document = get_document_for_worker(client, document_id)
    quality_report = dict(document.quality_report or {})
    quality_report["error_code"] = error_code
    return update_document(
        client,
        document.owner_user_id,
        document.id,
        DocumentUpdate(
            processing_status="failed",
            quality_status="failed",
            quality_report=quality_report,
        ),
    )


def get_document_for_worker(client: Client, document_id: UUID | str) -> Document:
    """Load a document by id for a trusted worker operation."""
    response = (
        client.table("documents")
        .select("*")
        .eq("id", _as_text(document_id))
        .limit(1)
        .execute()
    )
    row = _first_row(response)
    if row is None:
        raise RecordNotFound("Document not found")
    return Document.model_validate(row)


def get_pages_for_document(
    client: Client, document_id: UUID | str
) -> list[DocumentPage]:
    """Return all extracted pages for a document in page order."""
    response = (
        client.table("document_pages")
        .select("*")
        .eq("document_id", _as_text(document_id))
        .order("page_number")
        .execute()
    )
    return [DocumentPage.model_validate(row) for row in _rows(response)]


def delete_pages_for_document(client: Client, document_id: UUID | str) -> None:
    """Delete derived page rows before rebuilding a document."""
    (
        client.table("document_pages")
        .delete()
        .eq("document_id", _as_text(document_id))
        .execute()
    )


def delete_chunks_for_document(client: Client, document_id: UUID | str) -> None:
    """Delete derived chunk rows before rebuilding a document."""
    (
        client.table("document_chunks")
        .delete()
        .eq("document_id", _as_text(document_id))
        .execute()
    )


def delete_document_row(client: Client, document_id: UUID | str) -> None:
    """Delete a document row; pages, chunks, and jobs cascade from it."""
    (
        client.table("documents")
        .delete()
        .eq("id", _as_text(document_id))
        .execute()
    )


def get_chunks_for_document(
    client: Client,
    document_id: UUID | str,
    *,
    index_statuses: Sequence[IndexStatus] | None = None,
    index_schema_version: int | None = None,
) -> list[DocumentChunk]:
    """Return chunks for a document in stable chunk order."""
    query = (
        client.table("document_chunks")
        .select("*")
        .eq("document_id", _as_text(document_id))
    )
    if index_statuses:
        query = query.in_("index_status", list(index_statuses))
    if index_schema_version is not None:
        query = query.eq("index_schema_version", index_schema_version)
    response = query.order("chunk_index").execute()
    return [DocumentChunk.model_validate(row) for row in _rows(response)]


def get_chunks_for_case(
    client: Client, case_id: UUID | str
) -> list[DocumentChunk]:
    """Return every chunk in a case for reconciliation."""
    response = (
        client.table("document_chunks")
        .select("*")
        .eq("case_id", _as_text(case_id))
        .order("document_id")
        .order("chunk_index")
        .execute()
    )
    return [DocumentChunk.model_validate(row) for row in _rows(response)]


def mark_chunks_indexed(
    client: Client,
    chunk_ids: Sequence[UUID | str],
    *,
    embedding_model: str,
) -> int:
    """Mark chunk rows indexed with the embedding model that produced them."""
    ids = [_as_text(chunk_id) for chunk_id in chunk_ids]
    for start in range(0, len(ids), 100):
        batch = ids[start : start + 100]
        if not batch:
            continue
        (
            client.table("document_chunks")
            .update(
                {
                    "index_status": "indexed",
                    "embedding_model": embedding_model,
                }
            )
            .in_("id", batch)
            .execute()
        )
    return len(ids)


def mark_chunks_stale(
    client: Client,
    document_id: UUID | str,
    *,
    embedding_model: str,
    index_schema_version: int,
) -> int:
    """Mark indexed chunks stale when the model or schema version changed."""
    chunks = get_chunks_for_document(client, document_id)
    stale_ids = [
        chunk.id
        for chunk in chunks
        if chunk.index_status == "indexed"
        and (
            chunk.embedding_model != embedding_model
            or chunk.index_schema_version != index_schema_version
        )
    ]
    for start in range(0, len(stale_ids), 100):
        batch = stale_ids[start : start + 100]
        if not batch:
            continue
        (
            client.table("document_chunks")
            .update({"index_status": "stale"})
            .in_("id", [_as_text(chunk_id) for chunk_id in batch])
            .execute()
        )
    return len(stale_ids)


def _updated_job(response: Any) -> ProcessingJob:
    """Require an updated job row."""
    row = _first_row(response)
    if row is None:
        raise RepositoryError("Job update returned no row")
    return ProcessingJob.model_validate(row)
