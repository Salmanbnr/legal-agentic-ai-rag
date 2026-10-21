"""Typed repositories for case data with explicit ownership checks.

Every function receives the authenticated user id from the API layer. Reads,
updates, and child-record inserts verify that the case belongs to that user in
backend code, even though the service-role client bypasses RLS.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal, Sequence
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from supabase import Client

CaseStatus = Literal["active", "archived"]
QualityStatus = Literal[
    "pending",
    "ready",
    "ready_with_warnings",
    "needs_review",
    "failed",
    "password_required",
]
ProcessingStatus = Literal[
    "uploaded",
    "validating",
    "extracting",
    "ocr",
    "chunking",
    "embedding",
    "indexing",
    "ready",
    "needs_review",
    "failed",
]
IndexStatus = Literal["pending", "indexed", "failed", "stale"]
ExtractionMethod = Literal["native", "ocr", "none"]
JobType = Literal["ingest", "reindex", "delete"]
JobStatus = Literal["queued", "running", "succeeded", "failed"]


class RepositoryError(RuntimeError):
    """Base class for repository failures."""


class CaseAccessDenied(RepositoryError):
    """Raised when a case is missing or owned by another user."""


class DocumentAccessDenied(RepositoryError):
    """Raised when a document is missing or not owned by the user's case."""


class RecordNotFound(RepositoryError):
    """Raised when a record required for an operation does not exist."""


class RowModel(BaseModel):
    """Base model for database rows, tolerant of extra generated columns."""

    model_config = ConfigDict(extra="ignore")


class Case(RowModel):
    """A legal case owned by one authenticated user."""

    id: UUID
    owner_user_id: UUID
    title: str
    description: str | None = None
    status: CaseStatus
    created_at: datetime
    updated_at: datetime


class CaseUpdate(BaseModel):
    """Mutable case fields accepted by the repository."""

    title: str | None = None
    description: str | None = None
    status: CaseStatus | None = None


class Document(RowModel):
    """An uploaded original document and its processing metadata."""

    id: UUID
    case_id: UUID
    owner_user_id: UUID
    storage_path: str
    original_filename: str
    detected_mime: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    page_count: int | None = None
    document_type: str | None = None
    version_group_id: UUID | None = None
    duplicate_of: UUID | None = None
    quality_status: QualityStatus
    quality_report: dict[str, Any] = Field(default_factory=dict)
    processing_status: ProcessingStatus
    parser_version: str | None = None
    created_at: datetime
    updated_at: datetime


class DocumentPage(RowModel):
    """One extracted page with provenance and transformation metadata."""

    id: UUID
    case_id: UUID
    document_id: UUID
    page_number: int
    extraction_method: ExtractionMethod
    text_density: float | None = None
    ocr_confidence: float | None = None
    raw_text: str | None = None
    clean_text: str | None = None
    quality_flags: list[Any] = Field(default_factory=list)
    transform_log: list[Any] = Field(default_factory=list)
    created_at: datetime


class DocumentChunk(RowModel):
    """A citation-ready chunk of text derived from a document page range."""

    id: UUID
    case_id: UUID
    document_id: UUID
    document_version_id: UUID | None = None
    chunk_index: int
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    text_content: str
    content_hash: str
    source_anchor: str | None = None
    embedding_model: str | None = None
    index_schema_version: int
    index_status: IndexStatus
    quality_flags: list[Any] = Field(default_factory=list)
    created_at: datetime


class ProcessingJob(RowModel):
    """An idempotent ingestion, reindex, or delete job."""

    id: UUID
    case_id: UUID
    document_id: UUID
    job_type: JobType
    idempotency_key: str
    status: JobStatus
    attempts: int
    error_code: str | None = None
    error_summary: str | None = None
    pipeline_version: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    next_attempt_at: datetime | None = None
    created_at: datetime


class AuditEvent(RowModel):
    """A safe, append-only audit record."""

    id: UUID
    case_id: UUID | None = None
    actor_user_id: UUID | None = None
    event_type: str
    target_type: str | None = None
    target_id: str | None = None
    request_id: str | None = None
    safe_metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime


class Answer(RowModel):
    """A stored answer with its evidence manifest."""

    id: UUID
    case_id: UUID
    user_id: UUID
    question: str
    answer_payload: dict[str, Any] = Field(default_factory=dict)
    evidence_refs: list[Any] = Field(default_factory=list)
    graph_version: str | None = None
    final_status: str | None = None
    created_at: datetime


class DocumentCreate(BaseModel):
    """Fields accepted when registering an uploaded document."""

    storage_path: str = Field(min_length=1)
    original_filename: str = Field(min_length=1)
    detected_mime: str | None = None
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    page_count: int | None = Field(default=None, ge=0)
    document_type: str | None = None
    version_group_id: UUID | None = None
    duplicate_of: UUID | None = None
    quality_status: QualityStatus | None = None
    quality_report: dict[str, Any] | None = None
    processing_status: ProcessingStatus | None = None


class DocumentUpdate(BaseModel):
    """Mutable document metadata and processing fields."""

    detected_mime: str | None = None
    sha256: str | None = None
    size_bytes: int | None = Field(default=None, ge=0)
    page_count: int | None = Field(default=None, ge=0)
    document_type: str | None = None
    version_group_id: UUID | None = None
    duplicate_of: UUID | None = None
    quality_status: QualityStatus | None = None
    quality_report: dict[str, Any] | None = None
    processing_status: ProcessingStatus | None = None
    parser_version: str | None = None


class PageCreate(BaseModel):
    """Fields accepted for one extracted page."""

    page_number: int = Field(ge=1)
    extraction_method: ExtractionMethod
    text_density: float | None = Field(default=None, ge=0)
    ocr_confidence: float | None = Field(default=None, ge=0)
    raw_text: str | None = None
    clean_text: str | None = None
    quality_flags: list[Any] = Field(default_factory=list)
    transform_log: list[Any] = Field(default_factory=list)


class ChunkCreate(BaseModel):
    """Fields accepted for one citation-ready chunk."""

    chunk_index: int = Field(ge=0)
    page_start: int | None = Field(default=None, ge=1)
    page_end: int | None = Field(default=None, ge=1)
    section_path: list[str] = Field(default_factory=list)
    text_content: str = Field(min_length=1)
    content_hash: str = Field(min_length=1)
    source_anchor: str | None = None
    embedding_model: str | None = None
    index_schema_version: int = Field(default=1, ge=1)
    index_status: IndexStatus = "pending"
    quality_flags: list[Any] = Field(default_factory=list)


class JobCreate(BaseModel):
    """Fields accepted when enqueueing a processing job."""

    job_type: JobType
    idempotency_key: str = Field(min_length=1)
    pipeline_version: str | None = None


class AnswerCreate(BaseModel):
    """Fields accepted when storing an answer."""

    question: str = Field(min_length=1, max_length=20000)
    answer_payload: dict[str, Any]
    evidence_refs: list[Any] = Field(default_factory=list)
    graph_version: str | None = None
    final_status: str | None = None


class JobUpdate(BaseModel):
    """Mutable job state and sanitized failure metadata."""

    status: JobStatus | None = None
    attempts: int | None = Field(default=None, ge=0)
    error_code: str | None = None
    error_summary: str | None = None
    pipeline_version: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    next_attempt_at: datetime | None = None


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
    """Canonicalize a UUID-like value for PostgREST filters and payloads."""
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid UUID value") from exc


def _validate_sha256(value: str) -> str:
    """Require a lowercase 64-character SHA-256 digest."""
    if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError("Invalid SHA-256 digest")
    return value


def _pagination(limit: int, offset: int) -> tuple[int, int]:
    """Validate and convert limit/offset into an inclusive PostgREST range."""
    if not 1 <= limit <= 200:
        raise ValueError("limit must be between 1 and 200")
    if offset < 0:
        raise ValueError("offset must be non-negative")
    return offset, offset + limit - 1


def _require_case_row(
    client: Client, user_id: UUID | str, case_id: UUID | str
) -> dict[str, Any]:
    """Load a case and reject it unless the user explicitly owns it."""
    response = (
        client.table("cases")
        .select("*")
        .eq("id", _as_text(case_id))
        .limit(1)
        .execute()
    )
    row = _first_row(response)
    owner_id = row.get("owner_user_id") if row is not None else None
    if owner_id is None or _as_text(owner_id) != _as_text(user_id):
        raise CaseAccessDenied("Case not found or access denied")
    return row


def _require_document_row(
    client: Client,
    user_id: UUID | str,
    document_id: UUID | str,
    case_id: UUID | str | None = None,
) -> dict[str, Any]:
    """Load a document and verify its owner and case ownership."""
    response = (
        client.table("documents")
        .select("*")
        .eq("id", _as_text(document_id))
        .limit(1)
        .execute()
    )
    row = _first_row(response)
    owner_id = row.get("owner_user_id") if row is not None else None
    if owner_id is None or _as_text(owner_id) != _as_text(user_id):
        raise DocumentAccessDenied("Document not found or access denied")
    if case_id is not None and _as_text(row.get("case_id")) != _as_text(case_id):
        raise DocumentAccessDenied("Document not found or access denied")
    _require_case_row(client, user_id, row["case_id"])
    return row


def _require_job_row(
    client: Client, user_id: UUID | str, job_id: UUID | str
) -> dict[str, Any]:
    """Load a job and verify the user owns its case."""
    response = (
        client.table("processing_jobs")
        .select("*")
        .eq("id", _as_text(job_id))
        .limit(1)
        .execute()
    )
    row = _first_row(response)
    if row is None:
        raise RecordNotFound("Job not found or access denied")
    _require_case_row(client, user_id, row["case_id"])
    return row


def get_job_for_user(
    client: Client, user_id: UUID | str, job_id: UUID | str
) -> ProcessingJob:
    """Return a job only when the user owns the job's case.

    Raises:
        RecordNotFound: If the job is missing or belongs to another user.
    """
    return ProcessingJob.model_validate(_require_job_row(client, user_id, job_id))


def find_job_by_idempotency_key(
    client: Client, user_id: UUID | str, idempotency_key: str
) -> ProcessingJob | None:
    """Return an owned job for an idempotency key, if it already exists."""
    if not idempotency_key:
        raise ValueError("Idempotency key must not be empty")
    response = (
        client.table("processing_jobs")
        .select("*")
        .eq("idempotency_key", idempotency_key)
        .limit(1)
        .execute()
    )
    row = _first_row(response)
    if row is None:
        return None
    _require_case_row(client, user_id, row["case_id"])
    return ProcessingJob.model_validate(row)


def _inserted_row(response: Any, label: str) -> dict[str, Any]:
    """Require an inserted row so callers never receive a silent no-op."""
    row = _first_row(response)
    if row is None:
        raise RepositoryError(f"{label} insert returned no row")
    return row


def create_case(
    client: Client,
    user_id: UUID | str,
    *,
    title: str,
    description: str | None = None,
    status: CaseStatus = "active",
) -> Case:
    """Create a case owned by the authenticated user."""
    clean_title = title.strip()
    if not clean_title:
        raise ValueError("Case title must not be empty")
    response = (
        client.table("cases")
        .insert(
            {
                "owner_user_id": _as_text(user_id),
                "title": clean_title,
                "description": description,
                "status": status,
            }
        )
        .execute()
    )
    return Case.model_validate(_inserted_row(response, "Case"))


def get_case_for_user(
    client: Client, user_id: UUID | str, case_id: UUID | str
) -> Case:
    """Return a case only when it is owned by the authenticated user.

    Raises:
        CaseAccessDenied: If the case is missing or owned by another user.
    """
    return Case.model_validate(_require_case_row(client, user_id, case_id))


def list_cases(
    client: Client,
    user_id: UUID | str,
    *,
    status: CaseStatus | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[Case]:
    """List only the authenticated user's cases."""
    start, end = _pagination(limit, offset)
    query = client.table("cases").select("*").eq(
        "owner_user_id", _as_text(user_id)
    )
    if status is not None:
        query = query.eq("status", status)
    response = query.order("updated_at", desc=True).range(start, end).execute()
    return [Case.model_validate(row) for row in _rows(response)]


def update_case(
    client: Client,
    user_id: UUID | str,
    case_id: UUID | str,
    update: CaseUpdate,
) -> Case:
    """Update a case after verifying the authenticated user owns it."""
    _require_case_row(client, user_id, case_id)
    payload = update.model_dump(mode="json", exclude_unset=True)
    if not payload:
        raise ValueError("Case update must include at least one field")
    if "title" in payload:
        title = payload["title"]
        if title is None or not str(title).strip():
            raise ValueError("Case title must not be empty")
        payload["title"] = str(title).strip()
    if "status" in payload and payload["status"] is None:
        raise ValueError("Case status must not be null")
    response = (
        client.table("cases")
        .update(payload)
        .eq("id", _as_text(case_id))
        .eq("owner_user_id", _as_text(user_id))
        .execute()
    )
    return Case.model_validate(_inserted_row(response, "Case"))


def create_document(
    client: Client,
    user_id: UUID | str,
    case_id: UUID | str,
    data: DocumentCreate,
) -> Document:
    """Register a document after verifying the user owns the case."""
    _require_case_row(client, user_id, case_id)
    payload = {
        "case_id": _as_text(case_id),
        "owner_user_id": _as_text(user_id),
        **data.model_dump(mode="json", exclude_none=True),
    }
    response = client.table("documents").insert(payload).execute()
    return Document.model_validate(_inserted_row(response, "Document"))


def get_document_for_user(
    client: Client, user_id: UUID | str, document_id: UUID | str
) -> Document:
    """Return a document only when the user owns its case.

    Raises:
        DocumentAccessDenied: If the document is missing or not owned by the
            authenticated user's case.
    """
    return Document.model_validate(_require_document_row(client, user_id, document_id))


def find_document_by_sha256(
    client: Client,
    user_id: UUID | str,
    case_id: UUID | str,
    sha256: str,
) -> Document | None:
    """Find an active document with the same digest in an owned case."""
    _require_case_row(client, user_id, case_id)
    digest = _validate_sha256(sha256)
    response = (
        client.table("documents")
        .select("*")
        .eq("case_id", _as_text(case_id))
        .eq("owner_user_id", _as_text(user_id))
        .eq("sha256", digest)
        .order("created_at")
        .limit(20)
        .execute()
    )
    for row in _rows(response):
        quality_report = row.get("quality_report") or {}
        if not quality_report.get("deletion_requested"):
            return Document.model_validate(row)
    return None


def update_document(
    client: Client,
    user_id: UUID | str,
    document_id: UUID | str,
    update: DocumentUpdate,
) -> Document:
    """Update document metadata after verifying document and case ownership."""
    document = _require_document_row(client, user_id, document_id)
    payload = update.model_dump(mode="json", exclude_unset=True)
    if not payload:
        raise ValueError("Document update must include at least one field")
    response = (
        client.table("documents")
        .update(payload)
        .eq("id", _as_text(document_id))
        .eq("owner_user_id", _as_text(user_id))
        .eq("case_id", _as_text(document["case_id"]))
        .execute()
    )
    return Document.model_validate(_inserted_row(response, "Document"))


def list_documents(
    client: Client,
    user_id: UUID | str,
    case_id: UUID | str,
    *,
    processing_status: ProcessingStatus | None = None,
    limit: int = 100,
    offset: int = 0,
) -> list[Document]:
    """List documents for a case after verifying case ownership."""
    _require_case_row(client, user_id, case_id)
    start, end = _pagination(limit, offset)
    query = (
        client.table("documents")
        .select("*")
        .eq("case_id", _as_text(case_id))
        .eq("owner_user_id", _as_text(user_id))
    )
    if processing_status is not None:
        query = query.eq("processing_status", processing_status)
    response = query.order("created_at", desc=True).range(start, end).execute()
    return [Document.model_validate(row) for row in _rows(response)]


def get_chunks_by_ids(
    client: Client,
    user_id: UUID | str,
    case_id: UUID | str,
    chunk_ids: Sequence[UUID | str],
) -> list[DocumentChunk]:
    """Return case chunks by id after verifying case ownership."""
    _require_case_row(client, user_id, case_id)
    ids = [_as_text(chunk_id) for chunk_id in chunk_ids]
    if not ids:
        return []
    response = (
        client.table("document_chunks")
        .select("*")
        .eq("case_id", _as_text(case_id))
        .in_("id", ids)
        .execute()
    )
    return [DocumentChunk.model_validate(row) for row in _rows(response)]


def get_pages_for_documents(
    client: Client,
    user_id: UUID | str,
    case_id: UUID | str,
    document_ids: Sequence[UUID | str],
) -> list[DocumentPage]:
    """Return the pages of case documents after verifying case ownership."""
    _require_case_row(client, user_id, case_id)
    ids = [_as_text(document_id) for document_id in document_ids]
    if not ids:
        return []
    response = (
        client.table("document_pages")
        .select("*")
        .eq("case_id", _as_text(case_id))
        .in_("document_id", ids)
        .order("page_number")
        .execute()
    )
    return [DocumentPage.model_validate(row) for row in _rows(response)]


def insert_pages(
    client: Client,
    user_id: UUID | str,
    case_id: UUID | str,
    document_id: UUID | str,
    pages: list[PageCreate],
) -> list[DocumentPage]:
    """Insert extracted pages after verifying document and case ownership."""
    if not pages:
        return []
    _require_document_row(client, user_id, document_id, case_id)
    payload = [
        {
            "case_id": _as_text(case_id),
            "document_id": _as_text(document_id),
            **page.model_dump(mode="json", exclude_none=True),
        }
        for page in pages
    ]
    response = client.table("document_pages").insert(payload).execute()
    rows = _rows(response)
    if len(rows) != len(payload):
        raise RepositoryError("Document page insert returned an unexpected row count")
    return [DocumentPage.model_validate(row) for row in rows]


def update_page(
    client: Client,
    user_id: UUID | str,
    case_id: UUID | str,
    document_id: UUID | str,
    page: PageCreate,
) -> DocumentPage:
    """Replace one extracted page after verifying document ownership."""
    document = _require_document_row(client, user_id, document_id, case_id)
    payload = page.model_dump(mode="json", exclude_none=False)
    payload.pop("page_number", None)
    response = (
        client.table("document_pages")
        .update(payload)
        .eq("document_id", _as_text(document_id))
        .eq("page_number", page.page_number)
        .execute()
    )
    row = _first_row(response)
    if row is None:
        raise RecordNotFound("Document page not found")
    row_case_id = row.get("case_id")
    if row_case_id is None or _as_text(row_case_id) != _as_text(document["case_id"]):
        raise DocumentAccessDenied("Document page not found or access denied")
    return DocumentPage.model_validate(row)


def get_neighbor_chunks(
    client: Client,
    user_id: UUID | str,
    chunk_id: UUID | str,
    *,
    before: int = 1,
    after: int = 1,
) -> list[DocumentChunk]:
    """Return a chunk plus nearby chunks from the same document and index.

    The target chunk is resolved from the database first, then its document and
    case ownership are verified against the authenticated user. Neighbor lookup
    never crosses a document or an index schema version.
    """
    if before < 0 or after < 0:
        raise ValueError("before and after must be non-negative")
    response = (
        client.table("document_chunks")
        .select("*")
        .eq("id", _as_text(chunk_id))
        .limit(1)
        .execute()
    )
    target_row = _first_row(response)
    if target_row is None:
        raise RecordNotFound("Chunk not found or access denied")
    target = DocumentChunk.model_validate(target_row)
    _require_document_row(
        client, user_id, target.document_id, target.case_id
    )

    start = max(0, target.chunk_index - before)
    end = target.chunk_index + after
    response = (
        client.table("document_chunks")
        .select("*")
        .eq("document_id", _as_text(target.document_id))
        .eq("case_id", _as_text(target.case_id))
        .eq("index_schema_version", target.index_schema_version)
        .gte("chunk_index", start)
        .lte("chunk_index", end)
        .order("chunk_index")
        .execute()
    )
    return [DocumentChunk.model_validate(row) for row in _rows(response)]


def insert_chunks(
    client: Client,
    user_id: UUID | str,
    case_id: UUID | str,
    document_id: UUID | str,
    chunks: list[ChunkCreate],
) -> list[DocumentChunk]:
    """Insert citation-ready chunks after verifying ownership."""
    if not chunks:
        return []
    _require_document_row(client, user_id, document_id, case_id)
    payload = [
        {
            "case_id": _as_text(case_id),
            "document_id": _as_text(document_id),
            **chunk.model_dump(mode="json", exclude_none=True),
        }
        for chunk in chunks
    ]
    response = client.table("document_chunks").insert(payload).execute()
    rows = _rows(response)
    if len(rows) != len(payload):
        raise RepositoryError("Document chunk insert returned an unexpected row count")
    return [DocumentChunk.model_validate(row) for row in rows]


def create_job(
    client: Client,
    user_id: UUID | str,
    case_id: UUID | str,
    document_id: UUID | str,
    data: JobCreate,
) -> ProcessingJob:
    """Create an idempotent processing job after verifying ownership."""
    _require_document_row(client, user_id, document_id, case_id)
    payload = {
        "case_id": _as_text(case_id),
        "document_id": _as_text(document_id),
        **data.model_dump(mode="json", exclude_none=True),
    }
    response = client.table("processing_jobs").insert(payload).execute()
    return ProcessingJob.model_validate(_inserted_row(response, "Job"))


def update_job(
    client: Client,
    user_id: UUID | str,
    job_id: UUID | str,
    update: JobUpdate,
) -> ProcessingJob:
    """Update job state after verifying the user owns the job's case."""
    job = _require_job_row(client, user_id, job_id)
    payload = update.model_dump(mode="json", exclude_unset=True)
    if not payload:
        raise ValueError("Job update must include at least one field")
    response = (
        client.table("processing_jobs")
        .update(payload)
        .eq("id", _as_text(job_id))
        .eq("case_id", _as_text(job["case_id"]))
        .execute()
    )
    return ProcessingJob.model_validate(_inserted_row(response, "Job"))


def write_audit_event(
    client: Client,
    user_id: UUID | str,
    *,
    event_type: str,
    case_id: UUID | str | None = None,
    target_type: str | None = None,
    target_id: str | None = None,
    request_id: str | None = None,
    safe_metadata: dict[str, Any] | None = None,
) -> AuditEvent:
    """Append a safe audit event, verifying case ownership when linked.

    `safe_metadata` must never contain document text, prompts, completions,
    credentials, or tokens.
    """
    if case_id is not None:
        _require_case_row(client, user_id, case_id)
    payload = {
        "case_id": _as_text(case_id) if case_id is not None else None,
        "actor_user_id": _as_text(user_id),
        "event_type": event_type,
        "target_type": target_type,
        "target_id": target_id,
        "request_id": request_id,
        "safe_metadata": dict(safe_metadata or {}),
    }
    response = client.table("audit_events").insert(payload).execute()
    return AuditEvent.model_validate(_inserted_row(response, "Audit event"))


def list_answers(
    client: Client,
    user_id: UUID | str,
    case_id: UUID | str,
    *,
    limit: int = 20,
    offset: int = 0,
) -> list[Answer]:
    """List stored answers for a case, newest first, after ownership check."""
    _require_case_row(client, user_id, case_id)
    start, end = _pagination(limit, offset)
    response = (
        client.table("answers")
        .select("*")
        .eq("case_id", _as_text(case_id))
        .eq("user_id", _as_text(user_id))
        .order("created_at", desc=True)
        .range(start, end)
        .execute()
    )
    return [Answer.model_validate(row) for row in _rows(response)]


def insert_answer(
    client: Client,
    user_id: UUID | str,
    case_id: UUID | str,
    data: AnswerCreate,
) -> Answer:
    """Store an answer and its evidence manifest after verifying ownership.

    The evidence manifest holds evidence ids and page ranges only. Document
    text, prompts, and completions stay out of the manifest.
    """
    _require_case_row(client, user_id, case_id)
    payload = {
        "case_id": _as_text(case_id),
        "user_id": _as_text(user_id),
        "question": data.question,
        "answer_payload": data.answer_payload,
        "evidence_refs": list(data.evidence_refs),
        "graph_version": data.graph_version,
        "final_status": data.final_status,
    }
    response = client.table("answers").insert(payload).execute()
    return Answer.model_validate(_inserted_row(response, "Answer"))
