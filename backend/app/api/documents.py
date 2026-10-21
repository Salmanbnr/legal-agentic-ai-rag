"""Secure document upload, listing, retrieval, and deletion requests."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any, Literal
from uuid import UUID

from fastapi import (
    APIRouter,
    Depends,
    File,
    HTTPException,
    Query,
    Request,
    UploadFile,
    status,
)
from pydantic import BaseModel, Field
from supabase import Client

from app.auth.access import (
    require_case_access,
    require_document_access,
    require_job_access,
)
from app.auth.dependencies import CurrentUser, get_current_user
from app.core.config import Settings, get_settings
from app.core.rate_limit import InMemoryRateLimiter, RateLimitExceeded
from app.db.client import get_db_client
from app.db.repos import (
    Document,
    DocumentCreate,
    DocumentUpdate,
    JobCreate,
    JobStatus,
    JobType,
    ProcessingJob,
    ProcessingStatus,
    QualityStatus,
    create_document,
    create_job,
    find_document_by_sha256,
    find_job_by_idempotency_key,
    list_documents,
    update_document,
    write_audit_event,
)
from app.ingestion.fingerprint import compute_fingerprint
from app.ingestion.validation import UploadErrorCode, validate_upload
from app.indexing.keys import reindex_idempotency_key
from app.storage.supabase_storage import (
    StorageError,
    create_signed_url,
    delete_original,
    upload_original,
)

router = APIRouter(tags=["documents"])

logger = logging.getLogger(__name__)

#: The signed URL lifetime for opening one original, in seconds.
SIGNED_URL_SECONDS = 60

_UPLOAD_ERRORS: dict[UploadErrorCode, tuple[int, str]] = {
    "UNSUPPORTED_FILE": (
        status.HTTP_415_UNSUPPORTED_MEDIA_TYPE,
        "Unsupported or invalid file type",
    ),
    "FILE_TOO_LARGE": (
        status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
        "File exceeds the maximum upload size",
    ),
    "PROCESSING_FAILED": (
        status.HTTP_422_UNPROCESSABLE_ENTITY,
        "File could not be processed",
    ),
}


class QualityReportSummary(BaseModel):
    """The safe, countable quality facts a document owner may see.

    The report holds page numbers, counts, and error codes only. It never holds
    document text.
    """

    pages_total: int | None = None
    pages_extracted: int | None = None
    pages_empty: int | None = None
    empty_pages: list[int] = Field(default_factory=list)
    low_confidence_pages: list[int] = Field(default_factory=list)
    repeated_text_ratio: float | None = None
    parser_errors: list[dict[str, Any]] = Field(default_factory=list)


class DocumentResponse(BaseModel):
    """Safe document metadata; storage paths remain server-side."""

    id: UUID
    case_id: UUID
    original_filename: str
    detected_mime: str | None = None
    sha256: str | None = None
    size_bytes: int | None = None
    page_count: int | None = None
    document_type: str | None = None
    duplicate_of: UUID | None = None
    quality_status: QualityStatus
    processing_status: ProcessingStatus
    deletion_requested: bool = False
    quality_report: QualityReportSummary | None = None
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_document(cls, document: Document) -> "DocumentResponse":
        """Build a response without exposing internal quality reports or paths."""
        return cls(
            id=document.id,
            case_id=document.case_id,
            original_filename=document.original_filename,
            detected_mime=document.detected_mime,
            sha256=document.sha256,
            size_bytes=document.size_bytes,
            page_count=document.page_count,
            document_type=document.document_type,
            duplicate_of=document.duplicate_of,
            quality_status=document.quality_status,
            processing_status=document.processing_status,
            deletion_requested=bool(
                (document.quality_report or {}).get("deletion_requested")
            ),
            quality_report=_quality_summary(document),
            created_at=document.created_at,
            updated_at=document.updated_at,
        )


def _quality_summary(document: Document) -> QualityReportSummary | None:
    """Return the countable quality facts of a document, if there are any."""
    report = document.quality_report or {}
    if not report:
        return None
    return QualityReportSummary(
        pages_total=report.get("pages_total"),
        pages_extracted=report.get("pages_extracted"),
        pages_empty=report.get("pages_empty"),
        empty_pages=list(report.get("empty_pages") or []),
        low_confidence_pages=list(report.get("low_confidence_pages") or []),
        repeated_text_ratio=report.get("repeated_text_ratio"),
        parser_errors=list(report.get("parser_errors") or []),
    )


class JobResponse(BaseModel):
    """Safe processing-job state without internal failure summaries."""

    id: UUID
    case_id: UUID
    document_id: UUID
    job_type: JobType
    status: JobStatus
    attempts: int
    error_code: str | None = None
    pipeline_version: str | None = None
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime

    @classmethod
    def from_job(cls, job: ProcessingJob) -> "JobResponse":
        """Build a safe job response."""
        return cls(
            id=job.id,
            case_id=job.case_id,
            document_id=job.document_id,
            job_type=job.job_type,
            status=job.status,
            attempts=job.attempts,
            error_code=job.error_code,
            pipeline_version=job.pipeline_version,
            started_at=job.started_at,
            finished_at=job.finished_at,
            created_at=job.created_at,
        )


class UploadResponse(BaseModel):
    """Identifiers and validation state returned after an upload."""

    document_id: UUID
    job_id: UUID | None = None
    quality_status: QualityStatus
    duplicate_of: UUID | None = None
    error_code: UploadErrorCode | None = None


class DeleteResponse(BaseModel):
    """Acknowledgement that deletion was requested and queued."""

    document_id: UUID
    job_id: UUID
    deletion_requested: bool = True


class ReindexResponse(BaseModel):
    """Acknowledgement that a reindex job was requested."""

    document_id: UUID
    job_id: UUID
    job_type: JobType = "reindex"


class SignedUrlResponse(BaseModel):
    """A short-lived private download URL for one original document."""

    document_id: UUID
    filename: str
    mime: str | None = None
    url: str
    expires_in_seconds: int = SIGNED_URL_SECONDS


def _request_id(request: Request) -> str | None:
    """Read the middleware-assigned request id."""
    return getattr(request.state, "request_id", None)


def _read_upload_bytes(file: UploadFile) -> bytes:
    """Read validated upload bytes into memory for storage and hashing."""
    try:
        file.file.seek(0)
        data = file.file.read()
        file.file.seek(0)
    except (OSError, ValueError) as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File could not be processed",
        ) from exc
    if not isinstance(data, bytes) or not data:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="File could not be processed",
        )
    return data


def _raise_upload_error(error_code: UploadErrorCode) -> None:
    """Map a validation error code to a safe HTTP error."""
    status_code, detail = _UPLOAD_ERRORS.get(
        error_code,
        (status.HTTP_422_UNPROCESSABLE_ENTITY, "File could not be processed"),
    )
    raise HTTPException(status_code=status_code, detail=detail)


def _enforce_upload_rate_limit(request: Request, user: CurrentUser) -> None:
    """Reject uploads that exceed the app's per-user in-memory allowance."""
    limiter = getattr(request.app.state, "upload_rate_limiter", None)
    if not isinstance(limiter, InMemoryRateLimiter):
        return
    try:
        limiter.check(str(user.id))
    except RateLimitExceeded as exc:
        raise HTTPException(
            status_code=status.HTTP_429_TOO_MANY_REQUESTS,
            detail="Too many uploads. Try again later.",
            headers={"Retry-After": str(exc.retry_after)},
        ) from None


def _write_upload_audit(
    client: Client,
    user: CurrentUser,
    request: Request,
    document: Document,
    job: ProcessingJob | None,
) -> None:
    """Append a safe audit event for an accepted upload."""
    write_audit_event(
        client,
        user.id,
        event_type="document.uploaded",
        case_id=document.case_id,
        target_type="document",
        target_id=str(document.id),
        request_id=_request_id(request),
        safe_metadata={
            "detected_mime": document.detected_mime,
            "size_bytes": document.size_bytes,
            "page_count": document.page_count,
            "quality_status": document.quality_status,
            "duplicate_of": str(document.duplicate_of)
            if document.duplicate_of
            else None,
            "job_id": str(job.id) if job is not None else None,
        },
    )


@router.post(
    "/cases/{case_id}/documents",
    response_model=UploadResponse,
    status_code=status.HTTP_201_CREATED,
)
def upload_document(
    case_id: UUID,
    request: Request,
    file: Annotated[UploadFile, File()],
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> UploadResponse:
    """Validate and store one original, then enqueue ingestion."""
    require_case_access(case_id, user, client=client)
    _enforce_upload_rate_limit(request, user)

    validation = validate_upload(file, settings=settings)
    if validation.error_code and validation.error_code != "PASSWORD_REQUIRED":
        _raise_upload_error(validation.error_code)

    content = _read_upload_bytes(file)
    fingerprint = compute_fingerprint(content)
    existing = find_document_by_sha256(
        client, user.id, case_id, fingerprint.sha256
    )
    duplicate_of = (
        (existing.duplicate_of or existing.id)
        if existing is not None
        else None
    )

    storage_path = upload_original(
        user.id,
        case_id,
        content,
        content_type=validation.detected_mime or "application/octet-stream",
    )
    encrypted = validation.error_code == "PASSWORD_REQUIRED"
    try:
        document = create_document(
            client,
            user.id,
            case_id,
            DocumentCreate(
                storage_path=storage_path,
                original_filename=validation.original_filename,
                detected_mime=validation.detected_mime,
                sha256=fingerprint.sha256,
                size_bytes=fingerprint.size_bytes,
                page_count=validation.page_count,
                duplicate_of=duplicate_of,
                quality_status="password_required" if encrypted else None,
                quality_report={"error_code": "PASSWORD_REQUIRED"}
                if encrypted
                else None,
            ),
        )
    except Exception:
        try:
            delete_original(storage_path)
        except Exception:
            pass
        raise

    if encrypted:
        _write_upload_audit(client, user, request, document, None)
        return UploadResponse(
            document_id=document.id,
            job_id=None,
            quality_status=document.quality_status,
            duplicate_of=document.duplicate_of,
            error_code="PASSWORD_REQUIRED",
        )

    idempotency_key = f"{document.id}:{settings.ingest_pipeline_version}"
    try:
        job = create_job(
            client,
            user.id,
            case_id,
            document.id,
            JobCreate(
                job_type="ingest",
                idempotency_key=idempotency_key,
                pipeline_version=settings.ingest_pipeline_version,
            ),
        )
    except Exception:
        try:
            update_document(
                client,
                user.id,
                document.id,
                DocumentUpdate(
                    quality_status="failed",
                    processing_status="failed",
                    quality_report={"error_code": "PROCESSING_FAILED"},
                ),
            )
        except Exception:
            pass
        raise
    _write_upload_audit(client, user, request, document, job)
    return UploadResponse(
        document_id=document.id,
        job_id=job.id,
        quality_status=document.quality_status,
        duplicate_of=document.duplicate_of,
    )


@router.get(
    "/cases/{case_id}/documents",
    response_model=list[DocumentResponse],
)
def list_case_documents(
    case_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[DocumentResponse]:
    """List documents for an owned case with limit/offset pagination."""
    documents = list_documents(
        client, user.id, case_id, limit=limit, offset=offset
    )
    return [DocumentResponse.from_document(document) for document in documents]


@router.get("/documents/{document_id}", response_model=DocumentResponse)
def get_document(
    document_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
) -> DocumentResponse:
    """Return one document owned by the authenticated user."""
    document = require_document_access(document_id, user, client=client)
    return DocumentResponse.from_document(document)


@router.get(
    "/documents/{document_id}/signed-url",
    response_model=SignedUrlResponse,
)
def get_document_signed_url(
    document_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
) -> SignedUrlResponse:
    """Create a 60-second link to the original, after an ownership check.

    The link is minted from the stored storage path, never from a client value,
    and it expires quickly so it cannot be shared as a permanent download.
    """
    document = require_document_access(document_id, user, client=client)
    if (document.quality_report or {}).get("deletion_requested"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document deletion is in progress",
        )
    try:
        url = create_signed_url(
            document.storage_path,
            SIGNED_URL_SECONDS,
            client=client,
        )
    except StorageError as exc:
        logger.warning(
            "signed url creation failed",
            extra={
                "document_id": str(document.id),
                "error_class": type(exc).__name__,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The original file is temporarily unavailable",
        ) from None

    return SignedUrlResponse(
        document_id=document.id,
        filename=document.original_filename,
        mime=document.detected_mime,
        url=url,
        expires_in_seconds=SIGNED_URL_SECONDS,
    )


@router.get("/jobs/{job_id}", response_model=JobResponse)
def get_job(
    job_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
) -> JobResponse:
    """Return one processing job owned by the authenticated user."""
    job = require_job_access(job_id, user, client=client)
    return JobResponse.from_job(job)


@router.post(
    "/documents/{document_id}/reindex",
    response_model=ReindexResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_document_reindex(
    document_id: UUID,
    request: Request,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> ReindexResponse:
    """Queue an idempotent reindex job for one owned document."""
    document = require_document_access(document_id, user, client=client)
    quality_report = document.quality_report or {}
    if quality_report.get("deletion_requested"):
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Document deletion is in progress",
        )
    if document.quality_status == "password_required":
        raise HTTPException(
            status_code=status.HTTP_409_CONFLICT,
            detail="Password-protected documents cannot be reindexed",
        )

    idempotency_key = reindex_idempotency_key(document.id, settings)
    job = find_job_by_idempotency_key(client, user.id, idempotency_key)
    if job is None:
        job = create_job(
            client,
            user.id,
            document.case_id,
            document.id,
            JobCreate(
                job_type="reindex",
                idempotency_key=idempotency_key,
                pipeline_version=settings.ingest_pipeline_version,
            ),
        )

    write_audit_event(
        client,
        user.id,
        event_type="document.reindex_requested",
        case_id=document.case_id,
        target_type="document",
        target_id=str(document.id),
        request_id=_request_id(request),
        safe_metadata={
            "job_id": str(job.id),
            "embedding_model": settings.embedding_model,
            "index_schema_version": settings.index_schema_version,
        },
    )
    return ReindexResponse(
        document_id=document.id,
        job_id=job.id,
        job_type="reindex",
    )


@router.delete(
    "/documents/{document_id}",
    response_model=DeleteResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
def request_document_deletion(
    document_id: UUID,
    request: Request,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> DeleteResponse:
    """Mark a document for deletion and enqueue a worker job."""
    document = require_document_access(document_id, user, client=client)
    quality_report = dict(document.quality_report or {})
    quality_report["deletion_requested"] = True
    updated = update_document(
        client,
        user.id,
        document.id,
        DocumentUpdate(
            quality_status="needs_review",
            quality_report=quality_report,
            processing_status="needs_review",
        ),
    )

    idempotency_key = (
        f"{document.id}:delete:{settings.ingest_pipeline_version}"
    )
    job = find_job_by_idempotency_key(client, user.id, idempotency_key)
    if job is None:
        job = create_job(
            client,
            user.id,
            document.case_id,
            document.id,
            JobCreate(
                job_type="delete",
                idempotency_key=idempotency_key,
                pipeline_version=settings.ingest_pipeline_version,
            ),
        )

    write_audit_event(
        client,
        user.id,
        event_type="document.deletion_requested",
        case_id=updated.case_id,
        target_type="document",
        target_id=str(updated.id),
        request_id=_request_id(request),
        safe_metadata={"job_id": str(job.id), "deletion_requested": True},
    )
    return DeleteResponse(document_id=updated.id, job_id=job.id)
