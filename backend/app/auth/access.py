"""Case and document access checks with existence-safe 404 responses."""

from __future__ import annotations

from uuid import UUID

from fastapi import HTTPException, status
from supabase import Client

from app.auth.dependencies import CurrentUser
from app.db.client import get_service_client
from app.db.repos import (
    Case,
    CaseAccessDenied,
    Document,
    DocumentAccessDenied,
    ProcessingJob,
    RecordNotFound,
    get_case_for_user,
    get_document_for_user,
    get_job_for_user,
)

CASE_NOT_FOUND_MESSAGE = "Case not found"
DOCUMENT_NOT_FOUND_MESSAGE = "Document not found"
JOB_NOT_FOUND_MESSAGE = "Job not found"


def require_case_access(
    case_id: UUID | str,
    user: CurrentUser,
    *,
    client: Client | None = None,
) -> Case:
    """Return a case only when it belongs to the authenticated user.

    Raises:
        HTTPException: 404 with the same message for missing and foreign cases.
    """
    db = client or get_service_client()
    try:
        return get_case_for_user(db, user.id, case_id)
    except (CaseAccessDenied, ValueError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=CASE_NOT_FOUND_MESSAGE,
        ) from None


def require_document_access(
    document_id: UUID | str,
    user: CurrentUser,
    *,
    client: Client | None = None,
) -> Document:
    """Return a document only when it belongs to the authenticated user's case.

    Raises:
        HTTPException: 404 with the same message for missing and foreign
            documents.
    """
    db = client or get_service_client()
    try:
        return get_document_for_user(db, user.id, document_id)
    except (DocumentAccessDenied, ValueError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=DOCUMENT_NOT_FOUND_MESSAGE,
        ) from None


def require_job_access(
    job_id: UUID | str,
    user: CurrentUser,
    *,
    client: Client | None = None,
) -> ProcessingJob:
    """Return a job only when it belongs to the authenticated user's case.

    Raises:
        HTTPException: 404 with the same message for missing and foreign jobs.
    """
    db = client or get_service_client()
    try:
        return get_job_for_user(db, user.id, job_id)
    except (RecordNotFound, CaseAccessDenied, ValueError):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=JOB_NOT_FOUND_MESSAGE,
        ) from None
