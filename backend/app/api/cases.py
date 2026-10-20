"""Authenticated case CRUD endpoints."""

from __future__ import annotations

from datetime import datetime
from typing import Annotated, Self
from uuid import UUID

from fastapi import APIRouter, Depends, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
from supabase import Client

from app.auth.access import require_case_access
from app.auth.dependencies import CurrentUser, get_current_user
from app.db.client import get_db_client
from app.db.repos import (
    Case,
    CaseStatus,
    CaseUpdate,
    create_case as create_case_row,
    list_cases as list_case_rows,
    update_case as update_case_row,
    write_audit_event,
)

router = APIRouter(prefix="/cases", tags=["cases"])


class CaseCreateRequest(BaseModel):
    """Fields accepted when creating a case."""

    model_config = ConfigDict(extra="forbid")

    title: str = Field(min_length=1, max_length=300)
    description: str | None = Field(default=None, max_length=20000)

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str) -> str:
        """Reject whitespace-only titles and normalize surrounding spaces."""
        if not value.strip():
            raise ValueError("Title must not be empty")
        return value.strip()


class CaseUpdateRequest(BaseModel):
    """Mutable case fields; omitted fields are left unchanged."""

    model_config = ConfigDict(extra="forbid")

    title: str | None = Field(default=None, min_length=1, max_length=300)
    description: str | None = Field(default=None, max_length=20000)
    status: CaseStatus | None = None

    @field_validator("title")
    @classmethod
    def validate_title(cls, value: str | None) -> str | None:
        """Reject whitespace-only titles and normalize surrounding spaces."""
        if value is None:
            return None
        if not value.strip():
            raise ValueError("Title must not be empty")
        return value.strip()

    @model_validator(mode="after")
    def validate_update(self) -> Self:
        """Require at least one field and reject null required values."""
        if not self.model_fields_set:
            raise ValueError("At least one field must be provided")
        if "title" in self.model_fields_set and self.title is None:
            raise ValueError("Title must not be null")
        if "status" in self.model_fields_set and self.status is None:
            raise ValueError("Status must not be null")
        return self


class CaseResponse(BaseModel):
    """Safe case representation returned to the client."""

    model_config = ConfigDict(from_attributes=True)

    id: UUID
    title: str
    description: str | None = None
    status: CaseStatus
    created_at: datetime
    updated_at: datetime


def _request_id(request: Request) -> str | None:
    """Read the middleware-assigned request id without trusting headers."""
    return getattr(request.state, "request_id", None)


def _write_case_audit(
    client: Client,
    user: CurrentUser,
    request: Request,
    case: Case,
    event_type: str,
    safe_metadata: dict[str, object],
) -> None:
    """Append a safe audit event for a case mutation."""
    write_audit_event(
        client,
        user.id,
        event_type=event_type,
        case_id=case.id,
        target_type="case",
        target_id=str(case.id),
        request_id=_request_id(request),
        safe_metadata=safe_metadata,
    )


def _to_response(case: Case) -> CaseResponse:
    """Convert a repository row into the public response model."""
    return CaseResponse.model_validate(case)


@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
def create_case(
    payload: CaseCreateRequest,
    request: Request,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
) -> CaseResponse:
    """Create a case owned by the authenticated user."""
    case = create_case_row(
        client,
        user.id,
        title=payload.title,
        description=payload.description,
    )
    _write_case_audit(
        client,
        user,
        request,
        case,
        event_type="case.created",
        safe_metadata={"status": case.status},
    )
    return _to_response(case)


@router.get("", response_model=list[CaseResponse])
def list_cases(
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
    case_status: Annotated[CaseStatus | None, Query(alias="status")] = None,
    limit: Annotated[int, Query(ge=1, le=200)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[CaseResponse]:
    """List only cases owned by the authenticated user."""
    rows = list_case_rows(
        client,
        user.id,
        status=case_status,
        limit=limit,
        offset=offset,
    )
    return [_to_response(case) for case in rows]


@router.get("/{case_id}", response_model=CaseResponse)
def get_case(
    case_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
) -> CaseResponse:
    """Return one case owned by the authenticated user."""
    return _to_response(require_case_access(case_id, user, client=client))


@router.patch("/{case_id}", response_model=CaseResponse)
def update_case(
    case_id: UUID,
    payload: CaseUpdateRequest,
    request: Request,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
) -> CaseResponse:
    """Update title, description, or status for an owned case."""
    require_case_access(case_id, user, client=client)
    update = CaseUpdate.model_validate(payload.model_dump(exclude_unset=True))
    case = update_case_row(client, user.id, case_id, update)
    _write_case_audit(
        client,
        user,
        request,
        case,
        event_type="case.updated",
        safe_metadata={
            "changed_fields": sorted(payload.model_dump(exclude_unset=True).keys()),
            "status": case.status,
        },
    )
    return _to_response(case)


@router.delete("/{case_id}", response_model=CaseResponse)
def archive_case(
    case_id: UUID,
    request: Request,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
) -> CaseResponse:
    """Archive a case instead of deleting it."""
    require_case_access(case_id, user, client=client)
    case = update_case_row(client, user.id, case_id, CaseUpdate(status="archived"))
    _write_case_audit(
        client,
        user,
        request,
        case,
        event_type="case.archived",
        safe_metadata={"status": "archived"},
    )
    return _to_response(case)
