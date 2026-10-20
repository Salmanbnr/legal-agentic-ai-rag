"""Liveness endpoint."""

from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Request
from pydantic import BaseModel

router = APIRouter(tags=["system"])


class HealthResponse(BaseModel):
    """Health check payload."""

    status: Literal["ok"]
    service: str
    version: str


@router.get("/health", response_model=HealthResponse)
def health(request: Request) -> HealthResponse:
    """Return service liveness without touching external dependencies."""
    return HealthResponse(
        status="ok",
        service="legal-ai-backend",
        version=request.app.version,
    )
