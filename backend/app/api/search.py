"""Authenticated case search used for debugging retrieval quality.

The endpoint only returns evidence from a case owned by the authenticated
user. Document ids sent by the client are intersected with the case's own
documents, so a foreign id can never widen the search.
"""

from __future__ import annotations

from typing import Annotated
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, status
from supabase import Client

from app.auth.access import require_case_access
from app.auth.dependencies import CurrentUser, get_current_user
from app.core.config import Settings, get_settings
from app.db.client import get_db_client
from app.retrieval.coverage import assess_coverage
from app.retrieval.dense import RetrievalError
from app.retrieval.hybrid import hybrid_search
from app.retrieval.lexical import LexicalError
from app.retrieval.schemas import SearchRequest, SearchResponse

router = APIRouter(tags=["search"])


@router.post("/cases/{case_id}/search", response_model=SearchResponse)
def search_case(
    case_id: UUID,
    payload: SearchRequest,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> SearchResponse:
    """Search one owned case and return its evidence spans."""
    require_case_access(case_id, user, client=client)
    try:
        spans = hybrid_search(
            client,
            case_id,
            user,
            payload.query,
            top_k=payload.top_k,
            document_ids=payload.document_ids,
            settings=settings,
        )
    except (RetrievalError, LexicalError):
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search is temporarily unavailable",
        ) from None
    return SearchResponse(
        case_id=case_id,
        query=payload.query,
        spans=spans,
        coverage=assess_coverage(payload.query, spans),
    )
