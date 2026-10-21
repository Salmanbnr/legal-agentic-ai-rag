"""Lexical retrieval through the Postgres full-text search function.

The backend calls ``public.search_chunks_lexical`` through the service client.
The function filters by case id and by the supplied document id list, so the
authorized document ids are always applied in Postgres as well.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence
from uuid import UUID

from supabase import Client

logger = logging.getLogger(__name__)

LEXICAL_FUNCTION = "search_chunks_lexical"


class LexicalError(RuntimeError):
    """A safe lexical retrieval failure with a stable error code."""

    def __init__(self, error_code: str, *, retryable: bool = True) -> None:
        """Create a lexical error without including the query text."""
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class LexicalHit:
    """One lexical retrieval candidate."""

    chunk_id: str
    score: float
    document_id: str | None = None
    rank: int = 0


def lexical_search(
    client: Client,
    case_id: UUID | str,
    query: str,
    *,
    document_ids: Sequence[UUID | str],
    top_k: int,
) -> list[LexicalHit]:
    """Return chunks matching the query by full-text rank.

    Args:
        client: Service-role Supabase client used for the RPC call.
        case_id: Case whose chunks may be searched.
        query: Search text.
        document_ids: Already-authorized document ids applied as a filter.
        top_k: Maximum number of candidates to return.

    Returns:
        Lexical candidates ordered by descending rank.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    allowed = [str(value) for value in document_ids if str(value)]
    if not allowed or not query.strip():
        return []

    try:
        response = client.rpc(
            LEXICAL_FUNCTION,
            {
                "p_case_id": str(case_id),
                "p_query": query,
                "p_document_ids": allowed,
                "p_limit": top_k,
            },
        ).execute()
    except Exception as exc:
        logger.error(
            "lexical search failed",
            extra={"error_class": type(exc).__name__},
        )
        raise LexicalError("LEXICAL_SEARCH_FAILED") from None

    rows: Any = getattr(response, "data", None) or []
    hits: list[LexicalHit] = []
    for rank, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            continue
        chunk_id = str(row.get("id") or "")
        if not chunk_id:
            continue
        document_id = row.get("document_id")
        hits.append(
            LexicalHit(
                chunk_id=chunk_id,
                score=float(row.get("rank") or 0.0),
                document_id=str(document_id) if document_id else None,
                rank=rank,
            )
        )
    return hits
