"""Dense retrieval from the Pinecone case namespace.

The query is embedded with the same model that produced the stored vectors.
Results are always restricted to the case namespace and to an explicit list of
authorized document ids, so no request can read another case's vectors.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Sequence
from uuid import UUID

from app.core.config import Settings, get_settings
from app.embeddings.model import TextEmbedder, get_embedder
from app.indexing.pinecone_store import get_index, namespace_for_case

logger = logging.getLogger(__name__)


class RetrievalError(RuntimeError):
    """A safe retrieval failure with a stable, persistable error code."""

    def __init__(self, error_code: str, *, retryable: bool = True) -> None:
        """Create a retrieval error without including the query or documents."""
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class DenseHit:
    """One dense retrieval candidate."""

    chunk_id: str
    score: float
    document_id: str | None = None
    rank: int = 0


def _match_attribute(match: Any, name: str, default: Any = None) -> Any:
    """Read a field from a Pinecone match model or a plain mapping."""
    if isinstance(match, dict):
        return match.get(name, default)
    return getattr(match, name, default)


def dense_search(
    case_id: UUID | str,
    query: str,
    *,
    document_ids: Sequence[UUID | str],
    top_k: int,
    settings: Settings | None = None,
    embedder: TextEmbedder | None = None,
    index: Any | None = None,
) -> list[DenseHit]:
    """Return the closest chunk ids for one query inside one case namespace.

    Args:
        case_id: Case whose namespace is searched.
        query: Natural-language question.
        document_ids: Already-authorized document ids. The list is applied as a
            metadata filter, so a foreign document id cannot be searched.
        top_k: Maximum number of candidates to return.
        settings: Optional validated settings.
        embedder: Optional embedder override, mainly for tests.
        index: Optional Pinecone index override, mainly for tests.

    Returns:
        Dense candidates ordered by descending similarity.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    allowed = [str(value) for value in document_ids if str(value)]
    if not allowed or not query.strip():
        return []

    resolved = settings or get_settings()
    active_embedder = embedder or get_embedder(resolved)
    vector = active_embedder.embed_query(query)
    resolved_index = index or get_index(resolved)

    metadata_filter = {"document_id": {"$in": allowed}}
    try:
        response = resolved_index.query(
            vector=list(vector),
            top_k=top_k,
            namespace=namespace_for_case(case_id),
            filter=metadata_filter,
            include_metadata=True,
            include_values=False,
        )
    except Exception as exc:
        logger.error(
            "pinecone query failed",
            extra={"error_class": type(exc).__name__},
        )
        raise RetrievalError("DENSE_SEARCH_FAILED") from None

    matches = _match_attribute(response, "matches") or []
    hits: list[DenseHit] = []
    for rank, match in enumerate(matches, start=1):
        chunk_id = str(_match_attribute(match, "id", "") or "")
        if not chunk_id:
            continue
        metadata = _match_attribute(match, "metadata") or {}
        document_id = metadata.get("document_id") if isinstance(metadata, dict) else None
        hits.append(
            DenseHit(
                chunk_id=chunk_id,
                score=float(_match_attribute(match, "score", 0.0) or 0.0),
                document_id=str(document_id) if document_id else None,
                rank=rank,
            )
        )
    return hits
