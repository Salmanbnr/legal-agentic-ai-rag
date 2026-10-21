"""Hybrid dense and lexical retrieval for one authorized case.

Authorization is enforced in code before any candidate is fetched. The case is
checked against the authenticated user, the authorized document ids are loaded
from Postgres, and that list is applied as a filter in both Pinecone and
Postgres. A document id supplied by a caller is only honored when it belongs to
the case; foreign ids are dropped before any search runs.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Sequence
from uuid import UUID

from supabase import Client

from app.auth.dependencies import CurrentUser
from app.core.config import Settings, get_settings
from app.db.repos import (
    Document,
    DocumentChunk,
    DocumentPage,
    RepositoryError,
    get_case_for_user,
    get_chunks_by_ids,
    get_neighbor_chunks,
    get_pages_for_documents,
    list_documents,
)
from app.embeddings.model import TextEmbedder
from app.retrieval.dense import dense_search
from app.retrieval.fusion import FusedChunk, reciprocal_rank_fusion
from app.retrieval.evidence_meta import collect_quality_flags, page_signals
from app.retrieval.lexical import lexical_search
from app.retrieval.schemas import EvidenceSpan, RetrievalMethod

logger = logging.getLogger(__name__)

_CROSS_REFERENCE = re.compile(
    r"^\s*(?:subject to|pursuant to|as provided in|as set out in|"
    r"in accordance with|referred to in|refer to|see|under)\b",
    re.IGNORECASE,
)
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s*$|[:;][\"')\]]*\s*$")
_MAX_CASE_DOCUMENTS = 200


@dataclass(frozen=True, slots=True)
class _Selected:
    """A fused candidate paired with its stored chunk row."""

    fused: FusedChunk
    chunk: DocumentChunk


def starts_with_cross_reference(text: str) -> bool:
    """Return whether a chunk opens with a reference to another clause."""
    return bool(_CROSS_REFERENCE.match(text or ""))


def ends_mid_sentence(text: str) -> bool:
    """Return whether a chunk stops without terminal punctuation."""
    stripped = (text or "").rstrip()
    if not stripped:
        return False
    return not bool(_SENTENCE_END.search(stripped))


def needs_neighbor_expansion(text: str) -> bool:
    """Return whether adjacent chunk text should be attached to a chunk."""
    return starts_with_cross_reference(text) or ends_mid_sentence(text)


def _dedupe_by_content_hash(selected: Sequence[_Selected]) -> list[_Selected]:
    """Keep the best-ranked chunk for each content hash.

    Chunks with the same content hash are near-duplicates of each other, even
    when they live in different documents, and only the best-ranked copy is
    kept. Chunks with different hashes are never merged, so conflicting
    document versions both stay visible.
    """
    seen: set[str] = set()
    unique: list[_Selected] = []
    for item in selected:
        if item.chunk.content_hash in seen:
            continue
        seen.add(item.chunk.content_hash)
        unique.append(item)
    return unique


def _apply_source_diversity(
    selected: Sequence[_Selected],
    *,
    top_k: int,
    max_per_document: int,
) -> list[_Selected]:
    """Limit how many slots one document may take, then fill what is left."""
    chosen: list[_Selected] = []
    counts: dict[str, int] = {}
    for item in selected:
        if len(chosen) >= top_k:
            break
        document_id = str(item.chunk.document_id)
        if counts.get(document_id, 0) >= max_per_document:
            continue
        counts[document_id] = counts.get(document_id, 0) + 1
        chosen.append(item)

    if len(chosen) < top_k:
        chosen_ids = {str(item.chunk.id) for item in chosen}
        for item in selected:
            if len(chosen) >= top_k:
                break
            if str(item.chunk.id) in chosen_ids:
                continue
            chosen.append(item)
            chosen_ids.add(str(item.chunk.id))
    return chosen[:top_k]


def _page_signals(
    pages: Sequence[DocumentPage], chunk: DocumentChunk
) -> tuple[str | None, float | None]:
    """Return the extraction method and the worst page confidence of a chunk."""
    return page_signals(pages, chunk)


def _collect_flags(
    chunk: DocumentChunk, pages: Sequence[DocumentPage]
) -> list[Any]:
    """Merge chunk flags with the flags of the pages the chunk covers."""
    return collect_quality_flags(chunk, pages)


def _expand_with_neighbors(
    client: Client,
    user_id: UUID | str,
    chunk: DocumentChunk,
    text: str,
) -> tuple[str, int | None, int | None]:
    """Attach adjacent chunk text when a chunk starts or ends mid-thought.

    A cross reference at the start needs the preceding chunk, and a chunk that
    stops mid sentence needs the following one. The returned page range covers
    every attached chunk so the citation stays honest.
    """
    if not needs_neighbor_expansion(text):
        return text, chunk.page_start, chunk.page_end

    try:
        neighbors = get_neighbor_chunks(
            client, user_id, chunk.id, before=1, after=1
        )
    except RepositoryError as exc:
        logger.warning(
            "neighbor expansion skipped",
            extra={
                "chunk_id": str(chunk.id),
                "error_class": type(exc).__name__,
            },
        )
        return text, chunk.page_start, chunk.page_end

    by_index = {item.chunk_index: item for item in neighbors}
    before = by_index.get(chunk.chunk_index - 1)
    after = by_index.get(chunk.chunk_index + 1)

    parts: list[str] = []
    used = [chunk]
    if starts_with_cross_reference(text) and before is not None:
        parts.append(before.text_content)
        used.append(before)
    parts.append(text)
    if ends_mid_sentence(text) and after is not None:
        parts.append(after.text_content)
        used.append(after)

    expanded = "\n\n".join(part for part in parts if part)
    pages = [item.page_start for item in used if item.page_start is not None]
    page_ends = [item.page_end for item in used if item.page_end is not None]
    page_start = min(pages) if pages else chunk.page_start
    page_end = max(page_ends) if page_ends else chunk.page_end
    return expanded, page_start, page_end


def _retrieval_method(fused: FusedChunk) -> RetrievalMethod:
    """Return the method label for a fused candidate."""
    if len(fused.methods) > 1:
        return "hybrid"
    return fused.methods[0] if fused.methods else "dense"


def hybrid_search(
    client: Client,
    case_id: UUID | str,
    user: CurrentUser,
    query: str,
    *,
    top_k: int = 8,
    document_ids: Sequence[UUID | str] | None = None,
    settings: Settings | None = None,
    embedder: TextEmbedder | None = None,
    index: Any | None = None,
) -> list[EvidenceSpan]:
    """Search one authorized case and return citation-ready evidence spans.

    Args:
        client: Service-role Supabase client used for the lexical search.
        case_id: Case to search. The case must belong to ``user``.
        user: Authenticated user the request is executed for.
        query: Natural-language question.
        top_k: Maximum number of spans to return.
        document_ids: Optional document ids to restrict the search to. Ids that
            do not belong to the case are ignored.
        settings: Optional validated settings.
        embedder: Optional embedder override, mainly for tests.
        index: Optional Pinecone index override, mainly for tests.

    Returns:
        Evidence spans ordered by fused rank.
    """
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    resolved = settings or get_settings()

    # Authorization is code, never a prompt: the case must belong to the user.
    get_case_for_user(client, user.id, case_id)
    documents: list[Document] = list_documents(
        client, user.id, case_id, limit=_MAX_CASE_DOCUMENTS
    )
    authorized = {str(document.id) for document in documents}
    requested = (
        {str(value) for value in document_ids}
        if document_ids is not None
        else set(authorized)
    )
    effective_ids = sorted(requested & authorized)
    if not effective_ids:
        logger.info(
            "retrieval skipped because no authorized document matched",
            extra={"case_id": str(case_id), "requested_count": len(requested)},
        )
        return []

    candidate_count = max(
        top_k * resolved.retrieval_candidate_multiplier, top_k
    )
    dense_hits = dense_search(
        case_id,
        query,
        document_ids=effective_ids,
        top_k=candidate_count,
        settings=resolved,
        embedder=embedder,
        index=index,
    )
    lexical_hits = lexical_search(
        client,
        case_id,
        query,
        document_ids=effective_ids,
        top_k=candidate_count,
    )
    fused = reciprocal_rank_fusion(
        {
            "dense": [hit.chunk_id for hit in dense_hits],
            "lexical": [hit.chunk_id for hit in lexical_hits],
        }
    )
    if not fused:
        return []

    chunks = get_chunks_by_ids(
        client, user.id, case_id, [item.chunk_id for item in fused]
    )
    by_id = {str(chunk.id): chunk for chunk in chunks}
    selected = [
        _Selected(fused=item, chunk=by_id[item.chunk_id])
        for item in fused
        if item.chunk_id in by_id
    ]
    selected = _dedupe_by_content_hash(selected)
    selected = _apply_source_diversity(
        selected,
        top_k=top_k,
        max_per_document=resolved.retrieval_max_spans_per_document,
    )
    if not selected:
        return []

    pages = get_pages_for_documents(
        client,
        user.id,
        case_id,
        [str(item.chunk.document_id) for item in selected],
    )
    pages_by_document: dict[str, list[DocumentPage]] = {}
    for page in pages:
        pages_by_document.setdefault(str(page.document_id), []).append(page)
    documents_by_id = {str(document.id): document for document in documents}

    spans: list[EvidenceSpan] = []
    for rank, item in enumerate(selected, start=1):
        chunk = item.chunk
        document = documents_by_id.get(str(chunk.document_id))
        document_pages = pages_by_document.get(str(chunk.document_id), [])
        text = chunk.text_content
        page_start = chunk.page_start
        page_end = chunk.page_end
        if resolved.retrieval_neighbor_expansion:
            text, page_start, page_end = _expand_with_neighbors(
                client, user.id, chunk, text
            )
        extraction_method, confidence = _page_signals(document_pages, chunk)
        spans.append(
            EvidenceSpan(
                evidence_id=chunk.id,
                case_id=chunk.case_id,
                document_id=chunk.document_id,
                document_version_id=chunk.document_version_id,
                title=document.original_filename if document else "",
                page_start=page_start,
                page_end=page_end,
                section_path=list(chunk.section_path),
                quote_or_span=text,
                extraction_method=extraction_method,
                quality_status=(
                    document.quality_status if document else "needs_review"
                ),
                ocr_confidence=confidence,
                source_anchor=chunk.source_anchor,
                retrieval_method=_retrieval_method(item.fused),
                retrieval_rank=rank,
                quality_flags=_collect_flags(chunk, document_pages),
                retrieval_score=item.fused.score,
            )
        )

    logger.info(
        "hybrid search finished",
        extra={
            "case_id": str(case_id),
            "document_count": len(effective_ids),
            "dense_candidates": len(dense_hits),
            "lexical_candidates": len(lexical_hits),
            "span_count": len(spans),
        },
    )
    return spans
