"""Retrieval payload models shared by the search API and the agents.

Every evidence span carries the stored page range, the source anchor, and the
quality signals of the page it came from so a citation can be checked against
Postgres without re-reading the document.
"""

from __future__ import annotations

from typing import Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.db.repos import QualityStatus

RetrievalMethod = Literal["dense", "lexical", "hybrid"]


class EvidenceSpan(BaseModel):
    """One citation-ready piece of evidence returned by a search.

    The span is a projection of a stored chunk. ``quote_or_span`` holds the
    chunk text and may include an adjacent chunk when neighbor expansion was
    needed, but ``evidence_id`` and ``source_anchor`` always identify the
    primary chunk so the citation stays checkable.
    """

    evidence_id: UUID
    case_id: UUID
    document_id: UUID
    document_version_id: UUID | None = None
    title: str
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    quote_or_span: str
    extraction_method: str | None = None
    quality_status: QualityStatus
    ocr_confidence: float | None = None
    source_anchor: str | None = None
    retrieval_method: RetrievalMethod
    retrieval_rank: int = Field(ge=1)
    quality_flags: list[Any] = Field(default_factory=list)
    retrieval_score: float | None = None


class CoverageMetrics(BaseModel):
    """Simple, LLM-free coverage signals for a set of evidence spans."""

    span_count: int = 0
    documents_covered: int = 0
    top_rank_score_spread: float = 0.0
    low_quality_span_count: int = 0
    question_terms: int = 0


class SearchRequest(BaseModel):
    """Fields accepted by the case search endpoint."""

    model_config = ConfigDict(extra="forbid")

    query: str = Field(min_length=1, max_length=2000)
    top_k: int = Field(default=8, ge=1, le=50)
    document_ids: list[UUID] | None = None


class SearchResponse(BaseModel):
    """Evidence spans and coverage signals for one search."""

    case_id: UUID
    query: str
    spans: list[EvidenceSpan] = Field(default_factory=list)
    coverage: CoverageMetrics = Field(default_factory=CoverageMetrics)
