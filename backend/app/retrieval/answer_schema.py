"""Structured answer models returned by the baseline hybrid-RAG pipeline.

The answer is built from retrieved evidence only. Claims carry the evidence
ids they rely on so the citation validator can drop anything that was not
actually retrieved for the case.
"""

from __future__ import annotations

from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.retrieval.schemas import EvidenceSpan

ClaimType = Literal["direct", "inference"]

FinalStatus = Literal[
    "answered",
    "partial",
    "insufficient_evidence",
    "conflicting_evidence",
]


class AnswerClaim(BaseModel):
    """One factual statement with the evidence that supports it."""

    text: str = Field(min_length=1)
    evidence_ids: list[UUID] = Field(default_factory=list)
    type: ClaimType = "direct"
    confidence: float | None = None


class AnswerResponse(BaseModel):
    """The complete answer to a question, with its evidence."""

    answer_text: str = ""
    claims: list[AnswerClaim] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    final_status: FinalStatus = "answered"
    reason_codes: list[str] = Field(default_factory=list)
    evidence: list[EvidenceSpan] = Field(default_factory=list)


class AnswerDraft(BaseModel):
    """The model's draft, before the backend attaches the evidence.

    The model never fills the evidence list. The backend attaches only the
    spans that the citation validator accepted, so a hallucinated span can
    never reach the answer.
    """

    answer_text: str = ""
    claims: list[AnswerClaim] = Field(default_factory=list)
    conflicts: list[str] = Field(default_factory=list)
    limitations: list[str] = Field(default_factory=list)
    follow_up_questions: list[str] = Field(default_factory=list)
    final_status: FinalStatus = "answered"
    reason_codes: list[str] = Field(default_factory=list)


def draft_to_response(draft: AnswerDraft) -> AnswerResponse:
    """Turn a model draft into an answer without any evidence attached."""
    return AnswerResponse(**draft.model_dump(), evidence=[])
