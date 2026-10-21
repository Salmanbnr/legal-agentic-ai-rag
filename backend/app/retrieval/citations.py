"""Deterministic validation of the citations in a generated answer.

The model may only cite evidence that was actually retrieved for the case.
This module checks every claim against the retrieved spans, removes what it
cannot support, and turns an answer with no supported claim into a clean
abstention that says what was searched and what is missing.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Sequence
from uuid import UUID

from app.retrieval.answer_schema import AnswerClaim, AnswerResponse
from app.retrieval.schemas import EvidenceSpan

logger = logging.getLogger(__name__)

REASON_NO_SUPPORTED_CLAIMS = "NO_SUPPORTED_CLAIMS"
REASON_UNSUPPORTED_CLAIMS = "UNSUPPORTED_CLAIMS_REMOVED"
REASON_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
REASON_NO_EVIDENCE = "NO_EVIDENCE_RETRIEVED"


@dataclass(frozen=True, slots=True)
class CitationCheck:
    """The validated answer and what the validation changed."""

    answer: AnswerResponse
    used_evidence: list[EvidenceSpan] = field(default_factory=list)
    removed_claim_count: int = 0
    removed_evidence_ids: list[UUID] = field(default_factory=list)
    reason_codes: list[str] = field(default_factory=list)


def _has_valid_anchor(span: EvidenceSpan) -> bool:
    """Return whether a span carries a usable page anchor."""
    return span.page_start is not None and span.page_start >= 1


def _searched_scope(spans: Sequence[EvidenceSpan]) -> str:
    """Describe the documents and pages the search actually covered."""
    if not spans:
        return "no documents were searched for this question"
    parts: list[str] = []
    for span in spans:
        pages = (
            f"page {span.page_start}"
            if span.page_start == span.page_end
            else f"pages {span.page_start}-{span.page_end}"
        )
        title = span.title or "an untitled document"
        parts.append(f"{title} ({pages})")
    unique: list[str] = []
    for part in parts:
        if part not in unique:
            unique.append(part)
    return "; ".join(unique)


def build_insufficient_evidence_answer(
    question: str,
    spans: Sequence[EvidenceSpan],
    *,
    reason_codes: Sequence[str] = (),
    note: str | None = None,
) -> AnswerResponse:
    """Build a clean abstention that never claims anything is absent.

    The message states what was searched and what is missing. It never says
    that a clause or document does not exist, because absence from the
    retrieved evidence is not proof of absence from the case file.
    """
    codes = list(reason_codes) or [REASON_INSUFFICIENT]
    if not spans and REASON_NO_EVIDENCE not in codes:
        codes.insert(0, REASON_NO_EVIDENCE)
    scope = _searched_scope(spans)
    retrieved = len(spans)
    document = (
        "The retrieved evidence does not answer this question."
        if retrieved
        else "No evidence was retrieved for this question."
    )
    answer_text = (
        f"{document} The search for \"{question}\" covered {retrieved} "
        f"passage(s) from: {scope}. The material this question asks about may "
        "still exist in the case file, but it was not returned by the search, "
        "so this answer does not claim that it is absent. Try a broader or "
        "differently worded question, or check that the relevant document has "
        "been uploaded and processed."
    )
    if note:
        answer_text = f"{answer_text} {note}"
    return AnswerResponse(
        answer_text=answer_text,
        claims=[],
        conflicts=[],
        limitations=[
            "No claim could be supported by the retrieved evidence.",
        ],
        follow_up_questions=[],
        final_status="insufficient_evidence",
        reason_codes=codes,
        evidence=[],
    )


def validate_citations(
    answer: AnswerResponse,
    spans: Sequence[EvidenceSpan],
    *,
    question: str = "",
) -> CitationCheck:
    """Check every claim against the retrieved evidence.

    Claims may only cite evidence ids that were retrieved for this case and
    that carry a valid page anchor. Ids that do not are stripped; a claim that
    keeps no valid id is removed. When no claim survives, the answer becomes
    an abstention.

    Args:
        answer: The generated answer.
        spans: The evidence spans that were retrieved for this case.
        question: The question that was asked, used in the abstention message.

    Returns:
        The validated answer together with what the validation changed.
    """
    by_id = {span.evidence_id: span for span in spans}
    reason_codes: list[str] = list(answer.reason_codes)
    removed_ids: list[UUID] = []
    removed_claims = 0
    used: list[EvidenceSpan] = []
    used_ids: set[UUID] = set()
    validated_claims: list[AnswerClaim] = []

    for claim in answer.claims:
        valid: list[UUID] = []
        invalid: list[UUID] = []
        for evidence_id in claim.evidence_ids:
            span = by_id.get(evidence_id)
            if span is None or not _has_valid_anchor(span):
                invalid.append(evidence_id)
            else:
                valid.append(evidence_id)
        if invalid:
            removed_ids.extend(invalid)
            if REASON_UNSUPPORTED_CLAIMS not in reason_codes:
                reason_codes.append(REASON_UNSUPPORTED_CLAIMS)
        if not valid:
            removed_claims += 1
            continue
        validated_claims.append(claim.model_copy(update={"evidence_ids": valid}))
        for evidence_id in valid:
            if evidence_id not in used_ids:
                used_ids.add(evidence_id)
                used.append(by_id[evidence_id])

    if not validated_claims:
        logger.info(
            "answer had no supported claim",
            extra={
                "removed_claim_count": removed_claims,
                "removed_evidence_count": len(removed_ids),
            },
        )
        abstention = build_insufficient_evidence_answer(
            question,
            spans,
            reason_codes=[*reason_codes, REASON_NO_SUPPORTED_CLAIMS],
        )
        return CitationCheck(
            answer=abstention,
            used_evidence=[],
            removed_claim_count=removed_claims,
            removed_evidence_ids=removed_ids,
            reason_codes=list(abstention.reason_codes),
        )

    final_status = answer.final_status
    limitations = list(answer.limitations)
    if removed_claims:
        final_status = "partial" if final_status == "answered" else final_status
        limitations.append(
            f"{removed_claims} statement(s) were removed because they cited "
            "evidence that was not retrieved for this case."
        )

    validated = answer.model_copy(
        update={
            "claims": validated_claims,
            "limitations": limitations,
            "final_status": final_status,
            "reason_codes": reason_codes,
            "evidence": used,
        }
    )
    return CitationCheck(
        answer=validated,
        used_evidence=used,
        removed_claim_count=removed_claims,
        removed_evidence_ids=removed_ids,
        reason_codes=reason_codes,
    )
