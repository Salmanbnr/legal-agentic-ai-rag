"""The answer synthesizer.

The synthesizer only uses the claims that survived verification, the
conflicts, the gaps, and the original question. It never writes a factual
statement of its own: every sentence in the answer is a verified claim. Two
versions that disagree both stay visible.
"""

from __future__ import annotations

from typing import Any
from uuid import UUID

from pydantic import BaseModel, Field

from app.agents.contradiction_agent import ConflictRecord
from app.agents.state import DraftClaim, EvidenceGap, VerificationResult
from app.retrieval.answer_schema import AnswerClaim, AnswerResponse
from app.retrieval.schemas import EvidenceSpan

REASON_NO_VERIFIED_CLAIM = "NO_VERIFIED_CLAIM"
REASON_QUALIFIED_ANSWER = "QUALIFIED_ANSWER"
REASON_CONFLICTS_PRESENT = "CONFLICTS_PRESENT"
REASON_GAPS_PRESENT = "EVIDENCE_GAPS_PRESENT"
REASON_LOW_QUALITY_EVIDENCE = "LOW_QUALITY_EVIDENCE"

_SUPPORTED_LABELS = ("supported", "partially_supported")


class SynthesisInput(BaseModel):
    """Everything the synthesizer needs for one question."""

    question: str = ""
    claims: list[DraftClaim] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)
    conflicts: list[ConflictRecord] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    spans: list[EvidenceSpan] = Field(default_factory=list)


class SynthesisOutput(BaseModel):
    """The assembled answer."""

    answer: AnswerResponse
    reason_codes: list[str] = Field(default_factory=list)


def _verified_ids(results: list[VerificationResult]) -> set[str]:
    """Return the claim ids that the verifier accepted."""
    return {
        result.claim_id
        for result in results
        if result.supported and result.label in _SUPPORTED_LABELS
    }


def _used_spans(
    claims: list[DraftClaim], spans: list[EvidenceSpan]
) -> list[EvidenceSpan]:
    """Return the spans the used claims rely on, in citation order."""
    by_id = {span.evidence_id: span for span in spans}
    used: list[EvidenceSpan] = []
    seen: set[UUID] = set()
    for claim in claims:
        for evidence_id in claim.evidence_ids:
            span = by_id.get(evidence_id)
            if span is None or evidence_id in seen:
                continue
            seen.add(evidence_id)
            used.append(span)
    return used


def _answer_text(
    question: str, claims: list[DraftClaim], conflicts: list[ConflictRecord]
) -> str:
    """Build the answer prose from the verified claims only."""
    if not claims:
        return (
            f"The retrieved evidence does not answer the question about "
            f"{question.strip() or 'this matter'}. The material may exist in "
            "the case file but was not returned by the search, so this answer "
            "does not claim that it is absent."
        )
    direct = [claim.text for claim in claims if claim.type == "direct"]
    inference = [claim.text for claim in claims if claim.type == "inference"]
    parts = ["Based on the retrieved evidence:"]
    for text in direct:
        parts.append(f"- {text}")
    if inference:
        parts.append("The following is inference rather than a direct statement:")
        for text in inference:
            parts.append(f"- {text}")
    if conflicts:
        parts.append(
            "The evidence disagrees with itself on part of this question, so "
            "both readings are shown above."
        )
    return "\n".join(parts)


def synthesize_answer(data: SynthesisInput) -> SynthesisOutput:
    """Assemble the final answer from the verified claims.

    Args:
        data: The claims, the verification results, the conflicts, the gaps,
            and the evidence.

    Returns:
        The answer and the reason codes that explain its status.
    """
    verified = _verified_ids(data.verification_results)
    used_claims = [
        claim for claim in data.claims if claim.id in verified
    ]
    spans = _used_spans(used_claims, data.spans)
    reason_codes: list[str] = []

    answer_claims = [
        AnswerClaim(
            text=claim.text,
            evidence_ids=list(claim.evidence_ids),
            type=claim.type,
        )
        for claim in used_claims
    ]

    conflicts = [
        (
            f"{conflict.proposition_a} / {conflict.proposition_b}"
            + (f" ({conflict.field})" if conflict.field else "")
        )
        for conflict in data.conflicts
    ]
    if conflicts:
        reason_codes.append(REASON_CONFLICTS_PRESENT)

    limitations: list[str] = []
    for gap in data.gaps:
        limitations.append(gap.description)
    if data.gaps:
        reason_codes.append(REASON_GAPS_PRESENT)

    low_quality = [
        span for span in spans if span.quality_flags
    ]
    if low_quality:
        limitations.append(
            "Some of the supporting pages carry quality flags, so their "
            "wording may have been misread."
        )
        reason_codes.append(REASON_LOW_QUALITY_EVIDENCE)

    follow_up: list[str] = []
    for gap in data.gaps:
        follow_up.append(
            "Could you confirm whether the missing material is in the case "
            "file, or upload the document that should hold it?"
        )
        break

    if not used_claims:
        reason_codes.append(REASON_NO_VERIFIED_CLAIM)
        status = "insufficient_evidence"
        answer_text = _answer_text(data.question, [], data.conflicts)
    elif conflicts:
        status = "conflicting_evidence"
        answer_text = _answer_text(data.question, used_claims, data.conflicts)
    elif limitations or len(used_claims) < len(data.claims):
        status = "partial"
        reason_codes.append(REASON_QUALIFIED_ANSWER)
        answer_text = _answer_text(data.question, used_claims, data.conflicts)
    else:
        status = "answered"
        answer_text = _answer_text(data.question, used_claims, data.conflicts)

    answer = AnswerResponse(
        answer_text=answer_text,
        claims=answer_claims,
        conflicts=conflicts,
        limitations=limitations,
        follow_up_questions=follow_up,
        final_status=status,
        reason_codes=reason_codes,
        evidence=spans,
    )
    return SynthesisOutput(answer=answer, reason_codes=reason_codes)


def manifest_for(answer: AnswerResponse) -> list[dict[str, Any]]:
    """Return the evidence manifest: ids and pages only."""
    manifest: list[dict[str, Any]] = []
    for span in answer.evidence:
        manifest.append(
            {
                "evidence_id": str(span.evidence_id),
                "document_id": str(span.document_id),
                "page_start": span.page_start,
                "page_end": span.page_end,
                "source_anchor": span.source_anchor,
            }
        )
    return manifest
