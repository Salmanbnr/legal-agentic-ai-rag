"""The deterministic final gate.

The gate is the last check before an answer is stored. It verifies that the
answer has the right shape, that every material claim carries a valid
citation, that no instruction-like text from a document leaked into the
answer, and that the status matches the evidence that is actually there.
"""

from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, Field

from app.agents.answer_synthesizer import manifest_for
from app.agents.safety import InjectionFinding, scan_text_for_injection
from app.agents.state import DraftClaim, VerificationResult
from app.retrieval.answer_schema import AnswerResponse
from app.retrieval.schemas import EvidenceSpan

REASON_INVALID_CITATION = "MATERIAL_CLAIM_WITHOUT_CITATION"
REASON_INJECTION_REMOVED = "INSTRUCTION_LIKE_TEXT_REMOVED"
REASON_STATUS_MISMATCH = "STATUS_ADJUSTED_TO_EVIDENCE"
REASON_EMPTY_ANSWER = "ANSWER_WITHOUT_SUPPORTED_CLAIM"

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class FinalGateInput(BaseModel):
    """Everything the final gate checks."""

    question: str = ""
    answer: AnswerResponse
    claims: list[DraftClaim] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)
    spans: list[EvidenceSpan] = Field(default_factory=list)
    injection_findings: list[InjectionFinding] = Field(default_factory=list)


class FinalGateOutput(BaseModel):
    """The checked answer and the manifest to store."""

    answer: AnswerResponse
    evidence_manifest: list[dict[str, Any]] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    passed: bool = True


def _strip_instruction_like_text(text: str) -> tuple[str, bool]:
    """Remove sentences that look like embedded instructions.

    The removed text is reported, never followed, and never returned to the
    client.
    """
    if not text:
        return text, False
    sentences = _SENTENCE_SPLIT.split(text)
    kept: list[str] = []
    removed = False
    for sentence in sentences:
        if scan_text_for_injection(sentence):
            removed = True
            continue
        kept.append(sentence)
    return " ".join(part for part in kept if part).strip(), removed


def _valid_citations(
    claim: DraftClaim, spans_by_id: dict[Any, EvidenceSpan]
) -> bool:
    """Return whether a claim cites evidence that exists and has a page."""
    if not claim.evidence_ids:
        return False
    for evidence_id in claim.evidence_ids:
        span = spans_by_id.get(evidence_id)
        if span is None:
            return False
        if span.page_start is None or span.page_start < 1:
            return False
    return True


def run_final_gate(data: FinalGateInput) -> FinalGateOutput:
    """Check the answer and return the version that may be stored.

    Args:
        data: The answer, the claims, the verification results, and the
            evidence.

    Returns:
        The checked answer, the evidence manifest, and the reason codes.
    """
    answer = data.answer.model_copy(deep=True)
    reason_codes: list[str] = []
    spans_by_id = {span.evidence_id: span for span in data.spans}

    for claim in data.claims:
        if claim.type != "direct":
            continue
        if not _valid_citations(claim, spans_by_id):
            reason_codes.append(REASON_INVALID_CITATION)
            break

    cleaned_claims = []
    removed = False
    for claim in answer.claims:
        text, dropped = _strip_instruction_like_text(claim.text)
        removed = removed or dropped
        cleaned_claims.append(claim.model_copy(update={"text": text}))
    cleaned_text, text_removed = _strip_instruction_like_text(
        answer.answer_text
    )
    removed = removed or text_removed
    if removed:
        reason_codes.append(REASON_INJECTION_REMOVED)

    supported = [
        result
        for result in data.verification_results
        if result.supported and result.label in ("supported", "partially_supported")
    ]
    if not supported and answer.claims:
        reason_codes.append(REASON_EMPTY_ANSWER)

    status = answer.final_status
    if not answer.claims and status == "answered":
        status = "insufficient_evidence"
        reason_codes.append(REASON_STATUS_MISMATCH)
    if answer.conflicts and status == "answered":
        status = "conflicting_evidence"
        reason_codes.append(REASON_STATUS_MISMATCH)

    answer = answer.model_copy(
        update={
            "answer_text": cleaned_text,
            "claims": cleaned_claims,
            "final_status": status,
            "reason_codes": [*answer.reason_codes, *reason_codes],
        }
    )
    logger_codes = {
        "claim_count": len(answer.claims),
        "evidence_count": len(answer.evidence),
        "finding_count": len(data.injection_findings),
    }
    return FinalGateOutput(
        answer=answer,
        evidence_manifest=manifest_for(answer),
        reason_codes=reason_codes,
        passed=REASON_INVALID_CITATION not in reason_codes,
    )
