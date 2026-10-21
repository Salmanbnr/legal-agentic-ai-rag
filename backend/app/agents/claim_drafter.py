"""The claim drafter: turn extracted facts into short citable claims.

Every claim points at the evidence it came from. A fact the analyst marked as
certain becomes a direct claim, and a fact the analyst marked as uncertain or
ambiguous becomes an inference claim so the answer stays qualified. A fact
the analyst could not settle is not a claim at all; it stays a gap.
"""

from __future__ import annotations

import re
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.agents.state import DraftClaim, ExtractedFact
from app.retrieval.schemas import EvidenceSpan

ClaimType = Literal["direct", "inference"]

REASON_NO_FACTS = "NO_FACTS_TO_DRAFT"
REASON_UNCERTAIN_AS_INFERENCE = "UNCERTAIN_FACT_AS_INFERENCE"
REASON_SPANS_AS_CLAIMS = "EVIDENCE_PHRASES_USED_AS_CLAIMS"


class ClaimDraftInput(BaseModel):
    """Everything the drafter needs for one question."""

    question: str = ""
    facts: list[ExtractedFact] = Field(default_factory=list)
    spans: list[EvidenceSpan] = Field(default_factory=list)


class ClaimDraftOutput(BaseModel):
    """The drafted claims."""

    claims: list[DraftClaim] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


def _claim_text(fact: ExtractedFact) -> str:
    """Return the short claim text for one fact."""
    return fact.text.strip()


def _first_sentence(text: str) -> str:
    """Return the first sentence of a passage."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return parts[0] if parts else text.strip()


def draft_claims(data: ClaimDraftInput) -> ClaimDraftOutput:
    """Turn the extracted facts into short claims.

    When the facts analyst was skipped, the evidence passages themselves
    become the claims: the first sentence of a passage, quoting the whole
    passage, so the citation can still be checked.

    Args:
        data: The question, the extracted facts, and the evidence.

    Returns:
        The drafted claims, each with its evidence ids and its type.
    """
    claims: list[DraftClaim] = []
    reason_codes: list[str] = []
    for fact in data.facts:
        if not fact.evidence_ids:
            continue
        text = _claim_text(fact)
        if not text:
            continue
        if fact.uncertainty == "unknown":
            continue
        if fact.uncertainty in ("uncertain", "ambiguous"):
            if REASON_UNCERTAIN_AS_INFERENCE not in reason_codes:
                reason_codes.append(REASON_UNCERTAIN_AS_INFERENCE)
            claims.append(
                DraftClaim(
                    id=f"claim-{len(claims) + 1}",
                    text=f"It appears that {text[0].lower()}{text[1:]}",
                    evidence_ids=list(fact.evidence_ids),
                    type="inference",
                    subquestion_id=fact.subquestion_id,
                    quote=fact.quote,
                )
            )
            continue
        claims.append(
            DraftClaim(
                id=f"claim-{len(claims) + 1}",
                text=text,
                evidence_ids=list(fact.evidence_ids),
                type="direct",
                subquestion_id=fact.subquestion_id,
                quote=fact.quote,
            )
        )
    if not claims and data.spans:
        if REASON_SPANS_AS_CLAIMS not in reason_codes:
            reason_codes.append(REASON_SPANS_AS_CLAIMS)
        for span in data.spans:
            text = _first_sentence(span.quote_or_span)
            if not text:
                continue
            claims.append(
                DraftClaim(
                    id=f"claim-{len(claims) + 1}",
                    text=text,
                    evidence_ids=[span.evidence_id],
                    type="direct",
                    quote=span.quote_or_span,
                )
            )
    if not claims and not reason_codes:
        reason_codes.append(REASON_NO_FACTS)
    return ClaimDraftOutput(claims=claims, reason_codes=reason_codes)


def claim_evidence_ids(claim: DraftClaim) -> list[UUID]:
    """Return the evidence ids a claim relies on."""
    return list(claim.evidence_ids)
