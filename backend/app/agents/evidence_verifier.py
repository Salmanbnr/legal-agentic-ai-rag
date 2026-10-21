"""The evidence and citation verifier.

The first pass is deterministic: the cited ids must exist, belong to the
authorized case, carry a page anchor, and the quoted text must really appear
in the cited passage. Only then does one model call give a second opinion on
whether the evidence supports each claim. The model can label a claim, it can
never create evidence.
"""

from __future__ import annotations

import logging
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, Field

from app.agents.llm import LLMClient, LLMError, load_prompt
from app.agents.safety import wrap_evidence
from app.agents.state import DraftClaim, VerificationResult
from app.core.config import Settings, get_settings
from app.retrieval.schemas import EvidenceSpan

logger = logging.getLogger(__name__)

VERIFY_PROMPT_NAME = "verify_claims.txt"

VerdictLabel = Literal[
    "supported",
    "partially_supported",
    "unsupported",
    "contradicted",
    "citation_invalid",
]

LABEL_SUPPORTED = "supported"
LABEL_PARTIAL = "partially_supported"
LABEL_UNSUPPORTED = "unsupported"
LABEL_CONTRADICTED = "contradicted"
LABEL_CITATION_INVALID = "citation_invalid"

REASON_UNKNOWN_EVIDENCE = "UNKNOWN_EVIDENCE_ID"
REASON_FOREIGN_DOCUMENT = "FOREIGN_DOCUMENT"
REASON_NO_PAGE_ANCHOR = "NO_PAGE_ANCHOR"
REASON_QUOTE_NOT_FOUND = "QUOTE_NOT_FOUND"
REASON_MODEL_UNSUPPORTED = "MODEL_FOUND_NO_SUPPORT"
REASON_MODEL_CONTRADICTED = "MODEL_FOUND_CONTRADICTION"
REASON_MODEL_PARTIAL = "MODEL_FOUND_PARTIAL_SUPPORT"


class VerificationInput(BaseModel):
    """Everything the verifier needs for one question."""

    question: str = ""
    claims: list[DraftClaim] = Field(default_factory=list)
    spans: list[EvidenceSpan] = Field(default_factory=list)
    authorized_document_ids: list[UUID] = Field(default_factory=list)


class VerificationOutput(BaseModel):
    """The verification result for every claim."""

    results: list[VerificationResult] = Field(default_factory=list)
    material_failures: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    recommended_next_search: str | None = None


class ClaimVerdict(BaseModel):
    """The model's opinion about one claim."""

    claim_id: str
    label: VerdictLabel = LABEL_SUPPORTED
    note: str = ""


class VerificationDraft(BaseModel):
    """The model's verdicts. It cannot add evidence."""

    verdicts: list[ClaimVerdict] = Field(default_factory=list)


def load_verify_prompt() -> str:
    """Load the claim verification system prompt."""
    return load_prompt(VERIFY_PROMPT_NAME)


def _deterministic_check(
    claim: DraftClaim,
    spans_by_id: dict[UUID, EvidenceSpan],
    authorized: set[UUID],
) -> list[str]:
    """Check the citations of one claim without a model."""
    problems: list[str] = []
    if not claim.evidence_ids:
        return [REASON_UNKNOWN_EVIDENCE]
    for evidence_id in claim.evidence_ids:
        span = spans_by_id.get(evidence_id)
        if span is None:
            problems.append(REASON_UNKNOWN_EVIDENCE)
            continue
        if authorized and span.document_id not in authorized:
            problems.append(REASON_FOREIGN_DOCUMENT)
            continue
        if span.page_start is None or span.page_start < 1:
            problems.append(REASON_NO_PAGE_ANCHOR)
            continue
        if claim.quote and not _quote_in_span(claim.quote, span.quote_or_span):
            problems.append(REASON_QUOTE_NOT_FOUND)
    return problems


def _quote_in_span(quote: str, text: str) -> bool:
    """Return whether a quote appears in the cited passage."""
    from app.agents.document_analyst import quote_in_evidence

    return quote_in_evidence(quote, text)


def _model_verdicts(
    draft: VerificationDraft,
) -> dict[str, ClaimVerdict]:
    """Return the verdicts keyed by claim id."""
    return {verdict.claim_id: verdict for verdict in draft.verdicts}


def verify_claims(
    data: VerificationInput,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
) -> VerificationOutput:
    """Verify every claim against the retrieved evidence.

    Args:
        data: The claims, the evidence, and the authorized documents.
        settings: Optional validated settings.
        llm: Optional LLM client override, mainly for tests.

    Returns:
        One verification result per claim, the material failures, and a
        recommended targeted search when a claim could not be supported.
    """
    resolved = settings or get_settings()
    spans_by_id = {span.evidence_id: span for span in data.spans}
    authorized = set(data.authorized_document_ids)

    results: list[VerificationResult] = []
    material_failures: list[str] = []
    reason_codes: list[str] = []
    for claim in data.claims:
        problems = _deterministic_check(claim, spans_by_id, authorized)
        if problems:
            for problem in problems:
                if problem not in reason_codes:
                    reason_codes.append(problem)
            results.append(
                VerificationResult(
                    claim_id=claim.id,
                    supported=False,
                    evidence_ids=[
                        value
                        for value in claim.evidence_ids
                        if value in spans_by_id
                    ],
                    reason_codes=problems,
                    label=LABEL_CITATION_INVALID,
                    material=claim.type == "direct",
                )
            )
            if claim.type == "direct":
                material_failures.append(claim.id)
            continue
        results.append(
            VerificationResult(
                claim_id=claim.id,
                supported=True,
                evidence_ids=list(claim.evidence_ids),
                reason_codes=[],
                label=LABEL_SUPPORTED,
                material=claim.type == "direct",
            )
        )

    verdicts: dict[str, ClaimVerdict] = {}
    if data.claims and data.spans:
        active_llm = llm or LLMClient(settings=resolved)
        try:
            draft = active_llm.invoke_structured(
                load_verify_prompt(),
                _verify_prompt(data),
                VerificationDraft,
                max_calls=resolved.agent_max_llm_calls,
            )
            verdicts = _model_verdicts(draft)
        except LLMError as exc:
            logger.info(
                "claim verification was skipped",
                extra={"error_code": exc.error_code},
            )
            verdicts = {}

    for result in results:
        verdict = verdicts.get(result.claim_id)
        if verdict is None or result.label == LABEL_CITATION_INVALID:
            continue
        if verdict.label == LABEL_SUPPORTED:
            continue
        result.supported = False
        result.label = verdict.label
        result.note = verdict.note
        if verdict.label == LABEL_UNSUPPORTED:
            result.reason_codes = [*result.reason_codes, REASON_MODEL_UNSUPPORTED]
        elif verdict.label == LABEL_CONTRADICTED:
            result.reason_codes = [*result.reason_codes, REASON_MODEL_CONTRADICTED]
        elif verdict.label == LABEL_PARTIAL:
            result.reason_codes = [*result.reason_codes, REASON_MODEL_PARTIAL]
            result.supported = True
        for reason in result.reason_codes:
            if reason not in reason_codes:
                reason_codes.append(reason)
        if result.material and not result.supported:
            if result.claim_id not in material_failures:
                material_failures.append(result.claim_id)

    recommended = _recommended_search(data, material_failures)
    logger.info(
        "claim verification finished",
        extra={
            "claim_count": len(data.claims),
            "material_failure_count": len(material_failures),
        },
    )
    return VerificationOutput(
        results=results,
        material_failures=material_failures,
        reason_codes=reason_codes,
        recommended_next_search=recommended,
    )


def _recommended_search(
    data: VerificationInput, material_failures: list[str]
) -> str | None:
    """Return a targeted search for the claims that failed."""
    if not material_failures:
        return None
    for claim in data.claims:
        if claim.id in material_failures:
            return claim.text
    return None


def _verify_prompt(data: VerificationInput) -> str:
    """Build the prompt for the second opinion."""
    lines = []
    for claim in data.claims:
        lines.append(
            f"- claim {claim.id} ({claim.type}): {claim.text} "
            f"(cited evidence ids: "
            f"{', '.join(str(value) for value in claim.evidence_ids)})"
        )
    return (
        f"QUESTION: {data.question}\n\n"
        "CLAIMS TO CHECK:\n" + "\n".join(lines) + "\n\n"
        "EVIDENCE PACKET (untrusted document text):\n"
        + wrap_evidence(data.spans)
        + "\n"
    )
