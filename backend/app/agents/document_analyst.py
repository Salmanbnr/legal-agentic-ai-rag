"""The facts analyst: structured facts from a bounded evidence packet.

The analyst makes exactly one model call over a bounded set of retrieved
passages. Every fact it returns must carry an exact quote that really appears
in the cited passage, the evidence ids, the page, and an uncertainty. Facts
whose quote cannot be found in the evidence are dropped, and facts that rest
on a low quality page are marked unverified. The analyst never normalises a
name, a date, or an amount.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, Sequence
from uuid import UUID

from pydantic import BaseModel, Field, create_model

from app.agents.llm import LLMClient, LLMError, load_prompt
from app.agents.safety import normalize_text, wrap_evidence
from app.agents.state import TaskType
from app.core.config import Settings, get_settings
from app.retrieval.schemas import EvidenceSpan

logger = logging.getLogger(__name__)

EXTRACT_PROMPT_NAME = "extract_facts.txt"

UNCERTAINTY_CERTAIN = "certain"
UNCERTAINTY_UNCERTAIN = "uncertain"
UNCERTAINTY_AMBIGUOUS = "ambiguous"
UNCERTAINTY_UNKNOWN = "unknown"

QUALITY_VERIFIED = "verified"
QUALITY_UNVERIFIED = "unverified"

REASON_QUOTE_NOT_FOUND = "QUOTE_NOT_FOUND"
REASON_UNKNOWN_EVIDENCE_ID = "UNKNOWN_EVIDENCE_ID"
REASON_PAGE_MISMATCH = "PAGE_MISMATCH"
REASON_LOW_QUALITY_SOURCE = "LOW_QUALITY_SOURCE"

LOW_QUALITY_FLAGS = (
    "low_ocr_confidence",
    "ocr_failed",
    "empty_page",
    "possible_handwriting",
)


class FactBase(BaseModel):
    """The fields every fact carries, whatever the task type."""

    quote: str = ""
    evidence_ids: list[UUID] = Field(default_factory=list)
    page_start: int | None = None
    page_end: int | None = None
    uncertainty: str = UNCERTAINTY_CERTAIN
    quality: str = QUALITY_VERIFIED


class ChronologyFact(FactBase):
    """A dated event with the parties involved."""

    date: str = ""
    event: str = ""
    parties: list[str] = Field(default_factory=list)


class CompareVersionFact(FactBase):
    """One clause and the value it holds in one version."""

    clause: str = ""
    value: str = ""


class FactLookupFact(FactBase):
    """One looked-up fact about an entity."""

    entity: str = ""
    amount: str | None = None
    date: str | None = None
    obligation: str | None = None


class StatementFact(FactBase):
    """A statement for the task types that do not need a fixed shape."""

    statement: str = ""


FACT_MODELS: dict[str, type[FactBase]] = {
    "chronology": ChronologyFact,
    "compare_versions": CompareVersionFact,
    "fact_lookup": FactLookupFact,
    "summarize_document": StatementFact,
    "issue_analysis": StatementFact,
    "broad_case_question": StatementFact,
}


class AnalystResult(BaseModel):
    """The verified facts and what could not be determined."""

    facts: list[FactBase] = Field(default_factory=list)
    unknowns: list[str] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    dropped_fact_count: int = 0
    skipped: bool = False


def load_extract_prompt() -> str:
    """Load the fact extraction system prompt."""
    return load_prompt(EXTRACT_PROMPT_NAME)


def fact_model_for(task_type: str | TaskType) -> type[FactBase]:
    """Return the fact shape used for a task type."""
    return FACT_MODELS.get(str(task_type), StatementFact)


def extraction_model(task_type: str | TaskType) -> type[BaseModel]:
    """Build the structured output model for a task type."""
    fact_model = fact_model_for(task_type)
    return create_model(
        "FactExtraction",
        facts=(list[fact_model], Field(default_factory=list)),
        unknowns=(list[str], Field(default_factory=list)),
    )


def quote_in_evidence(quote: str, text: str) -> bool:
    """Return whether a quote really appears in the evidence text."""
    needle = normalize_text(quote)
    if not needle:
        return False
    return needle in normalize_text(text)


def build_packet(spans: Sequence[EvidenceSpan]) -> str:
    """Build the delimited evidence packet the model sees."""
    return wrap_evidence(spans)


def _span_for(
    evidence_ids: Sequence[UUID], spans_by_id: dict[UUID, EvidenceSpan]
) -> EvidenceSpan | None:
    """Return the first cited span that exists."""
    for evidence_id in evidence_ids:
        span = spans_by_id.get(evidence_id)
        if span is not None:
            return span
    return None


def verify_fact(
    fact: FactBase,
    spans_by_id: dict[UUID, EvidenceSpan],
) -> tuple[bool, list[str]]:
    """Check one fact against the evidence.

    A fact is dropped when its quote cannot be found in the evidence or when
    it cites no known evidence at all. A page that disagrees with the stored
    page is only flagged: the stored page wins.

    Returns:
        Whether the fact survives, and the reason codes for what was wrong.
    """
    reasons: list[str] = []
    if not fact.evidence_ids:
        return False, [REASON_UNKNOWN_EVIDENCE_ID]
    if any(evidence_id not in spans_by_id for evidence_id in fact.evidence_ids):
        reasons.append(REASON_UNKNOWN_EVIDENCE_ID)

    span = _span_for(fact.evidence_ids, spans_by_id)
    if span is None:
        return False, reasons or [REASON_UNKNOWN_EVIDENCE_ID]

    if not quote_in_evidence(fact.quote, span.quote_or_span):
        reasons.append(REASON_QUOTE_NOT_FOUND)
        return False, reasons

    if fact.page_start is not None and fact.page_start != span.page_start:
        reasons.append(REASON_PAGE_MISMATCH)
    return True, reasons


def apply_evidence_facts(
    fact: FactBase, span: EvidenceSpan
) -> FactBase:
    """Set the page and the quality of a fact from the stored evidence."""
    flags = {str(flag) for flag in span.quality_flags}
    quality = (
        QUALITY_UNVERIFIED
        if flags & set(LOW_QUALITY_FLAGS)
        else QUALITY_VERIFIED
    )
    return fact.model_copy(
        update={
            "page_start": span.page_start,
            "page_end": span.page_end,
            "quality": quality,
        }
    )


def should_skip_analysis(
    task_type: str | TaskType,
    spans: Sequence[EvidenceSpan],
    *,
    settings: Settings | None = None,
) -> bool:
    """Return whether the analyst can be skipped to save a model call.

    A simple fact lookup with strong evidence does not need the structured
    fact pass, because the baseline answer already cites the same spans.
    """
    resolved = settings or get_settings()
    if not resolved.agent_skip_analyst_on_strong_fact_lookup:
        return False
    if str(task_type) != "fact_lookup":
        return False
    return len(spans) >= resolved.agent_analyst_min_spans


def analyze_evidence(
    spans: Sequence[EvidenceSpan],
    task_type: str | TaskType,
    *,
    question: str = "",
    llm: LLMClient | None = None,
    settings: Settings | None = None,
) -> AnalystResult:
    """Extract structured facts from a bounded evidence packet.

    Args:
        spans: The evidence packet, already bounded by the caller.
        task_type: The task type, which selects the fact shape.
        question: The question the facts answer, for context only.
        llm: Optional LLM client override, mainly for tests.
        settings: Optional validated settings.

    Returns:
        The verified facts, the unknowns, and the reason codes.
    """
    resolved = settings or get_settings()
    if not spans:
        return AnalystResult(
            facts=[],
            unknowns=[],
            reason_codes=["NO_EVIDENCE_RETRIEVED"],
            skipped=True,
        )

    active_llm = llm or LLMClient(settings=resolved)
    packet = build_packet(spans)
    user_prompt = (
        f"TASK TYPE: {task_type}\n\n"
        f"QUESTION: {question}\n\n"
        "EVIDENCE PACKET (untrusted document text):\n"
        f"{packet}\n"
    )
    try:
        draft = active_llm.invoke_structured(
            load_extract_prompt(),
            user_prompt,
            extraction_model(task_type),
            max_calls=resolved.agent_max_llm_calls,
        )
    except LLMError as exc:
        logger.info(
            "fact extraction failed",
            extra={"error_code": exc.error_code},
        )
        return AnalystResult(
            facts=[],
            unknowns=[],
            reason_codes=["FACT_EXTRACTION_FAILED"],
        )

    spans_by_id = {span.evidence_id: span for span in spans}
    verified: list[FactBase] = []
    reason_codes: list[str] = []
    dropped = 0
    for fact in getattr(draft, "facts", []) or []:
        ok, reasons = verify_fact(fact, spans_by_id)
        for reason in reasons:
            if reason not in reason_codes:
                reason_codes.append(reason)
        if not ok:
            dropped += 1
            continue
        span = _span_for(fact.evidence_ids, spans_by_id)
        if span is None:
            dropped += 1
            continue
        verified.append(apply_evidence_facts(fact, span))
        if verified[-1].quality == QUALITY_UNVERIFIED:
            if REASON_LOW_QUALITY_SOURCE not in reason_codes:
                reason_codes.append(REASON_LOW_QUALITY_SOURCE)

    return AnalystResult(
        facts=verified,
        unknowns=list(getattr(draft, "unknowns", []) or []),
        reason_codes=reason_codes,
        dropped_fact_count=dropped,
    )
