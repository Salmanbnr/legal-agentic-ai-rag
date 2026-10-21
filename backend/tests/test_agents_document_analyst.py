"""Tests for the facts analyst."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from app.agents.document_analyst import (
    QUALITY_UNVERIFIED,
    QUALITY_VERIFIED,
    REASON_LOW_QUALITY_SOURCE,
    REASON_PAGE_MISMATCH,
    REASON_QUOTE_NOT_FOUND,
    REASON_UNKNOWN_EVIDENCE_ID,
    UNCERTAINTY_AMBIGUOUS,
    FactLookupFact,
    analyze_evidence,
    extraction_model,
    fact_model_for,
    quote_in_evidence,
    should_skip_analysis,
)
from app.core.config import Settings
from app.retrieval.schemas import EvidenceSpan

TIMESTAMP = "2026-01-01T00:00:00+00:00"


def _span(
    text: str,
    *,
    page_start: int = 3,
    page_end: int = 4,
    quality_flags: list[Any] | None = None,
    document_id=None,
) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=uuid4(),
        case_id=uuid4(),
        document_id=document_id or uuid4(),
        document_version_id=None,
        title="lease.pdf",
        page_start=page_start,
        page_end=page_end,
        section_path=[],
        quote_or_span=text,
        extraction_method="native",
        quality_status="ready",
        ocr_confidence=None,
        source_anchor=f"doc:1|p:{page_start}-{page_end}|c:0",
        retrieval_method="hybrid",
        retrieval_rank=1,
        quality_flags=list(quality_flags or []),
    )


class StubLLM:
    """LLM stand-in that returns a prepared extraction."""

    def __init__(self, draft: Any = None) -> None:
        """Create the stub with its extraction."""
        self.draft = draft
        self.calls: list[tuple[str, str]] = []

    def invoke_structured(
        self, system_prompt, user_prompt, schema, *, max_calls=None
    ):
        """Return the prepared extraction."""
        self.calls.append((system_prompt, user_prompt))
        if self.draft is None:
            from app.agents.llm import LLMError

            raise LLMError("LLM_INVALID_STRUCTURED_OUTPUT")
        return self.draft


def _extraction(facts: list[Any], unknowns: list[str] | None = None):
    """Build an extraction draft for a task type."""
    model = extraction_model("fact_lookup")
    return model(facts=facts, unknowns=unknowns or [])


def test_a_fabricated_quote_is_rejected() -> None:
    span = _span("The rent was 1,250 pounds per month.")
    draft = _extraction(
        [
            FactLookupFact(
                entity="The tenant",
                amount="9,999 pounds",
                quote="The rent was 9,999 pounds per month.",
                evidence_ids=[span.evidence_id],
                page_start=3,
                page_end=4,
            )
        ]
    )

    result = analyze_evidence(
        [span],
        "fact_lookup",
        question="How much was the rent?",
        llm=StubLLM(draft),
        settings=Settings(_env_file=None),
    )

    assert result.facts == []
    assert result.dropped_fact_count == 1
    assert REASON_QUOTE_NOT_FOUND in result.reason_codes


def test_a_real_quote_is_kept_with_the_stored_page() -> None:
    span = _span("The rent was 1,250 pounds per month.")
    draft = _extraction(
        [
            FactLookupFact(
                entity="The tenant",
                amount="1,250 pounds",
                quote="The rent was 1,250 pounds per month.",
                evidence_ids=[span.evidence_id],
                page_start=9,
                page_end=9,
            )
        ]
    )

    result = analyze_evidence(
        [span],
        "fact_lookup",
        question="How much was the rent?",
        llm=StubLLM(draft),
        settings=Settings(_env_file=None),
    )

    assert len(result.facts) == 1
    fact = result.facts[0]
    assert fact.page_start == 3
    assert fact.page_end == 4
    assert fact.quality == QUALITY_VERIFIED
    # The model gave page 9, the stored page wins.
    assert REASON_PAGE_MISMATCH in result.reason_codes


def test_an_unknown_evidence_id_is_rejected() -> None:
    span = _span("The rent was 1,250 pounds per month.")
    draft = _extraction(
        [
            FactLookupFact(
                entity="The tenant",
                quote="The rent was 1,250 pounds per month.",
                evidence_ids=[uuid4()],
            )
        ]
    )

    result = analyze_evidence(
        [span],
        "fact_lookup",
        llm=StubLLM(draft),
        settings=Settings(_env_file=None),
    )

    assert result.facts == []
    assert REASON_UNKNOWN_EVIDENCE_ID in result.reason_codes


def test_ambiguous_dates_return_two_candidates() -> None:
    span = _span("The notice was served on 3 May or 4 May 2021.")
    draft = _extraction(
        [
            FactLookupFact(
                entity="The notice",
                date="3 May 2021",
                quote="The notice was served on 3 May",
                evidence_ids=[span.evidence_id],
                uncertainty=UNCERTAINTY_AMBIGUOUS,
            ),
            FactLookupFact(
                entity="The notice",
                date="4 May 2021",
                quote="or 4 May 2021",
                evidence_ids=[span.evidence_id],
                uncertainty=UNCERTAINTY_AMBIGUOUS,
            ),
        ]
    )

    result = analyze_evidence(
        [span],
        "fact_lookup",
        llm=StubLLM(draft),
        settings=Settings(_env_file=None),
    )

    assert len(result.facts) == 2
    dates = sorted(fact.date for fact in result.facts)
    assert dates == ["3 May 2021", "4 May 2021"]
    assert all(
        fact.uncertainty == UNCERTAINTY_AMBIGUOUS for fact in result.facts
    )


def test_a_low_quality_page_marks_the_fact_unverified() -> None:
    span = _span(
        "The rent was 1,250 pounds per month.",
        quality_flags=["low_ocr_confidence"],
    )
    draft = _extraction(
        [
            FactLookupFact(
                entity="The tenant",
                amount="1,250 pounds",
                quote="The rent was 1,250 pounds per month.",
                evidence_ids=[span.evidence_id],
            )
        ]
    )

    result = analyze_evidence(
        [span],
        "fact_lookup",
        llm=StubLLM(draft),
        settings=Settings(_env_file=None),
    )

    assert len(result.facts) == 1
    assert result.facts[0].quality == QUALITY_UNVERIFIED
    assert REASON_LOW_QUALITY_SOURCE in result.reason_codes


def test_an_ocr_looking_value_is_never_normalised() -> None:
    span = _span("The claim was for 3,75O pounds.")
    draft = _extraction(
        [
            FactLookupFact(
                entity="The claim",
                amount="3,750 pounds",
                quote="The claim was for 3,75O pounds.",
                evidence_ids=[span.evidence_id],
            )
        ]
    )

    result = analyze_evidence(
        [span],
        "fact_lookup",
        llm=StubLLM(draft),
        settings=Settings(_env_file=None),
    )

    assert len(result.facts) == 1
    # The analyst does not rewrite what the document says.
    assert result.facts[0].quote == "The claim was for 3,75O pounds."


def test_the_unknowns_are_kept() -> None:
    span = _span("The rent was 1,250 pounds per month.")
    draft = _extraction([], unknowns=["The date of the agreement."])

    result = analyze_evidence(
        [span],
        "fact_lookup",
        llm=StubLLM(draft),
        settings=Settings(_env_file=None),
    )

    assert result.unknowns == ["The date of the agreement."]


def test_the_analyst_is_skipped_for_a_strong_fact_lookup() -> None:
    spans = [_span(f"text {index}") for index in range(3)]

    assert should_skip_analysis(
        "fact_lookup", spans, settings=Settings(_env_file=None)
    )
    assert not should_skip_analysis(
        "chronology", spans, settings=Settings(_env_file=None)
    )
    assert not should_skip_analysis(
        "fact_lookup",
        spans[:1],
        settings=Settings(_env_file=None),
    )


def test_the_skip_flag_can_be_turned_off() -> None:
    settings = Settings(_env_file=None).model_copy(
        update={"agent_skip_analyst_on_strong_fact_lookup": False}
    )
    spans = [_span(f"text {index}") for index in range(3)]

    assert not should_skip_analysis("fact_lookup", spans, settings=settings)


def test_an_empty_packet_is_reported_without_a_model_call() -> None:
    llm = StubLLM()

    result = analyze_evidence(
        [],
        "fact_lookup",
        llm=llm,
        settings=Settings(_env_file=None),
    )

    assert result.skipped is True
    assert result.facts == []
    assert llm.calls == []


def test_the_fact_shape_follows_the_task_type() -> None:
    from app.agents.document_analyst import (
        ChronologyFact,
        CompareVersionFact,
        StatementFact,
    )

    assert fact_model_for("chronology") is ChronologyFact
    assert fact_model_for("compare_versions") is CompareVersionFact
    assert fact_model_for("broad_case_question") is StatementFact
    assert "date" in ChronologyFact.model_fields
    assert "event" in ChronologyFact.model_fields
    assert "parties" in ChronologyFact.model_fields
    assert "clause" in CompareVersionFact.model_fields
    assert "value" in CompareVersionFact.model_fields
    for field in ("entity", "amount", "date", "obligation"):
        assert field in FactLookupFact.model_fields


def test_the_quote_check_ignores_case_and_whitespace() -> None:
    assert quote_in_evidence(
        "the tenant failed to pay",
        "The Tenant failed   to pay the rent.",
    )
    assert not quote_in_evidence("absent text", "The rent was paid.")


def test_the_prompt_treats_the_evidence_as_untrusted() -> None:
    from app.agents.document_analyst import load_extract_prompt

    prompt = load_extract_prompt().lower()

    assert "untrusted" in prompt
    assert "ignore them" in prompt or "never directions" in prompt
    assert "verbatim" in prompt
    assert "never normalise" in prompt or "never normalize" in prompt
