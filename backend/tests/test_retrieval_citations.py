"""Tests for deterministic citation validation."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.retrieval.answer_schema import AnswerClaim, AnswerResponse
from app.retrieval.citations import (
    REASON_INSUFFICIENT,
    REASON_NO_EVIDENCE,
    REASON_NO_SUPPORTED_CLAIMS,
    REASON_UNSUPPORTED_CLAIMS,
    build_insufficient_evidence_answer,
    validate_citations,
)
from app.retrieval.schemas import EvidenceSpan


def _span(
    *,
    page_start: int | None = 1,
    page_end: int | None = 1,
    title: str = "pleading.pdf",
) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=uuid4(),
        case_id=uuid4(),
        document_id=uuid4(),
        document_version_id=None,
        title=title,
        page_start=page_start,
        page_end=page_end,
        section_path=[],
        quote_or_span="Quoted text.",
        extraction_method="native",
        quality_status="ready",
        ocr_confidence=None,
        source_anchor=None,
        retrieval_method="hybrid",
        retrieval_rank=1,
        quality_flags=[],
    )


def test_a_valid_claim_is_kept() -> None:
    span = _span()
    answer = AnswerResponse(
        answer_text="The rent is unpaid.",
        claims=[
            AnswerClaim(
                text="The rent is unpaid.",
                evidence_ids=[span.evidence_id],
                type="direct",
            )
        ],
        final_status="answered",
    )

    check = validate_citations(answer, [span], question="Is the rent paid?")

    assert len(check.answer.claims) == 1
    assert check.answer.evidence == [span]
    assert check.answer.final_status == "answered"
    assert check.removed_claim_count == 0


def test_an_unknown_evidence_id_is_stripped() -> None:
    span = _span()
    unknown = uuid4()
    answer = AnswerResponse(
        answer_text="Two statements.",
        claims=[
            AnswerClaim(
                text="Supported.",
                evidence_ids=[span.evidence_id, unknown],
                type="direct",
            )
        ],
    )

    check = validate_citations(answer, [span], question="question")

    assert check.answer.claims[0].evidence_ids == [span.evidence_id]
    assert check.removed_evidence_ids == [unknown]
    assert REASON_UNSUPPORTED_CLAIMS in check.answer.reason_codes


def test_a_claim_with_only_unknown_ids_is_removed() -> None:
    span = _span()
    answer = AnswerResponse(
        answer_text="Unsupported.",
        claims=[
            AnswerClaim(text="Unsupported.", evidence_ids=[uuid4()]),
        ],
        final_status="answered",
    )

    check = validate_citations(answer, [span], question="question")

    assert check.answer.claims == []
    assert check.removed_claim_count == 1
    assert check.answer.final_status == "insufficient_evidence"
    assert REASON_NO_SUPPORTED_CLAIMS in check.answer.reason_codes


def test_a_claim_without_evidence_ids_is_removed() -> None:
    span = _span()
    answer = AnswerResponse(
        answer_text="No citation.",
        claims=[AnswerClaim(text="No citation.", evidence_ids=[])],
    )

    check = validate_citations(answer, [span], question="question")

    assert check.answer.claims == []
    assert check.removed_claim_count == 1


def test_a_span_without_a_page_anchor_cannot_be_cited() -> None:
    span = _span(page_start=None, page_end=None)
    answer = AnswerResponse(
        answer_text="Claim.",
        claims=[AnswerClaim(text="Claim.", evidence_ids=[span.evidence_id])],
    )

    check = validate_citations(answer, [span], question="question")

    assert check.answer.claims == []
    assert check.answer.final_status == "insufficient_evidence"


def test_removed_claims_downgrade_an_answered_status() -> None:
    span = _span()
    answer = AnswerResponse(
        answer_text="Mixed.",
        claims=[
            AnswerClaim(text="Supported.", evidence_ids=[span.evidence_id]),
            AnswerClaim(text="Unsupported.", evidence_ids=[uuid4()]),
        ],
        final_status="answered",
    )

    check = validate_citations(answer, [span], question="question")

    assert check.answer.final_status == "partial"
    assert len(check.answer.claims) == 1
    assert any("removed" in item for item in check.answer.limitations)


def test_the_answer_only_keeps_the_evidence_it_used() -> None:
    first = _span()
    second = _span()
    answer = AnswerResponse(
        answer_text="One statement.",
        claims=[
            AnswerClaim(text="One statement.", evidence_ids=[first.evidence_id])
        ],
    )

    check = validate_citations(answer, [first, second], question="question")

    assert check.answer.evidence == [first]


def test_no_evidence_gives_a_clean_abstention() -> None:
    answer = AnswerResponse(
        answer_text="Something.",
        claims=[AnswerClaim(text="Something.", evidence_ids=[uuid4()])],
    )

    check = validate_citations(answer, [], question="Who is liable?")

    assert check.answer.final_status == "insufficient_evidence"
    assert check.answer.claims == []
    assert check.answer.evidence == []
    assert REASON_NO_EVIDENCE in check.answer.reason_codes
    assert "Who is liable?" in check.answer.answer_text


def test_the_abstention_never_claims_something_is_absent() -> None:
    span = _span(title="lease.pdf", page_start=4, page_end=4)

    answer = build_insufficient_evidence_answer(
        "Is there a break clause?", [span]
    )

    text = answer.answer_text.lower()
    assert "does not exist" not in text
    assert "no such clause" not in text
    assert "may still exist" in answer.answer_text
    assert "lease.pdf" in answer.answer_text
    assert "page 4" in answer.answer_text
    assert answer.final_status == "insufficient_evidence"
    assert REASON_INSUFFICIENT in answer.reason_codes


def test_the_abstention_mentions_the_question_and_the_search_scope() -> None:
    answer = build_insufficient_evidence_answer("What is the rent?", [])

    assert "What is the rent?" in answer.answer_text
    assert "no documents were searched" in answer.answer_text
    assert REASON_NO_EVIDENCE in answer.reason_codes
