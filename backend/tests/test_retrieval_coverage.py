"""Tests for the LLM-free coverage helper."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.retrieval.coverage import assess_coverage
from app.retrieval.schemas import EvidenceSpan


def _span(
    *,
    document_id=None,
    score: float | None = 0.5,
    quality_flags: list[str] | None = None,
) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=uuid4(),
        case_id=uuid4(),
        document_id=document_id or uuid4(),
        document_version_id=None,
        title="pleading.pdf",
        page_start=1,
        page_end=1,
        section_path=[],
        quote_or_span="text",
        extraction_method="native",
        quality_status="ready",
        ocr_confidence=None,
        source_anchor=None,
        retrieval_method="hybrid",
        retrieval_rank=1,
        quality_flags=list(quality_flags or []),
        retrieval_score=score,
    )


def test_no_spans_gives_empty_metrics() -> None:
    metrics = assess_coverage("who is liable", [])

    assert metrics.span_count == 0
    assert metrics.documents_covered == 0
    assert metrics.top_rank_score_spread == 0.0
    assert metrics.low_quality_span_count == 0


def test_metrics_count_spans_documents_and_low_quality() -> None:
    first_document = uuid4()
    second_document = uuid4()
    spans = [
        _span(document_id=first_document, score=0.9),
        _span(document_id=first_document, score=0.5, quality_flags=["low_ocr_confidence"]),
        _span(document_id=second_document, score=0.4),
    ]

    metrics = assess_coverage("who is liable for the damages", spans)

    assert metrics.span_count == 3
    assert metrics.documents_covered == 2
    assert metrics.low_quality_span_count == 1
    assert metrics.top_rank_score_spread == pytest.approx(0.5)
    assert metrics.question_terms == 6


def test_score_spread_only_covers_the_top_ranked_spans() -> None:
    spans = [
        _span(score=0.9),
        _span(score=0.5),
        _span(score=0.4),
        _span(score=0.1),
    ]

    metrics = assess_coverage("question", spans, top_n=2)

    assert metrics.top_rank_score_spread == pytest.approx(0.4)


def test_a_single_span_has_no_spread() -> None:
    metrics = assess_coverage("question", [_span(score=0.7)])

    assert metrics.top_rank_score_spread == 0.0
