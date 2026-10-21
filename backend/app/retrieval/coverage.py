"""Simple, LLM-free coverage signals for a set of evidence spans.

The helper answers "how much evidence did this search surface?" without asking
a model anything. It only counts what the backend already knows.
"""

from __future__ import annotations

from typing import Sequence

from app.retrieval.schemas import CoverageMetrics, EvidenceSpan

LOW_QUALITY_PAGE_FLAGS = (
    "low_ocr_confidence",
    "ocr_failed",
    "empty_page",
    "possible_handwriting",
)

DEFAULT_TOP_N = 3


def _question_terms(question: str) -> int:
    """Return the number of distinct terms in a question."""
    return len({term for term in (question or "").lower().split() if term})


def assess_coverage(
    question: str,
    spans: Sequence[EvidenceSpan],
    *,
    top_n: int = DEFAULT_TOP_N,
) -> CoverageMetrics:
    """Summarize how much evidence a search returned.

    Args:
        question: The question that was searched for.
        spans: Evidence spans returned by the search.
        top_n: How many of the best-ranked spans take part in the score
            spread.

    Returns:
        Coverage metrics: span count, documents covered, the score spread
        across the top-ranked spans, and how many spans come from low quality
        pages.
    """
    if not spans:
        return CoverageMetrics(question_terms=_question_terms(question))

    scores = [
        span.retrieval_score for span in spans if span.retrieval_score is not None
    ]
    top_scores = scores[: max(1, top_n)]
    spread = (max(top_scores) - min(top_scores)) if len(top_scores) > 1 else 0.0

    low_quality = sum(
        1
        for span in spans
        if any(flag in LOW_QUALITY_PAGE_FLAGS for flag in span.quality_flags)
    )

    return CoverageMetrics(
        span_count=len(spans),
        documents_covered=len({span.document_id for span in spans}),
        top_rank_score_spread=round(float(spread), 6),
        low_quality_span_count=low_quality,
        question_terms=_question_terms(question),
    )
