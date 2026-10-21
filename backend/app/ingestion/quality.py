"""Transparent document and page quality assessment.

The report intentionally contains observable counts and page numbers instead of
one opaque quality score. Status mapping uses configurable thresholds and never
rewrites document text, names, dates, amounts, or numbers.
"""

from __future__ import annotations

import re
from collections import Counter
from typing import Any, Literal, Sequence

from pydantic import BaseModel, ConfigDict, Field

from app.core.config import Settings
from app.db.repos import PageCreate
from app.ingestion.ocr import OcrWord

QualityStatus = Literal[
    "ready",
    "ready_with_warnings",
    "needs_review",
    "failed",
]

_DERIVED_FLAGS = {
    "empty_page",
    "low_ocr_confidence",
    "possible_table",
    "possible_handwriting",
    "ocr_failed",
}
_PENDING_FLAGS = {"needs_ocr", "low_text_density"}


class QualityReport(BaseModel):
    """Observable quality facts for one document."""

    model_config = ConfigDict(frozen=True)

    pages_total: int = Field(ge=0)
    pages_extracted: int = Field(ge=0)
    pages_native: int = Field(ge=0)
    pages_ocr: int = Field(ge=0)
    pages_empty: int = Field(ge=0)
    empty_pages: list[int] = Field(default_factory=list)
    low_confidence_pages: list[int] = Field(default_factory=list)
    repeated_text_ratio: float = Field(ge=0.0, le=1.0)
    parser_errors: list[dict[str, Any]] = Field(default_factory=list)


def _page_text(page: PageCreate) -> str:
    """Return the best available page text without modifying it."""
    return (page.raw_text or page.clean_text or "").strip()


def _possible_table_from_text(text: str, min_rows: int) -> bool:
    """Detect repeated multi-column spacing in native or OCR text."""
    if not text.strip():
        return False
    rows = 0
    for line in text.splitlines():
        if "\t" in line:
            columns = [part for part in line.split("\t") if part.strip()]
        else:
            columns = [
                part for part in re.split(r"\s{2,}", line.strip()) if part.strip()
            ]
        if len(columns) >= 2:
            rows += 1
            if rows >= min_rows:
                return True
    return False


def _possible_table_from_words(words: Sequence[OcrWord], min_rows: int) -> bool:
    """Detect repeated wide horizontal gaps in OCR word geometry."""
    if len(words) < 4:
        return False
    groups: dict[tuple[int, int, int], list[OcrWord]] = {}
    for word in words:
        groups.setdefault((word.block, word.paragraph, word.line), []).append(word)

    heights = [word.height for word in words if word.height > 0]
    median_height = sorted(heights)[len(heights) // 2] if heights else 0
    minimum_gap = max(20, int(median_height * 2))
    tabular_rows = 0
    for line_words in groups.values():
        ordered = sorted(line_words, key=lambda item: item.left)
        if len(ordered) < 2:
            continue
        gaps = [
            right.left - (left.left + left.width)
            for left, right in zip(ordered, ordered[1:])
        ]
        if max(gaps) >= minimum_gap:
            tabular_rows += 1
            if tabular_rows >= min_rows:
                return True
    return False


def evaluate_page_flags(
    *,
    text: str | None,
    extraction_method: str,
    ocr_confidence: float | None,
    words: Sequence[OcrWord] = (),
    existing_flags: Sequence[Any] = (),
    settings: Settings,
    ocr_completed: bool = False,
    ocr_failed: bool = False,
) -> list[str]:
    """Return page quality flags derived from text, confidence, and layout.

    ``ocr_completed`` removes the transient ``needs_ocr`` and
    ``low_text_density`` flags once OCR has run. Existing non-quality flags are
    preserved so provenance is never silently discarded.
    """
    flags: list[str] = []
    for raw_flag in existing_flags:
        flag = str(raw_flag)
        if flag in _DERIVED_FLAGS:
            continue
        if ocr_completed and flag in _PENDING_FLAGS:
            continue
        flags.append(flag)

    clean_text = (text or "").strip()
    has_pending_ocr = "needs_ocr" in flags
    if not clean_text and (ocr_completed or not has_pending_ocr):
        flags.append("empty_page")

    if (
        extraction_method == "ocr"
        and ocr_confidence is not None
        and ocr_confidence < settings.ocr_confidence_threshold
    ):
        flags.append("low_ocr_confidence")

    if (
        extraction_method == "ocr"
        and ocr_confidence is not None
        and ocr_confidence < settings.ocr_handwriting_confidence_threshold
        and 0 < len(words) <= settings.ocr_handwriting_max_words
    ):
        flags.append("possible_handwriting")

    if words:
        possible_table = _possible_table_from_words(
            words, settings.ocr_table_min_rows
        )
    else:
        possible_table = _possible_table_from_text(
            clean_text, settings.ocr_table_min_rows
        )
    if possible_table:
        flags.append("possible_table")

    if ocr_failed:
        flags.append("ocr_failed")

    # Preserve deterministic order and remove duplicates.
    return list(dict.fromkeys(flags))


def _line_normalization(text: str) -> list[str]:
    """Return non-empty normalized lines for duplicate-text measurement."""
    return [
        re.sub(r"\s+", " ", line).strip().lower()
        for line in text.splitlines()
        if line.strip()
    ]


def _repeated_text_ratio(pages: Sequence[PageCreate]) -> float:
    """Measure repeated non-empty lines across the document."""
    lines: list[str] = []
    for page in pages:
        lines.extend(_line_normalization(_page_text(page)))
    if not lines:
        return 0.0
    counts = Counter(lines)
    duplicates = sum(count - 1 for count in counts.values() if count > 1)
    return round(duplicates / len(lines), 4)


def build_quality_report(
    pages: Sequence[PageCreate],
    *,
    confidence_threshold: float,
    parser_errors: Sequence[dict[str, Any]] = (),
) -> QualityReport:
    """Build the observable quality report for the document."""
    empty_pages: list[int] = []
    low_confidence_pages: list[int] = []
    pages_native = 0
    pages_ocr = 0

    for page in pages:
        text = _page_text(page)
        if not text:
            empty_pages.append(page.page_number)
        if page.extraction_method == "native":
            pages_native += 1
        if page.extraction_method == "ocr":
            pages_ocr += 1
        if (
            page.extraction_method == "ocr"
            and page.ocr_confidence is not None
            and page.ocr_confidence < confidence_threshold
        ) or "low_ocr_confidence" in page.quality_flags:
            low_confidence_pages.append(page.page_number)

    pages_total = len(pages)
    return QualityReport(
        pages_total=pages_total,
        pages_extracted=pages_total - len(empty_pages),
        pages_native=pages_native,
        pages_ocr=pages_ocr,
        pages_empty=len(empty_pages),
        empty_pages=sorted(empty_pages),
        low_confidence_pages=sorted(set(low_confidence_pages)),
        repeated_text_ratio=_repeated_text_ratio(pages),
        parser_errors=list(parser_errors),
    )


def map_quality_status(
    report: QualityReport,
    *,
    review_ratio_threshold: float,
    parser_error_threshold: int,
) -> QualityStatus:
    """Map report facts to a status using the documented simple rules."""
    if report.pages_extracted == 0:
        return "failed"
    if len(report.parser_errors) >= parser_error_threshold:
        return "failed"

    affected_pages = set(report.empty_pages) | set(report.low_confidence_pages)
    affected_ratio = (
        len(affected_pages) / report.pages_total if report.pages_total else 0.0
    )
    if affected_ratio > review_ratio_threshold:
        return "needs_review"
    if affected_pages:
        return "ready_with_warnings"
    return "ready"


def assess_quality(
    pages: Sequence[PageCreate],
    *,
    confidence_threshold: float,
    review_ratio_threshold: float,
    parser_error_threshold: int,
    parser_errors: Sequence[dict[str, Any]] = (),
) -> tuple[QualityReport, QualityStatus]:
    """Build the report and map it to a status in one step."""
    report = build_quality_report(
        pages,
        confidence_threshold=confidence_threshold,
        parser_errors=parser_errors,
    )
    status = map_quality_status(
        report,
        review_ratio_threshold=review_ratio_threshold,
        parser_error_threshold=parser_error_threshold,
    )
    return report, status
