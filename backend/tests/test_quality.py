"""Tests for transparent page flags and document quality status rules."""

from __future__ import annotations

import pytest

from app.core.config import Settings
from app.db.repos import PageCreate
from app.ingestion.ocr import OcrWord
from app.ingestion.quality import (
    assess_quality,
    build_quality_report,
    evaluate_page_flags,
    map_quality_status,
)


def _settings() -> Settings:
    return Settings(_env_file=None)


def _page(
    page_number: int,
    *,
    text: str | None = "Substantive page text.",
    method: str = "native",
    confidence: float | None = None,
    flags: list[str] | None = None,
) -> PageCreate:
    return PageCreate(
        page_number=page_number,
        extraction_method=method,
        raw_text=text,
        ocr_confidence=confidence,
        quality_flags=list(flags or []),
    )


def _status(pages: list[PageCreate], **overrides):
    settings = _settings()
    report, status = assess_quality(
        pages,
        confidence_threshold=settings.ocr_confidence_threshold,
        review_ratio_threshold=overrides.get(
            "review_ratio_threshold", settings.quality_review_ratio_threshold
        ),
        parser_error_threshold=overrides.get(
            "parser_error_threshold", settings.quality_parser_error_threshold
        ),
        parser_errors=overrides.get("parser_errors", ()),
    )
    return report, status


def test_ready_when_all_pages_have_text_and_no_warnings() -> None:
    report, status = _status([_page(1), _page(2), _page(3)])

    assert report.pages_extracted == 3
    assert report.pages_empty == 0
    assert report.low_confidence_pages == []
    assert status == "ready"


def test_ready_with_warnings_when_a_small_share_is_low_confidence() -> None:
    pages = [
        _page(1, method="ocr", confidence=92.0),
        _page(2, method="ocr", confidence=50.0, flags=["low_ocr_confidence"]),
        _page(3, method="ocr", confidence=91.0),
        _page(4, method="ocr", confidence=90.0),
        _page(5, method="ocr", confidence=93.0),
    ]

    report, status = _status(pages)

    assert report.low_confidence_pages == [2]
    assert status == "ready_with_warnings"


def test_needs_review_when_more_than_thirty_percent_are_affected() -> None:
    pages = [
        _page(1),
        _page(2),
        _page(3, text=None, method="ocr"),
        _page(4, method="ocr", confidence=40.0, flags=["low_ocr_confidence"]),
    ]

    report, status = _status(pages)

    assert report.empty_pages == [3]
    assert report.low_confidence_pages == [4]
    assert status == "needs_review"


def test_failed_when_no_page_has_usable_text() -> None:
    report, status = _status(
        [
            _page(1, text=None, method="none", flags=["empty_page"]),
            _page(2, text=None, method="none", flags=["empty_page"]),
        ]
    )

    assert report.pages_extracted == 0
    assert status == "failed"


def test_failed_when_parser_errors_repeat() -> None:
    pages = [_page(1), _page(2), _page(3), _page(4), _page(5)]
    parser_errors = [
        {"page_number": 2, "error_code": "OCR_FAILED"},
        {"page_number": 3, "error_code": "OCR_FAILED"},
        {"page_number": 4, "error_code": "OCR_FAILED"},
    ]

    report, status = _status(
        pages,
        parser_errors=parser_errors,
        parser_error_threshold=3,
    )

    assert len(report.parser_errors) == 3
    assert status == "failed"


def test_report_has_no_opaque_magic_score() -> None:
    report, _ = _status([_page(1)])

    assert "score" not in report.model_dump()


def test_noisy_page_gets_low_ocr_confidence_flag() -> None:
    flags = evaluate_page_flags(
        text="faint ambiguous text",
        extraction_method="ocr",
        ocr_confidence=22.0,
        words=[
            OcrWord(
                text="faint",
                confidence=22.0,
                left=10,
                top=10,
                width=60,
                height=20,
            )
        ],
        settings=_settings(),
        ocr_completed=True,
    )

    assert "low_ocr_confidence" in flags
    assert "possible_handwriting" in flags


def test_table_layout_and_empty_page_flags_are_explicit() -> None:
    words = []
    for line in range(1, 4):
        for column in range(2):
            words.append(
                OcrWord(
                    text=f"r{line}c{column}",
                    confidence=95.0,
                    left=50 + column * 500,
                    top=40 * line,
                    width=80,
                    height=20,
                    block=1,
                    paragraph=1,
                    line=line,
                    word=column + 1,
                )
            )

    table_flags = evaluate_page_flags(
        text="cell one    cell two",
        extraction_method="ocr",
        ocr_confidence=95.0,
        words=words,
        settings=_settings(),
        ocr_completed=True,
    )
    empty_flags = evaluate_page_flags(
        text=None,
        extraction_method="ocr",
        ocr_confidence=0.0,
        settings=_settings(),
        ocr_completed=True,
    )

    assert "possible_table" in table_flags
    assert "empty_page" in empty_flags


def test_repeated_text_ratio_measures_duplicate_lines() -> None:
    report = build_quality_report(
        [
            _page(1, text="Header\nUnique first page line."),
            _page(2, text="Header\nUnique second page line."),
        ],
        confidence_threshold=70.0,
    )

    assert report.repeated_text_ratio == pytest.approx(0.25)
