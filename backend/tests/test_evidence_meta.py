"""Tests for the page metadata shared by both retrieval paths."""

from __future__ import annotations

from uuid import uuid4

from app.db.repos import DocumentChunk, DocumentPage
from app.retrieval.evidence_meta import (
    MIXED_EXTRACTION_METHOD,
    collect_quality_flags,
    page_signals,
)

TIMESTAMP = "2026-01-01T00:00:00+00:00"


def _page(
    page_number: int,
    extraction_method: str = "native",
    ocr_confidence: float | None = None,
    quality_flags: list[str] | None = None,
) -> DocumentPage:
    return DocumentPage.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(uuid4()),
            "document_id": str(uuid4()),
            "page_number": page_number,
            "extraction_method": extraction_method,
            "text_density": 0.8,
            "ocr_confidence": ocr_confidence,
            "raw_text": None,
            "clean_text": None,
            "quality_flags": quality_flags or [],
            "transform_log": [],
            "created_at": TIMESTAMP,
        }
    )


def _chunk(
    page_start: int | None = 1,
    page_end: int | None = 1,
    quality_flags: list[str] | None = None,
) -> DocumentChunk:
    return DocumentChunk.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(uuid4()),
            "document_id": str(uuid4()),
            "document_version_id": None,
            "chunk_index": 0,
            "page_start": page_start,
            "page_end": page_end,
            "section_path": [],
            "text_content": "Evidence excerpt.",
            "content_hash": "b" * 64,
            "source_anchor": None,
            "embedding_model": None,
            "index_schema_version": 1,
            "index_status": "pending",
            "quality_flags": quality_flags or [],
            "created_at": TIMESTAMP,
        }
    )


def test_page_signals_reports_a_single_method_and_the_worst_confidence() -> None:
    pages = [
        _page(1, "ocr", 88.0),
        _page(2, "ocr", 61.5),
        _page(3, "ocr", 74.0),
    ]

    method, confidence = page_signals(pages, _chunk(1, 2))

    assert method == "ocr"
    assert confidence == 61.5


def test_page_signals_marks_a_range_with_two_methods_as_mixed() -> None:
    pages = [_page(1, "native"), _page(2, "ocr", 90.0)]

    method, confidence = page_signals(pages, _chunk(1, 2))

    assert method == MIXED_EXTRACTION_METHOD
    assert confidence == 90.0


def test_page_signals_ignores_pages_outside_the_range() -> None:
    pages = [_page(4, "native"), _page(5, "ocr", 40.0)]

    method, confidence = page_signals(pages, _chunk(4, 4))

    assert method == "native"
    assert confidence is None


def test_page_signals_without_a_page_range_returns_nothing() -> None:
    assert page_signals([_page(1)], _chunk(None, None)) == (None, None)


def test_collect_quality_flags_merges_and_deduplicates() -> None:
    pages = [
        _page(1, quality_flags=["low_ocr_confidence"]),
        _page(2, quality_flags=["low_ocr_confidence", "empty_page"]),
        _page(3, quality_flags=["possible_table"]),
    ]

    flags = collect_quality_flags(
        _chunk(1, 2, ["possible_handwriting"]), pages
    )

    assert flags == ["possible_handwriting", "low_ocr_confidence", "empty_page"]
