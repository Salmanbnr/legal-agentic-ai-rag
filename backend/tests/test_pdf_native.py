"""Tests for native PDF block extraction and page classification."""

from __future__ import annotations

from pathlib import Path

import pymupdf
import pytest

from app.ingestion.pdf_native import PdfExtractionError, extract_pdf_pages
from scripts.make_test_pdfs import GeneratedPdfs, generate_test_pdfs

DENSITY_THRESHOLD = 0.0002


@pytest.fixture()
def pdfs(tmp_path: Path) -> GeneratedPdfs:
    return generate_test_pdfs(tmp_path / "pdfs")


def test_clean_pdf_extracts_native_text(pdfs: GeneratedPdfs) -> None:
    pages = extract_pdf_pages(
        pdfs.clean_native.read_bytes(),
        density_threshold=DENSITY_THRESHOLD,
    )

    assert len(pages) == 1
    assert pages[0].page_number == 1
    assert pages[0].extraction_method == "native"
    assert "STATEMENT OF FACTS" in pages[0].raw_text
    assert pages[0].text_density > DENSITY_THRESHOLD
    assert pages[0].quality_flags == []


def test_image_only_pdf_is_marked_needs_ocr(pdfs: GeneratedPdfs) -> None:
    pages = extract_pdf_pages(
        pdfs.image_only.read_bytes(),
        density_threshold=DENSITY_THRESHOLD,
    )

    assert len(pages) == 1
    assert pages[0].extraction_method == "none"
    assert pages[0].raw_text == ""
    assert pages[0].text_density == 0.0
    assert "needs_ocr" in pages[0].quality_flags


def test_encrypted_pdf_raises_password_required(pdfs: GeneratedPdfs) -> None:
    with pytest.raises(PdfExtractionError) as error:
        extract_pdf_pages(
            pdfs.encrypted.read_bytes(),
            density_threshold=DENSITY_THRESHOLD,
        )

    assert error.value.error_code == "PASSWORD_REQUIRED"


def test_corrupt_pdf_raises_processing_failed() -> None:
    with pytest.raises(PdfExtractionError) as error:
        extract_pdf_pages(
            b"%PDF-1.7\nnot a real document",
            density_threshold=DENSITY_THRESHOLD,
        )

    assert error.value.error_code == "PROCESSING_FAILED"


def test_page_limit_is_enforced() -> None:
    document = pymupdf.open()
    try:
        document.new_page()
        document.new_page()
        data = document.tobytes()
    finally:
        document.close()

    with pytest.raises(PdfExtractionError) as error:
        extract_pdf_pages(
            data,
            density_threshold=DENSITY_THRESHOLD,
            max_pages=1,
        )

    assert error.value.error_code == "PAGE_LIMIT_EXCEEDED"


def test_two_column_reading_order_sorts_left_column_first() -> None:
    document = pymupdf.open()
    try:
        page = document.new_page(width=612, height=792)
        page.insert_textbox(
            pymupdf.Rect(50, 80, 290, 140),
            "LEFT TOP",
            fontsize=12,
        )
        page.insert_textbox(
            pymupdf.Rect(50, 180, 290, 240),
            "LEFT BOTTOM",
            fontsize=12,
        )
        page.insert_textbox(
            pymupdf.Rect(350, 80, 580, 140),
            "RIGHT TOP",
            fontsize=12,
        )
        page.insert_textbox(
            pymupdf.Rect(350, 180, 580, 240),
            "RIGHT BOTTOM",
            fontsize=12,
        )
        data = document.tobytes()
    finally:
        document.close()

    pages = extract_pdf_pages(data, density_threshold=0.0)
    text = pages[0].raw_text

    assert text.index("LEFT TOP") < text.index("LEFT BOTTOM")
    assert text.index("LEFT BOTTOM") < text.index("RIGHT TOP")
    assert text.index("RIGHT TOP") < text.index("RIGHT BOTTOM")
