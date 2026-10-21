"""Tests for Tesseract OCR helpers and PDF page rendering."""

from __future__ import annotations

import os
import shutil
from io import BytesIO
from pathlib import Path

import pymupdf
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.ingestion.ocr import (
    OcrError,
    load_image_page,
    ocr_image,
    render_pdf_page,
)


def _tesseract_command() -> str | None:
    configured = os.environ.get("TESSERACT_CMD")
    if configured and Path(configured).is_file():
        return configured
    found = shutil.which("tesseract")
    if found:
        return found
    common = Path(r"C:\Program Files\Tesseract-OCR\tesseract.exe")
    if common.is_file():
        return str(common)
    return None


TESSERACT = _tesseract_command()
requires_tesseract = pytest.mark.skipif(
    TESSERACT is None, reason="Tesseract binary is not available"
)


def _text_image(text: str = "LEGAL EVIDENCE PAGE") -> Image.Image:
    image = Image.new("RGB", (1000, 300), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=48)
    draw.text((60, 100), text, fill="black", font=font)
    return image


def test_load_image_page_decodes_png() -> None:
    image = _text_image()
    buffer = BytesIO()
    image.save(buffer, format="PNG")

    loaded = load_image_page(buffer.getvalue())

    assert loaded.mode == "RGB"
    assert loaded.size == (1000, 300)


def test_load_image_page_rejects_invalid_bytes() -> None:
    with pytest.raises(OcrError) as error:
        load_image_page(b"not an image")

    assert error.value.error_code == "IMAGE_DECODE_FAILED"


@requires_tesseract
def test_ocr_image_returns_text_confidence_and_boxes() -> None:
    result = ocr_image(
        _text_image("IMAGE ONLY SCANNED PAGE"),
        language="eng",
        tesseract_cmd=TESSERACT,
    )

    assert "IMAGE" in result.text
    assert result.mean_confidence > 70
    assert result.word_boxes
    first = result.word_boxes[0]
    assert first.width > 0
    assert first.height > 0
    assert 0 <= first.confidence <= 100


def test_render_pdf_page_returns_rendered_rgb() -> None:
    document = pymupdf.open()
    try:
        page = document.new_page(width=200, height=300)
        page.insert_text((20, 40), "Render me", fontsize=12)
        data = document.tobytes()
    finally:
        document.close()

    rendered = render_pdf_page(data, 1, dpi=150)

    assert rendered.mode == "RGB"
    assert rendered.width > 0
    assert rendered.height > 0
    assert data.startswith(b"%PDF-")


def test_render_pdf_page_rejects_missing_page() -> None:
    document = pymupdf.open()
    try:
        document.new_page()
        data = document.tobytes()
    finally:
        document.close()

    with pytest.raises(OcrError) as error:
        render_pdf_page(data, 2)

    assert error.value.error_code == "PDF_RENDER_FAILED"
