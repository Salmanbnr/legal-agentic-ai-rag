"""Tests for preprocessing and bounded OCR fallback behavior."""

from __future__ import annotations

import os
import shutil
from pathlib import Path

import cv2
import numpy as np
import pytest
from PIL import Image, ImageDraw, ImageFont

from app.ingestion import preprocess as preprocess_module
from app.ingestion.ocr import OcrResult
from app.ingestion.preprocess import (
    PreprocessedPage,
    estimate_skew_angle,
    ocr_page_with_fallback,
    preprocess_page,
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


def _document_image(width: int = 1200, height: int = 1600) -> Image.Image:
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=42)
    lines = [
        "IN THE MATTER OF A TEST LEGAL PROCEEDING",
        "STATEMENT OF FACTS",
        "1. This is a scanned document with enough text for orientation.",
        "2. Tesseract needs several lines to detect page rotation.",
        "3. The original page must remain untouched.",
        "4. A copied image is transformed only for OCR.",
        "5. No real person, date, or amount is represented.",
    ]
    for index, line in enumerate(lines):
        draw.text((100, 150 + index * 90), line, fill="black", font=font)
    return image


@requires_tesseract
def test_rotated_scanned_page_is_corrected() -> None:
    original = _document_image()
    rotated = cv2.rotate(np.array(original), cv2.ROTATE_90_CLOCKWISE)

    result = preprocess_page(rotated, tesseract_cmd=TESSERACT)

    orientation = result.transformations[0]
    assert orientation["step"] == "orientation"
    assert orientation["applied"] is True
    assert orientation["rotate_degrees"] in {90, 180, 270}
    assert result.image.shape[:2] == (original.height, original.width)


def test_deskew_estimates_correction_angle() -> None:
    image = np.array(_document_image())
    height, width = image.shape[:2]
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), 5.0, 1.0)
    rotated = cv2.warpAffine(
        image, matrix, (width, height), borderValue=(255, 255, 255)
    )

    angle = estimate_skew_angle(rotated)

    assert angle == pytest.approx(-5.0, abs=1.0)


def test_orientation_falls_back_to_no_rotation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def fail_osd(*args, **kwargs):
        raise RuntimeError("osd unavailable")

    monkeypatch.setattr(preprocess_module.pytesseract, "image_to_osd", fail_osd)
    image = np.full((200, 300, 3), 255, dtype=np.uint8)

    result = preprocess_page(image)

    orientation = result.transformations[0]
    assert orientation["applied"] is False
    assert orientation["reason"] == "osd_unavailable"


def test_preprocessing_keeps_original_array_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = np.array(_document_image(400, 600))
    before = original.copy()
    monkeypatch.setattr(
        preprocess_module,
        "_osd_rotation",
        lambda image, *, tesseract_cmd: (
            0,
            {
                "stage": "preprocess",
                "step": "orientation",
                "applied": False,
                "reason": "test",
            },
        ),
    )

    preprocess_page(original)

    assert np.array_equal(original, before)


def test_fallback_uses_one_preprocessed_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_ocr(image, *, language="eng", tesseract_cmd=None):
        calls.append("ocr")
        confidence = 40.0 if len(calls) == 1 else 88.0
        return OcrResult(
            text=f"attempt {len(calls)}",
            mean_confidence=confidence,
            word_boxes=[],
            language=language,
        )

    def fake_preprocess(image, *, tesseract_cmd=None, max_deskew_degrees=15.0):
        return PreprocessedPage(
            image=np.zeros((20, 20, 3), dtype=np.uint8),
            transformations=[
                {
                    "stage": "preprocess",
                    "step": "threshold",
                    "applied": True,
                }
            ],
        )

    monkeypatch.setattr(preprocess_module, "ocr_image", fake_ocr)
    monkeypatch.setattr(preprocess_module, "preprocess_page", fake_preprocess)

    result = ocr_page_with_fallback(
        np.zeros((20, 20, 3), dtype=np.uint8),
        confidence_threshold=70.0,
    )

    assert calls == ["ocr", "ocr"]
    assert result.variant == "preprocessed"
    assert result.result.mean_confidence == 88.0
    assert result.preprocessed_confidence == 88.0
    assert result.transformations[0]["step"] == "threshold"


def test_fallback_stops_after_original_when_confident(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []

    def fake_ocr(image, *, language="eng", tesseract_cmd=None):
        calls.append("ocr")
        return OcrResult(
            text="clear text",
            mean_confidence=91.0,
            word_boxes=[],
            language=language,
        )

    monkeypatch.setattr(preprocess_module, "ocr_image", fake_ocr)

    result = ocr_page_with_fallback(
        np.zeros((20, 20, 3), dtype=np.uint8),
        confidence_threshold=70.0,
    )

    assert calls == ["ocr"]
    assert result.variant == "original"
    assert result.preprocessed_confidence is None
