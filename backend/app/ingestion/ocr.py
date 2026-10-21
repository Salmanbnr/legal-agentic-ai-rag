"""Tesseract OCR for rendered PDF pages and standalone images.

Only derived text, confidence values, and word geometry are returned. The
original document bytes are never modified. Every failure is translated into a
stable error code so worker logs cannot leak parser output or file content.
"""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from typing import Any

import numpy as np
import pymupdf
import pytesseract
from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field
from pytesseract import Output

OCR_VERSION = "1.0.0"


class OcrError(RuntimeError):
    """A safe OCR failure carrying a stable, persistable error code."""

    def __init__(self, error_code: str) -> None:
        """Create an error without including image or OCR output details."""
        super().__init__(error_code)
        self.error_code = error_code


class OcrWord(BaseModel):
    """One recognized word with its pixel bounding box."""

    model_config = ConfigDict(frozen=True)

    text: str
    confidence: float = Field(ge=0.0, le=100.0)
    left: int = Field(ge=0)
    top: int = Field(ge=0)
    width: int = Field(ge=0)
    height: int = Field(ge=0)
    block: int = Field(default=0, ge=0)
    paragraph: int = Field(default=0, ge=0)
    line: int = Field(default=0, ge=0)
    word: int = Field(default=0, ge=0)


class OcrResult(BaseModel):
    """OCR text plus confidence and word geometry for one image."""

    model_config = ConfigDict(frozen=True)

    text: str
    mean_confidence: float = Field(ge=0.0, le=100.0)
    word_boxes: list[OcrWord] = Field(default_factory=list)
    language: str

    @property
    def word_count(self) -> int:
        """Return the number of recognized words."""
        return len(self.word_boxes)


def configure_tesseract(command: str | Path | None) -> None:
    """Configure an explicit Tesseract binary, otherwise leave PATH lookup."""
    if command is None:
        return
    resolved = Path(command)
    if not resolved.is_file():
        raise OcrError("TESSERACT_UNAVAILABLE")
    pytesseract.pytesseract.tesseract_cmd = str(resolved)


def _as_pil_image(image: Image.Image | np.ndarray) -> Image.Image:
    """Convert a PIL image or RGB/gray numpy array into a PIL RGB image."""
    if isinstance(image, Image.Image):
        return image.convert("RGB")
    if isinstance(image, np.ndarray):
        if image.ndim == 2:
            return Image.fromarray(image, mode="L").convert("RGB")
        if image.ndim == 3 and image.shape[2] == 3:
            return Image.fromarray(image, mode="RGB")
        if image.ndim == 3 and image.shape[2] == 4:
            return Image.fromarray(image, mode="RGBA").convert("RGB")
    raise OcrError("OCR_FAILED")


def _int_value(value: Any, default: int = 0) -> int:
    """Convert one Tesseract dictionary value to a non-negative integer."""
    try:
        return max(0, int(float(value)))
    except (TypeError, ValueError):
        return default


def _parse_ocr_data(data: dict[str, list[Any]], language: str) -> OcrResult:
    """Build ordered text, confidence, and word boxes from Tesseract output."""
    texts = data.get("text") or []
    confidences = data.get("conf") or []
    words: list[OcrWord] = []
    line_order: list[tuple[int, int, int]] = []
    line_words: dict[tuple[int, int, int], list[str]] = {}

    for index, raw_text in enumerate(texts):
        text = str(raw_text).strip()
        if not text:
            continue
        try:
            confidence = float(confidences[index])
        except (IndexError, TypeError, ValueError):
            continue
        if confidence < 0:
            continue

        block = _int_value((data.get("block_num") or [0] * len(texts))[index])
        paragraph = _int_value((data.get("par_num") or [0] * len(texts))[index])
        line = _int_value((data.get("line_num") or [0] * len(texts))[index])
        word_number = _int_value((data.get("word_num") or [0] * len(texts))[index])
        key = (block, paragraph, line)
        if key not in line_words:
            line_order.append(key)
            line_words[key] = []
        line_words[key].append(text)
        words.append(
            OcrWord(
                text=text,
                confidence=round(min(100.0, max(0.0, confidence)), 2),
                left=_int_value((data.get("left") or [0] * len(texts))[index]),
                top=_int_value((data.get("top") or [0] * len(texts))[index]),
                width=_int_value((data.get("width") or [0] * len(texts))[index]),
                height=_int_value((data.get("height") or [0] * len(texts))[index]),
                block=block,
                paragraph=paragraph,
                line=line,
                word=word_number,
            )
        )

    text = "\n".join(" ".join(line_words[key]) for key in line_order)
    mean_confidence = (
        round(sum(word.confidence for word in words) / len(words), 2)
        if words
        else 0.0
    )
    return OcrResult(
        text=text,
        mean_confidence=mean_confidence,
        word_boxes=words,
        language=language,
    )


def ocr_image(
    image: Image.Image | np.ndarray,
    *,
    language: str = "eng",
    psm: int = 3,
    tesseract_cmd: str | Path | None = None,
) -> OcrResult:
    """Run Tesseract on an image and return text, confidence, and word boxes.

    Args:
        image: PIL image or RGB/gray numpy array.
        language: Tesseract language code, for example ``eng`` or ``eng+fra``.
        psm: Tesseract page segmentation mode.
        tesseract_cmd: Optional explicit Tesseract executable.

    Raises:
        OcrError: With a safe code when Tesseract or the image is unusable.
    """
    configure_tesseract(tesseract_cmd)
    pil_image = _as_pil_image(image)
    try:
        data = pytesseract.image_to_data(
            pil_image,
            lang=language,
            config=f"--oem 3 --psm {psm}",
            output_type=Output.DICT,
        )
    except pytesseract.TesseractNotFoundError as exc:
        raise OcrError("TESSERACT_UNAVAILABLE") from exc
    except (pytesseract.TesseractError, OSError, RuntimeError, ValueError) as exc:
        raise OcrError("OCR_FAILED") from exc
    return _parse_ocr_data(data, language)


def render_pdf_page(
    data: bytes,
    page_number: int,
    *,
    dpi: int = 300,
) -> Image.Image:
    """Render one PDF page to a PIL RGB image without altering the original."""
    if page_number < 1:
        raise ValueError("page_number must be at least 1")
    if not 72 <= dpi <= 600:
        raise ValueError("dpi must be between 72 and 600")
    document: pymupdf.Document | None = None
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
        if document.needs_pass or document.is_encrypted:
            raise OcrError("PASSWORD_REQUIRED")
        if page_number > document.page_count:
            raise OcrError("PDF_RENDER_FAILED")
        page = document.load_page(page_number - 1)
        pixmap = page.get_pixmap(
            dpi=dpi, colorspace=pymupdf.csRGB, alpha=False
        )
        return Image.frombytes(
            "RGB", (pixmap.width, pixmap.height), pixmap.samples
        )
    except OcrError:
        raise
    except Exception as exc:
        raise OcrError("PDF_RENDER_FAILED") from exc
    finally:
        if document is not None:
            document.close()


def load_image_page(data: bytes, page_number: int = 1) -> Image.Image:
    """Decode one frame from a standalone image file as PIL RGB."""
    if page_number < 1:
        raise ValueError("page_number must be at least 1")
    try:
        with Image.open(BytesIO(data)) as image:
            image.seek(page_number - 1)
            return image.convert("RGB")
    except (
        UnidentifiedImageError,
        Image.DecompressionBombError,
        OSError,
        ValueError,
        EOFError,
        IndexError,
    ) as exc:
        raise OcrError("IMAGE_DECODE_FAILED") from exc
