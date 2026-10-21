"""Native PDF text extraction with reading-order and density classification."""

from __future__ import annotations

from typing import Literal

import pymupdf
from pydantic import BaseModel, ConfigDict, Field

PageExtractionMethod = Literal["native", "ocr", "none"]


class PdfExtractionError(RuntimeError):
    """A safe PDF extraction failure with a stable error code."""

    def __init__(self, error_code: str) -> None:
        """Create an error that can be persisted without parser details."""
        super().__init__(error_code)
        self.error_code = error_code


class ExtractedPage(BaseModel):
    """One page's text and extraction provenance."""

    model_config = ConfigDict(frozen=True)

    page_number: int = Field(ge=1)
    raw_text: str
    extraction_method: PageExtractionMethod
    text_density: float = Field(ge=0)
    quality_flags: list[str] = Field(default_factory=list)


def _sort_text_blocks(
    blocks: list[tuple[float, float, float, float, str]], page_width: float
) -> list[tuple[float, float, float, float, str]]:
    """Sort blocks by y, or by column then y for a simple two-column page."""
    if not blocks:
        return []
    left_blocks = [block for block in blocks if block[2] <= page_width * 0.55]
    right_blocks = [block for block in blocks if block[0] >= page_width * 0.45]
    two_column = len(left_blocks) >= 2 and len(right_blocks) >= 2

    if two_column:
        midpoint = page_width / 2
        return sorted(
            blocks,
            key=lambda block: (
                0 if (block[0] + block[2]) / 2 < midpoint else 1,
                round(block[1], 3),
                round(block[0], 3),
            ),
        )
    return sorted(
        blocks,
        key=lambda block: (round(block[1], 3), round(block[0], 3)),
    )


def _page_blocks(page: pymupdf.Page) -> list[tuple[float, float, float, float, str]]:
    """Return text-bearing blocks with finite coordinates."""
    blocks: list[tuple[float, float, float, float, str]] = []
    for raw_block in page.get_text("blocks"):
        if len(raw_block) < 7 or int(raw_block[6]) != 0:
            continue
        text = str(raw_block[4]).strip()
        if not text:
            continue
        blocks.append(
            (
                float(raw_block[0]),
                float(raw_block[1]),
                float(raw_block[2]),
                float(raw_block[3]),
                text,
            )
        )
    return blocks


def extract_pdf_pages(
    data: bytes,
    *,
    density_threshold: float,
    max_pages: int | None = None,
) -> list[ExtractedPage]:
    """Extract every PDF page and classify pages as native text or OCR-needed.

    ``text_density`` is non-whitespace characters per square point of page area.
    A page is native when it has text and its density meets the configured
    threshold; otherwise it is stored with extraction method ``none`` and a
    ``needs_ocr`` quality flag.
    """
    if density_threshold < 0:
        raise ValueError("density_threshold must be non-negative")
    if not isinstance(data, bytes) or not data.startswith(b"%PDF-"):
        raise PdfExtractionError("UNSUPPORTED_FILE")

    document: pymupdf.Document | None = None
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
        if document.needs_pass or document.is_encrypted:
            raise PdfExtractionError("PASSWORD_REQUIRED")
        if not document.is_pdf:
            raise PdfExtractionError("PROCESSING_FAILED")
        page_count = int(document.page_count)
        if page_count < 1:
            raise PdfExtractionError("PROCESSING_FAILED")
        if max_pages is not None and page_count > max_pages:
            raise PdfExtractionError("PAGE_LIMIT_EXCEEDED")

        pages: list[ExtractedPage] = []
        for index in range(page_count):
            page = document.load_page(index)
            width = max(float(page.rect.width), 1.0)
            height = max(float(page.rect.height), 1.0)
            area = width * height
            blocks = _page_blocks(page)
            ordered = _sort_text_blocks(blocks, width)
            raw_text = "\n\n".join(block[4] for block in ordered).strip()
            visible_characters = sum(
                1 for character in raw_text if not character.isspace()
            )
            density = visible_characters / area
            native = bool(raw_text) and density >= density_threshold
            pages.append(
                ExtractedPage(
                    page_number=index + 1,
                    raw_text=raw_text,
                    extraction_method="native" if native else "none",
                    text_density=density,
                    quality_flags=[] if native else ["needs_ocr", "low_text_density"],
                )
            )
        return pages
    except PdfExtractionError:
        raise
    except Exception as exc:
        # Never expose parser internals; the worker persists only the code.
        raise PdfExtractionError("PROCESSING_FAILED") from exc
    finally:
        if document is not None:
            document.close()
