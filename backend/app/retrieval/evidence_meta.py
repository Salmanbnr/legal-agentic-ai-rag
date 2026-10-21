"""Page-derived metadata for one evidence span.

A citation has to describe where its text came from in the same way whether it
was found by the baseline retriever or by the agent workflow. Both paths
therefore derive the extraction method, the worst OCR confidence, and the
quality flags of a chunk from the stored pages of its page range.

Only page numbers, methods, confidences, and flag names are returned. Document
text never passes through this module.
"""

from __future__ import annotations

from typing import Any, Sequence

from app.db.repos import DocumentChunk, DocumentPage

MIXED_EXTRACTION_METHOD = "mixed"


def page_signals(
    pages: Sequence[DocumentPage], chunk: DocumentChunk
) -> tuple[str | None, float | None]:
    """Return the extraction method and the worst page confidence of a chunk.

    Args:
        pages: The stored pages of the chunk's document.
        chunk: The chunk whose page range is described.

    Returns:
        ``(extraction_method, ocr_confidence)``. The method is ``mixed`` when
        the range covers more than one method. The confidence is the lowest
        value on the range, because a citation is only as reliable as its
        weakest page.
    """
    if chunk.page_start is None or chunk.page_end is None:
        return None, None
    relevant = [
        page
        for page in pages
        if chunk.page_start <= page.page_number <= chunk.page_end
    ]
    if not relevant:
        return None, None
    methods = {page.extraction_method for page in relevant}
    extraction_method = (
        next(iter(methods)) if len(methods) == 1 else MIXED_EXTRACTION_METHOD
    )
    confidences = [
        page.ocr_confidence
        for page in relevant
        if page.ocr_confidence is not None
    ]
    confidence = min(confidences) if confidences else None
    return extraction_method, confidence


def collect_quality_flags(
    chunk: DocumentChunk, pages: Sequence[DocumentPage]
) -> list[Any]:
    """Merge the chunk's flags with the flags of every page it covers."""
    flags: list[Any] = []
    for flag in chunk.quality_flags:
        if flag not in flags:
            flags.append(flag)
    if chunk.page_start is None or chunk.page_end is None:
        return flags
    for page in pages:
        if not chunk.page_start <= page.page_number <= chunk.page_end:
            continue
        for flag in page.quality_flags:
            if flag not in flags:
                flags.append(flag)
    return flags
