"""Shared legal document fixtures for the end-to-end tests.

The fixture is a three page PDF with one section per page so a retrieval hit
can be checked against a known page number.
"""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

import pymupdf

from app.core.config import Settings
from app.db.repos import PageCreate
from app.ingestion.chunking import chunk_pages, load_token_counter
from app.ingestion.normalize import normalize_pages
from app.ingestion.pdf_native import extract_pdf_pages

TIMESTAMP = "2026-01-01T00:00:00+00:00"
MODEL_CACHE = Path.home() / ".cache" / "huggingface" / "hub"

PAGE_ONE = """FACTS

The claimant is the owner of a residential property at 14 Bridge Street. The
respondent is the tenant of that property under a written tenancy agreement
dated the first day of April. The agreement fixes the monthly rent at 1,250
pounds and requires the rent to be paid in advance on the first day of each
month. The respondent took occupation in June and continued to use the property
as her only residence. The parties continued to perform their obligations
under the agreement for more than two years without any dispute about the
terms of the tenancy."""

PAGE_TWO = """THE BREACH

The tenant failed to pay the monthly rent for three consecutive months. The
landlord issued a notice requiring payment of the arrears within fourteen
days. The arrears were never paid and the landlord therefore commenced these
proceedings for possession of the property and for the sum of 3,750 pounds
representing three months of unpaid rent. The landlord also claimed the
reasonable costs of repairing the roof, which the tenant had neglected for a
period of more than eighteen months."""

PAGE_THREE = """THE REMEDY

The court ordered the claimant to recover 3,750 pounds from the respondent,
together with the costs of the claim. The judge held that the landlord was
entitled to terminate the tenancy because the rent had not been paid. The
court also awarded 500 pounds in compensation for the loss caused by the delay
in the repair of the roof. The order was served on the respondent on the
twelfth day of March and the arrears were paid in full the following week."""

EXPECTED_TEXT = {
    1: "monthly rent at 1,250",
    2: "three months of unpaid rent",
    3: "recover 3,750 pounds from the respondent",
}


def fixture_settings() -> Settings:
    """Return settings that read the locally cached model."""
    return Settings(_env_file=None).model_copy(
        update={"model_cache_dir": MODEL_CACHE}
    )


def pdf_bytes() -> bytes:
    """Build a three page PDF with one section per page."""
    document = pymupdf.open()
    try:
        for text in (PAGE_ONE, PAGE_TWO, PAGE_THREE):
            page = document.new_page()
            top = 72
            for line in text.splitlines():
                if not line.strip():
                    top += 10
                    continue
                page.insert_text((72, top), line, fontsize=11)
                top += 16
        return document.tobytes()
    finally:
        document.close()


def extracted_pages(data: bytes) -> list[PageCreate]:
    """Extract and normalize the generated PDF."""
    settings = fixture_settings()
    extracted = extract_pdf_pages(
        data,
        density_threshold=settings.pdf_native_text_density_threshold,
        max_pages=settings.max_pdf_pages,
    )
    pages = [
        PageCreate(
            page_number=page.page_number,
            extraction_method=page.extraction_method,
            raw_text=page.raw_text,
            quality_flags=list(page.quality_flags),
            transform_log=[],
        )
        for page in extracted
    ]
    normalized = normalize_pages(
        pages,
        header_footer_repeat_ratio=settings.header_footer_repeat_ratio,
        header_footer_max_lines=settings.header_footer_max_lines,
    )
    by_page = {result.page_number: result for result in normalized}
    for page in pages:
        result = by_page.get(page.page_number)
        if result is None:
            continue
        page.clean_text = result.clean_text
        page.transform_log = list(page.transform_log) + result.transformations
    return pages


def chunk_rows(
    pages: list[PageCreate], document_id: UUID, case_id: UUID
) -> list:
    """Chunk the pages into stored chunk rows."""
    from app.db.repos import DocumentChunk

    settings = fixture_settings()
    counter = load_token_counter(settings)
    payloads = chunk_pages(
        pages,
        document_id=document_id,
        token_counter=counter,
        min_tokens=settings.chunk_min_tokens,
        max_tokens=settings.chunk_max_tokens,
        overlap_min_tokens=settings.chunk_overlap_min_tokens,
        overlap_max_tokens=settings.chunk_overlap_max_tokens,
        index_schema_version=settings.index_schema_version,
    )
    rows = []
    for payload in payloads:
        rows.append(
            DocumentChunk.model_validate(
                {
                    "id": str(uuid4()),
                    "case_id": str(case_id),
                    "document_id": str(document_id),
                    "document_version_id": None,
                    **payload.model_dump(mode="json", exclude_none=True),
                    "created_at": TIMESTAMP,
                }
            )
        )
    return rows
