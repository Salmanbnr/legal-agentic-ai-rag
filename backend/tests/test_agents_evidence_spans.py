"""Tests for the page metadata the agent workflow attaches to its citations.

A citation from the agent path has to describe its page the same way a baseline
citation does: the extraction method, the worst OCR confidence, and the flags of
the pages it covers.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from uuid import UUID, uuid4

import pytest

from app.agents import graph
from app.agents.state import AgenticRAGState, RetrievedEvidence
from app.db.repos import Document, DocumentChunk, DocumentPage

TIMESTAMP = "2026-01-01T00:00:00+00:00"


def _document(document_id: UUID, case_id: UUID) -> Document:
    return Document.model_validate(
        {
            "id": str(document_id),
            "case_id": str(case_id),
            "owner_user_id": str(uuid4()),
            "storage_path": f"{uuid4()}/{case_id}/{uuid4()}",
            "original_filename": "Lease agreement.pdf",
            "detected_mime": "application/pdf",
            "sha256": "a" * 64,
            "size_bytes": 2048,
            "page_count": 8,
            "quality_status": "ready_with_warnings",
            "quality_report": {},
            "processing_status": "ready",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    )


def _chunk(document_id: UUID, case_id: UUID) -> DocumentChunk:
    return DocumentChunk.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "document_id": str(document_id),
            "document_version_id": None,
            "chunk_index": 2,
            "page_start": 7,
            "page_end": 8,
            "section_path": ["Clause 12"],
            "text_content": "The monthly rent is GBP 1,250.",
            "content_hash": "b" * 64,
            "source_anchor": "doc:1|p:7|c:0",
            "embedding_model": None,
            "index_schema_version": 1,
            "index_status": "indexed",
            "quality_flags": [],
            "created_at": TIMESTAMP,
        }
    )


def _page(
    document_id: UUID,
    case_id: UUID,
    page_number: int,
    extraction_method: str,
    ocr_confidence: float | None,
    quality_flags: list[str] | None = None,
) -> DocumentPage:
    return DocumentPage.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "document_id": str(document_id),
            "page_number": page_number,
            "extraction_method": extraction_method,
            "text_density": 0.7,
            "ocr_confidence": ocr_confidence,
            "raw_text": None,
            "clean_text": None,
            "quality_flags": quality_flags or [],
            "transform_log": [],
            "created_at": TIMESTAMP,
        }
    )


def _state(chunk: DocumentChunk) -> AgenticRAGState:
    return AgenticRAGState(
        user_id=uuid4(),
        case_id=chunk.case_id,
        question="What is the monthly rent?",
        retrieval_results=[
            RetrievedEvidence(
                subquestion_id="sq-1",
                evidence_id=chunk.id,
                document_id=chunk.document_id,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
            )
        ],
    )


def _deps() -> SimpleNamespace:
    return SimpleNamespace(
        client=object(),
        user=SimpleNamespace(id=uuid4()),
        settings=SimpleNamespace(agent_analyst_max_spans=10),
    )


def test_agent_citations_carry_the_page_method_and_confidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk(uuid4(), uuid4())
    pages = [
        _page(chunk.document_id, chunk.case_id, 7, "ocr", 82.0),
        _page(
            chunk.document_id,
            chunk.case_id,
            8,
            "ocr",
            58.0,
            ["low_ocr_confidence"],
        ),
    ]
    monkeypatch.setattr(graph, "get_chunks_by_ids", lambda *a, **k: [chunk])
    monkeypatch.setattr(
        graph,
        "list_documents",
        lambda *a, **k: [_document(chunk.document_id, chunk.case_id)],
    )
    monkeypatch.setattr(
        graph, "get_pages_for_documents", lambda *a, **k: pages
    )

    spans = graph._load_evidence_spans(_state(chunk), _deps())

    assert len(spans) == 1
    span = spans[0]
    assert span.extraction_method == "ocr"
    assert span.ocr_confidence == 58.0
    assert span.quality_flags == ["low_ocr_confidence"]
    assert span.quality_status == "ready_with_warnings"
    assert span.title == "Lease agreement.pdf"
    assert span.page_start == 7
    assert span.page_end == 8


def test_a_page_lookup_failure_keeps_the_evidence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    chunk = _chunk(uuid4(), uuid4())
    monkeypatch.setattr(graph, "get_chunks_by_ids", lambda *a, **k: [chunk])
    monkeypatch.setattr(
        graph,
        "list_documents",
        lambda *a, **k: [_document(chunk.document_id, chunk.case_id)],
    )

    def fail(*args: Any, **kwargs: Any) -> list[DocumentPage]:
        raise RuntimeError("pages unavailable")

    monkeypatch.setattr(graph, "get_pages_for_documents", fail)

    spans = graph._load_evidence_spans(_state(chunk), _deps())

    assert len(spans) == 1
    assert spans[0].quote_or_span == "The monthly rent is GBP 1,250."
    assert spans[0].extraction_method is None
    assert spans[0].ocr_confidence is None
