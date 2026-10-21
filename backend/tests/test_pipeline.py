"""Pipeline tests for native PDF, text, DOCX, image, and duplicate paths."""

from __future__ import annotations

import os
import shutil
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4
from zipfile import ZipFile

import pymupdf
import pytest
from fakes import FakeEmbedder, FakePineconeIndex
from PIL import Image, ImageDraw, ImageFont

from app.core.config import Settings
from app.db.repos import (
    ChunkCreate,
    Document,
    DocumentChunk,
    DocumentPage,
    ProcessingJob,
)
from app.indexing import indexer as indexer_module
from app.ingestion import pipeline as pipeline_module
from app.ingestion.ocr import OcrError, OcrResult, OcrWord
from app.ingestion.pipeline import (
    PIPELINE_VERSION,
    PipelineError,
    run_ingest_pipeline,
)
from app.ingestion.preprocess import OcrPageResult
from scripts.make_test_pdfs import GeneratedPdfs, generate_test_pdfs

TIMESTAMP = "2026-01-01T00:00:00+00:00"
DOCX_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"


class WordCounter:
    """Deterministic test tokenizer for chunk sizing."""

    def count(self, text: str) -> int:
        return len(text.split())


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


def _case_id() -> UUID:
    return uuid4()


def _document(
    *,
    case_id: UUID,
    owner_id: UUID,
    document_id: UUID | None = None,
    **overrides: Any,
) -> Document:
    resolved_id = document_id or uuid4()
    row: dict[str, Any] = {
        "id": str(resolved_id),
        "case_id": str(case_id),
        "owner_user_id": str(owner_id),
        "storage_path": f"{owner_id}/{case_id}/{resolved_id}",
        "original_filename": "clean.pdf",
        "detected_mime": "application/pdf",
        "sha256": "a" * 64,
        "size_bytes": 1024,
        "page_count": 1,
        "document_type": None,
        "version_group_id": None,
        "duplicate_of": None,
        "quality_status": "pending",
        "quality_report": {},
        "processing_status": "uploaded",
        "parser_version": None,
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    }
    row.update(overrides)
    return Document.model_validate(row)


def _job(case_id: UUID, document_id: UUID) -> ProcessingJob:
    return ProcessingJob.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "document_id": str(document_id),
            "job_type": "ingest",
            "idempotency_key": f"{document_id}:1.0.0",
            "status": "running",
            "attempts": 1,
            "error_code": None,
            "error_summary": None,
            "pipeline_version": "1.0.0",
            "started_at": TIMESTAMP,
            "finished_at": None,
            "next_attempt_at": TIMESTAMP,
            "created_at": TIMESTAMP,
        }
    )


def _ocr_result(
    text: str = "OCR extracted text.",
    confidence: float = 92.0,
) -> OcrResult:
    words = [
        OcrWord(
            text=token,
            confidence=confidence,
            left=index * 120,
            top=20,
            width=100,
            height=24,
            block=1,
            paragraph=1,
            line=1,
            word=index + 1,
        )
        for index, token in enumerate(text.split())
    ]
    return OcrResult(
        text=text,
        mean_confidence=confidence,
        word_boxes=words,
        language="eng",
    )


def _ocr_attempt(
    text: str = "OCR extracted text.",
    confidence: float = 92.0,
) -> OcrPageResult:
    result = _ocr_result(text, confidence)
    return OcrPageResult(
        result=result,
        variant="original",
        transformations=[],
        original_confidence=confidence,
        preprocessed_confidence=None,
    )


def _mixed_native_and_scanned_pdf() -> bytes:
    """Build a two-page PDF with native text on page 1 and an image on page 2."""
    document = pymupdf.open()
    try:
        native = document.new_page()
        for index in range(12):
            native.insert_text(
                (72, 72 + index * 18),
                f"Native page line {index + 1} with substantive legal text.",
                fontsize=12,
            )

        scanned = document.new_page(width=612, height=792)
        image = Image.new("RGB", (1200, 1600), "white")
        draw = ImageDraw.Draw(image)
        font = ImageFont.load_default(size=42)
        draw.text(
            (100, 100),
            "SCANNED PAGE REQUIRING OCR",
            fill="black",
            font=font,
        )
        buffer = BytesIO()
        image.save(buffer, format="PNG")
        scanned.insert_image(scanned.rect, stream=buffer.getvalue())
        return document.tobytes()
    finally:
        document.close()


def _png_bytes() -> bytes:
    image = Image.new("RGB", (800, 400), "white")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=42)
    draw.text((60, 150), "STANDALONE SCAN", fill="black", font=font)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


@dataclass
class PipelineHarness:
    """In-memory stand-ins for the worker's database helpers."""

    settings: Settings
    documents: dict[str, Document]
    pages: dict[str, list[DocumentPage]] = field(default_factory=dict)
    chunks: dict[str, list[DocumentChunk]] = field(default_factory=dict)
    statuses: list[str] = field(default_factory=list)
    page_deletes: int = 0
    chunk_deletes: int = 0
    inserted_pages: list[DocumentPage] = field(default_factory=list)
    updated_pages: list[DocumentPage] = field(default_factory=list)
    inserted_chunks: list[DocumentChunk] = field(default_factory=list)
    token_counter: Any = field(default_factory=WordCounter)
    embedder: Any = field(default_factory=FakeEmbedder)
    pinecone_index: Any = field(default_factory=FakePineconeIndex)

    def add_document(self, document: Document) -> Document:
        self.documents[str(document.id)] = document
        self.pages.setdefault(str(document.id), [])
        self.chunks.setdefault(str(document.id), [])
        return document

    def add_pages(self, document: Document, pages: list[DocumentPage]) -> None:
        self.pages[str(document.id)] = list(pages)


@pytest.fixture()
def harness(monkeypatch: pytest.MonkeyPatch) -> PipelineHarness:
    state = PipelineHarness(settings=Settings(_env_file=None), documents={})

    def fake_get_document(client, document_id):
        resolved = state.documents.get(str(document_id))
        if resolved is None:
            raise AssertionError("unknown document")
        return resolved

    def fake_update_document(client, user_id, document_id, update):
        existing = state.documents[str(document_id)]
        row = existing.model_dump()
        payload = update.model_dump(mode="json", exclude_unset=True)
        row.update(payload)
        updated = Document.model_validate(row)
        state.documents[str(document_id)] = updated
        if "processing_status" in payload:
            state.statuses.append(payload["processing_status"])
        return updated

    def fake_delete_pages(client, document_id):
        state.page_deletes += 1
        state.pages[str(document_id)] = []

    def fake_delete_chunks(client, document_id):
        state.chunk_deletes += 1
        state.chunks[str(document_id)] = []

    def fake_insert_chunks(
        client, user_id, case_id, document_id, chunks: list[ChunkCreate]
    ):
        created: list[DocumentChunk] = []
        for chunk in chunks:
            row = {
                "id": str(uuid4()),
                "case_id": str(case_id),
                "document_id": str(document_id),
                "document_version_id": None,
                **chunk.model_dump(mode="json", exclude_none=True),
                "created_at": TIMESTAMP,
            }
            created.append(DocumentChunk.model_validate(row))
        state.chunks[str(document_id)] = created
        state.inserted_chunks = created
        return created

    def fake_insert_pages(client, user_id, case_id, document_id, pages):
        created: list[DocumentPage] = []
        for page in pages:
            row = {
                "id": str(uuid4()),
                "case_id": str(case_id),
                "document_id": str(document_id),
                **page.model_dump(mode="json", exclude_none=True),
                "created_at": TIMESTAMP,
            }
            created.append(DocumentPage.model_validate(row))
        state.pages[str(document_id)] = created
        state.inserted_pages = created
        return created

    def fake_get_pages(client, document_id):
        return list(state.pages.get(str(document_id), []))

    def fake_update_page(client, user_id, case_id, document_id, page):
        existing_rows = state.pages.get(str(document_id), [])
        existing = next(
            (
                row
                for row in existing_rows
                if row.page_number == page.page_number
            ),
            None,
        )
        row = {
            "id": str(existing.id) if existing else str(uuid4()),
            "case_id": str(case_id),
            "document_id": str(document_id),
            **page.model_dump(mode="json", exclude_none=False),
            "created_at": existing.created_at if existing else TIMESTAMP,
        }
        updated = DocumentPage.model_validate(row)
        state.pages[str(document_id)] = [
            updated if item.page_number == page.page_number else item
            for item in existing_rows
        ]
        state.updated_pages.append(updated)
        return updated

    monkeypatch.setattr(
        pipeline_module, "get_document_for_worker", fake_get_document
    )
    monkeypatch.setattr(pipeline_module, "update_document", fake_update_document)
    monkeypatch.setattr(
        pipeline_module, "delete_pages_for_document", fake_delete_pages
    )
    monkeypatch.setattr(
        pipeline_module, "delete_chunks_for_document", fake_delete_chunks
    )
    monkeypatch.setattr(pipeline_module, "insert_pages", fake_insert_pages)
    monkeypatch.setattr(pipeline_module, "insert_chunks", fake_insert_chunks)
    monkeypatch.setattr(
        pipeline_module, "get_pages_for_document", fake_get_pages
    )
    monkeypatch.setattr(pipeline_module, "update_page", fake_update_page)
    monkeypatch.setattr(
        indexer_module, "get_document_for_worker", fake_get_document
    )
    monkeypatch.setattr(indexer_module, "update_document", fake_update_document)
    monkeypatch.setattr(
        indexer_module,
        "get_chunks_for_document",
        lambda client, document_id: list(state.chunks.get(str(document_id), [])),
    )

    def fake_mark_indexed_for_document(client, chunk_ids, *, embedding_model):
        wanted = {str(value) for value in chunk_ids}
        marked = 0
        for chunks in state.chunks.values():
            for chunk in chunks:
                if str(chunk.id) in wanted:
                    chunk.index_status = "indexed"
                    chunk.embedding_model = embedding_model
                    marked += 1
        return marked

    monkeypatch.setattr(
        indexer_module, "mark_chunks_indexed", fake_mark_indexed_for_document
    )
    return state


@pytest.fixture()
def pdfs(tmp_path: Path) -> GeneratedPdfs:
    return generate_test_pdfs(tmp_path / "pdfs")


def _run(
    harness: PipelineHarness,
    document: Document,
    data: bytes,
):
    job = _job(document.case_id, document.id)
    return run_ingest_pipeline(
        MagicMock(),
        job,
        settings=harness.settings,
        downloader=lambda path: data,
        token_counter=harness.token_counter,
        embedder=harness.embedder,
        pinecone_index=harness.pinecone_index,
    )


def test_clean_pdf_reaches_ready_with_indexed_chunks(
    harness: PipelineHarness, pdfs: GeneratedPdfs
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(
        _document(
            case_id=case_id,
            owner_id=owner_id,
            quality_report={"error_code": "STORAGE_ERROR"},
        )
    )

    outcome = _run(harness, document, pdfs.clean_native.read_bytes())

    assert outcome.status == "ready"
    assert outcome.page_count == 1
    assert outcome.needs_ocr_pages == 0
    assert outcome.chunk_count >= 1
    assert outcome.indexed_count == outcome.chunk_count
    assert harness.statuses[:4] == [
        "validating",
        "extracting",
        "extracting",
        "ocr",
    ]
    assert harness.statuses[-4:] == ["chunking", "embedding", "indexing", "ready"]
    assert harness.page_deletes == 1
    assert harness.chunk_deletes == 1

    stored = harness.documents[str(document.id)]
    assert stored.processing_status == "ready"
    assert stored.parser_version == "1.0.0"
    assert stored.page_count == 1
    assert stored.quality_status == "ready"
    assert stored.quality_report["pipeline_version"] == PIPELINE_VERSION
    assert stored.quality_report["parser_version"] == "1.0.0"
    assert stored.quality_report["pages_total"] == 1
    assert stored.quality_report["pages_native"] == 1
    assert stored.quality_report["pages_ocr"] == 0
    assert stored.quality_report["pages_empty"] == 0
    assert stored.quality_report["low_confidence_pages"] == []
    assert stored.quality_report["parser_errors"] == []
    assert "error_code" not in stored.quality_report
    assert stored.quality_report["indexing"]["chunk_count"] == outcome.chunk_count
    assert (
        stored.quality_report["indexing"]["indexed_vector_count"]
        == outcome.indexed_count
    )

    assert len(harness.pages[str(document.id)]) == 1
    page = harness.pages[str(document.id)][0]
    assert page.extraction_method == "native"
    assert "STATEMENT OF FACTS" in (page.raw_text or "")
    assert page.quality_flags == []
    assert page.clean_text

    chunks = harness.chunks[str(document.id)]
    assert len(chunks) == outcome.chunk_count
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert chunks[0].index_status == "indexed"
    assert chunks[0].embedding_model == harness.settings.embedding_model
    assert chunks[0].source_anchor == (
        f"doc:{document.id}|p:1-1|c:0"
    )

    namespace = str(case_id)
    assert len(harness.pinecone_index.vectors[namespace]) == outcome.chunk_count


def test_image_only_pdf_ocr_reaches_ready_with_chunks(
    harness: PipelineHarness,
    pdfs: GeneratedPdfs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(
        _document(case_id=case_id, owner_id=owner_id)
    )
    rendered_pages: list[int] = []

    def fake_render(data: bytes, page_number: int, *, dpi: int):
        rendered_pages.append(page_number)
        assert dpi == harness.settings.ocr_dpi
        return object()

    monkeypatch.setattr(pipeline_module, "render_pdf_page", fake_render)
    monkeypatch.setattr(
        pipeline_module,
        "ocr_page_with_fallback",
        lambda image, **kwargs: _ocr_attempt(
            "Image-only scanned page", confidence=94.0
        ),
    )

    outcome = _run(harness, document, pdfs.image_only.read_bytes())

    assert outcome.status == "ready"
    assert outcome.needs_ocr_pages == 0
    assert rendered_pages == [1]
    stored = harness.documents[str(document.id)]
    assert stored.quality_status == "ready"
    assert stored.quality_report["pages_ocr"] == 1
    assert stored.quality_report["pages_empty"] == 0
    assert stored.quality_report["low_confidence_pages"] == []

    page = harness.pages[str(document.id)][0]
    assert page.extraction_method == "ocr"
    assert page.raw_text == "Image-only scanned page"
    assert page.clean_text == "Image-only scanned page"
    assert page.ocr_confidence == 94.0
    assert "needs_ocr" not in page.quality_flags
    variant_selection = next(
        entry
        for entry in page.transform_log
        if entry.get("step") == "variant_selection"
    )
    assert variant_selection["winner"] == "original"

    chunks = harness.chunks[str(document.id)]
    assert len(chunks) == outcome.chunk_count
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1
    assert "Image-only scanned page" in chunks[0].text_content


def test_ocr_failure_is_recorded_as_safe_parser_error(
    harness: PipelineHarness,
    pdfs: GeneratedPdfs,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(
        _document(case_id=case_id, owner_id=owner_id)
    )
    monkeypatch.setattr(
        pipeline_module,
        "render_pdf_page",
        lambda data, page_number, *, dpi: object(),
    )

    def fail_ocr(image, **kwargs):
        raise OcrError("OCR_FAILED")

    monkeypatch.setattr(
        pipeline_module, "ocr_page_with_fallback", fail_ocr
    )

    outcome = _run(harness, document, pdfs.image_only.read_bytes())

    assert outcome.status == "failed"
    assert outcome.needs_ocr_pages == 1
    assert outcome.chunk_count == 0
    stored = harness.documents[str(document.id)]
    assert stored.quality_status == "failed"
    assert stored.processing_status == "failed"
    assert stored.quality_report["parser_errors"] == [
        {"page_number": 1, "error_code": "OCR_FAILED"}
    ]
    page = harness.pages[str(document.id)][0]
    assert page.extraction_method == "none"
    assert page.clean_text is None
    assert "ocr_failed" in page.quality_flags
    assert "empty_page" in page.quality_flags
    assert harness.chunks[str(document.id)] == []


@pytest.mark.skipif(
    TESSERACT is None, reason="Tesseract binary is not available"
)
def test_image_only_pdf_with_real_tesseract_reaches_ready(
    harness: PipelineHarness,
    pdfs: GeneratedPdfs,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(
        _document(case_id=case_id, owner_id=owner_id)
    )
    harness.settings = harness.settings.model_copy(
        update={"tesseract_cmd": TESSERACT}
    )

    outcome = _run(harness, document, pdfs.image_only.read_bytes())

    assert outcome.status == "ready"
    assert outcome.needs_ocr_pages == 0
    assert outcome.chunk_count >= 1
    stored = harness.documents[str(document.id)]
    assert stored.quality_status == "ready"
    page = harness.pages[str(document.id)][0]
    assert page.extraction_method == "ocr"
    assert "SCANNED" in (page.raw_text or "").upper()
    assert "SCANNED" in (page.clean_text or "").upper()
    assert page.ocr_confidence is not None
    assert page.ocr_confidence >= harness.settings.ocr_confidence_threshold
    assert "low_ocr_confidence" not in page.quality_flags
    chunks = harness.chunks[str(document.id)]
    assert len(chunks) == outcome.chunk_count
    assert any("SCANNED" in chunk.text_content.upper() for chunk in chunks)


def test_mixed_native_and_scanned_pdf_is_routed_page_by_page(
    harness: PipelineHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(
        _document(case_id=case_id, owner_id=owner_id)
    )
    rendered_pages: list[int] = []

    def fake_render(data: bytes, page_number: int, *, dpi: int):
        rendered_pages.append(page_number)
        return object()

    def fake_ocr(image, **kwargs):
        return _ocr_attempt("Scanned page evidence.", confidence=90.0)

    monkeypatch.setattr(pipeline_module, "render_pdf_page", fake_render)
    monkeypatch.setattr(
        pipeline_module, "ocr_page_with_fallback", fake_ocr
    )

    outcome = _run(harness, document, _mixed_native_and_scanned_pdf())

    assert outcome.status == "ready"
    assert outcome.page_count == 2
    assert outcome.chunk_count >= 1
    assert rendered_pages == [2]
    first_page, second_page = harness.pages[str(document.id)]
    assert first_page.extraction_method == "native"
    assert "Native page line 1" in (first_page.raw_text or "")
    assert second_page.extraction_method == "ocr"
    assert second_page.raw_text == "Scanned page evidence."
    assert harness.documents[str(document.id)].quality_status == "ready"

    chunks = harness.chunks[str(document.id)]
    assert chunks
    assert all(1 <= chunk.page_start <= chunk.page_end <= 2 for chunk in chunks)
    assert any(
        "Native page line 1" in chunk.text_content for chunk in chunks
    )
    assert any("Scanned page evidence." in chunk.text_content for chunk in chunks)


def test_running_pipeline_twice_does_not_duplicate_chunks(
    harness: PipelineHarness, pdfs: GeneratedPdfs
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(
        _document(case_id=case_id, owner_id=owner_id)
    )
    data = pdfs.clean_native.read_bytes()

    first = _run(harness, document, data)
    first_hashes = [
        chunk.content_hash for chunk in harness.chunks[str(document.id)]
    ]
    second = _run(harness, harness.documents[str(document.id)], data)

    assert first.page_count == second.page_count == 1
    assert first.chunk_count == second.chunk_count
    assert len(harness.pages[str(document.id)]) == 1
    assert len(harness.chunks[str(document.id)]) == second.chunk_count
    assert [
        chunk.content_hash for chunk in harness.chunks[str(document.id)]
    ] == first_hashes
    assert harness.page_deletes == 2
    assert harness.chunk_deletes == 2


def test_duplicate_copies_original_pages_without_downloading(
    harness: PipelineHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    original = harness.add_document(
        _document(
            case_id=case_id,
            owner_id=owner_id,
            parser_version="1.0.0",
            processing_status="chunking",
        )
    )
    harness.add_pages(
        original,
        [
            DocumentPage.model_validate(
                {
                    "id": str(uuid4()),
                    "case_id": str(case_id),
                    "document_id": str(original.id),
                    "page_number": 1,
                    "extraction_method": "native",
                    "text_density": 0.01,
                    "ocr_confidence": None,
                    "raw_text": "Original extracted text.",
                    "clean_text": "Original extracted text.",
                    "quality_flags": [],
                    "transform_log": [{"stage": "extract"}],
                    "created_at": TIMESTAMP,
                }
            )
        ],
    )
    duplicate = harness.add_document(
        _document(
            case_id=case_id,
            owner_id=owner_id,
            duplicate_of=original.id,
            sha256="b" * 64,
        )
    )
    job = _job(case_id, duplicate.id)

    def fail_download(path: str) -> bytes:
        raise AssertionError("Duplicate documents must not be downloaded")

    outcome = run_ingest_pipeline(
        MagicMock(),
        job,
        settings=harness.settings,
        downloader=fail_download,
        token_counter=harness.token_counter,
        embedder=harness.embedder,
        pinecone_index=harness.pinecone_index,
    )

    assert outcome.status == "ready"
    assert outcome.page_count == 1
    assert outcome.chunk_count >= 1
    page = harness.pages[str(duplicate.id)][0]
    assert page.raw_text == "Original extracted text."
    assert page.clean_text == "Original extracted text."
    assert page.transform_log[1]["action"] == "copied_duplicate"
    assert page.transform_log[1]["source_document_id"] == str(original.id)
    assert harness.documents[str(duplicate.id)].parser_version == "1.0.0"
    assert len(harness.chunks[str(duplicate.id)]) == outcome.chunk_count


def test_duplicate_waits_until_original_ocr_is_complete(
    harness: PipelineHarness,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    original = harness.add_document(
        _document(case_id=case_id, owner_id=owner_id)
    )
    harness.add_pages(
        original,
        [
            DocumentPage.model_validate(
                {
                    "id": str(uuid4()),
                    "case_id": str(case_id),
                    "document_id": str(original.id),
                    "page_number": 1,
                    "extraction_method": "none",
                    "text_density": 0.0,
                    "ocr_confidence": None,
                    "raw_text": None,
                    "clean_text": None,
                    "quality_flags": ["needs_ocr"],
                    "transform_log": [],
                    "created_at": TIMESTAMP,
                }
            )
        ],
    )
    duplicate = harness.add_document(
        _document(
            case_id=case_id,
            owner_id=owner_id,
            duplicate_of=original.id,
            sha256="b" * 64,
        )
    )

    with pytest.raises(PipelineError) as error:
        run_ingest_pipeline(
            MagicMock(),
            _job(case_id, duplicate.id),
            settings=harness.settings,
            downloader=lambda path: b"",
            token_counter=harness.token_counter,
        )

    assert error.value.error_code == "DUPLICATE_SOURCE_NOT_READY"
    assert error.value.retryable is True


def test_txt_is_one_native_page(harness: PipelineHarness) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(
        _document(
            case_id=case_id,
            owner_id=owner_id,
            original_filename="notes.txt",
            detected_mime="text/plain",
            page_count=None,
        )
    )

    outcome = _run(harness, document, b"First paragraph.\n\nSecond paragraph.")

    assert outcome.status == "ready"
    assert outcome.page_count == 1
    assert outcome.chunk_count >= 1
    page = harness.pages[str(document.id)][0]
    assert page.extraction_method == "native"
    assert page.raw_text == "First paragraph.\n\nSecond paragraph."
    assert page.clean_text == "First paragraph.\n\nSecond paragraph."
    assert harness.chunks[str(document.id)]


def test_docx_is_one_native_page(harness: PipelineHarness) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(
        _document(
            case_id=case_id,
            owner_id=owner_id,
            original_filename="brief.docx",
            detected_mime=(
                "application/vnd.openxmlformats-officedocument."
                "wordprocessingml.document"
            ),
            page_count=None,
        )
    )
    xml = (
        f'<w:document xmlns:w="{DOCX_NAMESPACE}"><w:body>'
        "<w:p><w:r><w:t>First paragraph</w:t></w:r></w:p>"
        "<w:p><w:r><w:t>Second paragraph</w:t></w:r></w:p>"
        "</w:body></w:document>"
    )
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", xml)

    outcome = _run(harness, document, buffer.getvalue())

    assert outcome.status == "ready"
    assert outcome.page_count == 1
    assert outcome.chunk_count >= 1
    page = harness.pages[str(document.id)][0]
    assert page.extraction_method == "native"
    assert "First paragraph" in (page.raw_text or "")
    assert "Second paragraph" in (page.raw_text or "")
    assert harness.chunks[str(document.id)]


def test_standalone_image_is_ocr_processed(
    harness: PipelineHarness,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(
        _document(
            case_id=case_id,
            owner_id=owner_id,
            original_filename="scan.png",
            detected_mime="image/png",
            page_count=None,
        )
    )
    monkeypatch.setattr(
        pipeline_module,
        "ocr_page_with_fallback",
        lambda image, **kwargs: _ocr_attempt(
            "Standalone scan", confidence=91.0
        ),
    )

    outcome = _run(harness, document, _png_bytes())

    assert outcome.status == "ready"
    assert outcome.needs_ocr_pages == 0
    assert outcome.chunk_count >= 1
    page = harness.pages[str(document.id)][0]
    assert page.extraction_method == "ocr"
    assert page.raw_text == "Standalone scan"
    assert page.clean_text == "Standalone scan"
    assert page.ocr_confidence == 91.0
    assert harness.chunks[str(document.id)]


def test_corrupt_pdf_raises_safe_pipeline_error(harness: PipelineHarness) -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = harness.add_document(
        _document(case_id=case_id, owner_id=owner_id)
    )

    with pytest.raises(PipelineError) as error:
        _run(harness, document, b"%PDF-1.7\nnot a real document")

    assert error.value.error_code == "PROCESSING_FAILED"
    assert "not a real document" not in str(error.value)
