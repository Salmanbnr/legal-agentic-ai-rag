"""Ingestion pipeline stages for extraction, OCR, quality, chunking, and indexing.

The pipeline is split into small stage functions. Normalization and chunking
persist derived rows, then the indexing stage embeds pending chunks, upserts
them to Pinecone, verifies counts, and marks the document ready or needs_review.
"""

from __future__ import annotations

import logging
import zipfile
from dataclasses import dataclass, field
from io import BytesIO
from pathlib import Path
from typing import Any, Callable, Sequence
from xml.etree import ElementTree

from PIL import Image
from supabase import Client

from app.core.config import Settings, get_settings
from app.db.repos import (
    ChunkCreate,
    Document,
    DocumentChunk,
    DocumentUpdate,
    PageCreate,
    ProcessingJob,
    insert_chunks,
    insert_pages,
    update_document,
    update_page,
)
from app.embeddings.model import EmbeddingError, TextEmbedder
from app.indexing.indexer import IndexingError, index_document_chunks
from app.indexing.pinecone_store import PineconeError
from app.ingestion.chunking import (
    CHUNKING_VERSION,
    ChunkingError,
    TokenCounter,
    chunk_pages,
    load_token_counter,
)
from app.ingestion.normalize import normalize_pages
from app.ingestion.ocr import (
    OCR_VERSION,
    OcrError,
    OcrWord,
    load_image_page,
    render_pdf_page,
)
from app.ingestion.pdf_native import (
    ExtractedPage,
    PdfExtractionError,
    extract_pdf_pages,
)
from app.ingestion.preprocess import OcrPageResult, ocr_page_with_fallback
from app.ingestion.quality import assess_quality, evaluate_page_flags
from app.jobs.store import (
    delete_chunks_for_document,
    delete_pages_for_document,
    get_document_for_worker,
    get_pages_for_document,
)
from app.storage.supabase_storage import StorageError, download_original

PIPELINE_VERSION = "1.0.0"
PARSER_VERSION = "1.0.0"

_MAX_DOCX_XML_BYTES = 20 * 1024 * 1024
_WORD_NAMESPACE = "http://schemas.openxmlformats.org/wordprocessingml/2006/main"
_IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tiff"}

logger = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """A safe pipeline failure carrying a stable, persistable error code."""

    def __init__(self, error_code: str, *, retryable: bool = True) -> None:
        """Create a pipeline error without parser or file content details."""
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable


@dataclass(frozen=True, slots=True)
class PipelineOutcome:
    """Result of a pipeline run through indexing."""

    status: str
    page_count: int
    needs_ocr_pages: int
    chunk_count: int = 0
    indexed_count: int = 0


@dataclass
class PipelineContext:
    """Mutable state shared by pipeline stages."""

    client: Client
    job: ProcessingJob
    settings: Settings
    downloader: Callable[[str], bytes]
    document: Document
    token_counter: TokenCounter | None = None
    embedder: TextEmbedder | None = None
    pinecone_index: Any | None = None
    pages: list[PageCreate] = field(default_factory=list)
    chunks: list[DocumentChunk] = field(default_factory=list)
    needs_ocr_pages: int = 0
    indexed_count: int = 0
    source_data: bytes | None = None
    source_type: str = "unknown"
    parser_errors: list[dict[str, Any]] = field(default_factory=list)


def _set_stage(context: PipelineContext, stage: str, **fields: Any) -> Document:
    """Persist a processing stage transition and refresh the document."""
    document = update_document(
        context.client,
        context.document.owner_user_id,
        context.document.id,
        DocumentUpdate(processing_status=stage, **fields),
    )
    context.document = document
    return document


def _quality_report(document: Document, **updates: Any) -> dict[str, Any]:
    """Merge safe pipeline metadata into a document's quality report."""
    report = dict(document.quality_report or {})
    report.update(updates)
    return report


def _source_type(document: Document) -> str:
    """Classify a document's source format for the OCR stage."""
    mime = (document.detected_mime or "").lower()
    extension = Path(document.original_filename).suffix.lower()
    if mime == "application/pdf" or extension == ".pdf":
        return "pdf"
    if mime.startswith("image/") or extension in _IMAGE_EXTENSIONS:
        return "image"
    if extension == ".txt" or mime == "text/plain":
        return "text"
    if (
        extension == ".docx"
        or mime
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        return "docx"
    return "unknown"


def _apply_page_flags(
    page: PageCreate,
    context: PipelineContext,
    *,
    words: Sequence[OcrWord] = (),
    ocr_completed: bool = False,
    ocr_failed: bool = False,
) -> None:
    """Update one page's quality flags without altering its text."""
    page.quality_flags = evaluate_page_flags(
        text=page.raw_text or page.clean_text,
        extraction_method=page.extraction_method,
        ocr_confidence=page.ocr_confidence,
        words=words,
        existing_flags=page.quality_flags,
        settings=context.settings,
        ocr_completed=ocr_completed,
        ocr_failed=ocr_failed,
    )


def stage_validating(context: PipelineContext) -> None:
    """Validate job/document consistency before reading any file content."""
    if context.job.job_type != "ingest":
        raise PipelineError("UNSUPPORTED_JOB_TYPE", retryable=False)
    if context.job.case_id != context.document.case_id:
        raise PipelineError("PROCESSING_FAILED", retryable=False)
    if context.document.quality_status == "password_required":
        raise PipelineError("PASSWORD_REQUIRED", retryable=False)
    report = context.document.quality_report or {}
    if report.get("deletion_requested"):
        raise PipelineError("DOCUMENT_DELETION_REQUESTED", retryable=False)
    _set_stage(context, "validating", parser_version=PARSER_VERSION)


def _decode_text(data: bytes) -> str:
    """Decode validated UTF-8 or BOM-marked UTF-16 text."""
    encoding = "utf-16" if data.startswith((b"\xff\xfe", b"\xfe\xff")) else "utf-8-sig"
    try:
        text = data.decode(encoding)
    except UnicodeDecodeError as exc:
        raise PipelineError("PROCESSING_FAILED", retryable=False) from exc
    return text.replace("\r\n", "\n").replace("\r", "\n")


def _extract_txt_pages(data: bytes) -> list[PageCreate]:
    """Treat a text file as one native page."""
    text = _decode_text(data).strip()
    if not text:
        raise PipelineError("EMPTY_DOCUMENT", retryable=False)
    return [
        PageCreate(
            page_number=1,
            extraction_method="native",
            raw_text=text,
            quality_flags=[],
            transform_log=[
                {"stage": "extract", "source": "txt", "parser_version": PARSER_VERSION}
            ],
        )
    ]


def _docx_text(data: bytes) -> str:
    """Extract paragraph text from a DOCX container without new dependencies."""
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            info = archive.getinfo("word/document.xml")
            if info.file_size > _MAX_DOCX_XML_BYTES:
                raise PipelineError("PROCESSING_FAILED", retryable=False)
            xml_bytes = archive.read(info)
    except (KeyError, OSError, RuntimeError, zipfile.BadZipFile) as exc:
        raise PipelineError("PROCESSING_FAILED", retryable=False) from exc

    try:
        root = ElementTree.fromstring(xml_bytes)
    except ElementTree.ParseError as exc:
        raise PipelineError("PROCESSING_FAILED", retryable=False) from exc

    paragraphs: list[str] = []
    paragraph_tag = f"{{{_WORD_NAMESPACE}}}p"
    text_tag = f"{{{_WORD_NAMESPACE}}}t"
    tab_tag = f"{{{_WORD_NAMESPACE}}}tab"
    break_tag = f"{{{_WORD_NAMESPACE}}}br"

    for paragraph in root.iter(paragraph_tag):
        parts: list[str] = []
        for node in paragraph.iter():
            if node.tag == text_tag:
                parts.append(node.text or "")
            elif node.tag == tab_tag:
                parts.append("\t")
            elif node.tag == break_tag:
                parts.append("\n")
        text = "".join(parts).strip()
        if text:
            paragraphs.append(text)
    return "\n\n".join(paragraphs)


def _extract_docx_pages(data: bytes) -> list[PageCreate]:
    """Treat a DOCX as one native page, preserving paragraph groups."""
    text = _docx_text(data).strip()
    if not text:
        raise PipelineError("EMPTY_DOCUMENT", retryable=False)
    return [
        PageCreate(
            page_number=1,
            extraction_method="native",
            raw_text=text,
            quality_flags=[],
            transform_log=[
                {
                    "stage": "extract",
                    "source": "docx",
                    "parser_version": PARSER_VERSION,
                }
            ],
        )
    ]


def _extract_image_pages() -> list[PageCreate]:
    """Represent a standalone image as one page awaiting OCR."""
    return [
        PageCreate(
            page_number=1,
            extraction_method="none",
            text_density=0.0,
            raw_text=None,
            quality_flags=["needs_ocr"],
            transform_log=[
                {
                    "stage": "extract",
                    "source": "image",
                    "parser_version": PARSER_VERSION,
                }
            ],
        )
    ]


def _page_from_extracted(page: ExtractedPage) -> PageCreate:
    """Convert a PDF extraction result into a database page payload."""
    return PageCreate(
        page_number=page.page_number,
        extraction_method=page.extraction_method,
        text_density=page.text_density,
        raw_text=page.raw_text or None,
        quality_flags=list(page.quality_flags),
        transform_log=[
            {
                "stage": "extract",
                "source": "pdf_native",
                "parser_version": PARSER_VERSION,
            }
        ],
    )


def _extract_source_pages(
    context: PipelineContext, data: bytes
) -> list[PageCreate]:
    """Dispatch extraction by detected MIME and filename extension."""
    document = context.document
    mime = (document.detected_mime or "").lower()
    extension = Path(document.original_filename).suffix.lower()

    if mime == "application/pdf" or extension == ".pdf":
        try:
            extracted = extract_pdf_pages(
                data,
                density_threshold=context.settings.pdf_native_text_density_threshold,
                max_pages=context.settings.max_pdf_pages,
            )
        except PdfExtractionError as exc:
            raise PipelineError(
                exc.error_code,
                retryable=exc.error_code
                not in {
                    "PASSWORD_REQUIRED",
                    "UNSUPPORTED_FILE",
                    "PAGE_LIMIT_EXCEEDED",
                },
            ) from exc
        return [_page_from_extracted(page) for page in extracted]

    if extension == ".txt" or mime == "text/plain":
        return _extract_txt_pages(data)

    if (
        extension == ".docx"
        or mime
        == "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    ):
        return _extract_docx_pages(data)

    if mime.startswith("image/") or extension in _IMAGE_EXTENSIONS:
        return _extract_image_pages()

    raise PipelineError("UNSUPPORTED_FILE", retryable=False)


def _duplicate_pages(context: PipelineContext) -> tuple[list[PageCreate], str]:
    """Copy pages from the duplicate's root document without re-extracting."""
    if context.document.duplicate_of is None:
        raise PipelineError("PROCESSING_FAILED", retryable=False)
    original = get_document_for_worker(context.client, context.document.duplicate_of)
    if original.id == context.document.id:
        raise PipelineError("PROCESSING_FAILED", retryable=False)
    if original.case_id != context.document.case_id:
        raise PipelineError("PROCESSING_FAILED", retryable=False)

    original_pages = get_pages_for_document(context.client, original.id)
    if not original_pages:
        raise PipelineError("DUPLICATE_SOURCE_NOT_READY", retryable=True)
    if any(
        page.extraction_method == "none" or "needs_ocr" in page.quality_flags
        for page in original_pages
    ):
        raise PipelineError("DUPLICATE_SOURCE_NOT_READY", retryable=True)

    pages: list[PageCreate] = []
    for page in original_pages:
        transform_log = list(page.transform_log)
        transform_log.append(
            {
                "stage": "extract",
                "action": "copied_duplicate",
                "source_document_id": str(original.id),
            }
        )
        pages.append(
            PageCreate(
                page_number=page.page_number,
                extraction_method=page.extraction_method,
                text_density=page.text_density,
                ocr_confidence=page.ocr_confidence,
                raw_text=page.raw_text,
                clean_text=page.clean_text,
                quality_flags=list(page.quality_flags),
                transform_log=transform_log,
            )
        )
    return pages, original.parser_version or PARSER_VERSION


def stage_extracting(context: PipelineContext) -> None:
    """Extract or copy pages, then replace this document's derived rows."""
    _set_stage(context, "extracting")

    if context.document.duplicate_of is not None:
        context.source_type = "duplicate"
        pages, parser_version = _duplicate_pages(context)
    else:
        try:
            data = context.downloader(context.document.storage_path)
        except StorageError as exc:
            raise PipelineError("STORAGE_ERROR", retryable=True) from exc
        context.source_data = data
        context.source_type = _source_type(context.document)
        pages = _extract_source_pages(context, data)
        parser_version = PARSER_VERSION

    if not pages:
        raise PipelineError("EMPTY_DOCUMENT", retryable=False)

    if context.document.duplicate_of is None:
        for page in pages:
            _apply_page_flags(page, context)

    # Rewriting from a clean slate keeps retries idempotent. Do this only after
    # extraction succeeds so a parser failure does not discard prior pages.
    # Chunks are deleted at the start of the chunking stage.
    delete_pages_for_document(context.client, context.document.id)
    insert_pages(
        context.client,
        context.document.owner_user_id,
        context.document.case_id,
        context.document.id,
        pages,
    )
    context.pages = pages
    context.needs_ocr_pages = sum(
        1
        for page in pages
        if "needs_ocr" in page.quality_flags
        or page.extraction_method == "none"
    )
    report = _quality_report(
        context.document,
        pipeline_version=PIPELINE_VERSION,
        parser_version=parser_version,
        needs_ocr_pages=context.needs_ocr_pages,
    )
    report.pop("error_code", None)
    if not context.needs_ocr_pages:
        report.pop("ocr_pending", None)
    quality_status = "needs_review" if context.needs_ocr_pages else "pending"
    _set_stage(
        context,
        "extracting",
        page_count=len(pages),
        parser_version=parser_version,
        quality_status=quality_status,
        quality_report=report,
    )


def _ocr_image_for_page(
    context: PipelineContext, page: PageCreate
) -> Image.Image:
    """Render or decode the source image for one OCR-needed page."""
    if context.source_data is None:
        raise OcrError("PROCESSING_FAILED")
    if context.source_type == "pdf":
        return render_pdf_page(
            context.source_data,
            page.page_number,
            dpi=context.settings.ocr_dpi,
        )
    if context.source_type == "image":
        return load_image_page(context.source_data, page.page_number)
    raise OcrError("PROCESSING_FAILED")


def _record_ocr_page(
    context: PipelineContext,
    page: PageCreate,
    attempt: OcrPageResult,
) -> None:
    """Persist the winning OCR result and its safe transformation trace."""
    page.extraction_method = "ocr"
    page.raw_text = attempt.result.text or ""
    page.clean_text = None
    page.ocr_confidence = attempt.result.mean_confidence
    _apply_page_flags(
        page,
        context,
        words=attempt.result.word_boxes,
        ocr_completed=True,
    )
    page.transform_log = list(page.transform_log) + [
        {
            "stage": "ocr",
            "step": "render",
            "source": context.source_type,
            "page_number": page.page_number,
            "dpi": context.settings.ocr_dpi,
        },
        *attempt.transformations,
        {
            "stage": "ocr",
            "step": "variant_selection",
            "winner": attempt.variant,
            "original_confidence": round(attempt.original_confidence, 2),
            "preprocessed_confidence": (
                round(attempt.preprocessed_confidence, 2)
                if attempt.preprocessed_confidence is not None
                else None
            ),
            "mean_confidence": round(attempt.result.mean_confidence, 2),
            "threshold": context.settings.ocr_confidence_threshold,
        },
        {
            "stage": "ocr",
            "step": "result",
            "language": context.settings.ocr_language,
            "word_count": attempt.result.word_count,
            "ocr_version": OCR_VERSION,
        },
    ]
    update_page(
        context.client,
        context.document.owner_user_id,
        context.document.case_id,
        context.document.id,
        page,
    )


def _record_ocr_failure(
    context: PipelineContext,
    page: PageCreate,
    error_code: str,
) -> None:
    """Persist one page-level OCR failure and keep the document moving."""
    context.parser_errors.append(
        {"page_number": page.page_number, "error_code": error_code}
    )
    page.extraction_method = "none"
    page.ocr_confidence = None
    _apply_page_flags(
        page,
        context,
        ocr_completed=True,
        ocr_failed=True,
    )
    page.transform_log = list(page.transform_log) + [
        {
            "stage": "ocr",
            "step": "failed",
            "error_code": error_code,
        }
    ]
    update_page(
        context.client,
        context.document.owner_user_id,
        context.document.case_id,
        context.document.id,
        page,
    )


def stage_ocr(context: PipelineContext) -> None:
    """OCR pages flagged ``needs_ocr`` with one bounded preprocessing retry."""
    _set_stage(context, "ocr")
    pending = [
        page
        for page in context.pages
        if "needs_ocr" in page.quality_flags
        or page.extraction_method == "none"
    ]
    if not pending:
        context.needs_ocr_pages = 0
        return

    unresolved = 0
    for page in pending:
        try:
            image = _ocr_image_for_page(context, page)
            attempt = ocr_page_with_fallback(
                image,
                language=context.settings.ocr_language,
                confidence_threshold=context.settings.ocr_confidence_threshold,
                tesseract_cmd=context.settings.tesseract_cmd,
                enable_preprocess=context.settings.ocr_preprocess_enabled,
                max_deskew_degrees=context.settings.ocr_max_deskew_degrees,
            )
            _record_ocr_page(context, page, attempt)
        except OcrError as exc:
            if exc.error_code == "TESSERACT_UNAVAILABLE":
                raise PipelineError(
                    "TESSERACT_UNAVAILABLE", retryable=False
                ) from exc
            _record_ocr_failure(context, page, exc.error_code)
            unresolved += 1

    context.needs_ocr_pages = unresolved
    logger.info(
        "OCR stage finished",
        extra={
            "job_id": str(context.job.id),
            "document_id": str(context.document.id),
            "ocr_pages": len(pending) - unresolved,
            "failed_pages": unresolved,
        },
    )


def stage_quality(context: PipelineContext) -> None:
    """Persist the transparent quality report and mapped quality status."""
    report, status = assess_quality(
        context.pages,
        confidence_threshold=context.settings.ocr_confidence_threshold,
        review_ratio_threshold=context.settings.quality_review_ratio_threshold,
        parser_error_threshold=context.settings.quality_parser_error_threshold,
        parser_errors=context.parser_errors,
    )
    report_payload = report.model_dump(mode="json")
    report_payload["pipeline_version"] = PIPELINE_VERSION
    report_payload["parser_version"] = (
        context.document.parser_version or PARSER_VERSION
    )
    _set_stage(
        context,
        "ocr",
        quality_status=status,
        quality_report=report_payload,
    )
    logger.info(
        "Quality assessment finished",
        extra={
            "job_id": str(context.job.id),
            "document_id": str(context.document.id),
            "quality_status": status,
            "pages_total": report.pages_total,
            "pages_empty": report.pages_empty,
            "low_confidence_pages": len(report.low_confidence_pages),
            "parser_errors": len(report.parser_errors),
        },
    )


def stage_normalizing(context: PipelineContext) -> None:
    """Normalize page text for retrieval while preserving ``raw_text``."""
    normalized = normalize_pages(
        context.pages,
        header_footer_repeat_ratio=context.settings.header_footer_repeat_ratio,
        header_footer_max_lines=context.settings.header_footer_max_lines,
    )
    by_page = {result.page_number: result for result in normalized}
    updated = 0
    for page in context.pages:
        result = by_page.get(page.page_number)
        if result is None:
            continue
        page.clean_text = result.clean_text
        page.transform_log = list(page.transform_log) + result.transformations
        update_page(
            context.client,
            context.document.owner_user_id,
            context.document.case_id,
            context.document.id,
            page,
        )
        updated += 1
    logger.info(
        "Normalization stage finished",
        extra={
            "job_id": str(context.job.id),
            "document_id": str(context.document.id),
            "normalized_pages": updated,
        },
    )


def stage_chunking(context: PipelineContext) -> None:
    """Build structure-aware chunks and persist them as pending."""
    _set_stage(context, "chunking")
    delete_chunks_for_document(context.client, context.document.id)

    counter = context.token_counter
    if counter is None:
        try:
            counter = load_token_counter(context.settings)
        except ChunkingError as exc:
            raise PipelineError(exc.error_code, retryable=False) from exc

    try:
        chunks: list[ChunkCreate] = chunk_pages(
            context.pages,
            document_id=context.document.id,
            token_counter=counter,
            min_tokens=context.settings.chunk_min_tokens,
            max_tokens=context.settings.chunk_max_tokens,
            overlap_min_tokens=context.settings.chunk_overlap_min_tokens,
            overlap_max_tokens=context.settings.chunk_overlap_max_tokens,
            index_schema_version=context.settings.index_schema_version,
        )
    except ChunkingError as exc:
        raise PipelineError(exc.error_code, retryable=False) from exc

    if chunks:
        context.chunks = insert_chunks(
            context.client,
            context.document.owner_user_id,
            context.document.case_id,
            context.document.id,
            chunks,
        )
    else:
        context.chunks = []

    logger.info(
        "Chunking stage finished",
        extra={
            "job_id": str(context.job.id),
            "document_id": str(context.document.id),
            "chunk_count": len(context.chunks),
            "pipeline_version": PIPELINE_VERSION,
            "chunking_version": CHUNKING_VERSION,
        },
    )


def stage_embedding(context: PipelineContext) -> None:
    """Enter the embedding stage and record its boundary."""
    _set_stage(context, "embedding")
    logger.info(
        "Embedding stage started",
        extra={
            "job_id": str(context.job.id),
            "document_id": str(context.document.id),
            "chunk_count": len(context.chunks),
            "embedding_model": context.settings.embedding_model,
        },
    )


def stage_indexing(context: PipelineContext) -> None:
    """Embed pending chunks, upsert vectors, and verify the index count."""
    try:
        result = index_document_chunks(
            context.client,
            context.document.id,
            settings=context.settings,
            embedder=context.embedder,
            index=context.pinecone_index,
        )
    except (IndexingError, EmbeddingError, PineconeError) as exc:
        raise PipelineError(exc.error_code, retryable=exc.retryable) from exc

    context.document = result.document
    context.indexed_count = result.indexed_count
    logger.info(
        "Indexing stage finished",
        extra={
            "job_id": str(context.job.id),
            "document_id": str(context.document.id),
            "chunk_count": result.chunk_count,
            "indexed_count": result.indexed_count,
            "status": result.status,
        },
    )


def stage_ready(context: PipelineContext) -> None:
    """Mark a document ready when a caller handles readiness explicitly."""
    _set_stage(context, "ready", quality_status="ready")


def run_ingest_pipeline(
    client: Client,
    job: ProcessingJob,
    *,
    settings: Settings | None = None,
    downloader: Callable[[str], bytes] | None = None,
    token_counter: TokenCounter | None = None,
    embedder: TextEmbedder | None = None,
    pinecone_index: Any | None = None,
) -> PipelineOutcome:
    """Run the ingest pipeline through chunking and Pinecone indexing."""
    resolved = settings or get_settings()
    document = get_document_for_worker(client, job.document_id)
    download = downloader or (
        lambda storage_path: download_original(storage_path, client=client)
    )
    context = PipelineContext(
        client=client,
        job=job,
        settings=resolved,
        downloader=download,
        document=document,
        token_counter=token_counter,
        embedder=embedder,
        pinecone_index=pinecone_index,
    )

    stage_validating(context)
    stage_extracting(context)
    stage_ocr(context)
    stage_quality(context)
    stage_normalizing(context)
    stage_chunking(context)
    stage_embedding(context)
    stage_indexing(context)

    return PipelineOutcome(
        status=context.document.processing_status,
        page_count=len(context.pages),
        needs_ocr_pages=context.needs_ocr_pages,
        chunk_count=len(context.chunks),
        indexed_count=context.indexed_count,
    )
