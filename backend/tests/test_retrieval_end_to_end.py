"""End-to-end retrieval test over a generated PDF with the real embedding model.

The test ingests a small legal PDF, chunks it, embeds it with the configured
sentence-transformers model, stores the vectors in the fake index, and then
searches the case with a keyword and with a paraphrased question. Both queries
must return the page-cited chunk that actually answers them.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest
from fakes import FakePineconeIndex
from legal_fixtures import (
    EXPECTED_TEXT,
    chunk_rows,
    extracted_pages,
    fixture_settings,
    pdf_bytes,
)

from app.auth.dependencies import CurrentUser
from app.db.repos import (
    Case,
    Document,
    DocumentChunk,
    DocumentPage,
)
from app.embeddings.model import EmbeddingError, get_embedder
from app.indexing.pinecone_store import namespace_for_case
from app.ingestion.chunking import ChunkingError, load_token_counter
from app.retrieval import hybrid as hybrid_module
from app.retrieval.hybrid import hybrid_search
from app.retrieval.lexical import lexical_search

TIMESTAMP = "2026-01-01T00:00:00+00:00"


class FakeLexicalQuery:
    """RPC query builder that returns the matched rows on execute."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        """Store the rows the function should return."""
        self._rows = rows

    def execute(self) -> Any:
        """Return the stored rows as a response payload."""
        from types import SimpleNamespace

        return SimpleNamespace(data=list(self._rows))


class FakeLexicalClient:
    """Service-client stand-in that mimics ``search_chunks_lexical``."""

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        """Store the chunks the function searches over."""
        self.chunks = chunks
        self.rpc_calls: list[tuple[str, dict[str, Any]]] = []

    def rpc(self, name: str, params: dict[str, Any]) -> FakeLexicalQuery:
        """Return the chunks matching the query terms."""
        self.rpc_calls.append((name, dict(params)))
        query = str(params.get("p_query") or "")
        allowed = {str(value) for value in params.get("p_document_ids") or []}
        terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if term]
        rows: list[dict[str, Any]] = []
        for chunk in self.chunks:
            if str(chunk.document_id) not in allowed:
                continue
            text = chunk.text_content.lower()
            matches = sum(1 for term in terms if term in text)
            if not matches:
                continue
            rows.append(
                {
                    "id": str(chunk.id),
                    "document_id": str(chunk.document_id),
                    "rank": matches / max(1, len(terms)),
                }
            )
        rows.sort(key=lambda row: -row["rank"])
        limit = int(params.get("p_limit") or 20)
        return FakeLexicalQuery(rows[:limit])


@dataclass
class RetrievalFixture:
    """In-memory case data used by the end-to-end search."""

    settings: Settings
    case: Case
    document: Document
    chunks: list[DocumentChunk]
    pages: list[DocumentPage]
    index: FakePineconeIndex
    client: FakeLexicalClient
    embedder: Any = None
    neighbours: dict[str, list[DocumentChunk]] = field(default_factory=dict)


@pytest.fixture()
def retrieval_fixture(monkeypatch: pytest.MonkeyPatch) -> RetrievalFixture:
    """Ingest a PDF, embed it, and wire the retrieval layer to memory."""
    settings = fixture_settings()
    try:
        embedder = get_embedder(settings)
    except EmbeddingError:
        pytest.skip("the embedding model is not available locally")

    case_id = uuid4()
    owner_id = uuid4()
    document_id = uuid4()
    case = Case.model_validate(
        {
            "id": str(case_id),
            "owner_user_id": str(owner_id),
            "title": "Rent dispute",
            "description": None,
            "status": "active",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    )
    document = Document.model_validate(
        {
            "id": str(document_id),
            "case_id": str(case_id),
            "owner_user_id": str(owner_id),
            "storage_path": f"{owner_id}/{case_id}/{document_id}",
            "original_filename": "rent-dispute.pdf",
            "detected_mime": "application/pdf",
            "sha256": "a" * 64,
            "size_bytes": 1024,
            "page_count": 3,
            "document_type": None,
            "version_group_id": None,
            "duplicate_of": None,
            "quality_status": "ready",
            "quality_report": {},
            "processing_status": "ready",
            "parser_version": "1.0.0",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    )

    try:
        pages = extracted_pages(pdf_bytes())
    except Exception as exc:  # pragma: no cover - defensive
        pytest.fail(f"PDF fixture could not be extracted: {exc}")
    chunks = chunk_rows(pages, document_id, case_id)
    if not chunks:
        pytest.fail("the fixture produced no chunks")

    page_rows = [
        DocumentPage.model_validate(
            {
                "id": str(uuid4()),
                "case_id": str(case_id),
                "document_id": str(document_id),
                "page_number": page.page_number,
                "extraction_method": "native",
                "text_density": 0.01,
                "ocr_confidence": None,
                "raw_text": page.raw_text,
                "clean_text": page.clean_text,
                "quality_flags": list(page.quality_flags),
                "transform_log": list(page.transform_log),
                "created_at": TIMESTAMP,
            }
        )
        for page in pages
    ]

    index = FakePineconeIndex()
    namespace = namespace_for_case(case_id)
    vectors = embedder.embed_texts([chunk.text_content for chunk in chunks])
    index.upsert(
        vectors=[
            (
                str(chunk.id),
                vector,
                {
                    "case_id": str(case_id),
                    "document_id": str(document_id),
                    "page_start": chunk.page_start,
                    "page_end": chunk.page_end,
                    "chunk_index": chunk.chunk_index,
                    "quality_flags": [],
                    "embedding_model": settings.embedding_model,
                    "index_schema_version": settings.index_schema_version,
                },
            )
            for chunk, vector in zip(chunks, vectors)
        ],
        namespace=namespace,
    )

    fixture = RetrievalFixture(
        settings=settings,
        case=case,
        document=document,
        chunks=chunks,
        pages=page_rows,
        index=index,
        client=FakeLexicalClient(chunks),
        embedder=embedder,
    )

    monkeypatch.setattr(
        hybrid_module,
        "get_case_for_user",
        lambda client, user_id, target: fixture.case,
    )
    monkeypatch.setattr(
        hybrid_module,
        "list_documents",
        lambda client, user_id, target, **kwargs: [fixture.document],
    )
    monkeypatch.setattr(
        hybrid_module,
        "get_chunks_by_ids",
        lambda client, user_id, target, chunk_ids: [
            chunk
            for chunk in fixture.chunks
            if str(chunk.id) in {str(value) for value in chunk_ids}
        ],
    )
    monkeypatch.setattr(
        hybrid_module,
        "get_pages_for_documents",
        lambda client, user_id, target, document_ids: list(fixture.pages),
    )
    monkeypatch.setattr(
        hybrid_module,
        "get_neighbor_chunks",
        lambda client, user_id, chunk_id, before=1, after=1: [
            chunk
            for chunk in fixture.chunks
            if str(chunk.document_id) == str(fixture.document.id)
            and abs(chunk.chunk_index - _chunk_index(fixture, chunk_id)) <= 1
        ],
    )
    yield fixture


def _chunk_index(fixture: RetrievalFixture, chunk_id: Any) -> int:
    """Return the chunk index of one stored chunk."""
    for chunk in fixture.chunks:
        if str(chunk.id) == str(chunk_id):
            return chunk.chunk_index
    return -1


def _search(fixture: RetrievalFixture, query: str) -> list:
    """Run a hybrid search against the fixture."""
    return hybrid_search(
        fixture.client,
        fixture.case.id,
        CurrentUser(id=fixture.case.owner_user_id, email="lawyer@example.com"),
        query,
        top_k=3,
        settings=fixture.settings,
        embedder=fixture.embedder,
        index=fixture.index,
    )


def test_keyword_search_returns_the_page_cited_chunk(
    retrieval_fixture: RetrievalFixture,
) -> None:
    spans = _search(retrieval_fixture, "arrears")

    assert spans
    assert any(span.page_start == 2 for span in spans)
    matched = next(span for span in spans if span.page_start == 2)
    assert EXPECTED_TEXT[2] in " ".join(matched.quote_or_span.split())
    assert matched.title == "rent-dispute.pdf"
    assert matched.source_anchor is not None
    assert matched.retrieval_rank == 1


def test_paraphrased_question_returns_the_answering_chunk(
    retrieval_fixture: RetrievalFixture,
) -> None:
    spans = _search(
        retrieval_fixture, "How much did the court order the landlord to recover?"
    )

    assert spans
    assert any(span.page_start == 3 for span in spans)
    matched = next(span for span in spans if span.page_start == 3)
    assert EXPECTED_TEXT[3] in " ".join(matched.quote_or_span.split())
    assert str(matched.evidence_id) in {
        str(chunk.id) for chunk in retrieval_fixture.chunks
    }


def test_search_results_never_leave_the_authorized_case(
    retrieval_fixture: RetrievalFixture,
) -> None:
    spans = _search(retrieval_fixture, "rent")

    assert spans
    assert {span.case_id for span in spans} == {retrieval_fixture.case.id}
    assert {span.document_id for span in spans} == {
        retrieval_fixture.document.id
    }
    assert all(span.quote_or_span for span in spans)


def test_lexical_search_is_called_with_the_authorized_documents(
    retrieval_fixture: RetrievalFixture,
) -> None:
    _search(retrieval_fixture, "rent")

    name, params = retrieval_fixture.client.rpc_calls[0]
    assert name == "search_chunks_lexical"
    assert params["p_case_id"] == str(retrieval_fixture.case.id)
    assert params["p_document_ids"] == [str(retrieval_fixture.document.id)]


def test_real_tokenizer_chunks_stay_within_the_configured_bounds(
    retrieval_fixture: RetrievalFixture,
) -> None:
    settings = retrieval_fixture.settings
    try:
        counter = load_token_counter(settings)
    except ChunkingError:
        pytest.skip("the tokenizer is not available locally")

    for chunk in retrieval_fixture.chunks:
        assert counter.count(chunk.text_content) <= settings.chunk_max_tokens
        assert chunk.source_anchor is not None
