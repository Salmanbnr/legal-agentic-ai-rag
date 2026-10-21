"""Tests for hybrid retrieval: authorization, fusion, dedupe, and diversity."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any
from uuid import UUID, uuid4

import pytest
from fakes import FakeEmbedder

from app.auth.dependencies import CurrentUser
from app.core.config import Settings
from app.db.repos import (
    Case,
    CaseAccessDenied,
    Document,
    DocumentChunk,
    DocumentPage,
)
from app.retrieval import hybrid as hybrid_module
from app.retrieval.dense import DenseHit
from app.retrieval.hybrid import hybrid_search
from app.retrieval.lexical import LexicalHit

TIMESTAMP = "2026-01-01T00:00:00+00:00"


def _case(case_id: UUID, owner_id: UUID) -> Case:
    return Case.model_validate(
        {
            "id": str(case_id),
            "owner_user_id": str(owner_id),
            "title": "Test case",
            "description": None,
            "status": "active",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    )


def _document(
    case_id: UUID,
    owner_id: UUID,
    *,
    document_id: UUID | None = None,
    quality_status: str = "ready",
    filename: str = "pleading.pdf",
) -> Document:
    resolved_id = document_id or uuid4()
    return Document.model_validate(
        {
            "id": str(resolved_id),
            "case_id": str(case_id),
            "owner_user_id": str(owner_id),
            "storage_path": f"{owner_id}/{case_id}/{resolved_id}",
            "original_filename": filename,
            "detected_mime": "application/pdf",
            "sha256": "a" * 64,
            "size_bytes": 1024,
            "page_count": 2,
            "document_type": None,
            "version_group_id": None,
            "duplicate_of": None,
            "quality_status": quality_status,
            "quality_report": {},
            "processing_status": "ready",
            "parser_version": "1.0.0",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    )


def _chunk(
    case_id: UUID,
    document_id: UUID,
    index: int,
    *,
    text: str = "The claimant seeks damages.",
    content_hash: str | None = None,
    page_start: int = 1,
    page_end: int = 1,
    quality_flags: list[Any] | None = None,
) -> DocumentChunk:
    resolved_hash = content_hash or f"{index:064d}"
    return DocumentChunk.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "document_id": str(document_id),
            "document_version_id": None,
            "chunk_index": index,
            "page_start": page_start,
            "page_end": page_end,
            "section_path": ["Article 1", "Clause 2"],
            "text_content": text,
            "content_hash": resolved_hash,
            "source_anchor": f"doc:{document_id}|p:{page_start}-{page_end}|c:{index}",
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "index_schema_version": 1,
            "index_status": "indexed",
            "quality_flags": list(quality_flags or []),
            "created_at": TIMESTAMP,
        }
    )


def _page(
    case_id: UUID,
    document_id: UUID,
    page_number: int,
    *,
    extraction_method: str = "native",
    ocr_confidence: float | None = None,
    quality_flags: list[Any] | None = None,
) -> DocumentPage:
    return DocumentPage.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "document_id": str(document_id),
            "page_number": page_number,
            "extraction_method": extraction_method,
            "text_density": 0.01,
            "ocr_confidence": ocr_confidence,
            "raw_text": "text",
            "clean_text": "text",
            "quality_flags": list(quality_flags or []),
            "transform_log": [],
            "created_at": TIMESTAMP,
        }
    )


@dataclass
class HybridHarness:
    """In-memory stand-ins for the retrieval data access layer."""

    settings: Settings
    case: Case
    documents: dict[str, Document] = field(default_factory=dict)
    chunks: dict[str, list[DocumentChunk]] = field(default_factory=dict)
    pages: list[DocumentPage] = field(default_factory=list)
    dense_calls: list[dict[str, Any]] = field(default_factory=list)
    lexical_calls: list[dict[str, Any]] = field(default_factory=list)
    dense_hits: list[DenseHit] = field(default_factory=list)
    lexical_hits: list[LexicalHit] = field(default_factory=list)

    def add_document(self, document: Document) -> Document:
        """Register a document owned by the case."""
        self.documents[str(document.id)] = document
        self.chunks.setdefault(str(document.id), [])
        return document

    def add_chunks(
        self, document: Document, chunks: list[DocumentChunk]
    ) -> list[DocumentChunk]:
        """Register chunks for a document."""
        self.chunks[str(document.id)] = list(chunks)
        return chunks

    def all_chunks(self) -> list[DocumentChunk]:
        """Return every registered chunk."""
        return [chunk for chunks in self.chunks.values() for chunk in chunks]


@pytest.fixture()
def harness(monkeypatch: pytest.MonkeyPatch) -> HybridHarness:
    """Provide a hybrid search harness wired to in-memory storage."""
    owner_id = uuid4()
    case = _case(uuid4(), owner_id)
    state = HybridHarness(settings=Settings(_env_file=None), case=case)

    def fake_get_case(client, user_id, case_id):
        if str(case_id) != str(state.case.id) or str(user_id) != str(
            state.case.owner_user_id
        ):
            raise CaseAccessDenied("Case not found")
        return state.case

    def fake_list_documents(
        client, user_id, case_id, *, processing_status=None, limit=100, offset=0
    ):
        if str(case_id) != str(state.case.id):
            raise CaseAccessDenied("Case not found")
        return list(state.documents.values())

    def fake_get_chunks_by_ids(client, user_id, case_id, chunk_ids):
        wanted = {str(value) for value in chunk_ids}
        return [
            chunk for chunk in state.all_chunks() if str(chunk.id) in wanted
        ]

    def fake_get_pages(client, user_id, case_id, document_ids):
        wanted = {str(value) for value in document_ids}
        return [
            page for page in state.pages if str(page.document_id) in wanted
        ]

    def fake_get_neighbors(
        client, user_id, chunk_id, *, before=1, after=1
    ):
        target = next(
            (chunk for chunk in state.all_chunks() if str(chunk.id) == str(chunk_id)),
            None,
        )
        if target is None:
            return []
        same_document = [
            chunk
            for chunk in state.all_chunks()
            if str(chunk.document_id) == str(target.document_id)
        ]
        return [
            chunk
            for chunk in same_document
            if target.chunk_index - before
            <= chunk.chunk_index
            <= target.chunk_index + after
        ]

    def fake_dense_search(case_id, query, **kwargs):
        state.dense_calls.append({"case_id": case_id, "query": query, **kwargs})
        return list(state.dense_hits)

    def fake_lexical_search(client, case_id, query, **kwargs):
        state.lexical_calls.append(
            {"case_id": case_id, "query": query, **kwargs}
        )
        return list(state.lexical_hits)

    monkeypatch.setattr(hybrid_module, "get_case_for_user", fake_get_case)
    monkeypatch.setattr(hybrid_module, "list_documents", fake_list_documents)
    monkeypatch.setattr(
        hybrid_module, "get_chunks_by_ids", fake_get_chunks_by_ids
    )
    monkeypatch.setattr(
        hybrid_module, "get_pages_for_documents", fake_get_pages
    )
    monkeypatch.setattr(
        hybrid_module, "get_neighbor_chunks", fake_get_neighbors
    )
    monkeypatch.setattr(hybrid_module, "dense_search", fake_dense_search)
    monkeypatch.setattr(hybrid_module, "lexical_search", fake_lexical_search)
    return state


def _user(harness: HybridHarness) -> CurrentUser:
    """Return the authenticated user that owns the harness case."""
    return CurrentUser(id=harness.case.owner_user_id, email="lawyer@example.com")


def _search(harness: HybridHarness, **kwargs) -> list:
    """Run a hybrid search against the harness."""
    options = {
        "settings": harness.settings,
        "embedder": FakeEmbedder(),
        "index": object(),
    }
    options.update(kwargs)
    return hybrid_search(
        object(),
        harness.case.id,
        _user(harness),
        options.pop("query", "who is liable"),
        **options,
    )


def test_foreign_document_ids_cannot_widen_the_search(
    harness: HybridHarness,
) -> None:
    case_id = harness.case.id
    owner_id = harness.case.owner_user_id
    own_document = harness.add_document(_document(case_id, owner_id))
    foreign_document = _document(uuid4(), owner_id)
    own_chunk = _chunk(case_id, own_document.id, 0)
    foreign_chunk = _chunk(foreign_document.case_id, foreign_document.id, 0)
    harness.add_chunks(own_document, [own_chunk])
    harness.dense_hits = [
        DenseHit(chunk_id=str(own_chunk.id), score=0.9),
        DenseHit(chunk_id=str(foreign_chunk.id), score=0.95),
    ]

    spans = _search(
        harness,
        document_ids=[own_document.id, foreign_document.id],
    )

    assert harness.dense_calls[0]["document_ids"] == [str(own_document.id)]
    assert harness.lexical_calls[0]["document_ids"] == [str(own_document.id)]
    assert [span.document_id for span in spans] == [own_document.id]


def test_a_user_cannot_search_a_case_they_do_not_own(
    harness: HybridHarness,
) -> None:
    other_case = _case(uuid4(), uuid4())

    with pytest.raises(CaseAccessDenied):
        hybrid_search(
            object(),
            other_case.id,
            _user(harness),
            "question",
            settings=harness.settings,
            embedder=FakeEmbedder(),
            index=object(),
        )


def test_document_ids_default_to_every_case_document(
    harness: HybridHarness,
) -> None:
    case_id = harness.case.id
    owner_id = harness.case.owner_user_id
    first = harness.add_document(_document(case_id, owner_id))
    second = harness.add_document(_document(case_id, owner_id))
    chunk = _chunk(case_id, first.id, 0)
    harness.add_chunks(first, [chunk])
    harness.dense_hits = [DenseHit(chunk_id=str(chunk.id), score=0.8)]

    spans = _search(harness)

    assert harness.dense_calls[0]["document_ids"] == sorted(
        [str(first.id), str(second.id)]
    )
    assert len(spans) == 1


def test_duplicate_chunks_with_the_same_hash_are_collapsed(
    harness: HybridHarness,
) -> None:
    case_id = harness.case.id
    owner_id = harness.case.owner_user_id
    document = harness.add_document(_document(case_id, owner_id))
    first = _chunk(case_id, document.id, 0, content_hash="a" * 64)
    second = _chunk(case_id, document.id, 1, content_hash="a" * 64)
    harness.add_chunks(document, [first, second])
    harness.dense_hits = [
        DenseHit(chunk_id=str(first.id), score=0.9),
        DenseHit(chunk_id=str(second.id), score=0.8),
    ]

    spans = _search(harness)

    assert len(spans) == 1
    assert spans[0].evidence_id == first.id


def test_conflicting_document_versions_are_both_visible(
    harness: HybridHarness,
) -> None:
    case_id = harness.case.id
    owner_id = harness.case.owner_user_id
    first_document = harness.add_document(
        _document(case_id, owner_id, filename="pleading-v1.pdf")
    )
    second_document = harness.add_document(
        _document(case_id, owner_id, filename="pleading-v2.pdf")
    )
    first_chunk = _chunk(
        case_id, first_document.id, 0, text="Version one text.", content_hash="a" * 64
    )
    second_chunk = _chunk(
        case_id,
        second_document.id,
        0,
        text="Version two text.",
        content_hash="b" * 64,
    )
    harness.add_chunks(first_document, [first_chunk])
    harness.add_chunks(second_document, [second_chunk])
    harness.dense_hits = [
        DenseHit(chunk_id=str(first_chunk.id), score=0.9),
        DenseHit(chunk_id=str(second_chunk.id), score=0.88),
    ]

    spans = _search(harness)

    assert {span.document_id for span in spans} == {
        first_document.id,
        second_document.id,
    }
    assert {span.title for span in spans} == {
        "pleading-v1.pdf",
        "pleading-v2.pdf",
    }


def test_low_quality_spans_keep_their_flags(harness: HybridHarness) -> None:
    case_id = harness.case.id
    owner_id = harness.case.owner_user_id
    document = harness.add_document(
        _document(case_id, owner_id, quality_status="ready_with_warnings")
    )
    chunk = _chunk(
        case_id,
        document.id,
        0,
        quality_flags=["low_ocr_confidence"],
    )
    harness.add_chunks(document, [chunk])
    harness.pages.append(
        _page(
            case_id,
            document.id,
            1,
            extraction_method="ocr",
            ocr_confidence=61.5,
            quality_flags=["low_ocr_confidence"],
        )
    )
    harness.dense_hits = [DenseHit(chunk_id=str(chunk.id), score=0.7)]

    spans = _search(harness)

    assert spans[0].quality_flags == ["low_ocr_confidence"]
    assert spans[0].quality_status == "ready_with_warnings"
    assert spans[0].ocr_confidence == pytest.approx(61.5)
    assert spans[0].extraction_method == "ocr"


def test_one_document_cannot_take_every_slot(harness: HybridHarness) -> None:
    case_id = harness.case.id
    owner_id = harness.case.owner_user_id
    crowded = harness.add_document(_document(case_id, owner_id))
    other = harness.add_document(_document(case_id, owner_id))
    crowded_chunks = [
        _chunk(case_id, crowded.id, index) for index in range(5)
    ]
    other_chunks = [
        _chunk(case_id, other.id, 0, content_hash="f" * 64)
    ]
    harness.add_chunks(crowded, crowded_chunks)
    harness.add_chunks(other, other_chunks)
    harness.dense_hits = [
        DenseHit(chunk_id=str(chunk.id), score=0.9 - index * 0.01)
        for index, chunk in enumerate(crowded_chunks + other_chunks)
    ]

    spans = _search(harness, top_k=4)

    crowded_count = sum(
        1 for span in spans if span.document_id == crowded.id
    )
    assert crowded_count <= harness.settings.retrieval_max_spans_per_document
    assert len(spans) == 4
    assert any(span.document_id == other.id for span in spans)


def test_neighbor_expansion_attaches_the_previous_chunk(
    harness: HybridHarness,
) -> None:
    case_id = harness.case.id
    owner_id = harness.case.owner_user_id
    document = harness.add_document(_document(case_id, owner_id))
    first = _chunk(
        case_id,
        document.id,
        0,
        text="The parties agreed the schedule. The claim is for rent.",
        page_start=1,
        page_end=1,
    )
    second = _chunk(
        case_id,
        document.id,
        1,
        text="Subject to clause 4, the amount is payable monthly.",
        page_start=2,
        page_end=2,
    )
    harness.add_chunks(document, [first, second])
    harness.dense_hits = [DenseHit(chunk_id=str(second.id), score=0.9)]

    spans = _search(harness)

    assert "The parties agreed the schedule." in spans[0].quote_or_span
    assert "Subject to clause 4" in spans[0].quote_or_span
    assert spans[0].page_start == 1
    assert spans[0].page_end == 2


def test_neighbor_expansion_attaches_the_next_chunk(
    harness: HybridHarness,
) -> None:
    case_id = harness.case.id
    owner_id = harness.case.owner_user_id
    document = harness.add_document(_document(case_id, owner_id))
    first = _chunk(
        case_id,
        document.id,
        0,
        text="The tenant failed to pay the rent",
        page_start=1,
        page_end=1,
    )
    second = _chunk(
        case_id,
        document.id,
        1,
        text="within thirty days of the notice.",
        page_start=1,
        page_end=1,
    )
    harness.add_chunks(document, [first, second])
    harness.dense_hits = [DenseHit(chunk_id=str(first.id), score=0.9)]

    spans = _search(harness)

    assert "within thirty days of the notice." in spans[0].quote_or_span
    assert spans[0].page_start == 1
    assert spans[0].page_end == 1


def test_a_complete_chunk_is_not_expanded(harness: HybridHarness) -> None:
    case_id = harness.case.id
    owner_id = harness.case.owner_user_id
    document = harness.add_document(_document(case_id, owner_id))
    first = _chunk(case_id, document.id, 0, text="The claim is for rent.")
    second = _chunk(case_id, document.id, 1, text="Another clause entirely.")
    harness.add_chunks(document, [first, second])
    harness.dense_hits = [DenseHit(chunk_id=str(first.id), score=0.9)]

    spans = _search(harness)

    assert spans[0].quote_or_span == "The claim is for rent."


def test_evidence_spans_carry_page_and_anchor_information(
    harness: HybridHarness,
) -> None:
    case_id = harness.case.id
    owner_id = harness.case.owner_user_id
    document = harness.add_document(
        _document(case_id, owner_id, filename="contract.pdf")
    )
    chunk = _chunk(
        case_id, document.id, 3, page_start=2, page_end=3
    )
    harness.add_chunks(document, [chunk])
    harness.pages.append(
        _page(case_id, document.id, 2, extraction_method="native")
    )
    harness.pages.append(
        _page(case_id, document.id, 3, extraction_method="ocr", ocr_confidence=88.0)
    )
    harness.dense_hits = [DenseHit(chunk_id=str(chunk.id), score=0.9)]

    spans = _search(harness)

    span = spans[0]
    assert span.evidence_id == chunk.id
    assert span.case_id == case_id
    assert span.document_id == document.id
    assert span.title == "contract.pdf"
    assert span.page_start == 2
    assert span.page_end == 3
    assert span.section_path == ["Article 1", "Clause 2"]
    assert span.source_anchor == f"doc:{document.id}|p:2-3|c:3"
    assert span.extraction_method == "mixed"
    assert span.ocr_confidence == pytest.approx(88.0)
    assert span.retrieval_method == "dense"
    assert span.retrieval_rank == 1
    assert span.retrieval_score is not None


def test_a_chunk_found_by_both_methods_is_labelled_hybrid(
    harness: HybridHarness,
) -> None:
    case_id = harness.case.id
    owner_id = harness.case.owner_user_id
    document = harness.add_document(_document(case_id, owner_id))
    chunk = _chunk(case_id, document.id, 0)
    harness.add_chunks(document, [chunk])
    harness.dense_hits = [DenseHit(chunk_id=str(chunk.id), score=0.9)]
    harness.lexical_hits = [LexicalHit(chunk_id=str(chunk.id), score=0.5)]

    spans = _search(harness)

    assert spans[0].retrieval_method == "hybrid"


def test_no_candidates_returns_no_spans(harness: HybridHarness) -> None:
    case_id = harness.case.id
    owner_id = harness.case.owner_user_id
    document = harness.add_document(_document(case_id, owner_id))
    harness.add_chunks(document, [])

    assert _search(harness) == []


def test_a_search_without_authorized_documents_returns_nothing(
    harness: HybridHarness,
) -> None:
    spans = _search(harness, document_ids=[str(uuid4())])

    assert spans == []
    assert harness.dense_calls == []
    assert harness.lexical_calls == []
