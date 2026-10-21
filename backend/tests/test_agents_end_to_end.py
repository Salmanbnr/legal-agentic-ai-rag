"""End-to-end test of the two specialist agents on a multi-document case.

The fixture builds two related documents, indexes them, and then asks a
chronology question. The research agent has to find both documents, and the
facts analyst has to return structured facts whose quotes really appear in the
evidence and whose pages come from the stored chunks.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any
from uuid import uuid4

import pytest
from fakes import FakePineconeIndex
from legal_fixtures import (
    chunk_rows,
    extracted_pages,
    fixture_settings,
    pdf_bytes,
)

from app.agents import graph as graph_module
from app.agents import research_agent as research_module
from app.agents.document_analyst import (
    ChronologyFact,
    analyze_evidence,
)
from app.agents.graph import run_agentic_rag
from app.agents.llm import LLMClient
from app.agents.research_agent import QueryRewrite
from app.auth.dependencies import CurrentUser
from app.db.repos import Case, Document, DocumentPage
from app.embeddings.model import EmbeddingError, get_embedder
from app.indexing.pinecone_store import namespace_for_case
from app.retrieval import baseline_rag as baseline_module
from app.retrieval.schemas import EvidenceSpan

TIMESTAMP = "2026-01-01T00:00:00+00:00"

SECOND_DOCUMENT_TEXT = """AMENDMENT

The parties signed an amendment to the tenancy agreement on 2 June 2021. The
amendment increased the monthly rent from 1,250 pounds to 1,400 pounds and
added a clause requiring the rent to be paid within seven days of the due
date. The landlord accepted the amended rent in a letter dated 9 June 2021."""


class GroundedChat:
    """Fake model that plans, rewrites, and extracts grounded facts."""

    def __init__(self) -> None:
        """Create the fake model."""
        self.calls: list[Any] = []
        self.facts: list[Any] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Return a runnable that answers for the requested schema."""
        outer = self

        class _Runnable:
            def invoke(self, messages: Any) -> Any:
                outer.calls.append(messages)
                return outer.answer(messages, schema)

        return _Runnable()

    def answer(self, messages: Any, schema: Any) -> Any:
        """Build the response for the requested schema."""
        fields = getattr(schema, "model_fields", {})
        if "subquestions" in fields:
            question = _question(messages)
            return schema(
                task_type="chronology",
                subquestions=[
                    {"text": question, "purpose": "Build the timeline."}
                ],
                rationale="One timeline retrieval.",
            )
        if "rewritten_query" in fields:
            return QueryRewrite(
                rewritten_query="amendment rent increase",
                strategy="synonym",
                terms_added=["amendment"],
            )
        if "answer_text" in fields:
            return schema(
                answer_text="The answer is in the evidence.",
                claims=[],
                final_status="partial",
                reason_codes=[],
            )
        return schema(facts=list(self.facts), unknowns=[])


def _question(messages: Any) -> str:
    """Return the question from a prompt."""
    match = re.search(r"QUESTION: (.+)", messages[-1][1], re.DOTALL)
    return match.group(1).strip() if match else "the question"


def _document(case_id, owner_id, filename: str) -> Document:
    return Document.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(case_id),
            "owner_user_id": str(owner_id),
            "storage_path": f"{owner_id}/{case_id}/{uuid4()}",
            "original_filename": filename,
            "detected_mime": "application/pdf",
            "sha256": "a" * 64,
            "size_bytes": 1024,
            "page_count": 1,
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


@dataclass
class ChronologyFixture:
    """The in-memory case used by the chronology test."""

    settings: Any
    case: Case
    documents: list[Document]
    chunks: list[Any]
    pages: list[DocumentPage]
    index: FakePineconeIndex
    embedder: Any = None
    user: CurrentUser = field(default_factory=lambda: CurrentUser(id=uuid4()))


@pytest.fixture()
def chronology_case(monkeypatch: pytest.MonkeyPatch) -> ChronologyFixture:
    """Build a two document case and index both documents."""
    settings = fixture_settings()
    try:
        embedder = get_embedder(settings)
    except EmbeddingError:
        pytest.skip("the embedding model is not available locally")

    case_id = uuid4()
    owner_id = uuid4()
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
    lease = _document(case_id, owner_id, "lease.pdf")
    amendment = _document(case_id, owner_id, "amendment.pdf")

    lease_pages = extracted_pages(pdf_bytes())
    lease_chunks = chunk_rows(lease_pages, lease.id, case_id)

    amendment_pages = extracted_pages(_second_pdf())
    amendment_chunks = chunk_rows(amendment_pages, amendment.id, case_id)

    chunks = [*lease_chunks, *amendment_chunks]
    if len(chunks) < 2:
        pytest.fail("the fixture produced too few chunks")

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
        for document_id, pages in (
            (lease.id, lease_pages),
            (amendment.id, amendment_pages),
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
                    "document_id": str(chunk.document_id),
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

    fixture = ChronologyFixture(
        settings=settings,
        case=case,
        documents=[lease, amendment],
        chunks=chunks,
        pages=page_rows,
        index=index,
        embedder=embedder,
    )
    fixture.user = CurrentUser(id=owner_id, email="lawyer@example.com")

    def fake_search(
        client, target_case_id, user, query, *, document_ids=None, **kwargs
    ):
        from app.retrieval.dense import dense_search
        from app.retrieval.fusion import reciprocal_rank_fusion

        allowed = [str(value) for value in (document_ids or [])]
        hits = dense_search(
            target_case_id,
            query,
            document_ids=allowed,
            top_k=8,
            settings=settings,
            embedder=embedder,
            index=index,
        )
        fused = reciprocal_rank_fusion({"dense": [hit.chunk_id for hit in hits]})
        by_id = {str(chunk.id): chunk for chunk in chunks}
        spans: list[EvidenceSpan] = []
        for rank, item in enumerate(fused, start=1):
            chunk = by_id.get(item.chunk_id)
            if chunk is None:
                continue
            if allowed and str(chunk.document_id) not in allowed:
                continue
            document = next(
                (
                    doc
                    for doc in fixture.documents
                    if str(doc.id) == str(chunk.document_id)
                ),
                None,
            )
            spans.append(
                EvidenceSpan(
                    evidence_id=chunk.id,
                    case_id=chunk.case_id,
                    document_id=chunk.document_id,
                    document_version_id=chunk.document_version_id,
                    title=document.original_filename if document else "",
                    page_start=chunk.page_start,
                    page_end=chunk.page_end,
                    section_path=list(chunk.section_path),
                    quote_or_span=chunk.text_content,
                    extraction_method="native",
                    quality_status="ready",
                    ocr_confidence=None,
                    source_anchor=chunk.source_anchor,
                    retrieval_method="hybrid",
                    retrieval_rank=rank,
                    quality_flags=[],
                    retrieval_score=item.score,
                )
            )
        return spans

    monkeypatch.setattr(research_module, "hybrid_search", fake_search)
    monkeypatch.setattr(baseline_module, "hybrid_search", fake_search)
    monkeypatch.setattr(graph_module, "get_case_for_user", lambda *a, **k: case)
    monkeypatch.setattr(
        graph_module,
        "list_documents",
        lambda *a, **k: fixture.documents,
    )
    monkeypatch.setattr(
        graph_module,
        "get_chunks_by_ids",
        lambda client, user_id, case_id, chunk_ids: [
            chunk
            for chunk in fixture.chunks
            if str(chunk.id) in {str(value) for value in chunk_ids}
        ],
    )
    return fixture


def _second_pdf() -> bytes:
    """Build the one page amendment document."""
    import pymupdf

    document = pymupdf.open()
    try:
        page = document.new_page()
        top = 72
        for line in SECOND_DOCUMENT_TEXT.splitlines():
            if not line.strip():
                top += 10
                continue
            page.insert_text((72, top), line, fontsize=11)
            top += 16
        return document.tobytes()
    finally:
        document.close()


def test_a_chronology_question_returns_structured_facts(
    chronology_case: ChronologyFixture,
) -> None:
    amendment_chunk = next(
        chunk
        for chunk in chronology_case.chunks
        if "amendment" in chunk.text_content.lower()
    )
    chat = GroundedChat()
    chat.facts = [
        ChronologyFact(
            date="2 June 2021",
            event="The parties signed an amendment to the tenancy agreement.",
            parties=["the parties"],
            quote="The parties signed an amendment to the tenancy agreement",
            evidence_ids=[amendment_chunk.id],
            page_start=99,
            page_end=99,
            uncertainty="certain",
        )
    ]
    llm = LLMClient(settings=chronology_case.settings, chat=chat)

    result = run_agentic_rag(
        object(),
        chronology_case.case.id,
        chronology_case.user,
        "What happened in June 2021?",
        settings=chronology_case.settings,
        llm=llm,
        embedder=chronology_case.embedder,
        index=chronology_case.index,
    )

    assert result.state.task_type == "chronology"
    facts = result.state.extracted_facts
    assert facts
    fact = facts[0]
    assert fact.fields["date"] == "2 June 2021"
    assert fact.fields["event"].startswith("The parties signed an amendment")
    assert fact.evidence_ids == [amendment_chunk.id]
    # The page comes from the stored chunk, never from the model.
    assert fact.page_start == amendment_chunk.page_start
    assert fact.page_end == amendment_chunk.page_end
    assert fact.quality == "verified"


def test_the_analyst_rejects_a_quote_that_is_not_in_the_evidence(
    chronology_case: ChronologyFixture,
) -> None:
    chunk = chronology_case.chunks[0]
    span = EvidenceSpan(
        evidence_id=chunk.id,
        case_id=chunk.case_id,
        document_id=chunk.document_id,
        document_version_id=None,
        title="lease.pdf",
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_path=[],
        quote_or_span=chunk.text_content,
        extraction_method="native",
        quality_status="ready",
        ocr_confidence=None,
        source_anchor=chunk.source_anchor,
        retrieval_method="hybrid",
        retrieval_rank=1,
        quality_flags=[],
    )
    chat = GroundedChat()
    chat.facts = [
        ChronologyFact(
            date="1 January 2020",
            event="A fabricated event.",
            parties=[],
            quote="A fabricated event that never appears in the evidence.",
            evidence_ids=[chunk.id],
        )
    ]
    llm = LLMClient(settings=chronology_case.settings, chat=chat)

    result = analyze_evidence(
        [span],
        "chronology",
        question="What happened?",
        llm=llm,
        settings=chronology_case.settings,
    )

    assert result.facts == []
    assert result.dropped_fact_count == 1
    assert "QUOTE_NOT_FOUND" in result.reason_codes
