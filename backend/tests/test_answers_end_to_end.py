"""End-to-end test of the baseline answer flow over a generated legal PDF.

The test ingests a small legal PDF, embeds it with the configured model, asks
a real question, and checks that the answer is cited to the right page. It
then asks about something that is not in the file and checks that the answer
is a clean abstention.
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

from app.agents.llm import LLMClient
from app.agents import graph as graph_module
from app.agents import research_agent as research_module
from app.auth.dependencies import CurrentUser
from app.db.repos import Case, Document, DocumentPage
from app.embeddings.model import EmbeddingError, get_embedder
from app.indexing.pinecone_store import namespace_for_case
from app.retrieval import baseline_rag as baseline_module
from app.agents.graph import run_agentic_rag
from app.retrieval.answer_schema import AnswerClaim, AnswerResponse
from app.retrieval.baseline_rag import answer_question

TIMESTAMP = "2026-01-01T00:00:00+00:00"

_EVIDENCE_PATTERN = re.compile(
    r"<<<EVIDENCE id=(?P<id>[0-9a-fA-F-]{36})[^\n]*>>>\n"
    r"(?P<text>.*?)\n<<<END EVIDENCE",
    re.DOTALL,
)
_QUESTION_PATTERN = re.compile(r"QUESTION: (?P<question>.+)", re.DOTALL)
_STOP_WORDS = {
    "the",
    "and",
    "was",
    "were",
    "did",
    "does",
    "that",
    "this",
    "with",
    "from",
    "what",
    "how",
    "much",
    "many",
    "there",
    "their",
    "they",
}


def _fact_values(fact_model: Any, text: str) -> dict[str, Any]:
    """Fill the task specific fields of a fact from the quoted text."""
    fields = getattr(fact_model, "model_fields", {})
    values: dict[str, Any] = {}
    amount = re.search(r"[\d,]+(?:\.\d+)?\s*(?:pounds|GBP|£)", text)
    date = re.search(r"\b\d{1,2}\s+[A-Z][a-z]+\s+\d{4}\b", text)
    if "amount" in fields:
        values["amount"] = amount.group(0) if amount else None
    if "date" in fields:
        values["date"] = date.group(0) if date else None
    if "entity" in fields:
        values["entity"] = (
            "the tenant" if "tenant" in text.lower() else "the party"
        )
    if "obligation" in fields:
        values["obligation"] = None
    if "event" in fields:
        values["event"] = text.strip()[:80]
    if "parties" in fields:
        values["parties"] = ["the tenant"] if "tenant" in text.lower() else []
    if "clause" in fields:
        clause = re.search(r"clause\s+\d+(?:\.\d+)*", text, re.IGNORECASE)
        values["clause"] = clause.group(0) if clause else ""
    if "value" in fields:
        values["value"] = text.strip()[:60]
    if "statement" in fields:
        values["statement"] = text.strip()[:120]
    return values


class GroundedChat:
    """Fake model that only cites evidence whose text matches the question."""

    def __init__(self) -> None:
        """Create the fake model."""
        self.calls: list[Any] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Return a runnable that answers from the prompt it receives."""
        outer = self

        class _Runnable:
            def invoke(self, messages: Any) -> Any:
                outer.calls.append(messages)
                return outer.answer(messages, schema)

        return _Runnable()

    def answer(self, messages: Any, schema: Any) -> Any:
        """Build a grounded answer from the evidence in the prompt."""
        user_prompt = messages[-1][1]
        fields = getattr(schema, "model_fields", {})
        if "subquestions" in fields:
            question = _QUESTION_PATTERN.search(user_prompt)
            return schema(
                task_type="fact_lookup",
                subquestions=[
                    {
                        "text": (
                            question.group("question").strip()
                            if question
                            else "the question"
                        ),
                        "purpose": "Answer the question.",
                    }
                ],
                rationale="One retrieval is enough.",
            )
        if "rewritten_query" in fields:
            from app.agents.research_agent import QueryRewrite

            return QueryRewrite(
                rewritten_query="", strategy="", terms_added=[]
            )
        if "verdicts" in fields:
            claim_ids = re.findall(r"^- claim (\S+)", user_prompt, re.MULTILINE)
            return schema(
                verdicts=[
                    {"claim_id": claim_id, "label": "supported", "note": ""}
                    for claim_id in claim_ids
                ]
            )
        if "answer_text" in fields:
            return self._answer_draft(user_prompt, schema)
        if "facts" in fields:
            return schema(facts=self._facts(user_prompt, schema), unknowns=[])
        if "conflicts" in fields:
            return schema(conflicts=[], gaps=[])
        question_match = _QUESTION_PATTERN.search(user_prompt)
        question = question_match.group("question") if question_match else ""
        terms = {
            term
            for term in re.findall(r"[a-z0-9]+", question.lower())
            if len(term) > 3 and term not in _STOP_WORDS
        }
        matched: list[str] = []
        quotes: list[str] = []
        for block in _EVIDENCE_PATTERN.finditer(user_prompt):
            text = block.group("text")
            lowered = text.lower()
            if any(term in lowered for term in terms):
                matched.append(block.group("id"))
                quotes.append(text)
        if not matched:
            return schema(
                answer_text=(
                    "The retrieved evidence does not answer this question."
                ),
                claims=[],
                final_status="insufficient_evidence",
                reason_codes=["NO_SUPPORTED_CLAIMS"],
            )
        return schema(
            answer_text=" ".join(quotes),
            claims=[
                AnswerClaim(
                    text=quote,
                    evidence_ids=[evidence_id],
                    type="direct",
                )
                for quote, evidence_id in zip(quotes, matched)
            ],
            final_status="answered",
            reason_codes=[],
        )

    def _answer_draft(self, user_prompt: str, schema: Any) -> Any:
        """Build the baseline answer draft from the evidence packet."""
        question_match = _QUESTION_PATTERN.search(user_prompt)
        question = question_match.group("question") if question_match else ""
        terms = {
            term
            for term in re.findall(r"[a-z0-9]+", question.lower())
            if len(term) > 3 and term not in _STOP_WORDS
        }
        matched: list[str] = []
        quotes: list[str] = []
        for block in _EVIDENCE_PATTERN.finditer(user_prompt):
            text = block.group("text")
            if any(term in text.lower() for term in terms):
                matched.append(block.group("id"))
                quotes.append(text)
        if not matched:
            return schema(
                answer_text=(
                    "The retrieved evidence does not answer this question."
                ),
                claims=[],
                final_status="insufficient_evidence",
                reason_codes=["NO_SUPPORTED_CLAIMS"],
            )
        return schema(
            answer_text=" ".join(quotes),
            claims=[
                AnswerClaim(
                    text=quote,
                    evidence_ids=[evidence_id],
                    type="direct",
                )
                for quote, evidence_id in zip(quotes, matched)
            ],
            final_status="answered",
            reason_codes=[],
        )

    def _facts(self, user_prompt: str, schema: Any) -> list[Any]:
        """Build one fact per evidence block that carries the question."""
        question_match = _QUESTION_PATTERN.search(user_prompt)
        question = question_match.group("question") if question_match else ""
        terms = {
            term
            for term in re.findall(r"[a-z0-9]+", question.lower())
            if len(term) > 3 and term not in _STOP_WORDS
        }
        fact_model = schema.model_fields["facts"].annotation
        facts: list[Any] = []
        for block in _EVIDENCE_PATTERN.finditer(user_prompt):
            text = block.group("text")
            if terms and not any(term in text.lower() for term in terms):
                continue
            values = _fact_values(fact_model, text)
            facts.append(
                fact_model(
                    quote=text,
                    evidence_ids=[block.group("id")],
                    uncertainty="certain",
                    **values,
                )
            )
        return facts


@dataclass
class CaseFixture:
    """The in-memory case, document, and index used by the test."""

    settings: Any
    case: Case
    document: Document
    chunks: list[Any]
    pages: list[DocumentPage]
    index: FakePineconeIndex
    embedder: Any = None
    user: CurrentUser = field(default_factory=lambda: CurrentUser(id=uuid4()))


@pytest.fixture()
def case_fixture(monkeypatch: pytest.MonkeyPatch) -> CaseFixture:
    """Ingest a PDF, embed it, and wire the answer flow to memory."""
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

    pages = extracted_pages(pdf_bytes())
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

    fixture = CaseFixture(
        settings=settings,
        case=case,
        document=document,
        chunks=chunks,
        pages=page_rows,
        index=index,
        embedder=embedder,
    )
    fixture.user = CurrentUser(id=owner_id, email="lawyer@example.com")

    monkeypatch.setattr(
        baseline_module,
        "hybrid_search",
        lambda client, case_id, user, query, **kwargs: _search(
            fixture, query, kwargs.get("top_k", 8)
        ),
    )
    monkeypatch.setattr(
        research_module,
        "hybrid_search",
        lambda client, case_id, user, query, **kwargs: _search(
            fixture, query, kwargs.get("top_k", 8)
        ),
    )
    monkeypatch.setattr(
        graph_module,
        "get_case_for_user",
        lambda client, user_id, target: fixture.case,
    )
    monkeypatch.setattr(
        graph_module,
        "list_documents",
        lambda client, user_id, target, **kwargs: [fixture.document],
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


def _search(fixture: CaseFixture, query: str, top_k: int) -> list:
    """Search the fixture with the same dense and lexical path as production."""
    from app.retrieval.dense import dense_search
    from app.retrieval.fusion import reciprocal_rank_fusion
    from app.retrieval.lexical import LexicalHit
    from app.retrieval.schemas import EvidenceSpan

    document_ids = [str(fixture.document.id)]
    dense_hits = dense_search(
        fixture.case.id,
        query,
        document_ids=document_ids,
        top_k=top_k,
        settings=fixture.settings,
        embedder=fixture.embedder,
        index=fixture.index,
    )
    terms = [term for term in re.findall(r"[a-z0-9]+", query.lower()) if term]
    lexical_hits = [
        LexicalHit(
            chunk_id=str(chunk.id),
            score=float(
                sum(1 for term in terms if term in chunk.text_content.lower())
            ),
            document_id=str(chunk.document_id),
        )
        for chunk in fixture.chunks
        if any(term in chunk.text_content.lower() for term in terms)
    ]
    fused = reciprocal_rank_fusion(
        {
            "dense": [hit.chunk_id for hit in dense_hits],
            "lexical": [hit.chunk_id for hit in lexical_hits],
        }
    )
    by_id = {str(chunk.id): chunk for chunk in fixture.chunks}
    spans: list[EvidenceSpan] = []
    for rank, item in enumerate(fused, start=1):
        chunk = by_id.get(item.chunk_id)
        if chunk is None:
            continue
        spans.append(
            EvidenceSpan(
                evidence_id=chunk.id,
                case_id=chunk.case_id,
                document_id=chunk.document_id,
                document_version_id=chunk.document_version_id,
                title=fixture.document.original_filename,
                page_start=chunk.page_start,
                page_end=chunk.page_end,
                section_path=list(chunk.section_path),
                quote_or_span=chunk.text_content,
                extraction_method="native",
                quality_status=fixture.document.quality_status,
                ocr_confidence=None,
                source_anchor=chunk.source_anchor,
                retrieval_method="hybrid",
                retrieval_rank=rank,
                quality_flags=[],
                retrieval_score=item.score,
            )
        )
    return spans[:top_k]


def _ask(fixture: CaseFixture, question: str):
    """Ask a question with the grounded fake model."""
    chat = GroundedChat()
    llm = LLMClient(settings=fixture.settings, chat=chat)
    result = answer_question(
        object(),
        fixture.case.id,
        fixture.user,
        question,
        settings=fixture.settings,
        llm=llm,
        embedder=fixture.embedder,
        index=fixture.index,
    )
    return result, chat


def test_a_question_about_the_file_gets_a_cited_answer(
    case_fixture: CaseFixture,
) -> None:
    result, _chat = _ask(
        case_fixture, "How much did the court order the landlord to recover?"
    )

    assert result.answer.final_status == "answered"
    assert result.answer.claims
    cited = {str(span.evidence_id) for span in result.answer.evidence}
    assert cited
    for claim in result.answer.claims:
        assert claim.evidence_ids
        assert all(str(evidence_id) in cited for evidence_id in claim.evidence_ids)
    page = result.answer.evidence[0].page_start
    assert EXPECTED_TEXT[page] in " ".join(result.answer.evidence[0].quote_or_span.split())


def test_a_question_about_something_absent_abstains_cleanly(
    case_fixture: CaseFixture,
) -> None:
    result, _chat = _ask(
        case_fixture,
        "What settlement amount was agreed at the arbitration hearing?",
    )

    assert result.answer.final_status == "insufficient_evidence"
    assert result.answer.claims == []
    text = result.answer.answer_text.lower()
    assert "does not exist" not in text
    assert "may still exist" in result.answer.answer_text
    assert "settlement amount" in result.answer.answer_text


def test_the_answer_never_cites_an_evidence_id_outside_the_case(
    case_fixture: CaseFixture,
) -> None:
    result, _chat = _ask(
        case_fixture, "What did the court order the claimant to recover?"
    )

    retrieved = {str(chunk.id) for chunk in case_fixture.chunks}
    for claim in result.answer.claims:
        assert all(str(evidence_id) in retrieved for evidence_id in claim.evidence_ids)


def test_the_agentic_mode_answers_the_same_question(
    case_fixture: CaseFixture,
) -> None:
    question = "How much did the court order the landlord to recover?"

    baseline, _chat = _ask(case_fixture, question)

    chat = GroundedChat()
    llm = LLMClient(settings=case_fixture.settings, chat=chat)
    agentic = run_agentic_rag(
        object(),
        case_fixture.case.id,
        case_fixture.user,
        question,
        settings=case_fixture.settings,
        llm=llm,
        embedder=case_fixture.embedder,
        index=case_fixture.index,
    )

    assert baseline.answer.final_status == "answered"
    assert agentic.answer.final_status == "answered"
    baseline_pages = {span.page_start for span in baseline.answer.evidence}
    agentic_pages = {span.page_start for span in agentic.answer.evidence}
    # The agentic answer cites every retrieved passage, so it covers at least
    # the pages the baseline cited.
    assert baseline_pages <= agentic_pages
    assert 3 in agentic_pages
    assert all(
        span.page_start is not None for span in agentic.answer.evidence
    )


def test_the_agentic_trace_shows_the_node_path(
    case_fixture: CaseFixture,
) -> None:
    chat = GroundedChat()
    llm = LLMClient(settings=case_fixture.settings, chat=chat)

    agentic = run_agentic_rag(
        object(),
        case_fixture.case.id,
        case_fixture.user,
        "How much did the court order the landlord to recover?",
        settings=case_fixture.settings,
        llm=llm,
        embedder=case_fixture.embedder,
        index=case_fixture.index,
    )

    summary = agentic.trace_summary()
    assert summary.graph_version == "agentic-rag-3"
    assert summary.node_path[0] == "authorize"
    assert "route" in summary.node_path
    assert "retrieve" in summary.node_path
    assert "synthesize" in summary.node_path
    assert summary.node_path[-1] == "final_gate"
    assert summary.steps == len(agentic.state.trace)
    assert summary.llm_calls >= 1
