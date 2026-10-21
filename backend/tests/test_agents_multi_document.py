"""End-to-end test of the agentic mode over a multi-document case.

Two documents in the same case state the same event with different dates. The
workflow must cite both, report the conflict, and refuse to pick a winner.
A question about something that is not in the file must abstain cleanly.
"""

from __future__ import annotations

import re
from typing import Any
from uuid import uuid4

import pytest

from app.agents import graph as graph_module
from app.agents import research_agent as research_module
from app.agents.document_analyst import ChronologyFact
from app.agents.evidence_verifier import ClaimVerdict
from app.agents.graph import run_agentic_rag
from app.agents.llm import LLMClient
from app.agents.router import RoutingDraft
from app.auth.dependencies import CurrentUser
from app.core.config import Settings
from app.db.repos import Case, Document
from app.retrieval.schemas import EvidenceSpan

TIMESTAMP = "2026-01-01T00:00:00+00:00"

LEASE_TEXT = "The agreement was signed on 1 April 2021 by the parties."
COPY_TEXT = "The agreement was signed on 2 April 2021 by the parties."


class StubLLM:
    """Fake model that plans, extracts two facts, and supports them."""

    def __init__(self, spans: list[EvidenceSpan]) -> None:
        """Create the stub with the evidence it will quote."""
        self.spans = spans
        self.calls: list[Any] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> Any:
        """Return a runnable that answers for the requested schema."""
        outer = self

        class _Runnable:
            def invoke(self, messages: Any) -> Any:
                outer.calls.append(messages)
                return outer.answer(messages, schema)

        return _Runnable()

    def answer(self, messages: Any, schema: Any) -> Any:
        """Return the prepared result for the schema."""
        fields = getattr(schema, "model_fields", {})
        question = ""
        for message in messages:
            match = re.search(r"QUESTION: (.+)", message[1], re.DOTALL)
            if match:
                question = match.group(1).strip()
        if "subquestions" in fields:
            return schema(
                task_type="chronology",
                subquestions=[
                    {
                        "text": question or "when was it signed",
                        "purpose": "Build the timeline.",
                    }
                ],
                rationale="One timeline retrieval.",
            )
        if "rewritten_query" in fields:
            from app.agents.research_agent import QueryRewrite

            return QueryRewrite(
                rewritten_query="", strategy="", terms_added=[]
            )
        if "verdicts" in fields:
            return schema(
                verdicts=[
                    {"claim_id": f"claim-{index}", "label": "supported"}
                    for index in range(1, len(self.spans) + 1)
                ]
            )
        if "conflicts" in fields and "facts" not in fields:
            return schema(conflicts=[], gaps=[])
        if "facts" in fields:
            return schema(facts=self._facts(schema), unknowns=[])
        if "answer_text" in fields:
            return schema(
                answer_text="The evidence is in the packet.",
                claims=[],
                final_status="partial",
                reason_codes=[],
            )
        return schema()

    def _facts(self, schema: Any) -> list[Any]:
        """Build one dated fact per document."""
        dates = ["1 April 2021", "2 April 2021"]
        facts: list[Any] = []
        for index, span in enumerate(self.spans):
            facts.append(
                ChronologyFact(
                    date=dates[index % len(dates)],
                    event="The agreement was signed",
                    parties=["the parties"],
                    quote=span.quote_or_span,
                    evidence_ids=[span.evidence_id],
                    uncertainty="certain",
                )
            )
        return facts


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


@pytest.fixture()
def multi_document_case(monkeypatch: pytest.MonkeyPatch):
    """Wire two documents that disagree about a date into the graph."""
    owner_id = uuid4()
    case_id = uuid4()
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
    copy = _document(case_id, owner_id, "lease-copy.pdf")
    spans = [
        EvidenceSpan(
            evidence_id=uuid4(),
            case_id=case_id,
            document_id=lease.id,
            document_version_id=None,
            title="lease.pdf",
            page_start=1,
            page_end=1,
            section_path=[],
            quote_or_span=LEASE_TEXT,
            extraction_method="native",
            quality_status="ready",
            ocr_confidence=None,
            source_anchor="doc:1|p:1-1|c:0",
            retrieval_method="hybrid",
            retrieval_rank=1,
            quality_flags=[],
            retrieval_score=0.7,
        ),
        EvidenceSpan(
            evidence_id=uuid4(),
            case_id=case_id,
            document_id=copy.id,
            document_version_id=None,
            title="lease-copy.pdf",
            page_start=1,
            page_end=1,
            section_path=[],
            quote_or_span=COPY_TEXT,
            extraction_method="native",
            quality_status="ready",
            ocr_confidence=None,
            source_anchor="doc:2|p:1-1|c:0",
            retrieval_method="hybrid",
            retrieval_rank=2,
            quality_flags=[],
            retrieval_score=0.6,
        ),
    ]

    class _Chunk:
        def __init__(self, span: EvidenceSpan) -> None:
            self.id = span.evidence_id
            self.case_id = span.case_id
            self.document_id = span.document_id
            self.document_version_id = None
            self.chunk_index = span.retrieval_rank
            self.page_start = span.page_start
            self.page_end = span.page_end
            self.section_path = []
            self.text_content = span.quote_or_span
            self.content_hash = str(span.evidence_id)
            self.source_anchor = span.source_anchor
            self.embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
            self.index_schema_version = 1
            self.index_status = "indexed"
            self.quality_flags = []

    chunks = [_Chunk(span) for span in spans]

    def fake_search(
        client, target_case_id, user, query, *, document_ids=None, **kwargs
    ):
        # Nothing in the file mentions an arbitration, so the search for that
        # question finds nothing at all.
        if "arbitration" in query.lower():
            return []
        return list(spans)

    monkeypatch.setattr(research_module, "hybrid_search", fake_search)
    monkeypatch.setattr(graph_module, "get_case_for_user", lambda *a, **k: case)
    monkeypatch.setattr(
        graph_module, "list_documents", lambda *a, **k: [lease, copy]
    )
    monkeypatch.setattr(
        graph_module,
        "get_chunks_by_ids",
        lambda client, user_id, target, chunk_ids: [
            chunk
            for chunk in chunks
            if str(chunk.id) in {str(value) for value in chunk_ids}
        ],
    )
    return {
        "case": case,
        "documents": [lease, copy],
        "spans": spans,
        "chunks": chunks,
        "user": CurrentUser(id=owner_id, email="lawyer@example.com"),
    }


def _run(fixture, question: str):
    """Run the agentic workflow against the multi-document fixture."""
    settings = Settings(_env_file=None)
    llm = LLMClient(settings=settings, chat=StubLLM(fixture["spans"]))
    return run_agentic_rag(
        object(),
        fixture["case"].id,
        fixture["user"],
        question,
        settings=settings,
        llm=llm,
    )


def test_the_agentic_mode_cites_both_documents_and_shows_the_conflict(
    multi_document_case,
) -> None:
    result = _run(multi_document_case, "When was the agreement signed?")

    assert result.answer.evidence
    documents = {span.document_id for span in result.answer.evidence}
    assert len(documents) == 2
    for span in result.answer.evidence:
        assert span.page_start == 1
        assert span.title in ("lease.pdf", "lease-copy.pdf")

    assert result.state.contradictions
    conflict = result.state.contradictions[0]
    assert conflict.field == "date"
    assert conflict.resolution_status == "unresolved"
    assert "1 April 2021" in conflict.proposition_a
    assert "2 April 2021" in conflict.proposition_b

    assert result.answer.final_status == "conflicting_evidence"
    assert result.answer.conflicts
    assert "1 April 2021" in result.answer.conflicts[0]
    assert "2 April 2021" in result.answer.conflicts[0]


def test_the_agentic_mode_abstains_when_the_evidence_is_missing(
    multi_document_case,
) -> None:
    result = _run(
        multi_document_case, "What was agreed at the arbitration hearing?"
    )

    assert result.answer.final_status in (
        "insufficient_evidence",
        "partial",
    )
    assert "does not claim that it is absent" in result.answer.answer_text
    for claim in result.answer.claims:
        assert claim.evidence_ids
