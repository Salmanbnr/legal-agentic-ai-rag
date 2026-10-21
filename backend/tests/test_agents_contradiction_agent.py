"""Tests for the contradiction and gap analyst."""

from __future__ import annotations

from typing import Any
from uuid import uuid4

import pytest

from app.agents.contradiction_agent import (
    REASON_UNSUPPORTED_CONFLICT,
    ConflictCandidate,
    ConflictDraft,
    ContradictionInput,
    analyze_contradictions,
)
from app.agents.state import ExtractedFact
from app.core.config import Settings
from app.db.repos import Document
from app.retrieval.schemas import EvidenceSpan

TIMESTAMP = "2026-01-01T00:00:00+00:00"


def _document(
    case_id,
    owner_id,
    *,
    filename: str = "lease.pdf",
    version_group_id=None,
) -> Document:
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
            "version_group_id": str(version_group_id) if version_group_id else None,
            "duplicate_of": None,
            "quality_status": "ready",
            "quality_report": {},
            "processing_status": "ready",
            "parser_version": "1.0.0",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    )


def _span(document_id, text: str, page: int = 1) -> EvidenceSpan:
    return EvidenceSpan(
        evidence_id=uuid4(),
        case_id=uuid4(),
        document_id=document_id,
        document_version_id=None,
        title="lease.pdf",
        page_start=page,
        page_end=page,
        section_path=[],
        quote_or_span=text,
        extraction_method="native",
        quality_status="ready",
        ocr_confidence=None,
        source_anchor="doc:1|p:1-1|c:0",
        retrieval_method="hybrid",
        retrieval_rank=1,
        quality_flags=[],
    )


def _fact(
    document_id,
    span: EvidenceSpan,
    *,
    event: str = "The agreement was signed",
    date: str = "1 April 2021",
) -> ExtractedFact:
    return ExtractedFact(
        id=f"fact-{uuid4()}",
        text=f"{event} on {date}.",
        evidence_ids=[span.evidence_id],
        page_start=span.page_start,
        page_end=span.page_end,
        uncertainty="certain",
        quality="verified",
        quote=span.quote_or_span,
        fields={"event": event, "date": date},
    )


class StubLLM:
    """LLM stand-in that returns a prepared conflict draft."""

    def __init__(self, draft: Any = None) -> None:
        """Create the stub with its draft."""
        self.draft = draft
        self.calls: list[Any] = []

    def invoke_structured(
        self, system_prompt, user_prompt, schema, *, max_calls=None
    ):
        """Return the prepared draft."""
        self.calls.append((system_prompt, user_prompt))
        if self.draft is None:
            from app.agents.llm import LLMError

            raise LLMError("LLM_INVALID_STRUCTURED_OUTPUT")
        return self.draft


def test_conflicting_dates_across_two_documents_are_both_reported() -> None:
    case_id = uuid4()
    owner_id = uuid4()
    first_document = _document(case_id, owner_id, filename="lease.pdf")
    second_document = _document(case_id, owner_id, filename="lease-copy.pdf")
    first_span = _span(first_document.id, "The agreement was signed on 1 April 2021.")
    second_span = _span(second_document.id, "The agreement was signed on 2 April 2021.")
    facts = [
        _fact(first_document.id, first_span, date="1 April 2021"),
        _fact(second_document.id, second_span, date="2 April 2021"),
    ]

    result = analyze_contradictions(
        ContradictionInput(
            task_type="chronology",
            question="When was the agreement signed?",
            facts=facts,
            spans=[first_span, second_span],
            documents=[first_document, second_document],
        ),
        settings=Settings(_env_file=None),
        llm=StubLLM(ConflictDraft(conflicts=[], gaps=[])),
    )

    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.field == "date"
    assert conflict.evidence_ids_a == [first_span.evidence_id]
    assert conflict.evidence_ids_b == [second_span.evidence_id]
    assert "different date" in conflict.possible_reasons
    # No rule is documented, so no winner is picked.
    assert conflict.resolution_status == "unresolved"
    assert conflict.proposition_a and conflict.proposition_b


def test_a_documented_supersession_rule_resolves_the_conflict() -> None:
    case_id = uuid4()
    owner_id = uuid4()
    first_document = _document(case_id, owner_id, filename="lease.pdf")
    second_document = _document(case_id, owner_id, filename="lease-amended.pdf")
    first_span = _span(
        first_document.id,
        "The agreement was signed on 1 April 2021 and is superseded by any "
        "later signed deed.",
    )
    second_span = _span(
        second_document.id,
        "This deed dated 2 April 2021 supersedes the agreement of 1 April 2021.",
    )
    facts = [
        _fact(first_document.id, first_span, date="1 April 2021"),
        _fact(second_document.id, second_span, date="2 April 2021"),
    ]

    result = analyze_contradictions(
        ContradictionInput(
            task_type="chronology",
            question="When was the agreement signed?",
            facts=facts,
            spans=[first_span, second_span],
            documents=[first_document, second_document],
        ),
        settings=Settings(_env_file=None),
        llm=StubLLM(ConflictDraft(conflicts=[], gaps=[])),
    )

    assert result.conflicts
    conflict = result.conflicts[0]
    assert conflict.resolution_status == "resolved"
    assert "supersedes" in conflict.resolution_note


def test_two_versions_of_one_document_are_reported() -> None:
    case_id = uuid4()
    owner_id = uuid4()
    group_id = uuid4()
    first_document = _document(
        case_id, owner_id, filename="lease.pdf", version_group_id=group_id
    )
    second_document = _document(
        case_id, owner_id, filename="lease-v2.pdf", version_group_id=group_id
    )
    first_span = _span(first_document.id, "The rent is 1,250 pounds.")
    second_span = _span(second_document.id, "The rent is 1,400 pounds.")

    result = analyze_contradictions(
        ContradictionInput(
            task_type="compare_versions",
            question="What is the rent?",
            facts=[],
            spans=[first_span, second_span],
            documents=[first_document, second_document],
        ),
        settings=Settings(_env_file=None),
        llm=StubLLM(ConflictDraft(conflicts=[], gaps=[])),
    )

    assert len(result.conflicts) == 1
    conflict = result.conflicts[0]
    assert conflict.field == "version"
    assert "different version" in conflict.possible_reasons
    assert conflict.resolution_status == "unresolved"


def test_the_model_cannot_create_evidence() -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = _document(case_id, owner_id)
    span = _span(document.id, "The rent is 1,250 pounds.")
    draft = ConflictDraft(
        conflicts=[
            ConflictCandidate(
                proposition_a="The rent is 1,250 pounds.",
                proposition_b="The rent is 1,400 pounds.",
                evidence_ids_a=[span.evidence_id],
                evidence_ids_b=[uuid4()],
            )
        ]
    )

    result = analyze_contradictions(
        ContradictionInput(
            task_type="fact_lookup",
            question="What is the rent?",
            facts=[],
            spans=[span],
            documents=[document],
        ),
        settings=Settings(_env_file=None),
        llm=StubLLM(draft),
    )

    assert result.conflicts == []
    assert REASON_UNSUPPORTED_CONFLICT in result.reason_codes


def test_uncovered_subquestions_become_gaps() -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = _document(case_id, owner_id)

    result = analyze_contradictions(
        ContradictionInput(
            task_type="chronology",
            question="What happened next?",
            facts=[],
            spans=[],
            documents=[document],
            uncovered_subquestions=["sq-2"],
        ),
        settings=Settings(_env_file=None),
        llm=StubLLM(ConflictDraft(conflicts=[], gaps=[])),
    )

    assert len(result.gaps) == 1
    assert result.gaps[0].subquestion_id == "sq-2"
    assert "may exist in the case file" in result.gaps[0].description


def test_the_same_document_does_not_count_as_a_conflict() -> None:
    case_id = uuid4()
    owner_id = uuid4()
    document = _document(case_id, owner_id)
    first_span = _span(document.id, "The rent is 1,250 pounds.")
    second_span = _span(document.id, "The rent is 1,400 pounds.")
    facts = [
        _fact(document.id, first_span, event="The rent was fixed", date="1 April"),
        _fact(document.id, second_span, event="The rent was fixed", date="2 April"),
    ]

    result = analyze_contradictions(
        ContradictionInput(
            task_type="chronology",
            question="When was the rent fixed?",
            facts=facts,
            spans=[first_span, second_span],
            documents=[document],
        ),
        settings=Settings(_env_file=None),
        llm=StubLLM(ConflictDraft(conflicts=[], gaps=[])),
    )

    assert result.conflicts == []
