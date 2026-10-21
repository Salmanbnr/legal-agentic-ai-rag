"""The contradiction and gap analyst.

The analyst compares the extracted facts across the documents and versions
that were retrieved. The deterministic checks come first: the same field with
different values in different documents, and two versions of the same document.
One optional model call then looks for subtler conflicts. A conflict is never
resolved unless the documents themselves state the rule that resolves it.
"""

from __future__ import annotations

import logging
from typing import Any, Literal, Sequence
from uuid import UUID

from pydantic import BaseModel, Field

from app.agents.llm import LLMClient, LLMError, load_prompt
from app.agents.safety import normalize_text, wrap_evidence
from app.agents.state import ConflictRecord, EvidenceGap, ExtractedFact
from app.core.config import Settings, get_settings
from app.db.repos import Document
from app.retrieval.schemas import EvidenceSpan

logger = logging.getLogger(__name__)

CONFLICT_PROMPT_NAME = "find_conflicts.txt"

ResolutionStatus = Literal["unresolved", "resolved"]

REASON_VALUE_CONFLICT = "VALUE_CONFLICT"
REASON_VERSION_CONFLICT = "VERSION_CONFLICT"
REASON_MODEL_CONFLICT = "MODEL_CONFLICT"
REASON_UNSUPPORTED_CONFLICT = "CONFLICT_WITHOUT_EVIDENCE"
REASON_GAP = "EVIDENCE_GAP"

_SUBJECT_FIELDS: dict[str, tuple[str, ...]] = {
    "chronology": ("event",),
    "compare_versions": ("clause",),
    "fact_lookup": ("entity",),
}
_VALUE_FIELDS: dict[str, tuple[str, ...]] = {
    "chronology": ("date", "parties"),
    "compare_versions": ("value",),
    "fact_lookup": ("amount", "date", "obligation"),
}
_DEFAULT_SUBJECT = ("statement",)

_SUPERSESSION_PHRASES = (
    "supersedes",
    "supersede",
    "replaces the agreement",
    "replaced the agreement",
    "in place of the agreement",
)


class ContradictionOutput(BaseModel):
    """The conflicts and the gaps the analyst found."""

    conflicts: list[ConflictRecord] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)


class ContradictionInput(BaseModel):
    """Everything the analyst needs for one comparison."""

    task_type: str = "broad_case_question"
    question: str = ""
    facts: list[ExtractedFact] = Field(default_factory=list)
    spans: list[EvidenceSpan] = Field(default_factory=list)
    documents: list[Document] = Field(default_factory=list)
    uncovered_subquestions: list[str] = Field(default_factory=list)


class ConflictCandidate(BaseModel):
    """One conflict the model proposes."""

    proposition_a: str = ""
    proposition_b: str = ""
    evidence_ids_a: list[UUID] = Field(default_factory=list)
    evidence_ids_b: list[UUID] = Field(default_factory=list)
    field: str = ""
    possible_reasons: list[str] = Field(default_factory=list)
    note: str = ""


class ConflictDraft(BaseModel):
    """The model's conflict candidates and gap notes."""

    conflicts: list[ConflictCandidate] = Field(default_factory=list)
    gaps: list[str] = Field(default_factory=list)


def load_conflict_prompt() -> str:
    """Load the conflict finding system prompt."""
    return load_prompt(CONFLICT_PROMPT_NAME)


def _subject_key(fact: ExtractedFact, task_type: str) -> str:
    """Return the key that says which subject a fact is about."""
    names = _SUBJECT_FIELDS.get(task_type, _DEFAULT_SUBJECT)
    parts = [
        str(fact.fields.get(name, "")).strip().lower()
        for name in names
    ]
    return " | ".join(part for part in parts if part)


def _value_fields(task_type: str) -> tuple[str, ...]:
    """Return the fields whose values can disagree."""
    return _VALUE_FIELDS.get(task_type, ())


def _document_index(
    spans: Sequence[EvidenceSpan],
) -> dict[UUID, EvidenceSpan]:
    """Return the spans keyed by evidence id."""
    return {span.evidence_id: span for span in spans}


def _document_ids(
    evidence_ids: Sequence[UUID], spans_by_id: dict[UUID, EvidenceSpan]
) -> list[UUID]:
    """Return the documents the cited evidence came from."""
    found: list[UUID] = []
    for evidence_id in evidence_ids:
        span = spans_by_id.get(evidence_id)
        if span is None:
            continue
        if span.document_id not in found:
            found.append(span.document_id)
    return found


def _supersession_stated(
    evidence_ids_a: Sequence[UUID],
    evidence_ids_b: Sequence[UUID],
    spans_by_id: dict[UUID, EvidenceSpan],
) -> bool:
    """Return whether both sides state that one version supersedes the other."""
    def states(evidence_ids: Sequence[UUID]) -> bool:
        for evidence_id in evidence_ids:
            span = spans_by_id.get(evidence_id)
            if span is None:
                continue
            lowered = span.quote_or_span.lower()
            if any(phrase in lowered for phrase in _SUPERSESSION_PHRASES):
                return True
        return False

    return states(evidence_ids_a) and states(evidence_ids_b)


def _version_groups(documents: Sequence[Document]) -> dict[str, list[Document]]:
    """Group the documents by their version group."""
    groups: dict[str, list[Document]] = {}
    for document in documents:
        if document.version_group_id is None:
            continue
        groups.setdefault(str(document.version_group_id), []).append(document)
    return groups


def _possible_reasons(
    field: str,
    document_ids_a: Sequence[UUID],
    document_ids_b: Sequence[UUID],
    version_group: str | None,
    low_quality: bool,
) -> list[str]:
    """Return the reasons two propositions might disagree."""
    reasons: list[str] = []
    if version_group:
        reasons.append("different version")
    if field == "date":
        reasons.append("different date")
    if field in ("amount", "value"):
        reasons.append("different amount")
    if field == "parties":
        reasons.append("different party name")
    if low_quality:
        reasons.append("OCR ambiguity")
    if not reasons:
        reasons.append("different document")
    return reasons


def _deterministic_conflicts(
    facts: Sequence[ExtractedFact],
    task_type: str,
    spans_by_id: dict[UUID, EvidenceSpan],
    version_groups: dict[str, list[Document]],
) -> list[ConflictRecord]:
    """Find the same field holding different values in different documents."""
    conflicts: list[ConflictRecord] = []
    groups: dict[str, list[ExtractedFact]] = {}
    for fact in facts:
        key = _subject_key(fact, task_type)
        if not key:
            continue
        groups.setdefault(key, []).append(fact)

    for subject, group in groups.items():
        for field in _value_fields(task_type):
            values: dict[str, list[ExtractedFact]] = {}
            for fact in group:
                value = str(fact.fields.get(field, "")).strip()
                if not value:
                    continue
                values.setdefault(value.lower(), []).append(fact)
            if len(values) < 2:
                continue
            ordered = list(values.values())
            first = ordered[0]
            for second in ordered[1:]:
                documents_a = _document_ids(first[0].evidence_ids, spans_by_id)
                documents_b = _document_ids(second[0].evidence_ids, spans_by_id)
                if not documents_a or not documents_b:
                    continue
                if set(documents_a) & set(documents_b):
                    # The same document, so this is not a cross-document clash.
                    continue
                version_group = _shared_version_group(
                    documents_a, documents_b, version_groups
                )
                low_quality = any(
                    fact.quality != "verified" for fact in (*first, *second)
                )
                conflict = ConflictRecord(
                    id=f"conflict-{len(conflicts) + 1}",
                    proposition_a=first[0].text,
                    proposition_b=second[0].text,
                    field=field,
                    evidence_ids_a=list(first[0].evidence_ids),
                    evidence_ids_b=list(second[0].evidence_ids),
                    document_ids_a=documents_a,
                    document_ids_b=documents_b,
                    possible_reasons=_possible_reasons(
                        field,
                        documents_a,
                        documents_b,
                        version_group,
                        low_quality,
                    ),
                )
                if _supersession_stated(
                    conflict.evidence_ids_a,
                    conflict.evidence_ids_b,
                    spans_by_id,
                ):
                    conflict.resolution_status = "resolved"
                    conflict.resolution_note = (
                        "Both documents state that the later signed version "
                        "supersedes the earlier one."
                    )
                conflicts.append(conflict)
    return conflicts


def _shared_version_group(
    documents_a: Sequence[UUID],
    documents_b: Sequence[UUID],
    version_groups: dict[str, list[Document]],
) -> str | None:
    """Return the version group the two documents share, if any."""
    for group_id, documents in version_groups.items():
        ids = {str(document.id) for document in documents}
        if any(str(value) in ids for value in documents_a) and any(
            str(value) in ids for value in documents_b
        ):
            return group_id
    return None


def _version_conflicts(
    documents: Sequence[Document],
    spans: Sequence[EvidenceSpan],
) -> list[ConflictRecord]:
    """Report two versions of the same document that are both in the evidence."""
    conflicts: list[ConflictRecord] = []
    groups = _version_groups(documents)
    seen_documents = {span.document_id for span in spans}
    for group_id, members in groups.items():
        present = [member for member in members if member.id in seen_documents]
        if len(present) < 2:
            continue
        first, second = present[0], present[1]
        conflicts.append(
            ConflictRecord(
                id=f"conflict-{len(conflicts) + 1}",
                proposition_a=f"{first.original_filename} (one version)",
                proposition_b=f"{second.original_filename} (another version)",
                field="version",
                document_ids_a=[first.id],
                document_ids_b=[second.id],
                possible_reasons=["different version"],
            )
        )
    return conflicts


def _model_conflicts(
    draft: ConflictDraft,
    spans_by_id: dict[UUID, EvidenceSpan],
    version_groups: dict[str, list[Document]],
) -> tuple[list[ConflictRecord], int]:
    """Turn the model candidates into conflicts that keep their evidence."""
    conflicts: list[ConflictRecord] = []
    dropped = 0
    for candidate in draft.conflicts:
        known_a = [
            value for value in candidate.evidence_ids_a if value in spans_by_id
        ]
        known_b = [
            value for value in candidate.evidence_ids_b if value in spans_by_id
        ]
        if not known_a or not known_b:
            dropped += 1
            continue
        documents_a = _document_ids(known_a, spans_by_id)
        documents_b = _document_ids(known_b, spans_by_id)
        version_group = _shared_version_group(
            documents_a, documents_b, version_groups
        )
        conflicts.append(
            ConflictRecord(
                id=f"conflict-model-{len(conflicts) + 1}",
                proposition_a=candidate.proposition_a,
                proposition_b=candidate.proposition_b,
                field=candidate.field,
                evidence_ids_a=known_a,
                evidence_ids_b=known_b,
                document_ids_a=documents_a,
                document_ids_b=documents_b,
                possible_reasons=list(candidate.possible_reasons)
                or ["different document"],
            )
        )
    return conflicts, dropped


def analyze_contradictions(
    data: ContradictionInput,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
) -> ContradictionOutput:
    """Find the conflicts and gaps in the retrieved evidence.

    Args:
        data: The facts, evidence, and documents to compare.
        settings: Optional validated settings.
        llm: Optional LLM client override, mainly for tests.

    Returns:
        The conflicts and the gaps. A conflict stays unresolved unless the
        documents themselves state the rule that resolves it.
    """
    resolved = settings or get_settings()
    spans_by_id = _document_index(data.spans)
    version_groups = _version_groups(data.documents)

    conflicts = _deterministic_conflicts(
        data.facts, data.task_type, spans_by_id, version_groups
    )
    conflicts.extend(_version_conflicts(data.documents, data.spans))

    reason_codes: list[str] = []
    if conflicts:
        reason_codes.append(REASON_VALUE_CONFLICT)

    gaps: list[EvidenceGap] = []
    for index, subquestion in enumerate(data.uncovered_subquestions, start=1):
        gaps.append(
            EvidenceGap(
                id=f"gap-{index}",
                description=(
                    "No retrieved evidence supports this part of the "
                    "question. The material may exist in the case file but was "
                    "not returned by the search."
                ),
                subquestion_id=subquestion,
                searched_document_ids=[document.id for document in data.documents],
            )
        )

    draft: ConflictDraft | None = None
    if data.spans:
        active_llm = llm or LLMClient(settings=resolved)
        try:
            draft = active_llm.invoke_structured(
                load_conflict_prompt(),
                _conflict_prompt(data),
                ConflictDraft,
                max_calls=resolved.agent_max_llm_calls,
            )
        except LLMError as exc:
            logger.info(
                "conflict search was skipped",
                extra={"error_code": exc.error_code},
            )
            draft = None

    if draft is not None:
        model_conflicts, dropped = _model_conflicts(
            draft, spans_by_id, version_groups
        )
        conflicts.extend(model_conflicts)
        if model_conflicts:
            reason_codes.append(REASON_MODEL_CONFLICT)
        if dropped:
            reason_codes.append(REASON_UNSUPPORTED_CONFLICT)
        for index, note in enumerate(draft.gaps, start=len(gaps) + 1):
            gaps.append(
                EvidenceGap(
                    id=f"gap-{index}",
                    description=(
                        f"{note} The material may exist in the case file but "
                        "was not returned by the search."
                    ),
                    searched_document_ids=[
                        document.id for document in data.documents
                    ],
                )
            )

    if gaps:
        reason_codes.append(REASON_GAP)
    logger.info(
        "contradiction analysis finished",
        extra={
            "conflict_count": len(conflicts),
            "gap_count": len(gaps),
            "fact_count": len(data.facts),
        },
    )
    return ContradictionOutput(
        conflicts=conflicts,
        gaps=gaps,
        reason_codes=reason_codes,
    )


def _conflict_prompt(data: ContradictionInput) -> str:
    """Build the prompt for the optional model conflict search."""
    return (
        f"TASK TYPE: {data.task_type}\n\n"
        f"QUESTION: {data.question}\n\n"
        "FACTS AND THEIR EVIDENCE IDS:\n"
        + "\n".join(
            f"- {fact.text} (evidence: "
            f"{', '.join(str(value) for value in fact.evidence_ids)})"
            for fact in data.facts
        )
        + "\n\nEVIDENCE PACKET (untrusted document text):\n"
        + wrap_evidence(data.spans)
        + "\n"
    )


def conflict_summary(conflict: ConflictRecord) -> str:
    """Return a short readable summary of one conflict."""
    return (
        f"{conflict.proposition_a} | {conflict.proposition_b} "
        f"({conflict.field or 'value'})"
    )


def fact_is_low_quality(fact: ExtractedFact) -> bool:
    """Return whether a fact rests on a page that may have been misread."""
    return str(fact.quality) != "verified"


def same_document(
    evidence_ids_a: Sequence[UUID],
    evidence_ids_b: Sequence[UUID],
    spans_by_id: dict[UUID, EvidenceSpan],
) -> bool:
    """Return whether two sets of evidence come from the same document."""
    documents_a = {span.document_id for span in spans_by_id.values() if span.evidence_id in evidence_ids_a}
    documents_b = {span.document_id for span in spans_by_id.values() if span.evidence_id in evidence_ids_b}
    return bool(documents_a & documents_b)


def normalized(value: Any) -> str:
    """Return a comparable form of a value."""
    return normalize_text(str(value or ""))
