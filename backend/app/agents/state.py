"""State models for the agentic retrieval workflow.

The state holds identifiers, references, and counts. Document text stays in
Postgres and in the evidence spans that the answer layer returns, so the
graph state never becomes a second copy of the case file.
"""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from app.observability.tracing import TraceEvent
from app.retrieval.answer_schema import AnswerClaim, AnswerResponse, FinalStatus

TaskType = Literal[
    "fact_lookup",
    "chronology",
    "compare_versions",
    "summarize_document",
    "issue_analysis",
    "broad_case_question",
]

SubquestionStatus = Literal[
    "pending",
    "retrieving",
    "retrieved",
    "gap",
    "answered",
]

REASON_AUTHORIZATION_FAILED = "AUTHORIZATION_FAILED"
REASON_BUDGET_EXCEEDED = "BUDGET_EXCEEDED"
REASON_WEAK_COVERAGE = "WEAK_COVERAGE"
REASON_NO_EVIDENCE = "NO_EVIDENCE_RETRIEVED"
REASON_PLAN_INVALID = "PLAN_INVALID"
REASON_GRAPH_VERSION = "AGENTIC_RAG_1"


class Subquestion(BaseModel):
    """One planned question inside the agent workflow."""

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str = Field(min_length=1)
    purpose: str = ""
    status: SubquestionStatus = "pending"


class RetrievedEvidence(BaseModel):
    """A reference to one retrieved span: ids, pages, and scores only."""

    model_config = ConfigDict(extra="forbid")

    subquestion_id: str
    evidence_id: UUID
    document_id: UUID
    document_version_id: UUID | None = None
    page_start: int | None = None
    page_end: int | None = None
    source_anchor: str | None = None
    retrieval_method: str = "hybrid"
    retrieval_rank: int = 1
    quality_flags: list[Any] = Field(default_factory=list)
    score: float | None = None


class ExtractedFact(BaseModel):
    """One fact taken from the retrieved evidence.

    The fact keeps the structured values the analyst produced, the ids and
    pages of the evidence behind it, and the exact quote that was checked
    against the evidence. The quote is a short excerpt, never a whole page.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    text: str
    evidence_ids: list[UUID] = Field(default_factory=list)
    subquestion_id: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    uncertainty: str = "certain"
    quality: str = "verified"
    quote: str = ""
    fields: dict[str, Any] = Field(default_factory=dict)


class ConflictRecord(BaseModel):
    """One disagreement between two pieces of evidence.

    A conflict stays unresolved unless the documents themselves state the
    rule that resolves it.
    """

    model_config = ConfigDict(extra="forbid")

    id: str
    proposition_a: str = ""
    proposition_b: str = ""
    field: str = ""
    evidence_ids_a: list[UUID] = Field(default_factory=list)
    evidence_ids_b: list[UUID] = Field(default_factory=list)
    document_ids_a: list[UUID] = Field(default_factory=list)
    document_ids_b: list[UUID] = Field(default_factory=list)
    possible_reasons: list[str] = Field(default_factory=list)
    resolution_status: Literal["unresolved", "resolved"] = "unresolved"
    resolution_note: str = ""


class EvidenceGap(BaseModel):
    """Something the workflow looked for and did not find."""

    model_config = ConfigDict(extra="forbid")

    id: str
    description: str
    subquestion_id: str | None = None
    searched_document_ids: list[UUID] = Field(default_factory=list)


class DraftClaim(AnswerClaim):
    """A claim drafted by the workflow before verification."""

    model_config = ConfigDict(extra="forbid")

    id: str = ""
    subquestion_id: str | None = None
    quote: str = ""


class VerificationResult(BaseModel):
    """The result of checking one draft claim against the evidence."""

    model_config = ConfigDict(extra="forbid")

    claim_id: str
    supported: bool = False
    evidence_ids: list[UUID] = Field(default_factory=list)
    reason_codes: list[str] = Field(default_factory=list)
    label: str = "supported"
    material: bool = False
    note: str = ""


class TracePlanStep(BaseModel):
    """One planned subquestion, as shown in the workflow panel."""

    model_config = ConfigDict(extra="forbid")

    text: str
    purpose: str = ""
    status: SubquestionStatus = "pending"


class AgentTraceSummary(BaseModel):
    """A short summary of one agentic run, safe to return to the client.

    The summary describes the workflow, not the case file: planned questions,
    the queries that were tried, the nodes that ran, and the counters. It never
    carries document text, prompts, or completions.
    """

    model_config = ConfigDict(extra="forbid")

    graph_version: str = "agentic-rag-3"
    node_path: list[str] = Field(default_factory=list)
    plan: list[TracePlanStep] = Field(default_factory=list)
    queries: list[str] = Field(default_factory=list)
    steps: int = 0
    llm_calls: int = 0
    tokens_used: int = 0
    retries_used: int = 0
    duration_ms: int = 0
    final_status: FinalStatus = "insufficient_evidence"
    reason_codes: list[str] = Field(default_factory=list)


class AgenticRAGState(BaseModel):
    """The state of one agentic retrieval run."""

    model_config = ConfigDict(extra="forbid")

    request_id: str | None = None
    user_id: UUID
    case_id: UUID
    question: str
    task_type: TaskType = "fact_lookup"

    authorized_document_ids: list[UUID] = Field(default_factory=list)
    plan: list[Subquestion] = Field(default_factory=list)
    retrieval_results: list[RetrievedEvidence] = Field(default_factory=list)
    recommended_next_search: str | None = None
    extracted_facts: list[ExtractedFact] = Field(default_factory=list)
    contradictions: list[ConflictRecord] = Field(default_factory=list)
    evidence_gaps: list[EvidenceGap] = Field(default_factory=list)
    draft_claims: list[DraftClaim] = Field(default_factory=list)
    verification_results: list[VerificationResult] = Field(default_factory=list)

    retries_used: int = 0
    max_retrieval_retries: int = 0
    steps_used: int = 0
    verification_rounds: int = 0
    llm_calls_used: int = 0
    tokens_used: int = 0
    deadline: float | None = None

    final_answer: AnswerResponse | None = None
    final_status: FinalStatus = "insufficient_evidence"
    reason_codes: Annotated[list[str], operator.add] = Field(
        default_factory=list
    )
    trace: Annotated[list[TraceEvent], operator.add] = Field(
        default_factory=list
    )
