"""The LangGraph orchestration of the agentic retrieval workflow.

The graph is a controlled workflow, not a chat loop. Every node is bounded by
the budget, the authorization node fails closed, and the workflow always ends
in a state that carries a final status and a reason code.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable
from uuid import UUID

from langgraph.graph import END, START, StateGraph
from supabase import Client

from app.agents.answer_synthesizer import SynthesisInput, synthesize_answer
from app.agents.budget import Budget, BudgetExceeded
from app.agents.claim_drafter import ClaimDraftInput, draft_claims
from app.agents.contradiction_agent import (
    ContradictionInput,
    analyze_contradictions,
)
from app.agents.document_analyst import analyze_evidence, should_skip_analysis
from app.agents.evidence_verifier import VerificationInput, verify_claims
from app.agents.final_gate import FinalGateInput, run_final_gate
from app.agents.llm import LLMBudgetExceeded, LLMClient, LLMError
from app.agents.quality_gate import (
    GATE_ABSTAIN,
    GATE_ACCEPT,
    GATE_ASK_CLARIFICATION,
    GATE_RETRY_RETRIEVAL,
    decide,
    coverage_summary,
)
from app.agents.research_agent import research
from app.agents.router import coordinator_node
from app.agents.safety import scan_spans_for_injection
from app.agents.state import (
    REASON_AUTHORIZATION_FAILED,
    REASON_BUDGET_EXCEEDED,
    AgenticRAGState,
    AgentTraceSummary,
    EvidenceGap,
    ExtractedFact,
    RetrievedEvidence,
    Subquestion,
    TracePlanStep,
)
from app.auth.dependencies import CurrentUser
from app.core.config import Settings, get_settings
from app.db.repos import (
    get_case_for_user,
    get_chunks_by_ids,
    get_pages_for_documents,
    list_documents,
)
from app.embeddings.model import TextEmbedder
from app.observability.tracing import trace_node
from app.retrieval.baseline_rag import answer_question
from app.retrieval.citations import build_insufficient_evidence_answer
from app.retrieval.evidence_meta import collect_quality_flags, page_signals
from app.retrieval.schemas import EvidenceSpan

logger = logging.getLogger(__name__)

GRAPH_VERSION = "agentic-rag-3"
MAX_PLAN_SIZE = 4
RETRIEVAL_TOP_K = 8
QUESTION_WORDS = {
    "the",
    "a",
    "an",
    "of",
    "and",
    "or",
    "to",
    "in",
    "on",
    "for",
    "what",
    "which",
    "who",
    "whom",
    "whose",
    "when",
    "where",
    "why",
    "how",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "do",
    "does",
    "did",
    "has",
    "have",
    "had",
    "that",
    "this",
    "these",
    "those",
    "there",
    "it",
    "its",
    "any",
    "all",
    "can",
    "could",
    "should",
    "would",
    "will",
    "shall",
    "may",
    "much",
    "many",
    "case",
    "file",
    "document",
    "documents",
    "please",
    "tell",
    "me",
    "about",
}


@dataclass
class AgentDeps:
    """The dependencies every node needs."""

    client: Client
    user: CurrentUser
    settings: Settings
    llm: LLMClient
    budget: Budget
    embedder: TextEmbedder | None = None
    index: Any | None = None


def _check_budget(state: AgenticRAGState, deps: AgentDeps) -> None:
    """Stop the run when one of the budget limits is reached."""
    deps.budget.check(
        steps=int(state.steps_used),
        llm_calls=int(state.llm_calls_used),
        tokens=int(state.tokens_used),
        retries=int(state.retries_used),
    )


def _new_codes(state: Any, codes: list[str]) -> list[str]:
    """Return the codes that are not already recorded on the state."""
    existing = set(getattr(state, "reason_codes", []) or [])
    return [code for code in codes if code not in existing]


def _fact_text(fact: Any) -> str:
    """Build a readable summary of one fact from its task specific fields."""
    parts: list[str] = []
    for key, value in fact.model_dump().items():
        if key in (
            "quote",
            "evidence_ids",
            "page_start",
            "page_end",
            "uncertainty",
            "quality",
        ):
            continue
        if value in (None, "", []):
            continue
        if isinstance(value, list):
            value = ", ".join(str(item) for item in value)
        parts.append(f"{key}: {value}")
    return "; ".join(parts)


def _load_documents(
    state: AgenticRAGState, deps: AgentDeps
) -> list[Any]:
    """Load the authorized documents, with their version groups."""
    try:
        return list_documents(
            deps.client, deps.user.id, state.case_id, limit=200
        )
    except Exception as exc:
        logger.warning(
            "documents could not be loaded",
            extra={"error_class": type(exc).__name__},
        )
        return []


def _load_evidence_spans(
    state: AgenticRAGState, deps: AgentDeps
) -> list[EvidenceSpan]:
    """Load the evidence text for the retrieved ids from Postgres.

    The state only keeps ids and pages, so the analyst packet is rebuilt from
    the stored chunk text. That is the same text the quote is checked against.
    """
    ids = [item.evidence_id for item in state.retrieval_results]
    if not ids:
        return []
    try:
        chunks = get_chunks_by_ids(
            deps.client, deps.user.id, state.case_id, ids
        )
        documents = list_documents(
            deps.client, deps.user.id, state.case_id, limit=200
        )
    except Exception as exc:
        logger.warning(
            "evidence text could not be loaded",
            extra={"error_class": type(exc).__name__},
        )
        return []
    # The page metadata only enriches a citation. A failure to read it must not
    # hide evidence that was already retrieved and authorised.
    pages: list[Any] = []
    try:
        pages = get_pages_for_documents(
            deps.client,
            deps.user.id,
            state.case_id,
            list({chunk.document_id for chunk in chunks}),
        )
    except Exception as exc:
        logger.warning(
            "page metadata could not be loaded",
            extra={"error_class": type(exc).__name__},
        )
    titles = {str(document.id): document for document in documents}
    pages_by_document: dict[str, list[Any]] = {}
    for page in pages:
        pages_by_document.setdefault(str(page.document_id), []).append(page)
    spans: list[EvidenceSpan] = []
    for chunk in chunks:
        document = titles.get(str(chunk.document_id))
        document_pages = pages_by_document.get(str(chunk.document_id), [])
        extraction_method, confidence = page_signals(document_pages, chunk)
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
                extraction_method=extraction_method,
                quality_status=(
                    document.quality_status if document else "needs_review"
                ),
                ocr_confidence=confidence,
                source_anchor=chunk.source_anchor,
                retrieval_method="hybrid",
                retrieval_rank=max(1, chunk.chunk_index),
                quality_flags=collect_quality_flags(chunk, document_pages),
            )
        )
    return spans[: deps.settings.agent_analyst_max_spans]


def broaden_query(text: str) -> str:
    """Broaden a retrieval query by dropping question words."""
    words = [
        word
        for word in (text or "").split()
        if word.lower().strip(".,:;?\"'") not in QUESTION_WORDS
    ]
    if not words:
        return (text or "").strip()
    return " ".join(words)


@trace_node("authorize", graph_version=GRAPH_VERSION)
def authorize_node(state: AgenticRAGState, deps: AgentDeps) -> dict[str, Any]:
    """Resolve the case and its documents, or fail closed."""
    _check_budget(state, deps)
    try:
        get_case_for_user(deps.client, deps.user.id, state.case_id)
        documents = list_documents(
            deps.client, deps.user.id, state.case_id, limit=200
        )
    except Exception as exc:
        logger.warning(
            "authorization failed, the run is stopped",
            extra={"error_class": type(exc).__name__},
        )
        return {
            "reason_codes": [REASON_AUTHORIZATION_FAILED],
            "final_status": "insufficient_evidence",
            "status": "denied",
        }
    return {
        "authorized_document_ids": [document.id for document in documents],
        "status": "ok",
        "details": {"document_count": len(documents)},
    }


@trace_node("route", graph_version=GRAPH_VERSION)
def route_node(state: AgenticRAGState, deps: AgentDeps) -> dict[str, Any]:
    """Classify the question and plan the subquestions."""
    _check_budget(state, deps)
    update = coordinator_node(state, deps.llm, settings=deps.settings)
    update["status"] = "ok"
    return update


@trace_node("build_plan", graph_version=GRAPH_VERSION)
def plan_node(state: AgenticRAGState, deps: AgentDeps) -> dict[str, Any]:
    """Finalise the plan: unique ids, pending status, bounded size."""
    _check_budget(state, deps)
    planned = list(state.plan)[:MAX_PLAN_SIZE]
    if not planned:
        planned = [
            Subquestion(
                id="sq-1",
                text=(state.question or "").strip() or "What does the file say?",
                purpose="Answer the question.",
            )
        ]
    final: list[Subquestion] = []
    seen: set[str] = set()
    for index, item in enumerate(planned, start=1):
        identifier = item.id if item.id not in seen else f"sq-{index}"
        seen.add(identifier)
        final.append(
            item.model_copy(
                update={"id": identifier, "status": "pending"}
            )
        )
    return {
        "plan": final,
        "status": "ok",
        "details": {"plan_count": len(final)},
    }


@trace_node("retrieve", graph_version=GRAPH_VERSION)
def retrieve_node(state: AgenticRAGState, deps: AgentDeps) -> dict[str, Any]:
    """Run the research agent over the planned subquestions."""
    _check_budget(state, deps)
    prior_spans = (
        _load_evidence_spans(state, deps) if state.retrieval_results else []
    )
    output = research(
        deps.client,
        state.case_id,
        deps.user,
        state.question,
        state.plan,
        state.authorized_document_ids,
        settings=deps.settings,
        llm=deps.llm,
        embedder=deps.embedder,
        index=deps.index,
        budget=deps.budget,
        steps_used=int(state.steps_used),
        llm_calls_used=int(state.llm_calls_used),
        retries_remaining=max(
            0,
            deps.settings.agent_max_retrieval_retries
            - int(state.retries_used),
        ),
        prior_results=list(state.retrieval_results),
        prior_spans=prior_spans,
    )

    known = {str(item.evidence_id) for item in state.retrieval_results}
    collected: list[RetrievedEvidence] = []
    for span in output.evidence:
        if str(span.evidence_id) in known:
            continue
        known.add(str(span.evidence_id))
        collected.append(
            RetrievedEvidence(
                subquestion_id=state.plan[0].id if state.plan else "sq-1",
                evidence_id=span.evidence_id,
                document_id=span.document_id,
                document_version_id=span.document_version_id,
                page_start=span.page_start,
                page_end=span.page_end,
                source_anchor=span.source_anchor,
                retrieval_method=span.retrieval_method,
                retrieval_rank=span.retrieval_rank,
                quality_flags=list(span.quality_flags),
                score=span.retrieval_score,
            )
        )

    updated_plan: list[Subquestion] = []
    for subquestion in state.plan:
        assessment = next(
            (
                item
                for item in output.coverage
                if item.subquestion_id == subquestion.id
            ),
            None,
        )
        status = "retrieved" if assessment and assessment.coverage_ok else "gap"
        updated_plan.append(subquestion.model_copy(update={"status": status}))

    results = [*state.retrieval_results, *collected]
    return {
        "retrieval_results": results,
        "evidence_gaps": [*state.evidence_gaps, *output.gaps],
        "recommended_next_search": output.recommended_next_search,
        "plan": updated_plan,
        "steps_used": int(state.steps_used) + max(0, output.steps_used),
        "llm_calls_used": int(state.llm_calls_used)
        + max(0, output.llm_calls_used),
        "reason_codes": _new_codes(state, output.reason_codes),
        "status": "ok",
        "details": {
            "span_count": len(results),
            "document_count": len({str(item.document_id) for item in results}),
        },
    }


@trace_node("assess_coverage", graph_version=GRAPH_VERSION)
def assess_coverage_node(
    state: AgenticRAGState, deps: AgentDeps
) -> dict[str, Any]:
    """Record how well the retrieved evidence covers the question."""
    _check_budget(state, deps)
    coverage = coverage_summary(state)
    return {
        "reason_codes": _new_codes(state, _coverage_codes(coverage)),
        "status": "ok",
        "details": coverage,
    }


def _coverage_codes(coverage: dict[str, int | bool]) -> list[str]:
    """Return the reason codes that describe the coverage."""
    if coverage["coverage_ok"]:
        return []
    if not coverage["span_count"]:
        return ["NO_EVIDENCE_RETRIEVED"]
    return ["WEAK_COVERAGE"]


@trace_node("rewrite_query", graph_version=GRAPH_VERSION)
def rewrite_query_node(
    state: AgenticRAGState, deps: AgentDeps
) -> dict[str, Any]:
    """Turn the recommended next search into one more planned subquestion."""
    _check_budget(state, deps)
    recommended = (state.recommended_next_search or "").strip()
    if not recommended:
        return {
            "retries_used": int(state.retries_used) + 1,
            "reason_codes": _new_codes(state, ["NO_RECOMMENDED_SEARCH"]),
            "status": "ok",
            "details": {"attempt": int(state.retries_used) + 1},
        }
    planned_texts = {item.text.strip().lower() for item in state.plan}
    if recommended.lower() in planned_texts:
        return {
            "retries_used": int(state.retries_used) + 1,
            "reason_codes": _new_codes(state, ["QUERY_REWRITTEN"]),
            "status": "ok",
            "details": {"attempt": int(state.retries_used) + 1},
        }
    plan = [
        *state.plan,
        Subquestion(
            id=f"sq-retry-{int(state.retries_used) + 1}",
            text=recommended,
            purpose="Follow up on the evidence that was not covered.",
        ),
    ]
    return {
        "plan": plan[: MAX_PLAN_SIZE + 1],
        "retries_used": int(state.retries_used) + 1,
        "reason_codes": _new_codes(state, ["QUERY_REWRITTEN"]),
        "status": "ok",
        "details": {"attempt": int(state.retries_used) + 1},
    }


@trace_node("mark_gap", graph_version=GRAPH_VERSION)
def mark_gap_node(state: AgenticRAGState, deps: AgentDeps) -> dict[str, Any]:
    """Record what could not be found, without claiming it is absent."""
    _check_budget(state, deps)
    gap = EvidenceGap(
        id=f"gap-{len(state.evidence_gaps) + 1}",
        description=(
            "The retrieved evidence does not cover every part of the "
            "question. The material may exist in the case file but was not "
            "returned by the search."
        ),
        searched_document_ids=list(state.authorized_document_ids),
    )
    return {
        "evidence_gaps": [*state.evidence_gaps, gap],
        "reason_codes": _new_codes(state, ["EVIDENCE_GAP"]),
        "status": "gap",
        "details": {"gap_count": len(state.evidence_gaps) + 1},
    }


@trace_node("extract_facts", graph_version=GRAPH_VERSION)
def extract_facts_node(
    state: AgenticRAGState, deps: AgentDeps
) -> dict[str, Any]:
    """Run the facts analyst over the retrieved evidence."""
    _check_budget(state, deps)
    spans = _load_evidence_spans(state, deps)
    if should_skip_analysis(
        state.task_type, spans, settings=deps.settings
    ):
        logger.info(
            "facts analysis skipped for a strong fact lookup",
            extra={"document_count": len({span.document_id for span in spans})},
        )
        return {
            "status": "ok",
            "reason_codes": _new_codes(state, ["ANALYST_SKIPPED"]),
            "details": {"span_count": len(spans), "claim_count": 0},
        }
    if not spans:
        return {
            "status": "ok",
            "reason_codes": _new_codes(state, ["NO_EVIDENCE_RETRIEVED"]),
            "details": {"span_count": 0, "claim_count": 0},
        }

    result = analyze_evidence(
        spans,
        state.task_type,
        question=state.question,
        llm=deps.llm,
        settings=deps.settings,
    )
    facts: list[ExtractedFact] = []
    for index, fact in enumerate(result.facts, start=1):
        fields = fact.model_dump(
            exclude={
                "quote",
                "evidence_ids",
                "page_start",
                "page_end",
                "uncertainty",
                "quality",
            }
        )
        facts.append(
            ExtractedFact(
                id=f"fact-{index}",
                text=_fact_text(fact),
                evidence_ids=list(fact.evidence_ids),
                page_start=fact.page_start,
                page_end=fact.page_end,
                uncertainty=fact.uncertainty,
                quality=fact.quality,
                fields={
                    key: value for key, value in fields.items() if value not in (None, "", [])
                },
            )
        )
    return {
        "extracted_facts": facts,
        "llm_calls_used": int(state.llm_calls_used) + 1,
        "reason_codes": _new_codes(state, result.reason_codes),
        "status": "ok",
        "details": {"span_count": len(spans), "claim_count": len(facts)},
    }


@trace_node("detect_conflicts", graph_version=GRAPH_VERSION)
def detect_conflicts_node(
    state: AgenticRAGState, deps: AgentDeps
) -> dict[str, Any]:
    """Compare the facts across the documents and versions that were found."""
    _check_budget(state, deps)
    spans = _load_evidence_spans(state, deps)
    documents = _load_documents(state, deps)
    uncovered = [
        subquestion.id
        for subquestion in state.plan
        if subquestion.status == "gap"
    ]
    result = analyze_contradictions(
        ContradictionInput(
            task_type=state.task_type,
            question=state.question,
            facts=list(state.extracted_facts),
            spans=spans,
            documents=documents,
            uncovered_subquestions=uncovered,
        ),
        settings=deps.settings,
        llm=deps.llm,
    )
    return {
        "contradictions": [*state.contradictions, *result.conflicts],
        "evidence_gaps": [*state.evidence_gaps, *result.gaps],
        "llm_calls_used": int(state.llm_calls_used) + 1,
        "reason_codes": _new_codes(state, result.reason_codes),
        "status": "ok",
        "details": {
            "conflict_count": len(result.conflicts),
            "gap_count": len(result.gaps),
            "fact_count": len(state.extracted_facts),
        },
    }


@trace_node("compose_claims", graph_version=GRAPH_VERSION)
def compose_claims_node(
    state: AgenticRAGState, deps: AgentDeps
) -> dict[str, Any]:
    """Turn the extracted facts into short citable claims."""
    _check_budget(state, deps)
    spans = _load_evidence_spans(state, deps)
    result = draft_claims(
        ClaimDraftInput(
            question=state.question,
            facts=list(state.extracted_facts),
            spans=spans,
        )
    )
    return {
        "draft_claims": result.claims,
        "reason_codes": _new_codes(state, result.reason_codes),
        "status": "ok",
        "details": {"claim_count": len(result.claims)},
    }


@trace_node("verify", graph_version=GRAPH_VERSION)
def verify_node(state: AgenticRAGState, deps: AgentDeps) -> dict[str, Any]:
    """Check every claim against the evidence, deterministically and then once."""
    _check_budget(state, deps)
    spans = _load_evidence_spans(state, deps)
    result = verify_claims(
        VerificationInput(
            question=state.question,
            claims=list(state.draft_claims),
            spans=spans,
            authorized_document_ids=list(state.authorized_document_ids),
        ),
        settings=deps.settings,
        llm=deps.llm,
    )
    update: dict[str, Any] = {
        "verification_results": result.results,
        "verification_rounds": int(state.verification_rounds) + 1,
        "reason_codes": _new_codes(state, result.reason_codes),
        "status": "ok",
        "details": {
            "claim_count": len(state.draft_claims),
            "verified_count": len(
                [item for item in result.results if item.supported]
            ),
            "unsupported_count": len(result.material_failures),
        },
    }
    if result.recommended_next_search:
        update["recommended_next_search"] = result.recommended_next_search
    return update


@trace_node("synthesize", graph_version=GRAPH_VERSION)
def synthesize_node(state: AgenticRAGState, deps: AgentDeps) -> dict[str, Any]:
    """Assemble the answer from the verified claims only."""
    _check_budget(state, deps)
    spans = _load_evidence_spans(state, deps)
    result = synthesize_answer(
        SynthesisInput(
            question=state.question,
            claims=list(state.draft_claims),
            verification_results=list(state.verification_results),
            conflicts=list(state.contradictions),
            gaps=list(state.evidence_gaps),
            spans=spans,
        )
    )
    return {
        "final_answer": result.answer,
        "final_status": result.answer.final_status,
        "reason_codes": _new_codes(state, result.reason_codes),
        "status": "ok",
        "details": {
            "claim_count": len(result.answer.claims),
            "evidence_count": len(result.answer.evidence),
        },
    }


@trace_node("final_gate", graph_version=GRAPH_VERSION)
def final_gate_node(state: AgenticRAGState, deps: AgentDeps) -> dict[str, Any]:
    """Run the deterministic checks before the answer may be stored."""
    _check_budget(state, deps)
    spans = _load_evidence_spans(state, deps)
    answer = state.final_answer
    if answer is None:
        answer = build_insufficient_evidence_answer(state.question, [])
    findings = scan_spans_for_injection(spans)
    result = run_final_gate(
        FinalGateInput(
            question=state.question,
            answer=answer,
            claims=list(state.draft_claims),
            verification_results=list(state.verification_results),
            spans=spans,
            injection_findings=findings,
        )
    )
    reason_codes = list(result.reason_codes)
    if findings:
        reason_codes.append("INSTRUCTION_LIKE_TEXT_IN_EVIDENCE")
    return {
        "final_answer": result.answer,
        "final_status": result.answer.final_status,
        "reason_codes": _new_codes(state, reason_codes),
        "status": "ok" if result.passed else "flagged",
        "details": {
            "claim_count": len(result.answer.claims),
            "evidence_count": len(result.answer.evidence),
            "finding_count": len(findings),
        },
    }


def _after_authorize(state: AgenticRAGState) -> str:
    """Route to the planner, or stop the run when authorization failed."""
    if REASON_AUTHORIZATION_FAILED in state.reason_codes:
        return "end"
    return "route"


def _after_assess_coverage(state: AgenticRAGState) -> str:
    """Route on the deterministic quality gate."""
    if REASON_BUDGET_EXCEEDED in state.reason_codes:
        return "final_gate"
    decision = decide(state)
    if decision == GATE_ACCEPT:
        return "accept"
    if decision == GATE_RETRY_RETRIEVAL:
        return "retry"
    if decision == GATE_ASK_CLARIFICATION:
        return "final_gate"
    return "abstain"


def _after_extract_facts(state: AgenticRAGState) -> str:
    """Run the contradiction analysis only when it can matter."""
    if REASON_BUDGET_EXCEEDED in state.reason_codes:
        return "final_gate"
    documents = len(
        {str(item.document_id) for item in state.retrieval_results}
    )
    if state.task_type == "compare_versions" or documents > 1:
        return "conflicts"
    return "claims"


def _after_verify(state: AgenticRAGState) -> str:
    """Send a material failure back for one targeted retrieval, or finish."""
    if REASON_BUDGET_EXCEEDED in state.reason_codes:
        return "final_gate"
    failures = [
        result
        for result in state.verification_results
        if result.material and not result.supported
    ]
    if not failures:
        return "synthesize"
    if state.verification_rounds >= 2:
        return "synthesize"
    if state.retries_used >= state.max_retrieval_retries:
        return "synthesize"
    return "revise"


def _guarded(target: str) -> Callable[[AgenticRAGState], str]:
    """Route to a target unless the budget is exhausted."""

    def edge(state: AgenticRAGState) -> str:
        if REASON_BUDGET_EXCEEDED in state.reason_codes:
            return "final_gate"
        return target

    return edge


def build_graph(deps: AgentDeps) -> Any:
    """Build the compiled agentic graph."""
    graph = StateGraph(AgenticRAGState)
    graph.add_node("authorize", lambda state: authorize_node(state, deps))
    graph.add_node("route", lambda state: route_node(state, deps))
    graph.add_node("build_plan", lambda state: plan_node(state, deps))
    graph.add_node("retrieve", lambda state: retrieve_node(state, deps))
    graph.add_node(
        "assess_coverage", lambda state: assess_coverage_node(state, deps)
    )
    graph.add_node(
        "rewrite_query", lambda state: rewrite_query_node(state, deps)
    )
    graph.add_node("mark_gap", lambda state: mark_gap_node(state, deps))
    graph.add_node(
        "extract_facts", lambda state: extract_facts_node(state, deps)
    )
    graph.add_node(
        "detect_conflicts", lambda state: detect_conflicts_node(state, deps)
    )
    graph.add_node(
        "compose_claims", lambda state: compose_claims_node(state, deps)
    )
    graph.add_node("verify", lambda state: verify_node(state, deps))
    graph.add_node("synthesize", lambda state: synthesize_node(state, deps))
    graph.add_node("final_gate", lambda state: final_gate_node(state, deps))

    graph.add_edge(START, "authorize")
    graph.add_conditional_edges(
        "authorize",
        _after_authorize,
        {"route": "route", "end": END},
    )
    graph.add_conditional_edges(
        "route",
        _guarded("build_plan"),
        {"build_plan": "build_plan", "final_gate": "final_gate"},
    )
    graph.add_conditional_edges(
        "build_plan",
        _guarded("retrieve"),
        {"retrieve": "retrieve", "final_gate": "final_gate"},
    )
    graph.add_conditional_edges(
        "retrieve",
        _guarded("assess_coverage"),
        {"assess_coverage": "assess_coverage", "final_gate": "final_gate"},
    )
    graph.add_conditional_edges(
        "assess_coverage",
        _after_assess_coverage,
        {
            "accept": "extract_facts",
            "retry": "rewrite_query",
            "abstain": "mark_gap",
            "final_gate": "final_gate",
        },
    )
    graph.add_conditional_edges(
        "rewrite_query",
        _guarded("retrieve"),
        {"retrieve": "retrieve", "final_gate": "final_gate"},
    )
    graph.add_edge("mark_gap", "synthesize")
    graph.add_conditional_edges(
        "extract_facts",
        _after_extract_facts,
        {
            "conflicts": "detect_conflicts",
            "claims": "compose_claims",
            "final_gate": "final_gate",
        },
    )
    graph.add_edge("detect_conflicts", "compose_claims")
    graph.add_edge("compose_claims", "verify")
    graph.add_conditional_edges(
        "verify",
        _after_verify,
        {
            "revise": "rewrite_query",
            "synthesize": "synthesize",
            "final_gate": "final_gate",
        },
    )
    graph.add_conditional_edges(
        "synthesize",
        _guarded("final_gate"),
        {"final_gate": "final_gate"},
    )
    graph.add_edge("final_gate", END)
    return graph.compile()


@dataclass(frozen=True, slots=True)
class AgenticResult:
    """The result of one agentic run."""

    state: AgenticRAGState
    answer: Any
    graph_version: str = GRAPH_VERSION

    def trace_summary(self) -> AgentTraceSummary:
        """Return the short trace summary of the run.

        The plan and the queries are the questions the workflow asked of the
        search index. They are model-authored strings, never document text, so
        the "how this was found" panel stays safe to show.
        """
        queries: list[str] = []
        for subquestion in self.state.plan:
            text = subquestion.text.strip()
            if text and text not in queries:
                queries.append(text)
        return AgentTraceSummary(
            graph_version=self.graph_version,
            node_path=[event.node for event in self.state.trace],
            plan=[
                TracePlanStep(
                    text=subquestion.text.strip(),
                    purpose=subquestion.purpose,
                    status=subquestion.status,
                )
                for subquestion in self.state.plan
            ],
            queries=queries,
            steps=len(self.state.trace),
            llm_calls=self.state.llm_calls_used,
            tokens_used=self.state.tokens_used,
            retries_used=self.state.retries_used,
            duration_ms=sum(event.duration_ms for event in self.state.trace),
            final_status=self.state.final_status,
            reason_codes=list(self.state.reason_codes),
        )


def run_agentic_rag(
    client: Client,
    case_id: UUID | str,
    user: CurrentUser,
    question: str,
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embedder: TextEmbedder | None = None,
    index: Any | None = None,
    request_id: str | None = None,
) -> AgenticResult:
    """Run the agentic workflow for one question.

    Args:
        client: Service-role Supabase client.
        case_id: Case to research; it must belong to ``user``.
        user: Authenticated user the request is executed for.
        question: The question to answer.
        settings: Optional validated settings.
        llm: Optional LLM client override, mainly for tests.
        embedder: Optional embedder override, mainly for tests.
        index: Optional Pinecone index override, mainly for tests.
        request_id: Optional request id for the trace.

    Returns:
        The final state, the answer, and the graph version.
    """
    resolved = settings or get_settings()
    budget = Budget.from_settings(resolved)
    deps = AgentDeps(
        client=client,
        user=user,
        settings=resolved,
        llm=llm or LLMClient(settings=resolved),
        budget=budget,
        embedder=embedder,
        index=index,
    )
    state = AgenticRAGState(
        request_id=request_id,
        user_id=user.id,
        case_id=UUID(str(case_id)),
        question=question,
        deadline=budget.deadline,
        max_retrieval_retries=resolved.agent_max_retrieval_retries,
    )
    graph = build_graph(deps)
    try:
        result = graph.invoke(
            state,
            config={"recursion_limit": resolved.agent_max_steps + 20},
        )
    except BudgetExceeded as exc:
        logger.warning(
            "agentic run exceeded its budget",
            extra={"error_code": exc.reason_code, "limit": exc.limit},
        )
        result = {
            "reason_codes": [exc.reason_code],
            "final_status": "insufficient_evidence",
            "final_answer": build_insufficient_evidence_answer(question, []),
        }
    final_state = AgenticRAGState.model_validate(dict(result))
    answer = final_state.final_answer or build_insufficient_evidence_answer(
        question, []
    )
    logger.info(
        "agentic run finished",
        extra={
            "graph_version": GRAPH_VERSION,
            "final_status": final_state.final_status,
            "steps": len(final_state.trace),
            "llm_calls": final_state.llm_calls_used,
            "tokens": final_state.tokens_used,
            "retries": final_state.retries_used,
        },
    )
    return AgenticResult(
        state=final_state,
        answer=answer,
        graph_version=GRAPH_VERSION,
    )
