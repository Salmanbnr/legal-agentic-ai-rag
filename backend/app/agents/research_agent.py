"""The research agent: bounded retrieval with one rewrite and one multi-hop.

The agent searches for every planned subquestion inside the authorized scope.
When the coverage is weak it spends exactly one model call rewriting the query
from what the earlier retrieval already showed, and when a retrieved passage
points at another clause or a date it spends one extra targeted search on that
reference. The agent returns evidence and gaps; it never concludes anything
and it can never widen the document scope.
"""

from __future__ import annotations

import logging
import re
from typing import Any, Literal, Sequence
from uuid import UUID

from pydantic import BaseModel, Field

from app.agents.llm import LLMClient, LLMError, load_prompt
from app.agents.safety import build_evidence_block
from app.agents.state import EvidenceGap, RetrievedEvidence, Subquestion
from app.auth.dependencies import CurrentUser
from app.core.config import Settings, get_settings
from app.embeddings.model import TextEmbedder
from app.retrieval.hybrid import hybrid_search
from app.retrieval.schemas import EvidenceSpan

logger = logging.getLogger(__name__)

REWRITE_PROMPT_NAME = "rewrite_query.txt"
RETRIEVAL_TOP_K = 8
MIN_SPANS_FOR_COVERAGE = 2

QueryStrategy = Literal["original", "rewrite", "multi_hop", "recommended"]

_CLAUSE_REFERENCE = re.compile(
    r"\b(?:subject to|pursuant to|under|per|referred? to in|see)\s+"
    r"(?:clause|sub-?clause|paragraph|section|article|schedule|annex)\s+"
    r"[0-9]{1,3}(?:\.[0-9]{1,3})*",
    re.IGNORECASE,
)
_DATE_REFERENCE = re.compile(
    r"\b(?:dated|agreement of|letter dated|deed dated)\s+"
    r"[0-9]{1,2}(?:st|nd|rd|th)?\s+[A-Za-z]+\s+[0-9]{4}",
    re.IGNORECASE,
)


class QueryAttempt(BaseModel):
    """One retrieval query the agent attempted."""

    subquestion_id: str
    query: str
    original_question: str = ""
    strategy: QueryStrategy = "original"
    rewrite_strategy: str = ""
    span_count: int = 0
    reason_codes: list[str] = Field(default_factory=list)


class CoverageAssessment(BaseModel):
    """How well the retrieved evidence covers one subquestion."""

    subquestion_id: str
    span_count: int = 0
    document_count: int = 0
    coverage_ok: bool = False


class ResearchOutput(BaseModel):
    """The result of one research pass."""

    evidence: list[EvidenceSpan] = Field(default_factory=list)
    queries_attempted: list[QueryAttempt] = Field(default_factory=list)
    coverage: list[CoverageAssessment] = Field(default_factory=list)
    gaps: list[EvidenceGap] = Field(default_factory=list)
    recommended_next_search: str | None = None
    reason_codes: list[str] = Field(default_factory=list)
    steps_used: int = 0
    llm_calls_used: int = 0


class QueryRewrite(BaseModel):
    """The planner's rewritten query and the strategy it used."""

    rewritten_query: str = ""
    strategy: str = ""
    terms_added: list[str] = Field(default_factory=list)


def load_rewrite_prompt() -> str:
    """Load the query rewriting system prompt."""
    return load_prompt(REWRITE_PROMPT_NAME)


def detect_cross_reference(text: str) -> str | None:
    """Return the clause or date reference inside a passage, if any."""
    clause = _CLAUSE_REFERENCE.search(text or "")
    if clause:
        return clause.group(0).strip()
    date = _DATE_REFERENCE.search(text or "")
    if date:
        return date.group(0).strip()
    return None


def _covers(spans: Sequence[EvidenceSpan]) -> bool:
    """Return whether the retrieved spans cover a subquestion."""
    return len(spans) >= MIN_SPANS_FOR_COVERAGE


def _merge(
    evidence: dict[str, EvidenceSpan], spans: Sequence[EvidenceSpan]
) -> int:
    """Merge spans into the evidence map, deduplicated by evidence id."""
    added = 0
    for span in spans:
        key = str(span.evidence_id)
        if key in evidence:
            continue
        evidence[key] = span
        added += 1
    return added


def _rewrite(
    question: str,
    subquestion: Subquestion,
    evidence: Sequence[EvidenceSpan],
    *,
    llm: LLMClient,
    settings: Settings,
) -> QueryRewrite | None:
    """Make one model call to rewrite a weak subquestion query.

    The rewrite may only use what the already retrieved evidence shows. The
    call fails closed: a bad or empty rewrite returns nothing and the agent
    keeps the original query.
    """
    if not evidence:
        return None
    blocks = [build_evidence_block(span) for span in evidence]
    user_prompt = (
        f"ORIGINAL QUESTION: {question}\n\n"
        f"SUBQUESTION: {subquestion.text}\n\n"
        "EVIDENCE ALREADY RETRIEVED (untrusted document text):\n"
        f"{chr(10).join(blocks)}\n"
    )
    try:
        rewrite = llm.invoke_structured(
            load_rewrite_prompt(),
            user_prompt,
            QueryRewrite,
            max_calls=settings.agent_max_llm_calls,
        )
    except LLMError as exc:
        logger.info(
            "query rewrite failed, the original query is kept",
            extra={"error_code": exc.error_code},
        )
        return None
    if not getattr(rewrite, "rewritten_query", "").strip():
        return None
    return rewrite


def research(
    client: Any,
    case_id: UUID | str,
    user: CurrentUser,
    question: str,
    subquestions: Sequence[Subquestion],
    authorized_document_ids: Sequence[UUID | str],
    *,
    settings: Settings | None = None,
    llm: LLMClient | None = None,
    embedder: TextEmbedder | None = None,
    index: Any | None = None,
    budget: Any | None = None,
    steps_used: int = 0,
    llm_calls_used: int = 0,
    retries_remaining: int = 0,
    prior_results: Sequence[RetrievedEvidence] = (),
    prior_spans: Sequence[EvidenceSpan] = (),
) -> ResearchOutput:
    """Retrieve evidence for every subquestion inside the authorized scope.

    Args:
        client: Service-role Supabase client.
        case_id: Case to research; it must belong to ``user``.
        user: Authenticated user the request is executed for.
        question: The original user question, kept for the audit trail.
        subquestions: The planned subquestions.
        authorized_document_ids: The documents the search may touch. The list
            is the only scope the agent has; it can never widen it.
        settings: Optional validated settings.
        llm: Optional LLM client override, mainly for tests.
        embedder: Optional embedder override, mainly for tests.
        index: Optional Pinecone index override, mainly for tests.
        budget: Optional budget checked before every search.
        steps_used: Steps already spent on this run.
        llm_calls_used: Model calls already spent on this run.
        retries_remaining: How many retrieval retries are still allowed. The
            query rewrite only happens while a retry is left.
        prior_results: Evidence references an earlier pass already found.
        prior_spans: The text of the earlier evidence, used to ground the
            query rewrite.

    Returns:
        The merged evidence, the queries attempted, the coverage assessment,
        the gaps, and a recommended next search.
    """
    resolved = settings or get_settings()
    scope = [str(value) for value in authorized_document_ids if str(value)]
    evidence: dict[str, EvidenceSpan] = {
        str(item.evidence_id): span for item, span in zip(prior_results, prior_spans)
    }
    attempts: list[QueryAttempt] = []
    coverage: list[CoverageAssessment] = []
    gaps: list[EvidenceGap] = []
    reason_codes: list[str] = []
    recommended: str | None = None
    steps = int(steps_used)
    calls = int(llm_calls_used)
    active_llm = llm or LLMClient(settings=resolved)

    def search(query: str) -> list[EvidenceSpan]:
        return hybrid_search(
            client,
            case_id,
            user,
            query,
            top_k=RETRIEVAL_TOP_K,
            document_ids=scope or None,
            settings=resolved,
            embedder=embedder,
            index=index,
        )

    def budget_allows(extra_steps: int = 1) -> bool:
        if budget is None:
            return True
        try:
            budget.check(
                steps=steps + extra_steps,
                llm_calls=calls,
                retries=0,
            )
        except Exception:
            return False
        return True

    for subquestion in subquestions:
        if subquestion.status == "retrieved":
            # This subquestion is already covered by an earlier pass.
            coverage.append(
                CoverageAssessment(
                    subquestion_id=subquestion.id,
                    span_count=0,
                    document_count=0,
                    coverage_ok=True,
                )
            )
            continue
        query = subquestion.text
        spans = search(query)
        steps += 1
        attempts.append(
            QueryAttempt(
                subquestion_id=subquestion.id,
                query=query,
                original_question=question,
                strategy="original",
                span_count=len(spans),
            )
        )
        _merge(evidence, spans)
        covered = _covers(spans)

        if not covered and retries_remaining > 0 and budget_allows(1):
            rewrite = _rewrite(
                question,
                subquestion,
                list(evidence.values()),
                llm=active_llm,
                settings=resolved,
            )
            if rewrite is not None:
                calls += 1
                rewritten = str(
                    getattr(rewrite, "rewritten_query", "") or ""
                ).strip()
                attempts.append(
                    QueryAttempt(
                        subquestion_id=subquestion.id,
                        query=rewritten,
                        original_question=question,
                        strategy="rewrite",
                        rewrite_strategy=str(
                            getattr(rewrite, "strategy", "") or ""
                        ),
                        span_count=0,
                    )
                )
                logger.info(
                    "query rewritten",
                    extra={
                        "subquestion_id": subquestion.id,
                        "rewrite_strategy": str(
                            getattr(rewrite, "strategy", "") or ""
                        ),
                        "original_span_count": len(spans),
                    },
                )
                if rewritten and rewritten != query:
                    rewritten_spans = search(rewritten)
                    steps += 1
                    attempts[-1].span_count = len(rewritten_spans)
                    _merge(evidence, rewritten_spans)
                    spans = [*spans, *rewritten_spans]
                    covered = _covers(spans)

        if resolved.agent_research_max_multi_hop > 0:
            reference = None
            for span in spans:
                reference = detect_cross_reference(span.quote_or_span)
                if reference:
                    break
            if reference and budget_allows(1):
                hop_query = f"{reference} {subquestion.text}".strip()
                hop_spans = search(hop_query)
                steps += 1
                attempts.append(
                    QueryAttempt(
                        subquestion_id=subquestion.id,
                        query=hop_query,
                        original_question=question,
                        strategy="multi_hop",
                        span_count=len(hop_spans),
                        reason_codes=["MULTI_HOP_FOLLOW_UP"],
                    )
                )
                _merge(evidence, hop_spans)
                logger.info(
                    "multi-hop follow-up searched",
                    extra={
                        "subquestion_id": subquestion.id,
                        "span_count": len(hop_spans),
                    },
                )

        document_ids = {str(span.document_id) for span in spans}
        coverage.append(
            CoverageAssessment(
                subquestion_id=subquestion.id,
                span_count=len(spans),
                document_count=len(document_ids),
                coverage_ok=covered,
            )
        )
        if not covered:
            gaps.append(
                EvidenceGap(
                    id=f"gap-{len(gaps) + 1}",
                    description=(
                        "The retrieved evidence does not cover this part of "
                        "the question. The material may exist in the case file "
                        "but was not returned by the search."
                    ),
                    subquestion_id=subquestion.id,
                    searched_document_ids=[
                        UUID(value)
                        for value in document_ids
                        if _is_uuid(value)
                    ],
                )
            )
            if recommended is None:
                recommended = subquestion.text

    if not evidence:
        reason_codes.append("NO_EVIDENCE_RETRIEVED")
    return ResearchOutput(
        evidence=list(evidence.values()),
        queries_attempted=attempts,
        coverage=coverage,
        gaps=gaps,
        recommended_next_search=recommended,
        reason_codes=reason_codes,
        steps_used=steps,
        llm_calls_used=calls,
    )


def _is_uuid(value: str) -> bool:
    """Return whether a string is a UUID."""
    try:
        UUID(value)
    except (TypeError, ValueError):
        return False
    return True
