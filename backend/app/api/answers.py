"""Ask a question about one owned case and read back its stored answers.

The baseline mode retrieves evidence, answers only from that evidence, and
validates every citation before the answer is stored. The agentic mode runs the
bounded workflow. Both modes are stored under the same payload shape, and the
history listing never carries document text.
"""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Annotated, Any, Literal, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request, status
from pydantic import BaseModel, ConfigDict, Field, ValidationError
from supabase import Client

from app.agents.answer_synthesizer import manifest_for
from app.agents.graph import GRAPH_VERSION, run_agentic_rag
from app.agents.llm import LLMClient, LLMError, LLMUsage
from app.agents.state import AgentTraceSummary
from app.auth.access import require_case_access
from app.auth.dependencies import CurrentUser, get_current_user
from app.core.config import Settings, get_settings
from app.db.client import get_db_client
from app.db.repos import (
    Answer,
    AnswerCreate,
    CaseAccessDenied,
    DocumentAccessDenied,
    QualityStatus,
    RecordNotFound,
    get_chunks_by_ids,
    get_document_for_user,
    get_pages_for_documents,
    insert_answer,
    list_answers,
    write_audit_event,
)
from app.retrieval.answer_schema import AnswerResponse, FinalStatus
from app.retrieval.baseline_rag import answer_question
from app.retrieval.dense import RetrievalError
from app.retrieval.evidence_meta import collect_quality_flags, page_signals
from app.retrieval.lexical import LexicalError

router = APIRouter(tags=["answers"])

logger = logging.getLogger(__name__)

BASELINE_GRAPH_VERSION = "baseline-rag-1"

#: The reserved payload key that keeps the agent workflow summary beside the
#: answer. Reading it never changes the shape of the answer itself.
ANSWER_TRACE_PAYLOAD_KEY = "trace"

#: A page read below this OCR confidence is flagged in the citation panel.
LOW_OCR_CONFIDENCE = 75.0

_KNOWN_FINAL_STATUSES: frozenset[str] = frozenset(
    {"answered", "partial", "insufficient_evidence", "conflicting_evidence"}
)


class AskRequest(BaseModel):
    """Fields accepted by the ask endpoint."""

    model_config = ConfigDict(extra="forbid")

    question: str = Field(min_length=1, max_length=20000)
    mode: Literal["baseline", "agentic"] = "baseline"


class LLMUsageSummary(BaseModel):
    """Safe call and token counters for one request."""

    calls: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @classmethod
    def from_usage(cls, usage: LLMUsage) -> "LLMUsageSummary":
        """Build a summary from the client counters."""
        return cls(
            calls=usage.calls,
            retries=usage.retries,
            prompt_tokens=usage.prompt_tokens,
            completion_tokens=usage.completion_tokens,
            total_tokens=usage.total_tokens,
        )


class AskResponse(BaseModel):
    """The stored answer with its evidence and request metadata."""

    answer_id: UUID
    case_id: UUID
    question: str
    answer: AnswerResponse
    request_id: str | None = None
    mode: Literal["baseline", "agentic"] = "baseline"
    usage: LLMUsageSummary = Field(default_factory=LLMUsageSummary)
    trace: AgentTraceSummary | None = None


class AnswerListItem(BaseModel):
    """One stored answer, safe to list in the chat history.

    The citation metadata (document title, page range, extraction method,
    confidence, quality flags) stays, but ``quote_or_span`` is blank on every
    span. The exact evidence text is fetched one citation at a time from the
    evidence endpoint, so listing a history never dumps the case file.
    """

    answer_id: UUID
    case_id: UUID
    question: str
    mode: Literal["baseline", "agentic"]
    final_status: FinalStatus
    created_at: datetime
    answer: AnswerResponse
    trace: AgentTraceSummary | None = None


class AnswersPage(BaseModel):
    """One page of stored answers, newest first."""

    items: list[AnswerListItem] = Field(default_factory=list)
    limit: int
    offset: int
    has_more: bool = False


class EvidenceDetail(BaseModel):
    """The exact stored evidence behind one citation.

    This is the only response that carries document text, and it carries one
    chunk at a time. Every field comes from Postgres, never from the model.
    """

    evidence_id: UUID
    case_id: UUID
    document_id: UUID
    document_title: str
    document_mime: str | None = None
    page_start: int | None = None
    page_end: int | None = None
    section_path: list[str] = Field(default_factory=list)
    text: str
    extraction_method: str | None = None
    ocr_confidence: float | None = None
    quality_status: QualityStatus
    quality_flags: list[Any] = Field(default_factory=list)
    quality_warnings: list[str] = Field(default_factory=list)
    source_anchor: str | None = None


def mode_for(graph_version: str | None) -> Literal["baseline", "agentic"]:
    """Return the answering mode a stored graph version belongs to."""
    return "agentic" if (graph_version or "").startswith("agentic") else "baseline"


def stored_payload(
    answer: AnswerResponse, trace: AgentTraceSummary | None
) -> dict[str, Any]:
    """Build the stored payload, keeping the workflow summary beside the answer."""
    payload = answer.model_dump(mode="json")
    payload[ANSWER_TRACE_PAYLOAD_KEY] = (
        trace.model_dump(mode="json") if trace is not None else None
    )
    return payload


def public_answer(payload: dict[str, Any]) -> AnswerResponse:
    """Return a stored answer without its workflow summary or evidence text."""
    clean = {
        key: value
        for key, value in (payload or {}).items()
        if key != ANSWER_TRACE_PAYLOAD_KEY
    }
    answer = AnswerResponse.model_validate(clean)
    return answer.model_copy(
        update={
            "evidence": [
                span.model_copy(update={"quote_or_span": ""})
                for span in answer.evidence
            ]
        }
    )


def trace_from_payload(payload: dict[str, Any]) -> AgentTraceSummary | None:
    """Return the stored workflow summary, if the row has a readable one."""
    raw = (payload or {}).get(ANSWER_TRACE_PAYLOAD_KEY)
    if not isinstance(raw, dict):
        return None
    try:
        return AgentTraceSummary.model_validate(raw)
    except ValidationError:
        return None


def _status_value(value: str | None) -> FinalStatus:
    """Return a known final status, defaulting to an honest abstention."""
    if value in _KNOWN_FINAL_STATUSES:
        return cast(FinalStatus, value)
    return "insufficient_evidence"


def _list_item(row: Answer) -> AnswerListItem:
    """Build one history item from a stored answer row."""
    payload = row.answer_payload or {}
    return AnswerListItem(
        answer_id=row.id,
        case_id=row.case_id,
        question=row.question,
        mode=mode_for(row.graph_version),
        final_status=_status_value(row.final_status),
        created_at=row.created_at,
        answer=public_answer(payload),
        trace=trace_from_payload(payload),
    )


def _flag_codes(flags: list[Any]) -> set[str]:
    """Return the flag names from a mixed list of strings and dictionaries."""
    codes: set[str] = set()
    for flag in flags:
        if isinstance(flag, str):
            codes.add(flag)
        elif isinstance(flag, dict) and isinstance(flag.get("code"), str):
            codes.add(flag["code"])
    return codes


def quality_warnings(
    flags: list[Any],
    confidence: float | None,
    quality_status: QualityStatus,
) -> list[str]:
    """Describe the quality of one citation in plain language.

    The wording is fixed, so a warning never carries document text and never
    invents a reason that the stored signals do not support.
    """
    codes = _flag_codes(flags)
    warnings: list[str] = []
    if "low_ocr_confidence" in codes or (
        confidence is not None and confidence < LOW_OCR_CONFIDENCE
    ):
        warnings.append(
            "Low OCR confidence on this page: a name, date, or amount may "
            "have been misread."
        )
    if "empty_page" in codes:
        warnings.append("No text could be read from this page.")
    if "possible_handwriting" in codes:
        warnings.append(
            "This page may contain handwriting, which is read less reliably."
        )
    if "possible_table" in codes:
        warnings.append(
            "This page may contain a table, which can be read out of order."
        )
    if quality_status == "needs_review":
        warnings.append("This document needs review before relying on it.")
    elif quality_status == "failed":
        warnings.append("This document could not be read reliably.")
    return warnings


def _request_id(request: Request) -> str | None:
    """Read the middleware-assigned request id."""
    return getattr(request.state, "request_id", None)


def _evidence_manifest(answer: AnswerResponse) -> list[dict[str, object]]:
    """Build the stored manifest: evidence ids and pages only."""
    return manifest_for(answer)


@router.post("/cases/{case_id}/ask", response_model=AskResponse)
def ask_case(
    case_id: UUID,
    payload: AskRequest,
    request: Request,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
    settings: Annotated[Settings, Depends(get_settings)],
) -> AskResponse:
    """Answer a question from the evidence of one owned case."""
    require_case_access(case_id, user, client=client)
    question = payload.question.strip()
    if not question:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question must not be empty",
        )
    if len(question) > settings.answer_max_question_length:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail="Question is too long",
        )

    request_id = _request_id(request)
    llm = LLMClient(settings=settings)
    try:
        if payload.mode == "agentic":
            result = run_agentic_rag(
                client,
                case_id,
                user,
                question,
                settings=settings,
                llm=llm,
                request_id=request_id,
            )
            answer = result.answer
            usage = result.state.llm_calls_used
            trace = result.trace_summary()
        else:
            baseline = answer_question(
                client,
                case_id,
                user,
                question,
                settings=settings,
                llm=llm,
            )
            answer = baseline.answer
            usage = baseline.usage.calls
            trace = None
    except (RetrievalError, LexicalError) as exc:
        logger.warning(
            "retrieval failed error_class=%s",
            type(exc).__name__,
            extra={
                "request_id": request_id,
                "case_id": str(case_id),
                "mode": payload.mode,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Search is temporarily unavailable",
        ) from None
    except LLMError as exc:
        logger.error(
            "answer generation failed error_code=%s retryable=%s",
            exc.error_code,
            exc.retryable,
            extra={
                "request_id": request_id,
                "case_id": str(case_id),
                "mode": payload.mode,
            },
        )
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="The answer service is temporarily unavailable",
        ) from None

    stored = insert_answer(
        client,
        user.id,
        case_id,
        AnswerCreate(
            question=question,
            answer_payload=stored_payload(answer, trace),
            evidence_refs=_evidence_manifest(answer),
            graph_version=(
                GRAPH_VERSION
                if payload.mode == "agentic"
                else BASELINE_GRAPH_VERSION
            ),
            final_status=answer.final_status,
        ),
    )
    write_audit_event(
        client,
        user.id,
        event_type="case.question_answered",
        case_id=case_id,
        target_type="answer",
        target_id=str(stored.id),
        request_id=request_id,
        safe_metadata={
            "final_status": answer.final_status,
            "claim_count": len(answer.claims),
            "evidence_count": len(answer.evidence),
            "mode": payload.mode,
            "llm_calls": usage,
            "graph_version": (
                GRAPH_VERSION if payload.mode == "agentic" else BASELINE_GRAPH_VERSION
            ),
            "reason_codes": list(answer.reason_codes)[:10],
        },
    )
    logger.info(
        "answer stored",
        extra={
            "request_id": request_id,
            "case_id": str(case_id),
            "mode": payload.mode,
            "final_status": answer.final_status,
            "claim_count": len(answer.claims),
            "evidence_count": len(answer.evidence),
            "llm_calls": usage,
        },
    )
    return AskResponse(
        answer_id=stored.id,
        case_id=case_id,
        question=question,
        answer=answer,
        request_id=request_id,
        mode=payload.mode,
        usage=LLMUsageSummary.from_usage(llm.usage),
        trace=trace,
    )


@router.get("/cases/{case_id}/answers", response_model=AnswersPage)
def list_case_answers(
    case_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
    limit: Annotated[int, Query(ge=1, le=100)] = 20,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> AnswersPage:
    """List the stored answers of one owned case, newest first.

    The listing carries the answer sections and the citation metadata only. The
    evidence text is fetched from the evidence endpoint, one citation at a time.
    """
    require_case_access(case_id, user, client=client)
    rows = list_answers(
        client, user.id, case_id, limit=limit + 1, offset=offset
    )
    has_more = len(rows) > limit
    return AnswersPage(
        items=[_list_item(row) for row in rows[:limit]],
        limit=limit,
        offset=offset,
        has_more=has_more,
    )


@router.get(
    "/cases/{case_id}/evidence/{evidence_id}", response_model=EvidenceDetail
)
def get_case_evidence(
    case_id: UUID,
    evidence_id: UUID,
    user: Annotated[CurrentUser, Depends(get_current_user)],
    client: Annotated[Client, Depends(get_db_client)],
) -> EvidenceDetail:
    """Return the stored text and page metadata behind one citation.

    Ownership is checked twice: the case must belong to the user, and the chunk
    must be part of that case. A citation that no longer resolves is reported as
    not found, because the document may have been deleted.
    """
    require_case_access(case_id, user, client=client)
    chunks = get_chunks_by_ids(client, user.id, case_id, [evidence_id])
    if not chunks:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence not found",
        )
    chunk = chunks[0]
    try:
        document = get_document_for_user(client, user.id, chunk.document_id)
    except (DocumentAccessDenied, RecordNotFound):
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence not found",
        ) from None
    try:
        pages = get_pages_for_documents(
            client, user.id, case_id, [chunk.document_id]
        )
    except CaseAccessDenied:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Evidence not found",
        ) from None

    extraction_method, confidence = page_signals(pages, chunk)
    flags = collect_quality_flags(chunk, pages)
    return EvidenceDetail(
        evidence_id=chunk.id,
        case_id=chunk.case_id,
        document_id=chunk.document_id,
        document_title=document.original_filename,
        document_mime=document.detected_mime,
        page_start=chunk.page_start,
        page_end=chunk.page_end,
        section_path=list(chunk.section_path),
        text=chunk.text_content,
        extraction_method=extraction_method,
        ocr_confidence=confidence,
        quality_status=document.quality_status,
        quality_flags=flags,
        quality_warnings=quality_warnings(
            flags, confidence, document.quality_status
        ),
        source_anchor=chunk.source_anchor,
    )
