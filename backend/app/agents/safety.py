"""Shared evidence delimiters and the prompt injection scanner.

Retrieved document text is untrusted data. It is always wrapped in explicit
delimiters before a model sees it, and the scanner flags sentences that look
like instructions. Flagged text is reported in the trace and never obeyed. No
agent in this package is given a shell, a code runner, a network tool, or a
direct database tool.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Sequence
from uuid import UUID

from app.retrieval.schemas import EvidenceSpan

EVIDENCE_OPEN = "<<<EVIDENCE"
EVIDENCE_CLOSE = "<<<END EVIDENCE"

SUSPICIOUS_PHRASES: tuple[str, ...] = (
    "ignore previous instructions",
    "ignore all previous instructions",
    "ignore the above instructions",
    "ignore your instructions",
    "disregard the instructions",
    "disregard previous instructions",
    "disregard all previous",
    "you are now",
    "act as a",
    "pretend to be",
    "system prompt",
    "reveal your",
    "print your instructions",
    "show me your prompt",
    "api key",
    "access token",
    "password",
    "secret key",
    "another case",
    "other case",
    "different case",
    "the other client",
    "send the",
    "email the",
    "forward the",
    "transfer the",
    "exfiltrate",
    "delete the document",
    "delete all documents",
    "run the following",
    "execute the following",
    "ignore the file",
)

_NORMALIZE = re.compile(r"\s+")
_WHITESPACE = re.compile(r"\s+")


def normalize_text(text: str) -> str:
    """Normalize whitespace and case for a text comparison."""
    return _NORMALIZE.sub(" ", text or "").strip().lower()


def scan_text_for_injection(text: str) -> list[str]:
    """Return the suspicious phrases found in a piece of text.

    The scanner only reports. It never changes the text and never follows an
    instruction found inside it.
    """
    if not text:
        return []
    lowered = text.lower()
    found: list[str] = []
    for phrase in SUSPICIOUS_PHRASES:
        if phrase in lowered and phrase not in found:
            found.append(phrase)
    return found


@dataclass(frozen=True, slots=True)
class InjectionFinding:
    """One piece of evidence that looks like it carries an instruction."""

    evidence_id: UUID | None = None
    document_id: UUID | None = None
    phrases: list[str] = field(default_factory=list)


def scan_spans_for_injection(
    spans: Sequence[EvidenceSpan],
) -> list[InjectionFinding]:
    """Return the evidence that looks like it carries an instruction."""
    findings: list[InjectionFinding] = []
    for span in spans:
        phrases = scan_text_for_injection(span.quote_or_span)
        if not phrases:
            continue
        findings.append(
            InjectionFinding(
                evidence_id=span.evidence_id,
                document_id=span.document_id,
                phrases=phrases,
            )
        )
    return findings


def build_evidence_block(span: EvidenceSpan) -> str:
    """Wrap one span in the shared evidence delimiters.

    The header carries the evidence id, the page, the document, and the
    quality flags so the model can cite the passage and the backend can check
    the citation afterwards.
    """
    flags = ", ".join(str(flag) for flag in span.quality_flags) or "none"
    page = (
        f"pages {span.page_start}-{span.page_end}"
        if span.page_start is not None and span.page_end != span.page_start
        else f"page {span.page_start}"
    )
    return (
        f"{EVIDENCE_OPEN} id={span.evidence_id} {page} "
        f"document={span.title!r} quality_flags={flags} "
        f"confidence={span.ocr_confidence}>>>\n"
        f"{span.quote_or_span}\n"
        f"{EVIDENCE_CLOSE} id={span.evidence_id}>>>"
    )


def wrap_evidence(spans: Iterable[EvidenceSpan]) -> str:
    """Wrap several spans in the shared evidence delimiters."""
    return "\n\n".join(build_evidence_block(span) for span in spans)


def contains_instruction_like_text(text: str) -> bool:
    """Return whether a text looks like it carries an embedded instruction."""
    return bool(scan_text_for_injection(text))


def safe_metadata(details: dict[str, Any]) -> dict[str, Any]:
    """Keep only the counts and identifiers that are safe to log."""
    allowed = {
        "span_count",
        "document_count",
        "claim_count",
        "fact_count",
        "conflict_count",
        "gap_count",
        "finding_count",
        "verified_count",
        "unsupported_count",
        "coverage_ok",
    }
    return {key: value for key, value in details.items() if key in allowed}
