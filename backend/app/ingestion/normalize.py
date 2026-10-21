"""Conservative per-page text normalization for retrieval and chunking.

Normalization never changes ``raw_text``. It writes a separate ``clean_text``
and appends an auditable transformation trace. The rules deliberately avoid
rewriting numbers, dates, currency, and section numbers: only whitespace,
Unicode compatibility forms, clearly valid end-of-line hyphen joins, and
repeated header/footer lines are changed.
"""

from __future__ import annotations

import re
import unicodedata
from collections import Counter
from dataclasses import dataclass
from typing import Any, Sequence

from app.db.repos import PageCreate

NORMALIZER_VERSION = "1.0.0"

_HORIZONTAL_WHITESPACE = re.compile(r"\s+")
_BLANK_RUN = re.compile(r"\n{3,}")
_HYPHENATED_LINE = re.compile(r"(?P<prefix>.*?)(?P<word>[A-Za-z]{2,})-$")
_LOWERCASE_WORD_START = re.compile(r"^(?P<word>[a-z][A-Za-z]*)")
_STRUCTURAL_NUMBER_ONLY = re.compile(
    r"^\s*(?:Article|Section|Clause|Paragraph|Para\.?|§)"
    r"\s+[IVXLCDM\d]+(?:\.[\d]+)*[.:]?\s*$",
    re.IGNORECASE,
)

_COMMON_JOINED_WORDS = frozenset(
    {
        "abovementioned",
        "abovesaid",
        "aforementioned",
        "aforesaid",
        "antitrust",
        "beforesaid",
        "codefendant",
        "coexist",
        "cohabit",
        "coplaintiff",
        "counterclaim",
        "counterpart",
        "countersign",
        "coworker",
        "hereafter",
        "hereby",
        "herefrom",
        "hereinafter",
        "hereinbefore",
        "herein",
        "hereof",
        "hereto",
        "heretofore",
        "hereunder",
        "hereunto",
        "herewith",
        "herewithal",
        "indemnified",
        "indemnify",
        "indemnifying",
        "insured",
        "invoice",
        "multicolumn",
        "multidistrict",
        "multiemployer",
        "multiline",
        "multipage",
        "multiunit",
        "nevertheless",
        "noncompete",
        "nonetheless",
        "nonprofit",
        "nonresident",
        "notwithstanding",
        "preapprove",
        "preapproved",
        "prearrange",
        "prearranged",
        "preauthorize",
        "preauthorized",
        "preorder",
        "prepackage",
        "prepackaged",
        "prepaid",
        "prepay",
        "prequalified",
        "prequalify",
        "presale",
        "preselect",
        "preselected",
        "reentry",
        "reimburse",
        "reissue",
        "reopen",
        "subaccount",
        "subclause",
        "subcommittee",
        "subcontract",
        "subcontractor",
        "sublease",
        "sublicense",
        "subparagraph",
        "subpoena",
        "subscribe",
        "thereafter",
        "thereby",
        "therefrom",
        "therein",
        "thereinafter",
        "thereof",
        "thereto",
        "theretofore",
        "thereunder",
        "thereunto",
        "therewith",
        "therewithal",
        "therefore",
        "unexpired",
        "unfair",
        "unissued",
        "unjust",
        "unknown",
        "unpaid",
        "unsecured",
        "undersigned",
        "underwriter",
        "whereas",
        "whereafter",
        "whereby",
        "wherefrom",
        "wherein",
        "whereof",
        "whereto",
        "whereunder",
        "whereunto",
        "whereupon",
        "wherewith",
        "wherewithal",
    }
)
_SUFFIXES = (
    "ability",
    "able",
    "al",
    "ance",
    "ary",
    "ate",
    "ation",
    "cation",
    "ction",
    "ence",
    "ency",
    "ful",
    "fy",
    "hood",
    "ibility",
    "ible",
    "ic",
    "ically",
    "ication",
    "ing",
    "isation",
    "ise",
    "ised",
    "ising",
    "ism",
    "ist",
    "istic",
    "ition",
    "ity",
    "ive",
    "ization",
    "ize",
    "ized",
    "izing",
    "less",
    "ly",
    "ment",
    "ness",
    "ology",
    "ous",
    "ously",
    "ry",
    "ship",
    "sion",
    "tion",
    "ty",
    "ually",
    "ure",
    "ution",
)


@dataclass(frozen=True, slots=True)
class NormalizedPage:
    """The safe normalized text and transformation trace for one page."""

    page_number: int
    clean_text: str | None
    transformations: list[dict[str, Any]]


@dataclass(slots=True)
class _PageLines:
    """Intermediate line representation used for document-wide comparisons."""

    page: PageCreate
    raw_text: str
    lines: list[str]
    line_changes: int
    nfkc_changed: bool


def _nfkc(text: str) -> str:
    """Apply Unicode NFKC compatibility normalization."""
    return unicodedata.normalize("NFKC", text)


def _normalize_lines(text: str) -> tuple[list[str], int]:
    """Strip line edges and collapse horizontal whitespace inside each line."""
    changes = 0
    lines: list[str] = []
    normalized_text = (
        text.replace("\r\n", "\n")
        .replace("\r", "\n")
        .replace("\u2028", "\n")
        .replace("\u2029", "\n")
    )
    for raw_line in normalized_text.split("\n"):
        line = raw_line.strip()
        collapsed = _HORIZONTAL_WHITESPACE.sub(" ", line)
        if collapsed != raw_line:
            changes += 1
        lines.append(collapsed)
    return lines, changes


def _candidate_indices(lines: Sequence[str], max_lines: int) -> set[int]:
    """Return indexes of the first and last non-empty lines on a page."""
    non_empty = [index for index, line in enumerate(lines) if line]
    return set(non_empty[:max_lines] + non_empty[-max_lines:])


def _line_key(line: str) -> str:
    """Build a case-insensitive comparison key for a repeated line."""
    return _HORIZONTAL_WHITESPACE.sub(" ", line).strip().casefold()


def _is_protected_line(line: str) -> bool:
    """Protect numeric-only and section-number-only lines from removal."""
    if not any(character.isalpha() for character in line):
        return True
    return bool(_STRUCTURAL_NUMBER_ONLY.match(line))


def _has_vowel(value: str) -> bool:
    """Return whether a lowercase word contains a vowel or ``y``."""
    return any(character in "aeiouy" for character in value)


def _is_plausible_hyphen_join(first: str, second: str) -> bool:
    """Join only when a common suffix makes the result clearly valid.

    Prefix-only evidence is not enough: joining ``well-`` and ``known`` would
    create the invalid word ``wellknown``. A recognizable word ending is a
    stronger signal, and unresolved cases stay hyphenated.
    """
    if not first.isalpha() or not second.isalpha():
        return False
    if len(first) < 2 or len(second) < 2:
        return False
    first_lower = first.lower()
    second_lower = second.lower()
    if not _has_vowel(first_lower) or not _has_vowel(second_lower):
        return False
    if len(first_lower) + len(second_lower) < 5:
        return False
    joined = f"{first_lower}{second_lower}"
    if joined in _COMMON_JOINED_WORDS:
        return True
    return any(second_lower.endswith(suffix) for suffix in _SUFFIXES)


def _join_hyphenated_lines(lines: Sequence[str]) -> tuple[list[str], int]:
    """Join clearly valid end-of-line hyphenations, including long words."""
    joined: list[str] = []
    count = 0
    index = 0
    while index < len(lines):
        line = lines[index]
        # A word may be split over more than two lines. Keep consuming
        # adjacent continuations while the joined text still ends in a hyphen.
        while line.endswith("-") and index + 1 < len(lines):
            match = _HYPHENATED_LINE.match(line)
            next_match = _LOWERCASE_WORD_START.match(lines[index + 1])
            if match is None or next_match is None:
                break
            first = match.group("word")
            second = next_match.group("word")
            if not _is_plausible_hyphen_join(first, second):
                break
            prefix = line[: -len(first) - 1]
            line = f"{prefix}{first}{lines[index + 1]}"
            count += 1
            index += 1
        joined.append(line)
        index += 1
    return joined, count


def _result_transformations(
    page: _PageLines,
    *,
    clean_text: str | None,
    removed_lines: int,
    joined_lines: int,
    blank_runs_collapsed: int,
    candidate_lines: int,
    ratio_threshold: float,
) -> list[dict[str, Any]]:
    """Build a text-free audit trail for one page's normalization."""
    return [
        {
            "stage": "normalize",
            "step": "unicode_nfkc",
            "version": NORMALIZER_VERSION,
            "changed": page.nfkc_changed,
        },
        {
            "stage": "normalize",
            "step": "whitespace",
            "version": NORMALIZER_VERSION,
            "changed_lines": page.line_changes,
            "blank_runs_collapsed": blank_runs_collapsed,
        },
        {
            "stage": "normalize",
            "step": "header_footer_removed",
            "version": NORMALIZER_VERSION,
            "candidate_lines": candidate_lines,
            "removed_lines": removed_lines,
            "repeat_ratio_threshold": ratio_threshold,
        },
        {
            "stage": "normalize",
            "step": "hyphenation_joined",
            "version": NORMALIZER_VERSION,
            "joined_lines": joined_lines,
        },
        {
            "stage": "normalize",
            "step": "result",
            "version": NORMALIZER_VERSION,
            "characters": len(clean_text or ""),
            "raw_characters": len(page.raw_text),
        },
    ]


def normalize_pages(
    pages: Sequence[PageCreate],
    *,
    header_footer_repeat_ratio: float = 0.60,
    header_footer_max_lines: int = 3,
) -> list[NormalizedPage]:
    """Normalize every page and remove only document-wide repeated headers.

    A header/footer line is removed only when it appears in a boundary slot on
    strictly more than ``header_footer_repeat_ratio`` of the document's pages.
    Numeric-only and section-number-only lines are never removed.
    """
    if not 0.0 <= header_footer_repeat_ratio <= 1.0:
        raise ValueError("header_footer_repeat_ratio must be between 0 and 1")
    if header_footer_max_lines < 1:
        raise ValueError("header_footer_max_lines must be at least 1")

    page_lines: list[_PageLines] = []
    for page in pages:
        raw_text = page.raw_text
        if raw_text is None:
            continue
        nfkc_text = _nfkc(raw_text)
        lines, line_changes = _normalize_lines(nfkc_text)
        page_lines.append(
            _PageLines(
                page=page,
                raw_text=raw_text,
                lines=lines,
                line_changes=line_changes,
                nfkc_changed=nfkc_text != raw_text,
            )
        )

    counts: Counter[str] = Counter()
    candidates_by_page: dict[int, set[int]] = {}
    for item in page_lines:
        candidates = _candidate_indices(item.lines, header_footer_max_lines)
        candidates_by_page[id(item)] = candidates
        for index in candidates:
            line = item.lines[index]
            if line and not _is_protected_line(line):
                counts[_line_key(line)] += 1

    total_pages = len(page_lines)
    repeated_keys = {
        key
        for key, count in counts.items()
        if total_pages >= 2
        and count >= 2
        and count > header_footer_repeat_ratio * total_pages
    }

    results: list[NormalizedPage] = []
    for item in page_lines:
        candidates = candidates_by_page[id(item)]
        filtered: list[str] = []
        removed = 0
        for index, line in enumerate(item.lines):
            if (
                index in candidates
                and line
                and not _is_protected_line(line)
                and _line_key(line) in repeated_keys
            ):
                removed += 1
                continue
            filtered.append(line)

        joined_lines, hyphen_joins = _join_hyphenated_lines(filtered)
        joined_text = "\n".join(joined_lines)
        blank_runs = len(_BLANK_RUN.findall(joined_text))
        clean_text = _BLANK_RUN.sub("\n\n", joined_text).strip() or None
        transformations = _result_transformations(
            item,
            clean_text=clean_text,
            removed_lines=removed,
            joined_lines=hyphen_joins,
            blank_runs_collapsed=blank_runs,
            candidate_lines=len(candidates),
            ratio_threshold=header_footer_repeat_ratio,
        )
        results.append(
            NormalizedPage(
                page_number=item.page.page_number,
                clean_text=clean_text,
                transformations=transformations,
            )
        )

    return results
