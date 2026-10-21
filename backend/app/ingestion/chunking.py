"""Structure-aware, token-bounded chunking for legal documents.

Chunks are built from normalized page text and split preferentially at headings,
numbered clauses, paragraph boundaries, and table rows. Every chunk records its
page range, heading trail, page quality flags, a stable content hash, and a
source anchor suitable for citation.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any, Literal, Protocol, Sequence
from uuid import UUID

from app.core.config import Settings
from app.db.repos import ChunkCreate, PageCreate

CHUNKING_VERSION = "1.0.0"

BlockKind = Literal["heading", "clause", "paragraph", "table_row"]

_STRUCTURAL_HEADING = re.compile(
    r"^(?P<kind>ARTICLE|SECTION|CLAUSE|PART|SCHEDULE|EXHIBIT|ANNEX|APPENDIX)"
    r"\s+(?P<number>[IVXLCDM]+|\d{1,3}(?:\.\d{1,3})*)"
    r"(?:\s*[.:\-—–]\s*|\s+)?(?P<title>.*)$",
    re.IGNORECASE,
)
_NUMBERED_HEADING = re.compile(
    r"^(?P<number>\d{1,3}(?:\.\d{1,3})*)\.?\s+(?P<title>.+)$"
)
_CLAUSE_START = re.compile(
    r"^(?:\d{1,3}(?:\.\d{1,3})*\.?|\([A-Za-z0-9]{1,4}\))\s+\S"
)
_CLAUSE_MARKER = re.compile(
    r"^(?P<marker>\d{1,3}(?:\.\d{1,3})*\.?|\([A-Za-z0-9]{1,4}\))\s+(?=\S)"
)
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")


class ChunkingError(RuntimeError):
    """A safe chunking failure with a stable, persistable error code."""

    def __init__(self, error_code: str) -> None:
        """Create a chunking error without including document text."""
        super().__init__(error_code)
        self.error_code = error_code


class TokenCounter(Protocol):
    """Minimal tokenizer interface needed by the chunker."""

    def count(self, text: str) -> int:
        """Return the embedding-tokenizer token count for ``text``."""
        ...


class EmbeddingTokenCounter:
    """Token counter backed by the configured embedding model tokenizer."""

    def __init__(self, tokenizer: Any) -> None:
        """Wrap a Hugging Face tokenizer."""
        self._tokenizer = tokenizer
        # Counting is not model inference. Remove the model's 512-token warning
        # so long paragraphs can be measured without noisy log output.
        try:
            self._tokenizer.model_max_length = int(1e12)
        except (AttributeError, TypeError, ValueError):
            pass

    def count(self, text: str) -> int:
        """Count embedding tokens without adding model special tokens."""
        if not text:
            return 0
        return len(self._tokenizer.encode(text, add_special_tokens=False))


def load_token_counter(settings: Settings) -> TokenCounter:
    """Load the configured embedding tokenizer lazily.

    Loading is deferred so the API and non-chunking jobs do not pay the model
    startup cost. Failures are reduced to a safe error code.
    """
    try:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(
            settings.embedding_model,
            cache_dir=str(settings.model_cache_dir),
            local_files_only=settings.chunk_tokenizer_local_files_only,
        )
    except Exception as exc:
        raise ChunkingError("TOKENIZER_UNAVAILABLE") from exc
    return EmbeddingTokenCounter(tokenizer)


@dataclass(frozen=True, slots=True)
class SourceBlock:
    """One structure-aware unit of text before token packing."""

    text: str
    page_number: int
    section_path: tuple[str, ...]
    kind: BlockKind
    token_count: int

    @property
    def protected(self) -> bool:
        """Return whether the block must not be split by the packer."""
        return self.kind in {"heading", "table_row"}


def _token_count(counter: TokenCounter, text: str) -> int:
    """Return a positive token count for non-empty text."""
    return max(1, counter.count(text))


def _is_table_row(line: str) -> bool:
    """Detect simple pipe-, tab-, or clearly column-spaced table rows.

    Multiple-space detection is intentionally conservative: a long sentence
    with extra spacing must not turn into one oversized protected row. Pages
    flagged ``possible_table`` by quality assessment are handled separately.
    """
    stripped = line.strip()
    if "|" in stripped or "\t" in line:
        return True
    if stripped.endswith((".", ";")):
        return False
    if len(stripped) > 300:
        return False
    columns = [part for part in re.split(r"\s{2,}", stripped) if part.strip()]
    return len(columns) >= 3


def _is_heading_title(title: str) -> bool:
    """Require a short title-like or uppercase heading tail."""
    cleaned = title.strip().strip("—-–: ")
    if not cleaned or len(cleaned) > 100:
        return False
    words = cleaned.split()
    if not words or len(words) > 12:
        return False
    if cleaned.endswith((".", ";", ",")):
        return False
    if cleaned.upper() == cleaned:
        return True
    alpha_words = [word for word in words if word and word[0].isalpha()]
    if not alpha_words:
        return False
    if len(words) <= 6 and all(word[0].isupper() for word in alpha_words):
        return True
    return False


def _is_all_caps_heading(line: str) -> bool:
    """Detect short, standalone all-caps headings."""
    if len(line) > 80:
        return False
    if line.endswith((".", ",", ";")):
        return False
    letters = [character for character in line if character.isalpha()]
    if len(letters) < 4 or any(character.islower() for character in letters):
        return False
    return len(line.split()) <= 12


def _detect_heading(line: str) -> tuple[int, str] | None:
    """Return a heading level and heading text, or ``None``."""
    structural = _STRUCTURAL_HEADING.match(line)
    if structural is not None:
        title = (structural.group("title") or "").strip()
        if not title or _is_heading_title(title):
            kind = structural.group("kind").upper()
            level = {"ARTICLE": 1, "PART": 1, "SCHEDULE": 1, "EXHIBIT": 1}.get(
                kind, 2
            )
            if kind in {"ANNEX", "APPENDIX"}:
                level = 1
            if kind == "CLAUSE":
                level = 3
            return level, line

    numbered = _NUMBERED_HEADING.match(line)
    if numbered is not None and _is_all_caps_heading(numbered.group("title")):
        number = numbered.group("number")
        return number.count(".") + 1, line

    if _is_all_caps_heading(line) and not _is_table_row(line):
        return (2 if _CLAUSE_START.match(line) else 1), line
    return None


def _update_heading_stack(
    stack: list[tuple[int, str]], heading: tuple[int, str]
) -> tuple[str, ...]:
    """Maintain a heading trail with simple numeric level precedence."""
    level, text = heading
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, text))
    return tuple(item[1] for item in stack)


def _make_block(
    text: str,
    *,
    page_number: int,
    section_path: tuple[str, ...],
    kind: BlockKind,
    counter: TokenCounter,
) -> SourceBlock | None:
    """Create a non-empty block with a token count."""
    cleaned = text.strip()
    if not cleaned:
        return None
    return SourceBlock(
        text=cleaned,
        page_number=page_number,
        section_path=section_path,
        kind=kind,
        token_count=_token_count(counter, cleaned),
    )


def build_blocks(
    pages: Sequence[PageCreate],
    *,
    token_counter: TokenCounter,
) -> list[SourceBlock]:
    """Convert normalized pages into heading-, clause-, and row-aware blocks."""
    blocks: list[SourceBlock] = []
    heading_stack: list[tuple[int, str]] = []

    for page in sorted(pages, key=lambda item: item.page_number):
        text = page.clean_text or ""
        if not text.strip():
            continue
        page_is_table = "possible_table" in page.quality_flags

        current_lines: list[str] = []
        current_kind: BlockKind = "paragraph"
        current_path: tuple[str, ...] = tuple(item[1] for item in heading_stack)

        def flush() -> None:
            """Flush the current paragraph or clause into a block."""
            nonlocal current_lines, current_kind, current_path
            if not current_lines:
                return
            block = _make_block(
                "\n".join(current_lines),
                page_number=page.page_number,
                section_path=current_path,
                kind=current_kind,
                counter=token_counter,
            )
            if block is not None:
                blocks.append(block)
            current_lines = []
            current_kind = "paragraph"
            current_path = tuple(item[1] for item in heading_stack)

        for raw_line in text.splitlines():
            line = raw_line.strip()
            if not line:
                flush()
                continue

            if _is_table_row(line):
                flush()
                block = _make_block(
                    line,
                    page_number=page.page_number,
                    section_path=tuple(item[1] for item in heading_stack),
                    kind="table_row",
                    counter=token_counter,
                )
                if block is not None:
                    blocks.append(block)
                continue

            heading = _detect_heading(line)
            if heading is not None:
                flush()
                current_path = _update_heading_stack(heading_stack, heading)
                block = _make_block(
                    line,
                    page_number=page.page_number,
                    section_path=current_path,
                    kind="heading",
                    counter=token_counter,
                )
                if block is not None:
                    blocks.append(block)
                continue

            if page_is_table:
                flush()
                block = _make_block(
                    line,
                    page_number=page.page_number,
                    section_path=tuple(item[1] for item in heading_stack),
                    kind="table_row",
                    counter=token_counter,
                )
                if block is not None:
                    blocks.append(block)
                continue

            if _CLAUSE_START.match(line):
                flush()
                current_kind = "clause"
                current_path = tuple(item[1] for item in heading_stack)
                current_lines = [line]
                continue

            if not current_lines:
                current_kind = "paragraph"
                current_path = tuple(item[1] for item in heading_stack)
            current_lines.append(line)

        flush()

    return blocks


def _split_words(text: str, limit: int, counter: TokenCounter) -> list[str]:
    """Split oversized text on word boundaries without exceeding the limit."""
    pieces: list[str] = []
    current: list[str] = []
    for word in text.split():
        candidate = " ".join([*current, word]) if current else word
        if current and counter.count(candidate) > limit:
            pieces.append(" ".join(current))
            current = [word]
            continue
        current.append(word)
    if current:
        pieces.append(" ".join(current))
    return pieces


def _split_block(
    block: SourceBlock,
    *,
    max_tokens: int,
    split_limit: int,
    counter: TokenCounter,
) -> list[SourceBlock]:
    """Split a long paragraph or clause at sentence, then word, boundaries.

    A numbered clause that fits within ``max_tokens`` is preserved even when it
    is larger than ``split_limit``. Other blocks are split below ``split_limit``
    so the packer can add overlap without exceeding the hard maximum.
    """
    if block.token_count <= split_limit or block.protected:
        return [block]
    if block.kind == "clause" and block.token_count <= max_tokens:
        return [block]

    text = block.text
    marker = ""
    limit = split_limit
    if block.kind == "clause":
        marker_match = _CLAUSE_MARKER.match(text)
        if marker_match is not None:
            marker = marker_match.group("marker")
            text = text[marker_match.end() :]
            # Leave room for the clause number on the first piece so "1."
            # never becomes an orphan chunk.
            limit = max(1, split_limit - _token_count(counter, f"{marker} "))

    pieces: list[SourceBlock] = []
    current: list[str] = []

    def flush() -> None:
        """Emit the current sentence group as a block."""
        nonlocal current
        if not current:
            return
        piece_text = " ".join(current).strip()
        pieces.append(
            SourceBlock(
                text=piece_text,
                page_number=block.page_number,
                section_path=block.section_path,
                kind=block.kind,
                token_count=_token_count(counter, piece_text),
            )
        )
        current = []

    for sentence in _SENTENCE_SPLIT.split(text):
        sentence = sentence.strip()
        if not sentence:
            continue
        sentence_tokens = _token_count(counter, sentence)
        if sentence_tokens > limit:
            flush()
            for word_piece in _split_words(sentence, limit, counter):
                pieces.append(
                    SourceBlock(
                        text=word_piece,
                        page_number=block.page_number,
                        section_path=block.section_path,
                        kind=block.kind,
                        token_count=_token_count(counter, word_piece),
                    )
                )
            continue
        candidate = " ".join([*current, sentence])
        if current and counter.count(candidate) > limit:
            flush()
        current.append(sentence)

    flush()
    if not pieces:
        return [block]
    if marker:
        first = pieces[0]
        piece_text = f"{marker} {first.text}"
        pieces[0] = SourceBlock(
            text=piece_text,
            page_number=first.page_number,
            section_path=first.section_path,
            kind=first.kind,
            token_count=_token_count(counter, piece_text),
        )
    return pieces


def _join_blocks(blocks: Sequence[SourceBlock]) -> str:
    """Join blocks with a paragraph separator."""
    return "\n\n".join(block.text.strip() for block in blocks if block.text.strip())


def _can_merge(
    previous: Sequence[SourceBlock],
    current: Sequence[SourceBlock],
    *,
    max_tokens: int,
    counter: TokenCounter,
) -> bool:
    """Allow merging small neighbors only within one section and page run."""
    if not previous or not current:
        return False
    if current[0].kind == "heading":
        return False
    if previous[-1].section_path != current[0].section_path:
        return False
    if current[0].page_number > previous[-1].page_number + 1:
        return False
    return counter.count(_join_blocks([*previous, *current])) <= max_tokens


def _pack_blocks(
    blocks: Sequence[SourceBlock],
    *,
    min_tokens: int,
    max_tokens: int,
    overlap_min_tokens: int,
    separator_tokens: int,
    counter: TokenCounter,
) -> list[list[SourceBlock]]:
    """Pack blocks into base chunks while preserving hard structural boundaries."""
    base_limit = max(1, max_tokens - overlap_min_tokens - separator_tokens)
    chunks: list[list[SourceBlock]] = []
    current: list[SourceBlock] = []
    current_tokens = 0

    def flush() -> None:
        """Emit the current base chunk."""
        nonlocal current, current_tokens
        if current:
            chunks.append(list(current))
            current = []
            current_tokens = 0

    for block in blocks:
        for piece in _split_block(
            block,
            max_tokens=max_tokens,
            split_limit=base_limit,
            counter=counter,
        ):
            if current and piece.page_number > current[-1].page_number + 1:
                flush()
            if piece.kind == "heading" and current:
                flush()

            extra_separator = separator_tokens if current else 0
            if current and current_tokens + piece.token_count + extra_separator > base_limit:
                flush()
                extra_separator = 0

            current.append(piece)
            current_tokens += piece.token_count + extra_separator

    flush()

    # Merge a small tail into its neighbor when both fit. This keeps the
    # configured minimum meaningful without crossing headings or page gaps.
    merged: list[list[SourceBlock]] = []
    for chunk in chunks:
        previous_is_small = bool(
            merged and counter.count(_join_blocks(merged[-1])) < min_tokens
        )
        current_is_small = counter.count(_join_blocks(chunk)) < min_tokens
        if (
            merged
            and (previous_is_small or current_is_small)
            and _can_merge(
                merged[-1], chunk, max_tokens=max_tokens, counter=counter
            )
        ):
            merged[-1] = [*merged[-1], *chunk]
            continue
        merged.append(list(chunk))
    return merged


def _tail_block(
    block: SourceBlock, limit: int, counter: TokenCounter
) -> SourceBlock | None:
    """Take a word-boundary tail from a block for overlap context."""
    if limit <= 0:
        return None
    if block.token_count <= limit:
        return block
    if block.protected:
        return None

    selected: list[str] = []
    for word in reversed(block.text.split()):
        candidate = " ".join([word, *selected])
        if counter.count(candidate) > limit:
            break
        selected.insert(0, word)
    if not selected:
        return None
    text = " ".join(selected)
    return SourceBlock(
        text=text,
        page_number=block.page_number,
        section_path=block.section_path,
        kind=block.kind,
        token_count=_token_count(counter, text),
    )


def _select_overlap_blocks(
    previous: Sequence[SourceBlock],
    *,
    page_number: int,
    target_tokens: int,
    minimum_tokens: int,
    counter: TokenCounter,
) -> list[SourceBlock]:
    """Select a contiguous same-page tail of the previous chunk for overlap."""
    selected: list[SourceBlock] = []
    total = 0
    for block in reversed(previous):
        if block.page_number != page_number:
            break
        if total >= target_tokens:
            break
        remaining = target_tokens - total
        if block.token_count <= remaining:
            selected.insert(0, block)
            total += block.token_count
            continue
        partial = _tail_block(block, remaining, counter)
        if partial is not None:
            selected.insert(0, partial)
            total += partial.token_count
        break
    if total < minimum_tokens:
        return []
    return selected


def _can_overlap(
    previous: Sequence[SourceBlock],
    current: Sequence[SourceBlock],
) -> bool:
    """Allow overlap only within one section and on the same page.

    Keeping overlap on one page makes citation page ranges unambiguous: a
    chunk's ``page_start`` is never pulled back to a previous page solely by
    duplicated context.
    """
    if not previous or not current:
        return False
    if current[0].kind == "heading":
        return False
    if previous[-1].section_path != current[0].section_path:
        return False
    return current[0].page_number == previous[-1].page_number


def _dedupe_flags(values: Sequence[Any]) -> list[Any]:
    """Preserve flag order while removing duplicates."""
    result: list[Any] = []
    for value in values:
        if value not in result:
            result.append(value)
    return result


def _finalize_chunks(
    chunks: Sequence[tuple[list[SourceBlock], str]],
    *,
    document_id: UUID | str,
    page_flags: dict[int, list[Any]],
    index_schema_version: int = 1,
) -> list[ChunkCreate]:
    """Build immutable chunk payloads with anchors, hashes, and page metadata."""
    payloads: list[ChunkCreate] = []
    for chunk_index, (blocks, text) in enumerate(chunks):
        page_start = min(block.page_number for block in blocks)
        page_end = max(block.page_number for block in blocks)
        flags: list[Any] = []
        for page_number in range(page_start, page_end + 1):
            flags.extend(page_flags.get(page_number, []))
        payloads.append(
            ChunkCreate(
                chunk_index=chunk_index,
                page_start=page_start,
                page_end=page_end,
                section_path=list(blocks[0].section_path),
                text_content=text,
                content_hash=hashlib.sha256(text.encode("utf-8")).hexdigest(),
                source_anchor=(
                    f"doc:{document_id}|p:{page_start}-{page_end}|c:{chunk_index}"
                ),
                index_schema_version=index_schema_version,
                index_status="pending",
                quality_flags=_dedupe_flags(flags),
            )
        )
    return payloads


def chunk_pages(
    pages: Sequence[PageCreate],
    *,
    document_id: UUID | str,
    token_counter: TokenCounter,
    min_tokens: int = 150,
    max_tokens: int = 220,
    overlap_min_tokens: int = 20,
    overlap_max_tokens: int = 40,
    index_schema_version: int = 1,
) -> list[ChunkCreate]:
    """Split normalized pages into overlapping, citation-ready chunks."""
    if min_tokens < 1:
        raise ValueError("min_tokens must be at least 1")
    if max_tokens < min_tokens:
        raise ValueError("max_tokens must be at least min_tokens")
    if overlap_min_tokens < 1 or overlap_max_tokens < overlap_min_tokens:
        raise ValueError("overlap token bounds are invalid")
    if overlap_max_tokens >= min_tokens:
        raise ValueError("overlap_max_tokens must be less than min_tokens")
    if max_tokens - overlap_min_tokens < min_tokens:
        raise ValueError(
            "max_tokens minus overlap_min_tokens must be at least min_tokens"
        )
    if index_schema_version < 1:
        raise ValueError("index_schema_version must be at least 1")

    ordered_pages = sorted(pages, key=lambda item: item.page_number)
    page_flags = {page.page_number: list(page.quality_flags) for page in ordered_pages}
    blocks = build_blocks(ordered_pages, token_counter=token_counter)
    if not blocks:
        return []

    separator_tokens = max(0, token_counter.count("\n\n"))
    base_chunks = _pack_blocks(
        blocks,
        min_tokens=min_tokens,
        max_tokens=max_tokens,
        overlap_min_tokens=overlap_min_tokens,
        separator_tokens=separator_tokens,
        counter=token_counter,
    )

    final: list[tuple[list[SourceBlock], str]] = []
    for index, base in enumerate(base_chunks):
        overlap: list[SourceBlock] = []
        if index > 0:
            previous = base_chunks[index - 1]
            if _can_overlap(previous, base):
                base_text = _join_blocks(base)
                base_tokens = token_counter.count(base_text)
                available = (
                    max_tokens - base_tokens - separator_tokens
                )
                if available >= overlap_min_tokens:
                    target = min(overlap_max_tokens, available)
                    overlap = _select_overlap_blocks(
                        previous,
                        page_number=base[0].page_number,
                        target_tokens=target,
                        minimum_tokens=overlap_min_tokens,
                        counter=token_counter,
                    )

        blocks = [*overlap, *base]
        text = _join_blocks(blocks)
        while overlap and token_counter.count(text) > max_tokens:
            overlap.pop(0)
            blocks = [*overlap, *base]
            text = _join_blocks(blocks)
        if text.strip():
            final.append((blocks, text.strip()))

    return _finalize_chunks(
        final,
        document_id=document_id,
        page_flags=page_flags,
        index_schema_version=index_schema_version,
    )
