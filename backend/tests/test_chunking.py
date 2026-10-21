"""Tests for structure-aware token chunking."""

from __future__ import annotations

from uuid import uuid4

import pytest

from app.db.repos import PageCreate
from app.ingestion.chunking import EmbeddingTokenCounter, chunk_pages


class WordCounter:
    """Deterministic token counter for tests."""

    def count(self, text: str) -> int:
        return len(text.split())


def _page(
    page_number: int,
    text: str | None,
    *,
    flags: list[str] | None = None,
) -> PageCreate:
    return PageCreate(
        page_number=page_number,
        extraction_method="native",
        raw_text=text,
        clean_text=text,
        quality_flags=list(flags or []),
    )


def _words(prefix: str, count: int) -> str:
    return " ".join(f"{prefix}{index}" for index in range(count))


def _sentence(prefix: str, word_count: int = 50) -> str:
    return f"{prefix}start {_words(prefix, word_count - 2)}."


def test_numbered_clause_stays_together() -> None:
    clause = "1. " + _words("clauseword", 205) + "."
    page = _page(1, "SECTION 1. DEFINITIONS\n\n" + clause)

    chunks = chunk_pages(
        [page],
        document_id=uuid4(),
        token_counter=WordCounter(),
    )

    assert any(clause in chunk.text_content for chunk in chunks)


def test_heading_is_a_hard_boundary_between_sections() -> None:
    page = _page(
        1,
        "SECTION 1. FIRST PART\n\n"
        + _words("alpha", 60)
        + "\n\nSECTION 2. SECOND PART\n\n"
        + _words("beta", 60),
    )

    chunks = chunk_pages(
        [page],
        document_id=uuid4(),
        token_counter=WordCounter(),
    )

    assert len(chunks) >= 2
    assert not any(
        "alpha0" in chunk.text_content and "beta0" in chunk.text_content
        for chunk in chunks
    )
    assert any(
        chunk.section_path == ["SECTION 1. FIRST PART"] for chunk in chunks
    )
    assert any(
        chunk.section_path == ["SECTION 2. SECOND PART"] for chunk in chunks
    )


def test_page_ranges_are_recorded_for_each_chunk() -> None:
    pages = [
        _page(1, _words("pageone", 190)),
        _page(2, _words("pagetwo", 190)),
    ]

    chunks = chunk_pages(
        pages,
        document_id=uuid4(),
        token_counter=WordCounter(),
    )

    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [
        (1, 1),
        (2, 2),
    ]


def test_overlap_does_not_pull_page_start_back_to_an_earlier_page() -> None:
    pages = [
        _page(1, _words("one", 160)),
        _page(
            2,
            _words("twoa", 30) + "\n\n" + _words("twob", 180),
        ),
    ]

    chunks = chunk_pages(
        pages,
        document_id=uuid4(),
        token_counter=WordCounter(),
    )

    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [
        (1, 2),
        (2, 2),
    ]
    assert "twoa0" in chunks[1].text_content
    assert "one0" not in chunks[1].text_content


def test_hashes_are_stable_between_runs() -> None:
    page = _page(1, _words("stable", 180))
    document_id = uuid4()

    first = chunk_pages(
        [page], document_id=document_id, token_counter=WordCounter()
    )
    second = chunk_pages(
        [page], document_id=document_id, token_counter=WordCounter()
    )

    assert [chunk.content_hash for chunk in first] == [
        chunk.content_hash for chunk in second
    ]
    assert [chunk.text_content for chunk in first] == [
        chunk.text_content for chunk in second
    ]


def test_short_heading_is_merged_with_a_fitting_clause() -> None:
    page = _page(
        1,
        "SECTION 1. DEFINITIONS\n\n1. " + _words("clause", 200) + ".",
    )

    chunks = chunk_pages(
        [page],
        document_id=uuid4(),
        token_counter=WordCounter(),
    )

    assert len(chunks) == 1
    assert chunks[0].text_content.startswith("SECTION 1. DEFINITIONS")
    assert 150 <= len(chunks[0].text_content.split()) <= 220


def test_small_adjacent_chunks_are_merged_when_they_fit() -> None:
    page = _page(1, _words("first", 100) + "\n\n" + _words("second", 100))

    chunks = chunk_pages(
        [page],
        document_id=uuid4(),
        token_counter=WordCounter(),
    )

    assert len(chunks) == 1
    assert 150 <= len(chunks[0].text_content.split()) <= 220


def test_empty_pages_are_skipped() -> None:
    pages = [
        _page(1, _words("first", 190)),
        _page(2, ""),
        _page(3, _words("third", 190)),
    ]

    chunks = chunk_pages(
        pages,
        document_id=uuid4(),
        token_counter=WordCounter(),
    )

    assert [(chunk.page_start, chunk.page_end) for chunk in chunks] == [
        (1, 1),
        (3, 3),
    ]


def test_extra_spaces_do_not_turn_a_long_paragraph_into_one_row() -> None:
    text = (("The parties agree to this provision.  " * 40)).strip()
    page = _page(1, text)

    chunks = chunk_pages(
        [page],
        document_id=uuid4(),
        token_counter=WordCounter(),
    )

    assert len(chunks) > 1
    assert all(len(chunk.text_content.split()) <= 220 for chunk in chunks)


def test_page_marked_as_possible_table_keeps_each_row_whole() -> None:
    page = _page(
        1,
        "Particulars | Amount\nFirst row | 100\nSecond row | 200",
        flags=["possible_table"],
    )

    chunks = chunk_pages(
        [page],
        document_id=uuid4(),
        token_counter=WordCounter(),
    )

    assert len(chunks) == 1
    assert "First row | 100" in chunks[0].text_content
    assert "Second row | 200" in chunks[0].text_content


def test_table_row_is_kept_together_even_when_oversized() -> None:
    row = "| " + " | ".join(f"cell{index}" for index in range(250)) + " |"
    page = _page(1, row)

    chunks = chunk_pages(
        [page],
        document_id=uuid4(),
        token_counter=WordCounter(),
    )

    assert len(chunks) == 1
    assert chunks[0].text_content == row


def test_page_quality_flags_are_carried_to_the_chunk() -> None:
    pages = [
        _page(1, _words("first", 50), flags=["low_ocr_confidence"]),
        _page(2, _words("second", 50), flags=["possible_table"]),
    ]

    chunks = chunk_pages(
        pages,
        document_id=uuid4(),
        token_counter=WordCounter(),
    )

    assert len(chunks) == 1
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 2
    assert chunks[0].quality_flags == ["low_ocr_confidence", "possible_table"]


def test_chunks_have_token_bounded_overlap() -> None:
    text = " ".join(_sentence(f"sentence{index}", 50) for index in range(8))
    page = _page(1, text)

    chunks = chunk_pages(
        [page],
        document_id=uuid4(),
        token_counter=WordCounter(),
    )

    assert len(chunks) >= 2
    token_counts = [len(chunk.text_content.split()) for chunk in chunks]
    assert all(150 <= count <= 220 for count in token_counts)

    first_words = chunks[0].text_content.split()
    second_words = chunks[1].text_content.split()
    overlap = 0
    for size in range(min(len(first_words), len(second_words)), 0, -1):
        if first_words[-size:] == second_words[:size]:
            overlap = size
            break
    assert 20 <= overlap <= 40


def test_embedding_tokenizer_bounds_real_subword_chunk_sizes() -> None:
    transformers = pytest.importorskip("transformers")
    try:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            "sentence-transformers/all-MiniLM-L6-v2",
            local_files_only=True,
        )
    except OSError:
        pytest.skip("embedding tokenizer is not present in the local cache")

    counter = EmbeddingTokenCounter(tokenizer)
    text = " ".join(
        _sentence(f"tokens{index}", 120) for index in range(12)
    )
    page = _page(1, text)

    chunks = chunk_pages(
        [page],
        document_id=uuid4(),
        token_counter=counter,
    )

    assert len(chunks) > 1
    assert all(counter.count(chunk.text_content) <= 220 for chunk in chunks)
    assert chunks[0].page_start == 1
    assert chunks[0].page_end == 1


def test_source_anchor_and_pending_status_are_set() -> None:
    document_id = uuid4()
    page = _page(1, _words("anchor", 160))

    chunks = chunk_pages(
        [page],
        document_id=document_id,
        token_counter=WordCounter(),
    )

    assert chunks[0].source_anchor == (
        f"doc:{document_id}|p:1-1|c:0"
    )
    assert chunks[0].index_status == "pending"
    assert chunks[0].content_hash == __import__("hashlib").sha256(
        chunks[0].text_content.encode("utf-8")
    ).hexdigest()
