"""Tests for conservative, auditable page normalization."""

from __future__ import annotations

from app.db.repos import PageCreate
from app.ingestion.normalize import NORMALIZER_VERSION, normalize_pages


def _page(page_number: int, text: str | None) -> PageCreate:
    return PageCreate(
        page_number=page_number,
        extraction_method="native",
        raw_text=text,
    )


def test_nfkc_and_whitespace_are_normalized_without_touching_raw_text() -> None:
    raw = "  The \ufb01ling   was\n\n\nserved.  "
    page = _page(1, raw)

    result = normalize_pages([page])[0]

    assert page.raw_text == raw
    assert result.clean_text == "The filing was\n\nserved."
    steps = [entry["step"] for entry in result.transformations]
    assert steps == [
        "unicode_nfkc",
        "whitespace",
        "header_footer_removed",
        "hyphenation_joined",
        "result",
    ]
    assert result.transformations[0]["version"] == NORMALIZER_VERSION


def test_only_clearly_valid_end_of_line_hyphenation_is_joined() -> None:
    raw = (
        "The indemnifi-\n"
        "cation provision applies.\n\n"
        "The amount is 100-\n"
        "200.\n\n"
        "Article 5-\n"
        "6 also applies."
    )

    result = normalize_pages([_page(1, raw)])[0]

    assert result.clean_text is not None
    assert "indemnification provision" in result.clean_text
    assert "100-\n200." in result.clean_text
    assert "Article 5-\n6" in result.clean_text
    assert result.transformations[3]["joined_lines"] == 1


def test_hyphenated_compounds_are_not_closed_into_invalid_words() -> None:
    raw = (
        "A well-\n"
        "known rule.\n\n"
        "A self-\n"
        "employed person.\n\n"
        "A cross-\n"
        "border case.\n\n"
        "The non-\n"
        "compliance clause.\n\n"
        "They co-\n"
        "operate.\n\n"
        "The there-\n"
        "fore clause.\n\n"
        "The where-\n"
        "as clause.\n\n"
        "An in-\n"
        "house rule."
    )

    result = normalize_pages([_page(1, raw)])[0]

    assert result.clean_text is not None
    assert "well-\nknown" in result.clean_text
    assert "self-\nemployed" in result.clean_text
    assert "cross-\nborder" in result.clean_text
    assert "noncompliance clause" in result.clean_text
    assert "cooperate." in result.clean_text
    assert "therefore clause" in result.clean_text
    assert "whereas clause" in result.clean_text
    assert "in-\nhouse rule" in result.clean_text
    assert result.transformations[3]["joined_lines"] == 4


def test_common_suffix_hyphenations_are_joined() -> None:
    raw = (
        "We author-\n"
        "ize the request.\n\n"
        "We classi-\n"
        "fy the claim.\n\n"
        "The capital-\n"
        "ism clause.\n\n"
        "A social-\n"
        "ist policy.\n\n"
        "A real-\n"
        "istic result.\n\n"
        "The legal-\n"
        "ity question."
    )

    result = normalize_pages([_page(1, raw)])[0]

    assert result.clean_text is not None
    assert "authorize the request" in result.clean_text
    assert "classify the claim" in result.clean_text
    assert "capitalism clause" in result.clean_text
    assert "socialist policy" in result.clean_text
    assert "realistic result" in result.clean_text
    assert "legality question" in result.clean_text
    assert result.transformations[3]["joined_lines"] == 6


def test_repeated_header_is_removed_only_above_the_threshold() -> None:
    pages = [
        _page(index, f"CONFIDENTIAL\nBody text for page {index}.")
        for index in range(1, 6)
    ]

    results = normalize_pages(pages, header_footer_repeat_ratio=0.60)

    assert all(result.clean_text is not None for result in results)
    assert all("CONFIDENTIAL" not in (result.clean_text or "") for result in results)
    assert all(
        result.transformations[2]["removed_lines"] == 1 for result in results
    )


def test_header_at_exactly_sixty_percent_is_kept() -> None:
    pages = []
    for index in range(1, 6):
        header = "CONFIDENTIAL" if index <= 3 else "DIFFERENT"
        pages.append(_page(index, f"{header}\nBody text for page {index}."))

    results = normalize_pages(pages, header_footer_repeat_ratio=0.60)

    assert results[0].clean_text is not None
    assert "CONFIDENTIAL" in results[0].clean_text
    assert results[0].transformations[2]["removed_lines"] == 0


def test_numeric_and_section_number_only_lines_are_protected() -> None:
    pages = [
        _page(1, "12345\nSection 1\nBody one."),
        _page(2, "12345\nSection 1\nBody two."),
        _page(3, "12345\nSection 1\nBody three."),
    ]

    results = normalize_pages(pages)

    for result in results:
        assert result.clean_text is not None
        assert result.clean_text.startswith("12345\nSection 1")
        assert result.transformations[2]["removed_lines"] == 0


def test_empty_page_has_no_clean_text() -> None:
    result = normalize_pages([_page(1, "   \n\n")])[0]

    assert result.clean_text is None


def test_transform_log_contains_counts_not_document_text() -> None:
    secret = "Privileged settlement amount is 987654."
    result = normalize_pages([_page(1, secret)])[0]

    rendered = str(result.transformations)

    assert secret not in rendered
    assert "987654" not in rendered
