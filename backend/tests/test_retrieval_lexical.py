"""Tests for lexical retrieval through the Postgres search function."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fakes import FakeSupabaseClient

from app.retrieval.lexical import LEXICAL_FUNCTION, lexical_search


def test_lexical_search_calls_the_search_function_with_filters() -> None:
    case_id = uuid4()
    document_id = str(uuid4())
    client = FakeSupabaseClient(
        [
            {"id": "chunk-1", "document_id": document_id, "rank": 0.42},
        ]
    )

    hits = lexical_search(
        client,
        case_id,
        "damages awarded",
        document_ids=[document_id],
        top_k=5,
    )

    name, params = client.rpc_calls[0]
    assert name == LEXICAL_FUNCTION
    assert params["p_case_id"] == str(case_id)
    assert params["p_query"] == "damages awarded"
    assert params["p_document_ids"] == [document_id]
    assert params["p_limit"] == 5

    assert [hit.chunk_id for hit in hits] == ["chunk-1"]
    assert hits[0].score == pytest.approx(0.42)
    assert hits[0].document_id == document_id
    assert hits[0].rank == 1


def test_lexical_search_does_not_call_the_function_without_documents() -> None:
    client = FakeSupabaseClient()

    hits = lexical_search(
        client,
        uuid4(),
        "question",
        document_ids=[],
        top_k=5,
    )

    assert hits == []
    assert client.rpc_calls == []


def test_lexical_search_ignores_an_empty_query() -> None:
    client = FakeSupabaseClient()

    hits = lexical_search(
        client,
        uuid4(),
        "  ",
        document_ids=[str(uuid4())],
        top_k=5,
    )

    assert hits == []
    assert client.rpc_calls == []


def test_lexical_search_skips_rows_without_an_id() -> None:
    client = FakeSupabaseClient(
        [
            {"id": None, "document_id": "doc", "rank": 1.0},
            {"id": "chunk-1", "document_id": "doc", "rank": 0.5},
        ]
    )

    hits = lexical_search(
        client,
        uuid4(),
        "question",
        document_ids=[str(uuid4())],
        top_k=5,
    )

    assert [hit.chunk_id for hit in hits] == ["chunk-1"]


def test_lexical_search_rejects_an_invalid_top_k() -> None:
    with pytest.raises(ValueError):
        lexical_search(
            FakeSupabaseClient(),
            uuid4(),
            "question",
            document_ids=[str(uuid4())],
            top_k=0,
        )
