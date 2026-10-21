"""Tests for dense retrieval from the Pinecone case namespace."""

from __future__ import annotations

from uuid import uuid4

import pytest
from fakes import FakeEmbedder, FakePineconeIndex

from app.core.config import Settings
from app.retrieval.dense import dense_search


def _stored(index: FakePineconeIndex, namespace: str, **documents: str) -> None:
    """Store one vector per document id in the fake index."""
    vectors = [
        (
            chunk_id,
            [1.0] * index.dimension,
            {"document_id": document_id, "case_id": namespace},
        )
        for chunk_id, document_id in documents.items()
    ]
    index.upsert(vectors=vectors, namespace=namespace)


def test_dense_search_returns_chunk_ids_with_scores() -> None:
    case_id = uuid4()
    document_id = str(uuid4())
    index = FakePineconeIndex()
    _stored(index, str(case_id), **{"chunk-1": document_id})
    index.set_score("chunk-1", 0.87)

    hits = dense_search(
        case_id,
        "who signed the agreement",
        document_ids=[document_id],
        top_k=5,
        settings=Settings(_env_file=None),
        embedder=FakeEmbedder(),
        index=index,
    )

    assert [hit.chunk_id for hit in hits] == ["chunk-1"]
    assert hits[0].score == pytest.approx(0.87)
    assert hits[0].document_id == document_id
    assert hits[0].rank == 1


def test_dense_search_filters_by_document_id() -> None:
    case_id = uuid4()
    allowed = str(uuid4())
    foreign = str(uuid4())
    index = FakePineconeIndex()
    _stored(
        index,
        str(case_id),
        **{"chunk-allowed": allowed, "chunk-foreign": foreign},
    )
    index.set_score("chunk-allowed", 0.9)
    index.set_score("chunk-foreign", 0.99)

    hits = dense_search(
        case_id,
        "question",
        document_ids=[allowed],
        top_k=10,
        settings=Settings(_env_file=None),
        embedder=FakeEmbedder(),
        index=index,
    )

    assert [hit.chunk_id for hit in hits] == ["chunk-allowed"]
    call = index.query_calls[0]
    assert call["namespace"] == str(case_id)
    assert call["filter"] == {"document_id": {"$in": [allowed]}}


def test_dense_search_never_queries_without_authorized_documents() -> None:
    index = FakePineconeIndex()

    hits = dense_search(
        uuid4(),
        "question",
        document_ids=[],
        top_k=10,
        settings=Settings(_env_file=None),
        embedder=FakeEmbedder(),
        index=index,
    )

    assert hits == []
    assert index.query_calls == []


def test_dense_search_ignores_an_empty_query() -> None:
    index = FakePineconeIndex()

    hits = dense_search(
        uuid4(),
        "   ",
        document_ids=[str(uuid4())],
        top_k=10,
        settings=Settings(_env_file=None),
        embedder=FakeEmbedder(),
        index=index,
    )

    assert hits == []
    assert index.query_calls == []


def test_dense_search_rejects_an_invalid_top_k() -> None:
    with pytest.raises(ValueError):
        dense_search(
            uuid4(),
            "question",
            document_ids=[str(uuid4())],
            top_k=0,
            settings=Settings(_env_file=None),
            embedder=FakeEmbedder(),
            index=FakePineconeIndex(),
        )


def test_dense_search_uses_the_configured_batch_size_for_the_query() -> None:
    embedder = FakeEmbedder()
    index = FakePineconeIndex()

    dense_search(
        uuid4(),
        "question",
        document_ids=[str(uuid4())],
        top_k=3,
        settings=Settings(_env_file=None),
        embedder=embedder,
        index=index,
    )

    assert embedder.texts == ["question"]
    assert len(index.query_calls[0]["vector"]) == embedder.dimension
