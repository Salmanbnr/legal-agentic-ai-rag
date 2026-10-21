"""Tests for the Pinecone namespace, dimension, upsert, and delete helpers."""

from __future__ import annotations

import sys
import types
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.indexing import pinecone_store
from app.indexing.pinecone_store import (
    PineconeDimensionError,
    PineconeError,
    create_index_if_missing,
    delete_document_vectors,
    describe_index_dimension,
    fetch_existing_ids,
    list_vector_ids,
    namespace_for_case,
    upsert_with_retry,
    verify_index_dimension,
)

from fakes import FakeControlPlane, FakeEmbedder, FakePineconeIndex


def test_namespace_uses_the_canonical_case_id() -> None:
    case_id = uuid4()

    assert namespace_for_case(case_id) == str(case_id)
    assert namespace_for_case(str(case_id)) == str(case_id)


def test_invalid_case_id_is_rejected() -> None:
    with pytest.raises(PineconeError) as error:
        namespace_for_case("not-a-uuid")

    assert error.value.error_code == "INVALID_CASE_ID"


def test_describe_index_dimension_reads_the_index() -> None:
    control = FakeControlPlane(dimension=384)

    assert describe_index_dimension(Settings(_env_file=None), client=control) == 384


def test_verify_index_dimension_accepts_a_matching_index() -> None:
    control = FakeControlPlane(dimension=384)

    dimension = verify_index_dimension(
        Settings(_env_file=None),
        embedder=FakeEmbedder(dimension=384),
        client=control,
    )

    assert dimension == 384


def test_verify_index_dimension_fails_clearly_on_mismatch() -> None:
    control = FakeControlPlane(dimension=768)

    with pytest.raises(PineconeDimensionError) as error:
        verify_index_dimension(
            Settings(_env_file=None),
            embedder=FakeEmbedder(dimension=384),
            client=control,
        )

    assert "384" in str(error.value)
    assert "768" in str(error.value)
    assert error.value.retryable is False


def test_create_index_creates_a_serverless_index_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    module = types.ModuleType("pinecone")
    module.ServerlessSpec = lambda **kwargs: kwargs  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "pinecone", module)
    control = FakeControlPlane(dimension=384, existing=False)

    status = create_index_if_missing(Settings(_env_file=None), client=control)

    assert status == "created"
    assert control.created
    assert control.created[0]["dimension"] == 384
    assert control.created[0]["metric"] == "cosine"


def test_create_index_is_a_no_op_when_the_index_exists() -> None:
    control = FakeControlPlane(dimension=384, existing=True)

    status = create_index_if_missing(Settings(_env_file=None), client=control)

    assert status == "exists"
    assert control.created == []


def test_upsert_with_retry_succeeds_after_a_transient_failure() -> None:
    index = FakePineconeIndex()
    index.upsert_failures = 2
    delays: list[float] = []

    upsert_with_retry(
        index,
        "case",
        [("id-1", [1.0] * 384, {})],
        settings=Settings(_env_file=None),
        sleep=delays.append,
        rng=lambda low, high: 1.0,
    )

    assert len(index.upsert_calls) == 3
    assert len(delays) == 2
    assert delays[1] > delays[0]
    assert list(index.vectors["case"]) == ["id-1"]


def test_upsert_with_retry_reports_a_safe_error_after_the_last_attempt() -> None:
    index = FakePineconeIndex()
    index.upsert_failures = 10

    with pytest.raises(PineconeError) as error:
        upsert_with_retry(
            index,
            "case",
            [("id-1", [1.0] * 384, {})],
            settings=Settings(_env_file=None),
            sleep=lambda seconds: None,
            rng=lambda low, high: 1.0,
        )

    assert error.value.error_code == "PINECONE_UPSERT_FAILED"
    assert len(index.upsert_calls) == 4


def test_fetch_existing_ids_returns_only_present_vectors() -> None:
    index = FakePineconeIndex()
    index.upsert(vectors=[("id-1", [1.0] * 384, {})], namespace="case")

    found = fetch_existing_ids(index, "case", ["id-1", "id-2"])

    assert found == {"id-1"}


def test_list_vector_ids_returns_the_namespace_contents() -> None:
    index = FakePineconeIndex()
    index.upsert(
        vectors=[("id-1", [1.0] * 384, {}), ("id-2", [1.0] * 384, {})],
        namespace="case",
    )

    assert list_vector_ids(index, "case") == {"id-1", "id-2"}


def test_delete_document_vectors_removes_only_the_given_ids() -> None:
    index = FakePineconeIndex()
    case_id = uuid4()
    namespace = namespace_for_case(case_id)
    index.upsert(
        vectors=[("id-1", [1.0] * 384, {}), ("id-2", [1.0] * 384, {})],
        namespace=namespace,
    )

    deleted = delete_document_vectors(
        case_id,
        uuid4(),
        chunk_ids=["id-1"],
        index=index,
        settings=Settings(_env_file=None),
    )

    assert deleted == 1
    assert list(index.vectors[namespace]) == ["id-2"]
    assert index.delete_calls[0][1] == namespace


def test_delete_document_vectors_without_ids_removes_nothing() -> None:
    index = FakePineconeIndex()

    deleted = delete_document_vectors(
        uuid4(),
        uuid4(),
        chunk_ids=[],
        index=index,
        settings=Settings(_env_file=None),
    )

    assert deleted == 0
    assert index.delete_calls == []


def test_store_module_exposes_a_namespace_helper() -> None:
    assert callable(pinecone_store.namespace_for_case)
