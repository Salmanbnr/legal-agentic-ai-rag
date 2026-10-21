"""Deterministic fakes for the embedding model, Pinecone index, and Supabase.

The fakes mirror only the small surface the backend uses, so tests can assert
batching, idempotency, authorization filters, and count mismatches without
network access.
"""

from __future__ import annotations

import math
from types import SimpleNamespace
from typing import Any, Iterable

import pytest


class FakeEmbedder:
    """Deterministic normalized embeddings with a fixed dimension."""

    def __init__(self, dimension: int = 384) -> None:
        """Create an embedder that reports the given output dimension."""
        self.dimension = dimension
        self.batches: list[list[str]] = []
        self.texts: list[str] = []

    def embed_texts(self, texts: Iterable[str]) -> list[list[float]]:
        """Return one deterministic unit vector per input text."""
        batch = [str(text) for text in texts]
        self.batches.append(batch)
        self.texts.extend(batch)
        return [self._vector(text) for text in batch]

    def embed_query(self, text: str) -> list[float]:
        """Return one deterministic unit vector for a query."""
        self.texts.append(str(text))
        return self._vector(str(text))

    def _vector(self, text: str) -> list[float]:
        values = [0.0] * self.dimension
        for position, byte in enumerate(text.encode("utf-8")):
            values[position % self.dimension] += (byte % 23) + 1
        norm = math.sqrt(sum(value * value for value in values))
        if norm == 0:
            return values
        return [value / norm for value in values]


class FakePineconeIndex:
    """In-memory stand-in for the Pinecone index data plane."""

    def __init__(self, dimension: int = 384) -> None:
        """Create an empty fake index."""
        self.dimension = dimension
        self.vectors: dict[str, dict[str, dict[str, Any]]] = {}
        self.upsert_calls: list[tuple[list[Any], str]] = []
        self.delete_calls: list[tuple[list[str], str]] = []
        self.fetch_calls: list[tuple[list[str], str]] = []
        self.list_calls: list[str] = []
        self.query_calls: list[dict[str, Any]] = []
        self.scores: dict[str, float] = {}
        self.upsert_failures = 0
        self.drop_upsert_ids: set[str] = set()

    def set_score(self, vector_id: str, score: float) -> None:
        """Set the similarity score a query returns for one vector id."""
        self.scores[str(vector_id)] = score

    def upsert(self, *, vectors: list[Any], namespace: str) -> None:
        """Store vectors, optionally failing to exercise the retry path."""
        self.upsert_calls.append((list(vectors), namespace))
        if self.upsert_failures > 0:
            self.upsert_failures -= 1
            raise RuntimeError("fake upsert failure")
        stored = self.vectors.setdefault(namespace, {})
        for item in vectors:
            if isinstance(item, dict):
                vector_id = str(item["id"])
                values = list(item["values"])
                metadata = dict(item.get("metadata") or {})
            else:
                vector_id, values, metadata = item
                vector_id = str(vector_id)
                values = list(values)
                metadata = dict(metadata or {})
            if len(values) != self.dimension:
                raise AssertionError("vector has the wrong dimension")
            if vector_id in self.drop_upsert_ids:
                continue
            stored[vector_id] = {"values": values, "metadata": metadata}

    def query(
        self,
        *,
        vector: list[float],
        top_k: int,
        namespace: str | None = None,
        filter: dict[str, Any] | None = None,
        include_metadata: bool = False,
        include_values: bool = False,
    ) -> Any:
        """Return the best matching vectors inside one namespace."""
        self.query_calls.append(
            {
                "vector": list(vector),
                "top_k": top_k,
                "namespace": namespace,
                "filter": filter,
            }
        )
        stored = self.vectors.get(namespace or "", {})
        allowed = self._allowed_documents(filter)
        matches = []
        for vector_id, entry in stored.items():
            metadata = entry.get("metadata") or {}
            if allowed is not None:
                document_id = str(metadata.get("document_id"))
                if document_id not in allowed:
                    continue
            score = self.scores.get(
                vector_id, self._similarity(vector, entry.get("values") or [])
            )
            matches.append(
                SimpleNamespace(
                    id=vector_id,
                    score=score,
                    metadata=metadata,
                )
            )
        matches.sort(key=lambda match: (-match.score, match.id))
        return SimpleNamespace(matches=matches[:top_k])

    @staticmethod
    def _similarity(query: list[float], values: list[float]) -> float:
        """Return the cosine similarity of two vectors."""
        if not query or not values:
            return 0.0
        size = min(len(query), len(values))
        dot = sum(query[index] * values[index] for index in range(size))
        query_norm = math.sqrt(sum(value * value for value in query[:size]))
        value_norm = math.sqrt(sum(value * value for value in values[:size]))
        if query_norm == 0 or value_norm == 0:
            return 0.0
        return dot / (query_norm * value_norm)

    @staticmethod
    def _allowed_documents(
        filter: dict[str, Any] | None,
    ) -> set[str] | None:
        """Return the document ids a metadata filter allows."""
        if not filter:
            return None
        condition = filter.get("document_id")
        if condition is None:
            return None
        if isinstance(condition, dict):
            values = condition.get("$in")
            if values is None:
                return None
            return {str(value) for value in values}
        return {str(condition)}

    def fetch(self, *, ids: list[str], namespace: str) -> dict[str, Any]:
        """Return the stored vectors for the requested ids."""
        self.fetch_calls.append((list(ids), namespace))
        stored = self.vectors.get(namespace, {})
        return {
            "vectors": {
                str(vector_id): stored[str(vector_id)]
                for vector_id in ids
                if str(vector_id) in stored
            }
        }

    def delete(self, *, ids: list[str], namespace: str) -> None:
        """Remove the requested ids from the namespace."""
        self.delete_calls.append((list(ids), namespace))
        stored = self.vectors.setdefault(namespace, {})
        for vector_id in ids:
            stored.pop(str(vector_id), None)

    def list(self, *, namespace: str, limit: int | None = None) -> list[list[str]]:
        """Return the namespace ids as a single page."""
        self.list_calls.append(namespace)
        stored = self.vectors.get(namespace, {})
        ids = list(stored)
        if limit is not None:
            ids = ids[:limit]
        return [ids]


@pytest.fixture()
def fake_embedder() -> FakeEmbedder:
    """Provide a deterministic embedder for indexing tests."""
    return FakeEmbedder()


class FakeControlPlane:
    """Minimal Pinecone control-plane stand-in."""

    def __init__(self, dimension: int = 384, *, existing: bool = False) -> None:
        """Create a control plane with a known dimension."""
        self.dimension = dimension
        self.existing = existing
        self.created: list[dict[str, object]] = []

    def has_index(self, name: str) -> bool:
        """Return whether the index already exists."""
        return self.existing

    def describe_index(self, name: str) -> dict[str, object]:
        """Return a description of the configured index."""
        return {"name": name, "dimension": self.dimension}

    def create_index(self, **kwargs: object) -> dict[str, object]:
        """Record an index creation request."""
        self.created.append(kwargs)
        self.existing = True
        return {"name": kwargs.get("name")}


@pytest.fixture()
def fake_index() -> FakePineconeIndex:
    """Provide an in-memory Pinecone index for indexing tests."""
    return FakePineconeIndex()


class FakeRpcQuery:
    """Minimal stand-in for a Supabase RPC query builder."""

    def __init__(self, rows: list[dict[str, Any]]) -> None:
        """Store the rows the RPC should return."""
        self._rows = rows

    def execute(self) -> Any:
        """Return the stored rows as a response payload."""
        return SimpleNamespace(data=list(self._rows))


class FakeSupabaseClient:
    """Service-client stand-in that answers a single RPC function."""

    def __init__(self, rows: list[dict[str, Any]] | None = None) -> None:
        """Create a client that returns the given RPC rows."""
        self.rows = list(rows or [])
        self.rpc_calls: list[tuple[str, dict[str, Any]]] = []

    def rpc(self, name: str, params: dict[str, Any]) -> FakeRpcQuery:
        """Record the RPC call and return its rows."""
        self.rpc_calls.append((name, dict(params)))
        return FakeRpcQuery(self.rows)


class FakeLLMResponse:
    """Chat completion stand-in with token counters."""

    def __init__(
        self,
        content: str = "",
        *,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        total_tokens: int | None = None,
    ) -> None:
        """Create a response with the given text and token usage."""
        self.content = content
        self.usage_metadata = {
            "input_tokens": prompt_tokens,
            "output_tokens": completion_tokens,
            "total_tokens": (
                total_tokens
                if total_tokens is not None
                else prompt_tokens + completion_tokens
            ),
        }


class FakeLLMRunnable:
    """Runnable stand-in that returns a prepared response."""

    def __init__(
        self,
        response: Any,
        calls: list[Any],
        failures: int = 0,
    ) -> None:
        """Create a runnable with a response and a failure budget."""
        self.response = response
        self.calls = calls
        self.failures = failures

    def invoke(self, messages: Any) -> Any:
        """Record the messages and return the prepared response."""
        self.calls.append(messages)
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("fake llm failure")
        return self.response


class FakeChat:
    """Chat model stand-in for the LLM wrapper."""

    def __init__(self, response: Any = None, failures: int = 0) -> None:
        """Create a chat model that returns the prepared response."""
        self.response = response
        self.failures = failures
        self.calls: list[Any] = []
        self.schemas: list[Any] = []

    def with_structured_output(self, schema: Any, **kwargs: Any) -> FakeLLMRunnable:
        """Return a runnable that answers with the prepared response."""
        self.schemas.append(schema)
        return FakeLLMRunnable(self.response, self.calls, self.failures)

    def invoke(self, messages: Any) -> Any:
        """Record the messages and return the prepared response."""
        self.calls.append(messages)
        if self.failures > 0:
            self.failures -= 1
            raise RuntimeError("fake llm failure")
        return self.response
