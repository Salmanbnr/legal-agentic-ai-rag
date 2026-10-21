"""Tests for the CPU embedding wrapper and its singleton behaviour."""

from __future__ import annotations

import math
import sys
import types
from pathlib import Path
from typing import Any

import pytest

from app.core.config import Settings
from app.embeddings import model as model_module
from app.embeddings.model import (
    EmbeddingError,
    SentenceTransformerEmbedder,
    clear_embedding_cache,
    load_embedding_model,
)


class RecordingModel:
    """Minimal sentence-transformers stand-in."""

    def __init__(self, dimension: int = 384) -> None:
        """Create a model with a fixed output dimension."""
        self.dimension = dimension
        self.batches: list[list[str]] = []

    def get_sentence_embedding_dimension(self) -> int:
        """Return the configured output dimension."""
        return self.dimension

    def encode(self, texts, **kwargs):
        """Return one unit vector per text and record the batch."""
        batch = list(texts)
        self.batches.append(batch)
        return [
            [1.0 if position == 0 else 0.0 for position in range(self.dimension)]
            for _ in batch
        ]


def test_embed_texts_uses_the_configured_batch_size() -> None:
    model = RecordingModel(dimension=8)
    embedder = SentenceTransformerEmbedder(model, batch_size=2)

    vectors = embedder.embed_texts(["a", "b", "c", "d", "e"])

    assert [len(batch) for batch in model.batches] == [2, 2, 1]
    assert len(vectors) == 5
    assert all(len(vector) == 8 for vector in vectors)
    assert embedder.dimension == 8


def test_embed_texts_requests_normalized_vectors() -> None:
    recorded: dict[str, Any] = {}

    class Model(RecordingModel):
        def encode(self, texts, **kwargs):
            recorded.update(kwargs)
            return super().encode(texts, **kwargs)

    embedder = SentenceTransformerEmbedder(Model(dimension=4), batch_size=2)
    embedder.embed_texts(["one", "two"])

    assert recorded["normalize_embeddings"] is True
    assert recorded["batch_size"] == 2
    assert recorded["show_progress_bar"] is False


def test_embed_query_returns_one_vector() -> None:
    embedder = SentenceTransformerEmbedder(RecordingModel(dimension=4))

    vector = embedder.embed_query("question")

    assert len(vector) == 4


def test_empty_input_returns_no_vectors() -> None:
    embedder = SentenceTransformerEmbedder(RecordingModel(dimension=4))

    assert embedder.embed_texts([]) == []


def test_dimension_mismatch_is_reported_without_text() -> None:
    class WrongModel(RecordingModel):
        def encode(self, texts, **kwargs):
            return [[0.0, 0.0] for _ in texts]

    embedder = SentenceTransformerEmbedder(WrongModel(dimension=4), batch_size=2)

    with pytest.raises(EmbeddingError) as error:
        embedder.embed_texts(["secret text"])

    assert error.value.error_code == "EMBEDDING_DIMENSION_MISMATCH"
    assert "secret text" not in str(error.value)


def test_model_load_failure_uses_a_safe_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setitem(sys.modules, "sentence_transformers", None)
    clear_embedding_cache()

    with pytest.raises(EmbeddingError) as error:
        load_embedding_model(Settings(_env_file=None))

    assert error.value.error_code == "EMBEDDING_MODEL_UNAVAILABLE"


def test_model_is_loaded_once_per_configuration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    created: list[RecordingModel] = []

    class FakeSentenceTransformer:
        def __init__(self, name: str, **kwargs: Any) -> None:
            self.name = name
            self.kwargs = kwargs
            created.append(self)

        def get_sentence_embedding_dimension(self) -> int:
            return 384

        def encode(self, texts, **kwargs):
            return [[0.0] * 384 for _ in texts]

    module = types.ModuleType("sentence_transformers")
    module.SentenceTransformer = FakeSentenceTransformer  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "sentence_transformers", module)
    clear_embedding_cache()

    settings = Settings(_env_file=None)
    try:
        first = load_embedding_model(settings)
        second = load_embedding_model(settings)

        assert first is second
        assert len(created) == 1
        assert created[0].kwargs["device"] == "cpu"
        assert created[0].kwargs["cache_folder"] == str(
            settings.model_cache_dir
        )
    finally:
        clear_embedding_cache()


def test_real_model_returns_normalized_vectors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pytest.importorskip("sentence_transformers")
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    settings = Settings(_env_file=None).model_copy(
        update={
            "model_cache_dir": (
                Path.home() / ".cache" / "huggingface" / "hub"
            )
        }
    )
    clear_embedding_cache()
    try:
        embedder = load_embedding_model(settings)
    except EmbeddingError:
        pytest.skip("the embedding model is not present in the local cache")
    try:
        vectors = embedder.embed_texts(
            ["The plaintiff filed a claim.", "The defendant answered."]
        )

        assert embedder.dimension == 384
        assert len(vectors) == 2
        for vector in vectors:
            assert len(vector) == 384
            norm = math.sqrt(sum(value * value for value in vector))
            assert abs(norm - 1.0) < 1e-4
    finally:
        clear_embedding_cache()
