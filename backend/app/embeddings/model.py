"""CPU sentence-transformer embeddings loaded once per process.

The model is imported lazily so the API and tests can start without the heavy
``sentence-transformers``/``torch`` stack installed. Vector values are
normalized for cosine similarity and the output dimension is verified by the
index startup check.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Protocol, Sequence

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)


class EmbeddingError(RuntimeError):
    """A safe embedding failure with a stable, persistable error code."""

    def __init__(self, error_code: str, *, retryable: bool = True) -> None:
        """Create an embedding error without including document text."""
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable


class TextEmbedder(Protocol):
    """Minimal embedding interface used by the indexer and query path."""

    dimension: int

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Return one normalized vector per input text."""
        ...

    def embed_query(self, text: str) -> list[float]:
        """Return one normalized vector for a query string."""
        ...


class SentenceTransformerEmbedder:
    """Batch wrapper around a sentence-transformers model.

    The wrapper preserves the configured batch size so both indexing and query
    embedding use the same deterministic path. ``normalize_embeddings=True``
    keeps vectors compatible with the cosine Pinecone index.
    """

    def __init__(self, model: Any, *, batch_size: int = 32) -> None:
        """Wrap an already-loaded model and validate its output dimension."""
        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        dimension = _model_dimension(model)
        if dimension is None or dimension < 1:
            raise EmbeddingError(
                "EMBEDDING_DIMENSION_UNAVAILABLE", retryable=False
            )
        self._model = model
        self._batch_size = batch_size
        self.dimension = dimension

    def embed_texts(self, texts: Sequence[str]) -> list[list[float]]:
        """Embed texts in batches of the configured size."""
        cleaned = [str(text) for text in texts]
        if not cleaned:
            return []

        vectors: list[list[float]] = []
        for start in range(0, len(cleaned), self._batch_size):
            batch = cleaned[start : start + self._batch_size]
            try:
                encoded = self._model.encode(
                    batch,
                    batch_size=self._batch_size,
                    normalize_embeddings=True,
                    convert_to_numpy=True,
                    show_progress_bar=False,
                )
            except Exception as exc:
                logger.error(
                    "embedding batch failed",
                    extra={"error_class": type(exc).__name__},
                )
                raise EmbeddingError("EMBEDDING_FAILED") from None

            for row in encoded:
                vector = [float(value) for value in row]
                if len(vector) != self.dimension:
                    raise EmbeddingError(
                        "EMBEDDING_DIMENSION_MISMATCH", retryable=False
                    )
                vectors.append(vector)

        if len(vectors) != len(cleaned):
            raise EmbeddingError("EMBEDDING_FAILED")
        return vectors

    def embed_query(self, text: str) -> list[float]:
        """Embed a single query string."""
        vectors = self.embed_texts([text])
        if not vectors:
            raise EmbeddingError("EMBEDDING_FAILED")
        return vectors[0]


def _model_dimension(model: Any) -> int | None:
    """Read the output dimension from a sentence-transformers model."""
    getter = getattr(model, "get_sentence_embedding_dimension", None)
    if callable(getter):
        try:
            value = getter()
        except Exception:
            return None
        if isinstance(value, int):
            return value
    value = getattr(model, "dimension", None)
    return value if isinstance(value, int) else None


_model_lock = threading.Lock()
_model_cache: dict[tuple[str, str, str, int], SentenceTransformerEmbedder] = {}


def load_embedding_model(settings: Settings | None = None) -> SentenceTransformerEmbedder:
    """Load and cache the configured embedding model once per process.

    The model runs on the configured device, which defaults to CPU. A missing
    model or dependency raises a safe error that names no local paths or
    secrets.
    """
    resolved = settings or get_settings()
    key = (
        resolved.embedding_model,
        resolved.embedding_device,
        str(resolved.model_cache_dir),
        resolved.embedding_batch_size,
    )
    with _model_lock:
        cached = _model_cache.get(key)
        if cached is not None:
            return cached

        try:
            from sentence_transformers import SentenceTransformer
        except Exception:
            raise EmbeddingError(
                "EMBEDDING_MODEL_UNAVAILABLE", retryable=False
            ) from None

        try:
            model = SentenceTransformer(
                resolved.embedding_model,
                device=resolved.embedding_device,
                cache_folder=str(resolved.model_cache_dir),
            )
        except Exception as exc:
            logger.error(
                "embedding model load failed",
                extra={
                    "embedding_model": resolved.embedding_model,
                    "error_class": type(exc).__name__,
                },
            )
            raise EmbeddingError(
                "EMBEDDING_MODEL_UNAVAILABLE", retryable=False
            ) from None

        embedder = SentenceTransformerEmbedder(
            model, batch_size=resolved.embedding_batch_size
        )
        _model_cache[key] = embedder
        logger.info(
            "embedding model loaded",
            extra={
                "embedding_model": resolved.embedding_model,
                "embedding_device": resolved.embedding_device,
                "embedding_dimension": embedder.dimension,
            },
        )
        return embedder


def get_embedder(settings: Settings | None = None) -> TextEmbedder:
    """Return the process-wide embedding singleton."""
    return load_embedding_model(settings)


def clear_embedding_cache() -> None:
    """Drop cached models; intended for tests and process shutdown."""
    with _model_lock:
        _model_cache.clear()
