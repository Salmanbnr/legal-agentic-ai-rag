"""Pinecone namespace, dimension, upsert, delete, and fetch helpers.

Pinecone is a derived index: vector ids are Postgres ``document_chunks.id``
values and document text never leaves Postgres. Each case owns one namespace.
"""

from __future__ import annotations

import logging
import random
import time
from typing import Any, Callable, Iterable, Sequence
from uuid import UUID

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

_DEFAULT_FETCH_BATCH = 1000


class PineconeError(RuntimeError):
    """A safe Pinecone failure with a stable, persistable error code."""

    def __init__(self, error_code: str, *, retryable: bool = True) -> None:
        """Create a Pinecone error without exposing client internals."""
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable


class PineconeDimensionError(PineconeError):
    """Raised when the index and embedding dimensions disagree."""

    def __init__(self, message: str) -> None:
        """Keep the human-readable dimension message for startup failures."""
        super().__init__("EMBEDDING_DIMENSION_MISMATCH", retryable=False)
        self.detail = message

    def __str__(self) -> str:
        return self.detail


def namespace_for_case(case_id: UUID | str) -> str:
    """Return the canonical case namespace used by every vector operation."""
    try:
        return str(UUID(str(case_id)))
    except (TypeError, ValueError) as exc:
        raise PineconeError("INVALID_CASE_ID", retryable=False) from exc


def _import_pinecone() -> Any:
    """Import the optional Pinecone client only when it is needed."""
    try:
        from pinecone import Pinecone
    except Exception:
        raise PineconeError("PINECONE_UNAVAILABLE", retryable=False) from None
    return Pinecone


def get_pinecone_client(settings: Settings | None = None) -> Any:
    """Create a Pinecone control-plane client from configuration."""
    resolved = settings or get_settings()
    Pinecone = _import_pinecone()
    try:
        return Pinecone(api_key=resolved.pinecone_api_key.get_secret_value())
    except Exception as exc:
        logger.error(
            "pinecone client creation failed",
            extra={"error_class": type(exc).__name__},
        )
        raise PineconeError("PINECONE_CONNECTION_FAILED") from None


def get_index(settings: Settings | None = None, *, client: Any | None = None) -> Any:
    """Return the configured index data-plane client."""
    resolved = settings or get_settings()
    control = client or get_pinecone_client(resolved)
    try:
        return control.Index(resolved.pinecone_index_name)
    except Exception as exc:
        logger.error(
            "pinecone index lookup failed",
            extra={"error_class": type(exc).__name__},
        )
        raise PineconeError("PINECONE_INDEX_UNAVAILABLE") from None


def _field(value: Any, name: str) -> Any:
    """Read a field from either a Pinecone model or a plain mapping."""
    if isinstance(value, dict):
        return value.get(name)
    return getattr(value, name, None)


def describe_index_dimension(
    settings: Settings | None = None, *, client: Any | None = None
) -> int:
    """Return the configured Pinecone index dimension."""
    resolved = settings or get_settings()
    control = client or get_pinecone_client(resolved)
    try:
        description = control.describe_index(resolved.pinecone_index_name)
    except Exception as exc:
        logger.error(
            "pinecone index description failed",
            extra={"error_class": type(exc).__name__},
        )
        raise PineconeError("PINECONE_INDEX_UNAVAILABLE") from None
    dimension = _field(description, "dimension")
    if not isinstance(dimension, int) or dimension < 1:
        raise PineconeError(
            "PINECONE_INDEX_DIMENSION_UNAVAILABLE", retryable=False
        )
    return dimension


def index_exists(control: Any, name: str) -> bool:
    """Return whether an index with ``name`` already exists."""
    has_index = getattr(control, "has_index", None)
    if callable(has_index):
        try:
            return bool(has_index(name))
        except Exception:
            pass
    try:
        listed = control.list_indexes()
    except Exception as exc:
        logger.error(
            "pinecone index listing failed",
            extra={"error_class": type(exc).__name__},
        )
        raise PineconeError("PINECONE_INDEX_UNAVAILABLE") from None
    for item in listed:
        item_name = _field(item, "name")
        if item_name is None and isinstance(item, str):
            item_name = item
        if item_name == name:
            return True
    return False


def wait_for_index_dimension(
    settings: Settings | None = None,
    *,
    client: Any | None = None,
    attempts: int = 30,
    delay_seconds: float = 2.0,
    sleep: Callable[[float], None] = time.sleep,
) -> int:
    """Wait until the index reports its dimension and return that dimension."""
    resolved = settings or get_settings()
    last_error: PineconeError | None = None
    for attempt in range(1, max(1, attempts) + 1):
        try:
            return describe_index_dimension(resolved, client=client)
        except PineconeError as exc:
            last_error = exc
            if attempt >= attempts:
                break
            sleep(delay_seconds)
    raise last_error or PineconeError("PINECONE_INDEX_UNAVAILABLE")


def delete_index(
    settings: Settings | None = None,
    *,
    name: str | None = None,
    client: Any | None = None,
) -> bool:
    """Delete the configured index, returning whether one was removed.

    The index holds derived vectors only, so it can be rebuilt from Postgres
    and from the original files. Deleting is reserved for an index created with
    an incompatible schema.
    """
    resolved = settings or get_settings()
    target = name or resolved.pinecone_index_name
    control = client or get_pinecone_client(resolved)
    if not index_exists(control, target):
        return False
    try:
        control.delete_index(target)
    except Exception as exc:
        logger.error(
            "pinecone index deletion failed",
            extra={"error_class": type(exc).__name__},
        )
        raise PineconeError("PINECONE_INDEX_DELETE_FAILED") from None
    logger.warning("pinecone index deleted", extra={"pinecone_index": target})
    return True


def create_index_if_missing(
    settings: Settings | None = None,
    *,
    dimension: int | None = None,
    client: Any | None = None,
) -> str:
    """Create the serverless index when absent and verify its dimension.

    Returns ``"created"`` or ``"exists"``. The operation is safe to rerun.
    """
    resolved = settings or get_settings()
    target_dimension = dimension or resolved.embedding_dimension
    control = client or get_pinecone_client(resolved)
    name = resolved.pinecone_index_name

    if index_exists(control, name):
        existing = describe_index_dimension(resolved, client=control)
        if existing != target_dimension:
            raise PineconeDimensionError(
                f"Pinecone index {name} has dimension {existing}, "
                f"expected {target_dimension}"
            )
        return "exists"

    try:
        from pinecone import ServerlessSpec
    except Exception:
        raise PineconeError("PINECONE_UNAVAILABLE", retryable=False) from None

    try:
        control.create_index(
            name=name,
            dimension=target_dimension,
            metric=resolved.pinecone_metric,
            spec=ServerlessSpec(
                cloud=resolved.pinecone_cloud,
                region=resolved.pinecone_region,
            ),
        )
    except Exception as exc:
        logger.error(
            "pinecone index creation failed",
            extra={"error_class": type(exc).__name__},
        )
        raise PineconeError("PINECONE_INDEX_CREATE_FAILED") from None

    wait_for_index_dimension(resolved, client=control)
    return "created"


def verify_index_dimension(
    settings: Settings | None = None,
    *,
    embedder: Any | None = None,
    client: Any | None = None,
    index_dimension: int | None = None,
) -> int:
    """Compare the embedding dimension with the configured index dimension.

    Raises:
        PineconeDimensionError: With a clear message when the dimensions differ.
    """
    resolved = settings or get_settings()
    if embedder is None:
        from app.embeddings.model import get_embedder

        embedder = get_embedder(resolved)
    dimension = (
        index_dimension
        if index_dimension is not None
        else describe_index_dimension(resolved, client=client)
    )
    if embedder.dimension != dimension:
        raise PineconeDimensionError(
            f"Embedding model outputs {embedder.dimension} dimensions but "
            f"Pinecone index {resolved.pinecone_index_name} has dimension "
            f"{dimension}"
        )
    if dimension != resolved.embedding_dimension:
        raise PineconeDimensionError(
            f"Pinecone index {resolved.pinecone_index_name} has dimension "
            f"{dimension} but EMBEDDING_DIMENSION is "
            f"{resolved.embedding_dimension}"
        )
    return dimension


def _iter_batches(values: Sequence[Any], size: int) -> Iterable[Sequence[Any]]:
    """Yield consecutive slices of at most ``size`` values."""
    for start in range(0, len(values), size):
        yield values[start : start + size]


def upsert_vectors(
    index: Any, namespace: str, vectors: Sequence[Any]
) -> None:
    """Upsert one batch of vectors into a case namespace."""
    if not vectors:
        return
    index.upsert(vectors=list(vectors), namespace=namespace)


def upsert_with_retry(
    index: Any,
    namespace: str,
    vectors: Sequence[Any],
    *,
    settings: Settings | None = None,
    sleep: Callable[[float], None] = time.sleep,
    rng: Callable[[float, float], float] = random.uniform,
) -> None:
    """Upsert one batch with capped exponential backoff and jitter."""
    resolved = settings or get_settings()
    attempts = resolved.pinecone_retry_attempts
    for attempt in range(1, attempts + 1):
        try:
            upsert_vectors(index, namespace, vectors)
            return
        except Exception as exc:
            if attempt >= attempts:
                logger.error(
                    "pinecone upsert failed",
                    extra={
                        "error_class": type(exc).__name__,
                        "attempt": attempt,
                    },
                )
                raise PineconeError("PINECONE_UPSERT_FAILED") from None
            nominal = min(
                resolved.pinecone_retry_max_seconds,
                resolved.pinecone_retry_base_seconds * (2 ** (attempt - 1)),
            )
            delay = min(
                resolved.pinecone_retry_max_seconds,
                nominal * rng(0.5, 1.5),
            )
            logger.warning(
                "pinecone upsert retry scheduled",
                extra={"attempt": attempt, "delay_seconds": round(delay, 2)},
            )
            sleep(delay)


def delete_vectors(
    index: Any,
    namespace: str,
    ids: Sequence[str],
    *,
    settings: Settings | None = None,
) -> int:
    """Delete vector ids from a namespace in batches."""
    resolved = settings or get_settings()
    cleaned = [str(value) for value in ids if str(value)]
    for batch in _iter_batches(cleaned, resolved.pinecone_delete_batch_size):
        index.delete(ids=list(batch), namespace=namespace)
    return len(cleaned)


def delete_document_vectors(
    case_id: UUID | str,
    document_id: UUID | str,
    *,
    chunk_ids: Sequence[str] | None = None,
    index: Any | None = None,
    settings: Settings | None = None,
) -> int:
    """Delete every vector id belonging to one document.

    When ``chunk_ids`` is omitted the ids are loaded from Postgres, keeping
    Postgres as the source of truth for what should exist in Pinecone.
    """
    resolved = settings or get_settings()
    ids = list(chunk_ids) if chunk_ids is not None else None
    if ids is None:
        from app.db.client import get_service_client
        from app.jobs.store import get_chunks_for_document

        chunks = get_chunks_for_document(
            get_service_client(resolved), document_id
        )
        ids = [str(chunk.id) for chunk in chunks]
    if not ids:
        return 0
    resolved_index = index or get_index(resolved)
    return delete_vectors(
        resolved_index,
        namespace_for_case(case_id),
        ids,
        settings=resolved,
    )


def fetch_existing_ids(
    index: Any,
    namespace: str,
    ids: Sequence[str],
    *,
    batch_size: int = _DEFAULT_FETCH_BATCH,
) -> set[str]:
    """Return the subset of ids currently present in the namespace."""
    cleaned = [str(value) for value in ids if str(value)]
    found: set[str] = set()
    for batch in _iter_batches(cleaned, max(1, batch_size)):
        try:
            response = index.fetch(ids=list(batch), namespace=namespace)
        except Exception as exc:
            logger.error(
                "pinecone fetch failed",
                extra={"error_class": type(exc).__name__},
            )
            raise PineconeError("PINECONE_FETCH_FAILED") from None
        vectors = _field(response, "vectors")
        if isinstance(vectors, dict):
            found.update(str(key) for key in vectors)
    return found


def list_vector_ids(
    index: Any,
    namespace: str,
    *,
    limit: int | None = None,
) -> set[str]:
    """List every vector id in a namespace, following Pinecone pagination."""
    found: set[str] = set()
    seen = 0
    try:
        if limit is None:
            pages = index.list(namespace=namespace)
        else:
            pages = index.list(namespace=namespace, limit=limit)
    except TypeError:
        pages = index.list(namespace=namespace)
    except Exception as exc:
        logger.error(
            "pinecone list failed",
            extra={"error_class": type(exc).__name__},
        )
        raise PineconeError("PINECONE_LIST_FAILED") from None

    try:
        for page in pages:
            if isinstance(page, str):
                found.add(page)
                seen += 1
            elif isinstance(page, dict):
                for value in page.get("ids", page.get("vectors", [])):
                    found.add(str(value))
                    seen += 1
            else:
                for value in page:
                    found.add(str(value))
                    seen += 1
            if limit is not None and seen >= limit:
                break
    except Exception as exc:
        logger.error(
            "pinecone list pagination failed",
            extra={"error_class": type(exc).__name__},
        )
        raise PineconeError("PINECONE_LIST_FAILED") from None
    return found
