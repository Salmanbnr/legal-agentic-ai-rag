"""Stable idempotency keys shared by the API and reconciliation jobs."""

from __future__ import annotations

from uuid import UUID

from app.core.config import Settings


def reindex_idempotency_key(
    document_id: UUID | str, settings: Settings
) -> str:
    """Return the reindex key for the current model and schema version."""
    return (
        f"{document_id}:reindex:{settings.ingest_pipeline_version}:"
        f"{settings.embedding_model}:{settings.index_schema_version}"
    )
