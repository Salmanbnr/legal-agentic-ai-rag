"""Startup checks that keep the embedding model and Pinecone index aligned."""

from __future__ import annotations

import logging

from app.core.config import Settings, get_settings
from app.embeddings.model import get_embedder
from app.indexing.pinecone_store import verify_index_dimension

logger = logging.getLogger(__name__)


def validate_index_startup(settings: Settings | None = None) -> int:
    """Load the embedding model and compare its size to the index dimension.

    Tests and explicitly disabled startup checks skip the heavy model load.
    Production and worker startup fail fast when the dimensions disagree.
    """
    resolved = settings or get_settings()
    if resolved.app_env == "test" or not resolved.startup_checks_enabled:
        return resolved.embedding_dimension

    embedder = get_embedder(resolved)
    dimension = verify_index_dimension(resolved, embedder=embedder)
    logger.info(
        "index dimension verified",
        extra={
            "embedding_model": resolved.embedding_model,
            "embedding_dimension": dimension,
            "pinecone_index": resolved.pinecone_index_name,
        },
    )
    return dimension
