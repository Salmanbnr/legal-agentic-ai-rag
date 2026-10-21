"""Tests for the startup dimension check between the model and the index."""

from __future__ import annotations

import pytest
from fakes import FakeControlPlane, FakeEmbedder

from app.core.config import Settings
from app.indexing import pinecone_store
from app.indexing import startup as startup_module
from app.indexing.pinecone_store import PineconeDimensionError
from app.indexing.startup import validate_index_startup


def test_startup_check_is_skipped_in_the_test_environment() -> None:
    settings = Settings(_env_file=None)

    assert settings.app_env == "test"
    assert validate_index_startup(settings) == settings.embedding_dimension


def test_startup_check_can_be_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None).model_copy(
        update={"app_env": "production", "startup_checks_enabled": False}
    )
    called: list[bool] = []

    monkeypatch.setattr(
        startup_module,
        "get_embedder",
        lambda *args, **kwargs: called.append(True),
    )

    assert validate_index_startup(settings) == settings.embedding_dimension
    assert called == []


def test_startup_check_fails_when_dimensions_differ(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(_env_file=None).model_copy(
        update={"app_env": "production", "startup_checks_enabled": True}
    )
    monkeypatch.setattr(
        startup_module,
        "get_embedder",
        lambda *args, **kwargs: FakeEmbedder(dimension=768),
    )
    monkeypatch.setattr(
        pinecone_store,
        "get_pinecone_client",
        lambda *args, **kwargs: FakeControlPlane(dimension=384),
    )

    with pytest.raises(PineconeDimensionError) as error:
        validate_index_startup(settings)

    assert "768" in str(error.value)
    assert "384" in str(error.value)
