"""Tests for environment-backed application configuration."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from app.core.config import (
    BACKEND_DIR,
    ConfigurationError,
    Settings,
    get_settings,
    load_settings,
)

REQUIRED_ENV = {
    "SUPABASE_URL": "https://validation-project.supabase.co",
    "SUPABASE_PUBLISHABLE_KEY": "publishable-key",
    "SUPABASE_SERVICE_ROLE_KEY": "service-role-key",
    "OPENAI_API_KEY": "openai-key",
    "OPENAI_MODEL": "gpt-test",
    "PINECONE_API_KEY": "pinecone-key",
    "EMBEDDING_MODEL": "sentence-transformers/all-MiniLM-L6-v2",
    "FRONTEND_ORIGIN": "http://localhost:3000",
}

ALL_ENV_KEYS = {
    "APP_ENV",
    "API_PREFIX",
    "SUPABASE_URL",
    "SUPABASE_PUBLISHABLE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "EMBEDDING_MODEL",
    "EMBEDDING_DEVICE",
    "MODEL_CACHE_DIR",
    "TEMP_DIR",
    "MAX_UPLOAD_MB",
    "MAX_PDF_PAGES",
    "AGENT_MAX_STEPS",
    "AGENT_MAX_RETRIEVAL_RETRIES",
    "AGENT_MAX_LLM_CALLS",
    "FRONTEND_ORIGIN",
}


@pytest.fixture(autouse=True)
def clean_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove ambient config so defaults are deterministic."""
    for key in ALL_ENV_KEYS:
        monkeypatch.delenv(key, raising=False)


def _set_required(monkeypatch: pytest.MonkeyPatch, **overrides: str) -> None:
    for key, value in REQUIRED_ENV.items():
        monkeypatch.setenv(key, value)
    for key, value in overrides.items():
        monkeypatch.setenv(key, value)


def _clear_required(monkeypatch: pytest.MonkeyPatch) -> None:
    for key in REQUIRED_ENV:
        monkeypatch.delenv(key, raising=False)


def test_settings_load_values_and_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    settings = Settings(_env_file=None)

    assert settings.app_env == "development"
    assert settings.api_prefix == "/api/v1"
    assert settings.supabase_url_value == "https://validation-project.supabase.co"
    assert settings.supabase_service_role_key.get_secret_value() == "service-role-key"
    assert settings.pinecone_index_name == "legal-evidence"
    assert settings.embedding_device == "cpu"
    assert settings.max_upload_mb == 50
    assert settings.max_pdf_pages == 1000
    assert settings.agent_max_steps == 12
    assert settings.agent_max_retrieval_retries == 2
    assert settings.agent_max_llm_calls == 12


def test_relative_storage_paths_resolve_under_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, TEMP_DIR=".tmp-example")
    settings = Settings(_env_file=None)

    assert settings.temp_dir == (BACKEND_DIR / ".tmp-example").resolve()
    assert settings.model_cache_dir == (BACKEND_DIR / ".cache/huggingface").resolve()


def test_load_settings_reports_missing_fields_without_values(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _clear_required(monkeypatch)

    with pytest.raises(ConfigurationError) as error:
        load_settings(env_file=None)

    message = str(error.value)
    assert "SUPABASE_URL" in message
    assert "OPENAI_API_KEY" in message
    assert "PINECONE_API_KEY" in message
    assert "Field required" in message


def test_validation_error_does_not_echo_invalid_value(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, SUPABASE_URL="sensitive-invalid-value")

    with pytest.raises(ConfigurationError) as error:
        load_settings(env_file=None)

    assert "SUPABASE_URL" in str(error.value)
    assert "sensitive-invalid-value" not in str(error.value)


def test_secret_values_are_redacted(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch, OPENAI_API_KEY="super-secret-openai-key")
    settings = Settings(_env_file=None)

    assert "super-secret-openai-key" not in repr(settings)
    assert "super-secret-openai-key" not in settings.model_dump_json()
    assert "**********" in settings.model_dump_json()


def test_api_prefix_must_be_rooted_without_trailing_slash(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, API_PREFIX="api/v1")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)

    monkeypatch.setenv("API_PREFIX", "/api/v1/")
    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_invalid_pinecone_index_name_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, PINECONE_INDEX_NAME="Legal Evidence")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_frontend_origin_must_be_a_bare_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, FRONTEND_ORIGIN="http://localhost:3000/workspace")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
