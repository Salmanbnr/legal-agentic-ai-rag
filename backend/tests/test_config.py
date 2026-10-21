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
    "LOG_LEVEL",
    "LOG_FILE",
    "SUPABASE_URL",
    "SUPABASE_PUBLISHABLE_KEY",
    "SUPABASE_SERVICE_ROLE_KEY",
    "OPENAI_API_KEY",
    "OPENAI_MODEL",
    "OPENAI_TIMEOUT_SECONDS",
    "OPENAI_MAX_RETRIES",
    "OPENAI_RETRY_BASE_SECONDS",
    "OPENAI_RETRY_MAX_SECONDS",
    "OPENAI_MAX_OUTPUT_TOKENS",
    "PINECONE_API_KEY",
    "PINECONE_INDEX_NAME",
    "PINECONE_CLOUD",
    "PINECONE_REGION",
    "PINECONE_METRIC",
    "PINECONE_UPSERT_BATCH_SIZE",
    "PINECONE_DELETE_BATCH_SIZE",
    "PINECONE_RETRY_ATTEMPTS",
    "PINECONE_RETRY_BASE_SECONDS",
    "PINECONE_RETRY_MAX_SECONDS",
    "EMBEDDING_MODEL",
    "EMBEDDING_DEVICE",
    "EMBEDDING_DIMENSION",
    "EMBEDDING_BATCH_SIZE",
    "MODEL_CACHE_DIR",
    "TEMP_DIR",
    "MAX_UPLOAD_MB",
    "MAX_PDF_PAGES",
    "UPLOAD_RATE_LIMIT_PER_MINUTE",
    "INGEST_PIPELINE_VERSION",
    "PDF_NATIVE_TEXT_DENSITY_THRESHOLD",
    "OCR_LANGUAGE",
    "OCR_DPI",
    "OCR_CONFIDENCE_THRESHOLD",
    "OCR_PREPROCESS_ENABLED",
    "OCR_HANDWRITING_CONFIDENCE_THRESHOLD",
    "OCR_HANDWRITING_MAX_WORDS",
    "OCR_TABLE_MIN_ROWS",
    "OCR_MAX_DESKEW_DEGREES",
    "TESSERACT_CMD",
    "QUALITY_REVIEW_RATIO_THRESHOLD",
    "QUALITY_PARSER_ERROR_THRESHOLD",
    "HEADER_FOOTER_REPEAT_RATIO",
    "HEADER_FOOTER_MAX_LINES",
    "CHUNK_MIN_TOKENS",
    "CHUNK_MAX_TOKENS",
    "CHUNK_OVERLAP_MIN_TOKENS",
    "CHUNK_OVERLAP_MAX_TOKENS",
    "CHUNK_TOKENIZER_LOCAL_FILES_ONLY",
    "INDEX_SCHEMA_VERSION",
    "RETRIEVAL_CANDIDATE_MULTIPLIER",
    "RETRIEVAL_MAX_SPANS_PER_DOCUMENT",
    "RETRIEVAL_NEIGHBOR_EXPANSION",
    "STARTUP_CHECKS_ENABLED",
    "WORKER_ENABLED",
    "WORKER_POLL_SECONDS",
    "WORKER_MAX_ATTEMPTS",
    "WORKER_BACKOFF_BASE_SECONDS",
    "WORKER_BACKOFF_MAX_SECONDS",
    "AGENT_MAX_STEPS",
    "AGENT_MAX_RETRIEVAL_RETRIES",
    "AGENT_MAX_LLM_CALLS",
    "AGENT_MAX_TOKENS",
    "AGENT_TIMEOUT_SECONDS",
    "AGENT_SKIP_ANALYST_ON_STRONG_FACT_LOOKUP",
    "AGENT_ANALYST_MIN_SPANS",
    "AGENT_ANALYST_MAX_SPANS",
    "AGENT_RESEARCH_MAX_MULTI_HOP",
    "ANSWER_MAX_EVIDENCE_SPANS",
    "ANSWER_MAX_QUESTION_LENGTH",
    "ANSWER_MAX_LLM_CALLS",
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
    assert settings.log_level == "INFO"
    assert settings.log_file is None
    assert settings.supabase_url_value == "https://validation-project.supabase.co"
    assert settings.supabase_service_role_key.get_secret_value() == "service-role-key"
    assert settings.pinecone_index_name == "legal-evidence"
    assert settings.pinecone_cloud == "aws"
    assert settings.pinecone_region == "us-east-1"
    assert settings.pinecone_metric == "cosine"
    assert settings.pinecone_upsert_batch_size == 100
    assert settings.pinecone_delete_batch_size == 1000
    assert settings.pinecone_retry_attempts == 4
    assert settings.pinecone_retry_base_seconds == 0.5
    assert settings.pinecone_retry_max_seconds == 8.0
    assert settings.embedding_device == "cpu"
    assert settings.embedding_dimension == 384
    assert settings.embedding_batch_size == 32
    assert settings.max_upload_mb == 50
    assert settings.max_pdf_pages == 1000
    assert settings.upload_rate_limit_per_minute == 30
    assert settings.ingest_pipeline_version == "1.0.0"
    assert settings.pdf_native_text_density_threshold == 0.0002
    assert settings.ocr_language == "eng"
    assert settings.ocr_dpi == 300
    assert settings.ocr_confidence_threshold == 70.0
    assert settings.ocr_preprocess_enabled is True
    assert settings.ocr_handwriting_confidence_threshold == 35.0
    assert settings.ocr_handwriting_max_words == 5
    assert settings.ocr_table_min_rows == 3
    assert settings.ocr_max_deskew_degrees == 15.0
    assert settings.tesseract_cmd is None
    assert settings.quality_review_ratio_threshold == 0.30
    assert settings.quality_parser_error_threshold == 3
    assert settings.header_footer_repeat_ratio == 0.60
    assert settings.header_footer_max_lines == 3
    assert settings.chunk_min_tokens == 150
    assert settings.chunk_max_tokens == 220
    assert settings.chunk_overlap_min_tokens == 20
    assert settings.chunk_overlap_max_tokens == 40
    assert settings.chunk_tokenizer_local_files_only is False
    assert settings.index_schema_version == 1
    assert settings.retrieval_candidate_multiplier == 4
    assert settings.retrieval_max_spans_per_document == 3
    assert settings.retrieval_neighbor_expansion is True
    assert settings.startup_checks_enabled is True
    assert settings.worker_enabled is True
    assert settings.worker_poll_seconds == 2.0
    assert settings.worker_max_attempts == 3
    assert settings.worker_backoff_base_seconds == 2.0
    assert settings.worker_backoff_max_seconds == 60.0
    assert settings.agent_max_steps == 24
    assert settings.agent_max_retrieval_retries == 2
    assert settings.agent_max_llm_calls == 12
    assert settings.agent_max_tokens == 20000
    assert settings.agent_timeout_seconds == 60
    assert settings.agent_skip_analyst_on_strong_fact_lookup is True
    assert settings.agent_analyst_min_spans == 3
    assert settings.agent_analyst_max_spans == 12
    assert settings.agent_research_max_multi_hop == 1
    assert settings.openai_timeout_seconds == 60.0
    assert settings.openai_max_retries == 3
    assert settings.openai_retry_base_seconds == 1.0
    assert settings.openai_retry_max_seconds == 20.0
    assert settings.openai_max_output_tokens == 1500
    assert settings.answer_max_evidence_spans == 10
    assert settings.answer_max_question_length == 2000
    assert settings.answer_max_llm_calls == 1


def test_relative_storage_paths_resolve_under_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, TEMP_DIR=".tmp-example")
    settings = Settings(_env_file=None)

    assert settings.temp_dir == (BACKEND_DIR / ".tmp-example").resolve()
    assert settings.model_cache_dir == (BACKEND_DIR / ".cache/huggingface").resolve()


def test_logging_values_are_normalised_and_resolved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, LOG_LEVEL="debug", LOG_FILE="logs/backend.log")
    settings = Settings(_env_file=None)

    assert settings.log_level == "DEBUG"
    assert settings.log_file == (BACKEND_DIR / "logs/backend.log").resolve()


def test_empty_log_file_disables_file_logging(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, LOG_FILE="   ")
    settings = Settings(_env_file=None)

    assert settings.log_file is None


def test_unknown_log_level_is_rejected(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch, LOG_LEVEL="VERBOSE")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


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


def test_worker_max_attempts_is_capped_at_three(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, WORKER_MAX_ATTEMPTS="4")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_ocr_language_rejects_shell_metacharacters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, OCR_LANGUAGE="eng;rm")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_empty_tesseract_command_uses_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, TESSERACT_CMD="  ")

    settings = Settings(_env_file=None)

    assert settings.tesseract_cmd is None


def test_missing_tesseract_command_path_is_rejected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, TESSERACT_CMD="missing-tesseract-binary")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_quality_ratio_threshold_must_be_a_fraction(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, QUALITY_REVIEW_RATIO_THRESHOLD="1.5")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_chunk_minimum_must_not_exceed_maximum(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(monkeypatch, CHUNK_MIN_TOKENS="230", CHUNK_MAX_TOKENS="220")

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_chunk_overlap_must_fit_inside_chunk_size(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(
        monkeypatch,
        CHUNK_MIN_TOKENS="150",
        CHUNK_MAX_TOKENS="220",
        CHUNK_OVERLAP_MIN_TOKENS="60",
        CHUNK_OVERLAP_MAX_TOKENS="160",
    )

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_worker_backoff_ceiling_must_be_at_least_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(
        monkeypatch,
        WORKER_BACKOFF_BASE_SECONDS="10",
        WORKER_BACKOFF_MAX_SECONDS="5",
    )

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_pinecone_retry_ceiling_must_be_at_least_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(
        monkeypatch,
        PINECONE_RETRY_BASE_SECONDS="10",
        PINECONE_RETRY_MAX_SECONDS="5",
    )

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_openai_retry_ceiling_must_be_at_least_base(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(
        monkeypatch,
        OPENAI_RETRY_BASE_SECONDS="10",
        OPENAI_RETRY_MAX_SECONDS="5",
    )

    with pytest.raises(ValidationError):
        Settings(_env_file=None)


def test_embedding_dimension_and_index_schema_version_are_configurable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _set_required(
        monkeypatch,
        EMBEDDING_DIMENSION="768",
        INDEX_SCHEMA_VERSION="3",
    )

    settings = Settings(_env_file=None)

    assert settings.embedding_dimension == 768
    assert settings.index_schema_version == 3


def test_get_settings_is_cached(monkeypatch: pytest.MonkeyPatch) -> None:
    _set_required(monkeypatch)
    get_settings.cache_clear()
    try:
        assert get_settings() is get_settings()
    finally:
        get_settings.cache_clear()
