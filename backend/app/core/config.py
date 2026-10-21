"""Application configuration loaded from environment variables.

Secrets are wrapped in :class:`pydantic.SecretStr` so they are redacted from
representations and serialized output. Validation errors are rewritten to name
the offending environment variables without echoing their values.
"""

from __future__ import annotations

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import (
    AnyHttpUrl,
    Field,
    SecretStr,
    ValidationError,
    field_validator,
    model_validator,
)
from pydantic_settings import BaseSettings, SettingsConfigDict

BACKEND_DIR = Path(__file__).resolve().parents[2]
ENV_FILE = BACKEND_DIR / ".env"


class ConfigurationError(RuntimeError):
    """Raised when required configuration is missing or invalid."""


class Settings(BaseSettings):
    """Validated runtime configuration for the backend."""

    model_config = SettingsConfigDict(
        env_file=ENV_FILE,
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    app_env: Literal["development", "test", "staging", "production"] = "development"
    api_prefix: str = "/api/v1"

    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    log_file: Path | None = None

    supabase_url: AnyHttpUrl
    supabase_publishable_key: SecretStr = Field(min_length=1)
    supabase_service_role_key: SecretStr = Field(min_length=1)

    openai_api_key: SecretStr = Field(min_length=1)
    openai_model: str = Field(min_length=1)
    openai_timeout_seconds: float = Field(default=60.0, ge=1.0, le=600.0)
    openai_max_retries: int = Field(default=3, ge=0, le=10)
    openai_retry_base_seconds: float = Field(default=1.0, ge=0.01, le=60.0)
    openai_retry_max_seconds: float = Field(default=20.0, ge=0.1, le=300.0)
    openai_max_output_tokens: int = Field(default=1500, ge=64, le=32768)

    pinecone_api_key: SecretStr = Field(min_length=1)
    pinecone_index_name: str = "legal-evidence"
    pinecone_cloud: str = Field(default="aws", min_length=1, max_length=32)
    pinecone_region: str = Field(default="us-east-1", min_length=1, max_length=64)
    pinecone_metric: Literal["cosine", "dotproduct", "euclidean"] = "cosine"
    pinecone_upsert_batch_size: int = Field(default=100, ge=1, le=1000)
    pinecone_delete_batch_size: int = Field(default=1000, ge=1, le=1000)
    pinecone_retry_attempts: int = Field(default=4, ge=1, le=10)
    pinecone_retry_base_seconds: float = Field(default=0.5, ge=0.01, le=60.0)
    pinecone_retry_max_seconds: float = Field(default=8.0, ge=0.1, le=300.0)

    embedding_model: str = Field(min_length=1)
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    embedding_dimension: int = Field(default=384, ge=1, le=8192)
    embedding_batch_size: int = Field(default=32, ge=1, le=512)
    model_cache_dir: Path = Path(".cache/huggingface")
    temp_dir: Path = Path(".tmp")

    max_upload_mb: int = Field(default=50, ge=1, le=1024)
    max_pdf_pages: int = Field(default=1000, ge=1, le=10000)
    upload_rate_limit_per_minute: int = Field(default=30, ge=1, le=1000)
    ingest_pipeline_version: str = Field(
        default="1.0.0", min_length=1, max_length=50
    )

    pdf_native_text_density_threshold: float = Field(
        default=0.0002, ge=0.0, le=1.0
    )

    ocr_language: str = Field(default="eng", min_length=1, max_length=32)
    ocr_dpi: int = Field(default=300, ge=72, le=600)
    ocr_confidence_threshold: float = Field(default=70.0, ge=0.0, le=100.0)
    ocr_preprocess_enabled: bool = True
    ocr_handwriting_confidence_threshold: float = Field(
        default=35.0, ge=0.0, le=100.0
    )
    ocr_handwriting_max_words: int = Field(default=5, ge=0, le=100)
    ocr_table_min_rows: int = Field(default=3, ge=1, le=50)
    ocr_max_deskew_degrees: float = Field(default=15.0, ge=0.0, le=45.0)
    tesseract_cmd: str | None = None

    quality_review_ratio_threshold: float = Field(default=0.30, ge=0.0, le=1.0)
    quality_parser_error_threshold: int = Field(default=3, ge=1, le=100)

    header_footer_repeat_ratio: float = Field(default=0.60, ge=0.0, le=1.0)
    header_footer_max_lines: int = Field(default=3, ge=1, le=20)

    chunk_min_tokens: int = Field(default=150, ge=1, le=10000)
    chunk_max_tokens: int = Field(default=220, ge=1, le=10000)
    chunk_overlap_min_tokens: int = Field(default=20, ge=1, le=1000)
    chunk_overlap_max_tokens: int = Field(default=40, ge=1, le=1000)
    chunk_tokenizer_local_files_only: bool = False
    index_schema_version: int = Field(default=1, ge=1, le=100000)

    retrieval_candidate_multiplier: int = Field(default=4, ge=1, le=20)
    retrieval_max_spans_per_document: int = Field(default=3, ge=1, le=50)
    retrieval_neighbor_expansion: bool = True

    startup_checks_enabled: bool = True

    worker_enabled: bool = True
    worker_poll_seconds: float = Field(default=2.0, ge=0.25, le=300.0)
    worker_max_attempts: int = Field(default=3, ge=1, le=3)
    worker_backoff_base_seconds: float = Field(default=2.0, ge=0.1, le=300.0)
    worker_backoff_max_seconds: float = Field(
        default=60.0, ge=0.1, le=3600.0
    )

    agent_max_steps: int = Field(default=24, ge=1, le=100)
    agent_max_retrieval_retries: int = Field(default=2, ge=0, le=10)
    agent_max_llm_calls: int = Field(default=12, ge=1, le=100)
    agent_max_tokens: int = Field(default=20000, ge=100, le=1000000)
    agent_timeout_seconds: int = Field(default=60, ge=1, le=600)
    agent_skip_analyst_on_strong_fact_lookup: bool = True
    agent_analyst_min_spans: int = Field(default=3, ge=1, le=50)
    agent_analyst_max_spans: int = Field(default=12, ge=1, le=50)
    agent_research_max_multi_hop: int = Field(default=1, ge=0, le=3)
    answer_max_evidence_spans: int = Field(default=10, ge=1, le=50)
    answer_max_question_length: int = Field(default=2000, ge=1, le=20000)
    answer_max_llm_calls: int = Field(default=1, ge=1, le=10)

    frontend_origin: AnyHttpUrl

    @field_validator("api_prefix")
    @classmethod
    def validate_api_prefix(cls, value: str) -> str:
        """Require a rooted prefix without a trailing slash."""
        if not value.startswith("/"):
            raise ValueError("API_PREFIX must start with '/'")
        if value != "/" and value.endswith("/"):
            raise ValueError("API_PREFIX must not end with '/'")
        return value

    @field_validator("pinecone_index_name")
    @classmethod
    def validate_index_name(cls, value: str) -> str:
        """Enforce the Pinecone index name character set and length."""
        if not value or len(value) > 45:
            raise ValueError("PINECONE_INDEX_NAME must be 1-45 characters")
        if not all(char.islower() or char.isdigit() or char == "-" for char in value):
            raise ValueError(
                "PINECONE_INDEX_NAME may contain only lowercase letters, digits, and hyphens"
            )
        return value

    @field_validator("frontend_origin")
    @classmethod
    def validate_frontend_origin(cls, value: AnyHttpUrl) -> AnyHttpUrl:
        """Require a bare origin suitable for CORS, not a deep link."""
        if value.path not in ("", "/") or value.query or value.fragment:
            raise ValueError(
                "FRONTEND_ORIGIN must be a bare origin without a path, query, or fragment"
            )
        return value

    @field_validator("model_cache_dir", "temp_dir", mode="after")
    @classmethod
    def resolve_backend_path(cls, value: Path) -> Path:
        """Resolve relative storage paths against the backend directory."""
        if value.is_absolute():
            return value
        return (BACKEND_DIR / value).resolve()

    @field_validator("log_level", mode="before")
    @classmethod
    def normalise_log_level(cls, value: str) -> str:
        """Accept a lower-case level name from the environment."""
        if isinstance(value, str):
            return value.strip().upper()
        return value

    @field_validator("log_file", mode="before")
    @classmethod
    def empty_log_file_is_none(cls, value: str | None) -> str | None:
        """Treat an empty log path as "console only"."""
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("log_file", mode="after")
    @classmethod
    def resolve_log_file(cls, value: Path | None) -> Path | None:
        """Resolve a relative log path against the backend directory."""
        if value is None or value.is_absolute():
            return value
        return (BACKEND_DIR / value).resolve()

    @field_validator("ocr_language")
    @classmethod
    def validate_ocr_language(cls, value: str) -> str:
        """Accept Tesseract language codes such as ``eng`` or ``eng+fra``."""
        if not re.fullmatch(r"[A-Za-z0-9_+-]+", value):
            raise ValueError(
                "OCR_LANGUAGE may contain only letters, digits, underscore, plus, and hyphen"
            )
        return value

    @field_validator("tesseract_cmd", mode="before")
    @classmethod
    def empty_tesseract_cmd_is_none(cls, value: str | None) -> str | None:
        """Treat an empty optional command as "use PATH"."""
        if isinstance(value, str):
            return value.strip() or None
        return value

    @field_validator("tesseract_cmd", mode="after")
    @classmethod
    def validate_tesseract_cmd(cls, value: str | None) -> str | None:
        """Fail fast when an explicitly configured Tesseract path is missing."""
        if value is not None and not Path(value).is_file():
            raise ValueError("TESSERACT_CMD must point to an existing file")
        return value

    @model_validator(mode="after")
    def validate_worker_backoff(self) -> "Settings":
        """Require the backoff ceiling to be at least the base delay."""
        if self.worker_backoff_max_seconds < self.worker_backoff_base_seconds:
            raise ValueError(
                "WORKER_BACKOFF_MAX_SECONDS must be greater than or equal to "
                "WORKER_BACKOFF_BASE_SECONDS"
            )
        return self

    @model_validator(mode="after")
    def validate_pinecone_backoff(self) -> "Settings":
        """Require the Pinecone retry ceiling to be at least the base delay."""
        if self.pinecone_retry_max_seconds < self.pinecone_retry_base_seconds:
            raise ValueError(
                "PINECONE_RETRY_MAX_SECONDS must be greater than or equal to "
                "PINECONE_RETRY_BASE_SECONDS"
            )
        return self

    @model_validator(mode="after")
    def validate_openai_backoff(self) -> "Settings":
        """Require the OpenAI retry ceiling to be at least the base delay."""
        if self.openai_retry_max_seconds < self.openai_retry_base_seconds:
            raise ValueError(
                "OPENAI_RETRY_MAX_SECONDS must be greater than or equal to "
                "OPENAI_RETRY_BASE_SECONDS"
            )
        return self

    @model_validator(mode="after")
    def validate_chunk_sizes(self) -> "Settings":
        """Keep chunk sizes ordered and leave room for the configured overlap."""
        if self.chunk_min_tokens > self.chunk_max_tokens:
            raise ValueError(
                "CHUNK_MIN_TOKENS must be less than or equal to CHUNK_MAX_TOKENS"
            )
        if self.chunk_overlap_min_tokens > self.chunk_overlap_max_tokens:
            raise ValueError(
                "CHUNK_OVERLAP_MIN_TOKENS must be less than or equal to "
                "CHUNK_OVERLAP_MAX_TOKENS"
            )
        if self.chunk_overlap_max_tokens >= self.chunk_min_tokens:
            raise ValueError(
                "CHUNK_OVERLAP_MAX_TOKENS must be less than CHUNK_MIN_TOKENS"
            )
        if self.chunk_max_tokens - self.chunk_overlap_min_tokens < self.chunk_min_tokens:
            raise ValueError(
                "CHUNK_MAX_TOKENS minus CHUNK_OVERLAP_MIN_TOKENS must be at "
                "least CHUNK_MIN_TOKENS"
            )
        return self

    @property
    def frontend_origin_value(self) -> str:
        """Return the configured frontend origin without a trailing slash."""
        return str(self.frontend_origin).rstrip("/")

    @property
    def supabase_url_value(self) -> str:
        """Return the configured Supabase URL without a trailing slash."""
        return str(self.supabase_url).rstrip("/")


def _describe_validation_error(exc: ValidationError) -> str:
    """Describe invalid fields by name only, never by submitted value."""
    details: list[str] = []
    for error in exc.errors(include_url=False):
        location = ".".join(str(part) for part in error.get("loc", ()) if part != "settings")
        field_name = location.upper() if location else "CONFIG"
        details.append(f"{field_name}: {error.get('msg', 'invalid value')}")
    return "; ".join(details) or "unknown configuration error"


def load_settings(env_file: Path | str | None = ENV_FILE) -> Settings:
    """Load settings from the environment and optional dotenv file.

    Raises:
        ConfigurationError: If required values are missing or invalid. The
            message names the affected variables but never includes values.
    """
    try:
        return Settings(_env_file=env_file)
    except ValidationError as exc:
        raise ConfigurationError(
            f"Invalid or missing configuration: {_describe_validation_error(exc)}"
        ) from None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide validated settings instance."""
    return load_settings()
