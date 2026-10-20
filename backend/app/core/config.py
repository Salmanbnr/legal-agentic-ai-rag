"""Application configuration loaded from environment variables.

Secrets are wrapped in :class:`pydantic.SecretStr` so they are redacted from
representations and serialized output. Validation errors are rewritten to name
the offending environment variables without echoing their values.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AnyHttpUrl, Field, SecretStr, ValidationError, field_validator
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

    supabase_url: AnyHttpUrl
    supabase_publishable_key: SecretStr = Field(min_length=1)
    supabase_service_role_key: SecretStr = Field(min_length=1)

    openai_api_key: SecretStr = Field(min_length=1)
    openai_model: str = Field(min_length=1)

    pinecone_api_key: SecretStr = Field(min_length=1)
    pinecone_index_name: str = "legal-evidence"

    embedding_model: str = Field(min_length=1)
    embedding_device: Literal["cpu", "cuda", "mps"] = "cpu"
    model_cache_dir: Path = Path(".cache/huggingface")
    temp_dir: Path = Path(".tmp")

    max_upload_mb: int = Field(default=50, ge=1, le=1024)
    max_pdf_pages: int = Field(default=1000, ge=1, le=10000)

    agent_max_steps: int = Field(default=12, ge=1, le=100)
    agent_max_retrieval_retries: int = Field(default=2, ge=0, le=10)
    agent_max_llm_calls: int = Field(default=12, ge=1, le=100)

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
