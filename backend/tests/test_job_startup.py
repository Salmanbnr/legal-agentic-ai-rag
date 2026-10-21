"""Tests for the job queue schema startup check."""

from __future__ import annotations

import pytest
from postgrest.exceptions import APIError

from app.core.config import Settings
from app.jobs.startup import (
    JOB_SCHEDULING_MIGRATION,
    JobSchemaError,
    _error_code,
    validate_job_schema,
)


class _FakeQuery:
    """Minimal PostgREST query stub."""

    def __init__(self, error: Exception | None) -> None:
        self.error = error

    def select(self, *_args, **_kwargs) -> "_FakeQuery":
        return self

    def limit(self, *_args, **_kwargs) -> "_FakeQuery":
        return self

    def execute(self):
        if self.error is not None:
            raise self.error
        return type("Response", (), {"data": []})()


class _FakeClient:
    """Minimal Supabase client stub."""

    def __init__(self, error: Exception | None = None) -> None:
        self.error = error

    def table(self, _name: str) -> _FakeQuery:
        return _FakeQuery(self.error)


def _settings(**overrides) -> Settings:
    return Settings(_env_file=None, **overrides)


def test_missing_column_names_the_migration() -> None:
    error = APIError(
        {
            "code": "42703",
            "details": None,
            "hint": None,
            "message": "column processing_jobs.next_attempt_at does not exist",
        }
    )

    with pytest.raises(JobSchemaError) as caught:
        validate_job_schema(
            _settings(app_env="production", startup_checks_enabled=True),
            client=_FakeClient(error),
        )

    assert JOB_SCHEDULING_MIGRATION in str(caught.value)
    assert "next_attempt_at" in str(caught.value)


def test_present_schema_passes() -> None:
    validate_job_schema(
        _settings(app_env="production", startup_checks_enabled=True),
        client=_FakeClient(),
    )


def test_other_errors_are_not_reported_as_schema_drift() -> None:
    error = APIError(
        {
            "code": "42P01",
            "details": None,
            "hint": None,
            "message": "relation does not exist",
        }
    )

    with pytest.raises(APIError):
        validate_job_schema(
            _settings(app_env="production", startup_checks_enabled=True),
            client=_FakeClient(error),
        )


def test_test_environment_skips_the_check() -> None:
    validate_job_schema(
        _settings(app_env="test"),
        client=_FakeClient(APIError({"code": "42703", "message": "missing"})),
    )


def test_disabled_startup_checks_skip_the_check() -> None:
    validate_job_schema(
        _settings(app_env="production", startup_checks_enabled=False),
        client=_FakeClient(APIError({"code": "42703", "message": "missing"})),
    )


def test_error_code_reads_sqlstate() -> None:
    assert _error_code(APIError({"code": "42703", "message": "missing"})) == (
        "42703"
    )
