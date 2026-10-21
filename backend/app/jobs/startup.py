"""Startup check that keeps the durable job queue aligned with the code."""

from __future__ import annotations

import logging

from postgrest.exceptions import APIError
from supabase import Client

from app.core.config import Settings, get_settings
from app.db.client import get_service_client

logger = logging.getLogger(__name__)

# The migration that adds the retry schedule the worker claims jobs with.
JOB_SCHEDULING_MIGRATION = "supabase/migrations/0002_job_scheduling.sql"

# PostgREST reports a missing column with this SQLSTATE.
UNDEFINED_COLUMN = "42703"


class JobSchemaError(RuntimeError):
    """Raised when the job queue schema is behind the running code."""


def validate_job_schema(
    settings: Settings | None = None,
    client: Client | None = None,
) -> None:
    """Fail fast when `processing_jobs` lacks the columns the worker needs.

    The worker claims work with a filter on `next_attempt_at`. If that column
    is missing, every poll fails and the queue silently stops moving, so the
    worker checks the queue at startup and names the migration to apply
    instead of logging the same failure forever.
    """
    resolved = settings or get_settings()
    if resolved.app_env == "test" or not resolved.startup_checks_enabled:
        return

    resolved_client = client or get_service_client(resolved)
    try:
        (
            resolved_client.table("processing_jobs")
            .select("id,next_attempt_at")
            .limit(1)
            .execute()
        )
    except APIError as exc:
        if _error_code(exc) != UNDEFINED_COLUMN:
            raise
        raise JobSchemaError(
            "The processing_jobs table is missing next_attempt_at, so the "
            f"worker cannot claim jobs. Apply {JOB_SCHEDULING_MIGRATION} "
            "before starting the API."
        ) from exc

    logger.info("job queue schema verified")


def _error_code(exc: APIError) -> str | None:
    """Return the SQLSTATE of a PostgREST error, if it carries one."""
    code = getattr(exc, "code", None)
    if isinstance(code, str):
        return code
    payload = getattr(exc, "json", None)
    if callable(payload):
        body = payload()
        if isinstance(body, dict) and isinstance(body.get("code"), str):
            return body["code"]
    return None
