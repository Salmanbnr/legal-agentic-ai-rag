"""Durable ingestion-job runner used by the API lifespan and CLI worker."""

from __future__ import annotations

import logging
import random
import threading
from datetime import datetime, timedelta, timezone
from typing import Callable

from supabase import Client

from app.core.config import Settings, get_settings
from app.db.client import get_service_client
from app.db.repos import ProcessingJob
from app.indexing.indexer import (
    DeleteResult,
    IndexResult,
    IndexingError,
    run_delete_document,
    run_reindex_document,
)
from app.ingestion.pipeline import (
    PIPELINE_VERSION,
    PipelineError,
    PipelineOutcome,
    run_ingest_pipeline,
)
from app.jobs.store import (
    claim_next_job,
    mark_document_failed,
    mark_job_failed,
    mark_job_succeeded,
    schedule_job_retry,
)

logger = logging.getLogger(__name__)


def calculate_backoff(
    attempt: int,
    *,
    base_seconds: float,
    max_seconds: float,
    rng: Callable[[float, float], float] = random.uniform,
) -> float:
    """Return a capped exponential delay with 50%-150% jitter."""
    if attempt < 1:
        raise ValueError("attempt must be at least 1")
    if base_seconds <= 0 or max_seconds <= 0:
        raise ValueError("backoff values must be positive")
    nominal = min(max_seconds, base_seconds * (2 ** (attempt - 1)))
    return min(max_seconds, nominal * rng(0.5, 1.5))


class JobWorker:
    """Poll, claim, and execute queued ingest, reindex, and delete jobs."""

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        client: Client | None = None,
        stop_event: threading.Event | None = None,
        pipeline: Callable[..., PipelineOutcome] = run_ingest_pipeline,
        reindex_pipeline: Callable[..., IndexResult] = run_reindex_document,
        delete_pipeline: Callable[..., DeleteResult] = run_delete_document,
        now_provider: Callable[[], datetime] | None = None,
        rng: Callable[[float, float], float] = random.uniform,
    ) -> None:
        """Build a worker with injectable dependencies for tests."""
        self.settings = settings or get_settings()
        self.client = client or get_service_client(self.settings)
        self.stop_event = stop_event or threading.Event()
        self.pipeline = pipeline
        self.reindex_pipeline = reindex_pipeline
        self.delete_pipeline = delete_pipeline
        self.now_provider = now_provider or (
            lambda: datetime.now(timezone.utc)
        )
        self.rng = rng

    def run_forever(self) -> None:
        """Poll until the stop event is set."""
        logger.info("ingestion worker started")
        while not self.stop_event.is_set():
            try:
                processed = self.run_once()
            except Exception as exc:
                logger.error(
                    "worker poll failed",
                    extra={"error_class": type(exc).__name__},
                )
                processed = False
            if not processed:
                self.stop_event.wait(self.settings.worker_poll_seconds)
        logger.info("ingestion worker stopped")

    def run_once(self) -> bool:
        """Claim and execute at most one queued job."""
        now = self.now_provider()
        job = claim_next_job(
            self.client,
            now=now,
            max_attempts=self.settings.worker_max_attempts,
        )
        if job is None:
            return False

        logger.info(
            "processing job claimed",
            extra={
                "job_id": str(job.id),
                "document_id": str(job.document_id),
                "job_type": job.job_type,
                "attempt": job.attempts,
            },
        )
        try:
            if job.job_type == "delete":
                delete_outcome = self.delete_pipeline(
                    self.client, job, settings=self.settings
                )
                logger.info(
                    "delete job succeeded",
                    extra={
                        "job_id": str(job.id),
                        "document_id": str(job.document_id),
                        "deleted_vector_count": getattr(
                            delete_outcome, "deleted_vector_count", 0
                        ),
                    },
                )
                # The document row was removed, cascading this job row.
                return True

            runner = (
                self.reindex_pipeline
                if job.job_type == "reindex"
                else self.pipeline
            )
            outcome = runner(self.client, job, settings=self.settings)
        except (PipelineError, IndexingError) as exc:
            self._handle_failure(job, exc.error_code, retryable=exc.retryable)
        except Exception as exc:
            logger.error(
                "processing job failed",
                extra={
                    "job_id": str(job.id),
                    "job_type": job.job_type,
                    "error_class": type(exc).__name__,
                },
            )
            self._handle_failure(job, "PROCESSING_FAILED", retryable=True)
        else:
            mark_job_succeeded(
                self.client,
                job.id,
                pipeline_version=PIPELINE_VERSION,
                now=self.now_provider(),
            )
            logger.info(
                "processing job succeeded",
                extra={
                    "job_id": str(job.id),
                    "document_id": str(job.document_id),
                    "job_type": job.job_type,
                    "stage": getattr(outcome, "status", "unknown"),
                    "page_count": getattr(outcome, "page_count", 0),
                    "indexed_count": getattr(outcome, "indexed_count", 0),
                },
            )
        return True

    def _handle_failure(
        self, job: ProcessingJob, error_code: str, *, retryable: bool
    ) -> None:
        """Persist retry scheduling or final failure with safe metadata."""
        now = self.now_provider()
        exhausted = job.attempts >= self.settings.worker_max_attempts
        if not retryable or exhausted:
            mark_job_failed(
                self.client,
                job.id,
                error_code=error_code,
                now=now,
            )
            logger.error(
                "ingestion job marked failed",
                extra={
                    "job_id": str(job.id),
                    "error_code": error_code,
                    "attempt": job.attempts,
                },
            )
        else:
            delay = calculate_backoff(
                job.attempts,
                base_seconds=self.settings.worker_backoff_base_seconds,
                max_seconds=self.settings.worker_backoff_max_seconds,
                rng=self.rng,
            )
            schedule_job_retry(
                self.client,
                job.id,
                attempts=job.attempts,
                next_attempt_at=now + timedelta(seconds=delay),
                error_code=error_code,
                now=now,
            )
            logger.warning(
                "ingestion job scheduled for retry",
                extra={
                    "job_id": str(job.id),
                    "error_code": error_code,
                    "attempt": job.attempts,
                    "delay_seconds": round(delay, 2),
                },
            )

        try:
            mark_document_failed(
                self.client,
                job.document_id,
                error_code=error_code,
            )
        except Exception as exc:
            logger.error(
                "failed to mark document failed",
                extra={
                    "job_id": str(job.id),
                    "document_id": str(job.document_id),
                    "error_class": type(exc).__name__,
                },
            )
