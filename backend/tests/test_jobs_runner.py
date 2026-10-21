"""Tests for job claiming, retries, and safe failure handling."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from app.core.config import Settings
from app.db.repos import ProcessingJob
from app.indexing.indexer import DeleteResult, IndexResult
from app.ingestion.pipeline import PIPELINE_VERSION, PipelineError, PipelineOutcome
from app.jobs import runner as runner_module
from app.jobs.runner import JobWorker, calculate_backoff

NOW = datetime(2026, 1, 1, 12, 0, tzinfo=timezone.utc)
TIMESTAMP = NOW.isoformat()


def _job(attempts: int = 1, job_type: str = "ingest") -> ProcessingJob:
    return ProcessingJob.model_validate(
        {
            "id": str(uuid4()),
            "case_id": str(uuid4()),
            "document_id": str(uuid4()),
            "job_type": job_type,
            "idempotency_key": str(uuid4()),
            "status": "running",
            "attempts": attempts,
            "error_code": None,
            "error_summary": None,
            "pipeline_version": "1.0.0",
            "started_at": TIMESTAMP,
            "finished_at": None,
            "next_attempt_at": TIMESTAMP,
            "created_at": TIMESTAMP,
        }
    )


def _worker(
    monkeypatch: pytest.MonkeyPatch,
    *,
    pipeline,
    job: ProcessingJob | None,
    reindex_pipeline=None,
    delete_pipeline=None,
) -> tuple[JobWorker, dict[str, list]]:
    calls: dict[str, list] = {
        "succeeded": [],
        "retry": [],
        "failed": [],
        "document_failed": [],
    }
    monkeypatch.setattr(runner_module, "claim_next_job", lambda *a, **k: job)
    monkeypatch.setattr(
        runner_module,
        "mark_job_succeeded",
        lambda *a, **k: calls["succeeded"].append((a, k)),
    )
    monkeypatch.setattr(
        runner_module,
        "schedule_job_retry",
        lambda *a, **k: calls["retry"].append((a, k)),
    )
    monkeypatch.setattr(
        runner_module,
        "mark_job_failed",
        lambda *a, **k: calls["failed"].append((a, k)),
    )
    monkeypatch.setattr(
        runner_module,
        "mark_document_failed",
        lambda *a, **k: calls["document_failed"].append((a, k)),
    )
    worker = JobWorker(
        settings=Settings(_env_file=None),
        client=MagicMock(),
        pipeline=pipeline,
        reindex_pipeline=reindex_pipeline
        or (lambda *a, **k: IndexResult("ready", 1, 1, None)),
        delete_pipeline=delete_pipeline
        or (lambda *a, **k: DeleteResult("deleted", 0)),
        now_provider=lambda: NOW,
        rng=lambda low, high: 1.0,
    )
    return worker, calls


def test_backoff_is_exponential_and_capped() -> None:
    assert calculate_backoff(
        1, base_seconds=2, max_seconds=60, rng=lambda a, b: 1.0
    ) == 2
    assert calculate_backoff(
        2, base_seconds=2, max_seconds=60, rng=lambda a, b: 1.0
    ) == 4
    assert calculate_backoff(
        3, base_seconds=2, max_seconds=60, rng=lambda a, b: 1.0
    ) == 8
    assert calculate_backoff(
        5, base_seconds=2, max_seconds=10, rng=lambda a, b: 1.0
    ) == 10


def test_backoff_applies_jitter() -> None:
    low = calculate_backoff(
        1, base_seconds=2, max_seconds=60, rng=lambda a, b: 0.5
    )
    high = calculate_backoff(
        1, base_seconds=2, max_seconds=60, rng=lambda a, b: 1.5
    )

    assert low == 1
    assert high == 3


def test_no_queued_job_returns_false(monkeypatch: pytest.MonkeyPatch) -> None:
    worker, calls = _worker(
        monkeypatch,
        pipeline=lambda *a, **k: PipelineOutcome("embedding", 1, 0),
        job=None,
    )

    assert worker.run_once() is False
    assert calls["succeeded"] == []


def test_successful_job_is_marked_succeeded(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(attempts=1)

    def pipeline(client, claimed_job, *, settings):
        assert claimed_job.id == job.id
        return PipelineOutcome("embedding", 2, 0)

    worker, calls = _worker(monkeypatch, pipeline=pipeline, job=job)

    assert worker.run_once() is True
    assert len(calls["succeeded"]) == 1
    _, kwargs = calls["succeeded"][0]
    assert kwargs["pipeline_version"] == PIPELINE_VERSION
    assert kwargs["now"] == NOW


def test_retryable_failure_schedules_backoff(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(attempts=1)

    def pipeline(client, claimed_job, *, settings):
        raise PipelineError("STORAGE_ERROR", retryable=True)

    worker, calls = _worker(monkeypatch, pipeline=pipeline, job=job)

    assert worker.run_once() is True
    assert calls["failed"] == []
    assert len(calls["retry"]) == 1
    _, kwargs = calls["retry"][0]
    assert kwargs["attempts"] == 1
    assert kwargs["next_attempt_at"] == NOW + timedelta(seconds=2)
    assert kwargs["error_code"] == "STORAGE_ERROR"
    assert calls["document_failed"][0][1]["error_code"] == "STORAGE_ERROR"


def test_failing_parser_marks_job_failed_on_last_attempt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(attempts=3)

    def pipeline(client, claimed_job, *, settings):
        raise PipelineError("PROCESSING_FAILED", retryable=True)

    worker, calls = _worker(monkeypatch, pipeline=pipeline, job=job)

    assert worker.run_once() is True
    assert calls["retry"] == []
    assert len(calls["failed"]) == 1
    _, kwargs = calls["failed"][0]
    assert kwargs["error_code"] == "PROCESSING_FAILED"
    assert "not a real document" not in kwargs["error_code"]


def test_nonretryable_failure_marks_job_failed_immediately(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(attempts=1)

    def pipeline(client, claimed_job, *, settings):
        raise PipelineError("PASSWORD_REQUIRED", retryable=False)

    worker, calls = _worker(monkeypatch, pipeline=pipeline, job=job)

    assert worker.run_once() is True
    assert calls["retry"] == []
    assert calls["failed"][0][1]["error_code"] == "PASSWORD_REQUIRED"


def test_unexpected_exception_uses_safe_error_code(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(attempts=3)

    def pipeline(client, claimed_job, *, settings):
        raise RuntimeError("internal parser detail")

    worker, calls = _worker(monkeypatch, pipeline=pipeline, job=job)

    assert worker.run_once() is True
    assert calls["failed"][0][1]["error_code"] == "PROCESSING_FAILED"


def test_reindex_job_uses_the_reindex_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(attempts=1, job_type="reindex")
    seen: list[str] = []

    def reindex_pipeline(client, claimed_job, *, settings):
        seen.append(claimed_job.job_type)
        return IndexResult("ready", 2, 2, None)

    worker, calls = _worker(
        monkeypatch,
        pipeline=lambda *a, **k: pytest.fail("ingest pipeline must not run"),
        job=job,
        reindex_pipeline=reindex_pipeline,
    )

    assert worker.run_once() is True
    assert seen == ["reindex"]
    assert len(calls["succeeded"]) == 1


def test_delete_job_runs_the_delete_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(attempts=1, job_type="delete")
    seen: list[str] = []

    def delete_pipeline(client, claimed_job, *, settings):
        seen.append(claimed_job.job_type)
        return DeleteResult("deleted", 3)

    worker, calls = _worker(
        monkeypatch,
        pipeline=lambda *a, **k: pytest.fail("ingest pipeline must not run"),
        job=job,
        delete_pipeline=delete_pipeline,
    )

    assert worker.run_once() is True
    assert seen == ["delete"]
    # The document row is removed, so the job row is gone with it.
    assert calls["succeeded"] == []


def test_failing_delete_job_is_marked_failed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    job = _job(attempts=1, job_type="delete")

    def delete_pipeline(client, claimed_job, *, settings):
        raise RuntimeError("storage failure")

    worker, calls = _worker(
        monkeypatch,
        pipeline=lambda *a, **k: pytest.fail("ingest pipeline must not run"),
        job=job,
        delete_pipeline=delete_pipeline,
    )

    assert worker.run_once() is True
    assert len(calls["retry"]) == 1
    assert calls["succeeded"] == []
