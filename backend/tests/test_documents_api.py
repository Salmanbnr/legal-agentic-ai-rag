"""API tests for secure document upload and document/job retrieval."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.api import documents as documents_api
from app.auth.dependencies import CurrentUser, get_current_user
from app.core.config import get_settings
from app.core.rate_limit import InMemoryRateLimiter
from app.db.client import get_db_client
from app.db.repos import Case, Document, ProcessingJob
from app.main import create_app
from scripts.make_test_pdfs import GeneratedPdfs, generate_test_pdfs

TIMESTAMP = "2026-01-01T00:00:00+00:00"
REQUEST_ID = "req-document-test"


def _case(case_id: UUID, owner_id: UUID) -> Case:
    return Case.model_validate(
        {
            "id": str(case_id),
            "owner_user_id": str(owner_id),
            "title": "Test case",
            "description": None,
            "status": "active",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    )


def _document(
    case_id: UUID,
    owner_id: UUID,
    document_id: UUID | None = None,
    **overrides: Any,
) -> Document:
    resolved_id = document_id or uuid4()
    row: dict[str, Any] = {
        "id": str(resolved_id),
        "case_id": str(case_id),
        "owner_user_id": str(owner_id),
        "storage_path": f"{owner_id}/{case_id}/{resolved_id}",
        "original_filename": "clean.pdf",
        "detected_mime": "application/pdf",
        "sha256": "a" * 64,
        "size_bytes": 1024,
        "page_count": 1,
        "document_type": None,
        "version_group_id": None,
        "duplicate_of": None,
        "quality_status": "pending",
        "quality_report": {
            "pages_total": 9,
            "pages_extracted": 9,
            "pages_empty": 0,
            "empty_pages": [],
            "low_confidence_pages": [4, 7],
            "repeated_text_ratio": 0.01,
            "parser_errors": [],
            "pipeline_version": "1.0.0",
        },
        "processing_status": "uploaded",
        "parser_version": None,
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    }
    row.update(overrides)
    return Document.model_validate(row)


def _job(
    case_id: UUID,
    document_id: UUID,
    job_id: UUID | None = None,
    **overrides: Any,
) -> ProcessingJob:
    resolved_id = job_id or uuid4()
    row: dict[str, Any] = {
        "id": str(resolved_id),
        "case_id": str(case_id),
        "document_id": str(document_id),
        "job_type": "ingest",
        "idempotency_key": f"{document_id}:1.0.0",
        "status": "queued",
        "attempts": 0,
        "error_code": None,
        "error_summary": "internal summary",
        "pipeline_version": "1.0.0",
        "started_at": None,
        "finished_at": None,
        "created_at": TIMESTAMP,
    }
    row.update(overrides)
    return ProcessingJob.model_validate(row)


@pytest.fixture()
def app():
    get_settings.cache_clear()
    return create_app()


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


@pytest.fixture()
def pdfs(tmp_path: Path) -> GeneratedPdfs:
    return generate_test_pdfs(tmp_path / "pdfs")


def _wire(app, user: CurrentUser, db: MagicMock | None = None) -> None:
    app.dependency_overrides[get_current_user] = lambda: user
    resolved_db = db if db is not None else MagicMock()
    app.dependency_overrides[get_db_client] = lambda: resolved_db


def _post_upload(
    client: TestClient,
    case_id: UUID,
    data: bytes,
    filename: str = "clean.pdf",
):
    return client.post(
        f"/api/v1/cases/{case_id}/documents",
        headers={"X-Request-ID": REQUEST_ID},
        files={"file": (filename, data, "application/pdf")},
    )


def _patch_successful_upload(
    monkeypatch: pytest.MonkeyPatch,
    user_id: UUID,
    case_id: UUID,
    document_id: UUID,
    job_id: UUID,
    *,
    existing: Document | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case = _case(case_id, user_id)
    storage_path = f"{user_id}/{case_id}/{uuid4()}"
    captured: dict[str, Any] = {"storage_path": storage_path}

    monkeypatch.setattr(
        documents_api, "require_case_access", lambda *args, **kwargs: case
    )
    monkeypatch.setattr(
        documents_api,
        "find_document_by_sha256",
        lambda *args, **kwargs: existing,
    )

    def fake_upload(
        owner_id: UUID,
        target_case_id: UUID,
        data: bytes,
        *,
        content_type: str = "application/octet-stream",
        client=None,
    ) -> str:
        captured["upload"] = {
            "owner_id": owner_id,
            "case_id": target_case_id,
            "data": data,
            "content_type": content_type,
        }
        return storage_path

    monkeypatch.setattr(documents_api, "upload_original", fake_upload)

    def fake_create_document(
        client, owner_id: UUID, target_case_id: UUID, data
    ) -> Document:
        captured["document_create"] = data
        return _document(
            target_case_id,
            owner_id,
            document_id,
            storage_path=data.storage_path,
            detected_mime=data.detected_mime,
            sha256=data.sha256,
            size_bytes=data.size_bytes,
            page_count=data.page_count,
            duplicate_of=data.duplicate_of,
            quality_status=data.quality_status or "pending",
            quality_report=data.quality_report or {},
        )

    monkeypatch.setattr(documents_api, "create_document", fake_create_document)

    def fake_create_job(
        client, owner_id: UUID, target_case_id: UUID, target_document_id: UUID, data
    ) -> ProcessingJob:
        captured["job_create"] = data
        return _job(target_case_id, target_document_id, job_id)

    monkeypatch.setattr(documents_api, "create_job", fake_create_job)

    audits: list[dict[str, Any]] = []
    monkeypatch.setattr(
        documents_api,
        "write_audit_event",
        lambda *args, **kwargs: audits.append(kwargs),
    )
    return captured, audits


def test_clean_pdf_upload_creates_document_job_and_audit(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch, pdfs: GeneratedPdfs
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    captured, audits = _patch_successful_upload(
        monkeypatch, user_id, case_id, document_id, job_id
    )

    response = _post_upload(client, case_id, pdfs.clean_native.read_bytes())

    assert response.status_code == 201
    body = response.json()
    assert body["document_id"] == str(document_id)
    assert body["job_id"] == str(job_id)
    assert body["quality_status"] == "pending"
    assert body["error_code"] is None

    created = captured["document_create"]
    assert created.storage_path == captured["storage_path"]
    assert created.original_filename == "clean.pdf"
    assert created.sha256 is not None and len(created.sha256) == 64
    assert created.page_count == 1
    assert created.duplicate_of is None
    assert captured["upload"]["content_type"] == "application/pdf"
    assert captured["job_create"].job_type == "ingest"
    assert captured["job_create"].idempotency_key == f"{document_id}:1.0.0"

    assert audits[0]["event_type"] == "document.uploaded"
    assert audits[0]["target_id"] == str(document_id)
    assert audits[0]["request_id"] == REQUEST_ID
    assert audits[0]["safe_metadata"]["quality_status"] == "pending"
    assert "original_filename" not in audits[0]["safe_metadata"]


def test_duplicate_upload_sets_duplicate_of_and_still_queues_job(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch, pdfs: GeneratedPdfs
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    existing_id = uuid4()
    root_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    existing = _document(case_id, user_id, existing_id, duplicate_of=root_id)
    _wire(app, user)
    captured, _ = _patch_successful_upload(
        monkeypatch,
        user_id,
        case_id,
        document_id,
        job_id,
        existing=existing,
    )

    response = _post_upload(client, case_id, pdfs.clean_native.read_bytes())

    assert response.status_code == 201
    assert response.json()["duplicate_of"] == str(root_id)
    assert captured["document_create"].duplicate_of == root_id
    assert captured["job_create"].job_type == "ingest"


def test_encrypted_pdf_is_stored_without_job(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch, pdfs: GeneratedPdfs
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    captured, audits = _patch_successful_upload(
        monkeypatch, user_id, case_id, document_id, job_id
    )
    job_calls: list[object] = []

    def fail_if_called(*args: object, **kwargs: object) -> None:
        job_calls.append(args)
        raise AssertionError("Encrypted uploads must not create a job")

    monkeypatch.setattr(documents_api, "create_job", fail_if_called)

    response = _post_upload(
        client, case_id, pdfs.encrypted.read_bytes(), "encrypted.pdf"
    )

    assert response.status_code == 201
    body = response.json()
    assert body["job_id"] is None
    assert body["quality_status"] == "password_required"
    assert body["error_code"] == "PASSWORD_REQUIRED"
    assert job_calls == []
    assert captured["document_create"].quality_status == "password_required"
    assert captured["document_create"].quality_report == {
        "error_code": "PASSWORD_REQUIRED"
    }
    assert audits[0]["safe_metadata"]["job_id"] is None


def test_wrong_magic_bytes_are_rejected_before_storage(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    monkeypatch.setattr(
        documents_api,
        "require_case_access",
        lambda *args, **kwargs: _case(case_id, user_id),
    )
    monkeypatch.setattr(
        documents_api,
        "upload_original",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Storage must not be touched")
        ),
    )

    response = _post_upload(client, case_id, b"not a pdf", "fake.pdf")

    assert response.status_code == 415
    assert response.json()["detail"] == "Unsupported or invalid file type"


def test_user_cannot_upload_into_another_users_case(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch, pdfs: GeneratedPdfs
) -> None:
    owner_id = uuid4()
    attacker_id = uuid4()
    case_id = uuid4()
    user = CurrentUser(id=attacker_id, email="attacker@example.com")
    cases = MagicMock()
    cases.select.return_value.eq.return_value.limit.return_value.execute.return_value.data = [
        {
            "id": str(case_id),
            "owner_user_id": str(owner_id),
            "title": "Owner case",
            "description": None,
            "status": "active",
            "created_at": TIMESTAMP,
            "updated_at": TIMESTAMP,
        }
    ]
    db = MagicMock()
    db.table.return_value = cases
    _wire(app, user, db)
    monkeypatch.setattr(
        documents_api,
        "upload_original",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Storage must not be touched")
        ),
    )

    response = _post_upload(client, case_id, pdfs.clean_native.read_bytes())

    assert response.status_code == 404
    assert response.json()["detail"] == "Case not found"


def test_list_documents_uses_pagination(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    captured: dict[str, Any] = {}

    def fake_list(
        client, owner_id, target_case_id, *, processing_status=None, limit=100, offset=0
    ):
        captured.update(
            {
                "owner_id": owner_id,
                "case_id": target_case_id,
                "limit": limit,
                "offset": offset,
            }
        )
        return [_document(case_id, user_id, document_id)]

    monkeypatch.setattr(documents_api, "list_documents", fake_list)

    response = client.get(
        f"/api/v1/cases/{case_id}/documents?limit=10&offset=5"
    )

    assert response.status_code == 200
    assert response.json()[0]["id"] == str(document_id)
    assert captured == {
        "owner_id": user_id,
        "case_id": case_id,
        "limit": 10,
        "offset": 5,
    }
    body = response.json()[0]
    assert "storage_path" not in body
    assert "owner_user_id" not in body
    # The owner sees the countable quality facts, never the internal report.
    report = body["quality_report"]
    assert set(report) <= {
        "pages_total",
        "pages_extracted",
        "pages_empty",
        "empty_pages",
        "low_confidence_pages",
        "repeated_text_ratio",
        "parser_errors",
    }
    assert report["low_confidence_pages"] == [4, 7]
    assert "pipeline_version" not in report
    assert "indexing" not in report


def test_get_document_returns_safe_metadata(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    document = _document(case_id, user_id, document_id)
    monkeypatch.setattr(
        documents_api,
        "require_document_access",
        lambda *args, **kwargs: document,
    )

    response = client.get(f"/api/v1/documents/{document_id}")

    assert response.status_code == 200
    assert response.json()["id"] == str(document_id)
    assert "storage_path" not in response.json()


def test_get_job_returns_safe_state(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    job = _job(case_id, document_id, job_id)
    monkeypatch.setattr(
        documents_api, "require_job_access", lambda *args, **kwargs: job
    )

    response = client.get(f"/api/v1/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["id"] == str(job_id)
    assert "error_summary" not in response.json()
    assert "idempotency_key" not in response.json()


def test_delete_marks_document_and_enqueues_delete_job(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    document = _document(
        case_id, user_id, document_id, quality_report={"previous": True}
    )
    updated = _document(
        case_id,
        user_id,
        document_id,
        quality_status="needs_review",
        quality_report={"previous": True, "deletion_requested": True},
    )
    delete_job = _job(
        case_id,
        document_id,
        job_id,
        job_type="delete",
        idempotency_key=f"{document_id}:delete:1.0.0",
    )
    captured: dict[str, Any] = {}
    audits: list[dict[str, Any]] = []

    monkeypatch.setattr(
        documents_api,
        "require_document_access",
        lambda *args, **kwargs: document,
    )

    def fake_update(client, owner_id, target_document_id, data):
        captured["update"] = data
        return updated

    monkeypatch.setattr(documents_api, "update_document", fake_update)
    monkeypatch.setattr(
        documents_api,
        "find_job_by_idempotency_key",
        lambda *args, **kwargs: None,
    )

    def fake_create_job(
        client, owner_id, target_case_id, target_document_id, data
    ):
        captured["job_create"] = data
        return delete_job

    monkeypatch.setattr(documents_api, "create_job", fake_create_job)
    monkeypatch.setattr(
        documents_api,
        "write_audit_event",
        lambda *args, **kwargs: audits.append(kwargs),
    )
    monkeypatch.setattr(
        documents_api,
        "delete_original",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Deletion must wait for the worker")
        ),
    )

    response = client.delete(
        f"/api/v1/documents/{document_id}",
        headers={"X-Request-ID": REQUEST_ID},
    )

    assert response.status_code == 202
    assert response.json() == {
        "document_id": str(document_id),
        "job_id": str(job_id),
        "deletion_requested": True,
    }
    assert captured["update"].quality_status == "needs_review"
    assert captured["update"].processing_status == "needs_review"
    assert captured["update"].quality_report["deletion_requested"] is True
    assert captured["job_create"].job_type == "delete"
    assert captured["job_create"].idempotency_key == f"{document_id}:delete:1.0.0"
    assert audits[0]["event_type"] == "document.deletion_requested"
    assert audits[0]["safe_metadata"] == {
        "job_id": str(job_id),
        "deletion_requested": True,
    }


def test_reindex_creates_a_reindex_job(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    document = _document(
        case_id,
        user_id,
        document_id,
        processing_status="ready",
        quality_status="ready",
    )
    reindex_job = _job(
        case_id,
        document_id,
        job_id,
        job_type="reindex",
        idempotency_key="reindex-key",
    )
    captured: dict[str, Any] = {}
    audits: list[dict[str, Any]] = []

    monkeypatch.setattr(
        documents_api,
        "require_document_access",
        lambda *args, **kwargs: document,
    )
    monkeypatch.setattr(
        documents_api,
        "find_job_by_idempotency_key",
        lambda *args, **kwargs: None,
    )

    def fake_create_job(
        client, owner_id, target_case_id, target_document_id, data
    ) -> ProcessingJob:
        captured["job_create"] = data
        return reindex_job

    monkeypatch.setattr(documents_api, "create_job", fake_create_job)
    monkeypatch.setattr(
        documents_api,
        "write_audit_event",
        lambda *args, **kwargs: audits.append(kwargs),
    )

    response = client.post(
        f"/api/v1/documents/{document_id}/reindex",
        headers={"X-Request-ID": REQUEST_ID},
    )

    assert response.status_code == 202
    assert response.json() == {
        "document_id": str(document_id),
        "job_id": str(job_id),
        "job_type": "reindex",
    }
    assert captured["job_create"].job_type == "reindex"
    assert captured["job_create"].idempotency_key == (
        f"{document_id}:reindex:1.0.0:"
        "sentence-transformers/all-MiniLM-L6-v2:1"
    )
    assert audits[0]["event_type"] == "document.reindex_requested"
    assert audits[0]["safe_metadata"]["job_id"] == str(job_id)


def test_reindex_reuses_an_existing_job(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    job_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    document = _document(case_id, user_id, document_id)
    existing = _job(case_id, document_id, job_id, job_type="reindex")

    monkeypatch.setattr(
        documents_api,
        "require_document_access",
        lambda *args, **kwargs: document,
    )
    monkeypatch.setattr(
        documents_api,
        "find_job_by_idempotency_key",
        lambda *args, **kwargs: existing,
    )
    monkeypatch.setattr(
        documents_api,
        "create_job",
        lambda *args, **kwargs: pytest.fail("Job must not be created twice"),
    )
    monkeypatch.setattr(
        documents_api, "write_audit_event", lambda *args, **kwargs: None
    )

    response = client.post(f"/api/v1/documents/{document_id}/reindex")

    assert response.status_code == 202
    assert response.json()["job_id"] == str(job_id)


def test_reindex_is_refused_while_deletion_is_requested(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    document_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    document = _document(
        case_id,
        user_id,
        document_id,
        quality_report={"deletion_requested": True},
    )

    monkeypatch.setattr(
        documents_api,
        "require_document_access",
        lambda *args, **kwargs: document,
    )
    monkeypatch.setattr(
        documents_api,
        "create_job",
        lambda *args, **kwargs: pytest.fail("Job must not be created"),
    )

    response = client.post(f"/api/v1/documents/{document_id}/reindex")

    assert response.status_code == 409


def test_upload_rate_limit_returns_429(
    app, client: TestClient, monkeypatch: pytest.MonkeyPatch, pdfs: GeneratedPdfs
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    _wire(app, user)
    _patch_successful_upload(
        monkeypatch, user_id, case_id, uuid4(), uuid4()
    )
    app.state.upload_rate_limiter = InMemoryRateLimiter(
        limit=1, window_seconds=60
    )

    first = _post_upload(client, case_id, pdfs.clean_native.read_bytes())
    second = _post_upload(client, case_id, pdfs.clean_native.read_bytes())

    assert first.status_code == 201
    assert second.status_code == 429
    assert second.headers["Retry-After"]
