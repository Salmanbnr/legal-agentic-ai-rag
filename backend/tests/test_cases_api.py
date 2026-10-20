"""API tests for authenticated case CRUD and audit events."""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest
from fastapi.testclient import TestClient

from app.auth.dependencies import CurrentUser, get_current_user
from app.core.config import get_settings
from app.db.client import get_db_client
from app.main import create_app

TIMESTAMP = "2026-01-01T00:00:00+00:00"
TEST_REQUEST_ID = "req-test-case"


def _case_row(
    case_id: UUID,
    owner_id: UUID,
    *,
    title: str = "Test case",
    status: str = "active",
) -> dict[str, Any]:
    return {
        "id": str(case_id),
        "owner_user_id": str(owner_id),
        "title": title,
        "description": None,
        "status": status,
        "created_at": TIMESTAMP,
        "updated_at": TIMESTAMP,
    }


def _audit_row(
    audit_id: UUID,
    case_id: UUID,
    user_id: UUID,
    event_type: str,
    request_id: str,
) -> dict[str, Any]:
    return {
        "id": str(audit_id),
        "case_id": str(case_id),
        "actor_user_id": str(user_id),
        "event_type": event_type,
        "target_type": "case",
        "target_id": str(case_id),
        "request_id": request_id,
        "safe_metadata": {},
        "created_at": TIMESTAMP,
    }


def _table_mock(
    *,
    select: list[dict[str, Any]] | None = None,
    insert: list[dict[str, Any]] | None = None,
    update: list[dict[str, Any]] | None = None,
) -> MagicMock:
    table = MagicMock()
    first = table.select.return_value.eq.return_value
    first.limit.return_value.execute.return_value.data = select
    first.order.return_value.range.return_value.execute.return_value.data = select
    table.insert.return_value.execute.return_value.data = insert
    table.update.return_value.eq.return_value.eq.return_value.execute.return_value.data = (
        update
    )
    return table


def _db_client(tables: dict[str, MagicMock]) -> MagicMock:
    client = MagicMock()
    client.table.side_effect = lambda name: tables[name]
    return client


def _override_auth(app, user: CurrentUser) -> None:
    app.dependency_overrides[get_current_user] = lambda: user


def _override_db(app, client: MagicMock) -> None:
    app.dependency_overrides[get_db_client] = lambda: client


@pytest.fixture()
def app():
    get_settings.cache_clear()
    return create_app()


@pytest.fixture()
def client(app):
    with TestClient(app) as test_client:
        yield test_client


def test_cases_require_a_bearer_token(client: TestClient) -> None:
    response = client.get("/api/v1/cases")

    assert response.status_code == 401
    assert response.json()["detail"] == "Invalid or expired authentication token"


def test_create_case_returns_201_and_writes_safe_audit(
    app, client: TestClient
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    audit_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    case_row = _case_row(case_id, user_id)
    cases = _table_mock(select=[case_row], insert=[case_row])
    audit = _table_mock(
        insert=[
            _audit_row(
                audit_id, case_id, user_id, "case.created", TEST_REQUEST_ID
            )
        ]
    )
    _override_auth(app, user)
    _override_db(app, _db_client({"cases": cases, "audit_events": audit}))

    response = client.post(
        "/api/v1/cases",
        headers={"X-Request-ID": TEST_REQUEST_ID},
        json={"title": "  Test case  ", "description": "A case description."},
    )

    assert response.status_code == 201
    body = response.json()
    assert body["id"] == str(case_id)
    assert body["title"] == "Test case"
    assert "owner_user_id" not in body

    case_payload = cases.insert.call_args.args[0]
    assert case_payload["owner_user_id"] == str(user_id)
    assert case_payload["title"] == "Test case"

    audit_payload = audit.insert.call_args.args[0]
    assert audit_payload["event_type"] == "case.created"
    assert audit_payload["actor_user_id"] == str(user_id)
    assert audit_payload["case_id"] == str(case_id)
    assert audit_payload["target_type"] == "case"
    assert audit_payload["target_id"] == str(case_id)
    assert audit_payload["request_id"] == TEST_REQUEST_ID
    assert audit_payload["safe_metadata"] == {"status": "active"}
    assert "title" not in audit_payload["safe_metadata"]
    assert "description" not in audit_payload["safe_metadata"]


def test_list_cases_filters_by_authenticated_owner(
    app, client: TestClient
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    cases = _table_mock(select=[_case_row(case_id, user_id)])
    _override_auth(app, user)
    _override_db(app, _db_client({"cases": cases}))

    response = client.get("/api/v1/cases")

    assert response.status_code == 200
    assert [row["id"] for row in response.json()] == [str(case_id)]
    cases.select.return_value.eq.assert_called_once_with(
        "owner_user_id", str(user_id)
    )


def test_user_b_cannot_read_user_a_case(
    app, client: TestClient
) -> None:
    user_a = uuid4()
    user_b = uuid4()
    case_id = uuid4()
    cases = _table_mock(select=[_case_row(case_id, user_a)])
    _override_auth(app, CurrentUser(id=user_b, email="b@example.com"))
    _override_db(app, _db_client({"cases": cases}))

    response = client.get(f"/api/v1/cases/{case_id}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Case not found"


def test_missing_case_returns_the_same_404_message(
    app, client: TestClient
) -> None:
    cases = _table_mock(select=[])
    _override_auth(app, CurrentUser(id=uuid4(), email="b@example.com"))
    _override_db(app, _db_client({"cases": cases}))

    response = client.get(f"/api/v1/cases/{uuid4()}")

    assert response.status_code == 404
    assert response.json()["detail"] == "Case not found"


def test_update_case_writes_audit_without_user_content(
    app, client: TestClient
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    case_row = _case_row(case_id, user_id)
    updated_row = _case_row(case_id, user_id, title="Updated case")
    cases = _table_mock(select=[case_row], update=[updated_row])
    audit = _table_mock(
        insert=[
            _audit_row(
                uuid4(), case_id, user_id, "case.updated", TEST_REQUEST_ID
            )
        ]
    )
    _override_auth(app, user)
    _override_db(app, _db_client({"cases": cases, "audit_events": audit}))

    response = client.patch(
        f"/api/v1/cases/{case_id}",
        headers={"X-Request-ID": TEST_REQUEST_ID},
        json={"title": "Updated case"},
    )

    assert response.status_code == 200
    assert response.json()["title"] == "Updated case"
    assert cases.update.call_args.args[0] == {"title": "Updated case"}

    audit_payload = audit.insert.call_args.args[0]
    assert audit_payload["event_type"] == "case.updated"
    assert audit_payload["request_id"] == TEST_REQUEST_ID
    assert audit_payload["safe_metadata"] == {
        "changed_fields": ["title"],
        "status": "active",
    }
    assert "Updated case" not in str(audit_payload["safe_metadata"])


def test_delete_archives_instead_of_deleting_and_writes_audit(
    app, client: TestClient
) -> None:
    user_id = uuid4()
    case_id = uuid4()
    user = CurrentUser(id=user_id, email="lawyer@example.com")
    case_row = _case_row(case_id, user_id)
    archived_row = _case_row(case_id, user_id, status="archived")
    cases = _table_mock(select=[case_row], update=[archived_row])
    audit = _table_mock(
        insert=[
            _audit_row(
                uuid4(), case_id, user_id, "case.archived", TEST_REQUEST_ID
            )
        ]
    )
    _override_auth(app, user)
    _override_db(app, _db_client({"cases": cases, "audit_events": audit}))

    response = client.delete(
        f"/api/v1/cases/{case_id}",
        headers={"X-Request-ID": TEST_REQUEST_ID},
    )

    assert response.status_code == 200
    assert response.json()["status"] == "archived"
    assert cases.update.call_args.args[0] == {"status": "archived"}
    cases.delete.assert_not_called()

    audit_payload = audit.insert.call_args.args[0]
    assert audit_payload["event_type"] == "case.archived"
    assert audit_payload["safe_metadata"] == {"status": "archived"}
    assert audit_payload["request_id"] == TEST_REQUEST_ID


def test_patch_with_no_fields_is_rejected(app, client: TestClient) -> None:
    _override_auth(app, CurrentUser(id=uuid4(), email="lawyer@example.com"))
    _override_db(app, _db_client({"cases": _table_mock(select=[])}))
    case_id = uuid4()

    response = client.patch(f"/api/v1/cases/{case_id}", json={})

    assert response.status_code == 422
