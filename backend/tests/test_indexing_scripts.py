"""Tests for the Pinecone index and reconciliation command line scripts."""

from __future__ import annotations

import json
from unittest.mock import MagicMock
from uuid import uuid4

import pytest

from scripts import create_index as create_index_module
from scripts import reconcile as reconcile_module
from app.indexing.reconcile import ReconcileFixResult, ReconcileReport


@pytest.fixture(autouse=True)
def fake_service_client(monkeypatch: pytest.MonkeyPatch) -> None:
    """Avoid building a real Supabase client in script tests."""
    monkeypatch.setattr(
        reconcile_module, "get_service_client", lambda settings: MagicMock()
    )


def _report(case_id, **overrides) -> ReconcileReport:
    values = {
        "case_id": case_id,
        "postgres_chunk_count": 2,
        "pinecone_vector_count": 2,
        "orphaned_ids": [],
        "missing_ids": [],
    }
    values.update(overrides)
    return ReconcileReport(**values)


def test_create_index_reports_the_result(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    calls: list[dict[str, object]] = []

    def fake_create(settings, *, dimension=None):
        calls.append({"dimension": dimension})
        return "created"

    monkeypatch.setattr(
        create_index_module, "create_index_if_missing", fake_create
    )
    monkeypatch.setattr(
        create_index_module, "describe_index_dimension", lambda settings: 384
    )

    assert create_index_module.main([]) == 0

    output = capsys.readouterr().out
    assert "created" in output
    assert "384" in output
    assert "legal-evidence-test" in output
    assert calls == [{"dimension": None}]


def test_create_index_accepts_a_dimension_override(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    def fake_create(settings, *, dimension=None):
        calls.append({"dimension": dimension})
        return "exists"

    monkeypatch.setattr(
        create_index_module, "create_index_if_missing", fake_create
    )
    monkeypatch.setattr(
        create_index_module, "describe_index_dimension", lambda settings: 768
    )

    assert create_index_module.main(["--dimension", "768"]) == 0
    assert calls == [{"dimension": 768}]


def test_recreate_requires_confirmation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        create_index_module,
        "delete_index",
        lambda *args, **kwargs: pytest.fail("delete must not run"),
    )

    with pytest.raises(SystemExit) as caught:
        create_index_module.main(["--recreate"])

    assert caught.value.code == 2


def test_recreate_deletes_then_rebuilds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    events: list[str] = []

    monkeypatch.setattr(
        create_index_module,
        "delete_index",
        lambda settings: events.append("delete") or True,
    )
    monkeypatch.setattr(
        create_index_module,
        "create_index_if_missing",
        lambda settings, *, dimension=None: events.append("create")
        or "created",
    )
    monkeypatch.setattr(
        create_index_module, "describe_index_dimension", lambda settings: 384
    )

    assert create_index_module.main(["--recreate", "--yes"]) == 0
    assert events == ["delete", "create"]

    output = capsys.readouterr().out
    assert "deleted" in output
    assert "created" in output


def test_reconcile_prints_a_clean_report(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    case_id = uuid4()
    monkeypatch.setattr(
        reconcile_module,
        "reconcile_case",
        lambda client, target, **kwargs: _report(case_id),
    )
    monkeypatch.setattr(
        reconcile_module,
        "fix_reconcile",
        lambda *args, **kwargs: pytest.fail("fix must not run"),
    )

    assert reconcile_module.main(["--case-id", str(case_id)]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["case_id"] == str(case_id)
    assert payload["orphaned_ids"] == []
    assert payload["missing_ids"] == []


def test_reconcile_reports_problems_with_a_nonzero_exit_code(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    case_id = uuid4()
    monkeypatch.setattr(
        reconcile_module,
        "reconcile_case",
        lambda client, target, **kwargs: _report(
            case_id, orphaned_ids=["orphan"], missing_ids=["missing"]
        ),
    )

    assert reconcile_module.main(["--case-id", str(case_id)]) == 1

    payload = json.loads(capsys.readouterr().out)
    assert payload["orphaned_ids"] == ["orphan"]
    assert payload["missing_ids"] == ["missing"]


def test_reconcile_fix_deletes_orphans_and_requeues_documents(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    case_id = uuid4()
    fixed: list[str] = []

    def fake_reconcile(client, target, **kwargs):
        return _report(case_id, orphaned_ids=["orphan"])

    def fake_fix(client, target, **kwargs):
        fixed.append(str(target))
        return ReconcileFixResult(
            report=_report(case_id, orphaned_ids=["orphan"]),
            deleted_orphan_count=1,
            requeued_document_ids=[str(uuid4())],
        )

    monkeypatch.setattr(reconcile_module, "reconcile_case", fake_reconcile)
    monkeypatch.setattr(reconcile_module, "fix_reconcile", fake_fix)

    assert reconcile_module.main(["--case-id", str(case_id), "--fix"]) == 0

    payload = json.loads(capsys.readouterr().out)
    assert payload["fixed"]["deleted_orphan_count"] == 1
    assert fixed == [str(case_id)]
