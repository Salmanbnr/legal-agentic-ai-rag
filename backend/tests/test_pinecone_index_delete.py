"""Tests for the index deletion helper."""

from __future__ import annotations

from typing import Any

import pytest

from app.core.config import Settings
from app.indexing.pinecone_store import PineconeError, delete_index


class _Control:
    """Minimal Pinecone control-plane stub."""

    def __init__(self, *, exists: bool, fail: bool = False) -> None:
        self.exists = exists
        self.fail = fail
        self.deleted: list[str] = []

    def list_indexes(self) -> Any:
        if not self.exists:
            return []
        return [type("Listing", (), {"name": "legal-evidence-test"})()]

    def delete_index(self, name: str) -> None:
        if self.fail:
            raise RuntimeError("upstream rejected the delete")
        self.deleted.append(name)


def _settings() -> Settings:
    return Settings(_env_file=None)


def test_delete_index_removes_an_existing_index() -> None:
    control = _Control(exists=True)

    assert delete_index(_settings(), client=control) is True
    assert control.deleted == ["legal-evidence-test"]


def test_delete_index_is_a_no_op_when_absent() -> None:
    control = _Control(exists=False)

    assert delete_index(_settings(), client=control) is False
    assert control.deleted == []


def test_delete_index_reports_a_safe_error() -> None:
    control = _Control(exists=True, fail=True)

    with pytest.raises(PineconeError) as caught:
        delete_index(_settings(), client=control)

    assert caught.value.error_code == "PINECONE_INDEX_DELETE_FAILED"
    assert "upstream" not in str(caught.value)
