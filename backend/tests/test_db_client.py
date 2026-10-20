"""Tests for the server-only Supabase client factory."""

from __future__ import annotations

from unittest.mock import MagicMock

from app.core.config import Settings
from app.db import client as db_client


def test_get_service_client_uses_service_role_key(monkeypatch) -> None:
    settings = Settings(_env_file=None)
    created = MagicMock()
    factory = MagicMock(return_value=created)
    monkeypatch.setattr(db_client, "create_client", factory)

    result = db_client.get_service_client(settings)

    assert result is created
    factory.assert_called_once_with(
        settings.supabase_url_value,
        settings.supabase_service_role_key.get_secret_value(),
    )
    factory.assert_called_once()
    assert (
        settings.supabase_service_role_key.get_secret_value()
        != settings.supabase_publishable_key.get_secret_value()
    )
