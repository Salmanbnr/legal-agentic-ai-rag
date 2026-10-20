"""Tests for Supabase token verification and the auth cache."""

from __future__ import annotations

import hashlib
from typing import Any
from uuid import UUID, uuid4

import httpx
import pytest
from fastapi import HTTPException

from app.auth import dependencies as auth
from app.core.config import Settings


@pytest.fixture(autouse=True)
def _clear_cache() -> None:
    auth.clear_auth_cache()
    yield
    auth.clear_auth_cache()


@pytest.fixture()
def settings() -> Settings:
    return Settings(_env_file=None)


def _user_payload(user_id: UUID, email: str = "lawyer@example.com") -> dict[str, Any]:
    return {"id": str(user_id), "email": email}


async def test_valid_token_returns_current_user(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    user_id = uuid4()

    async def fake_fetch(
        token: str,
        settings: Settings,
        transport: httpx.AsyncBaseTransport | None = None,
    ) -> dict[str, Any]:
        assert token == "good-token"
        return _user_payload(user_id)

    monkeypatch.setattr(auth, "_fetch_user", fake_fetch)

    user = await auth.get_current_user(
        settings=settings, authorization="Bearer good-token"
    )

    assert user.id == user_id
    assert user.email == "lawyer@example.com"


@pytest.mark.parametrize(
    "authorization",
    [None, "", "Basic abc", "Bearer", "Bearer one two", "bearer "],
)
async def test_missing_or_malformed_token_is_rejected(
    authorization: str | None, settings: Settings
) -> None:
    with pytest.raises(HTTPException) as error:
        await auth.get_current_user(
            settings=settings, authorization=authorization
        )

    assert error.value.status_code == 401
    assert error.value.detail == auth._UNAUTHORIZED_MESSAGE
    assert error.value.headers == {"WWW-Authenticate": "Bearer"}


async def test_bad_or_expired_token_is_rejected(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    async def fake_fetch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise auth.InvalidTokenError("rejected")

    monkeypatch.setattr(auth, "_fetch_user", fake_fetch)

    with pytest.raises(HTTPException) as error:
        await auth.get_current_user(
            settings=settings, authorization="Bearer expired-token"
        )

    assert error.value.status_code == 401
    assert error.value.detail == auth._UNAUTHORIZED_MESSAGE


async def test_auth_provider_failure_returns_503(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    async def fake_fetch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        raise auth.AuthProviderUnavailableError("down")

    monkeypatch.setattr(auth, "_fetch_user", fake_fetch)

    with pytest.raises(HTTPException) as error:
        await auth.get_current_user(
            settings=settings, authorization="Bearer any-token"
        )

    assert error.value.status_code == 503
    assert error.value.detail == "Authentication service unavailable"


async def test_missing_user_id_in_provider_payload_is_rejected(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    async def fake_fetch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        return {"email": "lawyer@example.com"}

    monkeypatch.setattr(auth, "_fetch_user", fake_fetch)

    with pytest.raises(HTTPException) as error:
        await auth.get_current_user(
            settings=settings, authorization="Bearer malformed-payload"
        )

    assert error.value.status_code == 401


async def test_successful_check_is_cached_without_raw_token(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    user_id = uuid4()
    calls = 0

    async def fake_fetch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _user_payload(user_id)

    monkeypatch.setattr(auth, "_fetch_user", fake_fetch)
    token = "cached-token-value"

    first = await auth.get_current_user(
        settings=settings, authorization=f"Bearer {token}"
    )
    second = await auth.get_current_user(
        settings=settings, authorization=f"Bearer {token}"
    )

    assert first == second
    assert calls == 1
    assert len(auth._USER_CACHE) == 1
    cache_key = next(iter(auth._USER_CACHE))
    assert cache_key == hashlib.sha256(token.encode("utf-8")).hexdigest()
    assert token not in cache_key
    entry = next(iter(auth._USER_CACHE.values()))
    assert not hasattr(entry, "token")


async def test_cache_expires_after_thirty_seconds(
    monkeypatch: pytest.MonkeyPatch, settings: Settings
) -> None:
    user_id = uuid4()
    calls = 0
    times = iter([100.0, 100.0, 131.0, 131.0])

    async def fake_fetch(*args: Any, **kwargs: Any) -> dict[str, Any]:
        nonlocal calls
        calls += 1
        return _user_payload(user_id)

    monkeypatch.setattr(auth, "_fetch_user", fake_fetch)
    monkeypatch.setattr(auth, "_now", lambda: next(times))

    await auth.get_current_user(
        settings=settings, authorization="Bearer expiring-token"
    )
    await auth.get_current_user(
        settings=settings, authorization="Bearer expiring-token"
    )

    assert calls == 2


def test_auth_ttl_and_timeout_are_configured() -> None:
    assert auth.AUTH_CACHE_TTL_SECONDS == 30.0
    assert auth.AUTH_TIMEOUT_SECONDS == 5.0


async def test_fetch_user_sends_expected_request(settings: Settings) -> None:
    user_id = uuid4()
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen["url"] = str(request.url)
        seen["authorization"] = request.headers.get("authorization", "")
        seen["apikey"] = request.headers.get("apikey", "")
        return httpx.Response(200, json=_user_payload(user_id))

    payload = await auth._fetch_user(
        "access-token",
        settings,
        transport=httpx.MockTransport(handler),
    )

    assert payload["id"] == str(user_id)
    assert seen["url"] == f"{settings.supabase_url_value}/auth/v1/user"
    assert seen["authorization"] == "Bearer access-token"
    assert (
        seen["apikey"] == settings.supabase_publishable_key.get_secret_value()
    )


@pytest.mark.parametrize("status_code", [401, 403])
async def test_fetch_user_maps_rejected_tokens_to_invalid(
    settings: Settings, status_code: int
) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(status_code))

    with pytest.raises(auth.InvalidTokenError):
        await auth._fetch_user("bad-token", settings, transport=transport)


@pytest.mark.parametrize("status_code", [429, 500, 502])
async def test_fetch_user_maps_provider_errors_to_unavailable(
    settings: Settings, status_code: int
) -> None:
    transport = httpx.MockTransport(lambda request: httpx.Response(status_code))

    with pytest.raises(auth.AuthProviderUnavailableError):
        await auth._fetch_user("any-token", settings, transport=transport)
