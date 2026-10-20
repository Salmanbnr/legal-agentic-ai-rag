"""Supabase access-token validation for FastAPI routes.

Successful token checks are cached in memory for 30 seconds under a SHA-256
hash of the token. The raw token is never stored in the cache or logged.
"""

from __future__ import annotations

import hashlib
import threading
import time
from dataclasses import dataclass
from typing import Annotated, Any
from uuid import UUID

import httpx
from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel, ConfigDict, ValidationError

from app.core.config import Settings, get_settings

AUTH_TIMEOUT_SECONDS = 5.0
AUTH_CACHE_TTL_SECONDS = 30.0
_CACHE_MAX_ENTRIES = 1024
_UNAUTHORIZED_MESSAGE = "Invalid or expired authentication token"


class CurrentUser(BaseModel):
    """The authenticated Supabase user resolved from an access token."""

    model_config = ConfigDict(extra="ignore", frozen=True)

    id: UUID
    email: str | None = None


class InvalidTokenError(Exception):
    """Raised when Supabase rejects the supplied access token."""


class AuthProviderUnavailableError(Exception):
    """Raised when Supabase Auth cannot be reached or returns a server error."""


@dataclass(frozen=True, slots=True)
class _CacheEntry:
    user: CurrentUser
    expires_at: float


_USER_CACHE: dict[str, _CacheEntry] = {}
_CACHE_LOCK = threading.Lock()


def _unauthorized() -> HTTPException:
    """Build a safe 401 response for any invalid credential shape."""
    return HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail=_UNAUTHORIZED_MESSAGE,
        headers={"WWW-Authenticate": "Bearer"},
    )


def _service_unavailable() -> HTTPException:
    """Build a safe 503 response when the auth provider is unavailable."""
    return HTTPException(
        status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
        detail="Authentication service unavailable",
    )


def _now() -> float:
    """Return a monotonic clock value for cache expiry checks."""
    return time.monotonic()


def _token_hash(token: str) -> str:
    """Return a stable cache key without retaining the token itself."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _extract_bearer_token(authorization: str | None) -> str:
    """Parse an Authorization header, rejecting missing or malformed values."""
    if not authorization:
        raise _unauthorized()
    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer" or not parts[1]:
        raise _unauthorized()
    token = parts[1]
    if len(token) > 8192:
        raise _unauthorized()
    return token


def _get_cached_user(token_hash: str, now: float) -> CurrentUser | None:
    """Return a live cached user and evict expired entries."""
    with _CACHE_LOCK:
        entry = _USER_CACHE.get(token_hash)
        if entry is None:
            return None
        if entry.expires_at <= now:
            _USER_CACHE.pop(token_hash, None)
            return None
        return entry.user


def _store_cached_user(token_hash: str, user: CurrentUser, now: float) -> None:
    """Cache a successful check for the configured TTL."""
    with _CACHE_LOCK:
        _USER_CACHE[token_hash] = _CacheEntry(
            user=user,
            expires_at=now + AUTH_CACHE_TTL_SECONDS,
        )
        if len(_USER_CACHE) <= _CACHE_MAX_ENTRIES:
            return
        expired = [
            key for key, entry in _USER_CACHE.items() if entry.expires_at <= now
        ]
        for key in expired:
            _USER_CACHE.pop(key, None)
        if len(_USER_CACHE) > _CACHE_MAX_ENTRIES:
            _USER_CACHE.clear()


def clear_auth_cache() -> None:
    """Clear the in-memory token cache (used by tests and shutdown hooks)."""
    with _CACHE_LOCK:
        _USER_CACHE.clear()


async def _fetch_user(
    token: str,
    settings: Settings,
    transport: httpx.AsyncBaseTransport | None = None,
) -> dict[str, Any]:
    """Fetch the Supabase user record for an access token.

    The token is sent only to the configured Supabase Auth endpoint. Errors
    intentionally omit tokens and response bodies.
    """
    url = f"{settings.supabase_url_value}/auth/v1/user"
    headers = {
        "Authorization": f"Bearer {token}",
        "apikey": settings.supabase_publishable_key.get_secret_value(),
    }
    try:
        async with httpx.AsyncClient(
            timeout=httpx.Timeout(AUTH_TIMEOUT_SECONDS),
            transport=transport,
        ) as client:
            response = await client.get(url, headers=headers)
    except httpx.HTTPError as exc:
        raise AuthProviderUnavailableError(
            "Supabase Auth request failed"
        ) from exc

    if response.status_code in (status.HTTP_401_UNAUTHORIZED, status.HTTP_403_FORBIDDEN):
        raise InvalidTokenError("Supabase rejected the token")
    if response.status_code != status.HTTP_200_OK:
        raise AuthProviderUnavailableError("Supabase Auth returned an error")

    try:
        payload = response.json()
    except ValueError as exc:
        raise AuthProviderUnavailableError("Supabase Auth returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise AuthProviderUnavailableError("Supabase Auth returned an invalid payload")
    return payload


def _to_current_user(payload: dict[str, Any]) -> CurrentUser:
    """Validate the fields required from the Supabase user payload."""
    try:
        return CurrentUser(id=payload["id"], email=payload.get("email"))
    except (KeyError, TypeError, ValidationError) as exc:
        raise InvalidTokenError("Supabase user payload was invalid") from exc


async def get_current_user(
    settings: Annotated[Settings, Depends(get_settings)],
    authorization: Annotated[str | None, Header(alias="Authorization")] = None,
) -> CurrentUser:
    """Resolve and verify the Bearer token for an API request.

    Raises:
        HTTPException: 401 for missing, malformed, expired, or invalid tokens;
            503 when Supabase Auth is unreachable.
    """
    token = _extract_bearer_token(authorization)
    token_hash = _token_hash(token)
    cached = _get_cached_user(token_hash, _now())
    if cached is not None:
        return cached

    try:
        payload = await _fetch_user(token, settings)
        user = _to_current_user(payload)
    except InvalidTokenError:
        raise _unauthorized() from None
    except AuthProviderUnavailableError:
        raise _service_unavailable() from None

    _store_cached_user(token_hash, user, _now())
    return user
