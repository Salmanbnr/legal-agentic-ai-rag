"""Tests for the in-memory per-user rate limiter."""

from __future__ import annotations

import pytest

from app.core.rate_limit import InMemoryRateLimiter, RateLimitExceeded


def test_limit_is_per_key_and_resets_after_window() -> None:
    limiter = InMemoryRateLimiter(limit=2, window_seconds=10)

    limiter.check("user-a", now=0.0)
    limiter.check("user-a", now=1.0)
    limiter.check("user-b", now=1.0)

    with pytest.raises(RateLimitExceeded) as error:
        limiter.check("user-a", now=2.0)

    assert error.value.retry_after == 8

    limiter.check("user-a", now=10.1)


def test_invalid_limit_is_rejected() -> None:
    with pytest.raises(ValueError):
        InMemoryRateLimiter(limit=0)
    with pytest.raises(ValueError):
        InMemoryRateLimiter(limit=1, window_seconds=0)
