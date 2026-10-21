"""Small in-memory rate limiter for low-volume single-user workloads."""

from __future__ import annotations

import math
import threading
import time
from collections import defaultdict, deque


class RateLimitExceeded(Exception):
    """Raised when a key has exhausted its allowance for the window."""

    def __init__(self, retry_after: int) -> None:
        """Store the minimum number of seconds before a retry can succeed."""
        super().__init__("Rate limit exceeded")
        self.retry_after = max(1, retry_after)


class InMemoryRateLimiter:
    """A thread-safe sliding-window limiter keyed by user id.

    This is intentionally process-local. A multi-worker deployment will need a
    shared limiter, but the current product is a single backend process.
    """

    def __init__(self, limit: int, window_seconds: int = 60) -> None:
        """Create a limiter allowing ``limit`` events per window."""
        if limit < 1:
            raise ValueError("limit must be at least 1")
        if window_seconds < 1:
            raise ValueError("window_seconds must be at least 1")
        self.limit = limit
        self.window_seconds = window_seconds
        self._events: dict[str, deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def check(self, key: str, now: float | None = None) -> None:
        """Record one event for ``key`` or raise :class:`RateLimitExceeded`."""
        current = time.monotonic() if now is None else now
        cutoff = current - self.window_seconds
        with self._lock:
            events = self._events[key]
            while events and events[0] <= cutoff:
                events.popleft()
            if len(events) >= self.limit:
                retry_after = math.ceil(events[0] + self.window_seconds - current)
                raise RateLimitExceeded(retry_after)
            events.append(current)

    def reset(self) -> None:
        """Clear all tracked keys, primarily for tests and local tooling."""
        with self._lock:
            self._events.clear()
