"""ASGI middleware for request correlation and request logging."""

from __future__ import annotations

import logging
import re
import time
import uuid
from typing import Any

from starlette.datastructures import Headers, MutableHeaders
from starlette.types import ASGIApp, Message, Receive, Scope, Send

logger = logging.getLogger(__name__)

REQUEST_ID_HEADER = "X-Request-ID"
_REQUEST_ID_PATTERN = re.compile(r"^[A-Za-z0-9._-]{1,128}$")
_QUIET_PATHS = frozenset({"/health"})


def normalise_request_id(value: str | None) -> str:
    """Return a safe client request id or generate a new opaque id."""
    if value and _REQUEST_ID_PATTERN.fullmatch(value):
        return value
    return uuid.uuid4().hex


class RequestIDMiddleware:
    """Attach a request id to the request state and response headers."""

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        headers = Headers(scope=scope)
        request_id = normalise_request_id(headers.get(REQUEST_ID_HEADER))
        scope.setdefault("state", {})
        state: dict[str, Any] = scope["state"]
        state["request_id"] = request_id

        async def send_with_request_id(message: Message) -> None:
            if message["type"] == "http.response.start":
                response_headers = MutableHeaders(scope=message)
                response_headers[REQUEST_ID_HEADER] = request_id
            await send(message)

        await self.app(scope, receive, send_with_request_id)


class RequestLogMiddleware:
    """Log one safe line per request: method, path, status, duration, id.

    Query strings, headers, and bodies are never logged, so document text and
    credentials cannot reach the log. Register this middleware inside
    :class:`RequestIDMiddleware` so the request id is already assigned.
    """

    def __init__(self, app: ASGIApp) -> None:
        self.app = app

    async def __call__(
        self, scope: Scope, receive: Receive, send: Send
    ) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return

        started = time.perf_counter()
        status_code = 500

        async def send_with_status(message: Message) -> None:
            nonlocal status_code
            if message["type"] == "http.response.start":
                status_code = int(message["status"])
            await send(message)

        try:
            await self.app(scope, receive, send_with_status)
        finally:
            duration_ms = int((time.perf_counter() - started) * 1000)
            path = str(scope.get("path", ""))
            state = scope.get("state") or {}
            request_id = (
                state.get("request_id") if isinstance(state, dict) else None
            )
            if path in _QUIET_PATHS and status_code < 400:
                level = logging.DEBUG
            elif status_code >= 500:
                level = logging.ERROR
            elif status_code >= 400:
                level = logging.WARNING
            else:
                level = logging.INFO
            logger.log(
                level,
                "request completed",
                extra={
                    "method": scope.get("method", ""),
                    "path": path,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                    "request_id": request_id or "unknown",
                },
            )
