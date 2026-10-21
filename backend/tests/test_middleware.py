"""Tests for request-id normalization and request logging."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import pytest
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import PlainTextResponse
from starlette.routing import Route
from starlette.testclient import TestClient

from app.core.middleware import (
    RequestIDMiddleware,
    RequestLogMiddleware,
    normalise_request_id,
)
from app.observability.logging import APP_LOGGER_NAME, configure_logging


@pytest.fixture(autouse=True)
def reset_app_logging() -> None:
    """Restore the shared app logger after each test."""
    yield
    logger = logging.getLogger(APP_LOGGER_NAME)
    for handler in list(logger.handlers):
        if getattr(handler, "_legal_ai_handler", False):
            logger.removeHandler(handler)
            handler.close()
    logger.setLevel(logging.NOTSET)
    logger.propagate = True


def _flush_app_handlers() -> None:
    for handler in logging.getLogger(APP_LOGGER_NAME).handlers:
        handler.flush()


async def _ok(request: Request) -> PlainTextResponse:
    return PlainTextResponse("ok")


async def _teapot(request: Request) -> PlainTextResponse:
    return PlainTextResponse("short and stout", status_code=418)


def _client() -> TestClient:
    app = Starlette(
        routes=[
            Route("/cases/{case_id}", _ok),
            Route("/teapot", _teapot),
        ]
    )
    app.add_middleware(RequestLogMiddleware)
    app.add_middleware(RequestIDMiddleware)
    return TestClient(app)


def _log_line(path: Path) -> str:
    lines = [
        line
        for line in path.read_text(encoding="utf-8").splitlines()
        if "request completed" in line
    ]
    assert len(lines) == 1
    return lines[0]


def test_safe_request_id_is_preserved() -> None:
    assert normalise_request_id("req-abc_123.4") == "req-abc_123.4"


def test_missing_request_id_generates_opaque_id() -> None:
    generated = normalise_request_id(None)

    assert re.fullmatch(r"[0-9a-f]{32}", generated)


def test_unsafe_request_id_is_replaced() -> None:
    for value in ("has space", "line\nbreak", "x" * 129, "semi;colon"):
        generated = normalise_request_id(value)
        assert generated != value
        assert re.fullmatch(r"[0-9a-f]{32}", generated)


def test_request_log_records_status_and_request_id(tmp_path: Path) -> None:
    log_path = tmp_path / "requests.log"
    configure_logging("INFO", log_path)

    response = _client().get(
        "/cases/case-1?token=super-secret",
        headers={"X-Request-ID": "req-1"},
    )
    _flush_app_handlers()

    assert response.status_code == 200
    line = _log_line(log_path)
    assert "INFO" in line
    assert "method=GET" in line
    assert "path=/cases/case-1" in line
    assert "status_code=200" in line
    assert "request_id=req-1" in line
    assert "duration_ms=" in line
    assert "super-secret" not in log_path.read_text(encoding="utf-8")


def test_client_errors_are_logged_as_warnings(tmp_path: Path) -> None:
    log_path = tmp_path / "requests.log"
    configure_logging("INFO", log_path)

    response = _client().get("/teapot")
    _flush_app_handlers()

    assert response.status_code == 418
    line = _log_line(log_path)
    assert "WARNING" in line
    assert "status_code=418" in line
