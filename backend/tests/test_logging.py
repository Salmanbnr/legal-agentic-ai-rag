"""Tests for application logging setup and its privacy guard."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from app.observability.logging import (
    APP_LOGGER_NAME,
    SafeFormatter,
    configure_logging,
)


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


def test_configure_logging_is_idempotent(tmp_path: Path) -> None:
    log_path = tmp_path / "app.log"

    configure_logging("INFO", log_path)
    configure_logging("INFO", log_path)

    handlers = logging.getLogger(APP_LOGGER_NAME).handlers
    assert len(handlers) == 2
    assert logging.getLogger(APP_LOGGER_NAME).level == logging.INFO


def test_configure_logging_creates_parent_directories(tmp_path: Path) -> None:
    log_path = tmp_path / "nested" / "logs" / "app.log"

    configure_logging("info", log_path)

    assert log_path.parent.is_dir()
    assert logging.getLogger(APP_LOGGER_NAME).level == logging.INFO


def test_only_allowlisted_extras_are_rendered(tmp_path: Path) -> None:
    log_path = tmp_path / "app.log"
    configure_logging("INFO", log_path)

    logging.getLogger("app.test").info(
        "answer stored",
        extra={
            "case_id": "case-1",
            "claim_count": 2,
            "document_text": "the rent is 5000 per month",
            "api_key": "sk-secret",
        },
    )
    _flush_app_handlers()

    text = log_path.read_text(encoding="utf-8")
    assert "answer stored" in text
    assert "case_id=case-1" in text
    assert "claim_count=2" in text
    assert "the rent is 5000 per month" not in text
    assert "sk-secret" not in text


def test_level_filters_lower_severity_records(tmp_path: Path) -> None:
    log_path = tmp_path / "app.log"
    configure_logging("WARNING", log_path)
    logger = logging.getLogger("app.test")

    logger.info("hidden info")
    logger.warning("visible warning")
    _flush_app_handlers()

    text = log_path.read_text(encoding="utf-8")
    assert "hidden info" not in text
    assert "visible warning" in text


def test_safe_formatter_keeps_message_without_extras() -> None:
    record = logging.LogRecord(
        name="app.test",
        level=logging.INFO,
        pathname=__file__,
        lineno=1,
        msg="plain message",
        args=(),
        exc_info=None,
    )

    assert SafeFormatter("%(message)s").format(record) == "plain message"
