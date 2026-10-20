"""Tests for request-id normalization."""

from __future__ import annotations

import re

from app.core.middleware import normalise_request_id


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
