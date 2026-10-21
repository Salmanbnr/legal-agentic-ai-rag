"""Tests for deterministic upload fingerprints."""

from __future__ import annotations

import hashlib

import pytest

from app.ingestion.fingerprint import compute_fingerprint


def test_fingerprint_matches_sha256_and_size() -> None:
    data = b"abc"

    fingerprint = compute_fingerprint(data)

    assert fingerprint.sha256 == hashlib.sha256(data).hexdigest()
    assert fingerprint.size_bytes == 3


def test_fingerprint_requires_bytes() -> None:
    with pytest.raises(TypeError):
        compute_fingerprint("not bytes")  # type: ignore[arg-type]
