"""Content fingerprints used for immutable uploads and duplicate detection."""

from __future__ import annotations

import hashlib

from pydantic import BaseModel, ConfigDict, Field


class FileFingerprint(BaseModel):
    """The SHA-256 digest and byte size of an uploaded original."""

    model_config = ConfigDict(frozen=True)

    sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    size_bytes: int = Field(ge=0)


def compute_fingerprint(data: bytes) -> FileFingerprint:
    """Compute a SHA-256 digest without retaining or logging file content."""
    if not isinstance(data, bytes):
        raise TypeError("Fingerprint input must be bytes")
    return FileFingerprint(
        sha256=hashlib.sha256(data).hexdigest(),
        size_bytes=len(data),
    )
