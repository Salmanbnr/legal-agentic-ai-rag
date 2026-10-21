"""Upload validation using extension, magic bytes, size, and PDF parsing."""

from __future__ import annotations

import os
import zipfile
from io import BytesIO
from pathlib import PurePosixPath
from typing import Literal

import pymupdf
from fastapi import UploadFile
from pydantic import BaseModel, ConfigDict

from app.core.config import Settings, get_settings

UploadErrorCode = Literal[
    "UNSUPPORTED_FILE",
    "FILE_TOO_LARGE",
    "PASSWORD_REQUIRED",
    "PROCESSING_FAILED",
]

ALLOWED_EXTENSIONS: frozenset[str] = frozenset(
    {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".txt", ".docx"}
)

MIME_BY_EXTENSION: dict[str, str] = {
    ".pdf": "application/pdf",
    ".png": "image/png",
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".tiff": "image/tiff",
    ".txt": "text/plain",
    ".docx": "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
}

_BINARY_SIGNATURES: tuple[bytes, ...] = (
    b"%PDF-",
    b"\x89PNG\r\n\x1a\n",
    b"\xff\xd8\xff",
    b"II*\x00",
    b"MM\x00*",
    b"PK\x03\x04",
    b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1",
)


class UploadValidation(BaseModel):
    """Typed, content-free result of validating one uploaded file."""

    model_config = ConfigDict(frozen=True)

    original_filename: str
    extension: str | None = None
    size_bytes: int = 0
    detected_mime: str | None = None
    page_count: int | None = None
    is_encrypted: bool = False
    error_code: UploadErrorCode | None = None

    @property
    def ok(self) -> bool:
        """Return whether the file can be ingested immediately."""
        return self.error_code is None


def _sanitize_filename(filename: str | None) -> str:
    """Return a display-only filename with no path or control characters."""
    normalized = (filename or "").replace("\\", "/")
    name = PurePosixPath(normalized).name
    cleaned = "".join(
        char for char in name if char.isprintable() and char not in "\x00\r\n"
    ).strip()
    return cleaned[:255]


def _measure_size(file: UploadFile) -> int:
    """Return the upload size without consuming the stream."""
    file.file.seek(0, os.SEEK_END)
    size = file.file.tell()
    file.file.seek(0)
    return max(0, int(size))


def _read_bytes(file: UploadFile) -> bytes:
    """Read the upload body and leave the stream at the beginning."""
    file.file.seek(0)
    data = file.file.read()
    file.file.seek(0)
    if not isinstance(data, bytes):
        raise ValueError("Upload did not contain bytes")
    return data


def _looks_like_text(data: bytes) -> bool:
    """Reject binary payloads while accepting common text encodings."""
    if not data:
        return False
    if any(data.startswith(signature) for signature in _BINARY_SIGNATURES):
        return False
    try:
        if data.startswith((b"\xff\xfe", b"\xfe\xff")):
            text = data.decode("utf-16")
        else:
            text = data.decode("utf-8-sig")
    except UnicodeDecodeError:
        return False
    sample = text[:4096]
    return all(
        ord(char) >= 32 or char in "\n\r\t\f\b"
        for char in sample
    )


def _is_docx(data: bytes) -> bool:
    """Check the ZIP container and required Word document entries."""
    if not data.startswith(b"PK\x03\x04"):
        return False
    try:
        with zipfile.ZipFile(BytesIO(data)) as archive:
            names = set(archive.namelist())
    except (OSError, zipfile.BadZipFile):
        return False
    return (
        "[Content_Types].xml" in names
        and "word/document.xml" in names
        and "word/vbaProject.bin" not in names
    )


def _matches_claimed_type(extension: str, data: bytes) -> bool:
    """Verify the claimed extension against bytes, not just the filename."""
    if extension == ".pdf":
        return data.startswith(b"%PDF-")
    if extension == ".png":
        return data.startswith(b"\x89PNG\r\n\x1a\n")
    if extension in {".jpg", ".jpeg"}:
        return data.startswith(b"\xff\xd8\xff")
    if extension == ".tiff":
        return data.startswith(b"II*\x00") or data.startswith(b"MM\x00*")
    if extension == ".txt":
        return _looks_like_text(data)
    if extension == ".docx":
        return _is_docx(data)
    return False


def _validate_pdf(
    data: bytes, max_pages: int
) -> tuple[UploadErrorCode | None, int | None, bool]:
    """Open a PDF, detect encryption/corruption, and enforce the page limit."""
    document: pymupdf.Document | None = None
    try:
        document = pymupdf.open(stream=data, filetype="pdf")
        is_encrypted = bool(document.needs_pass or document.is_encrypted)
        if is_encrypted:
            return "PASSWORD_REQUIRED", None, True
        if not document.is_pdf:
            return "PROCESSING_FAILED", None, False
        page_count = int(document.page_count)
        if page_count < 1:
            return "PROCESSING_FAILED", page_count, False
        if page_count > max_pages:
            return "PROCESSING_FAILED", page_count, False
        document.load_page(0)
        if page_count > 1:
            document.load_page(page_count - 1)
        return None, page_count, False
    except Exception:
        # PyMuPDF raises several implementation-specific errors for corrupt
        # files. Never expose parser details to the caller.
        return "PROCESSING_FAILED", None, False
    finally:
        if document is not None:
            document.close()


def validate_upload(
    file: UploadFile, *, settings: Settings | None = None
) -> UploadValidation:
    """Validate one upload and return a typed result with an error code."""
    resolved = settings or get_settings()
    filename = _sanitize_filename(file.filename)
    extension = PurePosixPath(filename).suffix.lower() if filename else None

    try:
        size_bytes = _measure_size(file)
    except (OSError, ValueError):
        return UploadValidation(
            original_filename=filename or "upload",
            extension=extension,
            error_code="PROCESSING_FAILED",
        )

    base = UploadValidation(
        original_filename=filename or "upload",
        extension=extension,
        size_bytes=size_bytes,
    )
    if extension is None or extension not in ALLOWED_EXTENSIONS:
        return base.model_copy(update={"error_code": "UNSUPPORTED_FILE"})

    max_bytes = resolved.max_upload_mb * 1024 * 1024
    if size_bytes > max_bytes:
        return base.model_copy(update={"error_code": "FILE_TOO_LARGE"})
    if size_bytes == 0:
        return base.model_copy(update={"error_code": "PROCESSING_FAILED"})

    try:
        data = _read_bytes(file)
    except (OSError, ValueError):
        return base.model_copy(update={"error_code": "PROCESSING_FAILED"})
    if len(data) > max_bytes:
        return base.model_copy(update={"error_code": "FILE_TOO_LARGE"})
    if not _matches_claimed_type(extension, data):
        return base.model_copy(update={"error_code": "UNSUPPORTED_FILE"})

    detected_mime = MIME_BY_EXTENSION[extension]
    if extension != ".pdf":
        return base.model_copy(update={"detected_mime": detected_mime})

    error_code, page_count, is_encrypted = _validate_pdf(
        data, resolved.max_pdf_pages
    )
    return base.model_copy(
        update={
            "detected_mime": detected_mime,
            "page_count": page_count,
            "is_encrypted": is_encrypted,
            "error_code": error_code,
        }
    )
