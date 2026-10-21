"""Validation tests for extension, magic, size, and PDF safety checks."""

from __future__ import annotations

from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

import pymupdf
import pytest
from fastapi import UploadFile

from app.core.config import Settings
from app.ingestion.validation import validate_upload
from scripts.make_test_pdfs import GeneratedPdfs, generate_test_pdfs


@pytest.fixture()
def settings() -> Settings:
    return Settings(_env_file=None)


@pytest.fixture()
def pdfs(tmp_path: Path) -> GeneratedPdfs:
    return generate_test_pdfs(tmp_path / "pdfs")


def _upload(data: bytes, filename: str) -> UploadFile:
    return UploadFile(filename=filename, file=BytesIO(data), size=len(data))


def _read(path: Path) -> bytes:
    return path.read_bytes()


def test_clean_native_pdf_passes(settings: Settings, pdfs: GeneratedPdfs) -> None:
    result = validate_upload(
        _upload(_read(pdfs.clean_native), "clean.pdf"), settings=settings
    )

    assert result.ok
    assert result.error_code is None
    assert result.detected_mime == "application/pdf"
    assert result.page_count == 1
    assert result.is_encrypted is False


def test_wrong_magic_bytes_are_rejected(
    settings: Settings, pdfs: GeneratedPdfs
) -> None:
    png_bytes = b"\x89PNG\r\n\x1a\n" + _read(pdfs.clean_native)

    result = validate_upload(_upload(png_bytes, "fake.pdf"), settings=settings)

    assert result.error_code == "UNSUPPORTED_FILE"
    assert result.detected_mime is None


def test_oversize_file_is_rejected(settings: Settings) -> None:
    small_settings = settings.model_copy(update={"max_upload_mb": 1})
    data = b"a" * (1024 * 1024 + 1)

    result = validate_upload(
        _upload(data, "large.txt"), settings=small_settings
    )

    assert result.error_code == "FILE_TOO_LARGE"
    assert result.size_bytes == len(data)


def test_encrypted_pdf_requires_password(
    settings: Settings, pdfs: GeneratedPdfs
) -> None:
    result = validate_upload(
        _upload(_read(pdfs.encrypted), "encrypted.pdf"), settings=settings
    )

    assert result.error_code == "PASSWORD_REQUIRED"
    assert result.is_encrypted is True
    assert result.detected_mime == "application/pdf"
    assert result.page_count is None


def test_pdf_page_limit_is_enforced(settings: Settings) -> None:
    document = pymupdf.open()
    try:
        document.new_page()
        document.new_page()
        data = document.tobytes()
    finally:
        document.close()

    small_settings = settings.model_copy(update={"max_pdf_pages": 1})
    result = validate_upload(_upload(data, "two-pages.pdf"), settings=small_settings)

    assert result.error_code == "PROCESSING_FAILED"
    assert result.page_count == 2


def test_corrupt_pdf_is_rejected(settings: Settings) -> None:
    result = validate_upload(
        _upload(b"%PDF-1.7\nnot actually a pdf", "broken.pdf"),
        settings=settings,
    )

    assert result.error_code == "PROCESSING_FAILED"
    assert result.detected_mime == "application/pdf"


def test_unsupported_extension_is_rejected(settings: Settings) -> None:
    result = validate_upload(
        _upload(b"plain text", "notes.exe"), settings=settings
    )

    assert result.error_code == "UNSUPPORTED_FILE"


def test_minimal_docx_is_accepted(settings: Settings) -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("word/document.xml", "<w:document />")

    result = validate_upload(
        _upload(buffer.getvalue(), "brief.docx"), settings=settings
    )

    assert result.ok
    assert result.detected_mime == (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    )


def test_docx_without_word_document_is_rejected(settings: Settings) -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("[Content_Types].xml", "<Types />")
        archive.writestr("other/data.txt", "not a word document")

    result = validate_upload(
        _upload(buffer.getvalue(), "fake.docx"), settings=settings
    )

    assert result.error_code == "UNSUPPORTED_FILE"


def test_text_must_not_be_binary(settings: Settings) -> None:
    result = validate_upload(
        _upload(b"\x00\x01\x02\x03", "binary.txt"), settings=settings
    )

    assert result.error_code == "UNSUPPORTED_FILE"


def test_magic_check_is_case_insensitive_for_extension(
    settings: Settings, pdfs: GeneratedPdfs
) -> None:
    result = validate_upload(
        _upload(_read(pdfs.clean_native), "CLEAN.PDF"), settings=settings
    )

    assert result.ok
    assert result.extension == ".pdf"


def test_filename_path_is_stripped(
    settings: Settings, pdfs: GeneratedPdfs
) -> None:
    result = validate_upload(
        _upload(_read(pdfs.clean_native), "..\\..\\secret\\brief.pdf"),
        settings=settings,
    )

    assert result.original_filename == "brief.pdf"
