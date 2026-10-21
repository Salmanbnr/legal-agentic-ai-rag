"""Tests for the local PDF fixture generator."""

from __future__ import annotations

from pathlib import Path

from fastapi import UploadFile

from app.core.config import get_settings
from app.ingestion.validation import validate_upload
from scripts.make_test_pdfs import generate_test_pdfs


def _validate(path: Path):
    with path.open("rb") as file:
        upload = UploadFile(filename=path.name, file=file)
        return validate_upload(upload, settings=get_settings())


def test_generated_pdfs_have_expected_validation_states(tmp_path: Path) -> None:
    generated = generate_test_pdfs(tmp_path)

    assert generated.clean_native.exists()
    assert generated.image_only.exists()
    assert generated.encrypted.exists()

    clean = _validate(generated.clean_native)
    image_only = _validate(generated.image_only)
    encrypted = _validate(generated.encrypted)

    assert clean.ok and clean.page_count == 1
    assert image_only.ok and image_only.page_count == 1
    assert encrypted.error_code == "PASSWORD_REQUIRED"
    assert encrypted.is_encrypted is True
