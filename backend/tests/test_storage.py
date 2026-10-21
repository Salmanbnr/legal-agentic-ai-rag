"""Tests for private Supabase Storage path and operation handling."""

from __future__ import annotations

from unittest.mock import MagicMock
from uuid import UUID, uuid4

import pytest

from app.storage.supabase_storage import (
    CASE_DOCUMENTS_BUCKET,
    StorageError,
    create_signed_url,
    delete_original,
    download_original,
    upload_original,
)


def _client() -> tuple[MagicMock, MagicMock]:
    bucket = MagicMock()
    client = MagicMock()
    client.storage.from_.return_value = bucket
    return client, bucket


def test_upload_uses_generated_uuid_path() -> None:
    client, bucket = _client()
    user_id = uuid4()
    case_id = uuid4()

    path = upload_original(
        user_id,
        case_id,
        b"original bytes",
        content_type="application/pdf",
        client=client,
    )

    parts = path.split("/")
    assert len(parts) == 3
    assert UUID(parts[0]) == user_id
    assert UUID(parts[1]) == case_id
    assert UUID(parts[2])
    assert "filing" not in path
    client.storage.from_.assert_called_once_with(CASE_DOCUMENTS_BUCKET)
    bucket.upload.assert_called_once_with(
        path,
        b"original bytes",
        {"content-type": "application/pdf", "upsert": "false"},
    )


def test_download_delete_and_signed_url() -> None:
    client, bucket = _client()
    path = f"{uuid4()}/{uuid4()}/{uuid4()}"
    bucket.download.return_value = b"stored bytes"
    bucket.create_signed_url.return_value = {
        "signedURL": "https://example.test/signed"
    }

    assert download_original(path, client=client) == b"stored bytes"
    delete_original(path, client=client)
    assert create_signed_url(path, 60, client=client) == (
        "https://example.test/signed"
    )

    bucket.download.assert_called_once_with(path)
    bucket.remove.assert_called_once_with([path])
    bucket.create_signed_url.assert_called_once_with(path, 60)


def test_signed_url_accepts_camel_case_response() -> None:
    client, bucket = _client()
    path = f"{uuid4()}/{uuid4()}/{uuid4()}"
    bucket.create_signed_url.return_value = {
        "signedUrl": "https://example.test/other"
    }

    assert create_signed_url(path, client=client) == (
        "https://example.test/other"
    )


def test_storage_rejects_path_traversal() -> None:
    client, _ = _client()

    with pytest.raises(StorageError):
        download_original("../secrets/file", client=client)


def test_storage_rejects_invalid_ids() -> None:
    client, _ = _client()

    with pytest.raises(StorageError):
        upload_original("../user", uuid4(), b"data", client=client)


def test_storage_failures_are_wrapped_without_details() -> None:
    client, bucket = _client()
    bucket.upload.side_effect = RuntimeError("provider detail")

    with pytest.raises(StorageError) as error:
        upload_original(uuid4(), uuid4(), b"data", client=client)

    assert str(error.value) == "Failed to store original file"
    assert "provider detail" not in str(error.value)


def test_signed_url_lifetime_is_validated() -> None:
    client, _ = _client()
    path = f"{uuid4()}/{uuid4()}/{uuid4()}"

    with pytest.raises(ValueError):
        create_signed_url(path, 0, client=client)
    with pytest.raises(ValueError):
        create_signed_url(path, 604801, client=client)


def test_missing_signed_url_is_an_error() -> None:
    client, bucket = _client()
    path = f"{uuid4()}/{uuid4()}/{uuid4()}"
    bucket.create_signed_url.return_value = {}

    with pytest.raises(StorageError):
        create_signed_url(path, client=client)
