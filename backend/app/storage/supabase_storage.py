"""Private Supabase Storage operations for immutable original documents."""

from __future__ import annotations

from uuid import UUID, uuid4

from supabase import Client

from app.db.client import get_service_client

CASE_DOCUMENTS_BUCKET = "case-documents"
_MAX_SIGNED_URL_SECONDS = 604800


class StorageError(RuntimeError):
    """Raised when a private storage operation fails."""


def _uuid_text(value: UUID | str) -> str:
    """Canonicalize an id used as a storage path segment."""
    try:
        return str(UUID(str(value)))
    except (TypeError, ValueError) as exc:
        raise StorageError("Invalid storage identifier") from exc


def _validate_storage_path(storage_path: str) -> str:
    """Accept only the generated user/case/object path shape."""
    path = (storage_path or "").strip()
    parts = path.split("/")
    if len(parts) != 3 or any(not part for part in parts):
        raise StorageError("Invalid storage path")
    try:
        for part in parts:
            UUID(part)
    except ValueError as exc:
        raise StorageError("Invalid storage path") from exc
    return path


def _bucket(client: Client | None = None):
    """Return the private case-documents bucket API."""
    resolved = client or get_service_client()
    return resolved.storage.from_(CASE_DOCUMENTS_BUCKET)


def upload_original(
    user_id: UUID | str,
    case_id: UUID | str,
    data: bytes,
    *,
    content_type: str = "application/octet-stream",
    client: Client | None = None,
) -> str:
    """Store an original under a generated key and return that key.

    The user-supplied filename is deliberately not part of the storage path.
    """
    if not isinstance(data, bytes) or not data:
        raise StorageError("Original file is empty")
    storage_path = f"{_uuid_text(user_id)}/{_uuid_text(case_id)}/{uuid4()}"
    options = {"content-type": content_type, "upsert": "false"}
    try:
        _bucket(client).upload(storage_path, data, options)
    except Exception as exc:
        raise StorageError("Failed to store original file") from exc
    return storage_path


def download_original(
    storage_path: str, *, client: Client | None = None
) -> bytes:
    """Download one original document from the private bucket."""
    path = _validate_storage_path(storage_path)
    try:
        data = _bucket(client).download(path)
    except Exception as exc:
        raise StorageError("Failed to download original file") from exc
    if not isinstance(data, bytes):
        raise StorageError("Storage returned an invalid payload")
    return data


def delete_original(storage_path: str, *, client: Client | None = None) -> None:
    """Delete one original document from the private bucket."""
    path = _validate_storage_path(storage_path)
    try:
        _bucket(client).remove([path])
    except Exception as exc:
        raise StorageError("Failed to delete original file") from exc


def create_signed_url(
    storage_path: str,
    seconds: int = 60,
    *,
    client: Client | None = None,
) -> str:
    """Create a short-lived private download URL for one original."""
    path = _validate_storage_path(storage_path)
    if not 1 <= seconds <= _MAX_SIGNED_URL_SECONDS:
        raise ValueError("Signed URL lifetime is out of range")
    try:
        response = _bucket(client).create_signed_url(path, seconds)
    except Exception as exc:
        raise StorageError("Failed to create download URL") from exc

    if isinstance(response, dict):
        signed_url = response.get("signedURL") or response.get("signedUrl")
    else:
        signed_url = getattr(response, "signedURL", None) or getattr(
            response, "signedUrl", None
        )
    if not isinstance(signed_url, str) or not signed_url:
        raise StorageError("Storage did not return a signed URL")
    return signed_url
