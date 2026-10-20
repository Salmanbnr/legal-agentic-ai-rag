"""Server-only Supabase client construction.

The service-role key bypasses row-level security and must never be exposed to
the browser or included in a frontend bundle.
"""

from __future__ import annotations

from supabase import Client, create_client

from app.core.config import Settings, get_settings


def get_service_client(settings: Settings | None = None) -> Client:
    """Create a Supabase client using the service-role key.

    Args:
        settings: Optional validated settings, primarily for tests or workers.

    Returns:
        A server-side Supabase client. Callers must still enforce ownership in
        repository code; the service role bypasses RLS by design.
    """
    resolved = settings or get_settings()
    return create_client(
        resolved.supabase_url_value,
        resolved.supabase_service_role_key.get_secret_value(),
    )
