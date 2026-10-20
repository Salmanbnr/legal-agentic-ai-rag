"""FastAPI application factory and entrypoint."""

from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.cases import router as cases_router
from app.api.health import router as health_router
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.middleware import RequestIDMiddleware

APP_VERSION = "0.1.0"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Configuration is validated while the app is created, so a missing
    environment variable fails startup instead of the first request.
    """
    resolved = settings or get_settings()
    app = FastAPI(
        title="Legal Evidence Workspace API",
        version=APP_VERSION,
        docs_url="/docs" if resolved.app_env != "production" else None,
        redoc_url=None,
    )
    app.state.settings = resolved

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[resolved.frontend_origin_value],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )
    app.add_middleware(RequestIDMiddleware)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(cases_router, prefix=resolved.api_prefix)
    return app


app = create_app()
