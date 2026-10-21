"""FastAPI application factory and entrypoint."""

from __future__ import annotations

import threading
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.answers import router as answers_router
from app.api.cases import router as cases_router
from app.api.documents import router as documents_router
from app.api.health import router as health_router
from app.api.search import router as search_router
from app.core.config import Settings, get_settings
from app.core.errors import register_exception_handlers
from app.core.middleware import RequestIDMiddleware, RequestLogMiddleware
from app.core.rate_limit import InMemoryRateLimiter
from app.indexing.startup import validate_index_startup
from app.jobs.runner import JobWorker
from app.jobs.startup import validate_job_schema
from app.observability.logging import configure_logging

APP_VERSION = "0.1.0"


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the FastAPI application.

    Configuration is validated while the app is created, so a missing
    environment variable fails startup instead of the first request.
    """
    resolved = settings or get_settings()
    configure_logging(resolved.log_level, resolved.log_file)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        """Validate the index, then run the in-process worker for the app."""
        validate_index_startup(resolved)
        stop_event = threading.Event()
        worker_thread: threading.Thread | None = None
        if resolved.worker_enabled:
            # Refuse to start a worker that could never claim a job.
            validate_job_schema(resolved)
            worker = JobWorker(settings=resolved, stop_event=stop_event)
            worker_thread = threading.Thread(
                target=worker.run_forever,
                name="ingestion-worker",
                daemon=True,
            )
            worker_thread.start()
        app.state.worker_thread = worker_thread
        app.state.worker_stop_event = stop_event
        try:
            yield
        finally:
            stop_event.set()
            if worker_thread is not None:
                worker_thread.join(timeout=5.0)

    app = FastAPI(
        title="Legal Evidence Workspace API",
        version=APP_VERSION,
        docs_url="/docs" if resolved.app_env != "production" else None,
        redoc_url=None,
        lifespan=lifespan,
    )
    app.state.settings = resolved
    app.state.upload_rate_limiter = InMemoryRateLimiter(
        limit=resolved.upload_rate_limit_per_minute,
        window_seconds=60,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[resolved.frontend_origin_value],
        allow_credentials=True,
        allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
        max_age=600,
    )
    app.add_middleware(RequestLogMiddleware)
    # Added last, so the outermost layer assigns the id before logging runs.
    app.add_middleware(RequestIDMiddleware)
    register_exception_handlers(app)
    app.include_router(health_router)
    app.include_router(cases_router, prefix=resolved.api_prefix)
    app.include_router(documents_router, prefix=resolved.api_prefix)
    app.include_router(search_router, prefix=resolved.api_prefix)
    app.include_router(answers_router, prefix=resolved.api_prefix)
    return app


app = create_app()
