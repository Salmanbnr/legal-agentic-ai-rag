"""Safe JSON error handling that never exposes stack traces or internals."""

from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from starlette.exceptions import HTTPException as StarletteHTTPException

logger = logging.getLogger(__name__)


def _request_id(request: Request) -> str:
    """Return the current request id, if middleware assigned one."""
    return getattr(request.state, "request_id", "unknown")


def _safe_validation_errors(exc: RequestValidationError) -> list[dict[str, str]]:
    """Keep validation metadata but drop submitted input values."""
    errors: list[dict[str, str]] = []
    for error in exc.errors():
        location = ".".join(str(part) for part in error.get("loc", ()))
        errors.append(
            {
                "loc": location,
                "type": str(error.get("type", "value_error")),
                "msg": str(error.get("msg", "Invalid value")),
            }
        )
    return errors


def register_exception_handlers(app: FastAPI) -> None:
    """Register JSON handlers for HTTP, validation, and unknown errors."""

    @app.exception_handler(StarletteHTTPException)
    async def handle_http_exception(
        request: Request, exc: StarletteHTTPException
    ) -> JSONResponse:
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail, "request_id": _request_id(request)},
            headers=exc.headers,
        )

    @app.exception_handler(RequestValidationError)
    async def handle_validation_error(
        request: Request, exc: RequestValidationError
    ) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "detail": "Request validation failed",
                "errors": _safe_validation_errors(exc),
                "request_id": _request_id(request),
            },
        )

    @app.exception_handler(Exception)
    async def handle_unexpected_error(
        request: Request, exc: Exception
    ) -> JSONResponse:
        request_id = _request_id(request)
        logger.error(
            "Unhandled error request_id=%s method=%s path=%s error_type=%s",
            request_id,
            request.method,
            request.url.path,
            type(exc).__name__,
        )
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error", "request_id": request_id},
            headers={"X-Request-ID": request_id},
        )
