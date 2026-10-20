"""Tests for the FastAPI app, health route, middleware, and safe errors."""

from __future__ import annotations

import re

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from pydantic import BaseModel

from app.core.config import get_settings
from app.main import APP_VERSION, create_app


@pytest.fixture()
def app() -> FastAPI:
    get_settings.cache_clear()
    return create_app()


@pytest.fixture()
def client(app: FastAPI) -> TestClient:
    with TestClient(app) as test_client:
        yield test_client


def test_health_returns_ok(client: TestClient) -> None:
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "legal-ai-backend",
        "version": APP_VERSION,
    }
    assert response.headers["X-Request-ID"]


def test_request_id_is_preserved_when_safe(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "req-abc_123"})

    assert response.headers["X-Request-ID"] == "req-abc_123"


def test_request_id_is_replaced_when_unsafe(client: TestClient) -> None:
    response = client.get("/health", headers={"X-Request-ID": "unsafe id!"})

    generated = response.headers["X-Request-ID"]
    assert generated != "unsafe id!"
    assert re.fullmatch(r"[0-9a-f]{32}", generated)


def test_unknown_route_returns_safe_json(client: TestClient) -> None:
    response = client.get("/does-not-exist")

    assert response.status_code == 404
    body = response.json()
    assert body["detail"] == "Not Found"
    assert body["request_id"] == response.headers["X-Request-ID"]


def test_unhandled_error_hides_internal_details(app: FastAPI) -> None:
    @app.get("/_test/boom")
    def boom() -> None:
        raise RuntimeError("confidential document text")

    with TestClient(app, raise_server_exceptions=False) as test_client:
        response = test_client.get("/_test/boom")

    assert response.status_code == 500
    assert response.json()["detail"] == "Internal server error"
    assert response.json()["request_id"] == response.headers["X-Request-ID"]
    assert "confidential document text" not in response.text
    assert "Traceback" not in response.text


def test_validation_errors_are_sanitized(app: FastAPI, client: TestClient) -> None:
    class Payload(BaseModel):
        name: str

    @app.post("/_test/payload")
    def payload(body: Payload) -> dict[str, str]:
        return {"name": body.name}

    response = client.post("/_test/payload", json={"name": 123})

    assert response.status_code == 422
    body = response.json()
    assert body["detail"] == "Request validation failed"
    assert body["request_id"] == response.headers["X-Request-ID"]
    assert body["errors"]
    assert set(body["errors"][0]) == {"loc", "type", "msg"}
    assert "Traceback" not in response.text


def test_cors_allows_configured_frontend(client: TestClient) -> None:
    response = client.options(
        "/health",
        headers={
            "Origin": "http://localhost:3000",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:3000"


def test_cors_rejects_unknown_origin(client: TestClient) -> None:
    response = client.options(
        "/health",
        headers={
            "Origin": "https://attacker.example",
            "Access-Control-Request-Method": "GET",
        },
    )

    assert "access-control-allow-origin" not in response.headers
