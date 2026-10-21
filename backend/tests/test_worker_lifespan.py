"""Tests for the FastAPI lifespan-managed worker thread."""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app


def test_lifespan_starts_and_stops_worker(monkeypatch) -> None:
    started = threading.Event()
    stopped = threading.Event()

    class FakeWorker:
        def __init__(self, *, settings, stop_event) -> None:
            self.settings = settings
            self.stop_event = stop_event

        def run_forever(self) -> None:
            started.set()
            while not self.stop_event.is_set():
                self.stop_event.wait(0.01)
            stopped.set()

    monkeypatch.setattr("app.main.JobWorker", FakeWorker)
    settings = Settings(_env_file=None, worker_enabled=True)
    app = create_app(settings)

    with TestClient(app):
        assert started.wait(1.0)
        assert app.state.worker_thread is not None
        assert app.state.worker_thread.is_alive()

    assert stopped.wait(1.0)
    assert app.state.worker_thread is not None
    assert not app.state.worker_thread.is_alive()
