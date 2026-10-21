"""Tests for the worker CLI entry point."""

from __future__ import annotations

from scripts import run_worker as run_worker_module


def test_parser_accepts_once_flag() -> None:
    args = run_worker_module.build_parser().parse_args(["--once"])

    assert args.once is True


def test_main_once_runs_a_single_poll(monkeypatch) -> None:
    calls: list[str] = []

    class FakeWorker:
        def __init__(self, *, settings) -> None:
            self.settings = settings

        def run_once(self) -> bool:
            calls.append("once")
            return True

        def run_forever(self) -> None:
            calls.append("forever")

    monkeypatch.setattr(run_worker_module, "JobWorker", FakeWorker)

    assert run_worker_module.main(["--once"]) == 0
    assert calls == ["once"]


def test_main_without_once_runs_forever(monkeypatch) -> None:
    calls: list[str] = []

    class FakeWorker:
        def __init__(self, *, settings) -> None:
            self.settings = settings

        def run_once(self) -> bool:
            calls.append("once")
            return True

        def run_forever(self) -> None:
            calls.append("forever")

    monkeypatch.setattr(run_worker_module, "JobWorker", FakeWorker)

    assert run_worker_module.main([]) == 0
    assert calls == ["forever"]


def test_main_validates_the_index_before_running(monkeypatch) -> None:
    checked: list[bool] = []

    class FakeWorker:
        def __init__(self, *, settings) -> None:
            self.settings = settings

        def run_once(self) -> bool:
            return True

        def run_forever(self) -> None:
            return None

    monkeypatch.setattr(run_worker_module, "JobWorker", FakeWorker)
    monkeypatch.setattr(
        run_worker_module,
        "validate_index_startup",
        lambda settings: checked.append(True),
    )

    assert run_worker_module.main(["--once"]) == 0
    assert checked == [True]
