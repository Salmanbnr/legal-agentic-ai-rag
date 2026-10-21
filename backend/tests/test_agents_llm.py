"""Tests for the single LLM access point."""

from __future__ import annotations

import sys
import types
from typing import Any

import pytest
from fakes import FakeChat, FakeLLMResponse

from app.agents.llm import (
    LLMBudgetExceeded,
    LLMClient,
    LLMError,
    load_prompt,
)
from app.core.config import Settings
from pydantic import BaseModel


class _Payload(BaseModel):
    """Small schema used to check structured output."""

    text: str
    count: int = 0


def _settings(**overrides: Any) -> Settings:
    """Return test settings with the given overrides."""
    return Settings(_env_file=None).model_copy(update=overrides)


def test_prompt_files_are_loadable() -> None:
    prompt = load_prompt("answer_baseline.txt")

    assert "EVIDENCE" in prompt.upper()
    assert "{question}" in prompt


def test_missing_prompt_uses_a_safe_error_code() -> None:
    with pytest.raises(LLMError) as error:
        load_prompt("missing_prompt.txt")

    assert error.value.error_code == "PROMPT_NOT_FOUND"
    assert error.value.retryable is False


def test_invoke_returns_text_and_counts_calls_and_tokens() -> None:
    chat = FakeChat(
        FakeLLMResponse(
            "The answer.",
            prompt_tokens=120,
            completion_tokens=45,
        )
    )
    client = LLMClient(settings=_settings(), chat=chat)

    text = client.invoke("system", "user")

    assert text == "The answer."
    usage = client.usage
    assert usage.calls == 1
    assert usage.prompt_tokens == 120
    assert usage.completion_tokens == 45
    assert usage.total_tokens == 165


def test_invoke_retries_with_exponential_backoff_and_jitter() -> None:
    chat = FakeChat(FakeLLMResponse("ok"), failures=2)
    delays: list[float] = []
    client = LLMClient(
        settings=_settings(openai_retry_base_seconds=1.0),
        chat=chat,
        sleep=delays.append,
        rng=lambda low, high: 1.0,
    )

    text = client.invoke("system", "user")

    assert text == "ok"
    assert len(delays) == 2
    assert delays[0] == pytest.approx(1.0)
    assert delays[1] == pytest.approx(2.0)
    assert client.usage.calls == 1
    assert client.usage.retries == 2


def test_jitter_changes_the_delay() -> None:
    chat = FakeChat(FakeLLMResponse("ok"), failures=1)
    delays: list[float] = []
    client = LLMClient(
        settings=_settings(),
        chat=chat,
        sleep=delays.append,
        rng=lambda low, high: 1.5,
    )

    client.invoke("system", "user")

    assert delays == [pytest.approx(1.5)]


def test_invoke_raises_a_safe_error_after_the_last_attempt() -> None:
    chat = FakeChat(FakeLLMResponse("ok"), failures=10)
    client = LLMClient(
        settings=_settings(openai_max_retries=2),
        chat=chat,
        sleep=lambda seconds: None,
        rng=lambda low, high: 1.0,
    )

    with pytest.raises(LLMError) as error:
        client.invoke("system", "user")

    assert error.value.error_code == "LLM_CALL_FAILED"
    assert "fake llm failure" not in str(error.value)
    assert len(chat.calls) == 3


def test_provider_client_errors_are_not_retried() -> None:
    class _AuthError(RuntimeError):
        status_code = 401

    class _FailingChat:
        def __init__(self) -> None:
            self.calls = 0

        def invoke(self, messages: Any) -> Any:
            self.calls += 1
            raise _AuthError("unauthorized")

    chat = _FailingChat()
    client = LLMClient(
        settings=_settings(openai_max_retries=3),
        chat=chat,
        sleep=lambda seconds: None,
    )

    with pytest.raises(LLMError) as error:
        client.invoke("system", "user")

    assert error.value.error_code == "LLM_CALL_FAILED"
    assert chat.calls == 1
    assert client.usage.retries == 0


def test_invoke_structured_returns_the_schema() -> None:
    chat = FakeChat(_Payload(text="hello", count=2))
    client = LLMClient(settings=_settings(), chat=chat)

    result = client.invoke_structured("system", "user", _Payload)

    assert isinstance(result, _Payload)
    assert result.text == "hello"
    assert chat.schemas == [_Payload]


def test_invoke_structured_validates_a_mapping() -> None:
    chat = FakeChat({"text": "hello", "count": 3})
    client = LLMClient(settings=_settings(), chat=chat)

    result = client.invoke_structured("system", "user", _Payload)

    assert result.count == 3


def test_invoke_structured_rejects_an_invalid_object() -> None:
    chat = FakeChat(object())
    client = LLMClient(settings=_settings(), chat=chat)

    with pytest.raises(LLMError) as error:
        client.invoke_structured("system", "user", _Payload)

    assert error.value.error_code == "LLM_INVALID_STRUCTURED_OUTPUT"


def test_the_call_budget_blocks_further_calls() -> None:
    chat = FakeChat(FakeLLMResponse("ok"))
    client = LLMClient(settings=_settings(), chat=chat)

    client.invoke("system", "user", max_calls=1)

    with pytest.raises(LLMBudgetExceeded):
        client.invoke("system", "user", max_calls=1)


def test_usage_can_be_reset() -> None:
    chat = FakeChat(FakeLLMResponse("ok", prompt_tokens=10, completion_tokens=5))
    client = LLMClient(settings=_settings(), chat=chat)

    client.invoke("system", "user")
    client.reset_usage()

    assert client.usage.calls == 0
    assert client.usage.total_tokens == 0


def test_the_provider_client_is_configured_with_timeout_and_zero_temperature(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, Any] = {}

    class FakeChatOpenAI:
        def __init__(self, **kwargs: Any) -> None:
            captured.update(kwargs)

        def invoke(self, messages: Any) -> Any:
            return FakeLLMResponse("ok")

    module = types.ModuleType("langchain_openai")
    module.ChatOpenAI = FakeChatOpenAI  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "langchain_openai", module)

    settings = _settings(openai_timeout_seconds=12.0)
    client = LLMClient(settings=settings)
    client.invoke("system", "user")

    assert captured["model"] == settings.openai_model
    assert captured["temperature"] == 0
    assert captured["timeout"] == 12.0
    assert captured["max_retries"] == 0
    assert captured["api_key"] == settings.openai_api_key.get_secret_value()
