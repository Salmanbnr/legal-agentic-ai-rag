"""The single LLM access point for the backend.

Every model call in the product goes through :class:`LLMClient`. It owns the
provider configuration, the timeout, the retry policy, the usage counters, and
the structured-output path. No other module imports the provider SDK.
"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, TypeVar

from pydantic import BaseModel

from app.core.config import Settings, get_settings

logger = logging.getLogger(__name__)

PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"

SchemaT = TypeVar("SchemaT", bound=BaseModel)


class LLMError(RuntimeError):
    """A safe LLM failure with a stable, persistable error code."""

    def __init__(self, error_code: str, *, retryable: bool = True) -> None:
        """Create an LLM error without including prompts or completions."""
        super().__init__(error_code)
        self.error_code = error_code
        self.retryable = retryable


class LLMBudgetExceeded(LLMError):
    """Raised when a request would exceed its LLM call budget."""

    def __init__(self) -> None:
        """Create a non-retryable budget error."""
        super().__init__("LLM_BUDGET_EXCEEDED", retryable=False)


@dataclass(frozen=True, slots=True)
class LLMUsage:
    """Call and token counters for one LLM client."""

    calls: int = 0
    retries: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


def _provider_status(exc: Exception | None) -> str:
    """Return the provider HTTP status of an SDK error without its message."""
    status = getattr(exc, "status_code", None)
    return str(status) if status is not None else "none"


def _is_retryable(exc: Exception) -> bool:
    """Return False for provider client errors that a retry cannot fix."""
    status = getattr(exc, "status_code", None)
    if isinstance(status, int) and 400 <= status < 500:
        return status in (408, 409, 429)
    return True


def load_prompt(name: str) -> str:
    """Load a prompt template from the packaged prompt directory."""
    path = PROMPTS_DIR / name
    try:
        return path.read_text(encoding="utf-8")
    except OSError as exc:
        raise LLMError("PROMPT_NOT_FOUND", retryable=False) from exc


class LLMClient:
    """Wrapper around ChatOpenAI with a timeout, retries, and usage counters.

    The provider client is created lazily so the API can start without the
    optional provider dependencies, and it can be replaced in tests.
    """

    def __init__(
        self,
        *,
        settings: Settings | None = None,
        chat: Any | None = None,
        sleep: Callable[[float], None] = time.sleep,
        rng: Callable[[float, float], float] = random.uniform,
    ) -> None:
        """Build a client with optional provider and timing overrides."""
        self.settings = settings or get_settings()
        self._chat = chat
        self._sleep = sleep
        self._rng = rng
        self._lock = threading.Lock()
        self._usage = LLMUsage()
        self._calls = 0
        self._retries = 0
        self._prompt_tokens = 0
        self._completion_tokens = 0
        self._total_tokens = 0

    @property
    def usage(self) -> LLMUsage:
        """Return a snapshot of the call and token counters."""
        with self._lock:
            return LLMUsage(
                calls=self._calls,
                retries=self._retries,
                prompt_tokens=self._prompt_tokens,
                completion_tokens=self._completion_tokens,
                total_tokens=self._total_tokens,
            )

    def reset_usage(self) -> None:
        """Reset the counters, mainly for tests and per-request budgets."""
        with self._lock:
            self._calls = 0
            self._retries = 0
            self._prompt_tokens = 0
            self._completion_tokens = 0
            self._total_tokens = 0

    def _chat_model(self) -> Any:
        """Return the provider chat model, creating it on first use."""
        if self._chat is None:
            try:
                from langchain_openai import ChatOpenAI
            except Exception:
                raise LLMError("LLM_PROVIDER_UNAVAILABLE", retryable=False) from None
            self._chat = ChatOpenAI(
                model=self.settings.openai_model,
                api_key=self.settings.openai_api_key.get_secret_value(),
                temperature=0,
                timeout=self.settings.openai_timeout_seconds,
                max_retries=0,
                max_tokens=self.settings.openai_max_output_tokens,
            )
        return self._chat

    def _record_usage(self, response: Any) -> None:
        """Add the token counters of one response to the totals."""
        usage = getattr(response, "usage_metadata", None)
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        if isinstance(usage, dict):
            prompt_tokens = int(usage.get("input_tokens") or 0)
            completion_tokens = int(usage.get("output_tokens") or 0)
            total_tokens = int(usage.get("total_tokens") or 0)
        if not total_tokens:
            metadata = getattr(response, "response_metadata", None) or {}
            token_usage = metadata.get("token_usage") or {}
            if isinstance(token_usage, dict):
                prompt_tokens = prompt_tokens or int(
                    token_usage.get("prompt_tokens") or 0
                )
                completion_tokens = completion_tokens or int(
                    token_usage.get("completion_tokens") or 0
                )
                total_tokens = total_tokens or int(
                    token_usage.get("total_tokens") or 0
                )
        with self._lock:
            self._prompt_tokens += prompt_tokens
            self._completion_tokens += completion_tokens
            self._total_tokens += total_tokens

    def _delay(self, attempt: int) -> float:
        """Return the capped exponential backoff delay with jitter."""
        nominal = min(
            self.settings.openai_retry_max_seconds,
            self.settings.openai_retry_base_seconds * (2 ** (attempt - 1)),
        )
        return min(
            self.settings.openai_retry_max_seconds,
            nominal * self._rng(0.5, 1.5),
        )

    def _run(self, runnable: Any, messages: list[tuple[str, str]]) -> Any:
        """Invoke a runnable with retries, counting calls and tokens."""
        attempts = self.settings.openai_max_retries + 1
        last_error: Exception | None = None
        for attempt in range(1, attempts + 1):
            try:
                response = runnable.invoke(messages)
            except Exception as exc:
                last_error = exc
                if attempt >= attempts or not _is_retryable(exc):
                    break
                with self._lock:
                    self._retries += 1
                delay = self._delay(attempt)
                logger.warning(
                    "llm call failed, retrying error_class=%s provider_status=%s",
                    type(exc).__name__,
                    _provider_status(exc),
                    extra={
                        "attempt": attempt,
                        "delay_seconds": round(delay, 2),
                    },
                )
                self._sleep(delay)
                continue
            with self._lock:
                self._calls += 1
            self._record_usage(response)
            return response
        logger.error(
            "llm call failed error_class=%s provider_status=%s",
            type(last_error).__name__ if last_error else "None",
            _provider_status(last_error),
        )
        raise LLMError("LLM_CALL_FAILED") from None

    def invoke(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_calls: int | None = None,
    ) -> str:
        """Run one completion and return its text.

        Args:
            system_prompt: Instructions that define the model's task.
            user_prompt: The user turn, which may contain untrusted evidence.
            max_calls: Optional cap on the calls this client may still make.

        Raises:
            LLMBudgetExceeded: When the call budget is already spent.
        """
        self._check_budget(max_calls)
        runnable = self._chat_model()
        response = self._run(
            runnable, [("system", system_prompt), ("human", user_prompt)]
        )
        content = getattr(response, "content", response)
        if isinstance(content, list):
            parts = [
                part.get("text", "")
                for part in content
                if isinstance(part, dict) and part.get("text")
            ]
            content = "".join(parts)
        return str(content or "")

    def invoke_structured(
        self,
        system_prompt: str,
        user_prompt: str,
        schema: type[SchemaT],
        *,
        max_calls: int | None = None,
    ) -> SchemaT:
        """Run one completion and parse it into ``schema``.

        Args:
            system_prompt: Instructions that define the model's task.
            user_prompt: The user turn, which may contain untrusted evidence.
            schema: Pydantic model the response must satisfy.
            max_calls: Optional cap on the calls this client may still make.

        Raises:
            LLMError: When the provider fails or returns an invalid object.
        """
        self._check_budget(max_calls)
        runnable = self._chat_model().with_structured_output(schema)
        response = self._run(
            runnable, [("system", system_prompt), ("human", user_prompt)]
        )
        if isinstance(response, schema):
            return response
        if isinstance(response, dict):
            try:
                return schema.model_validate(response)
            except Exception:
                raise LLMError("LLM_INVALID_STRUCTURED_OUTPUT") from None
        raise LLMError("LLM_INVALID_STRUCTURED_OUTPUT")

    def _check_budget(self, max_calls: int | None) -> None:
        """Raise when the optional call budget is already spent."""
        if max_calls is None:
            return
        with self._lock:
            used = self._calls + self._retries
        if used >= max_calls:
            raise LLMBudgetExceeded()
