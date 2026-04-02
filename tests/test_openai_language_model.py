import pathlib
import sys

import httpx

PROJECT_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.language_models.instance.openai_language_model import (
    OPENAI_RETRYABLE_EXCEPTIONS,
)
from src.utils import ExponentialBackoffStrategy, RetryHandler


def test_openai_retryable_exceptions_include_transient_transport_errors() -> None:
    assert httpx.ReadTimeout in OPENAI_RETRYABLE_EXCEPTIONS
    assert httpx.ConnectTimeout in OPENAI_RETRYABLE_EXCEPTIONS
    assert httpx.ReadError in OPENAI_RETRYABLE_EXCEPTIONS


def test_retry_handler_retries_transient_transport_failures() -> None:
    calls = {"count": 0}

    @RetryHandler.handle(
        max_retries=2,
        waiting_strategy=ExponentialBackoffStrategy(multiplier=0),
        retry_on=OPENAI_RETRYABLE_EXCEPTIONS,
    )
    def flaky() -> str:
        calls["count"] += 1
        if calls["count"] < 3:
            raise httpx.ReadTimeout("simulated timeout")
        return "ok"

    assert flaky() == "ok"
    assert calls["count"] == 3
