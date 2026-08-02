from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, TypeVar


T = TypeVar("T")
DEFAULT_BACKOFF_SECONDS = (0.5, 1.0)


@dataclass(frozen=True)
class RetryNotice:
    operation: str
    attempt: int
    next_attempt: int
    delay_seconds: float
    error_category: str


def safe_error_category(exc: BaseException) -> str:
    """Return a stable, non-sensitive category suitable for traces."""
    name = type(exc).__name__.lower()
    status = getattr(exc, "status_code", None)
    if status in {401, 403} or "authentication" in name or "permission" in name:
        return "authentication"
    if status == 429 or "ratelimit" in name or "rate_limit" in name:
        return "rate_limit"
    if status in {408, 409}:
        return f"http_{status}"
    if isinstance(status, int) and status >= 500:
        return "server_error"
    if "timeout" in name:
        return "timeout"
    if "connection" in name:
        return "connection"
    if "validation" in name or isinstance(exc, (ValueError, TypeError)):
        return "invalid_request"
    return "provider_error"


def is_retryable(exc: BaseException) -> bool:
    category = safe_error_category(exc)
    return category in {
        "rate_limit",
        "http_408",
        "http_409",
        "server_error",
        "timeout",
        "connection",
    }


def retry_call(
    operation: str,
    call: Callable[[], T],
    *,
    backoff_seconds: tuple[float, ...] = DEFAULT_BACKOFF_SECONDS,
    sleep: Callable[[float], Any] = time.sleep,
    on_retry: Callable[[RetryNotice], Any] | None = None,
) -> T:
    """Call once plus one retry per backoff value; never retry permanent errors."""
    total_attempts = len(backoff_seconds) + 1
    for attempt in range(1, total_attempts + 1):
        try:
            return call()
        except Exception as exc:
            if attempt >= total_attempts or not is_retryable(exc):
                raise
            notice = RetryNotice(
                operation=operation,
                attempt=attempt,
                next_attempt=attempt + 1,
                delay_seconds=backoff_seconds[attempt - 1],
                error_category=safe_error_category(exc),
            )
            if on_retry is not None:
                on_retry(notice)
            sleep(notice.delay_seconds)
    raise AssertionError("unreachable retry state")
