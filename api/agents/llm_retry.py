"""
api/agents/llm_retry.py

Shared retry utility for Gemini API calls.
Handles 429 Rate Limit errors with exponential backoff and dual-key rotation.
"""
from __future__ import annotations

import asyncio
import logging
import re
import time
from typing import Any, Callable

import google.generativeai as genai

logger = logging.getLogger(__name__)


def _extract_retry_delay(error_str: str) -> float:
    """Extract the recommended retry delay in seconds from a 429 error message."""
    match = re.search(r"retry in ([0-9.]+)s", error_str)
    if match:
        return float(match.group(1))
    return 0.0


def _is_rate_limit(exc: Exception) -> bool:
    return "429" in str(exc) or "quota" in str(exc).lower()


async def call_with_retry(
    fn: Callable,
    *args,
    api_keys: list[str],
    model_name: str,
    max_attempts: int = 6,
    **kwargs,
) -> Any:
    """
    Call a synchronous Gemini/Instructor function with automatic retry and
    dual-key rotation.

    Args:
        fn:           The instructor client factory callable — receives the
                      configured genai.GenerativeModel and returns a result.
                      Signature: fn(client) -> result
        api_keys:     List of API keys to rotate through.
        model_name:   The Gemini model name to use.
        max_attempts: Total attempts before raising.
        **kwargs:     Extra kwargs forwarded to fn.
    """
    import instructor

    last_exc: Exception | None = None
    key_index = 0

    for attempt in range(max_attempts):
        key = api_keys[key_index % len(api_keys)]
        genai.configure(api_key=key)
        model = genai.GenerativeModel(model_name=model_name)
        client = instructor.from_gemini(client=model, mode=instructor.Mode.GEMINI_JSON)

        try:
            result = await asyncio.to_thread(fn, client, *args, **kwargs)
            return result
        except Exception as exc:
            last_exc = exc
            if _is_rate_limit(exc):
                # Respect the API's suggested delay if present
                suggested = _extract_retry_delay(str(exc))
                wait = max(suggested, (attempt + 1) * 3)  # at least 3s * attempt
                logger.warning(
                    "gemini_rate_limit_retrying",
                    extra={
                        "attempt": attempt + 1,
                        "wait_s": wait,
                        "key_index": key_index % len(api_keys),
                    },
                )
                key_index += 1  # rotate key on next attempt
                await asyncio.sleep(wait)
            else:
                raise  # non-rate-limit errors bubble up immediately

    raise last_exc
