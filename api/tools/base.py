"""
api/tools/base.py

Base infrastructure for all tools:
 - ToolLogger  : writes every call to SharedContext.tool_call_log + DB trace_events
 - tool_call   : async decorator that wraps any tool coroutine with:
     • timing
     • retry-aware logging (retry_num tracked)
     • accepted/rejection recording
     • failure-mode routing constants

The orchestrator calls tools exclusively through call_tool_with_retry()
defined at the bottom of this module.

Failure-mode routing (orchestrator handles in code, not in prompts):
  "timeout"   → simplify / shorten input
  "empty"     → broaden / relax input
  "malformed" → fix format (not content)
"""

from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import logging
import time
from typing import Any, Callable, Coroutine

from api.models.context import SharedContext, ToolCall
from api.models.tools import ToolResult

logger = logging.getLogger(__name__)

MAX_TOOL_RETRIES = 2   # up to 2 retries (3 total attempts)


# ─── Hash helpers ─────────────────────────────────────────────────────────────

def _short_hash(obj: Any) -> str:
    serialized = json.dumps(obj, default=str, sort_keys=True)
    return hashlib.sha256(serialized.encode()).hexdigest()[:16]


# ─── Tool logger ──────────────────────────────────────────────────────────────

class ToolLogger:
    """
    Appends a ToolCall record to SharedContext.tool_call_log.
    This is the canonical log — trace_events DB writes happen separately
    in the orchestrator's _log_trace_event() method.
    """

    @staticmethod
    def log(
        ctx: SharedContext,
        tool_name: str,
        agent_id: str,
        input_data: dict,
        result: ToolResult,
        retry_num: int = 0,
        accepted: bool | None = None,
        rejection_reason: str | None = None,
    ) -> ToolCall:
        """
        Record a tool call in the shared context.
        accepted=None means the decision hasn't been made yet (pre-acceptance check).
        """
        call = ToolCall(
            tool=tool_name,
            input=input_data,
            output=result.data if result.success else {
                "error_type": result.error_type,
                "error_message": result.error_message,
            },
            latency_ms=int(result.latency_ms),
            success=result.success,
            error_type=result.error_type,
            accepted=accepted if accepted is not None else result.success,
            rejection_reason=rejection_reason,
            retry_num=retry_num,
            agent_id=agent_id,
        )
        ctx.tool_call_log.append(call)
        logger.info(
            "tool_call_logged",
            extra={
                "tool": tool_name,
                "agent_id": agent_id,
                "retry_num": retry_num,
                "success": result.success,
                "error_type": result.error_type,
                "latency_ms": call.latency_ms,
                "accepted": call.accepted,
                "job_id": str(ctx.job_id),
            },
        )
        return call


# ─── Input mutation helpers (called by orchestrator between retries) ──────────

def simplify_input(input_data: dict) -> dict:
    """
    Timeout strategy: shorten / reduce the query to be faster.
    Works on any dict with a 'query' or 'code' key.
    """
    result = dict(input_data)
    if "query" in result and isinstance(result["query"], str):
        # Trim to first sentence / first 100 chars
        q = result["query"]
        first_sentence = q.split(".")[0] + "." if "." in q else q[:100]
        result["query"] = first_sentence.strip()
    if "code" in result and isinstance(result["code"], str):
        # Trim code to first 20 lines
        lines = result["code"].splitlines()[:20]
        result["code"] = "\n".join(lines)
    result["_retry_strategy"] = "simplified"
    return result


def broaden_input(input_data: dict) -> dict:
    """
    Empty-result strategy: relax constraints / add wildcard.
    """
    result = dict(input_data)
    if "query" in result and isinstance(result["query"], str):
        result["query"] = result["query"].rstrip("?") + " OR related topics"
    if "filters" in result:
        result["filters"] = {}  # remove filters
    if "limit" in result:
        result["limit"] = min(result["limit"] * 2, 20)
    result["_retry_strategy"] = "broadened"
    return result


def fix_format(input_data: dict) -> dict:
    """
    Malformed-input strategy: clean up types and encoding.
    """
    result = {}
    for k, v in input_data.items():
        if k.startswith("_"):
            continue
        # Ensure string values are actually strings
        if isinstance(v, (bytes, bytearray)):
            result[k] = v.decode("utf-8", errors="replace")
        elif not isinstance(v, (str, int, float, bool, list, dict, type(None))):
            result[k] = str(v)
        else:
            result[k] = v
    result["_retry_strategy"] = "format_fixed"
    return result


def _mutate_input(input_data: dict, error_type: str) -> dict:
    """Route to the right mutation strategy based on failure type."""
    if error_type == "timeout":
        return simplify_input(input_data)
    elif error_type == "empty":
        return broaden_input(input_data)
    elif error_type == "malformed":
        return fix_format(input_data)
    return input_data


# ─── call_tool_with_retry ────────────────────────────────────────────────────

async def call_tool_with_retry(
    tool_fn: Callable[..., Coroutine[Any, Any, ToolResult]],
    input_data: dict,
    ctx: SharedContext,
    agent_id: str,
    tool_name: str,
    *,
    acceptance_check: Callable[[ToolResult], tuple[bool, str | None]] | None = None,
    max_retries: int = MAX_TOOL_RETRIES,
) -> ToolResult:
    """
    Call a tool with up to `max_retries` retries.

    Each retry:
    1. Mutates input_data based on the previous error_type (in code, not prompt)
    2. Logs the attempt with its retry_num
    3. Checks acceptance via acceptance_check(result) → (accepted: bool, reason: str|None)
       If the agent rejects a technically-successful result, it counts as a retry trigger.

    Returns the last ToolResult regardless of success.
    All attempts are logged to ctx.tool_call_log.
    """
    current_input = dict(input_data)
    last_result: ToolResult | None = None

    for attempt in range(max_retries + 1):
        t0 = time.perf_counter()
        result = await tool_fn(current_input)
        # Patch latency if tool didn't set it
        if result.latency_ms == 0:
            result.latency_ms = (time.perf_counter() - t0) * 1000

        # Determine acceptance
        accepted = result.success
        rejection_reason: str | None = None

        if result.success and acceptance_check:
            accepted, rejection_reason = acceptance_check(result)

        # Log this attempt
        ToolLogger.log(
            ctx=ctx,
            tool_name=tool_name,
            agent_id=agent_id,
            input_data=current_input,
            result=result,
            retry_num=attempt,
            accepted=accepted,
            rejection_reason=rejection_reason,
        )

        last_result = result

        # Stop if accepted
        if accepted:
            return result

        # Stop if out of retries
        if attempt >= max_retries:
            logger.warning(
                "tool_max_retries_reached",
                extra={
                    "tool": tool_name,
                    "agent_id": agent_id,
                    "attempts": attempt + 1,
                    "last_error_type": result.error_type,
                    "job_id": str(ctx.job_id),
                },
            )
            break

        # Mutate input for next attempt
        error_type = result.error_type or ("rejected" if not accepted else "unknown")
        current_input = _mutate_input(current_input, error_type)
        logger.info(
            "tool_retry",
            extra={
                "tool": tool_name,
                "agent_id": agent_id,
                "attempt": attempt + 1,
                "strategy": current_input.get("_retry_strategy", "unknown"),
                "job_id": str(ctx.job_id),
            },
        )

    return last_result  # type: ignore[return-value]
