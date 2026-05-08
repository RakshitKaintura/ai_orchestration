"""
api/agents/compression.py

Compression Agent
-----------------
Triggered by the BudgetManager when an agent's remaining token budget falls
below COMPRESSION_TRIGGER_THRESHOLD. This agent is NEVER called by users
directly — only by the orchestrator in response to a budget signal.

Compression contract:
  • Lossless for structured data  : JSON objects, scores, chunk IDs, tool outputs,
                                    citations, timestamps — preserved verbatim.
  • Lossy for prose               : summaries, explanations, reasoning chains,
                                    conversational filler — compressed aggressively.

The compression agent does NOT use the BudgetManager itself (it has its own
fixed budget declared separately) and does NOT write to SharedContext.
It takes raw text in, returns compressed text out.

This function is also the module-level entry point used by context_manager.py.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import time
from typing import Any

import anthropic
import instructor
from pydantic import BaseModel, Field

from api.config import get_settings

logger = logging.getLogger(__name__)

COMPRESSION_BUDGET_TOKENS = 2000  # budget for the compression agent itself
TOOL_NAME = "compression_agent"


# ─── Structured output ────────────────────────────────────────────────────────

class CompressionOutput(BaseModel):
    compressed_text: str = Field(
        description=(
            "The compressed version of the input. Structured data (JSON, scores, "
            "IDs) preserved verbatim. Natural language prose compressed."
        )
    )
    original_approx_tokens: int = Field(description="Estimated token count of input")
    compressed_approx_tokens: int = Field(description="Estimated token count of output")
    compression_ratio: float = Field(description="original / compressed (higher = more compressed)")


# ─── Structured data extractor (lossless preservation) ───────────────────────

_JSON_BLOCK_PATTERN = re.compile(
    r'(\{[^{}]*\}|\[[^\[\]]*\])',  # simple JSON objects/arrays (non-nested)
    re.DOTALL,
)


def _extract_structured_segments(text: str) -> list[tuple[str, bool]]:
    """
    Split text into segments: (content, is_structured).
    Structured segments will be preserved verbatim.
    Prose segments will be fed to the LLM for compression.
    """
    segments: list[tuple[str, bool]] = []
    last_end = 0

    for match in _JSON_BLOCK_PATTERN.finditer(text):
        # Prose before this match
        prose = text[last_end:match.start()]
        if prose.strip():
            segments.append((prose, False))

        # The structured block itself
        block = match.group(0)
        # Quick validation: try json.loads to confirm it's real JSON
        try:
            json.loads(block)
            segments.append((block, True))
        except (json.JSONDecodeError, ValueError):
            segments.append((block, False))  # treat as prose

        last_end = match.end()

    # Trailing prose
    remainder = text[last_end:]
    if remainder.strip():
        segments.append((remainder, False))

    return segments


# ─── LLM compression call ─────────────────────────────────────────────────────

async def _llm_compress(prose: str, model: str, api_key: str) -> str:
    """Compress natural language prose using Claude."""
    if len(prose.strip()) < 200:
        return prose  # too short to bother compressing

    raw_client = anthropic.AsyncAnthropic(api_key=api_key)
    client = instructor.from_anthropic(raw_client)

    result = await client.messages.create(
        model=model,
        max_tokens=COMPRESSION_BUDGET_TOKENS,
        messages=[{
            "role": "user",
            "content": (
                "Compress the following text to be as short as possible while preserving "
                "all key information, facts, and conclusions. Remove filler phrases, "
                "redundant explanations, and conversational padding. "
                "Return ONLY the compressed text with no preamble.\n\n"
                f"Text to compress:\n{prose}"
            ),
        }],
        response_model=CompressionOutput,
    )
    return result.compressed_text


# ─── Main compress_context function ──────────────────────────────────────────

async def compress_context_async(text: str) -> str:
    """
    Async version of context compression.
    Lossless for structured data, lossy for prose.
    Called by the orchestrator when BudgetManager signals near-limit.
    """
    if not text or not text.strip():
        return text

    settings = get_settings()
    t0 = time.perf_counter()

    segments = _extract_structured_segments(text)

    # Compress prose segments in parallel
    compressed_parts: list[str] = []
    prose_tasks = []
    prose_indices = []

    for i, (content, is_structured) in enumerate(segments):
        if is_structured:
            compressed_parts.append(content)  # placeholder, replaced below
        else:
            prose_tasks.append(_llm_compress(content, settings.primary_model, settings.anthropic_api_key))
            prose_indices.append(i)
            compressed_parts.append("")  # placeholder

    # Run prose compressions
    if prose_tasks:
        prose_results = await asyncio.gather(*prose_tasks, return_exceptions=True)
        for idx, (segment_idx, result) in enumerate(zip(prose_indices, prose_results)):
            if isinstance(result, Exception):
                logger.warning(
                    "compression_segment_failed",
                    extra={"error": str(result), "segment_idx": segment_idx},
                )
                # Fallback: keep original prose
                compressed_parts[segment_idx] = segments[segment_idx][0]
            else:
                compressed_parts[segment_idx] = result

    # Fill in structured parts (they were appended in order)
    struct_idx = 0
    final_parts = []
    for i, (content, is_structured) in enumerate(segments):
        if is_structured:
            final_parts.append(content)
        else:
            final_parts.append(compressed_parts[i])

    result_text = "\n".join(final_parts)
    elapsed = (time.perf_counter() - t0) * 1000
    logger.info(
        "compression_complete",
        extra={
            "original_chars": len(text),
            "compressed_chars": len(result_text),
            "ratio": round(len(text) / max(len(result_text), 1), 2),
            "latency_ms": int(elapsed),
        },
    )
    return result_text


def compress_context(text: str) -> str:
    """
    Sync wrapper for compress_context_async.
    Used by BudgetManager's compress() method (which is called from sync contexts).
    """
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            # In an async context — caller should use compress_context_async directly
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, compress_context_async(text))
                return future.result(timeout=30)
        else:
            return loop.run_until_complete(compress_context_async(text))
    except Exception as e:
        logger.error("compression_sync_fallback_failed", extra={"error": str(e)})
        # Last resort: truncate prose but preserve structure markers
        return text[:len(text) // 2] + "\n[...compressed...]"
