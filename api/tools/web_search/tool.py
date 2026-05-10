"""
api/tools/web_search/tool.py

Web Search Tool (production stub backed by curated fixtures).
"""

from __future__ import annotations

import asyncio
import time

from api.models.tools import ToolResult, WebSearchResult
from api.tools.web_search.fixtures import select_fixture

SEARCH_TIMEOUT_SECONDS = 5.0
TOOL_NAME = "web_search"


async def _execute_search(query: str, limit: int = 5) -> list[WebSearchResult]:
    """Simulate a web search with 0.3–0.8s realistic latency."""
    await asyncio.sleep(0.3 + hash(query) % 100 / 200)
    rows = select_fixture(query)
    return [
        WebSearchResult(url=url, title=title, snippet=snippet, relevance_score=score)
        for url, title, snippet, score in rows[:limit]
    ]


async def web_search(input_data: dict) -> ToolResult:
    """
    Tool #1 — Web Search

    Input schema:
        query   (str, required)  — search query
        limit   (int, optional)  — max results (default 5, max 10)

    Failure contract:
        malformed  → query missing or not a non-empty string
        timeout    → execution exceeds SEARCH_TIMEOUT_SECONDS
        empty      → no results found for query
    """
    t0 = time.perf_counter()

    query = input_data.get("query")
    if not isinstance(query, str) or not query.strip():
        return ToolResult.malformed(source=TOOL_NAME, message="'query' must be a non-empty string")

    limit = input_data.get("limit", 5)
    if not isinstance(limit, int) or limit < 1:
        limit = 5
    limit = min(limit, 10)

    try:
        results = await asyncio.wait_for(
            _execute_search(query.strip(), limit),
            timeout=SEARCH_TIMEOUT_SECONDS,
        )
    except asyncio.TimeoutError:
        latency = (time.perf_counter() - t0) * 1000
        return ToolResult.timeout(
            source=TOOL_NAME,
            latency_ms=latency,
            message=f"Web search timed out after {SEARCH_TIMEOUT_SECONDS}s",
        )

    latency = (time.perf_counter() - t0) * 1000

    if not results:
        return ToolResult.empty(
            source=TOOL_NAME, latency_ms=latency,
            message=f"No results found for query: '{query}'",
        )

    return ToolResult.ok(
        data=[r.model_dump() for r in results],
        source=TOOL_NAME,
        latency_ms=latency,
    )
