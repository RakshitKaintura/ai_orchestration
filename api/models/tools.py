"""
api/models/tools.py

Tool result schema with typed failure contracts.
Every tool in the system returns a ToolResult — never raises exceptions to callers.
The orchestrator inspects error_type to decide how to handle failures in code
(not in prompts).
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


class ToolResult(BaseModel):
    """
    Standardised return type for every tool.

    Failure contract:
    - success=True  → data contains the result, error_type is None
    - success=False → data is None, error_type is one of: "timeout" | "empty" | "malformed"

    The orchestrator handles each error_type differently:
    - "timeout"   → simplify/shorten the query and retry
    - "empty"     → broaden the query and retry
    - "malformed" → fix the input format and retry
    After max_retries (2) the orchestrator logs the failure and moves on.
    """

    # Core result
    success: bool = Field(description="True iff the tool produced usable data")
    data: Any | None = Field(
        default=None,
        description="The tool's output on success; None on failure",
    )

    # Failure contract fields
    error_type: Literal["timeout", "empty", "malformed"] | None = Field(
        default=None,
        description="Machine-readable failure category (None on success)",
    )
    error_message: str | None = Field(
        default=None,
        description="Human-readable description of what went wrong",
    )

    # Metadata
    latency_ms: float = Field(default=0.0, description="Tool execution time in milliseconds")
    source: str = Field(default="", description="Name of the tool that produced this result")
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    # ── Factory helpers ────────────────────────────────────────────────────────

    @classmethod
    def ok(cls, data: Any, source: str, latency_ms: float = 0.0) -> "ToolResult":
        """Create a successful result."""
        return cls(success=True, data=data, source=source, latency_ms=latency_ms)

    @classmethod
    def timeout(cls, source: str, latency_ms: float, message: str = "Tool timed out") -> "ToolResult":
        return cls(
            success=False,
            error_type="timeout",
            error_message=message,
            source=source,
            latency_ms=latency_ms,
        )

    @classmethod
    def empty(cls, source: str, latency_ms: float = 0.0, message: str = "Tool returned no results") -> "ToolResult":
        return cls(
            success=False,
            error_type="empty",
            error_message=message,
            source=source,
            latency_ms=latency_ms,
        )

    @classmethod
    def malformed(cls, source: str, message: str, latency_ms: float = 0.0) -> "ToolResult":
        return cls(
            success=False,
            error_type="malformed",
            error_message=message,
            source=source,
            latency_ms=latency_ms,
        )


class WebSearchResult(BaseModel):
    """A single result from the web search tool."""
    url: str
    title: str
    snippet: str
    relevance_score: float = Field(ge=0.0, le=1.0)


class CodeExecutionResult(BaseModel):
    """Output from the code execution sandbox tool."""
    stdout: str
    stderr: str
    exit_code: int
    execution_time_ms: float


class SQLQueryResult(BaseModel):
    """Output from the NL→SQL lookup tool."""
    sql: str = Field(description="The SQL query that was generated and executed")
    rows: list[dict[str, Any]] = Field(description="Result rows as list of dicts")
    row_count: int
    columns: list[str]


class SelfReflectionContradiction(BaseModel):
    """A single contradiction identified by the self-reflection tool."""
    span_a: str = Field(description="First text span")
    span_b: str = Field(description="Second text span that contradicts span_a")
    agent_a: str = Field(description="Agent that produced span_a")
    agent_b: str = Field(description="Agent that produced span_b")
    contradiction_description: str
