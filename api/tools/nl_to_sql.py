"""
api/tools/nl_to_sql.py

NL → SQL Structured Data Lookup Tool — Tool #3
Converts a natural-language question to a SQL query (via LLM + Instructor),
executes it against the products table in PostgreSQL, and returns the rows.

Schema available: products(id, name, category, price_usd, stock, rating, created_at)

Failure contract (enforced in code, not prompts):
  - malformed → input 'question' is not a non-empty string, OR LLM produces invalid SQL
               that fails to parse/execute due to syntax error
  - timeout   → DB query takes > DB_TIMEOUT_SECONDS
  - empty     → valid SQL executes but returns 0 rows
               (orchestrator can retry with broadened query)
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

import asyncpg
import anthropic
import instructor
from pydantic import BaseModel, Field

from api.config import get_settings
from api.models.tools import ToolResult, SQLQueryResult

DB_TIMEOUT_SECONDS = 5.0
TOOL_NAME = "nl_to_sql"

# ─── Schema description (injected into NL→SQL prompt) ─────────────────────────

_SCHEMA_DESCRIPTION = """
Table: products
Columns:
  id          SERIAL    PRIMARY KEY
  name        TEXT      product name
  category    TEXT      one of: Electronics, Books, Office, Software, Peripherals, Networking, Accessories
  price_usd   NUMERIC   price in US dollars
  stock       INT       units in stock
  rating      NUMERIC   customer rating 0.0 to 5.0
  created_at  DATE      when the product was listed

Write a single SELECT statement. Do not use DML (INSERT/UPDATE/DELETE/DROP).
Return only valid PostgreSQL SQL. No markdown, no explanation.
"""

# ─── Instructor structured output ────────────────────────────────────────────

class GeneratedSQL(BaseModel):
    sql: str = Field(description="A valid PostgreSQL SELECT statement")
    explanation: str = Field(description="One-sentence explanation of what the SQL does")


async def _generate_sql(question: str) -> GeneratedSQL:
    """Call Claude via Instructor to convert natural language to SQL."""
    settings = get_settings()
    raw_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    client = instructor.from_anthropic(raw_client)

    return await client.messages.create(
        model=settings.primary_model,
        max_tokens=300,
        messages=[{
            "role": "user",
            "content": (
                f"Convert this question to a PostgreSQL SELECT query.\n\n"
                f"{_SCHEMA_DESCRIPTION}\n\n"
                f"Question: {question}"
            ),
        }],
        response_model=GeneratedSQL,
    )


async def _execute_sql(sql: str, dsn: str, timeout: float) -> list[dict]:
    """Execute a SQL query and return rows as list of dicts."""
    conn = await asyncio.wait_for(
        asyncpg.connect(dsn), timeout=timeout
    )
    try:
        rows = await asyncio.wait_for(
            conn.fetch(sql), timeout=timeout
        )
        return [dict(row) for row in rows]
    finally:
        await conn.close()


def _build_dsn() -> str:
    s = get_settings()
    return f"postgresql://{s.postgres_user}:{s.postgres_password}@{s.postgres_host}:{s.postgres_port}/{s.postgres_db}"


def _is_safe_sql(sql: str) -> bool:
    """Basic safety check — reject DML and DDL."""
    normalized = sql.strip().upper()
    dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
                 "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE"]
    return all(not normalized.startswith(kw) for kw in dangerous)


# ─── Public tool function ─────────────────────────────────────────────────────

async def nl_to_sql(input_data: dict) -> ToolResult:
    """
    Tool #3 — NL → SQL Structured Lookup

    Input schema:
        question (str, required) — natural language question about the products data

    Failure contract:
        malformed  → question is not a non-empty string, or generated SQL is unsafe/invalid
        timeout    → LLM call or DB query exceeds DB_TIMEOUT_SECONDS
        empty      → SQL executes successfully but returns 0 rows
    """
    t0 = time.perf_counter()

    # ── Malformed check ───────────────────────────────────────────────────────
    question = input_data.get("question")
    if not isinstance(question, str) or not question.strip():
        return ToolResult.malformed(
            source=TOOL_NAME,
            message="'question' must be a non-empty string",
        )

    # ── LLM: NL → SQL ─────────────────────────────────────────────────────────
    try:
        generated = await asyncio.wait_for(
            _generate_sql(question.strip()),
            timeout=DB_TIMEOUT_SECONDS * 2,  # allow more time for LLM
        )
    except asyncio.TimeoutError:
        latency = (time.perf_counter() - t0) * 1000
        return ToolResult.timeout(
            source=TOOL_NAME,
            latency_ms=latency,
            message="SQL generation timed out",
        )
    except Exception as e:
        return ToolResult.malformed(
            source=TOOL_NAME,
            message=f"Failed to generate SQL: {e}",
        )

    sql = generated.sql.strip()

    # ── Safety check ──────────────────────────────────────────────────────────
    if not _is_safe_sql(sql):
        return ToolResult.malformed(
            source=TOOL_NAME,
            message=f"Generated SQL is not a safe SELECT statement: {sql[:100]}",
        )

    # ── Execute against DB ────────────────────────────────────────────────────
    dsn = _build_dsn()
    try:
        rows = await asyncio.wait_for(
            _execute_sql(sql, dsn, DB_TIMEOUT_SECONDS),
            timeout=DB_TIMEOUT_SECONDS + 1,
        )
    except asyncio.TimeoutError:
        latency = (time.perf_counter() - t0) * 1000
        return ToolResult.timeout(
            source=TOOL_NAME,
            latency_ms=latency,
            message=f"DB query timed out after {DB_TIMEOUT_SECONDS}s",
        )
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        err_msg = str(e)
        # Distinguish syntax error (malformed) from connection/runtime errors (timeout)
        if "syntax" in err_msg.lower() or "parse" in err_msg.lower():
            return ToolResult.malformed(
                source=TOOL_NAME,
                message=f"SQL syntax error: {err_msg[:200]}",
                latency_ms=latency,
            )
        return ToolResult.timeout(
            source=TOOL_NAME,
            latency_ms=latency,
            message=f"DB execution error: {err_msg[:200]}",
        )

    latency = (time.perf_counter() - t0) * 1000

    # ── Empty check ───────────────────────────────────────────────────────────
    if not rows:
        return ToolResult.empty(
            source=TOOL_NAME,
            latency_ms=latency,
            message=f"Query returned 0 rows for: '{question}'",
        )

    columns = list(rows[0].keys()) if rows else []
    result = SQLQueryResult(
        sql=sql,
        rows=rows,
        row_count=len(rows),
        columns=columns,
    )

    return ToolResult.ok(
        data=result.model_dump(),
        source=TOOL_NAME,
        latency_ms=latency,
    )
