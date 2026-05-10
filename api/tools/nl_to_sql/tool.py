"""
api/tools/nl_to_sql/tool.py

NL → SQL Structured Data Lookup Tool — Tool #3
"""

from __future__ import annotations

import asyncio
import time

import asyncpg
import google.generativeai as genai
import instructor
from pydantic import BaseModel, Field

from api.config import get_settings
from api.models.tools import SQLQueryResult, ToolResult

DB_TIMEOUT_SECONDS = 5.0
TOOL_NAME = "nl_to_sql"

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


class GeneratedSQL(BaseModel):
    sql: str = Field(description="A valid PostgreSQL SELECT statement")
    explanation: str = Field(description="One-sentence explanation of what the SQL does")


async def _generate_sql(question: str) -> GeneratedSQL:
    settings = get_settings()
    genai.configure(api_key=settings.google_api_key or settings.gemini_api_key)
    client = instructor.from_gemini(
        client=genai.GenerativeModel(model_name=settings.primary_model),
        mode=instructor.Mode.GEMINI_JSON,
    )
    return await asyncio.to_thread(
        client.chat.completions.create,
        messages=[{"role": "user", "content": (
            f"Convert this question to a PostgreSQL SELECT query.\n\n"
            f"{_SCHEMA_DESCRIPTION}\n\nQuestion: {question}"
        )}],
        response_model=GeneratedSQL,
    )


async def _execute_sql(sql: str, dsn: str, timeout: float) -> list[dict]:
    conn = await asyncio.wait_for(asyncpg.connect(dsn), timeout=timeout)
    try:
        rows = await asyncio.wait_for(conn.fetch(sql), timeout=timeout)
        return [dict(row) for row in rows]
    finally:
        await conn.close()


def _build_dsn() -> str:
    s = get_settings()
    return f"postgresql://{s.postgres_user}:{s.postgres_password}@{s.postgres_host}:{s.postgres_port}/{s.postgres_db}"


def _is_safe_sql(sql: str) -> bool:
    normalized = sql.strip().upper()
    dangerous = ["INSERT", "UPDATE", "DELETE", "DROP", "CREATE", "ALTER",
                 "TRUNCATE", "GRANT", "REVOKE", "EXEC", "EXECUTE"]
    return all(not normalized.startswith(kw) for kw in dangerous)


async def nl_to_sql(input_data: dict) -> ToolResult:
    """Tool #3 — NL → SQL Structured Lookup"""
    t0 = time.perf_counter()

    question = input_data.get("question")
    if not isinstance(question, str) or not question.strip():
        return ToolResult.malformed(source=TOOL_NAME, message="'question' must be a non-empty string")

    try:
        generated = await asyncio.wait_for(
            _generate_sql(question.strip()), timeout=DB_TIMEOUT_SECONDS * 2
        )
    except asyncio.TimeoutError:
        return ToolResult.timeout(source=TOOL_NAME, latency_ms=(time.perf_counter() - t0) * 1000,
                                  message="SQL generation timed out")
    except Exception as e:
        return ToolResult.malformed(source=TOOL_NAME, message=f"Failed to generate SQL: {e}")

    sql = generated.sql.strip()
    if not _is_safe_sql(sql):
        return ToolResult.malformed(source=TOOL_NAME,
                                    message=f"Generated SQL is not a safe SELECT statement: {sql[:100]}")

    dsn = _build_dsn()
    try:
        rows = await asyncio.wait_for(
            _execute_sql(sql, dsn, DB_TIMEOUT_SECONDS), timeout=DB_TIMEOUT_SECONDS + 1
        )
    except asyncio.TimeoutError:
        return ToolResult.timeout(source=TOOL_NAME, latency_ms=(time.perf_counter() - t0) * 1000,
                                  message=f"DB query timed out after {DB_TIMEOUT_SECONDS}s")
    except Exception as e:
        latency = (time.perf_counter() - t0) * 1000
        err_msg = str(e)
        if "syntax" in err_msg.lower() or "parse" in err_msg.lower():
            return ToolResult.malformed(source=TOOL_NAME, message=f"SQL syntax error: {err_msg[:200]}",
                                        latency_ms=latency)
        return ToolResult.timeout(source=TOOL_NAME, latency_ms=latency,
                                  message=f"DB execution error: {err_msg[:200]}")

    latency = (time.perf_counter() - t0) * 1000
    if not rows:
        return ToolResult.empty(source=TOOL_NAME, latency_ms=latency,
                                message=f"Query returned 0 rows for: '{question}'")

    result = SQLQueryResult(sql=sql, rows=rows, row_count=len(rows),
                            columns=list(rows[0].keys()) if rows else [])
    return ToolResult.ok(data=result.model_dump(), source=TOOL_NAME, latency_ms=latency)
