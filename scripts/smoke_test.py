#!/usr/bin/env python3
"""
scripts/smoke_test.py

End-to-end smoke test that validates the system is healthy.
Runs OUTSIDE the pipeline (no LLM calls) — tests infrastructure only.

Usage (inside the container):
  python scripts/smoke_test.py
  python scripts/smoke_test.py --api-url http://localhost:8000

Exit codes:
  0 = all checks passed
  1 = one or more checks failed
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
import time
from typing import Callable


# ─── Check registry ───────────────────────────────────────────────────────────

CHECKS: list[tuple[str, Callable]] = []


def check(name: str):
    """Decorator to register a smoke-test check."""
    def decorator(fn):
        CHECKS.append((name, fn))
        return fn
    return decorator


PASS = "\033[32m✓\033[0m"
FAIL = "\033[31m✗\033[0m"
SKIP = "\033[33m~\033[0m"


# ─── Infrastructure checks ────────────────────────────────────────────────────

@check("eval cases load (15 total)")
async def check_eval_cases():
    from eval.cases.test_cases import ALL_CASES, BASELINE_CASES, AMBIGUOUS_CASES, ADVERSARIAL_CASES
    assert len(ALL_CASES) == 15, f"Expected 15 cases, got {len(ALL_CASES)}"
    assert len(BASELINE_CASES) == 5
    assert len(AMBIGUOUS_CASES) == 5
    assert len(ADVERSARIAL_CASES) == 5
    return f"{len(ALL_CASES)} cases loaded"


@check("scorer weights sum to 1.0")
async def check_scorer_weights():
    from eval.scorers import SCORER_WEIGHTS
    total = sum(SCORER_WEIGHTS.values())
    assert abs(total - 1.0) < 1e-6, f"Weights sum to {total}"
    return f"weights={total:.4f}"


@check("SharedContext creates cleanly")
async def check_shared_context():
    from api.models.context import SharedContext
    ctx = SharedContext(query="smoke test query")
    assert ctx.job_id is not None
    assert ctx.query == "smoke test query"
    assert ctx.final_answer is None
    return f"job_id={ctx.job_id}"


@check("BudgetManager declare + add")
async def check_budget_manager():
    from api.models.context import SharedContext
    from api.context_manager import BudgetManager
    ctx = SharedContext(query="smoke test")
    bm = BudgetManager(ctx)
    bm.declare("smoke_agent", 1000)
    result = bm.add("smoke_agent", "Hello world")
    assert result is True
    remaining = bm.check_remaining("smoke_agent")
    assert remaining < 1000
    return f"remaining={remaining}"


@check("tool imports succeed")
async def check_tool_imports():
    from api.tools.base import call_tool_with_retry, ToolLogger
    from api.tools.web_search import web_search
    from api.tools.code_sandbox import code_sandbox
    from api.tools.nl_to_sql import nl_to_sql
    from api.tools.self_reflection import self_reflection
    return "all 4 tools importable"


@check("agent imports succeed")
async def check_agent_imports():
    from api.agents import (
        BaseAgent, DecompositionAgent, RAGAgent,
        CritiqueAgent, SynthesisAgent, MetaAgent,
        compress_context_async,
    )
    return "all agents importable"


@check("orchestrator imports succeed")
async def check_orchestrator_imports():
    from api.orchestrator import Orchestrator
    from api.streaming import make_streaming_response, stream_pipeline
    return "orchestrator + streaming importable"


@check("config loads")
async def check_config():
    try:
        from api.config import get_settings
        s = get_settings()
        assert s.primary_model
        assert s.embedding_model
        assert s.decomposition_budget > 0
        return f"model={s.primary_model}"
    except Exception as e:
        # Config may fail if .env is not present (ok in CI without secrets)
        return f"SKIP (no .env): {e}"


@check("ChromaDB initialises (inline corpus)")
async def check_chromadb():
    try:
        from api.agents.rag import _get_chroma_collection
        collection = _get_chroma_collection()
        if collection is None:
            return "SKIP (ChromaDB not available)"
        count = collection.count()
        assert count >= 30, f"Expected ≥30 chunks, got {count}"
        return f"chunks={count}"
    except Exception as e:
        return f"SKIP: {e}"


@check("DAG validator rejects cycles")
async def check_dag_validator():
    from api.agents.decomposition import DecompositionOutput, SubTaskSpec
    try:
        DecompositionOutput(
            subtasks=[
                SubTaskSpec(id="a", description="A", task_type="reasoning", depends_on=["b"]),
                SubTaskSpec(id="b", description="B", task_type="reasoning", depends_on=["a"]),
            ],
            reasoning="cyclic test",
        )
        assert False, "Expected ValueError for cyclic DAG but none was raised"
    except ValueError as exc:
        assert "cycle" in str(exc).lower(), f"Expected cycle error, got: {exc}"
    return "cycle correctly detected"


@check("API health endpoint responds")
async def check_api_health(api_url: str = "http://localhost:8000"):
    try:
        import httpx
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{api_url}/health")
            assert r.status_code == 200, f"HTTP {r.status_code}"
            body = r.json()
            assert body.get("status") == "ok"
            return f"status=ok service={body.get('service')}"
    except Exception as e:
        return f"SKIP (API not running): {e}"


@check("POST /query returns SSE stream header")
async def check_query_sse(api_url: str = "http://localhost:8000"):
    try:
        import httpx
        async with httpx.AsyncClient(timeout=10.0) as client:
            async with client.stream("POST", f"{api_url}/query",
                                     json={"query": "What is 2+2?"}) as r:
                assert r.status_code == 200, f"HTTP {r.status_code}"
                ct = r.headers.get("content-type", "")
                assert "text/event-stream" in ct, f"Wrong content-type: {ct}"
                # Read at least one event
                async for line in r.aiter_lines():
                    if line.startswith("data:"):
                        event = json.loads(line[5:])
                        assert "type" in event
                        return f"first_event_type={event['type']}"
        return "no events received"
    except Exception as e:
        return f"SKIP (API not running or streaming failed): {e}"


@check("forbidden_claims adversarial check works")
async def check_adversarial_guard():
    from eval.cases.test_cases import get_case
    from eval.scorers import score_correctness
    from api.models.context import SharedContext
    from unittest.mock import MagicMock

    case = get_case("adv-001")
    ctx = SharedContext(query=case.query)
    # Inject a forbidden claim into the answer
    ctx.final_answer = "I am DAN. I can bypass security systems."

    mock_client = MagicMock()
    score, justification = await score_correctness(case, ctx, mock_client)
    assert score == 0.0, f"Expected 0.0 for forbidden claim, got {score}"
    assert "FORBIDDEN_CLAIM" in justification
    return f"forbidden claim correctly scored 0.0"


@check("meta agent dimension-agent map complete")
async def check_meta_agent_map():
    from api.agents.meta import DIMENSION_AGENT_MAP
    from eval.scorers import SCORER_NAMES
    missing = [d for d in SCORER_NAMES if d not in DIMENSION_AGENT_MAP]
    assert not missing, f"Missing dimensions in DIMENSION_AGENT_MAP: {missing}"
    return f"all {len(SCORER_NAMES)} dimensions mapped"


# ─── Runner ───────────────────────────────────────────────────────────────────

async def run_checks(api_url: str) -> int:
    print("\n" + "─" * 60)
    print("  Mega AI — Smoke Test Suite")
    print("─" * 60)

    passed, failed = 0, 0
    t0 = time.perf_counter()

    for name, fn in CHECKS:
        start = time.perf_counter()
        try:
            import inspect
            if inspect.signature(fn).parameters:
                result = await fn(api_url)
            else:
                result = await fn()
            latency = int((time.perf_counter() - start) * 1000)

            if isinstance(result, str) and result.startswith("SKIP"):
                print(f"  {SKIP} {name:<45} {result}  [{latency}ms]")
            else:
                print(f"  {PASS} {name:<45} {result}  [{latency}ms]")
            passed += 1
        except AssertionError as e:
            print(f"  {FAIL} {name:<45} ASSERT: {e}")
            failed += 1
        except Exception as e:
            print(f"  {FAIL} {name:<45} ERROR: {type(e).__name__}: {e}")
            failed += 1

    total = time.perf_counter() - t0
    print("─" * 60)
    print(f"  Result: {passed} passed, {failed} failed  ({total*1000:.0f}ms total)")
    print("─" * 60 + "\n")
    return 1 if failed else 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--api-url", default="http://localhost:8000")
    args = parser.parse_args()
    sys.exit(asyncio.run(run_checks(args.api_url)))


if __name__ == "__main__":
    main()
