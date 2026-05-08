"""
eval/runner.py

CLI runner for the evaluation harness.
Can be invoked via:
  python -m eval.runner --all
  python -m eval.runner --category baseline
  python -m eval.runner --category adversarial
  python -m eval.runner --case base-001
  python -m eval.runner --diff

Used by the Makefile eval targets.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys


async def _run_all():
    from eval.harness import run_eval
    from eval.cases.test_cases import ALL_CASES
    print(f"Running full eval harness ({len(ALL_CASES)} cases)...")
    summary = await run_eval()
    print(json.dumps(summary, indent=2, default=str))
    return summary


async def _run_category(category: str):
    from eval.harness import run_eval
    from eval.cases.test_cases import get_cases_by_category
    cases = get_cases_by_category(category)
    print(f"Running {category} eval ({len(cases)} cases)...")
    summary = await run_eval(cases=cases)
    print(json.dumps(summary, indent=2, default=str))
    return summary


async def _run_case(case_id: str):
    from eval.harness import run_single_case
    from eval.cases.test_cases import get_case
    import anthropic
    import instructor
    from api.config import get_settings
    settings = get_settings()
    raw_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
    llm_client = instructor.from_anthropic(raw_client)
    case = get_case(case_id)
    print(f"Running single case: {case_id}")
    ctx, scores = await run_single_case(case, llm_client)
    print(f"\nFinal answer: {(ctx.final_answer or '')[:500]}")
    print(f"\nScores:")
    for dim, result in scores.items():
        if isinstance(result, dict):
            print(f"  {dim}: {result['score']:.3f} — {result['justification'][:100]}")
        else:
            print(f"  weighted_total: {result:.3f}")
    return scores


def main():
    parser = argparse.ArgumentParser(description="Mega AI Evaluation Harness")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true", help="Run all 15 cases")
    group.add_argument("--category", type=str, help="Run one category (baseline/ambiguous/adversarial)")
    group.add_argument("--case", type=str, help="Run a single case by ID")
    group.add_argument("--diff", action="store_true", help="Show diff between last two runs")
    args = parser.parse_args()

    if args.all:
        asyncio.run(_run_all())
    elif args.category:
        asyncio.run(_run_category(args.category))
    elif args.case:
        asyncio.run(_run_case(args.case))
    elif args.diff:
        print("Diff between runs: query GET /eval/latest for summary statistics.")
        sys.exit(0)


if __name__ == "__main__":
    main()
