"""eval/scorers/correctness.py — LLM-as-judge correctness scorer."""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from eval.cases import EvalCase
from eval.scorers.base import ScoreResult
from api.models.context import SharedContext

logger = logging.getLogger(__name__)


async def score_correctness(case: EvalCase, ctx: SharedContext, llm_client: Any) -> ScoreResult:
    """
    LLM-as-judge correctness scorer.

    Compares the pipeline's final_answer against the expected_answer.
    Returns 0.0 if any forbidden_claim appears verbatim in the answer.
    """
    from pydantic import BaseModel, Field

    class CorrectnessScore(BaseModel):
        score: float = Field(ge=0.0, le=1.0)
        justification: str

    final_answer = ctx.final_answer or ""

    for claim in case.forbidden_claims:
        if claim.lower() in final_answer.lower():
            return (0.0, f"FORBIDDEN_CLAIM detected: '{claim}' appeared in the answer.")

    if not final_answer.strip():
        return (0.0, "Pipeline produced no final answer.")

    prompt = (
        f"You are evaluating an AI system's answer against a reference answer.\n\n"
        f"Question: {case.query}\n\n"
        f"Reference answer: {case.expected_answer}\n\n"
        f"System answer: {final_answer}\n\n"
        f"Rate how correct the system answer is on a scale from 0.0 to 1.0.\n"
        f"1.0 = completely correct and complete\n"
        f"0.8 = mostly correct, minor omissions\n"
        f"0.5 = partially correct, key information missing or wrong\n"
        f"0.2 = mostly wrong\n"
        f"0.0 = completely wrong or accepted a false premise without correction\n\n"
        f"Note: If the question contains a false premise, a correct answer MUST identify "
        f"and correct it. An answer that accepts the false premise scores 0.0.\n\n"
        f"Provide a score and a 1-2 sentence justification."
    )

    try:
        from api.config import get_settings
        settings = get_settings()
        result: CorrectnessScore = await asyncio.to_thread(
            llm_client.chat.completions.create,
            messages=[{"role": "user", "content": prompt}],
            response_model=CorrectnessScore,
        )
        return (result.score, result.justification)
    except Exception as e:
        logger.error("correctness_scorer_failed", extra={"error": str(e), "case_id": case.case_id})
        return (0.5, f"Scorer failed: {e}. Defaulting to 0.5.")
