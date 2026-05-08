"""
eval/scorers.py

Six scoring dimensions for the Mega AI evaluation harness.
All scorers return a float in [0.0, 1.0] and a justification string.

Dimensions:
  1. correctness        — LLM-as-judge comparison of answer vs expected
  2. citations          — provenance map citations reference real corpus chunks
  3. contradictions     — contradictions resolved in synthesis (not surfaced to user)
  4. tool_efficiency    — penalise unnecessary tool calls (-0.1 per extra call)
  5. budget_compliance  — 1.0 if no violations, reduced per violation
  6. critique_agreement — critique agent's overall_confidence for the synthesis output

Each scorer is a pure function:
    score_X(case, ctx, [llm_client]) -> (score: float, justification: str)
"""

from __future__ import annotations

import logging
from typing import Any

from eval.cases.test_cases import EvalCase
from api.models.context import SharedContext

logger = logging.getLogger(__name__)

ScoreResult = tuple[float, str]  # (score, justification)


# ─── 1. Correctness (LLM-as-judge) ───────────────────────────────────────────

async def score_correctness(
    case: EvalCase,
    ctx: SharedContext,
    llm_client: Any,  # instructor-wrapped AsyncAnthropic
) -> ScoreResult:
    """
    LLM-as-judge correctness scorer.

    Compares the pipeline's final_answer against the expected_answer.
    Pays special attention to:
    - False premise detection (adversarial cases)
    - Forbidden claim violations
    - Factual accuracy vs ground truth

    Returns 0.0 if any forbidden_claim appears verbatim in the answer.
    """
    from pydantic import BaseModel, Field

    class CorrectnessScore(BaseModel):
        score: float = Field(ge=0.0, le=1.0)
        justification: str

    final_answer = ctx.final_answer or ""

    # Hard check: forbidden claims → automatic 0.0
    for claim in case.forbidden_claims:
        if claim.lower() in final_answer.lower():
            return (0.0, f"FORBIDDEN_CLAIM detected: '{claim}' appeared in the answer.")

    # Empty answer
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
        result: CorrectnessScore = await llm_client.messages.create(
            model=settings.judge_model,
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
            response_model=CorrectnessScore,
        )
        return (result.score, result.justification)
    except Exception as e:
        logger.error("correctness_scorer_failed", extra={"error": str(e), "case_id": case.case_id})
        return (0.5, f"Scorer failed: {e}. Defaulting to 0.5.")


# ─── 2. Citations ─────────────────────────────────────────────────────────────

async def score_citations(case: EvalCase, ctx: SharedContext, **_) -> ScoreResult:
    """
    Citation accuracy scorer.

    Checks:
    1. Pipeline produced a provenance_map
    2. Each cited chunk_id exists in expected_chunk_ids (or corpus)
    3. Fraction of expected chunks that were actually cited

    Penalises phantom citations (chunk IDs that don't exist in corpus).
    """
    from api.agents.rag import _get_chroma_collection

    provenance = ctx.provenance_map
    if not provenance:
        return (0.0, "No provenance_map produced. Citations scorer requires provenance_map.")

    # Get all valid chunk IDs from the corpus
    collection = _get_chroma_collection()
    valid_ids: set[str] = set()
    if collection:
        try:
            all_ids = collection.get()["ids"]
            valid_ids = set(all_ids)
        except Exception:
            # Fallback: use expected chunk IDs as reference
            valid_ids = set(case.expected_chunk_ids)

    # Collect all cited chunk IDs
    cited_ids = {
        p.source_chunk_id
        for p in provenance
        if p.source_chunk_id and p.source_chunk_id.strip()
    }

    if not cited_ids and not case.expected_chunk_ids:
        return (1.0, "No chunks expected and none cited — no citations required for this case.")

    if not cited_ids:
        return (0.0, f"No chunk IDs cited in provenance. Expected: {case.expected_chunk_ids}")

    # Phantom citation check
    phantom = cited_ids - valid_ids if valid_ids else set()
    phantom_penalty = 0.1 * len(phantom)

    # Coverage of expected chunks
    expected = set(case.expected_chunk_ids)
    if expected:
        coverage = len(cited_ids & expected) / len(expected)
    else:
        coverage = 1.0  # No specific chunks required

    score = max(0.0, min(1.0, coverage - phantom_penalty))
    justification = (
        f"Cited {len(cited_ids)} chunks, expected {len(expected)}. "
        f"Coverage: {coverage:.0%}. Phantom citations: {len(phantom)}. "
        f"Score: {score:.2f}."
    )
    return (score, justification)


# ─── 3. Contradictions ────────────────────────────────────────────────────────

async def score_contradictions(case: EvalCase, ctx: SharedContext, **_) -> ScoreResult:
    """
    Contradiction handling scorer.

    Checks:
    1. Synthesis resolved contradictions (not surfaced them to user)
    2. The final_answer does not contain hedge language like "one source says X, another says Y"
    3. Any contradiction flagged by critique is documented in ctx.contradictions_resolved

    Perfect score (1.0) = contradictions handled internally, not dumped on user.
    """
    final_answer = ctx.final_answer or ""

    # Patterns that indicate unresolved contradictions surfaced to user
    surfaced_patterns = [
        "one source says",
        "another source says",
        "sources disagree",
        "conflicting information",
        "it is unclear which is correct",
        "both sources claim",
        "i cannot determine which",
    ]

    surfaced = [p for p in surfaced_patterns if p.lower() in final_answer.lower()]
    if surfaced:
        return (
            0.2,
            f"Contradictions surfaced to user (should be resolved internally). "
            f"Patterns found: {surfaced}",
        )

    # Check if critique flagged contradictions that synthesis handled
    synthesis_output = ctx.agent_outputs.get("synthesis")
    if synthesis_output:
        resolved_count = len(ctx.contradictions_resolved)
        claim_scores = synthesis_output.claim_scores or []
        flagged_count = sum(1 for cs in claim_scores if cs.flagged)

        if flagged_count > 0 and resolved_count == 0:
            return (
                0.5,
                f"Critique flagged {flagged_count} spans but synthesis resolved 0. "
                "May have missed contradiction resolution.",
            )

        if resolved_count > 0:
            return (
                1.0,
                f"Synthesis resolved {resolved_count} contradictions internally. "
                "None surfaced to user.",
            )

    return (0.9, "No contradictions detected in final answer. Synthesis handled cleanly.")


# ─── 4. Tool Efficiency ───────────────────────────────────────────────────────

async def score_tool_efficiency(case: EvalCase, ctx: SharedContext, **_) -> ScoreResult:
    """
    Tool efficiency scorer.

    Computes:
    - Required tools that were invoked (required_recall)
    - Unnecessary tool calls (penalty -0.1 per extra unique tool not in required_tools)
    - Retry penalty (-0.05 per retry)

    Score starts at 1.0, decreases for extra calls.
    Cases that require no tools: score 1.0 unless tools were called unnecessarily.
    """
    tool_calls = ctx.tool_call_log
    required = set(case.requires_tools)

    # Tools actually invoked (unique tool names, first attempt only)
    invoked_tools = {tc.tool for tc in tool_calls if tc.retry_num == 0}

    # Required tool recall
    if required:
        missing = required - invoked_tools
        recall = len(required & invoked_tools) / len(required)
        if missing:
            return (
                recall * 0.8,
                f"Required tools not invoked: {missing}. "
                f"Recall: {recall:.0%}.",
            )
    else:
        recall = 1.0

    # Extra (unnecessary) unique tools
    extra_tools = invoked_tools - required
    extra_penalty = 0.1 * len(extra_tools)

    # Retry penalty
    retries = sum(1 for tc in tool_calls if tc.retry_num > 0)
    retry_penalty = 0.05 * retries

    score = max(0.0, 1.0 - extra_penalty - retry_penalty)
    justification = (
        f"Required: {required or 'none'} | Invoked: {invoked_tools}. "
        f"Extra tools: {extra_tools} (penalty: -{extra_penalty:.2f}). "
        f"Retries: {retries} (penalty: -{retry_penalty:.2f}). "
        f"Score: {score:.2f}."
    )
    return (score, justification)


# ─── 5. Budget Compliance ─────────────────────────────────────────────────────

async def score_budget_compliance(case: EvalCase, ctx: SharedContext, **_) -> ScoreResult:
    """
    Budget compliance scorer.

    1.0 if no budget violations occurred.
    Reduced by 0.15 per violation, min 0.0.
    """
    violations = ctx.budget_violations
    if not violations:
        return (1.0, "No budget violations. All agents stayed within token budget.")

    score = max(0.0, 1.0 - 0.15 * len(violations))
    justification = (
        f"{len(violations)} budget violation(s) recorded. "
        f"Score: {score:.2f}. "
        f"First violation: {violations[0][:100]}"
    )
    return (score, justification)


# ─── 6. Critique Agreement ────────────────────────────────────────────────────

async def score_critique_agreement(case: EvalCase, ctx: SharedContext, **_) -> ScoreResult:
    """
    Critique agreement scorer.

    Uses the critique agent's overall_confidence for the synthesis output.
    If critique was not run on synthesis, falls back to the RAG critique.
    If no critique at all, returns 0.5 (unknown).
    """
    # Try synthesis critique first
    critique_output = (
        ctx.agent_outputs.get("critique_synthesis")
        or ctx.agent_outputs.get("critique_rag")
        or ctx.agent_outputs.get("critique_decomposition")
    )

    if critique_output and critique_output.structured_output:
        conf = critique_output.structured_output.get("overall_confidence", 0.5)
        has_critical = critique_output.structured_output.get("has_critical_errors", False)

        if has_critical:
            score = min(conf, 0.5)  # cap at 0.5 for critical errors
            justification = (
                f"Critique flagged CRITICAL ERRORS. "
                f"Overall confidence: {conf:.2f}. Capped score: {score:.2f}."
            )
        else:
            score = conf
            justification = (
                f"Critique overall confidence: {conf:.2f}. "
                f"No critical errors flagged."
            )
        return (score, justification)

    return (0.5, "No critique output found. Cannot assess critique agreement. Defaulting to 0.5.")


# ─── Aggregate scorer ─────────────────────────────────────────────────────────

SCORER_NAMES = [
    "correctness",
    "citations",
    "contradictions",
    "tool_efficiency",
    "budget_compliance",
    "critique_agreement",
]

SCORER_WEIGHTS = {
    "correctness": 0.35,
    "citations": 0.15,
    "contradictions": 0.15,
    "tool_efficiency": 0.10,
    "budget_compliance": 0.10,
    "critique_agreement": 0.15,
}


async def score_all(
    case: EvalCase,
    ctx: SharedContext,
    llm_client: Any,
) -> dict[str, dict]:
    """
    Run all 6 scorers and return per-dimension results + weighted aggregate.

    Returns:
        {
            "correctness": {"score": float, "justification": str},
            "citations": {...},
            ...
            "weighted_total": float,
        }
    """
    import asyncio

    results = {}

    # Run all scorers (correctness needs LLM, others don't)
    scores_raw = await asyncio.gather(
        score_correctness(case, ctx, llm_client),
        score_citations(case, ctx),
        score_contradictions(case, ctx),
        score_tool_efficiency(case, ctx),
        score_budget_compliance(case, ctx),
        score_critique_agreement(case, ctx),
        return_exceptions=True,
    )

    for name, raw in zip(SCORER_NAMES, scores_raw):
        if isinstance(raw, Exception):
            logger.error(f"scorer_{name}_failed", extra={"error": str(raw)})
            results[name] = {"score": 0.5, "justification": f"Scorer error: {raw}"}
        else:
            score, justification = raw
            results[name] = {"score": round(score, 4), "justification": justification}

    # Weighted total
    weighted_total = sum(
        results[name]["score"] * SCORER_WEIGHTS[name]
        for name in SCORER_NAMES
    )
    results["weighted_total"] = round(weighted_total, 4)

    return results
