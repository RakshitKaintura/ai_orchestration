"""eval/scorers/contradictions.py — Contradiction handling scorer."""

from __future__ import annotations

import logging

from eval.cases import EvalCase
from eval.scorers.base import ScoreResult
from api.models.context import SharedContext

logger = logging.getLogger(__name__)

_SURFACED_PATTERNS = [
    "one source says", "another source says", "sources disagree",
    "conflicting information", "it is unclear which is correct",
    "both sources claim", "i cannot determine which",
]


async def score_contradictions(case: EvalCase, ctx: SharedContext, **_) -> ScoreResult:
    """
    Contradiction handling scorer.

    Perfect score (1.0) = contradictions handled internally, not dumped on user.
    Penalises language that surfaces conflicts directly to the user.
    """
    final_answer = ctx.final_answer or ""

    surfaced = [p for p in _SURFACED_PATTERNS if p.lower() in final_answer.lower()]
    if surfaced:
        return (
            0.2,
            f"Contradictions surfaced to user (should be resolved internally). "
            f"Patterns found: {surfaced}",
        )

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
