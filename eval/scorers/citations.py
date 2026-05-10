"""eval/scorers/citations.py — Citation accuracy scorer."""

from __future__ import annotations

import logging

from eval.cases import EvalCase
from eval.scorers.base import ScoreResult
from api.models.context import SharedContext

logger = logging.getLogger(__name__)


async def score_citations(case: EvalCase, ctx: SharedContext, **_) -> ScoreResult:
    """
    Citation accuracy scorer.

    Checks:
    1. Pipeline produced a provenance_map
    2. Each cited chunk_id exists in expected_chunk_ids (or corpus)
    3. Fraction of expected chunks that were actually cited

    Penalises phantom citations (chunk IDs that don't exist in corpus).
    """
    from api.agents.rag.retriever import _get_chroma_collection

    provenance = ctx.provenance_map
    if not provenance:
        if not case.expected_chunk_ids:
            return (1.0, "No provenance required and none produced.")
        return (0.0, "No provenance_map produced. Citations scorer requires provenance_map.")

    collection = _get_chroma_collection()
    valid_ids: set[str] = set()
    if collection:
        try:
            all_ids = collection.get()["ids"]
            valid_ids = set(all_ids)
        except Exception:
            valid_ids = set(case.expected_chunk_ids)

    cited_ids = {
        p.source_chunk_id
        for p in provenance
        if p.source_chunk_id and p.source_chunk_id.strip()
    }

    if not cited_ids and not case.expected_chunk_ids:
        return (1.0, "No chunks expected and none cited — no citations required for this case.")

    if not cited_ids:
        return (0.0, f"No chunk IDs cited in provenance. Expected: {case.expected_chunk_ids}")

    phantom = cited_ids - valid_ids if valid_ids else set()
    phantom_penalty = 0.1 * len(phantom)

    expected = set(case.expected_chunk_ids)
    coverage = len(cited_ids & expected) / len(expected) if expected else 1.0

    score = max(0.0, min(1.0, coverage - phantom_penalty))
    justification = (
        f"Cited {len(cited_ids)} chunks, expected {len(expected)}. "
        f"Coverage: {coverage:.0%}. Phantom citations: {len(phantom)}. "
        f"Score: {score:.2f}."
    )
    return (score, justification)
