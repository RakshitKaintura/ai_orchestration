"""
api/agents/meta/agent.py

Meta Agent — self-improving prompt rewriter. Post-eval, not pipeline.
"""

from __future__ import annotations

import asyncio
import logging
import uuid

import google.generativeai as genai
import instructor

from api.agents.meta.schemas import PromptRewriteProposal
from api.config import get_settings

logger = logging.getLogger(__name__)

# Dimension → responsible agent mapping
DIMENSION_AGENT_MAP = {
    "correctness": "synthesis",
    "citations": "rag",
    "contradictions": "synthesis",
    "tool_efficiency": "orchestrator",
    "budget_compliance": "decomposition",
    "critique_agreement": "critique",
}

AGENT_PROMPT_MAP = {
    "decomposition": "decomposition_agent",
    "rag": "rag_agent",
    "critique": "critique_agent",
    "synthesis": "synthesis_agent",
    "orchestrator": "orchestrator_agent",
}


class MetaAgent:
    """
    Post-eval meta agent. Not a pipeline agent — called by harness/worker.
    Does not inherit BaseAgent (no budget management, no SharedContext).
    """

    async def propose_rewrite(
        self,
        eval_summary: dict,
        case_results: list[dict],
        current_prompts: dict[str, str],
    ) -> PromptRewriteProposal | None:
        settings = get_settings()

        dim_stats = eval_summary.get("summary_by_dimension", {})
        if not dim_stats:
            logger.warning("meta_agent_no_dim_stats")
            return None

        worst_dim = min(dim_stats, key=lambda d: dim_stats[d].get("mean_score", 1.0))
        worst_score = dim_stats[worst_dim].get("mean_score", 1.0)

        if worst_score > 0.85:
            logger.info("meta_agent_no_rewrite_needed", extra={
                "worst_dim": worst_dim, "worst_score": worst_score
            })
            return None

        target_agent = DIMENSION_AGENT_MAP.get(worst_dim, "synthesis")
        current_prompt = current_prompts.get(target_agent, "")

        failed_cases = sorted(
            [c for c in case_results if c.get(worst_dim, 1.0) < 0.6],
            key=lambda c: c.get(worst_dim, 1.0),
        )[:5]

        failed_summary = "\n".join(
            f"- Case {c.get('case_id')}: {worst_dim}={c.get(worst_dim, 0):.2f} | "
            f"query='{c.get('query', '')[:100]}' | "
            f"answer='{c.get('final_answer', '')[:150]}'"
            for c in failed_cases
        )

        prompt = (
            f"You are the meta-agent in a self-improving AI evaluation system.\n\n"
            f"Eval summary:\n"
            f"  Worst dimension: {worst_dim} (mean score: {worst_score:.2f})\n"
            f"  Responsible agent: {target_agent}\n"
            f"  Overall mean score: {eval_summary.get('overall_mean', 0):.2f}\n\n"
            f"Failed cases ({len(failed_cases)} worst):\n{failed_summary or 'No cases below 0.6'}\n\n"
            f"Current system prompt for {target_agent}:\n{current_prompt[:2000] or '[No current prompt]'}\n\n"
            f"Propose a rewritten system prompt that specifically addresses the failures above.\n"
            f"Focus ONLY on the {worst_dim} dimension. Do NOT try to improve everything at once.\n"
            f"Be specific: quote the patterns from the failed cases that the new prompt should fix."
        )

        genai.configure(api_key=settings.google_api_key or settings.gemini_api_key)
        client = instructor.from_gemini(
            client=genai.GenerativeModel(model_name=settings.primary_model),
            mode=instructor.Mode.GEMINI_JSON,
        )

        try:
            proposal: PromptRewriteProposal = await asyncio.to_thread(
                client.chat.completions.create,
                messages=[{"role": "user", "content": prompt}],
                response_model=PromptRewriteProposal,
            )
            logger.info("meta_agent_proposal_generated", extra={
                "target_agent": proposal.target_agent,
                "target_dimension": proposal.target_dimension,
                "confidence": proposal.confidence,
            })
            return proposal
        except Exception as e:
            logger.error("meta_agent_failed", extra={"error": str(e)})
            return None


# ─── DB persistence helpers ───────────────────────────────────────────────────

async def save_rewrite_proposal(conn, run_id: str, proposal: PromptRewriteProposal) -> str:
    """Persist the meta agent's proposal to prompt_rewrites. Returns rewrite_id."""
    rewrite_id = str(uuid.uuid4())
    try:
        await conn.execute(
            """
            INSERT INTO prompt_rewrites
              (id, run_id, agent_id, dimension, analysis,
               prompt_before, prompt_after, diff_summary,
               expected_improvement, confidence, status)
            VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10, 'pending')
            """,
            rewrite_id, run_id, proposal.target_agent, proposal.target_dimension,
            proposal.analysis, "", proposal.rewritten_prompt, proposal.diff_summary,
            proposal.expected_improvement, proposal.confidence,
        )
        logger.info("rewrite_proposal_saved", extra={
            "rewrite_id": rewrite_id, "agent": proposal.target_agent
        })
    except Exception as e:
        logger.error("rewrite_proposal_save_failed", extra={"error": str(e)})
    return rewrite_id


async def apply_approved_rewrite(conn, rewrite_id: str) -> bool:
    """Apply an approved rewrite to agent_prompts. Returns True on success."""
    try:
        row = await conn.fetchrow(
            "SELECT agent_id, prompt_after FROM prompt_rewrites WHERE id = $1 AND status = 'approved'",
            rewrite_id,
        )
        if not row:
            logger.warning("apply_rewrite_not_found_or_not_approved", extra={"rewrite_id": rewrite_id})
            return False

        await conn.execute(
            """
            INSERT INTO agent_prompts (agent_id, version_label, system_prompt, is_active)
            SELECT $1,
                   'v' || (COALESCE(MAX(CAST(REPLACE(version_label, 'v', '') AS INTEGER)), 0) + 1),
                   $2, TRUE
            FROM agent_prompts WHERE agent_id = $1
            """,
            row["agent_id"], row["prompt_after"],
        )
        logger.info("rewrite_applied", extra={"rewrite_id": rewrite_id, "agent_id": row["agent_id"]})
        return True
    except Exception as e:
        logger.error("apply_rewrite_failed", extra={"error": str(e), "rewrite_id": rewrite_id})
        return False


async def fetch_current_prompts(conn) -> dict[str, str]:
    """Fetch active system prompts for all agents. Returns {agent_id: prompt}."""
    try:
        rows = await conn.fetch(
            "SELECT agent_id, system_prompt FROM agent_prompts WHERE is_active = TRUE"
        )
        return {row["agent_id"]: row["system_prompt"] for row in rows}
    except Exception as e:
        logger.warning("fetch_prompts_failed", extra={"error": str(e)})
        return {}
