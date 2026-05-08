"""
api/agents/meta.py

Meta Agent — Self-Improving Prompt Rewriter
--------------------------------------------
Reads failed eval cases, identifies the worst-performing dimension, and
proposes a rewritten system prompt for the responsible agent.

Called AFTER each eval run by the harness (not during pipeline execution).

Design constraints:
  - Proposes ONE rewrite per eval run (not one per failed case)
  - Rewrites are NEVER auto-applied — stored as 'pending' in prompt_rewrites
  - The proposal includes a structured diff (before/after)
  - Human reviews via POST /rewrites/{id}/review
  - On approval, a targeted re-eval runs on the previously failed cases
  - Score delta (before vs after) is stored in prompt_rewrites.delta

Structured output (Instructor):
  - target_agent:    which agent's prompt to rewrite
  - target_dimension: which eval dimension is worst
  - analysis:        why this agent/dimension is underperforming
  - rewritten_prompt: the new system prompt
  - diff_summary:    structured description of what changed and why
"""

from __future__ import annotations

import json
import logging
import uuid
from typing import Any

import anthropic
import instructor
from pydantic import BaseModel, Field

from api.config import get_settings

logger = logging.getLogger(__name__)

# Agent → their current system prompt header (agent_id → prompt_table_agent_id)
AGENT_PROMPT_MAP = {
    "decomposition": "decomposition_agent",
    "rag": "rag_agent",
    "critique": "critique_agent",
    "synthesis": "synthesis_agent",
    "orchestrator": "orchestrator_agent",
}

# Which dimension's failures are attributable to which agent
DIMENSION_AGENT_MAP = {
    "correctness": "synthesis",
    "citations": "rag",
    "contradictions": "synthesis",
    "tool_efficiency": "orchestrator",
    "budget_compliance": "decomposition",
    "critique_agreement": "critique",
}


# ─── Instructor structured output ────────────────────────────────────────────

class PromptRewriteProposal(BaseModel):
    target_agent: str = Field(
        description="Agent ID whose prompt should be rewritten (decomposition/rag/critique/synthesis/orchestrator)"
    )
    target_dimension: str = Field(
        description="The eval dimension this rewrite aims to improve"
    )
    analysis: str = Field(
        description=(
            "3-5 sentence analysis of WHY this agent/dimension is underperforming "
            "based on the failed cases and scores provided"
        )
    )
    rewritten_prompt: str = Field(
        description="The complete new system prompt for the target agent"
    )
    diff_summary: str = Field(
        description="Concise description of what changed between the current and new prompt, and why"
    )
    expected_improvement: str = Field(
        description="Specific, testable prediction of how this rewrite will improve the failing cases"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="How confident you are this rewrite will improve the target dimension",
    )


# ─── Meta Agent ───────────────────────────────────────────────────────────────

class MetaAgent:
    """
    Post-eval meta agent. Not a pipeline agent — called by the harness/worker.
    Does not inherit BaseAgent (no budget management, no SharedContext).
    """

    async def propose_rewrite(
        self,
        eval_summary: dict,
        case_results: list[dict],
        current_prompts: dict[str, str],
    ) -> PromptRewriteProposal | None:
        """
        Analyse eval results and propose a single prompt rewrite.

        Args:
            eval_summary:   The summary dict from harness.run_eval()
            case_results:   Per-case scores from eval_case_results table
            current_prompts: {agent_id: current_system_prompt}

        Returns:
            PromptRewriteProposal or None if no improvement is needed (all scores > 0.85)
        """
        settings = get_settings()

        # Find the worst-performing dimension
        dim_stats = eval_summary.get("summary_by_dimension", {})
        if not dim_stats:
            logger.warning("meta_agent_no_dim_stats")
            return None

        # Skip if all dimensions are performing well
        worst_dim = min(dim_stats, key=lambda d: dim_stats[d].get("mean_score", 1.0))
        worst_score = dim_stats[worst_dim].get("mean_score", 1.0)

        if worst_score > 0.85:
            logger.info("meta_agent_no_rewrite_needed", extra={
                "worst_dim": worst_dim, "worst_score": worst_score
            })
            return None

        # Identify which agent is responsible for the worst dimension
        target_agent = DIMENSION_AGENT_MAP.get(worst_dim, "synthesis")
        current_prompt = current_prompts.get(target_agent, "")

        # Find the worst-performing cases for this dimension
        failed_cases = sorted(
            [c for c in case_results if c.get(worst_dim, 1.0) < 0.6],
            key=lambda c: c.get(worst_dim, 1.0),
        )[:5]  # top 5 worst cases

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

        raw_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        client = instructor.from_anthropic(raw_client)

        try:
            proposal: PromptRewriteProposal = await client.messages.create(
                model=settings.primary_model,
                max_tokens=2000,
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


# ─── DB persistence ───────────────────────────────────────────────────────────

async def save_rewrite_proposal(
    conn,
    run_id: str,
    proposal: PromptRewriteProposal,
) -> str:
    """
    Persist the meta agent's proposal to prompt_rewrites table.
    Returns the rewrite_id (UUID).
    """
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
            rewrite_id,
            run_id,
            proposal.target_agent,
            proposal.target_dimension,
            proposal.analysis,
            "",  # prompt_before filled from agent_prompts table separately
            proposal.rewritten_prompt,
            proposal.diff_summary,
            proposal.expected_improvement,
            proposal.confidence,
        )
        logger.info("rewrite_proposal_saved", extra={
            "rewrite_id": rewrite_id, "agent": proposal.target_agent
        })
    except Exception as e:
        logger.error("rewrite_proposal_save_failed", extra={"error": str(e)})

    return rewrite_id


async def apply_approved_rewrite(conn, rewrite_id: str) -> bool:
    """
    Apply an approved rewrite to the agent_prompts table.
    Sets the old prompt inactive (trigger handles it) and inserts new.
    Returns True on success.
    """
    try:
        # Fetch the approved rewrite
        row = await conn.fetchrow(
            "SELECT agent_id, prompt_after FROM prompt_rewrites WHERE id = $1 AND status = 'approved'",
            rewrite_id,
        )
        if not row:
            logger.warning("apply_rewrite_not_found_or_not_approved", extra={"rewrite_id": rewrite_id})
            return False

        agent_id = row["agent_id"]
        new_prompt = row["prompt_after"]

        # Insert new prompt — the deactivate_old_prompts trigger handles the rest
        await conn.execute(
            """
            INSERT INTO agent_prompts (agent_id, version_label, system_prompt, is_active)
            SELECT $1,
                   'v' || (COALESCE(MAX(CAST(REPLACE(version_label, 'v', '') AS INTEGER)), 0) + 1),
                   $2, TRUE
            FROM agent_prompts WHERE agent_id = $1
            """,
            agent_id, new_prompt,
        )
        logger.info("rewrite_applied", extra={"rewrite_id": rewrite_id, "agent_id": agent_id})
        return True

    except Exception as e:
        logger.error("apply_rewrite_failed", extra={"error": str(e), "rewrite_id": rewrite_id})
        return False


async def fetch_current_prompts(conn) -> dict[str, str]:
    """
    Fetch the currently active system prompts for all agents.
    Returns {agent_id: system_prompt}.
    """
    try:
        rows = await conn.fetch(
            "SELECT agent_id, system_prompt FROM agent_prompts WHERE is_active = TRUE"
        )
        return {row["agent_id"]: row["system_prompt"] for row in rows}
    except Exception as e:
        logger.warning("fetch_prompts_failed", extra={"error": str(e)})
        return {}
