"""
api/context_manager.py

Context Budget Manager — tracks per-agent token consumption and enforces
budget limits. This is a hard enforcement layer, not a prompt instruction.

Design:
- Every agent MUST call declare() before execution to register its budget.
- Every agent MUST call check_remaining() before assembling large context blocks.
- If add() returns False, the caller is responsible for triggering compression.
- Policy violations are appended to SharedContext.budget_violations and logged.
  They are NEVER silently ignored or truncated.

Token counting uses tiktoken with the cl100k_base encoding (compatible with
both OpenAI and Anthropic models for planning purposes).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import tiktoken

if TYPE_CHECKING:
    from api.models.context import SharedContext

logger = logging.getLogger(__name__)

# cl100k_base is used by GPT-4, Claude (approximate), and embedding models.
# For actual Anthropic token counts, the SDK's count_tokens() can be used,
# but cl100k_base is a fast, good-enough approximation for budget enforcement.
_ENCODER = tiktoken.get_encoding("cl100k_base")

# How many tokens must remain before the budget manager auto-triggers compression.
COMPRESSION_TRIGGER_THRESHOLD = 500

# Minimum tokens required to be considered "safe to proceed" after compression.
MIN_SAFE_REMAINING = 200


@dataclass
class AgentBudgetState:
    """Tracks a single agent's token budget for one job."""
    agent_id: str
    max_tokens: int
    consumed: int = 0
    compression_count: int = 0  # how many times compression was triggered
    violation_count: int = 0

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.consumed)

    @property
    def utilization_pct(self) -> float:
        return (self.consumed / self.max_tokens) * 100 if self.max_tokens > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "agent_id": self.agent_id,
            "max_tokens": self.max_tokens,
            "consumed": self.consumed,
            "remaining": self.remaining,
            "utilization_pct": round(self.utilization_pct, 1),
            "compression_count": self.compression_count,
            "violation_count": self.violation_count,
        }


class BudgetManager:
    """
    Per-job context budget manager.

    Lifecycle:
    1. Orchestrator creates BudgetManager(ctx)
    2. Before each agent runs: bm.declare(agent_id, max_tokens)
    3. Agent assembles context: bm.check_remaining(agent_id) → int
    4. Agent wants to add text: bm.add(agent_id, text) → bool
       - True:  tokens tracked, proceed
       - False: budget exceeded, caller MUST compress before retrying
    5. After job: bm.audit_report() → dict with full per-agent stats
    """

    def __init__(self, ctx: "SharedContext") -> None:
        self._ctx = ctx
        self._states: dict[str, AgentBudgetState] = {}

    # ── Registration ──────────────────────────────────────────────────────────

    def declare(self, agent_id: str, max_tokens: int) -> None:
        """
        Register an agent's budget before execution.
        Must be called once per agent per job, before any add() calls.
        Raises if called twice for the same agent (prevents budget manipulation).
        """
        if agent_id in self._states:
            raise ValueError(
                f"BudgetManager: agent '{agent_id}' already declared. "
                "declare() must be called exactly once per agent per job."
            )
        if max_tokens <= 0:
            raise ValueError(
                f"BudgetManager: max_tokens must be > 0, got {max_tokens} for '{agent_id}'"
            )
        self._states[agent_id] = AgentBudgetState(
            agent_id=agent_id, max_tokens=max_tokens
        )
        logger.info(
            "budget_declared",
            extra={
                "agent_id": agent_id,
                "max_tokens": max_tokens,
                "job_id": str(self._ctx.job_id),
            },
        )

    # ── Token counting ────────────────────────────────────────────────────────

    @staticmethod
    def count_tokens(text: str) -> int:
        """
        Count tokens in text using cl100k_base encoding.
        Returns 0 for empty/None text.
        """
        if not text:
            return 0
        return len(_ENCODER.encode(text))

    # ── Budget checking ───────────────────────────────────────────────────────

    def check_remaining(self, agent_id: str) -> int:
        """
        Return the number of tokens remaining for an agent.
        Any agent can call this at any time to pre-check before assembling context.

        Raises KeyError if the agent was not declared (programming error).
        """
        self._require_declared(agent_id)
        return self._states[agent_id].remaining

    def would_exceed(self, agent_id: str, text: str) -> bool:
        """
        Check whether adding `text` would exceed the agent's budget.
        Does NOT modify state. Safe to call speculatively.
        """
        self._require_declared(agent_id)
        tokens = self.count_tokens(text)
        return tokens > self._states[agent_id].remaining

    def is_near_limit(self, agent_id: str, threshold: int = COMPRESSION_TRIGGER_THRESHOLD) -> bool:
        """
        Return True if the agent has fewer than `threshold` tokens remaining.
        The orchestrator uses this to decide whether to trigger compression
        proactively, before an overflow occurs.
        """
        self._require_declared(agent_id)
        return self._states[agent_id].remaining < threshold

    # ── Budget consumption ────────────────────────────────────────────────────

    def add(self, agent_id: str, text: str) -> bool:
        """
        Attempt to consume tokens for `text` against the agent's budget.

        Returns:
            True  — tokens consumed, agent may proceed
            False — budget would be exceeded; a policy violation is logged
                    and appended to SharedContext.budget_violations.
                    The caller MUST trigger compression before retrying.

        The text is NEVER truncated. This is a hard enforcement policy.
        """
        self._require_declared(agent_id)
        tokens = self.count_tokens(text)
        state = self._states[agent_id]

        if tokens > state.remaining:
            # Hard violation — log and refuse
            violation_msg = (
                f"BUDGET_VIOLATION | agent={agent_id} | "
                f"attempted={tokens} tokens | "
                f"remaining={state.remaining} | "
                f"max={state.max_tokens} | "
                f"job_id={self._ctx.job_id}"
            )
            state.violation_count += 1
            self._ctx.budget_violations.append(violation_msg)
            logger.warning(violation_msg)
            return False

        state.consumed += tokens
        logger.debug(
            "budget_consumed",
            extra={
                "agent_id": agent_id,
                "tokens_added": tokens,
                "remaining": state.remaining,
                "job_id": str(self._ctx.job_id),
            },
        )
        return True

    def force_add(self, agent_id: str, text: str) -> None:
        """
        Add tokens without budget enforcement.
        ONLY for structured/lossless data (tool outputs, scores, citations)
        that must be preserved even if budget is exceeded. Logs a warning.
        Use sparingly — every call is audited.
        """
        self._require_declared(agent_id)
        tokens = self.count_tokens(text)
        state = self._states[agent_id]

        if tokens > state.remaining:
            logger.warning(
                "budget_force_add: structured data exceeds budget",
                extra={
                    "agent_id": agent_id,
                    "tokens": tokens,
                    "remaining": state.remaining,
                    "job_id": str(self._ctx.job_id),
                },
            )
        state.consumed += tokens

    # ── Compression hook ──────────────────────────────────────────────────────

    def record_compression(self, agent_id: str, tokens_saved: int) -> None:
        """
        Called by the orchestrator after running the compression agent.
        Reduces the consumed count by `tokens_saved` and records the event.
        Tokens saved must be non-negative.
        """
        self._require_declared(agent_id)
        if tokens_saved < 0:
            raise ValueError("tokens_saved must be >= 0")

        state = self._states[agent_id]
        state.consumed = max(0, state.consumed - tokens_saved)
        state.compression_count += 1
        logger.info(
            "budget_compression_applied",
            extra={
                "agent_id": agent_id,
                "tokens_saved": tokens_saved,
                "new_consumed": state.consumed,
                "remaining": state.remaining,
                "job_id": str(self._ctx.job_id),
            },
        )

    async def compress_and_record(self, agent_id: str, text: str) -> str:
        """
        Run the compression agent on `text` and update the consumed count.
        Returns the compressed text. Called by orchestrator when is_near_limit().

        Lossless for structured data, lossy for prose (see agents/compression.py).
        """
        from api.agents.compression import compress_context_async  # lazy import to avoid circular

        original_tokens = self.count_tokens(text)
        compressed = await compress_context_async(text)
        compressed_tokens = self.count_tokens(compressed)
        tokens_saved = max(0, original_tokens - compressed_tokens)

        if tokens_saved > 0:
            self.record_compression(agent_id, tokens_saved)

        logger.info(
            "compress_and_record",
            extra={
                "agent_id": agent_id,
                "original_tokens": original_tokens,
                "compressed_tokens": compressed_tokens,
                "tokens_saved": tokens_saved,
                "job_id": str(self._ctx.job_id),
            },
        )
        return compressed

    # ── Reporting ─────────────────────────────────────────────────────────────

    def snapshot(self, agent_id: str) -> dict:
        """Return current budget state for a single agent (for SSE events)."""
        self._require_declared(agent_id)
        return self._states[agent_id].to_dict()

    def audit_report(self) -> dict:
        """
        Full per-agent budget report for end-of-job logging.
        Included in the trace_events payload for the job_complete event.
        """
        total_violations = sum(s.violation_count for s in self._states.values())
        return {
            "agents": {aid: state.to_dict() for aid, state in self._states.items()},
            "total_agents": len(self._states),
            "total_violations": total_violations,
            "job_id": str(self._ctx.job_id),
            "compliance": "PASS" if total_violations == 0 else "FAIL",
        }

    def get_all_states(self) -> dict[str, AgentBudgetState]:
        """Return all agent states (read-only access for orchestrator)."""
        return dict(self._states)

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _require_declared(self, agent_id: str) -> None:
        """Raise KeyError with a helpful message if agent was not declared."""
        if agent_id not in self._states:
            raise KeyError(
                f"BudgetManager: agent '{agent_id}' has not been declared. "
                "Call bm.declare(agent_id, max_tokens) before execution."
            )
