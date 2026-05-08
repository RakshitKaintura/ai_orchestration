"""
api/models/context.py

The SharedContext is the single object that flows between ALL agents in the
Mega AI pipeline. Agents never call each other directly — the orchestrator
mediates all handoffs by reading from and writing to this object.

Design constraints:
- Immutable once written: agents append to lists, they do not overwrite.
- All writes go through the orchestrator or BudgetManager.
- output_hash() provides a stable fingerprint for logging/deduplication.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, Literal
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


# ─── Sub-task (produced by Decomposition Agent) ───────────────────────────────

class SubTask(BaseModel):
    """
    A single atomic unit of work produced by the decomposition agent.
    Sub-tasks with depends_on entries must NOT execute until all their
    dependency sub-tasks have status == "done".
    """
    id: str = Field(
        default_factory=lambda: str(uuid4())[:8],
        description="Short unique ID within this job (e.g. 'st-a1b2')",
    )
    description: str = Field(
        description="Human-readable description of what this sub-task should do",
    )
    task_type: Literal["retrieval", "computation", "reasoning", "synthesis", "other"] = Field(
        default="reasoning",
        description="Typed category of this sub-task",
    )
    depends_on: list[str] = Field(
        default_factory=list,
        description="List of sub-task IDs that must complete before this one starts",
    )
    status: Literal["pending", "running", "done", "failed"] = Field(
        default="pending",
        description="Lifecycle status of this sub-task",
    )
    result: Any = Field(
        default=None,
        description="Output of this sub-task once completed",
    )
    assigned_agent: str | None = Field(
        default=None,
        description="Which agent the orchestrator assigned to this sub-task",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None


# ─── Tool call record ─────────────────────────────────────────────────────────

class ToolCall(BaseModel):
    """
    A record of a single tool invocation, including retry information.
    Every tool call — successful or not — is logged here.
    """
    id: str = Field(default_factory=lambda: str(uuid4())[:8])
    tool: str = Field(description="Name of the tool invoked (e.g. 'web_search')")
    input: dict[str, Any] = Field(description="Exact input passed to the tool")
    output: Any = Field(default=None, description="Raw output from the tool")
    latency_ms: int = Field(default=0, description="Wall-clock latency in milliseconds")
    success: bool = Field(default=False)
    error_type: str | None = Field(
        default=None,
        description="Failure category: 'timeout' | 'empty' | 'malformed' | None",
    )
    accepted: bool = Field(
        default=False,
        description="Whether the calling agent accepted and used this result",
    )
    rejection_reason: str | None = Field(
        default=None,
        description="If accepted=False, why the agent rejected the result",
    )
    retry_num: int = Field(
        default=0,
        description="0 = first attempt, 1 = first retry, 2 = second retry (max)",
    )
    agent_id: str | None = Field(default=None, description="Which agent made this call")
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Claim score (produced by Critique Agent) ─────────────────────────────────

class ClaimScore(BaseModel):
    """
    A confidence assessment for a specific text span within an agent's output.
    The critique agent MUST flag spans, not whole outputs.
    """
    span: str = Field(
        description="The exact verbatim text span being assessed (substring of agent output)",
    )
    confidence: float = Field(
        ge=0.0,
        le=1.0,
        description="Confidence that this claim is correct (0 = certainly wrong, 1 = certainly correct)",
    )
    flagged: bool = Field(
        default=False,
        description="True if the critique agent disagrees with or is uncertain about this claim",
    )
    reason: str | None = Field(
        default=None,
        description="If flagged=True, the reason for disagreement",
    )
    source_chunk_id: str | None = Field(
        default=None,
        description="If verifiable via a chunk, the chunk ID used to assess",
    )


# ─── Agent output ─────────────────────────────────────────────────────────────

class AgentOutput(BaseModel):
    """
    The complete output of a single agent run within a job.
    Written by the orchestrator after the agent completes.
    """
    agent_id: str
    output: str = Field(description="The main text output of the agent")
    structured_output: dict[str, Any] = Field(
        default_factory=dict,
        description="Any additional structured data (citations, subtasks, etc.)",
    )
    claim_scores: list[ClaimScore] = Field(
        default_factory=list,
        description="Populated by the critique agent after reviewing this output",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list,
        description="All tool calls made during this agent's execution",
    )
    token_count: int = Field(default=0, description="Total tokens consumed by this agent")
    latency_ms: int = Field(default=0, description="Wall-clock time for this agent's full run")
    input_hash: str | None = Field(
        default=None,
        description="SHA-256[:16] of the input prompt sent to the LLM",
    )
    output_hash: str | None = Field(
        default=None,
        description="SHA-256[:16] of the agent's output text",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)

    def compute_hashes(self) -> None:
        """Compute and store input/output hashes in place."""
        self.output_hash = hashlib.sha256(self.output.encode()).hexdigest()[:16]


# ─── Provenance entry (produced by Synthesis Agent) ───────────────────────────

class ProvenanceEntry(BaseModel):
    """
    Links a single sentence in the final answer back to its source.
    Every sentence in the synthesis output must have a provenance entry.
    """
    sentence: str = Field(description="The exact sentence from the final answer")
    source_agent: str = Field(description="Which agent produced the information")
    source_chunk_id: str | None = Field(
        default=None,
        description="If the sentence originates from a retrieved chunk, its ID",
    )
    source_chunk_excerpt: str | None = Field(
        default=None,
        description="A brief excerpt from the source chunk for traceability",
    )


# ─── Routing plan (produced by Orchestrator) ─────────────────────────────────

class RoutingPlan(BaseModel):
    """
    The orchestrator's decision about which agents to run, in what order,
    and with what budget. Logged to trace_events for auditability.
    """
    agents_selected: list[str] = Field(
        description="Ordered list of agent IDs to invoke",
    )
    per_agent_budget: dict[str, int] = Field(
        description="Token budget allocated to each agent",
    )
    reasoning: str = Field(
        description="The orchestrator's justification for these routing decisions",
    )
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ─── Shared Context (the single pipeline object) ─────────────────────────────

class SharedContext(BaseModel):
    """
    The central shared context object that flows through the entire pipeline.

    Invariants:
    - job_id is set at creation and never changes.
    - Agents append to lists (subtasks, tool_call_log, etc.) — they never overwrite.
    - The orchestrator is the only component that writes to agent_outputs.
    - final_answer and provenance_map are written exactly once by the synthesis agent.
    - budget_violations is append-only; violations are never removed.
    """

    # ── Identity ──────────────────────────────────────────────────────────────
    job_id: UUID = Field(default_factory=uuid4)
    query: str = Field(description="The original user query, unmodified")

    # ── Orchestration ─────────────────────────────────────────────────────────
    routing_plan: RoutingPlan | None = Field(
        default=None,
        description="The orchestrator's routing decision for this job",
    )

    # ── Sub-tasks (written by Decomposition Agent) ────────────────────────────
    subtasks: list[SubTask] = Field(
        default_factory=list,
        description="Decomposed sub-tasks. Must form a valid DAG.",
    )

    # ── Agent outputs (written by Orchestrator after each agent completes) ────
    agent_outputs: dict[str, AgentOutput] = Field(
        default_factory=dict,
        description="Keyed by agent_id. Set exactly once per agent per job.",
    )

    # ── Tool call log (every tool call across all agents) ─────────────────────
    tool_call_log: list[ToolCall] = Field(
        default_factory=list,
        description="Global append-only log of every tool call, including retries",
    )

    # ── Final output (written by Synthesis Agent) ─────────────────────────────
    final_answer: str | None = Field(
        default=None,
        description="The final answer surface to the user",
    )
    provenance_map: list[ProvenanceEntry] = Field(
        default_factory=list,
        description="Per-sentence provenance for every sentence in final_answer",
    )
    contradictions_resolved: list[str] = Field(
        default_factory=list,
        description="Description of each contradiction that was resolved (not surfaced)",
    )

    # ── Budget & compliance ───────────────────────────────────────────────────
    budget_violations: list[str] = Field(
        default_factory=list,
        description="Policy violations logged when an agent exceeds its token budget",
    )

    # ── Lifecycle ─────────────────────────────────────────────────────────────
    created_at: datetime = Field(default_factory=datetime.utcnow)
    completed_at: datetime | None = None

    # ── Utility methods ───────────────────────────────────────────────────────

    def context_hash(self) -> str:
        """
        SHA-256 fingerprint of the full context state.
        Used for deduplication and change detection in trace logging.
        """
        payload = self.model_dump(mode="json")
        serialized = json.dumps(payload, default=str, sort_keys=True)
        return hashlib.sha256(serialized.encode()).hexdigest()[:16]

    def get_pending_subtasks(self) -> list[SubTask]:
        """Return sub-tasks that are ready to run (all dependencies done)."""
        done_ids = {st.id for st in self.subtasks if st.status == "done"}
        return [
            st
            for st in self.subtasks
            if st.status == "pending"
            and all(dep in done_ids for dep in st.depends_on)
        ]

    def get_subtask_by_id(self, task_id: str) -> SubTask | None:
        return next((st for st in self.subtasks if st.id == task_id), None)

    def all_subtasks_done(self) -> bool:
        return all(st.status == "done" for st in self.subtasks)

    def has_budget_violations(self) -> bool:
        return len(self.budget_violations) > 0

    def summary_stats(self) -> dict[str, Any]:
        """
        Quick stats for logging and SSE budget_update events.
        """
        total_tokens = sum(
            ao.token_count for ao in self.agent_outputs.values()
        )
        total_tool_calls = len(self.tool_call_log)
        accepted_calls = sum(1 for tc in self.tool_call_log if tc.accepted)
        return {
            "total_tokens_consumed": total_tokens,
            "agents_completed": len(self.agent_outputs),
            "total_tool_calls": total_tool_calls,
            "tool_calls_accepted": accepted_calls,
            "tool_calls_rejected": total_tool_calls - accepted_calls,
            "budget_violations": len(self.budget_violations),
            "subtasks_total": len(self.subtasks),
            "subtasks_done": sum(1 for st in self.subtasks if st.status == "done"),
        }

    class Config:
        # Allow mutation for orchestrator to write outputs
        frozen = False
