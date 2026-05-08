"""
api/agents/decomposition.py

Decomposition Agent
-------------------
Given the user's query, breaks it into typed atomic sub-tasks with explicit
dependency relationships. Sub-tasks must form a valid DAG (no cycles).

The orchestrator checks SubTask.depends_on before scheduling any sub-task.
Sub-tasks with unresolved dependencies are held in "pending" status until
their dependencies complete.

Budget: 4000 tokens (from env: DECOMPOSITION_BUDGET)

Structured output via Instructor:
  - List of SubTask objects
  - Each with: id, description, task_type, depends_on list
  - Plus reasoning string explaining the decomposition

Validation:
  - DAG check: no cycles (topological sort must succeed)
  - Orphan check: all depends_on IDs must exist in the subtask list
  - Type check: task_type must be in the allowed enum
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Literal

import anthropic
import instructor
from pydantic import BaseModel, Field, model_validator

from api.agents.base import BaseAgent
from api.agents.compression import compress_context_async
from api.config import get_settings
from api.context_manager import BudgetManager
from api.models.context import AgentOutput, SharedContext, SubTask

logger = logging.getLogger(__name__)


# ─── Instructor structured output ────────────────────────────────────────────

class SubTaskSpec(BaseModel):
    """LLM-generated sub-task specification."""
    id: str = Field(description="Short unique ID, e.g. 'st-1', 'st-2'")
    description: str = Field(description="Clear description of what this sub-task does")
    task_type: Literal["retrieval", "computation", "reasoning", "synthesis", "other"]
    depends_on: list[str] = Field(
        default_factory=list,
        description="IDs of sub-tasks that must complete before this one starts. "
                    "Use [] for tasks with no dependencies.",
    )


class DecompositionOutput(BaseModel):
    subtasks: list[SubTaskSpec] = Field(
        description="Ordered list of atomic sub-tasks. Must form a valid DAG.",
        min_length=1,
    )
    reasoning: str = Field(
        description="Explanation of why the query was decomposed this way",
    )
    is_ambiguous: bool = Field(
        default=False,
        description="True if the query is underspecified or ambiguous",
    )
    ambiguity_notes: str = Field(
        default="",
        description="If is_ambiguous=True, describe what is unclear",
    )

    @model_validator(mode="after")
    def validate_dag(self) -> "DecompositionOutput":
        """Verify dependency graph is a valid DAG (no cycles, no missing IDs)."""
        ids = {st.id for st in self.subtasks}

        # Check all depends_on IDs exist
        for st in self.subtasks:
            for dep in st.depends_on:
                if dep not in ids:
                    raise ValueError(
                        f"SubTask '{st.id}' depends on '{dep}' which does not exist. "
                        f"Available IDs: {sorted(ids)}"
                    )

        # Topological sort (Kahn's algorithm) to detect cycles
        in_degree = {st.id: 0 for st in self.subtasks}
        adj: dict[str, list[str]] = {st.id: [] for st in self.subtasks}

        for st in self.subtasks:
            for dep in st.depends_on:
                adj[dep].append(st.id)
                in_degree[st.id] += 1

        queue = [sid for sid, deg in in_degree.items() if deg == 0]
        processed = 0
        while queue:
            node = queue.pop(0)
            processed += 1
            for neighbor in adj[node]:
                in_degree[neighbor] -= 1
                if in_degree[neighbor] == 0:
                    queue.append(neighbor)

        if processed != len(self.subtasks):
            raise ValueError(
                "Dependency graph contains a cycle. All dependencies must form a DAG."
            )

        return self


# ─── System prompt ────────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a decomposition agent in a multi-agent AI system.

Your job: break down user queries into atomic, typed sub-tasks with explicit dependencies.

Rules:
1. Each sub-task must be atomic (cannot be usefully split further).
2. Assign a task_type: retrieval, computation, reasoning, synthesis, or other.
3. Use depends_on to express which sub-tasks must complete before another starts.
4. Dependencies MUST form a DAG (no cycles). Never create circular dependencies.
5. For simple queries (e.g. "What is 2+2?"), return a single sub-task.
6. For ambiguous queries, set is_ambiguous=True and explain what is unclear.
7. Sub-task IDs must be short and unique: 'st-1', 'st-2', etc.

The sub-tasks you produce will be executed by other agents. Make descriptions specific."""


# ─── Agent ────────────────────────────────────────────────────────────────────

class DecompositionAgent(BaseAgent):
    agent_id = "decomposition"
    default_budget = 4000

    async def run(self) -> AgentOutput:
        settings = get_settings()
        t0 = time.perf_counter()

        # 1. Declare budget
        budget = settings.decomposition_budget
        self._declare_budget(budget)

        # 2. Build prompt
        prompt = (
            f"Query to decompose:\n{self.ctx.query}\n\n"
            f"Existing context:\n"
            f"  - Job ID: {self.ctx.job_id}\n"
            f"  - Prior agent outputs: {list(self.ctx.agent_outputs.keys()) or 'none'}"
        )
        input_hash = self._hash(prompt)

        # 3. Budget check before sending to LLM
        if not self._check_and_add(prompt):
            # Budget exceeded — compress then try again
            logger.warning("decomposition_budget_exceeded_compressing",
                           extra={"job_id": str(self.ctx.job_id)})
            compressed_prompt = await compress_context_async(prompt)
            if not self._check_and_add(compressed_prompt):
                # Still over — use minimal prompt
                compressed_prompt = f"Query: {self.ctx.query[:500]}"
                self.bm.force_add(self.agent_id, compressed_prompt)
            prompt = compressed_prompt

        # 4. LLM call via Instructor
        raw_client = anthropic.AsyncAnthropic(api_key=settings.anthropic_api_key)
        client = instructor.from_anthropic(raw_client)

        try:
            result: DecompositionOutput = await client.messages.create(
                model=settings.primary_model,
                max_tokens=1500,
                system=_SYSTEM_PROMPT,
                messages=[{"role": "user", "content": prompt}],
                response_model=DecompositionOutput,
            )
        except Exception as e:
            logger.error("decomposition_llm_failed",
                         extra={"error": str(e), "job_id": str(self.ctx.job_id)})
            # Graceful degradation: produce a single generic subtask
            result = DecompositionOutput(
                subtasks=[SubTaskSpec(
                    id="st-1",
                    description=f"Answer the query: {self.ctx.query[:200]}",
                    task_type="reasoning",
                    depends_on=[],
                )],
                reasoning=f"Fallback: LLM decomposition failed ({e}). Single generic task.",
                is_ambiguous=False,
            )

        # 5. Convert to SharedContext SubTask objects and write to context
        subtasks = []
        for spec in result.subtasks:
            st = SubTask(
                id=spec.id,
                description=spec.description,
                task_type=spec.task_type,
                depends_on=spec.depends_on,
                status="pending",
            )
            subtasks.append(st)

        self.ctx.subtasks = subtasks  # orchestrator mediates, but decomp writes its own output

        # 6. Build output text
        output_lines = [
            f"DECOMPOSITION ({len(subtasks)} sub-tasks):",
            f"Reasoning: {result.reasoning}",
        ]
        if result.is_ambiguous:
            output_lines.append(f"⚠ Ambiguity detected: {result.ambiguity_notes}")
        for st in subtasks:
            deps = f"depends on [{', '.join(st.depends_on)}]" if st.depends_on else "no dependencies"
            output_lines.append(f"  • [{st.task_type}] {st.id}: {st.description} ({deps})")
        output_text = "\n".join(output_lines)

        # 7. Track output tokens
        output_tokens = self.bm.count_tokens(output_text)
        self.bm.force_add(self.agent_id, output_text)  # lossless: structured output

        latency = int((time.perf_counter() - t0) * 1000)

        return self._make_output(
            output=output_text,
            structured={
                "subtasks": [s.model_dump() for s in subtasks],
                "reasoning": result.reasoning,
                "is_ambiguous": result.is_ambiguous,
                "ambiguity_notes": result.ambiguity_notes,
            },
            token_count=self.bm.get_all_states()[self.agent_id].consumed,
            latency_ms=latency,
            input_hash=input_hash,
        )
