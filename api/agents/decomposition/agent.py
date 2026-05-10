"""
api/agents/decomposition/agent.py

Decomposition Agent — breaks a query into typed atomic sub-tasks with DAG deps.
"""

from __future__ import annotations

import asyncio
import logging
import time

import google.generativeai as genai
import instructor

from api.agents.base import BaseAgent
from api.agents.compression import compress_context_async
from api.agents.decomposition.schemas import DecompositionOutput, SubTaskSpec
from api.agents.llm_retry import call_with_retry
from api.config import get_settings
from api.models.context import AgentOutput, SubTask

logger = logging.getLogger(__name__)

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


class DecompositionAgent(BaseAgent):
    agent_id = "decomposition"
    default_budget = 4000

    async def run(self) -> AgentOutput:
        settings = get_settings()
        t0 = time.perf_counter()

        budget = settings.decomposition_budget
        self._declare_budget(budget)

        prompt = (
            f"Query to decompose:\n{self.ctx.query}\n\n"
            f"Existing context:\n"
            f"  - Job ID: {self.ctx.job_id}\n"
            f"  - Prior agent outputs: {list(self.ctx.agent_outputs.keys()) or 'none'}"
        )
        input_hash = self._hash(prompt)

        if not self._check_and_add(prompt):
            logger.warning("decomposition_budget_exceeded_compressing",
                           extra={"job_id": str(self.ctx.job_id)})
            compressed_prompt = await compress_context_async(prompt)
            if not self._check_and_add(compressed_prompt):
                compressed_prompt = f"Query: {self.ctx.query[:500]}"
                self.bm.force_add(self.agent_id, compressed_prompt)
            prompt = compressed_prompt

        api_keys = [k for k in [settings.google_api_key, settings.gemini_api_key] if k]
        system_content = _SYSTEM_PROMPT
        user_content = prompt

        def _decomp(client):
            return client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system_content},
                    {"role": "user", "content": user_content},
                ],
                response_model=DecompositionOutput,
            )

        try:
            result: DecompositionOutput = await call_with_retry(
                _decomp, api_keys=api_keys, model_name=settings.primary_model
            )
        except Exception as e:
            logger.error("decomposition_llm_failed",
                         extra={"error": str(e), "job_id": str(self.ctx.job_id)})
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

        subtasks = [
            SubTask(
                id=spec.id,
                description=spec.description,
                task_type=spec.task_type,
                depends_on=spec.depends_on,
                status="pending",
            )
            for spec in result.subtasks
        ]
        self.ctx.subtasks = subtasks

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

        self.bm.force_add(self.agent_id, output_text)
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
