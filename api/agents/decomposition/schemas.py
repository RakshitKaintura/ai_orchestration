"""
api/agents/decomposition/schemas.py

Instructor-structured output models for the Decomposition Agent.
Isolated from agent logic so they can be imported independently by tests,
the orchestrator, and other agents without pulling in the full agent class.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, model_validator


class SubTaskSpec(BaseModel):
    """LLM-generated sub-task specification (Instructor output model)."""

    id: str = Field(description="Short unique ID, e.g. 'st-1', 'st-2'")
    description: str = Field(description="Clear description of what this sub-task does")
    task_type: Literal["retrieval", "computation", "reasoning", "synthesis", "other"]
    depends_on: list[str] = Field(
        default_factory=list,
        description=(
            "IDs of sub-tasks that must complete before this one starts. "
            "Use [] for tasks with no dependencies."
        ),
    )


class DecompositionOutput(BaseModel):
    """Full structured response from the Decomposition Agent LLM call."""

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

    def validate_dag(self) -> list[str]:
        """
        Validate that the dependency graph is a valid DAG.
        Returns a list of error strings (empty = valid).
        Uses Kahn's topological sort algorithm.
        """
        errors: list[str] = []
        ids = {st.id for st in self.subtasks}

        # Check all depends_on IDs exist
        for st in self.subtasks:
            for dep in st.depends_on:
                if dep not in ids:
                    errors.append(
                        f"SubTask '{st.id}' depends on '{dep}' which does not exist. "
                        f"Available IDs: {sorted(ids)}"
                    )

        if errors:
            return errors

        # Topological sort — detect cycles
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
            errors.append(
                "Dependency graph contains a cycle. All dependencies must form a DAG."
            )

        return errors

    @model_validator(mode="after")
    def _validate_dag_on_init(self) -> "DecompositionOutput":
        """Pydantic validator — raises ValueError on invalid DAG."""
        errors = self.validate_dag()
        if errors:
            raise ValueError("; ".join(errors))
        return self
