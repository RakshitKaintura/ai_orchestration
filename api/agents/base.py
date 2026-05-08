"""
api/agents/base.py

Base class for all Mega AI agents.

Every agent:
1. Declares its budget via BudgetManager before execution
2. Calls bm.check_remaining() before assembling context
3. Uses bm.add() to consume tokens — returns False → must compress first
4. Computes and stores input_hash + output_hash on its AgentOutput
5. Records latency

Agents communicate ONLY through SharedContext. They never call each other.
The orchestrator is the sole caller of agent.run().
"""

from __future__ import annotations

import hashlib
import logging
import time
from abc import ABC, abstractmethod
from typing import Any

from api.context_manager import BudgetManager
from api.models.context import AgentOutput, SharedContext

logger = logging.getLogger(__name__)


class BaseAgent(ABC):
    """Abstract base for all pipeline agents."""

    agent_id: str = "base"
    default_budget: int = 4000

    def __init__(self, ctx: SharedContext, bm: BudgetManager) -> None:
        self.ctx = ctx
        self.bm = bm

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def _declare_budget(self, budget: int | None = None) -> None:
        self.bm.declare(self.agent_id, budget or self.default_budget)

    def _check_and_add(self, text: str) -> bool:
        """
        Try to add text to the budget. If over budget, return False.
        Caller must trigger compression if False.
        """
        return self.bm.add(self.agent_id, text)

    def _make_output(
        self,
        output: str,
        structured: dict | None = None,
        token_count: int = 0,
        latency_ms: int = 0,
        input_hash: str | None = None,
    ) -> AgentOutput:
        ao = AgentOutput(
            agent_id=self.agent_id,
            output=output,
            structured_output=structured or {},
            token_count=token_count,
            latency_ms=latency_ms,
            input_hash=input_hash,
        )
        ao.compute_hashes()
        return ao

    @abstractmethod
    async def run(self) -> AgentOutput:
        """Execute the agent and return its output. Called by orchestrator only."""
        ...
