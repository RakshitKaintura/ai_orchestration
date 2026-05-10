"""
api/tools/registry.py

Tool registry — maps tool name strings to their callable functions.
Used by the orchestrator to resolve tool names to implementations
without importing all tools at module load time.
"""

from __future__ import annotations

from typing import Callable, Awaitable, Any

# Populated lazily on first access
_REGISTRY: dict[str, Callable[..., Awaitable[Any]]] | None = None


def _build_registry() -> dict[str, Callable[..., Awaitable[Any]]]:
    from api.tools.web_search.tool import web_search
    from api.tools.code_sandbox.tool import code_sandbox
    from api.tools.nl_to_sql.tool import nl_to_sql
    from api.tools.self_reflection.tool import self_reflection

    return {
        "web_search": web_search,
        "code_sandbox": code_sandbox,
        "nl_to_sql": nl_to_sql,
        "self_reflection": self_reflection,
    }


def get_tool(name: str) -> Callable[..., Awaitable[Any]]:
    """
    Look up a tool by name.

    Args:
        name: Tool name string (e.g. 'web_search')

    Returns:
        Async callable that accepts a dict input and returns ToolResult

    Raises:
        KeyError: If the tool name is not registered
    """
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()

    if name not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(f"Tool '{name}' not found. Available tools: {available}")

    return _REGISTRY[name]


def list_tools() -> list[str]:
    """Return all registered tool names."""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = _build_registry()
    return sorted(_REGISTRY.keys())
