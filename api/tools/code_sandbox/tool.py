"""
api/tools/code_sandbox/tool.py

Code Execution Sandbox — Tool #2
Runs a Python snippet in a subprocess and returns stdout, stderr, exit_code,
and execution_time_ms.

Security: this sandbox uses subprocess with restrictive flags. It is NOT
production-safe (no seccomp, no namespace isolation). See LIMITATIONS.md.
"""

from __future__ import annotations

import asyncio
import os
import sys
import tempfile
import time

from api.models.tools import CodeExecutionResult, ToolResult

EXEC_TIMEOUT_SECONDS = 10
TOOL_NAME = "code_sandbox"

_BLOCKED_PATTERNS = [
    "import os", "import sys", "import subprocess", "import socket",
    "__import__", "open(", "exec(", "eval(", "compile(",
    "getattr(", "setattr(", "delattr(", "__builtins__",
]


def _check_blocked(code: str) -> str | None:
    code_lower = code.lower().replace(" ", "")
    for pattern in _BLOCKED_PATTERNS:
        if pattern.replace(" ", "").lower() in code_lower:
            return pattern
    return None


async def _run_python(code: str) -> CodeExecutionResult:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False, encoding="utf-8") as f:
        f.write(code)
        script_path = f.name

    t0 = time.perf_counter()
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, script_path,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            env={
                "PATH": os.environ.get("PATH", ""),
                "PYTHONPATH": "",
                "HOME": tempfile.gettempdir(),
            },
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=EXEC_TIMEOUT_SECONDS
            )
            elapsed_ms = (time.perf_counter() - t0) * 1000
            return CodeExecutionResult(
                stdout=stdout_bytes.decode("utf-8", errors="replace"),
                stderr=stderr_bytes.decode("utf-8", errors="replace"),
                exit_code=proc.returncode or 0,
                execution_time_ms=elapsed_ms,
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            raise
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass


async def code_sandbox(input_data: dict) -> ToolResult:
    """
    Tool #2 — Code Execution Sandbox

    Input schema:
        code    (str, required)  — Python snippet to execute
        timeout (int, optional)  — override timeout in seconds (max 10)
    """
    t0 = time.perf_counter()

    code = input_data.get("code")
    if not isinstance(code, str) or not code.strip():
        return ToolResult.malformed(source=TOOL_NAME, message="'code' must be a non-empty string")

    blocked = _check_blocked(code)
    if blocked:
        return ToolResult.malformed(
            source=TOOL_NAME,
            message=f"Code contains blocked pattern: '{blocked}'. Remove unsafe operations.",
        )

    timeout = input_data.get("timeout", EXEC_TIMEOUT_SECONDS)
    if not isinstance(timeout, (int, float)) or timeout <= 0:
        timeout = EXEC_TIMEOUT_SECONDS
    timeout = min(float(timeout), EXEC_TIMEOUT_SECONDS)

    try:
        exec_result = await asyncio.wait_for(_run_python(code), timeout=timeout + 1)
    except asyncio.TimeoutError:
        latency = (time.perf_counter() - t0) * 1000
        return ToolResult.timeout(
            source=TOOL_NAME, latency_ms=latency,
            message=f"Code execution timed out after {timeout}s",
        )

    latency = (time.perf_counter() - t0) * 1000
    exec_result.execution_time_ms = latency

    return ToolResult.ok(data=exec_result.model_dump(), source=TOOL_NAME, latency_ms=latency)
