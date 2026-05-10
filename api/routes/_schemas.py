"""
api/routes/_schemas.py

Shared Pydantic request/response schemas used across route modules.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ErrorResponse(BaseModel):
    error_code: str = Field(description="Machine-readable error code")
    message: str = Field(description="Human-readable error description")
    job_id: str | None = Field(default=None, description="Relevant job ID if applicable")


class ReviewRequest(BaseModel):
    decision: str = Field(
        description="Human decision on the prompt rewrite proposal",
        pattern="^(approved|rejected)$",
        examples=["approved"],
    )


class ReRunRequest(BaseModel):
    rewrite_id: str | None = Field(
        default=None,
        description=(
            "UUID of an approved prompt rewrite to use. "
            "If omitted, uses the latest approved rewrite."
        ),
    )
