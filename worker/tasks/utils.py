"""
worker/tasks/utils.py

Shared utilities for Celery tasks.
"""
from __future__ import annotations

async def _get_asyncpg_conn():
    """Get a direct asyncpg connection for worker tasks."""
    import asyncpg
    from api.config import get_settings
    s = get_settings()
    return await asyncpg.connect(
        host=s.postgres_host,
        port=s.postgres_port,
        database=s.postgres_db,
        user=s.postgres_user,
        password=s.postgres_password,
    )
