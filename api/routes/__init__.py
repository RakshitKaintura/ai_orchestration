"""
api/routes/__init__.py

FastAPI router modules for Mega AI API.
Import and include these in the main app factory.
"""
from api.routes import query, trace, eval, rewrites

__all__ = ["query", "trace", "eval", "rewrites"]
