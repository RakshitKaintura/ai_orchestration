"""
api/app.py

FastAPI application factory.

Creates and configures the FastAPI application with all routers,
middleware, and lifecycle hooks. Import `app` from here for uvicorn.
"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from api.config import get_settings
from api.logging_config import configure_logging, get_logger
from api.routes import query, trace, eval, rewrites

settings = get_settings()
configure_logging(settings.log_level)
logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown hooks."""
    logger.info("mega_ai_startup", version="0.1.0", env=settings.log_level)
    try:
        from api.agents.rag.retriever import _get_chroma_collection
        _get_chroma_collection()
        logger.info("chromadb_initialised")
    except Exception as e:
        logger.warning("chromadb_init_failed", extra={"error": str(e)})
    yield
    logger.info("mega_ai_shutdown")


def create_app() -> FastAPI:
    """Application factory — creates and configures the FastAPI instance."""
    app = FastAPI(
        title="Mega AI — Multi-Agent LLM Orchestration",
        description=(
            "Production-grade multi-agent system with dynamic routing, "
            "RAG, critique/synthesis pipeline, self-improving eval loop, "
            "and real-time SSE streaming."
        ),
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
        lifespan=lifespan,
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ── Health check ──────────────────────────────────────────────────────────
    @app.get("/health", tags=["Internal"], summary="Health check")
    async def health_check() -> dict[str, str]:
        return {"status": "ok", "service": "mega-ai-api"}

    # ── Route modules ─────────────────────────────────────────────────────────
    app.include_router(query.router)
    app.include_router(trace.router)
    app.include_router(eval.router)
    app.include_router(rewrites.router)

    return app


# Module-level app instance for uvicorn / gunicorn
app = create_app()
