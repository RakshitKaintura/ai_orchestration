"""
api/config.py

Centralised settings loaded exclusively from environment variables.
No credentials are hardcoded here. The .env file is never committed to git.
"""

from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",  # ignore unknown env vars
    )

    # ── LLM Providers ────────────────────────────────────────────────────────
    google_api_key: str = Field(description="Google API key for Gemini models")
    gemini_api_key: str = Field(description="Gemini API key (alias for Google API key)")

    # ── Database ──────────────────────────────────────────────────────────────
    postgres_host: str = Field(default="db")
    postgres_port: int = Field(default=5432)
    postgres_db: str = Field(default="mega_ai")
    postgres_user: str = Field(default="mega_ai_user")
    postgres_password: str = Field(description="PostgreSQL password")

    @property
    def database_url(self) -> str:
        return (
            f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    @property
    def database_url_sync(self) -> str:
        """Sync DSN for Celery tasks that use psycopg2."""
        return (
            f"postgresql://{self.postgres_user}:{self.postgres_password}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )

    # ── Redis / Celery ────────────────────────────────────────────────────────
    redis_url: str = Field(default="redis://redis:6379/0")
    celery_broker_url: str = Field(default="redis://redis:6379/0")
    celery_result_backend: str = Field(default="redis://redis:6379/1")

    # ── API Server ────────────────────────────────────────────────────────────
    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)
    log_level: str = Field(default="info")

    # ── ChromaDB ─────────────────────────────────────────────────────────────
    chroma_persist_dir: str = Field(default="/app/chroma_data")
    chroma_collection_name: str = Field(default="mega_ai_corpus")

    # ── Context Budgets (tokens) ──────────────────────────────────────────────
    decomposition_budget: int = Field(default=4000)
    rag_budget: int = Field(default=6000)
    critique_budget: int = Field(default=4000)
    synthesis_budget: int = Field(default=5000)
    compression_budget: int = Field(default=2000)
    orchestrator_budget: int = Field(default=3000)

    # ── Eval ─────────────────────────────────────────────────────────────────
    auto_eval: bool = Field(default=False)

    # ─── LLM Model names ───────────────────────────────────────────────────────
    primary_model: str = Field(default="gemini-3.1-flash-lite")
    embedding_model: str = Field(default="models/gemini-embedding-2")
    judge_model: str = Field(default="gemini-3.1-flash-lite")


@lru_cache
def get_settings() -> Settings:
    """Return a cached singleton Settings instance."""
    return Settings()
