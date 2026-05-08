"""
api/logging_config.py

Structured logging configuration for Mega AI.
All log entries follow a consistent schema:
  timestamp, job_id, agent_id, event_type,
  input_hash, output_hash, latency_ms, token_count,
  policy_violations, payload

Uses structlog for structured JSON output that is machine-parseable
and compatible with the log query UI.
"""

from __future__ import annotations

import logging
import logging.config
import os
import sys
from pathlib import Path

import structlog


# ── Log file location ─────────────────────────────────────────────────────────
LOG_DIR = Path(os.environ.get("LOG_DIR", "/app/logs"))
LOG_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = LOG_DIR / "mega_ai.jsonl"


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog + stdlib logging.
    Call once at application startup.
    """
    log_level_int = getattr(logging, log_level.upper(), logging.INFO)

    # Stdlib logging config
    logging.config.dictConfig({
        "version": 1,
        "disable_existing_loggers": False,
        "formatters": {
            "json": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.processors.JSONRenderer(),
            },
            "console": {
                "()": structlog.stdlib.ProcessorFormatter,
                "processor": structlog.dev.ConsoleRenderer(colors=True),
            },
        },
        "handlers": {
            "console": {
                "class": "logging.StreamHandler",
                "stream": "ext://sys.stdout",
                "formatter": "console",
            },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "filename": str(LOG_FILE),
                "maxBytes": 100 * 1024 * 1024,  # 100 MB
                "backupCount": 5,
                "formatter": "json",
            },
        },
        "root": {
            "handlers": ["console", "file"],
            "level": log_level_int,
        },
    })

    # Structlog configuration
    structlog.configure(
        processors=[
            structlog.stdlib.add_log_level,
            structlog.stdlib.add_logger_name,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def get_logger(name: str) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger bound with the given name."""
    return structlog.get_logger(name)
