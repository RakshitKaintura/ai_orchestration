"""
tests/conftest.py

Shared pytest configuration and fixtures for the Mega AI test suite.
"""

from __future__ import annotations

import pytest
import pytest_asyncio


# Set asyncio mode globally so all async tests work without individual decorators
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "asyncio: mark test as asyncio"
    )


# ─── pytest-asyncio configuration ────────────────────────────────────────────

pytest_plugins = ["pytest_asyncio"]
