"""
Shared pytest fixtures and configuration.
"""
from __future__ import annotations

import pytest


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests (require network)"
    )

