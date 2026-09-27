"""Small shared helpers for the test suite."""

from __future__ import annotations

from datetime import UTC, datetime


def utc(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    """Build a UTC-aware datetime for test fixtures."""
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)
