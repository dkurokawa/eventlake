"""Small shared helpers for the test suite."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

from eventlake.storage import open_storage


def utc(
    year: int, month: int, day: int, hour: int = 0, minute: int = 0, second: int = 0
) -> datetime:
    """Build a UTC-aware datetime for test fixtures."""
    return datetime(year, month, day, hour, minute, second, tzinfo=UTC)


def keys(root: str | Path, prefix: str = "") -> list[str]:
    """Every stored key under `prefix`, whichever backend `root` is on."""
    return open_storage(root).list_keys(prefix)


def part_keys(root: str | Path, event_type: str, dt: str | None = None) -> list[str]:
    """Parquet file keys of an event type (optionally one partition day)."""
    where = f"{event_type}/dt={dt}/" if dt is not None else f"{event_type}/"
    return [k for k in keys(root, where) if k.rsplit("/", 1)[-1].startswith("part-")]


def read_table(root: str | Path, key: str) -> pa.Table:
    return pq.read_table(pa.BufferReader(open_storage(root).read_bytes(key)))
