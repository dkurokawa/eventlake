from __future__ import annotations

import uuid
from pathlib import Path
from typing import ClassVar

import pyarrow.parquet as pq
import pytest
from _helpers import utc

from eventlake.compact import compact
from eventlake.event import Event
from eventlake.lake import Lake
from eventlake.writer import Writer


class Ping(Event):
    event_type: ClassVar[str] = "ping"

    source: str


def partition_files(root: Path, event_type: str, dt: str) -> list[Path]:
    return sorted((root / event_type / f"dt={dt}").glob("part-*.parquet"))


def test_compact_merges_multiple_files_into_one(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="b"))

    assert len(partition_files(tmp_path, "ping", "2026-01-01")) == 2

    result = compact(tmp_path, "ping", "2026-01-01")
    assert result.files_before == 2
    assert result.files_after == 1
    assert result.rows_before == 2
    assert result.rows_after == 2

    files = partition_files(tmp_path, "ping", "2026-01-01")
    assert len(files) == 1
    table = pq.read_table(files[0])
    assert sorted(table.column("source").to_pylist()) == ["a", "b"]


def test_compact_drops_duplicate_event_ids_keeping_earliest_recorded_at(tmp_path: Path) -> None:
    event_id = uuid.uuid4()
    with Writer(tmp_path) as writer:
        writer.write(Ping(event_id=event_id, occurred_at=utc(2026, 1, 1), source="first"))

    import time

    time.sleep(0.01)

    with Writer(tmp_path) as writer:
        writer.write(Ping(event_id=event_id, occurred_at=utc(2026, 1, 1), source="second"))

    result = compact(tmp_path, "ping", "2026-01-01")
    assert result.rows_before == 2
    assert result.rows_after == 1

    files = partition_files(tmp_path, "ping", "2026-01-01")
    table = pq.read_table(files[0])
    assert table.column("source").to_pylist() == ["first"]


def test_compact_result_matches_lake_query_before_and_after(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="b"))
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="c"))

    before = sorted(Lake(tmp_path).events("ping").to_arrow_table().column("source").to_pylist())
    compact(tmp_path, "ping", "2026-01-01")
    after = sorted(Lake(tmp_path).events("ping").to_arrow_table().column("source").to_pylist())
    assert before == after == ["a", "b", "c"]


def test_compact_single_file_partition_is_a_cheap_noop(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    result = compact(tmp_path, "ping", "2026-01-01")
    assert result.files_before == 1
    assert result.files_after == 1
    assert result.rows_before == 1
    assert result.rows_after == 1
    assert len(partition_files(tmp_path, "ping", "2026-01-01")) == 1


def test_compact_missing_partition_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        compact(tmp_path, "ping", "2026-01-01")


def test_compact_failure_leaves_old_files_intact(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="b"))

    original_files = partition_files(tmp_path, "ping", "2026-01-01")
    assert len(original_files) == 2

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    # eventlake.compact imports this same module as `pq`; patching the shared
    # module object here affects its `pq.write_table` calls too.
    monkeypatch.setattr(pq, "write_table", boom)

    with pytest.raises(RuntimeError, match="disk full"):
        compact(tmp_path, "ping", "2026-01-01")

    remaining = partition_files(tmp_path, "ping", "2026-01-01")
    assert sorted(remaining) == sorted(original_files)
    leftovers = list((tmp_path / "ping" / "dt=2026-01-01").glob("*.tmp"))
    assert leftovers == []
