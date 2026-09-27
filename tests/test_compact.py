from __future__ import annotations

import uuid
from pathlib import Path
from typing import ClassVar

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
from _helpers import utc

from eventlake.compact import CompactionIncomplete, compact
from eventlake.event import Event, arrow_schema_for
from eventlake.lake import Lake
from eventlake.schema import SchemaRegistry
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


# --- M5: a single-file partition is rewritten if it has internal dupes -----


def test_compact_rewrites_single_file_with_internal_duplicate_event_id(
    tmp_path: Path,
) -> None:
    schema = arrow_schema_for(Ping)
    event_id = str(uuid.uuid4())
    occurred = utc(2026, 1, 1)
    table = pa.Table.from_pylist(
        [
            {
                "event_id": event_id,
                "occurred_at": occurred,
                "recorded_at": utc(2026, 1, 1, 0, 0, 1),
                "source": "first",
            },
            {
                "event_id": event_id,
                "occurred_at": occurred,
                "recorded_at": utc(2026, 1, 1, 0, 0, 2),
                "source": "second",
            },
        ],
        schema=schema,
    )
    partition_dir = tmp_path / "ping" / "dt=2026-01-01"
    partition_dir.mkdir(parents=True)
    pq.write_table(table, partition_dir / "part-manual.parquet")
    SchemaRegistry(tmp_path).register("ping", schema, allow_breaking=False)

    result = compact(tmp_path, "ping", "2026-01-01")
    assert result.files_before == 1
    assert result.files_after == 1
    assert result.rows_before == 2
    assert result.rows_after == 1

    files = partition_files(tmp_path, "ping", "2026-01-01")
    assert len(files) == 1
    assert files[0].name != "part-manual.parquet"  # rewritten, not left as-is
    out_table = pq.read_table(files[0])
    assert out_table.column("source").to_pylist() == ["first"]  # earliest recorded_at wins


def test_compact_single_file_without_duplicates_is_untouched(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    before = partition_files(tmp_path, "ping", "2026-01-01")
    assert len(before) == 1
    original_name = before[0].name

    result = compact(tmp_path, "ping", "2026-01-01")
    assert result.files_before == 1
    assert result.files_after == 1
    assert result.rows_before == 1
    assert result.rows_after == 1

    after = partition_files(tmp_path, "ping", "2026-01-01")
    assert len(after) == 1
    assert after[0].name == original_name  # untouched, not rewritten


# --- M4: a failed old-file delete raises CompactionIncomplete, not silent --


def test_compact_incomplete_when_old_file_deletion_fails_and_rerun_cleans_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="b"))

    original_files = partition_files(tmp_path, "ping", "2026-01-01")
    assert len(original_files) == 2
    stubborn_file = original_files[0]

    real_unlink = Path.unlink

    def flaky_unlink(self: Path, *args: object, **kwargs: object) -> None:
        if self == stubborn_file:
            raise OSError("permission denied")
        real_unlink(self, *args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr(Path, "unlink", flaky_unlink)

    with pytest.raises(CompactionIncomplete) as exc_info:
        compact(tmp_path, "ping", "2026-01-01")

    assert exc_info.value.leftover_files == [stubborn_file]
    assert exc_info.value.event_type == "ping"
    assert exc_info.value.dt == "2026-01-01"

    remaining = partition_files(tmp_path, "ping", "2026-01-01")
    assert stubborn_file in remaining
    assert len(remaining) == 2  # stubborn old file + the new merged file

    # Reading via Lake still gives correct, deduplicated results in this state.
    table = Lake(tmp_path).events("ping").to_arrow_table()
    assert sorted(table.column("source").to_pylist()) == ["a", "b"]

    monkeypatch.setattr(Path, "unlink", real_unlink)
    result = compact(tmp_path, "ping", "2026-01-01")
    assert result.files_after == 1

    final_files = partition_files(tmp_path, "ping", "2026-01-01")
    assert len(final_files) == 1
    final_table = pq.read_table(final_files[0])
    assert sorted(final_table.column("source").to_pylist()) == ["a", "b"]


def test_compact_validates_event_type_before_using_it_as_a_path() -> None:
    with pytest.raises(ValueError, match="invalid event_type"):
        compact("/tmp/does-not-matter", "../escape", "2026-01-01")


@pytest.mark.parametrize(
    "bad_dt",
    ["../x", "2026-1-1", "/abs/path", "", "20260101", "2026-01-01T00:00:00"],
)
def test_compact_validates_dt_before_using_it_as_a_path(tmp_path: Path, bad_dt: str) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    with pytest.raises(ValueError, match="invalid partition date"):
        compact(tmp_path, "ping", bad_dt)

    # Nothing should have been created or removed outside the real partition.
    assert (tmp_path / "ping" / "dt=2026-01-01").exists()
    assert not (tmp_path.parent / "x").exists()


# --- K3: a user "filename" field must not collide with the dedup tiebreak --


class Download(Event):
    event_type: ClassVar[str] = "download"

    filename: str
    size: int


def test_events_and_compact_work_with_a_real_filename_field(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Download(occurred_at=utc(2026, 1, 1), filename="a.txt", size=1))
    with Writer(tmp_path) as writer:
        writer.write(Download(occurred_at=utc(2026, 1, 1), filename="b.txt", size=2))

    table = Lake(tmp_path).events("download").to_arrow_table()
    assert table.num_rows == 2
    by_filename = dict(
        zip(table.column("filename").to_pylist(), table.column("size").to_pylist(), strict=True)
    )
    assert by_filename == {"a.txt": 1, "b.txt": 2}

    result = compact(tmp_path, "download", "2026-01-01")
    assert result.files_before == 2
    assert result.files_after == 1
    assert result.rows_after == 2

    files = partition_files(tmp_path, "download", "2026-01-01")
    assert len(files) == 1
    out_table = pq.read_table(files[0])
    assert "filename" in out_table.column_names
    assert "__eventlake_source_file" not in out_table.column_names
    by_filename_after = dict(
        zip(
            out_table.column("filename").to_pylist(),
            out_table.column("size").to_pylist(),
            strict=True,
        )
    )
    assert by_filename_after == {"a.txt": 1, "b.txt": 2}
