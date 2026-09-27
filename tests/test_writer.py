from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pyarrow.parquet as pq
import pytest
from _helpers import utc

from eventlake.event import Event
from eventlake.schema import SchemaChangeError
from eventlake.writer import Writer


class Ping(Event):
    event_type: ClassVar[str] = "ping"

    source: str


def read_all_partition_files(root: Path, event_type: str) -> list[Path]:
    return sorted((root / event_type).glob("dt=*/part-*.parquet"))


def test_write_creates_expected_partition_layout(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    files = read_all_partition_files(tmp_path, "ping")
    assert len(files) == 1
    assert files[0].parent.name == "dt=2026-01-01"
    table = pq.read_table(files[0])
    assert table.num_rows == 1
    assert (tmp_path / "_schemas" / "ping" / "v1.json").exists()


def test_late_event_partitioned_by_occurred_at_not_write_time(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2020, 5, 17), source="late"))

    files = read_all_partition_files(tmp_path, "ping")
    assert len(files) == 1
    assert files[0].parent.name == "dt=2020-05-17"


def test_write_many_splits_into_partitions_by_day(tmp_path: Path) -> None:
    events = [
        Ping(occurred_at=utc(2026, 1, 1), source="a"),
        Ping(occurred_at=utc(2026, 1, 2), source="b"),
        Ping(occurred_at=utc(2026, 1, 2), source="c"),
    ]
    with Writer(tmp_path) as writer:
        writer.write_many(events)

    files = read_all_partition_files(tmp_path, "ping")
    partitions = {f.parent.name for f in files}
    assert partitions == {"dt=2026-01-01", "dt=2026-01-02"}


def test_duplicate_event_id_within_same_writer_is_dropped(tmp_path: Path) -> None:
    event = Ping(occurred_at=utc(2026, 1, 1), source="a")
    with Writer(tmp_path) as writer:
        writer.write(event)
        writer.write(event)  # same event_id, second write must be dropped

    files = read_all_partition_files(tmp_path, "ping")
    assert len(files) == 1
    table = pq.read_table(files[0])
    assert table.num_rows == 1


def test_duplicate_event_id_across_flushes_in_same_writer_is_dropped(tmp_path: Path) -> None:
    event = Ping(occurred_at=utc(2026, 1, 1), source="a")
    with Writer(tmp_path, max_rows=1) as writer:
        writer.write(event)  # triggers immediate flush (max_rows=1)
        writer.write(event)  # duplicate, must still be dropped after flush

    total_rows = sum(pq.read_table(f).num_rows for f in read_all_partition_files(tmp_path, "ping"))
    assert total_rows == 1


def test_max_rows_triggers_automatic_flush(tmp_path: Path) -> None:
    writer = Writer(tmp_path, max_rows=2)
    writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    assert read_all_partition_files(tmp_path, "ping") == []
    writer.write(Ping(occurred_at=utc(2026, 1, 1), source="b"))
    # max_rows reached: flush should have already happened, before __exit__
    assert len(read_all_partition_files(tmp_path, "ping")) == 1
    writer.flush()


def test_flush_with_empty_buffer_is_a_noop(tmp_path: Path) -> None:
    writer = Writer(tmp_path)
    writer.flush()
    writer.flush()
    assert not (tmp_path / "ping").exists()


def test_writing_after_close_raises(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    with pytest.raises(RuntimeError):
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="b"))


def test_compatible_schema_change_registers_new_version(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    class PingV2(Event):
        event_type: ClassVar[str] = "ping"

        source: str
        note: str | None = None

    with Writer(tmp_path) as writer:
        writer.write(PingV2(occurred_at=utc(2026, 1, 2), source="b", note="hi"))

    from eventlake.schema import SchemaRegistry

    registry = SchemaRegistry(tmp_path)
    assert registry.versions("ping") == [1, 2]


def test_incompatible_schema_change_raises_and_keeps_buffer(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    class PingBroken(Event):
        event_type: ClassVar[str] = "ping"

        note: str  # `source` removed, `note` is required: breaking change

    writer = Writer(tmp_path, allow_breaking=False)
    writer.write(PingBroken(occurred_at=utc(2026, 1, 2), note="hi"))
    with pytest.raises(SchemaChangeError):
        writer.flush()

    from eventlake.schema import SchemaRegistry

    registry = SchemaRegistry(tmp_path)
    assert registry.versions("ping") == [1]

    # the buffer must still hold the event: flushing again (now allowed) succeeds
    writer._allow_breaking = True
    writer.flush()
    assert registry.versions("ping") == [1, 2]


def test_allow_breaking_true_accepts_incompatible_change_directly(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    class PingBroken(Event):
        event_type: ClassVar[str] = "ping"

        note: str

    with Writer(tmp_path, allow_breaking=True) as writer:
        writer.write(PingBroken(occurred_at=utc(2026, 1, 2), note="hi"))

    from eventlake.schema import SchemaRegistry

    registry = SchemaRegistry(tmp_path)
    assert registry.versions("ping") == [1, 2]


def test_exception_during_write_leaves_no_partial_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow.parquet

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    # eventlake.writer imports this same module as `pq`; patching the shared
    # module object here affects its `pq.write_table` calls too.
    monkeypatch.setattr(pyarrow.parquet, "write_table", boom)

    writer = Writer(tmp_path)
    with pytest.raises(RuntimeError, match="disk full"):
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
        writer.flush()

    partition_dir = tmp_path / "ping" / "dt=2026-01-01"
    if partition_dir.exists():
        assert list(partition_dir.iterdir()) == []
