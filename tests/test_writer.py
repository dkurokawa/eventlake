from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pyarrow.parquet as pq
import pytest
from _helpers import utc

from eventlake.event import Event
from eventlake.lake import Lake
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


def test_duplicate_event_id_within_the_buffer_is_dropped(tmp_path: Path) -> None:
    event = Ping(occurred_at=utc(2026, 1, 1), source="a")
    with Writer(tmp_path) as writer:
        writer.write(event)
        writer.write(event)  # same id, still buffered: dropped here

    total_rows = sum(pq.read_table(f).num_rows for f in read_all_partition_files(tmp_path, "ping"))
    assert total_rows == 1


def test_duplicate_across_flushes_is_removed_on_read(tmp_path: Path) -> None:
    # The writer forgets ids once they are flushed (so a long-lived writer's
    # memory stays bounded); the read path is what guarantees one row per id.
    event = Ping(occurred_at=utc(2026, 1, 1), source="a")
    with Writer(tmp_path, max_rows=1) as writer:
        writer.write(event)  # flushed immediately
        writer.write(event)  # written again as its own file

    assert Lake(tmp_path).events("ping").to_arrow_table().num_rows == 1


def test_seen_ids_do_not_grow_past_the_buffer(tmp_path: Path) -> None:
    writer = Writer(tmp_path, max_rows=10)
    for i in range(100):
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source=str(i)))
    writer.flush()
    assert len(writer._seen_event_ids) == 0


def test_concurrent_writes_from_threads_lose_nothing(tmp_path: Path) -> None:
    import threading

    writer = Writer(tmp_path, max_rows=7)

    def work(offset: int) -> None:
        for i in range(50):
            writer.write(Ping(occurred_at=utc(2026, 1, 1), source=f"{offset}-{i}"))

    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    writer.close()

    assert Lake(tmp_path).events("ping").to_arrow_table().num_rows == 8 * 50
    with pytest.raises(RuntimeError, match="closed"):
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="late"))


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

    # K6: the partition directory this failed write created is left in
    # place rather than rmdir'd - removing it here would race with another
    # Writer concurrently targeting the same partition. Only the tmp file
    # (never a valid part-*.parquet, never visible to a reader) is cleaned up.
    partition_dir = tmp_path / "ping" / "dt=2026-01-01"
    if partition_dir.exists():
        assert list(partition_dir.glob("part-*.parquet")) == []
        assert list(partition_dir.glob("*.tmp")) == []

    # An empty (or nonexistent) partition dir left behind like this must not
    # stop Lake from opening or from ignoring this (data-less) event type.
    from eventlake.lake import Lake

    lake = Lake(tmp_path)
    assert lake.describe() == []


def test_same_event_type_different_classes_v1_then_v2_in_one_writer(tmp_path: Path) -> None:
    """H1: buffering is per-class, so a V1/V2 mix in one Writer resolves
    through the normal per-event_type schema rules, not a single merged one."""

    class PingV1(Event):
        event_type: ClassVar[str] = "ping"

        source: str

    class PingV2(Event):
        event_type: ClassVar[str] = "ping"

        source: str
        note: str | None = None

    with Writer(tmp_path) as writer:
        writer.write(PingV1(occurred_at=utc(2026, 1, 1), source="a"))
        writer.write(PingV2(occurred_at=utc(2026, 1, 2), source="b", note="hi"))

    from eventlake.schema import SchemaRegistry

    registry = SchemaRegistry(tmp_path)
    assert registry.versions("ping") == [1, 2]

    from eventlake.lake import Lake

    table = Lake(tmp_path).events("ping").to_arrow_table()
    assert table.num_rows == 2
    sources = table.column("source").to_pylist()
    notes = table.column("note").to_pylist()
    by_source = dict(zip(sources, notes, strict=True))
    assert by_source == {"a": None, "b": "hi"}


def test_same_event_type_v1_flushed_after_v2_is_already_latest(tmp_path: Path) -> None:
    """K5: with v1 and v2 both already registered (v2 latest), flushing a
    V2 event and then a V1 event in the same Writer must not fail - v1 is
    reused as an exact match instead of being diffed against v2 (latest),
    which would look like an incompatible field removal."""

    class PingV1(Event):
        event_type: ClassVar[str] = "ping"

        source: str

    class PingV2(Event):
        event_type: ClassVar[str] = "ping"

        source: str
        note: str | None = None

    # Establish both versions first, in the "safe" order.
    with Writer(tmp_path) as writer:
        writer.write(PingV1(occurred_at=utc(2026, 1, 1), source="a"))
    with Writer(tmp_path) as writer:
        writer.write(PingV2(occurred_at=utc(2026, 1, 2), source="b", note="hi"))

    from eventlake.schema import SchemaRegistry

    registry = SchemaRegistry(tmp_path)
    assert registry.versions("ping") == [1, 2]

    # Now flush a V2 event (matches latest trivially) THEN a V1 event
    # (matches the older, non-latest v1) in a single Writer/flush.
    with Writer(tmp_path) as writer:
        writer.write(PingV2(occurred_at=utc(2026, 1, 3), source="c", note="yo"))
        writer.write(PingV1(occurred_at=utc(2026, 1, 4), source="d"))

    assert registry.versions("ping") == [1, 2]  # no new version created

    from eventlake.lake import Lake

    table = Lake(tmp_path).events("ping").to_arrow_table()
    assert table.num_rows == 4
    assert set(table.column("source").to_pylist()) == {"a", "b", "c", "d"}


def test_same_event_type_different_classes_incompatible_raises(tmp_path: Path) -> None:
    class PingV1(Event):
        event_type: ClassVar[str] = "ping"

        source: str

    class PingBrokenV2(Event):
        event_type: ClassVar[str] = "ping"

        note: str  # `source` dropped, `note` required: breaking

    # Not a `with` block: exiting it would call flush() again and re-raise,
    # same reasoning as test_incompatible_schema_change_raises_and_keeps_buffer.
    writer = Writer(tmp_path)
    writer.write(PingV1(occurred_at=utc(2026, 1, 1), source="a"))
    writer.write(PingBrokenV2(occurred_at=utc(2026, 1, 2), note="hi"))
    with pytest.raises(SchemaChangeError):
        writer.flush()


def test_partial_multiday_flush_failure_does_not_duplicate_on_retry(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good_event = Ping(occurred_at=utc(2026, 1, 1), source="a")
    bad_event = Ping(occurred_at=utc(2026, 1, 2), source="b")

    writer = Writer(tmp_path)
    writer.write(good_event)
    writer.write(bad_event)

    original_write_file = Writer._write_file

    def flaky(self: Writer, event_type: str, dt: str, table: object) -> None:
        if dt == "2026-01-02":
            raise RuntimeError("boom")
        original_write_file(self, event_type, dt, table)

    monkeypatch.setattr(Writer, "_write_file", flaky)

    with pytest.raises(RuntimeError, match="boom"):
        writer.flush()

    files_01 = list((tmp_path / "ping" / "dt=2026-01-01").glob("part-*.parquet"))
    assert len(files_01) == 1

    # The 2026-01-01 event must be gone from the buffer (already written);
    # only the 2026-01-02 event should still be pending.
    remaining = [e for buf in writer._buffers.values() for e in buf]
    assert [e.event_id for e in remaining] == [bad_event.event_id]

    monkeypatch.setattr(Writer, "_write_file", original_write_file)
    writer.flush()

    files_01_after = list((tmp_path / "ping" / "dt=2026-01-01").glob("part-*.parquet"))
    files_02_after = list((tmp_path / "ping" / "dt=2026-01-02").glob("part-*.parquet"))
    assert len(files_01_after) == 1  # unchanged - not duplicated
    assert len(files_02_after) == 1


class Tagged(Event):
    event_type: ClassVar[str] = "tagged"
    tags: list[str]


def test_mutating_a_list_after_write_does_not_change_what_is_stored(tmp_path: Path) -> None:
    event = Tagged(occurred_at=utc(2026, 1, 1), tags=["a"])
    with Writer(tmp_path) as writer:
        writer.write(event)
        event.tags.append("mutated-after-write")

    (part,) = (tmp_path / "tagged").rglob("part-*.parquet")
    assert pq.read_table(part).column("tags").to_pylist() == [["a"]]
