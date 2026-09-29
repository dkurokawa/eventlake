from __future__ import annotations

from pathlib import Path
from typing import Any, ClassVar

import pytest
from _helpers import keys, part_keys, read_table, utc

from eventlake.event import Event
from eventlake.lake import Lake
from eventlake.schema import SchemaChangeError
from eventlake.writer import Writer


class Ping(Event):
    event_type: ClassVar[str] = "ping"

    source: str


def test_write_creates_expected_partition_layout(lake_root: str | Path) -> None:
    with Writer(lake_root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    files = part_keys(lake_root, "ping")
    assert len(files) == 1
    assert files[0].split("/")[1] == "dt=2026-01-01"
    table = read_table(lake_root, files[0])
    assert table.num_rows == 1
    assert keys(lake_root, "_schemas/ping/") == ["_schemas/ping/v1.json"]


def test_late_event_partitioned_by_occurred_at_not_write_time(lake_root: str | Path) -> None:
    with Writer(lake_root) as writer:
        writer.write(Ping(occurred_at=utc(2020, 5, 17), source="late"))

    files = part_keys(lake_root, "ping")
    assert len(files) == 1
    assert files[0].split("/")[1] == "dt=2020-05-17"


def test_write_many_splits_into_partitions_by_day(lake_root: str | Path) -> None:
    events = [
        Ping(occurred_at=utc(2026, 1, 1), source="a"),
        Ping(occurred_at=utc(2026, 1, 2), source="b"),
        Ping(occurred_at=utc(2026, 1, 2), source="c"),
    ]
    with Writer(lake_root) as writer:
        writer.write_many(events)

    files = part_keys(lake_root, "ping")
    partitions = {f.split("/")[1] for f in files}
    assert partitions == {"dt=2026-01-01", "dt=2026-01-02"}


def test_duplicate_event_id_within_same_writer_is_dropped(lake_root: str | Path) -> None:
    event = Ping(occurred_at=utc(2026, 1, 1), source="a")
    with Writer(lake_root) as writer:
        writer.write(event)
        writer.write(event)  # same event_id, second write must be dropped

    files = part_keys(lake_root, "ping")
    assert len(files) == 1
    table = read_table(lake_root, files[0])
    assert table.num_rows == 1


def test_duplicate_event_id_within_the_buffer_is_dropped(lake_root: str | Path) -> None:
    event = Ping(occurred_at=utc(2026, 1, 1), source="a")
    with Writer(lake_root) as writer:
        writer.write(event)
        writer.write(event)  # same id, still buffered: dropped here

    total_rows = sum(read_table(lake_root, f).num_rows for f in part_keys(lake_root, "ping"))
    assert total_rows == 1


def test_duplicate_across_flushes_is_removed_on_read(lake_root: str | Path) -> None:
    # The writer forgets ids once they are flushed (so a long-lived writer's
    # memory stays bounded); the read path is what guarantees one row per id.
    event = Ping(occurred_at=utc(2026, 1, 1), source="a")
    with Writer(lake_root, max_rows=1) as writer:
        writer.write(event)  # flushed immediately
        writer.write(event)  # written again as its own file

    assert Lake(lake_root).events("ping").to_arrow_table().num_rows == 1


def test_seen_ids_do_not_grow_past_the_buffer(lake_root: str | Path) -> None:
    writer = Writer(lake_root, max_rows=10)
    for i in range(100):
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source=str(i)))
    writer.flush()
    assert len(writer._seen_event_ids) == 0


def test_concurrent_writes_from_threads_lose_nothing(lake_root: str | Path) -> None:
    import threading

    writer = Writer(lake_root, max_rows=7)

    def work(offset: int) -> None:
        for i in range(50):
            writer.write(Ping(occurred_at=utc(2026, 1, 1), source=f"{offset}-{i}"))

    threads = [threading.Thread(target=work, args=(t,)) for t in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    writer.close()

    assert Lake(lake_root).events("ping").to_arrow_table().num_rows == 8 * 50
    with pytest.raises(RuntimeError, match="closed"):
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="late"))


def test_max_rows_triggers_automatic_flush(lake_root: str | Path) -> None:
    writer = Writer(lake_root, max_rows=2)
    writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    assert part_keys(lake_root, "ping") == []
    writer.write(Ping(occurred_at=utc(2026, 1, 1), source="b"))
    # max_rows reached: flush should have already happened, before __exit__
    assert len(part_keys(lake_root, "ping")) == 1
    writer.flush()


def test_flush_with_empty_buffer_is_a_noop(lake_root: str | Path) -> None:
    writer = Writer(lake_root)
    writer.flush()
    writer.flush()
    assert keys(lake_root, "ping/") == []


def test_writing_after_close_raises(lake_root: str | Path) -> None:
    with Writer(lake_root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    with pytest.raises(RuntimeError):
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="b"))


def test_compatible_schema_change_registers_new_version(lake_root: str | Path) -> None:
    with Writer(lake_root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    class PingV2(Event):
        event_type: ClassVar[str] = "ping"

        source: str
        note: str | None = None

    with Writer(lake_root) as writer:
        writer.write(PingV2(occurred_at=utc(2026, 1, 2), source="b", note="hi"))

    from eventlake.schema import SchemaRegistry

    registry = SchemaRegistry(lake_root)
    assert registry.versions("ping") == [1, 2]


def test_incompatible_schema_change_raises_and_keeps_buffer(lake_root: str | Path) -> None:
    with Writer(lake_root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    class PingBroken(Event):
        event_type: ClassVar[str] = "ping"

        note: str  # `source` removed, `note` is required: breaking change

    writer = Writer(lake_root, allow_breaking=False)
    writer.write(PingBroken(occurred_at=utc(2026, 1, 2), note="hi"))
    with pytest.raises(SchemaChangeError):
        writer.flush()

    from eventlake.schema import SchemaRegistry

    registry = SchemaRegistry(lake_root)
    assert registry.versions("ping") == [1]

    # the buffer must still hold the event: flushing again (now allowed) succeeds
    writer._allow_breaking = True
    writer.flush()
    assert registry.versions("ping") == [1, 2]


def test_allow_breaking_true_accepts_incompatible_change_directly(lake_root: str | Path) -> None:
    with Writer(lake_root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    class PingBroken(Event):
        event_type: ClassVar[str] = "ping"

        note: str

    with Writer(lake_root, allow_breaking=True) as writer:
        writer.write(PingBroken(occurred_at=utc(2026, 1, 2), note="hi"))

    from eventlake.schema import SchemaRegistry

    registry = SchemaRegistry(lake_root)
    assert registry.versions("ping") == [1, 2]


def test_exception_during_write_leaves_no_partial_file(
    lake_root: str | Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    import pyarrow.parquet

    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("disk full")

    # eventlake.storage imports this same module as `pq`; patching the shared
    # module object here affects its `pq.write_table` calls too.
    monkeypatch.setattr(pyarrow.parquet, "write_table", boom)

    writer = Writer(lake_root)
    with pytest.raises(RuntimeError, match="disk full"):
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
        writer.flush()

    # K6: on a local root, the partition directory a failed write created is
    # left in place rather than rmdir'd (see LocalStorage.put_atomic) - only
    # the tmp file, never a valid part-*.parquet, is cleaned up. On S3 the
    # failure happens before anything is sent.
    assert part_keys(lake_root, "ping") == []
    if isinstance(lake_root, Path):
        assert list(lake_root.glob("ping/dt=*/.*.tmp")) == []

    # An empty (or nonexistent) partition dir left behind like this must not
    # stop Lake from opening or from ignoring this (data-less) event type.
    from eventlake.lake import Lake

    lake = Lake(lake_root)
    assert lake.describe() == []


def test_same_event_type_different_classes_v1_then_v2_in_one_writer(lake_root: str | Path) -> None:
    """H1: buffering is per-class, so a V1/V2 mix in one Writer resolves
    through the normal per-event_type schema rules, not a single merged one."""

    class PingV1(Event):
        event_type: ClassVar[str] = "ping"

        source: str

    class PingV2(Event):
        event_type: ClassVar[str] = "ping"

        source: str
        note: str | None = None

    with Writer(lake_root) as writer:
        writer.write(PingV1(occurred_at=utc(2026, 1, 1), source="a"))
        writer.write(PingV2(occurred_at=utc(2026, 1, 2), source="b", note="hi"))

    from eventlake.schema import SchemaRegistry

    registry = SchemaRegistry(lake_root)
    assert registry.versions("ping") == [1, 2]

    from eventlake.lake import Lake

    table = Lake(lake_root).events("ping").to_arrow_table()
    assert table.num_rows == 2
    sources = table.column("source").to_pylist()
    notes = table.column("note").to_pylist()
    by_source = dict(zip(sources, notes, strict=True))
    assert by_source == {"a": None, "b": "hi"}


def test_same_event_type_v1_flushed_after_v2_is_already_latest(lake_root: str | Path) -> None:
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
    with Writer(lake_root) as writer:
        writer.write(PingV1(occurred_at=utc(2026, 1, 1), source="a"))
    with Writer(lake_root) as writer:
        writer.write(PingV2(occurred_at=utc(2026, 1, 2), source="b", note="hi"))

    from eventlake.schema import SchemaRegistry

    registry = SchemaRegistry(lake_root)
    assert registry.versions("ping") == [1, 2]

    # Now flush a V2 event (matches latest trivially) THEN a V1 event
    # (matches the older, non-latest v1) in a single Writer/flush.
    with Writer(lake_root) as writer:
        writer.write(PingV2(occurred_at=utc(2026, 1, 3), source="c", note="yo"))
        writer.write(PingV1(occurred_at=utc(2026, 1, 4), source="d"))

    assert registry.versions("ping") == [1, 2]  # no new version created

    from eventlake.lake import Lake

    table = Lake(lake_root).events("ping").to_arrow_table()
    assert table.num_rows == 4
    assert set(table.column("source").to_pylist()) == {"a", "b", "c", "d"}


def test_same_event_type_different_classes_incompatible_raises(lake_root: str | Path) -> None:
    class PingV1(Event):
        event_type: ClassVar[str] = "ping"

        source: str

    class PingBrokenV2(Event):
        event_type: ClassVar[str] = "ping"

        note: str  # `source` dropped, `note` required: breaking

    # Not a `with` block: exiting it would call flush() again and re-raise,
    # same reasoning as test_incompatible_schema_change_raises_and_keeps_buffer.
    writer = Writer(lake_root)
    writer.write(PingV1(occurred_at=utc(2026, 1, 1), source="a"))
    writer.write(PingBrokenV2(occurred_at=utc(2026, 1, 2), note="hi"))
    with pytest.raises(SchemaChangeError):
        writer.flush()


def test_partial_multiday_flush_failure_does_not_duplicate_on_retry(
    lake_root: str | Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    good_event = Ping(occurred_at=utc(2026, 1, 1), source="a")
    bad_event = Ping(occurred_at=utc(2026, 1, 2), source="b")

    writer = Writer(lake_root)
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

    files_01 = part_keys(lake_root, "ping", "2026-01-01")
    assert len(files_01) == 1

    # The 2026-01-01 event must be gone from the buffer (already written);
    # only the 2026-01-02 event should still be pending.
    remaining = [e for buf in writer._buffers.values() for e in buf]
    assert [e.event_id for e in remaining] == [bad_event.event_id]

    monkeypatch.setattr(Writer, "_write_file", original_write_file)
    writer.flush()

    files_01_after = part_keys(lake_root, "ping", "2026-01-01")
    files_02_after = part_keys(lake_root, "ping", "2026-01-02")
    assert len(files_01_after) == 1  # unchanged - not duplicated
    assert len(files_02_after) == 1


class Tagged(Event):
    event_type: ClassVar[str] = "tagged"
    tags: list[str]


def test_mutating_a_list_after_write_does_not_change_what_is_stored(lake_root: str | Path) -> None:
    event = Tagged(occurred_at=utc(2026, 1, 1), tags=["a"])
    with Writer(lake_root) as writer:
        writer.write(event)
        event.tags.append("mutated-after-write")

    (part,) = part_keys(lake_root, "tagged")
    assert read_table(lake_root, part).column("tags").to_pylist() == [["a"]]


def test_failed_s3_put_leaves_no_partial_object(
    s3_client: object, s3_bucket: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Guarantee 1 on S3: a PutObject that fails midway leaves no object at
    the final key - nothing is ever visible under `part-*.parquet`."""
    from eventlake.storage import S3Storage

    class FailingPuts:
        def __init__(self, inner: object) -> None:
            self._inner = inner

        def put_object(self, **kwargs: Any) -> object:
            if str(kwargs["Key"]).endswith(".parquet"):
                raise ConnectionError("connection reset during upload")
            return self._inner.put_object(**kwargs)  # type: ignore[attr-defined]

        def __getattr__(self, name: str) -> object:
            return getattr(self._inner, name)

    store = S3Storage(s3_bucket, "lake", client=FailingPuts(s3_client))  # type: ignore[arg-type]
    writer = Writer(storage=store)
    writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    with pytest.raises(ConnectionError):
        writer.flush()
    # The schema version (a different PutObject) went through; the Parquet
    # file did not, and nothing partial exists under its key.
    assert store.list_keys("") == ["_schemas/ping/v1.json"]
    # The event is still buffered, so a retry can write it.
    assert len([e for buf in writer._buffers.values() for e in buf]) == 1
