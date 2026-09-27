from __future__ import annotations

import uuid
from datetime import datetime, timedelta
from pathlib import Path
from typing import ClassVar

import pytest
from _helpers import utc

from eventlake.event import Event
from eventlake.lake import Lake
from eventlake.writer import Writer


class Ping(Event):
    event_type: ClassVar[str] = "ping"

    source: str


class Reading(Event):
    event_type: ClassVar[str] = "reading"

    sensor_id: str
    value: float


def test_events_returns_all_rows(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
        writer.write(Ping(occurred_at=utc(2026, 1, 2), source="b"))

    lake = Lake(tmp_path)
    table = lake.events("ping").to_arrow_table()
    assert table.num_rows == 2
    assert set(table.column("source").to_pylist()) == {"a", "b"}


def test_fetchall_returns_timezone_aware_utc(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    rows = Lake(tmp_path).sql("SELECT occurred_at FROM ping").fetchall()
    assert len(rows) == 1
    occurred_at = rows[0][0]
    assert occurred_at.utcoffset() == timedelta(0)
    assert occurred_at == utc(2026, 1, 1)


def test_events_dedup_keeps_earliest_recorded_at(tmp_path: Path) -> None:
    event_id = uuid.uuid4()
    # Two separate Writer instances processing "the same" event: simulates a
    # retried job re-emitting the identical event_id.
    with Writer(tmp_path) as writer:
        writer.write(Ping(event_id=event_id, occurred_at=utc(2026, 1, 1), source="first"))

    import time

    time.sleep(0.01)

    with Writer(tmp_path) as writer:
        writer.write(Ping(event_id=event_id, occurred_at=utc(2026, 1, 1), source="second"))

    lake = Lake(tmp_path)
    table = lake.events("ping").to_arrow_table()
    assert table.num_rows == 1
    assert table.column("source").to_pylist() == ["first"]


def test_events_since_until_prunes_to_requested_partitions(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="d1"))
        writer.write(Ping(occurred_at=utc(2026, 1, 5), source="d5"))
        writer.write(Ping(occurred_at=utc(2026, 1, 10), source="d10"))

    lake = Lake(tmp_path)

    only_middle = lake.events("ping", since=utc(2026, 1, 3), until=utc(2026, 1, 7)).to_arrow_table()
    assert only_middle.column("source").to_pylist() == ["d5"]

    since_only = lake.events("ping", since=utc(2026, 1, 5)).to_arrow_table()
    assert set(since_only.column("source").to_pylist()) == {"d5", "d10"}

    until_only = lake.events("ping", until=utc(2026, 1, 5)).to_arrow_table()
    assert set(until_only.column("source").to_pylist()) == {"d1", "d5"}


def test_partition_files_only_lists_matching_dt_directories(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="d1"))
        writer.write(Ping(occurred_at=utc(2026, 1, 5), source="d5"))
        writer.write(Ping(occurred_at=utc(2026, 1, 10), source="d10"))

    lake = Lake(tmp_path)
    files = lake._partition_files("ping", since=utc(2026, 1, 3), until=utc(2026, 1, 7))
    assert len(files) == 1
    assert "dt=2026-01-05" in files[0]


def test_events_on_unknown_event_type_raises(tmp_path: Path) -> None:
    lake = Lake(tmp_path)
    with pytest.raises(ValueError, match="unknown event_type"):
        lake.events("nonexistent")


def test_events_no_rows_in_range_returns_empty_but_typed(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    lake = Lake(tmp_path)
    table = lake.events("ping", since=utc(2030, 1, 1)).to_arrow_table()
    assert table.num_rows == 0
    assert table.column_names == ["event_id", "occurred_at", "recorded_at", "source"]


def test_sql_exposes_each_event_type_as_a_view(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
        writer.write(Reading(occurred_at=utc(2026, 1, 1), sensor_id="s1", value=1.5))

    lake = Lake(tmp_path)
    count = lake.sql("SELECT COUNT(*) FROM ping").fetchone()
    assert count == (1,)
    joined = lake.sql(
        "SELECT (SELECT COUNT(*) FROM ping) + (SELECT COUNT(*) FROM reading) AS total"
    ).fetchone()
    assert joined == (2,)


def test_state_as_of_picks_latest_occurred_at_per_key(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Reading(occurred_at=utc(2026, 1, 1), sensor_id="s1", value=1.0))
        writer.write(Reading(occurred_at=utc(2026, 1, 2), sensor_id="s1", value=2.0))
        writer.write(Reading(occurred_at=utc(2026, 1, 1), sensor_id="s2", value=10.0))

    lake = Lake(tmp_path)
    result = lake.state_as_of("reading", key="sensor_id").to_arrow_table()
    sensor_ids = result.column("sensor_id").to_pylist()
    values = result.column("value").to_pylist()
    by_sensor = dict(zip(sensor_ids, values, strict=True))
    assert by_sensor == {"s1": 2.0, "s2": 10.0}


def test_state_as_of_respects_at_cutoff(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Reading(occurred_at=utc(2026, 1, 1), sensor_id="s1", value=1.0))
        writer.write(Reading(occurred_at=utc(2026, 1, 5), sensor_id="s1", value=5.0))

    lake = Lake(tmp_path)
    result = lake.state_as_of("reading", key="sensor_id", at=utc(2026, 1, 3)).to_arrow_table()
    assert result.column("value").to_pylist() == [1.0]


def test_state_as_of_future_at_returns_latest_overall(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Reading(occurred_at=utc(2026, 1, 1), sensor_id="s1", value=1.0))
        writer.write(Reading(occurred_at=utc(2026, 1, 5), sensor_id="s1", value=5.0))

    lake = Lake(tmp_path)
    result = lake.state_as_of("reading", key="sensor_id", at=utc(2099, 1, 1)).to_arrow_table()
    assert result.column("value").to_pylist() == [5.0]


def test_state_as_of_before_any_data_is_empty(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Reading(occurred_at=utc(2026, 1, 5), sensor_id="s1", value=5.0))

    lake = Lake(tmp_path)
    result = lake.state_as_of("reading", key="sensor_id", at=utc(2020, 1, 1)).to_arrow_table()
    assert result.num_rows == 0


def test_state_as_of_same_occurred_at_breaks_tie_by_recorded_at_then_event_id(
    tmp_path: Path,
) -> None:
    occurred = utc(2026, 1, 1)
    lower_id = uuid.UUID(int=1)
    higher_id = uuid.UUID(int=2)

    # Two events with the identical occurred_at for the same key: the one
    # with the later recorded_at should win, regardless of event_id order.
    with Writer(tmp_path) as writer:
        writer.write(Reading(event_id=lower_id, occurred_at=occurred, sensor_id="s1", value=1.0))

    import time

    time.sleep(0.01)

    with Writer(tmp_path) as writer:
        writer.write(Reading(event_id=higher_id, occurred_at=occurred, sensor_id="s1", value=2.0))

    lake = Lake(tmp_path)
    result = lake.state_as_of("reading", key="sensor_id").to_arrow_table()
    assert result.column("value").to_pylist() == [2.0]


def test_describe_reports_versions_partitions_files_rows(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
        writer.write(Ping(occurred_at=utc(2026, 1, 2), source="b"))
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 2), source="c"))

    lake = Lake(tmp_path)
    summaries = {s.event_type: s for s in lake.describe()}
    ping_summary = summaries["ping"]
    assert ping_summary.schema_versions == 1
    assert ping_summary.partitions == 2
    assert ping_summary.files == 3  # two Writer instances -> 3 flush files total
    assert ping_summary.rows == 3


def test_describe_on_empty_root_returns_empty_list(tmp_path: Path) -> None:
    lake = Lake(tmp_path / "does-not-exist")
    assert lake.describe() == []


# --- M1: Lake ignores empty dirs / names that don't look like event types --


def test_lake_opens_and_ignores_empty_and_invalid_directories(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    (tmp_path / "empty_type").mkdir()  # valid name, but no files yet
    (tmp_path / "Invalid-Name").mkdir()  # fails EVENT_TYPE_PATTERN
    (tmp_path / "empty_type" / "dt=2026-01-01").mkdir(parents=True)  # dir but no parquet file

    lake = Lake(tmp_path)  # must not raise
    assert {s.event_type for s in lake.describe()} == {"ping"}


def test_events_raises_for_naive_since_and_until(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    lake = Lake(tmp_path)
    with pytest.raises(ValueError, match="since"):
        lake.events("ping", since=datetime(2026, 1, 1))  # naive, no tzinfo
    with pytest.raises(ValueError, match="until"):
        lake.events("ping", until=datetime(2026, 1, 1))


def test_state_as_of_raises_for_naive_at(tmp_path: Path) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Reading(occurred_at=utc(2026, 1, 1), sensor_id="s1", value=1.0))

    lake = Lake(tmp_path)
    with pytest.raises(ValueError, match="at"):
        lake.state_as_of("reading", key="sensor_id", at=datetime(2026, 1, 1))


def test_events_dedup_tiebreak_by_filename_when_recorded_at_ties(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M3: when recorded_at ties exactly, dedup falls back to the source
    parquet filename (event_id can't help - it's constant within the
    PARTITION BY event_id window)."""
    import eventlake.writer as writer_module

    event_id = uuid.uuid4()
    fixed_recorded = utc(2026, 6, 1, 12)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> FrozenDatetime:
            return cls.fromisoformat(fixed_recorded.isoformat())

    monkeypatch.setattr(writer_module, "datetime", FrozenDatetime)

    with Writer(tmp_path) as writer:
        writer.write(Ping(event_id=event_id, occurred_at=utc(2026, 1, 1), source="first"))
    with Writer(tmp_path) as writer:
        writer.write(Ping(event_id=event_id, occurred_at=utc(2026, 1, 1), source="second"))

    lake = Lake(tmp_path)
    table = lake.events("ping").to_arrow_table()
    assert table.num_rows == 1
    # Deterministic (whichever filename sorts first), and repeatable.
    winner = table.column("source").to_pylist()[0]
    assert winner in {"first", "second"}
    table_again = Lake(tmp_path).events("ping").to_arrow_table()
    assert table_again.column("source").to_pylist()[0] == winner


def test_state_as_of_tie_break_by_event_id_when_occurred_and_recorded_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """M3: state_as_of's own tiebreak (occurred_at, recorded_at, event_id)
    is exercised even when both occurred_at AND recorded_at are identical -
    only event_id is left to decide."""
    import eventlake.writer as writer_module

    fixed_recorded = utc(2026, 6, 1, 12)

    class FrozenDatetime(datetime):
        @classmethod
        def now(cls, tz: object = None) -> FrozenDatetime:
            return cls.fromisoformat(fixed_recorded.isoformat())

    monkeypatch.setattr(writer_module, "datetime", FrozenDatetime)

    occurred = utc(2026, 1, 1)
    low_id = uuid.UUID(int=1)
    high_id = uuid.UUID(int=2)

    with Writer(tmp_path) as writer:
        writer.write(Reading(event_id=low_id, occurred_at=occurred, sensor_id="s1", value=1.0))
        writer.write(Reading(event_id=high_id, occurred_at=occurred, sensor_id="s1", value=2.0))

    lake = Lake(tmp_path)
    result = lake.state_as_of("reading", key="sensor_id").to_arrow_table()
    # occurred_at and recorded_at both tie exactly; event_id DESC decides.
    assert result.column("value").to_pylist() == [2.0]
