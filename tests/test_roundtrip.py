"""Hypothesis property test: every supported field type round-trips exactly
through Writer -> Parquet -> Lake."""

from __future__ import annotations

import enum
import tempfile
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import ClassVar

from hypothesis import given, settings
from hypothesis import strategies as st

from eventlake.event import Event
from eventlake.lake import Lake
from eventlake.writer import Writer


class Flavor(enum.Enum):
    SWEET = "sweet"
    SOUR = "sour"


class Sample(Event):
    event_type: ClassVar[str] = "sample"

    name: str
    count: int
    ratio: float
    active: bool
    on_date: date
    batch_id: uuid.UUID
    flavor: Flavor
    tags: list[str]
    note: str | None


_aware_datetimes = st.datetimes(
    min_value=datetime(2000, 1, 1),
    max_value=datetime(2099, 12, 31),
).map(lambda dt: dt.replace(tzinfo=UTC))

_safe_text = st.text(max_size=15)


@st.composite
def _samples(draw: st.DrawFn) -> Sample:
    return Sample(
        occurred_at=draw(_aware_datetimes),
        name=draw(_safe_text),
        count=draw(st.integers(min_value=-1_000_000, max_value=1_000_000)),
        ratio=draw(
            st.floats(
                allow_nan=False,
                allow_infinity=False,
                allow_subnormal=False,
                min_value=-1e12,
                max_value=1e12,
            )
        ),
        active=draw(st.booleans()),
        on_date=draw(st.dates(min_value=date(2000, 1, 1), max_value=date(2099, 12, 31))),
        batch_id=draw(st.uuids()),
        flavor=draw(st.sampled_from(list(Flavor))),
        tags=draw(st.lists(_safe_text, max_size=5)),
        note=draw(st.one_of(st.none(), _safe_text)),
    )


def _assert_round_trip(root: str | Path, sample: Sample) -> None:
    with Writer(root) as writer:
        writer.write(sample)

    lake = Lake(root)
    table = lake.events("sample").to_arrow_table()
    assert table.num_rows == 1
    row = table.to_pylist()[0]

    assert row["event_id"] == str(sample.event_id)
    assert row["occurred_at"] == sample.occurred_at
    assert row["name"] == sample.name
    assert row["count"] == sample.count
    assert row["ratio"] == sample.ratio
    assert row["active"] == sample.active
    assert row["on_date"] == sample.on_date
    assert row["batch_id"] == str(sample.batch_id)
    assert row["flavor"] == sample.flavor.value
    assert row["tags"] == sample.tags
    assert row["note"] == sample.note


# Local only: 25 examples against an S3 server would make the suite slow for
# no extra coverage - the value handling is the same on both backends. One
# fixed case below runs the same assertions against S3.
@given(_samples())
@settings(max_examples=25, deadline=None)
def test_property_round_trip_preserves_values(sample: Sample) -> None:
    with tempfile.TemporaryDirectory() as tmp_dir:
        _assert_round_trip(Path(tmp_dir), sample)


def test_round_trip_preserves_every_field_type(lake_root: str | Path) -> None:
    _assert_round_trip(
        lake_root,
        Sample(
            occurred_at=datetime(2026, 3, 4, 5, 6, 7, 890123, tzinfo=UTC),
            name="naïve ✓",
            count=-7,
            ratio=0.1,
            active=True,
            on_date=date(2026, 3, 4),
            batch_id=uuid.uuid4(),
            flavor=Flavor.SOUR,
            tags=["x", "y"],
            note=None,
        ),
    )
