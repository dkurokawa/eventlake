from __future__ import annotations

import enum
import uuid
from datetime import UTC, date, datetime, timedelta, timezone
from typing import ClassVar

import pyarrow as pa
import pytest
from pydantic import ValidationError

from eventlake.event import (
    EVENT_TYPE_PATTERN,
    Event,
    UnsupportedFieldType,
    arrow_schema_for,
    event_to_record,
    validate_event_type,
)


class Color(enum.Enum):
    RED = "red"
    BLUE = "blue"


class Kitchen(Event):
    event_type: ClassVar[str] = "kitchen"

    name: str
    quantity: int
    price: float
    in_stock: bool
    expires_on: date
    batch_id: uuid.UUID
    color: Color
    tags: list[str]
    note: str | None = None


def make_kitchen(**overrides: object) -> Kitchen:
    defaults: dict[str, object] = dict(
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        name="apple",
        quantity=3,
        price=1.5,
        in_stock=True,
        expires_on=date(2026, 6, 1),
        batch_id=uuid.uuid4(),
        color=Color.RED,
        tags=["fruit", "fresh"],
    )
    defaults.update(overrides)
    return Kitchen(**defaults)


def test_event_type_required() -> None:
    with pytest.raises(TypeError, match="event_type"):

        class Missing(Event):
            pass


def test_event_type_must_be_str() -> None:
    with pytest.raises(TypeError, match="must be a str"):

        class NotStr(Event):
            event_type: ClassVar[str] = 123  # type: ignore[assignment]


@pytest.mark.parametrize(
    "bad_event_type",
    [
        "",
        "../x",
        "/abs/path",
        "_schemas",
        "Kitchen",
        "kitchen-event",
        "kitchen.event",
        "1kitchen",
        "a" * 65,
    ],
)
def test_event_type_pattern_rejects_unsafe_names(bad_event_type: str) -> None:
    with pytest.raises(TypeError, match="invalid event_type"):
        type(
            "BadEventType",
            (Event,),
            {"__annotations__": {"event_type": ClassVar[str]}, "event_type": bad_event_type},
        )


def test_event_type_pattern_accepts_boundary_names() -> None:
    assert EVENT_TYPE_PATTERN.fullmatch("a")
    assert EVENT_TYPE_PATTERN.fullmatch("a" * 64)
    assert EVENT_TYPE_PATTERN.fullmatch("order_placed_v2")
    validate_event_type("order_placed_v2")  # does not raise


def test_validate_event_type_raises_value_error_directly() -> None:
    with pytest.raises(ValueError, match="invalid event_type"):
        validate_event_type("../etc")


def test_occurred_at_requires_tz() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_kitchen(occurred_at=datetime(2026, 1, 1))


def test_occurred_at_normalized_to_utc() -> None:
    jst = timezone(timedelta(hours=9))
    event = make_kitchen(occurred_at=datetime(2026, 1, 1, 9, 0, tzinfo=jst))
    assert event.occurred_at == datetime(2026, 1, 1, 0, 0, tzinfo=UTC)
    assert event.occurred_at.tzinfo == UTC


def test_recorded_at_requires_tz_when_set() -> None:
    with pytest.raises(ValidationError, match="timezone-aware"):
        make_kitchen(recorded_at=datetime(2026, 1, 1))


def test_recorded_at_defaults_to_none() -> None:
    event = make_kitchen()
    assert event.recorded_at is None


def test_user_defined_datetime_field_requires_tz_and_normalizes() -> None:
    class Scheduled(Event):
        event_type: ClassVar[str] = "scheduled"

        run_at: datetime
        cancelled_at: datetime | None = None

    with pytest.raises(ValidationError, match="timezone-aware"):
        Scheduled(occurred_at=datetime(2026, 1, 1, tzinfo=UTC), run_at=datetime(2026, 1, 2))

    jst = timezone(timedelta(hours=9))
    event = Scheduled(
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        run_at=datetime(2026, 1, 2, 9, 0, tzinfo=jst),
        cancelled_at=datetime(2026, 1, 2, 10, 0, tzinfo=jst),
    )
    assert event.run_at == datetime(2026, 1, 2, 0, 0, tzinfo=UTC)
    assert event.run_at.tzinfo == UTC
    assert event.cancelled_at == datetime(2026, 1, 2, 1, 0, tzinfo=UTC)

    # Optional datetime field left as None must stay None, not raise.
    event2 = Scheduled(
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        run_at=datetime(2026, 1, 2, tzinfo=UTC),
    )
    assert event2.cancelled_at is None


def test_list_of_datetime_requires_tz_on_every_element_and_normalizes() -> None:
    class Timeline(Event):
        event_type: ClassVar[str] = "timeline"

        moments: list[datetime]
        maybe_moments: list[datetime] | None = None

    with pytest.raises(ValidationError, match="timezone-aware"):
        Timeline(
            occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
            moments=[datetime(2026, 1, 1, tzinfo=UTC), datetime(2026, 1, 2)],  # 2nd naive
        )

    jst = timezone(timedelta(hours=9))
    event = Timeline(
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        moments=[datetime(2026, 1, 2, 9, 0, tzinfo=jst), datetime(2026, 1, 3, tzinfo=UTC)],
        maybe_moments=[datetime(2026, 1, 4, 9, 0, tzinfo=jst)],
    )
    assert event.moments == [
        datetime(2026, 1, 2, 0, 0, tzinfo=UTC),
        datetime(2026, 1, 3, 0, 0, tzinfo=UTC),
    ]
    assert all(m.tzinfo == UTC for m in event.moments)
    assert event.maybe_moments == [datetime(2026, 1, 4, 0, 0, tzinfo=UTC)]

    # Optional list left as None must stay None, not raise.
    event2 = Timeline(
        occurred_at=datetime(2026, 1, 1, tzinfo=UTC),
        moments=[datetime(2026, 1, 1, tzinfo=UTC)],
    )
    assert event2.maybe_moments is None


def test_reserved_field_prefix_rejected_at_class_definition_time() -> None:
    with pytest.raises(TypeError, match="reserved for eventlake's own use"):
        type(
            "BadReservedField",
            (Event,),
            {
                "__annotations__": {
                    "event_type": ClassVar[str],
                    "__eventlake_source_file": str,
                },
                "event_type": "bad_reserved_field",
            },
        )


def test_event_id_defaults_to_uuid4() -> None:
    a = make_kitchen()
    b = make_kitchen()
    assert isinstance(a.event_id, uuid.UUID)
    assert a.event_id != b.event_id


def test_extra_fields_forbidden() -> None:
    with pytest.raises(ValidationError):
        make_kitchen(unexpected_field="nope")


def test_events_are_frozen() -> None:
    event = make_kitchen()
    with pytest.raises(ValidationError):
        event.name = "banana"  # type: ignore[misc]


def test_arrow_schema_for_maps_supported_types() -> None:
    schema = arrow_schema_for(Kitchen)
    by_name = {f.name: f for f in schema}

    assert by_name["event_id"].type.equals(pa.string())
    assert not by_name["event_id"].nullable

    assert by_name["occurred_at"].type.equals(pa.timestamp("us", tz="UTC"))
    assert not by_name["occurred_at"].nullable

    assert by_name["recorded_at"].type.equals(pa.timestamp("us", tz="UTC"))
    assert by_name["recorded_at"].nullable

    assert by_name["name"].type.equals(pa.string())
    assert by_name["quantity"].type.equals(pa.int64())
    assert by_name["price"].type.equals(pa.float64())
    assert by_name["in_stock"].type.equals(pa.bool_())
    assert by_name["expires_on"].type.equals(pa.date32())
    assert by_name["batch_id"].type.equals(pa.string())
    assert by_name["color"].type.equals(pa.string())
    assert by_name["tags"].type.equals(pa.list_(pa.string()))

    assert by_name["note"].type.equals(pa.string())
    assert by_name["note"].nullable


def test_arrow_schema_field_order_matches_declaration() -> None:
    schema = arrow_schema_for(Kitchen)
    names = [f.name for f in schema]
    assert names[:3] == ["event_id", "occurred_at", "recorded_at"]
    assert names[3:] == [
        "name",
        "quantity",
        "price",
        "in_stock",
        "expires_on",
        "batch_id",
        "color",
        "tags",
        "note",
    ]


def test_list_supports_every_scalar_type_as_uuid_and_enum_become_strings() -> None:
    class Batch(Event):
        event_type: ClassVar[str] = "batch"

        ids: list[uuid.UUID]
        colors: list[Color]
        dates: list[date]
        moments: list[datetime]
        counts: list[int]

    schema = arrow_schema_for(Batch)
    by_name = {f.name: f for f in schema}
    assert by_name["ids"].type.equals(pa.list_(pa.string()))
    assert by_name["colors"].type.equals(pa.list_(pa.string()))
    assert by_name["dates"].type.equals(pa.list_(pa.date32()))
    assert by_name["moments"].type.equals(pa.list_(pa.timestamp("us", tz="UTC")))
    assert by_name["counts"].type.equals(pa.list_(pa.int64()))


def test_unsupported_field_type_dict() -> None:
    with pytest.raises(UnsupportedFieldType) as exc_info:

        class BadDict(Event):
            event_type: ClassVar[str] = "bad_dict"

            payload: dict[str, str]

    assert exc_info.value.field_name == "payload"


def test_unsupported_field_type_is_raised_at_class_definition_time() -> None:
    # No separate arrow_schema_for() call needed: the class statement itself
    # must raise, per __pydantic_init_subclass__.
    with pytest.raises(UnsupportedFieldType):

        class BadAtDefinition(Event):
            event_type: ClassVar[str] = "bad_at_definition"

            payload: dict[str, str]


def test_unsupported_field_type_nested_model() -> None:
    class Inner(Event):
        event_type: ClassVar[str] = "inner"

        x: int

    with pytest.raises(UnsupportedFieldType):

        class BadNested(Event):
            event_type: ClassVar[str] = "bad_nested"

            inner: Inner


def test_unsupported_field_type_list_of_non_primitive() -> None:
    with pytest.raises(UnsupportedFieldType):

        class BadListItem(Event):
            event_type: ClassVar[str] = "bad_list_item"

            groups: list[list[str]]


def test_unsupported_field_type_union_of_two_real_types() -> None:
    with pytest.raises(UnsupportedFieldType):

        class BadUnion(Event):
            event_type: ClassVar[str] = "bad_union"

            value: int | str


def test_event_to_record_converts_uuid_enum_and_list() -> None:
    batch_id = uuid.uuid4()
    event = make_kitchen(batch_id=batch_id, color=Color.BLUE, tags=["a", "b"])
    record = event_to_record(event)

    assert record["batch_id"] == str(batch_id)
    assert record["color"] == "blue"
    assert record["tags"] == ["a", "b"]
    assert record["name"] == "apple"
    assert record["note"] is None


@pytest.mark.parametrize("field", ["event_id", "occurred_at", "recorded_at"])
def test_managed_fields_cannot_be_redefined(field: str) -> None:
    with pytest.raises(TypeError, match="cannot be redefined"):
        type(
            "Redefines",
            (Event,),
            {
                "__annotations__": {field: "str | None", "event_type": "ClassVar[str]"},
                field: None,
                "event_type": "redefines",
                "__module__": __name__,
            },
        )
