"""The `Event` base class and the mapping from pydantic fields to Arrow types.

Every event written to an eventlake root is an immutable, typed record with
two fields eventlake itself manages (`event_id`, `recorded_at`) plus one the
caller must supply (`occurred_at`). Subclasses add whatever fields describe
the thing that happened and declare a class-level `event_type` name, which
doubles as the on-disk directory name for that kind of event.
"""

from __future__ import annotations

import enum
import types
import uuid
from datetime import UTC, date, datetime
from typing import Any, ClassVar, Union, get_args, get_origin

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, field_validator

_PRIMITIVE_ARROW_TYPES: dict[type, pa.DataType] = {
    str: pa.string(),
    int: pa.int64(),
    float: pa.float64(),
    bool: pa.bool_(),
}


class UnsupportedFieldType(TypeError):
    """Raised when an Event field has a type with no Arrow mapping."""

    def __init__(self, field_name: str, annotation: Any) -> None:
        self.field_name = field_name
        self.annotation = annotation
        super().__init__(f"field {field_name!r} has unsupported type {annotation!r}")


def _normalize_tz(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


class Event(BaseModel):
    """Base class for all eventlake events.

    Subclasses must declare a class variable ``event_type: ClassVar[str]``.
    Instances are frozen (immutable) and reject unknown fields.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    event_id: uuid.UUID = Field(default_factory=uuid.uuid4)
    occurred_at: datetime
    recorded_at: datetime | None = None

    event_type: ClassVar[str]

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        declared = cls.__dict__.get("event_type")
        if declared is None:
            raise TypeError(
                f"{cls.__name__} must define a class variable 'event_type: ClassVar[str]'"
            )
        if not isinstance(declared, str):
            raise TypeError(f"{cls.__name__}.event_type must be a str, got {declared!r}")

    @field_validator("occurred_at")
    @classmethod
    def _validate_occurred_at(cls, value: datetime) -> datetime:
        return _normalize_tz(value, "occurred_at")

    @field_validator("recorded_at")
    @classmethod
    def _validate_recorded_at(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _normalize_tz(value, "recorded_at")


def _primitive_arrow_type(field_name: str, annotation: Any) -> pa.DataType:
    """Map a primitive (non-container) annotation to an Arrow type.

    Raises UnsupportedFieldType for anything not in the supported set:
    str / int / float / bool / datetime / date / UUID / Enum subclass.
    """
    if annotation in _PRIMITIVE_ARROW_TYPES:
        return _PRIMITIVE_ARROW_TYPES[annotation]
    if annotation is uuid.UUID:
        return pa.string()
    if annotation is datetime:
        return pa.timestamp("us", tz="UTC")
    if annotation is date:
        return pa.date32()
    if isinstance(annotation, type) and issubclass(annotation, enum.Enum):
        return pa.string()
    raise UnsupportedFieldType(field_name, annotation)


def _list_item_arrow_type(field_name: str, annotation: Any) -> pa.DataType:
    """Map a `list[...]` item annotation. Only plain primitives are allowed."""
    if annotation in _PRIMITIVE_ARROW_TYPES:
        return _PRIMITIVE_ARROW_TYPES[annotation]
    raise UnsupportedFieldType(field_name, list[annotation])


def _is_optional_union(origin: Any) -> bool:
    return origin is Union or origin is types.UnionType


def _field_arrow_type(field_name: str, annotation: Any) -> tuple[pa.DataType, bool]:
    """Return (arrow_type, nullable) for a single pydantic field annotation."""
    origin = get_origin(annotation)
    nullable = False
    working = annotation

    if _is_optional_union(origin):
        args = get_args(working)
        non_none = [a for a in args if a is not type(None)]
        if len(args) != 2 or len(non_none) != 1:
            raise UnsupportedFieldType(field_name, annotation)
        nullable = True
        working = non_none[0]
        origin = get_origin(working)

    if origin is list:
        item_args = get_args(working)
        if len(item_args) != 1:
            raise UnsupportedFieldType(field_name, annotation)
        item_type = _list_item_arrow_type(field_name, item_args[0])
        return pa.list_(item_type), nullable

    return _primitive_arrow_type(field_name, working), nullable


def arrow_schema_for(cls: type[Event]) -> pa.Schema:
    """Compute the Arrow schema for an Event subclass from its pydantic fields.

    Raises UnsupportedFieldType if any field has a type with no Arrow mapping.
    """
    fields: list[pa.Field] = []
    for name, field_info in cls.model_fields.items():
        arrow_type, nullable = _field_arrow_type(name, field_info.annotation)
        fields.append(pa.field(name, arrow_type, nullable=nullable))
    return pa.schema(fields)


def _convert_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, uuid.UUID):
        return str(value)
    if isinstance(value, enum.Enum):
        return str(value.value)
    if isinstance(value, list):
        return [_convert_value(item) for item in value]
    return value


def event_to_record(event: Event) -> dict[str, Any]:
    """Convert an Event instance into a dict of Arrow-compatible values."""
    data = event.model_dump(mode="python")
    return {name: _convert_value(value) for name, value in data.items()}
