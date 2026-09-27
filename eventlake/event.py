"""The `Event` base class and the mapping from pydantic fields to Arrow types.

Every event written to an eventlake root is an immutable, typed record with
two fields eventlake itself manages (`event_id`, `recorded_at`) plus one the
caller must supply (`occurred_at`). Subclasses add whatever fields describe
the thing that happened and declare a class-level `event_type` name, which
doubles as the on-disk directory name for that kind of event.
"""

from __future__ import annotations

import enum
import re
import types
import uuid
from datetime import UTC, date, datetime
from typing import Any, ClassVar, Union, get_args, get_origin

import pyarrow as pa
from pydantic import BaseModel, ConfigDict, Field, model_validator

_PRIMITIVE_ARROW_TYPES: dict[type, pa.DataType] = {
    str: pa.string(),
    int: pa.int64(),
    float: pa.float64(),
    bool: pa.bool_(),
}

# event_type doubles as an on-disk directory name (and is embedded in SQL
# view names and file globs), so it's restricted to a safe, boring charset:
# lowercase, starts with a letter, no path separators or dots.
EVENT_TYPE_PATTERN = re.compile(r"^[a-z][a-z0-9_]{0,63}$")

# Reserved for eventlake's own internal use - no Event field may start with
# this prefix (checked at class-definition time). RESERVED_SOURCE_FILE_COLUMN
# is the concrete case: Lake and compact() add a synthetic column under this
# name (via DuckDB's read_parquet(..., filename=<name>)) to break dedup ties
# deterministically; a user field named plain "filename" must not collide
# with it, which is why the reserved name isn't just "filename".
RESERVED_FIELD_PREFIX = "__eventlake"
_MANAGED_FIELDS = frozenset({"event_id", "occurred_at", "recorded_at"})
RESERVED_SOURCE_FILE_COLUMN = "__eventlake_source_file"


class UnsupportedFieldType(TypeError):
    """Raised when an Event field has a type with no Arrow mapping."""

    def __init__(self, field_name: str, annotation: Any) -> None:
        self.field_name = field_name
        self.annotation = annotation
        super().__init__(f"field {field_name!r} has unsupported type {annotation!r}")


def validate_event_type(event_type: str) -> None:
    """Raise ValueError if `event_type` doesn't match `EVENT_TYPE_PATTERN`.

    Used both by Event subclasses (at class-definition time, where it's
    surfaced as a TypeError) and by anything else that takes an event_type
    string from outside and uses it to build a filesystem path or SQL
    identifier - `compact()` and the CLI - before that string ever touches
    a path.
    """
    if not EVENT_TYPE_PATTERN.fullmatch(event_type):
        raise ValueError(
            f"invalid event_type {event_type!r}: must match {EVENT_TYPE_PATTERN.pattern!r}"
        )


def _normalize_tz(value: datetime, field_name: str) -> datetime:
    if value.tzinfo is None:
        raise ValueError(f"{field_name} must be timezone-aware")
    return value.astimezone(UTC)


def _unwrap_optional(annotation: Any) -> Any:
    """Return the non-None member of `X | None`, or `annotation` unchanged."""
    origin = get_origin(annotation)
    if _is_optional_union(origin):
        non_none = [a for a in get_args(annotation) if a is not type(None)]
        if len(non_none) == 1:
            return non_none[0]
    return annotation


def _datetime_field_kind(annotation: Any) -> str | None:
    """Classify a field annotation for tz normalization purposes.

    Returns "scalar" for `datetime` (optionally wrapped in `| None`),
    "list" for `list[datetime]` (likewise optionally wrapped), else None.
    """
    unwrapped = _unwrap_optional(annotation)
    if unwrapped is datetime:
        return "scalar"
    origin = get_origin(unwrapped)
    if origin is list:
        args = get_args(unwrapped)
        if len(args) == 1 and args[0] is datetime:
            return "list"
    return None


class Event(BaseModel):
    """Base class for all eventlake events.

    Subclasses must declare a class variable ``event_type: ClassVar[str]``
    matching `EVENT_TYPE_PATTERN`. Instances are frozen (immutable) and
    reject unknown fields. Every `datetime` field (including ones a
    subclass adds) must be timezone-aware; values are normalized to UTC.
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
        try:
            validate_event_type(declared)
        except ValueError as exc:
            raise TypeError(f"{cls.__name__}.{exc}") from exc

    @classmethod
    def __pydantic_init_subclass__(cls, **kwargs: Any) -> None:
        """Fail at class-definition time, not at first write.

        `model_fields` is only fully resolved once pydantic has finished
        building the subclass, which is exactly what this hook (unlike
        plain `__init_subclass__`) guarantees has already happened.
        """
        super().__pydantic_init_subclass__(**kwargs)
        # Checked against the class's own raw __annotations__, not
        # model_fields: pydantic already excludes any leading-underscore
        # name from model_fields entirely (treating it as private, not a
        # field), so a literal `__eventlake_foo: str` in a class body can
        # never actually reach model_fields - but this guards the same name
        # arriving through other means (e.g. dynamic class construction)
        # and documents the reserved prefix as an explicit contract rather
        # than relying on that pydantic behavior as an implementation detail.
        own_annotations = cls.__dict__.get("__annotations__", {})
        # event_id / occurred_at / recorded_at drive deduplication, partitioning
        # and state_as_of ordering. A subclass that redefines one (say
        # `event_id: str | None = None`) would make distinct events look like
        # duplicates and silently drop them.
        for name in _MANAGED_FIELDS & set(own_annotations):
            raise TypeError(
                f"{cls.__name__}.{name}: {sorted(_MANAGED_FIELDS)} are defined by Event "
                f"and cannot be redefined in a subclass"
            )
        for name in own_annotations:
            if name.startswith(RESERVED_FIELD_PREFIX):
                raise TypeError(
                    f"{cls.__name__}.{name}: field names starting with "
                    f"{RESERVED_FIELD_PREFIX!r} are reserved for eventlake's own use"
                )
        arrow_schema_for(cls)

    @model_validator(mode="after")
    def _normalize_all_datetime_fields(self) -> Event:
        """Require tz-aware datetimes and normalize them to UTC.

        Applies to `occurred_at`, `recorded_at`, any datetime field a
        subclass declares, and every element of a `list[datetime]` field -
        not just the two eventlake owns. Uses `object.__setattr__` to update
        the field in place despite the model being frozen; this runs during
        construction, before the instance is handed back to the caller.
        """
        for name, field_info in type(self).model_fields.items():
            kind = _datetime_field_kind(field_info.annotation)
            if kind is None:
                continue
            value = getattr(self, name)
            if value is None:
                continue
            if kind == "scalar":
                object.__setattr__(self, name, _normalize_tz(value, name))
            else:
                normalized_list = [
                    _normalize_tz(item, f"{name}[{i}]") for i, item in enumerate(value)
                ]
                object.__setattr__(self, name, normalized_list)
        return self


def _primitive_arrow_type(field_name: str, annotation: Any) -> pa.DataType:
    """Map a scalar (non-container) annotation to an Arrow type.

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
        # Every scalar type we support standalone is also allowed as a list
        # item (UUID/Enum stored as strings, same as when used directly).
        item_type = _primitive_arrow_type(field_name, item_args[0])
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
