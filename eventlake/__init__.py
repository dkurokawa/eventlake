"""eventlake: a small append-only data lake for typed events."""

from .event import Event, UnsupportedFieldType, arrow_schema_for

__all__ = [
    "Event",
    "UnsupportedFieldType",
    "arrow_schema_for",
]

__version__ = "0.1.0"
