"""eventlake: a small append-only data lake for typed events."""

from .event import Event, UnsupportedFieldType, arrow_schema_for
from .lake import Lake, TypeSummary
from .writer import Writer

__all__ = [
    "Event",
    "UnsupportedFieldType",
    "arrow_schema_for",
    "Lake",
    "TypeSummary",
    "Writer",
]

__version__ = "0.1.0"
