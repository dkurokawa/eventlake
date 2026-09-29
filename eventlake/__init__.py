"""eventlake: a small append-only data lake for typed events."""

from .event import Event, UnsupportedFieldType, arrow_schema_for
from .lake import Lake, TypeSummary
from .storage import LocalStorage, Storage, open_storage
from .writer import Writer

__all__ = [
    "Event",
    "UnsupportedFieldType",
    "arrow_schema_for",
    "LocalStorage",
    "Lake",
    "Storage",
    "TypeSummary",
    "Writer",
    "open_storage",
]

__version__ = "0.1.0"
