"""Buffering and flushing events to partitioned Parquet files."""

from __future__ import annotations

import contextlib
import uuid
from collections import defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType

import pyarrow as pa
import pyarrow.parquet as pq

from .event import Event, arrow_schema_for, event_to_record
from .schema import SchemaRegistry

DEFAULT_MAX_ROWS = 10_000


class Writer:
    """Buffers events per event *class* and flushes them to Parquet.

    Buffering is keyed by the Python class, not by `event_type`: if two
    classes share an event_type (e.g. a V1 and a V2 of the same event,
    migrated mid-process), each is flushed with its own class's Arrow
    schema, and schema compatibility between them is resolved through the
    normal SchemaRegistry rules (a compatible V2 becomes a new version; an
    incompatible one raises unless allow_breaking is set).

    Use as a context manager so buffered events are flushed on exit::

        with Writer(root) as writer:
            writer.write(event)
    """

    def __init__(
        self,
        root: str | Path,
        *,
        max_rows: int = DEFAULT_MAX_ROWS,
        allow_breaking: bool = False,
    ) -> None:
        self._root = Path(root)
        self._max_rows = max_rows
        self._allow_breaking = allow_breaking
        self._registry = SchemaRegistry(self._root)
        self._buffers: dict[type[Event], list[Event]] = defaultdict(list)
        self._seen_event_ids: set[uuid.UUID] = set()
        self._closed = False

    def __enter__(self) -> Writer:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        if exc is None:
            self.flush()
        self._closed = True

    def write(self, event: Event) -> None:
        self.write_many([event])

    def write_many(self, events: Sequence[Event]) -> None:
        if self._closed:
            raise RuntimeError("writer is closed")
        for event in events:
            if event.event_id in self._seen_event_ids:
                continue
            self._seen_event_ids.add(event.event_id)
            recorded = event.model_copy(update={"recorded_at": datetime.now(UTC)})
            cls = type(recorded)
            self._buffers[cls].append(recorded)
            if len(self._buffers[cls]) >= self._max_rows:
                self._flush_class(cls)

    def flush(self) -> None:
        for cls in list(self._buffers.keys()):
            self._flush_class(cls)

    def _flush_class(self, cls: type[Event]) -> None:
        buffer = self._buffers.get(cls)
        if not buffer:
            return
        event_type = cls.event_type
        schema = arrow_schema_for(cls)
        # May raise SchemaChangeError; buffer stays intact if it does.
        self._registry.register(event_type, schema, allow_breaking=self._allow_breaking)
        self._write_partitions(cls, event_type, schema, buffer)

    def _write_partitions(
        self, cls: type[Event], event_type: str, schema: pa.Schema, events: list[Event]
    ) -> None:
        """Write one file per occurred_at day, removing each day's events
        from the buffer as soon as its file is safely on disk.

        If writing a later day's partition fails, the earlier days already
        written are gone from the buffer - a retry (another flush() call)
        only re-attempts what didn't make it, instead of writing duplicate
        files for partitions that already succeeded.
        """
        groups: dict[str, list[Event]] = defaultdict(list)
        for event in events:
            dt = event.occurred_at.astimezone(UTC).date().isoformat()
            groups[dt].append(event)

        remaining = list(events)
        self._buffers[cls] = remaining
        for dt, dt_events in groups.items():
            table = pa.Table.from_pylist([event_to_record(e) for e in dt_events], schema=schema)
            self._write_file(event_type, dt, table)
            written_ids = {e.event_id for e in dt_events}
            remaining = [e for e in remaining if e.event_id not in written_ids]
            self._buffers[cls] = remaining

    def _write_file(self, event_type: str, dt: str, table: pa.Table) -> None:
        partition_dir = self._root / event_type / f"dt={dt}"
        partition_dir.mkdir(parents=True, exist_ok=True)
        filename = f"part-{uuid.uuid4()}.parquet"
        final_path = partition_dir / filename
        tmp_path = partition_dir / f".{filename}.tmp"
        try:
            pq.write_table(table, tmp_path)
            tmp_path.rename(final_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            with contextlib.suppress(OSError):
                if not any(partition_dir.iterdir()):
                    partition_dir.rmdir()
            raise
