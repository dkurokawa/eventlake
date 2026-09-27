"""Buffering and flushing events to partitioned Parquet files."""

from __future__ import annotations

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
    """Buffers events per event type and flushes them to Parquet.

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
        self._buffers: dict[str, list[Event]] = defaultdict(list)
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
            event_type = type(recorded).event_type
            self._buffers[event_type].append(recorded)
            if len(self._buffers[event_type]) >= self._max_rows:
                self._flush_type(event_type)

    def flush(self) -> None:
        for event_type in list(self._buffers.keys()):
            self._flush_type(event_type)

    def _flush_type(self, event_type: str) -> None:
        buffer = self._buffers.get(event_type)
        if not buffer:
            return
        cls = type(buffer[0])
        schema = arrow_schema_for(cls)
        # May raise SchemaChangeError; buffer stays intact if it does.
        self._registry.register(event_type, schema, allow_breaking=self._allow_breaking)
        table = pa.Table.from_pylist([event_to_record(e) for e in buffer], schema=schema)
        self._write_partitions(event_type, table, buffer)
        self._buffers[event_type] = []

    def _write_partitions(self, event_type: str, table: pa.Table, events: list[Event]) -> None:
        groups: dict[str, list[int]] = defaultdict(list)
        for index, event in enumerate(events):
            dt = event.occurred_at.astimezone(UTC).date().isoformat()
            groups[dt].append(index)
        for dt, indices in groups.items():
            partition_table = table.take(pa.array(indices))
            self._write_file(event_type, dt, partition_table)

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
            raise
