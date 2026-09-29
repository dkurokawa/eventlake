"""Compacting the many small files a single partition accumulates over time."""

from __future__ import annotations

import re
import uuid
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import duckdb

from .event import RESERVED_SOURCE_FILE_COLUMN, validate_event_type
from .storage import Storage, resolve_storage, table_to_parquet_bytes

_PART_FILE = re.compile(r"part-[^/]+\.parquet")


def validate_dt(dt: str) -> None:
    """Raise ValueError unless `dt` is a strict `YYYY-MM-DD` date string.

    `dt` is embedded directly into a filesystem path (`dt=<dt>`), so a loose
    `date.fromisoformat()` isn't enough on its own (Python 3.11+ accepts a
    wider range of ISO 8601 forms there than the `dt=YYYY-MM-DD` layout
    uses) - the parsed date's canonical string form must round-trip back to
    exactly `dt`, which also rules out anything path-like (`../x`, an
    absolute path) since those never parse as a date at all.
    """
    try:
        parsed = date.fromisoformat(dt)
    except ValueError as exc:
        raise ValueError(f"invalid partition date {dt!r}: must be YYYY-MM-DD") from exc
    if str(parsed) != dt:
        raise ValueError(f"invalid partition date {dt!r}: must be YYYY-MM-DD")


@dataclass(frozen=True)
class CompactResult:
    event_type: str
    dt: str
    files_before: int
    files_after: int
    rows_before: int
    rows_after: int


class CompactionIncomplete(Exception):
    """Compaction wrote its new file but couldn't remove all the old ones.

    The new file (a complete, correctly-deduplicated merge of everything
    that was in the partition) is safely in place; the old files listed in
    `leftover_files` are still there too. Reading through `Lake` still
    produces correct, deduplicated results in this state - the surviving
    old files just contain rows that are now duplicates of what's in the
    new file, and `Lake.events()` already dedups by event_id. Re-running
    `compact()` on the same partition picks up the leftover file(s) too and
    finishes the cleanup.
    """

    def __init__(
        self, event_type: str, dt: str, new_file: Path, leftover_files: list[Path]
    ) -> None:
        self.event_type = event_type
        self.dt = dt
        self.new_file = new_file
        self.leftover_files = leftover_files
        leftover_list = ", ".join(str(f) for f in leftover_files)
        super().__init__(
            f"compaction of {event_type} dt={dt} wrote {new_file} but failed to remove "
            f"{len(leftover_files)} old file(s): {leftover_list}"
        )


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def compact(
    root: str | Path | None, event_type: str, dt: str, *, storage: Storage | None = None
) -> CompactResult:
    """Merge every file in `<root>/<event_type>/dt=<dt>/` into one file.

    Duplicate event_ids (from separate Writer instances writing overlapping
    data, or - rarely - accumulating within a single file) are dropped,
    keeping the one with the earliest recorded_at (ties broken by source
    filename) - the same rule `Lake.events()` uses at read time. A
    single-file partition with no duplicates is left untouched.

    The new file is written and renamed into place before the old files are
    removed, so a failure while writing it never loses rows: either the old
    files are all still there, or the new file is in place. If the new file
    is in place but removing an old file then fails, `CompactionIncomplete`
    is raised (see its docstring) rather than losing track of it.

    Concurrent writes to the same partition during a compaction are out of
    scope; run compaction when no writer is targeting that partition.
    """
    validate_event_type(event_type)
    validate_dt(dt)
    store = resolve_storage(root, storage)
    partition = f"{event_type}/dt={dt}"
    keys = [
        k for k in store.list_keys(f"{partition}/") if _PART_FILE.fullmatch(k[len(partition) + 1 :])
    ]
    if not keys:
        raise FileNotFoundError(f"no parquet files found for {event_type} dt={dt} under {root}")
    files = [store.uri(k) for k in keys]

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("SET TimeZone='UTC'")
        store.configure_duckdb(con)
        file_list = ", ".join(_quote_literal(f) for f in files)
        rows_row = con.execute(
            f"SELECT COALESCE(SUM(num_rows), 0) FROM parquet_file_metadata([{file_list}])"
        ).fetchone()
        assert rows_row is not None
        rows_before = int(rows_row[0])
        # filename=<reserved name>, not filename=true: see the comment on
        # Lake._from_files_expr - plain `filename=true` collides with a
        # real event field that happens to be named "filename".
        source_expr = (
            f"read_parquet([{file_list}], union_by_name=true, hive_partitioning=false, "
            f"filename={_quote_literal(RESERVED_SOURCE_FILE_COLUMN)})"
        )

        if len(files) == 1:
            distinct_count = con.sql(
                f"SELECT COUNT(DISTINCT event_id) FROM {source_expr}"
            ).fetchone()
            assert distinct_count is not None
            if distinct_count[0] == rows_before:
                # The only file has no duplicates: nothing to do.
                return CompactResult(event_type, dt, 1, 1, rows_before, rows_before)

        source_file = _quote_identifier(RESERVED_SOURCE_FILE_COLUMN)
        relation = con.sql(
            f"""
            SELECT * EXCLUDE (__rn, {source_file}) FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY event_id ORDER BY recorded_at ASC, {source_file} ASC
                ) AS __rn
                FROM {source_expr}
            ) WHERE __rn = 1
            """
        )
        table = relation.to_arrow_table()
    finally:
        con.close()

    rows_after = table.num_rows
    new_key = f"{partition}/part-{uuid.uuid4()}.parquet"
    store.put_atomic(new_key, table_to_parquet_bytes(table))

    leftover: list[Path] = []
    for old_key in keys:
        try:
            store.delete(old_key)
        except OSError:
            leftover.append(Path(store.uri(old_key)))

    if leftover:
        raise CompactionIncomplete(event_type, dt, Path(store.uri(new_key)), leftover)

    return CompactResult(event_type, dt, len(files), 1, rows_before, rows_after)
