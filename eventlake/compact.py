"""Compacting the many small files a single partition accumulates over time."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from .event import validate_event_type


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


def compact(root: str | Path, event_type: str, dt: str) -> CompactResult:
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
    root = Path(root)
    partition_dir = root / event_type / f"dt={dt}"
    files = sorted(partition_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found for {event_type} dt={dt} under {root}")

    rows_before = sum(pq.ParquetFile(f).metadata.num_rows for f in files)

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("SET TimeZone='UTC'")
        file_list = ", ".join(_quote_literal(str(f)) for f in files)
        source_expr = (
            f"read_parquet([{file_list}], union_by_name=true, "
            "hive_partitioning=false, filename=true)"
        )

        if len(files) == 1:
            distinct_count = con.sql(
                f"SELECT COUNT(DISTINCT event_id) FROM {source_expr}"
            ).fetchone()
            assert distinct_count is not None
            if distinct_count[0] == rows_before:
                # The only file has no duplicates: nothing to do.
                return CompactResult(event_type, dt, 1, 1, rows_before, rows_before)

        relation = con.sql(
            f"""
            SELECT * EXCLUDE (__rn, filename) FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY event_id ORDER BY recorded_at ASC, filename ASC
                ) AS __rn
                FROM {source_expr}
            ) WHERE __rn = 1
            """
        )
        table = relation.to_arrow_table()
    finally:
        con.close()

    rows_after = table.num_rows
    new_filename = f"part-{uuid.uuid4()}.parquet"
    new_path = partition_dir / new_filename
    tmp_path = partition_dir / f".{new_filename}.tmp"
    try:
        pq.write_table(table, tmp_path)
        tmp_path.rename(new_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    leftover: list[Path] = []
    for old_file in files:
        try:
            old_file.unlink()
        except OSError:
            leftover.append(old_file)

    if leftover:
        raise CompactionIncomplete(event_type, dt, new_path, leftover)

    return CompactResult(event_type, dt, len(files), 1, rows_before, rows_after)
