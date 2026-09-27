"""Compacting the many small files a single partition accumulates over time."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import duckdb
import pyarrow.parquet as pq


@dataclass(frozen=True)
class CompactResult:
    event_type: str
    dt: str
    files_before: int
    files_after: int
    rows_before: int
    rows_after: int


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def compact(root: str | Path, event_type: str, dt: str) -> CompactResult:
    """Merge every file in `<root>/<event_type>/dt=<dt>/` into one file.

    Duplicate event_ids (from separate Writer instances writing overlapping
    data) are dropped, keeping the one with the earliest recorded_at - the
    same rule `Lake.events()` uses at read time. The new file is written and
    renamed into place before the old files are removed, so a failure midway
    never loses rows: either the old files are all still there, or the new
    file is in place and the old files are removed.

    Concurrent writes to the same partition during a compaction are out of
    scope; run compaction when no writer is targeting that partition.
    """
    root = Path(root)
    partition_dir = root / event_type / f"dt={dt}"
    files = sorted(partition_dir.glob("part-*.parquet"))
    if not files:
        raise FileNotFoundError(f"no parquet files found for {event_type} dt={dt} under {root}")

    rows_before = sum(pq.ParquetFile(f).metadata.num_rows for f in files)

    if len(files) == 1:
        return CompactResult(event_type, dt, 1, 1, rows_before, rows_before)

    con = duckdb.connect(database=":memory:")
    try:
        con.execute("SET TimeZone='UTC'")
        file_list = ", ".join(_quote_literal(str(f)) for f in files)
        relation = con.sql(
            f"""
            SELECT * EXCLUDE (__rn) FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY event_id ORDER BY recorded_at ASC, event_id ASC
                ) AS __rn
                FROM read_parquet([{file_list}], union_by_name=true, hive_partitioning=false)
            ) WHERE __rn = 1
            """
        )
        table = relation.arrow().read_all()
    finally:
        con.close()

    rows_after = table.num_rows
    filename = f"part-{uuid.uuid4()}.parquet"
    new_path = partition_dir / filename
    tmp_path = partition_dir / f".{filename}.tmp"
    try:
        pq.write_table(table, tmp_path)
        tmp_path.rename(new_path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise

    for old_file in files:
        old_file.unlink()

    return CompactResult(event_type, dt, len(files), 1, rows_before, rows_after)
