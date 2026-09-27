"""Reading events back out of an eventlake root via DuckDB."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb
import pyarrow.parquet as pq

from .event import EVENT_TYPE_PATTERN, RESERVED_SOURCE_FILE_COLUMN
from .schema import SchemaRegistry


@dataclass(frozen=True)
class TypeSummary:
    event_type: str
    schema_versions: int
    partitions: int
    files: int
    rows: int


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"


def _quote_identifier(name: str) -> str:
    return '"' + name.replace('"', '""') + '"'


def _require_aware(value: datetime | None, name: str) -> None:
    if value is not None and value.tzinfo is None:
        raise ValueError(f"{name} must be timezone-aware, got a naive datetime: {value!r}")


class Lake:
    """A read-only view over an eventlake root, backed by an in-memory DuckDB."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)
        self._con = duckdb.connect(database=":memory:")
        # Keep timestamps as UTC on the way out; without this the session
        # picks up the OS-local timezone and re-labels every timestamp with
        # it (same instant, misleading label).
        self._con.execute("SET TimeZone='UTC'")
        self._registry = SchemaRegistry(self._root)
        for event_type in self._event_types():
            self._create_view(event_type)

    def _event_types(self) -> list[str]:
        """Directory names under root that look like a real, non-empty event type.

        Ignores `_schemas`, anything not matching EVENT_TYPE_PATTERN (so a
        stray or half-named directory can't shadow it), and any event_type
        directory that doesn't contain at least one Parquet file yet (an
        empty `read_parquet(glob)` with no matches raises in DuckDB, so
        these must never reach `_create_view`).
        """
        if not self._root.exists():
            return []
        result = []
        for p in sorted(self._root.iterdir()):
            if not p.is_dir() or p.name == "_schemas":
                continue
            if not EVENT_TYPE_PATTERN.fullmatch(p.name):
                continue
            if not any(p.glob("dt=*/part-*.parquet")):
                continue
            result.append(p.name)
        return result

    def _partition_dirs(
        self, event_type: str, *, since: datetime | None, until: datetime | None
    ) -> list[Path]:
        type_dir = self._root / event_type
        if not type_dir.exists():
            return []
        since_date = since.astimezone(UTC).date() if since is not None else None
        until_date = until.astimezone(UTC).date() if until is not None else None
        result: list[Path] = []
        for partition_dir in sorted(type_dir.glob("dt=*")):
            if not partition_dir.is_dir():
                continue
            try:
                dt = date.fromisoformat(partition_dir.name[len("dt=") :])
            except ValueError:
                continue
            if since_date is not None and dt < since_date:
                continue
            if until_date is not None and dt > until_date:
                continue
            result.append(partition_dir)
        return result

    def _partition_files(
        self, event_type: str, *, since: datetime | None, until: datetime | None
    ) -> list[str]:
        files: list[str] = []
        for partition_dir in self._partition_dirs(event_type, since=since, until=until):
            files.extend(str(p) for p in sorted(partition_dir.glob("part-*.parquet")))
        return files

    @staticmethod
    def _from_files_expr(files: list[str]) -> str:
        # hive_partitioning=false: the `dt=` directories are a physical
        # layout detail, not a data column. Without this, DuckDB auto-detects
        # the Hive layout and silently adds a `dt` column to every result.
        # filename=<reserved name>: adds a source-file column under a name
        # reserved for eventlake's own use (Event rejects it as a field name
        # at class-definition time - see RESERVED_FIELD_PREFIX), used as the
        # final, always-distinguishing dedup tiebreak below (event_id is
        # constant within the PARTITION BY event_id window, so it has no
        # power there; see _dedup_query). Plain `filename=true` would
        # collide with - and either error or shadow - a real event field
        # that happens to be named "filename".
        file_list = ", ".join(_quote_literal(f) for f in files)
        return (
            f"read_parquet([{file_list}], union_by_name=true, "
            f"hive_partitioning=false, filename={_quote_literal(RESERVED_SOURCE_FILE_COLUMN)})"
        )

    def _from_glob_expr(self, event_type: str) -> str:
        pattern = str(self._root / event_type / "dt=*" / "part-*.parquet")
        return (
            f"read_parquet({_quote_literal(pattern)}, union_by_name=true, "
            f"hive_partitioning=false, filename={_quote_literal(RESERVED_SOURCE_FILE_COLUMN)})"
        )

    @staticmethod
    def _dedup_query(from_expr: str) -> str:
        # event_id is constant within this PARTITION BY, so it cannot break
        # a tie by itself; recorded_at first (the documented rule: earliest
        # wins), then the reserved source-file column as a final,
        # always-distinct tiebreak.
        source_file = _quote_identifier(RESERVED_SOURCE_FILE_COLUMN)
        return f"""
            SELECT * EXCLUDE (__rn, {source_file}) FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY event_id ORDER BY recorded_at ASC, {source_file} ASC
                ) AS __rn
                FROM {from_expr}
            ) WHERE __rn = 1
        """

    def _create_view(self, event_type: str) -> None:
        view_sql = self._dedup_query(self._from_glob_expr(event_type))
        identifier = _quote_identifier(event_type)
        self._con.execute(f"CREATE OR REPLACE VIEW {identifier} AS {view_sql}")

    def _empty_relation(self, event_type: str) -> duckdb.DuckDBPyRelation:
        latest = self._registry.latest(event_type)
        if latest is None:
            raise ValueError(f"unknown event_type: {event_type!r}")
        empty_table = latest.schema.empty_table()  # noqa: F841 (used via SQL below)
        # DuckDB resolves `empty_table` by inspecting this frame's locals - it
        # is not an unused variable even though nothing calls it directly.
        return self._con.sql("SELECT * FROM empty_table")

    def events(
        self,
        event_type: str,
        *,
        since: datetime | None = None,
        until: datetime | None = None,
    ) -> duckdb.DuckDBPyRelation:
        """All events of `event_type`, deduplicated by event_id.

        Only reads the Parquet partitions whose dt falls within [since, until]
        (by occurred_at's UTC date). When the same event_id appears more than
        once (e.g. two Writer instances processed overlapping data), the copy
        with the earliest recorded_at wins; ties are broken by source filename
        (deterministic, but not meaningful - it only matters when recorded_at
        collides exactly).

        `since`/`until` must be timezone-aware; a naive datetime raises
        ValueError rather than being silently treated as local time.
        """
        _require_aware(since, "since")
        _require_aware(until, "until")
        files = self._partition_files(event_type, since=since, until=until)
        if not files:
            return self._empty_relation(event_type)
        return self._con.sql(self._dedup_query(self._from_files_expr(files)))

    def sql(self, query: str) -> duckdb.DuckDBPyRelation:
        """Run arbitrary SQL. Each event type is available as a same-named view."""
        return self._con.sql(query)

    def state_as_of(
        self,
        event_type: str,
        *,
        key: str,
        at: datetime | None = None,
    ) -> duckdb.DuckDBPyRelation:
        """Reconstruct the latest event per `key` as of a point in time.

        For each distinct value of the `key` column, returns the event with
        the greatest occurred_at that is <= `at` (or the greatest occurred_at
        overall, if `at` is None). Ties on occurred_at are broken by the most
        recent recorded_at, then by event_id, so the result is deterministic.

        `at` must be timezone-aware; a naive datetime raises ValueError
        rather than being silently treated as local time.
        """
        _require_aware(at, "at")
        files = self._partition_files(event_type, since=None, until=at)
        if not files:
            return self._empty_relation(event_type)

        deduped_expr = self._dedup_query(self._from_files_expr(files))
        where_clause = ""
        if at is not None:
            at_utc = at.astimezone(UTC)
            timestamp_literal = at_utc.strftime("%Y-%m-%d %H:%M:%S.%f")
            where_clause = f"WHERE occurred_at <= TIMESTAMP '{timestamp_literal}'"
        quoted_key = _quote_identifier(key)

        query = f"""
            SELECT * EXCLUDE (__rn2) FROM (
                SELECT *, ROW_NUMBER() OVER (
                    PARTITION BY {quoted_key}
                    ORDER BY occurred_at DESC, recorded_at DESC, event_id DESC
                ) AS __rn2
                FROM ({deduped_expr})
                {where_clause}
            ) WHERE __rn2 = 1
        """
        return self._con.sql(query)

    def describe(self) -> list[TypeSummary]:
        """Per event type: schema version count, partition count, file count, row count."""
        summaries: list[TypeSummary] = []
        for event_type in self._event_types():
            partition_dirs = self._partition_dirs(event_type, since=None, until=None)
            files: list[Path] = []
            for partition_dir in partition_dirs:
                files.extend(sorted(partition_dir.glob("part-*.parquet")))
            rows = sum(pq.ParquetFile(f).metadata.num_rows for f in files)
            summaries.append(
                TypeSummary(
                    event_type=event_type,
                    schema_versions=len(self._registry.versions(event_type)),
                    partitions=len(partition_dirs),
                    files=len(files),
                    rows=rows,
                )
            )
        return summaries
