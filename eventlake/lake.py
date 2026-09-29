"""Reading events back out of an eventlake root via DuckDB."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path

import duckdb

from .event import RESERVED_SOURCE_FILE_COLUMN, validate_event_type
from .schema import SchemaRegistry
from .storage import Storage, resolve_storage

# `<event_type>/dt=YYYY-MM-DD/part-<id>.parquet` - the layout Writer produces.
_PART_KEY = re.compile(
    r"(?P<et>[a-z][a-z0-9_]{0,63})/dt=(?P<dt>[0-9]{4}-[0-9]{2}-[0-9]{2})/part-[^/]+\.parquet"
)


def _is_date(value: str) -> bool:
    try:
        date.fromisoformat(value)
    except ValueError:
        return False
    return True


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

    def __init__(self, root: str | Path | None = None, *, storage: Storage | None = None) -> None:
        self._storage = resolve_storage(root, storage)
        self._con = duckdb.connect(database=":memory:")
        # Keep timestamps as UTC on the way out; without this the session
        # picks up the OS-local timezone and re-labels every timestamp with
        # it (same instant, misleading label).
        self._con.execute("SET TimeZone='UTC'")
        self._storage.configure_duckdb(self._con)
        self._registry = SchemaRegistry(storage=self._storage)
        for event_type in self._event_types():
            self._create_view(event_type)

    def _event_types(self) -> list[str]:
        """Event types under the root that have at least one Parquet file.

        Ignores `_schemas` and anything not matching the layout regex (so a
        stray or half-named directory can't shadow a real type). An event
        type with no Parquet file yet is never listed: an empty
        `read_parquet(glob)` with no matches raises in DuckDB, so these must
        never reach `_create_view`.
        """
        found: set[str] = set()
        for key in self._storage.list_keys(""):
            match = _PART_KEY.fullmatch(key)
            if match is not None and _is_date(match.group("dt")):
                found.add(match.group("et"))
        return sorted(found)

    def _partition_keys(
        self, event_type: str, *, since: datetime | None, until: datetime | None
    ) -> dict[str, list[str]]:
        """Parquet file keys of `event_type`, by partition date (sorted)."""
        # Validate before the value becomes part of a key: an event_type like
        # "../x" must never make the storage look outside the root.
        validate_event_type(event_type)
        since_date = since.astimezone(UTC).date() if since is not None else None
        until_date = until.astimezone(UTC).date() if until is not None else None
        partitions: dict[str, list[str]] = {}
        for key in self._storage.list_keys(f"{event_type}/"):
            match = _PART_KEY.fullmatch(key)
            if match is None or not _is_date(match.group("dt")):
                continue
            dt = date.fromisoformat(match.group("dt"))
            if since_date is not None and dt < since_date:
                continue
            if until_date is not None and dt > until_date:
                continue
            partitions.setdefault(match.group("dt"), []).append(key)
        return partitions

    def _partition_files(
        self, event_type: str, *, since: datetime | None, until: datetime | None
    ) -> list[str]:
        partitions = self._partition_keys(event_type, since=since, until=until)
        return [self._storage.uri(key) for keys in partitions.values() for key in keys]

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
        pattern = self._storage.uri(f"{event_type}/dt=*/part-*.parquet")
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
            partitions = self._partition_keys(event_type, since=None, until=None)
            uris = [self._storage.uri(key) for keys in partitions.values() for key in keys]
            summaries.append(
                TypeSummary(
                    event_type=event_type,
                    schema_versions=len(self._registry.versions(event_type)),
                    partitions=len(partitions),
                    files=len(uris),
                    rows=self._count_rows(uris),
                )
            )
        return summaries

    def _count_rows(self, uris: list[str]) -> int:
        # Footer metadata only - no row data is read - and the same path for
        # every storage backend.
        file_list = ", ".join(_quote_literal(u) for u in uris)
        row = self._con.execute(
            f"SELECT COALESCE(SUM(num_rows), 0) FROM parquet_file_metadata([{file_list}])"
        ).fetchone()
        assert row is not None
        return int(row[0])
