"""Behaviour that has to be identical on every backend, and the ways the
local and S3 backends are allowed to meet (moving a lake between them)."""

from __future__ import annotations

import re
from pathlib import Path
from typing import TYPE_CHECKING, ClassVar

import pytest
from _helpers import keys, utc

from eventlake.compact import compact
from eventlake.event import Event
from eventlake.lake import Lake
from eventlake.schema import SchemaRegistry
from eventlake.storage import LocalStorage, S3Storage, open_storage
from eventlake.writer import Writer

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class Ping(Event):
    event_type: ClassVar[str] = "ping"

    source: str


class PingV2(Event):
    event_type: ClassVar[str] = "ping"

    source: str
    note: str | None = None


# --- layout: what Athena / Spark / plain DuckDB globs rely on ---------------

_LAYOUT = re.compile(
    r"_schemas/[a-z][a-z0-9_]{0,63}/v[0-9]+\.json"
    r"|[a-z][a-z0-9_]{0,63}/dt=[0-9]{4}-[0-9]{2}-[0-9]{2}/part-[0-9a-f-]{36}\.parquet"
)


def test_key_layout_is_plain_hive_partitioning_on_every_backend(lake_root: str | Path) -> None:
    with Writer(lake_root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
        writer.write(Ping(occurred_at=utc(2026, 1, 2), source="b"))
    with Writer(lake_root) as writer:
        writer.write(PingV2(occurred_at=utc(2026, 1, 2), source="c", note="n"))

    all_keys = keys(lake_root)
    assert len(all_keys) == 5  # v1.json, v2.json and three Parquet files
    unexpected = [k for k in all_keys if _LAYOUT.fullmatch(k) is None]
    assert unexpected == []
    assert sum(k.endswith(".json") for k in all_keys) == 2  # v1 and v2
    assert sorted({k.split("/")[1] for k in all_keys if k.startswith("ping/")}) == [
        "dt=2026-01-01",
        "dt=2026-01-02",
    ]


# --- the same operations give the same results on both backends -------------


def _scenario(root: str | Path) -> dict[str, object]:
    with Writer(root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
        writer.write(Ping(occurred_at=utc(2026, 1, 2), source="b"))
    with Writer(root) as writer:
        writer.write(PingV2(occurred_at=utc(2026, 1, 2, 12), source="c", note="n"))
    lake = Lake(root)
    before = sorted(lake.sql("SELECT source, note FROM ping").fetchall(), key=repr)
    compact_result = compact(root, "ping", "2026-01-02")
    after = sorted(Lake(root).sql("SELECT source, note FROM ping").fetchall(), key=repr)
    (summary,) = Lake(root).describe()
    return {
        "before": before,
        "after": after,
        "compact": (
            compact_result.files_before,
            compact_result.files_after,
            compact_result.rows_before,
            compact_result.rows_after,
        ),
        "summary": (
            summary.schema_versions,
            summary.partitions,
            summary.files,
            summary.rows,
        ),
        "versions": SchemaRegistry(root).versions("ping"),
    }


def test_the_same_scenario_gives_the_same_results_locally_and_on_s3(
    tmp_path: Path, s3_bucket: str
) -> None:
    local = _scenario(tmp_path)
    s3 = _scenario(f"s3://{s3_bucket}/lake")
    assert local == s3
    assert local["before"] == local["after"]


# --- moving a lake between backends -----------------------------------------


def test_a_local_lake_copied_to_s3_reads_the_same(
    tmp_path: Path, s3_client: S3Client, s3_bucket: str
) -> None:
    """The `aws s3 sync` case: write locally, upload every file as-is, read it
    back through `s3://`."""
    local_root = tmp_path / "lake"
    with Writer(local_root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
        writer.write(Ping(occurred_at=utc(2026, 1, 2), source="b"))
    with Writer(local_root) as writer:
        writer.write(PingV2(occurred_at=utc(2026, 1, 3), source="c", note="n"))

    for path in sorted(local_root.rglob("*")):
        if path.is_file():
            key = "synced/" + path.relative_to(local_root).as_posix()
            s3_client.put_object(Bucket=s3_bucket, Key=key, Body=path.read_bytes())

    query = "SELECT source, note FROM ping ORDER BY source"
    local_rows = Lake(local_root).sql(query).fetchall()
    s3_rows = Lake(f"s3://{s3_bucket}/synced").sql(query).fetchall()
    assert s3_rows == local_rows == [("a", None), ("b", None), ("c", "n")]
    assert SchemaRegistry(f"s3://{s3_bucket}/synced").versions("ping") == [1, 2]
    assert Lake(f"s3://{s3_bucket}/synced").describe() == Lake(local_root).describe()


# --- the storage= argument --------------------------------------------------


def test_storage_argument_is_equivalent_to_a_root(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    with Writer(storage=store) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    assert Lake(storage=store).sql("SELECT source FROM ping").fetchall() == [("a",)]
    assert SchemaRegistry(storage=store).versions("ping") == [1]
    assert compact(None, "ping", "2026-01-01", storage=store).files_after == 1


def test_root_and_storage_together_are_refused(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    with pytest.raises(ValueError, match="not both"):
        Writer(tmp_path, storage=store)
    with pytest.raises(ValueError, match="not both"):
        Lake(tmp_path, storage=store)
    with pytest.raises(ValueError, match="not both"):
        SchemaRegistry(tmp_path, storage=store)
    with pytest.raises(ValueError, match="not both"):
        compact(tmp_path, "ping", "2026-01-01", storage=store)


def test_neither_root_nor_storage_is_refused() -> None:
    with pytest.raises(ValueError, match="required"):
        Writer()
    with pytest.raises(ValueError, match="required"):
        Lake()


# --- a key can never leave the root, on any backend --------------------------


def test_unsafe_event_types_and_dates_are_refused_before_any_key_is_built(
    lake_root: str | Path,
) -> None:
    with Writer(lake_root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    before = keys(lake_root)

    with pytest.raises(ValueError, match="invalid event_type"):
        Lake(lake_root).events("../escape")
    with pytest.raises(ValueError, match="invalid event_type"):
        Lake(lake_root).state_as_of("../escape", key="k")
    with pytest.raises(ValueError, match="invalid event_type"):
        compact(lake_root, "../escape", "2026-01-01")
    with pytest.raises(ValueError, match="invalid partition date"):
        compact(lake_root, "ping", "../2026-01-01")
    assert keys(lake_root) == before


def test_open_storage_gives_the_matching_backend(lake_root: str | Path) -> None:
    expected = LocalStorage if isinstance(lake_root, Path) else S3Storage
    assert isinstance(open_storage(lake_root), expected)
