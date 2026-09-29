"""Reading an S3 root: DuckDB httpfs configuration, credentials, secrets."""

from __future__ import annotations

from types import SimpleNamespace
from typing import TYPE_CHECKING, Any, ClassVar

import duckdb
import pytest
from _helpers import utc

from eventlake.event import Event
from eventlake.lake import Lake
from eventlake.storage import S3Storage
from eventlake.writer import Writer

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class Ping(Event):
    event_type: ClassVar[str] = "ping"

    source: str


class Owner(Event):
    event_type: ClassVar[str] = "owner"

    item: str
    owner: str


def test_lake_reads_an_s3_root_through_duckdb(s3_client: S3Client, s3_bucket: str) -> None:
    store = S3Storage(s3_bucket, "lake", client=s3_client)
    with Writer(storage=store) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
        writer.write(Ping(occurred_at=utc(2026, 1, 2), source="b"))

    lake = Lake(storage=store)
    assert sorted(lake.events("ping").to_arrow_table().column("source").to_pylist()) == ["a", "b"]
    assert lake.sql("SELECT COUNT(*) FROM ping").fetchall() == [(2,)]
    only_second = lake.events("ping", since=utc(2026, 1, 2)).to_arrow_table()
    assert only_second.column("source").to_pylist() == ["b"]


def test_lake_state_as_of_on_s3(s3_client: S3Client, s3_bucket: str) -> None:
    store = S3Storage(s3_bucket, "lake", client=s3_client)
    with Writer(storage=store) as writer:
        writer.write(Owner(occurred_at=utc(2026, 1, 1), item="x", owner="alice"))
        writer.write(Owner(occurred_at=utc(2026, 1, 5), item="x", owner="bob"))
    lake = Lake(storage=store)
    owners = lake.state_as_of("owner", key="item", at=utc(2026, 1, 2)).to_arrow_table()
    assert owners.column("owner").to_pylist() == ["alice"]


def test_describe_counts_rows_from_parquet_footers_on_s3(
    s3_client: S3Client, s3_bucket: str
) -> None:
    store = S3Storage(s3_bucket, "lake", client=s3_client)
    with Writer(storage=store) as writer:
        writer.write_many([Ping(occurred_at=utc(2026, 1, 1), source=str(n)) for n in range(5)])
    (summary,) = Lake(storage=store).describe()
    assert (summary.event_type, summary.partitions, summary.files, summary.rows) == (
        "ping",
        1,
        1,
        5,
    )
    assert summary.schema_versions == 1


def test_lake_opens_an_s3_uri_using_the_standard_boto3_endpoint(
    s3_client: S3Client, s3_bucket: str
) -> None:
    # No client is passed: credentials and the moto endpoint come from the
    # environment (AWS_ENDPOINT_URL_S3 is set by the s3_client fixture), the
    # way a user would point eventlake at MinIO.
    root = f"s3://{s3_bucket}/from-uri"
    with Writer(root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    assert Lake(root).sql("SELECT source FROM ping").fetchall() == [("a",)]


def test_an_empty_s3_root_is_an_empty_lake(s3_client: S3Client, s3_bucket: str) -> None:
    lake = Lake(storage=S3Storage(s3_bucket, "nothing-yet", client=s3_client))
    assert lake.describe() == []


# --- DuckDB configuration without a server ----------------------------------

_SECRET = "s3cr3t-value-that-must-not-leak"
_TOKEN = "session-token-that-must-not-leak"


class _RecordingConnection:
    def __init__(self, fail_on: str | None = None) -> None:
        self.statements: list[str] = []
        self._fail_on = fail_on

    def execute(self, sql: str) -> None:
        self.statements.append(sql)
        if self._fail_on is not None and sql.startswith(self._fail_on):
            raise duckdb.Error(f"could not run: {sql}")


def _fake_client(
    *,
    endpoint: str,
    region: str | None = "us-east-1",
    token: str | None = None,
    credentials: bool = True,
) -> Any:
    frozen = SimpleNamespace(access_key="AKIDEXAMPLE", secret_key=_SECRET, token=token)
    creds = SimpleNamespace(get_frozen_credentials=lambda: frozen) if credentials else None
    return SimpleNamespace(
        _request_signer=SimpleNamespace(_credentials=creds),
        meta=SimpleNamespace(endpoint_url=endpoint, region_name=region),
    )


def _secret_statement(con: _RecordingConnection) -> str:
    (statement,) = [s for s in con.statements if s.startswith("CREATE OR REPLACE SECRET")]
    return statement


def test_configure_duckdb_for_aws_uses_region_and_session_token() -> None:
    store = S3Storage(
        "my-bucket",
        "lake",
        client=_fake_client(endpoint="https://s3.eu-west-1.amazonaws.com", token=_TOKEN),
    )
    con = _RecordingConnection()
    store.configure_duckdb(con)  # type: ignore[arg-type]
    assert con.statements[:2] == ["INSTALL httpfs", "LOAD httpfs"]
    secret = _secret_statement(con)
    assert "TYPE s3" in secret and "PROVIDER config" in secret
    assert f"SESSION_TOKEN '{_TOKEN}'" in secret
    assert "REGION 'us-east-1'" in secret
    assert "ENDPOINT" not in secret  # DuckDB derives the AWS endpoint from the region
    assert "URL_STYLE" not in secret


def test_configure_duckdb_for_a_custom_endpoint_uses_path_style() -> None:
    store = S3Storage("my-bucket", "", client=_fake_client(endpoint="http://minio.local:9000"))
    con = _RecordingConnection()
    store.configure_duckdb(con)  # type: ignore[arg-type]
    secret = _secret_statement(con)
    assert "ENDPOINT 'minio.local:9000'" in secret
    assert "URL_STYLE 'path'" in secret
    assert "USE_SSL false" in secret
    assert "SESSION_TOKEN" not in secret


def test_configure_duckdb_for_a_custom_https_endpoint_keeps_ssl_on() -> None:
    store = S3Storage("my-bucket", "", client=_fake_client(endpoint="https://s3.example.test"))
    con = _RecordingConnection()
    store.configure_duckdb(con)  # type: ignore[arg-type]
    assert "USE_SSL true" in _secret_statement(con)


def test_configure_duckdb_uses_path_style_for_dotted_buckets_on_aws() -> None:
    store = S3Storage(
        "my.dotted.bucket",
        "",
        client=_fake_client(endpoint="https://s3.us-east-1.amazonaws.com", region=None),
    )
    con = _RecordingConnection()
    store.configure_duckdb(con)  # type: ignore[arg-type]
    secret = _secret_statement(con)
    assert "URL_STYLE 'path'" in secret
    assert "REGION" not in secret
    assert "ENDPOINT" not in secret


def test_configure_duckdb_without_credentials_fails_clearly() -> None:
    store = S3Storage(
        "my-bucket", "", client=_fake_client(endpoint="https://s3.amazonaws.com", credentials=False)
    )
    with pytest.raises(RuntimeError, match="no credentials"):
        store.configure_duckdb(_RecordingConnection())  # type: ignore[arg-type]


def test_a_rejected_secret_does_not_leak_it_into_the_exception() -> None:
    store = S3Storage(
        "my-bucket",
        "",
        client=_fake_client(endpoint="http://127.0.0.1:1", token=_TOKEN),
    )
    con = _RecordingConnection(fail_on="CREATE OR REPLACE SECRET")
    with pytest.raises(RuntimeError) as excinfo:
        store.configure_duckdb(con)  # type: ignore[arg-type]
    rendered = repr(excinfo.value) + str(excinfo.value)
    assert _SECRET not in rendered
    assert _TOKEN not in rendered
    assert excinfo.value.__cause__ is None
    assert excinfo.value.__suppress_context__


def test_the_storage_object_does_not_expose_credentials_in_its_repr() -> None:
    store = S3Storage("my-bucket", "lake", client=_fake_client(endpoint="http://x", token=_TOKEN))
    text = repr(store) + str(store)
    assert _SECRET not in text
    assert _TOKEN not in text
    assert "AKIDEXAMPLE" not in text
