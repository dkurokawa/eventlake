"""Where an eventlake root physically lives.

Every module that touches files goes through a `Storage`, so the same
Writer / SchemaRegistry / Lake / compact code runs against a local
directory or (see `S3Storage`) an object store. Paths handed to a Storage
are always *keys*: `/`-separated strings relative to the root.
"""

from __future__ import annotations

import os
import re
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast
from urllib.parse import urlparse

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


class Storage(Protocol):
    """The file operations eventlake needs, and nothing more."""

    def uri(self, key: str) -> str:
        """The absolute path / URI DuckDB should use to read `key` (globs allowed)."""
        ...

    def put_atomic(self, key: str, data: bytes) -> None:
        """Write `key` so that a reader sees all of it or none of it."""
        ...

    def put_exclusive(self, key: str, data: bytes) -> None:
        """Create `key` only if absent; raise FileExistsError otherwise. Never overwrites."""
        ...

    def read_bytes(self, key: str) -> bytes:
        """Return the content of `key`; FileNotFoundError if there is none."""
        ...

    def list_keys(self, prefix: str) -> list[str]:
        """Every key under `prefix` (recursive), sorted. Temporary files are not listed."""
        ...

    def delete(self, key: str) -> None:
        """Remove `key`; failure is an OSError (compact() reports it, not swallows it)."""
        ...

    def configure_duckdb(self, con: duckdb.DuckDBPyConnection) -> None:
        """Make `uri(...)` readable from `con` (credentials, extensions)."""
        ...


def _validate_key(key: str, *, allow_empty: bool = False) -> None:
    """Reject keys that could point outside the root.

    A key is built from an already-validated event_type / dt, so this is a
    second line of defence: `..` and `.` segments are refused outright, and
    so are empty segments (`a//b`), which mean different things to different
    backends.
    """
    if key == "" and allow_empty:
        return
    segments = key.split("/")
    if key.endswith("/") and allow_empty:
        segments = segments[:-1]
    if not segments or any(s in ("", ".", "..") for s in segments):
        raise ValueError(f"invalid storage key: {key!r}")


def table_to_parquet_bytes(table: pa.Table) -> bytes:
    """Serialize `table` to a complete Parquet file in memory."""
    sink = pa.BufferOutputStream()
    pq.write_table(table, sink)
    return bytes(sink.getvalue().to_pybytes())


class LocalStorage:
    """A root on the local filesystem."""

    def __init__(self, root: str | Path) -> None:
        self._root = Path(root)

    def _path(self, key: str) -> Path:
        _validate_key(key)
        return self._root.joinpath(*key.split("/"))

    def uri(self, key: str) -> str:
        # Not validated: callers pass globs (`dt=*`) built from checked parts.
        return str(self._root.joinpath(*key.split("/")))

    def put_atomic(self, key: str, data: bytes) -> None:
        # A failed write leaves the partition directory in place even if
        # it's now empty (this Writer just created it): removing it would
        # race with another Writer concurrently targeting the same
        # partition - it could rmdir the directory in the moment between
        # that other Writer's mkdir(exist_ok=True) no-op and its own write,
        # pulling the directory out from under a write that was otherwise
        # fine. An empty (or nonexistent) partition directory is already
        # invisible to Lake (see Lake._event_types), so there's nothing to
        # clean up here that matters.
        final_path = self._path(key)
        final_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = final_path.parent / f".{final_path.name}.tmp"
        try:
            tmp_path.write_bytes(data)
            tmp_path.rename(final_path)
        except BaseException:
            tmp_path.unlink(missing_ok=True)
            raise

    def put_exclusive(self, key: str, data: bytes) -> None:
        # Write the full content to a uniquely-named temp file first, then
        # atomically claim the target name with `os.link` (which fails with
        # FileExistsError if the target already exists, and never overwrites
        # it) - so a reader only ever sees the target absent or complete, and
        # two concurrent callers can't silently clobber each other.
        path = self._path(key)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
        tmp_path.write_bytes(data)
        try:
            os.link(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)

    def read_bytes(self, key: str) -> bytes:
        return self._path(key).read_bytes()

    def list_keys(self, prefix: str) -> list[str]:
        _validate_key(prefix, allow_empty=True)
        base = self._root.joinpath(*prefix.strip("/").split("/")) if prefix else self._root
        if not base.is_dir():
            return []
        keys: list[str] = []
        # followlinks: the root may contain symlinked directories, and the
        # rest of the library (like the README says) follows them.
        for dirpath, _dirnames, filenames in os.walk(base, followlinks=True):
            relative_dir = Path(dirpath).relative_to(self._root)
            for name in filenames:
                if name.startswith("."):
                    continue  # in-flight temp files are never part of the lake
                keys.append("/".join((*relative_dir.parts, name)))
        return sorted(keys)

    def delete(self, key: str) -> None:
        self._path(key).unlink()

    def configure_duckdb(self, con: duckdb.DuckDBPyConnection) -> None:
        return None


def open_storage(root: str | Path) -> Storage:
    """A `Storage` for `root`: `s3://bucket/prefix` is S3, anything else a local path."""
    if isinstance(root, str) and root.startswith("s3://"):
        return S3Storage.from_uri(root)
    return LocalStorage(root)


def resolve_storage(root: str | Path | None, storage: Storage | None) -> Storage:
    """The `Storage` for a `root` / `storage=` argument pair (exactly one)."""
    if storage is not None:
        if root is not None:
            raise ValueError("pass either a root or storage=, not both")
        return storage
    if root is None:
        raise ValueError("a root or storage= is required")
    return open_storage(root)


# S3 rejects a single PutObject above 5 GB. Parquet files are serialized in
# memory and written with one PutObject (no multipart upload - a multipart
# upload can leave a half-finished object behind), so a larger table is
# refused before anything is sent.
_MAX_SINGLE_PUT_BYTES = 5 * 1024**3

_BUCKET_PATTERN = re.compile(r"[a-z0-9][a-z0-9.-]{1,61}[a-z0-9]")


def parse_s3_uri(uri: str) -> tuple[str, str]:
    """Split `s3://<bucket>/<prefix>` into `(bucket, prefix)`, validating both.

    The prefix loses one trailing `/`; an empty prefix (the bucket root) is
    allowed. `.`, `..` and empty segments (`a//b`) are rejected: they mean
    different things to different tools and none of them belong in a root.
    """
    if not uri.startswith("s3://"):
        raise ValueError(f"not an s3:// URI: {uri!r}")
    bucket, _, prefix = uri[len("s3://") :].partition("/")
    if _BUCKET_PATTERN.fullmatch(bucket) is None:
        raise ValueError(f"invalid S3 bucket name in {uri!r}")
    if prefix.endswith("/"):
        prefix = prefix[:-1]
    if prefix and any(segment in ("", ".", "..") for segment in prefix.split("/")):
        raise ValueError(
            f"invalid S3 prefix in {uri!r}: '.', '..' and empty segments are not allowed"
        )
    return bucket, prefix


class S3Storage:
    """A root under `s3://<bucket>/<prefix>`, accessed with boto3.

    Needs the `s3` extra (`pip install 'eventlake[s3]'`). boto3 is imported
    here and nowhere else, so local use never loads it.
    """

    def __init__(self, bucket: str, prefix: str = "", *, client: S3Client | None = None) -> None:
        self._bucket = bucket
        self._prefix = prefix
        if client is None:
            try:
                import boto3
            except ImportError as exc:
                raise ImportError(
                    "S3 support needs the 's3' extra: pip install 'eventlake[s3]'"
                ) from exc
            client = boto3.client("s3")
        self._client = client

    @classmethod
    def from_uri(cls, uri: str, *, client: S3Client | None = None) -> S3Storage:
        bucket, prefix = parse_s3_uri(uri)
        return cls(bucket, prefix, client=client)

    def _full_key(self, key: str) -> str:
        return f"{self._prefix}/{key}" if self._prefix else key

    def uri(self, key: str) -> str:
        return f"s3://{self._bucket}/{self._full_key(key)}"

    def put_atomic(self, key: str, data: bytes) -> None:
        _validate_key(key)
        if len(data) > _MAX_SINGLE_PUT_BYTES:
            raise ValueError(
                f"{key!r} is {len(data)} bytes after serialization; a single S3 PutObject "
                f"is limited to {_MAX_SINGLE_PUT_BYTES} bytes and eventlake does not use "
                "multipart uploads. Flush smaller batches (lower max_rows)."
            )
        # One PutObject: the object appears whole or not at all.
        self._client.put_object(Bucket=self._bucket, Key=self._full_key(key), Body=data)

    def put_exclusive(self, key: str, data: bytes) -> None:
        from botocore.exceptions import ClientError

        _validate_key(key)
        try:
            self._client.put_object(
                Bucket=self._bucket, Key=self._full_key(key), Body=data, IfNoneMatch="*"
            )
        except ClientError as exc:
            code = exc.response.get("Error", {}).get("Code")
            status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
            # 412: the object already exists. 409: another conditional write
            # to the same key is in flight - either way we did not claim it.
            if code in ("PreconditionFailed", "ConditionalRequestConflict") or status in (
                409,
                412,
            ):
                raise FileExistsError(key) from exc
            raise

    def read_bytes(self, key: str) -> bytes:
        from botocore.exceptions import ClientError

        _validate_key(key)
        try:
            response = self._client.get_object(Bucket=self._bucket, Key=self._full_key(key))
        except ClientError as exc:
            if exc.response.get("Error", {}).get("Code") in ("NoSuchKey", "404"):
                raise FileNotFoundError(key) from exc
            raise
        return bytes(response["Body"].read())

    def list_keys(self, prefix: str) -> list[str]:
        _validate_key(prefix, allow_empty=True)
        directory = prefix if prefix == "" or prefix.endswith("/") else prefix + "/"
        strip = f"{self._prefix}/" if self._prefix else ""
        paginator = self._client.get_paginator("list_objects_v2")
        keys: list[str] = []
        for page in paginator.paginate(Bucket=self._bucket, Prefix=strip + directory):
            for item in page.get("Contents", []):
                relative = item["Key"][len(strip) :]
                if relative.rsplit("/", 1)[-1].startswith("."):
                    continue
                keys.append(relative)
        return sorted(keys)

    def delete(self, key: str) -> None:
        from botocore.exceptions import BotoCoreError, ClientError

        _validate_key(key)
        try:
            self._client.delete_object(Bucket=self._bucket, Key=self._full_key(key))
        except (ClientError, BotoCoreError) as exc:
            raise OSError(f"could not delete {self.uri(key)}: {exc}") from exc

    def configure_duckdb(self, con: duckdb.DuckDBPyConnection) -> None:
        """Load httpfs and hand DuckDB this client's credentials and endpoint.

        The credentials are frozen at this moment (see the README's note on
        temporary credentials). They travel inside a SQL string, so a failure
        here is re-raised without the statement or the driver's message.
        """
        # botocore has no public accessor for a client's credentials; the
        # request signer holds them (and refreshes them, for assumed roles).
        credentials = cast(Any, self._client)._request_signer._credentials
        if credentials is None:
            raise RuntimeError("the boto3 client has no credentials; DuckDB cannot read S3")
        frozen = credentials.get_frozen_credentials()
        options = [
            "TYPE s3",
            "PROVIDER config",
            f"KEY_ID {_quote_literal(frozen.access_key)}",
            f"SECRET {_quote_literal(frozen.secret_key)}",
        ]
        if frozen.token:
            options.append(f"SESSION_TOKEN {_quote_literal(frozen.token)}")
        region = self._client.meta.region_name
        if region:
            options.append(f"REGION {_quote_literal(region)}")
        endpoint = urlparse(self._client.meta.endpoint_url)
        on_aws = (endpoint.hostname or "").endswith(("amazonaws.com", "amazonaws.com.cn"))
        if not on_aws:
            # moto, MinIO and other S3-compatible servers: address the
            # endpoint explicitly, buckets in the path.
            options.append(f"ENDPOINT {_quote_literal(endpoint.netloc)}")
            options.append("URL_STYLE 'path'")
            options.append(f"USE_SSL {'true' if endpoint.scheme == 'https' else 'false'}")
        elif "." in self._bucket:
            # A dotted bucket name does not match the wildcard TLS
            # certificate of virtual-hosted-style addressing.
            options.append("URL_STYLE 'path'")

        # No secret values in these two statements.
        con.execute("INSTALL httpfs")
        con.execute("LOAD httpfs")
        try:
            con.execute(f"CREATE OR REPLACE SECRET ({', '.join(options)})")
        except duckdb.Error:
            raise RuntimeError("DuckDB rejected the S3 secret configuration") from None


def _quote_literal(value: str) -> str:
    return "'" + value.replace("'", "''") + "'"
