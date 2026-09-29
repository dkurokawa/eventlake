"""Where an eventlake root physically lives.

Every module that touches files goes through a `Storage`, so the same
Writer / SchemaRegistry / Lake / compact code runs against a local
directory or (see `S3Storage`) an object store. Paths handed to a Storage
are always *keys*: `/`-separated strings relative to the root.
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Protocol

import duckdb
import pyarrow as pa
import pyarrow.parquet as pq


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
    """A `Storage` for `root`."""
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
