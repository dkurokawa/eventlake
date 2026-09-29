"""Unit tests for the Storage implementations (local and S3 via moto)."""

from __future__ import annotations

import sys
import threading
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

from eventlake import storage as storage_module
from eventlake.storage import (
    LocalStorage,
    S3Storage,
    open_storage,
    parse_s3_uri,
    resolve_storage,
)

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client


# --- URI validation ---------------------------------------------------------


@pytest.mark.parametrize(
    ("uri", "expected"),
    [
        ("s3://my-bucket/lake", ("my-bucket", "lake")),
        ("s3://my-bucket/lake/", ("my-bucket", "lake")),
        ("s3://my-bucket/a/b/c", ("my-bucket", "a/b/c")),
        ("s3://my-bucket", ("my-bucket", "")),
        ("s3://my-bucket/", ("my-bucket", "")),
        ("s3://my.dotted.bucket/p", ("my.dotted.bucket", "p")),
    ],
)
def test_parse_s3_uri_accepts(uri: str, expected: tuple[str, str]) -> None:
    assert parse_s3_uri(uri) == expected


@pytest.mark.parametrize(
    "uri",
    [
        "s3://ab/p",  # too short
        "s3://" + "a" * 64 + "/p",  # too long
        "s3://UpperCase/p",
        "s3://under_score/p",
        "s3://-leading/p",
        "s3://trailing-/p",
        "s3:///p",
        "s3://bucket/a/../b",
        "s3://bucket/../b",
        "s3://bucket/./b",
        "s3://bucket/a//b",
        "s3://bucket/a//",
        "s3://bucket/a*",
        "s3://bucket/a?b",
        "s3://bucket/a[0-9]",
        "s3://bucket/a]",
        "s3://bucket/{a,b}",
        "s3://bucket/x/}",
        "s3://my..bucket/p",  # consecutive periods
        "s3://192.168.5.4/p",  # IP-address-shaped
        "s3://xn--bucket/p",
        "s3://sthree-bucket/p",
        "s3://amzn-s3-demo-bucket/p",
        "s3://bucket-s3alias/p",
        "s3://bucket--ol-s3/p",
        "s3://bucket--x-s3/p",
        "s3://bucket--table-s3/p",
        "s3://bucket.mrap/p",
        "https://bucket/p",
        "bucket/p",
    ],
)
def test_parse_s3_uri_rejects(uri: str) -> None:
    with pytest.raises(ValueError):
        parse_s3_uri(uri)


@pytest.mark.parametrize(
    ("bucket", "prefix"),
    [
        ("my-bucket", "a*"),
        ("my-bucket", "a?"),
        ("my-bucket", "[ab]"),
        ("my-bucket", "{a,b}"),
        ("my-bucket", "a/../b"),
        ("my-bucket", "a//b"),
        ("my..bucket", ""),
        ("10.0.0.1", ""),
        ("xn--bucket", ""),
        ("bucket--x-s3", ""),
        ("Upper", ""),
    ],
)
def test_direct_construction_is_validated_too(bucket: str, prefix: str) -> None:
    # The client is never used: validation happens first.
    with pytest.raises(ValueError):
        S3Storage(bucket, prefix, client=object())  # type: ignore[arg-type]


def test_open_storage_picks_the_backend_from_the_root(s3_client: S3Client, tmp_path: Path) -> None:
    assert isinstance(open_storage("s3://some-bucket/lake"), S3Storage)
    assert isinstance(open_storage(str(tmp_path)), LocalStorage)
    assert isinstance(open_storage(tmp_path), LocalStorage)


def test_s3_root_without_boto3_explains_the_extra(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # A None entry in sys.modules makes `import boto3` raise ImportError.
    monkeypatch.setitem(sys.modules, "boto3", None)
    with pytest.raises(ImportError, match=r"eventlake\[s3\]"):
        open_storage("s3://some-bucket/lake")


def test_resolve_storage_needs_exactly_one_of_root_and_storage(tmp_path: Path) -> None:
    local = LocalStorage(tmp_path)
    assert resolve_storage(None, local) is local
    assert isinstance(resolve_storage(tmp_path, None), LocalStorage)
    with pytest.raises(ValueError, match="not both"):
        resolve_storage(tmp_path, local)
    with pytest.raises(ValueError, match="required"):
        resolve_storage(None, None)


# --- LocalStorage -----------------------------------------------------------


def test_local_put_atomic_read_list_delete(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    store.put_atomic("a/dt=2026-01-01/part-1.parquet", b"one")
    store.put_atomic("a/dt=2026-01-02/part-2.parquet", b"two")
    store.put_atomic("b/x.json", b"{}")

    assert store.read_bytes("a/dt=2026-01-01/part-1.parquet") == b"one"
    assert store.list_keys("a/") == [
        "a/dt=2026-01-01/part-1.parquet",
        "a/dt=2026-01-02/part-2.parquet",
    ]
    assert store.list_keys("") == [
        "a/dt=2026-01-01/part-1.parquet",
        "a/dt=2026-01-02/part-2.parquet",
        "b/x.json",
    ]
    store.delete("b/x.json")
    assert store.list_keys("b/") == []
    assert store.list_keys("nothing-here/") == []


def test_local_list_keys_hides_temp_files(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    store.put_atomic("a/real.json", b"{}")
    (tmp_path / "a" / ".real.json.tmp").write_bytes(b"partial")
    assert store.list_keys("a/") == ["a/real.json"]


def test_local_read_missing_key_is_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        LocalStorage(tmp_path).read_bytes("nope/x.json")


def test_local_put_atomic_failure_leaves_no_temp_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = LocalStorage(tmp_path)

    def boom(self: Path, target: Any) -> None:
        raise RuntimeError("disk full")

    monkeypatch.setattr(Path, "rename", boom)
    with pytest.raises(RuntimeError, match="disk full"):
        store.put_atomic("a/x.parquet", b"data")
    assert store.list_keys("") == []
    assert list((tmp_path / "a").glob(".*")) == []


def test_local_put_exclusive_never_overwrites(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    store.put_exclusive("s/v1.json", b"first")
    with pytest.raises(FileExistsError):
        store.put_exclusive("s/v1.json", b"second")
    assert store.read_bytes("s/v1.json") == b"first"
    assert store.list_keys("s/") == ["s/v1.json"]
    assert list((tmp_path / "s").glob(".*")) == []  # no temp files left behind


@pytest.mark.parametrize("key", ["../x", "a/../../x", "a//b", "./a", "/abs", "a/./b", ""])
def test_local_keys_cannot_leave_the_root(tmp_path: Path, key: str) -> None:
    store = LocalStorage(tmp_path)
    with pytest.raises(ValueError):
        store.put_atomic(key, b"x")
    with pytest.raises(ValueError):
        store.read_bytes(key)


def test_local_list_keys_rejects_escaping_prefix(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        LocalStorage(tmp_path).list_keys("../")


def test_local_uri_is_an_absolute_style_path_and_keeps_globs(tmp_path: Path) -> None:
    store = LocalStorage(tmp_path)
    assert store.uri("a/dt=*/part-*.parquet") == str(tmp_path / "a" / "dt=*" / "part-*.parquet")
    assert store.uri("") == str(tmp_path)


# --- S3Storage (moto) -------------------------------------------------------


@pytest.fixture
def s3_store(s3_client: S3Client, s3_bucket: str) -> S3Storage:
    return S3Storage(s3_bucket, "lake", client=s3_client)


def test_s3_put_atomic_read_list_delete(s3_store: S3Storage) -> None:
    s3_store.put_atomic("a/dt=2026-01-01/part-1.parquet", b"one")
    s3_store.put_atomic("a/dt=2026-01-02/part-2.parquet", b"two")
    s3_store.put_atomic("b/x.json", b"{}")

    assert s3_store.read_bytes("a/dt=2026-01-01/part-1.parquet") == b"one"
    assert s3_store.list_keys("a/") == [
        "a/dt=2026-01-01/part-1.parquet",
        "a/dt=2026-01-02/part-2.parquet",
    ]
    assert s3_store.list_keys("") == [
        "a/dt=2026-01-01/part-1.parquet",
        "a/dt=2026-01-02/part-2.parquet",
        "b/x.json",
    ]
    s3_store.delete("b/x.json")
    assert s3_store.list_keys("b/") == []
    assert s3_store.list_keys("nothing-here/") == []


def test_s3_keys_live_under_the_prefix(
    s3_client: S3Client, s3_bucket: str, s3_store: S3Storage
) -> None:
    s3_store.put_atomic("a/x.json", b"{}")
    listing = s3_client.list_objects_v2(Bucket=s3_bucket)
    assert [o["Key"] for o in listing["Contents"]] == ["lake/a/x.json"]
    assert s3_store.uri("a/x.json") == f"s3://{s3_bucket}/lake/a/x.json"
    assert S3Storage(s3_bucket, "", client=s3_client).uri("a/x.json") == (
        f"s3://{s3_bucket}/a/x.json"
    )


def test_s3_list_keys_prefix_is_a_directory_not_a_string_prefix(s3_store: S3Storage) -> None:
    s3_store.put_atomic("ping/x.json", b"{}")
    s3_store.put_atomic("ping2/y.json", b"{}")
    assert s3_store.list_keys("ping/") == ["ping/x.json"]
    assert s3_store.list_keys("ping") == ["ping/x.json"]


def test_s3_list_keys_ignores_other_prefixes_in_the_bucket(
    s3_client: S3Client, s3_bucket: str, s3_store: S3Storage
) -> None:
    s3_client.put_object(Bucket=s3_bucket, Key="lake-other/x.json", Body=b"{}")
    s3_client.put_object(Bucket=s3_bucket, Key="elsewhere/x.json", Body=b"{}")
    s3_store.put_atomic("a/x.json", b"{}")
    assert s3_store.list_keys("") == ["a/x.json"]


def test_s3_list_keys_hides_dot_files(
    s3_client: S3Client, s3_bucket: str, s3_store: S3Storage
) -> None:
    s3_store.put_atomic("a/real.json", b"{}")
    s3_client.put_object(Bucket=s3_bucket, Key="lake/a/.real.json.tmp", Body=b"x")
    assert s3_store.list_keys("a/") == ["a/real.json"]


def test_s3_read_missing_key_is_file_not_found(s3_store: S3Storage) -> None:
    with pytest.raises(FileNotFoundError):
        s3_store.read_bytes("nope/x.json")


def test_s3_put_exclusive_never_overwrites(s3_store: S3Storage) -> None:
    s3_store.put_exclusive("s/v1.json", b"first")
    with pytest.raises(FileExistsError):
        s3_store.put_exclusive("s/v1.json", b"second")
    assert s3_store.read_bytes("s/v1.json") == b"first"


def test_s3_put_exclusive_has_exactly_one_winner_under_contention(s3_store: S3Storage) -> None:
    outcomes: list[str] = []
    barrier = threading.Barrier(6)

    def claim(n: int) -> None:
        barrier.wait()
        try:
            s3_store.put_exclusive("s/v1.json", f"writer-{n}".encode())
        except FileExistsError:
            outcomes.append("lost")
        else:
            outcomes.append("won")

    threads = [threading.Thread(target=claim, args=(n,)) for n in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert outcomes.count("won") == 1
    assert outcomes.count("lost") == 5


def _client_error(code: str, status: int) -> Exception:
    client_error = pytest.importorskip("botocore.exceptions").ClientError
    return client_error(
        {"Error": {"Code": code, "Message": "x"}, "ResponseMetadata": {"HTTPStatusCode": status}},
        "PutObject",
    )


class _StubClient:
    """Just enough of a boto3 client to make one call fail."""

    def __init__(self, error: Exception) -> None:
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def put_object(self, **kwargs: Any) -> None:
        self.calls.append(kwargs)
        raise self._error

    def delete_object(self, **kwargs: Any) -> None:
        raise self._error


@pytest.mark.parametrize(("code", "status"), [("PreconditionFailed", 412), ("Whatever", 412)])
def test_s3_put_exclusive_maps_a_412_to_file_exists(code: str, status: int) -> None:
    stub = _StubClient(_client_error(code, status))
    store = S3Storage("some-bucket", "lake", client=stub)  # type: ignore[arg-type]
    with pytest.raises(FileExistsError):
        store.put_exclusive("s/v1.json", b"x")
    assert len(stub.calls) == 1  # a 412 is final: no retry
    assert stub.calls[0]["IfNoneMatch"] == "*"


class _ScriptedClient:
    """Answers put_object from a list: an Exception is raised, anything else succeeds."""

    def __init__(self, outcomes: list[object]) -> None:
        self._outcomes = list(outcomes)
        self.calls = 0

    def put_object(self, **kwargs: Any) -> None:
        self.calls += 1
        outcome = self._outcomes.pop(0)
        if isinstance(outcome, Exception):
            raise outcome


@pytest.fixture
def sleeps(monkeypatch: pytest.MonkeyPatch) -> list[float]:
    recorded: list[float] = []
    monkeypatch.setattr(storage_module.time, "sleep", recorded.append)
    return recorded


def test_s3_put_exclusive_retries_a_409_and_then_succeeds(sleeps: list[float]) -> None:
    conflict = _client_error("ConditionalRequestConflict", 409)
    client = _ScriptedClient([conflict, conflict, "ok"])
    store = S3Storage("some-bucket", "lake", client=client)  # type: ignore[arg-type]
    store.put_exclusive("s/v1.json", b"x")  # claimed after two conflicts
    assert client.calls == 3
    assert sleeps == pytest.approx([0.05, 0.10])


def test_s3_put_exclusive_retried_409_then_412_means_the_key_exists(sleeps: list[float]) -> None:
    client = _ScriptedClient(
        [_client_error("ConditionalRequestConflict", 409), _client_error("PreconditionFailed", 412)]
    )
    store = S3Storage("some-bucket", "lake", client=client)  # type: ignore[arg-type]
    with pytest.raises(FileExistsError):
        store.put_exclusive("s/v1.json", b"x")
    assert client.calls == 2


def test_s3_put_exclusive_gives_up_on_a_persistent_409_with_the_original_error(
    sleeps: list[float],
) -> None:
    conflict = _client_error("ConditionalRequestConflict", 409)
    client = _ScriptedClient([conflict] * 6)
    store = S3Storage("some-bucket", "lake", client=client)  # type: ignore[arg-type]
    with pytest.raises(type(conflict)) as excinfo:
        store.put_exclusive("s/v1.json", b"x")
    assert excinfo.value is conflict  # not converted to FileExistsError
    assert client.calls == 6  # the first try plus five retries
    assert sleeps == pytest.approx([0.05, 0.10, 0.15, 0.20, 0.25])


def test_s3_put_exclusive_lets_other_errors_through() -> None:
    stub = _StubClient(_client_error("AccessDenied", 403))
    store = S3Storage("some-bucket", "lake", client=stub)  # type: ignore[arg-type]
    with pytest.raises(type(stub._error)):
        store.put_exclusive("s/v1.json", b"x")


def test_s3_failed_put_leaves_no_object(s3_client: S3Client, s3_bucket: str) -> None:
    stub = _StubClient(RuntimeError("connection reset"))
    store = S3Storage(s3_bucket, "lake", client=stub)  # type: ignore[arg-type]
    with pytest.raises(RuntimeError, match="connection reset"):
        store.put_atomic("a/dt=2026-01-01/part-1.parquet", b"data")
    assert "Contents" not in s3_client.list_objects_v2(Bucket=s3_bucket)


def test_s3_put_atomic_refuses_a_file_over_the_single_put_limit(
    s3_client: S3Client, s3_bucket: str, s3_store: S3Storage, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(storage_module, "_MAX_SINGLE_PUT_BYTES", 10)
    s3_store.put_atomic("a/ok.parquet", b"0123456789")  # exactly at the limit
    with pytest.raises(ValueError, match="single S3 PutObject"):
        s3_store.put_atomic("a/big.parquet", b"0123456789A")
    assert s3_store.list_keys("a/") == ["a/ok.parquet"]


def test_s3_delete_failure_is_an_oserror() -> None:
    stub = _StubClient(_client_error("AccessDenied", 403))
    store = S3Storage("some-bucket", "lake", client=stub)  # type: ignore[arg-type]
    with pytest.raises(OSError, match="could not delete"):
        store.delete("a/x.parquet")


def test_s3_list_keys_reads_every_page() -> None:
    class Paginator:
        def paginate(self, **kwargs: Any) -> list[dict[str, Any]]:
            assert kwargs["Prefix"] == "lake/a/"
            return [
                {"Contents": [{"Key": "lake/a/2.json"}]},
                {},  # a page without Contents
                {"Contents": [{"Key": "lake/a/1.json"}]},
            ]

    class Client:
        def get_paginator(self, name: str) -> Paginator:
            assert name == "list_objects_v2"
            return Paginator()

    store = S3Storage("some-bucket", "lake", client=Client())  # type: ignore[arg-type]
    assert store.list_keys("a/") == ["a/1.json", "a/2.json"]


@pytest.mark.parametrize("key", ["../x", "a//b", "a/../b", ""])
def test_s3_keys_cannot_leave_the_prefix(s3_store: S3Storage, key: str) -> None:
    with pytest.raises(ValueError):
        s3_store.put_atomic(key, b"x")
    with pytest.raises(ValueError):
        s3_store.put_exclusive(key, b"x")
    with pytest.raises(ValueError):
        s3_store.read_bytes(key)
    with pytest.raises(ValueError):
        s3_store.delete(key)
