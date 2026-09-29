"""Shared fixtures: an in-process moto S3 server for the S3 backend's tests.

Nothing here can reach real AWS: every client points at a 127.0.0.1 moto
server with dummy credentials, and the environment is scrubbed so boto3
neither reads ~/.aws nor asks the instance metadata service for anything.
The S3 tests are skipped when boto3 / moto are not installed, so the local
tests run without either.
"""

from __future__ import annotations

import socket
import uuid
from collections.abc import Iterator
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    from mypy_boto3_s3 import S3Client

_LOCAL_ENDPOINT_PREFIXES = ("http://127.0.0.1", "http://localhost")


@pytest.fixture(autouse=True)
def _isolate_aws_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in (
        "AWS_PROFILE",
        "AWS_DEFAULT_PROFILE",
        "AWS_SESSION_TOKEN",
        "AWS_ENDPOINT_URL",
        "AWS_ENDPOINT_URL_S3",
        "AWS_ROLE_ARN",
        "AWS_WEB_IDENTITY_TOKEN_FILE",
    ):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    monkeypatch.setenv("AWS_DEFAULT_REGION", "us-east-1")
    monkeypatch.setenv("AWS_EC2_METADATA_DISABLED", "true")
    # Files that do not exist: boto3 must not fall back to a real ~/.aws.
    monkeypatch.setenv("AWS_SHARED_CREDENTIALS_FILE", str(tmp_path / "no-such-credentials"))
    monkeypatch.setenv("AWS_CONFIG_FILE", str(tmp_path / "no-such-config"))


@pytest.fixture(scope="session")
def moto_endpoint() -> Iterator[str]:
    pytest.importorskip("boto3")
    pytest.importorskip("moto")
    from moto.server import ThreadedMotoServer

    # A free port picked here rather than `port=0` + get_host_and_port(),
    # which older moto releases don't have.
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    server = ThreadedMotoServer(ip_address="127.0.0.1", port=port, verbose=False)
    server.start()
    endpoint = f"http://127.0.0.1:{port}"
    assert endpoint.startswith(_LOCAL_ENDPOINT_PREFIXES), endpoint
    try:
        yield endpoint
    finally:
        server.stop()


@pytest.fixture
def s3_client(moto_endpoint: str, monkeypatch: pytest.MonkeyPatch) -> S3Client:
    import boto3

    # The same endpoint `open_storage("s3://...")` picks up (boto3 standard
    # variable), so URI-based roots also land on moto.
    monkeypatch.setenv("AWS_ENDPOINT_URL_S3", moto_endpoint)
    client = boto3.client(
        "s3",
        region_name="us-east-1",
        endpoint_url=moto_endpoint,
        aws_access_key_id="testing",
        aws_secret_access_key="testing",
    )
    # Never a real AWS endpoint.
    assert str(client.meta.endpoint_url).startswith(_LOCAL_ENDPOINT_PREFIXES)
    return client


@pytest.fixture
def s3_bucket(s3_client: S3Client) -> Iterator[str]:
    bucket = f"eventlake-test-{uuid.uuid4().hex[:12]}"
    s3_client.create_bucket(Bucket=bucket)
    try:
        yield bucket
    finally:
        paginator = s3_client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=bucket):
            for item in page.get("Contents", []):
                s3_client.delete_object(Bucket=bucket, Key=item["Key"])
        s3_client.delete_bucket(Bucket=bucket)


@pytest.fixture(params=["local", "s3"])
def lake_root(request: pytest.FixtureRequest, tmp_path: Path) -> str | Path:
    """A fresh, empty eventlake root on each backend: a directory, or an
    `s3://` URI in a fresh moto bucket (boto3 finds moto through the standard
    endpoint variable). Tests taking this run once per backend."""
    if request.param == "local":
        return tmp_path
    bucket: str = request.getfixturevalue("s3_bucket")
    return f"s3://{bucket}/lake"
