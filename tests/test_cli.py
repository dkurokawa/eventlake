from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from _helpers import part_keys, utc

from eventlake.cli import main
from eventlake.event import Event
from eventlake.writer import Writer


class Ping(Event):
    event_type: ClassVar[str] = "ping"

    source: str


def test_describe_on_empty_root(lake_root: str | Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["describe", str(lake_root)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "(no events)" in out


def test_describe_on_populated_root(
    lake_root: str | Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with Writer(lake_root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
        writer.write(Ping(occurred_at=utc(2026, 1, 2), source="b"))

    exit_code = main(["describe", str(lake_root)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "ping" in out
    assert "event_type" in out  # header


def test_compact_command(lake_root: str | Path, capsys: pytest.CaptureFixture[str]) -> None:
    with Writer(lake_root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    with Writer(lake_root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="b"))

    exit_code = main(["compact", str(lake_root), "ping", "2026-01-01"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "compacted ping dt=2026-01-01" in out
    assert "2 file(s) -> 1 file(s)" in out

    assert len(part_keys(lake_root, "ping", "2026-01-01")) == 1


def test_schema_command_with_no_schema(
    lake_root: str | Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = main(["schema", str(lake_root), "ping"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "no schema recorded" in out


def test_schema_command_shows_versions_and_diff(
    lake_root: str | Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with Writer(lake_root) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    class PingV2(Event):
        event_type: ClassVar[str] = "ping"

        source: str
        note: str | None = None

    with Writer(lake_root) as writer:
        writer.write(PingV2(occurred_at=utc(2026, 1, 2), source="b", note="hi"))

    exit_code = main(["schema", str(lake_root), "ping"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "v1 (" in out
    assert "v2 (" in out
    assert "diff from previous version" in out
    assert "+ note" in out


@pytest.mark.parametrize("command", ["describe", "compact", "schema"])
def test_help_says_the_root_may_be_an_s3_uri(
    command: str, capsys: pytest.CaptureFixture[str]
) -> None:
    with pytest.raises(SystemExit) as excinfo:
        main([command, "--help"])
    assert excinfo.value.code == 0
    assert "s3://bucket/prefix" in " ".join(capsys.readouterr().out.split())


def test_main_requires_a_command(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([])


def test_compact_command_rejects_unsafe_event_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid event_type"):
        main(["compact", str(tmp_path), "../escape", "2026-01-01"])


def test_compact_command_rejects_unsafe_dt(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid partition date"):
        main(["compact", str(tmp_path), "ping", "../x"])
    with pytest.raises(ValueError, match="invalid partition date"):
        main(["compact", str(tmp_path), "ping", "2026-1-1"])


def test_schema_command_rejects_unsafe_event_type(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="invalid event_type"):
        main(["schema", str(tmp_path), "/abs/path"])
