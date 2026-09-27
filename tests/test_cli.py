from __future__ import annotations

from pathlib import Path
from typing import ClassVar

import pytest
from _helpers import utc

from eventlake.cli import main
from eventlake.event import Event
from eventlake.writer import Writer


class Ping(Event):
    event_type: ClassVar[str] = "ping"

    source: str


def test_describe_on_empty_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["describe", str(tmp_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "(no events)" in out


def test_describe_on_populated_root(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
        writer.write(Ping(occurred_at=utc(2026, 1, 2), source="b"))

    exit_code = main(["describe", str(tmp_path)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "ping" in out
    assert "event_type" in out  # header


def test_compact_command(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="b"))

    exit_code = main(["compact", str(tmp_path), "ping", "2026-01-01"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "compacted ping dt=2026-01-01" in out
    assert "2 file(s) -> 1 file(s)" in out

    files = list((tmp_path / "ping" / "dt=2026-01-01").glob("part-*.parquet"))
    assert len(files) == 1


def test_schema_command_with_no_schema(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    exit_code = main(["schema", str(tmp_path), "ping"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "no schema recorded" in out


def test_schema_command_shows_versions_and_diff(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    with Writer(tmp_path) as writer:
        writer.write(Ping(occurred_at=utc(2026, 1, 1), source="a"))

    class PingV2(Event):
        event_type: ClassVar[str] = "ping"

        source: str
        note: str | None = None

    with Writer(tmp_path) as writer:
        writer.write(PingV2(occurred_at=utc(2026, 1, 2), source="b", note="hi"))

    exit_code = main(["schema", str(tmp_path), "ping"])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "v1 (" in out
    assert "v2 (" in out
    assert "diff from previous version" in out
    assert "+ note" in out


def test_main_requires_a_command(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        main([])
