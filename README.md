# eventlake

[![CI](https://github.com/dkurokawa/eventlake/actions/workflows/ci.yml/badge.svg)](https://github.com/dkurokawa/eventlake/actions/workflows/ci.yml)

A small event lake: typed events, appended (never overwritten) to
date-partitioned Parquet, queryable as of any point in time via DuckDB.

## Why

An application database keeps "the current value" - it overwrites rows as
they change. That's exactly what you want for serving traffic, and exactly
what you don't want for training data or analysis: the history of what a
value *used to be*, when it changed, who changed it, and in what order is
gone the moment it's overwritten.

eventlake gives an application a place to record what happened, as typed
events, and never mutates what it's already written:

- Every event is validated against a `pydantic` model before it's accepted.
- Events are appended to Parquet files, partitioned by the UTC date they
  occurred on (not the date they were written).
- Changing an event type's shape is detected at write time: additions that
  don't break existing readers are versioned automatically; anything that
  would break them is rejected unless you say otherwise.
- Reading goes through DuckDB, so "what was true as of a given moment" is a
  query, not a re-implementation of your event log's replay logic.

## Install

Requires Python 3.11+. This repo is managed with [uv](https://docs.astral.sh/uv/):

```bash
uv sync
```

## Quick tour

```python
from datetime import datetime, timezone
from typing import ClassVar

from eventlake.event import Event
from eventlake.writer import Writer
from eventlake.lake import Lake


class OrderPlaced(Event):
    event_type: ClassVar[str] = "order_placed"

    order_id: str
    amount: float


with Writer("./data") as writer:
    writer.write(
        OrderPlaced(
            order_id="o-1",
            amount=42.0,
            occurred_at=datetime.now(timezone.utc),
        )
    )

lake = Lake("./data")
print(lake.sql("SELECT order_id, amount FROM order_placed").fetchall())
```

### Events (`eventlake.event`)

Every event is a frozen, extra-forbidding `pydantic.BaseModel` subclassing
`Event`, with three fields eventlake manages plus whatever your event needs:

- `event_id: UUID` - defaults to a fresh `uuid4()`.
- `occurred_at: datetime` - required, must be timezone-aware; normalized to UTC.
- `recorded_at: datetime | None` - set by `Writer` when the event is written.

A subclass must declare `event_type: ClassVar[str]`; leaving it out is a
`TypeError` as soon as the class is defined.

Supported field types (anything else raises `UnsupportedFieldType` the first
time the class's Arrow schema is computed): `str`, `int`, `float`, `bool`,
timezone-aware `datetime`, `date`, `UUID`, `Enum` subclasses, `list[...]` of
any of the primitives above, and any of these wrapped in `X | None`.

### Writing (`eventlake.writer.Writer`)

```python
with Writer(root, max_rows=10_000, allow_breaking=False) as writer:
    writer.write(event)
    writer.write_many(events)
```

- Events are buffered per event type and flushed to Parquet when a type's
  buffer reaches `max_rows`, or when the `with` block exits (or `.flush()`
  is called explicitly).
- Files are written to a temp path and renamed into place, so a crash mid-write
  never leaves a half-written file where a reader could see it.
- Writing the same `event_id` twice through the same `Writer` silently drops
  the second copy. Duplicates that come from two different `Writer` runs
  (e.g. a retried batch job) aren't caught here - see below.

### Storage layout

```
<root>/
  _schemas/<event_type>/v<N>.json      # one file per schema version
  <event_type>/dt=YYYY-MM-DD/part-<uuid>.parquet
```

`dt` is the UTC date of `occurred_at`, not of the write - a late-arriving
event lands in the partition for when it actually happened. The layout is
plain Hive-style partitioning, so it's also readable directly from DuckDB,
Athena, or Spark without going through this library at all.

Only local paths are supported. Object storage (`s3://` etc.) is out of
scope for now (see "Not covered" below).

### Schema evolution (`eventlake.schema`)

Each write compares the event's current Arrow schema against the latest
registered version for that event type:

| Change | Compatible? |
| --- | --- |
| Add a nullable field | yes - new version registered |
| Loosen non-nullable -> nullable | yes - new version registered |
| Remove a field | no |
| Change a field's type | no |
| Add a non-nullable field | no |
| Tighten nullable -> non-nullable | no |

An incompatible change raises `SchemaChangeError` (with a human-readable
diff) unless the `Writer` was created with `allow_breaking=True`, in which
case it's registered anyway and a warning is logged. Old files keep their
old schema; reading merges schema versions by field name (DuckDB's
`union_by_name`), so a field that was later removed just shows up as
present-with-nulls once it's gone.

### Reading (`eventlake.lake.Lake`)

```python
lake = Lake(root)

lake.events("order_placed", since=start, until=end)  # a DuckDBPyRelation
lake.sql("SELECT * FROM order_placed WHERE amount > 100")
lake.state_as_of("order_placed", key="order_id", at=some_time)
lake.describe()  # per-type schema/partition/file/row counts
```

- `events()` only reads the Parquet partitions that fall inside
  `[since, until]`, and deduplicates by `event_id`, keeping the copy with the
  earliest `recorded_at` (ties broken by `event_id`). This is what catches
  the cross-`Writer` duplicates that writing doesn't.
- Every event type is also available as a same-named SQL view, so arbitrary
  joins across event types work with plain `lake.sql(...)`.
- `state_as_of(event_type, key=..., at=...)` reconstructs "what did we
  believe was true, as of `at`" by taking, for each distinct value of `key`,
  the event with the latest `occurred_at <= at` (ties broken by `recorded_at`,
  then `event_id`, so the result is always deterministic). Leave `at` unset
  to get the latest state overall.

Relations returned by `Lake` carry timezone-aware timestamp columns; convert
them with `.to_arrow_table()`, `.df()`, etc. rather than `.fetchall()` if
your DuckDB build needs `pytz` for that path (this project doesn't depend on
`pytz`, to keep it small).

### Compaction (`eventlake.compact`)

Every flush writes a new file, so a busy event type accumulates many small
files per day. `compact(root, event_type, dt)` merges one partition's files
into one, dropping duplicate `event_id`s the same way `events()` does. It
writes the merged file and renames it into place *before* deleting the old
files, so a crash mid-compaction either leaves the original files untouched
or leaves the merge complete - never a state with rows missing.

Running compaction concurrently with a writer targeting the same partition
is out of scope; run it when nothing is actively writing to that partition.

### CLI

```bash
eventlake describe <root>
eventlake compact <root> <event_type> <YYYY-MM-DD>
eventlake schema <root> <event_type>
```

## Example

`examples/label_history.py` is a self-contained demo (synthetic, seeded
random data - no external inputs) built around a common ML pipeline problem:
a model makes predictions, a human sometimes corrects them, and a "current
label" column in a normal database throws away the fact that the correction
ever happened. Run it with:

```bash
uv run python examples/label_history.py
```

It writes `PredictionMade` and `LabelCorrected` events, then uses DuckDB to
answer three questions a mutable table can't: what fraction of predictions
were later corrected, how that correction rate differs by model version, and
what the believed-correct label was as of a specific point in time.

## Not covered

- Object storage backends (`s3://` and similar) - local filesystem only, for now.
- Concurrent writers compacting the same partition at the same time.
- PyPI publishing and a hosted GitHub repository - this is a local, standalone project.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict eventlake
uv run pytest --cov=eventlake --cov-fail-under=90
```

## License

MIT - see [LICENSE](LICENSE).

---

## 日本語要約

eventlake は、型付きイベントを **追記だけ** で日付パーティション付きの Parquet に残し、
DuckDB で「任意時点の状態」を組み立て直せる小さなデータレイクです。

アプリの DB は「今の値」を上書きしていくため、過去の値・削除・変更の順序が失われます。
ML の学習データや分析に必要なのは、まさにその履歴です。eventlake では:

- イベントは `pydantic` の型で検証してから受け付ける
- `occurred_at`（実際に起きた日、書き込んだ日ではない）の UTC 日付でパーティション分割
- スキーマの変更（フィールド追加・削除・型変更）を書き込み時に検知し、読む側を壊す変更は
  既定で拒否する（`allow_breaking=True` で許可も可能）
- 読み込みは DuckDB 経由なので、`lake.state_as_of(...)` のような「ある時点の状態」を
  SQL 感覚で問い合わせられる

`examples/label_history.py` は、ML の予測とラベル修正の履歴を題材にした自己完結のデモです
(乱数で合成したデータのみを使用)。ローカルファイルシステム専用で、S3 等のクラウド対応、
PyPI 公開、GitHub リポジトリの作成は本プロジェクトの対象外です。
