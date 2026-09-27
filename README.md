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

**Every `datetime` field is timezone-required and UTC-normalized** - not
just the two above, and not just scalar fields: a subclass's own
`run_at: datetime`, `cancelled_at: datetime | None`, or `moments:
list[datetime]` fields (every element of the list) all get the same
treatment. A naive `datetime` anywhere in there is a `ValidationError`.

A subclass must declare `event_type: ClassVar[str]` matching
`^[a-z][a-z0-9_]{0,63}$` (it becomes a directory name, a SQL view name, and
part of a file glob, so it's kept to a boring, safe charset - no `.`, `/`,
uppercase, or leading digit). Leaving it out, or giving it a name that
doesn't match, is a `TypeError` as soon as the class is defined. The same
pattern is enforced on any event_type string handed to `compact()` or the
CLI directly, and on the partition date (`dt`, `YYYY-MM-DD`) `compact()`
and the CLI take - both are validated before they're ever used to build a
filesystem path, so `"../escape"` or a malformed date can't point outside
the root or write/delete files it shouldn't.

No field may be named starting with `__eventlake` (also a `TypeError` at
class-definition time) - that prefix is reserved for a synthetic column
`Lake`/`compact()` add internally to break a rare dedup tie deterministically
(see "Reading" below); a real field named plain `filename` is unaffected by
it and round-trips normally.

Supported field types: `str`, `int`, `float`, `bool`, timezone-aware
`datetime`, `date`, `UUID`, `Enum` subclasses, `list[...]` of any of those
(UUIDs and Enums become strings, like they do standalone), and any of these
wrapped in `X | None`. Anything else raises `UnsupportedFieldType` - **at
class-definition time**, not on first write; a bad field type is a
`TypeError` from the `class Foo(Event):` statement itself.

### Writing (`eventlake.writer.Writer`)

```python
with Writer(root, max_rows=10_000, allow_breaking=False) as writer:
    writer.write(event)
    writer.write_many(events)
```

- Events are buffered **per Python class**, not per `event_type` string, and
  flushed to Parquet when a class's buffer reaches `max_rows`, or when the
  `with` block exits (or `.flush()` is called explicitly). If two different
  classes share an `event_type` (a V1 and a V2 in the same process, say),
  each flushes with its own class's Arrow schema, and the two are
  reconciled through the normal schema-compatibility rules below - a
  compatible V2 becomes a new version, an incompatible one raises.
- Files are written to a temp path and renamed into place, so a crash
  mid-write never leaves a half-written file where a reader could see it -
  only the temp file is cleaned up on failure. The partition directory
  itself (possibly now empty, if this `Writer` just created it) is left in
  place rather than removed: deleting it would race with another `Writer`
  concurrently targeting the same partition. `Lake` already ignores an
  event-type directory with no Parquet files in it (see "Reading" below),
  so an empty one left behind like this is harmless.
- If a flush spans multiple days (multiple `occurred_at` partitions) and
  fails partway through, the days that were already written are removed
  from the buffer immediately - a later retry only re-attempts what didn't
  make it, instead of writing duplicate files for partitions that already
  succeeded.
- Writing the same `event_id` twice through the same `Writer` silently drops
  the second copy. Duplicates that come from two different `Writer` runs
  (e.g. a retried batch job) aren't caught here - see below.
- **Multiple `Writer`s appending to the same root at the same time is fine**,
  including when they're registering different, concurrent schema changes
  for the same `event_type` (see "Schema evolution" below). Running
  `compact()` on a partition a `Writer` is actively targeting is not - see
  "Compaction" below.

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

**Registering a version is safe across concurrent writers.** Each version
is claimed with an exclusive file create (`v<N>.json` is never overwritten -
see `SchemaRegistry._write`), so two `Writer`s racing to register a
compatible change against the same `event_type` can't silently clobber one
another: whichever loses the race re-reads the version the winner just
created and redoes its compatibility check against that, up to 10 times
before giving up with `SchemaRegistrationRace`. This only resolves cleanly
when the loser's schema is itself a compatible extension of whatever the
winner just registered (e.g. it's a superset); two independently-evolving,
mutually incompatible schemas racing for the same `event_type` will still
correctly raise for whichever one turns out not to fit - concurrent writers
aren't a way around the compatibility rules above, just a way to not lose a
version to a lost race.

A schema that exactly matches *any* already-registered version - not just
the latest one - is written under that version with no compatibility check
at all, so mixing two classes' flushes for the same `event_type` is
order-independent: if v1 and v2 are both already registered, flushing a v2
event and then a v1 event both succeed, regardless of which is "latest".

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
  earliest `recorded_at`. Ties on `recorded_at` are broken by the source
  Parquet file path, read into a column reserved for this purpose under the
  name `__eventlake_source_file` (see "Events" above for why it isn't just
  called `filename`) and excluded from the result - not `event_id`, which is
  constant across every row in that tiebreak window and so can't distinguish
  anything - a deterministic but otherwise meaningless fallback that only
  matters when two writes landed at the exact same `recorded_at`. This is
  what catches the cross-`Writer` duplicates that writing doesn't.
- Every event type is also available as a same-named SQL view (built with
  the same dedup rule), so arbitrary joins across event types work with
  plain `lake.sql(...)`.
- `state_as_of(event_type, key=..., at=...)` reconstructs "what did we
  believe was true, as of `at`" by taking, for each distinct value of `key`,
  the event with the latest `occurred_at <= at` (ties broken by the most
  recent `recorded_at`, then - here `event_id` *does* vary within the
  window, so it's a meaningful final tiebreak - by `event_id`, so the result
  is always deterministic). Leave `at` unset to get the latest state overall.
- `since`, `until`, and `at` must be timezone-aware `datetime`s. A naive one
  raises `ValueError` rather than silently being treated as local time -
  `Lake` never guesses which timezone you meant.

Relations returned by `Lake` carry timezone-aware (UTC) timestamp columns.
`.fetchall()`, `.to_arrow_table()` and `.df()` all work; `pytz` is a
dependency because DuckDB needs it to hand timezone-aware values to Python.

### Compaction (`eventlake.compact`)

Every flush writes a new file, so a busy event type accumulates many small
files per day. `compact(root, event_type, dt)` merges one partition's files
into one, dropping duplicate `event_id`s the same way `events()` does
(earliest `recorded_at`, ties broken by source file - see "Reading" above).
A single-file partition with no internal duplicates is left untouched; a
single file that *does* have duplicate `event_id`s in it (rare, but
possible) is still rewritten to remove them. Both `event_type` and `dt` are
validated (the same `EVENT_TYPE_PATTERN` rule, and a strict `YYYY-MM-DD` for
`dt`) before either is used to build a path - `compact(root, "../x", ...)`
or a malformed `dt` raises `ValueError` rather than touching anything.

It writes the merged file and renames it into place *before* deleting the
old files, so a crash while writing it never loses rows: either the
original files are all still there, or the new file is in place. If the new
file is in place but deleting one of the old files then fails (permissions,
a concurrent process holding it open, ...), `compact()` raises
`CompactionIncomplete` (with the list of files it couldn't remove) instead
of swallowing the problem. In that state, both the new file and the
leftover old file(s) exist together; reading through `Lake` still produces
correct, deduplicated results either way, since the leftovers just look
like ordinary cross-file duplicates to `events()`. Re-running `compact()`
on the same partition picks the leftover(s) up and finishes the cleanup.

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
what the believed-correct label was as of a specific point in time. With the
seed fixed in the script, the output is exactly:

```
=== eventlake label history demo ===
predictions: 200, later corrected: 62
overall correction rate: 31.0%

correction rate by model version (a DB overwritten in place
could not tell you this - it never held the wrong label at all):
  v1: 43/95 corrected (45.3%)
  v2: 19/105 corrected (18.1%)

as of 2026-01-02T00:00:00+00:00, 19 corrections were known
(a mutable 'current label' table cannot answer this at all - it only
 ever holds the latest value, with no record of when it changed)
```

## Not covered

- Object storage backends (`s3://` and similar) - local filesystem only, for now.
- Concurrent writers compacting the same partition at the same time.
- Untrusted lake roots. The root is assumed to be a directory you control:
  eventlake follows symlinks inside it, so a symlink planted there could
  redirect writes (or `compact`'s deletes) outside the root.
- PyPI publishing - install from a clone for now.

## Development

```bash
uv sync
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict eventlake
uv run pytest --cov=eventlake --cov-fail-under=90
```

CI also runs a second job against `uv sync --resolution lowest-direct` - every
dependency pinned to the lowest version its constraint in `pyproject.toml`
allows. That floor is not a guess: `duckdb>=1.5.0`, `pyarrow>=16.0.0`,
`pydantic>=2.5.0`, and `hypothesis>=6.70.0` are each the lowest version this
project actually installed and ran its full test suite against (see the
comments next to each in `pyproject.toml` for what broke below it).

## License

MIT - see [LICENSE](LICENSE).

---

## 日本語要約

eventlake は、型付きイベントを **追記だけ** で日付パーティション付きの Parquet に残し、
DuckDB で「任意時点の状態」を組み立て直せる小さなデータレイクです。

アプリの DB は「今の値」を上書きしていくため、過去の値・削除・変更の順序が失われます。
ML の学習データや分析に必要なのは、まさにその履歴です。eventlake では:

- イベントは `pydantic` の型で検証してから受け付ける。日時フィールドは（`list[datetime]` の
  各要素・ユーザー定義のものも含め）すべてタイムゾーン必須で UTC に正規化し、非対応の型は
  クラス定義した瞬間にエラーにする。`event_type` は安全な文字種のみに制限し、`compact()` や
  CLI が受け取る `event_type` ・パーティション日付もパスに使う前に同じ規則で検証する
  （`../` 等でルート外にファイルを作成・削除できないようにする）
- `occurred_at`（実際に起きた日、書き込んだ日ではない）の UTC 日付でパーティション分割
- スキーマの変更（フィールド追加・削除・型変更）を書き込み時に検知し、読む側を壊す変更は
  既定で拒否する（`allow_breaking=True` で許可も可能）。バージョン登録は排他制御されており、
  複数の Writer が同時に書き込んでも版が失われたり壊れたりしない。既存のいずれかの版と
  完全に一致するスキーマは、その版としてそのまま書けるので、同じ event_type の V1/V2 を
  混ぜて書く順序にも依存しない
- 読み込みは DuckDB 経由なので、`lake.state_as_of(...)` のような「ある時点の状態」を
  SQL 感覚で問い合わせられる。`since`/`until`/`at` にタイムゾーンなしの日時を渡すとエラーになる
  （暗黙にローカル時刻として扱わない）

`examples/label_history.py` は、ML の予測とラベル修正の履歴を題材にした自己完結のデモです
(乱数で合成したデータのみを使用)。ローカルファイルシステム専用で、S3 等のクラウド対応と
PyPI 公開は対象外です。保存先は自分が管理するディレクトリであることを前提にしています
（中のシンボリックリンクはたどるため、信頼できない場所を保存先にしないでください）。
