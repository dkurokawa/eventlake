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

To keep a lake in S3, install the `s3` extra (boto3, 1.35.2 or newer): `pip install 'eventlake[s3]'`
(or `uv sync --extra s3` from a clone). Without it nothing else changes - local
roots never import boto3.

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

`root` is a directory, or `s3://bucket/prefix` (see "S3" below). `Writer`,
`Lake`, `SchemaRegistry` and `compact()` all take it the same way, and accept a
keyword-only `storage=` (an `eventlake.Storage`) instead when you want to pass
the backend yourself; giving both is a `ValueError`.

- Events are buffered **per Python class**, not per `event_type` string, and
  flushed to Parquet when a class's buffer reaches `max_rows`, or when the
  `with` block exits (or `.flush()` is called explicitly). If two different
  classes share an `event_type` (a V1 and a V2 in the same process, say),
  each flushes with its own class's Arrow schema, and the two are
  reconciled through the normal schema-compatibility rules below - a
  compatible V2 becomes a new version, an incompatible one raises.
- Files are written to a temp path and renamed into place, so a crash
  mid-write never leaves a half-written file where a reader could see it -
  only the temp file is cleaned up on failure (on S3 each file is a single
  `PutObject` instead - see "S3"). The partition directory
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
- Writing the same `event_id` twice while it is still buffered silently drops
  the second copy. Once a batch is flushed the `Writer` forgets those ids (so
  a long-lived writer's memory stays bounded); duplicates across flushes, or
  from two different `Writer` runs, are removed when the lake is read - see
  below.
- A `Writer` is safe to share between threads, and `close()` flushes and then
  refuses further writes.
- **Multiple `Writer`s appending to the same root at the same time is fine**,
  including when they're registering different, concurrent schema changes
  for the same `event_type` (see "Schema evolution" below). Running
  `compact()` on a partition a `Writer` is actively targeting is not - see
  "Compaction" below.

### Using it in an application

Create one `Writer` when the process starts, share it, and close it on
shutdown. Record an event at the point where something happens - next to the
code that changes the database row, not by reading the row back later.

```python
# app startup
writer = Writer("/var/lib/myapp/events", max_rows=500)


# in a request handler, right where the state change happens
def correct_label(item_id: str, old: str, new: str, reviewer: str) -> None:
    db.update_label(item_id, new)  # the database keeps the current value
    writer.write(
        LabelCorrected(  # the lake keeps what happened
            item_id=item_id,
            old_label=old,
            new_label=new,
            reviewer=reviewer,
            occurred_at=datetime.now(timezone.utc),
        )
    )


# a timer (every few seconds) and on shutdown
writer.flush()
writer.close()
```

Events sit in memory until they are flushed, so a crash loses what was
buffered. Pick `max_rows` and the flush interval for how much you can afford
to lose; for events you must never lose, call `flush()` after writing them.

### Where this fits next to your database

| Approach | What you get | What you don't |
|---|---|---|
| Snapshot export (e.g. RDS → S3 → Athena) | The current state of every row, periodically | Anything overwritten or deleted between snapshots; the order things happened in |
| Change data capture (e.g. Debezium, DMS on the binlog/WAL) | Every row change, without touching application code | *Why* it changed and who did it - only the new column values |
| **Application events (this library)** | What happened, in the application's own terms, with the context the code has at that moment | Changes made outside the application (manual SQL, other services) |

They are complementary: snapshots answer "what is true now", events answer
"how did it get this way". eventlake is for the second question when the
application is the one that knows the answer.

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

The same layout is used under an `s3://bucket/prefix` root.

### S3

```python
# pip install 'eventlake[s3]'
with Writer("s3://my-bucket/events") as writer:
    writer.write(event)

lake = Lake("s3://my-bucket/events")
```

```bash
eventlake describe s3://my-bucket/events
eventlake compact s3://my-bucket/events order_placed 2026-01-01
```

Everything above works the same on an `s3://` root: same API, same layout,
same results. The rest of this section is what differs.

- **Credentials, region and endpoint come from boto3's standard chain**
  (environment variables, shared config, roles). Set `AWS_ENDPOINT_URL_S3`
  (or `AWS_ENDPOINT_URL`) to point at an S3-compatible server instead of AWS;
  eventlake has no settings of its own for this. The role needs
  `s3:GetObject`, `s3:PutObject` and `s3:ListBucket` on the prefix, plus
  `s3:DeleteObject` for `compact()`.
- **Reads go through DuckDB's `httpfs` extension.** DuckDB downloads that
  extension the first time it is installed, so the first `Lake(...)` on a
  machine needs network access to DuckDB's extension server. The credentials
  are handed to DuckDB as a session secret scoped to the root's `s3://` path;
  they are not written to disk and are kept out of exception messages.
  On the standard AWS endpoints the region DuckDB uses is the bucket's own
  (looked up with `HeadBucket`), not the client's. Any other endpoint - a VPC
  endpoint, MinIO, moto - is used exactly as the client has it, with the bucket
  in the path; an endpoint URL with a path in it (`https://host/prefix`) is
  refused, because DuckDB would drop the path and read somewhere else.
- **A `Lake` uses the credentials it was created with.** They are read once,
  when the `Lake` is constructed. With temporary credentials (an assumed role)
  that expire, create a new `Lake` instead of holding one for hours. (A
  `Writer` uses boto3's client directly, which refreshes its own credentials.)
- **Guarantees.** Same statements as for a local root, achieved differently:

  | Guarantee | Local root | S3 root |
  |---|---|---|
  | A reader never sees a partly written Parquet file | temp file, then rename | one `PutObject` per file (an object appears whole or not at all); files are serialized in memory first |
  | A schema version is never overwritten by a concurrent `Writer` | exclusive create (`os.link`) | `PutObject` with `If-None-Match: *` (S3 conditional writes); a `412` or `409` response means someone else claimed the version, and registration retries |
  | Reading, `describe()` and `compact()` give the same results | - | tested with the same test suite on both |
  | A lake written locally can be uploaded (`aws s3 sync`) and read | - | tested: local files copied object-for-object, then read through `s3://` |
  | The layout is plain Hive partitioning | yes | the same keys under the prefix |

- **Limits.** A single file above 5 GB (S3's single-`PutObject` limit, measured
  after serialization) raises `ValueError` before anything is sent; eventlake
  does not use multipart uploads, which is also why it never leaves an
  unfinished multipart upload behind. With the default `max_rows` of 10,000 a
  file is nowhere near that. Lowering `max_rows` is the remedy if you get there.
- **Tested against [moto](https://github.com/getmoto/moto)'s in-process S3
  server only.** The conditional write needs an S3 (or compatible) server that
  supports `If-None-Match` on `PutObject`; that is how AWS S3 behaves since
  August 2024, and how moto 5.0.15+ behaves. Other servers are unverified.

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
of swallowing the problem. Its `new_file` and `leftover_files` are keys
relative to the root - `str`, e.g. `ping/dt=2026-01-01/part-<uuid>.parquet` -
on every backend; before S3 support they were `pathlib.Path`s, so code that
compared them to paths needs to compare to keys now. In that state, both the new file and the
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

`<root>` is a directory or an `s3://bucket/prefix` (see "S3").

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

- Object storage other than S3 (GCS, Azure Blob). The `Storage` interface
  leaves room for them; nothing implements it.
- Multipart uploads: a single file above 5 GB on S3 is refused (see "S3").
- Refreshing credentials on a long-lived `Lake` (see "S3").
- Creating buckets or setting lifecycle rules: the bucket is yours.
- Athena / Glue table definitions. The layout is plain Hive partitioning, but
  eventlake has not been run against Athena or Spark.
- Concurrent writers compacting the same partition at the same time, and
  `compact()` running while a `Writer` targets the same partition - on S3 as
  on a local root.
- Wildcard characters in a local root's path: DuckDB expands `*`, `?`, `[` and
  `]` in the file names it reads, so a local root whose path contains them can
  read the wrong files. (An `s3://` prefix with such characters is refused.)
- Untrusted lake roots. The root is assumed to be a directory you control:
  eventlake follows symlinks inside it, so a symlink planted there could
  redirect writes (or `compact`'s deletes) outside the root.
- PyPI publishing - install from a clone for now.

## Development

```bash
uv sync --all-groups
uv run ruff check .
uv run ruff format --check .
uv run mypy --strict eventlake
uv run pytest --cov=eventlake --cov-fail-under=90
```

The storage-dependent tests run twice, once on a temporary directory and once
on an `s3://` root in a fresh bucket on an in-process moto server bound to
127.0.0.1 (dummy credentials; the tests refuse any other endpoint). They skip
themselves when boto3 or moto is not installed, so the local suite needs
neither. The first S3 read on a machine downloads DuckDB's `httpfs` extension.

CI also runs a second job against `uv sync --resolution lowest-direct` - every
dependency pinned to the lowest version its constraint in `pyproject.toml`
allows. That floor is not a guess: `duckdb>=1.5.0`, `pyarrow>=16.0.0`,
`pydantic>=2.5.0`, `hypothesis>=6.70.0`, `boto3>=1.35.2` (the first release
whose `PutObject` accepts `IfNoneMatch`) and `moto>=5.0.15` (the first whose
S3 server enforces it) are each the lowest version this
project actually installed and ran its full test suite against (see the
comments next to each in `pyproject.toml` for what broke below it).
A third job removes boto3 and moto after installing and runs the suite, to keep
the local backend free of them.

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
(乱数で合成したデータのみを使用)。保存先はローカルのディレクトリか、`s3://bucket/prefix`
（`pip install 'eventlake[s3]'` が必要。`Writer` / `Lake` / `compact` / CLI が同じ API で使えます）。
S3 では、Parquet は 1 回の `PutObject` で書き（途中まで書かれたファイルが読み手に見えない）、
スキーマ版の排他作成は `If-None-Match: *` の条件付き書き込みで行います。5 GB を超える 1 ファイルは
書く前にエラーにし、`Lake` は作成時点の資格情報を使います。S3 のテストは moto（プロセス内の
S3 サーバ）に対してのみ行っています。マルチパートアップロード、S3 以外のストレージ、
Athena / Glue の定義と PyPI 公開は対象外です。保存先は自分が管理する場所であることを前提にしています
（ローカルではシンボリックリンクをたどるため、信頼できない場所を保存先にしないでください）。

アプリへの組み込みは、プロセスの起動時に `Writer` を1つ作って共有し（スレッドから同時に呼んでよい）、
DB の行を書き換える箇所のすぐ隣でイベントを `write()` し、タイマーと終了時に `flush()` / `close()` します。
DB の写し（RDS → S3 → Athena など）は「今の値」を、CDC は「行の変化」を残せますが、
「なぜ・誰が変えたか」はアプリのイベントにしか残りません。eventlake はその層を担います。
