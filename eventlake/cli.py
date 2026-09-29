"""Command-line interface: `eventlake describe|compact|schema`."""

from __future__ import annotations

import argparse
import sys

from .compact import compact
from .event import validate_event_type
from .lake import Lake
from .schema import SchemaRegistry, diff_schemas


def _cmd_describe(args: argparse.Namespace) -> int:
    lake = Lake(args.root)
    summaries = lake.describe()
    if not summaries:
        print("(no events)")
        return 0
    print(f"{'event_type':<30} {'schema_v':>8} {'partitions':>10} {'files':>6} {'rows':>10}")
    for summary in summaries:
        print(
            f"{summary.event_type:<30} {summary.schema_versions:>8} "
            f"{summary.partitions:>10} {summary.files:>6} {summary.rows:>10}"
        )
    return 0


def _cmd_compact(args: argparse.Namespace) -> int:
    result = compact(args.root, args.event_type, args.dt)
    print(
        f"compacted {result.event_type} dt={result.dt}: "
        f"{result.files_before} file(s) -> {result.files_after} file(s), "
        f"{result.rows_before} row(s) -> {result.rows_after} row(s)"
    )
    return 0


def _cmd_schema(args: argparse.Namespace) -> int:
    # compact() validates its own event_type internally; the schema command
    # doesn't go through compact(), so it validates here before using the
    # value to build a filesystem path.
    validate_event_type(args.event_type)
    registry = SchemaRegistry(args.root)
    versions = registry.all(args.event_type)
    if not versions:
        print(f"(no schema recorded for {args.event_type!r})")
        return 0
    previous = None
    for schema_version in versions:
        print(f"v{schema_version.version} ({schema_version.created_at.isoformat()}):")
        for arrow_field in schema_version.schema:
            print(f"  {arrow_field.name}: {arrow_field.type} (nullable={arrow_field.nullable})")
        if previous is not None:
            diff = diff_schemas(previous.schema, schema_version.schema)
            print("  diff from previous version:")
            for line in diff.describe().splitlines():
                print(f"    {line}")
        previous = schema_version
    return 0


_ROOT_HELP = "lake root: a directory, or s3://bucket/prefix (needs the 's3' extra)"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="eventlake")
    subparsers = parser.add_subparsers(dest="command", required=True)

    describe_parser = subparsers.add_parser("describe", help="summarize each event type")
    describe_parser.add_argument("root", help=_ROOT_HELP)
    describe_parser.set_defaults(func=_cmd_describe)

    compact_parser = subparsers.add_parser("compact", help="compact one partition's files")
    compact_parser.add_argument("root", help=_ROOT_HELP)
    compact_parser.add_argument("event_type")
    compact_parser.add_argument("dt", help="partition date, YYYY-MM-DD")
    compact_parser.set_defaults(func=_cmd_compact)

    schema_parser = subparsers.add_parser("schema", help="show schema version history")
    schema_parser.add_argument("root", help=_ROOT_HELP)
    schema_parser.add_argument("event_type")
    schema_parser.set_defaults(func=_cmd_schema)

    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    result: int = args.func(args)
    return result


if __name__ == "__main__":
    sys.exit(main())
