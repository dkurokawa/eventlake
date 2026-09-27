"""Detecting and recording schema changes for each event type.

Every event type has a version history under `<root>/_schemas/<event_type>/`.
`diff_schemas` is a pure function comparing two Arrow schemas; `SchemaRegistry`
uses it to decide, at write time, whether a new schema is identical to the
latest registered version, a compatible extension of it, or a breaking change.
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import pyarrow as pa

logger = logging.getLogger(__name__)

# How many times register() will re-read the latest version and retry after
# losing a race to claim the next version number, before giving up.
_MAX_REGISTER_ATTEMPTS = 10

_TYPE_TOKENS: dict[str, pa.DataType] = {
    "string": pa.string(),
    "int64": pa.int64(),
    "float64": pa.float64(),
    "bool": pa.bool_(),
    "date32": pa.date32(),
    "timestamp_utc": pa.timestamp("us", tz="UTC"),
}


def _type_to_token(data_type: pa.DataType) -> str:
    if pa.types.is_list(data_type):
        return f"list<{_type_to_token(data_type.value_type)}>"
    for token, candidate in _TYPE_TOKENS.items():
        if candidate.equals(data_type):
            return token
    raise ValueError(f"cannot serialize arrow type: {data_type!r}")


def _token_to_type(token: str) -> pa.DataType:
    if token.startswith("list<") and token.endswith(">"):
        inner = _token_to_type(token[len("list<") : -1])
        return pa.list_(inner)
    if token not in _TYPE_TOKENS:
        raise ValueError(f"unknown type token: {token!r}")
    return _TYPE_TOKENS[token]


@dataclass(frozen=True)
class FieldTypeChange:
    name: str
    old_type: pa.DataType
    new_type: pa.DataType


@dataclass(frozen=True)
class FieldNullabilityChange:
    name: str
    old_nullable: bool
    new_nullable: bool


@dataclass(frozen=True)
class SchemaDiff:
    added: tuple[pa.Field, ...] = field(default_factory=tuple)
    removed: tuple[pa.Field, ...] = field(default_factory=tuple)
    type_changed: tuple[FieldTypeChange, ...] = field(default_factory=tuple)
    nullability_changed: tuple[FieldNullabilityChange, ...] = field(default_factory=tuple)

    def is_empty(self) -> bool:
        return not (self.added or self.removed or self.type_changed or self.nullability_changed)

    def is_compatible(self) -> bool:
        """Whether `new` can be read together with data written under `old`.

        Compatible: adding a nullable field, or loosening non-nullable -> nullable.
        Incompatible: removing a field, changing a type, adding a non-nullable
        field, or tightening nullable -> non-nullable.
        """
        if self.removed or self.type_changed:
            return False
        if any(not added_field.nullable for added_field in self.added):
            return False
        return not any(
            change.old_nullable and not change.new_nullable for change in self.nullability_changed
        )

    def describe(self) -> str:
        lines: list[str] = []
        for added_field in self.added:
            lines.append(
                f"+ {added_field.name}: {added_field.type} (nullable={added_field.nullable})"
            )
        for removed_field in self.removed:
            lines.append(
                f"- {removed_field.name}: {removed_field.type} (nullable={removed_field.nullable})"
            )
        for type_change in self.type_changed:
            lines.append(f"~ {type_change.name}: {type_change.old_type} -> {type_change.new_type}")
        for null_change in self.nullability_changed:
            lines.append(
                f"~ {null_change.name}: nullable {null_change.old_nullable} -> "
                f"{null_change.new_nullable}"
            )
        return "\n".join(lines) if lines else "(no changes)"


def diff_schemas(old: pa.Schema, new: pa.Schema) -> SchemaDiff:
    """Compare two Arrow schemas field-by-field, by name."""
    old_fields = {f.name: f for f in old}
    new_fields = {f.name: f for f in new}

    added = tuple(new_fields[name] for name in sorted(new_fields.keys() - old_fields.keys()))
    removed = tuple(old_fields[name] for name in sorted(old_fields.keys() - new_fields.keys()))

    type_changed: list[FieldTypeChange] = []
    nullability_changed: list[FieldNullabilityChange] = []
    for name in sorted(old_fields.keys() & new_fields.keys()):
        old_field, new_field = old_fields[name], new_fields[name]
        if not old_field.type.equals(new_field.type):
            type_changed.append(FieldTypeChange(name, old_field.type, new_field.type))
        elif old_field.nullable != new_field.nullable:
            nullability_changed.append(
                FieldNullabilityChange(name, old_field.nullable, new_field.nullable)
            )

    return SchemaDiff(
        added=added,
        removed=removed,
        type_changed=tuple(type_changed),
        nullability_changed=tuple(nullability_changed),
    )


class SchemaChangeError(Exception):
    """Raised when a write would introduce a breaking schema change."""

    def __init__(self, event_type: str, diff: SchemaDiff) -> None:
        self.event_type = event_type
        self.diff = diff
        super().__init__(f"incompatible schema change for {event_type!r}:\n{diff.describe()}")


class SchemaRegistrationRace(Exception):
    """Raised when concurrent writers can't agree on a schema version.

    register() retries (re-reading the latest version and re-diffing) when
    it loses a race to claim the next version number, but only up to
    _MAX_REGISTER_ATTEMPTS times - this is what fires if contention is high
    enough that it never wins.
    """

    def __init__(self, event_type: str, attempts: int) -> None:
        self.event_type = event_type
        self.attempts = attempts
        super().__init__(
            f"could not register a schema version for {event_type!r} after "
            f"{attempts} attempts (too much concurrent contention)"
        )


@dataclass(frozen=True)
class SchemaVersion:
    version: int
    schema: pa.Schema
    created_at: datetime


class SchemaRegistry:
    """Reads and writes `<root>/_schemas/<event_type>/v<N>.json` version files."""

    def __init__(self, root: Path) -> None:
        self._root = root

    def _dir(self, event_type: str) -> Path:
        return self._root / "_schemas" / event_type

    def versions(self, event_type: str) -> list[int]:
        directory = self._dir(event_type)
        if not directory.exists():
            return []
        found: list[int] = []
        for path in directory.glob("v*.json"):
            try:
                found.append(int(path.stem[1:]))
            except ValueError:
                continue
        return sorted(found)

    def load(self, event_type: str, version: int) -> SchemaVersion:
        path = self._dir(event_type) / f"v{version}.json"
        payload = json.loads(path.read_text())
        fields = [
            pa.field(f["name"], _token_to_type(f["type"]), nullable=f["nullable"])
            for f in payload["fields"]
        ]
        return SchemaVersion(
            version=payload["version"],
            schema=pa.schema(fields),
            created_at=datetime.fromisoformat(payload["created_at"]),
        )

    def latest(self, event_type: str) -> SchemaVersion | None:
        versions = self.versions(event_type)
        if not versions:
            return None
        return self.load(event_type, versions[-1])

    def all(self, event_type: str) -> list[SchemaVersion]:
        return [self.load(event_type, version) for version in self.versions(event_type)]

    def register(
        self, event_type: str, schema: pa.Schema, *, allow_breaking: bool
    ) -> SchemaVersion:
        """Ensure `schema` is registered as (a version of) `event_type`.

        Returns the version to write under. Raises SchemaChangeError if the
        schema is a breaking change and `allow_breaking` is False.

        Safe to call concurrently from multiple processes/threads against
        the same root: claiming a version number is exclusive (a version
        file is never overwritten), so a writer that loses the race to
        claim version N simply re-reads the (now updated) latest version
        and redoes the diff against it, up to _MAX_REGISTER_ATTEMPTS times.
        """
        for _attempt in range(_MAX_REGISTER_ATTEMPTS):
            current = self.latest(event_type)

            if current is None:
                try:
                    return self._write(event_type, 1, schema)
                except FileExistsError:
                    continue

            diff = diff_schemas(current.schema, schema)
            if diff.is_empty():
                return current

            if diff.is_compatible():
                try:
                    return self._write(event_type, current.version + 1, schema)
                except FileExistsError:
                    continue

            if not allow_breaking:
                raise SchemaChangeError(event_type, diff)

            logger.warning("breaking schema change for %s:\n%s", event_type, diff.describe())
            try:
                return self._write(event_type, current.version + 1, schema)
            except FileExistsError:
                continue

        raise SchemaRegistrationRace(event_type, _MAX_REGISTER_ATTEMPTS)

    def _write(self, event_type: str, version: int, schema: pa.Schema) -> SchemaVersion:
        """Exclusively create `v<version>.json`, or raise FileExistsError.

        Writes the full content to a uniquely-named temp file first, then
        atomically claims the target name with `os.link` (which fails with
        FileExistsError if the target already exists, and never overwrites
        it) - so a reader only ever sees the target absent or complete, and
        two concurrent callers can't silently clobber each other's version.
        """
        directory = self._dir(event_type)
        directory.mkdir(parents=True, exist_ok=True)
        created_at = datetime.now(UTC)
        payload = {
            "version": version,
            "created_at": created_at.isoformat(),
            "fields": [
                {"name": f.name, "type": _type_to_token(f.type), "nullable": f.nullable}
                for f in schema
            ],
        }
        path = directory / f"v{version}.json"
        tmp_path = directory / f".v{version}.{uuid.uuid4().hex}.tmp"
        tmp_path.write_text(json.dumps(payload, indent=2))
        try:
            os.link(tmp_path, path)
        finally:
            tmp_path.unlink(missing_ok=True)
        return SchemaVersion(version=version, schema=schema, created_at=created_at)
