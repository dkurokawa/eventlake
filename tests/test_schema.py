from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st

from eventlake.schema import (
    SchemaChangeError,
    SchemaRegistrationRace,
    SchemaRegistry,
    diff_schemas,
)

BASE_SCHEMA = pa.schema(
    [
        pa.field("event_id", pa.string(), nullable=False),
        pa.field("occurred_at", pa.timestamp("us", tz="UTC"), nullable=False),
        pa.field("name", pa.string(), nullable=False),
        pa.field("score", pa.float64(), nullable=True),
    ]
)


def test_diff_schemas_no_changes_is_empty() -> None:
    diff = diff_schemas(BASE_SCHEMA, BASE_SCHEMA)
    assert diff.is_empty()
    assert diff.is_compatible()
    assert diff.describe() == "(no changes)"


def test_diff_schemas_added_nullable_field_is_compatible() -> None:
    new_schema = BASE_SCHEMA.append(pa.field("note", pa.string(), nullable=True))
    diff = diff_schemas(BASE_SCHEMA, new_schema)
    assert not diff.is_empty()
    assert diff.is_compatible()
    assert len(diff.added) == 1
    assert diff.added[0].name == "note"


def test_diff_schemas_added_non_nullable_field_is_incompatible() -> None:
    new_schema = BASE_SCHEMA.append(pa.field("note", pa.string(), nullable=False))
    diff = diff_schemas(BASE_SCHEMA, new_schema)
    assert not diff.is_compatible()


def test_diff_schemas_removed_field_is_incompatible() -> None:
    new_schema = pa.schema([f for f in BASE_SCHEMA if f.name != "score"])
    diff = diff_schemas(BASE_SCHEMA, new_schema)
    assert len(diff.removed) == 1
    assert not diff.is_compatible()


def _with_replaced_field(schema: pa.Schema, name: str, replacement: pa.Field) -> pa.Schema:
    return pa.schema([replacement if f.name == name else f for f in schema])


def test_diff_schemas_type_changed_is_incompatible() -> None:
    replacement = pa.field("name", pa.int64(), nullable=False)
    new_schema = _with_replaced_field(BASE_SCHEMA, "name", replacement)
    diff = diff_schemas(BASE_SCHEMA, new_schema)
    assert len(diff.type_changed) == 1
    assert not diff.is_compatible()


def test_diff_schemas_loosening_nullability_is_compatible() -> None:
    replacement = pa.field("name", pa.string(), nullable=True)
    new_schema = _with_replaced_field(BASE_SCHEMA, "name", replacement)
    diff = diff_schemas(BASE_SCHEMA, new_schema)
    assert len(diff.nullability_changed) == 1
    assert diff.is_compatible()


def test_diff_schemas_tightening_nullability_is_incompatible() -> None:
    replacement = pa.field("score", pa.float64(), nullable=False)
    new_schema = _with_replaced_field(BASE_SCHEMA, "score", replacement)
    diff = diff_schemas(BASE_SCHEMA, new_schema)
    assert len(diff.nullability_changed) == 1
    assert not diff.is_compatible()


def test_registry_first_registration_creates_v1(tmp_path: Path) -> None:
    registry = SchemaRegistry(tmp_path)
    result = registry.register("widget", BASE_SCHEMA, allow_breaking=False)
    assert result.version == 1
    assert registry.versions("widget") == [1]


def test_registry_no_change_reuses_version(tmp_path: Path) -> None:
    registry = SchemaRegistry(tmp_path)
    registry.register("widget", BASE_SCHEMA, allow_breaking=False)
    result = registry.register("widget", BASE_SCHEMA, allow_breaking=False)
    assert result.version == 1
    assert registry.versions("widget") == [1]


def test_registry_compatible_change_creates_new_version(tmp_path: Path) -> None:
    registry = SchemaRegistry(tmp_path)
    registry.register("widget", BASE_SCHEMA, allow_breaking=False)
    new_schema = BASE_SCHEMA.append(pa.field("note", pa.string(), nullable=True))
    result = registry.register("widget", new_schema, allow_breaking=False)
    assert result.version == 2
    assert registry.versions("widget") == [1, 2]


def test_registry_incompatible_change_raises_without_allow_breaking(tmp_path: Path) -> None:
    registry = SchemaRegistry(tmp_path)
    registry.register("widget", BASE_SCHEMA, allow_breaking=False)
    new_schema = pa.schema([f for f in BASE_SCHEMA if f.name != "score"])
    with pytest.raises(SchemaChangeError) as exc_info:
        registry.register("widget", new_schema, allow_breaking=False)
    assert exc_info.value.event_type == "widget"
    assert registry.versions("widget") == [1]


def test_registry_incompatible_change_allowed_with_allow_breaking(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    registry = SchemaRegistry(tmp_path)
    registry.register("widget", BASE_SCHEMA, allow_breaking=False)
    new_schema = pa.schema([f for f in BASE_SCHEMA if f.name != "score"])
    with caplog.at_level(logging.WARNING):
        result = registry.register("widget", new_schema, allow_breaking=True)
    assert result.version == 2
    assert registry.versions("widget") == [1, 2]
    assert any("breaking schema change" in message for message in caplog.messages)


def test_registry_load_round_trips_schema_with_list_field(tmp_path: Path) -> None:
    schema_with_list = BASE_SCHEMA.append(pa.field("tags", pa.list_(pa.string()), nullable=True))
    registry = SchemaRegistry(tmp_path)
    registry.register("widget", schema_with_list, allow_breaking=False)
    loaded = registry.load("widget", 1)
    assert loaded.schema.equals(schema_with_list)


def test_registry_latest_returns_none_when_unregistered(tmp_path: Path) -> None:
    registry = SchemaRegistry(tmp_path)
    assert registry.latest("nonexistent") is None
    assert registry.versions("nonexistent") == []
    assert registry.all("nonexistent") == []


def test_registry_all_returns_ordered_versions(tmp_path: Path) -> None:
    registry = SchemaRegistry(tmp_path)
    registry.register("widget", BASE_SCHEMA, allow_breaking=False)
    new_schema = BASE_SCHEMA.append(pa.field("note", pa.string(), nullable=True))
    registry.register("widget", new_schema, allow_breaking=False)
    versions = registry.all("widget")
    assert [v.version for v in versions] == [1, 2]
    assert versions[0].schema.equals(BASE_SCHEMA)
    assert versions[1].schema.equals(new_schema)


# --- H3: version files are claimed exclusively, with retry on collision ----


def test_write_never_overwrites_an_existing_version_file(tmp_path: Path) -> None:
    registry = SchemaRegistry(tmp_path)
    registry.register("widget", BASE_SCHEMA, allow_breaking=False)  # v1

    # v1 already exists: a second _write for the same version must fail
    # instead of silently overwriting it.
    with pytest.raises(FileExistsError):
        registry._write("widget", 1, BASE_SCHEMA)

    # Simulate a v2.json that already exists (e.g. another process claimed
    # it) with content the registry has no in-memory knowledge of yet:
    # a same-version _write of different content must still fail, not clobber it.
    concurrent_schema = BASE_SCHEMA.append(pa.field("other", pa.string(), nullable=True))
    registry._write("widget", 2, concurrent_schema)
    with pytest.raises(FileExistsError):
        registry._write("widget", 2, BASE_SCHEMA)
    assert {f.name for f in registry.load("widget", 2).schema} == {
        f.name for f in concurrent_schema
    }


def test_register_retries_when_target_version_already_exists(tmp_path: Path) -> None:
    registry = SchemaRegistry(tmp_path)
    registry.register("widget", BASE_SCHEMA, allow_breaking=False)  # v1

    concurrent_schema = BASE_SCHEMA.append(pa.field("other", pa.string(), nullable=True))
    real_write = SchemaRegistry._write
    call_count = {"n": 0}

    def flaky_write(
        self: SchemaRegistry, event_type: str, version: int, schema: pa.Schema
    ) -> object:
        call_count["n"] += 1
        if call_count["n"] == 1 and version == 2:
            # Someone else wins the race for v2 first, with a schema we
            # didn't know about when we decided our own candidate was v2.
            real_write(self, event_type, 2, concurrent_schema)
            raise FileExistsError("simulated concurrent winner")
        return real_write(self, event_type, version, schema)

    registry._write = flaky_write.__get__(registry, SchemaRegistry)  # type: ignore[method-assign]

    # Our schema happens to already be a compatible extension of whatever
    # ends up as the new v2 (this is what makes a clean retry possible -
    # see the docstring on SchemaRegistrationRace for the case where it isn't).
    our_schema = concurrent_schema.append(pa.field("mine", pa.string(), nullable=True))
    result = registry.register("widget", our_schema, allow_breaking=False)

    assert call_count["n"] == 2
    assert result.version == 3
    assert registry.versions("widget") == [1, 2, 3]
    v2 = registry.load("widget", 2)
    assert {f.name for f in v2.schema} == {f.name for f in concurrent_schema}
    v3 = registry.load("widget", 3)
    assert {f.name for f in v3.schema} >= {"other", "mine"}


def test_register_gives_up_after_max_attempts_under_permanent_contention(
    tmp_path: Path,
) -> None:
    registry = SchemaRegistry(tmp_path)
    registry.register("widget", BASE_SCHEMA, allow_breaking=False)  # v1

    def always_taken(
        self: SchemaRegistry, event_type: str, version: int, schema: pa.Schema
    ) -> object:
        raise FileExistsError("always contended")

    registry._write = always_taken.__get__(registry, SchemaRegistry)  # type: ignore[method-assign]

    new_schema = BASE_SCHEMA.append(pa.field("note", pa.string(), nullable=True))
    with pytest.raises(SchemaRegistrationRace, match="widget"):
        registry.register("widget", new_schema, allow_breaking=False)


def test_two_threads_registering_different_compatible_schemas_lose_nothing(
    tmp_path: Path,
) -> None:
    """A real concurrency test: two threads race to claim v2 for real."""
    import threading

    registry = SchemaRegistry(tmp_path)
    registry.register("widget", BASE_SCHEMA, allow_breaking=False)  # v1

    schema_a = BASE_SCHEMA.append(pa.field("a", pa.string(), nullable=True))
    schema_ab = schema_a.append(pa.field("b", pa.string(), nullable=True))

    real_write = SchemaRegistry._write
    barrier = threading.Barrier(2)
    seen_threads: set[int] = set()
    seen_lock = threading.Lock()

    def synced_write(
        self: SchemaRegistry, event_type: str, version: int, schema: pa.Schema
    ) -> object:
        tid = threading.get_ident()
        with seen_lock:
            is_first_call_for_thread = tid not in seen_threads
            seen_threads.add(tid)
        if is_first_call_for_thread:
            barrier.wait(timeout=5)
            # schema_ab's thread yields briefly so schema_a's thread wins
            # the real filesystem race deterministically; schema_ab is a
            # compatible superset of schema_a, so its retry (as v3) succeeds.
            if {f.name for f in schema} == {f.name for f in schema_ab}:
                import time

                time.sleep(0.05)
        return real_write(self, event_type, version, schema)

    registry._write = synced_write.__get__(registry, SchemaRegistry)  # type: ignore[method-assign]

    results: list[object] = []
    errors: list[BaseException] = []
    results_lock = threading.Lock()

    def run(schema: pa.Schema) -> None:
        try:
            result = registry.register("widget", schema, allow_breaking=False)
            with results_lock:
                results.append(result)
        except BaseException as exc:  # noqa: BLE001
            with results_lock:
                errors.append(exc)

    t_a = threading.Thread(target=run, args=(schema_a,))
    t_ab = threading.Thread(target=run, args=(schema_ab,))
    t_a.start()
    t_ab.start()
    t_a.join(timeout=10)
    t_ab.join(timeout=10)

    assert not errors, errors
    assert registry.versions("widget") == [1, 2, 3]
    v2 = registry.load("widget", 2)
    v3 = registry.load("widget", 3)
    assert {f.name for f in v2.schema} == {f.name for f in schema_a}
    assert {f.name for f in v3.schema} == {f.name for f in schema_ab}


# --- Property-based tests (hypothesis) -------------------------------------

_FIELD_NAMES = ["a", "b", "c", "d"]
_ARROW_TYPES = [pa.string(), pa.int64(), pa.float64(), pa.bool_(), pa.date32()]

_field_strategy = st.tuples(
    st.sampled_from(_FIELD_NAMES),
    st.sampled_from(_ARROW_TYPES),
    st.booleans(),
)


@st.composite
def schemas(draw: st.DrawFn) -> pa.Schema:
    chosen_names = draw(
        st.lists(st.sampled_from(_FIELD_NAMES), min_size=1, max_size=len(_FIELD_NAMES), unique=True)
    )
    fields = []
    for name in chosen_names:
        arrow_type = draw(st.sampled_from(_ARROW_TYPES))
        nullable = draw(st.booleans())
        fields.append(pa.field(name, arrow_type, nullable=nullable))
    return pa.schema(fields)


@given(schemas())
def test_property_diff_of_schema_with_itself_is_empty(schema: pa.Schema) -> None:
    diff = diff_schemas(schema, schema)
    assert diff.is_empty()
    assert diff.is_compatible()


@given(schemas(), st.sampled_from(_FIELD_NAMES), st.sampled_from(_ARROW_TYPES))
def test_property_adding_nullable_field_is_always_compatible(
    schema: pa.Schema, new_name: str, new_type: pa.DataType
) -> None:
    existing_names = {f.name for f in schema}
    if new_name in existing_names:
        new_name = new_name + "_extra"
    new_schema = schema.append(pa.field(new_name, new_type, nullable=True))
    diff = diff_schemas(schema, new_schema)
    assert diff.is_compatible()
