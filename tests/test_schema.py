from __future__ import annotations

import logging
from pathlib import Path

import pyarrow as pa
import pytest
from hypothesis import given
from hypothesis import strategies as st

from eventlake.schema import SchemaChangeError, SchemaRegistry, diff_schemas

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
