from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ...iceberg import (
    IcebergTableState,
    resolve_relation,
    resolve_table,
)
from ...identifier import TableIdentifier


@dataclass(frozen=True, slots=True)
class ResolvedRelation:
    """One exact Iceberg table state and the schema active at that snapshot."""

    state: IcebergTableState
    schema: Any


def resolve_current(
    catalog: Any,
    identifier: TableIdentifier,
) -> ResolvedRelation:
    state, table = resolve_table(catalog, identifier)
    return ResolvedRelation(state, _schema_at_snapshot(table, state.snapshot_id))


def resolve_reference(
    catalog: Any,
    catalog_name: str,
    relation: dict[str, Any],
) -> ResolvedRelation:
    if relation.get("profile") != "iceberg":
        raise NotImplementedError(
            "the first StarRocks implementation supports only Iceberg relations"
        )
    if relation.get("catalog") != catalog_name:
        raise ValueError(
            "index metadata references an Iceberg catalog not bound to this session"
        )
    state, table = resolve_relation(catalog, relation)
    return ResolvedRelation(state, _schema_at_snapshot(table, state.snapshot_id))


def relation_sql(relation: ResolvedRelation) -> str:
    identifier = relation.state.identifier
    name = ".".join(
        quote_identifier(segment)
        for segment in (
            identifier.catalog,
            *identifier.namespace,
            identifier.name,
        )
    )
    return f"{name} FOR VERSION AS OF {relation.state.snapshot_id}"


def quote_identifier(value: str) -> str:
    return f"`{value.replace('`', '``')}`"


def _schema_at_snapshot(table: Any, snapshot_id: int) -> Any:
    snapshot_by_id = getattr(table, "snapshot_by_id", None)
    if not callable(snapshot_by_id):
        raise TypeError("Iceberg table does not expose snapshot lookup")
    snapshot = snapshot_by_id(snapshot_id)
    if snapshot is None:
        raise ValueError(f"Iceberg snapshot is no longer available: {snapshot_id}")
    schema_id = getattr(snapshot, "schema_id", None)
    schemas = getattr(table, "schemas", None)
    if schema_id is None or not callable(schemas):
        raise TypeError("Iceberg table does not expose snapshot schema lookup")
    available = schemas()
    if not isinstance(available, Mapping):
        raise TypeError("Iceberg table returned an invalid schema registry")
    schema = available.get(schema_id)
    if schema is None:
        raise ValueError(f"Iceberg schema is no longer available: {schema_id}")
    return schema
