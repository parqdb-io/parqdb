from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .identifier import TableIdentifier


@dataclass(frozen=True, slots=True)
class IcebergTableState:
    """Exact portable state of one Iceberg table."""

    identifier: TableIdentifier
    table_uuid: str
    snapshot_id: int

    def relation_dict(self) -> dict[str, object]:
        return {
            "profile": "iceberg",
            "catalog": self.identifier.catalog,
            "namespace": list(self.identifier.namespace),
            "name": self.identifier.name,
            "table-uuid": self.table_uuid,
            "snapshot-id": self.snapshot_id,
        }

    def relation_json(self) -> str:
        return json.dumps(self.relation_dict(), separators=(",", ":"))


def load_table_state(catalog: Any, identifier: TableIdentifier) -> IcebergTableState:
    state, _ = resolve_table(catalog, identifier)
    return state


def resolve_table(
    catalog: Any,
    identifier: TableIdentifier,
) -> tuple[IcebergTableState, Any]:
    """Resolve one current Iceberg table state and its catalog object."""
    table = load_table(catalog, identifier)
    return _table_state(table, identifier), table


def _table_state(table: Any, identifier: TableIdentifier) -> IcebergTableState:
    metadata = table.metadata
    table_uuid = str(metadata.table_uuid).lower()
    snapshot = table.current_snapshot()
    if snapshot is None:
        raise ValueError(f"Iceberg table has no current snapshot: {identifier!r}")
    snapshot_id = int(snapshot.snapshot_id)
    if snapshot_id <= 0:
        raise ValueError(f"Iceberg table has an invalid snapshot ID: {identifier!r}")
    return IcebergTableState(identifier, table_uuid, snapshot_id)


def validate_relation(catalog: Any, relation: dict[str, Any]) -> IcebergTableState:
    state, _ = resolve_relation(catalog, relation)
    return state


def resolve_relation(
    catalog: Any,
    relation: dict[str, Any],
) -> tuple[IcebergTableState, Any]:
    """Resolve and validate the exact Iceberg table referenced by metadata."""
    if relation.get("profile") != "iceberg":
        raise TypeError("relation is not an Iceberg reference")
    identifier = TableIdentifier(
        str(relation["catalog"]),
        tuple(str(segment) for segment in relation["namespace"]),
        str(relation["name"]),
    )
    table = load_table(catalog, identifier)
    current = _table_state(table, identifier)
    expected_uuid = str(relation["table-uuid"]).lower()
    if current.table_uuid != expected_uuid:
        raise ValueError(f"Iceberg table UUID changed: {identifier!r}")
    snapshot_id = int(relation["snapshot-id"])
    snapshot_by_id = getattr(table, "snapshot_by_id", None)
    if not callable(snapshot_by_id) or snapshot_by_id(snapshot_id) is None:
        raise ValueError(
            f"Iceberg snapshot is no longer available: {identifier!r} @ {snapshot_id}"
        )
    return IcebergTableState(identifier, current.table_uuid, snapshot_id), table


def load_table(catalog: Any, identifier: TableIdentifier) -> Any:
    return catalog.load_table((*identifier.namespace, identifier.name))


def table_provider_inputs(
    catalog: Any, relation: dict[str, Any]
) -> tuple[str, dict[str, str]]:
    _, table = resolve_relation(catalog, relation)
    metadata_location = str(table.metadata_location)
    properties = {
        str(key): str(value)
        for key, value in getattr(table.io, "properties", {}).items()
    }
    return metadata_location, properties
