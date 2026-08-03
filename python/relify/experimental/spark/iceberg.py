from __future__ import annotations

from typing import Any

from ...iceberg import IcebergTableState, load_table_state, validate_relation
from ...identifier import TableIdentifier

__all__ = ["IcebergTableState", "load_table_state", "validate_relation"]


def ensure_namespace(catalog: Any, namespace: tuple[str, ...]) -> None:
    if _namespace_exists(catalog, namespace):
        return
    try:
        catalog.create_namespace(namespace)
    except Exception:
        # Namespace creation is idempotent from Relify's perspective. A
        # concurrent builder may have won the create race.
        if not _namespace_exists(catalog, namespace):
            raise


def _namespace_exists(catalog: Any, namespace: tuple[str, ...]) -> bool:
    exists = getattr(catalog, "namespace_exists", None)
    if callable(exists):
        return bool(exists(namespace))
    return namespace in catalog.list_namespaces()


def spark_identifier(identifier: TableIdentifier) -> str:
    return ".".join(_quote(segment) for segment in identifier_parts(identifier))


def read_snapshot(spark: Any, identifier: TableIdentifier, snapshot_id: int) -> Any:
    return spark.read.option("versionAsOf", str(snapshot_id)).table(
        spark_identifier(identifier)
    )


def identifier_parts(identifier: TableIdentifier) -> tuple[str, ...]:
    return (identifier.catalog, *identifier.namespace, identifier.name)


def _quote(segment: str) -> str:
    return f"`{segment.replace('`', '``')}`"
