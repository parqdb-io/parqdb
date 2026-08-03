from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import pyarrow

from ..._repository import open_index_repository
from ...backends.v1 import BackendInfo, CapabilityReport
from ...build import BuildCoordinator
from ...builders.v1 import BuildContext, IndexBuilder
from ...catalog import IndexCatalog, IndexInfo
from ...identifier import TableIdentifier
from ...query import VectorQuery
from ._backend import STARROCKS_INFO, starrocks_report
from .iceberg import ResolvedRelation, relation_sql, resolve_current
from .planner import compile_query
from .table import Table


class Session:
    """Query-only Relify session over a caller-owned StarRocks Flight connection."""

    def __init__(
        self,
        connection: Any,
        *,
        index_catalog: str,
        iceberg_catalog: Any,
        catalog_name: str | None = None,
        metadata_root: str | None = None,
        storage_options: Mapping[str, str] | None = None,
        index_namespace: tuple[str, ...] = ("relify",),
    ) -> None:
        if not callable(getattr(connection, "cursor", None)):
            raise TypeError("connection must be an Arrow Flight SQL ADBC connection")
        if iceberg_catalog is None:
            raise TypeError("iceberg_catalog is required")
        resolved_catalog_name = catalog_name or getattr(
            iceberg_catalog,
            "name",
            None,
        )
        if not isinstance(resolved_catalog_name, str) or not resolved_catalog_name:
            raise ValueError(
                "catalog_name is required when the PyIceberg catalog has no name"
            )
        self.connection = connection
        self.iceberg_catalog = iceberg_catalog
        self.catalog_name = resolved_catalog_name
        if (
            not isinstance(index_namespace, tuple)
            or not index_namespace
            or any(
                not isinstance(segment, str) or not segment
                for segment in index_namespace
            )
        ):
            raise ValueError("index_namespace must contain non-empty string segments")
        self.index_namespace = index_namespace
        self._native = open_index_repository(
            index_catalog,
            metadata_root=metadata_root,
            storage_options=storage_options,
        )
        self._indexes = IndexCatalog(self._native)
        self._builds = BuildCoordinator(self, default_builder=None)

    @property
    def indexes(self) -> IndexCatalog:
        return self._indexes

    @property
    def default_builder(self) -> IndexBuilder | None:
        return self._builds.default_builder

    @property
    def backend(self) -> BackendInfo:
        return STARROCKS_INFO

    @property
    def capabilities(self) -> CapabilityReport:
        return starrocks_report()

    def table(self, identifier: str | TableIdentifier) -> Table:
        resolved = self._identifier(identifier)
        relation = self._resolve_source(resolved)
        self._verify_host_relation(relation)
        return Table(self, resolved)

    def to_sql(self, query: VectorQuery) -> str:
        """Compile a vector query into StarRocks SQL without executing it."""
        return compile_query(self, query).sql

    def collect(self, query: VectorQuery) -> pyarrow.Table:
        """Execute a vector query and collect one portable Arrow table."""
        compiled = compile_query(self, query)
        cursor = self.connection.cursor()
        try:
            cursor.execute(compiled.sql)
            reader = cursor.fetch_record_batch()
            schema = reader.schema
            batches = list(reader)
        finally:
            cursor.close()
        _validate_result(schema, batches, compiled.projection)
        return pyarrow.Table.from_batches(batches, schema=schema)

    def explain(self, query: VectorQuery) -> str:
        """Return StarRocks' textual plan for a vector query."""
        sql = compile_query(self, query).sql
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"EXPLAIN {sql}")
            rows = cursor.fetchall()
        finally:
            cursor.close()
        return "\n".join(
            str(row[0]) if len(row) == 1 else "\t".join(str(value) for value in row)
            for row in rows
        )

    def _identifier(self, value: str | TableIdentifier) -> TableIdentifier:
        if isinstance(value, TableIdentifier):
            if value.catalog != self.catalog_name:
                raise ValueError(
                    "table identifier belongs to a different Iceberg catalog"
                )
            return value
        if not isinstance(value, str):
            raise TypeError(
                "table identifier must be a string or relify.TableIdentifier"
            )
        parts = tuple(value.split("."))
        if len(parts) < 2 or any(not part for part in parts):
            raise ValueError("StarRocks table identifiers require namespace and name")
        return TableIdentifier(self.catalog_name, parts[:-1], parts[-1])

    def _resolve_source(self, identifier: TableIdentifier) -> ResolvedRelation:
        if identifier.catalog != self.catalog_name:
            raise ValueError("table belongs to a different Iceberg catalog")
        return resolve_current(self.iceberg_catalog, identifier)

    def _verify_host_relation(self, relation: ResolvedRelation) -> None:
        cursor = self.connection.cursor()
        try:
            cursor.execute(f"SELECT * FROM {relation_sql(relation)} LIMIT 0")
            cursor.fetch_record_batch()
        finally:
            cursor.close()

    def _resolve_build_relation(
        self,
        identifier: TableIdentifier,
    ) -> dict[str, object]:
        return self._resolve_source(identifier).state.relation_dict()

    def _build_context(self) -> BuildContext:
        return BuildContext(
            iceberg_catalog=self.iceberg_catalog,
            catalog_name=self.catalog_name,
            index_namespace=self.index_namespace,
        )

    def _index_repository(self) -> Any:
        return self._native

    def _list_table_indexes(
        self,
        identifier: TableIdentifier,
    ) -> list[IndexInfo]:
        source = self._resolve_source(identifier)
        return self.indexes.list_for(source.state.relation_dict())

    def _drop_table_index(
        self,
        identifier: TableIdentifier,
        index: str,
    ) -> None:
        matching = {info.name for info in self._list_table_indexes(identifier)}
        if index not in matching:
            from ..._native import IndexNotFoundError

            raise IndexNotFoundError(f"index not found: {index}")
        self._native.drop_index(index)


def connect(
    connection: Any,
    *,
    index_catalog: str,
    iceberg_catalog: Any,
    catalog_name: str | None = None,
    metadata_root: str | None = None,
    storage_options: Mapping[str, str] | None = None,
    index_namespace: tuple[str, ...] = ("relify",),
) -> Session:
    return Session(
        connection,
        index_catalog=index_catalog,
        iceberg_catalog=iceberg_catalog,
        catalog_name=catalog_name,
        metadata_root=metadata_root,
        storage_options=storage_options,
        index_namespace=index_namespace,
    )


def _validate_result(
    schema: pyarrow.Schema,
    batches: list[pyarrow.RecordBatch],
    projection: tuple[str, ...],
) -> None:
    expected_names = [*projection, "_distance"]
    if schema.names != expected_names:
        raise TypeError("StarRocks returned an unexpected vector-query result schema")
    distance_field = schema.field("_distance")
    if not pyarrow.types.is_float32(distance_field.type):
        raise TypeError("StarRocks _distance result must have Arrow float32 type")
    for batch in batches:
        if batch.schema != schema:
            raise TypeError(
                "StarRocks returned an unexpected vector-query result schema"
            )
        distance = batch.column("_distance")
        if distance.null_count or any(
            not math.isfinite(value) for value in distance.to_pylist()
        ):
            raise ValueError("squared-L2 distance is non-finite")
