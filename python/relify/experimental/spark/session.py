from __future__ import annotations

import io
from collections.abc import Mapping
from contextlib import redirect_stdout
from typing import Any

import pyarrow

from ..._repository import open_index_repository
from ...backends.v1 import BackendInfo, CapabilityReport
from ...build import BuildCoordinator
from ...builders.v1 import BuildContext, IndexBuilder
from ...catalog import IndexCatalog, IndexInfo
from ...identifier import TableIdentifier
from ...query import VectorQuery
from ._backend import SPARK_INFO, spark_report
from .iceberg import IcebergTableState, load_table_state, spark_identifier
from .parquet import ParquetTableState, validate_parquet_uri
from .planner import plan_query
from .table import Table


class Session:
    """Relify session backed by one SparkSession and an optional Iceberg catalog."""

    def __init__(
        self,
        spark: Any,
        *,
        index_catalog: str,
        iceberg_catalog: Any | None = None,
        catalog_name: str | None = None,
        metadata_root: str | None = None,
        storage_options: Mapping[str, str] | None = None,
        index_namespace: tuple[str, ...] = ("relify",),
    ) -> None:
        try:
            spark_context = spark.sparkContext
        except Exception as error:
            raise NotImplementedError(
                "the first Spark implementation supports Spark Classic only"
            ) from error
        if spark_context is None:
            raise TypeError("spark must be an active pyspark.sql.SparkSession")
        resolved_catalog_name = catalog_name or getattr(iceberg_catalog, "name", None)
        if iceberg_catalog is not None and (
            not isinstance(resolved_catalog_name, str) or not resolved_catalog_name
        ):
            raise ValueError(
                "catalog_name is required when the PyIceberg catalog has no name"
            )
        if iceberg_catalog is None and catalog_name is not None:
            raise ValueError("catalog_name requires an Iceberg catalog")
        if (
            not isinstance(index_namespace, tuple)
            or not index_namespace
            or any(
                not isinstance(segment, str) or not segment
                for segment in index_namespace
            )
        ):
            raise ValueError("index_namespace must contain non-empty string segments")
        self.spark = spark
        self.iceberg_catalog = iceberg_catalog
        self.catalog_name = resolved_catalog_name
        self.index_namespace = index_namespace
        self._native = open_index_repository(
            index_catalog,
            metadata_root=metadata_root,
            storage_options=storage_options,
        )
        self._indexes = IndexCatalog(self._native)
        self._builds = BuildCoordinator(
            self,
            default_builder=None,
        )
        self._parquet_tables: dict[TableIdentifier, ParquetTableState] = {}
        self._parquet_names: dict[str, TableIdentifier] = {}

    @property
    def indexes(self) -> IndexCatalog:
        return self._indexes

    @property
    def default_builder(self) -> IndexBuilder | None:
        return self._builds.default_builder

    @property
    def backend(self) -> BackendInfo:
        return SPARK_INFO

    @property
    def capabilities(self) -> CapabilityReport:
        return spark_report(iceberg=self.iceberg_catalog is not None)

    def table(self, identifier: str | TableIdentifier) -> Table:
        if isinstance(identifier, str) and identifier in self._parquet_names:
            resolved = self._parquet_names[identifier]
            state = self._parquet_tables[resolved]
            self.spark.read.parquet(state.uri)
            return Table(self, resolved)
        resolved = self._identifier(identifier)
        self._load_source_state(resolved)
        self.spark.table(spark_identifier(resolved))
        return Table(self, resolved)

    def register_parquet(self, name: str, uri: str) -> Table:
        """Register one portable Parquet source for Spark queries."""
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError("Parquet table name must be one non-empty identifier")
        if name in self._parquet_names:
            raise ValueError(f"Parquet table is already registered: {name}")
        identifier = TableIdentifier("relify", ("parquet",), name)
        state = ParquetTableState(identifier, validate_parquet_uri(uri))
        self.spark.read.parquet(state.uri)
        self._parquet_names[name] = identifier
        self._parquet_tables[identifier] = state
        return Table(self, identifier)

    def to_dataframe(self, query: VectorQuery) -> Any:
        """Compile a vector query into a native lazy PySpark DataFrame."""
        return plan_query(self, query)

    def collect(self, query: VectorQuery) -> pyarrow.Table:
        """Execute a vector query and collect one portable Arrow table."""
        table = self.to_dataframe(query).toArrow()
        if not isinstance(table, pyarrow.Table):
            raise TypeError("Spark DataFrame.toArrow() returned an invalid value")
        return table

    def explain(self, query: VectorQuery, *, verbose: bool = False) -> str:
        """Return Spark's public textual DataFrame plan without executing it."""
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        output = io.StringIO()
        with redirect_stdout(output):
            self.to_dataframe(query).explain(extended=verbose)
        plan = output.getvalue().rstrip()
        if not plan:
            raise RuntimeError("Spark returned an empty query plan")
        return plan

    def _identifier(self, value: str | TableIdentifier) -> TableIdentifier:
        if isinstance(value, TableIdentifier):
            if value in self._parquet_tables:
                return value
            if value.catalog != self.catalog_name:
                raise ValueError(
                    "table identifier belongs to a different Iceberg catalog"
                )
            return value
        if not isinstance(value, str):
            raise TypeError(
                "table identifier must be a string or relify.TableIdentifier"
            )
        if self.iceberg_catalog is None:
            raise ValueError(
                "table is not a registered Parquet source and no Iceberg catalog is bound"
            )
        if not isinstance(self.catalog_name, str):
            raise ValueError("the session has no Iceberg catalog name")
        parts = tuple(part for part in value.split(".") if part)
        if len(parts) < 2:
            raise ValueError("Spark table identifiers require namespace and name")
        return TableIdentifier(self.catalog_name, parts[:-1], parts[-1])

    def _load_source_state(
        self, identifier: TableIdentifier
    ) -> IcebergTableState | ParquetTableState:
        parquet = self._parquet_tables.get(identifier)
        if parquet is not None:
            return parquet
        if self.iceberg_catalog is None:
            raise ValueError("the session has no Iceberg catalog")
        if identifier.catalog != self.catalog_name:
            raise ValueError("table belongs to a different Iceberg catalog")
        return load_table_state(self.iceberg_catalog, identifier)

    def _resolve_build_relation(
        self,
        identifier: TableIdentifier,
    ) -> dict[str, object]:
        return self._load_source_state(identifier).relation_dict()

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
        source = self._load_source_state(identifier)
        return self.indexes.list_for(source.relation_dict())

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
    spark: Any,
    *,
    index_catalog: str,
    iceberg_catalog: Any | None = None,
    catalog_name: str | None = None,
    metadata_root: str | None = None,
    storage_options: Mapping[str, str] | None = None,
) -> Session:
    return Session(
        spark,
        index_catalog=index_catalog,
        iceberg_catalog=iceberg_catalog,
        catalog_name=catalog_name,
        metadata_root=metadata_root,
        storage_options=storage_options,
    )
