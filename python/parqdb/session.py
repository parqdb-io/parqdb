from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pyarrow

from ._native import _NativeSession
from .datafusion import (
    DataFrame,
    RuntimeEnvBuilder,
    SessionContext,
)
from .datafusion import (
    SessionConfig as DataFusionSessionConfig,
)
from .datafusion.expr import SortKey
from .identifier import TableIdentifier
from .maintenance import Maintenance
from .query import VectorQuery


@dataclass(frozen=True)
class ParquetPageCacheStats:
    """Allocation and lookup counters for the session's Parquet Page cache."""

    capacity: int
    resident_bytes: int
    retired_bytes: int
    page_count: int
    hits: int
    misses: int
    admissions: int
    evictions: int
    capacity_bypasses: int
    oversized_bypasses: int


class _EmbeddedSession(SessionContext):
    def __init__(
        self,
        root: str | os.PathLike[str],
        *,
        warehouse: str | None = None,
        storage_options: Mapping[str, str] | None = None,
        catalog_path: str | os.PathLike[str] | None = None,
        config: DataFusionSessionConfig | None = None,
        runtime: RuntimeEnvBuilder | None = None,
    ) -> None:
        self._root = Path(root).expanduser().resolve()
        options = dict(storage_options or {})
        if any(
            not isinstance(key, str) or not isinstance(value, str)
            for key, value in options.items()
        ):
            raise TypeError("storage_options keys and values must be strings")
        resolved_catalog_path = (
            Path(catalog_path).expanduser().resolve()
            if catalog_path is not None
            else self._root / "catalog.sqlite"
        )
        if config is not None and not isinstance(config, DataFusionSessionConfig):
            raise TypeError("config must be parqdb.datafusion.SessionConfig")
        if runtime is not None and not isinstance(runtime, RuntimeEnvBuilder):
            raise TypeError("runtime must be parqdb.datafusion.RuntimeEnvBuilder")
        self._native = _NativeSession(
            self._root,
            warehouse,
            options or None,
            resolved_catalog_path if catalog_path is not None else None,
            config.config_internal if config is not None else None,
            runtime.config_internal if runtime is not None else None,
        )
        self._warehouse = self._native.warehouse_root()
        self.ctx = self._native.context()
        self._query_names = count()
        self._maintenance = Maintenance(self)

    @property
    def root(self) -> Path:
        return self._root

    @property
    def warehouse(self) -> str:
        return self._warehouse

    @classmethod
    def global_ctx(cls) -> SessionContext:
        """Return DataFusion's global context without ParqDB session state."""
        return SessionContext.global_ctx()

    def enable_url_table(self) -> SessionContext:
        """Return a plain DataFusion context with URL table support enabled."""
        return _wrap_datafusion_context(self.ctx.enable_url_table())

    @property
    def maintenance(self) -> Maintenance:
        return self._maintenance

    def register_parquet(
        self,
        name: str,
        path: str | Path | Sequence[str | Path],
        table_partition_cols: list[tuple[str, str | pyarrow.DataType]] | None = None,
        parquet_pruning: bool = True,
        file_extension: str = ".parquet",
        skip_metadata: bool = True,
        schema: pyarrow.Schema | None = None,
        file_sort_order: Sequence[Sequence[SortKey]] | None = None,
    ) -> None:
        """Register an indexable Parquet base table in this session."""
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(
                "persistent Parquet tables require one path or wildcard pattern"
            )
        source = _absolute_source_reference(path)
        sort_order = _persistent_sort_order(file_sort_order)
        self._native.register_parquet(
            table_name=name,
            source=source,
            table_partition_cols=self._convert_table_partition_cols(
                table_partition_cols or []
            ),
            parquet_pruning=parquet_pruning,
            file_extension=file_extension,
            skip_metadata=skip_metadata,
            schema=schema,
            file_sort_order=sort_order,
        )

    def table(self, name: str) -> DataFrame:
        """Return a registered table using DataFusion's table semantics."""
        dataframe = super().table(name)
        definition_json = self._native.persistent_table_definition(name)
        if definition_json is None:
            return dataframe
        definition = json.loads(definition_json)
        raw_identifier = definition["identifier"]
        identifier = TableIdentifier(
            str(raw_identifier["catalog"]),
            tuple(str(segment) for segment in raw_identifier["namespace"]),
            str(raw_identifier["name"]),
        )
        build_source = None
        if definition["provider"] == "parquet":
            build_source = str(definition["properties"]["location"])
        return _EmbeddedSourceTable(
            dataframe,
            self,
            identifier,
            definition_json,
            build_source=build_source,
        )

    def deregister_table(self, name: str) -> None:
        """Remove a registered table and its ParqDB source binding."""
        self._native.drop_table_definition_if_exists(name)
        super().deregister_table(name)

    def parquet_page_cache_stats(self) -> ParquetPageCacheStats:
        """Return allocation and lookup counters for the Parquet Page cache."""
        return ParquetPageCacheStats(*self._native.parquet_page_cache_stats())

    def clear_parquet_page_cache(self) -> None:
        """Remove resident Parquet Pages from future cache lookups."""
        self._native.clear_parquet_page_cache()

    def to_dataframe(self, query: VectorQuery) -> DataFrame:
        """Compile a vector query into this session's lazy DataFrame."""
        source = self._resolve_query_source(query)
        self._prepare_index_tables(query, source)
        internal = self._native.plan_search(
            source,
            _index_namespace(query.source),
            list(query.query),
            query.index,
            query.column,
            query.probe_count,
            query.result_limit,
            list(query.projection) if query.projection is not None else None,
            query.predicate,
            query.bypass_index,
        )
        return DataFrame(internal)

    def collect(self, query: VectorQuery) -> pyarrow.Table:
        """Execute a vector query and collect one portable Arrow table."""
        dataframe = self.to_dataframe(query)
        return pyarrow.Table.from_batches(
            dataframe.collect(),
            schema=dataframe.schema(),
        )

    def to_sql(self, query: VectorQuery) -> str:
        """Compile a vector query to executable SQL in this session."""
        source = self._resolve_query_source(query)
        self._prepare_index_tables(query, source)
        return self._native.search_sql(
            source,
            _index_namespace(query.source),
            list(query.query),
            query.index,
            query.column,
            query.probe_count,
            query.result_limit,
            list(query.projection) if query.projection is not None else None,
            query.predicate,
            query.bypass_index,
        )

    def explain(self, query: VectorQuery, *, verbose: bool = False) -> str:
        """Return the resolved DataFusion plan without executing the query."""
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        return self._explain_query(query, verbose=verbose, analyze=False)

    def analyze(self, query: VectorQuery) -> str:
        """Execute a vector query and return its plan with runtime metrics."""
        return self._explain_query(query, verbose=False, analyze=True)

    def _explain_query(
        self, query: VectorQuery, *, verbose: bool, analyze: bool
    ) -> str:
        dataframe = self.to_dataframe(query)
        name = f"__parqdb_explain_{next(self._query_names)}"
        self.register_view(name, dataframe)
        try:
            if analyze:
                prefix = "EXPLAIN ANALYZE"
            elif verbose:
                prefix = "EXPLAIN VERBOSE"
            else:
                prefix = "EXPLAIN"
            plan = self.sql(f"{prefix} SELECT * FROM {_quote_identifier(name)}")
            return _format_explain(plan.collect())
        finally:
            self.deregister_table(name)

    def _resolve_query_source(self, query: VectorQuery) -> str:
        if not isinstance(query, VectorQuery):
            raise TypeError("query must be a parqdb.VectorQuery")
        return self._table_definition(query.source)

    def _table_definition(self, identifier: TableIdentifier) -> str:
        definition = self._native.persistent_table_definition_by_identifier(
            identifier.catalog,
            list(identifier.namespace),
            identifier.name,
        )
        if definition is None:
            raise ValueError(f"query source is not registered: {identifier!r}")
        return definition

    def _prepare_index_tables(self, query: VectorQuery, source: str) -> None:
        if query.bypass_index:
            return
        metadata = json.loads(
            self._native.select_index_metadata(
                source,
                _index_namespace(query.source),
                query.index,
                query.column,
            )
        )
        self._prepare_metadata_tables(metadata)

    def _prepare_metadata_tables(self, metadata: dict[str, Any]) -> None:
        snapshot_id = int(metadata["current-snapshot-id"])
        snapshot = next(
            snapshot
            for snapshot in metadata["snapshots"]
            if int(snapshot["snapshot-id"]) == snapshot_id
        )
        provider = snapshot["index-provider"]
        if not isinstance(provider.get("provider"), str):
            raise ValueError("index provider name must be a string")
        for table in snapshot["index-tables"].values():
            if not isinstance(table.get("definition-version"), int):
                raise ValueError("index table definition version must be an integer")
            if not isinstance(table.get("properties"), dict):
                raise ValueError("index table properties must be an object")


class _EmbeddedSourceTable(DataFrame):
    def __init__(
        self,
        dataframe: DataFrame,
        session: _EmbeddedSession,
        identifier: TableIdentifier,
        reference: str,
        *,
        build_source: str | None = None,
    ) -> None:
        if not reference.startswith("{"):
            build_source = reference if build_source is None else build_source
            reference = _parquet_table_json(identifier, reference)
        super().__init__(dataframe.df)
        self._session = session
        self._identifier = identifier
        self._build_source = build_source

    @property
    def identifier(self) -> TableIdentifier:
        return self._identifier


def _connect_embedded(
    root: str | os.PathLike[str],
    *,
    warehouse: str | None = None,
    storage_options: Mapping[str, str] | None = None,
    config: DataFusionSessionConfig | None = None,
    runtime: RuntimeEnvBuilder | None = None,
) -> _EmbeddedSession:
    return _EmbeddedSession(
        root,
        warehouse=warehouse,
        storage_options=storage_options,
        config=config,
        runtime=runtime,
    )


def _wrap_datafusion_context(internal: object) -> SessionContext:
    context = SessionContext.__new__(SessionContext)
    context.ctx = internal
    return context


def _format_explain(batches: Sequence[pyarrow.RecordBatch]) -> str:
    sections: list[str] = []
    for batch in batches:
        if batch.num_columns != 2:
            raise RuntimeError("DataFusion EXPLAIN returned an invalid schema")
        plan_types = batch.column(0).to_pylist()
        plans = batch.column(1).to_pylist()
        for plan_type, plan in zip(plan_types, plans, strict=True):
            if not isinstance(plan_type, str) or not isinstance(plan, str):
                raise RuntimeError("DataFusion EXPLAIN returned a non-string plan")
            sections.append(f"{plan_type}\n{plan}")
    if not sections:
        raise RuntimeError("DataFusion EXPLAIN returned no plan")
    return "\n".join(sections)


def _quote_identifier(identifier: str) -> str:
    return f'"{identifier.replace(chr(34), chr(34) * 2)}"'


def _validate_index_name(name: str) -> None:
    if not isinstance(name, str):
        raise TypeError("index name must be a string")
    if not name.strip():
        raise ValueError("index name must not be empty")


def _absolute_source_reference(source: str | Path) -> str:
    reference = os.fspath(source)
    if urlsplit(reference).scheme:
        return reference
    return os.fspath(Path(reference).expanduser().resolve())


def _persistent_sort_order(
    file_sort_order: Sequence[Sequence[SortKey]] | None,
) -> list[list[str]]:
    if file_sort_order is None:
        return []
    if any(not isinstance(key, str) for order in file_sort_order for key in order):
        raise NotImplementedError(
            "persistent Parquet tables currently require string file_sort_order keys"
        )
    return [list(order) for order in file_sort_order]  # type: ignore[arg-type]


def _parquet_table_json(identifier: TableIdentifier, uri: str) -> str:
    return json.dumps(
        {
            "identifier": {
                "catalog": identifier.catalog,
                "namespace": list(identifier.namespace),
                "name": identifier.name,
            },
            "provider": "parquet",
            "properties": {
                "definition-version": "1",
                "location": uri,
                "table-identity": uri,
            },
        },
        separators=(",", ":"),
    )


def _index_namespace(identifier: TableIdentifier) -> list[str]:
    return list(identifier.index_namespace)
