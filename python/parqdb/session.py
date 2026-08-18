from __future__ import annotations

import json
import os
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from itertools import count
from pathlib import Path
from types import MappingProxyType
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
from .iceberg import load_table_state, table_provider_inputs
from .identifier import TableIdentifier
from .maintenance import Maintenance
from .query import VectorQuery
from .runtime.catalog import IndexCatalog, IndexInfo


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
        iceberg: object | None = None,
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
        self._repository = self._native.index_repository()
        self._indexes = IndexCatalog(self._repository)
        self.ctx = self._native.context()
        self._query_names = count()
        self._maintenance = Maintenance(self)
        self._iceberg_catalog = iceberg
        self._iceberg_catalog_name = (
            _iceberg_catalog_name(iceberg) if iceberg is not None else None
        )
        self._iceberg_provider_inputs: dict[str, tuple[str, dict[str, str]]] = {}

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
        iceberg_identifier = self._iceberg_identifier(name)
        if iceberg_identifier is not None:
            state = load_table_state(self._iceberg_catalog, iceberg_identifier)
            reference = state.relation_json()
            dataframe = self._register_iceberg_relation(reference)
            return _EmbeddedSourceTable(
                dataframe,
                self,
                iceberg_identifier,
                reference,
                build_source=None,
            )
        dataframe = super().table(name)
        binding = self._native.persistent_table(name)
        if binding is None:
            return dataframe
        catalog, namespace, table_name, reference = binding
        return _EmbeddedSourceTable(
            dataframe,
            self,
            TableIdentifier(catalog, tuple(namespace), table_name),
            _parquet_relation_json(reference),
            build_source=reference,
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
        self._prepare_index_relations(query, source)
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

    def to_arrow(self, query: VectorQuery) -> pyarrow.Table:
        """Execute a vector query and collect its result as an Arrow table."""
        return self.collect(query)

    def to_sql(self, query: VectorQuery) -> str:
        """Compile a vector query to executable SQL in this session."""
        source = self._resolve_query_source(query)
        self._prepare_index_relations(query, source)
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
        return self._relation_reference(query.source)

    def _relation_reference(self, identifier: TableIdentifier) -> str:
        if (
            self._iceberg_catalog_name is not None
            and identifier.catalog == self._iceberg_catalog_name
        ):
            state = load_table_state(self._iceberg_catalog, identifier)
            reference = state.relation_json()
            self._register_iceberg_relation(reference)
            return reference
        source = self._native.persistent_table_source_by_identifier(
            identifier.catalog,
            list(identifier.namespace),
            identifier.name,
        )
        if source is None:
            raise ValueError(f"query source is not registered: {identifier!r}")
        return _parquet_relation_json(source)

    def _list_table_indexes(
        self,
        identifier: TableIdentifier,
    ) -> list[IndexInfo]:
        reference = self._relation_reference(identifier)
        return [
            IndexInfo(
                name=name,
                column=column,
                family=family,
                metric=metric,
                parameters=MappingProxyType(dict(parameters)),
                current_snapshot_id=current_snapshot_id,
            )
            for (
                name,
                column,
                family,
                metric,
                parameters,
                current_snapshot_id,
            ) in self._native.list_source_indexes(
                reference,
                _index_namespace(identifier),
            )
        ]

    def _drop_table_index(
        self,
        identifier: TableIdentifier,
        index: str,
    ) -> None:
        self._native.drop_source_index(
            self._relation_reference(identifier),
            _index_namespace(identifier),
            index,
        )

    def _prepare_index_relations(self, query: VectorQuery, source: str) -> None:
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
        self._prepare_metadata_relations(metadata)

    def _prepare_metadata_relations(self, metadata: dict[str, Any]) -> None:
        snapshot_id = int(metadata["current-snapshot-id"])
        snapshot = next(
            snapshot
            for snapshot in metadata["snapshots"]
            if int(snapshot["snapshot-id"]) == snapshot_id
        )
        for relation in snapshot["index-relations"].values():
            if not isinstance(relation, str):
                raise ValueError("index relation locations must be strings")

    def _register_iceberg_relation(self, reference: str) -> DataFrame:
        if self._iceberg_catalog is None:
            raise ValueError(
                "index metadata references Iceberg, but this session has no Iceberg catalog"
            )
        relation = json.loads(reference)
        if relation.get("catalog") != self._iceberg_catalog_name:
            raise ValueError(
                "index metadata references an Iceberg catalog not bound to this session"
            )
        inputs = self._iceberg_provider_inputs.get(reference)
        if inputs is None:
            inputs = table_provider_inputs(self._iceberg_catalog, relation)
            self._iceberg_provider_inputs[reference] = inputs
        metadata_location, properties = inputs
        return DataFrame(
            self._native.register_iceberg_relation(
                reference,
                metadata_location,
                properties,
            )
        )

    def _iceberg_identifier(self, name: str) -> TableIdentifier | None:
        if self._iceberg_catalog_name is None:
            return None
        if not isinstance(name, str):
            raise TypeError("table name must be a string")
        parts = tuple(part for part in name.split(".") if part)
        if not parts or parts[0] != self._iceberg_catalog_name:
            return None
        if len(parts) < 3:
            raise ValueError(
                "Iceberg table identifiers require catalog, namespace, and name"
            )
        return TableIdentifier(parts[0], parts[1:-1], parts[-1])


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
            reference = _parquet_relation_json(reference)
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
    iceberg: object | None = None,
    config: DataFusionSessionConfig | None = None,
    runtime: RuntimeEnvBuilder | None = None,
) -> _EmbeddedSession:
    return _EmbeddedSession(
        root,
        warehouse=warehouse,
        storage_options=storage_options,
        iceberg=iceberg,
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


def _parquet_relation_json(uri: str) -> str:
    return json.dumps({"profile": "parquet", "uri": uri}, separators=(",", ":"))


def _index_namespace(identifier: TableIdentifier) -> list[str]:
    return list(identifier.index_namespace)


def _iceberg_catalog_name(catalog: object) -> str:
    name = getattr(catalog, "name", None)
    if not isinstance(name, str) or not name:
        raise ValueError("the Iceberg catalog must expose a non-empty name")
    return name
