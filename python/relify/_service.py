from __future__ import annotations

import asyncio
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import Any, Protocol

import pyarrow

from .build import IndexStatus
from .builders.v1 import IndexBuilder
from .catalog import IndexInfo
from .config import IVF, Local, WriteOptions
from .datafusion import RuntimeEnvBuilder
from .datafusion import SessionConfig as DataFusionSessionConfig
from .datafusion.expr import SortKey
from .identifier import TableIdentifier
from .query import VectorQuery
from .session import (
    _connect_embedded,
    _EmbeddedSession,
    _EmbeddedSourceTable,
    _wrap_datafusion_context,
)


class AsyncBatchStream(Protocol):
    def schema(self) -> pyarrow.Schema: ...

    def __aiter__(self) -> AsyncBatchStream: ...

    async def __anext__(self) -> pyarrow.RecordBatch: ...

    async def aclose(self) -> None: ...


@dataclass(frozen=True, slots=True)
class TableDescriptor:
    identifier: TableIdentifier
    schema: pyarrow.Schema


class SessionService:
    """Transport-neutral operations backed by one embedded execution session."""

    def __init__(self, host: _EmbeddedSession) -> None:
        self._host = host
        self._closed = False

    @classmethod
    async def open(
        cls,
        root: str | Path | None,
        *,
        catalog: str | None,
        index_root: str | None,
        storage_options: Mapping[str, str] | None,
        iceberg: object | None,
        config: DataFusionSessionConfig | None,
        runtime: RuntimeEnvBuilder | None,
    ) -> SessionService:
        host = await asyncio.to_thread(
            partial(
                _connect_embedded,
                root,
                catalog=catalog,
                index_root=index_root,
                storage_options=storage_options,
                iceberg=iceberg,
                config=config,
                runtime=runtime,
            )
        )
        return cls(host)

    @property
    def host(self) -> _EmbeddedSession:
        self._ensure_open()
        return self._host

    async def register_parquet(
        self,
        name: str,
        path: str | Path | Sequence[str | Path],
        *,
        table_partition_cols: list[tuple[str, str | pyarrow.DataType]] | None,
        parquet_pruning: bool,
        file_extension: str,
        skip_metadata: bool,
        schema: pyarrow.Schema | None,
        file_sort_order: Sequence[Sequence[SortKey]] | None,
    ) -> None:
        self._ensure_open()
        await asyncio.to_thread(
            partial(
                self._host.register_parquet,
                name,
                path,
                table_partition_cols=table_partition_cols,
                parquet_pruning=parquet_pruning,
                file_extension=file_extension,
                skip_metadata=skip_metadata,
                schema=schema,
                file_sort_order=file_sort_order,
            )
        )

    async def list_tables(self) -> list[TableIdentifier]:
        self._ensure_open()
        definitions = await asyncio.to_thread(self._host._native.list_table_definitions)
        return [
            TableIdentifier(catalog, tuple(namespace), name)
            for catalog, namespace, name in definitions
        ]

    async def table(self, identifier: str | TableIdentifier) -> TableDescriptor:
        self._ensure_open()
        name = _table_reference(identifier)
        table = await asyncio.to_thread(self._host.table, name)
        if not isinstance(table, _EmbeddedSourceTable):
            raise ValueError(f"table is not a registered Relify source: {name}")
        return TableDescriptor(table.identifier, table.schema())

    async def deregister_table(self, identifier: str | TableIdentifier) -> None:
        self._ensure_open()
        await asyncio.to_thread(
            self._host.deregister_table,
            _table_reference(identifier),
        )

    async def stream(self, query: VectorQuery | str) -> AsyncBatchStream:
        self._ensure_open()
        if isinstance(query, str):
            if not query.strip():
                raise ValueError("SQL statement must not be empty")
            return await self._host._native.stream_sql(query)
        if not isinstance(query, VectorQuery):
            raise TypeError("query must be a relify.VectorQuery or SQL string")
        return await self._host._native.stream_search(
            **await self._prepared_search_arguments(query)
        )

    async def explain(
        self,
        query: VectorQuery | str,
        *,
        verbose: bool,
        analyze: bool,
    ) -> str:
        self._ensure_open()
        if isinstance(query, VectorQuery):
            stream = await self._host._native.stream_explain_search(
                **await self._prepared_search_arguments(query),
                verbose=verbose,
                analyze=analyze,
            )
        elif not isinstance(query, str):
            raise TypeError("query must be a relify.VectorQuery or SQL string")
        else:
            prefix = (
                "EXPLAIN ANALYZE"
                if analyze
                else "EXPLAIN VERBOSE"
                if verbose
                else "EXPLAIN"
            )
            stream = await self.stream(f"{prefix} {query}")
        batches = await _collect_batches(stream)
        return _format_explain_batches(batches)

    async def to_sql(self, query: VectorQuery) -> str:
        self._ensure_open()
        return await asyncio.to_thread(self._host.to_sql, query)

    async def create_index(
        self,
        identifier: TableIdentifier,
        index: str,
        *,
        column: str,
        key: list[str],
        config: IVF,
        builder: IndexBuilder | None,
        writer_options: WriteOptions | None,
        wait_timeout: timedelta | None,
    ) -> None:
        self._ensure_open()
        await asyncio.to_thread(
            partial(
                self._host._builds.create,
                identifier,
                index=index,
                column=column,
                key=key,
                config=config,
                builder=builder,
                writer_options=writer_options,
                wait_timeout=wait_timeout,
            )
        )

    async def refresh_index(
        self,
        identifier: TableIdentifier,
        index: str,
        *,
        config: IVF | None,
        builder: Local | None,
        writer_options: WriteOptions | None,
        wait_timeout: timedelta | None,
    ) -> None:
        self._ensure_open()
        await asyncio.to_thread(
            partial(
                self._host._builds.refresh,
                identifier,
                index=index,
                config=config,
                builder=builder,
                writer_options=writer_options,
                wait_timeout=wait_timeout,
            )
        )

    async def index_status(
        self, identifier: TableIdentifier, index: str
    ) -> IndexStatus:
        self._ensure_open()
        return await asyncio.to_thread(self._host._builds.status, identifier, index)

    async def wait_for_index(
        self,
        identifier: TableIdentifier,
        index: str,
        timeout: timedelta,
    ) -> None:
        self._ensure_open()
        await asyncio.to_thread(self._host._builds.wait, identifier, index, timeout)

    async def list_indexes(self, identifier: TableIdentifier) -> list[IndexInfo]:
        self._ensure_open()
        return await asyncio.to_thread(self._host._list_table_indexes, identifier)

    async def drop_index(self, identifier: TableIdentifier, index: str) -> None:
        self._ensure_open()
        await asyncio.to_thread(self._host._drop_table_index, identifier, index)

    def datafusion_context(self) -> Any:
        self._ensure_open()
        return _wrap_datafusion_context(self._host.ctx)

    async def close(self) -> None:
        self._closed = True

    def _prepare_vector_query(self, query: VectorQuery) -> str:
        source = self._host._resolve_query_source(query)
        self._host._prepare_index_relations(query, source)
        return source

    async def _prepared_search_arguments(self, query: VectorQuery) -> dict[str, Any]:
        source = await asyncio.to_thread(self._prepare_vector_query, query)
        return {
            "source": source,
            "query": list(query.query),
            "index": query.index,
            "column": query.column,
            "nprobe": query.probe_count,
            "limit": query.result_limit,
            "projection": (
                list(query.projection) if query.projection is not None else None
            ),
            "filter": query.predicate,
            "bypass_index": query.bypass_index,
        }

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("session is closed")


async def _collect_batches(stream: AsyncBatchStream) -> list[pyarrow.RecordBatch]:
    try:
        return [batch async for batch in stream]
    finally:
        await stream.aclose()


def _table_reference(identifier: str | TableIdentifier) -> str:
    if isinstance(identifier, str):
        if not identifier:
            raise ValueError("table identifier must not be empty")
        return identifier
    if not isinstance(identifier, TableIdentifier):
        raise TypeError("table identifier must be a string or relify.TableIdentifier")
    return ".".join((identifier.catalog, *identifier.namespace, identifier.name))


def _format_explain_batches(batches: list[pyarrow.RecordBatch]) -> str:
    sections: list[str] = []
    for batch in batches:
        if batch.num_columns != 2:
            raise RuntimeError("DataFusion EXPLAIN returned an invalid schema")
        for plan_type, plan in zip(
            batch.column(0).to_pylist(),
            batch.column(1).to_pylist(),
            strict=True,
        ):
            if not isinstance(plan_type, str) or not isinstance(plan, str):
                raise RuntimeError("DataFusion EXPLAIN returned a non-string plan")
            sections.append(f"{plan_type}\n{plan}")
    if not sections:
        raise RuntimeError("DataFusion EXPLAIN returned no plan")
    return "\n".join(sections)
