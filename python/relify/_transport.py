from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, Protocol

import pyarrow

from ._service import AsyncBatchStream, SessionService, TableDescriptor
from .build import IndexStatus
from .catalog import IndexInfo
from .config import IVF, WriteOptions
from .datafusion import RuntimeEnvBuilder
from .datafusion import SessionConfig as DataFusionSessionConfig
from .datafusion.expr import SortKey
from .identifier import TableIdentifier
from .query import VectorQuery


class SessionTransport(Protocol):
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
    ) -> None: ...

    async def list_tables(self) -> list[TableIdentifier]: ...

    async def table(self, identifier: str | TableIdentifier) -> TableDescriptor: ...

    async def deregister_table(self, identifier: str | TableIdentifier) -> None: ...

    async def stream(self, query: VectorQuery | str) -> AsyncBatchStream: ...

    async def explain(
        self,
        query: VectorQuery | str,
        *,
        verbose: bool,
        analyze: bool,
    ) -> str: ...

    async def to_sql(self, query: VectorQuery) -> str: ...

    async def create_index(
        self,
        identifier: TableIdentifier,
        index: str,
        *,
        column: str,
        key: list[str],
        config: IVF,
        writer_options: WriteOptions | None,
        wait_timeout: timedelta | None,
    ) -> None: ...

    async def refresh_index(
        self,
        identifier: TableIdentifier,
        index: str,
        *,
        config: IVF | None,
        writer_options: WriteOptions | None,
        wait_timeout: timedelta | None,
    ) -> None: ...

    async def index_status(
        self, identifier: TableIdentifier, index: str
    ) -> IndexStatus: ...

    async def wait_for_index(
        self,
        identifier: TableIdentifier,
        index: str,
        timeout: timedelta,
    ) -> None: ...

    async def list_indexes(self, identifier: TableIdentifier) -> list[IndexInfo]: ...

    async def drop_index(self, identifier: TableIdentifier, index: str) -> None: ...

    def datafusion_context(self) -> Any: ...

    async def close(self) -> None: ...


class InProcessTransport:
    """Calls the session service directly without serializing requests."""

    def __init__(self, service: SessionService) -> None:
        self._service = service

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
    ) -> InProcessTransport:
        return cls(
            await SessionService.open(
                root,
                catalog=catalog,
                index_root=index_root,
                storage_options=storage_options,
                iceberg=iceberg,
                config=config,
                runtime=runtime,
            )
        )

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
        await self._service.register_parquet(
            name,
            path,
            table_partition_cols=table_partition_cols,
            parquet_pruning=parquet_pruning,
            file_extension=file_extension,
            skip_metadata=skip_metadata,
            schema=schema,
            file_sort_order=file_sort_order,
        )

    async def list_tables(self) -> list[TableIdentifier]:
        return await self._service.list_tables()

    async def table(self, identifier: str | TableIdentifier) -> TableDescriptor:
        return await self._service.table(identifier)

    async def deregister_table(self, identifier: str | TableIdentifier) -> None:
        await self._service.deregister_table(identifier)

    async def stream(self, query: VectorQuery | str) -> AsyncBatchStream:
        return await self._service.stream(query)

    async def explain(
        self,
        query: VectorQuery | str,
        *,
        verbose: bool,
        analyze: bool,
    ) -> str:
        return await self._service.explain(
            query,
            verbose=verbose,
            analyze=analyze,
        )

    async def to_sql(self, query: VectorQuery) -> str:
        return await self._service.to_sql(query)

    async def create_index(
        self,
        identifier: TableIdentifier,
        index: str,
        *,
        column: str,
        key: list[str],
        config: IVF,
        writer_options: WriteOptions | None,
        wait_timeout: timedelta | None,
    ) -> None:
        await self._service.create_index(
            identifier,
            index,
            column=column,
            key=key,
            config=config,
            writer_options=writer_options,
            wait_timeout=wait_timeout,
        )

    async def refresh_index(
        self,
        identifier: TableIdentifier,
        index: str,
        *,
        config: IVF | None,
        writer_options: WriteOptions | None,
        wait_timeout: timedelta | None,
    ) -> None:
        await self._service.refresh_index(
            identifier,
            index,
            config=config,
            writer_options=writer_options,
            wait_timeout=wait_timeout,
        )

    async def index_status(
        self, identifier: TableIdentifier, index: str
    ) -> IndexStatus:
        return await self._service.index_status(identifier, index)

    async def wait_for_index(
        self,
        identifier: TableIdentifier,
        index: str,
        timeout: timedelta,
    ) -> None:
        await self._service.wait_for_index(identifier, index, timeout)

    async def list_indexes(self, identifier: TableIdentifier) -> list[IndexInfo]:
        return await self._service.list_indexes(identifier)

    async def drop_index(self, identifier: TableIdentifier, index: str) -> None:
        await self._service.drop_index(identifier, index)

    def datafusion_context(self) -> Any:
        return self._service.datafusion_context()

    async def close(self) -> None:
        await self._service.close()
