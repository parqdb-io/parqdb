from __future__ import annotations

import asyncio
import os
import threading
from collections.abc import Callable, Coroutine, Iterable, Mapping, Sequence
from concurrent.futures import Future
from datetime import timedelta
from pathlib import Path
from typing import Any, Self, SupportsFloat, TypeVar
from weakref import WeakSet

import pyarrow

from .build import IndexStatus
from .config import IVF, WriteOptions
from .datafusion import RuntimeEnvBuilder
from .datafusion import SessionConfig as DataFusionSessionConfig
from .datafusion.expr import SortKey
from .errors import UnsupportedOperationError
from .identifier import TableIdentifier
from .query import VectorQuery
from .runtime.service import AsyncBatchStream, TableDescriptor
from .table import IndexInfo, _normalize_query
from .transport.base import InProcessTransport, SessionTransport

_T = TypeVar("_T")


class AsyncSession:
    """Asynchronous deployment-independent ParqDB session."""

    def __init__(self, transport: SessionTransport) -> None:
        self._transport = transport
        self._streams: set[AsyncResultStream] = set()
        self._closed = False

    async def register_parquet(
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
        await self._transport.register_parquet(
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
        return await self._transport.list_tables()

    async def table(self, identifier: str | TableIdentifier) -> AsyncSourceTable:
        descriptor = await self._transport.table(identifier)
        return AsyncSourceTable(self, descriptor)

    async def deregister_table(self, identifier: str | TableIdentifier) -> None:
        await self._transport.deregister_table(identifier)

    async def stream(self, query: VectorQuery | str) -> AsyncResultStream:
        stream = AsyncResultStream(
            await self._transport.stream(query),
            on_close=self._streams.discard,
        )
        self._streams.add(stream)
        return stream

    async def collect(self, query: VectorQuery | str) -> pyarrow.Table:
        stream = await self.stream(query)
        try:
            batches = [batch async for batch in stream]
        finally:
            await stream.aclose()
        return pyarrow.Table.from_batches(batches, schema=stream.schema)

    async def sql(self, statement: str) -> pyarrow.Table:
        return await self.collect(statement)

    async def to_sql(self, query: VectorQuery) -> str:
        return await self._transport.to_sql(query)

    async def explain(
        self,
        query: VectorQuery | str,
        *,
        verbose: bool = False,
    ) -> str:
        if not isinstance(verbose, bool):
            raise TypeError("verbose must be a boolean")
        return await self._transport.explain(
            query,
            verbose=verbose,
            analyze=False,
        )

    async def analyze(self, query: VectorQuery | str) -> str:
        return await self._transport.explain(
            query,
            verbose=False,
            analyze=True,
        )

    def datafusion_context(self) -> Any:
        return self._transport.datafusion_context()

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        for stream in list(self._streams):
            await stream.aclose()
        self._streams.clear()
        await self._transport.close()

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.close()


class AsyncResultStream:
    """Portable asynchronous Arrow batch stream."""

    def __init__(
        self,
        stream: AsyncBatchStream,
        *,
        on_close: Callable[[AsyncResultStream], None],
    ) -> None:
        self._stream = stream
        self.schema = stream.schema()
        self._on_close: Callable[[AsyncResultStream], None] | None = on_close
        self._closed = False

    def __aiter__(self) -> AsyncResultStream:
        return self

    async def __anext__(self) -> pyarrow.RecordBatch:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._stream.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            await self._stream.aclose()
        finally:
            on_close, self._on_close = self._on_close, None
            if on_close is not None:
                on_close(self)


class AsyncSourceTable:
    """Asynchronous table facade with portable index and search operations."""

    def __init__(self, session: AsyncSession, descriptor: TableDescriptor) -> None:
        self._session = session
        self.identifier = descriptor.identifier
        self.schema = descriptor.schema

    async def create_index(
        self,
        index: str,
        *,
        column: str,
        key: list[str],
        config: IVF,
        writer_options: WriteOptions | None = None,
        wait_timeout: timedelta | None = None,
    ) -> None:
        await self._session._transport.create_index(
            self.identifier,
            index,
            column=column,
            key=key,
            config=config,
            writer_options=writer_options,
            wait_timeout=wait_timeout,
        )

    async def register_index(
        self,
        index: str,
        *,
        metadata_location: str,
    ) -> None:
        await self._session._transport.register_index(
            self.identifier,
            index,
            metadata_location=metadata_location,
        )

    async def refresh_index(
        self,
        index: str,
        *,
        config: IVF | None = None,
        writer_options: WriteOptions | None = None,
        wait_timeout: timedelta | None = None,
    ) -> None:
        await self._session._transport.refresh_index(
            self.identifier,
            index,
            config=config,
            writer_options=writer_options,
            wait_timeout=wait_timeout,
        )

    async def index_status(self, index: str) -> IndexStatus:
        return await self._session._transport.index_status(self.identifier, index)

    async def wait_for_index(
        self,
        index: str,
        *,
        timeout: timedelta = timedelta(minutes=5),
    ) -> None:
        await self._session._transport.wait_for_index(
            self.identifier,
            index,
            timeout,
        )

    async def list_indexes(self) -> list[IndexInfo]:
        return await self._session._transport.list_indexes(self.identifier)

    async def drop_index(self, index: str) -> None:
        await self._session._transport.drop_index(self.identifier, index)

    def search(
        self,
        query: Iterable[SupportsFloat],
        *,
        column: str | None = None,
        index: str | None = None,
    ) -> VectorQuery:
        return _vector_query(self.identifier, query, column=column, index=index)


class Session:
    """Blocking facade over the authoritative asynchronous session API."""

    def __init__(self, session: AsyncSession, bridge: _BlockingBridge) -> None:
        self._async = session
        self._bridge = bridge
        self._readers: WeakSet[pyarrow.RecordBatchReader] = WeakSet()
        self._closed = False

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
        _run_sync(
            self,
            self._async.register_parquet(
                name,
                path,
                table_partition_cols=table_partition_cols,
                parquet_pruning=parquet_pruning,
                file_extension=file_extension,
                skip_metadata=skip_metadata,
                schema=schema,
                file_sort_order=file_sort_order,
            ),
        )

    def list_tables(self) -> list[TableIdentifier]:
        return _run_sync(self, self._async.list_tables())

    def table(self, identifier: str | TableIdentifier) -> SourceTable:
        table = _run_sync(self, self._async.table(identifier))
        return SourceTable(self, TableDescriptor(table.identifier, table.schema))

    def deregister_table(self, identifier: str | TableIdentifier) -> None:
        _run_sync(self, self._async.deregister_table(identifier))

    def stream(self, query: VectorQuery | str) -> pyarrow.RecordBatchReader:
        stream = _run_sync(self, self._async._transport.stream(query))
        if hasattr(stream, "__arrow_c_stream__"):
            reader = pyarrow.RecordBatchReader.from_stream(stream)
        else:
            async_stream = AsyncResultStream(stream, on_close=lambda _stream: None)
            batches = _BlockingBatchIterator(self._bridge, async_stream)
            reader = pyarrow.RecordBatchReader.from_batches(
                async_stream.schema, batches
            )
        self._readers.add(reader)
        return reader

    def collect(self, query: VectorQuery | str) -> pyarrow.Table:
        return _run_sync(self, self._async.collect(query))

    def sql(self, statement: str) -> pyarrow.Table:
        return _run_sync(self, self._async.sql(statement))

    def to_sql(self, query: VectorQuery) -> str:
        return _run_sync(self, self._async.to_sql(query))

    def explain(
        self,
        query: VectorQuery | str,
        *,
        verbose: bool = False,
    ) -> str:
        return _run_sync(self, self._async.explain(query, verbose=verbose))

    def analyze(self, query: VectorQuery | str) -> str:
        return _run_sync(self, self._async.analyze(query))

    def datafusion_context(self) -> Any:
        return self._async.datafusion_context()

    @property
    def root(self) -> Path:
        return _embedded_host(self._async).root

    @property
    def warehouse(self) -> str:
        """Return the index warehouse bound to this embedded session."""
        return _embedded_host(self._async).warehouse

    @property
    def maintenance(self) -> Any:
        return _embedded_host(self._async).maintenance

    def parquet_page_cache_stats(self) -> Any:
        return _embedded_host(self._async).parquet_page_cache_stats()

    def clear_parquet_page_cache(self) -> None:
        _embedded_host(self._async).clear_parquet_page_cache()

    def close(self) -> None:
        if self._closed:
            return
        try:
            for reader in list(self._readers):
                reader.close()
            self._readers.clear()
            self._bridge.run(self._async.close())
        finally:
            self._closed = True
            self._bridge.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, *_exc: object) -> None:
        self.close()


def _run_sync(session: Session, operation: Coroutine[Any, Any, _T]) -> _T:
    if session._closed:
        operation.close()
        raise RuntimeError("session is closed")
    return session._bridge.run(operation)


def _embedded_host(session: AsyncSession) -> Any:
    transport = session._transport
    if not isinstance(transport, InProcessTransport):
        raise UnsupportedOperationError("operation is available only in embedded mode")
    return transport._service.host


class SourceTable:
    """Blocking table facade with portable index and search operations."""

    def __init__(self, session: Session, descriptor: TableDescriptor) -> None:
        self._session = session
        self.identifier = descriptor.identifier
        self.schema = descriptor.schema

    def create_index(
        self,
        index: str,
        *,
        column: str,
        key: list[str],
        config: IVF,
        writer_options: WriteOptions | None = None,
        wait_timeout: timedelta | None = None,
    ) -> None:
        _run_sync(
            self._session,
            self._session._async._transport.create_index(
                self.identifier,
                index,
                column=column,
                key=key,
                config=config,
                writer_options=writer_options,
                wait_timeout=wait_timeout,
            ),
        )

    def register_index(
        self,
        index: str,
        *,
        metadata_location: str,
    ) -> None:
        _run_sync(
            self._session,
            self._session._async._transport.register_index(
                self.identifier,
                index,
                metadata_location=metadata_location,
            ),
        )

    def refresh_index(
        self,
        index: str,
        *,
        config: IVF | None = None,
        writer_options: WriteOptions | None = None,
        wait_timeout: timedelta | None = None,
    ) -> None:
        _run_sync(
            self._session,
            self._session._async._transport.refresh_index(
                self.identifier,
                index,
                config=config,
                writer_options=writer_options,
                wait_timeout=wait_timeout,
            ),
        )

    def index_status(self, index: str) -> IndexStatus:
        return _run_sync(
            self._session,
            self._session._async._transport.index_status(self.identifier, index),
        )

    def wait_for_index(
        self,
        index: str,
        *,
        timeout: timedelta = timedelta(minutes=5),
    ) -> None:
        _run_sync(
            self._session,
            self._session._async._transport.wait_for_index(
                self.identifier,
                index,
                timeout,
            ),
        )

    def list_indexes(self) -> list[IndexInfo]:
        return _run_sync(
            self._session, self._session._async._transport.list_indexes(self.identifier)
        )

    def drop_index(self, index: str) -> None:
        _run_sync(
            self._session,
            self._session._async._transport.drop_index(self.identifier, index),
        )

    def search(
        self,
        query: Iterable[SupportsFloat],
        *,
        column: str | None = None,
        index: str | None = None,
    ) -> VectorQuery:
        return _vector_query(self.identifier, query, column=column, index=index)


class _BlockingBatchIterator:
    def __init__(self, bridge: _BlockingBridge, stream: AsyncResultStream) -> None:
        self._bridge = bridge
        self._stream = stream
        self._closed = False

    def __iter__(self) -> _BlockingBatchIterator:
        return self

    def __next__(self) -> pyarrow.RecordBatch:
        if self._closed:
            raise StopIteration
        try:
            return self._bridge.run(self._stream.__anext__())
        except StopAsyncIteration:
            self.close()
            raise StopIteration from None

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._bridge.run(self._stream.aclose())

    def __del__(self) -> None:
        if not self._closed:
            try:
                self.close()
            except Exception:
                pass


class _BlockingBridge:
    def __init__(self) -> None:
        self._loop = asyncio.new_event_loop()
        self._started = threading.Event()
        self._thread = threading.Thread(
            target=self._run_loop,
            name="parqdb-async-bridge",
            daemon=True,
        )
        self._closed = False
        self._thread.start()
        self._started.wait()

    def run(self, operation: Coroutine[Any, Any, _T]) -> _T:
        if self._closed:
            operation.close()
            raise RuntimeError("session is closed")
        future: Future[_T] = asyncio.run_coroutine_threadsafe(operation, self._loop)
        return future.result()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join()
        self._loop.close()

    def _run_loop(self) -> None:
        asyncio.set_event_loop(self._loop)
        self._started.set()
        self._loop.run_forever()


async def connect_async(
    root: str | os.PathLike[str] | None = None,
    *,
    warehouse: str | None = None,
    storage_options: Mapping[str, str] | None = None,
    iceberg: object | None = None,
    config: DataFusionSessionConfig | None = None,
    runtime: RuntimeEnvBuilder | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | timedelta | None = None,
) -> AsyncSession:
    location = os.fspath(root) if root is not None else None
    if isinstance(location, str) and _is_http_url(location):
        _reject_remote_embedded_options(
            warehouse=warehouse,
            storage_options=storage_options,
            iceberg=iceberg,
            config=config,
            runtime=runtime,
        )
        from .transport.http import HttpTransport

        transport: SessionTransport = await HttpTransport.open(
            location,
            headers=headers,
            timeout=timeout,
        )
    else:
        if headers is not None or timeout is not None:
            raise ValueError("headers and timeout require an http(s) server URL")
        if location is None:
            raise TypeError("connect requires a local root or http(s) server URL")
        transport = await InProcessTransport.open(
            location,
            warehouse=warehouse,
            storage_options=storage_options,
            iceberg=iceberg,
            config=config,
            runtime=runtime,
        )
    return AsyncSession(transport)


def connect(
    root: str | os.PathLike[str] | None = None,
    *,
    warehouse: str | None = None,
    storage_options: Mapping[str, str] | None = None,
    iceberg: object | None = None,
    config: DataFusionSessionConfig | None = None,
    runtime: RuntimeEnvBuilder | None = None,
    headers: Mapping[str, str] | None = None,
    timeout: float | timedelta | None = None,
) -> Session:
    bridge = _BlockingBridge()
    try:
        session = bridge.run(
            connect_async(
                root,
                warehouse=warehouse,
                storage_options=storage_options,
                iceberg=iceberg,
                config=config,
                runtime=runtime,
                headers=headers,
                timeout=timeout,
            )
        )
    except BaseException:
        bridge.close()
        raise
    return Session(session, bridge)


def _is_http_url(location: str) -> bool:
    return location.lower().startswith(("http://", "https://"))


def _reject_remote_embedded_options(**options: object) -> None:
    configured = [name for name, value in options.items() if value is not None]
    if configured:
        raise ValueError(
            f"{configured[0]} is configured by the ParqDB server, not the client"
        )


def _vector_query(
    identifier: TableIdentifier,
    query: Iterable[SupportsFloat],
    *,
    column: str | None,
    index: str | None,
) -> VectorQuery:
    return VectorQuery(
        source=identifier,
        query=_normalize_query(query),
        column=column,
        index=index,
    )
