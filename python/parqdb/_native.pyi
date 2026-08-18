import os
from collections.abc import Mapping, Sequence
from typing import Any

import pyarrow

class _IpcEncoder:
    def __init__(self, schema: pyarrow.Schema, max_chunk_bytes: int) -> None: ...
    def start(self) -> list[bytes]: ...
    def write(self, batch: pyarrow.RecordBatch) -> list[bytes]: ...
    def finish(self) -> list[bytes]: ...

class _IpcDecoder:
    def __init__(self, max_frame_bytes: int) -> None: ...
    def push(
        self, chunk: bytes
    ) -> tuple[pyarrow.Schema | None, pyarrow.RecordBatch | None, bool]: ...
    def finish(self) -> None: ...

class ParqDBError(RuntimeError): ...
class InvalidArgumentError(ParqDBError): ...
class InvalidSchemaError(ParqDBError): ...
class IndexNotFoundError(ParqDBError): ...
class AlreadyExistsError(ParqDBError): ...
class BuildAlreadyRunningError(ParqDBError): ...
class AmbiguousIndexError(ParqDBError): ...
class InvalidMetadataError(ParqDBError): ...
class CatalogError(ParqDBError): ...
class StorageError(ParqDBError): ...
class BackendError(ParqDBError): ...
class QueryQueueFullError(ParqDBError): ...
class QueryQueueTimeoutError(ParqDBError): ...

class _NativeQueryStream:
    def schema(self) -> Any: ...
    def __arrow_c_stream__(self, requested_schema: Any | None = ...) -> Any: ...
    def __aiter__(self) -> _NativeQueryStream: ...
    async def __anext__(self) -> Any: ...
    async def aclose(self) -> None: ...

def _new_session_config() -> Any: ...

class _ParquetWriterOptions:
    def __init__(
        self,
        compression: str,
        max_row_group_rows: int | None,
        target_file_size: int,
        write_batch_rows: int,
    ) -> None: ...

class _NativeSession:
    def __init__(
        self,
        state_root: str | os.PathLike[str],
        warehouse: str | None = ...,
        storage_options: Mapping[str, str] | None = ...,
        catalog_path: str | os.PathLike[str] | None = ...,
        config: Any | None = ...,
        runtime: Any | None = ...,
    ) -> None: ...
    def warehouse_root(self) -> str: ...
    def context(self) -> Any: ...
    def register_parquet(
        self,
        table_name: str,
        source: str,
        table_partition_cols: list[tuple[str, Any]],
        parquet_pruning: bool,
        file_extension: str,
        skip_metadata: bool,
        schema: Any | None,
        file_sort_order: list[list[str]],
    ) -> tuple[str, list[tuple[str, str, bool]]]: ...
    def persistent_table(
        self, table_name: str
    ) -> tuple[str, list[str], str, str] | None: ...
    def list_table_definitions(self) -> list[tuple[str, list[str], str]]: ...
    def persistent_table_source_by_identifier(
        self,
        catalog: str,
        namespace: Sequence[str],
        name: str,
    ) -> str | None: ...
    def drop_table_definition_if_exists(self, table_name: str) -> bool: ...
    def register_index(
        self,
        source: str,
        index_namespace: Sequence[str],
        name: str,
        metadata_location: str,
    ) -> None: ...
    def select_index_metadata(
        self,
        source: str,
        index_namespace: Sequence[str],
        index: str | None,
        column: str | None,
    ) -> str: ...
    def register_iceberg_relation(
        self,
        reference: str,
        metadata_location: str,
        file_io_properties: dict[str, str],
    ) -> Any: ...
    def parquet_page_cache_stats(
        self,
    ) -> tuple[int, int, int, int, int, int, int, int, int, int]: ...
    def clear_parquet_page_cache(self) -> None: ...
    def query_admission_stats(self) -> tuple[int, int]: ...
    def remove_orphans(
        self,
        older_than_ms: int,
        dry_run: bool,
    ) -> list[tuple[str, str, int]]: ...
    def list_source_indexes(
        self, source: str, index_namespace: Sequence[str]
    ) -> list[tuple[str, str, str, str, Mapping[str, str], int]]: ...
    def drop_source_index(
        self, source: str, index_namespace: Sequence[str], name: str
    ) -> None: ...
    async def submit_create_index(
        self,
        source: str,
        index_namespace: Sequence[str],
        index_name: str,
        vector_field: str,
        source_key_fields: Sequence[str],
        nlist: int,
        posting_encoding: str,
        metric: str,
        writer_options: _ParquetWriterOptions,
        partitions: int | None,
    ) -> None: ...
    async def submit_refresh_index(
        self,
        source: str,
        index_namespace: Sequence[str],
        index_name: str,
        nlist: int | None,
        posting_encoding: str | None,
        metric: str | None,
        writer_options: _ParquetWriterOptions,
        partitions: int | None,
    ) -> None: ...
    async def index_build_status(
        self,
        source: str,
        index_namespace: Sequence[str],
        index_name: str,
    ) -> tuple[
        str,
        float | None,
        str | None,
        int | None,
        int | None,
        int | None,
        str | None,
        str | None,
    ]: ...
    async def wait_for_index_build(
        self, source: str, index_namespace: Sequence[str], index_name: str
    ) -> None: ...
    def plan_search(
        self,
        source: str,
        index_namespace: Sequence[str],
        query: Sequence[float],
        index: str | None = ...,
        column: str | None = ...,
        nprobe: int | None = ...,
        limit: int = ...,
        projection: Sequence[str] | None = ...,
        filter: str | None = ...,
        bypass_index: bool = ...,
    ) -> object: ...
    async def stream_search(
        self,
        source: str,
        index_namespace: Sequence[str],
        query: Sequence[float],
        index: str | None = ...,
        column: str | None = ...,
        nprobe: int | None = ...,
        limit: int = ...,
        projection: Sequence[str] | None = ...,
        filter: str | None = ...,
        bypass_index: bool = ...,
    ) -> _NativeQueryStream: ...
    async def stream_explain_search(
        self,
        source: str,
        index_namespace: Sequence[str],
        query: Sequence[float],
        index: str | None = ...,
        column: str | None = ...,
        nprobe: int | None = ...,
        limit: int = ...,
        projection: Sequence[str] | None = ...,
        filter: str | None = ...,
        bypass_index: bool = ...,
        verbose: bool = ...,
        analyze: bool = ...,
    ) -> _NativeQueryStream: ...
    async def stream_sql(self, sql: str) -> _NativeQueryStream: ...
    def search_sql(
        self,
        source: str,
        index_namespace: Sequence[str],
        query: Sequence[float],
        index: str | None = ...,
        column: str | None = ...,
        nprobe: int | None = ...,
        limit: int = ...,
        projection: Sequence[str] | None = ...,
        filter: str | None = ...,
        bypass_index: bool = ...,
    ) -> str: ...
