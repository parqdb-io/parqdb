from __future__ import annotations

import asyncio
import json
import os
import time
from collections.abc import Mapping, Sequence
from datetime import timedelta
from pathlib import Path
from typing import Any, Self
from urllib.parse import quote, urlsplit

import httpx
import pyarrow

from ..build import IndexStatus
from ..config import IVF, WriteOptions
from ..datafusion.expr import SortKey
from ..errors import StreamExecutionError, UnsupportedOperationError
from ..identifier import TableIdentifier
from ..query import VectorQuery
from ..runtime.service import AsyncBatchStream, TableDescriptor
from ..table import IndexInfo
from .errors import raise_remote_build_error, raise_remote_error, unavailable
from .ipc import decode_ipc_stream
from .models import (
    ARROW_STREAM_MEDIA_TYPE,
    DEFAULT_IDENTIFIER_DELIMITER,
    encode_identifier_path,
    identifier_from_json,
    identifier_to_json,
    index_info_from_json,
    index_status_from_json,
    ivf_to_json,
    registration_to_json,
    schema_from_base64,
    vector_query_to_json,
    writer_options_to_json,
)

_ERROR_BODY_LIMIT = 1024 * 1024


class HttpTransport:
    """Versioned HTTP transport for the portable ParqDB service contract."""

    def __init__(self, client: httpx.AsyncClient, *, owns_client: bool) -> None:
        self._client = client
        self._owns_client = owns_client
        self._closed = False

    @classmethod
    async def open(
        cls,
        url: str,
        *,
        headers: Mapping[str, str] | None = None,
        timeout: float | timedelta | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> Self:
        base_url = _base_url(url)
        normalized_headers = _headers(headers)
        if client is not None:
            return cls(client, owns_client=False)
        request_timeout: httpx.Timeout
        if timeout is None:
            request_timeout = httpx.Timeout(connect=10, read=None, write=30, pool=30)
        else:
            seconds = _timeout_seconds(timeout)
            request_timeout = httpx.Timeout(seconds)
        return cls(
            httpx.AsyncClient(
                base_url=base_url,
                headers=normalized_headers,
                timeout=request_timeout,
            ),
            owns_client=True,
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
        self._ensure_open()
        if not isinstance(path, (str, os.PathLike)):
            raise TypeError(
                "persistent Parquet tables require one path or wildcard pattern"
            )
        table_path, params = _registration_path(name)
        body = registration_to_json(
            name,
            os.fspath(path),
            table_partition_cols=table_partition_cols,
            parquet_pruning=parquet_pruning,
            file_extension=file_extension,
            skip_metadata=skip_metadata,
            schema=schema,
            file_sort_order=file_sort_order,
        )
        await self._request_json("POST", table_path, params=params, json_body=body)

    async def list_tables(self) -> list[TableIdentifier]:
        self._ensure_open()
        tables: list[TableIdentifier] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            params = {"limit": "1000"}
            if token is not None:
                params["page_token"] = token
            body = await self._request_json("GET", "v1/table", params=params)
            values = body.get("tables")
            if not isinstance(values, list):
                raise StreamExecutionError(
                    "ParqDB server returned invalid table metadata"
                )
            tables.extend(identifier_from_json(value) for value in values)
            next_token = body.get("next_page_token")
            if next_token is None:
                return tables
            if not isinstance(next_token, str) or not next_token:
                raise StreamExecutionError(
                    "ParqDB server returned an invalid page token"
                )
            if next_token in seen_tokens:
                raise StreamExecutionError("ParqDB server repeated a table page token")
            seen_tokens.add(next_token)
            token = next_token

    async def table(self, identifier: str | TableIdentifier) -> TableDescriptor:
        self._ensure_open()
        resolved = await self._resolve_identifier(identifier)
        path, params = _table_path(resolved, "describe")
        body = await self._request_json(
            "POST",
            path,
            params=params,
            json_body={"reference": _identifier_reference(resolved)},
        )
        return TableDescriptor(
            identifier_from_json(body.get("identifier")),
            schema_from_base64(body.get("schema")),
        )

    async def deregister_table(self, identifier: str | TableIdentifier) -> None:
        self._ensure_open()
        resolved = await self._resolve_identifier(identifier)
        path, params = _table_path(resolved, "deregister")
        await self._request_json(
            "POST",
            path,
            params=params,
            json_body={"identifier": identifier_to_json(resolved)},
        )

    async def stream(self, query: VectorQuery | str) -> AsyncBatchStream:
        self._ensure_open()
        if isinstance(query, VectorQuery):
            path, params = _table_path(query.source, "query")
            payload = vector_query_to_json(query)
        elif isinstance(query, str):
            if not query.strip():
                raise ValueError("SQL statement must not be empty")
            path, params = "v1/sql", None
            payload = {"sql": query}
        else:
            raise TypeError("query must be a parqdb.VectorQuery or SQL string")
        return await self._request_stream(path, params=params, json_body=payload)

    async def explain(
        self,
        query: VectorQuery | str,
        *,
        verbose: bool,
        analyze: bool,
    ) -> str:
        self._ensure_open()
        if isinstance(query, VectorQuery):
            path, params = _table_path(query.source, "explain")
            payload: dict[str, Any] = {
                "query": vector_query_to_json(query),
                "verbose": verbose,
                "analyze": analyze,
            }
        elif isinstance(query, str):
            path, params = "v1/sql/explain", None
            payload = {"sql": query, "verbose": verbose, "analyze": analyze}
        else:
            raise TypeError("query must be a parqdb.VectorQuery or SQL string")
        body = await self._request_json("POST", path, params=params, json_body=payload)
        plan = body.get("plan")
        if not isinstance(plan, str):
            raise StreamExecutionError("ParqDB server returned an invalid query plan")
        return plan

    async def to_sql(self, query: VectorQuery) -> str:
        self._ensure_open()
        if not isinstance(query, VectorQuery):
            raise TypeError("query must be a parqdb.VectorQuery")
        path, params = _table_path(query.source, "to_sql")
        body = await self._request_json(
            "POST",
            path,
            params=params,
            json_body=vector_query_to_json(query),
        )
        sql = body.get("sql")
        if not isinstance(sql, str):
            raise StreamExecutionError("ParqDB server returned invalid SQL")
        return sql

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
        self._ensure_open()
        path, params = _table_path(identifier, "create_index")
        await self._request_json(
            "POST",
            path,
            params=params,
            json_body={
                "source": identifier_to_json(identifier),
                "index": index,
                "column": column,
                "key": key,
                "config": ivf_to_json(config),
                "writer_options": writer_options_to_json(writer_options),
            },
        )
        if wait_timeout is not None:
            await self.wait_for_index(identifier, index, wait_timeout)

    async def register_index(
        self,
        identifier: TableIdentifier,
        index: str,
        *,
        metadata_location: str,
    ) -> None:
        self._ensure_open()
        path, params = _index_path(identifier, index, "register")
        await self._request_json(
            "POST",
            path,
            params=params,
            json_body={
                "source": identifier_to_json(identifier),
                "index": index,
                "metadata_location": metadata_location,
            },
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
        self._ensure_open()
        path, params = _index_path(identifier, index, "refresh")
        await self._request_json(
            "POST",
            path,
            params=params,
            json_body={
                "source": identifier_to_json(identifier),
                "index": index,
                "config": None if config is None else ivf_to_json(config),
                "writer_options": writer_options_to_json(writer_options),
            },
        )
        if wait_timeout is not None:
            await self.wait_for_index(identifier, index, wait_timeout)

    async def index_status(
        self, identifier: TableIdentifier, index: str
    ) -> IndexStatus:
        self._ensure_open()
        path, params = _index_path(identifier, index, "stats")
        body = await self._request_json(
            "POST",
            path,
            params=params,
            json_body={
                "source": identifier_to_json(identifier),
                "index": index,
            },
        )
        try:
            return index_status_from_json(body)
        except ValueError as error:
            raise StreamExecutionError(
                "ParqDB server returned invalid index status"
            ) from error

    async def wait_for_index(
        self,
        identifier: TableIdentifier,
        index: str,
        timeout: timedelta,
    ) -> None:
        seconds = _timeout_seconds(timeout)
        deadline = time.monotonic() + seconds
        delay = 0.05
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for index: {index}")
            try:
                async with asyncio.timeout(remaining):
                    status = await self.index_status(identifier, index)
            except TimeoutError as error:
                raise TimeoutError(f"timed out waiting for index: {index}") from error
            if status.error is not None:
                raise_remote_build_error(status.error_code, status.error)
            if status.state == "ready":
                return
            if status.state == "failed":
                raise_remote_build_error(
                    status.error_code,
                    f"index build failed: {index}",
                )
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError(f"timed out waiting for index: {index}")
            await asyncio.sleep(min(delay, remaining))
            delay = min(delay * 1.5, 1.0)

    async def list_indexes(self, identifier: TableIdentifier) -> list[IndexInfo]:
        self._ensure_open()
        path, params = _table_path(identifier, "index/list")
        body = await self._request_json(
            "POST",
            path,
            params=params,
            json_body={"source": identifier_to_json(identifier)},
        )
        values = body.get("indexes")
        if not isinstance(values, list):
            raise StreamExecutionError("ParqDB server returned invalid index metadata")
        try:
            return [index_info_from_json(value) for value in values]
        except ValueError as error:
            raise StreamExecutionError(
                "ParqDB server returned invalid index metadata"
            ) from error

    async def drop_index(self, identifier: TableIdentifier, index: str) -> None:
        self._ensure_open()
        path, params = _index_path(identifier, index, "drop")
        await self._request_json(
            "POST",
            path,
            params=params,
            json_body={
                "source": identifier_to_json(identifier),
                "index": index,
            },
        )

    def datafusion_context(self) -> Any:
        raise UnsupportedOperationError(
            "datafusion_context is available only in embedded mode"
        )

    async def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._owns_client:
            await self._client.aclose()

    async def _resolve_identifier(
        self, identifier: str | TableIdentifier
    ) -> TableIdentifier:
        if isinstance(identifier, TableIdentifier):
            return identifier
        if not isinstance(identifier, str):
            raise TypeError(
                "table identifier must be a string or parqdb.TableIdentifier"
            )
        if not identifier:
            raise ValueError("table identifier must not be empty")
        candidates = [
            item
            for item in await self.list_tables()
            if identifier
            in {
                item.name,
                ".".join((*item.namespace, item.name)),
                ".".join((item.catalog, *item.namespace, item.name)),
            }
        ]
        if len(candidates) != 1:
            raise ValueError(
                f"table reference resolved to {len(candidates)} registered tables: "
                f"{identifier}"
            )
        return candidates[0]

    async def _request_json(
        self,
        method: str,
        path: str,
        *,
        params: Mapping[str, str] | None = None,
        json_body: object | None = None,
    ) -> dict[str, Any]:
        self._ensure_open()
        try:
            response = await self._client.request(
                method,
                path,
                params=params,
                json=json_body,
                headers={"accept": "application/json"},
            )
        except httpx.HTTPError as error:
            raise unavailable(error) from error
        body = _response_json(response)
        if response.is_error:
            raise_remote_error(response, body)
        if not isinstance(body, dict):
            raise StreamExecutionError("ParqDB server returned invalid JSON")
        return body

    async def _request_stream(
        self,
        path: str,
        *,
        params: Mapping[str, str] | None,
        json_body: object,
    ) -> AsyncBatchStream:
        request = self._client.build_request(
            "POST",
            path,
            params=params,
            json=json_body,
            headers={"accept": ARROW_STREAM_MEDIA_TYPE, "accept-encoding": "identity"},
        )
        try:
            response = await self._client.send(request, stream=True)
        except httpx.HTTPError as error:
            raise unavailable(error) from error
        request_id = response.headers.get("x-request-id")
        if response.is_error:
            try:
                payload = await _read_limited(response, _ERROR_BODY_LIMIT)
                body = _json_bytes(payload)
                raise_remote_error(response, body)
            finally:
                await response.aclose()
        content_type = response.headers.get("content-type", "").split(";", 1)[0]
        if content_type != ARROW_STREAM_MEDIA_TYPE:
            await response.aclose()
            raise StreamExecutionError(
                "ParqDB server returned an unexpected response type",
                request_id=request_id,
            )
        source = _HttpResponseStream(response)
        try:
            decoded = await decode_ipc_stream(source)
        except Exception as error:
            await source.aclose()
            raise StreamExecutionError(
                "ParqDB query stream ended before its Arrow schema",
                request_id=request_id,
            ) from error
        return _RemoteBatchStream(decoded, request_id=request_id)

    def _ensure_open(self) -> None:
        if self._closed:
            raise RuntimeError("session is closed")


class _HttpResponseStream:
    def __init__(self, response: httpx.Response) -> None:
        self._response = response
        self._iterator = response.aiter_raw().__aiter__()
        self._closed = False

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> bytes:
        if self._closed:
            raise StopAsyncIteration
        try:
            return await self._iterator.__anext__()
        except StopAsyncIteration:
            await self.aclose()
            raise

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._response.aclose()


class _RemoteBatchStream:
    def __init__(self, source: AsyncBatchStream, *, request_id: str | None) -> None:
        self._source = source
        self._request_id = request_id

    def schema(self) -> pyarrow.Schema:
        return self._source.schema()

    def __aiter__(self) -> Self:
        return self

    async def __anext__(self) -> pyarrow.RecordBatch:
        try:
            return await self._source.__anext__()
        except StopAsyncIteration:
            raise
        except Exception as error:
            raise StreamExecutionError(
                "ParqDB query failed while streaming its result",
                request_id=self._request_id,
            ) from error

    async def aclose(self) -> None:
        await self._source.aclose()


def _table_path(
    identifier: TableIdentifier,
    operation: str,
) -> tuple[str, dict[str, str]]:
    delimiter = _choose_delimiter(identifier)
    path, delimiter = encode_identifier_path(identifier, delimiter=delimiter)
    return f"v1/table/{path}/{operation}", {"delimiter": delimiter}


def _registration_path(name: str) -> tuple[str, dict[str, str]]:
    if not isinstance(name, str):
        raise TypeError("table name must be a string")
    if not name:
        raise ValueError("table name must not be empty")
    delimiter = _choose_delimiter_for_segments((name,))
    return f"v1/table/{quote(name, safe='')}/register", {"delimiter": delimiter}


def _index_path(
    identifier: TableIdentifier,
    index: str,
    operation: str,
) -> tuple[str, dict[str, str]]:
    if not isinstance(index, str):
        raise TypeError("index name must be a string")
    if not index.strip():
        raise ValueError("index name must not be empty")
    path, params = _table_path(identifier, f"index/{quote(index, safe='')}/{operation}")
    return path, params


def _choose_delimiter(identifier: TableIdentifier) -> str:
    return _choose_delimiter_for_segments((*identifier.namespace, identifier.name))


def _choose_delimiter_for_segments(segments: Sequence[str]) -> str:
    for candidate in (DEFAULT_IDENTIFIER_DELIMITER, "~", "!", "^", "|"):
        if all(candidate not in segment for segment in segments):
            return candidate
    raise ValueError("table identifier contains every supported route delimiter")


def _identifier_reference(identifier: TableIdentifier) -> str:
    return ".".join((identifier.catalog, *identifier.namespace, identifier.name))


def _base_url(value: str) -> str:
    if not isinstance(value, str):
        raise TypeError("server URL must be a string")
    parsed = urlsplit(value)
    if parsed.scheme not in {"http", "https"} or not parsed.netloc:
        raise ValueError("server URL must use http or https")
    if parsed.query or parsed.fragment:
        raise ValueError("server URL must not contain a query or fragment")
    return value.rstrip("/") + "/"


def _headers(value: Mapping[str, str] | None) -> dict[str, str]:
    headers = dict(value or {})
    if any(
        not isinstance(key, str) or not isinstance(item, str)
        for key, item in headers.items()
    ):
        raise TypeError("HTTP header names and values must be strings")
    return headers


def _timeout_seconds(value: float | timedelta) -> float:
    seconds = value.total_seconds() if isinstance(value, timedelta) else value
    if (
        not isinstance(seconds, (int, float))
        or isinstance(seconds, bool)
        or seconds <= 0
    ):
        raise ValueError("timeout must be positive")
    return float(seconds)


def _response_json(response: httpx.Response) -> object:
    if len(response.content) > _ERROR_BODY_LIMIT:
        raise StreamExecutionError("ParqDB server returned an oversized JSON response")
    return _json_bytes(response.content)


def _json_bytes(payload: bytes) -> object:
    if len(payload) > _ERROR_BODY_LIMIT:
        return None
    try:
        return json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


async def _read_limited(response: httpx.Response, limit: int) -> bytes:
    payload = bytearray()
    async for chunk in response.aiter_raw():
        payload.extend(chunk)
        if len(payload) > limit:
            return b""
    return bytes(payload)
