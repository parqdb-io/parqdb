from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from urllib.parse import unquote

from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ..datafusion import RuntimeEnvBuilder
from ..datafusion import SessionConfig as DataFusionSessionConfig
from ..identifier import TableIdentifier
from ..runtime.service import AsyncBatchStream, SessionService
from ..transport.errors import classify_server_error
from ..transport.ipc import encode_ipc_stream
from ..transport.models import (
    ARROW_STREAM_MEDIA_TYPE,
    DEFAULT_IDENTIFIER_DELIMITER,
    MAX_JSON_BODY_BYTES,
    MAX_LIST_TABLES_PAGE_SIZE,
    decode_identifier_path,
    identifier_from_json,
    identifier_matches_path,
    identifier_to_json,
    index_info_to_json,
    index_status_to_json,
    ivf_from_json,
    registration_from_json,
    table_descriptor_to_json,
    vector_query_from_json,
    writer_options_from_json,
)
from .openapi import openapi_document
from .source_policy import SourceUriPolicy


def create_http_app(
    root: str | Path,
    *,
    warehouse: str | None = None,
    storage_options: Mapping[str, str] | None = None,
    config: DataFusionSessionConfig | None = None,
    runtime: RuntimeEnvBuilder | None = None,
    allowed_source_prefixes: Sequence[str | Path] = (),
) -> Starlette:
    source_policy = SourceUriPolicy(allowed_source_prefixes)

    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        service = await SessionService.open(
            root,
            warehouse=warehouse,
            storage_options=storage_options,
            iceberg=None,
            config=config,
            runtime=runtime,
        )
        app.state.parqdb_service = service
        try:
            yield
        finally:
            await service.close()

    return _application(lifespan=lifespan, source_policy=source_policy)


def create_http_app_for_service(
    service: SessionService,
    *,
    allowed_source_prefixes: Sequence[str | Path] = (),
) -> Starlette:
    app = _application(source_policy=SourceUriPolicy(allowed_source_prefixes))
    app.state.parqdb_service = service
    return app


def _application(*, lifespan: Any = None, source_policy: SourceUriPolicy) -> Starlette:
    app = Starlette(
        debug=False,
        lifespan=lifespan,
        routes=[
            Route("/health", _health, methods=["GET"]),
            Route("/openapi.json", _openapi, methods=["GET"]),
            Route("/v1/table", _list_tables, methods=["GET"]),
            Route(
                "/v1/table/{table_id:path}/index/{index_name:path}/register",
                _register_index,
                methods=["POST"],
            ),
            Route(
                "/v1/table/{table_id:path}/register",
                _register_table,
                methods=["POST"],
            ),
            Route(
                "/v1/table/{table_id:path}/describe",
                _describe_table,
                methods=["POST"],
            ),
            Route(
                "/v1/table/{table_id:path}/query",
                _query_table,
                methods=["POST"],
            ),
            Route(
                "/v1/table/{table_id:path}/explain",
                _explain_table,
                methods=["POST"],
            ),
            Route(
                "/v1/table/{table_id:path}/to_sql",
                _to_sql,
                methods=["POST"],
            ),
            Route(
                "/v1/table/{table_id:path}/deregister",
                _deregister_table,
                methods=["POST"],
            ),
            Route(
                "/v1/table/{table_id:path}/create_index",
                _create_index,
                methods=["POST"],
            ),
            Route(
                "/v1/table/{table_id:path}/index/list",
                _list_indexes,
                methods=["POST"],
            ),
            Route(
                "/v1/table/{table_id:path}/index/{index_name:path}/stats",
                _index_status,
                methods=["POST"],
            ),
            Route(
                "/v1/table/{table_id:path}/index/{index_name:path}/refresh",
                _refresh_index,
                methods=["POST"],
            ),
            Route(
                "/v1/table/{table_id:path}/index/{index_name:path}/drop",
                _drop_index,
                methods=["POST"],
            ),
            Route("/v1/sql", _query_sql, methods=["POST"]),
            Route("/v1/sql/explain", _explain_sql, methods=["POST"]),
        ],
    )
    app.state.parqdb_source_policy = source_policy
    return app


async def _health(request: Request) -> Response:
    return _json_response(request, {"status": "ok"})


async def _openapi(request: Request) -> Response:
    return _json_response(request, openapi_document())


async def _list_tables(request: Request) -> Response:
    async def operation() -> Response:
        limit_text = request.query_params.get("limit", "100")
        token_text = request.query_params.get("page_token", "0")
        try:
            limit = int(limit_text)
            offset = int(token_text)
        except ValueError as error:
            raise ValueError("limit and page_token must be integers") from error
        if limit <= 0 or limit > MAX_LIST_TABLES_PAGE_SIZE:
            raise ValueError(f"limit must be between 1 and {MAX_LIST_TABLES_PAGE_SIZE}")
        if offset < 0:
            raise ValueError("page_token must not be negative")
        tables = await _service(request).list_tables()
        page = tables[offset : offset + limit]
        next_offset = offset + len(page)
        return _json_response(
            request,
            {
                "tables": [identifier_to_json(identifier) for identifier in page],
                "next_page_token": (
                    str(next_offset) if next_offset < len(tables) else None
                ),
            },
        )

    return await _handle(request, operation)


async def _register_table(request: Request) -> Response:
    async def operation() -> Response:
        values = registration_from_json(await _json_body(request))
        name = values.pop("name")
        source = values.pop("source")
        _validate_registration_route(request, name)
        values["path"] = _source_policy(request).authorize(source)
        await _service(request).register_parquet(name, **values)
        descriptor = await _service(request).table(name)
        return _json_response(
            request,
            table_descriptor_to_json(descriptor.identifier, descriptor.schema),
        )

    return await _handle(request, operation)


async def _describe_table(request: Request) -> Response:
    async def operation() -> Response:
        body = await _json_body(request)
        reference = body.get("reference")
        if not isinstance(reference, str) or not reference:
            raise ValueError("reference must be a non-empty string")
        descriptor = await _service(request).table(reference)
        _validate_route_identifier(request, descriptor.identifier)
        return _json_response(
            request,
            table_descriptor_to_json(descriptor.identifier, descriptor.schema),
        )

    return await _handle(request, operation)


async def _deregister_table(request: Request) -> Response:
    async def operation() -> Response:
        body = await _json_body(request)
        _only_fields(body, {"identifier"})
        identifier = _scoped_identifier(request, body, field="identifier")
        await _service(request).deregister_table(identifier)
        return _json_response(request, {"deregistered": True})

    return await _handle(request, operation)


async def _create_index(request: Request) -> Response:
    async def operation() -> Response:
        body = await _json_body(request)
        _only_fields(
            body,
            {"source", "index", "column", "key", "config", "writer_options"},
        )
        identifier = _scoped_identifier(request, body)
        index = _required_string(body.get("index"), "index")
        key = _string_array(body.get("key"), "key", allow_empty=False)
        await _service(request).create_index(
            identifier,
            index,
            column=_required_string(body.get("column"), "column"),
            key=key,
            config=ivf_from_json(body.get("config")),
            writer_options=writer_options_from_json(body.get("writer_options")),
            wait_timeout=None,
        )
        return _json_response(request, {"accepted": True}, status_code=202)

    return await _handle(request, operation)


async def _list_indexes(request: Request) -> Response:
    async def operation() -> Response:
        body = await _json_body(request)
        _only_fields(body, {"source"})
        identifier = _scoped_identifier(request, body)
        indexes = await _service(request).list_indexes(identifier)
        return _json_response(
            request,
            {"indexes": [index_info_to_json(index) for index in indexes]},
        )

    return await _handle(request, operation)


async def _register_index(request: Request) -> Response:
    async def operation() -> Response:
        body = await _json_body(request)
        _only_fields(
            body,
            {"source", "index", "metadata_location"},
        )
        identifier = _scoped_identifier(request, body)
        index = _scoped_index(request, body)
        metadata_location = _required_string(
            body.get("metadata_location"), "metadata_location"
        )
        await _service(request).register_index(
            identifier,
            index,
            metadata_location=metadata_location,
        )
        return _json_response(request, {"registered": True})

    return await _handle(request, operation)


async def _index_status(request: Request) -> Response:
    async def operation() -> Response:
        body = await _json_body(request)
        _only_fields(body, {"source", "index"})
        identifier = _scoped_identifier(request, body)
        index = _scoped_index(request, body)
        status = await _service(request).index_status(identifier, index)
        return _json_response(request, index_status_to_json(status))

    return await _handle(request, operation)


async def _refresh_index(request: Request) -> Response:
    async def operation() -> Response:
        body = await _json_body(request)
        _only_fields(body, {"source", "index", "config", "writer_options"})
        identifier = _scoped_identifier(request, body)
        index = _scoped_index(request, body)
        config_value = body.get("config")
        await _service(request).refresh_index(
            identifier,
            index,
            config=None if config_value is None else ivf_from_json(config_value),
            writer_options=writer_options_from_json(body.get("writer_options")),
            wait_timeout=None,
        )
        return _json_response(request, {"accepted": True}, status_code=202)

    return await _handle(request, operation)


async def _drop_index(request: Request) -> Response:
    async def operation() -> Response:
        body = await _json_body(request)
        _only_fields(body, {"source", "index"})
        identifier = _scoped_identifier(request, body)
        index = _scoped_index(request, body)
        await _service(request).drop_index(identifier, index)
        return _json_response(request, {"dropped": True})

    return await _handle(request, operation)


async def _query_table(request: Request) -> Response:
    async def operation() -> Response:
        query = vector_query_from_json(await _json_body(request))
        _validate_route_identifier(request, query.source)
        stream = await _service(request).stream(query)
        return await _stream_response(request, stream)

    return await _handle(request, operation)


async def _query_sql(request: Request) -> Response:
    async def operation() -> Response:
        body = await _json_body(request)
        _only_fields(body, {"sql"})
        sql = body.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("sql must be a non-empty string")
        stream = await _service(request).stream(sql)
        return await _stream_response(request, stream)

    return await _handle(request, operation)


async def _explain_table(request: Request) -> Response:
    async def operation() -> Response:
        body = await _json_body(request)
        _only_fields(body, {"query", "verbose", "analyze"})
        query = vector_query_from_json(body.get("query"))
        _validate_route_identifier(request, query.source)
        plan = await _service(request).explain(
            query,
            verbose=_boolean(body.get("verbose", False), "verbose"),
            analyze=_boolean(body.get("analyze", False), "analyze"),
        )
        return _json_response(request, {"plan": plan})

    return await _handle(request, operation)


async def _explain_sql(request: Request) -> Response:
    async def operation() -> Response:
        body = await _json_body(request)
        _only_fields(body, {"sql", "verbose", "analyze"})
        sql = body.get("sql")
        if not isinstance(sql, str) or not sql.strip():
            raise ValueError("sql must be a non-empty string")
        plan = await _service(request).explain(
            sql,
            verbose=_boolean(body.get("verbose", False), "verbose"),
            analyze=_boolean(body.get("analyze", False), "analyze"),
        )
        return _json_response(request, {"plan": plan})

    return await _handle(request, operation)


async def _to_sql(request: Request) -> Response:
    async def operation() -> Response:
        query = vector_query_from_json(await _json_body(request))
        _validate_route_identifier(request, query.source)
        sql = await _service(request).to_sql(query)
        return _json_response(request, {"sql": sql})

    return await _handle(request, operation)


async def _stream_response(
    request: Request, stream: AsyncBatchStream
) -> StreamingResponse:
    encoded = await encode_ipc_stream(stream)

    async def chunks() -> AsyncIterator[bytes]:
        try:
            async for chunk in encoded:
                yield chunk
        finally:
            await encoded.aclose()

    return StreamingResponse(
        chunks(),
        media_type=ARROW_STREAM_MEDIA_TYPE,
        headers={"x-request-id": _request_id(request)},
        background=BackgroundTask(encoded.aclose),
    )


async def _handle(
    request: Request,
    operation: Callable[[], Awaitable[Response]],
) -> Response:
    _request_id(request)
    try:
        return await operation()
    except Exception as error:
        detail = classify_server_error(error, _request_id(request))
        return JSONResponse(
            detail.json(),
            status_code=detail.status,
            headers={"x-request-id": detail.request_id},
        )


async def _json_body(request: Request) -> dict[str, Any]:
    content_type = request.headers.get("content-type", "").split(";", 1)[0]
    if content_type != "application/json":
        raise ValueError("content-type must be application/json")
    content_length = request.headers.get("content-length")
    if content_length is not None:
        try:
            length = int(content_length)
        except ValueError as error:
            raise ValueError("content-length must be an integer") from error
        if length < 0:
            raise ValueError("content-length must not be negative")
        if length > MAX_JSON_BODY_BYTES:
            raise ValueError("JSON request body is too large")
    payload = bytearray()
    async for chunk in request.stream():
        payload.extend(chunk)
        if len(payload) > MAX_JSON_BODY_BYTES:
            raise ValueError("JSON request body is too large")
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("request body must contain valid JSON") from error
    if not isinstance(value, dict):
        raise ValueError("request body must be a JSON object")
    return value


def _validate_route_identifier(
    request: Request,
    identifier: TableIdentifier,
) -> None:
    delimiter = request.query_params.get("delimiter", DEFAULT_IDENTIFIER_DELIMITER)
    path = decode_identifier_path(request.path_params["table_id"], delimiter=delimiter)
    if not identifier_matches_path(identifier, path):
        raise ValueError("route and request table identifiers do not match")


def _validate_registration_route(request: Request, name: str) -> None:
    delimiter = request.query_params.get("delimiter", DEFAULT_IDENTIFIER_DELIMITER)
    path = decode_identifier_path(request.path_params["table_id"], delimiter=delimiter)
    if path != (name,):
        raise ValueError("route and request table names do not match")


def _scoped_identifier(
    request: Request,
    body: Mapping[str, Any],
    *,
    field: str = "source",
) -> TableIdentifier:
    identifier = identifier_from_json(body.get(field))
    _validate_route_identifier(request, identifier)
    return identifier


def _scoped_index(request: Request, body: Mapping[str, Any]) -> str:
    index = _required_string(body.get("index"), "index")
    if unquote(request.path_params["index_name"]) != index:
        raise ValueError("route and request index names do not match")
    return index


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    if value is None:
        supplied = request.headers.get("x-request-id")
        value = (
            supplied
            if supplied
            and len(supplied) <= 128
            and all(32 <= ord(character) < 127 for character in supplied)
            else uuid.uuid4().hex
        )
        request.state.request_id = value
    return value


def _json_response(
    request: Request,
    value: object,
    *,
    status_code: int = 200,
) -> JSONResponse:
    return JSONResponse(
        value,
        status_code=status_code,
        headers={"x-request-id": _request_id(request)},
    )


def _service(request: Request) -> SessionService:
    service = getattr(request.app.state, "parqdb_service", None)
    if not isinstance(service, SessionService):
        raise RuntimeError("ParqDB server has not completed startup")
    return service


def _source_policy(request: Request) -> SourceUriPolicy:
    policy = getattr(request.app.state, "parqdb_source_policy", None)
    if not isinstance(policy, SourceUriPolicy):
        raise RuntimeError("ParqDB server source policy is not configured")
    return policy


def _only_fields(body: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ValueError(f"request contains unknown field: {unknown[0]}")


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _required_string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _string_array(value: object, name: str, *, allow_empty: bool) -> list[str]:
    if not isinstance(value, list) or (not value and not allow_empty):
        qualifier = "" if allow_empty else " non-empty"
        raise ValueError(f"{name} must be a{qualifier} array of strings")
    return [_required_string(item, f"{name} item") for item in value]
