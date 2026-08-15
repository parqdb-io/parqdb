from __future__ import annotations

import json
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from starlette.applications import Starlette
from starlette.background import BackgroundTask
from starlette.requests import Request
from starlette.responses import JSONResponse, Response, StreamingResponse
from starlette.routing import Route

from ._http_errors import classify_server_error
from ._http_models import (
    ARROW_STREAM_MEDIA_TYPE,
    DEFAULT_IDENTIFIER_DELIMITER,
    MAX_JSON_BODY_BYTES,
    MAX_LIST_TABLES_PAGE_SIZE,
    decode_identifier_path,
    identifier_matches_path,
    identifier_to_json,
    table_descriptor_to_json,
    vector_query_from_json,
)
from ._ipc import encode_ipc_stream
from ._service import AsyncBatchStream, SessionService
from .datafusion import RuntimeEnvBuilder
from .datafusion import SessionConfig as DataFusionSessionConfig


def create_http_app(
    root: str | Path | None = None,
    *,
    catalog: str | None = None,
    index_root: str | None = None,
    storage_options: Mapping[str, str] | None = None,
    config: DataFusionSessionConfig | None = None,
    runtime: RuntimeEnvBuilder | None = None,
) -> Starlette:
    @asynccontextmanager
    async def lifespan(app: Starlette) -> AsyncIterator[None]:
        service = await SessionService.open(
            root,
            catalog=catalog,
            index_root=index_root,
            storage_options=storage_options,
            iceberg=None,
            config=config,
            runtime=runtime,
        )
        app.state.relify_service = service
        try:
            yield
        finally:
            await service.close()

    return _application(lifespan=lifespan)


def create_http_app_for_service(service: SessionService) -> Starlette:
    app = _application()
    app.state.relify_service = service
    return app


def _application(*, lifespan: Any = None) -> Starlette:
    return Starlette(
        debug=False,
        lifespan=lifespan,
        routes=[
            Route("/health", _health, methods=["GET"]),
            Route("/openapi.json", _openapi, methods=["GET"]),
            Route("/v1/table", _list_tables, methods=["GET"]),
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
            Route("/v1/sql", _query_sql, methods=["POST"]),
            Route("/v1/sql/explain", _explain_sql, methods=["POST"]),
        ],
    )


async def _health(request: Request) -> Response:
    return _json_response(request, {"status": "ok"})


async def _openapi(request: Request) -> Response:
    return _json_response(request, _openapi_document())


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


def _validate_route_identifier(request: Request, identifier: Any) -> None:
    delimiter = request.query_params.get("delimiter", DEFAULT_IDENTIFIER_DELIMITER)
    path = decode_identifier_path(request.path_params["table_id"], delimiter=delimiter)
    if not identifier_matches_path(identifier, path):
        raise ValueError("route and request table identifiers do not match")


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


def _json_response(request: Request, value: object) -> JSONResponse:
    return JSONResponse(value, headers={"x-request-id": _request_id(request)})


def _service(request: Request) -> SessionService:
    service = getattr(request.app.state, "relify_service", None)
    if not isinstance(service, SessionService):
        raise RuntimeError("Relify server has not completed startup")
    return service


def _only_fields(body: Mapping[str, Any], allowed: set[str]) -> None:
    unknown = sorted(set(body) - allowed)
    if unknown:
        raise ValueError(f"request contains unknown field: {unknown[0]}")


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _openapi_document() -> dict[str, Any]:
    def json_content(schema: dict[str, Any]) -> dict[str, Any]:
        return {"application/json": {"schema": schema}}

    error_response = {
        "description": "Relify error",
        "content": json_content({"$ref": "#/components/schemas/ErrorResponse"}),
    }
    arrow_response = {
        "description": "Arrow IPC stream",
        "content": {
            ARROW_STREAM_MEDIA_TYPE: {"schema": {"type": "string", "format": "binary"}}
        },
    }
    table_parameters = [
        {
            "name": "id",
            "in": "path",
            "required": True,
            "schema": {"type": "string"},
        },
        {
            "name": "delimiter",
            "in": "query",
            "required": False,
            "schema": {
                "type": "string",
                "minLength": 1,
                "maxLength": 1,
                "default": DEFAULT_IDENTIFIER_DELIMITER,
            },
        },
    ]
    vector_body = {
        "required": True,
        "content": json_content({"$ref": "#/components/schemas/VectorQuery"}),
    }
    sql_body = {
        "required": True,
        "content": json_content({"$ref": "#/components/schemas/SqlQuery"}),
    }

    def json_ok(schema: dict[str, Any]) -> dict[str, Any]:
        return {
            "200": {"description": "Success", "content": json_content(schema)},
            "default": error_response,
        }

    return {
        "openapi": "3.1.0",
        "info": {"title": "Relify API", "version": "v1"},
        "paths": {
            "/v1/table": {
                "get": {
                    "operationId": "listTables",
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "schema": {
                                "type": "integer",
                                "minimum": 1,
                                "maximum": MAX_LIST_TABLES_PAGE_SIZE,
                                "default": 100,
                            },
                        },
                        {
                            "name": "page_token",
                            "in": "query",
                            "schema": {"type": "string"},
                        },
                    ],
                    "responses": json_ok(
                        {"$ref": "#/components/schemas/ListTablesResponse"}
                    ),
                }
            },
            "/v1/table/{id}/describe": {
                "parameters": table_parameters,
                "post": {
                    "operationId": "describeTable",
                    "requestBody": {
                        "required": True,
                        "content": json_content(
                            {"$ref": "#/components/schemas/DescribeTableRequest"}
                        ),
                    },
                    "responses": json_ok(
                        {"$ref": "#/components/schemas/TableDescriptor"}
                    ),
                },
            },
            "/v1/table/{id}/query": {
                "parameters": table_parameters,
                "post": {
                    "operationId": "queryTable",
                    "requestBody": vector_body,
                    "responses": {"200": arrow_response, "default": error_response},
                },
            },
            "/v1/table/{id}/explain": {
                "parameters": table_parameters,
                "post": {
                    "operationId": "explainTable",
                    "requestBody": {
                        "required": True,
                        "content": json_content(
                            {"$ref": "#/components/schemas/ExplainVectorQuery"}
                        ),
                    },
                    "responses": json_ok({"$ref": "#/components/schemas/PlanResponse"}),
                },
            },
            "/v1/table/{id}/to_sql": {
                "parameters": table_parameters,
                "post": {
                    "operationId": "renderQuerySql",
                    "requestBody": vector_body,
                    "responses": json_ok({"$ref": "#/components/schemas/SqlQuery"}),
                },
            },
            "/v1/sql": {
                "post": {
                    "operationId": "executeSql",
                    "requestBody": sql_body,
                    "responses": {"200": arrow_response, "default": error_response},
                },
            },
            "/v1/sql/explain": {
                "post": {
                    "operationId": "explainSql",
                    "requestBody": {
                        "required": True,
                        "content": json_content(
                            {"$ref": "#/components/schemas/ExplainSqlQuery"}
                        ),
                    },
                    "responses": json_ok({"$ref": "#/components/schemas/PlanResponse"}),
                },
            },
        },
        "components": {
            "schemas": {
                "TableIdentifier": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["catalog", "namespace", "name"],
                    "properties": {
                        "catalog": {"type": "string", "minLength": 1},
                        "namespace": {
                            "type": "array",
                            "items": {"type": "string", "minLength": 1},
                        },
                        "name": {"type": "string", "minLength": 1},
                    },
                },
                "VectorQuery": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source", "query"],
                    "properties": {
                        "source": {"$ref": "#/components/schemas/TableIdentifier"},
                        "query": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "number"},
                        },
                        "column": {"type": ["string", "null"]},
                        "index": {"type": ["string", "null"]},
                        "projection": {
                            "type": ["array", "null"],
                            "items": {"type": "string", "minLength": 1},
                        },
                        "result_limit": {"type": "integer", "minimum": 1},
                        "probe_count": {
                            "type": ["integer", "null"],
                            "minimum": 1,
                        },
                        "predicate": {"type": ["string", "null"]},
                        "bypass_index": {"type": "boolean"},
                    },
                },
                "SqlQuery": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["sql"],
                    "properties": {"sql": {"type": "string", "minLength": 1}},
                },
                "ExplainVectorQuery": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["query"],
                    "properties": {
                        "query": {"$ref": "#/components/schemas/VectorQuery"},
                        "verbose": {"type": "boolean", "default": False},
                        "analyze": {"type": "boolean", "default": False},
                    },
                },
                "ExplainSqlQuery": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["sql"],
                    "properties": {
                        "sql": {"type": "string", "minLength": 1},
                        "verbose": {"type": "boolean", "default": False},
                        "analyze": {"type": "boolean", "default": False},
                    },
                },
                "DescribeTableRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["reference"],
                    "properties": {"reference": {"type": "string", "minLength": 1}},
                },
                "TableDescriptor": {
                    "type": "object",
                    "required": ["identifier", "schema"],
                    "properties": {
                        "identifier": {"$ref": "#/components/schemas/TableIdentifier"},
                        "schema": {"type": "string", "format": "byte"},
                    },
                },
                "ListTablesResponse": {
                    "type": "object",
                    "required": ["tables", "next_page_token"],
                    "properties": {
                        "tables": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/TableIdentifier"},
                        },
                        "next_page_token": {"type": ["string", "null"]},
                    },
                },
                "PlanResponse": {
                    "type": "object",
                    "required": ["plan"],
                    "properties": {"plan": {"type": "string"}},
                },
                "ErrorResponse": {
                    "type": "object",
                    "required": ["error"],
                    "properties": {
                        "error": {
                            "type": "object",
                            "required": ["code", "message", "request_id"],
                            "properties": {
                                "code": {"type": "string"},
                                "message": {"type": "string"},
                                "request_id": {"type": "string"},
                            },
                        }
                    },
                },
            },
        },
    }
