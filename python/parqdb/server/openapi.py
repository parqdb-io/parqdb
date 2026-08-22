from __future__ import annotations

from typing import Any

from ..transport.models import (
    ARROW_STREAM_MEDIA_TYPE,
    DEFAULT_IDENTIFIER_DELIMITER,
    MAX_LIST_TABLES_PAGE_SIZE,
)


def openapi_document() -> dict[str, Any]:
    def json_content(schema: dict[str, Any]) -> dict[str, Any]:
        return {"application/json": {"schema": schema}}

    error_response = {
        "description": "ParqDB error",
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
        "info": {"title": "ParqDB API", "version": "v1"},
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
            "/v1/table/{id}/register": {
                "parameters": table_parameters,
                "post": {
                    "operationId": "registerParquetTable",
                    "requestBody": {
                        "required": True,
                        "content": json_content(
                            {"$ref": "#/components/schemas/RegisterParquetRequest"}
                        ),
                    },
                    "responses": json_ok(
                        {"$ref": "#/components/schemas/TableDescriptor"}
                    ),
                },
            },
            "/v1/table/{id}/deregister": {
                "parameters": table_parameters,
                "post": {
                    "operationId": "deregisterTable",
                    "requestBody": {
                        "required": True,
                        "content": json_content(
                            {"$ref": "#/components/schemas/ScopedTableRequest"}
                        ),
                    },
                    "responses": json_ok(
                        {"$ref": "#/components/schemas/DeregisterResponse"}
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
            "/v1/table/{id}/create_index": {
                "parameters": table_parameters,
                "post": {
                    "operationId": "createIndex",
                    "requestBody": {
                        "required": True,
                        "content": json_content(
                            {"$ref": "#/components/schemas/CreateIndexRequest"}
                        ),
                    },
                    "responses": {
                        "202": {
                            "description": "Index build accepted",
                            "content": json_content(
                                {"$ref": "#/components/schemas/AcceptedResponse"}
                            ),
                        },
                        "default": error_response,
                    },
                },
            },
            "/v1/table/{id}/index/list": {
                "parameters": table_parameters,
                "post": {
                    "operationId": "listIndexes",
                    "requestBody": {
                        "required": True,
                        "content": json_content(
                            {"$ref": "#/components/schemas/IndexTableRequest"}
                        ),
                    },
                    "responses": json_ok(
                        {"$ref": "#/components/schemas/ListIndexesResponse"}
                    ),
                },
            },
            "/v1/table/{id}/index/{index_name}/stats": {
                "parameters": [
                    *table_parameters,
                    {
                        "name": "index_name",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                ],
                "post": {
                    "operationId": "getIndexStatus",
                    "requestBody": {
                        "required": True,
                        "content": json_content(
                            {"$ref": "#/components/schemas/ScopedIndexRequest"}
                        ),
                    },
                    "responses": json_ok({"$ref": "#/components/schemas/IndexStatus"}),
                },
            },
            "/v1/table/{id}/index/{index_name}/register": {
                "parameters": [
                    *table_parameters,
                    {
                        "name": "index_name",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                ],
                "post": {
                    "operationId": "registerIndex",
                    "requestBody": {
                        "required": True,
                        "content": json_content(
                            {"$ref": "#/components/schemas/RegisterIndexRequest"}
                        ),
                    },
                    "responses": json_ok(
                        {"$ref": "#/components/schemas/RegisterIndexResponse"}
                    ),
                },
            },
            "/v1/table/{id}/index/{index_name}/refresh": {
                "parameters": [
                    *table_parameters,
                    {
                        "name": "index_name",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                ],
                "post": {
                    "operationId": "refreshIndex",
                    "requestBody": {
                        "required": True,
                        "content": json_content(
                            {"$ref": "#/components/schemas/RefreshIndexRequest"}
                        ),
                    },
                    "responses": {
                        "202": {
                            "description": "Index refresh accepted",
                            "content": json_content(
                                {"$ref": "#/components/schemas/AcceptedResponse"}
                            ),
                        },
                        "default": error_response,
                    },
                },
            },
            "/v1/table/{id}/index/{index_name}/drop": {
                "parameters": [
                    *table_parameters,
                    {
                        "name": "index_name",
                        "in": "path",
                        "required": True,
                        "schema": {"type": "string"},
                    },
                ],
                "post": {
                    "operationId": "dropIndex",
                    "requestBody": {
                        "required": True,
                        "content": json_content(
                            {"$ref": "#/components/schemas/ScopedIndexRequest"}
                        ),
                    },
                    "responses": json_ok(
                        {"$ref": "#/components/schemas/DropIndexResponse"}
                    ),
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
                "RegisterParquetRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "name",
                        "source",
                        "partition_schema",
                        "parquet_pruning",
                        "file_extension",
                        "skip_metadata",
                        "schema",
                        "file_sort_order",
                    ],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "source": {"type": "string", "minLength": 1},
                        "partition_schema": {
                            "type": ["string", "null"],
                            "format": "byte",
                        },
                        "parquet_pruning": {"type": "boolean"},
                        "file_extension": {"type": "string", "minLength": 1},
                        "skip_metadata": {"type": "boolean"},
                        "schema": {"type": ["string", "null"], "format": "byte"},
                        "file_sort_order": {
                            "type": "array",
                            "items": {
                                "type": "array",
                                "items": {"type": "string", "minLength": 1},
                            },
                        },
                    },
                },
                "ScopedTableRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["identifier"],
                    "properties": {
                        "identifier": {"$ref": "#/components/schemas/TableIdentifier"}
                    },
                },
                "IndexTableRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source"],
                    "properties": {
                        "source": {"$ref": "#/components/schemas/TableIdentifier"}
                    },
                },
                "IvfConfig": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["type", "nlist", "encoding", "metric"],
                    "properties": {
                        "type": {"const": "ivf"},
                        "nlist": {"type": "integer", "minimum": 1},
                        "encoding": {
                            "type": "string",
                            "enum": ["source", "lvq4", "lvq8"],
                        },
                        "metric": {
                            "type": "string",
                            "enum": ["l2_squared", "cosine"],
                        },
                    },
                },
                "WriteOptions": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "partitions",
                        "compression",
                        "target_file_size",
                        "max_row_group_rows",
                        "write_batch_rows",
                    ],
                    "properties": {
                        "partitions": {"type": ["integer", "null"], "minimum": 1},
                        "compression": {"type": "string", "minLength": 1},
                        "target_file_size": {"type": "integer", "minimum": 1},
                        "max_row_group_rows": {
                            "type": ["integer", "null"],
                            "minimum": 1,
                        },
                        "write_batch_rows": {"type": "integer", "minimum": 1},
                    },
                },
                "CreateIndexRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "source",
                        "index",
                        "column",
                        "key",
                        "config",
                        "writer_options",
                    ],
                    "properties": {
                        "source": {"$ref": "#/components/schemas/TableIdentifier"},
                        "index": {"type": "string", "minLength": 1},
                        "column": {"type": "string", "minLength": 1},
                        "key": {
                            "type": "array",
                            "minItems": 1,
                            "items": {"type": "string", "minLength": 1},
                        },
                        "config": {"$ref": "#/components/schemas/IvfConfig"},
                        "writer_options": {
                            "oneOf": [
                                {"$ref": "#/components/schemas/WriteOptions"},
                                {"type": "null"},
                            ]
                        },
                    },
                },
                "RegisterIndexRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "source",
                        "index",
                        "manifest_location",
                    ],
                    "properties": {
                        "source": {"$ref": "#/components/schemas/TableIdentifier"},
                        "index": {"type": "string", "minLength": 1},
                        "manifest_location": {"type": "string", "minLength": 1},
                    },
                },
                "RefreshIndexRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source", "index", "config", "writer_options"],
                    "properties": {
                        "source": {"$ref": "#/components/schemas/TableIdentifier"},
                        "index": {"type": "string", "minLength": 1},
                        "config": {
                            "oneOf": [
                                {"$ref": "#/components/schemas/IvfConfig"},
                                {"type": "null"},
                            ]
                        },
                        "writer_options": {
                            "oneOf": [
                                {"$ref": "#/components/schemas/WriteOptions"},
                                {"type": "null"},
                            ]
                        },
                    },
                },
                "ScopedIndexRequest": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["source", "index"],
                    "properties": {
                        "source": {"$ref": "#/components/schemas/TableIdentifier"},
                        "index": {"type": "string", "minLength": 1},
                    },
                },
                "IndexStatus": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "state",
                        "progress",
                        "phase",
                        "completed",
                        "total",
                        "current_snapshot_id",
                        "error",
                        "error_code",
                    ],
                    "properties": {
                        "state": {
                            "type": "string",
                            "enum": ["pending", "building", "ready", "failed"],
                        },
                        "progress": {"type": ["number", "null"]},
                        "phase": {"type": ["string", "null"]},
                        "completed": {"type": ["integer", "null"], "minimum": 0},
                        "total": {"type": ["integer", "null"], "minimum": 0},
                        "current_snapshot_id": {
                            "type": ["integer", "null"],
                            "minimum": 0,
                        },
                        "error": {"type": ["string", "null"]},
                        "error_code": {"type": ["string", "null"]},
                    },
                },
                "IndexInfo": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": [
                        "name",
                        "column",
                        "family",
                        "metric",
                        "parameters",
                        "current_snapshot_id",
                    ],
                    "properties": {
                        "name": {"type": "string", "minLength": 1},
                        "column": {"type": "string", "minLength": 1},
                        "family": {"type": "string", "minLength": 1},
                        "metric": {"type": "string", "minLength": 1},
                        "parameters": {
                            "type": "object",
                            "additionalProperties": {"type": "string"},
                        },
                        "current_snapshot_id": {"type": "integer", "minimum": 0},
                    },
                },
                "ListIndexesResponse": {
                    "type": "object",
                    "required": ["indexes"],
                    "properties": {
                        "indexes": {
                            "type": "array",
                            "items": {"$ref": "#/components/schemas/IndexInfo"},
                        }
                    },
                },
                "AcceptedResponse": {
                    "type": "object",
                    "required": ["accepted"],
                    "properties": {"accepted": {"const": True}},
                },
                "DeregisterResponse": {
                    "type": "object",
                    "required": ["deregistered"],
                    "properties": {"deregistered": {"const": True}},
                },
                "DropIndexResponse": {
                    "type": "object",
                    "required": ["dropped"],
                    "properties": {"dropped": {"const": True}},
                },
                "RegisterIndexResponse": {
                    "type": "object",
                    "required": ["registered"],
                    "properties": {"registered": {"const": True}},
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
