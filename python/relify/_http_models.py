from __future__ import annotations

import base64
import math
from collections.abc import Mapping, Sequence
from typing import Any
from urllib.parse import quote, unquote

import pyarrow

from .identifier import TableIdentifier
from .query import VectorQuery

ARROW_STREAM_MEDIA_TYPE = "application/vnd.apache.arrow.stream"
DEFAULT_IDENTIFIER_DELIMITER = "$"
MAX_JSON_BODY_BYTES = 1024 * 1024
MAX_LIST_TABLES_PAGE_SIZE = 1000


def identifier_to_json(identifier: TableIdentifier) -> dict[str, Any]:
    return {
        "catalog": identifier.catalog,
        "namespace": list(identifier.namespace),
        "name": identifier.name,
    }


def identifier_from_json(value: object) -> TableIdentifier:
    body = _object(value, "table identifier")
    _reject_unknown(body, {"catalog", "namespace", "name"}, "table identifier")
    catalog = _string(body.get("catalog"), "catalog")
    name = _string(body.get("name"), "name")
    namespace_value = body.get("namespace")
    if not isinstance(namespace_value, list):
        raise ValueError("namespace must be an array of strings")
    namespace = tuple(
        _string(segment, "namespace segment") for segment in namespace_value
    )
    return TableIdentifier(catalog, namespace, name)


def encode_identifier_path(
    identifier: TableIdentifier,
    *,
    delimiter: str = DEFAULT_IDENTIFIER_DELIMITER,
) -> tuple[str, str]:
    delimiter = _identifier_delimiter(delimiter)
    segments = (*identifier.namespace, identifier.name)
    if any(delimiter in segment for segment in segments):
        raise ValueError("identifier delimiter must not occur in a table identifier")
    return delimiter.join(quote(segment, safe="") for segment in segments), delimiter


def decode_identifier_path(value: str, *, delimiter: str) -> tuple[str, ...]:
    delimiter = _identifier_delimiter(delimiter)
    if not value:
        raise ValueError("table identifier must not be empty")
    segments = tuple(unquote(segment) for segment in value.split(delimiter))
    if any(not segment for segment in segments):
        raise ValueError("table identifier contains an empty segment")
    return segments


def identifier_matches_path(
    identifier: TableIdentifier,
    path_segments: Sequence[str],
) -> bool:
    return (*identifier.namespace, identifier.name) == tuple(path_segments)


def vector_query_to_json(query: VectorQuery) -> dict[str, Any]:
    return {
        "source": identifier_to_json(query.source),
        "query": list(query.query),
        "column": query.column,
        "index": query.index,
        "projection": (
            list(query.projection) if query.projection is not None else None
        ),
        "result_limit": query.result_limit,
        "probe_count": query.probe_count,
        "predicate": query.predicate,
        "bypass_index": query.bypass_index,
    }


def vector_query_from_json(value: object) -> VectorQuery:
    body = _object(value, "vector query")
    fields = {
        "source",
        "query",
        "column",
        "index",
        "projection",
        "result_limit",
        "probe_count",
        "predicate",
        "bypass_index",
    }
    _reject_unknown(body, fields, "vector query")
    vector_value = body.get("query")
    if not isinstance(vector_value, list) or not vector_value:
        raise ValueError("query must be a non-empty array of finite numbers")
    vector: list[float] = []
    for item in vector_value:
        if not isinstance(item, (int, float)) or isinstance(item, bool):
            raise ValueError("query must be a non-empty array of finite numbers")
        number = float(item)
        if not math.isfinite(number):
            raise ValueError("query must be a non-empty array of finite numbers")
        vector.append(number)

    projection_value = body.get("projection")
    projection: tuple[str, ...] | None
    if projection_value is None:
        projection = None
    elif isinstance(projection_value, list):
        projection = tuple(
            _string(item, "projection field") for item in projection_value
        )
    else:
        raise ValueError("projection must be an array of strings or null")

    result_limit = _positive_integer(body.get("result_limit", 10), "result_limit")
    probe_count_value = body.get("probe_count")
    probe_count = (
        None
        if probe_count_value is None
        else _positive_integer(probe_count_value, "probe_count")
    )
    bypass_index = body.get("bypass_index", False)
    if not isinstance(bypass_index, bool):
        raise ValueError("bypass_index must be a boolean")

    return VectorQuery(
        source=identifier_from_json(body.get("source")),
        query=tuple(vector),
        column=_optional_string(body.get("column"), "column"),
        index=_optional_string(body.get("index"), "index"),
        projection=projection,
        result_limit=result_limit,
        probe_count=probe_count,
        predicate=_optional_string(body.get("predicate"), "predicate"),
        bypass_index=bypass_index,
    )


def schema_to_base64(schema: pyarrow.Schema) -> str:
    return base64.b64encode(schema.serialize().to_pybytes()).decode("ascii")


def schema_from_base64(value: object) -> pyarrow.Schema:
    encoded = _string(value, "schema")
    try:
        payload = base64.b64decode(encoded, validate=True)
        return pyarrow.ipc.read_schema(pyarrow.BufferReader(payload))
    except (ValueError, pyarrow.ArrowException) as error:
        raise ValueError("schema is not a valid Arrow IPC schema message") from error


def table_descriptor_to_json(
    identifier: TableIdentifier,
    schema: pyarrow.Schema,
) -> dict[str, Any]:
    return {
        "identifier": identifier_to_json(identifier),
        "schema": schema_to_base64(schema),
    }


def _identifier_delimiter(value: object) -> str:
    delimiter = _string(value, "delimiter")
    if len(delimiter) != 1 or delimiter in {"/", "%"}:
        raise ValueError("delimiter must be one character other than '/' or '%'")
    return delimiter


def _object(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping) or any(not isinstance(key, str) for key in value):
        raise ValueError(f"{name} must be a JSON object")
    return value


def _reject_unknown(
    body: Mapping[str, Any],
    fields: set[str],
    name: str,
) -> None:
    unknown = sorted(set(body) - fields)
    if unknown:
        raise ValueError(f"{name} contains unknown field: {unknown[0]}")


def _string(value: object, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{name} must be a non-empty string")
    return value


def _optional_string(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _string(value, name)


def _positive_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value
