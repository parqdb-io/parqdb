from __future__ import annotations

import base64
import math
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Any
from urllib.parse import quote, unquote

import pyarrow

from ..build import IndexStatus
from ..config import IVF, WriteOptions
from ..identifier import TableIdentifier
from ..query import VectorQuery
from ..table import IndexInfo

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


def registration_to_json(
    name: str,
    source: str,
    *,
    table_partition_cols: list[tuple[str, str | pyarrow.DataType]] | None,
    parquet_pruning: bool,
    file_extension: str,
    skip_metadata: bool,
    schema: pyarrow.Schema | None,
    file_sort_order: Sequence[Sequence[object]] | None,
) -> dict[str, Any]:
    partition_schema = None
    if table_partition_cols is not None:
        partition_schema = schema_to_base64(
            pyarrow.schema(
                [
                    pyarrow.field(name, _arrow_type(data_type), nullable=False)
                    for name, data_type in table_partition_cols
                ]
            )
        )
    sort_order = _portable_sort_order(file_sort_order)
    return {
        "name": _string(name, "table name"),
        "source": _string(source, "source"),
        "partition_schema": partition_schema,
        "parquet_pruning": _boolean(parquet_pruning, "parquet_pruning"),
        "file_extension": _string(file_extension, "file_extension"),
        "skip_metadata": _boolean(skip_metadata, "skip_metadata"),
        "schema": schema_to_base64(schema) if schema is not None else None,
        "file_sort_order": sort_order,
    }


def registration_from_json(value: object) -> dict[str, Any]:
    body = _object(value, "registration")
    fields = {
        "name",
        "source",
        "partition_schema",
        "parquet_pruning",
        "file_extension",
        "skip_metadata",
        "schema",
        "file_sort_order",
    }
    _reject_unknown(body, fields, "registration")
    _require_fields(body, fields, "registration")
    partition_value = body.get("partition_schema")
    partition_schema = (
        None if partition_value is None else schema_from_base64(partition_value)
    )
    schema_value = body.get("schema")
    sort_order = _portable_sort_order(body.get("file_sort_order"))
    return {
        "name": _string(body.get("name"), "table name"),
        "source": _string(body.get("source"), "source"),
        "table_partition_cols": (
            None
            if partition_schema is None
            else [(field.name, field.type) for field in partition_schema]
        ),
        "parquet_pruning": _boolean(body.get("parquet_pruning"), "parquet_pruning"),
        "file_extension": _string(body.get("file_extension"), "file_extension"),
        "skip_metadata": _boolean(body.get("skip_metadata"), "skip_metadata"),
        "schema": None if schema_value is None else schema_from_base64(schema_value),
        "file_sort_order": sort_order,
    }


def ivf_to_json(config: IVF) -> dict[str, Any]:
    if not isinstance(config, IVF):
        raise TypeError("the first implementation supports only parqdb.IVF")
    return {
        "type": "ivf",
        "nlist": config.nlist,
        "encoding": config.encoding,
        "metric": config.metric,
    }


def ivf_from_json(value: object) -> IVF:
    body = _object(value, "index configuration")
    fields = {"type", "nlist", "encoding", "metric"}
    _reject_unknown(body, fields, "index configuration")
    _require_fields(body, fields, "index configuration")
    if body.get("type") != "ivf":
        raise ValueError("index configuration type must be 'ivf'")
    return IVF(
        nlist=_positive_integer(body.get("nlist"), "nlist"),
        encoding=_string(body.get("encoding"), "encoding"),
        metric=_string(body.get("metric"), "metric"),
    )


def writer_options_to_json(options: WriteOptions | None) -> dict[str, Any] | None:
    if options is None:
        return None
    if not isinstance(options, WriteOptions):
        raise TypeError("writer_options must be parqdb.WriteOptions")
    return {
        "partitions": options.partitions,
        "compression": options.compression,
        "target_file_size": options.target_file_size,
        "max_row_group_rows": options.max_row_group_rows,
        "write_batch_rows": options.write_batch_rows,
    }


def writer_options_from_json(value: object) -> WriteOptions | None:
    if value is None:
        return None
    body = _object(value, "writer options")
    fields = {
        "partitions",
        "compression",
        "target_file_size",
        "max_row_group_rows",
        "write_batch_rows",
    }
    _reject_unknown(body, fields, "writer options")
    _require_fields(body, fields, "writer options")
    partitions = _optional_positive_integer(body.get("partitions"), "partitions")
    max_row_group_rows = _optional_positive_integer(
        body.get("max_row_group_rows"), "max_row_group_rows"
    )
    return WriteOptions(
        partitions=partitions,
        compression=_string(body.get("compression"), "compression"),
        target_file_size=_positive_integer(
            body.get("target_file_size"), "target_file_size"
        ),
        max_row_group_rows=max_row_group_rows,
        write_batch_rows=_positive_integer(
            body.get("write_batch_rows"), "write_batch_rows"
        ),
    )


def index_status_to_json(status: IndexStatus) -> dict[str, Any]:
    return {
        "state": status.state,
        "progress": status.progress,
        "phase": status.phase,
        "completed": status.completed,
        "total": status.total,
        "current_snapshot_id": status.current_snapshot_id,
        "error": status.error,
        "error_code": status.error_code,
    }


def index_status_from_json(value: object) -> IndexStatus:
    body = _object(value, "index status")
    fields = {
        "state",
        "progress",
        "phase",
        "completed",
        "total",
        "current_snapshot_id",
        "error",
        "error_code",
    }
    _reject_unknown(body, fields, "index status")
    _require_fields(body, fields, "index status")
    state = body.get("state")
    if state not in {"pending", "building", "ready", "failed"}:
        raise ValueError("index status contains an invalid state")
    error = _optional_string(body.get("error"), "error")
    error_code = _optional_string(body.get("error_code"), "error_code")
    if (error is None) != (error_code is None):
        raise ValueError("index status error and error_code must both be null or set")
    return IndexStatus(
        state=state,
        progress=_optional_number(body.get("progress"), "progress"),
        phase=_optional_string(body.get("phase"), "phase"),
        completed=_optional_non_negative_integer(body.get("completed"), "completed"),
        total=_optional_non_negative_integer(body.get("total"), "total"),
        current_snapshot_id=_optional_non_negative_integer(
            body.get("current_snapshot_id"), "current_snapshot_id"
        ),
        error=error,
        error_code=error_code,
    )


def index_info_to_json(info: IndexInfo) -> dict[str, Any]:
    return {
        "name": info.name,
        "column": info.column,
        "family": info.family,
        "metric": info.metric,
        "parameters": dict(info.parameters),
        "current_snapshot_id": info.current_snapshot_id,
    }


def index_info_from_json(value: object) -> IndexInfo:
    body = _object(value, "index metadata")
    fields = {
        "name",
        "column",
        "family",
        "metric",
        "parameters",
        "current_snapshot_id",
    }
    _reject_unknown(body, fields, "index metadata")
    _require_fields(body, fields, "index metadata")
    parameters_value = _object(body.get("parameters"), "index parameters")
    parameters = {
        _string(key, "index parameter name"): _string(item, "index parameter value")
        for key, item in parameters_value.items()
    }
    return IndexInfo(
        name=_string(body.get("name"), "index name"),
        column=_string(body.get("column"), "index column"),
        family=_string(body.get("family"), "index family"),
        metric=_string(body.get("metric"), "index metric"),
        parameters=MappingProxyType(parameters),
        current_snapshot_id=_non_negative_integer(
            body.get("current_snapshot_id"), "current_snapshot_id"
        ),
    )


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


def _require_fields(
    body: Mapping[str, Any],
    fields: set[str],
    name: str,
) -> None:
    missing = sorted(fields - set(body))
    if missing:
        raise ValueError(f"{name} is missing required field: {missing[0]}")


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


def _non_negative_integer(value: object, name: str) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return value


def _optional_positive_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _positive_integer(value, name)


def _optional_non_negative_integer(value: object, name: str) -> int | None:
    if value is None:
        return None
    return _non_negative_integer(value, name)


def _optional_number(value: object, name: str) -> float | None:
    if value is None:
        return None
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise ValueError(f"{name} must be a finite number or null")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number or null")
    return result


def _boolean(value: object, name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{name} must be a boolean")
    return value


def _arrow_type(value: str | pyarrow.DataType) -> pyarrow.DataType:
    if isinstance(value, pyarrow.DataType):
        return value
    if not isinstance(value, str):
        raise TypeError("partition column types must be strings or PyArrow types")
    try:
        return pyarrow.type_for_alias(value)
    except ValueError as error:
        raise ValueError(f"unsupported partition column type: {value}") from error


def _portable_sort_order(
    value: object,
) -> list[list[str]]:
    if value is None:
        return []
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError("file_sort_order must be an array of field-name arrays")
    result: list[list[str]] = []
    for order in value:
        if not isinstance(order, Sequence) or isinstance(order, (str, bytes)):
            raise ValueError("file_sort_order must be an array of field-name arrays")
        result.append([_string(key, "sort field") for key in order])
    return result
