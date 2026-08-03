from __future__ import annotations

from importlib import import_module
from typing import Any

from ...backends.v1 import (
    BINARY,
    BOOLEAN,
    DATE,
    DOUBLE,
    FLOAT,
    INT,
    LONG,
    STRING,
    TIME,
    TIMESTAMP,
    TIMESTAMPTZ,
    CanonicalSchema,
    DecimalType,
    Field,
    ListType,
    LogicalType,
    MapType,
    StructType,
)


def canonical_schema(
    schema: Any,
    *,
    nullability_known: bool = True,
) -> CanonicalSchema:
    """Map a PySpark StructType to canonical Iceberg logical types."""
    types = _types()
    if not isinstance(schema, types.StructType):
        raise TypeError("Spark relation schema must be a StructType")
    return CanonicalSchema(
        tuple(
            Field(
                str(field.name),
                _logical_type(field.dataType, types),
                required=not field.nullable,
            )
            for field in schema.fields
        ),
        nullability_known=nullability_known,
    )


def _logical_type(data_type: Any, types: Any) -> LogicalType:
    if isinstance(data_type, types.BooleanType):
        return BOOLEAN
    if isinstance(data_type, types.IntegerType):
        return INT
    if isinstance(data_type, types.LongType):
        return LONG
    if isinstance(data_type, types.FloatType):
        return FLOAT
    if isinstance(data_type, types.DoubleType):
        return DOUBLE
    if isinstance(data_type, types.DecimalType):
        return DecimalType(data_type.precision, data_type.scale)
    if isinstance(data_type, types.DateType):
        return DATE
    time_type = getattr(types, "TimeType", None)
    if time_type is not None and isinstance(data_type, time_type):
        return TIME
    timestamp_ntz = getattr(types, "TimestampNTZType", None)
    if timestamp_ntz is not None and isinstance(data_type, timestamp_ntz):
        return TIMESTAMP
    if isinstance(data_type, types.TimestampType):
        return TIMESTAMPTZ
    if isinstance(data_type, types.StringType):
        return STRING
    if isinstance(data_type, types.BinaryType):
        return BINARY
    if isinstance(data_type, types.ArrayType):
        return ListType(
            _logical_type(data_type.elementType, types),
            element_required=not data_type.containsNull,
        )
    if isinstance(data_type, types.MapType):
        return MapType(
            _logical_type(data_type.keyType, types),
            _logical_type(data_type.valueType, types),
            value_required=not data_type.valueContainsNull,
        )
    if isinstance(data_type, types.StructType):
        return StructType(
            tuple(
                Field(
                    str(field.name),
                    _logical_type(field.dataType, types),
                    required=not field.nullable,
                )
                for field in data_type.fields
            )
        )
    raise TypeError(
        f"unsupported Spark type for Iceberg mapping: {data_type.simpleString()}"
    )


def _types() -> Any:
    try:
        return import_module("pyspark.sql.types")
    except ImportError as error:
        raise ImportError(
            "Spark support requires the 'spark' extra: pip install 'relify[spark]'"
        ) from error
