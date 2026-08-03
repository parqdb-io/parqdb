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
    UUID,
    CanonicalSchema,
    DecimalType,
    Field,
    FixedType,
    ListType,
    LogicalType,
    MapType,
    StructType,
)


def canonical_schema(schema: Any) -> CanonicalSchema:
    """Map a PyIceberg Schema to canonical Iceberg logical types."""
    types = _types()
    fields = getattr(schema, "fields", None)
    if fields is None:
        raise TypeError("Iceberg relation returned an invalid schema")
    return CanonicalSchema(
        tuple(
            Field(
                str(field.name),
                _logical_type(field.field_type, types),
                required=bool(field.required),
            )
            for field in fields
        )
    )


def _logical_type(field_type: Any, types: Any) -> LogicalType:
    mappings = (
        (types.BooleanType, BOOLEAN),
        (types.IntegerType, INT),
        (types.LongType, LONG),
        (types.FloatType, FLOAT),
        (types.DoubleType, DOUBLE),
        (types.DateType, DATE),
        (types.TimeType, TIME),
        (types.TimestampType, TIMESTAMP),
        (types.TimestamptzType, TIMESTAMPTZ),
        (types.StringType, STRING),
        (types.UUIDType, UUID),
        (types.BinaryType, BINARY),
    )
    for iceberg_type, canonical in mappings:
        if isinstance(field_type, iceberg_type):
            return canonical
    if isinstance(field_type, types.DecimalType):
        return DecimalType(field_type.precision, field_type.scale)
    if isinstance(field_type, types.FixedType):
        return FixedType(field_type.length)
    if isinstance(field_type, types.ListType):
        return ListType(
            _logical_type(field_type.element_type, types),
            element_required=bool(field_type.element_required),
        )
    if isinstance(field_type, types.MapType):
        return MapType(
            _logical_type(field_type.key_type, types),
            _logical_type(field_type.value_type, types),
            value_required=bool(field_type.value_required),
        )
    if isinstance(field_type, types.StructType):
        return StructType(
            tuple(
                Field(
                    str(field.name),
                    _logical_type(field.field_type, types),
                    required=bool(field.required),
                )
                for field in field_type.fields
            )
        )
    raise TypeError(f"unsupported Iceberg type: {field_type}")


def _types() -> Any:
    try:
        return import_module("pyiceberg.types")
    except ImportError as error:
        raise ImportError(
            "StarRocks support requires the 'starrocks' extra: "
            "pip install 'relify[starrocks]'"
        ) from error
