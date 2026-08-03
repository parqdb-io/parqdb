from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import TypeAlias

import pyarrow

from .planning import ResolvedSearch


class PrimitiveKind(StrEnum):
    BOOLEAN = "boolean"
    INT = "int"
    LONG = "long"
    FLOAT = "float"
    DOUBLE = "double"
    DATE = "date"
    TIME = "time"
    TIMESTAMP = "timestamp"
    TIMESTAMPTZ = "timestamptz"
    STRING = "string"
    UUID = "uuid"
    BINARY = "binary"


@dataclass(frozen=True, slots=True)
class PrimitiveType:
    kind: PrimitiveKind


@dataclass(frozen=True, slots=True)
class DecimalType:
    precision: int
    scale: int

    def __post_init__(self) -> None:
        if self.precision <= 0:
            raise ValueError("decimal precision must be positive")
        if self.scale < 0 or self.scale > self.precision:
            raise ValueError("decimal scale must be in 0..=precision")


@dataclass(frozen=True, slots=True)
class FixedType:
    length: int

    def __post_init__(self) -> None:
        if self.length <= 0:
            raise ValueError("fixed length must be positive")


@dataclass(frozen=True, slots=True)
class ListType:
    element_type: LogicalType
    element_required: bool


@dataclass(frozen=True, slots=True)
class MapType:
    key_type: LogicalType
    value_type: LogicalType
    value_required: bool


@dataclass(frozen=True, slots=True)
class Field:
    name: str
    field_type: LogicalType
    required: bool

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name:
            raise ValueError("field name must not be empty")


@dataclass(frozen=True, slots=True)
class StructType:
    fields: tuple[Field, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        _validate_unique_names(self.fields)


LogicalType: TypeAlias = (
    PrimitiveType | DecimalType | FixedType | ListType | MapType | StructType
)


@dataclass(frozen=True, slots=True)
class CanonicalSchema:
    """Host schema expressed with Relify's canonical Iceberg logical types."""

    fields: tuple[Field, ...]
    nullability_known: bool = True

    def __post_init__(self) -> None:
        object.__setattr__(self, "fields", tuple(self.fields))
        _validate_unique_names(self.fields)
        if not isinstance(self.nullability_known, bool):
            raise TypeError("nullability_known must be a boolean")

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(field.name for field in self.fields)

    def field(self, name: str) -> Field | None:
        return next((field for field in self.fields if field.name == name), None)


BOOLEAN = PrimitiveType(PrimitiveKind.BOOLEAN)
INT = PrimitiveType(PrimitiveKind.INT)
LONG = PrimitiveType(PrimitiveKind.LONG)
FLOAT = PrimitiveType(PrimitiveKind.FLOAT)
DOUBLE = PrimitiveType(PrimitiveKind.DOUBLE)
DATE = PrimitiveType(PrimitiveKind.DATE)
TIME = PrimitiveType(PrimitiveKind.TIME)
TIMESTAMP = PrimitiveType(PrimitiveKind.TIMESTAMP)
TIMESTAMPTZ = PrimitiveType(PrimitiveKind.TIMESTAMPTZ)
STRING = PrimitiveType(PrimitiveKind.STRING)
UUID = PrimitiveType(PrimitiveKind.UUID)
BINARY = PrimitiveType(PrimitiveKind.BINARY)


def schema_from_pyarrow(schema: pyarrow.Schema) -> CanonicalSchema:
    """Map a PyArrow schema to canonical Iceberg logical types."""
    if not isinstance(schema, pyarrow.Schema):
        raise TypeError("schema must be a pyarrow.Schema")
    return CanonicalSchema(
        tuple(
            Field(
                field.name,
                _type_from_pyarrow(field.type),
                required=not field.nullable,
            )
            for field in schema
        )
    )


def vector_fields(schema: CanonicalSchema) -> tuple[str, ...]:
    """Return top-level list<float> fields."""
    return tuple(
        field.name for field in schema.fields if _is_float_list(field.field_type)
    )


def validate_ivf_schemas(
    search: ResolvedSearch,
    *,
    source: CanonicalSchema,
    centroids: CanonicalSchema,
    postings: CanonicalSchema,
) -> None:
    """Validate IVF fields visible through one backend schema."""
    if "_distance" in source.names:
        raise TypeError("source table must not contain reserved column _distance")
    source_vector = source.field(search.vector_field)
    if source_vector is None or not _is_float_list(source_vector.field_type):
        raise TypeError("source vector field must be list<float>")

    cid = centroids.field("cid")
    centroid = centroids.field("centroid")
    if (
        cid is None
        or centroid is None
        or not _required_if_known(centroids, cid)
        or cid.field_type != INT
        or not _required_if_known(centroids, centroid)
        or not _is_float_list(
            centroid.field_type,
            required_elements=centroids.nullability_known,
        )
    ):
        raise TypeError("invalid IVF centroid schema")

    posting_cid = postings.field("cid")
    if (
        posting_cid is None
        or not _required_if_known(postings, posting_cid)
        or posting_cid.field_type != INT
    ):
        raise TypeError("invalid IVF postings cid field")

    for position, source_name in enumerate(search.source_key_fields, start=1):
        source_field = source.field(source_name)
        posting_field = postings.field(f"key_{position}")
        if (
            source_field is None
            or posting_field is None
            or not _required_if_known(postings, posting_field)
            or not _is_supported_key(source_field.field_type)
            or posting_field.field_type != source_field.field_type
        ):
            raise TypeError(f"invalid IVF postings key field: key_{position}")

    vector = postings.field("vector")
    if search.store_vectors:
        if (
            vector is None
            or not _required_if_known(postings, vector)
            or not _is_float_list(
                vector.field_type,
                required_elements=postings.nullability_known,
            )
        ):
            raise TypeError("invalid IVF postings vector field")
    elif vector is not None:
        raise TypeError("IVF postings vector field must be absent")


def _type_from_pyarrow(data_type: pyarrow.DataType) -> LogicalType:
    types = pyarrow.types
    if types.is_boolean(data_type):
        return BOOLEAN
    if types.is_int32(data_type):
        return INT
    if types.is_int64(data_type):
        return LONG
    if types.is_float32(data_type):
        return FLOAT
    if types.is_float64(data_type):
        return DOUBLE
    if types.is_decimal(data_type):
        return DecimalType(data_type.precision, data_type.scale)
    if types.is_date(data_type):
        return DATE
    if types.is_time(data_type):
        return TIME
    if types.is_timestamp(data_type):
        return TIMESTAMPTZ if data_type.tz is not None else TIMESTAMP
    if (
        types.is_string(data_type)
        or types.is_large_string(data_type)
        or _arrow_type_predicate(types, "is_string_view", data_type)
    ):
        return STRING
    if (
        types.is_binary(data_type)
        or types.is_large_binary(data_type)
        or _arrow_type_predicate(types, "is_binary_view", data_type)
    ):
        return BINARY
    if types.is_fixed_size_binary(data_type):
        if _is_arrow_uuid(data_type):
            return UUID
        return FixedType(data_type.byte_width)
    if types.is_list(data_type) or types.is_large_list(data_type):
        return ListType(
            _type_from_pyarrow(data_type.value_type),
            element_required=not data_type.value_field.nullable,
        )
    if types.is_struct(data_type):
        return StructType(
            tuple(
                Field(
                    field.name,
                    _type_from_pyarrow(field.type),
                    required=not field.nullable,
                )
                for field in data_type
            )
        )
    if types.is_map(data_type):
        return MapType(
            _type_from_pyarrow(data_type.key_type),
            _type_from_pyarrow(data_type.item_type),
            value_required=not data_type.item_field.nullable,
        )
    raise TypeError(f"unsupported Arrow type for Iceberg mapping: {data_type}")


def _is_arrow_uuid(data_type: pyarrow.DataType) -> bool:
    uuid = getattr(pyarrow, "UuidType", None)
    return uuid is not None and isinstance(data_type, uuid)


def _arrow_type_predicate(
    types: object, name: str, data_type: pyarrow.DataType
) -> bool:
    predicate = getattr(types, name, None)
    return callable(predicate) and bool(predicate(data_type))


def _is_float_list(
    field_type: LogicalType,
    *,
    required_elements: bool = False,
) -> bool:
    return (
        isinstance(field_type, ListType)
        and field_type.element_type == FLOAT
        and (field_type.element_required or not required_elements)
    )


def _is_supported_key(field_type: LogicalType) -> bool:
    return field_type in {BOOLEAN, INT, LONG, BINARY, STRING, DATE} or isinstance(
        field_type,
        FixedType,
    )


def _required_if_known(schema: CanonicalSchema, field: Field) -> bool:
    return not schema.nullability_known or field.required


def _validate_unique_names(fields: tuple[Field, ...]) -> None:
    names = [field.name for field in fields]
    if len(names) != len(set(names)):
        raise ValueError("schema field names must be unique")
