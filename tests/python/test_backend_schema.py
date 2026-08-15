from __future__ import annotations

import pyarrow as pa
import pytest
import relify
from relify.backends.v1 import (
    BINARY,
    FLOAT,
    INT,
    LONG,
    STRING,
    CanonicalSchema,
    Field,
    ListType,
    MapType,
    ResolvedSearch,
    schema_from_pyarrow,
    validate_ivf_schemas,
    vector_fields,
)


def test_pyarrow_schema_maps_to_canonical_iceberg_types() -> None:
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field(
                "embedding",
                pa.list_(pa.field("element", pa.float32(), nullable=True)),
                nullable=False,
            ),
            pa.field(
                "attributes",
                pa.map_(
                    pa.string(),
                    pa.field("value", pa.string(), nullable=False),
                ),
            ),
        ]
    )

    canonical = schema_from_pyarrow(schema)

    assert canonical.field("id") == Field("id", LONG, required=True)
    assert canonical.field("embedding") == Field(
        "embedding",
        ListType(FLOAT, element_required=False),
        required=True,
    )
    assert canonical.field("attributes") == Field(
        "attributes",
        MapType(STRING, STRING, value_required=True),
        required=False,
    )
    assert vector_fields(canonical) == ("embedding",)


def test_shared_ivf_schema_validation_accepts_optional_source_declarations() -> None:
    search = _search(posting_encoding="source")
    source = CanonicalSchema(
        (
            Field("id", LONG, required=False),
            Field(
                "embedding",
                ListType(FLOAT, element_required=False),
                required=False,
            ),
        )
    )
    centroids = CanonicalSchema(
        (
            Field("cid", INT, required=True),
            Field(
                "centroid",
                ListType(FLOAT, element_required=True),
                required=True,
            ),
        )
    )
    postings = CanonicalSchema(
        (
            Field("cid", INT, required=True),
            Field("key_1", LONG, required=True),
        )
    )

    validate_ivf_schemas(
        search,
        source=source,
        centroids=centroids,
        postings=postings,
    )


def test_shared_ivf_schema_validation_accepts_unknown_index_nullability() -> None:
    search = _search(posting_encoding="source")
    source = CanonicalSchema(
        (
            Field("id", LONG, required=False),
            Field(
                "embedding",
                ListType(FLOAT, element_required=False),
                required=False,
            ),
        ),
        nullability_known=False,
    )
    centroids = CanonicalSchema(
        (
            Field(
                "centroid",
                ListType(FLOAT, element_required=False),
                required=False,
            ),
            Field("cid", INT, required=False),
        ),
        nullability_known=False,
    )
    postings = CanonicalSchema(
        (
            Field("key_1", LONG, required=False),
            Field("cid", INT, required=False),
        ),
        nullability_known=False,
    )

    validate_ivf_schemas(
        search,
        source=source,
        centroids=centroids,
        postings=postings,
    )


def test_shared_ivf_schema_validation_rejects_known_optional_index_fields() -> None:
    search = _search(posting_encoding="source")
    source = CanonicalSchema(
        (
            Field("id", LONG, required=True),
            Field(
                "embedding",
                ListType(FLOAT, element_required=True),
                required=True,
            ),
        )
    )
    centroids = CanonicalSchema(
        (
            Field("cid", INT, required=False),
            Field(
                "centroid",
                ListType(FLOAT, element_required=True),
                required=True,
            ),
        )
    )
    postings = CanonicalSchema(
        (
            Field("cid", INT, required=True),
            Field("key_1", LONG, required=True),
        )
    )

    with pytest.raises(TypeError, match="invalid IVF centroid schema"):
        validate_ivf_schemas(
            search,
            source=source,
            centroids=centroids,
            postings=postings,
        )


def test_shared_ivf_schema_validation_rejects_unsupported_keys() -> None:
    search = _search(posting_encoding="source")
    source = CanonicalSchema(
        (
            Field("id", FLOAT, required=True),
            Field(
                "embedding",
                ListType(FLOAT, element_required=True),
                required=True,
            ),
        )
    )
    centroids = CanonicalSchema(
        (
            Field("cid", INT, required=True),
            Field(
                "centroid",
                ListType(FLOAT, element_required=True),
                required=True,
            ),
        )
    )
    postings = CanonicalSchema(
        (
            Field("cid", INT, required=True),
            Field("key_1", FLOAT, required=True),
        )
    )

    with pytest.raises(TypeError, match="invalid IVF postings key field"):
        validate_ivf_schemas(
            search,
            source=source,
            centroids=centroids,
            postings=postings,
        )


def test_shared_ivf_schema_validation_accepts_lvq_fields() -> None:
    search = _search(posting_encoding="lvq8")
    source = CanonicalSchema(
        (
            Field("id", LONG, required=True),
            Field("embedding", ListType(FLOAT, element_required=True), required=True),
        )
    )
    centroids = CanonicalSchema(
        (
            Field("cid", INT, required=True),
            Field("centroid", ListType(FLOAT, element_required=True), required=True),
        )
    )
    postings = CanonicalSchema(
        (
            Field("cid", INT, required=True),
            Field("key_1", LONG, required=True),
            Field("offset", FLOAT, required=True),
            Field("scale", FLOAT, required=True),
            Field("code", BINARY, required=True),
        )
    )

    validate_ivf_schemas(search, source=source, centroids=centroids, postings=postings)


def test_canonical_schema_rejects_duplicate_names() -> None:
    with pytest.raises(ValueError, match="must be unique"):
        CanonicalSchema(
            (
                Field("id", LONG, required=True),
                Field("id", LONG, required=True),
            )
        )


def _search(*, posting_encoding: str) -> ResolvedSearch:
    source = {"profile": "parquet", "uri": "file:///source"}
    return ResolvedSearch(
        source=relify.TableIdentifier("relify", ("test",), "source"),
        source_relation=source,
        index="source_embedding",
        centroids_relation={
            "profile": "parquet",
            "uri": "file:///centroids",
        },
        postings_relation={
            "profile": "parquet",
            "uri": "file:///postings",
        },
        query_vector=(0.0, 0.0),
        projection=("id",),
        predicate=None,
        limit=10,
        dimension=2,
        nlist=2,
        nprobe=1,
        source_key_fields=("id",),
        vector_field="embedding",
        posting_encoding=posting_encoding,
        needs_source=posting_encoding == "source",
        snapshot_id=1,
        family="ivf",
        index_schema_version=1,
        metric="l2_squared",
        ntotal=2,
    )
