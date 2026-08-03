from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pyarrow as pa
import relify
from pyiceberg.catalog import load_catalog
from pyiceberg.schema import Schema
from pyiceberg.types import (
    FloatType,
    IntegerType,
    ListType,
    LongType,
    NestedField,
    StringType,
)


def test_datafusion_queries_spark_style_iceberg_index(tmp_path: Path) -> None:
    iceberg = load_catalog(
        "lakehouse",
        type="sql",
        uri=f"sqlite:///{tmp_path / 'iceberg.sqlite'}",
        warehouse=(tmp_path / "iceberg").as_uri(),
    )
    iceberg.create_namespace(("analytics",))
    iceberg.create_namespace(("relify",))

    source = _create_table(
        iceberg,
        ("analytics", "documents"),
        Schema(
            NestedField(1, "document_id", LongType(), required=True),
            NestedField(
                2,
                "embedding",
                ListType(3, FloatType(), element_required=True),
                required=True,
            ),
            NestedField(4, "tenant_id", IntegerType(), required=True),
            NestedField(5, "title", StringType(), required=True),
        ),
        pa.schema(
            [
                pa.field("document_id", pa.int64(), nullable=False),
                _vector_field("embedding"),
                pa.field("tenant_id", pa.int32(), nullable=False),
                pa.field("title", pa.string(), nullable=False),
            ]
        ),
        [
            pa.array([1, 2, 3], type=pa.int64()),
            _vectors([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]),
            pa.array([42, 42, 7], type=pa.int32()),
            pa.array(["a", "b", "c"], type=pa.string()),
        ],
    )
    centroids = _create_table(
        iceberg,
        ("relify", "documents_embedding_centroids"),
        Schema(
            NestedField(1, "cid", IntegerType(), required=True),
            NestedField(
                2,
                "centroid",
                ListType(3, FloatType(), element_required=True),
                required=True,
            ),
        ),
        pa.schema(
            [
                pa.field("cid", pa.int32(), nullable=False),
                _vector_field("centroid"),
            ]
        ),
        [
            pa.array([0, 1], type=pa.int32()),
            _vectors([[0.5, 0.0], [10.0, 0.0]]),
        ],
    )
    postings_arrow_schema = pa.schema(
        [
            pa.field("cid", pa.int32(), nullable=False),
            pa.field("key_1", pa.int64(), nullable=False),
            _vector_field("vector"),
        ]
    )
    postings = _create_table(
        iceberg,
        ("relify", "documents_embedding_postings"),
        Schema(
            NestedField(1, "cid", IntegerType(), required=True),
            NestedField(2, "key_1", LongType(), required=True),
            NestedField(
                3,
                "vector",
                ListType(4, FloatType(), element_required=True),
                required=True,
            ),
        ),
        postings_arrow_schema,
        [
            pa.array([0, 0, 1], type=pa.int32()),
            pa.array([1, 2, 3], type=pa.int64()),
            _vectors([[0.0, 0.0], [1.0, 0.0], [10.0, 0.0]]),
        ],
    )

    catalog_path = tmp_path / "relify.sqlite"
    metadata_root = (tmp_path / "relify-metadata").as_uri()
    repository = relify._native._NativeIndexRepository(
        catalog_path,
        metadata_root,
        None,
    )
    source_reference = _relation(source, "lakehouse", ("analytics",), "documents")
    repository.publish_initial(
        index_name="documents_embedding",
        source_json=source_reference,
        vector_field="embedding",
        source_key_fields=["document_id"],
        builder="fixture",
        parameters={
            "dimension": "2",
            "nlist": "2",
            "ntotal": "3",
            "store_vectors": "true",
        },
        index_relations={
            "ivf_centroids": _relation(
                centroids,
                "lakehouse",
                ("relify",),
                "documents_embedding_centroids",
            ),
            "ivf_postings": _relation(
                postings,
                "lakehouse",
                ("relify",),
                "documents_embedding_postings",
            ),
        },
    )
    # The Relify snapshot must keep reading the published Iceberg snapshot even
    # after the table itself advances.
    postings.append(
        pa.Table.from_batches(
            [
                pa.RecordBatch.from_arrays(
                    [
                        pa.array([0], type=pa.int32()),
                        pa.array([2], type=pa.int64()),
                        _vectors([[0.0, 0.0]]),
                    ],
                    schema=postings_arrow_schema,
                )
            ],
            schema=postings_arrow_schema,
        )
    )

    session = relify.connect(
        catalog=f"sqlite://{catalog_path}",
        index_root=metadata_root,
        iceberg=iceberg,
    )
    cached = session.cache_index("documents_embedding")
    assert cached.relation_count == 2
    assert cached.resident_bytes > 0
    assert _search_documents(session) == {
        "document_id": [1, 2],
        "title": ["a", "b"],
        "_distance": [0.0, 1.0],
    }
    reopened = relify.connect(
        catalog=f"sqlite://{catalog_path}",
        index_root=metadata_root,
        iceberg=iceberg,
    )
    assert _search_documents(reopened) == {
        "document_id": [1, 2],
        "title": ["a", "b"],
        "_distance": [0.0, 1.0],
    }


def _create_table(
    catalog: Any,
    identifier: tuple[str, ...],
    schema: Schema,
    arrow_schema: pa.Schema,
    arrays: list[pa.Array],
) -> Any:
    table = catalog.create_table(identifier, schema=schema)
    batch = pa.RecordBatch.from_arrays(arrays, schema=arrow_schema)
    table.append(pa.Table.from_batches([batch], schema=arrow_schema))
    return table


def _search_documents(session: relify.Session) -> dict[str, list[object]]:
    documents = session.table("lakehouse.analytics.documents")
    query = (
        documents.search([0.0, 0.0], column="embedding")
        .where("tenant_id = 42")
        .nprobes(1)
        .limit(10)
        .select(["document_id", "title"])
    )
    return session.to_arrow(query).to_pydict()


def _vector_field(name: str) -> pa.Field:
    return pa.field(
        name,
        pa.list_(pa.field("element", pa.float32(), nullable=False)),
        nullable=False,
    )


def _vectors(values: list[list[float]]) -> pa.Array:
    return pa.array(
        values,
        type=pa.list_(pa.field("element", pa.float32(), nullable=False)),
    )


def _relation(
    table: Any,
    catalog: str,
    namespace: tuple[str, ...],
    name: str,
) -> str:
    snapshot = table.current_snapshot()
    assert snapshot is not None
    return json.dumps(
        {
            "profile": "iceberg",
            "catalog": catalog,
            "namespace": list(namespace),
            "name": name,
            "table-uuid": str(table.metadata.table_uuid).lower(),
            "snapshot-id": int(snapshot.snapshot_id),
        },
        separators=(",", ":"),
    )
