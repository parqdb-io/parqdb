from __future__ import annotations

from datetime import date
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import relify
from _support import WAIT, build_index, register_source, relation_files, vector_type


def invalid_source(case: str) -> tuple[pa.Table, str, list[str], int]:
    required_vector = vector_type()
    ids = pa.array([0, 1], type=pa.int64())
    vectors = pa.array([[0.0, 0.0], [1.0, 0.0]], type=required_vector)

    if case == "missing-vector":
        return (
            pa.table(
                {"id": ids},
                schema=pa.schema([pa.field("id", pa.int64(), nullable=False)]),
            ),
            "embedding",
            ["id"],
            1,
        )
    if case == "scalar-vector":
        return (
            pa.table(
                {"id": ids, "embedding": pa.array([0.0, 1.0], type=pa.float32())},
                schema=pa.schema(
                    [
                        pa.field("id", pa.int64(), nullable=False),
                        pa.field("embedding", pa.float32(), nullable=False),
                    ]
                ),
            ),
            "embedding",
            ["id"],
            1,
        )
    if case == "nullable-vector":
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("embedding", required_vector, nullable=True),
            ]
        )
        return (
            pa.Table.from_arrays(
                [ids, pa.array([[0.0, 0.0], None], type=required_vector)],
                schema=schema,
            ),
            "embedding",
            ["id"],
            1,
        )
    if case == "nullable-elements":
        nullable_vector = vector_type(nullable_elements=True)
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=False),
                pa.field("embedding", nullable_vector, nullable=False),
            ]
        )
        return (
            pa.Table.from_arrays(
                [
                    ids,
                    pa.array([[0.0, None], [1.0, 0.0]], type=nullable_vector),
                ],
                schema=schema,
            ),
            "embedding",
            ["id"],
            1,
        )
    if case == "inconsistent-dimension":
        return (
            pa.table(
                {
                    "id": ids,
                    "embedding": pa.array(
                        [[0.0, 0.0], [1.0, 0.0, 0.0]],
                        type=required_vector,
                    ),
                },
                schema=pa.schema(
                    [
                        pa.field("id", pa.int64(), nullable=False),
                        pa.field("embedding", required_vector, nullable=False),
                    ]
                ),
            ),
            "embedding",
            ["id"],
            1,
        )
    if case == "non-finite-vector":
        return (
            pa.table(
                {
                    "id": ids,
                    "embedding": pa.array(
                        [[0.0, float("nan")], [1.0, 0.0]],
                        type=required_vector,
                    ),
                },
                schema=pa.schema(
                    [
                        pa.field("id", pa.int64(), nullable=False),
                        pa.field("embedding", required_vector, nullable=False),
                    ]
                ),
            ),
            "embedding",
            ["id"],
            1,
        )
    if case == "nullable-key":
        schema = pa.schema(
            [
                pa.field("id", pa.int64(), nullable=True),
                pa.field("embedding", required_vector, nullable=False),
            ]
        )
        return (
            pa.Table.from_arrays(
                [pa.array([0, None], type=pa.int64()), vectors],
                schema=schema,
            ),
            "embedding",
            ["id"],
            1,
        )
    if case == "unsupported-key":
        return (
            pa.table(
                {
                    "id": pa.array([0.0, 1.0], type=pa.float32()),
                    "embedding": vectors,
                },
                schema=pa.schema(
                    [
                        pa.field("id", pa.float32(), nullable=False),
                        pa.field("embedding", required_vector, nullable=False),
                    ]
                ),
            ),
            "embedding",
            ["id"],
            1,
        )
    if case == "missing-key":
        return (
            pa.table(
                {"id": ids, "embedding": vectors},
                schema=pa.schema(
                    [
                        pa.field("id", pa.int64(), nullable=False),
                        pa.field("embedding", required_vector, nullable=False),
                    ]
                ),
            ),
            "embedding",
            ["missing"],
            1,
        )
    if case == "reserved-distance":
        return (
            pa.table(
                {
                    "id": ids,
                    "embedding": vectors,
                    "_distance": pa.array([0.0, 0.0], type=pa.float32()),
                },
                schema=pa.schema(
                    [
                        pa.field("id", pa.int64(), nullable=False),
                        pa.field("embedding", required_vector, nullable=False),
                        pa.field("_distance", pa.float32(), nullable=False),
                    ]
                ),
            ),
            "embedding",
            ["id"],
            1,
        )
    if case == "empty-source":
        return (
            pa.table(
                {
                    "id": pa.array([], type=pa.int64()),
                    "embedding": pa.array([], type=required_vector),
                },
                schema=pa.schema(
                    [
                        pa.field("id", pa.int64(), nullable=False),
                        pa.field("embedding", required_vector, nullable=False),
                    ]
                ),
            ),
            "embedding",
            ["id"],
            1,
        )
    if case == "nlist-exceeds-rows":
        return (
            pa.table(
                {"id": ids, "embedding": vectors},
                schema=pa.schema(
                    [
                        pa.field("id", pa.int64(), nullable=False),
                        pa.field("embedding", required_vector, nullable=False),
                    ]
                ),
            ),
            "embedding",
            ["id"],
            3,
        )
    raise AssertionError(f"unknown invalid source case: {case}")


@pytest.mark.parametrize(
    ("case", "error_type", "message"),
    [
        ("missing-vector", relify.InvalidSchemaError, "vector column not found"),
        ("scalar-vector", relify.InvalidSchemaError, "list<float>"),
        (
            "nullable-vector",
            relify.InvalidSchemaError,
            "must not contain nulls",
        ),
        (
            "nullable-elements",
            relify.InvalidSchemaError,
            "finite, non-null",
        ),
        (
            "inconsistent-dimension",
            relify.InvalidSchemaError,
            "same positive dimension",
        ),
        ("non-finite-vector", relify.InvalidSchemaError, "finite"),
        ("nullable-key", relify.BackendError, "must not contain nulls"),
        ("unsupported-key", relify.InvalidSchemaError, "unsupported source key type"),
        ("missing-key", relify.InvalidSchemaError, "key column not found"),
        ("reserved-distance", relify.InvalidSchemaError, "reserved column"),
        ("empty-source", relify.InvalidSchemaError, "at least one"),
        ("nlist-exceeds-rows", relify.InvalidArgumentError, "must not exceed ntotal"),
    ],
)
def test_invalid_sources_fail_without_publication(
    tmp_path: Path,
    case: str,
    error_type: type[Exception],
    message: str,
) -> None:
    source = tmp_path / f"{case}.parquet"
    table, column, key, nlist = invalid_source(case)
    pq.write_table(table, source)
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    with pytest.raises(error_type, match=message):
        build_index(vectors, column=column, key=key, nlist=nlist)

    status = vectors.index_status("vectors_embedding")
    assert status.state == "failed"
    assert status.current_snapshot_id is None
    assert status.error is not None
    assert session.indexes.list() == []


def test_nullable_source_schema_accepts_non_null_values(tmp_path: Path) -> None:
    source = tmp_path / "nullable-schema.parquet"
    nullable_vector = vector_type(nullable_elements=True)
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array([0, 1], type=pa.int32()),
                pa.array([[0.0, 0.0], [1.0, 0.0]], type=nullable_vector),
            ],
            schema=pa.schema(
                [
                    pa.field("id", pa.int32(), nullable=True),
                    pa.field("embedding", nullable_vector, nullable=True),
                ]
            ),
        ),
        source,
    )
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    build_index(vectors, nlist=1)

    hits = session.to_arrow(vectors.search([0.0, 0.0]).nprobes(1).limit(2))
    assert hits["id"].to_pylist() == [0, 1]
    snapshot = session.indexes.load("vectors_embedding").metadata["snapshots"][0]
    postings = pq.read_schema(
        relation_files(snapshot["index-relations"]["ivf_postings"])[0]
    )
    assert not postings.field("key_1").nullable
    assert postings.names == ["key_1"]


def test_duplicate_source_key_is_a_caller_contract(tmp_path: Path) -> None:
    source = tmp_path / "duplicates.parquet"
    required_vector = vector_type()
    pq.write_table(
        pa.table(
            {
                "document_id": pa.array(["same", "same"], type=pa.string()),
                "embedding": pa.array(
                    [[0.0, 0.0], [1.0, 0.0]],
                    type=required_vector,
                ),
            },
            schema=pa.schema(
                [
                    pa.field("document_id", pa.string(), nullable=False),
                    pa.field("embedding", required_vector, nullable=False),
                ]
            ),
        ),
        source,
    )
    session = relify.connect(tmp_path / "relify-data")
    documents = register_source(session, source, "documents")

    documents.create_index(
        "duplicate_key_index",
        column="embedding",
        key=["document_id"],
        config=relify.IVF(nlist=1),
        wait_timeout=WAIT,
    )

    status = documents.index_status("duplicate_key_index")
    assert status.state == "ready"
    snapshot = session.indexes.load("duplicate_key_index").metadata["snapshots"][0]
    assert snapshot["parameters"]["ntotal"] == "2"
    postings = pq.read_table(
        relation_files(snapshot["index-relations"]["ivf_postings"])
    )
    assert postings.num_rows == 2
    assert postings["key_1"].to_pylist() == ["same", "same"]


def test_supported_source_key_types_round_trip_through_postings(
    tmp_path: Path,
) -> None:
    source = tmp_path / "keys.parquet"
    keys = ["flag", "small", "large", "blob", "fixed", "name", "day"]
    schema = pa.schema(
        [
            pa.field("flag", pa.bool_(), nullable=False),
            pa.field("small", pa.int32(), nullable=False),
            pa.field("large", pa.int64(), nullable=False),
            pa.field("blob", pa.binary(), nullable=False),
            pa.field("fixed", pa.binary(4), nullable=False),
            pa.field("name", pa.string(), nullable=False),
            pa.field("day", pa.date32(), nullable=False),
            pa.field("embedding", vector_type(), nullable=False),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array([False, True], type=pa.bool_()),
                pa.array([-1, 1], type=pa.int32()),
                pa.array([-10, 10], type=pa.int64()),
                pa.array([b"a", b"b"], type=pa.binary()),
                pa.array([b"aaaa", b"bbbb"], type=pa.binary(4)),
                pa.array(["a", "b"], type=pa.string()),
                pa.array([date(2026, 1, 1), date(2026, 1, 2)], type=pa.date32()),
                pa.array([[0.0, 0.0], [1.0, 0.0]], type=vector_type()),
            ],
            schema=schema,
        ),
        source,
    )
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    build_index(vectors, key=keys)

    hits = session.to_arrow(vectors.search([0.5, 0.0]).nprobes(2).limit(2))
    assert set(hits["flag"].to_pylist()) == {False, True}
    assert hits["_distance"].to_pylist() == [0.25, 0.25]
    snapshot = session.indexes.load("vectors_embedding").metadata["snapshots"][0]
    posting_schema = pq.read_schema(
        relation_files(snapshot["index-relations"]["ivf_postings"])[0]
    )
    assert posting_schema == pa.schema(
        [
            pa.field("key_1", pa.bool_(), nullable=False),
            pa.field("key_2", pa.int32(), nullable=False),
            pa.field("key_3", pa.int64(), nullable=False),
            pa.field("key_4", pa.binary(), nullable=False),
            pa.field("key_5", pa.binary(4), nullable=False),
            pa.field("key_6", pa.string(), nullable=False),
            pa.field("key_7", pa.date32(), nullable=False),
        ]
    )
