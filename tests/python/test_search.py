from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import relify
from _support import build_index, register_source, vector_type, write_vectors

SOURCE = relify.TableIdentifier("datafusion", ("public",), "documents")


@pytest.mark.parametrize(
    "query",
    [
        [0.0, 1.0],
        (0.0, 1.0),
        np.asarray([0.0, 1.0], dtype=np.float32),
    ],
)
def test_table_search_accepts_one_dimensional_vector_inputs(
    query: object,
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    _, documents = indexed_documents
    assert documents.search(cast(Any, query)).query == (0.0, 1.0)


def test_table_search_uses_array_like_tolist(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    class ArrayLike:
        ndim = 1

        def tolist(self) -> list[float]:
            return [0.0, 1.0]

        def __iter__(self) -> Any:
            raise AssertionError("search should use tolist")

    _, documents = indexed_documents
    assert documents.search(cast(Any, ArrayLike())).query == (0.0, 1.0)


def test_table_search_rejects_batch_query_arrays(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    _, documents = indexed_documents
    query = np.asarray([[0.0, 1.0]], dtype=np.float32)

    with pytest.raises(ValueError, match="one-dimensional"):
        documents.search(query)


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "10", None])
def test_vector_query_limit_requires_a_positive_integer(value: object) -> None:
    query = relify.VectorQuery(source=SOURCE, query=(1.0, 2.0))
    with pytest.raises(ValueError, match="limit must be a positive integer"):
        query.limit(cast(Any, value))


@pytest.mark.parametrize("value", [True, 0, -1, 1.5, "10", None])
def test_vector_query_nprobes_requires_a_positive_integer(value: object) -> None:
    query = relify.VectorQuery(source=SOURCE, query=(1.0, 2.0))
    with pytest.raises(ValueError, match="nprobes must be a positive integer"):
        query.nprobes(cast(Any, value))


def test_vector_query_builder_is_immutable() -> None:
    query = relify.VectorQuery(source=SOURCE, query=(1.0, 2.0))

    configured = (
        query.select(["id"]).where("id > 10").limit(5).nprobes(3).bypass_vector_index()
    )

    assert query.projection is None
    assert query.result_limit == 10
    assert query.probe_count is None
    assert query.predicate is None
    assert query.bypass_index is False
    assert configured.projection == ("id",)
    assert configured.result_limit == 5
    assert configured.probe_count == 3
    assert configured.predicate == "id > 10"
    assert configured.bypass_index is True


def test_vector_query_is_portable_across_sessions_with_the_same_table(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents
    query = documents.search([0.0, 0.0]).limit(1).select(["id"])

    assert query.source == relify.TableIdentifier(
        "datafusion",
        ("public",),
        "documents",
    )
    assert not hasattr(query, "table")

    reopened = relify.connect(session.root)
    assert reopened.to_arrow(query)["id"].to_pylist() == [0]


@pytest.mark.parametrize("value", [None, 1, True, [], object()])
def test_vector_query_where_requires_a_string(value: object) -> None:
    query = relify.VectorQuery(source=SOURCE, query=(1.0, 2.0))

    with pytest.raises(TypeError, match="predicate must be a string"):
        query.where(cast(Any, value))


@pytest.mark.parametrize("value", ["", " ", "\t\n"])
def test_vector_query_where_rejects_empty_predicates(value: str) -> None:
    query = relify.VectorQuery(source=SOURCE, query=(1.0, 2.0))

    with pytest.raises(ValueError, match="predicate must not be empty"):
        query.where(value)


@pytest.mark.parametrize(
    ("query", "nprobe", "projection", "error_type", "message"),
    [
        ([0.0], None, None, relify.InvalidArgumentError, "exactly 2"),
        ([float("nan"), 0.0], None, None, relify.InvalidArgumentError, "finite"),
        ([0.0, 0.0], 3, None, relify.InvalidArgumentError, "nprobe"),
        ([0.0, 0.0], None, [], relify.InvalidArgumentError, "projection"),
        ([0.0, 0.0], None, ["id", "id"], relify.InvalidArgumentError, "projection"),
        (
            [0.0, 0.0],
            None,
            ["missing"],
            relify.InvalidArgumentError,
            "column not found",
        ),
    ],
)
def test_query_validation_reaches_native_execution(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
    query: list[float],
    nprobe: int | None,
    projection: list[str] | None,
    error_type: type[Exception],
    message: str,
) -> None:
    session, documents = indexed_documents
    request = documents.search(query)
    if nprobe is not None:
        request = request.nprobes(nprobe)
    if projection is not None:
        request = request.select(projection)

    with pytest.raises(error_type, match=message):
        session.to_arrow(request)


def test_index_selection_reports_missing_and_mismatched_indexes(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents

    with pytest.raises(relify.IndexNotFoundError):
        session.to_arrow(documents.search([0.0, 0.0], index="missing"))
    with pytest.raises(relify.IndexNotFoundError):
        session.to_arrow(documents.search([0.0, 0.0], column="missing"))
    with pytest.raises(relify.IndexNotFoundError):
        session.to_arrow(
            documents.search(
                [0.0, 0.0],
                index="vectors_embedding",
                column="missing",
            )
        )


def test_multiple_indexes_require_disambiguation(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("embedding_a", vector_type(), nullable=False),
            pa.field("embedding_b", vector_type(), nullable=False),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array([0, 1], type=pa.int64()),
                pa.array([[0.0, 0.0], [10.0, 0.0]], type=vector_type()),
                pa.array([[100.0, 0.0], [110.0, 0.0]], type=vector_type()),
            ],
            schema=schema,
        ),
        source,
    )
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    build_index(vectors, "index_a", column="embedding_a")
    build_index(vectors, "index_b", column="embedding_b")

    assert [index.name for index in vectors.list_indexes()] == ["index_a", "index_b"]
    with pytest.raises(relify.AmbiguousIndexError, match="index_a, index_b"):
        session.to_arrow(vectors.search([0.0, 0.0]))

    from_column = session.to_arrow(
        vectors.search([100.0, 0.0], column="embedding_b").limit(1)
    )
    assert from_column["id"].to_pylist() == [0]
    from_name = session.to_arrow(vectors.search([10.0, 0.0], index="index_a").limit(1))
    assert from_name["id"].to_pylist() == [1]


def test_explicit_index_remains_bound_to_its_source(tmp_path: Path) -> None:
    first = tmp_path / "first.parquet"
    second = tmp_path / "second.parquet"
    write_vectors(first, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    write_vectors(second, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    first_table = register_source(session, first, "first")
    build_index(first_table)

    with pytest.raises(relify.IndexNotFoundError):
        session.to_arrow(
            register_source(session, second, "second").search(
                [0.0, 0.0],
                index="vectors_embedding",
            )
        )


def test_where_filters_source_rows_before_top_k(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents

    result = session.to_arrow(
        documents.search([0.0, 0.0]).where("id >= 2").limit(1).select(["id"])
    )

    assert result["id"].to_pylist() == [2]


def test_where_supports_source_string_columns(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents

    result = session.to_arrow(
        documents.search([0.0, 0.0])
        .where("payload = 'row-1'")
        .select(["id", "payload"])
    )

    assert result["id"].to_pylist() == [1]
    assert result["payload"].to_pylist() == ["row-1"]


def test_where_excludes_null_predicate_results(tmp_path: Path) -> None:
    source = tmp_path / "nullable-filter.parquet"
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("category", pa.string(), nullable=True),
            pa.field("embedding", vector_type(), nullable=False),
        ]
    )
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array([0, 1, 2], type=pa.int64()),
                pa.array(["keep", None, "keep"], type=pa.string()),
                pa.array(
                    [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
                    type=vector_type(),
                ),
            ],
            schema=schema,
        ),
        source,
    )
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)
    build_index(vectors, nlist=1)

    result = session.to_arrow(
        vectors.search([0.0, 0.0]).where("category = 'keep'").select(["id"])
    )

    assert result["id"].to_pylist() == [0, 2]


def test_where_rejects_invalid_backend_expressions(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents

    with pytest.raises(relify.InvalidArgumentError, match="invalid filter"):
        session.to_arrow(documents.search([0.0, 0.0]).where("missing = 1"))


def test_arrow_results_can_be_registered_in_the_native_context(
    indexed_documents: tuple[relify.Session, relify.SourceTable],
) -> None:
    session, documents = indexed_documents
    query = documents.search([0.0, 0.0]).limit(1).select(["id"])

    result = session.to_arrow(query)
    assert result.column_names == ["id", "_distance"]
    assert result["id"].to_pylist() == [0]

    context = session.datafusion_context()
    context.register_record_batches("hits", [result.to_batches()])
    sql_result = context.sql("SELECT * FROM hits").collect()
    assert [value for batch in sql_result for value in batch["id"].to_pylist()] == [0]
    context.deregister_table("hits")
