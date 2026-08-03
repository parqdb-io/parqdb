from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import relify
from _support import register_source, vector_type, write_vectors


def test_exact_search_does_not_require_an_index(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(
        source,
        [0, 1, 2],
        [[0.0, 0.0], [4.0, 0.0], [10.0, 0.0]],
    )
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    result = session.to_arrow(
        vectors.search([9.0, 0.0]).bypass_vector_index().limit(2).select(["id"])
    )

    assert result["id"].to_pylist() == [2, 1]
    assert result["_distance"].to_pylist() == pytest.approx([1.0, 25.0])
    assert result.schema.field("_distance") == pa.field(
        "_distance",
        pa.float32(),
        nullable=False,
    )
    assert session.indexes.list() == []


def test_exact_search_applies_prefilter_before_top_k(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(
        source,
        [0, 1, 2],
        [[0.0, 0.0], [4.0, 0.0], [10.0, 0.0]],
    )
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    result = session.to_arrow(
        vectors.search([9.0, 0.0])
        .bypass_vector_index()
        .where("id < 2")
        .limit(1)
        .select(["id"])
    )

    assert result["id"].to_pylist() == [1]


@pytest.mark.parametrize(
    ("query", "message"),
    [
        (
            lambda table: table.search(
                [0.0, 0.0],
                column="embedding",
                index="unused",
            ).bypass_vector_index(),
            "index cannot be set",
        ),
        (
            lambda table: (
                table.search([0.0, 0.0], column="embedding")
                .nprobes(1)
                .bypass_vector_index()
            ),
            "nprobes cannot be set",
        ),
    ],
)
def test_exact_search_rejects_index_only_options(
    tmp_path: Path,
    query: Callable[[relify.SourceTable], relify.VectorQuery],
    message: str,
) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0], [[0.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    with pytest.raises(relify.InvalidArgumentError, match=message):
        session.to_arrow(query(vectors))


def test_exact_search_requires_column_for_multiple_vector_columns(
    tmp_path: Path,
) -> None:
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
                pa.array([0], type=pa.int64()),
                pa.array([[0.0, 0.0]], type=vector_type()),
                pa.array([[1.0, 0.0]], type=vector_type()),
            ],
            schema=schema,
        ),
        source,
    )
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    with pytest.raises(relify.InvalidArgumentError, match="multiple vector columns"):
        session.to_arrow(vectors.search([0.0, 0.0]).bypass_vector_index())

    result = session.to_arrow(
        vectors.search([1.0, 0.0], column="embedding_b")
        .bypass_vector_index()
        .select(["id"])
    )
    assert result["id"].to_pylist() == [0]


def test_exact_search_explain_plan_contains_no_index_join(tmp_path: Path) -> None:
    source = tmp_path / "vectors.parquet"
    write_vectors(source, [0, 1], [[0.0, 0.0], [1.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    vectors = register_source(session, source)

    plan = session.explain(vectors.search([0.0, 0.0]).bypass_vector_index())

    assert "relify_squared_l2" in plan
    assert "IvfTopKExec" in plan
    assert "HashJoinExec" not in plan
    assert "ivf_postings" not in plan
