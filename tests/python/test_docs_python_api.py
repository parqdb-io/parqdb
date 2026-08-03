from __future__ import annotations

from pathlib import Path

import pyarrow
import relify
from _support import WAIT, register_source, write_vectors
from relify.datafusion import col, functions


def test_python_api_build_search_and_compose_example(tmp_path: Path) -> None:
    source = tmp_path / "documents.parquet"
    write_vectors(
        source,
        [0, 1, 2, 3],
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
    )

    session = relify.connect(tmp_path / "relify-data")
    documents = register_source(session, source, "documents")
    documents.create_index(
        "documents_embedding",
        column="embedding",
        key=["id"],
        config=relify.IVF(nlist=2),
        wait_timeout=WAIT,
    )

    hits = (
        documents.search([0.0, 0.0], column="embedding")
        .where("id >= 1")
        .nprobes(2)
        .limit(2)
        .select(["id", "payload"])
    )
    document_stats = session.from_pydict(
        {
            "id": [1, 2, 3],
            "category": ["reference", "guide", "guide"],
            "popularity": [10, 20, 30],
        }
    )
    analysis = (
        session.to_dataframe(hits)
        .join(document_stats, on="id")
        .aggregate(
            "category",
            [
                functions.count(col("id")).alias("matches"),
                functions.avg(col("_distance")).alias("avg_distance"),
                functions.max(col("popularity")).alias("max_popularity"),
            ],
        )
        .sort("category")
    )
    result = pyarrow.Table.from_batches(analysis.collect(), schema=analysis.schema())
    assert result.to_pydict() == {
        "category": ["guide", "reference"],
        "matches": [1, 1],
        "avg_distance": [100.0, 1.0],
        "max_popularity": [20, 10],
    }
    optimized = analysis.optimized_logical_plan().display_indent()
    assert "Aggregate:" in optimized
    assert "Inner Join" in optimized
    assert "relify_squared_l2" in optimized

    query = documents.search([0.0, 0.0]).limit(2).select(["id"])
    df = session.to_dataframe(query)
    assert df.filter("_distance >= 0").collect()[0]["id"].to_pylist() == [0, 1]

    search_sql = session.to_sql(query)
    sql_result = session.sql(
        f"""
        SELECT *
        FROM ({search_sql}) AS vector_hits
        WHERE _distance >= 0
        """
    )
    assert sql_result.count() == 2

    exact = session.to_arrow(
        documents.search([0.0, 0.0], column="embedding").bypass_vector_index().limit(1)
    )
    assert exact["id"].to_pylist() == [0]


def test_python_api_catalog_recovery_example(tmp_path: Path) -> None:
    source = tmp_path / "documents.parquet"
    write_vectors(source, [0], [[0.0, 0.0]])
    session = relify.connect(tmp_path / "relify-data")
    documents = register_source(session, source, "documents")
    documents.create_index(
        "documents_embedding",
        column="embedding",
        key=["id"],
        config=relify.IVF(nlist=1),
        wait_timeout=WAIT,
    )

    entry = session.indexes.load("documents_embedding")
    session.indexes.drop("documents_embedding")
    session.indexes.register("recovered_index", entry.metadata_location)

    assert session.indexes.list() == ["recovered_index"]
