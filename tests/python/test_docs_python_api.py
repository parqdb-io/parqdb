from __future__ import annotations

from pathlib import Path

import parqdb
import pyarrow
from _support import (
    WAIT,
    drop_table_index_entry,
    load_table_index,
    register_source,
    register_table_index,
    write_vectors,
)


def test_python_api_build_search_and_compose_example(tmp_path: Path) -> None:
    source = tmp_path / "documents.parquet"
    write_vectors(
        source,
        [0, 1, 2, 3],
        [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]],
    )

    session = parqdb.connect(tmp_path / "parqdb-data")
    documents = register_source(session, source, "documents")
    documents.create_index(
        "documents_embedding",
        column="embedding",
        key=["id"],
        config=parqdb.IVF(nlist=2),
        wait_timeout=WAIT,
    )

    hits = (
        documents.search([0.0, 0.0], column="embedding")
        .where("id >= 1")
        .nprobes(2)
        .limit(2)
        .select(["id", "payload"])
    )
    context = session.datafusion_context()
    document_stats = context.from_pydict(
        {
            "id": [1, 2, 3],
            "category": ["reference", "guide", "guide"],
            "popularity": [10, 20, 30],
        }
    )
    context.register_view("document_stats", document_stats)
    analysis = context.sql(
        f"""
        SELECT
            category,
            COUNT(id) AS matches,
            AVG(_distance) AS avg_distance,
            MAX(popularity) AS max_popularity
        FROM ({session.to_sql(hits)}) AS vector_hits
        JOIN document_stats USING (id)
        GROUP BY category
        ORDER BY category
        """
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
    assert "parqdb_squared_l2" in optimized

    query = documents.search([0.0, 0.0]).limit(2).select(["id"])
    query_result = session.collect(query)
    context.register_record_batches("query_result", [query_result.to_batches()])
    df = context.table("query_result")
    assert df.filter("_distance >= 0").collect()[0]["id"].to_pylist() == [0, 1]

    search_sql = session.to_sql(query)
    sql_result = session.sql(
        f"""
        SELECT *
        FROM ({search_sql}) AS vector_hits
        WHERE _distance >= 0
        """
    )
    assert sql_result.num_rows == 2

    exact = session.to_arrow(
        documents.search([0.0, 0.0], column="embedding").bypass_vector_index().limit(1)
    )
    assert exact["id"].to_pylist() == [0]


def test_python_api_catalog_recovery_example(tmp_path: Path) -> None:
    source = tmp_path / "documents.parquet"
    write_vectors(source, [0], [[0.0, 0.0]])
    session = parqdb.connect(tmp_path / "parqdb-data")
    documents = register_source(session, source, "documents")
    documents.create_index(
        "documents_embedding",
        column="embedding",
        key=["id"],
        config=parqdb.IVF(nlist=1),
        wait_timeout=WAIT,
    )

    entry = load_table_index(session, documents, "documents_embedding")
    drop_table_index_entry(session, documents, "documents_embedding")
    register_table_index(
        documents,
        "recovered_index",
        entry.metadata_location,
    )

    assert session._indexes.list(namespace=documents.identifier.index_namespace) == [
        "recovered_index"
    ]
