"""Validate the portable IVF fixtures with an independent SQL engine."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb

FIXTURES = Path(__file__).parents[2] / "spec" / "fixtures" / "v1" / "valid"


def assert_query_result(
    actual: list[dict[str, object]],
    expected: list[dict[str, object]],
) -> None:
    distances = [row["_distance"] for row in actual]
    assert distances == sorted(distances)

    def canonical(row: dict[str, object]) -> str:
        return json.dumps(row, sort_keys=True)

    assert sorted(actual, key=canonical) == sorted(expected, key=canonical)


def quote_identifier(value: str) -> str:
    return '"' + value.replace('"', '""') + '"'


def execute_fixture_query(
    connection: duckdb.DuckDBPyConnection,
    directory: Path,
    case: dict[str, Any],
) -> list[dict[str, object]]:
    metadata = json.loads((directory / "metadata.json").read_text(encoding="utf-8"))
    snapshot = metadata["snapshots"][0]
    key_fields = snapshot["source-key-fields"]
    projection = ", ".join(
        f"s.{quote_identifier(field)}" for field in case["projection"]
    )
    key_join = " AND ".join(
        f"p.key_{position} = s.{quote_identifier(field)}"
        for position, field in enumerate(key_fields, start=1)
    )
    distance_vector = (
        "p.vector"
        if snapshot["parameters"]["store_vectors"] == "true"
        else f"s.{quote_identifier(snapshot['vector-field'])}"
    )
    filters = case["filter"] or {}
    filter_sql = " AND ".join(f"s.{quote_identifier(field)} = ?" for field in filters)
    where_clause = f"WHERE {filter_sql}" if filter_sql else ""
    sql = f"""
        WITH selected_clusters AS (
            SELECT cid
            FROM read_parquet(?)
            ORDER BY list_distance(centroid, ?::FLOAT[]), cid
            LIMIT ?
        ),
        candidates AS (
            SELECT
                {projection},
                pow(list_distance({distance_vector}, ?::FLOAT[]), 2) AS _distance
            FROM read_parquet(?, hive_partitioning = true) AS p
            INNER JOIN selected_clusters AS c USING (cid)
            INNER JOIN read_parquet(?) AS s ON {key_join}
            {where_clause}
        )
        SELECT *
        FROM candidates
        ORDER BY _distance
        LIMIT ?
    """
    parameters = [
        str(directory / "ivf_centroids.parquet"),
        case["query-vector"],
        case["nprobe"],
        case["query-vector"],
        str(directory / "ivf_postings" / "*" / "*.parquet"),
        str(directory / "source.parquet"),
        *filters.values(),
        case["k"],
    ]
    cursor = connection.execute(sql, parameters)
    columns = [description[0] for description in cursor.description]
    return [dict(zip(columns, row, strict=True)) for row in cursor.fetchall()]


def test_duckdb_reproduces_all_portable_query_results() -> None:
    connection = duckdb.connect()
    try:
        for directory in (FIXTURES, FIXTURES / "composite_no_vectors"):
            cases = json.loads((directory / "queries.json").read_text(encoding="utf-8"))
            for case in cases:
                assert_query_result(
                    execute_fixture_query(connection, directory, case),
                    case["expected"],
                )
    finally:
        connection.close()
