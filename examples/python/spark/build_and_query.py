"""Build and query an Iceberg index through experimental Spark support."""

from __future__ import annotations

import argparse
from collections.abc import Sequence
from typing import Any

import pyarrow
import relify


def run(
    session: Any,
    *,
    table: str,
    index: str,
    column: str,
    key: list[str],
    query_vector: list[float],
    nlist: int,
    nprobes: int,
    limit: int,
    projection: list[str],
    predicate: str | None,
) -> pyarrow.Table:
    """Build the index when absent, then collect one portable Arrow table."""
    documents = session.table(table)
    if index not in {info.name for info in documents.list_indexes()}:
        documents.create_index(
            index,
            column=column,
            key=key,
            config=relify.IVF(nlist=nlist),
        )
        documents.wait_for_index(index)

    query = (
        documents.search(query_vector, column=column, index=index)
        .nprobes(nprobes)
        .limit(limit)
        .select(projection)
    )
    if predicate is not None:
        query = query.where(predicate)
    return session.collect(query)


def main(argv: Sequence[str] | None = None) -> None:
    parser = _parser()
    args = parser.parse_args(argv)

    from pyiceberg.catalog import load_catalog
    from pyspark.sql import SparkSession  # pyright: ignore[reportMissingImports]

    spark = SparkSession.builder.appName("relify-spark-example").getOrCreate()
    try:
        iceberg = load_catalog(args.iceberg_catalog)
        session = relify.experimental.spark.connect(
            spark,
            index_catalog=args.index_catalog,
            iceberg_catalog=iceberg,
        )
        hits = run(
            session,
            table=args.table,
            index=args.index,
            column=args.column,
            key=args.key,
            query_vector=args.vector,
            nlist=args.nlist,
            nprobes=args.nprobes,
            limit=args.limit,
            projection=args.select,
            predicate=args.where,
        )
        print("Spark hits:", hits.to_pylist())
    finally:
        spark.stop()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Build and query a Relify IVF index using an existing Spark "
            "Iceberg catalog."
        )
    )
    parser.add_argument("--index-catalog", required=True)
    parser.add_argument("--iceberg-catalog", default="lakehouse")
    parser.add_argument("--table", required=True, help="namespace-qualified table")
    parser.add_argument("--index", default="documents_embedding")
    parser.add_argument("--column", default="embedding")
    parser.add_argument("--key", nargs="+", default=["document_id"])
    parser.add_argument("--vector", type=_vector, required=True)
    parser.add_argument("--nlist", type=int, default=1024)
    parser.add_argument("--nprobes", type=int, default=16)
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--select", nargs="+", default=["document_id"])
    parser.add_argument("--where")
    return parser


def _vector(value: str) -> list[float]:
    try:
        vector = [float(component) for component in value.split(",")]
    except ValueError as error:
        raise argparse.ArgumentTypeError(
            "vector must contain comma-separated numbers"
        ) from error
    if not vector:
        raise argparse.ArgumentTypeError("vector must not be empty")
    return vector


if __name__ == "__main__":
    main()
