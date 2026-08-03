"""Query an Iceberg index through experimental StarRocks support."""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from typing import Any

import pyarrow as pa
import relify


def run(
    session: Any,
    *,
    table: str,
    index: str | None,
    column: str,
    query_vector: list[float],
    nprobes: int,
    limit: int,
    projection: list[str],
    predicate: str | None,
) -> pa.Table:
    """Execute one query and return its Arrow result."""
    documents = session.table(table)
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

    import adbc_driver_flightsql.dbapi as flight_sql  # pyright: ignore[reportMissingImports]
    from adbc_driver_manager import (  # pyright: ignore[reportMissingImports]
        DatabaseOptions,
    )
    from pyiceberg.catalog import load_catalog

    connection = flight_sql.connect(
        uri=args.flight_uri,
        db_kwargs={
            DatabaseOptions.USERNAME.value: args.username,
            DatabaseOptions.PASSWORD.value: os.environ.get(
                args.password_environment,
                "",
            ),
        },
    )
    try:
        iceberg = load_catalog(args.iceberg_catalog)
        session = relify.experimental.starrocks.connect(
            connection,
            index_catalog=args.index_catalog,
            iceberg_catalog=iceberg,
            catalog_name=args.host_catalog,
        )
        result = run(
            session,
            table=args.table,
            index=args.index,
            column=args.column,
            query_vector=args.vector,
            nprobes=args.nprobes,
            limit=args.limit,
            projection=args.select,
            predicate=args.where,
        )
        print("StarRocks hits:", result.to_pylist())
    finally:
        connection.close()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Query a Relify Iceberg index through an existing StarRocks "
            "Flight SQL endpoint."
        )
    )
    parser.add_argument("--flight-uri", required=True)
    parser.add_argument("--username", default="root")
    parser.add_argument(
        "--password-environment",
        default="STARROCKS_PASSWORD",
        help="environment variable containing the Flight SQL password",
    )
    parser.add_argument("--index-catalog", required=True)
    parser.add_argument("--iceberg-catalog", default="lakehouse")
    parser.add_argument(
        "--host-catalog",
        help=(
            "matching Iceberg catalog name registered in StarRocks; "
            "defaults to the PyIceberg catalog name"
        ),
    )
    parser.add_argument("--table", required=True, help="namespace-qualified table")
    parser.add_argument("--index")
    parser.add_argument("--column", default="embedding")
    parser.add_argument("--vector", type=_vector, required=True)
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
