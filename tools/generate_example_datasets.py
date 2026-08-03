from __future__ import annotations

import argparse
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

VECTOR_TYPE = pa.list_(pa.field("element", pa.float32(), nullable=False))
DOCUMENT_SCHEMA = pa.schema(
    [
        pa.field("document_id", pa.int64(), nullable=False),
        pa.field("title", pa.string(), nullable=False),
        pa.field("tenant_id", pa.int64(), nullable=False),
        pa.field("status", pa.string(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("embedding", VECTOR_TYPE, nullable=False),
    ]
)
DOCUMENTS = [
    {
        "document_id": 1,
        "title": "Vector indexing",
        "tenant_id": 42,
        "status": "published",
        "category": "search",
        "embedding": [0.0, 0.0],
    },
    {
        "document_id": 2,
        "title": "Lakehouse tables",
        "tenant_id": 42,
        "status": "published",
        "category": "storage",
        "embedding": [1.0, 0.0],
    },
    {
        "document_id": 3,
        "title": "Query engines",
        "tenant_id": 42,
        "status": "draft",
        "category": "compute",
        "embedding": [2.0, 0.0],
    },
    {
        "document_id": 4,
        "title": "Approximate search",
        "tenant_id": 7,
        "status": "published",
        "category": "search",
        "embedding": [8.0, 0.0],
    },
    {
        "document_id": 5,
        "title": "Distributed execution",
        "tenant_id": 42,
        "status": "published",
        "category": "compute",
        "embedding": [9.0, 0.0],
    },
    {
        "document_id": 6,
        "title": "Catalog recovery",
        "tenant_id": 42,
        "status": "archived",
        "category": "storage",
        "embedding": [10.0, 0.0],
    },
]

DOCUMENT_STATS_SCHEMA = pa.schema(
    [
        pa.field("document_id", pa.int64(), nullable=False),
        pa.field("category", pa.string(), nullable=False),
        pa.field("popularity", pa.int64(), nullable=False),
    ]
)
DOCUMENT_STATS = [
    {"document_id": 1, "category": "search", "popularity": 91},
    {"document_id": 2, "category": "storage", "popularity": 74},
    {"document_id": 3, "category": "compute", "popularity": 68},
    {"document_id": 4, "category": "search", "popularity": 86},
    {"document_id": 5, "category": "compute", "popularity": 95},
    {"document_id": 6, "category": "storage", "popularity": 57},
]


def generate(output: Path) -> None:
    output.mkdir(parents=True, exist_ok=True)
    _write(output / "documents.parquet", DOCUMENT_SCHEMA, DOCUMENTS)
    _write(
        output / "document_stats.parquet",
        DOCUMENT_STATS_SCHEMA,
        DOCUMENT_STATS,
    )


def _write(path: Path, schema: pa.Schema, rows: list[dict[str, object]]) -> None:
    table = pa.Table.from_pylist(rows, schema=schema)
    pq.write_table(
        table,
        path,
        compression="zstd",
        use_dictionary=False,
        write_statistics=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(__file__).parents[1] / "python" / "relify" / "datasets",
    )
    args = parser.parse_args()
    generate(args.output)


if __name__ == "__main__":
    main()
