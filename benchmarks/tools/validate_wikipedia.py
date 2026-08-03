from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds

from benchmarks.tools.datasets import (
    inspect_matrix,
    inspect_parquet_source,
    sha256_file,
)

DATASET = "maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2"
REVISION = "ab22410c0589e39371431e3dd293e4f0fa0c4b26"
SPLIT = "train"
ROWS = 35_167_920
DIMENSION = 384
QUERY_START = 0
QUERIES = 100
GROUND_TRUTH_K = 100_000
ID_COLUMN = "id"
VECTOR_COLUMN = "emb"


def validate(args: argparse.Namespace) -> dict[str, object]:
    source = inspect_parquet_source(
        args.source_parquet,
        id_column=ID_COLUMN,
        vector_column=VECTOR_COLUMN,
    )
    if source.rows != ROWS or source.dimension != DIMENSION:
        raise ValueError("Wikipedia source shape does not match the pinned train split")

    seen = np.zeros(ROWS, dtype=np.bool_)
    scanned_rows = 0
    dataset = ds.dataset(source.path, format="parquet", exclude_invalid_files=True)
    for batch in dataset.to_batches(columns=[ID_COLUMN], batch_size=262_144):
        column = batch.column(0)
        if column.null_count:
            raise ValueError("Wikipedia source IDs must not contain NULLs")
        values = np.ascontiguousarray(
            column.to_numpy(zero_copy_only=False),
            dtype=np.int64,
        )
        if np.any(values < 0) or np.any(values >= ROWS):
            raise ValueError("Wikipedia source IDs must be in 0..rows-1")
        if np.any(seen[values]):
            raise ValueError("Wikipedia source IDs must be unique")
        seen[values] = True
        scanned_rows += len(values)
    if scanned_rows != ROWS or not np.all(seen):
        raise ValueError("Wikipedia source row count changed during validation")

    ground_truth = inspect_matrix(args.ground_truth, dtype=np.dtype("<u4"))
    if ground_truth.rows != QUERIES or ground_truth.dimension != GROUND_TRUTH_K:
        raise ValueError(f"ground truth must have shape ({QUERIES}, {GROUND_TRUTH_K})")
    neighbors = ground_truth.memmap()
    if int(neighbors.max()) >= ROWS:
        raise ValueError("ground truth contains an ID outside the source table")
    query_ids = np.arange(QUERY_START, QUERY_START + QUERIES, dtype=np.uint32)
    if not np.array_equal(neighbors[:, 0], query_ids):
        raise ValueError("ground-truth rank 1 must be the source-resident query row")

    return {
        "schema_version": 1,
        "dataset": DATASET,
        "revision": REVISION,
        "split": SPLIT,
        "source": {
            "path": str(source.path),
            "rows": source.rows,
            "dimension": source.dimension,
            "bytes": source.bytes,
            "id_column": source.id_column,
            "vector_column": source.vector_column,
        },
        "queries": {
            "source_id_start": QUERY_START,
            "count": QUERIES,
        },
        "ground_truth": {
            "path": str(ground_truth.path),
            "queries": ground_truth.rows,
            "k": ground_truth.dimension,
            "sha256": sha256_file(ground_truth.path),
        },
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        description="Validate the published Wikipedia benchmark inputs."
    )
    command.add_argument("--source-parquet", type=Path, required=True)
    command.add_argument("--ground-truth", type=Path, required=True)
    return command


def main() -> None:
    print(json.dumps(validate(parser().parse_args()), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
