"""Prepare the BigANN SIFT1B files for the standard ParqDB benchmark runners."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import uuid
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import get_context
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.tools.datasets import inspect_float_matrix, inspect_parquet_source

DATASET_NAME = "sift1b-bigann"
DEFAULT_ROWS_PER_FILE = 2_097_152
DEFAULT_ROW_GROUP_ROWS = 65_536
DEFAULT_COMPRESSION_LEVEL = 3
DEFAULT_WORKERS = 32
_INT32 = np.dtype("<i4")
_UINT32 = np.dtype("<u4")
_FLOAT32 = np.dtype("<f4")


@dataclass(frozen=True)
class SourceFileTask:
    base: Path
    destination: Path
    file_start: int
    file_stop: int
    dimension: int
    row_group_rows: int
    compression_level: int


def _inspect_vecs(path: Path, *, value_dtype: np.dtype[np.generic]) -> tuple[int, int]:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype=_INT32, count=1)
    if len(header) != 1 or int(header[0]) <= 0:
        raise ValueError(f"invalid vecs dimension header: {path}")
    dimension = int(header[0])
    row_bytes = _INT32.itemsize + dimension * value_dtype.itemsize
    size = path.stat().st_size
    if size == 0 or size % row_bytes:
        raise ValueError(f"vecs file size does not match its dimension: {path}")
    return size // row_bytes, dimension


def _read_vecs(
    stream: Any,
    *,
    rows: int,
    dimension: int,
    value_dtype: np.dtype[np.generic],
) -> np.ndarray:
    row_bytes = _INT32.itemsize + dimension * value_dtype.itemsize
    encoded = np.fromfile(stream, dtype=np.uint8, count=rows * row_bytes)
    if len(encoded) != rows * row_bytes:
        raise ValueError("vecs file ends before its declared row count")
    records = encoded.reshape(rows, row_bytes)
    headers = records[:, : _INT32.itemsize].view(_INT32).reshape(-1)
    if not np.all(headers == dimension):
        raise ValueError("vecs file contains an inconsistent dimension header")
    values = records[:, _INT32.itemsize :].view(value_dtype).reshape(rows, dimension)
    return np.ascontiguousarray(values)


def _embedding_table(start: int, vectors: np.ndarray, *, dimension: int) -> pa.Table:
    values = pa.array(vectors.reshape(-1), type=pa.float32())
    embeddings = pa.FixedSizeListArray.from_arrays(values, dimension)
    vector_type = pa.list_(pa.field("element", pa.float32(), nullable=False), dimension)
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("embedding", vector_type, nullable=False),
        ]
    )
    return pa.Table.from_arrays(
        [
            pa.array(np.arange(start, start + len(vectors), dtype=np.int64)),
            embeddings,
        ],
        schema=schema,
    )


def _write_source_file(task: SourceFileTask) -> int:
    vector_type = pa.list_(
        pa.field("element", pa.float32(), nullable=False), task.dimension
    )
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("embedding", vector_type, nullable=False),
        ]
    )
    row_bytes = _INT32.itemsize + task.dimension * np.dtype(np.uint8).itemsize
    with task.base.open("rb") as source:
        source.seek(task.file_start * row_bytes)
        with pq.ParquetWriter(
            task.destination,
            schema,
            compression="zstd",
            compression_level=task.compression_level,
            use_dictionary=False,
            use_byte_stream_split=["embedding.list.element"],
        ) as writer:
            for start in range(task.file_start, task.file_stop, task.row_group_rows):
                stop = min(start + task.row_group_rows, task.file_stop)
                values = _read_vecs(
                    source,
                    rows=stop - start,
                    dimension=task.dimension,
                    value_dtype=np.dtype(np.uint8),
                ).astype(np.float32, copy=False)
                writer.write_table(
                    _embedding_table(start, values, dimension=task.dimension),
                    row_group_size=stop - start,
                )
    return task.destination.stat().st_size


def _write_source(
    base: Path,
    destination: Path,
    *,
    rows: int,
    dimension: int,
    rows_per_file: int,
    row_group_rows: int,
    compression_level: int,
    workers: int,
) -> tuple[int, int]:
    destination.mkdir(parents=True)
    tasks = [
        SourceFileTask(
            base=base,
            destination=destination / f"part-{file_number:05d}.parquet",
            file_start=file_start,
            file_stop=min(file_start + rows_per_file, rows),
            dimension=dimension,
            row_group_rows=row_group_rows,
            compression_level=compression_level,
        )
        for file_number, file_start in enumerate(range(0, rows, rows_per_file))
    ]
    parquet_bytes = 0
    with ProcessPoolExecutor(
        max_workers=workers,
        mp_context=get_context("spawn"),
    ) as executor:
        for file_count, (task, file_bytes) in enumerate(
            zip(tasks, executor.map(_write_source_file, tasks), strict=True),
            start=1,
        ):
            parquet_bytes += file_bytes
            print(
                f"\rSIFT1B Parquet {100.0 * task.file_stop / rows:5.1f}% "
                f"({file_count} files)",
                end="",
                file=sys.stderr,
                flush=True,
            )
    print(file=sys.stderr)
    return file_count, parquet_bytes


def _write_queries(source: Path, destination: Path, *, dimension: int) -> int:
    rows, source_dimension = _inspect_vecs(source, value_dtype=np.dtype(np.uint8))
    if source_dimension != dimension:
        raise ValueError("base and query vector dimensions differ")
    with source.open("rb") as stream:
        values = _read_vecs(
            stream,
            rows=rows,
            dimension=dimension,
            value_dtype=np.dtype(np.uint8),
        ).astype(_FLOAT32, copy=False)
    with destination.open("wb") as stream:
        np.asarray([rows, dimension], dtype=_UINT32).tofile(stream)
        values.tofile(stream)
    return rows


def _write_ground_truth(
    source: Path,
    destination: Path,
    *,
    queries: int,
    source_rows: int,
) -> int:
    rows, k = _inspect_vecs(source, value_dtype=_INT32)
    if rows != queries:
        raise ValueError("ground-truth and query row counts differ")
    with source.open("rb") as stream:
        values = _read_vecs(
            stream,
            rows=rows,
            dimension=k,
            value_dtype=_INT32,
        )
    if np.any(values < 0) or np.any(values >= source_rows):
        raise ValueError("ground truth contains an ID outside the base-vector range")
    with destination.open("wb") as stream:
        np.asarray([rows, k], dtype=_UINT32).tofile(stream)
        values.astype(_UINT32, copy=False).tofile(stream)
    return k


def _load_manifest(output: Path) -> dict[str, Any] | None:
    path = output / "manifest.json"
    if not path.exists():
        if output.exists():
            raise ValueError(f"prepared SIFT1B output is incomplete: {output}")
        return None
    manifest = json.loads(path.read_text(encoding="utf-8"))
    required = [output / "source", output / "queries.fbin", output / "gt1000.bin"]
    if not all(path.exists() for path in required):
        raise ValueError(f"prepared SIFT1B output is incomplete: {output}")
    return manifest


def prepare(
    *,
    base: Path,
    queries: Path,
    ground_truth: Path,
    output: Path,
    rows_per_file: int = DEFAULT_ROWS_PER_FILE,
    row_group_rows: int = DEFAULT_ROW_GROUP_ROWS,
    compression_level: int = DEFAULT_COMPRESSION_LEVEL,
    workers: int = DEFAULT_WORKERS,
) -> dict[str, Any]:
    if rows_per_file <= 0 or row_group_rows <= 0:
        raise ValueError("rows-per-file and row-group-rows must be positive")
    if rows_per_file % row_group_rows:
        raise ValueError("rows-per-file must be a multiple of row-group-rows")
    if compression_level < 0:
        raise ValueError("compression-level must not be negative")
    if workers <= 0:
        raise ValueError("workers must be positive")

    base = base.expanduser().resolve()
    queries = queries.expanduser().resolve()
    ground_truth = ground_truth.expanduser().resolve()
    output = output.expanduser().resolve()
    if manifest := _load_manifest(output):
        return manifest

    rows, dimension = _inspect_vecs(base, value_dtype=np.dtype(np.uint8))
    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    source = temporary / "source"
    try:
        parquet_files, parquet_bytes = _write_source(
            base,
            source,
            rows=rows,
            dimension=dimension,
            rows_per_file=rows_per_file,
            row_group_rows=row_group_rows,
            compression_level=compression_level,
            workers=workers,
        )
        query_rows = _write_queries(
            queries, temporary / "queries.fbin", dimension=dimension
        )
        ground_truth_k = _write_ground_truth(
            ground_truth,
            temporary / "gt1000.bin",
            queries=query_rows,
            source_rows=rows,
        )
        source_info = inspect_parquet_source(source)
        query_info = inspect_float_matrix(temporary / "queries.fbin")
        if source_info.rows != rows or source_info.dimension != dimension:
            raise ValueError("written source Parquet does not match the base vectors")
        if query_info.rows != query_rows or query_info.dimension != dimension:
            raise ValueError("written query matrix does not match the query vectors")
        manifest = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "dataset": DATASET_NAME,
            "base": str(base),
            "base_bytes": base.stat().st_size,
            "queries": str(queries),
            "queries_bytes": queries.stat().st_size,
            "ground_truth": str(ground_truth),
            "ground_truth_bytes": ground_truth.stat().st_size,
            "rows": rows,
            "dimension": dimension,
            "query_rows": query_rows,
            "ground_truth_k": ground_truth_k,
            "distance": "l2_squared",
            "id": "zero-based base-vector row number",
            "source_parquet_files": parquet_files,
            "source_parquet_bytes": parquet_bytes,
            "source_parquet_compression": "ZSTD",
            "source_parquet_encoding": "BYTE_STREAM_SPLIT",
            "rows_per_file": rows_per_file,
            "row_group_rows": row_group_rows,
            "workers": workers,
        }
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        os.replace(temporary, output)
        return manifest
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="python -m benchmarks.tools.prepare_sift1b",
        description="Convert BigANN SIFT1B bvecs/ivecs files into benchmark Parquet.",
    )
    command.add_argument("--base", type=Path, required=True)
    command.add_argument("--queries", type=Path, required=True)
    command.add_argument("--ground-truth", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--rows-per-file", type=int, default=DEFAULT_ROWS_PER_FILE)
    command.add_argument("--row-group-rows", type=int, default=DEFAULT_ROW_GROUP_ROWS)
    command.add_argument(
        "--compression-level", type=int, default=DEFAULT_COMPRESSION_LEVEL
    )
    command.add_argument("--workers", type=int, default=DEFAULT_WORKERS)
    return command


def main(argv: list[str] | None = None) -> None:
    manifest = prepare(**vars(parser().parse_args(argv)))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
