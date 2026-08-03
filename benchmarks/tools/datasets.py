"""Dataset readers and fixture writers used by benchmark runners."""

from __future__ import annotations

import json
import os
import uuid
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from hashlib import sha256
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq

_HEADER_DTYPE = np.dtype("<u4")
_FLOAT32 = np.dtype("<f4")
_UINT32 = np.dtype("<u4")
_HEADER_BYTES = 2 * _HEADER_DTYPE.itemsize


@dataclass(frozen=True)
class MatrixFile:
    path: Path
    rows: int
    dimension: int
    dtype: np.dtype[np.generic]

    @property
    def payload_bytes(self) -> int:
        return self.rows * self.dimension * self.dtype.itemsize

    def memmap(self) -> np.memmap:
        return np.memmap(
            self.path,
            dtype=self.dtype,
            mode="r",
            offset=_HEADER_BYTES,
            shape=(self.rows, self.dimension),
        )


@dataclass(frozen=True)
class ParquetSource:
    path: Path
    rows: int
    dimension: int
    bytes: int
    id_column: str
    vector_column: str


def inspect_matrix(path: Path, *, dtype: np.dtype[np.generic]) -> MatrixFile:
    path = path.expanduser().resolve()
    with path.open("rb") as stream:
        header = np.fromfile(stream, dtype=_HEADER_DTYPE, count=2)
    if len(header) != 2:
        raise ValueError(f"matrix header is truncated: {path}")
    rows, dimension = (int(value) for value in header)
    if rows <= 0 or dimension <= 0:
        raise ValueError(f"matrix dimensions must be positive: {path}")
    matrix = MatrixFile(path, rows, dimension, dtype)
    expected_bytes = _HEADER_BYTES + matrix.payload_bytes
    actual_bytes = path.stat().st_size
    if actual_bytes != expected_bytes:
        raise ValueError(
            f"matrix size mismatch for {path}: expected {expected_bytes}, "
            f"found {actual_bytes}"
        )
    return matrix


def inspect_float_matrix(path: Path) -> MatrixFile:
    return inspect_matrix(path, dtype=_FLOAT32)


def load_float_matrix(path: Path, *, rows: int | None = None) -> np.ndarray:
    matrix = inspect_float_matrix(path)
    count = matrix.rows if rows is None else min(rows, matrix.rows)
    return np.ascontiguousarray(matrix.memmap()[:count], dtype=np.float32)


def load_ground_truth(
    path: Path,
    *,
    queries: int | None = None,
    k: int | None = None,
) -> np.ndarray:
    matrix = inspect_matrix(path, dtype=_UINT32)
    query_count = matrix.rows if queries is None else min(queries, matrix.rows)
    result_k = matrix.dimension if k is None else k
    if result_k <= 0 or result_k > matrix.dimension:
        raise ValueError(f"ground-truth K must be in 1..={matrix.dimension}")
    return np.ascontiguousarray(
        matrix.memmap()[:query_count, :result_k],
        dtype=np.int64,
    )


def inspect_parquet_source(
    path: Path,
    *,
    id_column: str = "id",
    vector_column: str = "embedding",
) -> ParquetSource:
    path = path.expanduser().resolve()
    dataset = ds.dataset(path, format="parquet", exclude_invalid_files=True)
    schema = dataset.schema
    if id_column not in schema.names:
        raise ValueError(f"benchmark ID column not found: {id_column}")
    if vector_column not in schema.names:
        raise ValueError(f"benchmark vector column not found: {vector_column}")
    id_field = schema.field(id_column)
    if id_field.type not in (pa.int32(), pa.int64()):
        raise ValueError("benchmark IDs must be INT32 or INT64")
    vector_field = schema.field(vector_column)
    if not (
        pa.types.is_list(vector_field.type)
        or pa.types.is_large_list(vector_field.type)
        or pa.types.is_fixed_size_list(vector_field.type)
    ):
        raise ValueError("benchmark vectors must be a list")
    if vector_field.type.value_type != pa.float32():
        raise ValueError("benchmark vector elements must be FLOAT32")

    batches = dataset.to_batches(columns=[vector_column], batch_size=1)
    try:
        first_batch = next(iter(batches))
    except StopIteration as error:
        raise ValueError("benchmark source is empty") from error
    first_vector = first_batch.column(0)[0].as_py()
    if first_vector is None or any(value is None for value in first_vector):
        raise ValueError("benchmark vectors must not contain NULLs")
    dimension = len(first_vector)
    if dimension <= 0:
        raise ValueError("benchmark vectors must not be empty")
    parquet_files = (
        [path]
        if path.is_file()
        else sorted(file for file in path.rglob("*.parquet") if file.is_file())
    )
    return ParquetSource(
        path=path,
        rows=dataset.count_rows(),
        dimension=dimension,
        bytes=sum(file.stat().st_size for file in parquet_files),
        id_column=id_column,
        vector_column=vector_column,
    )


def iter_parquet_vector_batches(
    source: ParquetSource,
    *,
    batch_rows: int = 65_536,
) -> Iterator[tuple[np.ndarray, np.ndarray]]:
    if batch_rows <= 0:
        raise ValueError("batch_rows must be positive")
    dataset = ds.dataset(source.path, format="parquet", exclude_invalid_files=True)
    for batch in dataset.to_batches(
        columns=[source.id_column, source.vector_column],
        batch_size=batch_rows,
    ):
        ids = batch.column(source.id_column)
        if ids.null_count:
            raise ValueError("benchmark IDs must not contain NULLs")
        embeddings = batch.column(source.vector_column)
        if embeddings.null_count or embeddings.values.null_count:
            raise ValueError("benchmark vectors must not contain NULLs")
        values = embeddings.flatten().to_numpy(zero_copy_only=False)
        if len(values) != len(batch) * source.dimension:
            raise ValueError("benchmark vector dimension is inconsistent")
        yield (
            np.ascontiguousarray(ids.to_numpy(zero_copy_only=False), dtype=np.int64),
            np.ascontiguousarray(
                values.reshape(len(batch), source.dimension),
                dtype=np.float32,
            ),
        )


def sample_parquet_vectors(
    source: ParquetSource,
    rows: int,
    *,
    seed: int,
    batch_rows: int = 65_536,
    progress: Callable[[int, int, str], None] | None = None,
) -> np.ndarray:
    if not 1 <= rows <= source.rows:
        raise ValueError(f"sample rows must be in 1..={source.rows}")
    selected = np.sort(
        np.random.default_rng(seed).choice(source.rows, size=rows, replace=False)
    )
    sample = np.empty((rows, source.dimension), dtype=np.float32)
    written = 0
    source_offset = 0
    for _, vectors in iter_parquet_vector_batches(source, batch_rows=batch_rows):
        lower = np.searchsorted(selected, source_offset, side="left")
        upper = np.searchsorted(selected, source_offset + len(vectors), side="left")
        batch_selected = selected[lower:upper] - source_offset
        count = len(batch_selected)
        sample[written : written + count] = vectors[batch_selected]
        written += count
        source_offset += len(vectors)
        if progress is not None:
            progress(source_offset, source.rows, "Parquet vectors")
    if source_offset != source.rows or written != rows:
        raise ValueError("benchmark source row count changed during sampling")
    return sample


def load_parquet_vectors_by_id(
    source: ParquetSource,
    ids: np.ndarray,
) -> np.ndarray:
    requested = np.ascontiguousarray(ids, dtype=np.int64)
    if requested.ndim != 1 or len(requested) == 0:
        raise ValueError("query IDs must be a non-empty one-dimensional array")
    if len(np.unique(requested)) != len(requested):
        raise ValueError("query IDs must be unique")

    dataset = ds.dataset(source.path, format="parquet", exclude_invalid_files=True)
    table = dataset.to_table(
        columns=[source.id_column, source.vector_column],
        filter=ds.field(source.id_column).isin(requested.tolist()),
    )
    found_ids = table[source.id_column]
    vectors = table[source.vector_column].combine_chunks()
    if found_ids.null_count or vectors.null_count or vectors.values.null_count:
        raise ValueError("query IDs and vectors must not contain NULLs")
    found = np.ascontiguousarray(
        found_ids.to_numpy(zero_copy_only=False),
        dtype=np.int64,
    )
    if len(found) != len(requested) or len(np.unique(found)) != len(found):
        raise ValueError("each query ID must resolve to exactly one source row")
    positions = {int(value): position for position, value in enumerate(found)}
    try:
        order = np.asarray(
            [positions[int(value)] for value in requested], dtype=np.int64
        )
    except KeyError as error:
        raise ValueError(f"query ID not found: {error.args[0]}") from error
    values = vectors.flatten().to_numpy(zero_copy_only=False)
    if len(values) != len(vectors) * source.dimension:
        raise ValueError("benchmark vector dimension is inconsistent")
    matrix = np.ascontiguousarray(
        values.reshape(len(vectors), source.dimension),
        dtype=np.float32,
    )
    return np.ascontiguousarray(matrix[order])


def write_canonical_parquet(
    base_path: Path,
    output_path: Path,
    *,
    limit_rows: int | None = None,
    rows_per_file: int = 262_144,
    row_group_rows: int = 32_768,
) -> dict[str, object]:
    if limit_rows is not None and limit_rows <= 0:
        raise ValueError("limit_rows must be positive")
    if rows_per_file <= 0 or row_group_rows <= 0:
        raise ValueError("Parquet row counts must be positive")
    if row_group_rows > rows_per_file:
        raise ValueError("row_group_rows must not exceed rows_per_file")

    matrix = inspect_float_matrix(base_path)
    rows = matrix.rows if limit_rows is None else min(limit_rows, matrix.rows)
    output_path = output_path.expanduser().resolve()
    if output_path.exists():
        raise FileExistsError(f"output already exists: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = output_path.with_name(f".{output_path.name}.tmp-{uuid.uuid4().hex}")
    temporary.mkdir()

    vector_type = pa.list_(pa.field("element", pa.float32(), nullable=False))
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("embedding", vector_type, nullable=False),
        ]
    )
    mapped = matrix.memmap()
    parquet_bytes = 0
    file_count = 0
    try:
        for start in range(0, rows, rows_per_file):
            stop = min(start + rows_per_file, rows)
            count = stop - start
            values = pa.array(mapped[start:stop].reshape(-1), type=pa.float32())
            offsets = pa.array(
                np.arange(
                    0,
                    (count + 1) * matrix.dimension,
                    matrix.dimension,
                    dtype=np.int64,
                ),
                type=pa.int64(),
            )
            vectors = pa.LargeListArray.from_arrays(offsets, values).cast(vector_type)
            table = pa.Table.from_arrays(
                [
                    pa.array(np.arange(start, stop, dtype=np.int64)),
                    vectors,
                ],
                schema=schema,
            )
            part = temporary / f"part-{file_count:05d}.parquet"
            pq.write_table(
                table,
                part,
                compression="NONE",
                row_group_size=row_group_rows,
            )
            parquet_bytes += part.stat().st_size
            file_count += 1

        manifest: dict[str, object] = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "source": str(matrix.path),
            "source_rows": matrix.rows,
            "rows": rows,
            "dimension": matrix.dimension,
            "dtype": "float32",
            "id": "zero-based base-file row number",
            "rows_per_file": rows_per_file,
            "row_group_rows": row_group_rows,
            "parquet_files": file_count,
            "parquet_bytes": parquet_bytes,
            "compression": "NONE",
        }
        (temporary / "_dataset.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, output_path)
        return manifest
    except BaseException:
        for child in temporary.iterdir():
            child.unlink()
        temporary.rmdir()
        raise


def matrix_description(matrix: MatrixFile) -> dict[str, object]:
    result = asdict(matrix)
    result["path"] = str(matrix.path)
    result["dtype"] = matrix.dtype.name
    result["payload_bytes"] = matrix.payload_bytes
    return result


def sha256_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def parquet_file_fingerprints(source: ParquetSource) -> list[dict[str, object]]:
    files = (
        [source.path]
        if source.path.is_file()
        else sorted(source.path.rglob("*.parquet"))
    )
    return [
        {
            "path": (
                file.name
                if source.path.is_file()
                else str(file.relative_to(source.path))
            ),
            "bytes": file.stat().st_size,
            "sha256": sha256_file(file),
        }
        for file in files
    ]
