from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.dataset as ds
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from benchmarks.tools.datasets import (
    inspect_float_matrix,
    inspect_parquet_source,
    iter_parquet_vector_batches,
    load_float_matrix,
    load_ground_truth,
    load_parquet_vectors_by_id,
    parquet_file_fingerprints,
    sample_parquet_vectors,
    sha256_file,
    write_canonical_parquet,
)


def write_matrix(path: Path, values: np.ndarray) -> None:
    rows, dimension = values.shape
    path.write_bytes(
        np.asarray([rows, dimension], dtype="<u4").tobytes()
        + np.ascontiguousarray(values).tobytes()
    )


def test_diskann_matrix_and_ground_truth_loading(tmp_path: Path) -> None:
    base = tmp_path / "base.bin"
    values = np.arange(24, dtype="<f4").reshape(6, 4)
    write_matrix(base, values)

    matrix = inspect_float_matrix(base)
    assert matrix.rows == 6
    assert matrix.dimension == 4
    assert matrix.payload_bytes == 96
    np.testing.assert_array_equal(load_float_matrix(base, rows=2), values[:2])

    ground_truth = tmp_path / "ground-truth.bin"
    expected = np.asarray([[3, 1, 4], [2, 0, 5]], dtype="<u4")
    write_matrix(ground_truth, expected)
    np.testing.assert_array_equal(
        load_ground_truth(ground_truth, queries=1, k=2),
        [[3, 1]],
    )


def test_diskann_matrix_rejects_wrong_file_size(tmp_path: Path) -> None:
    path = tmp_path / "truncated.bin"
    path.write_bytes(np.asarray([2, 4], dtype="<u4").tobytes())
    with pytest.raises(ValueError, match="matrix size mismatch"):
        inspect_float_matrix(path)


def test_canonical_parquet_is_streamed_with_stable_ids(tmp_path: Path) -> None:
    base = tmp_path / "base.bin"
    values = np.arange(48, dtype="<f4").reshape(12, 4)
    write_matrix(base, values)
    output = tmp_path / "source"

    manifest = write_canonical_parquet(
        base,
        output,
        limit_rows=10,
        rows_per_file=4,
        row_group_rows=2,
    )

    assert manifest["rows"] == 10
    assert manifest["dimension"] == 4
    assert manifest["parquet_files"] == 3
    assert manifest["compression"] == "NONE"
    source = inspect_parquet_source(output)
    assert source.rows == 10
    assert source.dimension == 4
    assert source.bytes == manifest["parquet_bytes"]

    table = ds.dataset(output, format="parquet").to_table()
    assert table["id"].to_pylist() == list(range(10))
    np.testing.assert_array_equal(
        np.asarray(table["embedding"].to_pylist(), dtype=np.float32),
        values[:10],
    )

    batches = list(iter_parquet_vector_batches(source, batch_rows=3))
    np.testing.assert_array_equal(
        np.concatenate([ids for ids, _ in batches]),
        np.arange(10),
    )
    np.testing.assert_array_equal(
        np.concatenate([vectors for _, vectors in batches]),
        values[:10],
    )
    selected = np.sort(np.random.default_rng(42).choice(10, size=3, replace=False))
    np.testing.assert_array_equal(
        sample_parquet_vectors(source, 3, seed=42, batch_rows=3),
        values[selected],
    )
    fingerprints = parquet_file_fingerprints(source)
    assert len(fingerprints) == 3
    assert all(len(str(item["sha256"])) == 64 for item in fingerprints)
    assert (
        sha256_file(output / str(fingerprints[0]["path"]))
        == (fingerprints[0]["sha256"])
    )

    with pytest.raises(FileExistsError, match="output already exists"):
        write_canonical_parquet(base, output)


def test_original_parquet_columns_and_source_queries(tmp_path: Path) -> None:
    values = np.arange(24, dtype=np.float32).reshape(6, 4)
    vector_type = pa.list_(pa.field("item", pa.float32(), nullable=True))
    source_path = tmp_path / "source.parquet"
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(range(6), type=pa.int32()),
                pa.array([f"row-{value}" for value in range(6)]),
                pa.array(values.tolist(), type=vector_type),
            ],
            schema=pa.schema(
                [
                    pa.field("id", pa.int32(), nullable=True),
                    pa.field("text", pa.string(), nullable=True),
                    pa.field("emb", vector_type, nullable=True),
                ]
            ),
        ),
        source_path,
    )

    source = inspect_parquet_source(
        source_path,
        id_column="id",
        vector_column="emb",
    )
    assert source.rows == 6
    assert source.dimension == 4
    assert source.id_column == "id"
    assert source.vector_column == "emb"
    ids, vectors = zip(*iter_parquet_vector_batches(source, batch_rows=2), strict=True)
    np.testing.assert_array_equal(np.concatenate(ids), np.arange(6))
    np.testing.assert_array_equal(np.concatenate(vectors), values)
    np.testing.assert_array_equal(
        load_parquet_vectors_by_id(source, np.asarray([5, 1])),
        values[[5, 1]],
    )
