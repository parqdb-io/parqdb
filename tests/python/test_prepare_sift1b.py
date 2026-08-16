from __future__ import annotations

from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import pytest

from benchmarks.tools import prepare_sift1b
from benchmarks.tools.datasets import (
    inspect_float_matrix,
    inspect_parquet_source,
    iter_parquet_vector_batches,
    load_ground_truth,
)


def _write_vecs(path: Path, values: np.ndarray, *, dtype: np.dtype[np.generic]) -> None:
    matrix = np.asarray(values, dtype=dtype, order="C")
    with path.open("wb") as stream:
        for row in matrix:
            np.asarray([len(row)], dtype="<i4").tofile(stream)
            row.tofile(stream)


def test_prepare_sift1b_converts_vecs_to_compressed_parquet(tmp_path: Path) -> None:
    base = tmp_path / "base.bvecs"
    queries = tmp_path / "query.bvecs"
    ground_truth = tmp_path / "groundtruth.ivecs"
    _write_vecs(
        base,
        np.asarray(
            [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
            dtype=np.uint8,
        ),
        dtype=np.dtype(np.uint8),
    )
    _write_vecs(
        queries,
        np.asarray([[1, 2, 3], [8, 9, 10]], dtype=np.uint8),
        dtype=np.dtype(np.uint8),
    )
    _write_vecs(
        ground_truth,
        np.asarray([[0, 1], [3, 2]], dtype="<i4"),
        dtype=np.dtype("<i4"),
    )

    output = tmp_path / "sift1b"
    manifest = prepare_sift1b.prepare(
        base=base,
        queries=queries,
        ground_truth=ground_truth,
        output=output,
        rows_per_file=2,
        row_group_rows=1,
        workers=2,
    )

    assert manifest["rows"] == 4
    assert manifest["dimension"] == 3
    assert manifest["query_rows"] == 2
    assert manifest["ground_truth_k"] == 2
    assert manifest["source_parquet_files"] == 2
    source = inspect_parquet_source(output / "source")
    assert (source.rows, source.dimension) == (4, 3)
    batches = list(iter_parquet_vector_batches(source, batch_rows=2))
    assert np.array_equal(np.concatenate([ids for ids, _ in batches]), np.arange(4))
    assert np.array_equal(
        np.concatenate([vectors for _, vectors in batches]),
        np.arange(12, dtype=np.float32).reshape(4, 3),
    )
    assert np.array_equal(
        inspect_float_matrix(output / "queries.fbin").memmap(),
        np.asarray([[1, 2, 3], [8, 9, 10]], dtype=np.float32),
    )
    assert np.array_equal(
        load_ground_truth(output / "gt1000.bin"),
        np.asarray([[0, 1], [3, 2]], dtype=np.int64),
    )
    metadata = pq.ParquetFile(output / "source" / "part-00000.parquet").metadata
    vector = metadata.row_group(0).column(1)
    assert vector.compression == "ZSTD"
    assert "BYTE_STREAM_SPLIT" in vector.encodings
    assert (
        prepare_sift1b.prepare(
            base=base,
            queries=queries,
            ground_truth=ground_truth,
            output=output,
        )
        == manifest
    )


def test_prepare_sift1b_rejects_invalid_layout(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="multiple"):
        prepare_sift1b.prepare(
            base=tmp_path / "missing.bvecs",
            queries=tmp_path / "missing-query.bvecs",
            ground_truth=tmp_path / "missing.ivecs",
            output=tmp_path / "output",
            rows_per_file=3,
            row_group_rows=2,
        )


def test_prepare_sift1b_rejects_invalid_worker_count(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="workers"):
        prepare_sift1b.prepare(
            base=tmp_path / "missing.bvecs",
            queries=tmp_path / "missing-query.bvecs",
            ground_truth=tmp_path / "missing.ivecs",
            output=tmp_path / "output",
            workers=0,
        )
