from __future__ import annotations

import hashlib
import sys
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from benchmarks.tools import prepare_gist
from benchmarks.tools.datasets import inspect_float_matrix, load_ground_truth


def test_prepare_gist_downloads_and_converts_in_bounded_files(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    h5py = pytest.importorskip("h5py")
    source = tmp_path / "fixture.hdf5"
    train = np.arange(12, dtype=np.float32).reshape(4, 3)
    queries = train[:2] + 0.25
    neighbors = np.asarray([[0, 1], [1, 0]], dtype=np.int32)
    with h5py.File(source, "w") as dataset:
        dataset.create_dataset("train", data=train)
        dataset.create_dataset("test", data=queries)
        dataset.create_dataset("neighbors", data=neighbors)

    monkeypatch.setattr(prepare_gist, "DATASET_BYTES", source.stat().st_size)
    monkeypatch.setattr(
        prepare_gist,
        "DATASET_SHA256",
        hashlib.sha256(source.read_bytes()).hexdigest(),
    )
    monkeypatch.setattr(prepare_gist, "ROWS", 4)
    monkeypatch.setattr(prepare_gist, "DIMENSION", 3)
    monkeypatch.setattr(prepare_gist, "QUERIES", 2)
    monkeypatch.setattr(prepare_gist, "GROUND_TRUTH_K", 2)

    root = tmp_path / "prepared"
    manifest = prepare_gist.prepare(root, url=source.as_uri())
    output = root / prepare_gist.DATASET_NAME

    assert manifest["rows"] == 4
    assert manifest["dimension"] == 3
    assert manifest["download_sha256"] == prepare_gist.DATASET_SHA256
    assert manifest["source_parquet_files"] == 1
    table = ds.dataset(output / "source", format="parquet").to_table()
    assert table["id"].to_pylist() == [0, 1, 2, 3]
    assert table["embedding"].to_pylist() == train.tolist()
    query_matrix = inspect_float_matrix(output / "queries.bin")
    np.testing.assert_array_equal(query_matrix.memmap(), queries)
    np.testing.assert_array_equal(
        load_ground_truth(output / "gt100.bin"),
        neighbors,
    )

    assert prepare_gist.prepare(root, url=source.as_uri()) == manifest
