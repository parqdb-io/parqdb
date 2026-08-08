"""Download and prepare the public ANN-Benchmarks GIST1M dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import sys
import time
import urllib.request
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DATASET_NAME = "gist-960-euclidean"
DATASET_URL = f"https://ann-benchmarks.com/{DATASET_NAME}.hdf5"
DATASET_BYTES = 3_844_648_288
DATASET_SHA256 = "8e95831936bfdbfa0a56086942e2cf98cd703517c67f985914183eb4cdbf026a"
DATASET_REVISION = 'etag-"34da1d8a80764582ee4b0c0839b7c32a-459"'
ROWS = 1_000_000
DIMENSION = 960
QUERIES = 1_000
GROUND_TRUTH_K = 100
DOWNLOAD_CHUNK_BYTES = 8 * 1024 * 1024


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(DOWNLOAD_CHUNK_BYTES):
            digest.update(chunk)
    return digest.hexdigest()


def _validate_download(path: Path) -> None:
    if path.stat().st_size != DATASET_BYTES:
        raise ValueError(
            f"download size mismatch: expected {DATASET_BYTES}, "
            f"found {path.stat().st_size}"
        )
    checksum = _sha256(path)
    if checksum != DATASET_SHA256:
        raise ValueError(
            f"download checksum mismatch: expected {DATASET_SHA256}, found {checksum}"
        )


def _download(url: str, destination: Path) -> None:
    if destination.exists() and destination.stat().st_size == DATASET_BYTES:
        _validate_download(destination)
        return
    if destination.exists():
        raise ValueError(
            f"existing download has the wrong size: {destination}; remove it to retry"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".part")
    downloaded = temporary.stat().st_size if temporary.exists() else 0
    headers = {"Range": f"bytes={downloaded}-"} if downloaded else {}
    request = urllib.request.Request(url, headers=headers)
    started = time.monotonic()
    with urllib.request.urlopen(request) as response:
        if downloaded and getattr(response, "status", None) != 206:
            downloaded = 0
            temporary.unlink()
        mode = "ab" if downloaded else "wb"
        with temporary.open(mode) as stream:
            while chunk := response.read(DOWNLOAD_CHUNK_BYTES):
                stream.write(chunk)
                downloaded += len(chunk)
                elapsed = max(time.monotonic() - started, 0.001)
                percent = 100.0 * downloaded / DATASET_BYTES
                print(
                    f"\rGIST download {percent:5.1f}% "
                    f"({downloaded / 2**30:.2f} GiB, "
                    f"{downloaded / elapsed / 2**20:.1f} MiB/s)",
                    end="",
                    file=sys.stderr,
                    flush=True,
                )
    print(file=sys.stderr)
    _validate_download(temporary)
    os.replace(temporary, destination)


def _write_matrix(path: Path, values: np.ndarray, dtype: str) -> None:
    matrix = np.asarray(values, dtype=dtype, order="C")
    if matrix.ndim != 2:
        raise ValueError("benchmark matrices must be two-dimensional")
    with path.open("wb") as stream:
        np.asarray(matrix.shape, dtype="<u4").tofile(stream)
        matrix.tofile(stream)


def _write_source(train: Any, destination: Path) -> tuple[int, int]:
    vector_type = pa.list_(pa.field("element", pa.float32(), nullable=False))
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("embedding", vector_type, nullable=False),
        ]
    )
    rows_per_file = 16_384
    row_group_rows = 4_096
    parquet_bytes = 0
    file_count = 0
    for start in range(0, ROWS, rows_per_file):
        stop = min(start + rows_per_file, ROWS)
        vectors = np.asarray(train[start:stop], dtype=np.float32, order="C")
        if vectors.shape != (stop - start, DIMENSION):
            raise ValueError("GIST train matrix changed during conversion")
        values = pa.array(vectors.reshape(-1), type=pa.float32())
        offsets = pa.array(
            np.arange(
                0,
                (len(vectors) + 1) * DIMENSION,
                DIMENSION,
                dtype=np.int32,
            )
        )
        embeddings = pa.ListArray.from_arrays(offsets, values).cast(vector_type)
        table = pa.Table.from_arrays(
            [
                pa.array(np.arange(start, stop, dtype=np.int64)),
                embeddings,
            ],
            schema=schema,
        )
        part = destination / f"part-{file_count:05d}.parquet"
        pq.write_table(
            table,
            part,
            compression="NONE",
            use_dictionary=False,
            row_group_size=row_group_rows,
        )
        parquet_bytes += part.stat().st_size
        file_count += 1
        print(
            f"\rGIST Parquet {100.0 * stop / ROWS:5.1f}%",
            end="",
            file=sys.stderr,
            flush=True,
        )
    print(file=sys.stderr)
    return file_count, parquet_bytes


def _prepared_manifest(output: Path) -> dict[str, Any] | None:
    manifest_path = output / "manifest.json"
    if not manifest_path.exists():
        if output.exists():
            raise ValueError(f"prepared GIST dataset is incomplete: {output}")
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    expected = {
        "schema_version": 1,
        "dataset": DATASET_NAME,
        "revision": DATASET_REVISION,
        "download_bytes": DATASET_BYTES,
        "download_sha256": DATASET_SHA256,
        "rows": ROWS,
        "dimension": DIMENSION,
        "queries": QUERIES,
        "ground_truth_k": GROUND_TRUTH_K,
        "distance": "euclidean",
    }
    if any(manifest.get(key) != value for key, value in expected.items()):
        raise ValueError(f"prepared GIST manifest is incompatible: {manifest_path}")
    required = [output / "source", output / "queries.bin", output / "gt100.bin"]
    if not all(path.exists() for path in required):
        raise ValueError(f"prepared GIST dataset is incomplete: {output}")
    return manifest


def prepare(root: Path, *, url: str = DATASET_URL) -> dict[str, Any]:
    root = root.expanduser().resolve()
    download = root / "downloads" / f"{DATASET_NAME}.hdf5"
    output = root / DATASET_NAME
    if manifest := _prepared_manifest(output):
        return manifest
    _download(url, download)

    try:
        import h5py  # pyright: ignore[reportMissingImports]
    except ImportError as error:
        raise RuntimeError("GIST preparation requires h5py") from error

    temporary = output.with_name(f".{output.name}.tmp-{uuid.uuid4().hex}")
    source = temporary / "source"
    source.mkdir(parents=True)
    try:
        with h5py.File(download, "r") as dataset:
            train = dataset["train"]
            test = dataset["test"]
            neighbors = dataset["neighbors"]
            if train.shape != (ROWS, DIMENSION):
                raise ValueError(f"unexpected GIST train shape: {train.shape}")
            if test.shape != (QUERIES, DIMENSION):
                raise ValueError(f"unexpected GIST test shape: {test.shape}")
            if neighbors.shape != (QUERIES, GROUND_TRUTH_K):
                raise ValueError(
                    f"unexpected GIST ground-truth shape: {neighbors.shape}"
                )
            file_count, parquet_bytes = _write_source(train, source)
            _write_matrix(temporary / "queries.bin", test[:], "<f4")
            neighbor_values = np.asarray(neighbors[:], dtype=np.int64)
            if np.any(neighbor_values < 0) or np.any(neighbor_values >= ROWS):
                raise ValueError("GIST ground truth contains an invalid source ID")
            _write_matrix(temporary / "gt100.bin", neighbor_values, "<u4")

        manifest = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "dataset": DATASET_NAME,
            "revision": DATASET_REVISION,
            "url": url,
            "download_bytes": DATASET_BYTES,
            "download_sha256": DATASET_SHA256,
            "rows": ROWS,
            "dimension": DIMENSION,
            "queries": QUERIES,
            "ground_truth_k": GROUND_TRUTH_K,
            "distance": "euclidean",
            "id": "zero-based train row number",
            "source_parquet_files": file_count,
            "source_parquet_bytes": parquet_bytes,
            "source_parquet_compression": "NONE",
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
        prog="python -m benchmarks.tools.prepare_gist",
        description="Download and convert ANN-Benchmarks GIST1M.",
    )
    command.add_argument("--root", type=Path, required=True)
    command.add_argument("--url", default=DATASET_URL)
    return command


def main(argv: list[str] | None = None) -> None:
    manifest = prepare(**vars(parser().parse_args(argv)))
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
