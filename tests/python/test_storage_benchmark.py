from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parents[2]))

from benchmarks.tools.datasets import write_canonical_parquet


def write_matrix(path: Path, values: np.ndarray, dtype: str) -> None:
    with path.open("wb") as stream:
        np.asarray(values.shape, dtype="<u4").tofile(stream)
        np.asarray(values, dtype=dtype).tofile(stream)


def test_storage_backed_prepare_and_query_smoke(tmp_path: Path) -> None:
    pytest.importorskip("faiss")
    repository = Path(__file__).parents[2]
    rng = np.random.default_rng(42)
    base = rng.standard_normal((128, 8), dtype=np.float32)
    queries = base[[3, 17, 89]]
    ids = np.arange(len(base))
    expected = []
    for query in queries:
        distances = np.einsum("ij,ij->i", base - query, base - query)
        expected.append(np.lexsort((ids, distances))[:16])

    base_path = tmp_path / "base.bin"
    query_path = tmp_path / "query.bin"
    ground_truth_path = tmp_path / "ground-truth.bin"
    source = tmp_path / "source"
    prepared = tmp_path / "prepared"
    write_matrix(base_path, base, "<f4")
    write_matrix(query_path, queries, "<f4")
    write_matrix(ground_truth_path, np.asarray(expected), "<u4")
    write_canonical_parquet(
        base_path,
        source,
        rows_per_file=64,
        row_group_rows=32,
    )

    for implementation in ("parqdb", "faiss"):
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.tools.prepare_storage",
                "--implementation",
                implementation,
                "--source-parquet",
                str(source),
                "--output",
                str(prepared),
                "--nlist",
                "4",
                "--threads",
                "2",
                "--shard-rows",
                "64",
                "--batch-rows",
                "32",
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )

    metadata = json.loads(
        (prepared / "storage-indexes.json").read_text(encoding="utf-8")
    )
    assert set(metadata["indexes"]) == {"parqdb", "faiss"}
    assert metadata["indexes"]["faiss"]["inverted_lists_type"] == (
        "OnDiskInvertedLists"
    )
    assert metadata["indexes"]["faiss"]["payload_bytes"] > 0
    assert all(
        index["page_cache_evicted_after_build"]
        for index in metadata["indexes"].values()
    )

    for implementation in ("parqdb", "faiss"):
        output = tmp_path / f"{implementation}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.tools.storage_query",
                "--implementation",
                implementation,
                "--prepared",
                str(prepared),
                "--query-file",
                str(query_path),
                "--ground-truth",
                str(ground_truth_path),
                "--num-queries",
                "3",
                "--nprobe",
                "4",
                "--k",
                "4",
                "--curve-nprobe-values",
                "1,4",
                "--curve-k-values",
                "4,16",
                "--search-repetitions",
                "1",
                "--warmup-queries",
                "1",
                "--threads",
                "2",
                "--minimum-payload-bytes",
                "0",
                "--allow-resource-mismatch",
                "--output",
                str(output),
            ],
            cwd=repository,
            check=True,
        )
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result["suite"] == "storage-backed-ivf"
        assert result["result"]["implementation"] == implementation
        full_probe = [
            point for point in result["result"]["search_curve"] if point["nprobe"] == 4
        ]
        assert all(point["recall_at_k"] >= 0.75 for point in full_probe)
        assert all(
            "cgroup_memory_file_bytes" in point["resource_usage"]
            for point in result["result"]["search_curve"]
        )

    index_root = tmp_path / "standard-indexes"
    common_build_arguments = [
        "--source-parquet",
        str(source),
        "--id-column",
        "id",
        "--vector-column",
        "embedding",
        "--nlist",
        "4",
        "--threads",
        "2",
        "--index-root",
        str(index_root),
        "--rebuild",
        "--no-progress",
    ]
    for module, encoding in (
        ("benchmarks.build", "lvq8"),
        ("benchmarks.tools.faiss", "sq8"),
    ):
        operation = ["build"] if module == "benchmarks.tools.faiss" else []
        subprocess.run(
            [
                sys.executable,
                "-m",
                module,
                *operation,
                *common_build_arguments,
                "--encoding",
                encoding,
            ],
            cwd=repository,
            check=True,
            capture_output=True,
            text=True,
        )
    for implementation in ("parqdb", "faiss"):
        output = tmp_path / f"reused-{implementation}.json"
        subprocess.run(
            [
                sys.executable,
                "-m",
                "benchmarks.tools.storage_query",
                "--implementation",
                implementation,
                "--index-root",
                str(index_root),
                "--query-file",
                str(query_path),
                "--ground-truth",
                str(ground_truth_path),
                "--num-queries",
                "3",
                "--nprobe",
                "4",
                "--k",
                "16",
                "--curve-nprobe-values",
                "1,4",
                "--curve-k-values",
                "16",
                "--search-repetitions",
                "1",
                "--warmup-queries",
                "1",
                "--threads",
                "2",
                "--minimum-payload-bytes",
                "0",
                "--allow-resource-mismatch",
                "--output",
                str(output),
            ],
            cwd=repository,
            check=True,
        )
        result = json.loads(output.read_text(encoding="utf-8"))
        assert result["index_input"] == "benchmark-index-root"
        assert result["result"]["runtime"]["storage"] in {
            "Parquet IVF with bounded decompressed Page cache",
            "OnDiskInvertedLists",
        }
        assert not result["result"]["index"]["page_cache_evicted_before_query"]
