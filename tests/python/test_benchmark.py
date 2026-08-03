from __future__ import annotations

import json
import subprocess
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def write_matrix(path: Path, values: np.ndarray) -> None:
    with path.open("wb") as stream:
        np.asarray(values.shape, dtype="<u4").tofile(stream)
        np.asarray(values).tofile(stream)


def test_build_benchmark_smoke(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    output = tmp_path / "result.json"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.build",
            "--rows",
            "64",
            "--dimension",
            "4",
            "--nlist",
            "4",
            "--seed",
            "42",
            "--output",
            str(output),
        ],
        cwd=repository,
        check=True,
    )
    result = json.loads(output.read_text(encoding="utf-8"))

    assert result["dataset"]["kind"] == "synthetic"
    assert result["dataset"]["rows"] == 64
    assert result["dataset"]["dimension"] == 4
    assert result["dataset"]["queries"] is None
    assert result["dataset"]["source_parquet_bytes"] > 0
    relify_result = result["results"][0]
    assert relify_result["implementation"] == "relify"
    assert relify_result["preparation_seconds"] > 0
    assert relify_result["build_seconds"] > 0
    assert relify_result["build_points_per_second"] > 0
    assert relify_result["managed_bytes"] > 0
    assert relify_result["training_rows"] == 64
    assert len(relify_result["build_seconds_samples"]) == 1
    assert len(relify_result["preparation_seconds_samples"]) == 1
    assert result["source_revision"]
    assert result["software"]["rustc"].startswith("rustc ")
    assert result["benchmark"] == "build"
    implementations = {
        implementation["implementation"]: implementation
        for implementation in result["results"]
    }
    if faiss_result := implementations.get("faiss"):
        assert faiss_result["omp_threads"] > 0
        assert faiss_result["parallel_mode"] == 1


def test_build_then_query_uses_original_parquet_and_source_queries(
    tmp_path: Path,
) -> None:
    repository = Path(__file__).parents[2]
    rng = np.random.default_rng(42)
    base = rng.standard_normal((32, 4), dtype=np.float32)
    queries = base[:2]
    ids = np.arange(len(base), dtype=np.int64)
    expected = []
    for query in queries:
        distance = np.einsum("ij,ij->i", base - query, base - query)
        expected.append(np.lexsort((ids, distance))[:4])

    ground_truth_path = tmp_path / "ground-truth.bin"
    source = tmp_path / "source.parquet"
    output = tmp_path / "result.json"
    work_root = tmp_path / "work"
    write_matrix(ground_truth_path, np.asarray(expected, dtype="<u4"))
    vector_type = pa.list_(pa.field("item", pa.float32(), nullable=True))
    pq.write_table(
        pa.Table.from_arrays(
            [
                pa.array(ids, type=pa.int32()),
                pa.array([f"row-{value}" for value in ids]),
                pa.array(base.tolist(), type=vector_type),
            ],
            schema=pa.schema(
                [
                    pa.field("id", pa.int32(), nullable=True),
                    pa.field("text", pa.string(), nullable=True),
                    pa.field("emb", vector_type, nullable=True),
                ]
            ),
        ),
        source,
    )
    build_output = tmp_path / "build.json"
    build_command = [
        sys.executable,
        "-m",
        "benchmarks.build",
        "--source-parquet",
        str(source),
        "--id-column",
        "id",
        "--vector-column",
        "emb",
        "--dataset-name",
        "example/original-parquet",
        "--dataset-revision",
        "abc123",
        "--dataset-split",
        "train",
        "--nlist",
        "4",
        "--threads",
        "2",
        "--work-root",
        str(work_root),
        "--output",
        str(build_output),
    ]
    subprocess.run(build_command, cwd=repository, check=True)

    command = [
        sys.executable,
        "-m",
        "benchmarks.query",
        "--source-parquet",
        str(source),
        "--id-column",
        "id",
        "--vector-column",
        "emb",
        "--dataset-name",
        "example/original-parquet",
        "--dataset-revision",
        "abc123",
        "--dataset-split",
        "train",
        "--query-source-start",
        "0",
        "--ground-truth",
        str(ground_truth_path),
        "--num-queries",
        "2",
        "--nlist",
        "4",
        "--nprobe",
        "4",
        "--k",
        "2",
        "--curve-k-values",
        "1,2,4",
        "--curve-nprobe-values",
        "1,4",
        "--search-repetitions",
        "1",
        "--warmup-queries",
        "1",
        "--threads",
        "2",
        "--work-root",
        str(work_root),
        "--output",
        str(output),
    ]
    subprocess.run(
        command,
        cwd=repository,
        check=True,
    )

    build_result = json.loads(build_output.read_text(encoding="utf-8"))
    result = json.loads(output.read_text(encoding="utf-8"))
    assert build_result["benchmark"] == "build"
    assert result["benchmark"] == "query"
    assert result["dataset"]["kind"] == "example/original-parquet"
    assert result["dataset"]["revision"] == "abc123"
    assert result["dataset"]["split"] == "train"
    assert result["dataset"]["rows"] == 32
    assert result["dataset"]["dimension"] == 4
    assert result["dataset"]["queries"] == 2
    assert result["dataset"]["source_parquet"] == str(source)
    assert result["dataset"]["id_column"] == "id"
    assert result["dataset"]["vector_column"] == "emb"
    assert result["dataset"]["query_source_id_range"] == [0, 1]
    assert result["dataset"]["ground_truth"] == str(ground_truth_path)
    assert result["parameters"]["threads"] == 2
    assert result["parameters"]["index_root"] == str(work_root / "indexes")
    assert result["resources"]["cpus"] >= 2
    assert all(
        not implementation["index_reused"] for implementation in build_result["results"]
    )
    assert all(
        implementation["build_resource_usage"]["peak_rss_bytes"] > 0
        for implementation in build_result["results"]
    )
    assert all(
        implementation["peak_memory_bytes"] > 0
        for implementation in build_result["results"]
    )
    built_implementations = {
        implementation["implementation"] for implementation in build_result["results"]
    }
    assert all(
        (work_root / "indexes" / implementation / "benchmark-artifact.json").is_file()
        for implementation in built_implementations
    )

    subprocess.run(build_command, cwd=repository, check=True)
    reused = json.loads(build_output.read_text(encoding="utf-8"))
    assert all(implementation["index_reused"] for implementation in reused["results"])
    assert all("build_seconds" not in item for item in result["results"])
    assert all(item["index_bytes"] > 0 for item in result["results"])


def test_search_curve_chart_is_valid_svg(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    points = [
        {
            "k": k,
            "nprobe": nprobe,
            "recall_at_k": min(1.0, nprobe / 4096),
            "latency_ms_p50": float(nprobe * k / 100),
            "latency_ms_p95": float(nprobe * k / 80),
        }
        for k in (10000, 20000, 100000)
        for nprobe in (1, 64, 4096)
    ]
    run = {
        "schema_version": 1,
        "dataset": {"rows": 100_000, "dimension": 128, "queries": 50},
        "parameters": {
            "nlist": 4096,
            "search_repetitions": 3,
        },
        "resources": {"cpus": 10, "memory_limit_bytes": 16 * 1024**3},
        "software": {"machine": "arm64"},
        "results": [
            {"implementation": "relify", "search_curve": points},
            {"implementation": "faiss", "search_curve": points},
        ],
    }
    result_path = tmp_path / "result.json"
    output = tmp_path / "search.svg"
    result_path.write_text(json.dumps(run), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.tools.render_search_results",
            str(result_path),
            "--k-values",
            "10000,20000,100000",
            "--output",
            str(output),
        ],
        cwd=repository,
        check=True,
    )
    encoded = output.read_text(encoding="utf-8")
    root = ET.fromstring(encoded)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert "Large-k IVF Recall-Latency" in encoded
    assert "Recall@10K" in encoded
    assert "Recall@100K" in encoded
    assert "intra-query parallel" in encoded
    assert "10 vCPUs" in encoded
    assert "Faiss" in encoded


def test_build_time_chart_is_valid_svg(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    run = {
        "schema_version": 1,
        "dataset": {"rows": 1_000_000, "dimension": 128},
        "parameters": {"nlist": 4096},
        "resources": {"cpus": 10, "memory_limit_bytes": 16 * 1024**3},
        "software": {"machine": "arm64"},
        "results": [
            {"implementation": "relify", "build_seconds": 28.7},
            {"implementation": "faiss", "build_seconds": 49.3},
        ],
    }
    result_path = tmp_path / "result.json"
    output = tmp_path / "build.svg"
    result_path.write_text(json.dumps(run), encoding="utf-8")
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.tools.render_build_results",
            str(result_path),
            "--output",
            str(output),
        ],
        cwd=repository,
        check=True,
    )
    encoded = output.read_text(encoding="utf-8")
    root = ET.fromstring(encoded)
    assert root.tag == "{http://www.w3.org/2000/svg}svg"
    assert "Persisted IVF-Flat Build Time" in encoded
    assert "Faiss" in encoded
    assert "49.30s" in encoded
    assert "10 vCPUs" in encoded


def test_committed_benchmark_charts_match_raw_results(tmp_path: Path) -> None:
    repository = Path(__file__).parents[2]
    search_results = repository / "benchmarks" / "results" / "macos-arm64-2026-07-29"
    build_output = tmp_path / "build.svg"
    search_output = tmp_path / "search.svg"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.tools.render_build_results",
            str(search_results / "1m.json"),
            "--output",
            str(build_output),
        ],
        cwd=repository,
        check=True,
    )
    subprocess.run(
        [
            sys.executable,
            "-m",
            "benchmarks.tools.render_search_results",
            str(search_results / "1m.json"),
            "--k-values",
            "10000,20000,100000",
            "--output",
            str(search_output),
        ],
        cwd=repository,
        check=True,
    )
    assert (
        build_output.read_bytes()
        == (repository / "assets" / "build-time.svg").read_bytes()
    )
    assert (
        search_output.read_bytes()
        == (repository / "assets" / "search-recall-latency.svg").read_bytes()
    )
