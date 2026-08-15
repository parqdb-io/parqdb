from __future__ import annotations

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import shutil
import statistics
import sys
import tempfile
import time
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from benchmarks.tools.datasets import (
    ParquetSource,
    inspect_parquet_source,
    iter_parquet_vector_batches,
    load_ground_truth,
    load_parquet_vectors_by_id,
    sample_parquet_vectors,
)
from benchmarks.tools.harness import (
    FAISS_PARALLEL_MODE,
    KMEANS_MAX_ITERATIONS,
    KMEANS_SEED,
    BuildProgressBar,
    CounterProgressBar,
    benchmark_revision,
    command_version,
    directory_bytes,
    load_vectors,
    measure_search_curve,
    parse_positive_ints,
    sync_file,
)
from benchmarks.tools.resources import (
    ResourceMonitor,
    affinity_cpu_ids,
    cgroup_environment,
    effective_cpu_count,
)
from benchmarks.tools.search_adapters import (
    configure_relify_session,
    faiss_search,
    relify_search,
)

if TYPE_CHECKING:
    import relify

UNTIMED_RELIFY_BUILD_PHASES = {
    None,
    "pending",
    "scanning_source",
    "reading_training_vectors",
}
TRAINING_SAMPLING = {
    "relify": "streaming-reservoir-v1",
    "faiss": "uniform-without-replacement-v1",
}
RELIFY_POSTINGS_LAYOUT = "hive-cid-file-v1"
RELIFY_ENCODINGS = ("lvq8", "lvq4", "source")
FAISS_ENCODINGS = ("sq8", "sq4", "flat")


@dataclass(frozen=True)
class BuildMeasurement:
    preparation_seconds: float
    build_seconds: float
    resource_usage: dict[str, float | int | None]


def peak_memory_bytes(resource_usage: dict[str, float | int | None]) -> int | None:
    value = resource_usage.get("peak_rss_bytes")
    return value if isinstance(value, int) else None


def source_fingerprint(path: Path) -> str:
    path = path.expanduser().resolve()
    files = (
        [path]
        if path.is_file()
        else sorted(file for file in path.rglob("*.parquet") if file.is_file())
    )
    digest = hashlib.sha256()
    digest.update(os.fspath(path).encode())
    for file in files:
        stat = file.stat()
        digest.update(
            os.fspath(file.relative_to(path) if path.is_dir() else file.name).encode()
        )
        digest.update(str(stat.st_size).encode())
        digest.update(str(stat.st_mtime_ns).encode())
    return digest.hexdigest()


def artifact_signature(
    implementation: str,
    source: ParquetSource,
    *,
    rows: int,
    dimension: int,
    nlist: int,
    version: str,
    encoding: str | None = None,
) -> dict[str, Any]:
    signature = {
        "schema_version": 2,
        "implementation": implementation,
        "implementation_version": version,
        "source": str(source.path),
        "source_fingerprint": source_fingerprint(source.path),
        "id_column": source.id_column,
        "vector_column": source.vector_column,
        "rows": rows,
        "dimension": dimension,
        "nlist": nlist,
        "kmeans_max_iterations": KMEANS_MAX_ITERATIONS,
        "kmeans_seed": KMEANS_SEED,
        "training_sampling": TRAINING_SAMPLING[implementation],
        "build_timing": "training-to-persistence-v1",
    }
    supported = RELIFY_ENCODINGS if implementation == "relify" else FAISS_ENCODINGS
    if encoding not in supported:
        raise ValueError(f"{implementation} artifact requires a supported encoding")
    if implementation == "relify":
        signature["postings_layout"] = RELIFY_POSTINGS_LAYOUT
    signature["encoding"] = encoding
    return signature


def artifact_directory(
    implementation: str,
    *,
    index_root: Path | None,
    work_root: Path | None,
    rebuild: bool,
) -> Any:
    if index_root is None:
        return tempfile.TemporaryDirectory(
            prefix=f"{implementation}-index-benchmark-",
            dir=work_root,
        )
    directory = index_root / implementation
    if rebuild and directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True, exist_ok=True)
    return nullcontext(directory)


def load_artifact(
    directory: Path,
    signature: dict[str, Any],
) -> dict[str, Any] | None:
    manifest_path = directory / "benchmark-artifact.json"
    if not manifest_path.exists():
        if any(directory.iterdir()):
            raise RuntimeError(
                f"incomplete benchmark artifact at {directory}; use --rebuild"
            )
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("signature") != signature:
        raise RuntimeError(
            f"benchmark artifact parameters do not match {directory}; use --rebuild"
        )
    build_result = manifest.get("build_result")
    if not isinstance(build_result, dict):
        raise RuntimeError(f"invalid benchmark artifact manifest at {directory}")
    result = dict(build_result)
    result["index_reused"] = True
    return result


def write_artifact(
    directory: Path,
    signature: dict[str, Any],
    build_result: dict[str, Any],
) -> None:
    manifest_path = directory / "benchmark-artifact.json"
    temporary_path = manifest_path.with_suffix(".json.tmp")
    payload = {
        "signature": signature,
        "build_result": build_result,
    }
    temporary_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary_path.replace(manifest_path)


def create_relify_index_with_progress(
    table: relify.SourceTable,
    *,
    id_column: str,
    vector_column: str,
    nlist: int,
    encoding: str,
    threads: int,
    show_progress: bool,
) -> BuildMeasurement:
    import relify

    bar = BuildProgressBar("Relify", enabled=show_progress)
    preparation_started = time.perf_counter()
    build_started: float | None = None
    build_resources: ResourceMonitor | None = None
    table.create_index(
        "benchmark_embedding",
        column=vector_column,
        key=[id_column],
        config=relify.IVF(nlist=nlist, encoding=encoding),
        builder=relify.Local(threads=threads),
    )
    deadline = time.monotonic() + timedelta(hours=24).total_seconds()
    try:
        while True:
            status = table.index_status("benchmark_embedding")
            if (
                build_started is None
                and status.phase not in UNTIMED_RELIFY_BUILD_PHASES
            ):
                build_started = time.perf_counter()
                build_resources = ResourceMonitor()
                build_resources.__enter__()
            if status.state == "ready":
                if build_started is None or build_resources is None:
                    build_started = time.perf_counter()
                    build_resources = ResourceMonitor()
                    build_resources.__enter__()
                build_seconds = time.perf_counter() - build_started
                build_resources.__exit__(None, None, None)
                bar.close(success=True)
                return BuildMeasurement(
                    preparation_seconds=build_started - preparation_started,
                    build_seconds=build_seconds,
                    resource_usage=build_resources.result().as_dict(),
                )
            bar.update(status)
            if status.state == "failed":
                table.wait_for_index(
                    "benchmark_embedding",
                    timeout=timedelta(seconds=1),
                )
                raise RuntimeError(status.error or "Relify index build failed")
            if time.monotonic() >= deadline:
                raise TimeoutError("timed out waiting for Relify index build")
            time.sleep(0.01)
    except BaseException:
        if build_resources is not None and build_resources.metrics is None:
            build_resources.__exit__(None, None, None)
        bar.close(success=False)
        raise


def write_source(path: Path, vectors: np.ndarray) -> None:
    vector_type = pa.list_(pa.field("element", pa.float32(), nullable=False))
    schema = pa.schema(
        [
            pa.field("id", pa.int64(), nullable=False),
            pa.field("embedding", vector_type, nullable=False),
        ]
    )
    offsets = pa.array(
        np.arange(
            0,
            (len(vectors) + 1) * vectors.shape[1],
            vectors.shape[1],
            dtype=np.int64,
        ),
        type=pa.int64(),
    )
    vector_array = pa.LargeListArray.from_arrays(
        offsets,
        pa.array(vectors.reshape(-1), type=pa.float32()),
    ).cast(vector_type)
    table = pa.Table.from_arrays(
        [
            pa.array(np.arange(len(vectors)), type=pa.int64()),
            vector_array,
        ],
        schema=schema,
    )
    pq.write_table(table, path, compression="NONE")


def add_source_vectors(
    index: Any,
    source: ParquetSource,
    *,
    progress: CounterProgressBar | None = None,
) -> None:
    added = 0
    for batch_ids, batch_vectors in iter_parquet_vector_batches(source):
        index.add_with_ids(batch_vectors, batch_ids)
        added += len(batch_vectors)
        if progress is not None:
            progress.update(added, source.rows, "Parquet vectors")
    if added != source.rows or index.ntotal != source.rows:
        raise ValueError("source row count changed while populating the Faiss index")


def benchmark_relify(
    source: ParquetSource,
    queries: np.ndarray | None,
    expected: np.ndarray | None,
    *,
    rows: int,
    dimension: int,
    nlist: int,
    encoding: str,
    k_values: tuple[int, ...],
    nprobe_values: tuple[int, ...],
    search_repetitions: int,
    warmup_queries: int,
    measure_search: bool,
    threads: int,
    work_root: Path | None,
    index_root: Path | None,
    rebuild: bool,
    build_missing: bool,
    show_progress: bool,
    page_cache_capacity_bytes: int | None,
) -> dict[str, Any]:
    import relify

    signature = artifact_signature(
        "relify",
        source,
        rows=rows,
        dimension=dimension,
        nlist=nlist,
        version=importlib.metadata.version("relify"),
        encoding=encoding,
    )
    with artifact_directory(
        "relify",
        index_root=index_root,
        work_root=work_root,
        rebuild=rebuild,
    ) as directory:
        root = Path(directory)
        result = load_artifact(root, signature)
        if result is None:
            if not build_missing:
                raise RuntimeError(
                    f"Relify benchmark artifact is missing at {root}; "
                    "run python -m benchmarks.build first"
                )
            session = relify.connect(root / "relify-data")
            configure_relify_session(session, threads)
            session.register_parquet("benchmark", source.path)
            table = session.table("benchmark")
            assert isinstance(table, relify.SourceTable)

            measurement = create_relify_index_with_progress(
                table,
                id_column=source.id_column,
                vector_column=source.vector_column,
                nlist=nlist,
                encoding=encoding,
                threads=threads,
                show_progress=show_progress,
            )

            result = {
                "implementation": "relify",
                "workload": (
                    "IVF training, assignment, Parquet persistence, and catalog "
                    "publication; preparation excluded"
                ),
                "training_rows": min(rows, nlist * 256),
                "training_sampling": TRAINING_SAMPLING["relify"],
                "postings_layout": RELIFY_POSTINGS_LAYOUT,
                "encoding": encoding,
                "kmeans_max_iterations": KMEANS_MAX_ITERATIONS,
                "kmeans_seed": KMEANS_SEED,
                "preparation_seconds": measurement.preparation_seconds,
                "build_seconds": measurement.build_seconds,
                "build_points_per_second": rows / measurement.build_seconds,
                "managed_bytes": directory_bytes(session.root / "indexes"),
                "peak_memory_bytes": peak_memory_bytes(measurement.resource_usage),
                "build_resource_usage": measurement.resource_usage,
                "threads": threads,
                "index_reused": False,
            }
            write_artifact(root, signature, result)
        if measure_search:
            if queries is None or expected is None:
                raise ValueError("queries and ground truth are required for search")
            query_config = relify.SessionConfig()
            if page_cache_capacity_bytes is not None:
                query_config.set(
                    "relify.parquet.page_cache.capacity",
                    str(page_cache_capacity_bytes),
                )
            query_session = relify.connect(root / "relify-data", config=query_config)
            configure_relify_session(query_session, threads)
            query_table = query_session.table("benchmark")
            assert isinstance(query_table, relify.SourceTable)
            result["cache_kind"] = "bounded decompressed Parquet page cache"
            result["search_curve"] = measure_search_curve(
                relify_search(
                    query_session,
                    query_table,
                    id_column=source.id_column,
                    vector_column=source.vector_column,
                ),
                queries,
                expected,
                k_values=k_values,
                nprobe_values=nprobe_values,
                repetitions=search_repetitions,
                warmup_queries=warmup_queries,
                progress=CounterProgressBar("Relify query", enabled=show_progress),
            )
            result["page_cache"] = asdict(query_session.parquet_page_cache_stats())
        return result


def faiss_module() -> Any | None:
    try:
        import faiss  # pyright: ignore[reportMissingImports]
    except ImportError:
        return None
    return faiss


def create_faiss_index(
    faiss: Any,
    quantizer: Any,
    *,
    dimension: int,
    nlist: int,
    encoding: str,
) -> Any:
    if encoding == "flat":
        return faiss.IndexIVFFlat(quantizer, dimension, nlist)
    quantizer_type = {
        "sq8": faiss.ScalarQuantizer.QT_8bit,
        "sq4": faiss.ScalarQuantizer.QT_4bit,
    }[encoding]
    return faiss.IndexIVFScalarQuantizer(
        quantizer,
        dimension,
        nlist,
        quantizer_type,
        faiss.METRIC_L2,
    )


def benchmark_faiss(
    source: ParquetSource,
    queries: np.ndarray | None,
    expected: np.ndarray | None,
    *,
    rows: int,
    dimension: int,
    nlist: int,
    encoding: str,
    k_values: tuple[int, ...],
    nprobe_values: tuple[int, ...],
    search_repetitions: int,
    warmup_queries: int,
    measure_search: bool,
    threads: int,
    work_root: Path | None,
    index_root: Path | None,
    rebuild: bool,
    build_missing: bool,
    show_progress: bool,
) -> dict[str, Any] | None:
    faiss = faiss_module()
    if faiss is None:
        return None
    faiss.omp_set_num_threads(threads)

    version = getattr(faiss, "__version__", "unknown")
    signature = artifact_signature(
        "faiss",
        source,
        rows=rows,
        dimension=dimension,
        nlist=nlist,
        version=version,
        encoding=encoding,
    )
    with artifact_directory(
        "faiss",
        index_root=index_root,
        work_root=work_root,
        rebuild=rebuild,
    ) as directory:
        root = Path(directory)
        index_path = root / "benchmark.faiss"
        result = load_artifact(root, signature)
        if result is None:
            if not build_missing:
                raise RuntimeError(
                    f"Faiss benchmark artifact is missing at {root}; "
                    "run python -m benchmarks.tools.faiss build first"
                )
            preparation_started = time.perf_counter()
            training_size = min(rows, nlist * 256)
            sample_progress = CounterProgressBar("Faiss sample", enabled=show_progress)
            sample_succeeded = False
            try:
                training = sample_parquet_vectors(
                    source,
                    training_size,
                    seed=KMEANS_SEED,
                    progress=sample_progress.update,
                )
                sample_succeeded = True
            finally:
                sample_progress.close(success=sample_succeeded)
            if show_progress:
                print("Faiss: preparing index", file=sys.stderr, flush=True)
            gc.collect()
            pa.default_memory_pool().release_unused()
            quantizer = faiss.IndexFlatL2(dimension)
            index = create_faiss_index(
                faiss,
                quantizer,
                dimension=dimension,
                nlist=nlist,
                encoding=encoding,
            )
            index.parallel_mode = FAISS_PARALLEL_MODE
            index.cp.niter = KMEANS_MAX_ITERATIONS
            index.cp.seed = KMEANS_SEED
            preparation_seconds = time.perf_counter() - preparation_started

            with ResourceMonitor() as build_resources:
                if show_progress:
                    print("Faiss: training centroids", file=sys.stderr, flush=True)
                started = time.perf_counter()
                index.train(training)
                del training
                if show_progress:
                    print("Faiss: populating index", file=sys.stderr, flush=True)
                add_progress = CounterProgressBar("Faiss add", enabled=show_progress)
                add_succeeded = False
                try:
                    add_source_vectors(index, source, progress=add_progress)
                    add_succeeded = True
                finally:
                    add_progress.close(success=add_succeeded)
                if show_progress:
                    print("Faiss: writing index", file=sys.stderr, flush=True)
                faiss.write_index(index, str(index_path))
                sync_file(index_path)
                build_seconds = time.perf_counter() - started

            result = {
                "implementation": "faiss",
                "version": version,
                "encoding": encoding,
                "workload": (
                    f"Faiss IVF-{encoding.upper()} training, "
                    "population, and write_index "
                    "persistence; preparation excluded"
                ),
                "training_rows": training_size,
                "training_sampling": TRAINING_SAMPLING["faiss"],
                "kmeans_max_iterations": KMEANS_MAX_ITERATIONS,
                "kmeans_seed": KMEANS_SEED,
                "omp_threads": faiss.omp_get_max_threads(),
                "parallel_mode": FAISS_PARALLEL_MODE,
                "preparation_seconds": preparation_seconds,
                "build_seconds": build_seconds,
                "build_points_per_second": rows / build_seconds,
                "managed_bytes": index_path.stat().st_size,
                "build_resource_usage": build_resources.result().as_dict(),
                "index_reused": False,
            }
            result["peak_memory_bytes"] = peak_memory_bytes(
                result["build_resource_usage"]
            )
            write_artifact(root, signature, result)
            del index, quantizer
        elif not index_path.is_file():
            raise RuntimeError(f"Faiss artifact is missing {index_path}")
        if measure_search:
            if queries is None or expected is None:
                raise ValueError("queries and ground truth are required for search")
            load_started = time.perf_counter()
            persisted_index = faiss.read_index(str(index_path))
            persisted_index.parallel_mode = FAISS_PARALLEL_MODE
            result["index_load_seconds"] = time.perf_counter() - load_started
            result["index_resident_bytes"] = index_path.stat().st_size
            result["cache_kind"] = "Faiss in-process index"
            result["search_curve"] = measure_search_curve(
                faiss_search(persisted_index),
                queries,
                expected,
                k_values=k_values,
                nprobe_values=nprobe_values,
                repetitions=search_repetitions,
                warmup_queries=warmup_queries,
                progress=CounterProgressBar("Faiss query", enabled=show_progress),
            )
        return result


def curve_point(
    result: dict[str, Any],
    *,
    nprobe: int,
    k: int,
) -> dict[str, Any]:
    return next(
        point
        for point in result["search_curve"]
        if point["nprobe"] == nprobe and point["k"] == k
    )


def aggregate_build_trials(
    trials: list[dict[str, Any]],
    *,
    points: int,
) -> dict[str, Any]:
    if not trials:
        raise ValueError("at least one benchmark trial is required")
    result = dict(trials[0])
    preparation_samples = [trial["preparation_seconds"] for trial in trials]
    build_samples = [trial["build_seconds"] for trial in trials]
    managed_samples = [trial["managed_bytes"] for trial in trials]
    peak_memory_samples = [trial["peak_memory_bytes"] for trial in trials]
    measured_peaks = [value for value in peak_memory_samples if isinstance(value, int)]
    build_seconds = statistics.median(build_samples)
    result.update(
        {
            "preparation_seconds": statistics.median(preparation_samples),
            "preparation_seconds_samples": preparation_samples,
            "build_seconds": build_seconds,
            "build_seconds_samples": build_samples,
            "build_points_per_second": points / build_seconds,
            "managed_bytes": int(statistics.median(managed_samples)),
            "managed_bytes_samples": managed_samples,
            "peak_memory_bytes": (
                int(statistics.median(measured_peaks))
                if len(measured_peaks) == len(peak_memory_samples)
                else None
            ),
            "peak_memory_bytes_samples": peak_memory_samples,
            "build_resource_usage_samples": [
                trial["build_resource_usage"] for trial in trials
            ],
        }
    )
    return result


def summarize_query(
    result: dict[str, Any],
    *,
    headline_nprobe: int,
    headline_k: int,
) -> dict[str, Any]:
    headline = curve_point(result, nprobe=headline_nprobe, k=headline_k)
    summary = {
        key: value
        for key, value in result.items()
        if key
        not in {
            "build_seconds",
            "build_points_per_second",
            "build_resource_usage",
            "index_reused",
            "kmeans_max_iterations",
            "kmeans_seed",
            "training_rows",
            "workload",
        }
    }
    summary["index_bytes"] = summary.pop("managed_bytes")
    summary.update(
        {
            "recall_at_k": headline["recall_at_k"],
            "query_latency_ms_p50": headline["latency_ms_p50"],
            "query_latency_ms_p95": headline["latency_ms_p95"],
        }
    )
    return summary


def validate_args(
    args: argparse.Namespace,
    *,
    rows: int,
    dimension: int,
    threads: int,
    k_values: tuple[int, ...],
    nprobe_values: tuple[int, ...],
) -> None:
    if rows <= 0 or dimension <= 0 or args.repetitions <= 0 or threads <= 0:
        raise ValueError("rows, dimension, threads, and repetitions must be positive")
    if not 1 <= args.nlist <= rows:
        raise ValueError("nlist must be in 1..=rows")
    if args.operation == "build":
        return
    if args.num_queries <= 0 or args.search_repetitions <= 0 or args.warmup_queries < 0:
        raise ValueError(
            "num-queries and search-repetitions must be positive; warmup-queries "
            "must be non-negative"
        )
    if (
        args.page_cache_capacity_bytes is not None
        and args.page_cache_capacity_bytes < 0
    ):
        raise ValueError("page-cache-capacity-bytes must be non-negative")
    if args.query_source_start is not None and args.query_source_start < 0:
        raise ValueError("query-source-start must be non-negative")
    if max(nprobe_values) > args.nlist:
        raise ValueError("nprobe values must be in 1..=nlist")
    if max(k_values) > rows:
        raise ValueError("k values must be in 1..=rows")


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    rng = np.random.default_rng(args.seed)
    k_values = tuple(
        sorted(
            {
                args.k,
                *parse_positive_ints(args.curve_k_values, name="curve-k-values"),
            }
        )
    )
    nprobe_values = tuple(
        sorted(
            {
                args.nprobe,
                *parse_positive_ints(
                    args.curve_nprobe_values,
                    name="curve-nprobe-values",
                ),
            }
        )
    )
    threads = args.threads or effective_cpu_count()
    available_cpus = affinity_cpu_ids()
    cgroup = cgroup_environment()
    if threads > len(available_cpus):
        raise ValueError(
            f"threads must not exceed the {len(available_cpus)} CPUs visible "
            "to this process"
        )
    work_root = args.work_root.expanduser().resolve() if args.work_root else None
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)
    index_root = (
        args.index_root.expanduser().resolve()
        if args.index_root is not None
        else (work_root / "indexes" if work_root is not None else None)
    )
    if index_root is not None:
        if args.repetitions != 1:
            raise ValueError("persistent indexes require --repetitions 1")
        index_root.mkdir(parents=True, exist_ok=True)
    elif args.rebuild:
        raise ValueError("--rebuild requires --index-root or --work-root")

    if args.operation == "query" and index_root is None:
        raise ValueError("query requires --index-root or --work-root")

    if args.source_parquet is not None:
        source_info = inspect_parquet_source(
            args.source_parquet,
            id_column=args.id_column,
            vector_column=args.vector_column,
        )
        rows = source_info.rows
        dimension = source_info.dimension
        if args.operation == "query":
            assert args.ground_truth is not None
            if args.query_file is not None:
                queries = load_vectors(args.query_file)[: args.num_queries]
                query_source_ids = None
            else:
                assert args.query_source_start is not None
                query_source_ids = np.arange(
                    args.query_source_start,
                    args.query_source_start + args.num_queries,
                    dtype=np.int64,
                )
                queries = load_parquet_vectors_by_id(source_info, query_source_ids)
            expected = load_ground_truth(
                args.ground_truth,
                queries=len(queries),
                k=max(k_values),
            )
            if expected.shape != (len(queries), max(k_values)):
                raise ValueError("ground truth does not cover every benchmark query")
        else:
            queries = None
            expected = None
            query_source_ids = None
        source_context = nullcontext(source_info)
        dataset_kind = args.dataset_name or "parquet"
        source_bytes = source_info.bytes
    else:
        if args.operation == "query":
            raise ValueError("query requires --source-parquet")
        if args.ground_truth is not None:
            raise ValueError("--ground-truth requires --source-parquet")
        if args.base is None:
            base = rng.standard_normal(
                (args.rows, args.dimension),
                dtype=np.float32,
            )
            dataset_kind = "synthetic"
        else:
            base = load_vectors(args.base)
            dataset_kind = str(args.base)
        rows, dimension = base.shape
        queries = None
        expected = None
        query_source_ids = None
        source_context = tempfile.TemporaryDirectory(
            prefix="relify-source-benchmark-",
            dir=work_root,
        )
        source_bytes = 0

    if queries is not None:
        if len(queries) == 0:
            raise ValueError("at least one query is required")
        if len(queries) != args.num_queries:
            raise ValueError("query input does not contain --num-queries rows")
        if dimension != queries.shape[1]:
            raise ValueError("base and query dimensions differ")
    validate_args(
        args,
        rows=rows,
        dimension=dimension,
        threads=threads,
        k_values=k_values,
        nprobe_values=nprobe_values,
    )

    with source_context as source_value:
        if args.source_parquet is None:
            source_path = Path(source_value) / "source.parquet"
            write_source(source_path, base)
            source = inspect_parquet_source(source_path)
            source_bytes = source.bytes
        else:
            assert isinstance(source_value, ParquetSource)
            source = source_value
        if args.implementation == "relify":
            relify_trials = [
                benchmark_relify(
                    source,
                    queries,
                    expected,
                    rows=rows,
                    dimension=dimension,
                    nlist=args.nlist,
                    encoding=args.encoding,
                    k_values=k_values,
                    nprobe_values=nprobe_values,
                    search_repetitions=args.search_repetitions,
                    warmup_queries=args.warmup_queries,
                    measure_search=args.operation == "query",
                    threads=threads,
                    work_root=work_root,
                    index_root=index_root,
                    rebuild=args.rebuild,
                    build_missing=args.operation == "build",
                    show_progress=not args.no_progress,
                    page_cache_capacity_bytes=args.page_cache_capacity_bytes,
                )
                for trial in range(args.repetitions)
            ]
            if args.operation == "build":
                implementation_result = aggregate_build_trials(
                    relify_trials,
                    points=rows,
                )
            else:
                implementation_result = summarize_query(
                    relify_trials[0],
                    headline_nprobe=args.nprobe,
                    headline_k=args.k,
                )
        else:
            faiss_trials = [
                benchmark_faiss(
                    source,
                    queries,
                    expected,
                    rows=rows,
                    dimension=dimension,
                    nlist=args.nlist,
                    encoding=args.encoding,
                    k_values=k_values,
                    nprobe_values=nprobe_values,
                    search_repetitions=args.search_repetitions,
                    warmup_queries=args.warmup_queries,
                    measure_search=args.operation == "query",
                    threads=threads,
                    work_root=work_root,
                    index_root=index_root,
                    rebuild=args.rebuild,
                    build_missing=args.operation == "build",
                    show_progress=not args.no_progress,
                )
                for trial in range(args.repetitions)
            ]
            if faiss_trials[0] is None:
                raise RuntimeError("faiss-cpu is required for the Faiss benchmark")
            completed_trials = [trial for trial in faiss_trials if trial is not None]
            if args.operation == "build":
                implementation_result = aggregate_build_trials(
                    completed_trials,
                    points=rows,
                )
            else:
                implementation_result = summarize_query(
                    completed_trials[0],
                    headline_nprobe=args.nprobe,
                    headline_k=args.k,
                )

        implementation_version = implementation_result.get("version")
        if implementation_version is None:
            implementation_version = importlib.metadata.version("relify")
        result: dict[str, Any] = {
            "schema_version": 1,
            "generated_at_utc": datetime.now(UTC).isoformat(),
            "benchmark_revision": benchmark_revision(),
            "implementation_revision": os.environ.get(
                "BENCHMARK_IMPLEMENTATION_REVISION",
                implementation_version,
            ),
            "benchmark": args.operation,
            "dataset": {
                "kind": dataset_kind,
                "revision": args.dataset_revision,
                "split": args.dataset_split,
                "rows": rows,
                "dimension": dimension,
                "queries": len(queries) if queries is not None else None,
                "source_parquet": str(source.path),
                "source_parquet_bytes": source_bytes,
                "id_column": source.id_column,
                "vector_column": source.vector_column,
                "query_file": (
                    str(args.query_file.expanduser().resolve())
                    if args.query_file
                    else None
                ),
                "query_source_id_range": (
                    [int(query_source_ids[0]), int(query_source_ids[-1])]
                    if query_source_ids is not None
                    else None
                ),
                "ground_truth": (
                    str(args.ground_truth.expanduser().resolve())
                    if args.ground_truth
                    else None
                ),
            },
            "parameters": {
                "implementation": args.implementation,
                "index_root": str(index_root) if index_root is not None else None,
                "seed": args.seed,
                "nlist": args.nlist,
                "encoding": args.encoding,
                "repetitions": args.repetitions,
                "threads": threads,
            },
            "resources": {
                "cpus": len(available_cpus),
                "memory_limit_bytes": cgroup["memory_max_bytes"],
            },
            "software": {
                "platform": platform.platform(),
                "machine": platform.machine(),
                "python": platform.python_version(),
                "numpy": np.__version__,
                "pyarrow": pa.__version__,
                "rustc": command_version("rustc", "--version"),
                args.implementation: implementation_version,
            },
            "results": [implementation_result],
        }
        if args.operation == "build":
            result["parameters"]["rebuild"] = args.rebuild
        else:
            result["parameters"].update(
                {
                    "nprobe": args.nprobe,
                    "k": args.k,
                    "curve_nprobe_values": list(nprobe_values),
                    "curve_k_values": list(k_values),
                    "search_repetitions": args.search_repetitions,
                    "warmup_queries": args.warmup_queries,
                    "page_cache_capacity_bytes": args.page_cache_capacity_bytes,
                    "query_mode": "one-query-at-a-time with intra-query parallelism",
                    "point_order_seed": KMEANS_SEED,
                }
            )
        return result


def add_common_arguments(
    command: argparse.ArgumentParser,
    *,
    implementation: str,
) -> None:
    command.add_argument("--id-column", default="id")
    command.add_argument("--vector-column", default="embedding")
    command.add_argument("--dataset-name")
    command.add_argument("--dataset-revision")
    command.add_argument("--dataset-split")
    command.add_argument("--nlist", type=int, default=256)
    encodings = RELIFY_ENCODINGS if implementation == "relify" else FAISS_ENCODINGS
    command.add_argument(
        "--encoding",
        choices=encodings,
        default=encodings[0],
    )
    command.add_argument("--threads", type=int)
    command.add_argument("--work-root", type=Path)
    command.add_argument("--index-root", type=Path)
    command.add_argument("--output", type=Path)
    command.add_argument("--no-progress", action="store_true")


def build_parser(implementation: str = "relify") -> argparse.ArgumentParser:
    program = (
        "python -m benchmarks.build"
        if implementation == "relify"
        else "python -m benchmarks.tools.faiss build"
    )
    command = argparse.ArgumentParser(
        prog=program,
        description=f"Build a persisted {implementation} IVF benchmark index.",
    )
    source = command.add_mutually_exclusive_group()
    source.add_argument("--base", type=Path)
    source.add_argument("--source-parquet", type=Path)
    command.add_argument("--rows", type=int, default=100_000)
    command.add_argument("--dimension", type=int, default=128)
    command.add_argument("--seed", type=int, default=KMEANS_SEED)
    command.add_argument("--repetitions", type=int, default=1)
    command.add_argument("--rebuild", action="store_true")
    add_common_arguments(command, implementation=implementation)
    command.set_defaults(
        implementation=implementation,
        operation="build",
        query_file=None,
        query_source_start=None,
        ground_truth=None,
        num_queries=0,
        nprobe=1,
        k=1,
        curve_nprobe_values="1",
        curve_k_values="1",
        search_repetitions=1,
        warmup_queries=1,
        page_cache_capacity_bytes=None,
    )
    return command


def query_parser(implementation: str = "relify") -> argparse.ArgumentParser:
    program = (
        "python -m benchmarks.query"
        if implementation == "relify"
        else "python -m benchmarks.tools.faiss query"
    )
    command = argparse.ArgumentParser(
        prog=program,
        description=f"Query a persisted {implementation} IVF benchmark index.",
    )
    command.add_argument("--source-parquet", type=Path, required=True)
    queries = command.add_mutually_exclusive_group(required=True)
    queries.add_argument("--query-file", type=Path)
    queries.add_argument("--query-source-start", type=int)
    command.add_argument("--ground-truth", type=Path, required=True)
    command.add_argument("--num-queries", type=int, default=100)
    command.add_argument("--nprobe", type=int, default=16)
    command.add_argument("--k", type=int, default=10)
    command.add_argument(
        "--curve-nprobe-values",
        default="1,2,4,8,16,32,64,128,256",
    )
    command.add_argument("--curve-k-values", default="100,1000,10000")
    command.add_argument("--search-repetitions", type=int, default=3)
    command.add_argument("--warmup-queries", type=int, default=5)
    if implementation == "relify":
        command.add_argument("--page-cache-capacity-bytes", type=int)
    else:
        command.set_defaults(page_cache_capacity_bytes=None)
    add_common_arguments(command, implementation=implementation)
    command.set_defaults(
        implementation=implementation,
        operation="query",
        base=None,
        rows=0,
        dimension=0,
        seed=KMEANS_SEED,
        repetitions=1,
        rebuild=False,
    )
    return command


def main(
    operation: str,
    implementation: str = "relify",
    argv: list[str] | None = None,
) -> None:
    if operation == "build":
        args = build_parser(implementation).parse_args(argv)
    elif operation == "query":
        args = query_parser(implementation).parse_args(argv)
    else:
        raise ValueError(f"unknown benchmark operation: {operation}")
    result = benchmark(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")
