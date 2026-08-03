from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import relify

from benchmarks.tools.datasets import load_ground_truth
from benchmarks.tools.harness import (
    FAISS_PARALLEL_MODE,
    KMEANS_SEED,
    command_version,
    directory_bytes,
    evict_file,
    evict_tree,
    load_vectors,
    measure_search_curve,
    parse_positive_ints,
    source_revision,
)
from benchmarks.tools.resources import (
    affinity_cpu_ids,
    cgroup_environment,
    cgroup_snapshot,
)
from benchmarks.tools.search_adapters import (
    configure_relify_session,
    faiss_search,
    relify_search,
)

GIB = 1024**3


def _load_metadata(root: Path) -> tuple[Path, dict[str, Any]]:
    root = root.expanduser().resolve()
    path = root / "storage-indexes.json"
    try:
        metadata = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read storage preparation metadata: {path}") from error
    if metadata.get("schema_version") != 1:
        raise ValueError("unsupported storage preparation metadata version")
    return root, metadata


def _read_artifact(path: Path, implementation: str) -> dict[str, Any]:
    try:
        artifact = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"cannot read benchmark artifact: {path}") from error
    signature = artifact.get("signature")
    if not isinstance(signature, dict) or signature.get("schema_version") != 2:
        raise ValueError(f"unsupported benchmark artifact: {path}")
    if signature.get("implementation") != implementation:
        raise ValueError(f"benchmark artifact implementation mismatch: {path}")
    return artifact


def _load_index_root(root: Path) -> tuple[Path, dict[str, Any]]:
    root = root.expanduser().resolve()
    artifacts = {
        implementation: _read_artifact(
            root / implementation / "benchmark-artifact.json",
            implementation,
        )
        for implementation in ("relify", "faiss")
        if (root / implementation / "benchmark-artifact.json").is_file()
    }
    if not artifacts:
        raise ValueError(f"index root contains no benchmark artifacts: {root}")

    signatures = [artifact["signature"] for artifact in artifacts.values()]
    identity_fields = (
        "source",
        "source_fingerprint",
        "rows",
        "dimension",
        "id_column",
        "vector_column",
        "nlist",
    )
    expected = signatures[0]
    if any(
        signature.get(field) != expected.get(field)
        for signature in signatures[1:]
        for field in identity_fields
    ):
        raise ValueError("benchmark artifacts do not describe the same index workload")

    indexes: dict[str, dict[str, Any]] = {}
    if "relify" in artifacts:
        warehouse = root / "relify" / "relify-data"
        payload = warehouse / "indexes"
        if not payload.is_dir():
            raise ValueError(f"Relify benchmark index is missing: {payload}")
        indexes["relify"] = {
            "root": str(warehouse),
            "table": "benchmark",
            "index": "benchmark_embedding",
            "id_column": expected["id_column"],
            "vector_column": expected["vector_column"],
            "payload_bytes": directory_bytes(payload),
            "store_vectors": True,
            "artifact": str(root / "relify" / "benchmark-artifact.json"),
        }
    if "faiss" in artifacts:
        index = root / "faiss" / "benchmark.faiss"
        if not index.is_file():
            raise ValueError(f"Faiss benchmark index is missing: {index}")
        indexes["faiss"] = {
            "index": str(index),
            "payload_bytes": index.stat().st_size,
            "mmap_embedded_inverted_lists": True,
            "artifact": str(root / "faiss" / "benchmark-artifact.json"),
        }

    return root, {
        "schema_version": 1,
        "source": {
            "path": expected["source"],
            "rows": expected["rows"],
            "dimension": expected["dimension"],
            "id_column": expected["id_column"],
            "vector_column": expected["vector_column"],
        },
        "parameters": {
            "nlist": expected["nlist"],
            "kmeans_seed": expected["kmeans_seed"],
            "kmeans_max_iterations": expected["kmeans_max_iterations"],
        },
        "indexes": indexes,
    }


def _validate_resources(
    args: argparse.Namespace,
    implementation: dict[str, Any],
) -> dict[str, str | int | None]:
    cpus = affinity_cpu_ids()
    environment = cgroup_environment()
    if args.allow_resource_mismatch:
        return environment
    if len(cpus) != args.threads:
        raise RuntimeError(
            f"constrained query requires {args.threads} visible CPUs; found {len(cpus)}"
        )
    if environment["memory_max_bytes"] != args.memory_limit_bytes:
        raise RuntimeError(
            f"Suite B requires memory.max={args.memory_limit_bytes}; "
            f"found {environment['memory_max_bytes']}"
        )
    if environment["memory_swap_max_bytes"] != 0:
        raise RuntimeError(
            "Suite B requires memory.swap.max=0; "
            f"found {environment['memory_swap_max_bytes']}"
        )
    if cgroup_snapshot().swap_current_bytes != 0:
        raise RuntimeError("Suite B cgroup already has non-zero swap usage")
    if implementation["payload_bytes"] <= args.minimum_payload_bytes:
        raise RuntimeError(
            f"index payload must exceed {args.minimum_payload_bytes} bytes"
        )
    return environment


def _evict_index(implementation: str, index: dict[str, Any]) -> list[str]:
    if implementation == "relify":
        payload = Path(index["root"]) / "indexes"
        evict_tree(payload)
        return [str(payload)]
    paths = [Path(index["index"])]
    if inverted_lists := index.get("inverted_lists"):
        paths.append(Path(inverted_lists))
    for path in paths:
        evict_file(path)
    return [str(path) for path in paths]


def _open_relify(
    implementation: dict[str, Any],
    *,
    threads: int,
) -> tuple[Any, dict[str, Any]]:
    session = relify.connect(Path(implementation["root"]))
    configure_relify_session(session, threads)
    table = session.table(implementation["table"])
    if not isinstance(table, relify.SourceTable):
        raise TypeError("prepared Relify table is not a source table")
    return relify_search(
        session,
        table,
        id_column=implementation.get("id_column", "id"),
        vector_column=implementation.get("vector_column", "embedding"),
    ), {
        "version": importlib.metadata.version("relify"),
        "storage": "Parquet IVF queried without cache_index",
    }


def _open_faiss(
    implementation: dict[str, Any],
    *,
    threads: int,
) -> tuple[Any, dict[str, Any]]:
    import faiss  # pyright: ignore[reportMissingImports]

    faiss.omp_set_num_threads(threads)
    if implementation.get("mmap_embedded_inverted_lists"):
        flags = faiss.IO_FLAG_MMAP | faiss.IO_FLAG_READ_ONLY
        index = faiss.read_index(implementation["index"], flags)
        read_only = True
    else:
        index = faiss.read_index(implementation["index"])
        read_only = False
    index.parallel_mode = FAISS_PARALLEL_MODE
    inverted_lists = faiss.downcast_InvertedLists(
        faiss.extract_index_ivf(index).invlists
    )
    if type(inverted_lists).__name__ != "OnDiskInvertedLists":
        raise RuntimeError("prepared Faiss index does not use OnDiskInvertedLists")
    actual_filename = inverted_lists.filename
    if expected_filename := implementation.get("inverted_lists"):
        expected_path = Path(expected_filename).resolve()
        actual_path = Path(actual_filename).resolve()
        if actual_path != expected_path:
            raise RuntimeError("Faiss on-disk payload path does not match metadata")
    return faiss_search(index), {
        "version": getattr(faiss, "__version__", "unknown"),
        "storage": type(inverted_lists).__name__,
        "index": str(Path(implementation["index"]).resolve()),
        "inverted_lists": actual_filename or "embedded mmap",
        "read_only": read_only,
        "omp_threads": faiss.omp_get_max_threads(),
    }


def benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if (
        args.num_queries <= 0
        or args.search_repetitions <= 0
        or args.warmup_queries <= 0
        or args.threads <= 0
        or args.memory_limit_bytes <= 0
        or args.minimum_payload_bytes < 0
    ):
        raise ValueError(
            "num-queries, search-repetitions, warmup-queries, threads, and "
            "memory-limit-bytes must be positive; minimum-payload-bytes must "
            "be non-negative"
        )
    if args.index_root is not None:
        root, metadata = _load_index_root(args.index_root)
        input_kind = "benchmark-index-root"
    else:
        root, metadata = _load_metadata(args.prepared)
        input_kind = "prepared-storage-indexes"
    if args.implementation not in metadata["indexes"]:
        raise ValueError(f"{args.implementation} has not been prepared")
    implementation = metadata["indexes"][args.implementation]
    environment = _validate_resources(args, implementation)

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
    rows = int(metadata["source"]["rows"])
    dimension = int(metadata["source"]["dimension"])
    nlist = int(metadata["parameters"]["nlist"])
    if max(k_values) > rows:
        raise ValueError(f"K must not exceed the {rows} source rows")
    if max(nprobe_values) > nlist:
        raise ValueError(f"nprobe must not exceed nlist={nlist}")
    queries = load_vectors(args.query_file)[: args.num_queries]
    if len(queries) != args.num_queries:
        raise ValueError("query file does not contain --num-queries rows")
    if queries.shape[1] != dimension:
        raise ValueError("query dimension does not match the prepared source")
    expected = load_ground_truth(
        args.ground_truth,
        queries=len(queries),
        k=max(k_values),
    )
    if expected.shape != (len(queries), max(k_values)):
        raise ValueError("ground truth does not cover the query workload")
    if np.any(expected >= rows):
        raise ValueError("ground truth contains an ID outside the source table")

    evicted_paths = _evict_index(args.implementation, implementation)
    opened_at = time.perf_counter()
    if args.implementation == "relify":
        search, runtime = _open_relify(implementation, threads=args.threads)
    else:
        search, runtime = _open_faiss(implementation, threads=args.threads)
    open_seconds = time.perf_counter() - opened_at
    curve = measure_search_curve(
        search,
        queries,
        expected,
        k_values=k_values,
        nprobe_values=nprobe_values,
        repetitions=args.search_repetitions,
        warmup_queries=args.warmup_queries,
    )

    cgroup_reads_available = all(
        point["resource_usage"]["cgroup_read_bytes"] is not None for point in curve
    )
    physical_read_counter = "cgroup" if cgroup_reads_available else "process"
    physical_read_bytes = sum(
        point["resource_usage"][
            "cgroup_read_bytes" if cgroup_reads_available else "read_bytes"
        ]
        or 0
        for point in curve
    )
    if not args.allow_resource_mismatch:
        if any(
            point["resource_usage"]["cgroup_swap_current_bytes"] != 0 for point in curve
        ):
            raise RuntimeError("Suite B used swap during a measured query point")
        if physical_read_bytes == 0 and not args.allow_zero_physical_reads:
            raise RuntimeError("Suite B measured no physical reads")

    return {
        "schema_version": 1,
        "suite": "storage-backed-ivf-flat",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "source_revision": source_revision(),
        "prepared_root": str(root),
        "index_input": input_kind,
        "dataset": {
            **metadata["source"],
            "queries": len(queries),
            "query_file": str(args.query_file.expanduser().resolve()),
            "ground_truth": str(args.ground_truth.expanduser().resolve()),
        },
        "parameters": {
            "nlist": nlist,
            "nprobe": args.nprobe,
            "k": args.k,
            "curve_nprobe_values": list(nprobe_values),
            "curve_k_values": list(k_values),
            "search_repetitions": args.search_repetitions,
            "warmup_queries": args.warmup_queries,
            "point_order_seed": KMEANS_SEED,
            "threads": args.threads,
            "memory_limit_bytes": args.memory_limit_bytes,
            "minimum_payload_bytes": args.minimum_payload_bytes,
            "allow_resource_mismatch": args.allow_resource_mismatch,
            "allow_zero_physical_reads": args.allow_zero_physical_reads,
        },
        "resources": {
            "cpus": len(affinity_cpu_ids()),
            "memory_limit_bytes": environment["memory_max_bytes"],
            "physical_read_counter": physical_read_counter,
            "physical_read_bytes": physical_read_bytes,
        },
        "software": {
            "platform": platform.platform(),
            "machine": platform.machine(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pyarrow": pa.__version__,
            "rustc": command_version("rustc", "--version"),
        },
        "result": {
            "implementation": args.implementation,
            "index": {
                **implementation,
                "page_cache_evicted_before_query": True,
                "evicted_paths": evicted_paths,
            },
            "runtime": runtime,
            "index_open_seconds": open_seconds,
            "search_curve": curve,
        },
    }


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="python -m benchmarks.tools.storage_query",
        description="Query a prepared storage-backed IVF index.",
    )
    command.add_argument(
        "--implementation",
        choices=("relify", "faiss"),
        required=True,
    )
    index = command.add_mutually_exclusive_group(required=True)
    index.add_argument("--prepared", type=Path)
    index.add_argument("--index-root", type=Path)
    command.add_argument("--query-file", type=Path, required=True)
    command.add_argument("--ground-truth", type=Path, required=True)
    command.add_argument("--num-queries", type=int, default=100)
    command.add_argument("--nprobe", type=int, default=64)
    command.add_argument("--k", type=int, default=10)
    command.add_argument(
        "--curve-nprobe-values",
        default="1,4,16,64,256,1024,4096",
    )
    command.add_argument("--curve-k-values", default="10,1000,10000,20000")
    command.add_argument("--search-repetitions", type=int, default=5)
    command.add_argument("--warmup-queries", type=int, default=5)
    command.add_argument("--threads", type=int, default=1)
    command.add_argument("--memory-limit-bytes", type=int, default=2 * GIB)
    command.add_argument("--minimum-payload-bytes", type=int, default=48 * GIB)
    command.add_argument("--allow-resource-mismatch", action="store_true")
    command.add_argument("--allow-zero-physical-reads", action="store_true")
    command.add_argument("--output", type=Path)
    return command


def main(argv: list[str] | None = None) -> None:
    args = parser().parse_args(argv)
    result = benchmark(args)
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if args.output is None:
        print(encoded, end="")
    else:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
