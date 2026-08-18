from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
import time
from datetime import timedelta
from pathlib import Path
from typing import Any

import parqdb

from benchmarks.tools.datasets import (
    ParquetSource,
    inspect_parquet_source,
    iter_parquet_vector_batches,
    sample_parquet_vectors,
)
from benchmarks.tools.harness import (
    KMEANS_MAX_ITERATIONS,
    KMEANS_SEED,
    directory_bytes,
    evict_tree,
    sync_file,
    sync_tree,
)
from benchmarks.tools.ivf import (
    FAISS_ENCODINGS,
    PARQDB_ENCODINGS,
    create_faiss_index,
)


def prepare_parqdb(
    source: ParquetSource,
    destination: Path,
    *,
    nlist: int,
    encoding: str,
    threads: int,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"ParqDB destination already exists: {destination}")
    try:
        session = parqdb.connect(destination)
        session.datafusion_context().sql(
            f"SET datafusion.execution.target_partitions = '{threads}'"
        ).collect()
        session.register_parquet("benchmark", source.path)
        table = session.table("benchmark")
        assert isinstance(table, parqdb.SourceTable)
        started = time.perf_counter()
        table.create_index(
            "benchmark_embedding",
            column="embedding",
            key=["id"],
            config=parqdb.IVF(nlist=nlist, encoding=encoding),
            wait_timeout=timedelta(hours=24),
        )
        sync_tree(destination)
        result = {
            "root": str(destination),
            "table": "benchmark",
            "index": "benchmark_embedding",
            "build_seconds": time.perf_counter() - started,
            "payload_bytes": directory_bytes(destination / "indexes"),
            "managed_bytes": directory_bytes(destination),
            "encoding": encoding,
        }
        evict_tree(destination / "indexes")
        result["page_cache_evicted_after_build"] = True
        return result
    except BaseException:
        shutil.rmtree(destination, ignore_errors=True)
        raise


def prepare_faiss(
    source: ParquetSource,
    destination: Path,
    *,
    nlist: int,
    encoding: str,
    threads: int,
    shard_rows: int,
    batch_rows: int,
) -> dict[str, Any]:
    if destination.exists():
        raise FileExistsError(f"Faiss destination already exists: {destination}")
    destination.mkdir(parents=True)
    try:
        import faiss  # pyright: ignore[reportMissingImports]
        from faiss.contrib.ondisk import merge_ondisk

        faiss.omp_set_num_threads(threads)
        training_rows = min(source.rows, nlist * 256)
        training = sample_parquet_vectors(
            source,
            training_rows,
            seed=KMEANS_SEED,
            batch_rows=batch_rows,
        )
        index = create_faiss_index(
            faiss,
            faiss.IndexFlatL2(source.dimension),
            dimension=source.dimension,
            nlist=nlist,
            encoding=encoding,
        )
        index.cp.niter = KMEANS_MAX_ITERATIONS
        index.cp.seed = KMEANS_SEED
        started = time.perf_counter()
        index.train(training)
        del training
        trained_path = destination / "trained.faiss"
        faiss.write_index(index, str(trained_path))

        shard_directory = destination / "shards"
        shard_directory.mkdir()
        shard_paths = []
        shard = faiss.clone_index(index)
        shard_number = 0
        for ids, vectors in iter_parquet_vector_batches(
            source,
            batch_rows=batch_rows,
        ):
            shard.add_with_ids(vectors, ids)
            if shard.ntotal >= shard_rows:
                shard_path = shard_directory / f"part-{shard_number:05d}.faiss"
                faiss.write_index(shard, str(shard_path))
                shard_paths.append(shard_path)
                shard_number += 1
                shard = faiss.clone_index(index)
        if shard.ntotal:
            shard_path = shard_directory / f"part-{shard_number:05d}.faiss"
            faiss.write_index(shard, str(shard_path))
            shard_paths.append(shard_path)

        inverted_lists_path = destination / "inverted-lists.bin"
        merge_ondisk(
            index,
            [str(path) for path in shard_paths],
            str(inverted_lists_path),
        )
        index_path = destination / "index.faiss"
        faiss.write_index(index, str(index_path))
        sync_file(inverted_lists_path)
        sync_file(index_path)
        shutil.rmtree(shard_directory)
        trained_path.unlink()
        result = {
            "index": str(index_path),
            "inverted_lists": str(inverted_lists_path),
            "build_seconds": time.perf_counter() - started,
            "payload_bytes": inverted_lists_path.stat().st_size,
            "metadata_bytes": index_path.stat().st_size,
            "inverted_lists_type": "OnDiskInvertedLists",
            "encoding": encoding,
            "training_rows": training_rows,
            "shard_rows": shard_rows,
            "shards": len(shard_paths),
        }
        evict_tree(destination)
        result["page_cache_evicted_after_build"] = True
        return result
    except BaseException:
        shutil.rmtree(destination)
        raise


def _write_metadata(path: Path, metadata: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        dir=path.parent,
        prefix=f".{path.name}.",
        delete=False,
    ) as stream:
        json.dump(metadata, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    os.replace(temporary, path)
    path.chmod(0o644)


def prepare(args: argparse.Namespace) -> dict[str, Any]:
    supported = PARQDB_ENCODINGS if args.implementation == "parqdb" else FAISS_ENCODINGS
    encoding = args.encoding or supported[0]
    if encoding not in supported:
        raise ValueError(f"unsupported {args.implementation} encoding: {encoding}")
    source = inspect_parquet_source(args.source_parquet)
    if not 1 <= args.nlist <= source.rows:
        raise ValueError(f"nlist must be in 1..={source.rows}")
    if args.threads <= 0 or args.shard_rows <= 0 or args.batch_rows <= 0:
        raise ValueError("threads, shard-rows, and batch-rows must be positive")
    output = args.output.expanduser().resolve()
    output.mkdir(parents=True, exist_ok=True)
    metadata_path = output / "storage-indexes.json"
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        expected = metadata["source"]
        if (
            expected["path"] != str(source.path)
            or expected["rows"] != source.rows
            or expected["dimension"] != source.dimension
            or metadata["parameters"]["nlist"] != args.nlist
            or metadata["parameters"]["threads"] != args.threads
        ):
            raise ValueError("existing storage preparation uses different inputs")
    else:
        metadata = {
            "schema_version": 1,
            "source": {
                "path": str(source.path),
                "rows": source.rows,
                "dimension": source.dimension,
                "bytes": source.bytes,
            },
            "parameters": {
                "nlist": args.nlist,
                "kmeans_seed": KMEANS_SEED,
                "kmeans_max_iterations": KMEANS_MAX_ITERATIONS,
                "threads": args.threads,
            },
            "indexes": {},
        }
    if args.implementation in metadata["indexes"]:
        raise ValueError(f"{args.implementation} is already prepared")

    if args.implementation == "parqdb":
        result = prepare_parqdb(
            source,
            output / "parqdb",
            nlist=args.nlist,
            encoding=encoding,
            threads=args.threads,
        )
    else:
        result = prepare_faiss(
            source,
            output / "faiss",
            nlist=args.nlist,
            encoding=encoding,
            threads=args.threads,
            shard_rows=args.shard_rows,
            batch_rows=args.batch_rows,
        )
    metadata["indexes"][args.implementation] = result
    _write_metadata(metadata_path, metadata)
    return metadata


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(
        prog="python -m benchmarks.tools.prepare_storage",
        description="Prepare persistent indexes for the storage-backed suite.",
    )
    command.add_argument(
        "--implementation",
        choices=("parqdb", "faiss"),
        required=True,
    )
    command.add_argument("--source-parquet", type=Path, required=True)
    command.add_argument("--output", type=Path, required=True)
    command.add_argument("--nlist", type=int, default=8_192)
    command.add_argument(
        "--encoding",
        choices=tuple(dict.fromkeys((*PARQDB_ENCODINGS, *FAISS_ENCODINGS))),
        help="defaults to lvq8 for ParqDB and sq8 for Faiss",
    )
    command.add_argument("--threads", type=int, default=32)
    command.add_argument("--shard-rows", type=int, default=1_048_576)
    command.add_argument("--batch-rows", type=int, default=65_536)
    return command


def main(argv: list[str] | None = None) -> None:
    metadata = prepare(parser().parse_args(argv))
    print(json.dumps(metadata, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
