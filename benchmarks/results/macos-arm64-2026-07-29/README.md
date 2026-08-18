# macOS arm64 embedded IVF-Flat benchmark

These measurements compare persisted IVF-Flat construction and memory-resident
search on one recorded system. Each implementation may use all execution
threads within one query. These are not universal performance claims.

## Configuration

- ParqDB source revision: `1b3b9f7c7767feab7692e6f453c2113fa3487c95`
- Dataset: 1,000,000 seeded standard-normal synthetic `float32` vectors
- Dimension: 128
- Queries: 50, with exact brute-force ground truth
- IVF: `nlist=4,096`
- Training: all 1,000,000 vectors, 20 iterations, and seed 42
- Index builds: one per implementation, measured independently from queries
- Search repetitions: three per query after five warmup queries
- Faiss search: 10 OpenMP threads, `parallel_mode=1` (inverted-list parallelism)
- Sweep: `k=10,000,20,000,100,000`,
  `nprobe=1,4,16,64,256,1,024,4,096`
- Host: arm64 macOS, 10 logical CPUs, 16 GiB memory
- Software: Python 3.12.13, Rust 1.96.0, ParqDB 0.1.0, Faiss 1.14.3

## Method

Both implementations build and persist an IVF-Flat index from the same
uncompressed Parquet source. ParqDB retains 532.5 MiB as decoded Arrow buffers,
while Faiss loads a 497.9 MiB in-process index. Each executes one query at a
time and may parallelize work within that query.

## Build Results

| Implementation | Build time | Persisted data |
|---|---:|---:|
| ParqDB | 28.70 s | 654.2 MiB |
| Faiss | 49.33 s | 497.9 MiB |

## Selected Results

| K | nprobe | Implementation | Recall@K | p50 ms | p95 ms |
|---:|---:|---|---:|---:|---:|
| 10,000 | 64 | ParqDB | 0.0808 | 1.971 | 2.222 |
| 10,000 | 64 | Faiss | 0.0812 | 1.415 | 1.445 |
| 20,000 | 64 | ParqDB | 0.0710 | 2.220 | 2.350 |
| 20,000 | 64 | Faiss | 0.0709 | 1.852 | 1.899 |
| 100,000 | 64 | ParqDB | 0.0482 | 2.232 | 2.369 |
| 100,000 | 64 | Faiss | 0.0480 | 3.678 | 3.867 |
| 10,000 | 4,096 | ParqDB | 1.0000 | 8.381 | 8.681 |
| 10,000 | 4,096 | Faiss | 1.0000 | 7.794 | 8.617 |
| 20,000 | 4,096 | ParqDB | 1.0000 | 10.127 | 10.756 |
| 20,000 | 4,096 | Faiss | 1.0000 | 10.645 | 12.278 |
| 100,000 | 4,096 | ParqDB | 1.0000 | 18.481 | 19.530 |
| 100,000 | 4,096 | Faiss | 1.0000 | 41.704 | 47.239 |

The complete 21-point curve per implementation, p50/p95/mean latency, QPS, and
all 6,300 per-query latency samples are in [`1m.json`](1m.json).

This historical result predates the separate build and query runners. Use the
current procedure in [`benchmarks/README.md`](../../README.md) for new results.
