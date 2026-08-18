# macOS arm64 persisted-build and search benchmark

These measurements compare persisted IVF-Flat construction and resident-index
single-query search on one recorded system. They are not universal performance
claims.

## Configuration

- ParqDB source revision: `aad9580544c1f8ccffdfd9bb511c46680b3e10ae`
- Dataset: seeded standard-normal synthetic `float32` vectors
- Sizes: 100,000; 500,000; and 1,000,000 vectors
- Dimension: 128
- Queries: 50, with exact brute-force ground truth
- IVF: `nlist=256`, headline `nprobe=16`, headline `k=10`
- Training: at most 256 points per centroid, 20 iterations, seed 42
- Build repetitions: three independent persisted indexes, reported as medians
- Search: resident-index, serial single-query, three repetitions per query
- Large-k sweep on 1M:
  `k=100,1,000,10,000`, `nprobe=1,2,4,8,16,32,64,128,256`
- Host: arm64 macOS, 10 logical CPUs, 16 GiB memory
- Software: Python 3.12.13, Rust 1.96.0, ParqDB 0.1.0, Faiss 1.14.3

## Persisted Build

Both timers start from the same uncompressed Parquet source. ParqDB stops after
Parquet index data, metadata, and catalog publication are complete. Faiss stops
after `write_index` and file synchronization.

| Vectors | Implementation | Seconds | Vectors/s | Persisted MiB | Recall@10 |
|---:|---|---:|---:|---:|---:|
| 100,000 | ParqDB | 0.285 | 351,010 | 65.5 | 0.268 |
| 100,000 | Faiss | 0.322 | 310,438 | 49.7 | 0.276 |
| 500,000 | ParqDB | 0.865 | 577,870 | 325.9 | 0.274 |
| 500,000 | Faiss | 0.576 | 868,230 | 248.1 | 0.270 |
| 1,000,000 | ParqDB | 1.608 | 621,828 | 651.7 | 0.306 |
| 1,000,000 | Faiss | 1.087 | 919,570 | 496.0 | 0.282 |

Source fixture generation and exact ground-truth computation are outside both
timers. Faiss includes Parquet decoding, training, addition, serialization, and
file synchronization; its old in-memory-only timing is no longer used.

## Large-k Recall-Latency

Faiss is reopened with `read_index`; ParqDB is reopened through a new session
and its complete index snapshot is decoded into Arrow memory. Index loading is
outside query latency. At 1M vectors, ParqDB retained approximately 530.6 MiB
and loaded it in 0.116 seconds; Faiss loaded its index in 0.294 seconds. Both
execute one query at a time.

| K | nprobe | Implementation | Recall@K | p50 ms | p95 ms |
|---:|---:|---|---:|---:|---:|
| 100 | 16 | ParqDB | 0.2582 | 15.158 | 16.551 |
| 100 | 16 | Faiss | 0.2482 | 0.577 | 0.613 |
| 1,000 | 16 | ParqDB | 0.2111 | 15.105 | 16.564 |
| 1,000 | 16 | Faiss | 0.2092 | 0.766 | 0.778 |
| 10,000 | 16 | ParqDB | 0.1688 | 17.071 | 19.433 |
| 10,000 | 16 | Faiss | 0.1670 | 2.381 | 2.410 |
| 100 | 256 | ParqDB | 1.0000 | 24.980 | 26.606 |
| 100 | 256 | Faiss | 1.0000 | 7.820 | 8.394 |
| 1,000 | 256 | ParqDB | 1.0000 | 27.425 | 28.307 |
| 1,000 | 256 | Faiss | 1.0000 | 8.106 | 8.567 |
| 10,000 | 256 | ParqDB | 1.0000 | 40.515 | 43.851 |
| 10,000 | 256 | Faiss | 1.0000 | 10.770 | 10.928 |

The complete 36-point curve per implementation, p50/p95/mean latency, QPS, and
all 10,800 per-query latency samples are in [`1m.json`](1m.json).

This historical result predates the separate build and query runners. Use the
current procedure in [`benchmarks/README.md`](../../README.md) for new results.
