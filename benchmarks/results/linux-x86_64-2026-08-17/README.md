# Linux x86_64 SIFT1B benchmark

This run builds and queries a persisted IVF-LVQ8 index over the complete
BigANN SIFT1B base set. It reports the fastest measured setting above 0.90
Recall@10 under the stated resource limit, not a complete recall curve.

## Configuration

- Build revision: `75dad2ea7af3fcf9681efc8acbfbf8bbe0c88bc2`
- Query revision: `0355b29dd4df0511050a851e7f54899e25363b9b`
- Dataset: 1,000,000,000 vectors, dimension 128, squared Euclidean distance
- Source: 477 Parquet files, 116.31 GiB
- IVF: `nlist=65,536`, `nprobe=32`, LVQ8, `k=10`
- Training: 16,777,216 sampled vectors, seed 42
- Build resources: 64 vCPU, 128 GiB memory, 300 GiB spill limit
- Query resources: one physical core with two SMT threads, 4 GiB memory
- Query I/O: Linux Direct I/O for local index data, 640 MiB decompressed Page Cache
- Queries: 100 distinct BigANN queries after 10 warmup queries
- Software: Python 3.12.2, PyArrow 25.0.0, ParqDB 0.1.0rc2

The index files were evicted from the host page cache before starting the query
container. Parquet metadata remained on the buffered path; index data ranges
bypassed the host page cache. Query latency excludes index loading, query
loading, and warmup.

## Results

| Metric | Result |
| --- | ---: |
| Timed build | 2,159.7 s |
| Build throughput | 463,034 vectors/s |
| Index size | 132.96 GiB |
| Build peak cgroup memory | 113.84 GiB |
| Build peak process RSS | 61.61 GiB |
| Recall@10 | 0.903 |
| Query p50 | 63.05 ms |
| Query p95 | 97.34 ms |
| Query p99 | 186.94 ms |
| Mean query latency | 67.88 ms |
| Query throughput | 14.73 queries/s |
| Measured query reads | 9.49 GiB |

The query cgroup peaked at 3.57 GiB without swapping. The published index was
33 times larger than the query memory limit.

## Raw Data

- [`sift1b-lvq8-build.json`](sift1b-lvq8-build.json)
- [`sift1b-lvq8-query.json`](sift1b-lvq8-query.json)

Dataset preparation and benchmark commands are documented in
[`benchmarks/README.md`](../../README.md).
