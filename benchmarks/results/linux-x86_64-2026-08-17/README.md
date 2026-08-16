# Linux x86_64 SIFT1B benchmark

This run builds and queries a persisted IVF-LVQ8 index over the complete
BigANN SIFT1B base set. It records one constrained-resource run and is not a
universal performance claim.

## Configuration

- Relify source revision: `75dad2ea7af3fcf9681efc8acbfbf8bbe0c88bc2`
- Dataset: 1,000,000,000 vectors, dimension 128, squared Euclidean distance
- Source: 477 Parquet files, 116.31 GiB
- IVF: `nlist=65,536`, `nprobe=64`, LVQ8, `k=10`
- Training: 16,777,216 sampled vectors, seed 42
- Build resources: 64 vCPU, 128 GiB memory, 300 GiB spill limit
- Query resources: one physical core with two SMT threads, 4 GiB memory
- Queries: 100 distinct BigANN queries after 10 warmup queries
- Software: Python 3.12.2, PyArrow 25.0.0, Relify 0.1.0rc2

The index files were evicted from the host page cache before starting the query
container. Query latency excludes index loading, query loading, and warmup.

## Results

| Metric | Result |
| --- | ---: |
| Timed build | 2,159.7 s |
| Build throughput | 463,034 vectors/s |
| Index size | 132.96 GiB |
| Build peak cgroup memory | 113.84 GiB |
| Build peak process RSS | 61.61 GiB |
| Recall@10 | 0.940 |
| Query p50 | 137.38 ms |
| Query p95 | 223.44 ms |
| Query p99 | 291.52 ms |
| Mean query latency | 149.25 ms |
| Query throughput | 6.70 queries/s |
| Measured query reads | 13.55 GiB |

The query cgroup reached its 4 GiB memory limit without swapping. The published
index was 33 times larger than the query memory limit.

## Raw Data

- [`sift1b-lvq8-build.json`](sift1b-lvq8-build.json)
- [`sift1b-lvq8-query.json`](sift1b-lvq8-query.json)

Dataset preparation and benchmark commands are documented in
[`benchmarks/README.md`](../../README.md).
