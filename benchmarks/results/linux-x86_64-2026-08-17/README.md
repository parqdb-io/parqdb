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
- Query I/O: Linux Direct I/O for local index data
- Latency workload: 100 distinct BigANN queries after 10 warmup queries,
  640 MiB decompressed Page Cache
- Batch workload: 10,000 distinct BigANN queries without warmup, 512 MiB
  decompressed Page Cache
- Software: Python 3.12.2, PyArrow 25.0.0, Relify 0.1.0rc2

The index files were evicted from the host page cache before starting the query
container. Parquet metadata remained on the buffered path; index data ranges
bypassed the host page cache. Query latency excludes index loading, query
loading, and warmup.

## Build and single-query latency

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

## Batch throughput

Each scenario ran in a fresh 2 vCPU / 4 GiB container. The sequential path
submitted one query at a time. The batch path scanned the union of selected IVF
clusters once per batch and maintained an independent Top-K for each query.

| Mode | Batch size | Time | QPS | Recall@10 | Direct reads | Speedup |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Sequential | - | 650.6 s | 15.37 | 0.90094 | 847.6 GiB | 1.00x |
| Batch | 512 | 574.1 s | 17.42 | 0.90094 | 739.0 GiB | 1.13x |
| Batch | 1,024 | 500.9 s | 19.96 | 0.90099 | 646.0 GiB | 1.30x |

Batch sizes 2,048, 4,096, and 8,192 exceeded the 4 GiB cgroup limit. The 2,048
run completed two batches before it was killed; the larger sizes were killed
before returning their first batch. These are memory-bound failures, not
latency measurements. They show that the current union-cluster scan needs
stronger execution-state bounds before larger batches are practical under this
resource limit.

SIFT contains many tied distances. The query contract orders by distance but
does not define a secondary key, so parallel Top-K may choose different tied
candidates. The observed Recall@10 difference is 5 matches out of 100,000
returned rows.

## Raw Data

- [`sift1b-lvq8-build.json`](sift1b-lvq8-build.json)
- [`sift1b-lvq8-query.json`](sift1b-lvq8-query.json)
- [`sift1b-lvq8-batch-query.json`](sift1b-lvq8-batch-query.json)

Dataset preparation and benchmark commands are documented in
[`benchmarks/README.md`](../../README.md).
