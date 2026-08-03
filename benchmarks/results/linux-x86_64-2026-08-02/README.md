# Linux x86_64 Wikipedia 35M query benchmark

These measurements compare persisted IVF-Flat indexes using one-query-at-a-time
search. They describe one recorded run and are not universal performance
claims.

## Configuration

- Relify source revision: `b07e2892f83000494d9d86d5ed4a775628a5b221`
- Dataset: 35,167,820 Wikipedia vectors, dimension 384
- Queries: 100, with published exact GT@20,000
- IVF: `nlist=8,192`, `k=20,000`
- Training: 2,097,152 sampled vectors, 20 iterations, seed 42
- Sweep: `nprobe=1,2,4,8,16,32,64,128,256,512,1,024,2,048,4,096`
- Search: one repetition per query after five warmup queries
- Query resources: 32 CPU workers; memory was not container-limited
- Software: Python 3.12.2, Rust 1.96.0, Relify 0.1.0, Faiss 1.14.3

This run predates the 32 vCPU, 128 GiB query-container specification and should
be rerun in that container before publication.

## Method

Relify and Faiss ran in separate processes and reused previously built indexes.
Index loading and five warmup queries are outside query latency. Each point
contains 100 samples and an exact Recall@20,000 measurement.

## Results

| nprobe | Relify Recall | Relify p50 ms | Faiss Recall | Faiss p50 ms |
|---:|---:|---:|---:|---:|
| 1 | 0.094114 | 4.282 | 0.097021 | 2.474 |
| 2 | 0.156719 | 4.681 | 0.158420 | 3.391 |
| 4 | 0.242494 | 5.440 | 0.249832 | 4.463 |
| 8 | 0.356691 | 6.218 | 0.364576 | 6.182 |
| 16 | 0.483165 | 7.240 | 0.488984 | 8.190 |
| 32 | 0.612945 | 7.697 | 0.617649 | 10.759 |
| 64 | 0.734095 | 9.976 | 0.738576 | 14.122 |
| 128 | 0.832671 | 13.803 | 0.835711 | 19.368 |
| 256 | 0.903227 | 22.114 | 0.905601 | 26.041 |
| 512 | 0.949652 | 36.968 | 0.952102 | 38.835 |
| 1,024 | 0.978007 | 63.974 | 0.980016 | 63.975 |
| 2,048 | 0.992684 | 115.001 | 0.993711 | 113.536 |
| 4,096 | 0.998802 | 217.179 | 0.998920 | 214.265 |

At equal `nprobe`, Relify has slightly lower recall, so latency should be read
together with recall rather than as an equal-work quality comparison.

## Raw Data

- [`wikipedia-35m-relify.json`](wikipedia-35m-relify.json): Relify result and
  all 1,300 measured query samples.
- [`wikipedia-35m-faiss.json`](wikipedia-35m-faiss.json): Faiss result and all
  1,300 measured query samples.

The generic benchmark entry point and dataset preparation procedure are
documented in [`benchmarks/README.md`](../../README.md).
