# Benchmarks

Relify has two benchmark entry points:

- `python -m benchmarks.build` builds and persists an IVF-Flat index.
- `python -m benchmarks.query` measures Recall@K and query latency against an
  existing index.

Build and query run in separate **32 vCPU, 128 GiB** containers. The benchmark
uses the upstream Parquet files directly. It does not create a normalized source
table or a vector-only copy before indexing.

## Standard Dataset

The release benchmark uses the `train` split of
[`maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2`](https://huggingface.co/datasets/maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2)
at revision `ab22410c0589e39371431e3dd293e4f0fa0c4b26`.

| Property | Value |
| --- | --- |
| Source rows | 35,167,920 |
| Parquet files | 145 |
| Unique key | `id` (`INT32`, values `0..35,167,919`) |
| Vector | `emb` (`LIST<FLOAT>`, 384 dimensions) |
| Queries | Source rows with `id=0..99`, ordered by `id` |
| Distance | Squared Euclidean distance |
| Ground truth | Exact GT@100,000 using original source IDs |

The query rows remain in the indexed source. Their self-match is therefore rank
1. This changes Recall@20,000 by at most `1 / 20,000` and avoids rewriting the
upstream dataset solely to create a disjoint query set.

Dataset acquisition and exact ground-truth generation are outside the benchmark
runner. Prepare the pinned upstream Parquet directory and the GT file published
with the Relify benchmark release, then set:

```bash
SOURCE=/data/wikipedia/data
GT=/data/wikipedia-train-gt100000-100.bin
INDEX_ROOT=/benchmarks/indexes/wikipedia-nlist8192
RESULT_ROOT=/benchmarks/results
```

The GT file uses the DiskANN matrix layout: two little-endian `uint32` header
values (`100`, `100000`), followed by row-major `uint32` source IDs. Validate the
prepared inputs before running an experiment:

```bash
uv run --no-sync python -m benchmarks.tools.validate_wikipedia \
  --source-parquet "$SOURCE" \
  --ground-truth "$GT"
```

Validation checks the fixed dataset shape, the complete unique-key domain, the
GT shape and ID range, and the source-resident self-match. It reports the GT
checksum and does not rewrite the source data.

## Build

Run each implementation in a fresh container. Both implementations read the
same original Parquet columns and persist an IVF-Flat index with `nlist=8192`.
Each implementation draws a seeded uniform sample without replacement of up to
256 training vectors per centroid; sample memory is bounded by that limit.
Input inspection and training-sample materialization are recorded as preparation
and excluded from `build_seconds`. The build timer includes centroid training,
full-data assignment, index organization, persistence, and publication.

```bash
uv run --no-sync python -m benchmarks.build \
  --implementations relify \
  --source-parquet "$SOURCE" \
  --id-column id \
  --vector-column emb \
  --dataset-name maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2 \
  --dataset-revision ab22410c0589e39371431e3dd293e4f0fa0c4b26 \
  --dataset-split train \
  --nlist 8192 \
  --threads 32 \
  --index-root "$INDEX_ROOT" \
  --rebuild \
  --output "$RESULT_ROOT/build-relify.json"
```

Repeat with `--implementations faiss`, change the output to `build-faiss.json`,
and add `--require-faiss`. Build results
include preparation time, build time, vectors per second, persisted index bytes,
and peak process RSS. The peak covers preparation as well as the timed build.
Raw cgroup counters remain in the result for diagnostics, but they are not used
as the comparable peak-memory metric because a cgroup may contain unrelated
processes outside an isolated benchmark container.

## Query

Run each implementation in a fresh **32 vCPU, 128 GiB** container. Index loading,
query extraction from the source table, and warmup are outside query latency.
Queries execute one at a time with intra-query parallelism.

The release workload reports `Recall@20,000` and sweeps `nprobe` from 1 to 4,096.
The published GT supports workloads up to `K=100,000` without changing the
dataset contract.

```bash
uv run --no-sync python -m benchmarks.query \
  --implementations relify \
  --source-parquet "$SOURCE" \
  --id-column id \
  --vector-column emb \
  --dataset-name maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2 \
  --dataset-revision ab22410c0589e39371431e3dd293e4f0fa0c4b26 \
  --dataset-split train \
  --query-source-start 0 \
  --ground-truth "$GT" \
  --num-queries 100 \
  --nlist 8192 \
  --nprobe 64 \
  --k 20000 \
  --curve-nprobe-values 1,2,4,8,16,32,64,128,256,512,1024,2048,4096 \
  --curve-k-values 20000 \
  --search-repetitions 1 \
  --warmup-queries 5 \
  --threads 32 \
  --index-root "$INDEX_ROOT" \
  --output "$RESULT_ROOT/query-relify.json"
```

Repeat with `--implementations faiss`, change the output to `query-faiss.json`,
and add `--require-faiss`. Equal `nprobe` does not imply equal recall, so
comparisons must report latency and Recall@K together.

## Constrained Memory

The same persisted indexes may also be queried in a **1 vCPU, 2 GiB** container
to characterize storage-backed execution. Relify reads its Parquet index
without `cache_index`; Faiss opens the standard persisted index with read-only
mmap. The runner evicts the implementation's index files from the page cache
immediately before opening them. This is a resource profile of the same query
workload, not a different dataset or index.

Committed results must preserve the raw JSON and identify the declared container
resources. Machine-specific CPU affinity and NUMA configuration are execution
details and are not part of the portable benchmark contract.
