# Benchmarks

Relify has two benchmark entry points:

- `python -m benchmarks.build` builds and persists an IVF index.
- `python -m benchmarks.query` measures Recall@K and query latency against an
  existing index.

The benchmark runs Relify with LVQ8 by default and also supports `lvq4` and
`flat`. The standalone Faiss runner uses IVF-SQ8 by default and also supports
`sq4` and `flat`.

## Reproduce on GIST1M

One command downloads the public
[ANN-Benchmarks GIST1M dataset](https://github.com/erikbern/ann-benchmarks#data-sets),
checks out and compiles a specified Relify commit, then runs Relify and Faiss in
separate containers:

```bash
curl -fsSL \
  https://raw.githubusercontent.com/petrizhang/relify/main/benchmarks/container/run-gist.sh \
  | sh -s -- <RELIFY_COMMIT>
```

The script selects Docker or Podman and uses one image for dataset preparation,
the isolated Relify and Faiss runs, and result merging. Relify and Faiss run in
separate resource-constrained containers. The image-owned benchmark harness is
fixed across both runs, including the Python, NumPy, and PyArrow versions. The
requested Relify commit is compiled and installed without replacing those
runtime dependencies. Allow at least 20 GiB of free disk.
`relify-gist-benchmark/` retains downloads, compiled artifacts, indexes, and raw
JSON results for reuse. Merged build and query results are written under
`relify-gist-benchmark/current/`.

The default workload uses all 1,000,000 base vectors, 100 of the 1,000 public
queries, GT@100, `nlist=1024`, Relify LVQ8, Faiss SQ8, and an `nprobe` sweep of
`1,4,16,64,256`. Set `RELIFY_ENCODING=flat` and `FAISS_ENCODING=flat` to
compare exact-vector indexes. Resource defaults are 8 vCPU and 16 GiB per
implementation container.

The image definitions are under [`benchmarks/container`](container). The GIST
image uses [`prepare_gist.py`](tools/prepare_gist.py), which downloads
`https://ann-benchmarks.com/gist-960-euclidean.hdf5`, verifies its SHA-256, and
converts it to Parquet in bounded batches. Download and conversion are dataset
preparation and are not included in index build time.

## Wikipedia 35M

The larger release benchmark uses the `train` split of
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

Wikipedia remains a manual dataset because the benchmark intentionally reads the
upstream Parquet files without rewriting them. Prepare the pinned upstream
directory and the GT file published with the Relify benchmark release, then set:

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

## Build Wikipedia

Run each implementation in a fresh **32 vCPU, 128 GiB** container. Both read the
same original Parquet columns. The default compares Relify LVQ8 with Faiss
IVF-SQ8. Both use `nlist=8192`. Each implementation draws a seeded bounded
sample of up to 256 training vectors per centroid.
Input inspection and training-sample materialization are recorded as preparation
and excluded from `build_seconds`. The build timer includes centroid training,
full-data assignment, index organization, persistence, and publication.

```bash
uv run --no-sync python -m benchmarks.build \
  --source-parquet "$SOURCE" \
  --id-column id \
  --vector-column emb \
  --dataset-name maloyan/wikipedia-22-12-en-embeddings-all-MiniLM-L6-v2 \
  --dataset-revision ab22410c0589e39371431e3dd293e4f0fa0c4b26 \
  --dataset-split train \
  --nlist 8192 \
  --encoding lvq8 \
  --threads 32 \
  --index-root "$INDEX_ROOT" \
  --rebuild \
  --output "$RESULT_ROOT/build-relify.json"
```

Run the Faiss baseline with the same source and workload arguments using
`python -m benchmarks.tools.faiss build --encoding sq8`. Change the output to
`build-faiss.json`. Build results
include preparation time, build time, vectors per second, persisted index bytes,
and peak process RSS. The peak covers preparation as well as the timed build.
Raw cgroup counters remain in the result for diagnostics, but they are not used
as the comparable peak-memory metric because a cgroup may contain unrelated
processes outside an isolated benchmark container.

Merge the independently executed results only after both containers complete:

```bash
python -m benchmarks.tools.merge_results \
  "$RESULT_ROOT/build-relify.json" \
  "$RESULT_ROOT/build-faiss.json" \
  --output "$RESULT_ROOT/build-comparison.json"
```

## Query Wikipedia

Run each implementation in a fresh **32 vCPU, 128 GiB** container. Index loading,
query extraction from the source table, and warmup are outside query latency.
Queries execute one at a time with intra-query parallelism.

The release workload reports `Recall@20,000` and sweeps `nprobe` from 1 to 4,096.
The published GT supports workloads up to `K=100,000` without changing the
dataset contract.

```bash
uv run --no-sync python -m benchmarks.query \
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
  --encoding lvq8 \
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

Run the Faiss baseline with the same query arguments using
`python -m benchmarks.tools.faiss query --encoding sq8`, and change the output
to `query-faiss.json`. Equal `nprobe` does not imply equal recall, so comparisons
must report latency and Recall@K together.

Use `benchmarks.tools.merge_results` to produce `query-comparison.json` from the
two query results. The merger rejects different datasets, resource limits,
benchmark revisions, or common workload parameters.

## Constrained Memory

The same persisted indexes may also be queried in a **1 vCPU, 2 GiB** container
to characterize storage-backed execution. Relify reads its Parquet index
without `cache_index`; Faiss opens the standard persisted index with read-only
mmap. Queries run once in source order without actively evicting the page cache,
so frequently accessed index partitions may remain cached. This is a resource
profile of the same query workload, not a different dataset or index.

Committed results must preserve the raw JSON and identify the declared container
resources. Machine-specific CPU affinity and NUMA configuration are execution
details and are not part of the portable benchmark contract.
