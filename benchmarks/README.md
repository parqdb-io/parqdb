# Benchmarks

Relify has two benchmark entry points:

- `python -m benchmarks.build` builds and persists an IVF index.
- `python -m benchmarks.query` measures Recall@K and query latency against an
  existing index.

The benchmark runs Relify with LVQ8 by default and also supports `lvq4` and
`flat`. The standalone Faiss runner uses IVF-SQ8 by default and also supports
`sq4` and `flat`.

## GIST1M

[`prepare_gist.py`](tools/prepare_gist.py) downloads the public
[ANN-Benchmarks GIST1M dataset](https://github.com/erikbern/ann-benchmarks#data-sets),
verifies its SHA-256, and converts it to Parquet in bounded batches:

```bash
uv run --no-sync python -m benchmarks.tools.prepare_gist \
  --root /data/relify-benchmarks
```

The prepared directory contains one million 960-dimensional base vectors, 1,000
queries, and GT@100. Download and conversion are dataset preparation and are not
included in index build time. Use `benchmarks.build`, `benchmarks.query`, and the
standalone `benchmarks.tools.faiss` runner with the prepared files. Run the two
implementations under equivalent resource limits, then combine their JSON files
with `benchmarks.tools.merge_results`.

## SIFT1B

The standard BigANN files are already available in many ANN benchmark
environments. Relify's preparation command reads the compact `uint8` base and
query `bvecs` files directly; it does not use a converted `fvecs` copy. It
writes a normal Parquet source table with `id: INT64` and
`embedding: FixedSizeList<FLOAT, 128>`, plus an `fbin` query matrix and
canonical ground-truth matrix for the existing build and query runners.

```bash
uv run --group benchmark --no-sync python -m benchmarks.tools.prepare_sift1b \
  --base /home/share/vector_data/sift1b/sift1b_base.bvecs \
  --queries /home/share/vector_data/sift1b/sift1b_query.bvecs \
  --ground-truth /home/share/vector_data/sift1b/sift1b_groundtruth.ivecs \
  --output /storage_ssd/relify-benchmarks/datasets/sift1b \
  --workers 32
```

The source layout is fixed for reproducibility: 65,536 rows per row group and
2,097,152 rows per file, producing 477 Parquet files for the complete 1B-row
dataset. Embedding values use `BYTE_STREAM_SPLIT` and `ZSTD(3)`; IDs are stored
plain. Preparation is outside the timed index-build measurement.

Build an IVF-LVQ8 index through the normal public API path. The spill limit is
explicit because the 1B-row postings shuffle exceeds DataFusion's 100 GiB
default:

```bash
DATASET=/data/sift1b
INDEX_ROOT=/benchmarks/indexes/sift1b-lvq8-65536

uv run --group benchmark --no-sync python -m benchmarks.build \
  --source-parquet "$DATASET/source" \
  --dataset-name sift1b-bigann \
  --nlist 65536 \
  --encoding lvq8 \
  --threads 64 \
  --max-temp-directory-size-bytes 322122547200 \
  --index-root "$INDEX_ROOT" \
  --rebuild \
  --output /benchmarks/results/sift1b-lvq8-65536-build.json
```

Query against the prepared BigANN queries and ground truth:

```bash
uv run --group benchmark --no-sync python -m benchmarks.query \
  --source-parquet "$DATASET/source" \
  --query-file "$DATASET/queries.fbin" \
  --ground-truth "$DATASET/gt1000.bin" \
  --dataset-name sift1b-bigann \
  --num-queries 100 \
  --nlist 65536 \
  --encoding lvq8 \
  --nprobe 32 \
  --k 10 \
  --curve-nprobe-values 32 \
  --curve-k-values 10 \
  --search-repetitions 1 \
  --warmup-queries 10 \
  --threads 2 \
  --index-io direct \
  --page-cache-capacity-bytes 687194767 \
  --index-root "$INDEX_ROOT" \
  --output /benchmarks/results/sift1b-lvq8-65536-query.json
```

The recorded 64 vCPU build and 2 vCPU / 4 GiB query run is archived in
[`results/linux-x86_64-2026-08-17`](results/linux-x86_64-2026-08-17/README.md).

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

Run each implementation in a fresh environment limited to **32 vCPU and 128
GiB**. Both read the same original Parquet columns. The default compares Relify
LVQ8 with Faiss IVF-SQ8. Both use `nlist=8192`. Each implementation draws a
seeded bounded sample of up to 256 training vectors per centroid.
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

Merge the independently executed results only after both runs complete:

```bash
python -m benchmarks.tools.merge_results \
  "$RESULT_ROOT/build-relify.json" \
  "$RESULT_ROOT/build-faiss.json" \
  --output "$RESULT_ROOT/build-comparison.json"
```

## Query Wikipedia

Run each implementation in a fresh environment limited to **32 vCPU and 128
GiB**. Index loading, query extraction from the source table, and warmup are
outside query latency. Queries execute one at a time with intra-query
parallelism.

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

The same persisted indexes may also be queried in an environment limited to **1
vCPU and 2 GiB** to characterize storage-backed execution. Relify reads its
Parquet index through its bounded decompressed Page cache; Faiss opens the
standard persisted index with read-only mmap. Queries run once in source order
without actively evicting the cache, so frequently accessed index partitions
may remain resident. This is a resource profile of the same query workload, not
a different dataset or index.

Committed results must preserve the raw JSON and identify the declared resource
limits. Machine-specific CPU affinity and NUMA configuration are execution
details and are not part of the portable benchmark contract.
