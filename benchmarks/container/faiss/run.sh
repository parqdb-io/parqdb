#!/bin/sh
set -eu

benchmark_root="${BENCHMARK_ROOT:-/benchmark}"
dataset="$benchmark_root/datasets/gist-960-euclidean"
threads="${THREADS:-$(nproc)}"
nlist="${NLIST:-1024}"
encoding="${ENCODING:-sq8}"
num_queries="${NUM_QUERIES:-100}"
nprobe="${NPROBE:-64}"
curve_nprobe_values="${CURVE_NPROBE_VALUES:-1,4,16,64,256}"
result_root="$benchmark_root/results/faiss-$encoding-nlist$nlist"
index_root="$benchmark_root/indexes/faiss-$encoding-nlist$nlist"
current_root="$benchmark_root/current"

test -f "$dataset/manifest.json" || {
    echo "GIST1M is not prepared under $dataset" >&2
    exit 2
}
mkdir -p "$result_root" "$index_root" "$current_root"

python -m benchmarks.tools.faiss build \
    --source-parquet "$dataset/source" \
    --dataset-name gist-960-euclidean \
    --dataset-revision 'etag-"34da1d8a80764582ee4b0c0839b7c32a-459"' \
    --dataset-split train \
    --nlist "$nlist" \
    --encoding "$encoding" \
    --threads "$threads" \
    --index-root "$index_root" \
    --output "$result_root/build.json"

python -m benchmarks.tools.faiss query \
    --source-parquet "$dataset/source" \
    --dataset-name gist-960-euclidean \
    --dataset-revision 'etag-"34da1d8a80764582ee4b0c0839b7c32a-459"' \
    --dataset-split train \
    --query-file "$dataset/queries.bin" \
    --ground-truth "$dataset/gt100.bin" \
    --num-queries "$num_queries" \
    --nlist "$nlist" \
    --encoding "$encoding" \
    --nprobe "$nprobe" \
    --k 10 \
    --curve-nprobe-values "$curve_nprobe_values" \
    --curve-k-values 10 \
    --search-repetitions 1 \
    --warmup-queries 5 \
    --threads "$threads" \
    --index-root "$index_root" \
    --output "$result_root/query.json"

cp "$result_root/build.json" "$current_root/build-faiss.json"
cp "$result_root/query.json" "$current_root/query-faiss.json"

printf '\nBuild result: %s\nQuery result: %s\n' \
    "$result_root/build.json" \
    "$result_root/query.json"
