#!/bin/sh
set -eu

commit="${1:-}"
if [ -z "$commit" ]; then
    echo "usage: run-relify-benchmark <relify-commit>" >&2
    exit 2
fi

repository_url="${RELIFY_REPOSITORY:-https://github.com/petrizhang/relify.git}"
benchmark_root="${BENCHMARK_ROOT:-/benchmark}"
checkout="$benchmark_root/checkouts/relify"
dataset="$benchmark_root/datasets/gist-960-euclidean"
cache_root="$benchmark_root/cache"

test -f "$dataset/manifest.json" || {
    echo "GIST1M is not prepared under $dataset" >&2
    exit 2
}
mkdir -p "$checkout" "$cache_root"
if [ ! -d "$checkout/.git" ]; then
    git clone --filter=blob:none --no-checkout "$repository_url" "$checkout"
fi

git -C "$checkout" fetch --depth=1 origin "$commit"
git -C "$checkout" checkout --detach FETCH_HEAD
resolved_commit="$(git -C "$checkout" rev-parse HEAD)"
short_commit="$(git -C "$checkout" rev-parse --short=12 HEAD)"
runtime="$benchmark_root/runtimes/relify-$short_commit"

threads="${THREADS:-$(nproc)}"
nlist="${NLIST:-1024}"
encoding="${ENCODING:-lvq8}"
num_queries="${NUM_QUERIES:-100}"
nprobe="${NPROBE:-64}"
curve_nprobe_values="${CURVE_NPROBE_VALUES:-1,4,16,64,256}"
result_root="$benchmark_root/results/$short_commit/relify-$encoding-nlist$nlist"
index_root="$benchmark_root/indexes/$short_commit/relify-$encoding-nlist$nlist"
current_root="$benchmark_root/current"

export UV_CACHE_DIR="$cache_root/uv"
export CARGO_HOME="$cache_root/cargo"
export CARGO_TARGET_DIR="$cache_root/target/$short_commit"
export RUSTUP_HOME="/root/.rustup"
export BENCHMARK_IMPLEMENTATION_REVISION="$resolved_commit"

mkdir -p "$result_root" "$index_root" "$current_root"
rm -rf "$runtime"
python -m venv --system-site-packages "$runtime"
"$runtime/bin/python" -m pip install \
    --no-index \
    --no-build-isolation \
    --no-deps \
    "$checkout"
python="$runtime/bin/python"
cd /opt/relify

"$python" -m benchmarks.build \
    --source-parquet "$dataset/source" \
    --dataset-name gist-960-euclidean \
    --dataset-revision 'etag-"34da1d8a80764582ee4b0c0839b7c32a-459"' \
    --dataset-split train \
    --nlist "$nlist" \
    --encoding "$encoding" \
    --threads "$threads" \
    --index-root "$index_root" \
    --output "$result_root/build.json"

"$python" -m benchmarks.query \
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

cp "$result_root/build.json" "$current_root/build-relify.json"
cp "$result_root/query.json" "$current_root/query-relify.json"

printf '\nRelify commit: %s\nBuild result:  %s\nQuery result:  %s\n' \
    "$resolved_commit" \
    "$result_root/build.json" \
    "$result_root/query.json"
