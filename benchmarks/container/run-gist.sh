#!/bin/sh
set -eu

commit="${1:-}"
if [ -z "$commit" ]; then
    echo "usage: benchmarks/container/run-gist.sh <relify-commit>" >&2
    exit 2
fi

if [ -n "${CONTAINER_ENGINE:-}" ]; then
    engine="$CONTAINER_ENGINE"
elif command -v docker >/dev/null 2>&1; then
    engine=docker
elif command -v podman >/dev/null 2>&1; then
    engine=podman
else
    echo "Docker or Podman is required" >&2
    exit 2
fi

root="${BENCHMARK_ROOT:-$PWD/relify-gist-benchmark}"
cpus="${BENCHMARK_CPUS:-8}"
memory="${BENCHMARK_MEMORY:-16g}"
image="${BENCHMARK_IMAGE:-petrizhang/relify-benchmark:v1}"
mkdir -p "$root"
root="$(cd "$root" && pwd)"

run() {
    "$engine" run --rm --cpus "$cpus" --memory "$memory" \
        -v "$root:/benchmark" "$@"
}

run "$image" prepare
run \
    -e ENCODING="${RELIFY_ENCODING:-lvq8}" \
    -e THREADS="$cpus" \
    "$image" relify "$commit"
run \
    -e ENCODING="${FAISS_ENCODING:-sq8}" \
    -e THREADS="$cpus" \
    "$image" faiss
run "$image" merge \
    /benchmark/current/build-relify.json \
    /benchmark/current/build-faiss.json \
    --output /benchmark/current/build-comparison.json
run "$image" merge \
    /benchmark/current/query-relify.json \
    /benchmark/current/query-faiss.json \
    --output /benchmark/current/query-comparison.json

printf '\nCombined results: %s\n' "$root/current"
