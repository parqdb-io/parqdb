#!/bin/sh
set -eu

mode="${1:-}"
case "$mode" in
    prepare)
        shift
        exec python -m benchmarks.tools.prepare_gist \
            --root "${BENCHMARK_ROOT:-/benchmark}/datasets" "$@"
        ;;
    relify)
        shift
        exec /usr/local/bin/run-relify "$@"
        ;;
    faiss)
        shift
        exec /usr/local/bin/run-faiss "$@"
        ;;
    merge)
        shift
        exec python -m benchmarks.tools.merge_results "$@"
        ;;
    *)
        echo "usage: run-relify-benchmark {prepare|relify|faiss|merge} ..." >&2
        exit 2
        ;;
esac
