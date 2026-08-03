UV_CACHE_DIR ?= .uv-cache
UV_RUN = UV_CACHE_DIR=$(UV_CACHE_DIR) uv run --no-sync
TEST_ENV ?= tests/test-env.toml
PACKAGE_TARGET_DIR := $(CURDIR)/target/package
PYTHON_SOURCES := python tests benchmarks examples/python tools spec/fixtures/v1/generate.py
SEARCH_BENCHMARK_RESULT := benchmarks/results/macos-arm64-2026-07-29/1m.json

.PHONY: sync develop format lint test test-python test-rust test-interop test-capabilities test-s3 test-hdfs test-spark-iceberg test-starrocks test-remote-storage audit verify-datafusion-vendor fixtures datasets benchmark-smoke benchmark-chart sbom package verify-package check

sync:
	UV_CACHE_DIR=$(UV_CACHE_DIR) uv sync --no-install-project

develop:
	$(UV_RUN) maturin develop

format:
	$(UV_RUN) ruff check --fix $(PYTHON_SOURCES)
	$(UV_RUN) ruff format $(PYTHON_SOURCES)
	cargo fmt --all

lint:
	$(UV_RUN) ruff check $(PYTHON_SOURCES)
	$(UV_RUN) ruff format --check $(PYTHON_SOURCES)
	$(UV_RUN) python tools/check_docs.py
	$(UV_RUN) pyright
	cargo fmt --all --check
	cargo clippy --workspace --all-targets -- -D warnings

test-python:
	$(UV_RUN) pytest

test-rust:
	cargo test --workspace

test-interop:
	$(UV_RUN) pytest tests/interop

test-capabilities:
	$(UV_RUN) pytest --capabilities

test-s3:
	@command -v docker >/dev/null 2>&1 || { echo "test-s3 requires Docker"; exit 1; }
	@set -eu; \
	trap 'docker compose -f tests/integration/compose.yaml down --volumes' EXIT; \
	docker compose -f tests/integration/compose.yaml up --detach --wait minio; \
	$(UV_RUN) pytest \
		--test-env tests/integration/minio.toml \
		--require s3 \
		tests/python/test_s3_storage.py

test-hdfs:
	$(UV_RUN) pytest \
		--test-env tests/integration/minidfs.toml \
		--require hdfs \
		tests/integration/test_hdfs_storage.py

test-spark-iceberg:
	$(UV_RUN) --extra spark pytest \
		--test-env $(TEST_ENV) \
		--require spark \
		tests/python/test_spark_iceberg.py

test-starrocks:
	$(UV_RUN) --extra starrocks pytest \
		--test-env $(TEST_ENV) \
		--require iceberg,starrocks \
		tests/integration/test_starrocks_iceberg.py

test-remote-storage: test-s3 test-hdfs

test: test-python test-rust

audit:
	$(UV_RUN) pip-audit
	cargo deny check --hide-inclusion-graph

verify-datafusion-vendor:
	$(UV_RUN) python tools/sync_datafusion.py 54.0.0 --check

fixtures:
	$(UV_RUN) python spec/fixtures/v1/generate.py

datasets:
	$(UV_RUN) python tools/generate_example_datasets.py

benchmark-smoke:
	$(UV_RUN) pytest tests/python/test_benchmark.py -q -k 'build_benchmark_smoke or build_then_query'

benchmark-chart:
	$(UV_RUN) python -m benchmarks.tools.render_build_results $(SEARCH_BENCHMARK_RESULT) --output assets/build-time.svg
	$(UV_RUN) python -m benchmarks.tools.render_search_results $(SEARCH_BENCHMARK_RESULT) --k-values 10000,20000,100000 --output assets/search-recall-latency.svg

sbom:
	$(UV_RUN) python tools/generate_sbom.py

package: sbom
	rm -rf dist "$(PACKAGE_TARGET_DIR)"
	mkdir -p dist
	@set -eu; \
	trap 'rm -rf "$(PACKAGE_TARGET_DIR)"' EXIT; \
	CARGO_TARGET_DIR="$(PACKAGE_TARGET_DIR)" UV_CACHE_DIR=$(UV_CACHE_DIR) \
		uv run --no-sync maturin build --release --locked --compatibility pypi --out dist

verify-package: package
	@wheel_count="$$(find dist -maxdepth 1 -name 'relify-*.whl' | wc -l)"; \
	test "$$wheel_count" -eq 1; \
	wheel="$$(find dist -maxdepth 1 -name 'relify-*.whl' -print)"; \
	$(UV_RUN) python tools/verify_release.py "$$wheel"

check: lint test
