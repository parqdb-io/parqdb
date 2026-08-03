# Changelog

All notable changes to Relify will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0rc1] - 2026-08-04

### Added

- A portable specification for index catalogs, immutable metadata snapshots,
  Parquet and Iceberg relation references, IVF-Flat schemas, and vector-query
  semantics.
- Shared conformance fixtures for valid and invalid metadata, index tables, and
  ordered IVF results, including independent execution through DuckDB.
- An embedded DataFusion session with persistent Parquet table registration,
  SQLite-backed table and index catalogs, and support for files, directories,
  and nested wildcard patterns over local filesystems, S3, and HDFS.
- Parallel Rust IVF-Flat construction with bounded-memory deterministic
  sampling and k-means training, SIMD/GEMM distance kernels, composite source
  keys, optional vector storage in postings, and configurable Parquet output.
- A table-centered Python query API with automatic or explicit index
  selection, SQL-string filters, projection, `nprobes`, large result limits,
  exact-search fallback, and transparent source-table resolution.
- Portable Arrow collection plus lazy DataFusion DataFrame, executable SQL,
  `EXPLAIN`, and runtime analysis terminals for local vector queries.
- Asynchronous index creation and refresh, source-scoped status and waiting,
  immutable snapshot publication, resident index caching, and
  reachability-based orphan removal.
- A Spark Classic backend that builds IVF indexes in Iceberg, queries Parquet
  and Iceberg index tables through native PySpark DataFrame plans, and shares
  published indexes with the local backend.
- A query-only StarRocks backend that executes complete IVF plans over Arrow
  Flight SQL while binding every Iceberg source and index table to its exact
  snapshot.
- A versioned third-party backend SDK with lazy entry-point discovery, typed
  capability reports, shared query resolution and canonical schema validation,
  and reusable backend contract tests.
- An independent builder SDK and shared table lifecycle that allow Local,
  Spark, StarRocks, and third-party query sessions to select compatible
  construction engines without a backend/build capability matrix.
- Bundled example datasets, runnable Local, Spark, and StarRocks examples, and
  a capability-driven integration-test framework for optional environments.
- Reproducible persisted-build and large-`k` recall-latency benchmarks with
  Faiss comparison.
- Verified Maturin wheels with locked Python and Rust dependencies, CycloneDX
  SBOMs, license and vulnerability audits, and isolated build-and-search smoke
  tests.

### Security

- Canonical URI validation, warehouse confinement, immutable metadata writes,
  catalog compare-and-swap publication, cross-process build coordination, and
  conservative garbage-collection retention protect index state from partial
  publication and premature deletion.
