# Changelog

All notable changes to Relify will be documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and releases use [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added

- Reusable IVF centroid artifacts used by source, LVQ4, and LVQ8 logical
  indexes.
- Cosine distance and `list<double>` source-vector support through canonical
  float conversion.
- Portable source, LVQ4, and LVQ8 conformance fixtures under IVF schema
  version 1.
- A reproducible GIST benchmark runner for Relify and Faiss Flat, SQ4, and SQ8
  comparisons.
- Native, process-scoped index build coordination with observable status,
  bounded parallelism, failure retention, and retry.
- Bounded incremental Arrow IPC encoding and decoding with transport-level
  backpressure and no full-result buffering.
- An HTTP transport and Python ASGI server using the same session facades,
  source and index lifecycle, error hierarchy, and managed query streams as
  embedded execution. Server-side source registration uses an explicit URI
  prefix allowlist.

### Changed

- The public Python API now uses portable session and table facades instead of
  inheriting DataFusion objects. DataFusion remains available through an
  explicit embedded-only escape hatch.
- Relify now has one supported DataFusion execution path. The public backend,
  builder, Spark, and StarRocks extension surfaces have been removed while the
  open index specification remains available to other engines.
- IVF postings no longer copy full source vectors. The public `IVF`
  configuration selects `source`, `lvq4`, or `lvq8` encoding and
  `l2_squared` or `cosine` distance.
- Portable SQL terminals now reject DDL, DML, `COPY`, and session-mutating
  statements before execution.
- Index names are scoped by their owning table identifier. Build status and
  transport errors expose stable failure codes for embedded/HTTP parity.

### Fixed

- LVQ distance evaluation now honors non-zero Arrow array offsets when cached
  postings batches are sliced.

### Removed

- The versioned backend and builder SDKs, capability registry, experimental
  Spark and StarRocks modules, and their public configuration objects.

## [0.1.0rc2] - 2026-08-04

### Fixed

- Embedded DataFusion DataFrame text and HTML representations now resolve the
  bundled formatter from the Relify namespace in clean wheel installations.

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
