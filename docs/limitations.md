# Current Limitations

This page defines the implementation boundary for Relify 0.1. Features in the
specification or roadmap are not release commitments unless listed as
implemented here or in the [Python API](python-api.md).

## Release Platforms

- Standard CPython 3.11 through 3.14.
- Linux x86_64 with glibc 2.28 or later.
- macOS arm64 11 or later.
- No free-threaded Python wheels.
- No initial Windows, Linux arm64, or macOS x86_64 binary release.

Other targets may build from source but are not part of the tested 0.1 binary
matrix.

## Index and Query Semantics

- IVF is the only implemented index family. The local backend supports exact
  vectors and LVQ4/LVQ8 encodings; experimental backends support a narrower
  subset.
- Squared L2 is the only distance metric.
- Query vectors are one-dimensional; batch queries are not implemented.
- Results order by distance; the relative order of equal-distance rows is
  unspecified.
- Vector columns must have one fixed dimension and finite `float32` elements.
- The current API accepts SQL-string source filters; it does not expose a
  backend-neutral expression object for filters.

## Catalog and Coordination

- SQLite is the only index catalog implementation.
- The Python catalog facade exposes one root index namespace.
- There is no remote catalog service or multi-node transaction coordinator.
- Local builds coordinate through the local catalog and filesystem state.
- The Spark development catalog assumes one coordinating driver; cross-driver
  build coordination is not implemented.

## Storage

- Local index construction writes Parquet.
- Spark index construction writes Iceberg.
- Local storage access supports `file`, S3, and HDFS.
- Parquet locations have no snapshot isolation. Replacing files under a
  registered path is the application's consistency responsibility.
- Iceberg reads validate exact table identity and snapshots through PyIceberg.

## Local DataFusion

The stable local backend supports Parquet source registration, local IVF
construction, exact fallback, filtering, projection, DataFusion SQL and
DataFrame composition, index caching, refresh, catalog recovery, and orphan
removal. The local builder does not write Iceberg indexes.

## Spark

The experimental backend supports Spark Classic 4.0 and 4.1, initial Iceberg
index construction, native PySpark DataFrame queries, and queries over
compatible Parquet indexes. It does not support:

- Spark Connect;
- index refresh;
- cross-driver build coordination;
- Iceberg maintenance;
- a remote Relify index catalog; or
- production distributed benchmarks and conformance guarantees.

## StarRocks

The experimental backend requires StarRocks 3.5.1 or later and supports
query-only access to Iceberg source and index tables through Arrow Flight SQL.
It does not:

- build indexes with StarRocks compute;
- use StarRocks native vector indexes;
- query Parquet indexes through `FILES()`;
- create or configure the StarRocks Iceberg catalog; or
- expose a DataFrame facade.

Construction can be delegated explicitly to a compatible Spark builder.

## Operational Scope

Relify is an embedded library and compute-engine extension, not a managed
vector service. Replication, high-availability catalog deployment, query
admission control, tenant isolation, and high-concurrency online serving are
outside the 0.1 scope.
