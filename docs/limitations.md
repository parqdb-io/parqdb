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

- IVF is the only implemented index family. The local backend supports
  source-encoded, LVQ4, and LVQ8 postings; experimental backends support only
  source-encoded L2 IVF indexes.
- Squared L2 and cosine are supported by the local backend.
- Query vectors are one-dimensional; batch queries are not implemented.
- Results order by distance; the relative order of equal-distance rows is
  unspecified.
- Vector columns must have one fixed dimension and finite `float32` or
  `float64` elements. Computation canonicalizes them to `float32`.
- The current API accepts SQL-string source filters; it does not expose a
  backend-neutral expression object for filters.
- Ready shared-IVF centroid artifacts are retained conservatively; automatic
  reclamation after the last logical index is dropped is not implemented.

## Catalog and Coordination

- SQLite is the only index catalog implementation.
- The Python catalog facade exposes one root index namespace.
- There is no remote catalog service or multi-node transaction coordinator.
- Local builds coordinate through the local catalog and filesystem state.

## Storage

- Local index construction writes Parquet.
- Local storage access supports `file`, S3, and HDFS.
- Parquet locations have no snapshot isolation. Replacing files under a
  registered path is the application's consistency responsibility.
- Iceberg reads validate exact table identity and snapshots through PyIceberg.

## Local DataFusion

The stable local backend supports Parquet source registration, local IVF
construction, exact fallback, filtering, projection, DataFusion SQL and
DataFrame composition, decompressed Parquet page caching, refresh, catalog
recovery, and orphan removal. The local builder does not write Iceberg indexes.

## Spark

The experimental backend supports Spark Classic 4.0 and 4.1 and native
PySpark DataFrame queries over compatible source-encoded L2 IVF indexes in
Parquet and Iceberg. It does not support:

- Spark Connect;
- index construction or refresh;
- Iceberg maintenance;
- a remote Relify index catalog; or
- production distributed benchmarks and conformance guarantees.

## StarRocks

The experimental backend requires StarRocks 3.5.1 or later and supports
query-only access to source-encoded L2 IVF indexes and their Iceberg source
tables through Arrow Flight SQL. It does not:

- build indexes with StarRocks compute;
- use StarRocks native vector indexes;
- query Parquet indexes through `FILES()`;
- create or configure the StarRocks Iceberg catalog; or
- expose a DataFrame facade.

Construction may be delegated explicitly to a compatible third-party builder.

## Operational Scope

Relify is an embedded library and compute-engine extension, not a managed
vector service. Replication, high-availability catalog deployment, query
admission control, tenant isolation, and high-concurrency online serving are
outside the 0.1 scope.
