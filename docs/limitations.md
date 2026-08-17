# Current Limitations

This page defines the implemented boundary. The specification and RFCs may
describe behavior that is not yet exposed by a released runtime.

## Platforms

- CPython 3.11 through 3.14.
- Linux x86_64 with glibc 2.28 or later.
- macOS arm64 11 or later.
- No initial Windows, Linux arm64, macOS x86_64, or free-threaded wheels.

## Indexes and Queries

- IVF is the only index family. Postings may use source, LVQ4, or LVQ8
  encoding.
- Squared L2 and cosine are supported.
- Query vectors are one-dimensional; batch query execution is not implemented.
- Equal-distance result order is unspecified.
- Source vectors must have one fixed dimension and finite `float32` or
  `float64` elements. Computation canonicalizes values to `float32`.
- Source filters are SQL strings rather than a typed expression API.
- Automatic reclamation of an unused shared centroid artifact is not
  implemented.

## Catalog and Storage

- SQLite is the only catalog implementation.
- Index construction writes Parquet. Iceberg writing is not implemented.
- Storage access supports canonical `file`, S3, and HDFS locations.
- Parquet sources have no snapshot isolation. Replacing registered files in
  place is the application's responsibility.
- Iceberg reads require PyIceberg and validate exact table identity and
  snapshot.

## Runtime

- The supported execution runtime is embedded DataFusion.
- Client/server transport is under development and is not a stable deployment
  mode.
- Distributed engine adapters are not bundled.
- Build submission is process-local. Published metadata and index snapshots
  are durable, but in-progress jobs do not survive process restart.
- ParqDB does not provide replication, a highly available catalog, tenant
  isolation, authentication, or a managed service control plane.
