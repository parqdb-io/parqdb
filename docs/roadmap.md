# Roadmap

Relify develops one narrow path at a time. Current priorities follow the
[unified embedded and client/server API RFC](rfcs/20260815-unified-embedded-client-server-api.md).

## Current Foundation

- SQLite catalog with persistent Parquet source definitions.
- Immutable Parquet index publication over `file`, S3, and HDFS warehouses.
- IVF source, LVQ4, and LVQ8 postings with shared centroids.
- Squared-L2 and cosine search over `float32` and `float64` source vectors.
- Embedded DataFusion planning, filtering, projection, exact fallback, and SQL
  composition.
- Bounded metadata, planning, centroid, and decompressed Parquet page caches.
- Process-scoped query admission and cancellable managed Arrow streams.
- Portable synchronous and asynchronous session facades.
- Process-scoped native index build coordination.
- Bounded incremental Arrow IPC stream encoding and decoding.
- Query-only HTTP transport and Python ASGI server with incremental Arrow IPC,
  cancellation, queue-error mapping, and restart coverage.
- Atomic refresh and reachability-based orphan removal.

## Next

1. Add remote table and index lifecycle after URI authorization,
   table-scoped identity, and interrupted-build tests pass.
2. Run one conformance suite against embedded and HTTP transports.

## Later

- Batch vector queries with physical grouping by selected clusters.
- Additional index families and quantization schemes.
- Iceberg writing and a transactional shared catalog.
- Browser and non-Python clients over the stable HTTP protocol.
- Distributed SQL integration through a standalone reference compiler rather
  than a Python compute-engine plugin framework.

These items are directions, not release commitments. A feature becomes part of
the supported surface only after its implementation, conformance tests, and
operational limits are documented.
