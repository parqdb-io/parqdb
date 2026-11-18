# Roadmap

ParqDB develops one narrow path at a time. Current priorities follow the
[unified embedded and client/server API RFC](rfcs/20260815-unified-embedded-client-server-api.md).

## Current Foundation

- SQLite catalog with persistent Parquet source definitions.
- Immutable Parquet index publication over `file`, S3, and HDFS warehouses.
- `parqdb publish` for immutable IVF-LVQ4/LVQ8 publication to a local or
  S3-compatible prefix, including public HTTP Range and CORS checks.
- IVF source, LVQ4, and LVQ8 postings with shared centroids.
- Squared-L2 and cosine search over `float32` and `float64` source vectors.
- Embedded DataFusion planning, filtering, projection, exact fallback, and SQL
  composition.
- Bounded metadata, planning, centroid, and decompressed Parquet page caches.
- Process-scoped query admission and cancellable managed Arrow streams.
- Portable synchronous and asynchronous session facades.
- Process-scoped native index build coordination.
- Bounded incremental Arrow IPC stream encoding and decoding.
- Atomic refresh and reachability-based orphan removal.

## Experimental

These surfaces are available, but they are not the first supported product path
and may still change:

- Browser/WASM client (`@parqdb/browser`) that queries published IVF-LVQ4 and
  IVF-LVQ8 indexes over HTTP Range, without a query server.
- HTTP transport and Python ASGI server with incremental Arrow IPC, source URI
  authorization, remote index lifecycle, cancellation, and restart coverage.

## Next

1. Run the complete portable conformance suite against embedded and HTTP
   transports as the public surface grows.
2. Harden authentication, deployment guidance, and non-Python protocol tests.

## Later

- Batch vector queries with physical grouping by selected clusters.
- Additional index families and quantization schemes.
- Iceberg writing and a transactional shared catalog.
- Additional non-Python clients, and a stable (non-experimental) HTTP protocol
  for them.
- Distributed SQL integration through a standalone reference compiler rather
  than a Python compute-engine plugin framework.

These items are directions, not release commitments or dates. A feature becomes
part of the supported surface only after its implementation, conformance tests,
and operational limits are documented.
