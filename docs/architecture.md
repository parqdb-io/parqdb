# Architecture

Relify separates the public API, transport boundary, execution service,
catalog, immutable storage, and vector kernels.

```text
Session / AsyncSession
        |
        v
SessionTransport
        +--> InProcessTransport ------------------+
        |                                         |
        +--> HttpTransport --> HTTP / Arrow IPC   |
                                  |               |
                                  v               v
                              ASGI adapter --> SessionService
        |
        v
private embedded host
        |
        +--> LocalSession --> DataFusion
        |        |
        |        +--> RelifyRuntime
        |        +--> Parquet page cache
        |        +--> query admission
        |
        +--> SQLite catalog
        +--> Parquet / Iceberg storage
```

The asynchronous service path is authoritative. The blocking facade uses one
long-lived event-loop bridge and does not implement a second query path.

## Python Boundaries

`Session` and `SourceTable` expose portable table, index, and query operations.
They do not inherit DataFusion objects. `VectorQuery`, `IVF`, and
`WriteOptions` are immutable request values that can cross a future transport.

`SessionTransport` is private. `InProcessTransport` delegates directly to
`SessionService` without serialization. `HttpTransport` sends versioned JSON
requests and incrementally decodes Arrow IPC query results. Both return the same
public values and errors.

`SessionService` is the only Python layer that coordinates portable operations
with the private embedded host. DataFusion-specific registration and planning
stay below this boundary. Applications that deliberately need bundled
DataFusion can call `session.datafusion_context()`; that context is outside
embedded/remote parity guarantees.

The ASGI adapter owns deployment policy rather than execution semantics. It
canonicalizes each requested Parquet source and checks it against the server's
file-root and object-store-prefix allowlist before invoking `SessionService`.
The service never accepts client-supplied storage credentials. Index builds use
the same process-scoped coordinator as embedded sessions; HTTP create and
refresh calls acknowledge submission, while clients poll portable index status.

The Python package mirrors those ownership boundaries: `relify.server` owns
the public ASGI factory and server-only deployment policy; `relify.transport`
owns the portable HTTP, Arrow IPC, and in-process transports; and
`relify.runtime` owns the transport-neutral session service and repository
bridge. The public `relify.server` import remains stable while its server
implementation evolves behind that package boundary.

## Rust Components

- `relify-local` owns the embedded DataFusion session, query planning, Parquet
  access, caches, query admission, index construction runtime, and maintenance.
- `relify-index` loads, validates, discovers, and publishes immutable index
  metadata.
- `relify-catalog` owns structured table/index identifiers and atomic SQLite
  publication operations.
- `relify-meta` defines the portable metadata model.
- `relify-storage` resolves canonical `file`, S3, and HDFS locations under a
  managed warehouse root.
- `relify-kmeans` owns sampling, training, assignment, and cluster recovery.
- `relify-kernels` owns runtime-selected SIMD and GEMM kernels.
- `relify-iceberg` binds exact Iceberg snapshots to DataFusion providers.
- `parallite` provides partitioned local execution for index construction.

Lower-level crates do not depend on Python API types. Metadata and catalog
crates do not depend on DataFusion or numerical kernels.

## Query Runtime

Each process owns one `RelifyRuntime`, shared by embedded sessions. It contains
the Tokio runtime, DataFusion runtime environment, memory budget, Parquet page
cache, and query admission controller. Query concurrency and queue limits are
process resources rather than per-request state.

Native SQL and vector queries return `ManagedQueryStream`. The stream owns the
DataFusion stream, cancellation token, and admission permit. Exhaustion,
explicit close, cancellation, or Arrow reader destruction releases the permit.

The Python ASGI adapter invokes the same long-lived `SessionService` used by the
embedded transport. It does not plan queries or own execution resources. The
network boundary uses the Arrow IPC streaming format. Encoding pulls one record
batch at a time and emits bounded byte chunks through a shared executor. The
native incremental decoder accepts arbitrary transport boundaries, returns at
most one batch per pull, and bounds each incomplete IPC frame. Neither side
collects a complete result before forwarding it. Closing the HTTP response
closes the managed native stream and releases query admission.

Centroid routing uses a bounded native matrix path for ordinary indexes and a
relational path for larger matrices. Selected cluster IDs become static
postings predicates. Hive-style `cid=<value>` partitions allow file pruning
before Parquet decoding.

The physical optimizer can replace distance projection plus distance Top-K
with `IvfTopKExec`. The fused operator reads Arrow buffers directly, computes
distance with native kernels, and materializes payload columns only for retained
rows. Unsupported plans remain on the general DataFusion path.

## Storage and Caches

Index metadata and relations are immutable. Their URI is therefore a complete
cache-consistency boundary. Metadata, centroid matrices, postings manifests,
and decompressed Parquet pages use bounded caches; publishing a new snapshot
creates new immutable locations instead of mutating cached objects.

The Parquet page cache sits inside the reader path. File, row-group, and column
pruning happen before cache lookup. A hit avoids storage I/O and decompression
while preserving normal decoder semantics.

## Publication

A build writes to a new warehouse prefix and validates every relation before
publication:

```text
<warehouse>/indexes/<index-uuid>/<snapshot-id>/
<warehouse>/metadata/<index-uuid>/v<sequence>.metadata.json
```

The SQLite register or compare-and-swap commit is the publication point. A
failed build cannot expose partial data. Orphan removal retains unreachable
objects for a minimum safety period and rechecks catalog reachability before
deletion.
