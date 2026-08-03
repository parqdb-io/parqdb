# Architecture

Relify separates portable index state, catalog publication, managed storage,
construction, and query integration.

```text
Python API
    -> relify-local (DataFusion + Parquet/Iceberg reads)
    -> relify.experimental.spark (Spark + Parquet/Iceberg reads)
    -> relify.experimental.starrocks (StarRocks + Iceberg reads)
    -> third-party BackendPlugin sessions
    -> Local / Spark / third-party IndexBuilder implementations

relify-local -------------------------\
relify.experimental.spark -------------+--> relify-index
relify.experimental.starrocks --------/         |-> relify-core
                                                 |-> relify-catalog
                                                 |-> relify-meta
                                                 `-> relify-storage

relify-core -> relify-catalog
            -> relify-meta

relify-local -> relify-kmeans -> relify-kernels
             -> parallite
             -> relify-iceberg
             -> DataFusion
```

## Component Boundaries

`relify-meta` defines and validates the portable metadata types described by
the specification. It does not perform catalog, storage, or query operations.

`relify-catalog` defines one top-level registry for runtime tables and Relify
indexes. Its index side maps a structured identifier to the URI of the current
immutable metadata document and owns atomic register, compare-and-swap commit,
drop, and discovery. Its table side stores provider-defined definitions needed
to reconstruct local external tables. It does not read metadata files, source
data, or index tables.

`relify-storage` resolves absolute `file`, `s3`, and `hdfs` URIs through the
Arrow object-store interface. A `Warehouse` restricts Relify-managed metadata
and index data to one configured URI prefix.

`relify-kernels` owns the runtime-selected SIMD squared-L2 kernels, dense row
norms, and the platform GEMM boundary shared by index construction and local
query execution. It has no dependency on Arrow, DataFusion, index metadata,
catalogs, or storage.

`relify-kmeans` owns deterministic sampling, Lloyd training, centroid
assignment, and empty-cluster recovery over dense `f32` matrices. It depends
only on `relify-kernels` and the `parallite` execution context, not on Arrow,
DataFusion, index metadata, catalogs, or storage.

`relify-core` defines backend-neutral construction options, query intent,
portable build artifacts, and publication results. It depends only on
`relify-meta` and `relify-catalog`; it does not depend on Arrow, DataFusion,
Parquet, object stores, numerical kernels, or a concrete runtime.

`relify-index` owns immutable metadata I/O, catalog loading and discovery,
implicit index selection, and the publication transaction shared by backends.
It accepts portable `RelationReference` values and backend-produced
`IndexArtifacts`; it does not depend on Arrow, DataFusion, Spark, Parquet,
Iceberg, numerical kernels, or a concrete runtime.

`relify-iceberg` resolves one portable Iceberg reference into a DataFusion
provider. It verifies the table UUID and binds the exact snapshot ID from
Relify metadata. It is compiled into the same native extension as DataFusion;
no provider crosses a Python or dynamic-library FFI boundary.

`relify-local` is the embedded DataFusion implementation. It composes the
catalog, warehouse, query, cache, coordination, and maintenance capabilities
behind `LocalSession`, and supplies the native runtime used by `Local`. The
name describes embedded execution, not a filesystem restriction. The local
builder writes Parquet, while the query resolver reads both Parquet and exact
Iceberg snapshots.

`relify.experimental.spark` is the Spark Classic implementation. It reads
Parquet relations through Spark and optionally binds one PyIceberg catalog
under the same logical catalog name used by Spark. Spark owns table reads,
bounded-sample MLlib block training, Arrow-batch assignment, the required
`(cid, key...)` range shuffle and partition-local sort, distributed Iceberg
data writes, and the native DataFrame query plan.
PyIceberg supplies exact table UUID and snapshot references and creates index
tables with the canonical schema before Spark appends data. The session never
creates a DataFusion context and does not use private PySpark JVM objects.

`relify.experimental.starrocks` is the StarRocks query implementation. It
binds a caller-owned Arrow Flight SQL ADBC connection and one PyIceberg catalog
under the same logical name already registered in StarRocks. PyIceberg
verifies table UUIDs and schemas at exact snapshots; the compiler emits fully
qualified StarRocks SQL with `VERSION AS OF` for every source and index table.
StarRocks owns centroid routing, postings pruning, source joins, filtering,
distance evaluation, Top-K, and distributed execution. The session neither
creates a DataFusion context nor transfers candidates through Python.

`parallite` provides the local partitioned execution context used by local IVF
construction.

The Python package exposes a stable local session and independent experimental
Spark and StarRocks sessions. Its modules separate configuration, catalog
access, asynchronous build state, maintenance, query intent, dialect
compilation, and session composition. All concrete tables reuse the same
table-centered lifecycle. Each session owns one `BuildCoordinator`;
independent sessions do not share a global build queue.
The coordinator pins the source relation at submission, validates builder
capabilities, owns status and publication, and retains only active operations
and operation failures.
Once publication succeeds, status is always loaded from the catalog. Python
sessions share one process-wide Tokio runtime; this is an execution resource,
not query or catalog state.

`relify.backends.v1` is the versioned public extension surface. Third-party
distributions register one lazy `BackendPlugin` through the
`relify.backends` package entry-point group. The plugin publishes typed static
capabilities and creates one concrete caller-owned session. A bound session
reports the currently available subset and reasons for declared capabilities
blocked by engine version or runtime configuration. Direct integration imports
remain the user API; the registry serves discovery and configuration-driven
applications. Index catalog implementations are not backend plugins.
`relify.open_index_catalog` exposes engine-independent index loading,
source-centered discovery, and selection without exposing native bindings;
concrete sessions use the same facade.

`relify.builders.v1` is the independent construction extension surface.
Builders advertise source/index profiles and accept an immutable
`BuildRequest` containing the pinned source reference, logical index
configuration, keys, and writer options. A builder returns portable
`BuildOutput`; the coordinator publishes it through `relify-index` and invokes
the supplied discard action if publication fails. `Local` builds
Parquet-to-Parquet through Rust. `Spark(spark_session)` builds
Iceberg-to-Iceberg and may be passed to a Spark, StarRocks, or other
Iceberg-bound table. Query backend capabilities contain no build matrix.

`VectorQuery` is an immutable backend-independent value containing a structured
table identifier and logical search options. It neither holds a session nor
imports DataFusion or Arrow. A concrete session resolves that identifier and
owns terminal compilation and execution. The local session exposes DataFusion,
Arrow, SQL, explain, and analyze terminals; the Spark session currently
exposes native PySpark DataFrame and explain terminals; and the StarRocks
session exposes StarRocks SQL and explain terminals. Every session collects a
portable `pyarrow.Table`, preserving the required result schema even when no
rows match.

For indexed queries, shared Python planning validates backend-independent
inputs and interprets the selected metadata snapshot into one immutable
`ResolvedSearch`. It contains exact portable source and index relation
references, query and index parameters, projection, predicate, and the derived
source-resolution requirement. Spark and StarRocks retain their own schema
mapping, relation resolution, native plan compilation, and execution. This
shares observable Relify semantics without introducing a universal physical
plan.

For queries, `relify-local` resolves catalog state into one immutable
`ResolvedSearch` and compiles the DataFusion plan. Each native `LocalSession`
owns one DataFusion `SessionContext`; Parquet I/O, build ETL, and query planning
share it. Python requests that native plan and wraps the returned DataFrame; it
does not register index relations or compile a second copy of the query.
The Python `Session` is the same mutable context instead of containing or
accepting another one. The complete DataFusion Python API is built into the same
extension and published under `relify.datafusion`; it does not depend on or
replace the separately distributed `datafusion` package.

`Session.register_parquet` uses DataFusion's normal named-table registration
and stores the versioned table definition in the session catalog. Reopening the
same catalog reconstructs the provider before lookup, so
`Session.table("documents")` resolves without another registration call.
Definitions are keyed by the fully qualified DataFusion table identifier and
cannot be silently rebound. The native execution binding retains the exact
registered `TableProvider`, so explicit schemas, partition columns, sort
information, builds, and queries all see the same table definition. A source
binding does not materialize or cache table data; scan behavior remains the
provider's responsibility. The Parquet definition codec and provider
reconstruction live in `relify-local`; Python passes typed registration options
and resolves table capabilities through the canonical identifier.

The local catalog registry is unified; the metadata authorities beneath it are
not. SQLite owns local Parquet definitions and Relify index mappings. In a
Spark session, the configured Iceberg catalog remains authoritative for source
and index tables, while the Relify index catalog stores only the current
metadata pointer. Relify metadata references exact Iceberg table snapshots
without copying their schemas, manifests, or file lists.

Relify registers a stateless Rust distance UDF as the logical expression and
generic execution fallback. A physical optimizer rule recognizes a supported
distance projection followed by distance-only Top-K and replaces both with one
`IvfTopKExec`. The fused operator consumes Arrow batches, reads vector buffers
directly, calls the runtime-selected SIMD squared-L2 kernel, and materializes
the remaining projection expressions only for retained rows. Its Top-K
threshold is shared across input partitions and accepted as a DataFusion
dynamic filter, so later batches can reject candidates without rebuilding
sort-key rows. Query and index validation occur before physical execution; the
per-vector hot loop does not repeat null, type, dimension, or finite-value
checks. Unsupported plan shapes retain the ordinary UDF, projection, and
Top-K operators. Runtime metrics separate distance computation, candidate
selection, final candidate sorting, and result projection, with counters for
candidate decisions and retained-batch memory.

Relify routes centroid matrices up to a bounded in-process size with its native
SIMD kernel; larger matrices remain in the relational plan as a centroid
distance projection, Top-K, and postings semi-join. For uncached Parquet
indexes, native routing attaches the selected CIDs as a static scan predicate.
The local writer stores postings in Hive-style `cid=<value>` partitions with
one Parquet file per non-empty cluster. It hash-partitions construction work by
`cid` first, so one cluster reaches exactly one Hive writer. The local reader
binds `cid` as a partition column, allowing DataFusion to remove unselected
files while planning the scan. A full probe needs neither centroid routing nor
a cluster predicate. IVF postings store exact vectors by default,
so key/vector-only projections without a source filter can compute and rank
results from index tables alone. Other projections and filters resolve source
rows by key. Callers can continue from the resulting lazy DataFrame directly or
compile complete SQL over the session's registered source and index relations.
Observable index metadata and query semantics remain in Rust.

Index caching is a session execution capability, not catalog state. The local
session materializes every relation in one current index snapshot as decoded
Arrow memory and transparently substitutes those relations while the snapshot
remains current. Cached IVF postings are coalesced up to the DataFusion batch
size and keep an immutable `cid`-to-range directory. Each query creates a
lightweight provider containing only the selected range descriptors; the scan
slices the shared Arrow batches lazily across the configured execution
partitions. It does not scan the complete cached relation, construct a filtered
`MemTable`, or copy vector buffers. The scan accepts DataFusion runtime filters
only for `cid` and `key_i` columns. Cached index-only searches use an optimized
DataFrame compiler beside the general SQL compiler; both consume the same
resolved search and shared source-to-index column mapping. Refresh, registration,
or removal invalidates the corresponding session cache; dropping the cache never
changes catalog metadata or durable index data.

## Current Scope

The local path uses SQLite, persistent Parquet table definitions, a Rust
builder, one Rust-owned DataFusion context, and optional exact Iceberg
resolution. `Local()` is its default builder. The experimental Spark path uses
one caller-owned Spark Classic session, optional Parquet sources, and one
matching PyIceberg catalog; `relify.experimental.Spark(session)` is its default
builder. Experimental StarRocks has no default builder and accepts an explicit
compatible builder. All sessions expose the root Relify index namespace, share
`relify-index`, and can query compatible physical index-table profiles
independent of which builder produced them.

Spark refresh, cross-driver build coordination, remote Relify index catalogs,
and Spark Connect are not current capabilities. Additional engines add
concrete session types around the shared metadata, `ResolvedSearch`, and
`VectorQuery` semantics. Independently distributed integrations register a
`BackendPlugin`; they do not inject a replaceable backend object into another
session. See [`python-api.md`](python-api.md) for implemented behavior and
[`backends.md`](backends.md) for the extension contract.

## Publication

Each local build writes Parquet index tables directly to a newly allocated
immutable warehouse prefix:

```text
<warehouse>/indexes/<index-uuid>/<snapshot-id>/
```

Spark instead creates named Iceberg tables in the configured `relify`
namespace and binds their exact snapshots. After the index tables are
complete, both paths create an immutable Relify metadata object under the
metadata warehouse. The catalog register or compare-and-swap commit is the
publication point. A failure before that point never exposes a partial index;
the builder removes newly created unpublished tables when its catalog supports
purge.

Catalog transactions and storage guarantees are independent. The warehouse
does not require atomic directory rename, so the same publication flow works
for local files, S3, and HDFS.

## Build, Query, and Maintenance Lifetimes

Resolving an indexed query is read-only. The plan selects one exact immutable
index snapshot and does not create a lease, acquire a maintenance reference, or
write session state.

Before construction, a native builder takes an exclusive cross-process
reservation for the complete structured index identifier. Coordination is
scoped to the exact catalog database, so distinct SQLite catalogs in one
directory cannot block each other. Once a build allocates an immutable snapshot
root, the reservation records that root until publication succeeds or the build
fails.

SQLite catalog commits and drops atomically record when their previous metadata
document lost catalog reachability. Orphan removal uses that timestamp rather
than file creation time and enforces a seven-day minimum retention period. It
rechecks catalog reachability before deletion. Recent uncommitted objects are
also retained for at least seven days, while active build reservations provide
additional protection for cooperating local builders.

This keeps query planning free of coordination writes. Retention is the
cross-process and remote-storage reader-safety boundary; a query that outlives
it is not guaranteed to survive collection of its selected snapshot.
