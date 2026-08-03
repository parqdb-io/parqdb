# Backend and Compute Crate Boundaries

- Status: Accepted
- Date: 2026-07-30

## Context

The initial Rust implementation placed backend-neutral query models, embedded
DataFusion execution, Parquet I/O, SQLite session composition, SIMD distance
kernels, and K-means training in `relify-core`. That made the local
implementation usable, but it forced a future backend to depend on another
backend's execution engine and storage stack.

K-means and query execution also shared the same numerical kernels. Moving only
K-means would either duplicate those kernels or make an embedded query backend
depend on a clustering algorithm for generic distance computation.

## Decision

The workspace uses these dependency boundaries:

- `relify-core` defines backend-neutral construction options, query intent,
  portable build artifacts, and publication results.
- `relify-local` implements embedded execution with DataFusion, Parquet, the
  SQLite catalog, local coordination, caching, and maintenance.
- `relify-kmeans` owns deterministic sampling, Lloyd training, centroid
  assignment, and empty-cluster recovery.
- `relify-kernels` owns shared SIMD distance, row-norm, and GEMM primitives.
- `parallite` remains the local partitioned execution runtime used by
  `relify-kmeans`.

`relify-core` does not define a universal backend trait. Each execution backend
compiles the shared query intent into its own plan model. The SQLite
implementation in `relify-catalog` is feature-gated so consumers of the catalog
interfaces and identifiers do not have to link SQLite.

At the Python boundary, `VectorQuery` is an immutable logical value containing
a structured table identifier rather than a table, session, or engine plan.
Terminal compilation and execution methods belong to each concrete session.

## Consequences

Future DataFusion-independent backends can depend on `relify-core`,
`relify-meta`, and `relify-catalog` without pulling in DataFusion, Parquet,
object stores, SQLite, or local numerical execution. The embedded Python
extension depends on `relify-local`, while its public Python API remains
backend-independent at the query-intent boundary.

K-means can be tested and evolved independently of Arrow, metadata, catalogs,
and storage. Query execution and K-means continue to share one implementation
of the performance-sensitive numerical kernels.
