# Core Concepts

Relify keeps vector indexes as table data and executes search inside a SQL
runtime. The persisted index, catalog state, and execution runtime have separate
responsibilities.

## Source Table

A source table contains the original vectors and payload columns. Relify does
not copy those rows into a private database. Each indexed source declares one
or more key columns that join index candidates back to source rows.

The embedded runtime persists Parquet source definitions. Iceberg references
identify an exact table UUID and snapshot when a PyIceberg catalog is bound.

## Open Vector Index

An index consists of immutable JSON metadata and ordinary relations. The IVF
family stores:

- centroids used to choose candidate clusters; and
- postings that map clusters to source keys and, for LVQ4 or LVQ8, quantized
  vector codes.

Logical indexes over the same source state, vector field, metric, and cluster
count may share one immutable centroid artifact while keeping independent
postings. The [specification](../spec/README.md) defines the metadata and table
schemas independently of the Python runtime.

## Catalog and Snapshots

The catalog maps a registered source and index name to the current immutable
metadata document. The embedded implementation uses SQLite. Posting rows and
vectors remain in Parquet or Iceberg rather than in the catalog.

Every successful build or refresh publishes a new immutable snapshot. The
catalog update is the publication point, so readers see either the previous
snapshot or the complete replacement, never partially written relations.

Parquet locations do not provide source snapshot isolation. Applications must
not replace registered Parquet files in place while an index depends on them.

## Session and Execution

`relify.connect` returns a portable session facade. The embedded transport calls
the session service directly; the service owns a private DataFusion host and a
native Rust `LocalSession`. Public `Session` and `SourceTable` objects do not
inherit DataFusion classes.

`VectorQuery` is immutable and contains only table identity and search options.
Execution returns Arrow data. Embedded-only DataFusion operations are available
through the explicit `session.datafusion_context()` escape hatch.

## Query Lifecycle

An indexed query has four logical stages:

1. select the nearest IVF centroids;
2. scan postings for those cluster IDs;
3. apply source filters and compute distance; and
4. order by distance and retain the requested limit.

Relify may fuse or prune these stages physically without changing the index
format or result semantics.
