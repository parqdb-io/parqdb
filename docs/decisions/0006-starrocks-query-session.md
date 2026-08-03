# StarRocks Query Session

- Status: Accepted
- Date: 2026-07-30

## Context

Relify needs an OLAP reader that demonstrates one Spark-built Iceberg index can
be queried by another engine without export, conversion, or reconstruction.
StarRocks already owns distributed Iceberg scans, joins, filtering, Top-K, and
query scheduling. Relify should supply portable index discovery and query
semantics without introducing a generic backend container or moving candidate
data through Python.

StarRocks exposes Arrow Flight SQL from version 3.5.1. Its Iceberg catalog
supports exact snapshot queries through `VERSION AS OF`. A Relify reader must
also verify the table UUID and snapshot availability required by the Iceberg
relation profile; resolving only a StarRocks table name is insufficient.

## Decision

`relify.experimental.starrocks` is a third concrete session. It binds:

- one caller-owned Arrow Flight SQL ADBC connection;
- one PyIceberg catalog;
- the same logical catalog name already registered in StarRocks; and
- one Relify index catalog.

The StarRocks execution layer is query-only. It exposes table lookup,
`VectorQuery` compilation, collection, and explain. The table still shares
Relify's catalog lifecycle and may delegate construction to an independent
builder as defined by [ADR 0008](0008-independent-index-builders.md).
StarRocks itself does not build, refresh, cache, or garbage-collect index data,
create a StarRocks catalog, or use StarRocks' native vector indexes.

PyIceberg verifies each source and index table UUID, resolves the schema active
at the referenced snapshot, and rejects an unavailable snapshot. The
StarRocks compiler renders every table as its fully qualified identifier with
`FOR VERSION AS OF`. The logical catalog name is not translated.

The compiler emits ordinary StarRocks SQL:

- centroid Top-K remains a subquery;
- postings are pruned by joining selected cluster IDs;
- source predicates are evaluated in a source-only CTE before candidate
  resolution;
- source resolution is omitted for eligible key/vector-only projections;
- `l2_distance` computes squared L2 distance directly and the public
  `_distance` casts its result to `FLOAT`; and
- final results are ordered and limited by the public squared distance.

The ADBC connection remains owned by the caller. Relify opens and closes a
cursor per operation and returns one Arrow table. It validates the result field
order, `_distance` type, nulls, and finite values before returning a result.
The table retains its schema when the query returns no rows.

The first implementation accepts only Iceberg relation references and a SQLite
development index catalog. Parquet `FILES()`, a remote index catalog, analytical
SQL composition helpers, construction using StarRocks compute, refresh, cache,
and garbage collection are separate future work.

## Consequences

Spark can build canonical Iceberg index tables and publish Relify metadata,
while StarRocks reads the same exact table snapshots and executes the complete
IVF query in its own distributed runtime. Neither `relify-local` nor DataFusion
is instantiated in a StarRocks session.

The query-intent and repository boundaries remain backend-independent;
StarRocks-specific SQL and ADBC behavior remain isolated under
`relify.experimental.starrocks`. A real integration test loads the shared specification
fixtures into Iceberg and compares StarRocks results with their expected
outputs.

SQLite is sufficient for one development client but is not production
coordination for concurrent writers. Production readiness still requires a
remote Relify index catalog, a maintained StarRocks/Iceberg conformance
environment, and reproducible operational benchmarks.
