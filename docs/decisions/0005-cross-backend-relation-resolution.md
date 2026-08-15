# Cross-Backend Relation Resolution

- Status: Accepted
- Date: 2026-07-30

## Context

Relify metadata already identifies every source and index table with a portable
`RelationReference`. The first DataFusion implementation nevertheless assumed
that every reference was Parquet, while the first Spark implementation assumed
that every reference was Iceberg. The metadata was portable, but a physical
index could not actually move between the two query runtimes.

## Decision

Query sessions resolve each relation from its metadata profile:

- DataFusion reads Parquet URIs through `relify-storage` and reads Iceberg
  references through a native Iceberg `TableProvider` compiled into the same
  DataFusion extension.
- Spark reads Parquet URIs through `SparkSession.read.parquet` and reads
  Iceberg references through the caller's registered Spark and PyIceberg
  catalog.

Iceberg reads verify the table UUID and select the exact referenced snapshot.
Parquet reads preserve the canonical URI as the relation state. The Relify
catalog stores neither engine-specific plans nor converted copies of index
tables.

One session binds at most one logical Iceberg catalog name. Metadata that
references another catalog fails resolution instead of being silently rebound.
Each runtime may expose different construction capabilities. The current local
builder writes Parquet; compatible Iceberg indexes may be published by a
conforming external builder.

## Consequences

A compatible Iceberg index can be queried by DataFusion, and a locally built
Parquet index can be queried by Spark, provided both sessions use the same
Relify index catalog and can access the referenced storage. Cross-backend
interoperability does not require another index format, export step, or metadata
field.

DataFusion's Iceberg integration is a shared Rust component rather than a
Python DataFusion bridge. This preserves one execution context and avoids
cross-library `TableProvider` FFI.
