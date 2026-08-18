# ParqDB Spec

ParqDB is a library for building and querying vector indexes over existing data
tables. This specification defines the portable metadata, table schemas,
and query semantics shared across compute engines.

## Background and Motivation

Vector indexes are commonly stored in engine-specific formats and cannot be
reused by another engine that accesses the same source data. ParqDB standardizes
the metadata, table schemas, and query semantics required to share vector
indexes across engines.

## Goals

- Keep source data in existing host-engine tables.
- Store index structures in open table formats.
- Allow different engines to discover and query the same logical index.
- Reuse Iceberg catalogs and host-engine storage readers and query execution.

## Overview

A ParqDB index is an access path for a source table. The source table remains
authoritative for source rows. Index tables contain auxiliary data
used to accelerate queries, such as centroids and postings.

Each logical index has a catalog identifier. The catalog binds that index to a
source table and points to the current immutable metadata file. Each index
snapshot describes the warehouse-relative index data for that binding.

```text
catalog identifier
    -> source table
    -> index metadata
        -> warehouse-relative index tables
```

The runtime resolves the catalog-bound source through its host engine and
index relations through the session warehouse. Query results preserve source
columns and add index-family result fields.

All indexes are published and loaded through a catalog. A catalog commit
publishes metadata only; table consistency is defined by the applicable
relation profile. Iceberg references bind exact table snapshots. Parquet
references identify tables by URI and rely on publisher-managed
consistency.

## Specification

### Terms

- **Index** -- A logical access path for a source table.
- **Index identifier** -- An Iceberg-style namespace and name that identify a
  logical index within a catalog.
- **Index snapshot** -- An immutable logical state for an index family and its
  warehouse-relative index tables. The catalog owns the source binding.
- **Source table** -- The host-engine table whose rows are indexed.
- **Index table** -- A table that stores auxiliary data for an index
  family.
- **Metadata file** -- An immutable, self-contained JSON document that stores
  index state and snapshot history.
- **Catalog** -- A naming layer that maps an index identifier to its current
  metadata file and source table.
- **Warehouse** -- The single URI prefix used by a session for all index
  metadata and data.
- **Iceberg catalog** -- An Iceberg catalog registered at runtime under the
  logical name used by Iceberg relation references.
- **Resolution context** -- Runtime configuration used by a relation profile to
  resolve relation references.
- **Relation reference** -- A profile-tagged, portable identifier for one
  logical table and, when supported by the profile, its exact state.
- **Relation profile** -- Rules for resolving a relation reference and
  interpreting its identity, exact state, and storage guarantees.

### Type System

Table schemas use the
[Apache Iceberg type system](https://iceberg.apache.org/docs/latest/schemas/)
as their canonical type system.

Common primitive types are:

| Type | Definition |
|---|---|
| `boolean` | Boolean value. |
| `int` | Signed 32-bit integer. |
| `long` | Signed 64-bit integer. |
| `float` | IEEE 754 binary32 value. |
| `double` | IEEE 754 binary64 value. |
| `decimal(P, S)` | Fixed-point decimal. |
| `date` | Date without time or time zone. |
| `time` | Time without date or time zone. |
| `timestamp` | Timestamp without time zone. |
| `timestamptz` | UTC-adjusted timestamp. |
| `string` | UTF-8 string. |
| `uuid` | Universally unique identifier. |
| `fixed(L)` | Fixed-length binary. |
| `binary` | Variable-length binary. |

Iceberg also defines `struct<...>`, `list<T>`, and `map<K, V>`. Struct fields,
list elements, and map values define nullability independently. Map keys are
required.

Relation profiles define physical mappings for non-Iceberg storage. Schema
conformance is determined from the underlying table or file schema. A compute
engine may expose a conservatively nullable query schema; that query schema
does not change the nullability encoded by the storage format.

### Table Schema Compatibility

Field names identify top-level table fields and are compared as exact
sequences of Unicode code points without case folding or normalization. Format
version 1 does not define nested field paths.

A table satisfies a required schema when:

- every required field is present with the exact field name;
- its canonical type, type parameters, and nullability match exactly; and
- collection element types and nullability match exactly.

Field order is not significant. A table may contain additional fields unless
an index-family spec forbids them. Readers may ignore additional fields, and
writers must not encode required behavior in them. Host-engine coercions do not
change schema compatibility.

### Error Conditions

An operation described as failing or producing an error terminates without
fallback or partial results. Error identifiers in this spec name semantic
conditions; concrete exception, status, and protocol representations are
implementation-specific.

### Specification Index

Core:

- [`metadata.md`](metadata.md): index metadata format and snapshots.
- [`catalog.md`](catalog.md): catalog state and atomic metadata commits.

Relation profiles:

- [`storage/parquet.md`](storage/parquet.md)
- [`storage/iceberg.md`](storage/iceberg.md)

IVF:

- [`ivf/index-schema.md`](ivf/index-schema.md)
- [`ivf/query.md`](ivf/query.md)

Non-normative test vectors:

- [`fixtures/v1/`](fixtures/v1/): valid and invalid metadata, Parquet tables,
  LVQ codes, and ordered IVF query results for format version 1.
