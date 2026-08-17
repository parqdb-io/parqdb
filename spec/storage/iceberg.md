# Iceberg Relation Profile

## Overview

This profile represents source and index tables as Apache Iceberg tables.
An Iceberg catalog resolves table metadata, and a host engine reads the
resolved tables and executes queries. ParqDB does not implement an Iceberg
table reader.

The ParqDB index catalog tracks index metadata. Iceberg catalogs independently
resolve the data tables referenced by an index snapshot. At runtime, each
Iceberg catalog is registered under the logical name stored in its relation
references.

## Type System

Iceberg table schemas use ParqDB's canonical Iceberg types directly. The
reader verifies type, nullability, and collection-element requirements through
the host engine.

## Relation Reference

An Iceberg relation reference contains exactly:

```json
{
  "profile": "iceberg",
  "catalog": "lakehouse",
  "namespace": ["analytics"],
  "name": "documents",
  "table-uuid": "<Iceberg table UUID>",
  "snapshot-id": 123
}
```

`catalog` is the non-empty UTF-8 logical name under which the Iceberg catalog
is registered with the reader. `namespace` is the ordered sequence of
namespace segments and may be empty. Each segment and `name` must be a
non-empty UTF-8 string. Together, `catalog`, `namespace`, and `name` form the
complete runtime identifier of the table.

Format version 1 does not define catalog aliases or name translation. A reader
uses `catalog` unchanged to select its runtime catalog registration. A host
engine used by that reader must expose the same catalog under the same name.

`table-uuid` is the lowercase textual Iceberg table UUID. `snapshot-id` is
the exact table snapshot used by the ParqDB index snapshot.

The resolution context for this profile is a registry from logical catalog
names to Iceberg catalog implementations. A reader resolves each reference
through its named catalog, verifies `table-uuid`, and reads exactly
`snapshot-id`. An unregistered catalog, missing table, UUID mismatch, or
unavailable snapshot is an error.

Catalog registration is runtime configuration. Catalog endpoints, credentials,
and implementation-specific properties are not stored in index metadata. Two
readers may use different implementations or credentials for the same logical
catalog name, but that name must resolve the same table identities referenced
by the metadata.

`namespace` and `name` locate a table. `table-uuid` is its stable identity, and
`snapshot-id` identifies its exact state. Two source references match when
their `table-uuid` and `snapshot-id` are equal; their locators may differ after
a table rename. A later index snapshot may update the locator without changing
the logical index identity.

## Publication and Consistency

The writer completes and validates all referenced snapshots before committing
new ParqDB metadata. The metadata commit is the publication point.

No transaction is required across the source table, index tables, and ParqDB
catalog. Consistency comes from publishing immutable table UUID and snapshot
references in one metadata file. Failed commits may leave orphan metadata or
Iceberg snapshots.

Iceberg remains authoritative for each referenced table's UUID, schema,
partition specification, snapshots, manifests, and files. ParqDB metadata does
not duplicate those fields; it records only the exact relation references that
compose one logical index snapshot.

A reader that cannot read an exact snapshot must reject this profile rather
than read another snapshot.
