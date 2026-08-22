# Iceberg Table Provider Profile

## Overview

The `iceberg` table provider represents a source as one exact Apache Iceberg
snapshot. A native provider reads the pinned metadata file and exposes the
snapshot to the host engine. This profile does not define an Iceberg index
provider.

The ParqDB catalog tracks table definitions and index metadata independently of
an external Iceberg catalog. Catalog endpoints, credentials, and object-store
configuration remain runtime-only.

## Type System

Iceberg table schemas use ParqDB's canonical Iceberg types directly. The
reader verifies type, nullability, and collection-element requirements through
the host engine.

## Table Definition

An exact Iceberg source uses this `table-definition` shape:

```json
{
  "identifier": {
    "catalog": "datafusion",
    "namespace": ["analytics"],
    "name": "documents"
  },
  "provider": "iceberg",
  "properties": {
    "definition-version": "1",
    "location": "<absolute Iceberg metadata-file URI>",
    "table-identity": "<Iceberg table UUID>",
    "option.table-uuid": "<Iceberg table UUID>",
    "option.snapshot-id": "123"
  }
}
```

`identifier` is the logical runtime name. Every catalog, namespace, and name
segment is a non-empty UTF-8 string. `location` pins one immutable Iceberg
metadata file. `table-identity` and `option.table-uuid` contain the same
lowercase Iceberg table UUID; `option.snapshot-id` is the positive decimal ID
of a snapshot retained by that metadata file.

A reader loads `location`, verifies the UUID, and reads exactly the retained
snapshot ID. A missing metadata file, UUID mismatch, or unavailable snapshot is
an error. Two table definitions have the same semantic identity when their
provider and `table-identity` match; the full definition fingerprint also
includes the pinned metadata location and snapshot.

## Publication and Consistency

The writer completes and validates all referenced snapshots before committing
new ParqDB metadata. The metadata commit is the publication point.

No transaction is required across the source table, index tables, and ParqDB
catalog. Consistency comes from publishing the exact table definition in one
immutable metadata file. Failed commits may leave orphan metadata or Iceberg
snapshots.

Iceberg remains authoritative for the table UUID, schema, partition
specification, snapshots, manifests, and files. ParqDB records only the
properties required to verify and reopen the selected state.

A reader that cannot read an exact snapshot must reject this provider rather
than read another snapshot.
