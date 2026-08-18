# ParqDB Catalog Spec

## Overview

A catalog maps an identifier to the current metadata location and source-table
binding of one ParqDB index. It owns naming, discovery, registration, and
metadata publication. It does not execute index queries or store a per-index
warehouse.

This spec defines operation semantics, not an API, SQL syntax, metastore
schema, or network protocol.

A ParqDB index catalog is supplied to a ParqDB operation at runtime. An index
identifier uses the
[Apache Iceberg `TableIdentifier` model](https://iceberg.apache.org/javadoc/latest/org/apache/iceberg/catalog/TableIdentifier.html):
an ordered namespace and a name. Namespace segments and names are non-empty
UTF-8 strings and are compared as exact sequences of Unicode code points. The
index catalog name identifies the runtime index catalog and is not part of the
index identifier. It is distinct from the logical Iceberg catalog name stored
in an Iceberg relation reference.

This spec does not define a dotted textual representation because catalog,
namespace, and name segments may require host-specific quoting. Catalog identity
and index identifiers are not stored in index metadata.

A **metadata location** is the absolute URI of an immutable metadata file. A
**base metadata location** is the location on which an update was based.
Every metadata location belongs to the single warehouse configured for the
session using the catalog.

## Catalog Operations

A read-only catalog implements `load`. A mutable catalog also implements
`register`, `commit`, and `drop`. `rename` and `list` are optional. These names
describe semantics and do not prescribe public method names.

| Operation | Semantics |
|---|---|
| `load(identifier)` | Return the current metadata location and source binding, or `INDEX_NOT_FOUND`. |
| `register(identifier, source, metadata-location)` | Create a source-bound mapping only if the identifier is absent. |
| `commit(identifier, source, base, new)` | Replace the current location only if it equals `base`. |
| `drop(identifier)` | Remove the mapping, or `INDEX_NOT_FOUND`. |
| `rename(source, destination)` | Move a mapping if the source exists and destination does not. |
| `list(namespace)` | Return identifiers visible in the namespace, or `NAMESPACE_NOT_FOUND`. |

Mutating operations are atomic for each identifier they modify. A `load` of an
affected identifier observes either the state before or the state after a
successful mutation, never an intermediate state. This spec does not require a
global order across different identifiers or a transactional `list`.

`load` returns one complete metadata file and must not combine files or
substitute an older location.

A successful `register` or `commit` must reference metadata that satisfies:

- the metadata format and snapshot invariants in
  [`metadata.md`](metadata.md);
- warehouse-relative location syntax and family-defined index-table roles; and
- for `commit`, unchanged `index-uuid` and logical identity fields.

Data-dependent validation belongs to the publisher, not the catalog.

## Commit

To update an index, a writer:

1. loads the current metadata location and metadata;
2. completes and validates new index tables;
3. writes a new complete metadata file to a fresh location; and
4. commits the new location using the loaded location as `base`.

The successful compare-and-swap in step 4 is the publication point. Readers
that already loaded the previous file continue to observe its immutable state.

The commit publishes only the metadata file and source binding. It does not
atomically commit, snapshot, or retain the source table or warehouse objects.

When concurrent commits use the same base, at most one succeeds. A losing
writer reloads current metadata and reapplies its change only if it remains
valid. A catalog without this guarantee must be read-only.

The compare-and-swap mechanism is implementation-specific. Iceberg describes
the same model for
[metastore-backed metadata commits](https://iceberg.apache.org/spec/#metastore-tables).

## Index Lifecycle

Creating an index writes initial metadata with a new `index-uuid`, then
registers it under an absent identifier and source binding. Registering an
existing metadata file can attach the same immutable index to a catalog-bound
source without changing index identity or copying warehouse objects.

The runtime name of the ParqDB index catalog is not stored in metadata. Rename
therefore leaves metadata, `index-uuid`, snapshots, and index data unchanged.

`drop` removes catalog visibility but does not delete metadata or data.
Garbage collection may remove an object only after no catalog mapping or
retained metadata references it and the implementation's reader-safety
retention period has elapsed. The retention period begins when a published
object loses catalog reachability, not when its files were created. A reader
that outlives that period is not guaranteed to complete after collection.

Failed registers and commits may leave reclaimable orphan objects. An
implementation must conservatively retain objects that may belong to an
in-progress operation. This spec does not define reader tracking,
garbage-collection bookkeeping, or retention duration.

## Errors

| Error | Condition |
|---|---|
| `NAMESPACE_NOT_FOUND` | A required namespace does not exist. |
| `INDEX_NOT_FOUND` | The identifier does not exist. |
| `ALREADY_EXISTS` | A register or rename destination exists. |
| `INVALID_INDEX_METADATA` | Metadata violates format or snapshot invariants. |
| `UNSUPPORTED_FORMAT_VERSION` | The metadata format version is unsupported. |
| `INDEX_UUID_MISMATCH` | New metadata belongs to another index. |
| `COMMIT_CONFLICT` | The supplied base location is no longer current. |

Concrete exception and protocol representations are implementation-specific.
