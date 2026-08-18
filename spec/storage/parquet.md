# Parquet Relation Profile

## Overview

This profile represents source and index tables as Parquet files resolved by
the host engine. ParqDB does not parse Parquet.

The resolution context for this profile supplies host-engine access to the URI
schemes used by the selected index snapshot.

Parquet provides no table UUID, snapshot identity, or portable multi-file
transaction. ParqDB metadata that references Parquet is still published through
a catalog. The publisher is responsible for source consistency, complete
table contents, and reader-writer coordination.

## Type Mapping

| Iceberg type | Parquet representation |
|---|---|
| `boolean` | `BOOLEAN` |
| `int` | `INT32` |
| `long` | `INT64` |
| `float` | `FLOAT` |
| `double` | `DOUBLE` |
| `string` | `BYTE_ARRAY` annotated as `STRING` |
| `uuid` | `FIXED_LEN_BYTE_ARRAY(16)` annotated as `UUID` |
| `binary` | Unannotated `BYTE_ARRAY` |
| `fixed(L)` | `FIXED_LEN_BYTE_ARRAY(L)` |
| `date` | `INT32` annotated as `DATE` |
| `list<string>` | Parquet `LIST` with required `STRING` elements |
| `list<float>` | Parquet `LIST` with required `FLOAT` elements |
| `list<double>` | Parquet `LIST` with required `DOUBLE` elements |
| `map<string, string>` | Parquet `MAP` with required `STRING` keys and values |
| `map<string, long>` | Parquet `MAP` with required `STRING` keys and `INT64` values |
| `map<string, uuid>` | Parquet `MAP` with required `STRING` keys and UUID-annotated `FIXED_LEN_BYTE_ARRAY(16)` values |

Family schemas determine which mappings are used.

## IVF Postings Layout

For IVF schema version `1`, an `ivf_postings` Parquet relation is discovered
from exactly one `manifest.json` below the relation root. Readers must not list
the relation prefix. The strict manifest contains `format-version`, `nlist`,
`ntotal`, the hierarchy's `cid-offsets`, and the complete ordered `files`
inventory. Each file entry contains `path`, `cid-bucket`, inclusive `min-cid`
and `max-cid`, `rows`, and `size`.

The canonical object paths are
`cid_bucket=<six-digit-root-id>/part-<five-digit-sequence>.parquet`.
`cid_bucket` is physical layout metadata and is not exposed as a relation
column. Each file belongs to one hierarchical root and may contain multiple
consecutive CIDs. The required physical `INT32` `cid` column remains in every
Parquet file.

Each row group contains exactly one non-null CID, although one CID may span
consecutive row groups and files. Exact `cid` min/max statistics are required
in every row group. A reader first intersects selected CIDs with manifest file
ranges, then reads candidate footers and attaches an explicit row-group access
plan. A residual `cid` predicate remains a correctness check; expression
optimizer handling of a large `IN` list must not control physical pruning.

Writers create all data files before creating `manifest.json` with
create-if-absent semantics. File sizes and row totals in the manifest must
match the completed objects. This layout does not change the logical
`ivf_postings` schema in the IVF index specification.

## Relation Reference

A Parquet relation reference contains exactly:

```json
{
  "profile": "parquet",
  "uri": "<absolute table URI or URI pattern>"
}
```

No other field is defined. The canonical `uri` is the table identity and is
compared byte-for-byte. A source URI may contain `*` wildcards in its path.
The pattern itself is the identity; metadata does not expand it into a file
list. Index-table writers should use concrete URIs.

The URI must:

1. use a lowercase scheme;
2. contain no user information, query, fragment, `.` segment, `..` segment, or
   repeated path separator;
3. lowercase a DNS host name;
4. use uppercase hexadecimal percent encodings;
5. not encode unreserved characters or `/`; and
6. identify one table root or one `*` wildcard pattern.

A trailing `/` is significant. Readers may support a subset of URI schemes and
must reject unsupported schemes. The URI must resolve to one stable logical
table for the duration of a query.

## Publication and Consistency

A catalog commit atomically publishes metadata, not the referenced Parquet
contents. Before making metadata current, the publisher must ensure that every
referenced table is complete and satisfies the selected index snapshot.
Schema validation uses the Parquet file schema, not a compute engine's inferred
query schema. The publisher performs this validation before publication;
readers are not required to reopen Parquet footers for every query.

A publisher may replace a Parquet table in place using a host operation such
as `INSERT OVERWRITE`. This profile provides no isolation for a reader that
overlaps that replacement, no atomicity across multiple tables, and no
recovery guarantee after a partial write. The publisher must coordinate those
operations externally.

Writing the index tables for a new snapshot to fresh URIs and retaining old
contents can provide stable reads and usable history, but this profile does not
require it. A writer that replaces a URI must not assume that index snapshots
referring to the old contents remain readable.

For a Parquet source, metadata captures only its URI or URI pattern. It does not
capture a file listing, content hash, or snapshot. A reader validates URI and
schema but cannot detect content replacement at the same URI or changes to the
set of files matched by the same pattern.
