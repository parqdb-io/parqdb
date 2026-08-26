# Parquet Provider Profile

## Overview

The `parquet` table and index providers represent source and index tables as
Parquet files resolved by the host engine.

Runtime storage configuration supplies access to the URI schemes used by the
selected index snapshot. Credentials are not stored in provider definitions.

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

For IVF schema version `1`, an `ivf_postings` Parquet table is discovered
from exactly one `manifest.json` below the table root. Readers must not list
the table prefix. The strict manifest contains `format-version`, `nlist`,
`ntotal`, the hierarchy's `cid-offsets`, and the complete ordered `files`
inventory. Each file entry contains `path`, `cid-bucket`, inclusive `min-cid`
and `max-cid`, `rows`, `size`, and the lowercase whole-object `sha256`.

The canonical object paths are
`cid_bucket=<six-digit-root-id>/part-<five-digit-sequence>.parquet`.
`cid_bucket` is physical layout metadata and is not exposed as a table
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

## Provider Definitions

A Parquet source uses a `table-definition`:

```json
{
  "identifier": {
    "catalog": "datafusion",
    "namespace": ["public"],
    "name": "documents"
  },
  "provider": "parquet",
  "properties": {
    "definition-version": "1",
    "location": "<absolute table URI or URI pattern>",
    "table-identity": "<stable logical identity>"
  }
}
```

`identifier` is the runtime table name. Provider properties are versioned,
non-secret strings. `location` is canonical and may contain `*` wildcards for
a source table; the pattern itself is persisted rather than an expanded file
list. `table-identity` is the provider's stable semantic identity and defaults
to the serialized logical identifier when omitted. An implementation may
persist additional versioned properties required to reconstruct schema,
partition columns, sort order, and scan options.

A Parquet index uses an `index-provider-definition` with `provider` equal to
`parquet`. Its `index-table-definition` has `definition-version` equal to `1`
and a required `location` property. The optional `layout` value
`artifact-manifest` means `location` names the authoritative top-level
`manifest.json`; otherwise it names a Parquet table directly. Index-table
locations are warehouse-relative unless the provider explicitly supports an
absolute external location.

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

For a Parquet source, metadata captures provider properties rather than a file
inventory. It does not capture a content hash or snapshot. A reader validates
location and schema but cannot detect content replacement at the same location
or changes to the set of files matched by the same pattern.
