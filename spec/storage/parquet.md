# Parquet Relation Profile

## Overview

This profile represents source and index tables as Parquet files resolved by
the host engine. Relify does not parse Parquet.

The resolution context for this profile supplies host-engine access to the URI
schemes used by the selected index snapshot.

Parquet provides no table UUID, snapshot identity, or portable multi-file
transaction. Relify metadata that references Parquet is still published through
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

For IVF schema version `1`, an `ivf_postings` Parquet relation must use `cid`
as a Hive-style partition column. Every non-empty cluster is stored as exactly
one Parquet file below a directory named `cid=<value>`. The `cid` field is
omitted from the Parquet file schema and reconstructed as a required `INT32`
partition column by the reader.

This layout lets a reader resolve a selected set of clusters to files while
planning the scan. It does not change the logical `ivf_postings` schema in the
IVF index specification.

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
