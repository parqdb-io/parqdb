# ParqDB Index Metadata Spec

## Overview

A ParqDB catalog identifier points to the current metadata file for one logical
index. Metadata files are immutable, self-contained JSON documents. The
catalog identifier is not stored in metadata.

Metadata storage follows the model used by
[Apache Iceberg tables](https://iceberg.apache.org/spec/#table-metadata) and
[views](https://iceberg.apache.org/view-spec/#overview): object state is stored
in immutable metadata files and a catalog atomically tracks the current file.

ParqDB metadata describes the logical index. It does not replace or copy the
metadata of the source and index tables. An index snapshot binds one source
table to the exact states of the tables that contain its index data. Each
referenced table remains governed by its own relation profile.

### Metadata Location

The catalog stores the current metadata location for an index. A metadata
location is the absolute URI of an immutable metadata file. Readers continue
to use the metadata file they loaded until they refresh the index, subject to
the reader-safety retention boundary defined by the catalog implementation.

Catalog publication and table publication are independent. A catalog commit
makes one metadata file current but does not add snapshot or transaction
semantics to the tables referenced by that file. Those guarantees are
defined by each relation profile.

## Specification

### JSON Serialization

Metadata is one UTF-8 JSON object conforming to
[RFC 8259](https://www.rfc-editor.org/rfc/rfc8259).

- Field names and string values are case-sensitive.
- UUIDs use lowercase `8-4-4-4-12` hexadecimal strings.
- `int` and `long` are exact signed 32-bit and 64-bit JSON integers.
- Timestamps are Unix epoch milliseconds stored as `long`.
- Maps are JSON objects with unique keys; lists preserve order.
- Writers must not encode required behavior in unknown fields.

A reader must reject duplicate keys, missing required fields, values of the
wrong type, and out-of-range integers. It may ignore unknown fields in a
supported format version.

### Index Metadata

The root object contains:

| Requirement | Field name | Type | Description |
|---|---|---|---|
| required | `format-version` | `int` | Metadata format version; must be `1`. |
| required | `index-uuid` | `string` | Stable UUID of the logical index. |
| required | `location` | `string` | Base URI for metadata files. |
| required | `last-updated-ms` | `long` | Creation time of this metadata file. |
| required | `last-sequence-number` | `long` | Greatest snapshot sequence number ever allocated. |
| required | `current-snapshot-id` | `long` | Current index snapshot. |
| required | `snapshots` | `list<index-snapshot>` | Current and retained index snapshots. |
| required | `snapshot-log` | `list<snapshot-log-entry>` | Current-snapshot history. |
| optional | `properties` | `map<string, string>` | Non-semantic properties. |

`index-uuid` is generated at index creation, must be globally unique, and must
remain unchanged after a refresh. `location` must be an absolute URI without
user information, query, or fragment and must remain unchanged in format
version 1.

`last-updated-ms` must be no earlier than the base metadata file or any retained
snapshot or snapshot-log timestamp.

`properties` may be ignored and must not affect index interpretation, reference
resolution, or query results.

#### Index Snapshots

Each element of `snapshots` is one immutable logical index state and contains:

| Requirement | Field name | Type | Description |
|---|---|---|---|
| required | `snapshot-id` | `long` | Positive ID unique within the index. |
| required | `sequence-number` | `long` | Positive, monotonically increasing commit sequence. |
| required | `timestamp-ms` | `long` | Snapshot creation time. |
| required | `summary` | `map<string, string>` | Non-semantic provenance; may be empty. |
| required | `source` | `relation-reference` | Bound source table. |
| required | `vector-field` | `string` | Source vector field. |
| required | `source-key-fields` | `list<string>` | Ordered source unique key. |
| required | `index-family` | `string` | Index-family identifier. |
| required | `index-schema-version` | `int` | Index-family schema version. |
| required | `metric` | `string` | Distance metric. |
| required | `parameters` | `map<string, string>` | Family-defined parameters. |
| required | `index-relations` | `map<string, relation-reference>` | Family-defined index tables. |

String fields, field names, map keys, and parameter values must be non-empty
unless their defining section states otherwise. `source-key-fields` must be
non-empty and contain no duplicates. `summary` may be ignored and must not
affect query results.

Snapshot IDs and sequence numbers must be unique within `snapshots`, and no
sequence number may exceed `last-sequence-number`. Snapshot IDs identify
logical states; sequence numbers order commits. The two values are independent,
and the order of `snapshots` has no semantic meaning.

Format version 1 defines relation profiles `parquet` and `iceberg` and index
family `ivf`. The IVF family defines metrics `l2_squared` and `cosine`. An
index-family spec defines its supported schema versions, metrics, parameter
syntax, and index-table roles. Missing required roles are invalid; unknown
roles are allowed only when that family declares them optional.

A reader must reject an unsupported `format-version`, relation profile,
`index-family`, `index-schema-version`, or metric. It must not substitute a
supported profile, family, schema version, or metric.

The following logical-identity fields must remain equal across index snapshots:

- source profile and profile-defined stable identity;
- `vector-field`;
- `source-key-fields`;
- `index-family`; and
- `metric`.

Changing an identity field creates a new `index-uuid`. Changing a source table
snapshot, family parameters, schema version, or physical index table creates a
new snapshot of the existing index.

An index snapshot is a ParqDB metadata object, not an Iceberg table snapshot.
It may compose several independently committed table states. ParqDB
metadata stores only the references needed to bind that composition; table
schemas, partition specifications, manifests, data files, and table snapshot
history remain in the referenced table's metadata.

A retained index snapshot is usable only while its table references continue
to satisfy their profiles and the snapshot's schema and data constraints.

#### Relation References

A `relation-reference` is a JSON object discriminated by its required
`profile` string. The applicable relation profile defines every remaining
field, its serialization, locator, stable identity, exact-state semantics, and
host-engine resolution. A reference must contain exactly the fields defined by
that profile.

References identify logical tables, not their constituent files. Format
version 1 relation references are defined by
[`storage/iceberg.md`](storage/iceberg.md) and
[`storage/parquet.md`](storage/parquet.md).

Resolving a selected index snapshot requires the runtime resolution contexts
selected by its relation references. Iceberg references select a registered
Iceberg catalog by their `catalog` field; Parquet references use the Parquet
resolution context. Source and index tables may use different profiles or
different registered Iceberg catalogs. Catalog connections, resolution
contexts, and credentials are not stored in metadata. A reader must reject a
snapshot when a required registration or context is unavailable.

#### Snapshot Log

Initial metadata has one snapshot:

```text
last-sequence-number = 1
current-snapshot-id = S
snapshots = [snapshot S with sequence-number 1]
snapshot-log = [entry for snapshot S]
```

To create a snapshot, a writer:

1. generates a positive `snapshot-id` not previously used by the index;
2. allocates `base.last-sequence-number + 1`;
3. adds exactly one immutable snapshot with that ID and sequence number;
4. sets `last-sequence-number` and `current-snapshot-id`; and
5. appends one snapshot-log entry.

The catalog compare-and-swap serializes updates. A writer that loses a commit
race must reload metadata before allocating another sequence number. Snapshot
IDs are never reused, and `last-sequence-number` never decreases.

`current-snapshot-id` must identify exactly one retained snapshot. Snapshot
selection is either `current` or `exact(S)`, where `S` is a positive retained
ID. A non-positive `S` fails with `INVALID_QUERY_INPUT`; a missing snapshot
fails with `INDEX_SNAPSHOT_NOT_FOUND`. Implementations must not fall back to
another snapshot.

Each `snapshot-log-entry` contains:

| Requirement | Field name | Type | Description |
|---|---|---|---|
| required | `timestamp-ms` | `long` | Time the snapshot became current. |
| required | `snapshot-id` | `long` | Snapshot that became current. |

Log timestamps are non-decreasing and list order breaks equal-timestamp ties.
Every entry references a retained snapshot; the final entry references
`current-snapshot-id`.

A metadata update may remove non-current snapshots and their log entries while
preserving `last-sequence-number`. A rollback sets `current-snapshot-id` to a
retained snapshot and appends a log entry; it does not modify or create a
snapshot. The publisher validates the target snapshot's table availability
before commit.

### Source Binding

For a selected index snapshot, a reader verifies that the source table:

1. matches the identity and snapshot encoded by `source`;
2. contains `vector-field` and every `source-key-fields` field; and
3. satisfies the applicable index-family schema.

The reader must not substitute another source, index snapshot, table, or table
snapshot. An index is an access path for its source: public results
preserve source columns and add only the fields defined by the family query
spec.

When a relation profile separates its locator from stable identity, a runtime
source reference may use a different locator only when its stable identity and
exact state match the metadata source.

## Appendix A: Metadata Example

The following non-normative example defines one IVF index snapshot whose source
and index tables are resolved through the registered `lakehouse` Iceberg
catalog:

```json
{
  "format-version": 1,
  "index-uuid": "2f1c7f5e-3c43-4a44-8f2a-cf560c4db8d1",
  "location": "s3://warehouse/parqdb/documents_embedding",
  "last-updated-ms": 1750000000000,
  "last-sequence-number": 1,
  "current-snapshot-id": 701,
  "snapshots": [
    {
      "snapshot-id": 701,
      "sequence-number": 1,
      "timestamp-ms": 1750000000000,
      "summary": {
        "operation": "create"
      },
      "source": {
        "profile": "iceberg",
        "catalog": "lakehouse",
        "namespace": [
          "analytics"
        ],
        "name": "documents",
        "table-uuid": "1e2d3c4b-5a69-4788-9123-456789abcdef",
        "snapshot-id": 101
      },
      "vector-field": "embedding",
      "source-key-fields": [
        "document_id"
      ],
      "index-family": "ivf",
      "index-schema-version": 1,
      "metric": "l2_squared",
      "parameters": {
        "dimension": "2",
        "nlist": "2",
        "ntotal": "3",
        "posting_encoding": "source",
        "ivf_centroids_fingerprint": "73a6be1d-5c50-4f9f-a70b-035ca68b105d",
        "ivf_centroids_uuid": "fe985f6d-3592-4385-a1ca-71347057a210",
        "ivf_centroids_metadata_location": "s3://warehouse/parqdb/centroid-artifacts/fe985f6d/v1.metadata.json"
      },
      "index-relations": {
        "ivf_centroids": {
          "profile": "iceberg",
          "catalog": "lakehouse",
          "namespace": [
            "parqdb"
          ],
          "name": "documents_embedding_centroids",
          "table-uuid": "3a4b5c6d-7e8f-4901-a234-56789abcdef0",
          "snapshot-id": 201
        },
        "ivf_postings": {
          "profile": "iceberg",
          "catalog": "lakehouse",
          "namespace": [
            "parqdb"
          ],
          "name": "documents_embedding",
          "table-uuid": "4b5c6d7e-8f90-4a12-b345-6789abcdef01",
          "snapshot-id": 202
        }
      }
    }
  ],
  "snapshot-log": [
    {
      "timestamp-ms": 1750000000000,
      "snapshot-id": 701
    }
  ],
  "properties": {
    "owner": "search-team"
  }
}
```
