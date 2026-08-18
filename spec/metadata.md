# ParqDB Index Metadata Spec

## Overview

A ParqDB catalog identifier points to the current immutable metadata file for
one logical index. The catalog owns the index name and source-table binding.
The session owns one warehouse containing every metadata and index object.
None of those deployment-specific values is duplicated in index metadata.

Metadata files are immutable, self-contained UTF-8 JSON documents. A catalog
commit atomically changes which file is current; readers may continue using a
previously loaded file subject to the catalog's retention boundary.

## JSON Serialization

Metadata is one JSON object conforming to RFC 8259.

- Field names and string values are case-sensitive.
- UUIDs use lowercase `8-4-4-4-12` hexadecimal strings.
- `int` and `long` are exact signed 32-bit and 64-bit JSON integers.
- Timestamps are Unix epoch milliseconds stored as `long`.
- Maps have unique keys and lists preserve order.
- Unknown fields, duplicate keys, missing required fields, incorrect types,
  and out-of-range integers are invalid.

## Warehouse-relative Locations

Index relation locations and nested metadata locations are strings relative to
the session warehouse. A location:

- is non-empty and not absolute;
- uses `/`, never `\\`;
- contains no empty, `.`, or `..` path segment;
- contains no query or fragment; and
- may end in `/` to identify a directory relation.

The metadata file itself is identified by an absolute catalog location and
must also be inside the session warehouse. Metadata contains no `location`
field.

## Index Metadata

The root object contains:

| Requirement | Field | Type | Definition |
|---|---|---|---|
| required | `format-version` | `int` | Must be `1`. |
| required | `index-uuid` | `string` | Stable, non-nil UUID of the logical index. |
| required | `last-updated-ms` | `long` | Creation time of this metadata file. |
| required | `last-sequence-number` | `long` | Greatest sequence number ever allocated. |
| required | `current-snapshot-id` | `long` | Current retained snapshot. |
| required | `snapshots` | `list<index-snapshot>` | Current and retained snapshots. |
| required | `snapshot-log` | `list<snapshot-log-entry>` | Current-snapshot history. |
| optional | `properties` | `map<string,string>` | Non-semantic properties. |

`index-uuid` remains unchanged across refreshes. `last-updated-ms` is no
earlier than the base metadata, retained snapshots, or snapshot-log entries.
Properties do not affect interpretation or query results.

### Index Snapshot

Each snapshot contains:

| Requirement | Field | Type | Definition |
|---|---|---|---|
| required | `snapshot-id` | `long` | Positive ID unique within the index. |
| required | `sequence-number` | `long` | Positive commit sequence. |
| required | `timestamp-ms` | `long` | Snapshot creation time. |
| required | `summary` | `map<string,string>` | Non-semantic provenance; may be empty. |
| required | `vector-field` | `string` | Vector field in the catalog-bound source. |
| required | `source-key-fields` | `list<string>` | Ordered source key; non-empty and unique. |
| required | `indexed-rows` | `long` | Positive number of represented source rows. |
| required | `index-family` | `string` | Index-family identifier. |
| required | `index-schema-version` | `int` | Family schema version. |
| required | `metric` | `string` | Distance metric. |
| required | `parameters` | `map<string,string>` | Family-defined parameters. |
| required | `index-relations` | `map<string,string>` | Family roles to warehouse-relative Parquet locations. |

Metadata contains no source URI, source identifier, source version, source
snapshot, or source fingerprint. The catalog binds the metadata to one exact
runtime source relation.

Snapshot and sequence IDs are unique. No sequence exceeds
`last-sequence-number`. Snapshot list order has no meaning. The family spec
defines supported schema versions, metrics, parameters, and the exact set of
relation roles.

These logical identity fields remain equal across retained snapshots:

- `vector-field`;
- `source-key-fields`;
- `index-family`; and
- `metric`.

Changing an identity field creates a new index UUID. Row count, parameters,
schema version, and physical relations may change in a successor snapshot.

## Snapshot Log and Updates

Initial metadata has sequence number `1`, one current snapshot, and one log
entry. To publish a successor, a writer:

1. generates a new positive snapshot ID;
2. allocates `base.last-sequence-number + 1`;
3. appends one snapshot using that ID and sequence;
4. advances `last-sequence-number` and `current-snapshot-id`;
5. appends one snapshot-log entry; and
6. commits the fresh metadata location with catalog compare-and-swap.

Each log entry contains required `timestamp-ms` and `snapshot-id` fields. Log
timestamps are non-decreasing, every entry refers to a retained snapshot, and
the final entry refers to `current-snapshot-id`.

A metadata update may remove non-current snapshots and their log entries while
preserving `last-sequence-number`. Snapshot IDs are never reused.

## Catalog-owned Source Binding

For a selected index, a reader validates that the catalog-bound source:

1. contains `vector-field` and every `source-key-fields` field;
2. satisfies the family-defined vector and key type requirements; and
3. has exactly `indexed-rows` rows when an existing index is registered.

The row count and schema checks are compatibility guardrails, not a content
fingerprint. Registration is an administrative assertion that the selected
source contains the rows represented by the index. Query results preserve
source columns and add only family-defined result fields.

## Example

```json
{
  "format-version": 1,
  "index-uuid": "2f1c7f5e-3c43-4a44-8f2a-cf560c4db8d1",
  "last-updated-ms": 1750000000000,
  "last-sequence-number": 1,
  "current-snapshot-id": 701,
  "snapshots": [
    {
      "snapshot-id": 701,
      "sequence-number": 1,
      "timestamp-ms": 1750000000000,
      "summary": {"operation": "create"},
      "vector-field": "embedding",
      "source-key-fields": ["document_id"],
      "indexed-rows": 1000000,
      "index-family": "ivf",
      "index-schema-version": 1,
      "metric": "l2_squared",
      "parameters": {
        "dimension": "768",
        "nlist": "4096",
        "ntotal": "1000000",
        "posting_encoding": "lvq8",
        "ivf_centroids_fingerprint": "3ad6988a-389e-53de-aaa6-f210345fd894",
        "ivf_centroids_uuid": "fe985f6d-3592-4385-a1ca-71347057a210",
        "ivf_centroids_metadata_location": "metadata/fe985f6d-3592-4385-a1ca-71347057a210/v1.metadata.json"
      },
      "index-relations": {
        "ivf_centroids": "indexes/fe985f6d35924385a1ca71347057a210/1/ivf_centroids/",
        "ivf_postings": "indexes/2f1c7f5e3c434a448f2acf560c4db8d1/701/ivf_postings/"
      }
    }
  ],
  "snapshot-log": [
    {"timestamp-ms": 1750000000000, "snapshot-id": 701}
  ],
  "properties": {}
}
```
