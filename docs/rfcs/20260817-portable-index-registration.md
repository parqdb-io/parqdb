# Catalog-Owned Index Registration

## Problem

ParqDB metadata currently binds an index snapshot to physical source and index
relation URIs. A copied index cannot be attached to a catalog at a new location
without preserving the original paths or rewriting immutable metadata.

ParqDB needs one explicit operation that registers an existing index without
training centroids, rebuilding postings, copying files, or rewriting metadata.
The catalog should own the association between a registered source table and
the index. Index metadata should describe only the index.

## Decisions

| Question | Decision |
| --- | --- |
| Does index metadata identify its source? | No. It contains no source URI, source ID, source version, or source content fingerprint. |
| Who associates an index with a source table? | The catalog. |
| What source compatibility is recorded? | The fields already used by the index and snapshot-level `indexed-rows`. |
| Who stores physical locations? | The catalog stores the metadata location and artifact root. |
| Does registration rewrite or copy files? | No. |
| Is directory scanning supported? | No. The caller supplies one metadata location. |
| Where is the public API? | `SourceTable.register_index(name, metadata_location=...)`. |
| Does the catalog need a separate import primitive? | No. Registration reuses the existing `IndexCatalog::register`. |

## Scope

This RFC defines:

- source-free index metadata for new metadata versions;
- relative index artifact locations;
- a catalog-owned source-to-index association;
- an `indexed-rows` snapshot field; and
- validation and atomic publication of an existing index.

It does not define warehouse scanning, metadata rewriting, file relocation,
source identity, source content hashing, or repair of incomplete artifacts.

## 1. Metadata Contract

A portable index snapshot does not contain a `source` relation reference. Its
source-dependent fields are limited to the fields the index actually uses:

- `vector-field`;
- `source-key-fields`; and
- `indexed-rows`.

`indexed-rows` is a required positive `long` containing the number of source
rows represented by that snapshot:

```json
{
  "vector-field": "embedding",
  "source-key-fields": ["document_id"],
  "indexed-rows": 1000000000
}
```

The metadata does not copy the complete source schema. Unreferenced source
columns have no effect on registration.

The existing index-family fields continue to describe vector type and
dimension requirements, distance metric, encoding, and artifact schemas. A
reader validates the registered source's referenced fields against those
requirements.

Index relations are relative to an artifact root rather than absolute source
or warehouse URIs:

```json
{
  "index-relations": {
    "ivf-centroids": "centroids/ivf_centroids/",
    "ivf-postings": "snapshots/4820/ivf_postings/"
  }
}
```

A relative location must not be empty, absolute, contain `..`, or escape the
artifact root after normalization. The root metadata `location` field is also
removed: the catalog already stores the immutable metadata location.

These changes require a new metadata format version. Version 1 retains its
current URI-bound interpretation.

## 2. Catalog Contract

The catalog is the authority for deployment-specific relationships:

```text
registered source table
    -> logical index name
    -> immutable index metadata location
    -> index artifact root
```

The source table entry already owns the source provider and physical URI. The
logical index entry stores the source-table association, metadata location,
artifact root, index UUID, and status. Neither association nor physical
location is duplicated in index metadata.

Reusable centroid records remain catalog records. Their fingerprint identifies
and validates the centroid artifact within ParqDB; it is not a source identity
or a source-content check. Centroid reuse is scoped by the catalog's source
table association so the same fingerprint is not used to infer that two source
tables contain the same data.

ParqDB does not add `register_existing_index` to the catalog. The existing
`IndexCatalog::register` already accepts loaded and validated metadata and
atomically publishes the logical index mapping. Its registration input is
extended with the source-table association and artifact root required by the
source-free format, conceptually:

```text
register(
    source_table_identifier,
    index_identifier,
    index_metadata_location,
    artifact_root,
    validated_index_metadata,
)
```

Both `create_index` and `register_index` call this same operation. Index
creation constructs and writes the metadata first; index registration loads an
existing metadata document first. An existing index name fails with
`AlreadyExistsError`; the operation never silently replaces an index.

Registration handles reusable centroids through the existing
`load_ivf_centroids`, `claim_ivf_centroids`, and `publish_ivf_centroids`
lifecycle. When the matching centroid is already ready it is reused. When it
is absent, registration claims the descriptor and publishes the existing
validated centroid metadata without writing centroid data. This follows the
same publication order as index creation and does not require a combined
centroid-and-index transaction.

Metadata parsing and storage inspection stay outside the catalog so every
catalog implementation receives the same validated input.

## 3. Registration API

Registration is table-scoped:

```python
table.register_index(
    "benchmark_embedding",
    metadata_location="s3://indexes/sift1b/v2.metadata.json",
)
```

The selected `SourceTable` supplies the source association. The metadata
location determines the default artifact root unless the storage profile or
API call supplies one explicitly.

Registration performs these steps:

1. load and validate the immutable index metadata;
2. require `vector-field` and every `source-key-fields` field to exist and to
   satisfy the index-family type requirements;
3. count the rows eligible for indexing and require the result to equal
   `indexed-rows`;
4. resolve all relative index relations against the artifact root;
5. validate that the centroid and postings relations are permitted, readable,
   and structurally compatible with the metadata;
6. make the validated centroid ready through the existing centroid lifecycle;
   and
7. publish the logical index with `IndexCatalog::register`.

It does not train centroids, assign rows, encode vectors, write Parquet, copy
artifacts, or generate metadata. A failure before the final register call
leaves the logical index absent. It may leave a validated ready centroid, which
matches the existing behavior when index creation fails after centroid
publication.

Schema and row count are intentional guardrails, not proof that two datasets
have identical contents. If source values or keys change while referenced
fields and row count remain compatible, registration cannot detect the stale
index. Calling `register_index` is an explicit administrative assertion that
the selected source is the one represented by the index.

## 4. Publication and Sharing

A publisher distributes only the index package:

```text
package/
├── index metadata
├── centroid metadata
├── centroid Parquet
└── postings Parquet
```

A recipient:

1. registers the intended source table at any supported URI;
2. makes the index package available at any supported artifact root; and
3. calls `table.register_index` with the metadata location.

The publisher and recipient do not coordinate source UUIDs, version strings,
filesystem paths, catalog files, or content manifests.

## 5. Query and Lifecycle Behavior

After registration, query planning loads the source association and artifact
root from the catalog, then interprets the index metadata normally. A
registered index behaves like one published by `create_index`:

- `index_status` reports `ready`;
- selection and query planning use the normal repository path;
- refresh creates a normal successor snapshot;
- dropping the index removes catalog reachability; and
- maintenance treats its metadata and relations as reachable artifacts.

Registration does not change file ownership or retention policy.

## 6. Compatibility

Version 1 metadata continues to load and query without modification. It may be
registered only when its URI-bound references are already valid in the current
deployment; this is not portable registration.

Portable registration requires the new source-free format. Version 1 metadata
is never silently rewritten or reinterpreted.

## Alternatives Rejected

### Put source ID and version in index metadata

This duplicates the catalog association and introduces a distributed dataset
naming contract that publishers and recipients must coordinate.

### Store or derive a source content fingerprint

Registration is an explicit administrative operation. Content hashing adds
cost and identity semantics that are not needed for the intended workflow.
ParqDB deliberately checks only the referenced fields and `indexed-rows`.

### Copy the complete source schema into index metadata

Columns the index does not use are irrelevant. Recording the full schema would
reject safe source changes without adding index correctness.

### Rewrite metadata during registration

Rewriting makes registration a migration and publication operation and breaks
the immutable metadata contract.

### Scan a warehouse and infer catalog entries

Directory layout is not a logical contract. Scanning is ambiguous, expensive
on object stores, and cannot determine the intended source table or index name.

## Implementation Sequence

1. define the source-free metadata version and add `indexed-rows`;
2. move source association and artifact-root resolution into the catalog path;
3. extend the existing `IndexCatalog::register` input for that association;
4. reuse the existing centroid publication lifecycle;
5. expose `SourceTable.register_index`; and
6. add local-file and object-store publication round-trip tests.
