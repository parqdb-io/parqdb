# Single Publication Manifest

- Status: Accepted
- Target: `0.2.0rc4`
- Supersedes: [Static HTTP Index Packages](20260818-static-http-index-packages.md)

## Summary

ParqDB publishes one immutable index artifact with one authoritative
`manifest.json`. The manifest contains the complete physical index inventory
and may include source-table and embedding-model inventories. Native ParqDB
binds the same manifest through its catalog; browsers open it directly.

The change is intentionally incompatible with the `0.2.0rc2` and
`0.2.0rc3` publication and catalog formats. ParqDB has not released a stable
v1 format, so no compatibility reader or catalog migration is provided.
SQLite `user_version` remains `1`.

## Decision

### One immutable data-plane document

Every publication has this shape:

```text
manifest.json
centroids.parquet
ivf_postings/
  cid_bucket=000000/part-00000.parquet
  ...
documents.parquet                         # optional
models/...                               # optional
```

The manifest is written last and is the publication commit marker. It is the
only file inventory. The format removes:

- `source-manifest.json`;
- `ivf_postings/manifest.json`;
- published `roots.parquet`;
- published native `vN.metadata.json`; and
- the term "package" from the public format and APIs.

The immutable identity field is `artifact-uuid`.

### Index is required; source and embedding are optional

The required `index`, `hierarchy`, and `postings` sections are sufficient to
return source keys and `_distance`. An optional `source` section enables
payload lookup. An optional `embedding` section enables clients to produce a
query vector with the exact pinned model used during construction.

Build input and published content are separate decisions. A builder always
reads a source, but the publisher uploads it only when explicitly requested.
The minimal publication contains only the index artifact.

### Root topology without root routing

Hierarchical training assigns a variable number of leaf clusters to each root.
`cid-offsets` is the sole authoritative root-to-leaf mapping:

```text
root r -> [cid-offsets[r], cid-offsets[r + 1])
```

The offsets begin at zero, end at `nlist`, and are strictly increasing; their
interval lengths need not be equal. Root count is derived as
`len(cid-offsets) - 1` and is not stored separately.

Root centroid vectors are build-time state. They are not published and are not
used for query routing. Native and browser readers rank every leaf centroid
globally, then use the root grouping only to improve physical locality and
select bucket files.

### Catalog is the mutable control plane

A catalog maps a local source and logical index name to a current manifest
location and local snapshot history:

```text
(source, index name) -> current snapshot -> manifest location
```

The catalog owns names, lifecycle state, refresh history, build status,
tombstones, and garbage-collection reachability. It may cache validated
manifest summary fields for discovery, but it does not own or duplicate the
physical object inventory.

The manifest owns the vector field, index dimensions, encoding, hierarchy, centroid and postings
objects, source inventory, model inventory, sizes, and hashes.

One manifest may be registered by multiple catalogs under different local
names. `artifact-uuid` identifies immutable content; `current_snapshot_id` is
the local logical-index version selected by one catalog.

### No object listing

Given an exact manifest location, registration and querying issue no prefix or
directory listing. Readers construct explicit files and Parquet row-group
access plans from manifest entries. Build writers record objects as they are
created, publishers upload that known inventory, and verification uses exact
HEAD or Range requests.

Offline warehouse orphan discovery may list a catalog-managed prefix. It is
not part of registration or query execution.

## Native API

A separately registered source may consume an index-only artifact:

```python
session.register_parquet("documents", "s3://private/documents.parquet")
documents = session.table("documents")

documents.register_index(
    "embedding",
    manifest_location="https://data.example.com/wiki/v1/manifest.json",
)
```

Registration validates the manifest, source key schema, source row count,
vector field and dimension, centroid layout, postings schema, and every
explicit object boundary before atomically creating the catalog binding.

The query surface remains table-centered:

```python
documents.search(vector, index="embedding").nprobes(64).limit(10)
```

No index-only native query API is introduced.

## Browser API

```ts
const index = await ParqDB.open(
  'https://data.example.com/wiki/v1/manifest.json',
)
const hits = await index.search(vector, { nprobe: 64, k: 10 })
```

Without `source`, results contain source keys and `_distance`. With `source`,
an application may perform explicit payload lookup. The core index reader does
not silently join payload columns.

## Publication CLI

Index-only publication is the default:

```bash
parqdb publish \
  --source documents.parquet \
  --key id \
  --vector-column embedding \
  --nlist 4096 \
  --destination s3://bucket/wiki/v1
```

Source and model publication are explicit:

```bash
parqdb publish ... --include-source --include-model
```

`--include-model` requires a text build with a pinned embedding descriptor.
The manifest is uploaded only after every selected object.

## Refresh and immutability

Refresh builds a new artifact and manifest, validates it, then atomically
changes the catalog pointer:

```text
manifest A <- catalog current
manifest B <- build and validate
catalog CAS: A -> B
```

Publications use new immutable prefixes such as `/v1/` and `/v2/`; a manifest
is never overwritten. Stable channel aliases are deployment policy and are
outside format version 1.

## Validation and conformance

The Rust and TypeScript parsers reject unknown fields and validate the same
portable integer, UUID, path, size, hash, source, hierarchy, and postings
constraints. Native and browser conformance tests use the same fixtures and
must agree on selected CIDs, returned source keys, and distances within the
declared floating-point tolerance.

Performance regression data records recall, cold and warm latency, HTTP
request count, transferred index bytes, source lookup bytes, Range coalescing,
and cache hits for a fixed query corpus.
