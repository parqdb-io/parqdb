# Immutable Publication Manifest

## Overview

A version 1 publication is one immutable IVF-LVQ4 or IVF-LVQ8 index snapshot
that can be copied to object storage and queried without a catalog or object
listing. A publication may additionally include a source table for payload
lookup and an embedding model for text queries. Both are optional.

The publication root contains one authoritative `manifest.json`. Builders
record every object while writing it and create the manifest last with
create-if-absent semantics. Uploaders preserve relative paths and bytes and
publish the manifest only after every referenced object.

There is no nested postings manifest, source manifest, root-centroid object, or
native metadata document in the publication format.

## Manifest

The top-level `manifest.json` is a strict JSON object with these required
fields:

- `format-version`: exactly `1`;
- `artifact-uuid`: a non-nil lowercase UUID identifying this immutable
  publication;
- `index`: the complete portable IVF query contract;
- `hierarchy`: the root-to-leaf CID topology and leaf-centroid object; and
- `postings.files`: the canonical inventory of postings Parquet objects.

It may additionally contain:

- `source`: an immutable source-table inventory supporting key-based payload
  lookup; and
- `embedding`: an immutable embedding-model descriptor and asset inventory.

Unknown fields are invalid. Every object descriptor contains a relative
`path`, positive `size`, and lowercase whole-object `sha256`. Paths are unique
across the complete publication and cannot contain empty, `.`, or `..`
segments, a query, fragment, or URI scheme.

`ntotal`, row positions, row counts, and object sizes cannot exceed
`9007199254740991`. This lets JavaScript validate every JSON integer exactly.
A source key of canonical type `long` is returned as `bigint` and is never
converted to a JavaScript `number`.

The required `index.vector-field` records the source vector column used to
build the artifact. The index accepts only `l2_squared` or `cosine`, only `lvq4` or `lvq8`
postings, and only source-key types `boolean`, `int`, `long`, `binary`,
`string`, `date`, or `fixed(L)` for a positive canonical decimal `L`.

## Hierarchy and Centroids

`cid-offsets` has at least two strictly increasing entries, begins with zero,
and ends at `nlist`. The number of roots is `len(cid-offsets) - 1`; root `r`
owns the variable-cardinality CID interval
`[cid-offsets[r], cid-offsets[r + 1])` and physical bucket `r`.

Root centroid vectors are build-time state and are not published. Readers
never perform root-to-child routing.

`centroids.parquet` has one LVQ8 row per leaf CID, ordered by
`(cid_bucket, cid)`, and exactly one row group per root. Row-group sizes are
variable and must equal the corresponding CID interval length. A centroid row
contains required `cid: int`, `cid_bucket: int`, `offset: float`,
`scale: float`, and `code: binary` columns; `code` contains exactly
`dimension` LVQ8 bytes. Readers rank all leaf centroids globally.

## Postings

Each postings descriptor contains `cid-bucket`, inclusive `min-cid` and
`max-cid`, positive `rows`, and the ordinary object fields. Paths are rooted at
`ivf_postings/cid_bucket=<six-digit-root-id>/`.

Each postings row group contains exactly one non-null `cid`, with equal and
exact minimum and maximum statistics. A CID may span consecutive row groups or
files. File entries are ordered by `(cid-bucket, min-cid, max-cid, path)`, each
file range is contained by its variable-cardinality root interval, and the sum
of file `rows` equals `ntotal`.

The top-level manifest is the only postings inventory. A reader constructs
explicit files and row-group access plans from it and never scans the postings
directory.

## Optional Source

When `source` is present, it declares `rows`, `row-group-rows`, ordered
`columns`, one `key`, and a canonical `files` inventory. Version 1 payload
lookup requires a non-null `long` key that is dense and ordered from zero.
Source file descriptors additionally contain half-open `row-begin` and
`row-end` positions. The ordered file ranges must partition `[0, rows)`, and
`rows` must equal index `ntotal`.

When `source` is absent, portable queries return only the index source-key
fields and `_distance`. A native catalog may bind the publication to a
separately registered compatible source table.

## Optional Embedding Model

When `embedding` is present, it declares the pinned model repository and
revision, runtime contract, dimension, input template, parity probe, and a
complete immutable asset inventory. Its dimension must equal the index
dimension. Vector-only clients ignore this section.

## Catalog Binding

The publication manifest is immutable data-plane state. A native catalog is a
mutable control plane mapping a local source and index name to the current
manifest location and local snapshot history. Catalog state is not part of a
publication and must not duplicate its physical object inventory.

Browser readers open the manifest directly. Native readers find it through
the catalog binding. Both use the same hierarchy, centroid, postings, and
distance contract.

## HTTP Reading

A reader fetches the exact top-level manifest and never lists the publication
prefix. Large Parquet objects must support single HTTP byte ranges with a
matching `206 Partial Content` and `Content-Range`. A reader validates footer
CID statistics and constructs an explicit selected-row-group plan; it does not
depend on SQL `IN`-list optimization for physical pruning.

If source objects are present, payload lookup also uses their explicit
inventory and never lists a source prefix.
