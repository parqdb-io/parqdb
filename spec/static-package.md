# Static HTTP Index Package

## Overview

A version 1 static package is one immutable IVF-LVQ4 or IVF-LVQ8 index
snapshot that can be copied to an HTTP object prefix and queried without a
catalog or object listing. The snapshot directory is already the package; no
export or repacking step is defined.

The package root contains `manifest.json`, `roots.parquet`,
`centroids.parquet`, and the native `ivf_postings` Parquet relation. Builders
create the top-level manifest last with create-if-absent semantics. Uploaders
preserve all relative paths and bytes and publish the top-level manifest only
after its referenced objects.

## Manifest

The top-level `manifest.json` is a strict JSON object with these required
fields:

- `format-version`: exactly `1`;
- `package-uuid`: a non-nil lowercase UUID;
- `index`: `metric`, `posting-encoding`, `dimension`, `nlist`, `ntotal`, and
  ordered `source-key-fields`;
- `hierarchy`: `root-count`, `cid-offsets`, `centroid-encoding` (exactly
  `lvq8`), and descriptors for the roots and leaf-centroid Parquet objects; and
- `postings.files`: the canonical inventory of postings Parquet objects.

Unknown fields are invalid. Each object descriptor contains a relative `path`,
positive `size`, and lowercase whole-object `sha256`. Each postings descriptor
also contains `cid-bucket`, inclusive `min-cid` and `max-cid`, and positive
`rows`. Paths are unique and cannot contain empty, `.`, or `..` segments, a
query, fragment, or URI scheme.

`ntotal`, `rows`, and `size` cannot exceed `9007199254740991`. This limit lets
JavaScript validate the JSON integers exactly. A source key of canonical type
`long` is still returned as `bigint` and is never converted to a JavaScript
`number`.

The package accepts only `l2_squared` or `cosine`, only `lvq4` or `lvq8`, and
only source-key types `boolean`, `int`, `long`, `binary`, `string`, `date`, or
`fixed(L)` for a positive canonical decimal `L`.

## Hierarchy and Objects

`cid-offsets` has `root-count + 1` strictly increasing entries, begins with
zero, and ends at `nlist`. Root `r` owns the CID interval
`[cid-offsets[r], cid-offsets[r + 1])` and physical bucket `r`.

`roots.parquet` has one row per root. It records topology and is not a
query-time pruning layer. `centroids.parquet` has one row per leaf CID, ordered
by `(cid_bucket, cid)`, and exactly one row group per root. A leaf centroid row
contains required `offset: float`, `scale: float`, and `code: binary` columns;
the code is exactly `dimension` LVQ8 bytes. A row group cannot cross a root
boundary. Readers rank all leaf centroids globally; they must not restrict
leaf routing to selected roots.

Postings paths are rooted at
`ivf_postings/cid_bucket=<six-digit-root-id>/`. The top-level manifest repeats
the Parquet file inventory required by an HTTP reader; the nested
`ivf_postings/manifest.json` remains the discovery entry for the native
Parquet relation and is not fetched by a browser.

Each postings row group contains exactly one non-null `cid`, with equal and
exact minimum and maximum statistics. A CID may span consecutive row groups or
files. File entries are ordered by `(cid-bucket, min-cid, max-cid, path)`, and
the sum of file `rows` equals `ntotal`.

## HTTP Reading

A reader fetches the exact top-level manifest and never lists the package
prefix. Large Parquet objects must support single HTTP byte ranges with a
matching `206 Partial Content` and `Content-Range`. A reader validates footer
CID statistics and constructs an explicit selected-row-group plan; it does not
depend on SQL `IN`-list optimization for physical pruning.

The portable query result contains the configured source keys and `_distance`.
No source-table payload join, credentials, mutable refresh, or source-encoded
postings are part of version 1.
