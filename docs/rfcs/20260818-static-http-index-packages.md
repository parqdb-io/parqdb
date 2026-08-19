# Static HTTP Index Packages

- Status: Proposed
- Date: 2026-08-18

## Problem

ParqDB indexes are currently consumed through a catalog-backed native session.
Their Parquet postings use one Hive partition and one file per `cid`. A reader
must list that directory before it can map selected clusters to files. This
layout has two problems for publication:

1. a large index creates many small objects; and
2. a browser cannot efficiently discover and query those objects through
   ordinary HTTP byte-range requests without reproducing object-store listing
   behavior.

ParqDB needs an immutable, self-describing index package that can be published
to a public Hugging Face dataset repository or public S3 bucket and queried
directly in a browser. The storage format must remain standard Parquet. It must
also be usable by native ParqDB so the browser path is not a separate index
format.

## Decisions

| Question | Decision |
| --- | --- |
| What can host a version 1 package? | A public Hugging Face dataset revision or public S3 objects exposed through HTTP. |
| How does a reader discover objects? | One exact `manifest.json` request. Readers never list the package prefix. |
| Which index representations are supported? | IVF-LVQ4 and IVF-LVQ8. |
| Which clustering strategy is used? | Hierarchical K-means by default and for every version 1 package. The hierarchy controls CID order and layout, not query-time root pruning. |
| How are leaf CIDs assigned? | Each root owns one contiguous CID interval. |
| What is the postings partition key? | `cid_bucket`, equal to the hierarchical root ID. |
| Is `cid_bucket` part of the logical postings schema? | No. It is private physical layout metadata. |
| Is `cid` stored in Parquet? | Yes, as a required physical `INT32` column. |
| What is the row-group boundary rule? | A row group contains exactly one CID. A CID may span consecutive row groups and files. |
| What is the file boundary rule? | A file contains one `cid_bucket`, may contain several consecutive CIDs, and is closed only at a row-group boundary. |
| Where are row-group offsets recorded? | Only in the Parquet footer, not duplicated in `manifest.json`. |
| Does row-group pruning depend on an SQL `IN` predicate? | No. Selected CIDs are a typed scan input and always produce an explicit row-group access plan. |
| What does a query return? | The configured source-key fields and `_distance`. |
| Where does distance and top-k execution run? | In a WebAssembly kernel called by a TypeScript client. |
| Is a package mutable? | No. Its manifest and every referenced object are immutable. |

## Scope

Version 1 includes:

- public, unauthenticated Hugging Face dataset and S3 HTTP objects;
- an immutable package manifest;
- global leaf-centroid IVF routing over hierarchy-ordered CIDs;
- LVQ4 and LVQ8 postings;
- standard Parquet objects selected with HTTP range requests;
- a TypeScript client and WebAssembly distance/top-k kernel; and
- native ParqDB reading of the same package without object listing.

Version 1 does not include:

- private repositories, signed requests, cookies, or other credentials;
- the `source` postings encoding or exact reranking against source vectors;
- fetching source payload columns or joining results back to a source table;
- mutable package refresh or following a moving branch;
- server-side query execution;
- Iceberg publication; or
- a general-purpose remote Parquet query engine.

## 1. Package Layout

A package is a directory-like immutable object prefix:

```text
manifest.json
roots.parquet
centroids.parquet
ivf_postings/
  manifest.json
  cid_bucket=000000/
    part-00000.parquet
    part-00001.parquet
  cid_bucket=000001/
    part-00000.parquet
```

The Hive-looking directory names are only canonical object names. Readers do
not infer a Hive schema and do not expose `cid_bucket` as a query column.
The top-level `manifest.json` is the sole browser discovery entry point and
lists every Parquet object that may be read. `ivf_postings/manifest.json` is
the native Parquet relation manifest; a browser does not fetch it. Keeping the
relation manifest in the package lets the same snapshot remain a normal
catalog-backed ParqDB index without object listing.

All manifest paths are relative to the directory containing `manifest.json`.
A path is non-empty, uses `/`, and contains no empty, `.` or `..` segment,
query, fragment, or URI scheme. Resolving a path must remain below the package
root.

Object names are not content-addressed in version 1. Immutability is therefore
a storage requirement: a builder writes a new snapshot prefix and never
replaces an object reachable from its manifest. Publishing that snapshot is a
byte-for-byte copy, not a format conversion.

## 2. Hierarchical Clustering and CID Assignment

Hierarchical K-means is the default clustering strategy. `Auto` mode is
removed: a build must not silently choose a different published topology based
on the sampled data. Flat K-means remains available when a caller explicitly
requests it, and its raw output cannot be published as a version 1 static
package.

Training produces `R` root centroids. Root `r` owns `children[r]` leaf
centroids. Leaf centroids are appended in ascending root order, producing the
prefix sum:

```text
cid_offsets[0] = 0
cid_offsets[r + 1] = cid_offsets[r] + children[r]
cid_offsets[R] = nlist
```

The storage bucket and CID interval for root `r` are:

```text
cid_bucket = r
cid in [cid_offsets[r], cid_offsets[r + 1])
```

This is a format invariant, not an implementation coincidence. Child counts
need not be equal, so implementations must use `cid_offsets`; division by a
fixed bucket width is not generally correct.

The K-means model must retain root centroids and `cid_offsets` along with leaf
centroids. Root training uses 512 sampled points per root. After assigning the
complete training sample to roots, the implementation reserves one leaf for
each root and distributes the remaining leaf budget in proportion to each
root's remaining population. Integer quotas use largest remainders with root
ID as the deterministic tie-breaker. No root receives more leaves than its
sample population, and the allocation sums exactly to `nlist`.

If a root partition is empty, the implementation re-seeds it from the farthest
point in a populated donor partition and continues root Lloyd iterations over
the complete training sample. After three unsuccessful recovery rounds, the
implementation emits a warning, trains all leaves with flat K-means,
deterministically orders those leaves by the trained root anchors, and
synthesizes a valid hierarchy. Only the leaf-training algorithm falls back;
the persisted topology and package invariants do not.

Root IDs, leaf CIDs, and their ordering are local to one immutable centroid
artifact. They do not claim stable identity across independently trained
artifacts.

## 3. Manifest Version 1

`manifest.json` is one strict RFC 8259 JSON object. Unknown fields, duplicate
keys, missing required fields, non-canonical integers, and paths that escape
the package are invalid.

An abbreviated example is:

```json
{
  "format-version": 1,
  "package-uuid": "249343c7-9989-48d8-b2ca-d0caa62ba940",
  "index": {
    "metric": "l2_squared",
    "posting-encoding": "lvq8",
    "dimension": 768,
    "nlist": 4,
    "ntotal": 1000000,
    "source-key-fields": [
      {"name": "document_id", "type": "long"}
    ]
  },
  "hierarchy": {
    "root-count": 2,
    "cid-offsets": [0, 2, 4],
    "centroid-encoding": "lvq8",
    "roots": {
      "path": "roots.parquet",
      "size": 7184,
      "sha256": "12907f719aba1156f2ed59222b41e114213752e31dd8c39b56d96d00f5b018d6"
    },
    "centroids": {
      "path": "centroids.parquet",
      "size": 14841,
      "sha256": "af853c8e65da24292a27b07cc70025e720f1eaa3e6afabfe39efac58a56dca92"
    }
  },
  "postings": {
    "files": [
      {
        "path": "ivf_postings/cid_bucket=000000/part-00000.parquet",
        "cid-bucket": 0,
        "min-cid": 0,
        "max-cid": 1,
        "rows": 300000,
        "size": 231093221,
        "sha256": "e1550cc51520095ad34357a61f5a23532b72885d083c6ddfd83936f9b139cbc7"
      },
      {
        "path": "ivf_postings/cid_bucket=000000/part-00001.parquet",
        "cid-bucket": 0,
        "min-cid": 1,
        "max-cid": 1,
        "rows": 200000,
        "size": 153782114,
        "sha256": "bf40320451c794954fecdffdc749f17c150bd2c59e4777f06c3846b226f196ef"
      },
      {
        "path": "ivf_postings/cid_bucket=000001/part-00000.parquet",
        "cid-bucket": 1,
        "min-cid": 2,
        "max-cid": 2,
        "rows": 250000,
        "size": 192882117,
        "sha256": "92ada15d0cc7e83b1538e591094f5f52666822317fcef66e8bd24f7b0d965232"
      },
      {
        "path": "ivf_postings/cid_bucket=000001/part-00001.parquet",
        "cid-bucket": 1,
        "min-cid": 3,
        "max-cid": 3,
        "rows": 250000,
        "size": 192714092,
        "sha256": "34a20a29af7deace8cb94d4bd565895e4b05bda54592072dde81f8805b1f0963"
      }
    ]
  }
}
```

`cid-offsets` contains exactly `root-count + 1` entries. Required constraints
are:

- `format-version` is `1`;
- `package-uuid` is a non-nil lowercase UUID;
- `metric` is `l2_squared` or `cosine`;
- `posting-encoding` is `lvq4` or `lvq8`;
- `dimension`, `nlist`, `ntotal`, and `root-count` are positive;
- `cid-offsets` starts at zero, is strictly increasing, and ends at `nlist`;
- every `cid-bucket` is in `[0, root-count)`;
- every file's inclusive CID range is non-empty and lies inside its bucket's
  half-open CID interval;
- file entries are ordered by `(cid-bucket, min-cid, max-cid, path)`;
- files from different buckets never overlap;
- consecutive files in one bucket may overlap at exactly one boundary CID
  when that CID spans files;
- `ntotal`, every `rows`, and every `size` are positive integers no greater
  than `9007199254740991`, so a browser can validate them without precision
  loss;
- the sum of postings file `rows` equals `ntotal`;
- `sha256` is the lowercase digest of the complete referenced object.

The manifest deliberately contains only file-level postings ranges. Parquet
footers remain the sole authority for row groups, column chunks, page indexes,
compression, and byte offsets.

Source-key types use the canonical IVF grammar: `boolean`, `int`, `long`,
`binary`, `string`, `date`, or `fixed(L)` for positive canonical `L`. A
JavaScript-facing API represents `long` values as `bigint`, not `number`.

## 4. Parquet Contracts

All data objects are ordinary Parquet files. No TAR, custom pack container, or
sidecar row-group offset table is introduced.

### Root centroids

`roots.parquet` contains exactly one row per root:

| Field | Type | Constraint |
| --- | --- | --- |
| `cid_bucket` | `int` | Required, unique, ascending, in `[0, R)`. |
| `cid_begin` | `int` | Required, equal to `cid_offsets[cid_bucket]`. |
| `cid_end` | `int` | Required, equal to `cid_offsets[cid_bucket + 1]`. |
| `centroid` | `list<float>` | Required, exactly `dimension` finite elements. |

The file is small enough to fetch as one object in version 1.

### Leaf centroids

`centroids.parquet` contains exactly one row per leaf CID:

| Field | Type | Constraint |
| --- | --- | --- |
| `cid` | `int` | Required, unique, ascending, in `[0, nlist)`. |
| `cid_bucket` | `int` | Required, equal to the root owning `cid`. |
| `offset` | `float` | Required, finite LVQ8 lower bound. |
| `scale` | `float` | Required, finite and non-negative LVQ8 scale. |
| `code` | `binary` | Required, exactly `dimension` bytes. |

Rows are ordered by `(cid_bucket, cid)`. A leaf-centroid row group cannot cross
a `cid_bucket` boundary. Version 1 readers still rank every leaf centroid
globally: row-group boundaries preserve physical topology and future planning
options, but must not be used to prune roots and change the selected CIDs.

### Postings

The physical postings schema begins with required `cid: int`, followed by the
source-key fields and the canonical LVQ4 or LVQ8 fields. `cid_bucket` is not a
physical postings column and is not reconstructed as a logical column.

Within each file, rows are ordered by `cid`. The writer enforces these boundary
rules:

1. one row group contains rows for exactly one CID;
2. a CID may occupy multiple consecutive row groups;
3. a CID may continue in the next file in the same bucket;
4. a file contains only one bucket;
5. a file may contain multiple consecutive CIDs; and
6. files close only after a complete row group has been flushed.

Row-group and file size targets are soft physical tuning parameters, not part
of the package's logical identity. Initial defaults should target row groups
in the 8--32 MiB range and files in the 128--256 MiB range. A writer uses its
encoded in-progress row-group size to flush a large CID and its completed byte
count to rotate files. No single CID is allowed to defeat both limits merely
to preserve a one-CID boundary.

The writer enables Parquet statistics for `cid`. Every postings row group's
`cid` minimum and maximum must both equal its single CID. A package validator
checks this invariant before publication.

## 5. Build and Publication

The index builder creates a self-contained snapshot in this order:

1. train and validate the hierarchical centroid model;
2. write every postings file and the native relation manifest to a fresh
   snapshot prefix;
3. write package-local `roots.parquet` and `centroids.parquet`;
4. close all Parquet writers and collect exact object sizes and SHA-256
   digests;
5. validate schemas, hierarchy, file ranges, and row-group CID statistics;
6. create `manifest.json` last; and
7. return the completed snapshot root only after the manifest create succeeds.

An interrupted build may leave unreachable objects but cannot expose a valid
partial package because the entry-point manifest does not exist. The builder
uses create-if-absent behavior for every object, including `manifest.json`.

No package exporter or service-specific publisher is required. A user may copy
the completed snapshot directory with ordinary tools such as `cp`, an S3 sync,
or a Hugging Face dataset upload. The upload operation must preserve relative
paths and bytes, target a fresh immutable prefix or revision, and make the
top-level manifest visible only after all referenced Parquet objects have been
uploaded. When an upload tool cannot guarantee that order, the user uploads
`manifest.json` in a final operation.

Hugging Face publication identifies an immutable dataset commit, not a moving
branch such as `main`. S3 publication uses a fresh immutable key prefix. Bucket
policy and CORS configuration are deployment concerns, but a publisher must
verify the HTTP contract before reporting success.

## 6. HTTP Contract

A package host must support unauthenticated HTTPS `GET` requests. Parquet
objects must support a single byte range and return a valid `206 Partial
Content` response with a matching `Content-Range`. Browser-visible CORS policy
must permit the requests and expose the response headers needed to validate
ranges and object size.

The client rejects a large Parquet request when the server ignores `Range` and
returns the complete object with `200 OK`. Small objects intentionally fetched
in full, including `manifest.json` and normally `roots.parquet`, may return
`200 OK`.

Redirects are permitted only when the final response preserves byte-range and
CORS behavior. The resolved URL must remain HTTPS. Version 1 does not attach
authorization headers or credentials to redirects or object requests.

## 7. Browser Query Execution

The TypeScript client opens an immutable package URL and performs:

```text
GET manifest.json
    -> validate format and query parameters
Range GET centroids.parquet footer
Range GET all LVQ8 leaf-centroid column chunks
    -> WASM distance/top-k globally selects leaf CIDs
manifest file-range lookup
    -> select postings files whose CID intervals intersect selected CIDs
Range GET selected postings footers
    -> select row groups whose cid min=max is selected
Range GET selected postings column chunks
    -> WASM LVQ distance/top-k
return source keys + _distance
```

Root centroids are not part of version 1 query routing. Selecting roots first
would make the candidate leaf set differ from native global `nprobe` routing
and can reduce recall. The hierarchy instead makes related globally selected
CIDs more likely to map to nearby row groups and ranges.

Footer-derived postings row-group selection is explicit. The client does not download a
whole candidate file and hope that a generic predicate optimizer prunes it.
Adjacent selected ranges may be coalesced to reduce requests, but coalescing
must be bounded so one small query cannot accidentally fetch a complete large
file.

The selected CID set is query control data, not an SQL expression. Its size
does not pass through an optimizer's `IN`-list expansion, simplification, or
statistics-pruning threshold. The browser represents it as a sorted unique
integer set or bitmap and tests row-group `cid` statistics directly.

The WebAssembly kernel implements global LVQ8 leaf-centroid routing, LVQ4/LVQ8
posting distance, and bounded top-k selection. TypeScript owns manifest parsing, HTTP
requests, Parquet metadata planning, cancellation, and the public result API.

For cosine, the client applies the same normalized-vector and reported-distance
contract as native ParqDB. Browser and native implementations must pass the
same conformance vectors and return the same source keys and distances within
the format's floating-point tolerance.

## 8. Native ParqDB Integration

The main ParqDB package supports this format; it is not browser-only. A native
reader given a package manifest:

- fetches that exact object instead of issuing `ListObjectsV2` or an equivalent
  prefix listing;
- builds its bounded file-range index from manifest entries;
- reads Parquet footers for candidate files;
- attaches an explicit DataFusion `ParquetAccessPlan` to each selected file;
  and
- retains the exact `cid` filter because files may contain multiple CIDs.

File-level CID range selection alone is inexact. The provider must not report
the filter as exact merely because it selected candidate files. The physical
`cid` column and final predicate preserve correctness, while the access plan
provides deterministic row-group pruning.

Native query planning must not communicate cluster selection to the postings
provider only as `cid IN (...)`. The centroid-routing result is carried as a
typed `CidSelection` input to the manifested postings scan. The provider uses
that value directly to:

1. select candidate manifest files;
2. fetch their footers;
3. match the selected CIDs against each row group's single-CID statistics;
4. construct a `ParquetAccessPlan` with only those row groups enabled; and
5. attach that plan to the `PartitionedFile` before DataFusion builds the
   Parquet execution plan.

An SQL predicate or selected-cluster semi-join may remain above the scan as a
correctness guard, but it is not the physical pruning mechanism. This avoids
DataFusion or another engine skipping statistics pruning when an `IN` list is
large, rewritten, or above an optimizer threshold.

This path is fail-closed. When the requested CID set does not cover every
distinct row-group CID in a candidate file, the provider must not attach an
all-row-groups plan. Missing CID statistics, mixed-CID row groups, an
access-plan length that does not match the footer, or failure to carry
`CidSelection` to the physical scan makes the package or plan invalid.
Scanning every row group is valid only when the selected set actually covers
every CID represented by that file.

The package reader shares manifest validation and Parquet layout validation
with the browser conformance suite. Native ParqDB may additionally use local
files, authenticated object-store clients, page caches, and direct I/O, but
those facilities do not change the portable package contract.

Ordinary external source Parquet relations remain outside this decision and
may still require listing. The no-list guarantee applies to manifested index
packages.

## 9. Immutability, Caching, and Validation

The manifest URL identifies one immutable package. Clients may cache parsed
manifests and Parquet metadata by the manifest URL plus object path. A client
must not treat a mutable Hugging Face branch URL or overwriteable S3 prefix as
an immutable cache key.

The browser client additionally keeps a byte-bounded in-memory LRU below the
Parquet reader. Its default 32 MiB budget is shared by every object opened by
one client. It stores validated, coalesced ranges exactly as fetched, keyed by
immutable object URL, declared size, start, and end, and deduplicates covered
in-flight loads. Cache admission never expands a cold HTTP request. Partial
HTTP requests still bypass the browser's implicit `206` cache; a covering LRU
hit is resolved before `fetch` is called. Applications may supply one cache to
both the index and a manifested source relation so the combined working set
remains under one budget.

Whole-object SHA-256 digests support publication validation, mirroring, and
full-download verification. A range reader cannot prove a whole-object digest
from one fragment; it still validates HTTP range boundaries, expected object
size, Parquet metadata, schemas, CID ranges, and all decoded buffer lengths.

Before reading a selected row group, a client verifies that its CID statistics
are present, non-null, equal (`min == max`), inside the file's manifest range,
and inside the owning bucket's CID interval. Missing or inconsistent
statistics make the package invalid rather than triggering an unbounded scan.

All integer arithmetic for offsets, lengths, row counts, and allocation sizes
is checked. The browser enforces configurable limits for manifest bytes,
object count, footer bytes, range bytes, decoded rows, vector dimension,
`nlist`, and `k` before allocating memory.

## 10. Compatibility

There is no compatibility path for the development-only one-CID-per-file
layout. ParqDB has not published version 1, so the IVF and Parquet version 1
specifications will be updated in place and existing development indexes must
be rebuilt.

When accepted and implemented, this RFC supersedes
[ADR 0010](../decisions/0010-hive-partitioned-parquet-postings.md). SQLite
catalog `user_version` remains `1`; this storage-format change does not create
a catalog schema migration.

## Alternatives Rejected

### One file per CID

It makes cluster selection simple but creates too many small objects, performs
poorly on object stores, and prevents a large CID from following the general
file-size policy.

### TAR or a custom pack container

It reduces object count but introduces a second container index, custom
tooling, and recovery rules around Parquet. Standard Parquet files already
provide byte-addressable metadata and column chunks.

### Multiple CIDs in one row group

It improves row-group fill for tiny clusters but makes CID selection depend on
reading unrelated postings. Version 1 chooses predictable range reads over
maximum row-group density.

### Never split one CID

A skewed cluster can be much larger than both the row-group and file targets.
Allowing one CID to span consecutive row groups and files bounds memory and
object size without mixing clusters.

### Hash buckets

`cid % bucket_count` scatters adjacent leaf CIDs and discards the physical
locality already provided by hierarchical routing. Root-aligned range buckets
let root selection directly choose storage partitions.

### Fixed-width CID buckets

Division by a fixed width works only when every root owns the same number of
children. Hierarchical training distributes leaves in proportion to sampled
root populations, so the persisted `cid_offsets` are authoritative.

### Record row-group offsets in the manifest

This duplicates Parquet footer metadata and creates two authorities that can
disagree. The manifest narrows files; the footer narrows row groups and column
chunks.

### Rely only on generic Parquet predicate pruning

The previous multi-cluster layout produced near-full reads under constrained
memory. Both native and browser readers construct an explicit row-group access
plan from validated footer statistics.

### Allow topology-changing flat fallback

Raw flat output would make the same user configuration produce incompatible
physical topologies depending on training data. The permitted fallback instead
constructs and validates the requested hierarchy, so `cid_bucket == root_id`
and every package invariant remain true.

## Implementation Sequence

1. change the default K-means mode to hierarchical, remove `Auto`, retain root
   centroids and CID offsets, and add hierarchy invariants;
2. define strict manifest types, canonical JSON fixtures, and package
   validators;
3. replace the Hive-CID writer with a bucketed writer that preserves physical
   `cid`, explicitly flushes row groups, rotates files, and emits manifest
   entries;
4. replace the Hive-CID provider with a manifest-backed native provider using
   explicit `ParquetAccessPlan` row-group selection and no listing;
5. update the unreleased IVF and Parquet version 1 specifications and regenerate
   fixtures;
6. add the TypeScript HTTP/Parquet planner and WebAssembly query kernel;
7. add host-independent HTTP capability validation and document byte-preserving
   uploads to public Hugging Face datasets and S3; and
8. add cross-runtime conformance, corruption, CORS, range, skewed-CID, and
   billion-scale manifest tests. Native tests must use CID selections larger
   than optimizer `IN`-list thresholds and assert the exact accessible row
   groups and requested object-store byte ranges.
