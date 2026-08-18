# IVF Index Schema

## 1. Scope

This document defines index family `ivf` at `index-schema-version = 1`.
Supported metrics are `l2_squared` and `cosine`. Supported postings encodings
are `source`, `lvq4`, and `lvq8`.

An IVF index references one immutable centroid artifact and owns exactly one
postings relation. Logical indexes over the same source state, vector field,
dimension, metric, cluster count, and clustering profile may reference the
same centroid artifact. They never share postings.

## 2. Logical Index Metadata

An IVF snapshot must contain exactly these parameters:

| Key | Definition |
|---|---|
| `dimension` | Vector dimension `D`. |
| `nlist` | Number of IVF clusters `C`. |
| `ntotal` | Number of indexed source rows `N`. |
| `posting_encoding` | `source`, `lvq4`, or `lvq8`. |
| `ivf_centroids_fingerprint` | Deterministic UUID identifying the IVF centroid descriptor. |
| `ivf_centroids_uuid` | UUID of the referenced IVF centroid artifact. |
| `ivf_centroids_metadata_location` | Warehouse-relative location of its immutable metadata file. |

`dimension`, `nlist`, and `ntotal` use the canonical base-10 representation of
a positive integer, without a sign or leading zero. `dimension` and `nlist`
must not exceed `2147483647`; `ntotal` must not exceed
`9223372036854775807`; and `nlist` must not exceed `ntotal`.

The snapshot contains exactly these index relation roles:

| Role | Definition |
|---|---|
| `ivf_centroids` | The centroid relation named by the IVF centroid metadata. |
| `ivf_postings` | The postings owned by this logical index. |

The snapshot's `ivf_centroids` location must equal the centroid metadata's
`centroids` location. Its vector field, dimension, metric, and `nlist` must
equal the corresponding descriptor fields. `indexed-rows` must equal
parameter `ntotal`.

## 3. IVF Centroid Metadata

An IVF centroid metadata document has these fields:

| Field | Definition |
|---|---|
| `format-version` | `1`. |
| `artifact-uuid` | Non-nil lowercase UUID of this immutable artifact. |
| `fingerprint` | Fingerprint of `descriptor`. |
| `created-at-ms` | Non-negative Unix epoch time in milliseconds. |
| `descriptor` | Semantic identity defined below. |
| `centroids` | Warehouse-relative Parquet location for `ivf_centroids`. |
| `roots` | Warehouse-relative Parquet location for `ivf_roots`. |

The descriptor contains, in order, `vector-field`, `dimension`, `metric`,
`nlist`, and `clustering-profile-version`. Version `1` is the only clustering
profile defined here.

The fingerprint is UUIDv5 using namespace
`2fb71e63-a27c-4fc5-9d6d-5070698dc398`. The UUID name is a semantic descriptor
encoded as compact UTF-8 JSON with fields in the order above and no
insignificant whitespace. JSON strings leave non-ASCII characters unescaped
and use the shortest RFC 8259 escape for characters that must be escaped.

The fingerprint is a lookup key. Readers still compare every descriptor field
before using the artifact. The catalog scopes centroid reuse by its exact
source-table binding; the fingerprint itself is source-free.

## 4. Types

Index relations use these canonical Iceberg types:

| Type | Definition |
|---|---|
| `int` | Signed 32-bit integer. |
| `long` | Signed 64-bit integer. |
| `float` | IEEE 754 binary32 value. |
| `list<float>` | Ordered list with required `float` elements. |
| `binary` | Byte sequence. |
| `T_i` | Exact source-key type for field `i`. |

Source vectors may use `list<float>` or `list<double>`. Implementations convert
vector elements to finite `float` values before training, assignment, encoding,
or distance evaluation; a value that is not representable as finite `float` is
invalid. The source relation itself is not rewritten.

Each `key_i` corresponds to source key field `i` and uses the same canonical
type and value. Supported source-key types are `boolean`, `int`, `long`,
`binary`, `fixed(L)`, `string`, and `date`. String, binary, and fixed values
are compared byte-for-byte. Integer key types are signed.

## 5. IVF Centroids

Clustering profile version `1` uses hierarchical K-means. Training produces
root centroids and leaf centroids. Root `r` owns one non-empty contiguous leaf
CID interval `[cid_offsets[r], cid_offsets[r + 1])`; the first offset is zero,
the final offset is `C`, and offsets are strictly increasing. Implementations
must not infer this mapping by fixed-width division. Flat K-means may be
requested as an implementation-specific operation, but its result does not
satisfy this profile unless its leaves are deterministically reordered and a
valid hierarchy is synthesized. Implementations may use that conversion as a
warned fallback when bounded recovery cannot keep every trained root partition
non-empty. For hierarchical training, the exact `C`-leaf budget is allocated
across non-empty roots in proportion to their sampled populations, with at
least one leaf per root and no more leaves than assigned sample rows.

`ivf_roots` has this schema:

| Field | Type | Constraint |
|---|---|---|
| `cid_bucket` | `int` | Required; unique; ascending root ID. |
| `cid_begin` | `int` | Required; inclusive first leaf CID. |
| `cid_end` | `int` | Required; exclusive final leaf CID. |
| `centroid` | `list<float>` | Required; exactly `D` finite elements. |

The root rows define ordered, adjacent, non-empty ranges that cover `[0, C)`.

`ivf_centroids` has this schema:

| Field | Type | Constraint |
|---|---|---|
| `cid` | `int` | Required; unique; in `[0, C)`. |
| `cid_bucket` | `int` | Required; root whose interval contains `cid`. |
| `centroid` | `list<float>` | Required; exactly `D` finite elements. |

The relation contains exactly `C` rows. Centroid training is implementation
specific. A source row is assigned to the centroid with the smallest squared
Euclidean distance; equal distances select the smaller `cid`.

For `cosine`, source vectors are normalized before training and assignment.
Training persists the centroids produced from those normalized inputs. A
centroid is generally their arithmetic mean and is not required to have unit
norm. Assignment and query routing use squared-L2 distance to the persisted
centroid as written; implementations must not normalize it again.

## 6. Postings

Every `ivf_postings` row starts with:

| Field | Type | Constraint |
|---|---|---|
| `cid` | `int` | Required; references `ivf_centroids.cid`. |
| `key_i`, `i = 1..K` | `T_i` | Required; source key field `i`. |

Additional fields depend on `posting_encoding`:

| Encoding | Additional fields | Candidate vector |
|---|---|---|
| `source` | None | Canonical vector read from the source row. |
| `lvq4` | `offset: float`, `scale: float`, `code: binary` | LVQ4 reconstruction. |
| `lvq8` | `offset: float`, `scale: float`, `code: binary` | LVQ8 reconstruction. |

Fields not listed for the selected encoding must be absent. Every postings
field is required. The relation contains exactly `N` rows, each source key
tuple occurs exactly once, and every posting resolves to exactly one source
row. Row position and physical file order have no semantic meaning.

Full source vectors are never stored in postings.

## 7. LVQ Encoding

LVQ encodes each canonical source vector independently. For vector `x`:

```text
offset = min(x_i)
upper  = max(x_i)
max_code = 15 for lvq4, otherwise 255
scale    = (upper - offset) / max_code
```

When `upper > offset`:

```text
code_i = clamp(
    round(max_code * (x_i - offset) / (upper - offset)),
    0,
    max_code
)
```

`round` selects the nearest integer and resolves an exact half toward the
larger integer. Implementations use sufficient intermediate precision to
apply this rule before storing the code. When `upper = offset`, every code is
zero. The inclusive range `[0, max_code]` intentionally provides 16 distinct
codes for LVQ4 and 256 for LVQ8.

LVQ8 stores `code_i` in byte `i`. LVQ4 stores even dimension `i` in the low
nibble of byte `i / 2` and the following odd dimension in its high nibble. An
unused final high nibble is zero. Code lengths are `D` bytes for LVQ8 and
`ceil(D / 2)` bytes for LVQ4.

The reconstructed value is:

```text
x_hat_i = offset + scale * code_i
```

`offset`, `scale`, and reconstructed values are finite; `scale` is
non-negative. For `cosine`, LVQ encodes the normalized source vector and the
reconstruction is not normalized again.

## 8. Source Contract

The source relation contains exactly `N` rows. Its ordered source-key fields
form a unique, non-null key. Its vector field is non-null; every vector has
exactly `D` non-null elements that produce finite canonical `float` values.
Cosine vectors additionally have a non-zero norm.

The source may contain additional payload columns. They are not copied into
the index. `_distance` is reserved and must not be a source field.

Writers may rely on source-key uniqueness and are not required to verify it.

## 9. Physical Layout

Parquet LVQ `code` is stored as `BYTE_ARRAY`; PLAIN encoding without a
dictionary is recommended. Parquet postings retain physical `cid` and use the
manifested, root-bucketed layout in the Parquet relation profile. One postings
row group contains exactly one CID. File and row-group boundaries are physical
tuning details subject to those invariants and have no logical identity.
