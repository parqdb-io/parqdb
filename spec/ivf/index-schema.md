# IVF Index Schema

## 1. Scope

This document defines index family `ivf` at `index-schema-version = 1`.
Supported metrics are `l2_squared` and `cosine`. Supported postings encodings
are `source`, `lvq4`, and `lvq8`.

An IVF index consists of one immutable shared centroid artifact and one logical
index postings relation. Logical indexes over the same source state, vector
field, dimension, metric, cluster count, and clustering profile may reference
the same centroid artifact.

## 2. Logical Index Metadata

An IVF snapshot must contain exactly these parameters:

| Key | Definition |
|---|---|
| `dimension` | Vector dimension `D`. |
| `nlist` | Number of IVF clusters `C`. |
| `ntotal` | Number of indexed source rows `N`. |
| `posting_encoding` | `source`, `lvq4`, or `lvq8`. |
| `shared_ivf_fingerprint` | Deterministic UUID identifying the shared IVF descriptor. |
| `shared_ivf_uuid` | UUID of the referenced shared IVF artifact. |
| `shared_ivf_metadata_location` | Absolute URI of its immutable metadata file. |

`dimension`, `nlist`, and `ntotal` use the canonical base-10 representation of
a positive integer, without a sign or leading zero. `dimension` and `nlist`
must not exceed `2147483647`; `ntotal` must not exceed
`9223372036854775807`; and `nlist` must not exceed `ntotal`.

The snapshot contains exactly these index relation roles:

| Role | Definition |
|---|---|
| `ivf_centroids` | The centroid relation named by the shared IVF metadata. |
| `ivf_postings` | The postings owned by this logical index. |

The snapshot's `ivf_centroids` reference must equal the shared metadata's
centroid reference. Its source, vector field, dimension, metric, and `nlist`
must equal the corresponding shared descriptor fields.

## 3. Shared IVF Metadata

A shared IVF metadata document has these fields:

| Field | Definition |
|---|---|
| `format-version` | `1`. |
| `artifact-uuid` | Non-nil lowercase UUID of this immutable artifact. |
| `fingerprint` | Fingerprint of `descriptor`. |
| `location` | Absolute base URI assigned to the artifact. |
| `created-at-ms` | Non-negative Unix epoch time in milliseconds. |
| `descriptor` | Semantic identity defined below. |
| `centroids` | Relation reference for `ivf_centroids`. |

The descriptor contains, in order, `source`, `vector-field`, `dimension`,
`metric`, `nlist`, and `clustering-profile-version`. Version `1` is the only
clustering profile defined here.

The fingerprint is UUIDv5 using namespace
`2fb71e63-a27c-4fc5-9d6d-5070698dc398`. The UUID name is a semantic descriptor
encoded as compact UTF-8 JSON with fields in the order above and no
insignificant whitespace. Its `source` contains `profile` followed by `uri` for
Parquet, or `profile`, `table-uuid`, and `snapshot-id` for Iceberg. Iceberg
locator fields do not affect the fingerprint. JSON strings leave non-ASCII
characters unescaped and use the shortest RFC 8259 escape for characters that
must be escaped.

The fingerprint is a lookup key. Readers must still compare every semantic
descriptor field before using the artifact. Source references are compared by
profile-defined exact state, so Iceberg locator changes do not prevent reuse.

For a Parquet source, the files resolved by the descriptor URI must remain
unchanged for the lifetime of the shared artifact. Changed source contents or
a changed wildcard expansion require a new source URI and therefore a new
descriptor. Parquet cannot otherwise distinguish the two source states.

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
vector elements to `float` before training, assignment, encoding, or distance
evaluation. The source relation itself is not rewritten.

Each `key_i` corresponds to source key field `i` and uses the same canonical
type and value. Supported source-key types are `boolean`, `int`, `long`,
`binary`, `fixed(L)`, `string`, and `date`. String, binary, and fixed values
are compared byte-for-byte. Integer key types are signed.

## 5. Shared Centroids

`ivf_centroids` has this schema:

| Field | Type | Constraint |
|---|---|---|
| `cid` | `int` | Required; unique; in `[0, C)`. |
| `centroid` | `list<float>` | Required; exactly `D` finite elements. |

The relation contains exactly `C` rows. Centroid training is implementation
specific. A source row is assigned to the centroid with the smallest squared
Euclidean distance; equal distances select the smaller `cid`.

For `cosine`, source vectors are normalized before training and assignment.
Persisted centroids are used as written and are not normalized again.

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
levels = 15 for lvq4, otherwise 255
scale  = (upper - offset) / levels
```

When `upper > offset`:

```text
code_i = clamp(round(levels * (x_i - offset) / (upper - offset)), 0, levels)
```

`round` selects the nearest integer and resolves an exact half toward the
larger integer. Implementations use sufficient intermediate precision to
apply this rule before storing the code. When `upper = offset`, every code is
zero.

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
exactly `D` non-null finite elements. Cosine vectors additionally have a
non-zero norm.

The source may contain additional payload columns. They are not copied into
the index. `_distance` is reserved and must not be a source field.

Writers may rely on source-key uniqueness and are not required to verify it.

## 9. Physical Layout

The logical schema does not assign meaning to file boundaries, row groups,
Parquet encodings, compression, or row order. Parquet LVQ `code` is stored as
`BYTE_ARRAY`; PLAIN encoding without a dictionary is recommended. Writers may
partition postings by `cid` without changing index semantics.
