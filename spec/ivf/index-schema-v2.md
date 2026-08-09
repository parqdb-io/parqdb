# IVF Index Schema Version 2

IVF schema version 2 separates cluster routing from the vector representation
stored in the postings table. It supports metric `l2_squared` and the same
source-key types and `ivf_centroids` relation as
[schema version 1](index-schema.md).

## Metadata

An IVF v2 snapshot contains exactly these parameters:

| Key | Definition |
|---|---|
| `dimension` | Positive vector dimension. |
| `nlist` | Positive number of IVF clusters. |
| `ntotal` | Positive number of indexed points. |
| `posting_encoding` | One of `source`, `flat`, `lvq4`, or `lvq8`. |

The numeric parameters and relation roles follow the constraints in schema
version 1. The v1 parameter `store_vectors` must not appear in a v2 snapshot.

## Postings

Every `ivf_postings` row contains the required `cid` and `key_i` fields defined
by schema version 1. Its remaining fields are determined by
`posting_encoding`:

| Encoding | Additional required fields |
|---|---|
| `source` | None. Candidate vectors are read from the source table. |
| `flat` | `vector: list<float>`. |
| `lvq4` | `offset: float`, `scale: float`, `code: binary`. |
| `lvq8` | `offset: float`, `scale: float`, `code: binary`. |

Fields not listed for the selected encoding must be absent. Every postings
field is required. The source-key, row-count, and cluster-assignment
requirements from schema version 1 continue to apply.

Every `lvq4` code value must contain exactly `ceil(D / 2)` bytes. Every `lvq8`
code value must contain exactly `D` bytes.

## LVQ Encoding

LVQ encodes each source vector independently. For a source vector `x`, define:

```text
offset = min(x_i)
upper  = max(x_i)
levels = 15 for lvq4, otherwise 255
scale  = (upper - offset) / levels
```

When `upper` is greater than `offset`, dimension `i` is encoded as:

```text
code_i = clamp(round(levels * (x_i - offset) / (upper - offset)), 0, levels)
```

`round` selects the nearest integer and resolves an exact half toward the
larger integer. Implementations must use sufficient intermediate precision to
preserve this rule before storing the code. When `upper` equals `offset`, every
code is zero. `offset`, `scale`, and every reconstructed value must be finite;
`scale` must be non-negative.

For `lvq8`, byte `i` stores `code_i`. For `lvq4`, byte `i / 2` stores an even
dimension in its low nibble and the following odd dimension in its high
nibble. The unused high nibble for an odd dimension must be zero.

The reconstructed value used by search is:

```text
x_hat_i = offset + scale * code_i
```

LVQ candidate distances and ordering are computed from `x_hat`, not the exact
source vector. Cluster selection continues to use the exact centroids stored
in `ivf_centroids`.

## Physical Layout

The logical schema does not assign meaning to postings row order, file
boundaries, Parquet row groups, encodings, or compression. In the Parquet
profile, `code` is stored as `BYTE_ARRAY`. PLAIN encoding without dictionary is
recommended so implementations can scan packed codes sequentially without
dictionary indirection. Compression remains an implementation choice.
Implementations may organize postings by `cid` without changing the published
index semantics.
