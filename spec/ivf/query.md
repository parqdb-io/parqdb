# IVF Query Spec

## Overview

This spec defines query semantics for IVF
[schema version 1](index-schema.md) and
[schema version 2](index-schema-v2.md). It does not prescribe SQL syntax,
functions, operators, or an execution strategy.

## Inputs

| Input | Type | Description |
|---|---|---|
| `query-vector` | `list<float>` | Query vector. |
| `nprobe` | positive integer | Number of clusters to search. |
| `k` | positive integer | Maximum result count. |
| `filter` | optional row predicate | Source rows eligible for the result. |
| `projection` | optional list of source field names | Source fields returned before `_distance`. |

`query-vector` must be non-null, contain exactly `dimension` finite values, and
have required elements. `nprobe` must be in `[1, nlist]`, and `k` must be
positive. When supplied, `projection` must be non-empty, contain no duplicate
names, and reference source fields only. When omitted, it is the source schema
fields in schema order.

The representation of `filter` is an API or protocol concern and is not
specified here. It may reference source columns only. A source row is eligible
when the predicate evaluates to `true`; `false` and `null` both exclude the
row.

## Candidate Vector

The index schema and posting encoding determine the vector used to rank each
candidate:

| Schema | Posting configuration | Candidate vector |
|---|---|---|
| v1 | `store_vectors = false` | Exact vector from the source row. |
| v1 | `store_vectors = true` | Exact `ivf_postings.vector`. |
| v2 | `posting_encoding = source` | Exact vector from the source row. |
| v2 | `posting_encoding = flat` | Exact `ivf_postings.vector`. |
| v2 | `posting_encoding = lvq4` or `lvq8` | Reconstructed vector `x_hat` defined by schema v2. |

An LVQ candidate vector is an approximation of the source vector. Search does
not rerank LVQ candidates against the source vector.

## Search

Given the selected index snapshot, its source table, and its IVF index tables, a
query:

1. computes the squared Euclidean distance from `query-vector` to every
   centroid;
2. selects `nprobe` clusters ordered by `(distance ASC, cid ASC)`;
3. selects postings whose `cid` belongs to those clusters;
4. resolves candidate source-key tuples to source rows when required for
   filtering, the candidate vector, or result fields;
5. discards candidates whose source rows do not satisfy `filter`, when
   supplied;
6. computes squared Euclidean distance from `query-vector` to the candidate
   vector defined above;
7. selects at most `k` candidates ordered by `distance ASC`; and
8. returns the projected source fields in projection order followed by
   `_distance`.

If selected clusters contain fewer than `k` candidates, all candidates are
returned. Setting `nprobe = nlist` evaluates every indexed point, but LVQ
distances remain approximate. The relative order of candidates with equal
distance is unspecified.

## Distance

For query vector `q` and candidate vector `x` of dimension `D`:

```text
distance(q, x) = SUM((q[i] - x[i]) * (q[i] - x[i])), i = 0..D-1
```

The square root is not computed. Query, centroid, posting-vector, reconstructed,
and source-vector elements have canonical type `float`, and `_distance` has
canonical type `float`.
Intermediate precision, evaluation order, reassociation, and fused operations
are implementation-specific. Results from different engines need not be
bit-for-bit identical.

A query fails without returning partial results if its final distance is
non-finite.

## Source Resolution

```text
ivf_postings.(key_1, ..., key_K)
    -> source[source-key-fields]
```

The ordered posting fields `key_1` through `key_K` correspond to
`source-key-fields` in the same order. Every candidate must resolve to exactly
one source row.

Source resolution is part of the query. Callers are not required to join
postings to the source table. An implementation may elide source resolution
when no source-row filter requires it and the projection contains only
source-key fields. The vector field may also be returned without source
resolution when the posting contains its exact value.

## Result

The result contains the source fields selected by `projection`, in projection
order, followed by non-null `float` `_distance`. When `projection` is omitted,
the result contains every source field in source schema order followed by
`_distance`. Source names, canonical types, nullability, and values are
preserved.

`_distance` contains the distance to the candidate vector. It is exact for
`source` and `flat` representations and approximate for LVQ representations.
The result is ordered by `distance ASC`. The relative order of rows with equal
distance is unspecified.

Callers may apply ordinary relational projection, filtering, joins, and
aggregation to the result. Such filtering is post-search and may return fewer
than `k` rows; it is distinct from the optional source-row filter, which is
applied before Top-K.

## Appendix A: Query Example

The following non-normative example uses `dimension = 2`, `nlist = 2`, and
`ntotal = 3`.

Source table:

| `document_id` | `embedding` |
|---|---|
| `a` | `[0.0, 0.0]` |
| `b` | `[1.0, 0.0]` |
| `c` | `[10.0, 0.0]` |

`ivf_centroids`:

| `cid` | `centroid` |
|---|---|
| `0` | `[0.5, 0.0]` |
| `1` | `[10.0, 0.0]` |

Schema v1 `ivf_postings` with `store_vectors = true`:

| `cid` | `key_1` | `vector` |
|---|---|---|
| `0` | `a` | `[0.0, 0.0]` |
| `0` | `b` | `[1.0, 0.0]` |
| `1` | `c` | `[10.0, 0.0]` |

For `query-vector = [0.0, 0.0]`, `nprobe = 1`, and `k = 2`, cluster `0` is
selected. The result is:

| `document_id` | `embedding` | `_distance` |
|---|---|---|
| `a` | `[0.0, 0.0]` | `0.0` |
| `b` | `[1.0, 0.0]` | `1.0` |
