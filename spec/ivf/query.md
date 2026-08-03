# IVF Query Spec

## Overview

This spec defines query semantics for the IVF index schema in
[`index-schema.md`](index-schema.md). It does not prescribe SQL syntax,
functions, operators, or an execution strategy.

Version 1 defines IVF search with squared Euclidean distance and an optional
source-row filter.

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

## Search

Given the selected index snapshot, its source table, and its IVF index tables, a
query:

1. computes the squared Euclidean distance from `query-vector` to every
   centroid;
2. selects `nprobe` clusters ordered by `(distance ASC, cid ASC)`;
3. selects postings whose `cid` belongs to those clusters;
4. resolves candidate source-key tuples to source rows when required for
   filtering, distance evaluation, or result fields;
5. discards candidates whose source rows do not satisfy `filter`, when
   supplied;
6. computes squared Euclidean distance from `query-vector` to
   `ivf_postings.vector` when `store_vectors` is `true`, or to the resolved
   source vector otherwise;
7. selects at most `k` candidates ordered by `distance ASC`; and
8. returns the projected source fields in projection order followed by
   `_distance`.

The final distance must be computed from the exact indexed-point vector, not a
centroid or approximation. A stored posting vector is exact by the index-schema
requirements. If selected clusters contain fewer than `k` candidates, all
candidates are returned. Setting `nprobe = nlist` evaluates every indexed
point. The relative order of candidates with equal distance is unspecified.

## Distance

For vectors `q` and `x` of dimension `D`:

```text
distance(q, x) = SUM((q[i] - x[i]) * (q[i] - x[i])), i = 0..D-1
```

The square root is not computed. Query, centroid, posting-vector, and
source-vector elements have canonical type `float`, and `_distance` has
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
when `store_vectors` is `true`, no source-row filter requires it, and the
consumer's projection contains only source-key fields and the vector field.

## Result

The result contains the source fields selected by `projection`, in projection
order, followed by non-null `float` `_distance`. When `projection` is omitted,
the result contains every source field in source schema order followed by
`_distance`. Source names, canonical types, nullability, and values are
preserved.

`_distance` contains the final source-vector distance. The result is ordered by
`distance ASC`. The relative order of rows with equal distance is unspecified.

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

`ivf_postings` with `store_vectors = true`:

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
