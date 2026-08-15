# IVF Query Spec

## 1. Inputs

| Input | Type | Definition |
|---|---|---|
| `query-vector` | `list<float>` or `list<double>` | Query vector. |
| `nprobe` | positive integer | Number of clusters to search. |
| `k` | positive integer | Maximum result count. |
| `filter` | optional row predicate | Source rows eligible before Top-K. |
| `projection` | optional source field list | Fields returned before `_distance`. |

The query vector contains exactly `dimension` finite values. `nprobe` is in
`[1, nlist]`, and `k` is positive. A projection is non-empty, contains no
duplicates, and names source fields only. An omitted projection selects all
source fields in schema order.

## 2. Search

A query:

1. converts the query vector to canonical `float` values;
2. normalizes it when the metric is `cosine`;
3. selects `nprobe` centroids ordered by squared-L2 distance and then `cid`;
4. reads postings in the selected clusters;
5. resolves source rows required by source encoding, filtering, or projection;
6. applies the optional source predicate;
7. computes candidate distance from the representation defined below;
8. retains at most `k` rows ordered by distance; and
9. returns projected source fields followed by `_distance`.

When fewer than `k` eligible candidates exist, all are returned. Equal final
distances have unspecified order.

## 3. Candidate Distance

For `l2_squared`:

```text
distance(q, x) = SUM((q[i] - x[i]) * (q[i] - x[i])), i = 0..D-1
```

For `cosine`, source and query vectors are normalized by their L2 norm before
use, and the reported value is:

```text
distance(q, x) = squared_l2(normalize(q), normalize(x)) / 2
```

A zero-norm cosine query is invalid. Centroids are used as persisted for
routing. LVQ reconstructs `x_hat` from normalized source data and reports
`squared_l2(normalize(q), x_hat) / 2`; `x_hat` is not normalized again.

`source` encoding therefore returns exact distances under the canonical float
contract. LVQ4 and LVQ8 return approximate distances and do not rerank against
the source vector.

Intermediate precision and operation order are implementation specific.
Different engines need not return bit-identical floating-point values. A query
fails without partial results if a final distance is non-finite.

## 4. Source Resolution

```text
ivf_postings.(key_1, ..., key_K)
    -> source[source-key-fields]
```

Every candidate resolves to exactly one source row. Implementations may defer
source resolution until after Top-K when neither source encoding nor a source
filter requires the row earlier.

## 5. Result

The result contains projected source fields in projection order followed by a
required `float` field named `_distance`. Source field names, canonical types,
nullability, and values are preserved.

Post-search relational filtering, joins, and aggregation operate on this
result and are distinct from the optional pre-Top-K source filter.

## Appendix A: Example

For centroids `[0.5, 0.0]` and `[10.0, 0.0]`, source vectors `[0.0, 0.0]`,
`[1.0, 0.0]`, and `[10.0, 0.0]`, query `[0.0, 0.0]`, `nprobe = 1`, and
`k = 2`, cluster `0` is selected. Source encoding returns:

| source key | `_distance` |
|---|---|
| `a` | `0.0` |
| `b` | `1.0` |
