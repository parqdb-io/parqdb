# IVF Index Spec

## Metadata

This spec defines index family `ivf` at `index-schema-version = 1`.
It supports metric `l2_squared`.

An IVF index snapshot must contain exactly the following semantic parameters:

| Key | Symbol | Definition |
|---|---|---|
| `dimension` | `D` | Vector dimension. |
| `nlist` | `C` | Number of IVF clusters. |
| `ntotal` | `N` | Number of indexed points. |
| `store_vectors` | `S` | Whether postings contain exact source vectors. |

`dimension`, `nlist`, and `ntotal` must use the canonical base-10
representation of a positive integer: one or more ASCII digits with no sign or
leading zero. `dimension` and `nlist` must be no greater than `2147483647`;
`ntotal` must be no greater than `9223372036854775807`. `nlist` must not exceed
`ntotal`.

`store_vectors` must be `true` or `false`.

`K` is the number of fields in the index snapshot's `source-key-fields` list and
must be positive.

The index snapshot must contain these index-table roles:

| Requirement | Role name | Description |
|---|---|---|
| required | `ivf_centroids` | Stores one centroid for each IVF cluster. |
| required | `ivf_postings` | Associates each source-key tuple with one IVF cluster. |

No other index-table role is defined for IVF schema version `1`. Every
resolved index table must be scoped to the selected index snapshot and expose
the corresponding schema below.

## Schema

IVF means IVF-Flat: IVF cluster pruning followed by exact distance evaluation
over vectors assigned to the selected clusters. When `store_vectors` is
`true`, the postings table contains an exact copy of each indexed source vector
and can evaluate candidate distances without reading source vectors. When it is
`false`, candidate vectors are read from the source table.

The schema is independent of query syntax and execution strategy.

## Types

The index tables use the following canonical Iceberg types:

| Type | Description |
|---|---|
| `int` | Iceberg signed 32-bit integer. |
| `long` | Iceberg signed 64-bit integer. |
| `float` | Iceberg IEEE 754 binary32 value. |
| `list<float>` | Iceberg ordered list with required `float` elements. |
| `T_i` | Type variable for source unique-key field `i`, as defined below. |

The `cid` values must be non-negative even though their type is signed. This
restriction does not apply to integer-valued source unique keys.

### Source-Key Types

The columns `key_1` through `key_K` correspond, in order, to the source
unique-key fields. The numeric suffix defines the field's position in a
composite key.

`T_i` is a schema type variable, not a persisted datatype. It denotes the exact
canonical Iceberg type of source unique-key field `i` as reported by the host
engine or mapped by the applicable non-Iceberg relation profile.

The `ivf_postings` table copies the source unique key so that each posting can
be joined directly to its source row. Each unique-key field must use one of the
following canonical Iceberg types:

- `boolean`;
- `int` or `long`;
- `binary` or `fixed(L)`;
- `string`;
- `date`.

Only signed integer keys are supported. Types not listed above must not be used
as source unique-key fields. All type parameters are part of the column type.

Key values are compared exactly under their declared canonical type. `string`,
`binary`, and `fixed(L)` values are compared byte-for-byte.

## Construction

Centroid training is implementation-specific. A writer may use any training
algorithm that produces `nlist` finite centroids with the required dimension.

Every source row is assigned to the centroid with the smallest squared
Euclidean distance under the arithmetic defined in
[`query.md`](query.md#distance). Equal distances are resolved by the smaller
`cid`.

The writer copies the source row's unique-key fields into the corresponding
`key_i` fields of `ivf_postings`. When `store_vectors` is `true`, it also copies
the source vector into `ivf_postings.vector`. Table row order is not
significant.

## Index Tables

### IVF Centroids (`ivf_centroids`)

Each row in this index table represents one IVF cluster and its centroid.

| Requirement | Field name | Type | Description |
|---|---|---|---|
| required | `cid` | `int` | Cluster ID; primary key. |
| required | `centroid` | `list<float>` | Cluster centroid. |

The table must satisfy the following requirements:

1. `cid` is in the range `[0, C)`.
2. `centroid` contains exactly `D` values.
3. Every value in `centroid` is finite.
4. The table contains exactly `C` rows.

### IVF Postings (`ivf_postings`)

Each row in this index table represents one indexed point and its IVF cluster
assignment.

| Requirement | Field name | Type | Description |
|---|---|---|---|
| required | `cid` | `int` | Cluster ID; references `ivf_centroids.cid`. |
| required | `key_i`, `i = 1..K` | `T_i` | Source unique-key field `i`. |
| required when `store_vectors = true` | `vector` | `list<float>` | Exact source vector. |

The table must satisfy the following requirements:

1. Every `cid` resolves to exactly one row in `ivf_centroids`.
2. The tuple `(key_1, ..., key_K)` is unique.
3. Each `key_i` has the same canonical Iceberg type, type parameters, and value
   as its corresponding source unique-key field.
4. When `store_vectors` is `true`, `vector` has exactly the same value as the
   corresponding source vector. When `store_vectors` is `false`, the `vector`
   field must be absent.
5. The table contains exactly `N` rows.

After cluster pruning, an execution path can evaluate exact candidate distances
from `ivf_postings.vector` when vectors are stored. Otherwise it joins selected
source-key tuples to the source table and reads source vectors there.

## Source Table

The indexed source is the host-engine table bound by the selected index
snapshot as defined by [`../metadata.md`](../metadata.md). The snapshot's
`vector-field` names the source vector column, and its ordered
`source-key-fields` list names the source unique key.

IVF schema version 1 defines a full index over the selected source table:

1. the source table contains exactly `N` rows;
2. every source row has one unique, non-null source-key tuple;
3. every source row corresponds to exactly one `ivf_postings` row; and
4. every `ivf_postings` row resolves to exactly one source row.

Writers may rely on source-key uniqueness and are not required to verify it
while building an index.

### Vector Column

The vector field of every source row must:

1. have canonical type `list<float>`; its schema may declare elements optional;
2. be non-null;
3. contain exactly `D` elements;
4. contain only non-null, finite elements.

The source may contain additional payload or metadata columns. Those columns
are not copied into the IVF index.

The source table must not contain a column named `_distance`, which is
reserved by the IVF public query result.
