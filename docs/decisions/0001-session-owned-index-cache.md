# Session-Owned Index Cache

- Status: Accepted
- Date: 2026-07-28

## Context

Repeated local searches should be measurable after the complete index has been
loaded into memory, matching the lifecycle of an in-memory Faiss index. Cache
state is execution-local and must not become durable catalog metadata.

## Decision

Index caching is owned by the execution session. The local DataFusion
session exposes `cache_index`, `is_index_cached`, and `uncache_index`.
`cache_index` resolves the current snapshot through the index catalog and
materializes all of its index relations as decoded Arrow data. Generic
relations use DataFusion `DataFrame.cache()`. IVF postings are streamed into
batches no larger than the session's DataFusion batch size and retain a
zero-copy `cid`-to-range directory behind one stable `TableProvider`. Each scan
derives selected ranges from ordinary `cid` predicates, slices the shared Arrow
batches lazily, and uses the session's current target partition count. It
accepts DataFusion runtime filters only for `cid` and `key_i` columns. Query
planning does not allocate a provider per query. Uncached queries choose native
or relational cluster selection using the centroid-matrix heuristic, then
expose a static or dynamic cluster filter to the Parquet scan.

Stable relation names emitted by `Session.to_sql(query)` resolve the current session cache
when each physical scan is planned and otherwise fall back to immutable
Parquet. They do not retain a cached provider after `uncache_index` or index
invalidation. An already planned or running query may retain its selected
provider until that query completes.

The first implementation is explicit and session-scoped. It does not implement
capacity-based eviction, partial-index caching, or source-table caching.
Changing the session batch size requires rebuilding the cache.

## Consequences

Catalog implementations remain independent of execution engines and memory
management. Future host-engine sessions may map the operation to their native
cache primitive or report that it is unsupported. Index publication invalidates
stale session state without changing durable index data.
