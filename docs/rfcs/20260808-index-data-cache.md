# Bounded Index Data Cache

## Summary

Replace the current all-or-nothing in-memory index path with a bounded,
read-through cache for decoded index data. Published Parquet or Iceberg data
remains the source of truth. A cache hit supplies decoded Arrow data; a miss
uses the standard DataFusion Parquet path and may populate the cache. Queries
use the same logical and physical scan contract in both cases.

The first implementation covers index relations in the local DataFusion
backend. It does not cache source tables or introduce a persistent local copy
of an index.

## Motivation

The current `cache_index()` implementation materializes every relation in one
index snapshot. Cached IVF postings then use `CachedIvfScanExec`, while an
uncached query uses DataFusion's Parquet scan. This path was useful for
measuring in-memory performance, but it is not a stable serving architecture:

- an index must fit entirely in memory;
- cache memory has no capacity or eviction policy;
- cached and uncached queries use different scan implementations;
- warming a new snapshot duplicates substantial data before the old snapshot
  is released;
- repeated Parquet decoding is avoided only after explicit whole-index
  materialization.

The index files are already immutable, partitioned, and selectively scanned.
Relify should retain those properties and cache only the hot decoded fragments
that actual queries touch.

## Goals

- Keep published index relations as the only durable source of truth.
- Bound cache memory independently of total index size.
- Reuse decoded Arrow data across queries and avoid repeated Parquet decoding
  for hot fragments.
- Preserve file and row-group pruning before cache lookup.
- Preserve projection, filtering, distance, and Top-K semantics across hits and
  misses.
- Use one cache-aware index scan contract instead of separate memory-only and
  storage-backed query implementations.
- Remain correct during concurrent queries, eviction, index refresh, and
  session shutdown.
- Provide enough metrics to distinguish storage I/O, Parquet decoding, cache
  lookup, and vector computation.

## Non-Goals

- Caching arbitrary source tables or SQL query results.
- A generic cache for every DataFusion data source.
- A distributed cache shared by multiple processes or machines.
- A persistent cache that must survive process restart.
- Replacing DataFusion's Parquet decoder.
- Adding a raw byte-range or local-disk cache in the first implementation.
- Automatically choosing a cache capacity from machine memory.

## User Experience

An index query always reads the current immutable index snapshot. When an
eligible decoded fragment is resident, the query reuses it. Otherwise, Relify
reads and decodes the fragment from the published relation. The query result
does not depend on whether any fragment was cached.

The cache has a session-level byte capacity. A zero capacity disables decoded
data caching without changing the scan plan's semantics. The exact public
configuration name and default are left unresolved by this RFC.

Read-through loading is automatic. The existing `cache_index()` and
`is_index_cached()` methods describe whole-index residency and do not fit a
partially resident cache; this RFC removes them. An explicit clear operation
and cache statistics replace `uncache_index()` and the boolean status method.
Workload-directed prewarming can be designed separately if cold-query latency
requires it.

## Reference Design

### Ownership and Scope

`relify-local` owns one `IndexDataCache` per `LocalSession`. Catalog and
metadata crates remain independent of Arrow, DataFusion, and memory policy.
The cache accepts only relations belonging to a validated, published index
snapshot.

The cache is an execution resource. It is never serialized into index metadata
or the SQLite catalog. Closing the session releases it without changing the
published index.

### Source of Truth and Identity

Cache correctness relies on exact immutable relation identity:

- a managed Parquet relation is identified by its published index snapshot,
  relation role, and canonical object path;
- an Iceberg relation is additionally bound to its table UUID and exact
  snapshot ID;
- an object version or ETag is included when the storage implementation
  provides one.

Relify-managed index objects must not be overwritten after publication. When a
new index snapshot is published, its relation identity changes. Old cache
entries may be removed eagerly to reclaim memory, but identity separation, not
invalidation timing, provides correctness.

### Cache Unit

The logical cache unit is a complete, unfiltered decoded fragment from one
physical Parquet row group. A fragment contains an ordered projection and one
or more Arrow `RecordBatch` values, each no larger than the configured
DataFusion batch size.

The key contains at least:

```text
exact relation identity
object path and available object version
row-group ordinal
ordered physical projection
```

Filters, query vectors, `nprobes`, and `k` are not part of the key. File and
row-group pruning happen before loading a fragment; row-level filters and
runtime filters are applied after the complete fragment is obtained. Filtered
or partially decoded output is not admitted because it cannot be safely reused
by another query. The projection identity includes physical field identities
and types, not only column names. Process restart naturally separates entries
decoded by different Relify or Arrow versions.

Row-group granularity is the proposed contract, not a requirement to implement
a new Parquet decoder. The DataFusion integration may use
`ParquetAccessPlan`, its standard Parquet source, and Arrow's standard decoder
to request complete row groups. The implementation must validate this adapter
with a prototype before the existing cache path is removed.

### Read Path

The cache-aware index scan performs these steps:

1. Resolve the exact published index snapshot.
2. Apply index pruning, including IVF cluster-to-file pruning.
3. Apply DataFusion file and row-group pruning.
4. Construct fragment keys for the remaining row groups and projection.
5. Return resident fragments directly.
6. Read cache misses through the standard DataFusion Parquet path.
7. Admit only successfully and completely decoded fragments.
8. Apply row-level and runtime filters, then continue to distance and Top-K
   execution.

Hits and misses may occur in one query. Their output schema, partitioning, and
ordering contract must be identical. Cache-specific operators must not contain
IVF distance or Top-K logic; they only provide decoded index data.

### Concurrency

Concurrent misses for the same key use single-flight loading. One task reads
and decodes the fragment while other tasks await the same result. A failed or
cancelled load is not cached, and another query may retry it.

Cache values are reference counted. Eviction removes an entry from future
lookup but cannot invalidate a fragment held by a running query. Capacity
accounting must continue to include an evicted value until its final query
reference is released; otherwise concurrent scans can exceed the configured
budget by repeatedly replacing pinned entries.

When capacity cannot be reserved without violating the hard cache budget, a
miss remains a normal storage-backed read and bypasses admission. Cache
pressure must not fail an otherwise valid query.

### Capacity and Eviction

The cache is weighted by retained Arrow buffer bytes plus entry overhead, not
by row or entry count. The implementation must avoid double-counting shared
buffers inside one value and must account conservatively when exact ownership
cannot be determined.

The first eviction policy is byte-weighted LRU. An entry larger than the total
capacity is never admitted. Admission occurs only after a complete fragment is
available, so a cancelled scan cannot leave a partial entry.

SLRU or frequency-aware admission may be added later if scans that are touched
once displace useful hot data. That policy change does not alter cache keys or
the scan contract.

The cache budget controls retained cache data. Temporary decoder buffers,
query output, distance state, and Top-K state remain query memory and continue
to follow DataFusion's execution-memory behavior.

### Existing Cache Migration

The final implementation removes the complete `CachedIndex` materialization,
`CachedIvfPostings`, `CachedIvfScanExec`, and the whole-index cache API.
Temporary coexistence is allowed only while performance and correctness are
compared during development.

The optimized cached postings scan currently maintains a `cid`-to-range
directory and over-partitions selected work. Equivalent cluster pruning and
parallelism must be preserved by planning physical files and row groups, not by
recreating a whole-index memory structure.

ADR 0001 remains the record of the first explicit cache design. Once this RFC
is implemented, a short follow-up ADR will mark that design as superseded.

### Observability

Session-level cache statistics must include:

- capacity, live bytes, and entry count;
- hit, miss, and single-flight wait counts;
- hit and miss bytes;
- admitted, evicted, oversized, and capacity-bypassed entries;
- storage-read and decode time for misses.

The cache-aware scan must expose per-plan hit and miss metrics through
`EXPLAIN ANALYZE`. Metrics must make it possible to prove that a warm query did
not read or decode a resident fragment.

### Failure and Consistency Behavior

- Missing or corrupt cache data is treated as a miss; the published relation
  remains authoritative.
- A storage or decode failure is returned exactly as it would be without the
  cache.
- Publishing or selecting a new snapshot cannot return data from an older
  snapshot because its exact relation identity differs.
- Eviction and explicit cache removal do not affect already running queries.
- Cache entries are not written back to index storage.

## Rollout and Validation

Implementation proceeds in three stages:

1. Build a row-group adapter prototype and prove that complete fragments can be
   mixed with standard DataFusion misses without changing results, projection,
   pruning, or runtime-filter behavior.
2. Add the bounded cache, single-flight loading, and metrics while retaining
   the old path only as a benchmark comparison.
3. Run correctness, concurrency, memory-pressure, GIST, and Wikipedia
   benchmarks, then remove the old complete-index path.

The implementation is ready to replace the old path only when it demonstrates:

- identical results for uncached, partially cached, and fully warm queries;
- no repeated Parquet decoding for cache hits;
- bounded live cache memory under concurrent eviction and pinned readers;
- no material cold-query regression attributable to cache integration;
- preserved scaling across configured DataFusion execution parallelism.

## Drawbacks

- Row-group cache entries may be too large for a small cache when an index was
  written with large row groups.
- Projection-specific entries can duplicate decoded columns across query
  shapes.
- Applying row-level filters after cache lookup may give up decoder-level
  filtering for cacheable scans. The prototype must measure this tradeoff and
  preserve row-group pruning.
- A session-owned cache duplicates hot data across multiple sessions in one
  process.
- Reference-counted readers make strict accounting more complex than a simple
  LRU whose values are never pinned.

## Alternatives

### Keep the complete explicit cache

This gives the simplest warm path and strong benchmark performance, but it
requires the complete index to fit in memory and preserves two independent
scan implementations. It does not meet the serving or memory-safety goals.

### Cache compressed byte ranges only

A `ParquetFileReaderFactory` or object-store layer can cache remote byte
ranges. This is useful for object-storage latency but does not remove Parquet
decompression, decoding, or Arrow allocation, which remained material in local
profiling. It is a possible second cache level, not a replacement for the
decoded cache.

### Rely on the operating-system page cache

The OS page cache avoids repeated local disk I/O but still leaves Parquet
decoding and Arrow materialization on every query. It also provides no portable
control over remote object storage.

### Cache native vector-index structures

A purpose-built in-memory postings representation can be faster, but it creates
a second index representation with separate correctness, invalidation, and
scan semantics. This RFC keeps Arrow fragments reusable by the normal relational
execution path.

### Cache arbitrary DataFusion plans

`DataFrame.cache()` materializes a complete result as a memory table. It does
not provide bounded fragment admission or transparent partial hits and would
turn query-specific plans into cache keys. The proposed cache is limited to
immutable Relify index relations.

## Prior Art

[Databend](https://github.com/databendlabs/databend/blob/main/src/query/storages/fuse/src/io/read/block/block_reader_merge_io_async.rs)
checks an in-memory cache of decoded Arrow arrays before its raw column-data
cache and remote storage. It keys data by immutable block path, column identity,
offset, and length, and only admits complete unfiltered arrays. This RFC adopts
the separation between decoded and raw caching but keeps DataFusion's reader
and Relify's index identities.

[StarRocks 4.x](https://docs.starrocks.io/docs/data_source/data_cache/)
similarly separates an in-memory Page Cache for decompressed data from a disk
Block Cache for fixed-size ranges of remote files.
[Velox](https://facebookincubator.github.io/velox/develop/memory.html)
combines a process-wide file cache with explicit memory-pressure reclamation.
These systems own their storage readers; Relify must provide the equivalent
boundary around DataFusion rather than importing an engine-specific reader.

## Unresolved Questions

The following decisions require review or prototype evidence before this RFC
is accepted:

1. What public session option sets cache capacity, and should the default be
   disabled or a fixed conservative size?
2. Can the DataFusion adapter request individual complete row groups without a
   material planning or cold-query overhead?
3. Should the first key cache an ordered multi-column projection, or should it
   cache independently decoded columns and reconstruct batches from column
   hits?
4. Should Iceberg index relations enter the first implementation, or follow
   after the Parquet path is validated?
5. Does byte-weighted LRU resist the expected IVF access pattern, or is SLRU
   required from the start?

## Future Work

- Add a raw memory or local-disk byte-range cache below the decoded cache for
  remote object storage.
- Coordinate cache and DataFusion query memory through a shared memory
  arbitration mechanism when DataFusion exposes a suitable reclaim callback.
- Share immutable cache entries across compatible sessions in one process.
- Add asynchronous or workload-directed prefetching.
- Let additional index families reuse the same cache-aware relation scan.
