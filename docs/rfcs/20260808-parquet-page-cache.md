# DecompressedParquetPageCache

## Problem

Relify currently exposes two index scan paths:

- `cache_index()` materializes a complete index as decoded Arrow data;
- an uncached query reads Parquet through DataFusion.

The first path is fast only when the complete index fits in memory. It has no
capacity or eviction policy, duplicates data during snapshot refresh, and
implements scan behavior separately from the storage-backed path.

Relify needs one storage-backed path with a bounded read-through cache.
Published Parquet or Iceberg data remains authoritative. Query correctness must
not depend on which data is resident.

## Decisions

| Question | Decision |
| --- | --- |
| What is cached? | Complete, validated, decompressed Parquet pages. Values remain in their Parquet encoding. |
| What is the cache granularity? | One physical page from one leaf column in one Parquet row group. |
| Where is the cache integrated? | At the Page-provider boundary: lookup before Page-body I/O, admission after validation and decompression, and output before definition-level and value decoding. |
| How is pruning preserved? | File, row-group, and column pruning determine which column chunks are opened before page lookup. |
| How are correctness and concurrency handled? | StarRocks-compatible file identity, per-page single-flight loading, atomic admission, and reference-counted page buffers. |
| How is memory controlled? | A session-owned budget that defaults to 20% of the machine memory limit, allocation-lifetime accounting, oversized-page bypass, and byte-weighted LRU eviction. |

`DecompressedParquetPageCache` is a Parquet reader cache, not an Arrow column
cache or the operating system's file cache. Arrow arrays created by a query may
reference cached Page buffers, but arrays and execution batches are not cache
entries.

## Scope

The cache abstraction is independent of vector indexes. The first
implementation enables it only for Parquet index relations in the local
DataFusion backend. It does not initially admit source-table Pages, query
results, or arbitrary DataFusion plans.

It also does not add:

- a persistent cache across process restarts;
- a cache shared by multiple processes;
- a local-disk cache;
- a replacement Parquet decoder;
- workload-driven automatic cache sizing; or
- a cache for non-Parquet formats.

## Architecture

```mermaid
flowchart LR
    query[Query Parquet data] --> prune[Prune files, row groups, and columns]
    prune --> chunk[Open selected column chunk]
    chunk --> page[Read next physical page]
    page --> lookup{Parquet Page cached}
    lookup -->|Yes| decode[Standard value decoder]
    lookup -->|No| read[Read compressed page bytes]
    read --> unpack[Validate and decompress page]
    unpack --> admit[Admit complete page]
    admit --> decode
    decode --> arrays[Query-local Arrow arrays]
    arrays --> search[Distance and Top-K]
```

Hits and misses return the same Parquet `Page` representation to the standard
decoder. The cache changes how a Page is obtained, not index semantics or query
results.

## StarRocks Reference Design

StarRocks 4.1 has a general remote-data cache below its file readers and a
separate in-memory Page cache inside its Parquet `PageReader`. The latter is the
reference for this RFC.

The StarRocks reader receives the start and length of a column chunk and walks
its Pages sequentially. For each current Page offset it:

1. looks up a key composed of the file cache key and Page offset;
2. on a hit, deserializes the cached Page header, obtains the cached body, and
   advances to the next Page without reading the body from storage; or
3. on a miss, reads the Page header and body, optionally decompresses the body,
   and admits the complete Page.

The cached header supplies the compressed and uncompressed lengths needed to
advance the reader. This path does not require a Parquet OffsetIndex. StarRocks
may cache either compressed or decompressed bodies according to compression
benefit; Relify deliberately admits only decompressed bodies so a warm hit also
avoids decompression.

StarRocks constructs a compact file key from a 64-bit filename hash, a cache
type prefix, and a 32-bit value derived from modification time or file size. It
appends the 64-bit Page offset for Page-cache lookup. Modification time is
preferred; file size is the fallback for sources that do not expose
modification time because their files are not overwritten.

StarRocks returns a reference-counted `PageHandle` with each hit. An entry held
by an active reader is not eligible for LRU eviction. Its LRU implementation
may temporarily exceed capacity when all candidates are pinned. Relify follows
the same lifetime model, with the stricter admission behavior defined in Clause
6.

## 1. Cache Unit

### Cached value

A cache entry contains one complete Parquet `DataPage`, `DataPageV2`, or
`DictionaryPage` after decompression. It retains:

- the Page metadata required by the standard decoder;
- the decompressed, still encoded payload as reference-counted `Bytes`;
- the compressed span needed to advance the physical reader; and
- its retained-memory charge.

The cache never stores a filtered Page fragment. Row selections and runtime
filters are query state applied after lookup.

Each entry records the canonical physical span and decoded size:

```rust
struct PageDescriptor {
    page_type: PageType,
    compressed_range: Range<u64>,
    uncompressed_length: u32,
}
```

`compressed_range` starts at the Page header and ends after the compressed
body. Its end is also the next physical Page offset. The descriptor is cached
metadata used to advance the reader and validate other metadata; it is not part
of the cache key.

### Cache key

The conceptual key is:

```rust
struct ParquetPageKey {
    file_cache_key: FileCacheKey,
    page_offset: u64,
}
```

`page_offset` is the physical offset of the Page header. `file_cache_key`
follows StarRocks: it hashes the filename and cache type, then includes the
file's modification time when available or its file size otherwise. The file
metadata used by the scan supplies these values.

Row-group and leaf-column identity may be retained as entry metadata for
validation and diagnostics. They are not part of the key because the file
cache key and Page offset identify the cached Page.

The cache does not accept arbitrary `(offset, length)` reads. The reader begins
at the column-chunk offset from Parquet metadata. On a miss it learns the Page
span from the Page header; on a hit the cached descriptor supplies the same
span. The end of that span becomes the next Page offset. No OffsetIndex is
required.

### Why Page granularity

A Page is the smallest reusable Parquet unit that can be validated and
decompressed independently. It preserves column projection, does not depend on
DataFusion batch boundaries, and has one clear backing allocation for
accounting and eviction.

Rejected units are:

| Unit | Reason not selected |
| --- | --- |
| Complete index | Requires the complete index to fit in memory. |
| File or row group | Retains unrelated columns and pages. |
| Decoded row-group column | Couples residency to Arrow materialization and batch assembly. |
| Compressed byte range | Is not Page-aware and still pays Page decompression on every hit. |
| Filtered Arrow batch | Is query-specific and unsafe to reuse. |

## 2. Reader Boundary

The relevant DataFusion path is:

```text
DataSourceExec
  -> FileStream / ParquetMorselizer
  -> ParquetPushDecoder
  -> Page production
  -> definition-level and value decoders
  -> RecordBatch
```

`AsyncFileReader` returns compressed file ranges. Caching those ranges is a
byte-range cache, not `DecompressedParquetPageCache`. The cache belongs at the
next boundary, where a reader can identify a physical Page and either:

1. return a resident decompressed `Page`; or
2. read its header and compressed body, validate and decompress it, atomically
   admit it, and return the same `Page` representation.

Lookup happens at the current sequential Page offset before reading its body.
The first Page offset comes from the column-chunk metadata. A miss reads the
header to discover the compressed body length; a hit obtains that length from
the cached Page metadata. Both paths then advance to the same next Page offset.

DataFusion 54's `ParquetFileReaderFactory` and `AsyncFileReader` expose byte
ranges below this boundary. `ParquetPushDecoder` does not currently expose a
public Page-provider hook. The first implementation gate is therefore a small
prototype that provides this boundary through either:

- a generic upstream Page-provider hook; or
- a Relify-owned cache-aware `PageReader` adapter that reuses arrow-rs Page and
  value decoders.

Relify vendors the required arrow-rs/DataFusion reader change and keeps it
narrowly scoped to Page production and lifetime. It must not fork or copy the
value decoders. A cache implemented only in `AsyncFileReader`, or by retaining
output `RecordBatch` objects, does not satisfy this RFC.

The cache and adapter are generic index infrastructure. Their interfaces
contain no cluster, distance, quantization, or Top-K concepts.

## 3. PLAIN BYTE_ARRAY Fast Path

LVQ codes can be stored as required Parquet `BYTE_ARRAY` values using PLAIN
encoding. Their fixed byte length remains an index-schema requirement and is
validated when a Page is first consumed.

The DataFusion scan must request Arrow `BinaryView`. For a PLAIN `BYTE_ARRAY`,
arrow-rs parses each four-byte length prefix and creates query-local View
metadata whose payload points into the Page buffer. It does not copy the code
bytes.

```text
DecompressedParquetPageCache entry: decompressed PLAIN payload
                                   |
                                   +-- BinaryViewArray for query A
                                   +-- BinaryViewArray for query B
```

The `BinaryViewArray` retains a reference to the Page buffer. Evicting the Page
from lookup cannot invalidate an array used by a running query. The allocation
is released only after both the cache and all active arrays drop their
references.

PLAIN does not remove all decoder work. Each query still parses length prefixes
and creates View metadata. It avoids payload materialization, which is the
dominant copy for large LVQ codes. Other Parquet types continue through their
standard decoders and may copy values into Arrow buffers.

## 4. Pruning and Filtering

Caching must not weaken the storage-backed scan path.

| Existing optimization | Required behavior with the cache |
| --- | --- |
| File pruning | Reject files before creating Page requests. |
| Row-group pruning | Reject row groups before creating Page requests. |
| Column projection | Request Pages only for projected leaf columns. |
| Predicate pushdown | Reuse Pages, then apply the query's predicate and row selection through the standard decoder. |
| Dynamic filters | Apply new file, row-group, and row-selection information through the standard scan path; dynamic values never enter cache keys. |

A query that consumes only part of a Page may still admit the complete Page
because the cache entry is immutable and query-independent. Dictionary Pages
use the same cache and retain their normal dependency ordering within a column
chunk.

## 5. Consistency and Concurrency

File identity and invalidation follow StarRocks. The cache key uses the filename
and modification time when the storage exposes one. If modification time is
unavailable, it uses file size; this is intended for sources whose files are not
overwritten. A modified file therefore receives a different cache key when its
reported modification time or fallback size changes.

Deleting a file or removing a table does not synchronously invalidate its cache
entries. Once metadata no longer selects the file, new scans cannot reach those
keys. Unreferenced entries remain until normal LRU eviction. Eviction removes
the lookup mapping, while a running reader's Page handle keeps the underlying
buffer alive until that reader releases it.

An entry becomes visible only after the complete Page has been read, parsed,
decompressed, and structurally validated. Failed, partial, corrupt, or
cancelled loads are not cached. The original storage or decode error is
returned to the query.

Parquet CRC handling remains the responsibility of the standard reader when a
file provides it. CRC is not a cache key, descriptor field, or
`DecompressedParquetPageCache` requirement.

Concurrent misses for the same key use single-flight loading. One task reads
and prepares the Page; other tasks await the same result. Requests for different
Pages proceed independently.

Eviction and explicit clearing remove an entry from future lookup but do not
invalidate references held by running decoders or Arrow arrays.

## 6. Memory and Eviction

`relify-local` owns one `DecompressedParquetPageCache` per `LocalSession`. The
cache is an execution resource and is never serialized into index metadata or
the catalog. Closing the session releases its cache references.

Capacity is charged against the decompressed Page allocation and entry
overhead. The memory reservation remains attached to the backing allocation
until its final Page or Arrow reference is released. Removing a pinned entry
from LRU lookup therefore moves its bytes to a retired count; it does not
release the reservation early.

Query-local `BinaryView` metadata, numeric Arrow buffers, distance state, and
Top-K state are query memory rather than `DecompressedParquetPageCache` memory.

The initial policy is byte-weighted LRU. It must:

- bypass admission for a Page larger than the complete cache capacity;
- evict least-recently-used resident Pages before admission;
- include retired but pinned allocations in the memory budget; and
- allow a valid query to continue without admission under cache pressure.

Frequency-aware admission may replace LRU if benchmarks show that sequential
scans evict useful hot Pages.

## 7. Interfaces and Metrics

Capacity is a session variable. It accepts either a percentage of the machine
memory limit or an absolute byte size:

```python
session.set("relify.parquet_page_cache_capacity", "20%")
session.set("relify.parquet_page_cache_capacity", "4GiB")
```

The same variable is available through SQL:

```sql
SET relify.parquet_page_cache_capacity = '4GiB';
```

The default is `20%`. The percentage is resolved from the machine memory limit
when the session creates its cache. `0` disables admission and retires resident
entries. Operational methods are:

```python
stats = session.parquet_page_cache_stats()
session.clear_parquet_page_cache()
```

The existing `cache_index()`, `is_index_cached()`, and `uncache_index()` APIs
are removed because bounded partial residency cannot be represented by a
whole-index boolean.

The private cache operation is conceptually:

```rust
async fn get_or_load_page(
    key: ParquetPageKey,
    load: LoadPage,
) -> Result<Arc<CachedParquetPage>>;
```

Session and `EXPLAIN ANALYZE` metrics must report:

- capacity, resident bytes, retired bytes, and Page count;
- Page hits, misses, single-flight waits, admissions, and evictions;
- hit, miss, compressed-read, and decompressed bytes;
- oversized and capacity bypasses; and
- storage-read, decompression, and value-decoder time.

A warm Page hit must perform no storage read or decompression. Value decoding
still occurs; the PLAIN `BYTE_ARRAY` fast path should show View construction
without payload copying.

## Rollout

1. Prototype the Page-provider boundary and prove that a hit avoids reading the
   compressed Page body while preserving the standard arrow-rs decoder.
2. Benchmark `FIXED_LEN_BYTE_ARRAY` materialization against PLAIN `BYTE_ARRAY`
   with `BinaryView` for complete Parquet scan, LVQ distance, and Top-K.
3. Implement identity, single-flight loading, accounting, eviction, and
   metrics.
4. Test cold, partial-hit, and warm scans under pruning, concurrent loads,
   cancellation, file replacement, file deletion, and memory pressure.
5. Run GIST and Wikipedia benchmarks, then remove the whole-index cache path.

The old path is removed only after the new path demonstrates:

- identical results for cold, partial-hit, and warm queries;
- no storage I/O or decompression for resident Pages;
- no LVQ code-payload copy on the PLAIN `BYTE_ARRAY` path;
- bounded memory with concurrent eviction and pinned readers;
- no material cold-query regression; and
- preserved pruning and parallel scaling.

## Prior Art

Databend and StarRocks place caches inside storage readers they own. Neither
provides a reader crate that Relify can adopt unchanged.

| System | Cache design | Difference from Relify |
| --- | --- | --- |
| Databend Fuse | Decoded block columns and compressed column-chunk bytes | Fuse controls its writer, block metadata, and Parquet deserializer; a managed block is typically one single-row-group file. |
| StarRocks external Parquet | Remote byte-range cache plus Page caching in its Parquet reader | This RFC follows its sequential Page discovery, file identity, deletion, and handle-lifetime model, while always caching decompressed bodies and enforcing a hard admission budget. |
| Relify | Decompressed Parquet Pages consumed by arrow-rs decoders | Relify keeps open Parquet files and must add a narrow Page-provider boundary to DataFusion's reader. |

The PLAIN `BYTE_ARRAY` layout changes the trade-off relative to Databend's
decoded-column cache. The Page payload can remain authoritative in memory while
Arrow `BinaryView` arrays reference it directly, so Relify does not need a
second decoded representation for LVQ codes.

Relevant implementations:

- [DataFusion 54 Parquet source](https://github.com/apache/datafusion/blob/54.0.0/datafusion/datasource-parquet/src/source.rs)
- [arrow-rs 58 PageReader](https://github.com/apache/arrow-rs/blob/58.3.0/parquet/src/column/page.rs)
- [Databend Fuse block reader](https://github.com/databendlabs/databend/blob/3c73a7f73585d8ace3ffcd568c4bdc7dd30f71c0/src/query/storages/fuse/src/io/read/block/block_reader_merge_io_async.rs)
- [StarRocks 4.1 Parquet PageReader](https://github.com/StarRocks/starrocks/blob/4.1.0/be/src/formats/parquet/page_reader.cpp)
- [StarRocks file cache key](https://github.com/StarRocks/starrocks/blob/4.1.0/be/src/formats/parquet/utils.cpp)
- [StarRocks Page cache](https://github.com/StarRocks/starrocks/blob/4.1.0/be/src/cache/mem_cache/page_cache.h)
- [StarRocks reference-counted LRU](https://github.com/StarRocks/starrocks/blob/4.1.0/be/src/util/lru_cache.cpp)
- [StarRocks Data Cache deletion behavior](https://docs.starrocks.io/docs/using_starrocks/caching/block_cache/)

## Alternatives

- **Complete explicit cache:** provides the fastest warm path but requires the
  complete index to fit in memory and preserves two scan implementations.
- **Decoded row-group-column cache:** avoids repeated value decoding but couples
  residency to Arrow materialization and complicates batch alignment, Page
  lifetime, and shared-buffer accounting.
- **Compressed byte-range cache:** avoids remote reads but repeats Page parsing,
  decompression, and value decoding. It may become a lower cache tier.
- **OS file cache:** helps local file I/O but not remote object stores or
  Parquet decompression. It is separate from
  `DecompressedParquetPageCache`.
- **Native vector structures:** may be faster but create another index format
  and separate scan semantics.

## Unresolved Questions

1. Can the Page-provider boundary be implemented as a narrow maintained adapter
   without copying the arrow-rs decoder?
2. What Page size best balances I/O granularity, decompression, View metadata,
   and cache admission for LVQ codes?
3. Is byte-weighted LRU sufficient for IVF access patterns, or is admission
   control required initially?
4. Should Iceberg enter the first implementation or follow the local Parquet
   prototype?

## Future Work

- Add a compressed memory or local-disk byte-range cache below
  `DecompressedParquetPageCache`.
- Coordinate `DecompressedParquetPageCache` and DataFusion query memory under
  shared pressure.
- Share Page entries across compatible sessions.
- Add workload-directed Page prefetching.
- Reuse the page-aware scan path across additional index families.
