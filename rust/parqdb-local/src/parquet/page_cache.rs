//! Bounded cache for complete decompressed Parquet Pages.

use std::mem::size_of;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use datafusion::datasource::listing::PartitionedFile;
use datafusion::execution::memory_pool::{MemoryLimit, MemoryPool};
use datafusion::prelude::SessionConfig;
use datafusion_datasource_parquet::ParquetPageCacheFactory;
use hashlink::LinkedHashMap;
use parquet::column::page::{CachedPage, PageCache};

use crate::config::parquet_page_cache_capacity;
use crate::runtime::effective_memory_limit;

const AUTOMATIC_CAPACITY_DIVISOR: usize = 5;
const FALLBACK_CAPACITY: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct ParquetFileCacheKey {
    location: Arc<str>,
    modified_ns: Option<i64>,
    size: u64,
}

impl ParquetFileCacheKey {
    pub(crate) fn new(location: impl Into<Arc<str>>, modified_ns: Option<i64>, size: u64) -> Self {
        Self {
            location: location.into(),
            modified_ns,
            size,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ParquetPageKey {
    file: ParquetFileCacheKey,
    page_offset: u64,
}

#[derive(Debug)]
struct CacheEntry {
    page: Arc<CachedPage>,
    charge: usize,
}

impl CacheEntry {
    fn pinned(&self) -> bool {
        Arc::strong_count(&self.page) > 1
    }
}

#[derive(Debug, Default)]
struct CacheState {
    entries: LinkedHashMap<ParquetPageKey, CacheEntry>,
    resident_bytes: usize,
}

#[derive(Debug, Default)]
struct CacheAccounting {
    live_bytes: AtomicUsize,
}

#[derive(Debug)]
struct TrackedPageBuffer {
    buffer: Bytes,
    accounting: Arc<CacheAccounting>,
    charge: usize,
}

impl AsRef<[u8]> for TrackedPageBuffer {
    fn as_ref(&self) -> &[u8] {
        self.buffer.as_ref()
    }
}

impl Drop for TrackedPageBuffer {
    fn drop(&mut self) {
        self.accounting
            .live_bytes
            .fetch_sub(self.charge, Ordering::Relaxed);
    }
}

#[derive(Debug, Default)]
struct CacheCounters {
    hits: AtomicU64,
    misses: AtomicU64,
    admissions: AtomicU64,
    evictions: AtomicU64,
    capacity_bypasses: AtomicU64,
    oversized_bypasses: AtomicU64,
}

/// Allocation and lookup counters for one runtime's Parquet Page cache.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ParquetPageCacheStats {
    /// Active byte capacity.
    pub capacity: usize,
    /// Bytes retained by entries currently eligible for lookup.
    pub resident_bytes: usize,
    /// Bytes removed from lookup but still referenced by active queries.
    pub retired_bytes: usize,
    /// Number of entries currently eligible for lookup.
    pub page_count: usize,
    /// Successful Page lookups.
    pub hits: u64,
    /// Unsuccessful Page lookups.
    pub misses: u64,
    /// Pages admitted to the cache.
    pub admissions: u64,
    /// Pages removed from lookup.
    pub evictions: u64,
    /// Admissions bypassed because all eviction candidates were pinned.
    pub capacity_bypasses: u64,
    /// Admissions bypassed because one Page exceeded the capacity.
    pub oversized_bypasses: u64,
}

#[derive(Debug)]
pub(crate) struct DecompressedParquetPageCache {
    capacity: AtomicUsize,
    state: Mutex<CacheState>,
    accounting: Arc<CacheAccounting>,
    counters: CacheCounters,
}

impl DecompressedParquetPageCache {
    pub(crate) fn new(capacity: usize) -> Self {
        Self {
            capacity: AtomicUsize::new(capacity),
            state: Mutex::new(CacheState::default()),
            accounting: Arc::new(CacheAccounting::default()),
            counters: CacheCounters::default(),
        }
    }

    pub(crate) fn bind(self: &Arc<Self>, file: ParquetFileCacheKey) -> Arc<dyn PageCache> {
        Arc::new(FilePageCache {
            cache: Arc::clone(self),
            file,
        })
    }

    fn capacity(&self) -> usize {
        self.capacity.load(Ordering::Relaxed)
    }

    pub(crate) fn set_capacity(&self, capacity: usize) {
        let mut state = self.state();
        self.capacity.store(capacity, Ordering::Relaxed);
        if capacity == 0 {
            self.retire_all(&mut state);
            return;
        }
        while self.live_bytes() > capacity && self.evict_one_unpinned(&mut state) {}
    }

    pub(crate) fn clear(&self) {
        self.retire_all(&mut self.state());
    }

    pub(crate) fn stats(&self) -> ParquetPageCacheStats {
        let state = self.state();
        let live_bytes = self.live_bytes();
        ParquetPageCacheStats {
            capacity: self.capacity.load(Ordering::Relaxed),
            resident_bytes: state.resident_bytes,
            retired_bytes: live_bytes.saturating_sub(state.resident_bytes),
            page_count: state.entries.len(),
            hits: self.counters.hits.load(Ordering::Relaxed),
            misses: self.counters.misses.load(Ordering::Relaxed),
            admissions: self.counters.admissions.load(Ordering::Relaxed),
            evictions: self.counters.evictions.load(Ordering::Relaxed),
            capacity_bypasses: self.counters.capacity_bypasses.load(Ordering::Relaxed),
            oversized_bypasses: self.counters.oversized_bypasses.load(Ordering::Relaxed),
        }
    }

    fn get(&self, key: &ParquetPageKey) -> Option<Arc<CachedPage>> {
        let mut state = self.state();
        if let Some(entry) = state.entries.to_back(key) {
            self.counters.hits.fetch_add(1, Ordering::Relaxed);
            Some(Arc::clone(&entry.page))
        } else {
            self.counters.misses.fetch_add(1, Ordering::Relaxed);
            None
        }
    }

    fn insert(&self, key: ParquetPageKey, page: CachedPage) -> Arc<CachedPage> {
        let mut state = self.state();
        if let Some(entry) = state.entries.to_back(&key) {
            return Arc::clone(&entry.page);
        }

        let charge = page
            .page()
            .buffer()
            .len()
            .saturating_add(size_of::<CachedPage>())
            .saturating_add(size_of::<ParquetPageKey>());
        let capacity = self.capacity.load(Ordering::Relaxed);
        if charge > capacity {
            self.counters
                .oversized_bypasses
                .fetch_add(1, Ordering::Relaxed);
            return Arc::new(page);
        }

        while self.live_bytes().saturating_add(charge) > capacity {
            if !self.evict_one_unpinned(&mut state) {
                self.counters
                    .capacity_bypasses
                    .fetch_add(1, Ordering::Relaxed);
                return Arc::new(page);
            }
        }

        self.accounting
            .live_bytes
            .fetch_add(charge, Ordering::Relaxed);
        let accounting = Arc::clone(&self.accounting);
        let page = page.map_buffer(|buffer| {
            Bytes::from_owner(TrackedPageBuffer {
                buffer,
                accounting,
                charge,
            })
        });
        let page = Arc::new(page);
        state.resident_bytes += charge;
        state.entries.insert(
            key,
            CacheEntry {
                page: Arc::clone(&page),
                charge,
            },
        );
        self.counters.admissions.fetch_add(1, Ordering::Relaxed);
        page
    }

    fn evict_one_unpinned(&self, state: &mut CacheState) -> bool {
        for _ in 0..state.entries.len() {
            let Some((key, entry)) = state.entries.pop_front() else {
                return false;
            };
            if entry.pinned() {
                state.entries.insert(key, entry);
                continue;
            }
            state.resident_bytes -= entry.charge;
            self.counters.evictions.fetch_add(1, Ordering::Relaxed);
            return true;
        }
        false
    }

    fn retire_all(&self, state: &mut CacheState) {
        let entries = std::mem::take(&mut state.entries);
        state.resident_bytes = 0;
        self.counters
            .evictions
            .fetch_add(entries.len() as u64, Ordering::Relaxed);
        drop(entries);
    }

    fn live_bytes(&self) -> usize {
        self.accounting.live_bytes.load(Ordering::Relaxed)
    }

    fn state(&self) -> std::sync::MutexGuard<'_, CacheState> {
        self.state
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }
}

#[derive(Debug)]
pub(crate) struct ParqDBParquetPageCacheFactory {
    cache: Arc<DecompressedParquetPageCache>,
    default_capacity: usize,
}

impl ParqDBParquetPageCacheFactory {
    pub(crate) fn new(cache: Arc<DecompressedParquetPageCache>, default_capacity: usize) -> Self {
        Self {
            cache,
            default_capacity,
        }
    }
}

impl ParquetPageCacheFactory for ParqDBParquetPageCacheFactory {
    fn update_config(&self, config: &SessionConfig) {
        self.cache
            .set_capacity(parquet_page_cache_capacity(config).unwrap_or(self.default_capacity));
    }

    fn create_page_cache(
        &self,
        object_store_url: &str,
        file: &PartitionedFile,
    ) -> Option<Arc<dyn PageCache>> {
        if self.cache.capacity() == 0 {
            return None;
        }
        let object = &file.object_meta;
        let location = format!("{object_store_url}#{}", object.location);
        let modified_ns = object.last_modified.timestamp_nanos_opt();
        Some(
            self.cache
                .bind(ParquetFileCacheKey::new(location, modified_ns, object.size)),
        )
    }
}

pub(crate) fn automatic_page_cache_capacity(memory_pool: &dyn MemoryPool) -> usize {
    match memory_pool.memory_limit() {
        MemoryLimit::Finite(limit) => limit / AUTOMATIC_CAPACITY_DIVISOR,
        MemoryLimit::Infinite | MemoryLimit::Unknown => effective_memory_limit()
            .map_or(FALLBACK_CAPACITY, |limit| {
                limit / AUTOMATIC_CAPACITY_DIVISOR
            }),
    }
}

#[derive(Debug)]
struct FilePageCache {
    cache: Arc<DecompressedParquetPageCache>,
    file: ParquetFileCacheKey,
}

impl PageCache for FilePageCache {
    fn get(&self, page_offset: u64) -> Option<Arc<CachedPage>> {
        self.cache.get(&ParquetPageKey {
            file: self.file.clone(),
            page_offset,
        })
    }

    fn insert(&self, page: CachedPage) -> Arc<CachedPage> {
        self.cache.insert(
            ParquetPageKey {
                file: self.file.clone(),
                page_offset: page.compressed_range().start,
            },
            page,
        )
    }
}

#[cfg(test)]
mod tests {
    use std::io::Cursor;
    use std::sync::Barrier;
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::array::{ArrayRef, Int32Array, RecordBatch};
    use arrow::datatypes::{DataType, Field, Schema};
    use parquet::arrow::ArrowWriter;
    use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
    use parquet::arrow::push_decoder::ParquetPushDecoderBuilder;
    use parquet::basic::{Compression, Encoding};
    use parquet::column::page::Page;
    use parquet::file::properties::WriterProperties;
    use parquet::file::reader::{ChunkReader, Length};
    use parquet::{DecodeResult, file::metadata::ParquetMetaData};

    use super::*;

    #[derive(Debug, Clone)]
    struct CountingBytes {
        bytes: Bytes,
        reads: Arc<AtomicUsize>,
    }

    impl Length for CountingBytes {
        fn len(&self) -> u64 {
            self.bytes.len() as u64
        }
    }

    impl ChunkReader for CountingBytes {
        type T = std::io::Cursor<Bytes>;

        fn get_read(&self, start: u64) -> parquet::errors::Result<Self::T> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            Ok(Cursor::new(
                self.bytes.slice(usize::try_from(start).unwrap()..),
            ))
        }

        fn get_bytes(&self, start: u64, length: usize) -> parquet::errors::Result<Bytes> {
            self.reads.fetch_add(1, Ordering::Relaxed);
            let start = usize::try_from(start).unwrap();
            Ok(self.bytes.slice(start..start + length))
        }
    }

    fn parquet_bytes() -> Bytes {
        let schema = Arc::new(Schema::new(vec![Field::new(
            "value",
            DataType::Int32,
            false,
        )]));
        let values: ArrayRef = Arc::new(Int32Array::from_iter_values(0..4_096));
        let batch = RecordBatch::try_new(Arc::clone(&schema), vec![values]).unwrap();
        let properties = WriterProperties::builder()
            .set_compression(Compression::SNAPPY)
            .set_data_page_row_count_limit(128)
            .build();
        let mut bytes = Vec::new();
        let mut writer = ArrowWriter::try_new(&mut bytes, schema, Some(properties)).unwrap();
        writer.write(&batch).unwrap();
        writer.close().unwrap();
        Bytes::from(bytes)
    }

    fn cached_page(offset: u64, length: usize, value: u8) -> CachedPage {
        CachedPage::try_new(
            Page::DataPage {
                buf: Bytes::from(vec![value; length]),
                num_values: u32::try_from(length).unwrap(),
                encoding: Encoding::PLAIN,
                def_level_encoding: Encoding::RLE,
                rep_level_encoding: Encoding::RLE,
                statistics: None,
            },
            offset..offset + u64::try_from(length).unwrap(),
            true,
        )
        .unwrap()
    }

    fn read_all(
        bytes: &Bytes,
        reads: &Arc<AtomicUsize>,
        page_cache: Arc<dyn PageCache>,
    ) -> Vec<i32> {
        let input = CountingBytes {
            bytes: bytes.clone(),
            reads: Arc::clone(reads),
        };
        let builder = ParquetRecordBatchReaderBuilder::try_new(input)
            .unwrap()
            .with_page_cache(page_cache);
        reads.store(0, Ordering::Relaxed);
        builder
            .build()
            .unwrap()
            .flat_map(|batch| {
                batch
                    .unwrap()
                    .column(0)
                    .as_any()
                    .downcast_ref::<Int32Array>()
                    .unwrap()
                    .values()
                    .to_vec()
            })
            .collect()
    }

    fn read_all_with_push_decoder(
        bytes: &Bytes,
        metadata: Arc<ParquetMetaData>,
        page_cache: Arc<dyn PageCache>,
        allow_storage_reads: bool,
    ) -> (Vec<i32>, usize) {
        let mut decoder = ParquetPushDecoderBuilder::try_new_decoder(metadata)
            .unwrap()
            .with_page_cache(page_cache)
            .build()
            .unwrap();
        let mut values = Vec::new();
        let mut range_count = 0;

        loop {
            match decoder.try_decode().unwrap() {
                DecodeResult::NeedsData(ranges) => {
                    assert!(allow_storage_reads, "warm decoder requested {ranges:?}");
                    range_count += ranges.len();
                    let chunks = ranges
                        .iter()
                        .map(|range| {
                            bytes.slice(
                                usize::try_from(range.start).unwrap()
                                    ..usize::try_from(range.end).unwrap(),
                            )
                        })
                        .collect();
                    decoder.push_ranges(ranges, chunks).unwrap();
                }
                DecodeResult::Data(batch) => values.extend_from_slice(
                    batch
                        .column(0)
                        .as_any()
                        .downcast_ref::<Int32Array>()
                        .unwrap()
                        .values(),
                ),
                DecodeResult::Finished => return (values, range_count),
            }
        }
    }

    #[test]
    fn warm_pages_bypass_reader_io_and_reuse_decompressed_payloads() {
        let bytes = parquet_bytes();
        let reads = Arc::new(AtomicUsize::new(0));
        let cache = Arc::new(DecompressedParquetPageCache::new(16 * 1024 * 1024));
        let file =
            ParquetFileCacheKey::new("memory://page-cache.parquet", Some(1), bytes.len() as u64);

        let cold = read_all(&bytes, &reads, cache.bind(file.clone()));
        let cold_reads = reads.load(Ordering::Relaxed);
        assert!(cold_reads > 0);
        assert!(cache.stats().admissions > 1);

        let warm = read_all(&bytes, &reads, cache.bind(file));
        assert_eq!(warm, cold);
        assert_eq!(reads.load(Ordering::Relaxed), 0);
        assert!(cache.stats().hits > 1);
    }

    #[test]
    fn warm_push_decoder_skips_column_chunk_ranges() {
        let bytes = parquet_bytes();
        let metadata = ParquetRecordBatchReaderBuilder::try_new(bytes.clone())
            .unwrap()
            .metadata()
            .clone();
        let cache = Arc::new(DecompressedParquetPageCache::new(16 * 1024 * 1024));
        let file =
            ParquetFileCacheKey::new("memory://push-decoder.parquet", Some(1), bytes.len() as u64);

        let (cold, cold_ranges) = read_all_with_push_decoder(
            &bytes,
            Arc::clone(&metadata),
            cache.bind(file.clone()),
            true,
        );
        assert!(cold_ranges > 0);

        let (warm, warm_ranges) =
            read_all_with_push_decoder(&bytes, metadata, cache.bind(file), false);
        assert_eq!(warm, cold);
        assert_eq!(warm_ranges, 0);
    }

    #[test]
    fn file_identity_prevents_stale_page_reuse() {
        let bytes = parquet_bytes();
        let reads = Arc::new(AtomicUsize::new(0));
        let cache = Arc::new(DecompressedParquetPageCache::new(16 * 1024 * 1024));
        let old_file =
            ParquetFileCacheKey::new("memory://replacement.parquet", Some(1), bytes.len() as u64);
        let new_file =
            ParquetFileCacheKey::new("memory://replacement.parquet", Some(2), bytes.len() as u64);

        read_all(&bytes, &reads, cache.bind(old_file));
        reads.store(0, Ordering::Relaxed);
        read_all(&bytes, &reads, cache.bind(new_file));

        assert!(reads.load(Ordering::Relaxed) > 0);
    }

    #[test]
    fn concurrent_admission_returns_one_canonical_page() {
        let cache = Arc::new(DecompressedParquetPageCache::new(16 * 1024));
        let file = ParquetFileCacheKey::new("memory://concurrent.parquet", Some(1), 4_096);
        let pages = cache.bind(file);
        let barrier = Arc::new(Barrier::new(3));
        let handles = (1..=2)
            .map(|value| {
                let pages = Arc::clone(&pages);
                let barrier = Arc::clone(&barrier);
                std::thread::spawn(move || {
                    barrier.wait();
                    pages.insert(cached_page(0, 512, value))
                })
            })
            .collect::<Vec<_>>();
        barrier.wait();
        let admitted = handles
            .into_iter()
            .map(|handle| handle.join().unwrap())
            .collect::<Vec<_>>();

        assert!(Arc::ptr_eq(&admitted[0], &admitted[1]));
        assert_eq!(cache.stats().admissions, 1);
        assert_eq!(cache.stats().page_count, 1);
    }

    #[test]
    fn pinned_pages_bypass_admission_and_remain_accounted_after_clear() {
        let cache = Arc::new(DecompressedParquetPageCache::new(16 * 1024));
        let file = ParquetFileCacheKey::new("memory://capacity.parquet", Some(1), 4_096);
        let pages = cache.bind(file);

        drop(pages.insert(cached_page(0, 512, 1)));
        let one_page_capacity = cache.stats().resident_bytes;
        cache.set_capacity(one_page_capacity);
        let pinned = pages.get(0).unwrap();

        drop(pages.insert(cached_page(512, 512, 2)));
        let bypassed = cache.stats();
        assert_eq!(bypassed.page_count, 1);
        assert_eq!(bypassed.capacity_bypasses, 1);

        cache.clear();
        let cleared = cache.stats();
        assert_eq!(cleared.resident_bytes, 0);
        assert_eq!(cleared.retired_bytes, one_page_capacity);

        drop(pinned);
        assert_eq!(cache.stats().retired_bytes, 0);
    }

    #[test]
    fn unreferenced_pages_are_evictable() {
        let cache = Arc::new(DecompressedParquetPageCache::new(16 * 1024));
        let file = ParquetFileCacheKey::new("memory://eviction.parquet", Some(1), 4_096);
        let pages = cache.bind(file);

        drop(pages.insert(cached_page(0, 512, 1)));
        cache.set_capacity(cache.stats().resident_bytes);
        drop(pages.insert(cached_page(512, 512, 2)));

        let stats = cache.stats();
        assert_eq!(stats.page_count, 1);
        assert_eq!(stats.evictions, 1);
        assert!(pages.get(0).is_none());
        assert!(pages.get(512).is_some());
    }

    #[test]
    fn retired_arrow_buffers_remain_in_the_capacity_budget() {
        let cache = Arc::new(DecompressedParquetPageCache::new(16 * 1024));
        let file = ParquetFileCacheKey::new("memory://retired.parquet", Some(1), 4_096);
        let pages = cache.bind(file);

        drop(pages.insert(cached_page(0, 512, 1)));
        let one_page_capacity = cache.stats().resident_bytes;
        cache.set_capacity(one_page_capacity);
        let retained_buffer = pages.get(0).unwrap().page().buffer().clone();

        drop(pages.insert(cached_page(512, 512, 2)));
        let retired = cache.stats();
        assert_eq!(retired.page_count, 0);
        assert_eq!(retired.resident_bytes, 0);
        assert_eq!(retired.retired_bytes, one_page_capacity);
        assert_eq!(retired.capacity_bypasses, 1);

        drop(retained_buffer);
        assert_eq!(cache.stats().retired_bytes, 0);
    }
}
