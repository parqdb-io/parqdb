use std::collections::{HashMap, VecDeque};
use std::sync::{Arc, Mutex};

use bytes::Bytes;
use relify_meta::IndexMetadata;
use relify_storage::Warehouse;
use uuid::Uuid;

use crate::{Error, Result};

type MetadataCacheConfigResolver = dyn Fn() -> MetadataCacheConfig + Send + Sync;

/// Default maximum number of immutable metadata documents retained in memory.
pub const DEFAULT_METADATA_CACHE_ENTRIES: usize = 128;

/// Default metadata-cache budget, measured using serialized document sizes.
pub const DEFAULT_METADATA_CACHE_BYTES: usize = 16 * 1024 * 1024;

/// Bounds for the in-memory immutable metadata cache.
///
/// `max_bytes` is accounted from each document's serialized size. It provides
/// a stable cache budget, but is not an exact process-memory limit because the
/// parsed metadata also has allocator and object overhead.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct MetadataCacheConfig {
    /// Maximum number of cached metadata documents.
    pub max_entries: usize,
    /// Maximum total serialized size of cached metadata documents.
    pub max_bytes: usize,
}

impl MetadataCacheConfig {
    /// Creates explicit metadata-cache bounds.
    #[must_use]
    pub const fn new(max_entries: usize, max_bytes: usize) -> Self {
        Self {
            max_entries,
            max_bytes,
        }
    }
}

impl Default for MetadataCacheConfig {
    fn default() -> Self {
        Self::new(DEFAULT_METADATA_CACHE_ENTRIES, DEFAULT_METADATA_CACHE_BYTES)
    }
}

/// Immutable index metadata storage under one managed warehouse.
#[derive(Clone)]
pub struct MetadataStore {
    warehouse: Warehouse,
    cache: Arc<Mutex<MetadataCache>>,
    resolve_cache_config: Arc<MetadataCacheConfigResolver>,
}

impl std::fmt::Debug for MetadataStore {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("MetadataStore")
            .field("warehouse", &self.warehouse)
            .field("cache", &self.cache)
            .finish_non_exhaustive()
    }
}

#[derive(Debug)]
struct MetadataCache {
    config: MetadataCacheConfig,
    resident_bytes: usize,
    entries: HashMap<String, CachedMetadata>,
    order: VecDeque<String>,
}

#[derive(Debug)]
struct CachedMetadata {
    metadata: Arc<IndexMetadata>,
    serialized_bytes: usize,
}

impl MetadataStore {
    /// Creates a metadata store rooted in `warehouse`.
    #[must_use]
    pub fn open(warehouse: Warehouse) -> Self {
        Self::open_with_cache_config(warehouse, MetadataCacheConfig::default())
    }

    /// Creates a metadata store with explicit in-memory cache bounds.
    #[must_use]
    pub fn open_with_cache_config(warehouse: Warehouse, cache_config: MetadataCacheConfig) -> Self {
        Self::open_with_cache_config_resolver(warehouse, move || cache_config)
    }

    /// Creates a metadata store whose cache bounds are resolved before each
    /// cache access.
    ///
    /// This keeps a host session's configuration authoritative while allowing
    /// standalone stores to use fixed bounds through
    /// [`Self::open_with_cache_config`].
    #[must_use]
    pub fn open_with_cache_config_resolver(
        warehouse: Warehouse,
        resolve_cache_config: impl Fn() -> MetadataCacheConfig + Send + Sync + 'static,
    ) -> Self {
        let cache_config = resolve_cache_config();
        Self {
            warehouse,
            cache: Arc::new(Mutex::new(MetadataCache::new(cache_config))),
            resolve_cache_config: Arc::new(resolve_cache_config),
        }
    }

    /// Returns the active metadata-cache bounds.
    #[must_use]
    pub fn cache_config(&self) -> MetadataCacheConfig {
        self.cache().config
    }

    /// Returns the stable root recorded in metadata for an index UUID.
    pub fn index_location(&self, index_uuid: Uuid) -> Result<String> {
        Ok(self
            .warehouse
            .location(&format!("metadata/{index_uuid}"), true)?)
    }

    /// Validates and writes the first immutable metadata document.
    pub async fn write_initial(&self, metadata: &IndexMetadata) -> Result<String> {
        metadata.validate()?;
        self.write(metadata, "v1.metadata.json").await
    }

    /// Validates and writes an immutable successor metadata document.
    pub async fn write_update(
        &self,
        base: &IndexMetadata,
        metadata: &IndexMetadata,
    ) -> Result<String> {
        metadata.validate_update_from(base)?;
        let version = u64::try_from(metadata.last_sequence_number).map_err(|_| {
            Error::InvalidMetadata("metadata sequence number is out of range".into())
        })?;
        self.write(
            metadata,
            &format!("v{version}-{}.metadata.json", metadata.current_snapshot_id),
        )
        .await
    }

    /// Loads and decodes one immutable metadata document.
    pub async fn load(&self, location: &str) -> Result<IndexMetadata> {
        if let Some(metadata) = self.cache().get(location) {
            return Ok(metadata.as_ref().clone());
        }
        self.load_from_storage(location).await
    }

    /// Removes one metadata document from the in-memory cache.
    pub fn invalidate(&self, location: &str) {
        self.cache().remove(location);
    }

    pub(crate) async fn load_from_storage(&self, location: &str) -> Result<IndexMetadata> {
        let bytes = self.warehouse.read(location).await?;
        let serialized_bytes = bytes.len();
        let metadata = Arc::new(IndexMetadata::from_json_slice(&bytes)?);
        self.cache()
            .insert(location, Arc::clone(&metadata), serialized_bytes);
        Ok(metadata.as_ref().clone())
    }

    async fn write(&self, metadata: &IndexMetadata, filename: &str) -> Result<String> {
        let expected_location = self.index_location(metadata.index_uuid)?;
        if metadata.location != expected_location {
            return Err(Error::InvalidMetadata(
                "metadata location does not match the configured warehouse".into(),
            ));
        }
        let destination = self.warehouse.location(
            &format!("metadata/{}/{filename}", metadata.index_uuid),
            false,
        )?;
        let mut bytes = serde_json::to_vec_pretty(metadata)?;
        bytes.push(b'\n');
        let serialized_bytes = bytes.len();
        self.warehouse
            .put_new(&destination, Bytes::from(bytes))
            .await?;
        self.cache()
            .insert(&destination, Arc::new(metadata.clone()), serialized_bytes);
        Ok(destination)
    }

    fn cache(&self) -> std::sync::MutexGuard<'_, MetadataCache> {
        let mut cache = self
            .cache
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner);
        let config = (self.resolve_cache_config)();
        if cache.config != config {
            cache.set_config(config);
        }
        cache
    }
}

impl MetadataCache {
    fn new(config: MetadataCacheConfig) -> Self {
        Self {
            config,
            resident_bytes: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    fn get(&mut self, location: &str) -> Option<Arc<IndexMetadata>> {
        let metadata = Arc::clone(&self.entries.get(location)?.metadata);
        self.touch(location);
        Some(metadata)
    }

    fn insert(&mut self, location: &str, metadata: Arc<IndexMetadata>, serialized_bytes: usize) {
        self.remove(location);
        if self.config.max_entries == 0
            || self.config.max_bytes == 0
            || serialized_bytes > self.config.max_bytes
        {
            return;
        }
        self.entries.insert(
            location.to_owned(),
            CachedMetadata {
                metadata,
                serialized_bytes,
            },
        );
        self.resident_bytes += serialized_bytes;
        self.order.push_back(location.to_owned());
        self.enforce_bounds();
    }

    fn set_config(&mut self, config: MetadataCacheConfig) {
        self.config = config;
        self.enforce_bounds();
    }

    fn touch(&mut self, location: &str) {
        if let Some(position) = self.order.iter().position(|entry| entry == location) {
            self.order.remove(position);
        }
        self.order.push_back(location.to_owned());
    }

    fn remove(&mut self, location: &str) {
        if let Some(entry) = self.entries.remove(location) {
            self.resident_bytes -= entry.serialized_bytes;
        }
        if let Some(position) = self.order.iter().position(|entry| entry == location) {
            self.order.remove(position);
        }
    }

    fn enforce_bounds(&mut self) {
        while self.entries.len() > self.config.max_entries
            || self.resident_bytes > self.config.max_bytes
        {
            let Some(evicted) = self.order.pop_front() else {
                break;
            };
            if let Some(entry) = self.entries.remove(&evicted) {
                self.resident_bytes -= entry.serialized_bytes;
            }
        }
    }
}
