//! Session-scoped planning state for immutable index relations.

mod centroid;
mod hive_cid;

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::sync::{Arc, Mutex, RwLock};

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{ScanArgs, ScanResult, Session, TableProvider};
use datafusion::common::DataFusionError;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use hashlink::LinkedHashMap;
use relify_storage::StorageRegistry;
use tokio::sync::OnceCell;

use centroid::CentroidCache;
pub(super) use centroid::CentroidMatrix;
use hive_cid::HiveCidParquetProvider;

use crate::parquet::uniform_dataset_listing_table;
use crate::{Error, Result};

const DEFAULT_PARQUET_PROVIDER_CACHE_ENTRIES: usize = 128;
const DEFAULT_PARQUET_PROVIDER_CACHE_BYTES: usize = 64 * 1024 * 1024;
const PLAIN_PROVIDER_CHARGE: usize = 4 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum IndexRelationLayout {
    Plain,
    HiveCid,
}

impl IndexRelationLayout {
    fn filter_pushdown(self, filter: &Expr) -> TableProviderFilterPushDown {
        match self {
            Self::Plain => TableProviderFilterPushDown::Inexact,
            Self::HiveCid if hive_cid::cid_filter_values(filter).is_some() => {
                TableProviderFilterPushDown::Exact
            }
            Self::HiveCid => TableProviderFilterPushDown::Unsupported,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ParquetProviderKey {
    relation_key: String,
    layout: IndexRelationLayout,
}

type Provider = Arc<dyn TableProvider>;
type ProviderCell = Arc<OnceCell<ProviderValue>>;

struct ProviderValue {
    provider: Provider,
    charge: usize,
}

struct ProviderCacheEntry {
    cell: ProviderCell,
    charge: usize,
}

#[derive(Default)]
struct ProviderCacheState {
    entries: LinkedHashMap<ParquetProviderKey, ProviderCacheEntry>,
    resident_bytes: usize,
}

/// Resolves explicit providers and caches immutable Parquet planning state.
pub(super) struct IndexRelationProviderRegistry {
    storage: StorageRegistry,
    registered: RwLock<HashMap<String, Provider>>,
    parquet: Mutex<ProviderCacheState>,
    parquet_entry_capacity: usize,
    parquet_byte_capacity: usize,
    centroids: CentroidCache,
}

impl Default for IndexRelationProviderRegistry {
    fn default() -> Self {
        Self::new(StorageRegistry::default())
    }
}

impl IndexRelationProviderRegistry {
    pub(super) fn new(storage: StorageRegistry) -> Self {
        Self::with_capacities(
            storage,
            DEFAULT_PARQUET_PROVIDER_CACHE_ENTRIES,
            DEFAULT_PARQUET_PROVIDER_CACHE_BYTES,
            128,
            256 * 1024 * 1024,
        )
    }

    fn with_capacities(
        storage: StorageRegistry,
        parquet_entry_capacity: usize,
        parquet_byte_capacity: usize,
        centroid_entry_capacity: usize,
        centroid_byte_capacity: usize,
    ) -> Self {
        Self {
            storage,
            registered: RwLock::new(HashMap::new()),
            parquet: Mutex::new(ProviderCacheState::default()),
            parquet_entry_capacity,
            parquet_byte_capacity,
            centroids: CentroidCache::new(centroid_entry_capacity, centroid_byte_capacity),
        }
    }

    pub(super) fn registered(&self, relation_key: &str) -> Result<Option<Provider>> {
        Ok(self
            .registered
            .read()
            .map_err(|_| provider_lock_error())?
            .get(relation_key)
            .cloned())
    }

    pub(super) fn register(&self, relation_key: String, provider: Provider) -> Result<Provider> {
        let mut registered = self.registered.write().map_err(|_| provider_lock_error())?;
        Ok(Arc::clone(
            registered.entry(relation_key).or_insert(provider),
        ))
    }

    pub(super) async fn get_or_create_parquet(
        &self,
        relation_key: &str,
        layout: IndexRelationLayout,
        state: &dyn Session,
    ) -> Result<Provider> {
        if let Some(provider) = self.registered(relation_key)? {
            return Ok(provider);
        }

        let key = ParquetProviderKey {
            relation_key: relation_key.to_owned(),
            layout,
        };
        self.get_or_create(key, || async {
            match layout {
                IndexRelationLayout::Plain => {
                    let (listing, _) = uniform_dataset_listing_table(
                        &self.storage,
                        state,
                        relation_key,
                        Vec::new(),
                    )
                    .await?;
                    Ok(ProviderValue {
                        provider: listing as Provider,
                        charge: PLAIN_PROVIDER_CHARGE,
                    })
                }
                IndexRelationLayout::HiveCid => {
                    let provider =
                        HiveCidParquetProvider::load(&self.storage, relation_key, state).await?;
                    let charge = provider.resident_size();
                    Ok(ProviderValue {
                        provider: Arc::new(provider) as Provider,
                        charge,
                    })
                }
            }
        })
        .await
    }

    pub(super) async fn deferred_parquet_provider(
        self: &Arc<Self>,
        relation_key: &str,
        layout: IndexRelationLayout,
        state: &dyn Session,
    ) -> Result<Provider> {
        let schema = self
            .get_or_create_parquet(relation_key, layout, state)
            .await?
            .schema();
        Ok(Arc::new(DeferredParquetProvider {
            schema,
            relation_key: relation_key.to_owned(),
            layout,
            registry: Arc::clone(self),
        }))
    }

    pub(super) async fn get_or_load_centroids<F, Fut>(
        &self,
        relation_key: &str,
        load: F,
    ) -> Result<Arc<CentroidMatrix>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<CentroidMatrix>>,
    {
        self.centroids.get_or_load(relation_key, load).await
    }

    async fn get_or_create<F, Fut>(&self, key: ParquetProviderKey, create: F) -> Result<Provider>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<ProviderValue>>,
    {
        if self.parquet_entry_capacity == 0 || self.parquet_byte_capacity == 0 {
            return create().await.map(|value| value.provider);
        }
        let cell = {
            let mut state = self.provider_state();
            if let Some(entry) = state.entries.to_back(&key) {
                Arc::clone(&entry.cell)
            } else {
                let cell = Arc::new(OnceCell::new());
                state.entries.insert(
                    key.clone(),
                    ProviderCacheEntry {
                        cell: Arc::clone(&cell),
                        charge: 0,
                    },
                );
                cell
            }
        };

        let value = match cell.get_or_try_init(create).await {
            Ok(value) => value,
            Err(error) => {
                self.remove_provider_cell(&key, &cell);
                return Err(error);
            }
        };
        let mut state = self.provider_state();
        if value.charge > self.parquet_byte_capacity {
            remove_matching_provider_cell(&mut state, &key, &cell);
        } else if let Some(entry) = state.entries.get_mut(&key)
            && Arc::ptr_eq(&entry.cell, &cell)
            && entry.charge == 0
        {
            entry.charge = value.charge;
            state.resident_bytes = state.resident_bytes.saturating_add(value.charge);
            state.entries.to_back(&key);
            self.evict_providers(&mut state);
        }
        Ok(Arc::clone(&value.provider))
    }

    fn evict_providers(&self, state: &mut ProviderCacheState) {
        while state.entries.len() > self.parquet_entry_capacity
            || state.resident_bytes > self.parquet_byte_capacity
        {
            let pending = state.entries.len();
            let mut evicted = false;
            for _ in 0..pending {
                let Some((key, entry)) = state.entries.pop_front() else {
                    return;
                };
                if entry.charge == 0 {
                    state.entries.insert(key, entry);
                    continue;
                }
                state.resident_bytes = state.resident_bytes.saturating_sub(entry.charge);
                evicted = true;
                break;
            }
            if !evicted {
                break;
            }
        }
    }

    fn remove_provider_cell(&self, key: &ParquetProviderKey, cell: &ProviderCell) {
        remove_matching_provider_cell(&mut self.provider_state(), key, cell);
    }

    fn provider_state(&self) -> std::sync::MutexGuard<'_, ProviderCacheState> {
        self.parquet
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
    }

    #[cfg(test)]
    fn parquet_stats(&self) -> (usize, usize) {
        let state = self.provider_state();
        (state.entries.len(), state.resident_bytes)
    }
}

fn remove_matching_provider_cell(
    state: &mut ProviderCacheState,
    key: &ParquetProviderKey,
    cell: &ProviderCell,
) {
    if state
        .entries
        .get(key)
        .is_some_and(|entry| Arc::ptr_eq(&entry.cell, cell))
        && let Some(entry) = state.entries.remove(key)
    {
        state.resident_bytes = state.resident_bytes.saturating_sub(entry.charge);
    }
}

struct DeferredParquetProvider {
    schema: SchemaRef,
    relation_key: String,
    layout: IndexRelationLayout,
    registry: Arc<IndexRelationProviderRegistry>,
}

impl fmt::Debug for DeferredParquetProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeferredParquetProvider")
            .field("relation_key", &self.relation_key)
            .field("layout", &self.layout)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for DeferredParquetProvider {
    fn schema(&self) -> SchemaRef {
        Arc::clone(&self.schema)
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        limit: Option<usize>,
    ) -> datafusion::common::Result<Arc<dyn ExecutionPlan>> {
        self.registry
            .get_or_create_parquet(&self.relation_key, self.layout, state)
            .await
            .map_err(|error| DataFusionError::External(Box::new(error)))?
            .scan(state, projection, filters, limit)
            .await
    }

    async fn scan_with_args<'a>(
        &self,
        state: &dyn Session,
        args: ScanArgs<'a>,
    ) -> datafusion::common::Result<ScanResult> {
        self.registry
            .get_or_create_parquet(&self.relation_key, self.layout, state)
            .await
            .map_err(|error| DataFusionError::External(Box::new(error)))?
            .scan_with_args(state, args)
            .await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> datafusion::common::Result<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|filter| self.layout.filter_pushdown(filter))
            .collect())
    }
}

fn provider_lock_error() -> Error {
    Error::InvalidArgument("relation provider lock is poisoned".into())
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};

    use arrow::datatypes::Schema;
    use datafusion::datasource::MemTable;

    use super::*;

    fn empty_provider() -> Provider {
        Arc::new(MemTable::try_new(Arc::new(Schema::empty()), vec![Vec::new()]).unwrap())
    }

    fn value(charge: usize) -> ProviderValue {
        ProviderValue {
            provider: empty_provider(),
            charge,
        }
    }

    fn key(relation_key: &str) -> ParquetProviderKey {
        ParquetProviderKey {
            relation_key: relation_key.into(),
            layout: IndexRelationLayout::Plain,
        }
    }

    fn registry(entries: usize, bytes: usize) -> IndexRelationProviderRegistry {
        IndexRelationProviderRegistry::with_capacities(
            StorageRegistry::default(),
            entries,
            bytes,
            2,
            1024,
        )
    }

    #[tokio::test]
    async fn concurrent_misses_build_one_provider() {
        let registry = registry(2, 1024);
        let builds = Arc::new(AtomicUsize::new(0));
        let first_builds = Arc::clone(&builds);
        let second_builds = Arc::clone(&builds);

        let (first, second) = tokio::join!(
            registry.get_or_create(key("relation"), || async move {
                first_builds.fetch_add(1, Ordering::Relaxed);
                tokio::task::yield_now().await;
                Ok(value(64))
            }),
            registry.get_or_create(key("relation"), || async move {
                second_builds.fetch_add(1, Ordering::Relaxed);
                Ok(value(64))
            })
        );
        let first = first.unwrap();
        let second = second.unwrap();

        assert!(Arc::ptr_eq(&first, &second));
        assert_eq!(builds.load(Ordering::Relaxed), 1);
        assert_eq!(registry.parquet_stats(), (1, 64));
    }

    #[tokio::test]
    async fn provider_cache_is_bounded_by_entries_and_bytes() {
        let registry = registry(2, 100);
        for (relation_key, charge) in [("a", 40), ("b", 40), ("c", 70)] {
            registry
                .get_or_create(key(relation_key), || async move { Ok(value(charge)) })
                .await
                .unwrap();
        }

        assert_eq!(registry.parquet_stats(), (1, 70));
    }

    #[tokio::test]
    async fn oversized_provider_does_not_evict_resident_provider() {
        let registry = registry(2, 64);
        registry
            .get_or_create(key("resident"), || async { Ok(value(32)) })
            .await
            .unwrap();
        registry
            .get_or_create(key("oversized"), || async { Ok(value(65)) })
            .await
            .unwrap();

        assert_eq!(registry.parquet_stats(), (1, 32));
    }

    #[tokio::test]
    async fn failed_provider_creation_is_removed_and_retried() {
        let registry = registry(2, 64);
        let attempts = AtomicUsize::new(0);
        let error = registry
            .get_or_create(key("relation"), || async {
                attempts.fetch_add(1, Ordering::Relaxed);
                Err(Error::InvalidArgument("failed".into()))
            })
            .await
            .unwrap_err();
        assert!(matches!(error, Error::InvalidArgument(_)));
        assert_eq!(registry.parquet_stats(), (0, 0));

        registry
            .get_or_create(key("relation"), || async {
                attempts.fetch_add(1, Ordering::Relaxed);
                Ok(value(32))
            })
            .await
            .unwrap();
        assert_eq!(attempts.load(Ordering::Relaxed), 2);
        assert_eq!(registry.parquet_stats(), (1, 32));
    }

    #[tokio::test]
    async fn deferred_provider_does_not_pin_an_evicted_manifest() {
        let registry = Arc::new(registry(1, 1024));
        let provider = registry
            .get_or_create(key("a"), || async { Ok(value(32)) })
            .await
            .unwrap();
        let provider_weak = Arc::downgrade(&provider);
        let deferred = DeferredParquetProvider {
            schema: provider.schema(),
            relation_key: "a".into(),
            layout: IndexRelationLayout::Plain,
            registry: Arc::clone(&registry),
        };
        drop(provider);

        registry
            .get_or_create(key("b"), || async { Ok(value(32)) })
            .await
            .unwrap();

        assert!(provider_weak.upgrade().is_none());
        assert_eq!(deferred.relation_key, "a");
    }
}
