//! Session-scoped planning state for immutable index relations.

mod bounded_cache;
mod centroid;
mod manifested_cid;

use std::collections::HashMap;
use std::fmt;
use std::future::Future;
use std::sync::{Arc, RwLock};

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{ScanArgs, ScanResult, Session, TableProvider};
use datafusion::common::DataFusionError;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use parqdb_storage::StorageRegistry;

use bounded_cache::BoundedAsyncCache;
use centroid::CentroidCache;
pub(super) use centroid::CentroidNavigator;
use manifested_cid::ManifestedCidParquetProvider;

use crate::config::{IndexIoMode, IndexRelationCacheConfig};
use crate::parquet::uniform_dataset_listing_table;
use crate::{Error, Result};

const PLAIN_PROVIDER_CHARGE: usize = 4 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum IndexRelationLayout {
    Plain,
    ManifestedCid,
}

impl IndexRelationLayout {
    fn filter_pushdown(self, filter: &Expr) -> TableProviderFilterPushDown {
        match self {
            Self::Plain => TableProviderFilterPushDown::Inexact,
            Self::ManifestedCid if manifested_cid::cid_filter_values(filter).is_some() => {
                TableProviderFilterPushDown::Inexact
            }
            Self::ManifestedCid => TableProviderFilterPushDown::Unsupported,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ParquetProviderKey {
    relation_key: String,
    layout: IndexRelationLayout,
}

type Provider = Arc<dyn TableProvider>;

/// Resolves explicit providers and caches immutable Parquet planning state.
pub(super) struct IndexRelationProviderRegistry {
    storage: StorageRegistry,
    registered: RwLock<HashMap<String, Provider>>,
    parquet: BoundedAsyncCache<ParquetProviderKey, Provider>,
    manifested: BoundedAsyncCache<String, Arc<ManifestedCidParquetProvider>>,
    centroids: CentroidCache,
    index_io: IndexIoMode,
}

impl Default for IndexRelationProviderRegistry {
    fn default() -> Self {
        Self::new(
            StorageRegistry::default(),
            IndexRelationCacheConfig::default(),
            IndexIoMode::Buffered,
        )
    }
}

impl IndexRelationProviderRegistry {
    pub(super) fn new(
        storage: StorageRegistry,
        config: IndexRelationCacheConfig,
        index_io: IndexIoMode,
    ) -> Self {
        Self {
            storage,
            registered: RwLock::new(HashMap::new()),
            parquet: BoundedAsyncCache::new(config.manifest_max_entries, config.manifest_max_bytes),
            manifested: BoundedAsyncCache::new(
                config.manifest_max_entries,
                config.manifest_max_bytes,
            ),
            centroids: CentroidCache::new(config.centroid_max_entries, config.centroid_max_bytes),
            index_io,
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
        if layout == IndexRelationLayout::ManifestedCid {
            return Ok(self.get_or_create_manifested(relation_key, state).await? as Provider);
        }

        let key = ParquetProviderKey {
            relation_key: relation_key.to_owned(),
            layout,
        };
        self.parquet
            .get_or_try_insert(key, || async {
                match layout {
                    IndexRelationLayout::Plain => {
                        let (listing, _) = uniform_dataset_listing_table(
                            &self.storage,
                            state,
                            relation_key,
                            Vec::new(),
                        )
                        .await?;
                        Ok((listing as Provider, PLAIN_PROVIDER_CHARGE))
                    }
                    IndexRelationLayout::ManifestedCid => unreachable!("handled above"),
                }
            })
            .await
    }

    async fn get_or_create_manifested(
        &self,
        relation_key: &str,
        state: &dyn Session,
    ) -> Result<Arc<ManifestedCidParquetProvider>> {
        self.manifested
            .get_or_try_insert(relation_key.to_owned(), || async {
                let provider = ManifestedCidParquetProvider::load(
                    &self.storage,
                    relation_key,
                    state,
                    self.index_io,
                )
                .await?;
                let charge = provider.resident_size();
                Ok((Arc::new(provider), charge))
            })
            .await
    }

    pub(super) async fn manifested_cid_provider(
        &self,
        relation_key: &str,
        cids: &[i32],
        state: &dyn Session,
    ) -> Result<Provider> {
        if self.registered(relation_key)?.is_some() {
            return Err(Error::InvalidArgument(
                "typed CID selection is unavailable for an overridden postings provider".into(),
            ));
        }
        let provider = self
            .get_or_create_manifested(relation_key, state)
            .await?
            .with_cid_selection(cids)?;
        Ok(Arc::new(provider))
    }

    pub(super) async fn validate_manifested_cid_identity(
        &self,
        relation_key: &str,
        nlist: usize,
        ntotal: usize,
        cid_offsets: &[usize],
        state: &dyn Session,
    ) -> Result<()> {
        self.get_or_create_manifested(relation_key, state)
            .await?
            .validate_identity(nlist, ntotal, cid_offsets)
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
    ) -> Result<Arc<CentroidNavigator>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<CentroidNavigator>>,
    {
        self.centroids.get_or_load(relation_key, load).await
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
    use arrow::datatypes::Schema;
    use datafusion::datasource::MemTable;

    use super::*;

    fn empty_provider() -> Provider {
        Arc::new(MemTable::try_new(Arc::new(Schema::empty()), vec![Vec::new()]).unwrap())
    }

    fn key(relation_key: &str) -> ParquetProviderKey {
        ParquetProviderKey {
            relation_key: relation_key.into(),
            layout: IndexRelationLayout::Plain,
        }
    }

    fn registry(entries: usize, bytes: usize) -> IndexRelationProviderRegistry {
        IndexRelationProviderRegistry::new(
            StorageRegistry::default(),
            IndexRelationCacheConfig {
                manifest_max_entries: entries,
                manifest_max_bytes: bytes,
                centroid_max_entries: 2,
                centroid_max_bytes: 1024,
            },
            IndexIoMode::Buffered,
        )
    }

    #[tokio::test]
    async fn deferred_provider_does_not_pin_an_evicted_manifest() {
        let registry = Arc::new(registry(1, 1024));
        let provider = registry
            .parquet
            .get_or_try_insert(key("a"), || async { Ok((empty_provider(), 32)) })
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
            .parquet
            .get_or_try_insert(key("b"), || async { Ok((empty_provider(), 32)) })
            .await
            .unwrap();

        assert!(provider_weak.upgrade().is_none());
        assert_eq!(deferred.relation_key, "a");
    }
}
