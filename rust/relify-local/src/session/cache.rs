//! Session-scoped decoded index cache.

use std::collections::{BTreeMap, HashMap};
use std::fmt;
use std::sync::{Arc, RwLock, Weak};

use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{DataFusionError, Result as DataFusionResult, Statistics};
use datafusion::dataframe::DataFrame;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use relify_catalog::Error as CatalogError;

use super::catalog::local_index_identifier;
use super::{IndexCacheInfo, LocalSession};
use crate::ivf::{CachedIvfPostings, read_centroids};
use crate::{Error, Result};

pub(super) struct CachedIndex {
    metadata_location: String,
    info: IndexCacheInfo,
    relations: BTreeMap<String, CachedRelation>,
}

struct CachedRelation {
    key: String,
    provider: Arc<dyn TableProvider>,
    kind: CachedRelationKind,
}

enum CachedRelationKind {
    Generic,
    IvfCentroids(Arc<[f32]>),
}

struct CacheAwareRelation {
    key: String,
    fallback: Arc<dyn TableProvider>,
    cache: Weak<RwLock<HashMap<String, CachedIndex>>>,
}

impl LocalSession {
    /// Materializes every relation in the current index snapshot as Arrow memory.
    pub async fn cache_index(&self, name: &str) -> Result<IndexCacheInfo> {
        let _guard = self.coordination.read()?;
        let loaded = self.load_entry(&local_index_identifier(name)?).await?;
        if let Some(info) = self.cached_info(name, Some(&loaded.entry.metadata_location))? {
            return Ok(info);
        }

        let snapshot = loaded.metadata.current_snapshot()?;
        let dimension = snapshot.parameter_usize("dimension")?;
        let nlist = snapshot.parameter_usize("nlist")?;
        let mut relations = BTreeMap::new();
        let mut resident_bytes = 0usize;
        for (role, reference) in &snapshot.index_relations {
            let key = super::source::relation_key(reference);
            if role == "ivf_postings" {
                let dataframe = self.relation_dataframe(reference, role).await?;
                let batch_size = self.context.state().config().batch_size();
                let (provider, postings_bytes) =
                    CachedIvfPostings::load(dataframe, batch_size).await?;
                resident_bytes = resident_bytes.saturating_add(postings_bytes);
                relations.insert(
                    role.clone(),
                    CachedRelation {
                        key,
                        provider,
                        kind: CachedRelationKind::Generic,
                    },
                );
                continue;
            }
            let cached = self
                .relation_dataframe(reference, role)
                .await?
                .cache()
                .await?;
            let batches = cached.clone().collect().await?;
            resident_bytes = resident_bytes.saturating_add(
                batches
                    .iter()
                    .map(arrow::record_batch::RecordBatch::get_array_memory_size)
                    .sum::<usize>(),
            );
            let kind = if role == "ivf_centroids" {
                let values = {
                    let schema = Arc::clone(cached.schema().inner());
                    let batch = concat_batches(&schema, &batches)?;
                    Ok::<Arc<[f32]>, Error>(Arc::from(
                        read_centroids(&batch, nlist, dimension)?.into_boxed_slice(),
                    ))
                }?;
                resident_bytes =
                    resident_bytes.saturating_add(std::mem::size_of_val(values.as_ref()));
                CachedRelationKind::IvfCentroids(values)
            } else {
                CachedRelationKind::Generic
            };
            relations.insert(
                role.clone(),
                CachedRelation {
                    key,
                    provider: cached.into_view(),
                    kind,
                },
            );
        }

        let info = IndexCacheInfo {
            name: name.to_owned(),
            snapshot_id: snapshot.snapshot_id,
            relation_count: relations.len(),
            resident_bytes,
        };
        let cached = CachedIndex {
            metadata_location: loaded.entry.metadata_location,
            info: info.clone(),
            relations,
        };
        self.index_cache
            .write()
            .map_err(|_| cache_lock_error())?
            .insert(name.to_owned(), cached);
        Ok(info)
    }

    /// Returns whether the backend contains the current snapshot of an index.
    pub fn is_index_cached(&self, name: &str) -> Result<bool> {
        let _guard = self.coordination.read()?;
        let identifier = local_index_identifier(name)?;
        let entry = match self.catalog.load(&identifier) {
            Ok(entry) => entry,
            Err(CatalogError::IndexNotFound(_)) => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        Ok(self
            .cached_info(name, Some(&entry.metadata_location))?
            .is_some())
    }

    /// Removes one explicitly materialized index snapshot from this backend.
    pub fn uncache_index(&self, name: &str) -> Result<bool> {
        local_index_identifier(name)?;
        Ok(self
            .index_cache
            .write()
            .map_err(|_| cache_lock_error())?
            .remove(name)
            .is_some())
    }

    pub(super) fn invalidate_index_cache(&self, name: &str) -> Result<()> {
        self.index_cache
            .write()
            .map_err(|_| cache_lock_error())?
            .remove(name);
        Ok(())
    }

    pub(super) fn cached_relation_dataframe(&self, key: &str) -> Result<Option<DataFrame>> {
        let cache = self.index_cache.read().map_err(|_| cache_lock_error())?;
        let provider = cache
            .values()
            .flat_map(|index| index.relations.values())
            .find(|relation| relation.key == key)
            .map(|relation| Arc::clone(&relation.provider));
        provider
            .map(|provider| self.context.read_table(provider).map_err(Error::from))
            .transpose()
    }

    pub(super) fn is_relation_cached(&self, key: &str) -> Result<bool> {
        let cache = self.index_cache.read().map_err(|_| cache_lock_error())?;
        Ok(cache
            .values()
            .flat_map(|index| index.relations.values())
            .any(|relation| relation.key == key))
    }

    pub(super) fn cached_centroid_values(&self, key: &str) -> Result<Option<Arc<[f32]>>> {
        let cache = self.index_cache.read().map_err(|_| cache_lock_error())?;
        Ok(cache
            .values()
            .flat_map(|index| index.relations.values())
            .find(|relation| relation.key == key)
            .and_then(|relation| match &relation.kind {
                CachedRelationKind::IvfCentroids(values) => Some(Arc::clone(values)),
                CachedRelationKind::Generic => None,
            }))
    }

    pub(super) fn cache_aware_relation(
        &self,
        key: &str,
        fallback: Arc<dyn TableProvider>,
    ) -> Arc<dyn TableProvider> {
        Arc::new(CacheAwareRelation {
            key: key.to_owned(),
            fallback,
            cache: Arc::downgrade(&self.index_cache),
        })
    }

    fn cached_info(
        &self,
        name: &str,
        metadata_location: Option<&str>,
    ) -> Result<Option<IndexCacheInfo>> {
        let cache = self.index_cache.read().map_err(|_| cache_lock_error())?;
        Ok(cache.get(name).and_then(|cached| {
            metadata_location
                .is_none_or(|location| cached.metadata_location == location)
                .then(|| cached.info.clone())
        }))
    }
}

impl CacheAwareRelation {
    fn provider(&self) -> DataFusionResult<Arc<dyn TableProvider>> {
        let Some(cache) = self.cache.upgrade() else {
            return Ok(Arc::clone(&self.fallback));
        };
        let cache = cache.read().map_err(|_| {
            DataFusionError::Execution("Relify index cache lock is poisoned".into())
        })?;
        Ok(cache
            .values()
            .flat_map(|index| index.relations.values())
            .find(|relation| relation.key == self.key)
            .map_or_else(
                || Arc::clone(&self.fallback),
                |relation| Arc::clone(&relation.provider),
            ))
    }
}

impl fmt::Debug for CacheAwareRelation {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CacheAwareRelation")
            .field("key", &self.key)
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for CacheAwareRelation {
    fn schema(&self) -> SchemaRef {
        self.fallback.schema()
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
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        self.provider()?
            .scan(state, projection, filters, limit)
            .await
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        // The provider can change between planning and execution. Keep filters
        // above the scan so either the cached or Parquet path remains correct.
        Ok(self
            .fallback
            .supports_filters_pushdown(filters)?
            .into_iter()
            .map(|support| match support {
                TableProviderFilterPushDown::Unsupported => {
                    TableProviderFilterPushDown::Unsupported
                }
                TableProviderFilterPushDown::Exact | TableProviderFilterPushDown::Inexact => {
                    TableProviderFilterPushDown::Inexact
                }
            })
            .collect())
    }

    fn statistics(&self) -> Option<Statistics> {
        self.fallback.statistics()
    }
}

fn cache_lock_error() -> Error {
    Error::InvalidArgument("index cache lock is poisoned".into())
}
