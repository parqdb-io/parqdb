//! Session-scoped planning state for immutable index tables.

mod bounded_cache;
mod centroid;
mod manifested_cid;

use std::collections::{BTreeMap, BTreeSet, HashMap};
use std::fmt;
use std::future::Future;
use std::sync::{Arc, RwLock};

use arrow::datatypes::SchemaRef;
use async_trait::async_trait;
use datafusion::catalog::{ScanArgs, ScanResult, Session, TableProvider};
use datafusion::common::DataFusionError;
use datafusion::execution::SessionState;
use datafusion::logical_expr::{Expr, TableProviderFilterPushDown, TableType};
use datafusion::physical_plan::ExecutionPlan;
use parqdb_index::{
    CompactIndexRequest, CreateIndexRequest, IndexProvider, IndexProviderFactory, IndexSelection,
    IndexTableRole, IndexWriteInput, IndexWritePlan, UpdateIndexRequest,
};
use parqdb_meta::{
    IndexArtifactManifest, IndexProviderDefinition, IndexSnapshot, IndexTableDefinition,
    PostingEncoding,
};
use parqdb_storage::{StorageRegistry, Warehouse};
use uuid::Uuid;

use bounded_cache::BoundedAsyncCache;
use centroid::CentroidCache;
pub(super) use centroid::CentroidNavigator;
use manifested_cid::ManifestedCidParquetProvider;

use crate::config::{IndexIoMode, IndexTableCacheConfig};
use crate::parquet::uniform_dataset_listing_table;
use crate::{Error, Result};

const PLAIN_PROVIDER_CHARGE: usize = 4 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(super) enum IndexTableLayout {
    Plain,
    ManifestedCid,
}

impl IndexTableLayout {
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
    table_key: String,
    layout: IndexTableLayout,
}

type Provider = Arc<dyn TableProvider>;

#[derive(Clone, PartialEq, Eq)]
pub(super) struct IndexTableBinding {
    pub(super) provider: IndexProviderDefinition,
    pub(super) role: IndexTableRole,
    pub(super) table: IndexTableDefinition,
}

/// Resolves explicit providers and caches immutable Parquet planning state.
pub(super) struct IndexTableProviderRegistry {
    storage: StorageRegistry,
    registered: RwLock<HashMap<String, Provider>>,
    parquet: BoundedAsyncCache<ParquetProviderKey, Provider>,
    manifested: BoundedAsyncCache<String, Arc<ManifestedCidParquetProvider>>,
    centroids: CentroidCache,
    index_io: IndexIoMode,
}

impl Default for IndexTableProviderRegistry {
    fn default() -> Self {
        Self::new(
            StorageRegistry::default(),
            IndexTableCacheConfig::default(),
            IndexIoMode::Buffered,
        )
    }
}

impl IndexTableProviderRegistry {
    pub(super) fn new(
        storage: StorageRegistry,
        config: IndexTableCacheConfig,
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

    pub(super) fn registered(&self, table_key: &str) -> Result<Option<Provider>> {
        Ok(self
            .registered
            .read()
            .map_err(|_| provider_lock_error())?
            .get(table_key)
            .cloned())
    }

    pub(super) async fn get_or_create_parquet(
        &self,
        table_key: &str,
        layout: IndexTableLayout,
        state: &dyn Session,
    ) -> Result<Provider> {
        if let Some(provider) = self.registered(table_key)? {
            return Ok(provider);
        }
        if layout == IndexTableLayout::ManifestedCid {
            return Ok(self.get_or_create_manifested(table_key, state).await? as Provider);
        }

        let key = ParquetProviderKey {
            table_key: table_key.to_owned(),
            layout,
        };
        self.parquet
            .get_or_try_insert(key, || async {
                match layout {
                    IndexTableLayout::Plain => {
                        let (listing, _) = uniform_dataset_listing_table(
                            &self.storage,
                            state,
                            table_key,
                            Vec::new(),
                        )
                        .await?;
                        Ok((listing as Provider, PLAIN_PROVIDER_CHARGE))
                    }
                    IndexTableLayout::ManifestedCid => unreachable!("handled above"),
                }
            })
            .await
    }

    async fn get_or_create_manifested(
        &self,
        table_key: &str,
        state: &dyn Session,
    ) -> Result<Arc<ManifestedCidParquetProvider>> {
        self.manifested
            .get_or_try_insert(table_key.to_owned(), || async {
                let provider = ManifestedCidParquetProvider::load(
                    &self.storage,
                    table_key,
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
        table_key: &str,
        cids: &[i32],
        state: &dyn Session,
    ) -> Result<Provider> {
        if self.registered(table_key)?.is_some() {
            return Err(Error::InvalidArgument(
                "typed CID selection is unavailable for an overridden postings provider".into(),
            ));
        }
        let provider = self
            .get_or_create_manifested(table_key, state)
            .await?
            .with_cid_selection(cids)?;
        Ok(Arc::new(provider))
    }

    pub(super) async fn validate_manifested_cid_identity(
        &self,
        table_key: &str,
        nlist: usize,
        ntotal: usize,
        cid_offsets: &[usize],
        state: &dyn Session,
    ) -> Result<()> {
        self.get_or_create_manifested(table_key, state)
            .await?
            .validate_identity(nlist, ntotal, cid_offsets)
    }

    pub(super) async fn deferred_parquet_provider(
        self: &Arc<Self>,
        table_key: &str,
        layout: IndexTableLayout,
        state: &dyn Session,
    ) -> Result<Provider> {
        let schema = self
            .get_or_create_parquet(table_key, layout, state)
            .await?
            .schema();
        Ok(Arc::new(DeferredParquetProvider {
            schema,
            table_key: table_key.to_owned(),
            layout,
            registry: Arc::clone(self),
        }))
    }

    pub(super) async fn get_or_load_centroids<F, Fut>(
        &self,
        table_key: &str,
        load: F,
    ) -> Result<Arc<CentroidNavigator>>
    where
        F: FnOnce() -> Fut,
        Fut: Future<Output = Result<CentroidNavigator>>,
    {
        self.centroids.get_or_load(table_key, load).await
    }
}

struct DeferredParquetProvider {
    schema: SchemaRef,
    table_key: String,
    layout: IndexTableLayout,
    registry: Arc<IndexTableProviderRegistry>,
}

impl fmt::Debug for DeferredParquetProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("DeferredParquetProvider")
            .field("table_key", &self.table_key)
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
            .get_or_create_parquet(&self.table_key, self.layout, state)
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
            .get_or_create_parquet(&self.table_key, self.layout, state)
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
    Error::InvalidArgument("table provider lock is poisoned".into())
}

pub(super) struct ParquetIndexProviderFactory {
    registry: Arc<IndexTableProviderRegistry>,
    warehouse: Warehouse,
}

impl ParquetIndexProviderFactory {
    pub(super) fn new(registry: Arc<IndexTableProviderRegistry>, warehouse: Warehouse) -> Self {
        Self {
            registry,
            warehouse,
        }
    }
}

#[async_trait]
impl IndexProviderFactory for ParquetIndexProviderFactory {
    async fn open(
        &self,
        session: &dyn Session,
        definition: &IndexProviderDefinition,
    ) -> parqdb_index::Result<Arc<dyn IndexProvider>> {
        if definition.provider != "parquet" || !definition.properties.is_empty() {
            return Err(parqdb_index::Error::InvalidProvider(
                "Parquet index provider v1 requires provider=parquet and no provider properties"
                    .into(),
            ));
        }
        let state = session
            .as_any()
            .downcast_ref::<SessionState>()
            .cloned()
            .ok_or_else(|| {
                parqdb_index::Error::InvalidProvider(
                    "Parquet index provider requires a DataFusion SessionState".into(),
                )
            })?;
        Ok(Arc::new(ParquetIndexProvider {
            registry: Arc::clone(&self.registry),
            state,
            warehouse: self.warehouse.clone(),
        }))
    }
}

struct ParquetIndexProvider {
    registry: Arc<IndexTableProviderRegistry>,
    state: SessionState,
    warehouse: Warehouse,
}

#[async_trait]
impl IndexProvider for ParquetIndexProvider {
    async fn validate_snapshot(&self, snapshot: &IndexSnapshot) -> parqdb_index::Result<()> {
        if !matches!(
            PostingEncoding::from_snapshot(snapshot)?,
            PostingEncoding::Lvq4 | PostingEncoding::Lvq8
        ) {
            return Ok(());
        }
        let postings = snapshot.index_tables.get("ivf_postings").ok_or_else(|| {
            parqdb_index::Error::InvalidMetadata("missing table role: ivf_postings".into())
        })?;
        let centroids = snapshot.index_tables.get("ivf_centroids").ok_or_else(|| {
            parqdb_index::Error::InvalidMetadata("missing table role: ivf_centroids".into())
        })?;
        if postings != centroids {
            return Err(parqdb_index::Error::InvalidMetadata(
                "artifact centroids and postings must reference one manifest".into(),
            ));
        }
        let location = self.table_location(postings)?;
        let provider = self
            .registry
            .get_or_create_manifested(&location, &self.state)
            .await
            .map_err(|error| parqdb_index::Error::InvalidProvider(error.to_string()))?;
        let manifest = provider.artifact_manifest().ok_or_else(|| {
            parqdb_index::Error::InvalidProvider(
                "artifact-manifest layout did not resolve an artifact manifest".into(),
            )
        })?;
        validate_artifact_snapshot(snapshot, &manifest)
    }

    async fn open_index_table(
        &self,
        role: &IndexTableRole,
        table: &IndexTableDefinition,
        selection: &IndexSelection,
    ) -> parqdb_index::Result<Provider> {
        let mut location = self.table_location(table)?;
        if table.properties.get("layout").map(String::as_str) == Some("artifact-manifest") {
            let manifest_location = self.table_location(table)?;
            location = match role.as_str() {
                "ivf_centroids" => {
                    let provider = self
                        .registry
                        .get_or_create_manifested(&manifest_location, &self.state)
                        .await
                        .map_err(|error| parqdb_index::Error::InvalidProvider(error.to_string()))?;
                    let manifest = provider.artifact_manifest().ok_or_else(|| {
                        parqdb_index::Error::InvalidProvider(
                            "artifact-manifest layout did not resolve an artifact manifest".into(),
                        )
                    })?;
                    parqdb_index::resolve_artifact_object(
                        &manifest_location,
                        &manifest.hierarchy.centroids.path,
                    )?
                }
                "ivf_postings" => manifest_location,
                other => {
                    return Err(parqdb_index::Error::InvalidProvider(format!(
                        "artifact manifest does not contain index table role: {other}"
                    )));
                }
            };
        }
        let provider = if role.as_str() == "ivf_postings" {
            match selection.cids.as_deref() {
                Some(cids) => {
                    self.registry
                        .manifested_cid_provider(&location, cids, &self.state)
                        .await
                }
                None => {
                    self.registry
                        .get_or_create_parquet(
                            &location,
                            IndexTableLayout::ManifestedCid,
                            &self.state,
                        )
                        .await
                }
            }
        } else {
            self.registry
                .get_or_create_parquet(&location, IndexTableLayout::Plain, &self.state)
                .await
        };
        provider.map_err(|error| parqdb_index::Error::InvalidProvider(error.to_string()))
    }

    async fn plan_create(
        &self,
        _request: CreateIndexRequest,
        _input: IndexWriteInput,
    ) -> parqdb_index::Result<Box<dyn IndexWritePlan>> {
        Err(parqdb_index::Error::UnsupportedProviderOperation(
            "Parquet provider writes are currently driven by the local IVF builder".into(),
        ))
    }

    async fn plan_update(
        &self,
        _request: UpdateIndexRequest,
        _input: IndexWriteInput,
    ) -> parqdb_index::Result<Box<dyn IndexWritePlan>> {
        Err(parqdb_index::Error::UnsupportedProviderOperation(
            "incremental update".into(),
        ))
    }

    async fn plan_compact(
        &self,
        _request: CompactIndexRequest,
    ) -> parqdb_index::Result<Box<dyn IndexWritePlan>> {
        Err(parqdb_index::Error::UnsupportedProviderOperation(
            "compaction".into(),
        ))
    }

    async fn managed_table_locations(
        &self,
        tables: &BTreeMap<IndexTableRole, IndexTableDefinition>,
    ) -> parqdb_index::Result<Vec<String>> {
        let mut locations = BTreeSet::new();
        for table in tables.values() {
            locations.insert(self.table_location(table)?);
        }
        Ok(locations.into_iter().collect())
    }

    async fn delete_index_tables(
        &self,
        _tables: &BTreeMap<IndexTableRole, IndexTableDefinition>,
    ) -> parqdb_index::Result<()> {
        Err(parqdb_index::Error::UnsupportedProviderOperation(
            "provider-owned garbage collection".into(),
        ))
    }
}

impl ParquetIndexProvider {
    fn validate_table_definition(table: &IndexTableDefinition) -> parqdb_index::Result<()> {
        if table.definition_version != 1 {
            return Err(parqdb_index::Error::InvalidProvider(format!(
                "unsupported Parquet index table definition version: {}",
                table.definition_version
            )));
        }
        table.required_property("location")?;
        let expected_properties = match table.properties.get("layout").map(String::as_str) {
            None => 1,
            Some("artifact-manifest") => 2,
            Some(layout) => {
                return Err(parqdb_index::Error::InvalidProvider(format!(
                    "unsupported Parquet index table layout: {layout}"
                )));
            }
        };
        if table.properties.len() != expected_properties {
            return Err(parqdb_index::Error::InvalidProvider(format!(
                "unknown Parquet index table properties: {}",
                table
                    .properties
                    .keys()
                    .filter(|name| !matches!(name.as_str(), "layout" | "location"))
                    .cloned()
                    .collect::<Vec<_>>()
                    .join(", ")
            )));
        }
        Ok(())
    }

    fn table_location(&self, table: &IndexTableDefinition) -> parqdb_index::Result<String> {
        Self::validate_table_definition(table)?;
        let stored_location = table.required_property("location")?;
        if url::Url::parse(stored_location).is_ok() {
            Ok(stored_location.to_owned())
        } else {
            self.warehouse
                .location(stored_location, stored_location.ends_with('/'))
                .map_err(parqdb_index::Error::Storage)
        }
    }
}

fn validate_artifact_snapshot(
    snapshot: &IndexSnapshot,
    manifest: &IndexArtifactManifest,
) -> parqdb_index::Result<()> {
    let artifact_uuid = snapshot
        .parameters
        .get("artifact_uuid")
        .and_then(|value| Uuid::parse_str(value).ok());
    let source_keys = manifest
        .index
        .source_key_fields
        .iter()
        .map(|field| field.name.as_str())
        .collect::<Vec<_>>();
    if artifact_uuid != Some(manifest.artifact_uuid)
        || snapshot.vector_field != manifest.index.vector_field
        || snapshot
            .source_key_fields
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
            != source_keys
        || snapshot.metric != manifest.index.metric.as_str()
        || PostingEncoding::from_snapshot(snapshot)? != manifest.index.posting_encoding
        || snapshot.parameter_usize("dimension")?
            != usize::try_from(manifest.index.dimension).unwrap_or_default()
        || snapshot.parameter_usize("nlist")?
            != usize::try_from(manifest.index.nlist).unwrap_or_default()
        || snapshot.parameter_usize("ntotal")?
            != usize::try_from(manifest.index.ntotal).unwrap_or_default()
    {
        return Err(parqdb_index::Error::InvalidMetadata(
            "catalog snapshot does not match its artifact manifest".into(),
        ));
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use arrow::datatypes::Schema;
    use datafusion::datasource::MemTable;

    use super::*;

    fn empty_provider() -> Provider {
        Arc::new(MemTable::try_new(Arc::new(Schema::empty()), vec![Vec::new()]).unwrap())
    }

    fn key(table_key: &str) -> ParquetProviderKey {
        ParquetProviderKey {
            table_key: table_key.into(),
            layout: IndexTableLayout::Plain,
        }
    }

    fn registry(entries: usize, bytes: usize) -> IndexTableProviderRegistry {
        IndexTableProviderRegistry::new(
            StorageRegistry::default(),
            IndexTableCacheConfig {
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
            table_key: "a".into(),
            layout: IndexTableLayout::Plain,
            registry: Arc::clone(&registry),
        };
        drop(provider);

        registry
            .parquet
            .get_or_try_insert(key("b"), || async { Ok((empty_provider(), 32)) })
            .await
            .unwrap();

        assert!(provider_weak.upgrade().is_none());
        assert_eq!(deferred.table_key, "a");
    }
}
