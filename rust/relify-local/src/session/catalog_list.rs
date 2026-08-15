//! `DataFusion` catalog-list integration.

use std::sync::Arc;

use datafusion::catalog::{CatalogProvider, CatalogProviderList};
use relify_catalog::{
    CatalogEntry, CatalogTombstone, IndexCatalog, IndexIdentifier, IvfCentroidsCatalogEntry,
    IvfCentroidsClaim, IvfCentroidsClaimResult, TableCatalog, TableDefinition, TableIdentifier,
};
use relify_meta::{IndexMetadata, IvfCentroidsDescriptor, IvfCentroidsMetadata, RelationReference};
use uuid::Uuid;

pub(super) struct RelifyCatalogList {
    catalogs: Arc<dyn CatalogProviderList>,
    indexes: Arc<dyn IndexCatalog>,
    tables: Option<Arc<dyn TableCatalog>>,
}

impl std::fmt::Debug for RelifyCatalogList {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("RelifyCatalogList")
            .field("catalog_names", &self.catalogs.catalog_names())
            .field("persistent_tables", &self.tables.is_some())
            .finish_non_exhaustive()
    }
}

impl RelifyCatalogList {
    pub(super) fn new(
        catalogs: Arc<dyn CatalogProviderList>,
        indexes: Arc<dyn IndexCatalog>,
        tables: Option<Arc<dyn TableCatalog>>,
    ) -> Self {
        Self {
            catalogs,
            indexes,
            tables,
        }
    }

    fn tables(&self) -> relify_catalog::Result<&dyn TableCatalog> {
        self.tables
            .as_deref()
            .ok_or(relify_catalog::Error::UnsupportedOperation(
                "persistent tables",
            ))
    }
}

impl CatalogProviderList for RelifyCatalogList {
    fn register_catalog(
        &self,
        name: String,
        catalog: Arc<dyn CatalogProvider>,
    ) -> Option<Arc<dyn CatalogProvider>> {
        self.catalogs.register_catalog(name, catalog)
    }

    fn catalog_names(&self) -> Vec<String> {
        self.catalogs.catalog_names()
    }

    fn catalog(&self, name: &str) -> Option<Arc<dyn CatalogProvider>> {
        self.catalogs.catalog(name)
    }
}

impl IndexCatalog for RelifyCatalogList {
    fn load(&self, identifier: &IndexIdentifier) -> relify_catalog::Result<CatalogEntry> {
        self.indexes.load(identifier)
    }

    fn register(
        &self,
        identifier: &IndexIdentifier,
        metadata_location: &str,
        metadata: &IndexMetadata,
    ) -> relify_catalog::Result<()> {
        self.indexes
            .register(identifier, metadata_location, metadata)
    }

    fn commit(
        &self,
        identifier: &IndexIdentifier,
        base_metadata_location: &str,
        new_metadata_location: &str,
        base_metadata: &IndexMetadata,
        new_metadata: &IndexMetadata,
    ) -> relify_catalog::Result<()> {
        self.indexes.commit(
            identifier,
            base_metadata_location,
            new_metadata_location,
            base_metadata,
            new_metadata,
        )
    }

    fn drop(&self, identifier: &IndexIdentifier) -> relify_catalog::Result<()> {
        IndexCatalog::drop(self.indexes.as_ref(), identifier)
    }

    fn list(&self, namespace: &[String]) -> relify_catalog::Result<Vec<IndexIdentifier>> {
        self.indexes.list(namespace)
    }

    fn list_all(&self) -> relify_catalog::Result<Vec<IndexIdentifier>> {
        self.indexes.list_all()
    }

    fn find_by_source(
        &self,
        namespace: &[String],
        source: &RelationReference,
    ) -> relify_catalog::Result<Vec<CatalogEntry>> {
        self.indexes.find_by_source(namespace, source)
    }

    fn list_tombstones(&self) -> relify_catalog::Result<Vec<CatalogTombstone>> {
        self.indexes.list_tombstones()
    }

    fn purge_tombstone(&self, tombstone: &CatalogTombstone) -> relify_catalog::Result<bool> {
        self.indexes.purge_tombstone(tombstone)
    }

    fn load_ivf_centroids(
        &self,
        fingerprint: &str,
    ) -> relify_catalog::Result<IvfCentroidsCatalogEntry> {
        self.indexes.load_ivf_centroids(fingerprint)
    }

    fn claim_ivf_centroids(
        &self,
        descriptor: &IvfCentroidsDescriptor,
        owner: Uuid,
        lease_duration_ms: i64,
    ) -> relify_catalog::Result<IvfCentroidsClaimResult> {
        self.indexes
            .claim_ivf_centroids(descriptor, owner, lease_duration_ms)
    }

    fn renew_ivf_centroids_claim(
        &self,
        claim: &IvfCentroidsClaim,
        lease_duration_ms: i64,
    ) -> relify_catalog::Result<()> {
        self.indexes
            .renew_ivf_centroids_claim(claim, lease_duration_ms)
    }

    fn publish_ivf_centroids(
        &self,
        claim: &IvfCentroidsClaim,
        metadata_location: &str,
        metadata: &IvfCentroidsMetadata,
    ) -> relify_catalog::Result<IvfCentroidsCatalogEntry> {
        self.indexes
            .publish_ivf_centroids(claim, metadata_location, metadata)
    }

    fn abandon_ivf_centroids(
        &self,
        claim: &IvfCentroidsClaim,
        error: &str,
    ) -> relify_catalog::Result<()> {
        self.indexes.abandon_ivf_centroids(claim, error)
    }

    fn list_ivf_centroids(&self) -> relify_catalog::Result<Vec<IvfCentroidsCatalogEntry>> {
        self.indexes.list_ivf_centroids()
    }
}

impl TableCatalog for RelifyCatalogList {
    fn create_table(&self, definition: &TableDefinition) -> relify_catalog::Result<()> {
        self.tables()?.create_table(definition)
    }

    fn load_table(&self, identifier: &TableIdentifier) -> relify_catalog::Result<TableDefinition> {
        self.tables()?.load_table(identifier)
    }

    fn list_tables(
        &self,
        catalog: &str,
        namespace: &[String],
    ) -> relify_catalog::Result<Vec<TableIdentifier>> {
        self.tables()?.list_tables(catalog, namespace)
    }

    fn drop_table(&self, identifier: &TableIdentifier) -> relify_catalog::Result<()> {
        self.tables()?.drop_table(identifier)
    }
}
