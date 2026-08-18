//! `DataFusion` catalog-list integration.

use std::sync::Arc;

use datafusion::catalog::{CatalogProvider, CatalogProviderList};
use parqdb_catalog::{
    CatalogEntry, CatalogTombstone, IndexCatalog, IndexIdentifier, IvfCentroidsCatalogEntry,
    IvfCentroidsClaim, IvfCentroidsClaimResult, TableCatalog, TableDefinition, TableIdentifier,
};
use parqdb_meta::{IndexMetadata, IvfCentroidsDescriptor, IvfCentroidsMetadata, RelationReference};
use uuid::Uuid;

pub(super) struct ParqDBCatalogList {
    catalogs: Arc<dyn CatalogProviderList>,
    indexes: Arc<dyn IndexCatalog>,
    tables: Option<Arc<dyn TableCatalog>>,
}

impl std::fmt::Debug for ParqDBCatalogList {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("ParqDBCatalogList")
            .field("catalog_names", &self.catalogs.catalog_names())
            .field("persistent_tables", &self.tables.is_some())
            .finish_non_exhaustive()
    }
}

impl ParqDBCatalogList {
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

    fn tables(&self) -> parqdb_catalog::Result<&dyn TableCatalog> {
        self.tables
            .as_deref()
            .ok_or(parqdb_catalog::Error::UnsupportedOperation(
                "persistent tables",
            ))
    }
}

impl CatalogProviderList for ParqDBCatalogList {
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

impl IndexCatalog for ParqDBCatalogList {
    fn load(&self, identifier: &IndexIdentifier) -> parqdb_catalog::Result<CatalogEntry> {
        self.indexes.load(identifier)
    }

    fn register(
        &self,
        identifier: &IndexIdentifier,
        source: &RelationReference,
        metadata_location: &str,
        metadata: &IndexMetadata,
    ) -> parqdb_catalog::Result<()> {
        self.indexes
            .register(identifier, source, metadata_location, metadata)
    }

    fn commit(
        &self,
        identifier: &IndexIdentifier,
        source: &RelationReference,
        base_metadata_location: &str,
        new_metadata_location: &str,
        base_metadata: &IndexMetadata,
        new_metadata: &IndexMetadata,
    ) -> parqdb_catalog::Result<()> {
        self.indexes.commit(
            identifier,
            source,
            base_metadata_location,
            new_metadata_location,
            base_metadata,
            new_metadata,
        )
    }

    fn drop(&self, identifier: &IndexIdentifier) -> parqdb_catalog::Result<()> {
        IndexCatalog::drop(self.indexes.as_ref(), identifier)
    }

    fn list(&self, namespace: &[String]) -> parqdb_catalog::Result<Vec<IndexIdentifier>> {
        self.indexes.list(namespace)
    }

    fn list_all(&self) -> parqdb_catalog::Result<Vec<IndexIdentifier>> {
        self.indexes.list_all()
    }

    fn find_by_source(
        &self,
        namespace: &[String],
        source: &RelationReference,
    ) -> parqdb_catalog::Result<Vec<CatalogEntry>> {
        self.indexes.find_by_source(namespace, source)
    }

    fn list_tombstones(&self) -> parqdb_catalog::Result<Vec<CatalogTombstone>> {
        self.indexes.list_tombstones()
    }

    fn purge_tombstone(&self, tombstone: &CatalogTombstone) -> parqdb_catalog::Result<bool> {
        self.indexes.purge_tombstone(tombstone)
    }

    fn load_ivf_centroids(
        &self,
        source: &RelationReference,
        fingerprint: &str,
    ) -> parqdb_catalog::Result<IvfCentroidsCatalogEntry> {
        self.indexes.load_ivf_centroids(source, fingerprint)
    }

    fn claim_ivf_centroids(
        &self,
        source: &RelationReference,
        descriptor: &IvfCentroidsDescriptor,
        owner: Uuid,
        lease_duration_ms: i64,
    ) -> parqdb_catalog::Result<IvfCentroidsClaimResult> {
        self.indexes
            .claim_ivf_centroids(source, descriptor, owner, lease_duration_ms)
    }

    fn renew_ivf_centroids_claim(
        &self,
        claim: &IvfCentroidsClaim,
        lease_duration_ms: i64,
    ) -> parqdb_catalog::Result<()> {
        self.indexes
            .renew_ivf_centroids_claim(claim, lease_duration_ms)
    }

    fn publish_ivf_centroids(
        &self,
        claim: &IvfCentroidsClaim,
        metadata_location: &str,
        metadata: &IvfCentroidsMetadata,
    ) -> parqdb_catalog::Result<IvfCentroidsCatalogEntry> {
        self.indexes
            .publish_ivf_centroids(claim, metadata_location, metadata)
    }

    fn abandon_ivf_centroids(
        &self,
        claim: &IvfCentroidsClaim,
        error: &str,
    ) -> parqdb_catalog::Result<()> {
        self.indexes.abandon_ivf_centroids(claim, error)
    }

    fn list_ivf_centroids(&self) -> parqdb_catalog::Result<Vec<IvfCentroidsCatalogEntry>> {
        self.indexes.list_ivf_centroids()
    }

    fn purge_ivf_centroids(
        &self,
        entry: &IvfCentroidsCatalogEntry,
    ) -> parqdb_catalog::Result<bool> {
        self.indexes.purge_ivf_centroids(entry)
    }
}

impl TableCatalog for ParqDBCatalogList {
    fn create_table(&self, definition: &TableDefinition) -> parqdb_catalog::Result<()> {
        self.tables()?.create_table(definition)
    }

    fn load_table(&self, identifier: &TableIdentifier) -> parqdb_catalog::Result<TableDefinition> {
        self.tables()?.load_table(identifier)
    }

    fn list_tables(
        &self,
        catalog: &str,
        namespace: &[String],
    ) -> parqdb_catalog::Result<Vec<TableIdentifier>> {
        self.tables()?.list_tables(catalog, namespace)
    }

    fn drop_table(&self, identifier: &TableIdentifier) -> parqdb_catalog::Result<()> {
        self.tables()?.drop_table(identifier)
    }
}
