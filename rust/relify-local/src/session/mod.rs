//! Embedded Relify session composition.

mod build;
mod catalog;
mod catalog_list;
mod query;
mod source;

pub use build::LocalBuildOptions;
pub use source::PersistentParquetOptions;

use std::collections::{BTreeMap, HashMap};
use std::ffi::OsString;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};

use datafusion::prelude::SessionContext;
use datafusion_datasource_parquet::{ParquetPageCacheFactory, ParquetPageCacheFactoryConfig};
#[cfg(test)]
use relify_catalog::{CatalogEntry, Error as CatalogError, IndexIdentifier};
use relify_catalog::{IndexCatalog, SqliteCatalog, TableCatalog};
use relify_index::{IndexRepository, MetadataCacheConfig, MetadataStore};
#[cfg(test)]
use relify_meta::{IndexSnapshot, RelationReference};
use relify_storage::{StorageRegistry, Warehouse};

#[cfg(test)]
use self::catalog::validate_index_name;
use self::catalog_list::RelifyCatalogList;
#[cfg(test)]
use crate::SearchRequest;
use crate::config::{LocalSessionOptions, metadata_cache_config};
use crate::coordination::SessionCoordination;
use crate::durability::{create_dir_all, sync_directory};
use crate::local_uri::directory_to_file_uri;
#[cfg(test)]
use crate::local_uri::file_uri_to_path;
use crate::parquet::{
    DecompressedParquetPageCache, ParquetPageCacheStats, ParquetStore,
    RelifyParquetPageCacheFactory, automatic_page_cache_capacity,
};
use crate::{Error, Result};

/// One top-level source-table field returned by [`LocalSession::describe`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceField {
    /// Field name.
    pub name: String,
    /// Arrow display form of the field type.
    pub data_type: String,
    /// Whether the field is nullable.
    pub nullable: bool,
}

/// Canonical source URI and schema reported by a session.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct SourceDescription {
    /// Canonical absolute source-table URI.
    pub uri: String,
    /// Top-level source fields in schema order.
    pub fields: Vec<SourceField>,
}

/// Published index information for one exact source-table state.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexInfo {
    /// Root-namespace index name.
    pub name: String,
    /// Source column containing vectors.
    pub column: String,
    /// Index-family identifier.
    pub family: String,
    /// Distance metric identifier.
    pub metric: String,
    /// Family-defined canonical parameters.
    pub parameters: BTreeMap<String, String>,
    /// Current Relify index snapshot ID.
    pub current_snapshot_id: i64,
}

/// Single-node Parquet session with an independent catalog and warehouse.
#[derive(Clone)]
pub struct LocalSession {
    state_root: PathBuf,
    catalog: Arc<dyn IndexCatalog>,
    table_catalog: Arc<dyn TableCatalog>,
    coordination: SessionCoordination,
    warehouse: Warehouse,
    indexes: IndexRepository,
    parquet: ParquetStore,
    parquet_page_cache: Arc<DecompressedParquetPageCache>,
    context: SessionContext,
    source_bindings: Arc<RwLock<HashMap<String, source::SourceBinding>>>,
    relation_providers: Arc<RwLock<HashMap<String, Arc<dyn datafusion::catalog::TableProvider>>>>,
    sql_relations: Arc<RwLock<HashMap<String, String>>>,
}

impl LocalSession {
    /// Opens the local shortcut: `SQLite` catalog and `file` warehouse under
    /// `root`.
    pub fn open(root: impl AsRef<Path>) -> Result<Self> {
        Self::open_with_options(root, LocalSessionOptions::default())
    }

    /// Opens the local shortcut with explicit `DataFusion` initialization
    /// options.
    pub fn open_with_options(root: impl AsRef<Path>, options: LocalSessionOptions) -> Result<Self> {
        let root = prepare_root(root.as_ref())?;
        let warehouse = directory_to_file_uri(&root)?;
        let catalog = Arc::new(SqliteCatalog::open(root.join("catalog.sqlite"))?);
        Self::from_parts(
            root.clone(),
            &root,
            Arc::clone(&catalog) as Arc<dyn IndexCatalog>,
            Some(catalog as Arc<dyn TableCatalog>),
            &warehouse,
            HashMap::new(),
            options,
        )
    }

    /// Opens a `SQLite` catalog under `state_root` with a separately configured
    /// `file`, `s3`, or `hdfs` warehouse.
    pub fn open_with_warehouse(
        state_root: impl AsRef<Path>,
        warehouse: &str,
        storage_options: HashMap<String, String>,
    ) -> Result<Self> {
        Self::open_with_warehouse_options(
            state_root,
            warehouse,
            storage_options,
            LocalSessionOptions::default(),
        )
    }

    /// Opens a separately configured warehouse with explicit `DataFusion`
    /// initialization options.
    pub fn open_with_warehouse_options(
        state_root: impl AsRef<Path>,
        warehouse: &str,
        storage_options: HashMap<String, String>,
        options: LocalSessionOptions,
    ) -> Result<Self> {
        let state_root = prepare_root(state_root.as_ref())?;
        let catalog = Arc::new(SqliteCatalog::open(state_root.join("catalog.sqlite"))?);
        Self::from_parts(
            state_root.clone(),
            &state_root,
            Arc::clone(&catalog) as Arc<dyn IndexCatalog>,
            Some(catalog as Arc<dyn TableCatalog>),
            warehouse,
            storage_options,
            options,
        )
    }

    /// Opens an explicitly located `SQLite` catalog with an independent
    /// warehouse.
    pub fn open_sqlite(
        database: impl AsRef<Path>,
        warehouse: &str,
        storage_options: HashMap<String, String>,
    ) -> Result<Self> {
        Self::open_sqlite_with_options(
            database,
            warehouse,
            storage_options,
            LocalSessionOptions::default(),
        )
    }

    /// Opens an explicit `SQLite` catalog with `DataFusion` initialization
    /// options.
    pub fn open_sqlite_with_options(
        database: impl AsRef<Path>,
        warehouse: &str,
        storage_options: HashMap<String, String>,
        options: LocalSessionOptions,
    ) -> Result<Self> {
        let database = absolute_path(database.as_ref())?;
        let file_name = database
            .file_name()
            .ok_or_else(|| Error::InvalidArgument("SQLite catalog path has no file name".into()))?
            .to_owned();
        let state_root = database
            .parent()
            .ok_or_else(|| Error::InvalidArgument("SQLite catalog path has no parent".into()))?;
        let state_root = prepare_root(state_root)?;
        let database = state_root.join(file_name);
        let coordination_root = prepare_root(&sqlite_coordination_root(&database)?)?;
        let catalog = Arc::new(SqliteCatalog::open(&database)?);
        Self::from_parts(
            state_root,
            &coordination_root,
            Arc::clone(&catalog) as Arc<dyn IndexCatalog>,
            Some(catalog as Arc<dyn TableCatalog>),
            warehouse,
            storage_options,
            options,
        )
    }

    /// Opens a session using a caller-supplied catalog and local warehouse.
    pub fn with_catalog(root: impl AsRef<Path>, catalog: Arc<dyn IndexCatalog>) -> Result<Self> {
        Self::with_catalog_and_options(root, catalog, LocalSessionOptions::default())
    }

    /// Opens a caller-supplied catalog with `DataFusion` initialization options.
    pub fn with_catalog_and_options(
        root: impl AsRef<Path>,
        catalog: Arc<dyn IndexCatalog>,
        options: LocalSessionOptions,
    ) -> Result<Self> {
        let root = prepare_root(root.as_ref())?;
        let warehouse = directory_to_file_uri(&root)?;
        Self::from_parts(
            root.clone(),
            &root,
            catalog,
            None,
            &warehouse,
            HashMap::new(),
            options,
        )
    }

    /// Opens a session with independent catalog and warehouse implementations.
    pub fn with_catalog_and_warehouse(
        state_root: impl AsRef<Path>,
        catalog: Arc<dyn IndexCatalog>,
        warehouse: &str,
        storage_options: HashMap<String, String>,
    ) -> Result<Self> {
        Self::with_catalog_and_warehouse_options(
            state_root,
            catalog,
            warehouse,
            storage_options,
            LocalSessionOptions::default(),
        )
    }

    /// Opens independent catalog and warehouse implementations with explicit
    /// `DataFusion` initialization options.
    pub fn with_catalog_and_warehouse_options(
        state_root: impl AsRef<Path>,
        catalog: Arc<dyn IndexCatalog>,
        warehouse: &str,
        storage_options: HashMap<String, String>,
        options: LocalSessionOptions,
    ) -> Result<Self> {
        let state_root = prepare_root(state_root.as_ref())?;
        Self::from_parts(
            state_root.clone(),
            &state_root,
            catalog,
            None,
            warehouse,
            storage_options,
            options,
        )
    }

    fn from_parts(
        state_root: PathBuf,
        coordination_root: &Path,
        catalog: Arc<dyn IndexCatalog>,
        table_catalog: Option<Arc<dyn TableCatalog>>,
        warehouse_root: &str,
        storage_options: HashMap<String, String>,
        options: LocalSessionOptions,
    ) -> Result<Self> {
        sync_directory(&state_root)?;
        let registry = StorageRegistry::new(storage_options);
        let warehouse = Warehouse::open(warehouse_root, registry.clone())?;
        let (session_config, runtime) = options.into_parts();
        let runtime = Arc::new(runtime.build()?);
        let automatic_capacity = automatic_page_cache_capacity(runtime.memory_pool.as_ref());
        let initial_capacity = crate::config::parquet_page_cache_capacity(&session_config)
            .unwrap_or(automatic_capacity);
        let parquet_page_cache = Arc::new(DecompressedParquetPageCache::new(initial_capacity));
        let page_cache_factory: Arc<dyn ParquetPageCacheFactory> = Arc::new(
            RelifyParquetPageCacheFactory::new(Arc::clone(&parquet_page_cache), automatic_capacity),
        );
        let session_config = session_config.with_extension(Arc::new(
            ParquetPageCacheFactoryConfig::new(page_cache_factory),
        ));
        let context = crate::query::relify_session_context(session_config, runtime);
        let catalog_list = Arc::new(RelifyCatalogList::new(
            context.state().catalog_list().clone(),
            catalog,
            table_catalog,
        ));
        context.register_catalog_list(Arc::clone(&catalog_list) as _);
        let index_catalog = Arc::clone(&catalog_list) as Arc<dyn IndexCatalog>;
        let config_context = context.clone();
        let metadata =
            MetadataStore::open_with_cache_config_resolver(warehouse.clone(), move || {
                metadata_cache_config(&config_context.copied_config())
            });
        Ok(Self {
            catalog: Arc::clone(&index_catalog),
            table_catalog: catalog_list as Arc<dyn TableCatalog>,
            coordination: SessionCoordination::open(coordination_root)?,
            indexes: IndexRepository::new(index_catalog, metadata),
            parquet: ParquetStore::with_context(registry, context.clone()),
            parquet_page_cache,
            context,
            source_bindings: Arc::new(RwLock::new(HashMap::new())),
            relation_providers: Arc::new(RwLock::new(HashMap::new())),
            sql_relations: Arc::new(RwLock::new(HashMap::new())),
            state_root,
            warehouse,
        })
    }

    /// Returns the local directory containing catalog state.
    #[must_use]
    pub fn state_root(&self) -> &Path {
        &self.state_root
    }

    /// Returns the canonical managed warehouse URI.
    #[must_use]
    pub fn warehouse_root(&self) -> &str {
        self.warehouse.root()
    }

    /// Returns a backend-neutral handle to this session's index repository.
    #[must_use]
    pub fn index_repository(&self) -> IndexRepository {
        self.indexes.clone()
    }

    /// Returns the active metadata-cache bounds.
    #[must_use]
    pub fn metadata_cache_config(&self) -> MetadataCacheConfig {
        self.indexes.metadata_store().cache_config()
    }

    /// Returns allocation and lookup counters for the Parquet Page cache.
    #[must_use]
    pub fn parquet_page_cache_stats(&self) -> ParquetPageCacheStats {
        self.parquet_page_cache.stats()
    }

    /// Removes all resident Parquet Pages from future cache lookups.
    ///
    /// Pages referenced by active queries remain alive until those queries
    /// release their Arrow buffers.
    pub fn clear_parquet_page_cache(&self) {
        self.parquet_page_cache.clear();
    }

    /// Returns the session's shared `DataFusion` execution context.
    #[must_use]
    pub fn context(&self) -> SessionContext {
        self.context.clone()
    }
}

fn prepare_root(root: &Path) -> Result<PathBuf> {
    create_dir_all(root)?;
    Ok(root.canonicalize()?)
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() {
        Ok(path.to_owned())
    } else {
        Ok(std::env::current_dir()?.join(path))
    }
}

fn sqlite_coordination_root(database: &Path) -> Result<PathBuf> {
    let file_name = database
        .file_name()
        .ok_or_else(|| Error::InvalidArgument("SQLite catalog path has no file name".into()))?;
    let mut coordination_name = OsString::from(".");
    coordination_name.push(file_name);
    coordination_name.push(".relify");
    Ok(database.with_file_name(coordination_name))
}

#[cfg(test)]
mod tests;
