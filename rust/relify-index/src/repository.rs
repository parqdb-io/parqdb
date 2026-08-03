use std::sync::Arc;

use relify_catalog::{CatalogEntry, Error as CatalogError, IndexCatalog, IndexIdentifier};
use relify_meta::{IndexMetadata, RelationReference};

use crate::{Error, MetadataStore, Result};

/// One catalog entry and its validated immutable metadata document.
#[derive(Debug, Clone)]
pub struct LoadedIndex {
    /// Catalog pointer used to load this state.
    pub entry: CatalogEntry,
    /// Validated metadata stored at the catalog pointer.
    pub metadata: IndexMetadata,
}

/// Backend-neutral access to one Relify index catalog and metadata store.
#[derive(Clone)]
pub struct IndexRepository {
    catalog: Arc<dyn IndexCatalog>,
    metadata: MetadataStore,
}

impl IndexRepository {
    /// Creates a repository from independent catalog and metadata authorities.
    #[must_use]
    pub const fn new(catalog: Arc<dyn IndexCatalog>, metadata: MetadataStore) -> Self {
        Self { catalog, metadata }
    }

    /// Returns the catalog used for identifier mappings and publication.
    #[must_use]
    pub fn catalog(&self) -> &dyn IndexCatalog {
        self.catalog.as_ref()
    }

    /// Returns the immutable metadata store.
    #[must_use]
    pub const fn metadata_store(&self) -> &MetadataStore {
        &self.metadata
    }

    /// Returns whether an identifier is currently published.
    pub fn exists(&self, identifier: &IndexIdentifier) -> Result<bool> {
        match self.catalog.load(identifier) {
            Ok(_) => Ok(true),
            Err(CatalogError::IndexNotFound(_)) => Ok(false),
            Err(error) => Err(error.into()),
        }
    }

    /// Lists identifiers registered directly in a namespace.
    pub fn list(&self, namespace: &[String]) -> Result<Vec<IndexIdentifier>> {
        Ok(self.catalog.list(namespace)?)
    }

    /// Loads the current catalog entry and validates its metadata document.
    pub async fn load(&self, identifier: &IndexIdentifier) -> Result<LoadedIndex> {
        let entry = self.catalog.load(identifier)?;
        let metadata = self.metadata.load(&entry.metadata_location).await?;
        Ok(LoadedIndex { entry, metadata })
    }

    /// Loads all indexes bound to one exact source-table state.
    pub async fn find_by_source(
        &self,
        namespace: &[String],
        source: &RelationReference,
    ) -> Result<Vec<LoadedIndex>> {
        let entries = self.catalog.find_by_source(namespace, source)?;
        let mut loaded = Vec::with_capacity(entries.len());
        for entry in entries {
            let metadata = self.metadata.load(&entry.metadata_location).await?;
            loaded.push(LoadedIndex { entry, metadata });
        }
        Ok(loaded)
    }

    /// Selects an explicit index or the sole index matching source and column.
    pub async fn select(
        &self,
        namespace: &[String],
        source: &RelationReference,
        identifier: Option<&IndexIdentifier>,
        vector_field: Option<&str>,
    ) -> Result<LoadedIndex> {
        if let Some(identifier) = identifier {
            let loaded = self.load(identifier).await?;
            let snapshot = loaded.metadata.current_snapshot()?;
            if snapshot.source.exact_state_key() != source.exact_state_key()
                || vector_field.is_some_and(|field| snapshot.vector_field != field)
            {
                return Err(Error::IndexNotFound(identifier.to_string()));
            }
            return Ok(loaded);
        }

        let mut matches = Vec::new();
        for loaded in self.find_by_source(namespace, source).await? {
            if vector_field.is_none_or(|field| {
                loaded
                    .metadata
                    .current_snapshot()
                    .is_ok_and(|snapshot| snapshot.vector_field == field)
            }) {
                matches.push(loaded);
            }
        }
        match matches.len() {
            0 => Err(Error::IndexNotFound(source.exact_state_key())),
            1 => Ok(matches.remove(0)),
            _ => Err(Error::AmbiguousIndex(
                matches
                    .iter()
                    .map(|loaded| loaded.entry.identifier.clone())
                    .collect(),
            )),
        }
    }

    /// Loads and registers an existing metadata file.
    pub async fn register(
        &self,
        identifier: &IndexIdentifier,
        metadata_location: &str,
    ) -> Result<()> {
        let metadata = self.metadata.load(metadata_location).await?;
        self.catalog
            .register(identifier, metadata_location, &metadata)?;
        Ok(())
    }

    /// Removes an index mapping without deleting metadata or index tables.
    pub fn drop(&self, identifier: &IndexIdentifier) -> Result<()> {
        Ok(IndexCatalog::drop(self.catalog.as_ref(), identifier)?)
    }
}
