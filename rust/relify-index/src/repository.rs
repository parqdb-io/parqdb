use std::sync::Arc;

use relify_catalog::{
    CatalogEntry, Error as CatalogError, IndexCatalog, IndexIdentifier, IvfCentroidsCatalogEntry,
};
use relify_meta::{
    DistanceMetric, IndexMetadata, IndexSnapshot, IvfCentroidsMetadata, IvfCentroidsReference,
    RelationReference, ivf_centroids_reference,
};

use crate::{Error, MetadataStore, Result};

/// One catalog entry and its validated immutable metadata document.
#[derive(Debug, Clone)]
pub struct LoadedIndex {
    /// Catalog pointer used to load this state.
    pub entry: CatalogEntry,
    /// Validated metadata stored at the catalog pointer.
    pub metadata: IndexMetadata,
}

/// One cataloged and validated immutable IVF centroid artifact.
#[derive(Debug, Clone)]
pub struct LoadedIvfCentroids {
    /// Catalog pointer used to load this state.
    pub entry: IvfCentroidsCatalogEntry,
    /// Validated metadata stored at the catalog pointer.
    pub metadata: IvfCentroidsMetadata,
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

    /// Loads one ready IVF centroid artifact by fingerprint.
    pub async fn load_ivf_centroids(&self, fingerprint: &str) -> Result<LoadedIvfCentroids> {
        let entry = self.catalog.load_ivf_centroids(fingerprint)?;
        self.load_ivf_centroids_entry(entry).await
    }

    /// Loads and validates the centroid artifact referenced by a logical index.
    pub async fn load_ivf_centroids_reference(
        &self,
        reference: &IvfCentroidsReference,
    ) -> Result<LoadedIvfCentroids> {
        reference.validate()?;
        let metadata = self
            .metadata
            .load_ivf_centroids(&reference.metadata_location)
            .await?;
        validate_ivf_centroids_identity(
            &reference.fingerprint,
            reference.artifact_uuid,
            &metadata,
        )?;
        Ok(LoadedIvfCentroids {
            entry: IvfCentroidsCatalogEntry {
                fingerprint: reference.fingerprint.clone(),
                artifact_uuid: reference.artifact_uuid,
                metadata_location: reference.metadata_location.clone(),
            },
            metadata,
        })
    }

    /// Loads and validates the centroid artifact required by one logical IVF snapshot.
    pub async fn load_snapshot_ivf_centroids(
        &self,
        snapshot: &IndexSnapshot,
    ) -> Result<LoadedIvfCentroids> {
        let reference = ivf_centroids_reference(snapshot)?;
        let loaded = self.load_ivf_centroids_reference(&reference).await?;
        let descriptor = &loaded.metadata.descriptor;
        let metric = DistanceMetric::from_metadata(&snapshot.metric).ok_or_else(|| {
            Error::InvalidMetadata(format!("unsupported IVF metric: {}", snapshot.metric))
        })?;
        let centroids = snapshot
            .index_relations
            .get("ivf_centroids")
            .ok_or_else(|| Error::InvalidMetadata("missing relation role: ivf_centroids".into()))?;
        if descriptor.source.exact_state_key() != snapshot.source.exact_state_key()
            || descriptor.vector_field != snapshot.vector_field
            || usize::try_from(descriptor.dimension).ok()
                != Some(snapshot.parameter_usize("dimension")?)
            || usize::try_from(descriptor.nlist).ok() != Some(snapshot.parameter_usize("nlist")?)
            || descriptor.metric != metric
            || loaded.metadata.centroids != *centroids
        {
            return Err(Error::InvalidMetadata(
                "logical index does not match its IVF centroid artifact".into(),
            ));
        }
        Ok(loaded)
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
        let metadata = self.metadata.load_from_storage(metadata_location).await?;
        self.catalog
            .register(identifier, metadata_location, &metadata)?;
        Ok(())
    }

    /// Removes an index mapping without deleting metadata or index tables.
    pub fn drop(&self, identifier: &IndexIdentifier) -> Result<()> {
        Ok(IndexCatalog::drop(self.catalog.as_ref(), identifier)?)
    }

    async fn load_ivf_centroids_entry(
        &self,
        entry: IvfCentroidsCatalogEntry,
    ) -> Result<LoadedIvfCentroids> {
        let metadata = self
            .metadata
            .load_ivf_centroids(&entry.metadata_location)
            .await?;
        validate_ivf_centroids_identity(&entry.fingerprint, entry.artifact_uuid, &metadata)?;
        Ok(LoadedIvfCentroids { entry, metadata })
    }
}

fn validate_ivf_centroids_identity(
    fingerprint: &str,
    artifact_uuid: uuid::Uuid,
    metadata: &IvfCentroidsMetadata,
) -> Result<()> {
    if metadata.fingerprint != fingerprint || metadata.artifact_uuid != artifact_uuid {
        return Err(Error::InvalidMetadata(
            "IVF centroid identity does not match its metadata".into(),
        ));
    }
    Ok(())
}
