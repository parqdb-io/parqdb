//! Embedded catalog operations and index discovery.

use parqdb_catalog::{CatalogEntry, Error as CatalogError, IndexIdentifier};
use parqdb_index::{Error as IndexError, LoadedIndex};
use parqdb_meta::{IndexMetadata, RelationReference};

use super::source::canonical_source;
use super::{IndexInfo, LocalSession};
use crate::maintenance::{self, MaintenanceObject};
use crate::{Error, Result};

impl LocalSession {
    /// Returns whether a root-namespace index exists.
    pub fn index_exists(&self, name: &str) -> Result<bool> {
        self.index_exists_identifier(&local_index_identifier(name)?)
    }

    pub(super) fn index_exists_identifier(&self, identifier: &IndexIdentifier) -> Result<bool> {
        let _guard = self.coordination.read()?;
        self.indexes.exists(identifier).map_err(Into::into)
    }

    /// Lists root-namespace index names.
    pub fn list_indexes(&self) -> Result<Vec<String>> {
        let _guard = self.coordination.read()?;
        Ok(self
            .indexes
            .list(&[])?
            .into_iter()
            .map(|identifier| identifier.name().to_owned())
            .collect())
    }

    /// Loads validated index metadata as pretty-printed JSON.
    pub async fn load_index_json(&self, name: &str) -> Result<String> {
        let _guard = self.coordination.read()?;
        let loaded = self.load_entry(&local_index_identifier(name)?).await?;
        Ok(serde_json::to_string_pretty(&loaded.metadata)?)
    }

    /// Loads the current metadata location and validated metadata JSON.
    pub async fn load_index_entry(&self, name: &str) -> Result<(String, String)> {
        let _guard = self.coordination.read()?;
        let loaded = self.load_entry(&local_index_identifier(name)?).await?;
        Ok((
            loaded.entry.metadata_location,
            serde_json::to_string_pretty(&loaded.metadata)?,
        ))
    }

    /// Registers an existing metadata file in the root namespace.
    pub async fn register_index(&self, name: &str, metadata_location: &str) -> Result<()> {
        let _guard = self.coordination.write()?;
        self.indexes
            .register(&local_index_identifier(name)?, metadata_location)
            .await?;
        Ok(())
    }

    /// Removes a root-namespace index mapping without deleting index data.
    pub fn drop_index(&self, name: &str) -> Result<()> {
        let _guard = self.coordination.write()?;
        self.indexes.drop(&local_index_identifier(name)?)?;
        Ok(())
    }

    /// Finds or removes unreachable objects owned by this session.
    pub async fn remove_orphans(
        &self,
        older_than_ms: i64,
        dry_run: bool,
    ) -> Result<Vec<MaintenanceObject>> {
        let _guard = self.coordination.write()?;
        let active_roots = self.coordination.active_build_roots()?;
        maintenance::remove_orphans(
            &self.warehouse,
            self.indexes.metadata_store(),
            self.catalog.as_ref(),
            &active_roots,
            older_than_ms,
            dry_run,
        )
        .await
    }

    /// Lists published indexes for the exact state of a Parquet source.
    pub async fn list_source_indexes(&self, source: &str) -> Result<Vec<IndexInfo>> {
        let source_uri = canonical_source(&self.warehouse.registry(), source)?;
        let source = RelationReference::Parquet { uri: source_uri };
        self.list_relation_indexes(&source).await
    }

    /// Lists published indexes for one exact portable source reference.
    pub async fn list_relation_indexes(
        &self,
        source: &RelationReference,
    ) -> Result<Vec<IndexInfo>> {
        self.list_relation_indexes_in(&[], source).await
    }

    /// Lists published indexes for a source within one catalog namespace.
    pub async fn list_relation_indexes_in(
        &self,
        namespace: &[String],
        source: &RelationReference,
    ) -> Result<Vec<IndexInfo>> {
        source.validate()?;
        let _guard = self.coordination.read()?;
        let loaded = match self.indexes.find_by_source(namespace, source).await {
            Ok(indexes) => indexes,
            Err(IndexError::Catalog(CatalogError::NamespaceNotFound(_))) => Vec::new(),
            Err(error) => return Err(error.into()),
        };
        let mut indexes = Vec::with_capacity(loaded.len());
        for index in loaded {
            indexes.push(index_info(&index.entry, &index.metadata)?);
        }
        Ok(indexes)
    }

    /// Removes an index mapping after verifying its exact source-table state.
    pub async fn drop_source_index(&self, source: &str, name: &str) -> Result<()> {
        let source_uri = canonical_source(&self.warehouse.registry(), source)?;
        let source = RelationReference::Parquet { uri: source_uri };
        self.drop_relation_index(&source, name).await
    }

    /// Removes an index mapping after verifying its exact source relation.
    pub async fn drop_relation_index(&self, source: &RelationReference, name: &str) -> Result<()> {
        self.drop_relation_index_in(&[], source, name).await
    }

    /// Removes a source-bound index within one catalog namespace.
    pub async fn drop_relation_index_in(
        &self,
        namespace: &[String],
        source: &RelationReference,
        name: &str,
    ) -> Result<()> {
        source.validate()?;
        let identifier = namespaced_index_identifier(namespace, name)?;
        let loaded = self.load_entry(&identifier).await.map_err(|error| {
            if matches!(error, Error::Catalog(CatalogError::NamespaceNotFound(_))) {
                Error::IndexNotFound(name.to_owned())
            } else {
                error
            }
        })?;
        if loaded.metadata.current_snapshot()?.source.exact_state_key() != source.exact_state_key()
        {
            return Err(Error::IndexNotFound(name.to_owned()));
        }
        let _guard = self.coordination.write()?;
        self.indexes.drop(&identifier)?;
        Ok(())
    }

    /// Selects one index for an exact portable source relation.
    pub async fn select_index(
        &self,
        source: &RelationReference,
        index: Option<&str>,
        column: Option<&str>,
    ) -> Result<LoadedIndex> {
        self.select_index_in(&[], source, index, column).await
    }

    /// Selects one source-bound index within one catalog namespace.
    pub async fn select_index_in(
        &self,
        namespace: &[String],
        source: &RelationReference,
        index: Option<&str>,
        column: Option<&str>,
    ) -> Result<LoadedIndex> {
        let identifier = index
            .map(|name| namespaced_index_identifier(namespace, name))
            .transpose()?;
        self.indexes
            .select(namespace, source, identifier.as_ref(), column)
            .await
            .map_err(|error| match error {
                IndexError::Catalog(CatalogError::NamespaceNotFound(_)) => {
                    Error::IndexNotFound(index.or(column).unwrap_or("index for source").to_owned())
                }
                error => error.into(),
            })
    }

    /// Returns the selected metadata document for one exact source relation.
    pub async fn select_index_metadata(
        &self,
        source: &RelationReference,
        index: Option<&str>,
        column: Option<&str>,
    ) -> Result<String> {
        self.select_index_metadata_in(&[], source, index, column)
            .await
    }

    /// Returns selected index metadata within one catalog namespace.
    pub async fn select_index_metadata_in(
        &self,
        namespace: &[String],
        source: &RelationReference,
        index: Option<&str>,
        column: Option<&str>,
    ) -> Result<String> {
        source.validate()?;
        let loaded = self
            .select_index_in(namespace, source, index, column)
            .await?;
        Ok(serde_json::to_string(&loaded.metadata)?)
    }

    pub(super) async fn load_entry(&self, identifier: &IndexIdentifier) -> Result<LoadedIndex> {
        self.indexes.load(identifier).await.map_err(Into::into)
    }
}

pub(super) fn local_index_identifier(name: &str) -> Result<IndexIdentifier> {
    namespaced_index_identifier(&[], name)
}

pub(super) fn namespaced_index_identifier(
    namespace: &[String],
    name: &str,
) -> Result<IndexIdentifier> {
    validate_index_name(name)?;
    Ok(IndexIdentifier::new(namespace.to_vec(), name)?)
}

pub(super) fn validate_index_name(name: &str) -> Result<()> {
    let mut chars = name.chars();
    if !chars
        .next()
        .is_some_and(|character| character == '_' || character.is_ascii_alphabetic())
        || !chars.all(|character| character == '_' || character.is_ascii_alphanumeric())
    {
        return Err(Error::InvalidArgument(format!(
            "index name must match [A-Za-z_][A-Za-z0-9_]*: {name}"
        )));
    }
    Ok(())
}

fn index_info(entry: &CatalogEntry, metadata: &IndexMetadata) -> Result<IndexInfo> {
    let snapshot = metadata.current_snapshot()?;
    Ok(IndexInfo {
        name: entry.identifier.name().to_owned(),
        column: snapshot.vector_field.clone(),
        family: snapshot.index_family.clone(),
        metric: snapshot.metric.clone(),
        parameters: snapshot.parameters.clone(),
        current_snapshot_id: snapshot.snapshot_id,
    })
}
