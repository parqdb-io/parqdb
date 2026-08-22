//! Embedded catalog operations and index discovery.

use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use arrow::datatypes::{DataType, Schema};
use parqdb_catalog::{CatalogEntry, Error as CatalogError, IndexIdentifier};
use parqdb_index::{Error as IndexError, LoadedIndex, new_snapshot_id, resolve_artifact_object};
use parqdb_meta::{
    IndexArtifactManifest, IndexMetadata, IndexSnapshot, PostingEncoding, RelationReference,
    SnapshotLogEntry, ivf_centroids_reference,
};
use uuid::Uuid;

use super::index_relation::IndexRelationLayout;
use super::source::{canonical_source, validate_index_source_schema};
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

    /// Registers an existing immutable index artifact for one Parquet source.
    pub async fn register_source_index(
        &self,
        source: &str,
        name: &str,
        manifest_location: &str,
    ) -> Result<()> {
        let source_uri = canonical_source(&self.warehouse.registry(), source)?;
        self.register_relation_index(
            &RelationReference::Parquet { uri: source_uri },
            name,
            manifest_location,
        )
        .await
    }

    /// Registers an existing immutable index artifact for one exact source relation.
    pub async fn register_relation_index(
        &self,
        source: &RelationReference,
        name: &str,
        manifest_location: &str,
    ) -> Result<()> {
        self.register_relation_index_in(&[], source, name, manifest_location)
            .await
    }

    /// Registers an existing immutable index artifact in one catalog namespace.
    pub async fn register_relation_index_in(
        &self,
        namespace: &[String],
        source: &RelationReference,
        name: &str,
        manifest_location: &str,
    ) -> Result<()> {
        source.validate()?;
        let identifier = namespaced_index_identifier(namespace, name)?;
        let _guard = self.coordination.write()?;
        if self.indexes.exists(&identifier)? {
            return Err(CatalogError::AlreadyExists(identifier).into());
        }
        let binding = self.bind_relation(source).await?;
        let manifest = self
            .indexes
            .metadata_store()
            .load_artifact_manifest(manifest_location)
            .await?;
        let metadata = registered_artifact_metadata(&manifest, manifest_location)?;
        self.validate_registration(&binding, &metadata).await?;
        let metadata_location = self
            .indexes
            .metadata_store()
            .write_initial(&metadata)
            .await?;
        self.indexes
            .catalog()
            .register(&identifier, source, &metadata_location, &metadata)?;
        Ok(())
    }

    async fn validate_registration(
        &self,
        source: &super::source::SourceBinding,
        metadata: &IndexMetadata,
    ) -> Result<()> {
        for snapshot in &metadata.snapshots {
            validate_index_source_schema(source.schema.as_ref(), snapshot)?;
            for (role, location) in &snapshot.index_relations {
                if role == "artifact_manifest" && url::Url::parse(location).is_ok() {
                    parqdb_meta::validate_absolute_location(location)?;
                } else {
                    self.indexes
                        .metadata_store()
                        .resolve_location(location, location.ends_with('/'))?;
                }
            }
        }
        let snapshot = metadata.current_snapshot()?;
        let source_rows = self
            .context
            .read_table(source.provider.clone())?
            .count()
            .await?;
        if i64::try_from(source_rows).ok() != Some(snapshot.indexed_rows) {
            return Err(Error::InvalidSchema(format!(
                "source row count {source_rows} does not match indexed-rows {}",
                snapshot.indexed_rows
            )));
        }

        if let Some(location) = snapshot.index_relations.get("artifact_manifest") {
            return self
                .validate_artifact_registration(source, snapshot, location)
                .await;
        }

        self.validate_warehouse_registration(source, snapshot).await
    }

    async fn validate_artifact_registration(
        &self,
        source: &super::source::SourceBinding,
        snapshot: &IndexSnapshot,
        location: &str,
    ) -> Result<()> {
        let manifest_location = if url::Url::parse(location).is_ok() {
            location.to_owned()
        } else {
            self.indexes
                .metadata_store()
                .resolve_location(location, false)?
        };
        let manifest = self
            .indexes
            .metadata_store()
            .load_artifact_manifest(&manifest_location)
            .await?;
        let centroid_location =
            resolve_artifact_object(&manifest_location, &manifest.hierarchy.centroids.path)?;
        let centroids = self.parquet.read(&centroid_location, None).await?;
        if centroids.num_rows() != snapshot.parameter_usize("nlist")? {
            return Err(Error::InvalidSchema(
                "artifact centroid row count does not match nlist".into(),
            ));
        }
        let cid_offsets = manifest
            .hierarchy
            .cid_offsets
            .iter()
            .map(|value| usize::try_from(*value).unwrap_or_default())
            .collect::<Vec<_>>();
        self.index_relation_providers
            .validate_manifested_cid_identity(
                &manifest_location,
                snapshot.parameter_usize("nlist")?,
                snapshot.parameter_usize("ntotal")?,
                &cid_offsets,
                &self.context.state(),
            )
            .await?;
        let postings = self
            .index_relation_providers
            .get_or_create_parquet(
                &manifest_location,
                IndexRelationLayout::ManifestedCid,
                &self.context.state(),
            )
            .await?;
        validate_postings_schema(source.schema.as_ref(), postings.schema().as_ref(), snapshot)
    }

    async fn validate_warehouse_registration(
        &self,
        source: &super::source::SourceBinding,
        snapshot: &IndexSnapshot,
    ) -> Result<()> {
        let centroids = self
            .indexes
            .load_ivf_centroids_reference(&source.reference, &ivf_centroids_reference(snapshot)?)
            .await?;
        let centroids_location = self.indexes.metadata_store().resolve_location(
            &centroids.metadata.centroids,
            centroids.metadata.centroids.ends_with('/'),
        )?;
        let centroids_batch = self.parquet.read(&centroids_location, None).await?;
        crate::ivf::read_centroids(
            &centroids_batch,
            snapshot.parameter_usize("nlist")?,
            snapshot.parameter_usize("dimension")?,
        )?;
        let roots_location = self.indexes.metadata_store().resolve_location(
            &centroids.metadata.roots,
            centroids.metadata.roots.ends_with('/'),
        )?;
        let roots_batch = self.parquet.read(&roots_location, None).await?;
        let (_, cid_offsets) = crate::ivf::read_roots(
            &roots_batch,
            snapshot.parameter_usize("nlist")?,
            snapshot.parameter_usize("dimension")?,
        )?;
        crate::ivf::validate_centroid_buckets(&centroids_batch, &cid_offsets)?;

        let postings_location = snapshot
            .index_relations
            .get("ivf_postings")
            .ok_or_else(|| Error::InvalidMetadata("missing relation role: ivf_postings".into()))?;
        let postings_location = self
            .indexes
            .metadata_store()
            .resolve_location(postings_location, postings_location.ends_with('/'))?;
        self.index_relation_providers
            .validate_manifested_cid_identity(
                &postings_location,
                snapshot.parameter_usize("nlist")?,
                snapshot.parameter_usize("ntotal")?,
                &cid_offsets,
                &self.context.state(),
            )
            .await?;
        let postings = self
            .index_relation_providers
            .get_or_create_parquet(
                &postings_location,
                IndexRelationLayout::ManifestedCid,
                &self.context.state(),
            )
            .await?;
        validate_postings_schema(source.schema.as_ref(), postings.schema().as_ref(), snapshot)?;
        let posting_rows = self.context.read_table(postings)?.count().await?;
        if i64::try_from(posting_rows).ok() != Some(snapshot.indexed_rows) {
            return Err(Error::InvalidSchema(format!(
                "postings row count {posting_rows} does not match indexed-rows {}",
                snapshot.indexed_rows
            )));
        }
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
        if loaded.entry.source.exact_state_key() != source.exact_state_key() {
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

fn registered_artifact_metadata(
    manifest: &IndexArtifactManifest,
    manifest_location: &str,
) -> Result<IndexMetadata> {
    parqdb_meta::validate_absolute_location(manifest_location)?;
    let timestamp_ms = i64::try_from(
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map_err(|error| Error::InvalidArgument(error.to_string()))?
            .as_millis(),
    )
    .map_err(|_| Error::InvalidArgument("current timestamp is out of range".into()))?;
    let snapshot_id = new_snapshot_id();
    let source_key_fields = manifest
        .index
        .source_key_fields
        .iter()
        .map(|field| field.name.clone())
        .collect();
    let snapshot = IndexSnapshot {
        snapshot_id,
        sequence_number: 1,
        timestamp_ms,
        summary: BTreeMap::from([("operation".into(), "register".into())]),
        vector_field: manifest.index.vector_field.clone(),
        source_key_fields,
        indexed_rows: manifest.index.ntotal,
        index_family: "ivf".into(),
        index_schema_version: parqdb_meta::IVF_SCHEMA_VERSION,
        metric: manifest.index.metric.as_str().into(),
        parameters: BTreeMap::from([
            ("artifact_uuid".into(), manifest.artifact_uuid.to_string()),
            ("dimension".into(), manifest.index.dimension.to_string()),
            ("nlist".into(), manifest.index.nlist.to_string()),
            ("ntotal".into(), manifest.index.ntotal.to_string()),
            (
                "posting_encoding".into(),
                manifest.index.posting_encoding.as_str().into(),
            ),
        ]),
        index_relations: BTreeMap::from([(
            "artifact_manifest".into(),
            manifest_location.to_owned(),
        )]),
    };
    let metadata = IndexMetadata {
        format_version: 1,
        index_uuid: Uuid::new_v4(),
        last_updated_ms: timestamp_ms,
        last_sequence_number: 1,
        current_snapshot_id: snapshot_id,
        snapshots: vec![snapshot],
        snapshot_log: vec![SnapshotLogEntry {
            timestamp_ms,
            snapshot_id,
        }],
        properties: BTreeMap::new(),
    };
    metadata.validate()?;
    Ok(metadata)
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

fn validate_postings_schema(
    source: &Schema,
    postings: &Schema,
    snapshot: &IndexSnapshot,
) -> Result<()> {
    let encoding = PostingEncoding::from_snapshot(snapshot)?;
    let expected_fields =
        1 + snapshot.source_key_fields.len() + usize::from(encoding != PostingEncoding::Source) * 3;
    if postings.fields().len() != expected_fields {
        return Err(Error::InvalidSchema(
            "ivf_postings fields do not match index metadata".into(),
        ));
    }
    require_posting_field(postings, "cid", &DataType::Int32)?;
    for (position, source_key) in snapshot.source_key_fields.iter().enumerate() {
        let source_type = source
            .field_with_name(source_key)
            .map_err(|_| {
                Error::InvalidSchema(format!("source key column not found: {source_key}"))
            })?
            .data_type();
        let expected = match source_type {
            DataType::Utf8View | DataType::LargeUtf8 => DataType::Utf8,
            DataType::BinaryView | DataType::LargeBinary => DataType::Binary,
            other => other.clone(),
        };
        require_posting_field(postings, &format!("key_{}", position + 1), &expected)?;
    }
    if encoding != PostingEncoding::Source {
        require_posting_field(postings, "offset", &DataType::Float32)?;
        require_posting_field(postings, "scale", &DataType::Float32)?;
        require_posting_field(postings, "code", &DataType::BinaryView)?;
    }
    Ok(())
}

fn require_posting_field(schema: &Schema, name: &str, data_type: &DataType) -> Result<()> {
    let field = schema
        .field_with_name(name)
        .map_err(|_| Error::InvalidSchema(format!("ivf_postings is missing {name}")))?;
    if !equivalent_posting_type(field.data_type(), data_type) {
        return Err(Error::InvalidSchema(format!(
            "ivf_postings.{name} must have type {data_type}"
        )));
    }
    Ok(())
}

fn equivalent_posting_type(actual: &DataType, expected: &DataType) -> bool {
    actual == expected
        || matches!(
            (actual, expected),
            (
                DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8,
                DataType::Utf8 | DataType::Utf8View | DataType::LargeUtf8
            ) | (
                DataType::Binary | DataType::BinaryView | DataType::LargeBinary,
                DataType::Binary | DataType::BinaryView | DataType::LargeBinary
            )
        )
}
