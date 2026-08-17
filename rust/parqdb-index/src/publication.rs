use std::collections::BTreeMap;
use std::time::{SystemTime, UNIX_EPOCH};

use parqdb_catalog::{IndexCatalog, IndexIdentifier};
use parqdb_core::{IndexArtifacts, PublishedIndex};
use parqdb_meta::{IndexMetadata, IndexSnapshot, RelationReference, SnapshotLogEntry};
use uuid::Uuid;

use crate::{Error, MetadataStore, Result};

/// Inputs for publishing the first snapshot of an index.
pub struct InitialIndex<'a> {
    /// Catalog identifier to register.
    pub identifier: IndexIdentifier,
    /// Stable UUID for the index lifetime.
    pub index_uuid: Uuid,
    /// Unique ID of the first immutable snapshot.
    pub snapshot_id: i64,
    /// Portable reference to the indexed source table.
    pub source: RelationReference,
    /// Vector field in the source table.
    pub vector_field: &'a str,
    /// Source fields that identify one row.
    pub source_key_fields: &'a [String],
    /// Backend identity recorded in the snapshot summary.
    pub builder: &'a str,
    /// Immutable relations and parameters produced by construction.
    pub build: IndexArtifacts,
}

/// Inputs for atomically publishing a refreshed index snapshot.
pub struct RefreshedIndex<'a> {
    /// Catalog identifier whose current metadata is replaced.
    pub identifier: IndexIdentifier,
    /// Catalog location observed before construction.
    pub base_metadata_location: &'a str,
    /// Metadata document observed before construction.
    pub base_metadata: &'a IndexMetadata,
    /// Unique ID of the new immutable snapshot.
    pub snapshot_id: i64,
    /// Portable reference to the indexed source table.
    pub source: RelationReference,
    /// Backend identity recorded in the snapshot summary.
    pub builder: &'a str,
    /// Immutable relations and parameters produced by construction.
    pub build: IndexArtifacts,
}

/// Writes and atomically registers the first metadata document for an index.
pub async fn publish_initial(
    catalog: &dyn IndexCatalog,
    metadata_store: &MetadataStore,
    request: InitialIndex<'_>,
) -> Result<PublishedIndex> {
    let timestamp_ms = now_ms()?;
    let IndexArtifacts {
        format,
        parameters,
        index_relations,
    } = request.build;
    let snapshot = IndexSnapshot {
        snapshot_id: request.snapshot_id,
        sequence_number: 1,
        timestamp_ms,
        summary: summary("create", request.builder),
        source: request.source,
        vector_field: request.vector_field.to_owned(),
        source_key_fields: request.source_key_fields.to_vec(),
        index_family: format.family,
        index_schema_version: format.schema_version,
        metric: format.metric,
        parameters,
        index_relations,
    };
    let metadata = IndexMetadata {
        format_version: 1,
        index_uuid: request.index_uuid,
        location: metadata_store.index_location(request.index_uuid)?,
        last_updated_ms: timestamp_ms,
        last_sequence_number: 1,
        current_snapshot_id: request.snapshot_id,
        snapshots: vec![snapshot],
        snapshot_log: vec![SnapshotLogEntry {
            timestamp_ms,
            snapshot_id: request.snapshot_id,
        }],
        properties: BTreeMap::new(),
    };
    let metadata_location = metadata_store.write_initial(&metadata).await?;
    catalog.register(&request.identifier, &metadata_location, &metadata)?;
    Ok(PublishedIndex {
        identifier: request.identifier,
        metadata_location,
        metadata,
    })
}

/// Writes and atomically commits a refreshed metadata document.
pub async fn publish_refresh(
    catalog: &dyn IndexCatalog,
    metadata_store: &MetadataStore,
    request: RefreshedIndex<'_>,
) -> Result<PublishedIndex> {
    let base_snapshot = request.base_metadata.current_snapshot()?;
    let IndexArtifacts {
        format,
        parameters,
        index_relations,
    } = request.build;
    let timestamp_ms = now_ms()?.max(request.base_metadata.last_updated_ms);
    let sequence_number = request
        .base_metadata
        .last_sequence_number
        .checked_add(1)
        .ok_or_else(|| Error::InvalidMetadata("last sequence number is exhausted".into()))?;
    let snapshot = IndexSnapshot {
        snapshot_id: request.snapshot_id,
        sequence_number,
        timestamp_ms,
        summary: summary("refresh", request.builder),
        source: request.source,
        vector_field: base_snapshot.vector_field.clone(),
        source_key_fields: base_snapshot.source_key_fields.clone(),
        index_family: format.family,
        index_schema_version: format.schema_version,
        metric: format.metric,
        parameters,
        index_relations,
    };
    let mut metadata = request.base_metadata.clone();
    metadata.last_updated_ms = timestamp_ms;
    metadata.last_sequence_number = sequence_number;
    metadata.current_snapshot_id = request.snapshot_id;
    metadata.snapshots.push(snapshot);
    metadata.snapshot_log.push(SnapshotLogEntry {
        timestamp_ms,
        snapshot_id: request.snapshot_id,
    });
    let metadata_location = metadata_store
        .write_update(request.base_metadata, &metadata)
        .await?;
    catalog.commit(
        &request.identifier,
        request.base_metadata_location,
        &metadata_location,
        request.base_metadata,
        &metadata,
    )?;
    Ok(PublishedIndex {
        identifier: request.identifier,
        metadata_location,
        metadata,
    })
}

/// Allocates a positive, process-independent snapshot identifier.
#[must_use]
pub fn new_snapshot_id() -> i64 {
    let bytes = Uuid::new_v4().into_bytes();
    let mut first = [0_u8; 8];
    first.copy_from_slice(&bytes[..8]);
    (i64::from_be_bytes(first) & i64::MAX).max(1)
}

fn now_ms() -> Result<i64> {
    let millis = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_err(|error| Error::InvalidTimestamp(error.to_string()))?
        .as_millis();
    i64::try_from(millis)
        .map_err(|_| Error::InvalidTimestamp("current timestamp is out of range".into()))
}

fn summary(operation: &str, builder: &str) -> BTreeMap<String, String> {
    BTreeMap::from([
        ("operation".into(), operation.into()),
        ("builder".into(), builder.into()),
    ])
}
