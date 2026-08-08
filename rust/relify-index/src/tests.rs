use std::collections::BTreeMap;
use std::sync::Arc;

use relify_catalog::{IndexIdentifier, SqliteCatalog};
use relify_core::{IndexArtifacts, IndexFormat};
use relify_meta::{IndexMetadata, IndexSnapshot, RelationReference, SnapshotLogEntry};
use relify_storage::{StorageRegistry, Warehouse};
use tempfile::TempDir;
use uuid::Uuid;

use crate::{
    IndexRepository, InitialIndex, MetadataStore, RefreshedIndex, new_snapshot_id, publish_initial,
    publish_refresh,
};

fn repository(temporary: &TempDir) -> IndexRepository {
    let catalog = Arc::new(SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap());
    let root = url::Url::from_directory_path(temporary.path().join("warehouse"))
        .unwrap()
        .to_string();
    let metadata = MetadataStore::open(Warehouse::open(&root, StorageRegistry::default()).unwrap());
    IndexRepository::new(catalog, metadata)
}

fn source(uri: &str) -> RelationReference {
    RelationReference::Parquet {
        uri: uri.to_owned(),
    }
}

fn artifacts(uri: &str, nlist: usize) -> IndexArtifacts {
    IndexArtifacts {
        format: IndexFormat::ivf_v1(),
        parameters: BTreeMap::from([
            ("dimension".into(), "2".into()),
            ("nlist".into(), nlist.to_string()),
            ("ntotal".into(), "3".into()),
            ("store_vectors".into(), "true".into()),
        ]),
        index_relations: BTreeMap::from([
            ("ivf_centroids".into(), source(&format!("{uri}/centroids"))),
            ("ivf_postings".into(), source(&format!("{uri}/postings"))),
        ]),
    }
}

fn metadata_document(store: &MetadataStore) -> IndexMetadata {
    let index_uuid = Uuid::new_v4();
    let snapshot_id = 701;
    let timestamp_ms = 1_750_000_000_000;
    let build = artifacts("file:///indexes/metadata-test", 2);
    IndexMetadata {
        format_version: 1,
        index_uuid,
        location: store.index_location(index_uuid).unwrap(),
        last_updated_ms: timestamp_ms,
        last_sequence_number: 1,
        current_snapshot_id: snapshot_id,
        snapshots: vec![IndexSnapshot {
            snapshot_id,
            sequence_number: 1,
            timestamp_ms,
            summary: BTreeMap::new(),
            source: source("file:///data/documents.parquet"),
            vector_field: "embedding".into(),
            source_key_fields: vec!["document_id".into()],
            index_family: "ivf".into(),
            index_schema_version: 1,
            metric: "l2_squared".into(),
            parameters: build.parameters,
            index_relations: build.index_relations,
        }],
        snapshot_log: vec![SnapshotLogEntry {
            timestamp_ms,
            snapshot_id,
        }],
        properties: BTreeMap::new(),
    }
}

#[tokio::test]
async fn metadata_store_writes_immutable_validated_documents() {
    let temporary = TempDir::new().unwrap();
    let repository = repository(&temporary);
    let store = repository.metadata_store();
    let base = metadata_document(store);
    let base_location = store.write_initial(&base).await.unwrap();
    assert_eq!(store.load(&base_location).await.unwrap(), base);
    assert!(store.write_initial(&base).await.is_err());

    let mut next = base.clone();
    let mut snapshot = base.current_snapshot().unwrap().clone();
    snapshot.snapshot_id = 702;
    snapshot.sequence_number = 2;
    snapshot.timestamp_ms += 1;
    next.last_updated_ms += 1;
    next.last_sequence_number = 2;
    next.current_snapshot_id = snapshot.snapshot_id;
    next.snapshots.push(snapshot);
    next.snapshot_log.push(SnapshotLogEntry {
        timestamp_ms: next.last_updated_ms,
        snapshot_id: next.current_snapshot_id,
    });
    let next_location = store.write_update(&base, &next).await.unwrap();

    assert_ne!(next_location, base_location);
    assert!(next_location.ends_with("/v2-702.metadata.json"));
    assert_eq!(store.load(&next_location).await.unwrap(), next);
    assert!(store.write_update(&base, &next).await.is_err());
}

#[tokio::test]
async fn metadata_store_rejects_foreign_locations() {
    let temporary = TempDir::new().unwrap();
    let repository = repository(&temporary);
    let store = repository.metadata_store();
    let mut metadata = metadata_document(store);
    metadata.location = "file:///tmp/other/metadata/".into();

    assert!(store.write_initial(&metadata).await.is_err());
}

#[tokio::test]
async fn repository_loads_discovers_and_selects_published_indexes() {
    let temporary = TempDir::new().unwrap();
    let repository = repository(&temporary);
    let identifier = IndexIdentifier::root("documents_embedding").unwrap();
    let source = source("file:///data/documents.parquet");
    let snapshot_id = new_snapshot_id();
    let published = publish_initial(
        repository.catalog(),
        repository.metadata_store(),
        InitialIndex {
            identifier: identifier.clone(),
            index_uuid: Uuid::new_v4(),
            snapshot_id,
            source: source.clone(),
            vector_field: "embedding",
            source_key_fields: &["document_id".into()],
            builder: "test",
            build: artifacts("file:///indexes/v1", 2),
        },
    )
    .await
    .unwrap();

    assert!(repository.exists(&identifier).unwrap());
    let listed = repository.list(&[]).unwrap();
    assert_eq!(listed.len(), 1);
    assert_eq!(listed[0], identifier);
    assert_eq!(
        repository
            .load(&identifier)
            .await
            .unwrap()
            .entry
            .metadata_location,
        published.metadata_location
    );
    assert_eq!(
        repository
            .select(&[], &source, None, Some("embedding"))
            .await
            .unwrap()
            .metadata
            .current_snapshot()
            .unwrap()
            .snapshot_id,
        snapshot_id
    );
}

#[tokio::test]
async fn publication_uses_the_builder_format_descriptor() {
    let temporary = TempDir::new().unwrap();
    let repository = repository(&temporary);
    let mut build = artifacts("file:///indexes/v2", 2);
    build.format = IndexFormat::ivf_v2();
    build.parameters.remove("store_vectors");
    build
        .parameters
        .insert("posting_encoding".into(), "lvq8".into());

    let published = publish_initial(
        repository.catalog(),
        repository.metadata_store(),
        InitialIndex {
            identifier: IndexIdentifier::root("documents_embedding_v2").unwrap(),
            index_uuid: Uuid::new_v4(),
            snapshot_id: new_snapshot_id(),
            source: source("file:///data/documents.parquet"),
            vector_field: "embedding",
            source_key_fields: &["document_id".into()],
            builder: "test",
            build,
        },
    )
    .await
    .unwrap();

    let snapshot = published.metadata.current_snapshot().unwrap();
    assert_eq!(snapshot.index_family, "ivf");
    assert_eq!(snapshot.index_schema_version, 2);
    assert_eq!(snapshot.parameters["posting_encoding"], "lvq8");
}

#[tokio::test]
async fn repository_rejects_missing_and_ambiguous_selection() {
    let temporary = TempDir::new().unwrap();
    let repository = repository(&temporary);
    let source = source("file:///data/documents.parquet");
    assert!(repository.select(&[], &source, None, None).await.is_err());

    for name in ["one", "two"] {
        publish_initial(
            repository.catalog(),
            repository.metadata_store(),
            InitialIndex {
                identifier: IndexIdentifier::root(name).unwrap(),
                index_uuid: Uuid::new_v4(),
                snapshot_id: new_snapshot_id(),
                source: source.clone(),
                vector_field: "embedding",
                source_key_fields: &["document_id".into()],
                builder: "test",
                build: artifacts(&format!("file:///indexes/{name}"), 2),
            },
        )
        .await
        .unwrap();
    }
    assert!(matches!(
        repository.select(&[], &source, None, None).await,
        Err(crate::Error::AmbiguousIndex(indexes)) if indexes.len() == 2
    ));
}

#[tokio::test]
async fn repository_publication_refreshes_with_catalog_cas() {
    let temporary = TempDir::new().unwrap();
    let repository = repository(&temporary);
    let identifier = IndexIdentifier::root("documents_embedding").unwrap();
    let source = source("file:///data/documents.parquet");
    let initial = publish_initial(
        repository.catalog(),
        repository.metadata_store(),
        InitialIndex {
            identifier: identifier.clone(),
            index_uuid: Uuid::new_v4(),
            snapshot_id: new_snapshot_id(),
            source: source.clone(),
            vector_field: "embedding",
            source_key_fields: &["document_id".into()],
            builder: "test",
            build: artifacts("file:///indexes/v1", 2),
        },
    )
    .await
    .unwrap();
    let refreshed = publish_refresh(
        repository.catalog(),
        repository.metadata_store(),
        RefreshedIndex {
            identifier: identifier.clone(),
            base_metadata_location: &initial.metadata_location,
            base_metadata: &initial.metadata,
            snapshot_id: new_snapshot_id(),
            source,
            builder: "test",
            build: artifacts("file:///indexes/v2", 3),
        },
    )
    .await
    .unwrap();
    assert_eq!(
        repository
            .load(&identifier)
            .await
            .unwrap()
            .entry
            .metadata_location,
        refreshed.metadata_location
    );
}

#[tokio::test]
async fn duplicate_publication_does_not_replace_catalog_state() {
    let temporary = TempDir::new().unwrap();
    let repository = repository(&temporary);
    let identifier = IndexIdentifier::root("documents_embedding").unwrap();
    let first = publish_initial(
        repository.catalog(),
        repository.metadata_store(),
        InitialIndex {
            identifier: identifier.clone(),
            index_uuid: Uuid::new_v4(),
            snapshot_id: new_snapshot_id(),
            source: source("file:///data/documents.parquet"),
            vector_field: "embedding",
            source_key_fields: &["document_id".into()],
            builder: "local",
            build: artifacts("file:///indexes/first", 2),
        },
    )
    .await
    .unwrap();
    let duplicate = publish_initial(
        repository.catalog(),
        repository.metadata_store(),
        InitialIndex {
            identifier: identifier.clone(),
            index_uuid: Uuid::new_v4(),
            snapshot_id: new_snapshot_id(),
            source: source("file:///data/documents.parquet"),
            vector_field: "embedding",
            source_key_fields: &["document_id".into()],
            builder: "spark",
            build: artifacts("file:///indexes/duplicate", 2),
        },
    )
    .await;

    assert!(matches!(
        duplicate,
        Err(crate::Error::Catalog(relify_catalog::Error::AlreadyExists(
            _
        )))
    ));
    assert_eq!(
        repository
            .load(&identifier)
            .await
            .unwrap()
            .entry
            .metadata_location,
        first.metadata_location
    );
}
