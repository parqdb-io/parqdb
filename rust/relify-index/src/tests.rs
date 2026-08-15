use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use relify_catalog::{IndexIdentifier, SharedIvfClaimResult, SqliteCatalog};
use relify_core::{IndexArtifacts, IndexFormat};
use relify_meta::{
    DistanceMetric, IVF_CLUSTERING_PROFILE_VERSION, IndexMetadata, IndexSnapshot,
    RelationReference, SharedIvfDescriptor, SharedIvfMetadata, SharedIvfReference,
    SnapshotLogEntry,
};
use relify_storage::{StorageRegistry, Warehouse};
use tempfile::TempDir;
use uuid::Uuid;

use crate::{
    IndexRepository, InitialIndex, MetadataCacheConfig, MetadataStore, RefreshedIndex,
    new_snapshot_id, publish_initial, publish_refresh,
};

fn repository(temporary: &TempDir) -> IndexRepository {
    let catalog = Arc::new(SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap());
    let root = url::Url::from_directory_path(temporary.path().join("warehouse"))
        .unwrap()
        .to_string();
    let metadata = MetadataStore::open(Warehouse::open(&root, StorageRegistry::default()).unwrap());
    IndexRepository::new(catalog, metadata)
}

fn metadata_store(temporary: &TempDir, cache_capacity: usize) -> MetadataStore {
    metadata_store_with_config(
        temporary,
        MetadataCacheConfig::new(cache_capacity, usize::MAX),
    )
}

fn metadata_store_with_config(
    temporary: &TempDir,
    cache_config: MetadataCacheConfig,
) -> MetadataStore {
    let root = url::Url::from_directory_path(temporary.path().join("warehouse"))
        .unwrap()
        .to_string();
    MetadataStore::open_with_cache_config(
        Warehouse::open(&root, StorageRegistry::default()).unwrap(),
        cache_config,
    )
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

fn shared_ivf_document(store: &MetadataStore) -> SharedIvfMetadata {
    let descriptor = SharedIvfDescriptor {
        source: source("file:///data/documents.parquet"),
        vector_field: "embedding".into(),
        dimension: 2,
        metric: DistanceMetric::L2Squared,
        nlist: 2,
        clustering_profile_version: IVF_CLUSTERING_PROFILE_VERSION,
    };
    let artifact_uuid = Uuid::new_v4();
    SharedIvfMetadata {
        format_version: 1,
        artifact_uuid,
        fingerprint: descriptor.fingerprint().unwrap(),
        location: store.shared_ivf_location(artifact_uuid).unwrap(),
        created_at_ms: 1_750_000_000_000,
        descriptor,
        centroids: source("file:///indexes/shared/centroids"),
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
async fn metadata_store_bounds_index_and_shared_documents_in_one_cache() {
    let temporary = TempDir::new().unwrap();
    let store = metadata_store(&temporary, 1);
    let index = metadata_document(&store);
    let index_location = store.write_initial(&index).await.unwrap();
    let shared = shared_ivf_document(&store);
    let shared_location = store.write_shared_ivf(&shared).await.unwrap();

    for location in [&index_location, &shared_location] {
        std::fs::remove_file(url::Url::parse(location).unwrap().to_file_path().unwrap()).unwrap();
    }

    assert!(store.load(&index_location).await.is_err());
    assert_eq!(
        store.load_shared_ivf(&shared_location).await.unwrap(),
        shared
    );
}

#[tokio::test]
async fn repository_validates_shared_ivf_catalog_and_reference_identity() {
    let temporary = TempDir::new().unwrap();
    let repository = repository(&temporary);
    let shared = shared_ivf_document(repository.metadata_store());
    let location = repository
        .metadata_store()
        .write_shared_ivf(&shared)
        .await
        .unwrap();
    let claim = match repository
        .catalog()
        .claim_shared_ivf(&shared.descriptor, Uuid::new_v4(), 60_000)
        .unwrap()
    {
        SharedIvfClaimResult::Claimed(claim) => claim,
        other => panic!("first caller must own the claim, received {other:?}"),
    };
    repository
        .catalog()
        .publish_shared_ivf(&claim, &location, &shared)
        .unwrap();

    let loaded = repository
        .load_shared_ivf(&shared.fingerprint)
        .await
        .unwrap();
    assert_eq!(loaded.metadata, shared);
    let reference = SharedIvfReference::new(
        &loaded.entry.fingerprint,
        loaded.entry.artifact_uuid,
        &loaded.entry.metadata_location,
    )
    .unwrap();
    assert_eq!(
        repository
            .load_shared_ivf_reference(&reference)
            .await
            .unwrap()
            .metadata,
        loaded.metadata
    );

    let mismatched = SharedIvfReference::new(
        reference.fingerprint,
        Uuid::new_v4(),
        reference.metadata_location,
    )
    .unwrap();
    assert!(
        repository
            .load_shared_ivf_reference(&mismatched)
            .await
            .is_err()
    );

    let malformed = SharedIvfReference {
        fingerprint: loaded.entry.fingerprint.to_uppercase(),
        artifact_uuid: loaded.entry.artifact_uuid,
        metadata_location: loaded.entry.metadata_location,
    };
    assert!(
        repository
            .load_shared_ivf_reference(&malformed)
            .await
            .is_err()
    );
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
async fn metadata_store_reuses_cached_immutable_documents_across_clones() {
    let temporary = TempDir::new().unwrap();
    let writer = metadata_store(&temporary, 1);
    let metadata = metadata_document(&writer);
    let location = writer.write_initial(&metadata).await.unwrap();
    let reader = metadata_store(&temporary, 1);
    assert_eq!(reader.load(&location).await.unwrap(), metadata);
    let cached_reader = reader.clone();
    std::fs::remove_file(url::Url::parse(&location).unwrap().to_file_path().unwrap()).unwrap();

    assert_eq!(cached_reader.load(&location).await.unwrap(), metadata);
    cached_reader.invalidate(&location);
    assert!(cached_reader.load(&location).await.is_err());
}

#[tokio::test]
async fn metadata_store_evicts_the_least_recently_used_document() {
    let temporary = TempDir::new().unwrap();
    let store = metadata_store(&temporary, 2);
    let first = metadata_document(&store);
    let first_location = store.write_initial(&first).await.unwrap();
    let second = metadata_document(&store);
    let second_location = store.write_initial(&second).await.unwrap();

    assert_eq!(store.load(&first_location).await.unwrap(), first);

    let third = metadata_document(&store);
    let third_location = store.write_initial(&third).await.unwrap();
    for location in [&first_location, &second_location, &third_location] {
        std::fs::remove_file(url::Url::parse(location).unwrap().to_file_path().unwrap()).unwrap();
    }

    assert_eq!(store.load(&first_location).await.unwrap(), first);
    assert_eq!(store.load(&third_location).await.unwrap(), third);
    assert!(store.load(&second_location).await.is_err());
}

#[tokio::test]
async fn metadata_store_rejects_documents_larger_than_its_byte_budget() {
    let temporary = TempDir::new().unwrap();
    let writer = metadata_store(&temporary, 1);
    let metadata = metadata_document(&writer);
    let location = writer.write_initial(&metadata).await.unwrap();
    let document_size = usize::try_from(
        std::fs::metadata(url::Url::parse(&location).unwrap().to_file_path().unwrap())
            .unwrap()
            .len(),
    )
    .unwrap();
    let reader =
        metadata_store_with_config(&temporary, MetadataCacheConfig::new(1, document_size - 1));

    assert_eq!(reader.load(&location).await.unwrap(), metadata);
    std::fs::remove_file(url::Url::parse(&location).unwrap().to_file_path().unwrap()).unwrap();

    assert!(reader.load(&location).await.is_err());
}

#[tokio::test]
async fn metadata_store_evicts_entries_to_enforce_its_byte_budget() {
    let temporary = TempDir::new().unwrap();
    let writer = metadata_store(&temporary, 2);
    let first = metadata_document(&writer);
    let first_location = writer.write_initial(&first).await.unwrap();
    let second = metadata_document(&writer);
    let second_location = writer.write_initial(&second).await.unwrap();
    let document_size = [&first_location, &second_location]
        .into_iter()
        .map(|location| {
            usize::try_from(
                std::fs::metadata(url::Url::parse(location).unwrap().to_file_path().unwrap())
                    .unwrap()
                    .len(),
            )
            .unwrap()
        })
        .max()
        .unwrap();
    let reader = metadata_store_with_config(&temporary, MetadataCacheConfig::new(2, document_size));

    assert_eq!(reader.load(&first_location).await.unwrap(), first);
    assert_eq!(reader.load(&second_location).await.unwrap(), second);
    for location in [&first_location, &second_location] {
        std::fs::remove_file(url::Url::parse(location).unwrap().to_file_path().unwrap()).unwrap();
    }

    assert!(reader.load(&first_location).await.is_err());
    assert_eq!(reader.load(&second_location).await.unwrap(), second);
}

#[tokio::test]
async fn metadata_store_applies_resolved_bounds_before_cache_access() {
    let temporary = TempDir::new().unwrap();
    let root = url::Url::from_directory_path(temporary.path().join("warehouse"))
        .unwrap()
        .to_string();
    let config = Arc::new(Mutex::new(MetadataCacheConfig::new(2, usize::MAX)));
    let resolved_config = Arc::clone(&config);
    let store = MetadataStore::open_with_cache_config_resolver(
        Warehouse::open(&root, StorageRegistry::default()).unwrap(),
        move || *resolved_config.lock().unwrap(),
    );
    let first = metadata_document(&store);
    let first_location = store.write_initial(&first).await.unwrap();
    let second = metadata_document(&store);
    let second_location = store.write_initial(&second).await.unwrap();

    *config.lock().unwrap() = MetadataCacheConfig::new(1, usize::MAX);
    assert_eq!(
        store.cache_config(),
        MetadataCacheConfig::new(1, usize::MAX)
    );
    for location in [&first_location, &second_location] {
        std::fs::remove_file(url::Url::parse(location).unwrap().to_file_path().unwrap()).unwrap();
    }

    assert!(store.load(&first_location).await.is_err());
    assert_eq!(store.load(&second_location).await.unwrap(), second);
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
async fn repository_registration_requires_metadata_to_exist_in_storage() {
    let temporary = TempDir::new().unwrap();
    let repository = repository(&temporary);
    let metadata = metadata_document(repository.metadata_store());
    let location = repository
        .metadata_store()
        .write_initial(&metadata)
        .await
        .unwrap();
    std::fs::remove_file(url::Url::parse(&location).unwrap().to_file_path().unwrap()).unwrap();
    let identifier = IndexIdentifier::root("missing_metadata").unwrap();

    assert!(repository.register(&identifier, &location).await.is_err());
    assert!(!repository.exists(&identifier).unwrap());
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
