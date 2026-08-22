use std::collections::BTreeMap;
use std::sync::{Arc, Mutex};

use parqdb_catalog::{CatalogEntry, IndexIdentifier, IvfCentroidsClaimResult, SqliteCatalog};
use parqdb_core::{IndexArtifacts, IndexFormat};
use parqdb_meta::{
    DistanceMetric, IVF_CLUSTERING_PROFILE_VERSION, IndexMetadata, IndexSnapshot,
    IvfCentroidsDescriptor, IvfCentroidsMetadata, IvfCentroidsReference, RelationReference,
    SnapshotLogEntry,
};
use parqdb_storage::{StorageRegistry, Warehouse};
use tempfile::TempDir;
use uuid::Uuid;

use crate::{
    Error, IndexRepository, InitialIndex, LoadedIndex, MetadataCacheConfig, MetadataStore,
    RefreshedIndex, new_snapshot_id, publish_initial, publish_refresh,
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

fn artifacts(store: &MetadataStore, relative_root: &str, nlist: usize) -> IndexArtifacts {
    let centroid_uuid = Uuid::new_v4();
    let uri = store.resolve_location(relative_root, true).unwrap();
    IndexArtifacts {
        format: IndexFormat::ivf(DistanceMetric::L2Squared),
        parameters: BTreeMap::from([
            ("dimension".into(), "2".into()),
            ("nlist".into(), nlist.to_string()),
            ("ntotal".into(), "3".into()),
            ("posting_encoding".into(), "source".into()),
            (
                "ivf_centroids_fingerprint".into(),
                Uuid::new_v4().to_string(),
            ),
            ("ivf_centroids_uuid".into(), centroid_uuid.to_string()),
            (
                "ivf_centroids_metadata_location".into(),
                format!("metadata/{centroid_uuid}/v1.metadata.json"),
            ),
        ]),
        index_relations: BTreeMap::from([
            ("ivf_centroids".into(), source(&format!("{uri}centroids"))),
            ("ivf_postings".into(), source(&format!("{uri}postings"))),
        ]),
    }
}

fn metadata_document(store: &MetadataStore) -> IndexMetadata {
    let index_uuid = Uuid::new_v4();
    let snapshot_id = 701;
    let timestamp_ms = 1_750_000_000_000;
    let build = artifacts(store, "indexes/metadata-test", 2);
    IndexMetadata {
        format_version: 1,
        index_uuid,
        last_updated_ms: timestamp_ms,
        last_sequence_number: 1,
        current_snapshot_id: snapshot_id,
        snapshots: vec![IndexSnapshot {
            snapshot_id,
            sequence_number: 1,
            timestamp_ms,
            summary: BTreeMap::new(),
            vector_field: "embedding".into(),
            source_key_fields: vec!["document_id".into()],
            indexed_rows: 3,
            index_family: "ivf".into(),
            index_schema_version: 1,
            metric: "l2_squared".into(),
            parameters: build.parameters,
            index_relations: build
                .index_relations
                .into_iter()
                .map(|(role, reference)| match reference {
                    RelationReference::Parquet { uri } => {
                        (role, store.relative_location(&uri).unwrap())
                    }
                    RelationReference::Iceberg { .. } => unreachable!(),
                })
                .collect(),
        }],
        snapshot_log: vec![SnapshotLogEntry {
            timestamp_ms,
            snapshot_id,
        }],
        properties: BTreeMap::new(),
    }
}

fn ivf_centroids_document(_store: &MetadataStore) -> IvfCentroidsMetadata {
    let descriptor = IvfCentroidsDescriptor {
        vector_field: "embedding".into(),
        dimension: 2,
        metric: DistanceMetric::L2Squared,
        nlist: 2,
        clustering_profile_version: IVF_CLUSTERING_PROFILE_VERSION,
    };
    let artifact_uuid = Uuid::new_v4();
    IvfCentroidsMetadata {
        format_version: 1,
        artifact_uuid,
        fingerprint: descriptor.fingerprint().unwrap(),
        created_at_ms: 1_750_000_000_000,
        descriptor,
        centroids: "indexes/centroid-artifacts/centroids/".into(),
        roots: "indexes/centroid-artifacts/roots/".into(),
    }
}

#[tokio::test]
async fn repository_validates_the_centroid_artifact_against_the_logical_snapshot() {
    let temporary = TempDir::new().unwrap();
    let repository = repository(&temporary);
    let store = repository.metadata_store();
    let centroids = ivf_centroids_document(store);
    let centroid_location = store.write_ivf_centroids(&centroids).await.unwrap();
    let mut metadata = metadata_document(store);
    let snapshot = metadata.snapshots.first_mut().unwrap();
    snapshot
        .index_relations
        .insert("ivf_centroids".into(), centroids.centroids.clone());
    snapshot.parameters.insert(
        "ivf_centroids_fingerprint".into(),
        centroids.fingerprint.clone(),
    );
    snapshot.parameters.insert(
        "ivf_centroids_uuid".into(),
        centroids.artifact_uuid.to_string(),
    );
    snapshot.parameters.insert(
        "ivf_centroids_metadata_location".into(),
        store.relative_location(&centroid_location).unwrap(),
    );
    let source = source("file:///data/documents.parquet");
    let loaded = LoadedIndex {
        entry: CatalogEntry {
            identifier: IndexIdentifier::root("documents_embedding").unwrap(),
            metadata_location: "file:///metadata.json".into(),
            source: source.clone(),
        },
        metadata,
    };

    repository
        .load_snapshot_ivf_centroids(&loaded)
        .await
        .unwrap();

    let mut mismatched = loaded;
    mismatched.metadata.snapshots[0].vector_field = "other_embedding".into();
    assert!(
        repository
            .load_snapshot_ivf_centroids(&mismatched)
            .await
            .is_err()
    );
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
async fn metadata_store_bounds_index_and_centroid_documents_in_one_cache() {
    let temporary = TempDir::new().unwrap();
    let store = metadata_store(&temporary, 1);
    let index = metadata_document(&store);
    let index_location = store.write_initial(&index).await.unwrap();
    let centroids = ivf_centroids_document(&store);
    let centroid_location = store.write_ivf_centroids(&centroids).await.unwrap();

    for location in [&index_location, &centroid_location] {
        std::fs::remove_file(url::Url::parse(location).unwrap().to_file_path().unwrap()).unwrap();
    }

    assert!(store.load(&index_location).await.is_err());
    assert_eq!(
        store.load_ivf_centroids(&centroid_location).await.unwrap(),
        centroids
    );
}

#[tokio::test]
async fn repository_validates_ivf_centroids_catalog_and_reference_identity() {
    let temporary = TempDir::new().unwrap();
    let repository = repository(&temporary);
    let centroids = ivf_centroids_document(repository.metadata_store());
    let location = repository
        .metadata_store()
        .write_ivf_centroids(&centroids)
        .await
        .unwrap();
    let source = source("file:///data/documents.parquet");
    let claim = match repository
        .catalog()
        .claim_ivf_centroids(&source, &centroids.descriptor, Uuid::new_v4(), 60_000)
        .unwrap()
    {
        IvfCentroidsClaimResult::Claimed(claim) => claim,
        other => panic!("first caller must own the claim, received {other:?}"),
    };
    repository
        .catalog()
        .publish_ivf_centroids(&claim, &location, &centroids)
        .unwrap();

    let loaded = repository
        .load_ivf_centroids(&source, &centroids.fingerprint)
        .await
        .unwrap();
    assert_eq!(loaded.metadata, centroids);
    let reference = IvfCentroidsReference::new(
        &loaded.entry.fingerprint,
        loaded.entry.artifact_uuid,
        repository
            .metadata_store()
            .relative_location(&loaded.entry.metadata_location)
            .unwrap(),
    )
    .unwrap();
    assert_eq!(
        repository
            .load_ivf_centroids_reference(&source, &reference)
            .await
            .unwrap()
            .metadata,
        loaded.metadata
    );

    let mismatched = IvfCentroidsReference::new(
        reference.fingerprint,
        Uuid::new_v4(),
        reference.metadata_location,
    )
    .unwrap();
    assert!(
        repository
            .load_ivf_centroids_reference(&source, &mismatched)
            .await
            .is_err()
    );

    let malformed = IvfCentroidsReference {
        fingerprint: loaded.entry.fingerprint.to_uppercase(),
        artifact_uuid: loaded.entry.artifact_uuid,
        metadata_location: loaded.entry.metadata_location,
    };
    assert!(
        repository
            .load_ivf_centroids_reference(&source, &malformed)
            .await
            .is_err()
    );
}

#[tokio::test]
async fn metadata_store_rejects_absolute_artifact_locations() {
    let temporary = TempDir::new().unwrap();
    let repository = repository(&temporary);
    let store = repository.metadata_store();
    let mut metadata = metadata_document(store);
    metadata.snapshots[0]
        .index_relations
        .insert("ivf_postings".into(), "file:///tmp/other/postings/".into());

    assert!(store.write_initial(&metadata).await.is_err());

    let mut centroids = ivf_centroids_document(store);
    centroids.centroids = "file:///tmp/other/centroids/".into();
    assert!(matches!(
        store.write_ivf_centroids(&centroids).await,
        Err(Error::InvalidMetadata(_))
    ));
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
            build: artifacts(repository.metadata_store(), "indexes/v1", 2),
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

    assert!(
        repository
            .register(
                &identifier,
                &source("file:///data/documents.parquet"),
                &location,
            )
            .await
            .is_err()
    );
    assert!(!repository.exists(&identifier).unwrap());
}

#[tokio::test]
async fn publication_uses_the_builder_format_descriptor() {
    let temporary = TempDir::new().unwrap();
    let repository = repository(&temporary);
    let mut build = artifacts(repository.metadata_store(), "indexes/v2", 2);
    build.format = IndexFormat::ivf(DistanceMetric::Cosine);

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
    assert_eq!(snapshot.index_schema_version, 1);
    assert_eq!(snapshot.metric, "cosine");
    assert_eq!(snapshot.parameters["posting_encoding"], "source");
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
                build: artifacts(repository.metadata_store(), &format!("indexes/{name}"), 2),
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
            build: artifacts(repository.metadata_store(), "indexes/v1", 2),
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
            build: artifacts(repository.metadata_store(), "indexes/v2", 3),
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
            build: artifacts(repository.metadata_store(), "indexes/first", 2),
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
            build: artifacts(repository.metadata_store(), "indexes/duplicate", 2),
        },
    )
    .await;

    assert!(matches!(
        duplicate,
        Err(crate::Error::Catalog(parqdb_catalog::Error::AlreadyExists(
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
