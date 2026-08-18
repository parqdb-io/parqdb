//! Integration tests for the `SQLite` index catalog.

use std::collections::BTreeMap;
use std::sync::{Arc, Barrier};

use parqdb_catalog::{
    CatalogTombstone, Error, IndexCatalog, IndexIdentifier, IvfCentroidsCatalogEntry,
    IvfCentroidsClaim, IvfCentroidsClaimResult, SqliteCatalog, TableCatalog, TableDefinition,
    TableIdentifier,
};
use parqdb_meta::{
    DistanceMetric, IndexMetadata, IvfCentroidsDescriptor, IvfCentroidsMetadata, RelationReference,
};
use rusqlite::Connection;
use tempfile::TempDir;
use uuid::Uuid;

mod common;

use common::{assert_index_catalog_contract, file_uri, metadata, refreshed};

fn test_source() -> RelationReference {
    RelationReference::Parquet {
        uri: "file:///parqdb-test-source.parquet".into(),
    }
}

trait TestCatalogExt {
    fn register_test(
        &self,
        identifier: &IndexIdentifier,
        metadata_location: &str,
        metadata: &IndexMetadata,
    ) -> parqdb_catalog::Result<()>;

    fn commit_test(
        &self,
        identifier: &IndexIdentifier,
        base_metadata_location: &str,
        new_metadata_location: &str,
        base_metadata: &IndexMetadata,
        new_metadata: &IndexMetadata,
    ) -> parqdb_catalog::Result<()>;

    fn claim_ivf_centroids_test(
        &self,
        descriptor: &IvfCentroidsDescriptor,
        owner: Uuid,
        lease_duration_ms: i64,
    ) -> parqdb_catalog::Result<IvfCentroidsClaimResult>;

    fn publish_ivf_centroids_test(
        &self,
        claim: &IvfCentroidsClaim,
        metadata_location: &str,
        metadata: &IvfCentroidsMetadata,
    ) -> parqdb_catalog::Result<IvfCentroidsCatalogEntry>;

    fn load_ivf_centroids_test(
        &self,
        fingerprint: &str,
    ) -> parqdb_catalog::Result<IvfCentroidsCatalogEntry>;
}

impl TestCatalogExt for SqliteCatalog {
    fn register_test(
        &self,
        identifier: &IndexIdentifier,
        metadata_location: &str,
        metadata: &IndexMetadata,
    ) -> parqdb_catalog::Result<()> {
        IndexCatalog::register(
            self,
            identifier,
            &test_source(),
            metadata_location,
            metadata,
        )
    }

    fn commit_test(
        &self,
        identifier: &IndexIdentifier,
        base_metadata_location: &str,
        new_metadata_location: &str,
        base_metadata: &IndexMetadata,
        new_metadata: &IndexMetadata,
    ) -> parqdb_catalog::Result<()> {
        IndexCatalog::commit(
            self,
            identifier,
            &test_source(),
            base_metadata_location,
            new_metadata_location,
            base_metadata,
            new_metadata,
        )
    }

    fn claim_ivf_centroids_test(
        &self,
        descriptor: &IvfCentroidsDescriptor,
        owner: Uuid,
        lease_duration_ms: i64,
    ) -> parqdb_catalog::Result<IvfCentroidsClaimResult> {
        IndexCatalog::claim_ivf_centroids(
            self,
            &test_source(),
            descriptor,
            owner,
            lease_duration_ms,
        )
    }

    fn publish_ivf_centroids_test(
        &self,
        claim: &IvfCentroidsClaim,
        metadata_location: &str,
        metadata: &IvfCentroidsMetadata,
    ) -> parqdb_catalog::Result<IvfCentroidsCatalogEntry> {
        IndexCatalog::publish_ivf_centroids(self, claim, metadata_location, metadata)
    }

    fn load_ivf_centroids_test(
        &self,
        fingerprint: &str,
    ) -> parqdb_catalog::Result<IvfCentroidsCatalogEntry> {
        IndexCatalog::load_ivf_centroids(self, &test_source(), fingerprint)
    }
}

#[test]
fn sqlite_catalog_satisfies_index_catalog_contract() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    assert_index_catalog_contract(&catalog, temporary.path());
}

#[test]
fn ivf_centroids_claim_publish_and_lookup_are_atomic() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let descriptor = ivf_centroids_descriptor(temporary.path());
    let first_owner = Uuid::new_v4();
    let second_owner = Uuid::new_v4();
    let claim = match catalog
        .claim_ivf_centroids_test(&descriptor, first_owner, 60_000)
        .unwrap()
    {
        IvfCentroidsClaimResult::Claimed(claim) => claim,
        other => panic!("first caller must own the claim, received {other:?}"),
    };
    assert!(matches!(
        catalog
            .claim_ivf_centroids_test(&descriptor, second_owner, 60_000)
            .unwrap(),
        IvfCentroidsClaimResult::Busy { .. }
    ));
    catalog.renew_ivf_centroids_claim(&claim, 60_000).unwrap();

    let metadata = ivf_centroids_metadata(temporary.path(), &descriptor);
    let metadata_location = file_uri(&temporary.path().join("ivf-centroids-v1.metadata.json"));
    assert!(matches!(
        catalog.publish_ivf_centroids_test(&claim, "ivf-centroids-v1.metadata.json", &metadata),
        Err(Error::InvalidMetadata(_))
    ));
    let published = catalog
        .publish_ivf_centroids_test(&claim, &metadata_location, &metadata)
        .unwrap();

    assert_eq!(
        catalog
            .load_ivf_centroids_test(&published.fingerprint)
            .unwrap(),
        published
    );
    assert_eq!(
        catalog.list_ivf_centroids().unwrap(),
        std::slice::from_ref(&published)
    );
    assert!(matches!(
        catalog
            .claim_ivf_centroids_test(&descriptor, second_owner, 60_000)
            .unwrap(),
        IvfCentroidsClaimResult::Ready(entry) if entry == published
    ));

    let mut changed = published.clone();
    changed.metadata_location.push_str(".changed");
    assert!(!catalog.purge_ivf_centroids(&changed).unwrap());
    assert!(catalog.purge_ivf_centroids(&published).unwrap());
    assert!(catalog.list_ivf_centroids().unwrap().is_empty());
    assert!(matches!(
        catalog.load_ivf_centroids_test(&published.fingerprint),
        Err(Error::IvfCentroidsNotFound(_))
    ));
}

#[test]
fn ivf_centroids_listing_returns_multiple_ready_entries_in_fingerprint_order() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let mut descriptors = [
        ivf_centroids_descriptor(temporary.path()),
        ivf_centroids_descriptor(temporary.path()),
    ];
    descriptors[1].vector_field = "other_embedding".into();
    let mut published = Vec::new();

    for (position, descriptor) in descriptors.iter().enumerate() {
        let claim = match catalog
            .claim_ivf_centroids_test(descriptor, Uuid::new_v4(), 60_000)
            .unwrap()
        {
            IvfCentroidsClaimResult::Claimed(claim) => claim,
            other => panic!("distinct descriptor must create a claim, received {other:?}"),
        };
        let metadata = ivf_centroids_metadata(temporary.path(), descriptor);
        let metadata_location = file_uri(
            &temporary
                .path()
                .join(format!("ivf-centroids-{position}.metadata.json")),
        );
        published.push(
            catalog
                .publish_ivf_centroids_test(&claim, &metadata_location, &metadata)
                .unwrap(),
        );
    }

    published.sort_by(|left, right| left.fingerprint.cmp(&right.fingerprint));
    assert_eq!(catalog.list_ivf_centroids().unwrap(), published);
}

#[test]
fn publishing_ivf_centroids_with_a_mismatched_fingerprint_is_rejected() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let descriptor = ivf_centroids_descriptor(temporary.path());
    let claim = match catalog
        .claim_ivf_centroids_test(&descriptor, Uuid::new_v4(), 60_000)
        .unwrap()
    {
        IvfCentroidsClaimResult::Claimed(claim) => claim,
        other => panic!("first caller must own the claim, received {other:?}"),
    };
    let mut other_descriptor = descriptor;
    other_descriptor.vector_field = "other_embedding".into();
    let metadata = ivf_centroids_metadata(temporary.path(), &other_descriptor);
    let metadata_location = file_uri(&temporary.path().join("ivf-centroids-v1.metadata.json"));

    assert!(matches!(
        catalog.publish_ivf_centroids_test(&claim, &metadata_location, &metadata),
        Err(Error::InvalidMetadata(message))
            if message.contains("published IVF centroid fingerprint does not match claim")
    ));
}

#[test]
fn expired_ivf_centroids_claim_cannot_be_renewed_or_published() {
    let temporary = TempDir::new().unwrap();
    let database = temporary.path().join("catalog.sqlite");
    let catalog = SqliteCatalog::open(&database).unwrap();
    let descriptor = ivf_centroids_descriptor(temporary.path());
    let claim = match catalog
        .claim_ivf_centroids_test(&descriptor, Uuid::new_v4(), 60_000)
        .unwrap()
    {
        IvfCentroidsClaimResult::Claimed(claim) => claim,
        other => panic!("first caller must own the claim, received {other:?}"),
    };
    Connection::open(&database)
        .unwrap()
        .execute("UPDATE ivf_centroid_artifacts SET lease_expires_ms = 0", [])
        .unwrap();
    let metadata = ivf_centroids_metadata(temporary.path(), &descriptor);
    let metadata_location = file_uri(&temporary.path().join("ivf-centroids-v1.metadata.json"));

    assert!(matches!(
        catalog.renew_ivf_centroids_claim(&claim, 60_000),
        Err(Error::IvfCentroidsClaimLost(_))
    ));
    assert!(matches!(
        catalog.publish_ivf_centroids_test(&claim, &metadata_location, &metadata),
        Err(Error::IvfCentroidsClaimLost(_))
    ));
}

#[test]
fn expired_ivf_centroids_claim_uses_compare_and_swap_publication() {
    let temporary = TempDir::new().unwrap();
    let database = temporary.path().join("catalog.sqlite");
    let catalog = SqliteCatalog::open(&database).unwrap();
    let descriptor = ivf_centroids_descriptor(temporary.path());
    let first = match catalog
        .claim_ivf_centroids_test(&descriptor, Uuid::new_v4(), 60_000)
        .unwrap()
    {
        IvfCentroidsClaimResult::Claimed(claim) => claim,
        other => panic!("first caller must own the claim, received {other:?}"),
    };
    Connection::open(&database)
        .unwrap()
        .execute("UPDATE ivf_centroid_artifacts SET lease_expires_ms = 0", [])
        .unwrap();
    let second = match catalog
        .claim_ivf_centroids_test(&descriptor, Uuid::new_v4(), 60_000)
        .unwrap()
    {
        IvfCentroidsClaimResult::Claimed(claim) => claim,
        other => panic!("expired claim must be replaced, received {other:?}"),
    };
    let metadata = ivf_centroids_metadata(temporary.path(), &descriptor);
    let metadata_location = file_uri(&temporary.path().join("ivf-centroids-v1.metadata.json"));

    assert!(matches!(
        catalog.publish_ivf_centroids_test(&first, &metadata_location, &metadata),
        Err(Error::IvfCentroidsClaimLost(_))
    ));
    catalog
        .publish_ivf_centroids_test(&second, &metadata_location, &metadata)
        .unwrap();
}

#[test]
fn abandoned_ivf_centroids_claim_can_be_retried() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let descriptor = ivf_centroids_descriptor(temporary.path());
    let claim = match catalog
        .claim_ivf_centroids_test(&descriptor, Uuid::new_v4(), 60_000)
        .unwrap()
    {
        IvfCentroidsClaimResult::Claimed(claim) => claim,
        other => panic!("first caller must own the claim, received {other:?}"),
    };

    catalog
        .abandon_ivf_centroids(&claim, "fixture failure")
        .unwrap();
    assert!(matches!(
        catalog.load_ivf_centroids_test(&claim.fingerprint),
        Err(Error::IvfCentroidsNotFound(_))
    ));
    assert!(matches!(
        catalog
            .claim_ivf_centroids_test(&descriptor, Uuid::new_v4(), 60_000)
            .unwrap(),
        IvfCentroidsClaimResult::Claimed(_)
    ));
}

#[test]
fn concurrent_ivf_centroids_claims_allow_exactly_one_builder() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let descriptor = ivf_centroids_descriptor(temporary.path());
    let barrier = Arc::new(Barrier::new(3));
    let handles = [Uuid::new_v4(), Uuid::new_v4()].map(|owner| {
        let catalog = catalog.clone();
        let descriptor = descriptor.clone();
        let barrier = Arc::clone(&barrier);
        std::thread::spawn(move || {
            barrier.wait();
            catalog.claim_ivf_centroids_test(&descriptor, owner, 60_000)
        })
    });
    barrier.wait();
    let results = handles.map(|handle| handle.join().unwrap().unwrap());

    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, IvfCentroidsClaimResult::Claimed(_)))
            .count(),
        1
    );
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, IvfCentroidsClaimResult::Busy { .. }))
            .count(),
        1
    );
}

#[test]
fn ivf_centroids_reuse_follows_iceberg_exact_state_across_renames() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let table_uuid = Uuid::new_v4();
    let descriptor = ivf_centroids_descriptor(temporary.path());
    let first_source = RelationReference::Iceberg {
        catalog: "first".into(),
        namespace: vec!["analytics".into()],
        name: "documents".into(),
        table_uuid,
        snapshot_id: 101,
    };
    let claim = match IndexCatalog::claim_ivf_centroids(
        &catalog,
        &first_source,
        &descriptor,
        Uuid::new_v4(),
        60_000,
    )
    .unwrap()
    {
        IvfCentroidsClaimResult::Claimed(claim) => claim,
        other => panic!("first caller must own the claim, received {other:?}"),
    };
    let metadata = ivf_centroids_metadata(temporary.path(), &descriptor);
    let metadata_location = file_uri(&temporary.path().join("ivf-centroids-v1.metadata.json"));
    let published =
        IndexCatalog::publish_ivf_centroids(&catalog, &claim, &metadata_location, &metadata)
            .unwrap();

    let renamed_source = RelationReference::Iceberg {
        catalog: "second".into(),
        namespace: vec!["renamed".into()],
        name: "vectors".into(),
        table_uuid,
        snapshot_id: 101,
    };
    assert!(matches!(
        IndexCatalog::claim_ivf_centroids(
            &catalog,
            &renamed_source,
            &descriptor,
            Uuid::new_v4(),
            60_000,
        )
        .unwrap(),
        IvfCentroidsClaimResult::Ready(entry) if entry == published
    ));

    let different_snapshot = RelationReference::Iceberg {
        catalog: "second".into(),
        namespace: vec!["renamed".into()],
        name: "vectors".into(),
        table_uuid,
        snapshot_id: 102,
    };
    assert!(matches!(
        IndexCatalog::claim_ivf_centroids(
            &catalog,
            &different_snapshot,
            &descriptor,
            Uuid::new_v4(),
            60_000,
        )
        .unwrap(),
        IvfCentroidsClaimResult::Claimed(_)
    ));
}

#[test]
fn root_namespace_exists_in_a_new_catalog() {
    let temporary = TempDir::new().unwrap();
    let database = temporary.path().join("catalog.sqlite");
    let catalog = SqliteCatalog::open(&database).unwrap();
    assert!(catalog.list(&[]).unwrap().is_empty());

    let connection = Connection::open(database).unwrap();
    let application_id: i64 = connection
        .query_row("PRAGMA application_id", [], |row| row.get(0))
        .unwrap();
    let user_version: i64 = connection
        .query_row("PRAGMA user_version", [], |row| row.get(0))
        .unwrap();
    assert_eq!(application_id, 0x5051_4442);
    assert_eq!(user_version, 1);
}

#[test]
fn table_definitions_share_the_sqlite_catalog_and_survive_reopen() {
    let temporary = TempDir::new().unwrap();
    let database = temporary.path().join("catalog.sqlite");
    let identifier =
        TableIdentifier::new("datafusion", vec!["public".into()], "documents").unwrap();
    let definition = TableDefinition::new(
        identifier.clone(),
        "parquet",
        BTreeMap::from([
            ("location".into(), "/data/*/documents/*.parquet".into()),
            ("schema".into(), "arrow-ipc:example".into()),
        ]),
    )
    .unwrap();

    SqliteCatalog::open(&database)
        .unwrap()
        .create_table(&definition)
        .unwrap();
    let reopened = SqliteCatalog::open(&database).unwrap();

    assert_eq!(reopened.load_table(&identifier).unwrap(), definition);
    assert_eq!(
        reopened
            .list_tables("datafusion", &["public".into()])
            .unwrap(),
        [identifier]
    );
}

#[test]
fn table_identifiers_cannot_be_silently_rebound() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let identifier =
        TableIdentifier::new("datafusion", vec!["public".into()], "documents").unwrap();
    let first = TableDefinition::new(
        identifier.clone(),
        "parquet",
        BTreeMap::from([("location".into(), "/data/first/*.parquet".into())]),
    )
    .unwrap();
    let second = TableDefinition::new(
        identifier.clone(),
        "parquet",
        BTreeMap::from([("location".into(), "/data/second/*.parquet".into())]),
    )
    .unwrap();

    catalog.create_table(&first).unwrap();
    assert!(matches!(
        catalog.create_table(&second),
        Err(Error::TableAlreadyExists(found)) if found == identifier
    ));
    assert_eq!(catalog.load_table(&identifier).unwrap(), first);
}

#[test]
fn dropping_a_table_definition_preserves_indexes_and_external_data() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let identifier =
        TableIdentifier::new("datafusion", vec!["public".into()], "documents").unwrap();
    let definition = TableDefinition::new(
        identifier.clone(),
        "parquet",
        BTreeMap::from([("location".into(), "/data/documents/".into())]),
    )
    .unwrap();
    catalog.create_table(&definition).unwrap();

    catalog.drop_table(&identifier).unwrap();

    assert!(matches!(
        catalog.load_table(&identifier),
        Err(Error::TableNotFound(found)) if found == identifier
    ));
    assert!(matches!(
        catalog.drop_table(&identifier),
        Err(Error::TableNotFound(found)) if found == identifier
    ));
}

#[test]
fn namespaces_are_structural_and_isolated() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let metadata = metadata(temporary.path());
    let location = file_uri(&temporary.path().join("v1.metadata.json"));
    let root = IndexIdentifier::root("root-index").unwrap();
    catalog.register_test(&root, &location, &metadata).unwrap();
    let nested_location = file_uri(&temporary.path().join("nested.metadata.json"));
    let mut nested_metadata = metadata.clone();
    nested_metadata.index_uuid = Uuid::new_v4();
    let nested = IndexIdentifier::new(vec!["a".into(), "b".into()], "documents").unwrap();
    catalog
        .register_test(&nested, &nested_location, &nested_metadata)
        .unwrap();

    assert_eq!(
        catalog.list(nested.namespace()).unwrap(),
        std::slice::from_ref(&nested)
    );
    assert_eq!(catalog.list_all().unwrap(), [nested, root]);
    assert!(matches!(
        catalog.list(&["a.b".into()]),
        Err(Error::NamespaceNotFound(_))
    ));
}

#[test]
fn register_rejects_duplicates_and_invalid_metadata() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let identifier = IndexIdentifier::root("documents").unwrap();
    let metadata = metadata(temporary.path());
    let location = file_uri(&temporary.path().join("v1.metadata.json"));
    catalog
        .register_test(&identifier, &location, &metadata)
        .unwrap();

    assert!(matches!(
        catalog.register_test(&identifier, &location, &metadata),
        Err(Error::AlreadyExists(_))
    ));

    let invalid_location = file_uri(&temporary.path().join("invalid.metadata.json"));
    let mut invalid_metadata = metadata.clone();
    invalid_metadata.format_version = 2;
    let invalid = IndexIdentifier::root("invalid").unwrap();
    assert!(matches!(
        catalog.register_test(&invalid, &invalid_location, &invalid_metadata),
        Err(Error::InvalidMetadata(_))
    ));
    assert!(matches!(
        catalog.load(&invalid),
        Err(Error::IndexNotFound(_))
    ));
}

#[test]
fn commit_replaces_only_the_expected_base() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let identifier = IndexIdentifier::root("documents").unwrap();
    let base = metadata(temporary.path());
    let next = refreshed(&base, 702);
    let losing = refreshed(&base, 703);
    let base_location = file_uri(&temporary.path().join("v1.metadata.json"));
    let next_location = file_uri(&temporary.path().join("v2.metadata.json"));
    let losing_location = file_uri(&temporary.path().join("v2-losing.metadata.json"));
    catalog
        .register_test(&identifier, &base_location, &base)
        .unwrap();

    catalog
        .commit_test(&identifier, &base_location, &next_location, &base, &next)
        .unwrap();
    assert_eq!(
        catalog.load(&identifier).unwrap().metadata_location,
        next_location
    );
    assert!(matches!(
        catalog.commit_test(
            &identifier,
            &base_location,
            &losing_location,
            &base,
            &losing
        ),
        Err(Error::CommitConflict(_))
    ));
    let tombstones = catalog.list_tombstones().unwrap();
    assert_eq!(tombstones.len(), 1);
    assert_eq!(tombstones[0].metadata_location, base_location);
    assert!(tombstones[0].unreachable_since_ms > 0);
}

#[test]
fn drop_records_reachability_loss_and_register_clears_it() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let identifier = IndexIdentifier::root("documents").unwrap();
    let recovered = IndexIdentifier::root("recovered").unwrap();
    let metadata = metadata(temporary.path());
    let location = file_uri(&temporary.path().join("v1.metadata.json"));
    catalog
        .register_test(&identifier, &location, &metadata)
        .unwrap();

    catalog.drop(&identifier).unwrap();
    let tombstones = catalog.list_tombstones().unwrap();
    assert_eq!(tombstones.len(), 1);
    assert_eq!(tombstones[0].metadata_location, location);
    assert!(tombstones[0].unreachable_since_ms > 0);

    catalog
        .register_test(&recovered, &location, &metadata)
        .unwrap();
    assert!(catalog.list_tombstones().unwrap().is_empty());
}

#[test]
fn tombstone_purge_uses_compare_and_delete() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let identifier = IndexIdentifier::root("documents").unwrap();
    let metadata = metadata(temporary.path());
    let location = file_uri(&temporary.path().join("v1.metadata.json"));
    catalog
        .register_test(&identifier, &location, &metadata)
        .unwrap();
    catalog.drop(&identifier).unwrap();
    let tombstone = catalog.list_tombstones().unwrap().pop().unwrap();
    let stale = CatalogTombstone {
        metadata_location: tombstone.metadata_location.clone(),
        unreachable_since_ms: tombstone.unreachable_since_ms - 1,
    };

    assert!(!catalog.purge_tombstone(&stale).unwrap());
    assert!(catalog.purge_tombstone(&tombstone).unwrap());
    assert!(catalog.list_tombstones().unwrap().is_empty());
}

#[test]
fn concurrent_commits_allow_exactly_one_winner() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let identifier = IndexIdentifier::root("documents").unwrap();
    let base = metadata(temporary.path());
    let first = refreshed(&base, 702);
    let second = refreshed(&base, 703);
    let base_location = file_uri(&temporary.path().join("v1.metadata.json"));
    let first_location = file_uri(&temporary.path().join("v2-first.metadata.json"));
    let second_location = file_uri(&temporary.path().join("v2-second.metadata.json"));
    catalog
        .register_test(&identifier, &base_location, &base)
        .unwrap();

    let barrier = Arc::new(Barrier::new(3));
    let handles =
        [(first_location, first), (second_location, second)].map(|(new_location, new_metadata)| {
            let catalog = catalog.clone();
            let barrier = Arc::clone(&barrier);
            let base_location = base_location.clone();
            let base = base.clone();
            let identifier = identifier.clone();
            std::thread::spawn(move || {
                barrier.wait();
                catalog.commit_test(
                    &identifier,
                    &base_location,
                    &new_location,
                    &base,
                    &new_metadata,
                )
            })
        });
    barrier.wait();
    let results = handles.map(|handle| handle.join().unwrap());
    assert_eq!(results.iter().filter(|result| result.is_ok()).count(), 1);
    assert_eq!(
        results
            .iter()
            .filter(|result| matches!(result, Err(Error::CommitConflict(_))))
            .count(),
        1
    );
}

#[test]
fn commit_rejects_uuid_changes_without_publication() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let identifier = IndexIdentifier::root("documents").unwrap();
    let base = metadata(temporary.path());
    let base_location = file_uri(&temporary.path().join("v1.metadata.json"));
    catalog
        .register_test(&identifier, &base_location, &base)
        .unwrap();

    let mut wrong_uuid = refreshed(&base, 702);
    wrong_uuid.index_uuid = Uuid::new_v4();
    let wrong_uuid_location = file_uri(&temporary.path().join("wrong-uuid.metadata.json"));
    assert!(matches!(
        catalog.commit_test(
            &identifier,
            &base_location,
            &wrong_uuid_location,
            &base,
            &wrong_uuid
        ),
        Err(Error::IndexUuidMismatch(_))
    ));

    assert_eq!(
        catalog.load(&identifier).unwrap().metadata_location,
        base_location
    );
}

#[test]
fn commit_rejects_rewriting_an_existing_snapshot() {
    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();
    let identifier = IndexIdentifier::root("documents").unwrap();
    let base = metadata(temporary.path());
    let base_location = file_uri(&temporary.path().join("v1.metadata.json"));
    catalog
        .register_test(&identifier, &base_location, &base)
        .unwrap();

    let mut rewritten = refreshed(&base, 702);
    rewritten.snapshots[0]
        .summary
        .insert("rewritten".into(), "true".into());
    let rewritten_location = file_uri(&temporary.path().join("rewritten.metadata.json"));
    assert!(matches!(
        catalog.commit_test(
            &identifier,
            &base_location,
            &rewritten_location,
            &base,
            &rewritten
        ),
        Err(Error::InvalidMetadata(_))
    ));
    assert_eq!(
        catalog.load(&identifier).unwrap().metadata_location,
        base_location
    );
}

#[test]
fn rejects_an_unversioned_existing_schema() {
    let temporary = TempDir::new().unwrap();
    let database = temporary.path().join("catalog.sqlite");
    let connection = Connection::open(&database).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE indexes (
                 name TEXT PRIMARY KEY,
                 metadata_location TEXT NOT NULL,
                 index_uuid TEXT NOT NULL UNIQUE,
                 source_identity TEXT NOT NULL
             );
             CREATE INDEX indexes_by_source ON indexes(source_identity);",
        )
        .unwrap();
    drop(connection);

    assert!(matches!(
        SqliteCatalog::open(&database),
        Err(Error::UnsupportedSchemaVersion(0))
    ));
}

#[test]
fn rejects_an_unrelated_unversioned_database_without_modifying_it() {
    let temporary = TempDir::new().unwrap();
    let database = temporary.path().join("catalog.sqlite");
    let connection = Connection::open(&database).unwrap();
    connection
        .execute("CREATE TABLE application_data (id INTEGER PRIMARY KEY)", [])
        .unwrap();
    drop(connection);

    assert!(matches!(
        SqliteCatalog::open(&database),
        Err(Error::UnsupportedApplicationId(0))
    ));

    let connection = Connection::open(database).unwrap();
    let objects = connection
        .prepare(
            "SELECT name FROM sqlite_schema
             WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
             ORDER BY name",
        )
        .unwrap()
        .query_map([], |row| row.get::<_, String>(0))
        .unwrap()
        .collect::<rusqlite::Result<Vec<_>>>()
        .unwrap();
    assert_eq!(objects, ["application_data"]);
}

#[test]
fn rejects_unsupported_schema_versions() {
    for version in [2, 3, 4, 5, 6] {
        let temporary = TempDir::new().unwrap();
        let database = temporary.path().join("catalog.sqlite");
        let connection = Connection::open(&database).unwrap();
        connection
            .execute_batch(&format!(
                "CREATE TABLE indexes (name TEXT PRIMARY KEY);
                 PRAGMA user_version={version};"
            ))
            .unwrap();
        drop(connection);

        assert!(matches!(
            SqliteCatalog::open(&database),
            Err(Error::UnsupportedSchemaVersion(actual)) if actual == version
        ));
    }
}

#[test]
fn rejects_the_current_schema_without_the_parqdb_application_id() {
    let temporary = TempDir::new().unwrap();
    let database = temporary.path().join("catalog.sqlite");
    let connection = Connection::open(&database).unwrap();
    connection
        .execute_batch(
            "CREATE TABLE indexes (name TEXT PRIMARY KEY);
             PRAGMA user_version=1;",
        )
        .unwrap();
    drop(connection);

    assert!(matches!(
        SqliteCatalog::open(&database),
        Err(Error::UnsupportedApplicationId(0))
    ));
}

fn ivf_centroids_descriptor(root: &std::path::Path) -> IvfCentroidsDescriptor {
    let _ = root;
    IvfCentroidsDescriptor {
        vector_field: "embedding".into(),
        dimension: 2,
        metric: DistanceMetric::L2Squared,
        nlist: 2,
        clustering_profile_version: 1,
    }
}

fn ivf_centroids_metadata(
    _root: &std::path::Path,
    descriptor: &IvfCentroidsDescriptor,
) -> IvfCentroidsMetadata {
    IvfCentroidsMetadata {
        format_version: 1,
        artifact_uuid: Uuid::new_v4(),
        fingerprint: descriptor.fingerprint().unwrap(),
        created_at_ms: 1_750_000_000_000,
        descriptor: descriptor.clone(),
        centroids: "centroids/".into(),
        roots: "roots/".into(),
    }
}
