use std::collections::BTreeMap;
use std::path::Path;

use relify_catalog::{Error, IndexCatalog, IndexIdentifier};
use relify_meta::{IndexMetadata, IndexSnapshot, RelationReference, SnapshotLogEntry};
use url::Url;
use uuid::Uuid;

pub(crate) fn file_uri(path: &Path) -> String {
    Url::from_file_path(path).unwrap().into()
}

pub(crate) fn directory_uri(path: &Path) -> String {
    Url::from_directory_path(path).unwrap().into()
}

pub(crate) fn metadata(root: &Path) -> IndexMetadata {
    let index_uuid = Uuid::parse_str("2f1c7f5e-3c43-4a44-8f2a-cf560c4db8d1").unwrap();
    let timestamp_ms = 1_750_000_000_000;
    let relation = |name: &str| RelationReference::Parquet {
        uri: directory_uri(&root.join(name)),
    };
    let snapshot = IndexSnapshot {
        snapshot_id: 701,
        sequence_number: 1,
        timestamp_ms,
        summary: BTreeMap::new(),
        source: RelationReference::Parquet {
            uri: file_uri(&root.join("source.parquet")),
        },
        vector_field: "embedding".into(),
        source_key_fields: vec!["document_id".into()],
        index_family: "ivf".into(),
        index_schema_version: 1,
        metric: "l2_squared".into(),
        parameters: BTreeMap::from([
            ("dimension".into(), "2".into()),
            ("nlist".into(), "2".into()),
            ("ntotal".into(), "4".into()),
            ("store_vectors".into(), "true".into()),
        ]),
        index_relations: BTreeMap::from([
            ("ivf_centroids".into(), relation("centroids")),
            ("ivf_postings".into(), relation("postings")),
        ]),
    };
    IndexMetadata {
        format_version: 1,
        index_uuid,
        location: directory_uri(&root.join("metadata").join(index_uuid.to_string())),
        last_updated_ms: timestamp_ms,
        last_sequence_number: 1,
        current_snapshot_id: snapshot.snapshot_id,
        snapshots: vec![snapshot],
        snapshot_log: vec![SnapshotLogEntry {
            timestamp_ms,
            snapshot_id: 701,
        }],
        properties: BTreeMap::new(),
    }
}

pub(crate) fn refreshed(base: &IndexMetadata, snapshot_id: i64) -> IndexMetadata {
    let mut next = base.clone();
    let mut snapshot = base.snapshots[0].clone();
    snapshot.snapshot_id = snapshot_id;
    snapshot.sequence_number = 2;
    snapshot.timestamp_ms += 1;
    next.last_updated_ms += 1;
    next.last_sequence_number = 2;
    next.current_snapshot_id = snapshot_id;
    next.snapshots.push(snapshot);
    next.snapshot_log.push(SnapshotLogEntry {
        timestamp_ms: next.last_updated_ms,
        snapshot_id,
    });
    next
}

pub(crate) fn assert_index_catalog_contract(catalog: &dyn IndexCatalog, root: &Path) {
    let identifier = IndexIdentifier::new(vec!["analytics".into()], "documents").unwrap();
    let metadata = metadata(root);
    let location = file_uri(&root.join("v1.metadata.json"));

    assert!(matches!(
        catalog.list(identifier.namespace()),
        Err(Error::NamespaceNotFound(_))
    ));
    catalog.register(&identifier, &location, &metadata).unwrap();
    assert_eq!(
        catalog.list(identifier.namespace()).unwrap(),
        std::slice::from_ref(&identifier)
    );

    let loaded = catalog.load(&identifier).unwrap();
    assert_eq!(loaded.identifier, identifier);
    assert_eq!(loaded.metadata_location, location);

    let source = &metadata.current_snapshot().unwrap().source;
    let discovered = catalog
        .find_by_source(identifier.namespace(), source)
        .unwrap();
    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].identifier, identifier);

    let next = refreshed(&metadata, 702);
    let next_location = file_uri(&root.join("v2.metadata.json"));
    catalog
        .commit(&identifier, &location, &next_location, &metadata, &next)
        .unwrap();
    assert_eq!(
        catalog.load(&identifier).unwrap().metadata_location,
        next_location
    );

    catalog.drop(&identifier).unwrap();
    assert!(catalog.list(identifier.namespace()).unwrap().is_empty());
    assert!(matches!(
        catalog.load(&identifier),
        Err(Error::IndexNotFound(_))
    ));
}
