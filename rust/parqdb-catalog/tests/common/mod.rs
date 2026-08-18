use std::collections::BTreeMap;
use std::path::Path;

use parqdb_catalog::{Error, IndexCatalog, IndexIdentifier};
use parqdb_meta::{IndexMetadata, IndexSnapshot, RelationReference, SnapshotLogEntry};
use url::Url;
use uuid::Uuid;

pub(crate) fn file_uri(path: &Path) -> String {
    Url::from_file_path(path).unwrap().into()
}

pub(crate) fn source(root: &Path) -> RelationReference {
    RelationReference::Parquet {
        uri: file_uri(&root.join("source.parquet")),
    }
}

pub(crate) fn metadata(_root: &Path) -> IndexMetadata {
    let index_uuid = Uuid::parse_str("2f1c7f5e-3c43-4a44-8f2a-cf560c4db8d1").unwrap();
    let timestamp_ms = 1_750_000_000_000;
    let snapshot = IndexSnapshot {
        snapshot_id: 701,
        sequence_number: 1,
        timestamp_ms,
        summary: BTreeMap::new(),
        vector_field: "embedding".into(),
        source_key_fields: vec!["document_id".into()],
        indexed_rows: 4,
        index_family: "ivf".into(),
        index_schema_version: 1,
        metric: "l2_squared".into(),
        parameters: BTreeMap::from([
            ("dimension".into(), "2".into()),
            ("nlist".into(), "2".into()),
            ("ntotal".into(), "4".into()),
            ("posting_encoding".into(), "source".into()),
            (
                "ivf_centroids_fingerprint".into(),
                "73a6be1d-5c50-4f9f-a70b-035ca68b105d".into(),
            ),
            (
                "ivf_centroids_uuid".into(),
                "fe985f6d-3592-4385-a1ca-71347057a210".into(),
            ),
            (
                "ivf_centroids_metadata_location".into(),
                "centroid-artifacts/metadata.json".into(),
            ),
        ]),
        index_relations: BTreeMap::from([
            ("ivf_centroids".into(), "centroids/".into()),
            ("ivf_postings".into(), "postings/".into()),
        ]),
    };
    IndexMetadata {
        format_version: 1,
        index_uuid,
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
    let source = source(root);
    let location = file_uri(&root.join("v1.metadata.json"));

    assert!(matches!(
        catalog.list(identifier.namespace()),
        Err(Error::NamespaceNotFound(_))
    ));
    catalog
        .register(&identifier, &source, &location, &metadata)
        .unwrap();
    assert_eq!(
        catalog.list(identifier.namespace()).unwrap(),
        std::slice::from_ref(&identifier)
    );

    let loaded = catalog.load(&identifier).unwrap();
    assert_eq!(loaded.identifier, identifier);
    assert_eq!(loaded.metadata_location, location);

    let discovered = catalog
        .find_by_source(identifier.namespace(), &source)
        .unwrap();
    assert_eq!(discovered.len(), 1);
    assert_eq!(discovered[0].identifier, identifier);

    let next = refreshed(&metadata, 702);
    let next_location = file_uri(&root.join("v2.metadata.json"));
    catalog
        .commit(
            &identifier,
            &source,
            &location,
            &next_location,
            &metadata,
            &next,
        )
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
