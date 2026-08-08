//! Integration tests for Relify metadata validation.

use std::collections::BTreeMap;
use std::sync::Arc;

use relify_meta::{
    Error, IndexFamily, IndexFamilyRegistry, IndexMetadata, IndexSnapshot, PostingEncoding,
    RelationReference, SnapshotLogEntry,
};
use uuid::Uuid;

fn parquet(uri: &str) -> RelationReference {
    RelationReference::Parquet {
        uri: uri.to_owned(),
    }
}

fn valid_snapshot() -> IndexSnapshot {
    IndexSnapshot {
        snapshot_id: 701,
        sequence_number: 1,
        timestamp_ms: 1_750_000_000_000,
        summary: BTreeMap::from([("operation".into(), "create".into())]),
        source: parquet("file:///tmp/source.parquet"),
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
            (
                "ivf_centroids".into(),
                parquet("file:///tmp/centroids.parquet"),
            ),
            (
                "ivf_postings".into(),
                parquet("file:///tmp/postings.parquet"),
            ),
        ]),
    }
}

fn valid_metadata() -> IndexMetadata {
    IndexMetadata {
        format_version: 1,
        index_uuid: Uuid::parse_str("2f1c7f5e-3c43-4a44-8f2a-cf560c4db8d1").unwrap(),
        location: "file:///tmp/relify/index".into(),
        last_updated_ms: 1_750_000_000_000,
        last_sequence_number: 1,
        current_snapshot_id: 701,
        snapshots: vec![valid_snapshot()],
        snapshot_log: vec![SnapshotLogEntry {
            timestamp_ms: 1_750_000_000_000,
            snapshot_id: 701,
        }],
        properties: BTreeMap::new(),
    }
}

fn valid_update(base: &IndexMetadata) -> IndexMetadata {
    let mut update = base.clone();
    let mut snapshot = base.snapshots[0].clone();
    snapshot.snapshot_id = 702;
    snapshot.sequence_number = 2;
    snapshot.timestamp_ms += 1;
    update.last_updated_ms += 1;
    update.last_sequence_number = 2;
    update.current_snapshot_id = snapshot.snapshot_id;
    update.snapshots.push(snapshot);
    update.snapshot_log.push(SnapshotLogEntry {
        timestamp_ms: update.last_updated_ms,
        snapshot_id: update.current_snapshot_id,
    });
    update
}

#[test]
fn accepts_valid_metadata_round_trip() {
    let metadata = valid_metadata();
    metadata.validate().unwrap();

    let json = serde_json::to_vec(&metadata).unwrap();
    assert_eq!(IndexMetadata::from_json_slice(&json).unwrap(), metadata);
}

#[test]
fn accepts_both_vector_storage_modes() {
    for value in ["true", "false"] {
        let mut metadata = valid_metadata();
        metadata.snapshots[0]
            .parameters
            .insert("store_vectors".into(), value.into());

        metadata.validate().unwrap();
        assert_eq!(
            metadata.snapshots[0]
                .parameter_bool("store_vectors")
                .unwrap(),
            value == "true"
        );
    }
}

#[test]
fn accepts_ivf_v2_posting_encodings() {
    for encoding in ["source", "flat", "lvq4", "lvq8"] {
        let mut metadata = valid_metadata();
        metadata.snapshots[0].index_schema_version = 2;
        metadata.snapshots[0].parameters.remove("store_vectors");
        metadata.snapshots[0]
            .parameters
            .insert("posting_encoding".into(), encoding.into());

        metadata.validate().unwrap();
        assert_eq!(
            PostingEncoding::from_snapshot(&metadata.snapshots[0])
                .unwrap()
                .as_str(),
            encoding
        );
    }
}

struct TestFamily;

impl IndexFamily for TestFamily {
    fn name(&self) -> &'static str {
        "test_family"
    }

    fn validate(&self, snapshot: &IndexSnapshot) -> relify_meta::Result<()> {
        if snapshot.index_schema_version == 7 && snapshot.metric == "test_metric" {
            Ok(())
        } else {
            Err(Error::new("invalid test-family snapshot"))
        }
    }
}

#[test]
fn validates_registered_index_family_without_changing_metadata_core() {
    let mut metadata = valid_metadata();
    metadata.snapshots[0].index_family = "test_family".into();
    metadata.snapshots[0].index_schema_version = 7;
    metadata.snapshots[0].metric = "test_metric".into();

    assert!(metadata.validate().is_err());

    let mut registry = IndexFamilyRegistry::new();
    registry.register(Arc::new(TestFamily)).unwrap();
    metadata.validate_with_registry(&registry).unwrap();

    let json = serde_json::to_vec(&metadata).unwrap();
    assert_eq!(
        IndexMetadata::from_json_slice_with_registry(&json, &registry).unwrap(),
        metadata
    );
}

#[test]
fn rejects_duplicate_and_noncanonical_family_registrations() {
    struct InvalidFamily;

    impl IndexFamily for InvalidFamily {
        fn name(&self) -> &'static str {
            "Invalid-Family"
        }

        fn validate(&self, _snapshot: &IndexSnapshot) -> relify_meta::Result<()> {
            Ok(())
        }
    }

    let mut registry = IndexFamilyRegistry::new();
    registry.register(Arc::new(TestFamily)).unwrap();
    assert!(registry.register(Arc::new(TestFamily)).is_err());
    assert!(registry.register(Arc::new(InvalidFamily)).is_err());
}

#[test]
fn rejects_parameters_from_another_ivf_schema_version() {
    let mut metadata = valid_metadata();
    metadata.snapshots[0].index_schema_version = 2;

    assert!(metadata.validate().is_err());

    metadata.snapshots[0].parameters.remove("store_vectors");
    metadata.snapshots[0]
        .parameters
        .insert("posting_encoding".into(), "unknown".into());
    assert!(metadata.validate().is_err());
}

#[test]
fn accepts_valid_metadata_update() {
    let base = valid_metadata();
    valid_update(&base).validate_update_from(&base).unwrap();
}

#[test]
fn rejects_mutating_a_retained_snapshot() {
    let base = valid_metadata();
    let mut update = valid_update(&base);
    update.snapshots[0]
        .summary
        .insert("rewritten".into(), "true".into());

    assert!(update.validate().is_ok());
    assert!(update.validate_update_from(&base).is_err());
}

#[test]
fn rejects_replacing_logical_identity_with_the_same_uuid() {
    let base = valid_metadata();
    let mut update = valid_update(&base);
    update.snapshots.remove(0);
    update.snapshots[0].source = parquet("file:///tmp/other-source.parquet");
    update.snapshot_log = vec![SnapshotLogEntry {
        timestamp_ms: update.last_updated_ms,
        snapshot_id: update.current_snapshot_id,
    }];

    assert!(update.validate().is_ok());
    assert!(update.validate_update_from(&base).is_err());
}

#[test]
fn rejects_skipping_a_snapshot_sequence_number() {
    let base = valid_metadata();
    let mut update = valid_update(&base);
    update.snapshots[1].sequence_number = 3;
    update.last_sequence_number = 3;

    assert!(update.validate().is_ok());
    assert!(update.validate_update_from(&base).is_err());
}

#[test]
fn accepts_rolling_back_to_a_retained_snapshot() {
    let original = valid_metadata();
    let base = valid_update(&original);
    let mut update = base.clone();
    update.last_updated_ms += 1;
    update.current_snapshot_id = original.current_snapshot_id;
    update.snapshot_log.push(SnapshotLogEntry {
        timestamp_ms: update.last_updated_ms,
        snapshot_id: update.current_snapshot_id,
    });

    assert!(update.validate().is_ok());
    update.validate_update_from(&base).unwrap();
}

#[test]
fn rejects_unknown_ivf_relation_role() {
    let mut metadata = valid_metadata();
    metadata.snapshots[0]
        .index_relations
        .insert("future_role".into(), parquet("file:///tmp/future.parquet"));

    assert!(metadata.validate().is_err());
}

#[test]
fn rejects_unknown_ivf_parameter() {
    let mut metadata = valid_metadata();
    metadata.snapshots[0]
        .parameters
        .insert("future-option".into(), "1".into());

    assert!(metadata.validate().is_err());
}

#[test]
fn rejects_noncanonical_store_vectors_parameter() {
    for value in ["True", "1", "yes", ""] {
        let mut metadata = valid_metadata();
        metadata.snapshots[0]
            .parameters
            .insert("store_vectors".into(), value.into());

        assert!(metadata.validate().is_err(), "{value:?} must be rejected");
    }
}

#[test]
fn rejects_identity_changes_across_snapshots() {
    let mut metadata = valid_metadata();
    let mut second = metadata.snapshots[0].clone();
    second.snapshot_id = 702;
    second.sequence_number = 2;
    second.vector_field = "other_embedding".into();
    metadata.snapshots.push(second);
    metadata.last_sequence_number = 2;
    metadata.current_snapshot_id = 702;
    metadata.snapshot_log.push(SnapshotLogEntry {
        timestamp_ms: metadata.last_updated_ms,
        snapshot_id: 702,
    });

    assert!(metadata.validate().is_err());
}

#[test]
fn rejects_snapshot_log_later_than_metadata() {
    let mut metadata = valid_metadata();
    metadata.snapshot_log[0].timestamp_ms = metadata.last_updated_ms + 1;

    assert!(metadata.validate().is_err());
}

#[test]
fn rejects_negative_snapshot_log_timestamp() {
    let mut metadata = valid_metadata();
    metadata.snapshot_log[0].timestamp_ms = -1;

    assert!(metadata.validate().is_err());
}

#[test]
fn rejects_empty_map_keys() {
    let mut metadata = valid_metadata();
    metadata.properties.insert(String::new(), "value".into());

    assert!(metadata.validate().is_err());
}

#[test]
fn rejects_duplicate_json_map_keys() {
    let json = serde_json::to_string(&valid_metadata()).unwrap();
    let duplicate = json.replacen(
        "\"dimension\":\"2\"",
        "\"dimension\":\"2\",\"dimension\":\"3\"",
        1,
    );
    assert_ne!(duplicate, json);

    assert!(IndexMetadata::from_json_slice(duplicate.as_bytes()).is_err());
}

#[test]
fn rejects_non_lowercase_uuid_json() {
    let json = serde_json::to_string(&valid_metadata()).unwrap();
    let uppercase = json.replacen(
        "2f1c7f5e-3c43-4a44-8f2a-cf560c4db8d1",
        "2F1C7F5E-3C43-4A44-8F2A-CF560C4DB8D1",
        1,
    );

    assert!(IndexMetadata::from_json_slice(uppercase.as_bytes()).is_err());
}

#[test]
fn rejects_unknown_relation_reference_fields() {
    let json = r#"{
        "profile": "parquet",
        "uri": "file:///tmp/source.parquet",
        "unexpected": "value"
    }"#;

    assert!(serde_json::from_str::<RelationReference>(json).is_err());
}

#[test]
fn enforces_canonical_parquet_uris() {
    let invalid = [
        "S3://bucket/index",
        "s3://Bucket/index",
        "s3://bucket/a//b",
        "s3://bucket/a/./b",
        "s3://bucket/a/../b",
        "s3://bucket/a%2fb",
        "s3://bucket/%41",
        "s3://user@bucket/index",
        "s3://bucket/index?version=1",
        "s3://bucket/index#fragment",
    ];

    for uri in invalid {
        assert!(parquet(uri).validate().is_err(), "{uri} must be rejected");
    }

    for uri in [
        "file:///tmp/index.parquet",
        "file:///tmp/partition=*/part-*.parquet",
        "s3://bucket/index",
        "s3://bucket/partition=*/part-*.parquet",
        "s3://bucket/path%20with%20spaces",
    ] {
        parquet(uri).validate().unwrap();
    }
}
