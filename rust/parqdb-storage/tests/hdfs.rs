//! HDFS integration coverage for the public warehouse contract.

#![cfg(feature = "hdfs-integration")]

use bytes::Bytes;
use hdfs_native_object_store::minidfs::MiniDfs;
use parqdb_storage::{StorageRegistry, Warehouse};
use std::time::{SystemTime, UNIX_EPOCH};

#[tokio::test]
async fn hdfs_warehouse_round_trip() {
    let cluster = std::env::var("PARQDB_TEST_HDFS_URI")
        .is_err()
        .then(MiniDfs::default);
    let base = std::env::var("PARQDB_TEST_HDFS_URI")
        .unwrap_or_else(|_| cluster.as_ref().unwrap().url.clone());
    let suffix = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_nanos();
    let root = format!(
        "{}/parqdb-{}-{suffix}",
        base.trim_end_matches('/'),
        std::process::id()
    );
    let warehouse = Warehouse::open(&root, StorageRegistry::default()).unwrap();
    let location = warehouse
        .location("metadata/integration/v1.json", false)
        .unwrap();

    warehouse
        .put_new(&location, Bytes::from_static(b"hdfs-metadata"))
        .await
        .unwrap();

    assert_eq!(
        warehouse.read(&location).await.unwrap(),
        Bytes::from_static(b"hdfs-metadata")
    );
    assert_eq!(warehouse.head(&location).await.unwrap().size, 13);
    assert_eq!(warehouse.list("metadata").await.unwrap().len(), 1);
    assert!(
        warehouse
            .put_new(&location, Bytes::from_static(b"replacement"))
            .await
            .is_err()
    );

    warehouse.delete(&location).await.unwrap();
    assert!(warehouse.read(&location).await.is_err());
    assert!(warehouse.list("metadata").await.unwrap().is_empty());
}
