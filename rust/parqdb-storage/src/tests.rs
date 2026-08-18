use std::collections::HashMap;
use std::sync::Arc;

use bytes::Bytes;
use object_store::path::Path as ObjectPath;
use tempfile::TempDir;
use url::Url;

use crate::warehouse::object_parents_below;
use crate::{StorageRegistry, Warehouse};

#[test]
fn derives_locations_for_each_supported_scheme() {
    let cases = [
        ("file:///tmp/parqdb", "file:///tmp/parqdb/indexes/a/"),
        ("s3://bucket/parqdb", "s3://bucket/parqdb/indexes/a/"),
        (
            "hdfs://namenode:8020/parqdb",
            "hdfs://namenode:8020/parqdb/indexes/a/",
        ),
    ];
    for (root, expected) in cases {
        if root.starts_with("hdfs:") {
            let mut root = Url::parse(root).unwrap();
            if !root.path().ends_with('/') {
                root.set_path(&format!("{}/", root.path()));
            }
            let mut location = root.join("indexes/a").unwrap();
            location.set_path(&format!("{}/", location.path()));
            assert_eq!(location.as_str(), expected);
        } else if root.starts_with("s3:") {
            let mut root = Url::parse(root).unwrap();
            root.set_path("/parqdb/");
            let mut location = root.join("indexes/a").unwrap();
            location.set_path(&format!("{}/", location.path()));
            assert_eq!(location.as_str(), expected);
        } else {
            let warehouse = Warehouse::open(root, StorageRegistry::new(HashMap::new())).unwrap();
            assert_eq!(warehouse.location("indexes/a", true).unwrap(), expected);
        }
    }
}

#[test]
fn warehouse_relative_locations_preserve_directory_semantics() {
    let warehouse =
        Warehouse::open("file:///tmp/parqdb/", StorageRegistry::new(HashMap::new())).unwrap();

    assert_eq!(
        warehouse
            .relative_location("file:///tmp/parqdb/indexes/a/")
            .unwrap(),
        "indexes/a/"
    );
    assert_eq!(
        warehouse
            .relative_location("file:///tmp/parqdb/metadata/a.json")
            .unwrap(),
        "metadata/a.json"
    );
}

#[tokio::test]
async fn writes_reads_lists_and_deletes_local_objects() {
    let temporary = TempDir::new().unwrap();
    let root = Url::from_directory_path(temporary.path()).unwrap();
    let warehouse = Warehouse::open(root.as_str(), StorageRegistry::new(HashMap::new())).unwrap();
    let location = warehouse.location("metadata/id/v1.json", false).unwrap();
    warehouse
        .put_new(&location, Bytes::from_static(b"metadata"))
        .await
        .unwrap();

    assert_eq!(
        warehouse.read(&location).await.unwrap(),
        Bytes::from_static(b"metadata")
    );
    assert_eq!(warehouse.list("metadata").await.unwrap().len(), 1);
    assert!(warehouse.put_new(&location, Bytes::new()).await.is_err());
    warehouse.delete(&location).await.unwrap();
    assert!(warehouse.read(&location).await.is_err());
}

#[tokio::test]
async fn expands_nested_star_patterns_without_crossing_path_segments() {
    let temporary = TempDir::new().unwrap();
    for relative in [
        "documents/p0/data/part-0.parquet",
        "documents/p1/data/part-1.parquet",
        "documents/p1/other/part-2.parquet",
        "documents/p2/data/notes.txt",
    ] {
        let path = temporary.path().join(relative);
        std::fs::create_dir_all(path.parent().unwrap()).unwrap();
        std::fs::write(path, relative).unwrap();
    }
    let registry = StorageRegistry::default();
    let pattern = temporary
        .path()
        .join("documents/*/data/*.parquet")
        .to_string_lossy()
        .into_owned();
    let pattern = Url::from_file_path(pattern).unwrap().to_string();

    let matches = registry.expand(&pattern).await.unwrap();

    assert_eq!(matches.len(), 2);
    assert!(matches[0].ends_with("/documents/p0/data/part-0.parquet"));
    assert!(matches[1].ends_with("/documents/p1/data/part-1.parquet"));
}

#[tokio::test]
async fn reports_star_patterns_without_matches() {
    let temporary = TempDir::new().unwrap();
    let pattern = temporary
        .path()
        .join("documents/*/*.parquet")
        .to_string_lossy()
        .into_owned();
    let pattern = Url::from_file_path(pattern).unwrap().to_string();

    assert!(StorageRegistry::default().expand(&pattern).await.is_err());
}

#[test]
fn rejects_locations_outside_the_warehouse() {
    let warehouse =
        Warehouse::open("file:///tmp/parqdb/", StorageRegistry::new(HashMap::new())).unwrap();
    assert!(warehouse.managed("file:///tmp/other/file").is_err());
    assert!(warehouse.managed("s3://bucket/parqdb/file").is_err());
}

#[test]
fn rejects_unsupported_or_noncanonical_locations() {
    let registry = StorageRegistry::default();
    assert!(Warehouse::open("gs://bucket/parqdb", registry.clone()).is_err());
    assert!(Warehouse::open("file://host/tmp/parqdb", registry).is_err());
}

#[test]
fn scheme_specific_options_do_not_break_local_resolution() {
    let registry = StorageRegistry::new(HashMap::from([("aws_region".into(), "us-east-1".into())]));
    assert!(registry.resolve("file:///tmp/source.parquet").is_ok());
}

#[test]
fn caller_supplied_store_replaces_scheme_construction() {
    let temporary = TempDir::new().unwrap();
    let registry = StorageRegistry::default();
    let store = object_store::local::LocalFileSystem::new_with_prefix(temporary.path()).unwrap();

    assert!(
        registry
            .register_store("s3://fixture-bucket/", Arc::new(store))
            .unwrap()
            .is_none()
    );
    let resolved = registry
        .resolve("s3://fixture-bucket/path/to/object")
        .unwrap();
    assert_eq!(resolved.base_url().as_str(), "s3://fixture-bucket/");
    assert_eq!(resolved.path().as_ref(), "path/to/object");
}

#[test]
fn derives_only_object_parents_below_the_warehouse_root() {
    let root = ObjectPath::from("parqdb");
    let object = ObjectPath::from(
        "parqdb/indexes/0123456789abcdef0123456789abcdef/1/ivf_postings/part.parquet",
    );
    assert_eq!(
        object_parents_below(&root, &object),
        [
            "parqdb/indexes/0123456789abcdef0123456789abcdef/1/ivf_postings",
            "parqdb/indexes/0123456789abcdef0123456789abcdef/1",
            "parqdb/indexes/0123456789abcdef0123456789abcdef",
            "parqdb/indexes",
        ]
        .map(ObjectPath::from)
    );
    assert!(object_parents_below(&root, &ObjectPath::from("outside/file")).is_empty());
    assert!(object_parents_below(&root, &ObjectPath::from("parqdb/file")).is_empty());
}

#[test]
fn constructs_s3_and_hdfs_adapters_without_accessing_remote_storage() {
    let s3 = Warehouse::open(
        "s3://bucket/parqdb",
        StorageRegistry::new(HashMap::from([("aws_region".into(), "us-east-1".into())])),
    )
    .unwrap();
    assert_eq!(s3.root(), "s3://bucket/parqdb/");

    let hdfs = Warehouse::open("hdfs://namenode:8020/parqdb", StorageRegistry::default()).unwrap();
    assert_eq!(hdfs.root(), "hdfs://namenode:8020/parqdb/");
}
