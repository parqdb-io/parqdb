//! Executes the portable catalog operation fixture against `SQLite`.

use std::collections::BTreeMap;
use std::fs;
use std::path::{Path, PathBuf};

use relify_catalog::{Error, IndexCatalog, IndexIdentifier, Result, SqliteCatalog};
use relify_meta::IndexMetadata;
use serde_json::Value;
use tempfile::TempDir;

fn fixture_root() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../../spec/fixtures/v1")
        .canonicalize()
        .unwrap()
}

fn load_metadata(root: &Path, trace: &Value) -> BTreeMap<String, IndexMetadata> {
    trace["metadata"]
        .as_object()
        .unwrap()
        .iter()
        .map(|(name, relative_path)| {
            let bytes = fs::read(root.join(relative_path.as_str().unwrap())).unwrap();
            (
                name.clone(),
                IndexMetadata::from_json_slice(&bytes).unwrap(),
            )
        })
        .collect()
}

fn error_code(error: &Error) -> &'static str {
    match error {
        Error::IndexNotFound(_) => "INDEX_NOT_FOUND",
        Error::AlreadyExists(_) => "ALREADY_EXISTS",
        Error::CommitConflict(_) => "COMMIT_CONFLICT",
        other => panic!("unexpected catalog fixture error: {other}"),
    }
}

#[test]
fn sqlite_catalog_matches_the_portable_operation_trace() {
    let root = fixture_root();
    let trace: Value =
        serde_json::from_slice(&fs::read(root.join("catalog.json")).unwrap()).unwrap();
    let namespace = trace["identifier"]["namespace"]
        .as_array()
        .unwrap()
        .iter()
        .map(|segment| segment.as_str().unwrap().to_owned())
        .collect();
    let identifier =
        IndexIdentifier::new(namespace, trace["identifier"]["name"].as_str().unwrap()).unwrap();
    let metadata = load_metadata(&root, &trace);
    let locations = trace["locations"].as_object().unwrap();
    let location = |name: &str| locations[name].as_str().unwrap();

    let temporary = TempDir::new().unwrap();
    let catalog = SqliteCatalog::open(temporary.path().join("catalog.sqlite")).unwrap();

    for operation in trace["operations"].as_array().unwrap() {
        let result: Result<Option<String>> = match operation["operation"].as_str().unwrap() {
            "load" => catalog
                .load(&identifier)
                .map(|entry| Some(entry.metadata_location)),
            "register" => {
                let metadata_name = operation["metadata"].as_str().unwrap();
                let location_name = operation["location"].as_str().unwrap();
                catalog
                    .register(
                        &identifier,
                        location(location_name),
                        &metadata[metadata_name],
                    )
                    .map(|()| None)
            }
            "commit" => {
                let base_metadata = operation["base-metadata"].as_str().unwrap();
                let base_location = operation["base-location"].as_str().unwrap();
                let new_metadata = operation["new-metadata"].as_str().unwrap();
                let new_location = operation["new-location"].as_str().unwrap();
                catalog
                    .commit(
                        &identifier,
                        location(base_location),
                        location(new_location),
                        &metadata[base_metadata],
                        &metadata[new_metadata],
                    )
                    .map(|()| None)
            }
            "drop" => catalog.drop(&identifier).map(|()| None),
            other => panic!("unknown catalog fixture operation: {other}"),
        };

        match operation["expect"].as_str().unwrap() {
            "OK" => {
                let output = result.unwrap();
                if let Some(expected_location) = operation.get("expect-location") {
                    assert_eq!(
                        output.as_deref(),
                        Some(location(expected_location.as_str().unwrap()))
                    );
                }
            }
            expected_error => {
                let error = result.unwrap_err();
                assert_eq!(error_code(&error), expected_error);
            }
        }
    }
}
