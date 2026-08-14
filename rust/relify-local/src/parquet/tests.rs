#![allow(clippy::cast_precision_loss)]

// Managed Parquet relation tests.

use std::io::Cursor;
use std::sync::Arc;

use arrow::array::{
    Array, Float32Array, Float32Builder, Int32Array, Int64Array, ListBuilder, StringArray,
};
use arrow::datatypes::{DataType, Field, Schema};
use bytes::Bytes;
use datafusion::prelude::{col, lit};
use object_store::PutMode;
use parquet::arrow::ArrowWriter;
use relify_storage::StorageRegistry;
use tempfile::TempDir;
use url::Url;

use super::*;

fn batch(ids: &[i64]) -> RecordBatch {
    let scores = Float32Array::from_iter_values(ids.iter().map(|id| *id as f32 * 0.5));
    let mut vectors = ListBuilder::new(Float32Builder::new()).with_field(Arc::new(Field::new(
        "element",
        DataType::Float32,
        false,
    )));
    for id in ids {
        vectors.values().append_slice(&[*id as f32, 1.0]);
        vectors.append(true);
    }
    let vectors = vectors.finish();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("score", DataType::Float32, false),
            Field::new("embedding", vectors.data_type().clone(), false),
        ])),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(scores),
            Arc::new(vectors),
        ],
    )
    .unwrap()
}

fn postings() -> RecordBatch {
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("cid", DataType::Int32, false),
            Field::new("key_1", DataType::Int64, false),
        ])),
        vec![
            Arc::new(Int32Array::from(vec![0, 0, 1, 1])),
            Arc::new(Int64Array::from(vec![10, 20, 30, 40])),
        ],
    )
    .unwrap()
}

fn relation(temporary: &TempDir, name: &str) -> (ParquetStore, String) {
    let registry = StorageRegistry::default();
    let root = Url::from_directory_path(temporary.path()).unwrap();
    (
        ParquetStore::new(registry),
        child_location(root.as_str(), name, true).unwrap(),
    )
}

async fn put_part(store: &ParquetStore, relation: &str, name: &str, batch: &RecordBatch) {
    let location = child_location(relation, name, false).unwrap();
    put_parquet_file(store, &location, batch).await;
}

async fn put_parquet_file(store: &ParquetStore, location: &str, batch: &RecordBatch) {
    let mut bytes = Vec::new();
    {
        let mut writer = ArrowWriter::try_new(
            Cursor::new(&mut bytes),
            batch.schema(),
            Some(ParquetWriterOptions::default().writer_properties().unwrap()),
        )
        .unwrap();
        writer.write(batch).unwrap();
        writer.close().unwrap();
    }
    let resolved = store.registry().resolve(location).unwrap();
    resolved
        .store()
        .put_opts(
            resolved.path(),
            Bytes::from(bytes).into(),
            PutMode::Create.into(),
        )
        .await
        .unwrap();
}

#[test]
fn writer_options_are_explicit_and_validated() {
    let defaults = ParquetWriterOptions::default();
    assert_eq!(defaults.compression, "uncompressed");
    assert_eq!(defaults.max_row_group_rows, None);
    assert_eq!(defaults.target_file_size, 512 * 1024 * 1024);
    assert_eq!(defaults.write_batch_rows, 8_192);
    assert!(defaults.validate().is_ok());

    let invalid_compression = ParquetWriterOptions {
        compression: "invalid".into(),
        ..defaults.clone()
    };
    assert!(invalid_compression.validate().is_err());
    let invalid_size = ParquetWriterOptions {
        target_file_size: 0,
        ..defaults
    };
    assert!(invalid_size.validate().is_err());
}

#[tokio::test]
async fn round_trip_preserves_schema_and_values() {
    let temporary = TempDir::new().unwrap();
    let (store, location) = relation(&temporary, "relation");
    let expected = batch(&[1, 2, 3]);

    store.write(&location, &expected).await.unwrap();
    assert_eq!(store.schema(&location).await.unwrap(), expected.schema());

    let actual = store.read(&location, None).await.unwrap();
    assert_eq!(actual.schema(), expected.schema());
    assert_eq!(actual.num_rows(), 3);
    assert_eq!(
        actual
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[1, 2, 3]
    );
}

#[tokio::test]
async fn projection_preserves_requested_order() {
    let temporary = TempDir::new().unwrap();
    let (store, location) = relation(&temporary, "relation");
    store.write(&location, &batch(&[1, 2])).await.unwrap();

    let projected = store.read(&location, Some(&["score", "id"])).await.unwrap();
    assert_eq!(
        projected
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        ["score", "id"]
    );
    assert!(store.read(&location, Some(&["missing"])).await.is_err());
}

#[tokio::test]
async fn hive_writer_allows_postings_without_vectors() {
    let temporary = TempDir::new().unwrap();
    let (store, location) = relation(&temporary, "postings");
    let expected = postings();
    let context = store.context();
    context
        .register_batch("postings", expected.clone())
        .unwrap();
    let dataframe = context.table("postings").await.unwrap();

    store
        .write_hive_cid_dataframe(&location, dataframe, 1, &ParquetWriterOptions::default())
        .await
        .unwrap();

    let dataframe = store
        .partitioned_dataframe(&location, vec![("cid".into(), DataType::Int32)])
        .await
        .unwrap();
    let schema = Arc::clone(dataframe.schema().inner());
    let actual = concat_batches(&schema, &dataframe.collect().await.unwrap()).unwrap();
    assert_eq!(actual.num_rows(), expected.num_rows());
    assert_eq!(
        actual
            .schema()
            .field_with_name("key_1")
            .unwrap()
            .data_type(),
        &DataType::Int64
    );
    assert_eq!(
        actual.schema().field_with_name("cid").unwrap().data_type(),
        &DataType::Int32
    );
}

#[tokio::test]
async fn uniform_provider_infers_schema_from_one_file() {
    let temporary = TempDir::new().unwrap();
    let (store, location) = relation(&temporary, "uniform");
    let expected = batch(&[1, 2]);
    put_part(&store, &location, "a.parquet", &expected).await;

    let corrupt_location = child_location(&location, "z.parquet", false).unwrap();
    let resolved = store.registry().resolve(&corrupt_location).unwrap();
    resolved
        .store()
        .put_opts(
            resolved.path(),
            Bytes::from_static(b"not parquet").into(),
            PutMode::Create.into(),
        )
        .await
        .unwrap();

    let provider = store
        .uniform_dataset_provider(&location, Vec::new())
        .await
        .unwrap();

    assert_eq!(provider.schema(), expected.schema());
}

#[tokio::test]
async fn uniform_provider_accepts_a_single_file_location() {
    let temporary = TempDir::new().unwrap();
    let store = ParquetStore::new(StorageRegistry::default());
    let root = Url::from_directory_path(temporary.path()).unwrap();
    let location = child_location(root.as_str(), "single.parquet", false).unwrap();
    let expected = batch(&[1, 2]);
    put_parquet_file(&store, &location, &expected).await;

    let provider = store
        .uniform_dataset_provider(&location, Vec::new())
        .await
        .unwrap();

    assert_eq!(provider.schema(), expected.schema());
}

#[tokio::test]
async fn cluster_filter_prunes_unrelated_row_groups() {
    let temporary = TempDir::new().unwrap();
    let (store, location) = relation(&temporary, "postings");
    let options = ParquetWriterOptions {
        max_row_group_rows: Some(2),
        ..ParquetWriterOptions::default()
    };
    store
        .write_batch(&location, &postings(), &options)
        .await
        .unwrap();

    let filtered = store.read_clusters(&location, &[1]).await.unwrap();
    assert_eq!(filtered.num_rows(), 2);
    let explain = store
        .dataframe(&location)
        .await
        .unwrap()
        .filter(col("cid").eq(lit(1_i32)))
        .unwrap()
        .explain(true, true)
        .unwrap()
        .collect()
        .await
        .unwrap();
    let plans = explain
        .iter()
        .flat_map(|batch| {
            batch
                .column(1)
                .as_any()
                .downcast_ref::<StringArray>()
                .unwrap()
                .iter()
                .flatten()
        })
        .collect::<Vec<_>>()
        .join("\n");
    assert!(plans.contains("row_groups_pruned_statistics"), "{plans}");
    assert!(plans.contains("2 total → 1 matched"), "{plans}");
}

#[tokio::test]
async fn reads_a_multi_file_relation() {
    let temporary = TempDir::new().unwrap();
    let (store, location) = relation(&temporary, "relation");
    put_part(&store, &location, "part-00000.parquet", &batch(&[1, 2])).await;
    put_part(&store, &location, "part-00001.parquet", &batch(&[3, 4])).await;

    let actual = store.read(&location, Some(&["id"])).await.unwrap();
    let mut ids = actual
        .column(0)
        .as_any()
        .downcast_ref::<Int64Array>()
        .unwrap()
        .values()
        .to_vec();
    ids.sort_unstable();
    assert_eq!(ids, [1, 2, 3, 4]);
}

#[tokio::test]
async fn preserves_an_empty_relation_schema() {
    let temporary = TempDir::new().unwrap();
    let (store, location) = relation(&temporary, "relation");
    let expected = batch(&[]);
    store.write(&location, &expected).await.unwrap();

    let actual = store.read(&location, None).await.unwrap();
    assert_eq!(actual.schema(), expected.schema());
    assert_eq!(actual.num_rows(), 0);
}

#[tokio::test]
async fn rejects_empty_and_corrupt_relation_roots() {
    let temporary = TempDir::new().unwrap();
    let (store, empty) = relation(&temporary, "empty");
    assert!(store.read(&empty, None).await.is_err());

    let corrupt = child_location(
        Url::from_directory_path(temporary.path()).unwrap().as_str(),
        "corrupt",
        true,
    )
    .unwrap();
    let resolved = store
        .registry()
        .resolve(&child_location(&corrupt, "part-00000.parquet", false).unwrap())
        .unwrap();
    resolved
        .store()
        .put_opts(
            resolved.path(),
            Bytes::from_static(b"not parquet").into(),
            PutMode::Create.into(),
        )
        .await
        .unwrap();
    assert!(store.read(&corrupt, None).await.is_err());
}

#[tokio::test]
async fn relation_writes_are_immutable() {
    let temporary = TempDir::new().unwrap();
    let (store, location) = relation(&temporary, "relation");
    store.write(&location, &batch(&[1, 2])).await.unwrap();
    let before = store.read(&location, None).await.unwrap();

    assert!(store.write(&location, &batch(&[3, 4])).await.is_err());
    assert_eq!(store.read(&location, None).await.unwrap(), before);
}
