#![allow(clippy::cast_precision_loss)]

// Local IVF construction tests.

use arrow::array::{Array, Float32Builder, Int32Array, Int64Array, ListBuilder, StringArray};
use datafusion::prelude::SessionContext;
use tempfile::TempDir;

use super::*;
use crate::ivf::read_centroids;

fn source(ids: &[i64], tenants: &[&str], vectors: &[[f32; 2]]) -> RecordBatch {
    let mut embeddings = ListBuilder::new(Float32Builder::new()).with_field(Arc::new(Field::new(
        "element",
        DataType::Float32,
        false,
    )));
    for vector in vectors {
        embeddings.values().append_slice(vector);
        embeddings.append(true);
    }
    let embeddings = embeddings.finish();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("tenant", DataType::Utf8, false),
            Field::new("embedding", embeddings.data_type().clone(), false),
        ])),
        vec![
            Arc::new(Int64Array::from(ids.to_vec())),
            Arc::new(StringArray::from(tenants.to_vec())),
            Arc::new(embeddings),
        ],
    )
    .unwrap()
}

fn example_source() -> RecordBatch {
    source(
        &[10, 20, 30, 40],
        &["a", "a", "b", "b"],
        &[[0.0, 0.0], [0.1, 0.0], [10.0, 0.0], [10.1, 0.0]],
    )
}

#[test]
fn builds_ivf_relations_with_source_keys_in_postings() {
    let artifacts = build_ivf_tables(
        &example_source(),
        "embedding",
        &["tenant".into(), "id".into()],
        2,
    )
    .unwrap();

    assert_eq!(artifacts.dimension, 2);
    assert_eq!(artifacts.ntotal, 4);
    assert_eq!(artifacts.centroids.num_rows(), 2);
    assert_eq!(artifacts.postings.num_rows(), 4);
    assert_eq!(
        artifacts
            .centroids
            .schema()
            .fields()
            .iter()
            .map(|field| (field.name().as_str(), field.is_nullable()))
            .collect::<Vec<_>>(),
        [("cid", false), ("centroid", false)]
    );
    assert_eq!(
        artifacts
            .postings
            .schema()
            .fields()
            .iter()
            .map(|field| (field.name().as_str(), field.data_type().clone()))
            .collect::<Vec<_>>(),
        [
            ("cid", DataType::Int32),
            ("key_1", DataType::Utf8),
            ("key_2", DataType::Int64),
            (
                "vector",
                DataType::List(Arc::new(Field::new("element", DataType::Float32, false,))),
            ),
        ]
    );
    let cids = artifacts
        .postings
        .column(0)
        .as_any()
        .downcast_ref::<Int32Array>()
        .unwrap()
        .values();
    assert!(cids.windows(2).all(|pair| pair[0] <= pair[1]));
}

#[test]
fn build_is_deterministic() {
    let source = source(
        &(0..32).collect::<Vec<_>>(),
        &vec!["tenant"; 32],
        &(0..32)
            .map(|value| [value as f32, (value % 3) as f32])
            .collect::<Vec<_>>(),
    );
    let left = build_ivf_tables(&source, "embedding", &["id".into()], 4).unwrap();
    let right = build_ivf_tables(&source, "embedding", &["id".into()], 4).unwrap();

    assert_eq!(
        read_centroids(&left.centroids, 4, 2).unwrap(),
        read_centroids(&right.centroids, 4, 2).unwrap()
    );
    assert_eq!(left.postings, right.postings);
}

#[tokio::test]
async fn production_training_samples_across_the_complete_source() {
    let ids = (0..300).collect::<Vec<_>>();
    let tenants = vec!["tenant"; 300];
    let vectors = (0..300)
        .map(|row| {
            if row < 256 {
                [0.0, 0.0]
            } else {
                [100.0, 100.0]
            }
        })
        .collect::<Vec<_>>();
    let source = source(&ids, &tenants, &vectors);
    let context = SessionContext::new();
    context.register_batch("source", source).unwrap();
    let dataframe = context.table("source").await.unwrap();

    let trained = train_centroids(
        dataframe,
        "embedding",
        1,
        &ParalliteContext::default(),
        &LocalBuildProgress::default(),
    )
    .await
    .unwrap();

    assert!(trained.centroids[0] > 0.0);
    assert!((trained.centroids[0] - trained.centroids[1]).abs() < f32::EPSILON);
}

#[test]
fn rejects_invalid_build_requests_before_training() {
    let source = example_source();
    assert!(build_ivf_tables(&source, "", &["id".into()], 2).is_err());
    assert!(build_ivf_tables(&source, "embedding", &[], 2).is_err());
    assert!(build_ivf_tables(&source, "embedding", &["id".into(), "id".into()], 2).is_err());
    assert!(build_ivf_tables(&source, "embedding", &["id".into()], 0).is_err());
    assert!(build_ivf_tables(&source, "embedding", &["id".into()], 5).is_err());
    assert!(build_ivf_tables(&source, "missing", &["id".into()], 2).is_err());
    assert!(build_ivf_tables(&source, "embedding", &["missing".into()], 2).is_err());
}

#[test]
fn accepts_duplicate_composite_source_keys_as_a_source_contract() {
    let source = source(
        &[10, 10, 20],
        &["a", "a", "a"],
        &[[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]],
    );
    let artifacts =
        build_ivf_tables(&source, "embedding", &["tenant".into(), "id".into()], 2).unwrap();

    assert_eq!(artifacts.ntotal, 3);
    assert_eq!(artifacts.postings.num_rows(), 3);
}

#[test]
fn rejects_non_finite_vectors() {
    let source = source(&[1, 2], &["a", "a"], &[[0.0, f32::NAN], [1.0, 2.0]]);
    assert!(build_ivf_tables(&source, "embedding", &["id".into()], 1).is_err());
}

#[test]
fn source_vector_and_key_schema_may_be_nullable() {
    let source = example_source();
    for nullable_field in ["id", "embedding"] {
        let schema = Arc::new(Schema::new(
            source
                .schema()
                .fields()
                .iter()
                .map(|field| {
                    Field::new(
                        field.name(),
                        field.data_type().clone(),
                        field.name() == nullable_field,
                    )
                })
                .collect::<Vec<_>>(),
        ));
        let nullable = RecordBatch::try_new(schema, source.columns().to_vec()).unwrap();

        assert!(build_ivf_tables(&nullable, "embedding", &["id".into()], 1).is_ok());
    }
}

#[test]
fn signed_and_string_keys_use_the_same_postings_schema() {
    let source = source(&[-1, 20], &["a", "b"], &[[0.0, 0.0], [10.0, 0.0]]);
    let artifacts =
        build_ivf_tables(&source, "embedding", &["tenant".into(), "id".into()], 2).unwrap();

    assert_eq!(artifacts.postings.num_columns(), 4);
    assert_eq!(artifacts.postings.schema().field(1).name(), "key_1");
    assert_eq!(artifacts.postings.schema().field(2).name(), "key_2");
    assert_eq!(artifacts.postings.schema().field(3).name(), "vector");
}

#[test]
fn vector_storage_can_be_disabled() {
    let artifacts = build_ivf_tables_with_vector_storage(
        &example_source(),
        "embedding",
        &["id".into()],
        2,
        false,
    )
    .unwrap();

    assert!(!artifacts.store_vectors);
    assert_eq!(
        artifacts
            .postings
            .schema()
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        ["cid", "key_1"]
    );
}

#[tokio::test]
async fn writes_only_the_two_ivf_relations() {
    let temporary = TempDir::new().unwrap();
    let build = build_ivf(
        &example_source(),
        "embedding",
        &["tenant".into(), "id".into()],
        2,
        temporary.path(),
    )
    .await
    .unwrap();

    assert_eq!(build.parameters["dimension"], "2");
    assert_eq!(build.parameters["nlist"], "2");
    assert_eq!(build.parameters["ntotal"], "4");
    assert_eq!(build.parameters["store_vectors"], "true");
    assert_eq!(
        build
            .index_relations
            .keys()
            .map(String::as_str)
            .collect::<Vec<_>>(),
        ["ivf_centroids", "ivf_postings"]
    );
    for reference in build.index_relations.values() {
        let RelationReference::Parquet { uri } = reference else {
            panic!("local builder must return Parquet references");
        };
        assert!(
            crate::local_uri::file_uri_to_path(uri)
                .unwrap()
                .join("part-00000.parquet")
                .is_file()
        );
    }
}

#[test]
fn writer_count_respects_target_size_and_explicit_partitions() {
    let source = example_source();
    let keys = ["tenant".into(), "id".into()];
    let row_width =
        estimate_posting_row_width(source.schema().as_ref(), &keys, 2, PostingEncoding::Flat);
    let one_file = ParquetWriterOptions {
        target_file_size: usize::MAX,
        ..ParquetWriterOptions::default()
    };
    assert_eq!(writer_count(&one_file, None, 1_000, row_width, 8), 1);

    let automatic = ParquetWriterOptions {
        target_file_size: 1,
        ..ParquetWriterOptions::default()
    };
    assert_eq!(writer_count(&automatic, None, 1_000, row_width, 3), 3);
    assert_eq!(writer_count(&automatic, Some(3), 1_000, row_width, 3), 3);
}

#[test]
fn postings_row_groups_follow_cluster_cardinality_with_safe_bounds() {
    let automatic = ParquetWriterOptions::default();
    assert_eq!(
        resolved_postings_writer_options(&automatic, 1_000_000, 256).max_row_group_rows,
        Some(8_192)
    );
    assert_eq!(
        resolved_postings_writer_options(&automatic, 2_000_000, 256).max_row_group_rows,
        Some(8_192)
    );
    assert_eq!(
        resolved_postings_writer_options(&automatic, 100, 100).max_row_group_rows,
        Some(MIN_AUTO_ROW_GROUP_ROWS)
    );
    assert_eq!(
        resolved_postings_writer_options(&automatic, 100_000_000, 256).max_row_group_rows,
        Some(MAX_AUTO_ROW_GROUP_ROWS)
    );
}

#[test]
fn explicit_postings_row_group_size_overrides_automatic_size() {
    let explicit = ParquetWriterOptions {
        max_row_group_rows: Some(16_384),
        ..ParquetWriterOptions::default()
    };
    assert_eq!(
        resolved_postings_writer_options(&explicit, 1_000_000, 256).max_row_group_rows,
        Some(16_384)
    );
}
