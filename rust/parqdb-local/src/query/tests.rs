#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

// DataFusion query integration tests.

use std::collections::BTreeMap;
use std::fs::{copy, create_dir_all};
use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, Float32Array, Float32Builder, Int64Array, ListBuilder, StringArray};
use arrow::compute::concat_batches;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::util::display::array_value_to_string;
use datafusion::common::ScalarValue;
use datafusion::logical_expr::registry::FunctionRegistry;
use datafusion::prelude::{Expr, SessionContext, col};
use object_store::local::LocalFileSystem;
use parqdb_meta::{DistanceMetric, IndexMetadata, PostingEncoding};
use parqdb_storage::StorageRegistry;
use serde_json::Value;
use tempfile::TempDir;

use super::*;
use crate::{ClusterSelection, ResolvedSearch};

fn query_literal(query: &[f32]) -> Expr {
    let values = query
        .iter()
        .copied()
        .map(|value| ScalarValue::Float32(Some(value)))
        .collect::<Vec<_>>();
    Expr::Literal(
        ScalarValue::List(ScalarValue::new_list(&values, &DataType::Float32, false)),
        None,
    )
}
use crate::builder::{IvfTables, build_ivf_tables};
use crate::parquet::ParquetStore;

fn source() -> RecordBatch {
    let mut vectors = ListBuilder::new(Float32Builder::new()).with_field(Arc::new(Field::new(
        "element",
        DataType::Float32,
        false,
    )));
    for vector in [[0.0, 0.0], [1.0, 0.0], [10.0, 0.0], [11.0, 0.0]] {
        vectors.values().append_slice(&vector);
        vectors.append(true);
    }
    let vectors = vectors.finish();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Utf8, false),
            Field::new("title", DataType::Utf8, false),
            Field::new("embedding", vectors.data_type().clone(), false),
        ])),
        vec![
            Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
            Arc::new(StringArray::from(vec!["zero", "one", "ten", "eleven"])),
            Arc::new(vectors),
        ],
    )
    .unwrap()
}

fn shared_query_fixture() -> (TempDir, ParquetStore, IndexMetadata) {
    let temporary = TempDir::new().unwrap();
    let fixture_source = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../spec/fixtures/v1/valid");
    let fixture_destination = temporary.path().join("v1/valid");
    let source_destination = fixture_destination.join("source");
    create_dir_all(&source_destination).unwrap();
    copy(
        fixture_source.join("source.parquet"),
        source_destination.join("part-00000.parquet"),
    )
    .unwrap();
    copy(
        fixture_source.join("ivf_centroids.parquet"),
        fixture_destination.join("ivf_centroids.parquet"),
    )
    .unwrap();
    let postings_source = fixture_source.join("ivf_postings");
    let postings_destination = temporary.path().join("v1/valid/ivf_postings");
    let bucket = "cid_bucket=000000";
    create_dir_all(postings_destination.join(bucket)).unwrap();
    copy(
        postings_source.join("manifest.json"),
        postings_destination.join("manifest.json"),
    )
    .unwrap();
    copy(
        postings_source.join(bucket).join("part-00000.parquet"),
        postings_destination.join(bucket).join("part-00000.parquet"),
    )
    .unwrap();
    let registry = StorageRegistry::default();
    registry
        .register_store(
            "s3://parqdb-fixtures/",
            Arc::new(LocalFileSystem::new_with_prefix(temporary.path()).unwrap()),
        )
        .unwrap();
    let metadata = IndexMetadata::from_json_slice(include_bytes!(
        "../../../../spec/fixtures/v1/valid/metadata.json"
    ))
    .unwrap();
    (temporary, ParquetStore::new(registry), metadata)
}

fn resolved_search(nlist: usize, cluster_selection: ClusterSelection) -> ResolvedSearch {
    ResolvedSearch {
        source_relation_key: "file:///source".into(),
        query: vec![0.0, 0.0],
        vector_field: "embedding".into(),
        source_key_fields: vec!["id".into()],
        postings_relation_key: Some("file:///postings".into()),
        posting_encoding: PostingEncoding::Source,
        metric: DistanceMetric::L2Squared,
        source_vector_is_f64: false,
        cluster_selection: Some(cluster_selection),
        nlist: Some(nlist),
        ntotal: Some(nlist),
        projection: vec!["id".into()],
        filter: None,
        limit: 10,
    }
}

#[test]
fn datafusion_cluster_filter_uses_inline_relation_and_full_scan_modes() {
    let inline = resolved_search(64, ClusterSelection::Native(vec![0, 3, 7]));
    assert!(!datafusion_cluster_relation_required(&inline).unwrap());
    let inline_sql =
        compile_datafusion_sql(&inline, Some("source"), Some("postings"), None, None).unwrap();
    assert!(inline_sql.contains("p.\"cid\" IN (0, 3, 7)"));

    let relation = resolved_search(256, ClusterSelection::Native((0..129).collect()));
    assert!(datafusion_cluster_relation_required(&relation).unwrap());
    let embedded_relation_sql =
        compile_datafusion_sql(&relation, Some("source"), Some("postings"), None, None).unwrap();
    assert!(
        embedded_relation_sql
            .contains("parqdb_selected_clusters(\"cid\") AS (\n        VALUES (0), (1), (2), (3)")
    );
    assert!(embedded_relation_sql.contains(
        "LEFT SEMI JOIN \"parqdb_selected_clusters\" AS selected ON p.\"cid\" = selected.\"cid\""
    ));
    let relation_sql = compile_datafusion_sql(
        &relation,
        Some("source"),
        Some("postings"),
        None,
        Some("selected_clusters"),
    )
    .unwrap();
    assert!(relation_sql.contains(
        "LEFT SEMI JOIN \"selected_clusters\" AS selected ON p.\"cid\" = selected.\"cid\""
    ));
    assert!(!relation_sql.contains("p.\"cid\" IN"));

    let all = resolved_search(64, ClusterSelection::All);
    assert!(!datafusion_cluster_relation_required(&all).unwrap());
    let all_sql =
        compile_datafusion_sql(&all, Some("source"), Some("postings"), None, None).unwrap();
    assert!(all_sql.contains("SELECT * FROM \"postings\" AS p"));
    assert!(!all_sql.contains("p.\"cid\" IN"));
    assert!(!all_sql.contains("SEMI JOIN"));
}

#[test]
fn datafusion_cluster_filter_rejects_invalid_selected_cids() {
    for selected in [vec![], vec![-1], vec![0, 0], vec![64]] {
        let resolved = resolved_search(64, ClusterSelection::Native(selected));
        assert!(datafusion_cluster_relation_required(&resolved).is_err());
    }
}

#[test]
fn source_encoding_requires_the_source_relation() {
    let mut resolved = resolved_search(2, ClusterSelection::All);
    assert!(datafusion_source_relation_required(&resolved).unwrap());
    resolved.projection = vec!["title".into()];
    assert!(datafusion_source_relation_required(&resolved).unwrap());
}

#[test]
fn datafusion_cluster_filter_builds_relational_centroid_top_k() {
    let resolved = resolved_search(
        4096,
        ClusterSelection::Relational {
            centroids_relation_key: "file:///centroids".into(),
            nprobe: 64,
        },
    );

    assert!(datafusion_centroid_relation_required(&resolved).unwrap());
    assert!(!datafusion_cluster_relation_required(&resolved).unwrap());
    assert!(
        compile_datafusion_sql(&resolved, Some("source"), Some("postings"), None, None).is_err()
    );
    let sql = compile_datafusion_sql(
        &resolved,
        Some("source"),
        Some("postings"),
        Some("centroids"),
        None,
    )
    .unwrap();

    assert!(sql.contains("FROM \"centroids\" AS c"));
    assert!(sql.contains(
        "ORDER BY parqdb_squared_l2(c.\"centroid\", make_array(CAST(0 AS REAL), CAST(0 AS REAL))) \
         ASC, c.\"cid\" ASC"
    ));
    assert!(sql.contains("LIMIT 64"));
    assert!(sql.contains(
        "LEFT SEMI JOIN parqdb_selected_clusters AS selected ON p.\"cid\" = selected.\"cid\""
    ));
}

#[test]
fn cluster_router_uses_centroid_matrix_size() {
    assert!(use_native_cluster_routing(1024, 1024));
    assert!(use_native_cluster_routing(8192, 384));
    assert!(!use_native_cluster_routing(65_536, 384));
    assert!(!use_native_cluster_routing(usize::MAX, 2));
}

#[tokio::test]
async fn relational_cluster_routing_matches_native_routing() {
    let source = source();
    let artifacts = build_ivf_tables(&source, "embedding", &["id".into()], 2).unwrap();
    let snapshot = snapshot(&artifacts);
    let selected =
        selected_cluster_ids(&snapshot, &artifacts.centroids, &[0.0, 0.0], Some(1)).unwrap();
    let context = SessionContext::new();
    context.register_udf(squared_l2_udf());
    context.register_batch("source", source).unwrap();
    context
        .register_batch("centroids", artifacts.centroids)
        .unwrap();
    context
        .register_batch("postings", artifacts.postings)
        .unwrap();

    let native = resolved_search(2, ClusterSelection::Native(selected));
    let relational = resolved_search(
        2,
        ClusterSelection::Relational {
            centroids_relation_key: "file:///centroids".into(),
            nprobe: 1,
        },
    );
    let native_sql =
        compile_datafusion_sql(&native, Some("source"), Some("postings"), None, None).unwrap();
    let relational_sql = compile_datafusion_sql(
        &relational,
        Some("source"),
        Some("postings"),
        Some("centroids"),
        None,
    )
    .unwrap();
    let relational_plan = context
        .sql(&relational_sql)
        .await
        .unwrap()
        .create_physical_plan()
        .await
        .unwrap();
    let relational_plan = datafusion::physical_plan::displayable(relational_plan.as_ref())
        .indent(false)
        .to_string();
    assert!(relational_plan.contains("TopK(fetch=1)"));
    assert!(relational_plan.contains("join_type=RightSemi"));
    let native_batches = context
        .sql(&native_sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();
    let relational_batches = context
        .sql(&relational_sql)
        .await
        .unwrap()
        .collect()
        .await
        .unwrap();

    let ids = |batches: &[RecordBatch]| {
        batches
            .iter()
            .flat_map(|batch| {
                let values = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .unwrap();
                (0..batch.num_rows())
                    .map(|row| values.value(row).to_owned())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>()
    };
    assert_eq!(ids(&relational_batches), ids(&native_batches));
}

#[tokio::test]
async fn squared_l2_udf_takes_the_query_vector_as_an_argument() {
    let context = SessionContext::new();
    context.register_udf(squared_l2_udf());
    context.register_batch("source", source()).unwrap();
    let distance = context.udf("parqdb_squared_l2").unwrap();

    let near_zero = context
        .table("source")
        .await
        .unwrap()
        .select(vec![
            col("id"),
            distance
                .call(vec![col("embedding"), query_literal(&[0.0, 0.0])])
                .alias("distance"),
        ])
        .unwrap()
        .sort(vec![
            col("distance").sort(true, false),
            col("id").sort(true, false),
        ])
        .unwrap()
        .limit(0, Some(1))
        .unwrap()
        .collect()
        .await
        .unwrap();
    let near_ten = context
        .table("source")
        .await
        .unwrap()
        .select(vec![
            col("id"),
            distance
                .call(vec![col("embedding"), query_literal(&[10.0, 0.0])])
                .alias("distance"),
        ])
        .unwrap()
        .sort(vec![
            col("distance").sort(true, false),
            col("id").sort(true, false),
        ])
        .unwrap()
        .limit(0, Some(1))
        .unwrap()
        .collect()
        .await
        .unwrap();

    assert_eq!(
        near_zero[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "a"
    );
    assert_eq!(
        near_ten[0]
            .column(0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap()
            .value(0),
        "c"
    );
}

fn assert_fixture_result(case: &Value, result: &[RecordBatch]) {
    let mut actual = result
        .iter()
        .flat_map(|batch| {
            let distances = batch
                .column(1)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap();
            (0..batch.num_rows())
                .map(|row| {
                    (
                        array_value_to_string(batch.column(0).as_ref(), row).unwrap(),
                        distances.value(row),
                    )
                })
                .collect::<Vec<_>>()
        })
        .collect::<Vec<_>>();
    assert!(
        actual
            .windows(2)
            .all(|rows| rows[0].1.total_cmp(&rows[1].1).is_le()),
        "{}",
        case["name"]
    );
    let mut expected = case["expected"]
        .as_array()
        .unwrap()
        .iter()
        .map(|row| {
            (
                row["document_id"].as_str().unwrap().to_owned(),
                row["_distance"].as_f64().unwrap() as f32,
            )
        })
        .collect::<Vec<_>>();
    let canonical_order = |left: &(String, f32), right: &(String, f32)| {
        left.1
            .total_cmp(&right.1)
            .then_with(|| left.0.cmp(&right.0))
    };
    actual.sort_by(canonical_order);
    expected.sort_by(canonical_order);
    assert_eq!(actual, expected, "{}", case["name"]);
}

#[tokio::test]
async fn datafusion_execution_matches_the_shared_query_fixtures() {
    let (_temporary, parquet, metadata) = shared_query_fixture();
    let snapshot = metadata.current_snapshot().unwrap();
    let fixture_root = "s3://parqdb-fixtures/v1/valid/";
    let source_uri = format!("{fixture_root}source/");
    let centroids_uri = format!(
        "{fixture_root}{}",
        snapshot.index_relations["ivf_centroids"]
    );
    let postings_uri = format!("{fixture_root}{}", snapshot.index_relations["ivf_postings"]);
    let centroids = parquet.read(&centroids_uri, None).await.unwrap();
    let source = parquet.read(&source_uri, None).await.unwrap();
    let postings = parquet.dataframe(&postings_uri).await.unwrap();
    let postings_schema = Arc::clone(postings.schema().inner());
    let postings = concat_batches(&postings_schema, &postings.collect().await.unwrap()).unwrap();
    let context = SessionContext::new();
    context.register_udf(squared_l2_udf());
    context.register_batch("source", source).unwrap();
    context.register_batch("postings", postings).unwrap();
    let cases: Vec<Value> = serde_json::from_slice(include_bytes!(
        "../../../../spec/fixtures/v1/valid/queries.json"
    ))
    .unwrap();

    for case in cases {
        let query = case["query-vector"]
            .as_array()
            .unwrap()
            .iter()
            .map(|value| value.as_f64().unwrap() as f32)
            .collect::<Vec<_>>();
        let nprobe = usize::try_from(case["nprobe"].as_u64().unwrap()).unwrap();
        let limit = usize::try_from(case["k"].as_u64().unwrap()).unwrap();
        let filter = case["filter"]["status"]
            .as_str()
            .map(|status| format!("status = '{status}'"));
        let projection = case["projection"]
            .as_array()
            .unwrap()
            .iter()
            .map(|field| field.as_str().unwrap().to_owned())
            .collect();
        let selected = selected_cluster_ids(snapshot, &centroids, &query, Some(nprobe)).unwrap();
        let resolved = ResolvedSearch {
            source_relation_key: source_uri.clone(),
            query,
            vector_field: snapshot.vector_field.clone(),
            source_key_fields: snapshot.source_key_fields.clone(),
            postings_relation_key: Some(postings_uri.clone()),
            posting_encoding: PostingEncoding::from_snapshot(snapshot).unwrap(),
            metric: DistanceMetric::L2Squared,
            source_vector_is_f64: false,
            cluster_selection: Some(if nprobe == snapshot.parameter_usize("nlist").unwrap() {
                ClusterSelection::All
            } else {
                ClusterSelection::Native(selected)
            }),
            nlist: Some(snapshot.parameter_usize("nlist").unwrap()),
            ntotal: Some(snapshot.parameter_usize("ntotal").unwrap()),
            projection,
            filter,
            limit,
        };
        let source_name = datafusion_source_relation_required(&resolved)
            .unwrap()
            .then_some("source");
        let sql =
            compile_datafusion_sql(&resolved, source_name, Some("postings"), None, None).unwrap();
        let result = context.sql(&sql).await.unwrap().collect().await.unwrap();
        assert_fixture_result(&case, &result);
    }
}

fn snapshot(artifacts: &IvfTables) -> IndexSnapshot {
    IndexSnapshot {
        snapshot_id: 1,
        sequence_number: 1,
        timestamp_ms: 1,
        summary: BTreeMap::new(),
        vector_field: "embedding".into(),
        source_key_fields: vec!["id".into()],
        indexed_rows: i64::try_from(artifacts.ntotal).unwrap(),
        index_family: "ivf".into(),
        index_schema_version: 1,
        metric: "l2_squared".into(),
        parameters: BTreeMap::from([
            ("dimension".into(), artifacts.dimension.to_string()),
            ("nlist".into(), artifacts.nlist.to_string()),
            ("ntotal".into(), artifacts.ntotal.to_string()),
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
                "metadata/fe985f6d-3592-4385-a1ca-71347057a210/v1.metadata.json".into(),
            ),
        ]),
        index_relations: BTreeMap::new(),
    }
}

#[test]
fn executes_spec_distance_ordering_projection_and_exact_distance() {
    let source = source();
    let artifacts = build_ivf_tables(&source, "embedding", &["id".into()], 2).unwrap();
    let snapshot = snapshot(&artifacts);
    let (batches, schema) = execute(&SearchInput {
        snapshot: &snapshot,
        source: &source,
        centroids: &artifacts.centroids,
        postings: &artifacts.postings,
        query: &[0.5, 0.0],
        nprobe: Some(2),
        limit: 2,
        projection: Some(&["id".into(), "title".into()]),
    })
    .unwrap();

    assert_eq!(
        schema
            .fields()
            .iter()
            .map(|field| field.name().as_str())
            .collect::<Vec<_>>(),
        ["id", "title", "_distance"]
    );
    let mut ids = batches[0]
        .column(0)
        .as_any()
        .downcast_ref::<StringArray>()
        .unwrap()
        .iter()
        .flatten()
        .collect::<Vec<_>>();
    ids.sort_unstable();
    assert_eq!(ids, ["a", "b"]);
    assert_eq!(
        batches[0]
            .column(2)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap()
            .values(),
        &[0.25, 0.25]
    );
}

#[test]
fn validates_query_inputs_and_relation_cardinality() {
    let source = source();
    let artifacts = build_ivf_tables(&source, "embedding", &["id".into()], 2).unwrap();
    let snapshot = snapshot(&artifacts);
    let execute_with = |query: &[f32], nprobe: Option<usize>, limit: usize| {
        execute(&SearchInput {
            snapshot: &snapshot,
            source: &source,
            centroids: &artifacts.centroids,
            postings: &artifacts.postings,
            query,
            nprobe,
            limit,
            projection: None,
        })
    };

    assert!(execute_with(&[0.0], Some(1), 1).is_err());
    assert!(execute_with(&[f32::NAN, 0.0], Some(1), 1).is_err());
    assert!(execute_with(&[0.0, 0.0], Some(0), 1).is_err());
    assert!(execute_with(&[0.0, 0.0], Some(3), 1).is_err());
    assert!(execute_with(&[0.0, 0.0], Some(1), 0).is_err());
}

#[test]
fn rejects_invalid_projection() {
    let source = source();
    let artifacts = build_ivf_tables(&source, "embedding", &["id".into()], 2).unwrap();
    let snapshot = snapshot(&artifacts);
    let run = |projection: &[String]| {
        execute(&SearchInput {
            snapshot: &snapshot,
            source: &source,
            centroids: &artifacts.centroids,
            postings: &artifacts.postings,
            query: &[0.0, 0.0],
            nprobe: Some(2),
            limit: 2,
            projection: Some(projection),
        })
    };

    assert!(run(&[]).is_err());
    assert!(run(&["id".into(), "id".into()]).is_err());
    assert!(run(&["missing".into()]).is_err());
}

#[test]
fn supports_long_source_keys_without_a_special_mode() {
    let mut vectors = ListBuilder::new(Float32Builder::new()).with_field(Arc::new(Field::new(
        "element",
        DataType::Float32,
        false,
    )));
    for vector in [[0.0, 0.0], [10.0, 0.0]] {
        vectors.values().append_slice(&vector);
        vectors.append(true);
    }
    let vectors = vectors.finish();
    let source = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("source_pid", DataType::Int64, false),
            Field::new("embedding", vectors.data_type().clone(), false),
        ])),
        vec![Arc::new(Int64Array::from(vec![0, 1])), Arc::new(vectors)],
    )
    .unwrap();
    let artifacts = build_ivf_tables(&source, "embedding", &["source_pid".into()], 2).unwrap();
    let mut snapshot = snapshot(&artifacts);
    snapshot.source_key_fields = vec!["source_pid".into()];

    let (batches, _) = execute(&SearchInput {
        snapshot: &snapshot,
        source: &source,
        centroids: &artifacts.centroids,
        postings: &artifacts.postings,
        query: &[10.0, 0.0],
        nprobe: Some(2),
        limit: 1,
        projection: Some(&["source_pid".into()]),
    })
    .unwrap();
    assert_eq!(
        batches[0]
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap()
            .values(),
        &[1]
    );
}
