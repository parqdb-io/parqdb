//! Fused Top-K optimizer and execution tests.

use std::sync::Arc;

use arrow::array::{Array, Float32Array, Float32Builder, Int64Array, ListBuilder};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::MemTable;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};

use super::{IvfTopKExec, relify_session_context};

fn batch(ids: impl IntoIterator<Item = i64>) -> RecordBatch {
    let ids = ids.into_iter().collect::<Vec<_>>();
    let mut vectors = ListBuilder::new(Float32Builder::new()).with_field(Arc::new(Field::new(
        "element",
        DataType::Float32,
        false,
    )));
    for id in &ids {
        vectors
            .values()
            .append_slice(&[f32::from(u16::try_from(*id).unwrap()), 0.0]);
        vectors.append(true);
    }
    let vectors = vectors.finish();
    RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("id", DataType::Int64, false),
            Field::new("embedding", vectors.data_type().clone(), false),
        ])),
        vec![Arc::new(Int64Array::from(ids)), Arc::new(vectors)],
    )
    .unwrap()
}

fn find_fused(plan: &Arc<dyn ExecutionPlan>) -> Option<&IvfTopKExec> {
    if let Some(fused) = plan.downcast_ref::<IvfTopKExec>() {
        return Some(fused);
    }
    plan.children()
        .into_iter()
        .find_map(|child| find_fused(child))
}

fn metric_sum(plan: &Arc<dyn ExecutionPlan>, name: &str) -> usize {
    find_fused(plan)
        .unwrap()
        .metrics()
        .unwrap()
        .iter()
        .filter(|metric| metric.value().name() == name)
        .map(|metric| metric.value().as_usize())
        .sum()
}

#[tokio::test]
async fn fuses_distance_projection_topk_and_updates_its_dynamic_filter() {
    let context = relify_session_context();
    let first = batch(0..8);
    let second = batch(100..108);
    let schema = first.schema();
    context
        .register_table(
            "vectors",
            Arc::new(MemTable::try_new(schema, vec![vec![first, second]]).unwrap()),
        )
        .unwrap();

    let dataframe = context
        .sql(
            "SELECT id, \
             relify_squared_l2(embedding, make_array(CAST(0 AS REAL), CAST(0 AS REAL))) \
             AS _distance \
             FROM vectors ORDER BY _distance ASC LIMIT 3",
        )
        .await
        .unwrap();
    let plan = dataframe.create_physical_plan().await.unwrap();
    let formatted = displayable(plan.as_ref()).indent(false).to_string();
    assert!(formatted.contains("IvfTopKExec"), "{formatted}");
    assert!(!formatted.contains("ProjectionExec"), "{formatted}");
    assert!(!formatted.contains("SortExec: TopK"), "{formatted}");

    let dynamic_filter = find_fused(&plan).unwrap().dynamic_filter();
    let results = collect(Arc::clone(&plan), context.task_ctx())
        .await
        .unwrap();
    let ids = results
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();
    let distances = results
        .iter()
        .flat_map(|batch| {
            batch
                .column(1)
                .as_any()
                .downcast_ref::<Float32Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();

    assert_eq!(ids, [0, 1, 2]);
    assert_eq!(distances, [0.0, 1.0, 4.0]);
    assert!(
        dynamic_filter.current().unwrap().to_string().contains('<'),
        "{dynamic_filter}"
    );
    assert_eq!(metric_sum(&plan, "input_batches"), 2);
    assert_eq!(metric_sum(&plan, "distance_evaluations"), 16);
    assert_eq!(metric_sum(&plan, "selection_candidates"), 8);
    assert_eq!(metric_sum(&plan, "selection_discarded"), 5);
    assert_eq!(metric_sum(&plan, "selection_passes"), 1);
    assert_eq!(metric_sum(&plan, "dynamic_filter_pruned"), 8);
    assert_eq!(metric_sum(&plan, "retained_batches_peak"), 1);
    assert!(metric_sum(&plan, "retained_bytes_peak") > 0);
    assert!(metric_sum(&plan, "distance_compute") > 0);
    assert!(metric_sum(&plan, "selection_compute") > 0);
    assert!(metric_sum(&plan, "candidate_sort_compute") > 0);
    assert!(metric_sum(&plan, "projection_compute") > 0);
}

#[tokio::test]
async fn large_top_k_uses_batch_selection_and_retains_only_projected_columns() {
    let context = relify_session_context();
    let first = batch(0..2_000);
    let second = batch(2_000..3_000);
    let schema = first.schema();
    context
        .register_table(
            "vectors",
            Arc::new(MemTable::try_new(schema, vec![vec![first, second]]).unwrap()),
        )
        .unwrap();

    let dataframe = context
        .sql(
            "SELECT id, \
             relify_squared_l2(embedding, make_array(CAST(0 AS REAL), CAST(0 AS REAL))) \
             AS _distance \
             FROM vectors ORDER BY _distance ASC LIMIT 1500",
        )
        .await
        .unwrap();
    let plan = dataframe.create_physical_plan().await.unwrap();
    let results = collect(Arc::clone(&plan), context.task_ctx())
        .await
        .unwrap();
    let ids = results
        .iter()
        .flat_map(|batch| {
            batch
                .column(0)
                .as_any()
                .downcast_ref::<Int64Array>()
                .unwrap()
                .values()
                .iter()
                .copied()
        })
        .collect::<Vec<_>>();

    assert_eq!(ids.len(), 1_500);
    assert_eq!(ids.first(), Some(&0));
    assert_eq!(ids.last(), Some(&1_499));
    assert_eq!(metric_sum(&plan, "selection_candidates"), 2_000);
    assert_eq!(metric_sum(&plan, "selection_discarded"), 500);
    assert_eq!(metric_sum(&plan, "selection_passes"), 1);
    assert_eq!(metric_sum(&plan, "dynamic_filter_pruned"), 1_000);
    assert!(metric_sum(&plan, "retained_bytes_peak") < 100_000);
}
