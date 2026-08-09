//! Fused Top-K optimizer and execution tests.

use std::sync::Arc;

use arrow::array::{
    Array, ArrayRef, BinaryArray, BinaryViewArray, Float32Array, Float32Builder, Int64Array,
    ListBuilder,
};
use arrow::buffer::Buffer;
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use datafusion::catalog::MemTable;
use datafusion::execution::runtime_env::RuntimeEnvBuilder;
use datafusion::physical_plan::{ExecutionPlan, collect, displayable};

use arrow_data::ByteView;
use relify_kernels::{LvqBits, detect};

use super::exec::DistanceInput;
use super::selector::compute_batch_distances;
use super::{IvfTopKExec, relify_session_context};
use crate::config::relify_session_config;

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

#[test]
fn lvq_distance_uses_only_the_sliced_code_buffer() {
    let values = [&[99_u8, 99][..], &[3_u8, 4][..], &[5_u8, 6][..]];
    let arrays: [ArrayRef; 2] = [
        Arc::new(BinaryArray::from_iter_values(values)),
        Arc::new(BinaryViewArray::from_iter_values(values)),
    ];

    for codes in arrays {
        let batch = RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("code", codes.data_type().clone(), false),
                Field::new("offset", DataType::Float32, false),
                Field::new("scale", DataType::Float32, false),
            ])),
            vec![
                codes,
                Arc::new(Float32Array::from(vec![0.0, 0.0, 0.0])),
                Arc::new(Float32Array::from(vec![1.0, 1.0, 1.0])),
            ],
        )
        .unwrap()
        .slice(1, 2);
        let mut distances = Vec::new();

        compute_batch_distances(
            &mut distances,
            &batch,
            DistanceInput::Lvq {
                bits: LvqBits::Eight,
                code_index: 0,
                offset_index: 1,
                scale_index: 2,
            },
            &[0.0, 0.0],
            *detect(),
        )
        .unwrap();

        assert_eq!(distances, [25.0, 61.0]);
    }
}

#[test]
fn lvq_distance_scans_plain_byte_array_strides() {
    let code_size = 16;
    let stride = code_size + 4;
    let mut values = vec![0_u8; 2 * stride];
    values[4..4 + code_size].fill(1);
    values[stride + 4..stride + 4 + code_size].fill(2);
    let views = [0, stride]
        .into_iter()
        .map(|offset| {
            ByteView::new(
                u32::try_from(code_size).unwrap(),
                &values[offset + 4..offset + 8],
            )
            .with_buffer_index(0)
            .with_offset(u32::try_from(offset + 4).unwrap())
            .as_u128()
        })
        .collect::<Vec<_>>()
        .into();
    let codes = BinaryViewArray::new(views, vec![Buffer::from(values)], None);
    let batch = RecordBatch::try_new(
        Arc::new(Schema::new(vec![
            Field::new("code", DataType::BinaryView, false),
            Field::new("offset", DataType::Float32, false),
            Field::new("scale", DataType::Float32, false),
        ])),
        vec![
            Arc::new(codes),
            Arc::new(Float32Array::from(vec![0.0, 0.0])),
            Arc::new(Float32Array::from(vec![1.0, 1.0])),
        ],
    )
    .unwrap();
    let mut distances = Vec::new();

    compute_batch_distances(
        &mut distances,
        &batch,
        DistanceInput::Lvq {
            bits: LvqBits::Eight,
            code_index: 0,
            offset_index: 1,
            scale_index: 2,
        },
        &[0.0; 16],
        *detect(),
    )
    .unwrap();

    assert_eq!(distances, [16.0, 64.0]);
}

#[tokio::test]
async fn fuses_distance_projection_topk_and_updates_its_dynamic_filter() {
    let context = relify_session_context(
        relify_session_config(),
        Arc::new(RuntimeEnvBuilder::default().build().unwrap()),
    );
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
    let context = relify_session_context(
        relify_session_config(),
        Arc::new(RuntimeEnvBuilder::default().build().unwrap()),
    );
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
