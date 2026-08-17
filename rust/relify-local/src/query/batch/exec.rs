use std::fmt::{self, Formatter};
use std::sync::Arc;

use arrow::array::{Array, Float32Array, Int32Array};
use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, Result as DataFusionResult};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::MemoryConsumer;
use datafusion::physical_expr::expressions::Column;
use datafusion::physical_expr::{Distribution, EquivalenceProperties, PhysicalSortExpr};
use datafusion::physical_plan::execution_plan::{CardinalityEffect, EmissionType};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet, RecordOutput,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, Partitioning,
    PlanProperties, SendableRecordBatchStream, Statistics,
};
use futures::StreamExt;
use relify_kernels::{DistanceKernel, LvqBatchView, LvqBits, detect};
use relify_meta::DistanceMetric;

use super::{BatchCandidateSelector, BatchRoutes, QUERY_ID_COLUMN};
use crate::query::lvq_code_rows;

#[derive(Debug, Clone, Copy)]
pub(crate) struct BatchLvqInput {
    pub(crate) bits: LvqBits,
    pub(crate) cid_index: usize,
    pub(crate) code_index: usize,
    pub(crate) offset_index: usize,
    pub(crate) scale_index: usize,
}

impl BatchLvqInput {
    fn validate(self, schema: &SchemaRef, dimension: usize) -> DataFusionResult<()> {
        let field = |index: usize| {
            schema.fields().get(index).ok_or_else(|| {
                DataFusionError::Plan(format!("batch IVF input column {index} does not exist"))
            })
        };
        let cid = field(self.cid_index)?;
        let code = field(self.code_index)?;
        let offset = field(self.offset_index)?;
        let scale = field(self.scale_index)?;
        if dimension == 0
            || cid.is_nullable()
            || cid.data_type() != &DataType::Int32
            || code.is_nullable()
            || !matches!(code.data_type(), DataType::Binary | DataType::BinaryView)
            || offset.is_nullable()
            || offset.data_type() != &DataType::Float32
            || scale.is_nullable()
            || scale.data_type() != &DataType::Float32
        {
            return Err(DataFusionError::Plan(
                "batch IVF input does not match the required LVQ schema".into(),
            ));
        }
        Ok(())
    }
}

/// Computes partition-local Top-K results for every query in one postings scan.
#[derive(Debug)]
pub(crate) struct BatchIvfTopKExec {
    input: Arc<dyn ExecutionPlan>,
    routes: BatchRoutes,
    distance_input: BatchLvqInput,
    retained_columns: Vec<usize>,
    fetch: usize,
    metric: DistanceMetric,
    metrics: ExecutionPlanMetricsSet,
    schema: SchemaRef,
    properties: Arc<PlanProperties>,
}

/// Merges partition-local batch Top-K rows after their candidate sets are bounded.
#[derive(Debug)]
pub(crate) struct BatchTopKMergeExec {
    input: Arc<dyn ExecutionPlan>,
    query_count: usize,
    fetch: usize,
    metrics: ExecutionPlanMetricsSet,
    properties: Arc<PlanProperties>,
}

impl BatchIvfTopKExec {
    pub(crate) fn try_new(
        input: Arc<dyn ExecutionPlan>,
        routes: BatchRoutes,
        distance_input: BatchLvqInput,
        retained_columns: Vec<usize>,
        output_names: Vec<String>,
        fetch: usize,
        metric: DistanceMetric,
    ) -> DataFusionResult<Self> {
        if fetch == 0 {
            return Err(DataFusionError::Plan(
                "batch IVF Top-K fetch must be positive".into(),
            ));
        }
        if retained_columns.is_empty() || retained_columns.len() != output_names.len() {
            return Err(DataFusionError::Plan(
                "batch IVF output columns must be non-empty and uniquely named".into(),
            ));
        }
        let input_schema = input.schema();
        distance_input.validate(&input_schema, routes.dimension())?;
        let mut seen = std::collections::HashSet::with_capacity(retained_columns.len());
        let mut fields = Vec::with_capacity(retained_columns.len() + 2);
        fields.push(Field::new(QUERY_ID_COLUMN, DataType::UInt32, false));
        for (index, name) in retained_columns.iter().copied().zip(output_names) {
            if !seen.insert(index) || name.is_empty() {
                return Err(DataFusionError::Plan(
                    "batch IVF output columns must be non-empty and unique".into(),
                ));
            }
            let input_field = input_schema.fields().get(index).ok_or_else(|| {
                DataFusionError::Plan(format!("batch IVF output column {index} does not exist"))
            })?;
            fields.push(input_field.as_ref().clone().with_name(name));
        }
        fields.push(Field::new("_distance", DataType::Float32, false));
        let schema = Arc::new(Schema::new(fields));
        let mut equivalence = EquivalenceProperties::new(Arc::clone(&schema));
        equivalence.add_ordering([
            PhysicalSortExpr {
                expr: Arc::new(Column::new(QUERY_ID_COLUMN, 0)),
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            },
            PhysicalSortExpr {
                expr: Arc::new(Column::new("_distance", schema.fields().len() - 1)),
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            },
        ]);
        let properties = PlanProperties::new(
            equivalence,
            Partitioning::UnknownPartitioning(input.output_partitioning().partition_count()),
            EmissionType::Final,
            input.boundedness(),
        );
        Ok(Self {
            input,
            routes,
            distance_input,
            retained_columns,
            fetch,
            metric,
            metrics: ExecutionPlanMetricsSet::new(),
            schema,
            properties: Arc::new(properties),
        })
    }

    fn clone_with_input(&self, input: Arc<dyn ExecutionPlan>) -> DataFusionResult<Self> {
        let mut cloned = Self::try_new(
            input,
            self.routes.clone(),
            self.distance_input,
            self.retained_columns.clone(),
            self.schema
                .fields()
                .iter()
                .skip(1)
                .take(self.retained_columns.len())
                .map(|field| field.name().clone())
                .collect(),
            self.fetch,
            self.metric,
        )?;
        cloned.metrics = self.metrics.clone();
        Ok(cloned)
    }
}

impl DisplayAs for BatchIvfTopKExec {
    fn fmt_as(&self, format: DisplayFormatType, formatter: &mut Formatter<'_>) -> fmt::Result {
        match format {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                formatter,
                "BatchIvfTopKExec: queries={}, clusters={}, fetch={}, encoding={:?}",
                self.routes.query_count(),
                self.routes.distinct_cids().count(),
                self.fetch,
                self.distance_input.bits
            ),
            DisplayFormatType::TreeRender => write!(formatter, "fetch={}", self.fetch),
        }
    }
}

impl ExecutionPlan for BatchIvfTopKExec {
    fn name(&self) -> &'static str {
        "BatchIvfTopKExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::UnspecifiedDistribution]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![true]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [input] = children.as_slice() else {
            return Err(DataFusionError::Internal(
                "BatchIvfTopKExec requires exactly one input".into(),
            ));
        };
        Ok(Arc::new(self.clone_with_input(Arc::clone(input))?))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let mut input = self.input.execute(partition, Arc::clone(&context))?;
        let schema = Arc::clone(&self.schema);
        let output_schema = Arc::clone(&schema);
        let routes = self.routes.clone();
        let distance_input = self.distance_input;
        let retained_columns = self.retained_columns.clone();
        let fetch = self.fetch;
        let metric = self.metric;
        let kernel = *detect();
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let input_batches = MetricBuilder::new(&self.metrics).counter("input_batches", partition);
        let input_rows = MetricBuilder::new(&self.metrics).counter("input_rows", partition);
        let distance_evaluations =
            MetricBuilder::new(&self.metrics).counter("distance_evaluations", partition);
        let retained_batches_peak =
            MetricBuilder::new(&self.metrics).gauge("retained_batches_peak", partition);
        let retained_bytes_peak =
            MetricBuilder::new(&self.metrics).gauge("retained_bytes_peak", partition);
        let batch_compute =
            MetricBuilder::new(&self.metrics).subset_time("batch_compute", partition);
        let reservation = MemoryConsumer::new(format!("BatchIvfTopKExec[{partition}]"))
            .register(&context.runtime_env().memory_pool);

        let future = async move {
            let mut selector = BatchCandidateSelector::try_new(
                routes.query_count(),
                fetch,
                retained_columns,
                reservation,
            )?;
            let mut distances = Vec::new();
            while let Some(batch) = input.next().await {
                let batch = batch?;
                input_batches.add(1);
                input_rows.add(batch.num_rows());
                let timer = baseline.elapsed_compute().timer();
                let batch_timer = batch_compute.timer();
                let batch_id = selector.begin_batch(&batch)?;
                let evaluations = score_batch(
                    &batch,
                    batch_id,
                    &routes,
                    distance_input,
                    metric,
                    kernel,
                    &mut distances,
                    &mut selector,
                )?;
                distance_evaluations.add(evaluations);
                retained_batches_peak.set_max(selector.retained_batch_count());
                retained_bytes_peak.set_max(selector.retained_bytes());
                selector.end_batch(batch_id);
                batch_timer.done();
                timer.done();
            }
            drop(input);
            let timer = baseline.elapsed_compute().timer();
            let output = selector.finish(output_schema)?.record_output(&baseline);
            baseline.done();
            timer.done();
            Ok::<RecordBatch, DataFusionError>(output)
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(future),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        Ok(Arc::new(Statistics::new_unknown(self.schema.as_ref())))
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::LowerEqual
    }
}

impl BatchTopKMergeExec {
    pub(crate) fn try_new(
        input: Arc<dyn ExecutionPlan>,
        query_count: usize,
        fetch: usize,
    ) -> DataFusionResult<Self> {
        if query_count == 0 || fetch == 0 {
            return Err(DataFusionError::Plan(
                "batch Top-K merge requires positive query and fetch counts".into(),
            ));
        }
        let schema = input.schema();
        if schema.fields().len() < 3
            || schema.field(0).name() != QUERY_ID_COLUMN
            || schema.field(0).data_type() != &DataType::UInt32
            || schema
                .fields()
                .last()
                .is_none_or(|field| field.name() != "_distance")
            || schema
                .fields()
                .last()
                .is_none_or(|field| field.data_type() != &DataType::Float32)
        {
            return Err(DataFusionError::Plan(
                "batch Top-K merge input has an invalid result schema".into(),
            ));
        }
        let mut equivalence = EquivalenceProperties::new(Arc::clone(&schema));
        equivalence.add_ordering([
            PhysicalSortExpr {
                expr: Arc::new(Column::new(QUERY_ID_COLUMN, 0)),
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            },
            PhysicalSortExpr {
                expr: Arc::new(Column::new("_distance", schema.fields().len() - 1)),
                options: SortOptions {
                    descending: false,
                    nulls_first: false,
                },
            },
        ]);
        let properties = PlanProperties::new(
            equivalence,
            Partitioning::UnknownPartitioning(1),
            EmissionType::Final,
            input.boundedness(),
        );
        Ok(Self {
            input,
            query_count,
            fetch,
            metrics: ExecutionPlanMetricsSet::new(),
            properties: Arc::new(properties),
        })
    }
}

impl DisplayAs for BatchTopKMergeExec {
    fn fmt_as(&self, format: DisplayFormatType, formatter: &mut Formatter<'_>) -> fmt::Result {
        match format {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                formatter,
                "BatchTopKMergeExec: queries={}, fetch={}",
                self.query_count, self.fetch
            ),
            DisplayFormatType::TreeRender => write!(formatter, "fetch={}", self.fetch),
        }
    }
}

impl ExecutionPlan for BatchTopKMergeExec {
    fn name(&self) -> &'static str {
        "BatchTopKMergeExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![Distribution::SinglePartition]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![false]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [input] = children.as_slice() else {
            return Err(DataFusionError::Internal(
                "BatchTopKMergeExec requires exactly one input".into(),
            ));
        };
        let mut cloned = Self::try_new(Arc::clone(input), self.query_count, self.fetch)?;
        cloned.metrics = self.metrics.clone();
        Ok(Arc::new(cloned))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        if partition != 0 {
            return Err(DataFusionError::Internal(
                "BatchTopKMergeExec has only one output partition".into(),
            ));
        }
        let mut input = self.input.execute(0, Arc::clone(&context))?;
        let schema = self.schema();
        let output_schema = Arc::clone(&schema);
        let query_count = self.query_count;
        let fetch = self.fetch;
        let distance_index = schema.fields().len() - 1;
        let retained_columns = (1..distance_index).collect::<Vec<_>>();
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let input_batches = MetricBuilder::new(&self.metrics).counter("input_batches", partition);
        let input_rows = MetricBuilder::new(&self.metrics).counter("input_rows", partition);
        let retained_batches_peak =
            MetricBuilder::new(&self.metrics).gauge("retained_batches_peak", partition);
        let retained_bytes_peak =
            MetricBuilder::new(&self.metrics).gauge("retained_bytes_peak", partition);
        let merge_compute =
            MetricBuilder::new(&self.metrics).subset_time("merge_compute", partition);
        let reservation =
            MemoryConsumer::new("BatchTopKMergeExec").register(&context.runtime_env().memory_pool);
        let future = async move {
            let mut selector =
                BatchCandidateSelector::try_new(query_count, fetch, retained_columns, reservation)?;
            while let Some(batch) = input.next().await {
                let batch = batch?;
                input_batches.add(1);
                input_rows.add(batch.num_rows());
                let timer = baseline.elapsed_compute().timer();
                let merge_timer = merge_compute.timer();
                let query_ids = batch
                    .column(0)
                    .as_any()
                    .downcast_ref::<arrow::array::UInt32Array>()
                    .ok_or_else(|| {
                        DataFusionError::Execution("invalid batch query ID column".into())
                    })?;
                let distances = batch
                    .column(distance_index)
                    .as_any()
                    .downcast_ref::<Float32Array>()
                    .ok_or_else(|| {
                        DataFusionError::Execution("invalid batch distance column".into())
                    })?;
                if query_ids.null_count() != 0 || distances.null_count() != 0 {
                    return Err(DataFusionError::Execution(
                        "batch Top-K rows must not contain null query IDs or distances".into(),
                    ));
                }
                let batch_id = selector.begin_batch(&batch)?;
                let mut start = 0;
                while start < batch.num_rows() {
                    let query_ordinal = usize::try_from(query_ids.value(start)).map_err(|_| {
                        DataFusionError::Execution("batch query ID exceeds usize".into())
                    })?;
                    let mut end = start + 1;
                    while end < batch.num_rows() && query_ids.value(end) == query_ids.value(start) {
                        end += 1;
                    }
                    selector.update(
                        batch_id,
                        query_ordinal,
                        start,
                        &distances.values()[start..end],
                    )?;
                    start = end;
                }
                retained_batches_peak.set_max(selector.retained_batch_count());
                retained_bytes_peak.set_max(selector.retained_bytes());
                selector.end_batch(batch_id);
                merge_timer.done();
                timer.done();
            }
            drop(input);
            let timer = baseline.elapsed_compute().timer();
            let output = selector.finish(output_schema)?.record_output(&baseline);
            baseline.done();
            timer.done();
            Ok::<RecordBatch, DataFusionError>(output)
        };
        Ok(Box::pin(RecordBatchStreamAdapter::new(
            schema,
            futures::stream::once(future),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn partition_statistics(&self, _partition: Option<usize>) -> DataFusionResult<Arc<Statistics>> {
        Ok(Arc::new(Statistics::new_unknown(self.schema().as_ref())))
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::LowerEqual
    }
}

#[allow(clippy::too_many_arguments)]
fn score_batch(
    batch: &arrow::record_batch::RecordBatch,
    batch_id: u64,
    routes: &BatchRoutes,
    input: BatchLvqInput,
    metric: DistanceMetric,
    kernel: DistanceKernel,
    distances: &mut Vec<f32>,
    selector: &mut BatchCandidateSelector,
) -> DataFusionResult<usize> {
    if batch.num_rows() == 0 {
        return Ok(0);
    }
    let cids = batch
        .column(input.cid_index)
        .as_any()
        .downcast_ref::<Int32Array>()
        .ok_or_else(|| DataFusionError::Execution("invalid batch IVF cid column".into()))?;
    if cids.null_count() != 0 {
        return Err(DataFusionError::Execution(
            "batch IVF cid column must not contain nulls".into(),
        ));
    }

    let mut start = 0;
    let mut evaluations = 0_usize;
    while start < batch.num_rows() {
        let cid = cids.value(start);
        let mut end = start + 1;
        while end < batch.num_rows() && cids.value(end) == cid {
            end += 1;
        }
        let query_ordinals = routes.query_ordinals(cid);
        if !query_ordinals.is_empty() {
            evaluations =
                evaluations.saturating_add((end - start).saturating_mul(query_ordinals.len()));
            let span = batch.slice(start, end - start);
            let offsets = span
                .column(input.offset_index)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| DataFusionError::Execution("invalid LVQ offset column".into()))?;
            let scales = span
                .column(input.scale_index)
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| DataFusionError::Execution("invalid LVQ scale column".into()))?;
            let codes = lvq_code_rows(
                span.column(input.code_index).as_ref(),
                input.bits.code_size(routes.dimension()),
            )?;
            let view = LvqBatchView::try_new_rows(
                input.bits,
                routes.dimension(),
                codes,
                offsets.values(),
                scales.values(),
            )
            .map_err(|error| DataFusionError::Execution(error.to_string()))?;
            distances.resize(span.num_rows(), 0.0);
            for query_ordinal in query_ordinals {
                let query = routes.query(*query_ordinal).ok_or_else(|| {
                    DataFusionError::Internal("batch route refers to a missing query".into())
                })?;
                kernel
                    .lvq_squared_l2_rows(&view, query, distances)
                    .map_err(|error| DataFusionError::Execution(error.to_string()))?;
                if metric == DistanceMetric::Cosine {
                    for distance in distances.iter_mut() {
                        *distance *= 0.5;
                    }
                }
                selector.update(batch_id, *query_ordinal, start, distances)?;
            }
        }
        start = end;
    }
    Ok(evaluations)
}

#[cfg(test)]
mod tests {
    use arrow::array::{Array, ArrayRef, BinaryArray, StringArray, UInt32Array};
    use arrow::record_batch::RecordBatch;
    use datafusion::catalog::MemTable;
    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion::physical_plan::collect;
    use relify_kernels::encode_lvq_rows;

    use super::*;
    use crate::config::relify_session_config;
    use crate::query::relify_session_context;

    fn postings() -> RecordBatch {
        let encoded = encode_lvq_rows(
            &[0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 11.0, 0.0],
            2,
            LvqBits::Eight,
        )
        .unwrap();
        let codes = encoded.codes().chunks_exact(2).collect::<Vec<_>>();
        RecordBatch::try_from_iter([
            (
                "cid",
                Arc::new(Int32Array::from(vec![0, 0, 0, 0])) as ArrayRef,
            ),
            (
                "key_1",
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])) as ArrayRef,
            ),
            ("code", Arc::new(BinaryArray::from_vec(codes)) as ArrayRef),
            (
                "offset",
                Arc::new(Float32Array::from(encoded.offsets().to_vec())) as ArrayRef,
            ),
            (
                "scale",
                Arc::new(Float32Array::from(encoded.scales().to_vec())) as ArrayRef,
            ),
        ])
        .unwrap()
    }

    #[tokio::test]
    async fn computes_partition_local_top_k_for_each_query() {
        let context = relify_session_context(
            relify_session_config(),
            Arc::new(datafusion::execution::runtime_env::RuntimeEnv::default()),
        );
        let batch = postings();
        context
            .register_table(
                "postings",
                Arc::new(MemTable::try_new(batch.schema(), vec![vec![batch]]).unwrap()),
            )
            .unwrap();
        let input = context
            .table("postings")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let routes = BatchRoutes::try_new(
            vec![vec![0.0, 0.0], vec![10.0, 0.0]],
            vec![vec![0], vec![0]],
        )
        .unwrap();
        let plan: Arc<dyn ExecutionPlan> = Arc::new(
            BatchIvfTopKExec::try_new(
                input,
                routes,
                BatchLvqInput {
                    bits: LvqBits::Eight,
                    cid_index: 0,
                    code_index: 2,
                    offset_index: 3,
                    scale_index: 4,
                },
                vec![1],
                vec!["id".into()],
                2,
                DistanceMetric::L2Squared,
            )
            .unwrap(),
        );

        let batches = collect(plan, context.task_ctx()).await.unwrap();
        let result = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
        let query_ids = result
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let ids = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!(query_ids.values(), &[0, 0, 1, 1]);
        assert_eq!(
            (0..ids.len()).map(|row| ids.value(row)).collect::<Vec<_>>(),
            ["a", "b", "c", "d"]
        );
    }

    #[tokio::test]
    async fn merges_partition_local_results_into_global_top_k() {
        let context = relify_session_context(
            relify_session_config(),
            Arc::new(datafusion::execution::runtime_env::RuntimeEnv::default()),
        );
        let batch = postings();
        context
            .register_table(
                "postings",
                Arc::new(
                    MemTable::try_new(
                        batch.schema(),
                        vec![vec![batch.slice(0, 2)], vec![batch.slice(2, 2)]],
                    )
                    .unwrap(),
                ),
            )
            .unwrap();
        let input = context
            .table("postings")
            .await
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let routes = BatchRoutes::try_new(
            vec![vec![0.0, 0.0], vec![10.0, 0.0]],
            vec![vec![0], vec![0]],
        )
        .unwrap();
        let local: Arc<dyn ExecutionPlan> = Arc::new(
            BatchIvfTopKExec::try_new(
                input,
                routes,
                BatchLvqInput {
                    bits: LvqBits::Eight,
                    cid_index: 0,
                    code_index: 2,
                    offset_index: 3,
                    scale_index: 4,
                },
                vec![1],
                vec!["id".into()],
                2,
                DistanceMetric::L2Squared,
            )
            .unwrap(),
        );
        let coalesced: Arc<dyn ExecutionPlan> = Arc::new(CoalescePartitionsExec::new(local));
        let merged: Arc<dyn ExecutionPlan> =
            Arc::new(BatchTopKMergeExec::try_new(coalesced, 2, 2).unwrap());

        let batches = collect(merged, context.task_ctx()).await.unwrap();
        let result = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
        let query_ids = result
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .unwrap();
        let ids = result
            .column(1)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();

        assert_eq!(query_ids.values(), &[0, 0, 1, 1]);
        assert_eq!(
            (0..ids.len()).map(|row| ids.value(row)).collect::<Vec<_>>(),
            ["a", "b", "c", "d"]
        );
    }
}
