//! Fused IVF Top-K physical execution node.

use std::cmp::Ordering;
use std::fmt::{self, Formatter};
use std::sync::Arc;
use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering as AtomicOrdering};

use arrow::compute::SortOptions;
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::config::ConfigOptions;
use datafusion::common::{DataFusionError, Result as DataFusionResult, ScalarValue};
use datafusion::execution::TaskContext;
use datafusion::execution::memory_pool::MemoryConsumer;
use datafusion::logical_expr::Operator;
use datafusion::physical_expr::expressions::{
    BinaryExpr, Column, DynamicFilterPhysicalExpr, Literal, lit,
};
use datafusion::physical_expr::projection::ProjectionExpr;
use datafusion::physical_expr::{
    Distribution, EquivalenceProperties, PhysicalExpr, PhysicalSortExpr,
};
use datafusion::physical_plan::execution_plan::{CardinalityEffect, EmissionType};
use datafusion::physical_plan::filter_pushdown::{FilterDescription, FilterPushdownPhase};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricBuilder, MetricsSet, RecordOutput,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, ExecutionPlanProperties, PlanProperties,
    SendableRecordBatchStream, Statistics,
};
use futures::StreamExt;

use super::DISTANCE_COLUMN;
use super::selector::{CandidateSelector, compute_batch_distances, retained_input_columns};
use relify_kernels::{LvqBits, detect};

#[derive(Debug, Clone, Copy)]
pub(super) enum VectorKind {
    List,
    LargeList,
    FixedSizeList,
}

#[derive(Debug, Clone, Copy)]
pub(super) enum DistanceInput {
    Dense {
        vector_index: usize,
        vector_kind: VectorKind,
    },
    Lvq {
        bits: LvqBits,
        code_index: usize,
        offset_index: usize,
        scale_index: usize,
    },
}

impl DistanceInput {
    pub(super) fn dense(schema: &SchemaRef, vector_index: usize, dimension: usize) -> Option<Self> {
        Some(Self::Dense {
            vector_index,
            vector_kind: VectorKind::from_schema(schema, vector_index, dimension)?,
        })
    }

    pub(super) fn lvq(
        schema: &SchemaRef,
        bits: LvqBits,
        code_index: usize,
        offset_index: usize,
        scale_index: usize,
        dimension: usize,
    ) -> Option<Self> {
        let code_size = i32::try_from(bits.code_size(dimension)).ok()?;
        let code = schema.fields().get(code_index)?;
        let offset = schema.fields().get(offset_index)?;
        let scale = schema.fields().get(scale_index)?;
        if code.is_nullable()
            || code.data_type() != &DataType::FixedSizeBinary(code_size)
            || offset.is_nullable()
            || offset.data_type() != &DataType::Float32
            || scale.is_nullable()
            || scale.data_type() != &DataType::Float32
        {
            return None;
        }
        Some(Self::Lvq {
            bits,
            code_index,
            offset_index,
            scale_index,
        })
    }

    fn display(self) -> String {
        match self {
            Self::Dense { vector_index, .. } => format!("vector_col={vector_index}"),
            Self::Lvq {
                bits,
                code_index,
                offset_index,
                scale_index,
            } => {
                let encoding = match bits {
                    LvqBits::Four => "lvq4",
                    LvqBits::Eight => "lvq8",
                };
                format!(
                    "encoding={encoding}, code_col={code_index}, offset_col={offset_index}, scale_col={scale_index}"
                )
            }
        }
    }
}

impl VectorKind {
    pub(super) fn from_schema(schema: &SchemaRef, index: usize, dimension: usize) -> Option<Self> {
        let field = schema.fields().get(index)?;
        if field.is_nullable() {
            return None;
        }
        match field.data_type() {
            DataType::List(element)
                if !element.is_nullable() && element.data_type() == &DataType::Float32 =>
            {
                Some(Self::List)
            }
            DataType::LargeList(element)
                if !element.is_nullable() && element.data_type() == &DataType::Float32 =>
            {
                Some(Self::LargeList)
            }
            DataType::FixedSizeList(element, size)
                if !element.is_nullable()
                    && element.data_type() == &DataType::Float32
                    && usize::try_from(*size).ok() == Some(dimension) =>
            {
                Some(Self::FixedSizeList)
            }
            _ => None,
        }
    }
}

#[derive(Debug)]
struct DynamicThreshold {
    value: AtomicU32,
    completed_partitions: AtomicUsize,
    partition_count: usize,
    expression: Arc<DynamicFilterPhysicalExpr>,
    distance_expr: Arc<dyn PhysicalExpr>,
}

impl DynamicThreshold {
    fn new(
        partition_count: usize,
        distance_index: usize,
        expression: Arc<DynamicFilterPhysicalExpr>,
    ) -> Self {
        Self {
            value: AtomicU32::new(f32::INFINITY.to_bits()),
            completed_partitions: AtomicUsize::new(0),
            partition_count,
            expression,
            distance_expr: Arc::new(Column::new(DISTANCE_COLUMN, distance_index)),
        }
    }

    #[inline]
    fn load(&self) -> f32 {
        f32::from_bits(self.value.load(AtomicOrdering::Relaxed))
    }

    fn sync_from_expression(&self) {
        let Ok(current) = self.expression.current() else {
            return;
        };
        let Some(binary) = current.downcast_ref::<BinaryExpr>() else {
            return;
        };
        if !matches!(binary.op(), Operator::Lt | Operator::LtEq) {
            return;
        }
        let Some(literal) = binary.right().downcast_ref::<Literal>() else {
            return;
        };
        if let ScalarValue::Float32(Some(value)) = literal.value() {
            self.tighten_atomic(*value);
        }
    }

    fn tighten(&self, candidate: f32) -> DataFusionResult<()> {
        if !self.tighten_atomic(candidate) {
            return Ok(());
        }
        self.expression.update(Arc::new(BinaryExpr::new(
            Arc::clone(&self.distance_expr),
            Operator::Lt,
            lit(ScalarValue::Float32(Some(candidate))),
        )))
    }

    fn tighten_atomic(&self, candidate: f32) -> bool {
        let mut current = self.value.load(AtomicOrdering::Relaxed);
        loop {
            let threshold = f32::from_bits(current);
            if candidate.total_cmp(&threshold) != Ordering::Less {
                return false;
            }
            match self.value.compare_exchange_weak(
                current,
                candidate.to_bits(),
                AtomicOrdering::AcqRel,
                AtomicOrdering::Relaxed,
            ) {
                Ok(_) => return true,
                Err(observed) => current = observed,
            }
        }
    }

    fn partition_complete(&self) {
        if self
            .completed_partitions
            .fetch_add(1, AtomicOrdering::AcqRel)
            + 1
            == self.partition_count
        {
            self.expression.mark_complete();
        }
    }
}

#[derive(Debug)]
pub(super) struct IvfTopKExec {
    input: Arc<dyn ExecutionPlan>,
    projection: Vec<ProjectionExpr>,
    schema: SchemaRef,
    distance_index: usize,
    distance_input: DistanceInput,
    query: Arc<[f32]>,
    fetch: usize,
    requires_single_partition: bool,
    dynamic_threshold: Arc<DynamicThreshold>,
    metrics: ExecutionPlanMetricsSet,
    properties: Arc<PlanProperties>,
}

impl IvfTopKExec {
    #[allow(clippy::too_many_arguments)]
    pub(super) fn new(
        input: Arc<dyn ExecutionPlan>,
        projection: Vec<ProjectionExpr>,
        schema: SchemaRef,
        distance_index: usize,
        distance_input: DistanceInput,
        query: Vec<f32>,
        fetch: usize,
        requires_single_partition: bool,
        dynamic_filter: Arc<DynamicFilterPhysicalExpr>,
    ) -> Self {
        let mut equivalence = EquivalenceProperties::new(Arc::clone(&schema));
        equivalence.add_ordering([PhysicalSortExpr {
            expr: Arc::new(Column::new(DISTANCE_COLUMN, distance_index)),
            options: SortOptions {
                descending: false,
                nulls_first: false,
            },
        }]);
        let properties = PlanProperties::new(
            equivalence,
            input.output_partitioning().clone(),
            EmissionType::Final,
            input.boundedness(),
        );
        let partition_count = input.output_partitioning().partition_count();
        Self {
            input,
            projection,
            schema,
            distance_index,
            distance_input,
            query: query.into(),
            fetch,
            requires_single_partition,
            dynamic_threshold: Arc::new(DynamicThreshold::new(
                partition_count,
                distance_index,
                dynamic_filter,
            )),
            metrics: ExecutionPlanMetricsSet::new(),
            properties: Arc::new(properties),
        }
    }

    fn clone_with_input(&self, input: Arc<dyn ExecutionPlan>) -> Self {
        let mut cloned = Self::new(
            input,
            self.projection.clone(),
            Arc::clone(&self.schema),
            self.distance_index,
            self.distance_input,
            self.query.to_vec(),
            self.fetch,
            self.requires_single_partition,
            Arc::clone(&self.dynamic_threshold.expression),
        );
        cloned.metrics = self.metrics.clone();
        cloned
    }

    #[cfg(test)]
    pub(super) fn dynamic_filter(&self) -> Arc<DynamicFilterPhysicalExpr> {
        Arc::clone(&self.dynamic_threshold.expression)
    }
}

impl DisplayAs for IvfTopKExec {
    fn fmt_as(&self, format: DisplayFormatType, formatter: &mut Formatter<'_>) -> fmt::Result {
        match format {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                formatter,
                "IvfTopKExec: fetch={}, {}, dimension={}, dynamic_filter=[{}]",
                self.fetch,
                self.distance_input.display(),
                self.query.len(),
                self.dynamic_threshold.expression
            ),
            DisplayFormatType::TreeRender => write!(formatter, "fetch={}", self.fetch),
        }
    }
}

impl ExecutionPlan for IvfTopKExec {
    fn name(&self) -> &'static str {
        "IvfTopKExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![&self.input]
    }

    fn required_input_distribution(&self) -> Vec<Distribution> {
        vec![if self.requires_single_partition {
            Distribution::SinglePartition
        } else {
            Distribution::UnspecifiedDistribution
        }]
    }

    fn benefits_from_input_partitioning(&self) -> Vec<bool> {
        vec![!self.requires_single_partition]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let [input] = children.as_slice() else {
            return Err(DataFusionError::Internal(
                "IvfTopKExec requires exactly one input".into(),
            ));
        };
        Ok(Arc::new(self.clone_with_input(Arc::clone(input))))
    }

    fn reset_state(self: Arc<Self>) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let filter = Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::new(Column::new(DISTANCE_COLUMN, self.distance_index))],
            lit(true),
        ));
        Ok(Arc::new(Self::new(
            Arc::clone(&self.input),
            self.projection.clone(),
            Arc::clone(&self.schema),
            self.distance_index,
            self.distance_input,
            self.query.to_vec(),
            self.fetch,
            self.requires_single_partition,
            filter,
        )))
    }

    fn execute(
        &self,
        partition: usize,
        context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let mut input = self.input.execute(partition, Arc::clone(&context))?;
        let schema = self.schema();
        let output_schema = Arc::clone(&schema);
        let projection = self.projection.clone();
        let retained_input_columns = retained_input_columns(&projection, self.distance_index);
        let distance_index = self.distance_index;
        let distance_input = self.distance_input;
        let query = Arc::clone(&self.query);
        let fetch = self.fetch;
        let kernel = *detect();
        let threshold = Arc::clone(&self.dynamic_threshold);
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let input_batches = MetricBuilder::new(&self.metrics).counter("input_batches", partition);
        let input_rows = MetricBuilder::new(&self.metrics).counter("input_rows", partition);
        let distance_evaluations =
            MetricBuilder::new(&self.metrics).counter("distance_evaluations", partition);
        let selection_candidates =
            MetricBuilder::new(&self.metrics).counter("selection_candidates", partition);
        let selection_discarded =
            MetricBuilder::new(&self.metrics).counter("selection_discarded", partition);
        let selection_passes =
            MetricBuilder::new(&self.metrics).counter("selection_passes", partition);
        let dynamic_filter_pruned =
            MetricBuilder::new(&self.metrics).counter("dynamic_filter_pruned", partition);
        let retained_batches_peak =
            MetricBuilder::new(&self.metrics).gauge("retained_batches_peak", partition);
        let retained_bytes_peak =
            MetricBuilder::new(&self.metrics).gauge("retained_bytes_peak", partition);
        let distance_compute =
            MetricBuilder::new(&self.metrics).subset_time("distance_compute", partition);
        let selection_compute =
            MetricBuilder::new(&self.metrics).subset_time("selection_compute", partition);
        let candidate_sort_compute =
            MetricBuilder::new(&self.metrics).subset_time("candidate_sort_compute", partition);
        let projection_compute =
            MetricBuilder::new(&self.metrics).subset_time("projection_compute", partition);
        let reservation = MemoryConsumer::new(format!("IvfTopKExec[{partition}]"))
            .register(&context.runtime_env().memory_pool);

        let future = async move {
            let mut selector = CandidateSelector::new(fetch, retained_input_columns, reservation);
            let mut distances = Vec::new();
            while let Some(batch) = input.next().await {
                let batch = batch?;
                input_batches.add(1);
                input_rows.add(batch.num_rows());
                threshold.sync_from_expression();
                let dynamic_limit = threshold.load();
                let timer = baseline.elapsed_compute().timer();
                let distance_timer = distance_compute.timer();
                compute_batch_distances(&mut distances, &batch, distance_input, &query, kernel)?;
                distance_timer.done();
                distance_evaluations.add(distances.len());
                let selection_timer = selection_compute.timer();
                let stats = selector.update_batch(batch, &distances, dynamic_limit)?;
                selection_timer.done();
                selection_candidates.add(stats.candidates);
                selection_discarded.add(stats.discarded);
                selection_passes.add(stats.passes);
                dynamic_filter_pruned.add(stats.pruned);
                retained_batches_peak.set_max(selector.retained_batch_count());
                retained_bytes_peak.set_max(selector.retained_bytes());
                if let Some(local_limit) = selector.threshold() {
                    threshold.tighten(local_limit)?;
                }
                timer.done();
            }
            drop(input);
            let timer = baseline.elapsed_compute().timer();
            let selection_timer = selection_compute.timer();
            let stats = selector.compact_retained();
            selection_timer.done();
            selection_discarded.add(stats.discarded);
            selection_passes.add(stats.passes);
            let output = selector
                .finish(
                    &projection,
                    distance_index,
                    output_schema,
                    &candidate_sort_compute,
                    &projection_compute,
                )?
                .record_output(&baseline);
            baseline.done();
            threshold.partition_complete();
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
        let statistics = Statistics::new_unknown(self.schema.as_ref());
        Ok(Arc::new(statistics.with_fetch(Some(self.fetch), 0, 1)?))
    }

    fn with_fetch(&self, limit: Option<usize>) -> Option<Arc<dyn ExecutionPlan>> {
        match limit {
            Some(limit) if limit > 0 && limit < self.fetch => Some(Arc::new(Self::new(
                Arc::clone(&self.input),
                self.projection.clone(),
                Arc::clone(&self.schema),
                self.distance_index,
                self.distance_input,
                self.query.to_vec(),
                limit,
                self.requires_single_partition,
                Arc::clone(&self.dynamic_threshold.expression),
            ))
                as Arc<dyn ExecutionPlan>),
            Some(limit) if limit >= self.fetch => {
                Some(Arc::new(self.clone_with_input(Arc::clone(&self.input))))
            }
            Some(0) | None => None,
            Some(_) => unreachable!("positive limits below fetch handled above"),
        }
    }

    fn fetch(&self) -> Option<usize> {
        Some(self.fetch)
    }

    fn cardinality_effect(&self) -> CardinalityEffect {
        CardinalityEffect::LowerEqual
    }

    fn gather_filters_for_pushdown(
        &self,
        _phase: FilterPushdownPhase,
        parent_filters: Vec<Arc<dyn PhysicalExpr>>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<FilterDescription> {
        // Top-K is a semantic barrier for parent filters. Its own runtime
        // threshold is consumed by this fused operator rather than pushed to
        // the input, which would evaluate the distance expression twice.
        Ok(FilterDescription::all_unsupported(
            &parent_filters,
            &self.children(),
        ))
    }
}
