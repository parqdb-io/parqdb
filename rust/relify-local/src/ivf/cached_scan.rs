//! Filter-aware scans over session-cached IVF postings.

use std::cmp::Reverse;
use std::collections::{BTreeMap, BTreeSet, BinaryHeap};
use std::fmt;
use std::sync::Arc;

use arrow::array::BooleanArray;
use arrow::compute::filter_record_batch;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider};
use datafusion::common::{DataFusionError, Result as DataFusionResult, ScalarValue};
use datafusion::execution::TaskContext;
use datafusion::logical_expr::{Expr, Operator, TableProviderFilterPushDown, TableType};
use datafusion::physical_expr::expressions::DynamicFilterPhysicalExpr;
use datafusion::physical_expr::utils::collect_columns;
use datafusion::physical_expr::{EquivalenceProperties, PhysicalExpr};
use datafusion::physical_plan::execution_plan::{Boundedness, EmissionType, SchedulingType};
use datafusion::physical_plan::filter_pushdown::{
    ChildPushdownResult, FilterPushdownPhase, FilterPushdownPropagation, PushedDown,
};
use datafusion::physical_plan::metrics::{
    BaselineMetrics, ExecutionPlanMetricsSet, MetricsSet, RecordOutput,
};
use datafusion::physical_plan::stream::RecordBatchStreamAdapter;
use datafusion::physical_plan::{
    DisplayAs, DisplayFormatType, ExecutionPlan, Partitioning, PlanProperties,
    SendableRecordBatchStream,
};
use futures::StreamExt;

use super::CachedIvfPostings;

const SCAN_PARTITIONS_PER_WORKER: usize = 4;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BatchRange {
    batch: usize,
    start: usize,
    length: usize,
}

struct CachedIvfProvider {
    postings: Arc<CachedIvfPostings>,
}

struct CachedIvfScanExec {
    postings: Arc<CachedIvfPostings>,
    partitions: Arc<[Arc<[BatchRange]>]>,
    projection: Option<Vec<usize>>,
    runtime_filters: Vec<Arc<dyn PhysicalExpr>>,
    metrics: ExecutionPlanMetricsSet,
    properties: Arc<PlanProperties>,
}

impl fmt::Debug for CachedIvfScanExec {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CachedIvfScanExec")
            .field("partitions", &self.partitions.len())
            .field("ranges", &range_count(&self.partitions))
            .field("projection", &self.projection)
            .field("runtime_filters", &self.runtime_filters.len())
            .finish_non_exhaustive()
    }
}

impl CachedIvfPostings {
    pub(crate) fn provider(self: &Arc<Self>) -> Arc<dyn TableProvider> {
        Arc::new(CachedIvfProvider {
            postings: Arc::clone(self),
        })
    }
}

impl fmt::Debug for CachedIvfProvider {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        formatter
            .debug_struct("CachedIvfProvider")
            .field("clusters", &self.postings.cluster_ids().count())
            .finish_non_exhaustive()
    }
}

#[async_trait]
impl TableProvider for CachedIvfProvider {
    fn schema(&self) -> SchemaRef {
        self.postings.schema()
    }

    fn table_type(&self) -> TableType {
        TableType::Temporary
    }

    async fn scan(
        &self,
        state: &dyn Session,
        projection: Option<&Vec<usize>>,
        filters: &[Expr],
        _limit: Option<usize>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        let cids = selected_cids(&self.postings, filters);
        let target_partitions = state
            .config()
            .target_partitions()
            .saturating_mul(SCAN_PARTITIONS_PER_WORKER);
        let partitions = route_ranges(&self.postings, &cids, target_partitions).into();
        Ok(Arc::new(CachedIvfScanExec::try_new(
            Arc::clone(&self.postings),
            partitions,
            projection.cloned(),
            vec![],
        )?))
    }

    fn supports_filters_pushdown(
        &self,
        filters: &[&Expr],
    ) -> DataFusionResult<Vec<TableProviderFilterPushDown>> {
        Ok(filters
            .iter()
            .map(|filter| {
                if cid_filter_values(filter).is_some() {
                    TableProviderFilterPushDown::Exact
                } else {
                    TableProviderFilterPushDown::Unsupported
                }
            })
            .collect())
    }
}

impl CachedIvfScanExec {
    fn try_new(
        postings: Arc<CachedIvfPostings>,
        partitions: Arc<[Arc<[BatchRange]>]>,
        projection: Option<Vec<usize>>,
        runtime_filters: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DataFusionResult<Self> {
        let schema = match &projection {
            Some(projection) => Arc::new(postings.schema().project(projection)?),
            None => postings.schema(),
        };
        let properties = PlanProperties::new(
            EquivalenceProperties::new(schema),
            Partitioning::UnknownPartitioning(partitions.len()),
            EmissionType::Incremental,
            Boundedness::Bounded,
        )
        .with_scheduling_type(SchedulingType::Cooperative);
        Ok(Self {
            postings,
            partitions,
            projection,
            runtime_filters,
            metrics: ExecutionPlanMetricsSet::new(),
            properties: Arc::new(properties),
        })
    }

    fn with_runtime_filters(
        &self,
        runtime_filters: Vec<Arc<dyn PhysicalExpr>>,
    ) -> DataFusionResult<Self> {
        Self::try_new(
            Arc::clone(&self.postings),
            Arc::clone(&self.partitions),
            self.projection.clone(),
            runtime_filters,
        )
    }
}

impl DisplayAs for CachedIvfScanExec {
    fn fmt_as(&self, format: DisplayFormatType, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match format {
            DisplayFormatType::Default | DisplayFormatType::Verbose => write!(
                formatter,
                "CachedIvfScanExec: partitions={}, ranges={}, runtime_filters={}",
                self.partitions.len(),
                range_count(&self.partitions),
                self.runtime_filters.len()
            ),
            DisplayFormatType::TreeRender => write!(
                formatter,
                "partitions={}, ranges={}",
                self.partitions.len(),
                range_count(&self.partitions)
            ),
        }
    }
}

impl ExecutionPlan for CachedIvfScanExec {
    fn name(&self) -> &'static str {
        "CachedIvfScanExec"
    }

    fn properties(&self) -> &Arc<PlanProperties> {
        &self.properties
    }

    fn children(&self) -> Vec<&Arc<dyn ExecutionPlan>> {
        vec![]
    }

    fn with_new_children(
        self: Arc<Self>,
        children: Vec<Arc<dyn ExecutionPlan>>,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        if children.is_empty() {
            Ok(self)
        } else {
            Err(DataFusionError::Internal(
                "CachedIvfScanExec cannot have children".into(),
            ))
        }
    }

    fn reset_state(self: Arc<Self>) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        Ok(Arc::new(
            self.with_runtime_filters(self.runtime_filters.clone())?,
        ))
    }

    fn execute(
        &self,
        partition: usize,
        _context: Arc<TaskContext>,
    ) -> DataFusionResult<SendableRecordBatchStream> {
        let ranges = self.partitions.get(partition).cloned().ok_or_else(|| {
            DataFusionError::Internal(format!("CachedIvfScanExec has no partition {partition}"))
        })?;
        let schema = self.schema();
        let postings = Arc::clone(&self.postings);
        let projection = self.projection.clone();
        let runtime_filters = self.runtime_filters.clone();
        let baseline = BaselineMetrics::new(&self.metrics, partition);
        let stream = futures::stream::iter(0..ranges.len()).map(move |index| {
            let timer = baseline.elapsed_compute().timer();
            let range = ranges[index];
            let mut batch = postings.batch(range.batch).slice(range.start, range.length);
            if let Some(projection) = &projection {
                batch = batch.project(projection)?;
            }
            batch = apply_runtime_filters(batch, &runtime_filters)?;
            timer.done();
            Ok(batch.record_output(&baseline))
        });
        Ok(Box::pin(datafusion::physical_plan::coop::cooperative(
            RecordBatchStreamAdapter::new(schema, stream),
        )))
    }

    fn metrics(&self) -> Option<MetricsSet> {
        Some(self.metrics.clone_inner())
    }

    fn handle_child_pushdown_result(
        &self,
        phase: FilterPushdownPhase,
        child_pushdown_result: ChildPushdownResult,
        _config: &datafusion::common::config::ConfigOptions,
    ) -> DataFusionResult<FilterPushdownPropagation<Arc<dyn ExecutionPlan>>> {
        let mut runtime_filters = self.runtime_filters.clone();
        let mut accepted = false;
        let filters = child_pushdown_result
            .parent_filters
            .into_iter()
            .map(|candidate| {
                if phase == FilterPushdownPhase::Post
                    && supports_runtime_filter(&candidate.filter, &self.schema())
                {
                    runtime_filters.push(candidate.filter);
                    accepted = true;
                    PushedDown::Yes
                } else {
                    PushedDown::No
                }
            })
            .collect();
        let updated_node = accepted
            .then(|| self.with_runtime_filters(runtime_filters))
            .transpose()?
            .map(|plan| Arc::new(plan) as Arc<dyn ExecutionPlan>);
        Ok(FilterPushdownPropagation {
            filters,
            updated_node,
        })
    }
}

fn route_ranges(
    postings: &CachedIvfPostings,
    cids: &[i32],
    target_partitions: usize,
) -> Vec<Arc<[BatchRange]>> {
    let mut ranges = selected_ranges(postings, cids);
    if ranges.is_empty() {
        return vec![Arc::from([])];
    }
    ranges.sort_unstable_by_key(|range| Reverse(range.length));
    let partition_count = target_partitions.max(1).min(ranges.len());
    let mut partitions = vec![Vec::new(); partition_count];
    let mut loads = (0..partition_count)
        .map(|partition| Reverse((0usize, partition)))
        .collect::<BinaryHeap<_>>();
    for range in ranges {
        let Reverse((rows, partition)) = loads.pop().expect("at least one cache partition");
        partitions[partition].push(range);
        loads.push(Reverse((rows + range.length, partition)));
    }
    partitions
        .into_iter()
        .map(|ranges| Arc::from(ranges.into_boxed_slice()))
        .collect()
}

fn selected_cids(postings: &CachedIvfPostings, filters: &[Expr]) -> Vec<i32> {
    let mut selected = None::<BTreeSet<i32>>;
    for values in filters.iter().filter_map(cid_filter_values) {
        selected = Some(match selected {
            Some(current) => current.intersection(&values).copied().collect(),
            None => values,
        });
    }
    selected.map_or_else(
        || postings.cluster_ids().collect(),
        |values| values.into_iter().collect(),
    )
}

fn cid_filter_values(filter: &Expr) -> Option<BTreeSet<i32>> {
    match filter {
        Expr::InList(list) if !list.negated && is_cid_column(&list.expr) => list
            .list
            .iter()
            .map(cid_literal)
            .collect::<Option<BTreeSet<_>>>(),
        Expr::BinaryExpr(binary) => match binary.op {
            Operator::Eq => {
                if is_cid_column(&binary.left) {
                    cid_literal(&binary.right).map(|value| BTreeSet::from([value]))
                } else if is_cid_column(&binary.right) {
                    cid_literal(&binary.left).map(|value| BTreeSet::from([value]))
                } else {
                    None
                }
            }
            Operator::Or => {
                let mut values = cid_filter_values(&binary.left)?;
                values.extend(cid_filter_values(&binary.right)?);
                Some(values)
            }
            Operator::And => {
                let left = cid_filter_values(&binary.left)?;
                let right = cid_filter_values(&binary.right)?;
                Some(left.intersection(&right).copied().collect())
            }
            _ => None,
        },
        _ => None,
    }
}

fn is_cid_column(expression: &Expr) -> bool {
    matches!(expression, Expr::Column(column) if column.name == "cid")
}

fn cid_literal(expression: &Expr) -> Option<i32> {
    let Expr::Literal(value, _) = expression else {
        return None;
    };
    match value {
        ScalarValue::Int8(Some(value)) => Some(i32::from(*value)),
        ScalarValue::Int16(Some(value)) => Some(i32::from(*value)),
        ScalarValue::Int32(value) => *value,
        ScalarValue::Int64(Some(value)) => i32::try_from(*value).ok(),
        ScalarValue::UInt8(Some(value)) => Some(i32::from(*value)),
        ScalarValue::UInt16(Some(value)) => Some(i32::from(*value)),
        ScalarValue::UInt32(Some(value)) => i32::try_from(*value).ok(),
        ScalarValue::UInt64(Some(value)) => i32::try_from(*value).ok(),
        _ => None,
    }
}

fn selected_ranges(postings: &CachedIvfPostings, cids: &[i32]) -> Vec<BatchRange> {
    let mut ranges = BTreeMap::<usize, Vec<(usize, usize)>>::new();
    for cid in cids.iter().copied().collect::<BTreeSet<_>>() {
        for fragment in postings.fragments(cid).into_iter().flatten() {
            ranges
                .entry(fragment.batch)
                .or_default()
                .push((fragment.start, fragment.length));
        }
    }
    let mut selected = Vec::new();
    for (batch, mut ranges) in ranges {
        ranges.sort_unstable_by_key(|(start, _)| *start);
        let mut ranges = ranges.into_iter();
        let Some((mut start, length)) = ranges.next() else {
            continue;
        };
        let mut end = start + length;
        for (next_start, next_length) in ranges {
            if next_start == end {
                end += next_length;
            } else {
                selected.push(BatchRange {
                    batch,
                    start,
                    length: end - start,
                });
                start = next_start;
                end = next_start + next_length;
            }
        }
        selected.push(BatchRange {
            batch,
            start,
            length: end - start,
        });
    }
    selected
}

fn supports_runtime_filter(filter: &Arc<dyn PhysicalExpr>, schema: &SchemaRef) -> bool {
    let columns = collect_columns(filter);
    filter.downcast_ref::<DynamicFilterPhysicalExpr>().is_some()
        && !columns.is_empty()
        && columns.into_iter().all(|column| {
            schema
                .fields()
                .get(column.index())
                .is_some_and(|field| is_routing_column(field.name()))
        })
}

fn is_routing_column(name: &str) -> bool {
    name == "cid"
        || name.strip_prefix("key_").is_some_and(|suffix| {
            !suffix.starts_with('0') && suffix.bytes().all(|byte| byte.is_ascii_digit())
        })
}

fn apply_runtime_filters(
    mut batch: RecordBatch,
    filters: &[Arc<dyn PhysicalExpr>],
) -> DataFusionResult<RecordBatch> {
    for filter in filters {
        let values = filter.evaluate(&batch)?.into_array(batch.num_rows())?;
        let mask = values
            .as_any()
            .downcast_ref::<BooleanArray>()
            .ok_or_else(|| {
                DataFusionError::Execution(
                    "cached IVF runtime filter did not produce boolean values".into(),
                )
            })?;
        batch = filter_record_batch(&batch, mask)?;
    }
    Ok(batch)
}

fn range_count(partitions: &[Arc<[BatchRange]>]) -> usize {
    partitions.iter().map(|ranges| ranges.len()).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use arrow::array::{ArrayRef, Int32Array, Int64Array};
    use datafusion::common::ScalarValue;
    use datafusion::logical_expr::Operator;
    use datafusion::physical_expr::expressions::{BinaryExpr, Column, lit as physical_lit};
    use datafusion::physical_plan::collect;
    use datafusion::physical_plan::filter_pushdown::ChildFilterPushdownResult;
    use datafusion::prelude::{SessionConfig, SessionContext, col, lit};

    fn postings_batch(cids: Vec<i32>, keys: Vec<i64>) -> RecordBatch {
        RecordBatch::try_from_iter([
            ("cid", Arc::new(Int32Array::from(cids)) as ArrayRef),
            ("key_1", Arc::new(Int64Array::from(keys)) as ArrayRef),
        ])
        .unwrap()
    }

    fn cached_postings() -> Arc<CachedIvfPostings> {
        let first = postings_batch(vec![0, 0, 1, 1], vec![0, 1, 2, 3]);
        let second = postings_batch(vec![2, 2, 3, 3], vec![4, 5, 6, 7]);
        Arc::new(CachedIvfPostings::from_partitions(&[vec![first, second]], 8).unwrap())
    }

    fn dynamic_filter(name: &str, index: usize, minimum: ScalarValue) -> Arc<dyn PhysicalExpr> {
        let column = Arc::new(Column::new(name, index)) as Arc<dyn PhysicalExpr>;
        Arc::new(DynamicFilterPhysicalExpr::new(
            vec![Arc::clone(&column)],
            Arc::new(BinaryExpr::new(
                column,
                Operator::GtEq,
                physical_lit(minimum),
            )),
        ))
    }

    fn keys(batches: &[RecordBatch]) -> Vec<i64> {
        let mut keys = batches
            .iter()
            .flat_map(|batch| {
                batch
                    .column_by_name("key_1")
                    .unwrap()
                    .as_any()
                    .downcast_ref::<Int64Array>()
                    .unwrap()
                    .values()
                    .iter()
                    .copied()
            })
            .collect::<Vec<_>>();
        keys.sort_unstable();
        keys
    }

    #[test]
    fn adjacent_cluster_ranges_share_one_scan_range() {
        let parent = postings_batch(vec![0, 0, 1, 1, 2, 2], (0..6).collect());
        let postings = CachedIvfPostings::from_partitions(&[vec![parent]], 8).unwrap();

        let adjacent = selected_ranges(&postings, &[0, 1]);
        let separated = selected_ranges(&postings, &[0, 2]);

        assert_eq!(
            adjacent
                .iter()
                .map(|range| range.length)
                .collect::<Vec<_>>(),
            [4]
        );
        assert_eq!(
            separated
                .iter()
                .map(|range| range.length)
                .collect::<Vec<_>>(),
            [2, 2]
        );
    }

    #[tokio::test]
    async fn provider_routes_clusters_and_pushes_projection_into_the_scan() {
        let postings = cached_postings();
        let provider = postings.provider();
        let context = SessionContext::new();
        let dataframe = context
            .read_table(provider)
            .unwrap()
            .filter(col("cid").in_list(vec![lit(0_i32), lit(2_i32)], false))
            .unwrap()
            .select_columns(&["key_1"])
            .unwrap();
        let plan = dataframe.create_physical_plan().await.unwrap();
        let formatted = datafusion::physical_plan::displayable(plan.as_ref())
            .indent(false)
            .to_string();
        let batches = collect(plan, context.task_ctx()).await.unwrap();

        assert!(formatted.contains("CachedIvfScanExec"), "{formatted}");
        assert!(!formatted.contains("FilterExec"), "{formatted}");
        assert!(!formatted.contains("DataSourceExec"), "{formatted}");
        assert_eq!(keys(&batches), [0, 1, 4, 5]);
        assert!(
            batches
                .iter()
                .all(|batch| batch.schema().fields().len() == 1)
        );
    }

    #[test]
    fn provider_pushes_down_only_cid_selection_filters() {
        let provider = cached_postings().provider();
        let cid = col("cid").in_list(vec![lit(0_i32), lit(2_i32)], false);
        let key = col("key_1").gt(lit(2_i64));

        assert_eq!(
            provider.supports_filters_pushdown(&[&cid, &key]).unwrap(),
            [
                TableProviderFilterPushDown::Exact,
                TableProviderFilterPushDown::Unsupported,
            ]
        );
    }

    #[tokio::test]
    async fn provider_overpartitions_cached_scan_work() {
        let partitions = (0_i32..8)
            .map(|cid| vec![postings_batch(vec![cid], vec![i64::from(cid)])])
            .collect::<Vec<_>>();
        let postings = Arc::new(CachedIvfPostings::from_partitions(&partitions, 8).unwrap());
        let cids = (0_i32..8).collect::<Vec<_>>();
        let provider = postings.provider();
        let context =
            SessionContext::new_with_config(SessionConfig::new().with_target_partitions(2));
        let plan = context
            .read_table(provider)
            .unwrap()
            .filter(col("cid").in_list(cids.into_iter().map(lit).collect(), false))
            .unwrap()
            .create_physical_plan()
            .await
            .unwrap();
        let formatted = datafusion::physical_plan::displayable(plan.as_ref())
            .indent(false)
            .to_string();

        assert!(formatted.contains("partitions=8"), "{formatted}");
    }

    #[tokio::test]
    async fn independent_cluster_views_can_execute_concurrently() {
        let postings = cached_postings();
        let provider = postings.provider();
        let context = SessionContext::new();
        let left = context
            .read_table(Arc::clone(&provider))
            .unwrap()
            .filter(col("cid").in_list(vec![lit(0_i32), lit(1_i32)], false))
            .unwrap()
            .select_columns(&["key_1"])
            .unwrap();
        let right = context
            .read_table(provider)
            .unwrap()
            .filter(col("cid").in_list(vec![lit(2_i32), lit(3_i32)], false))
            .unwrap()
            .select_columns(&["key_1"])
            .unwrap();

        let (left, right) = tokio::join!(left.collect(), right.collect());

        assert_eq!(keys(&left.unwrap()), [0, 1, 2, 3]);
        assert_eq!(keys(&right.unwrap()), [4, 5, 6, 7]);
    }

    #[tokio::test]
    async fn scan_applies_key_runtime_filters_before_emitting_batches() {
        let postings = cached_postings();
        let partitions = route_ranges(&postings, &[0, 1, 2, 3], 2).into();
        let filter = dynamic_filter("key_1", 1, ScalarValue::Int64(Some(5)));
        let plan =
            Arc::new(CachedIvfScanExec::try_new(postings, partitions, None, vec![filter]).unwrap())
                as Arc<dyn ExecutionPlan>;
        let context = SessionContext::new();

        let batches = collect(plan, context.task_ctx()).await.unwrap();

        assert_eq!(keys(&batches), [5, 6, 7]);
    }

    #[test]
    fn runtime_filter_pushdown_accepts_only_cid_and_key_columns() {
        let schema = cached_postings().schema();
        let cid = dynamic_filter("cid", 0, ScalarValue::Int32(Some(1)));
        let key = dynamic_filter("key_1", 1, ScalarValue::Int64(Some(1)));
        let unrelated = dynamic_filter("vector", 2, ScalarValue::Int64(Some(1)));

        assert!(supports_runtime_filter(&cid, &schema));
        assert!(supports_runtime_filter(&key, &schema));
        assert!(!supports_runtime_filter(&unrelated, &schema));
        assert!(!supports_runtime_filter(
            &(Arc::new(Column::new("key_1", 1)) as Arc<dyn PhysicalExpr>),
            &schema
        ));
        assert!(is_routing_column("key_10"));
        assert!(!is_routing_column("key_0"));
        assert!(!is_routing_column("key_01"));
    }

    #[test]
    fn physical_pushdown_attaches_only_supported_runtime_filters() {
        let postings = cached_postings();
        let partitions = route_ranges(&postings, &[0, 1], 1).into();
        let plan = CachedIvfScanExec::try_new(postings, partitions, None, vec![]).unwrap();
        let key = dynamic_filter("key_1", 1, ScalarValue::Int64(Some(1)));
        let unrelated = dynamic_filter("vector", 2, ScalarValue::Int64(Some(1)));
        let result = plan
            .handle_child_pushdown_result(
                FilterPushdownPhase::Post,
                ChildPushdownResult {
                    parent_filters: vec![
                        ChildFilterPushdownResult {
                            filter: key,
                            child_results: vec![],
                        },
                        ChildFilterPushdownResult {
                            filter: unrelated,
                            child_results: vec![],
                        },
                    ],
                    self_filters: vec![],
                },
                &datafusion::common::config::ConfigOptions::new(),
            )
            .unwrap();

        assert!(matches!(
            result.filters.as_slice(),
            [PushedDown::Yes, PushedDown::No]
        ));
        let updated = result.updated_node.unwrap();
        assert_eq!(
            updated
                .downcast_ref::<CachedIvfScanExec>()
                .unwrap()
                .runtime_filters
                .len(),
            1
        );
    }
}
