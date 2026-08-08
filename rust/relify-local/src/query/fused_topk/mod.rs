//! Fused `DataFusion` IVF distance projection and Top-K rule.

use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float32Array, LargeListArray, ListArray};
use arrow::datatypes::{DataType, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::Result as DataFusionResult;
use datafusion::common::config::ConfigOptions;
use datafusion::common::tree_node::{Transformed, TransformedResult, TreeNode};
use datafusion::execution::session_state::{SessionState, SessionStateBuilder};
use datafusion::physical_expr::PhysicalExpr;
use datafusion::physical_expr::expressions::{Column, DynamicFilterPhysicalExpr, lit};
use datafusion::physical_optimizer::PhysicalOptimizerRule;
use datafusion::physical_optimizer::optimizer::PhysicalOptimizer;
use datafusion::physical_plan::ExecutionPlan;
use datafusion::physical_plan::projection::ProjectionExec;
use datafusion::physical_plan::sorts::sort::SortExec;
use datafusion::prelude::{SessionConfig, SessionContext};

use self::exec::{DistanceInput, IvfTopKExec};
use super::{lvq_squared_l2_udf, squared_l2_udf};
use relify_kernels::LvqBits;

mod exec;
mod selector;

const DISTANCE_COLUMN: &str = "_distance";
const DISTANCE_UDF: &str = "relify_squared_l2";
const LVQ4_DISTANCE_UDF: &str = "relify_lvq4_l2";
const LVQ8_DISTANCE_UDF: &str = "relify_lvq8_l2";

#[cfg(test)]
mod tests;

/// Builds the single `DataFusion` context used by local Relify sessions.
pub(crate) fn relify_session_context() -> SessionContext {
    let base = SessionContext::new_with_config(SessionConfig::new());
    let mut rules = PhysicalOptimizer::new().rules;
    // The distance Projection must still be explicit when this rule runs.
    // EnforceSorting has already attached `fetch`, while the first
    // ProjectionPushdown would otherwise absorb the expression into the scan.
    let insertion = rules
        .iter()
        .position(|rule| rule.name() == "ProjectionPushdown")
        .unwrap_or(rules.len());
    rules.insert(insertion, Arc::new(FuseIvfTopK));
    let state: SessionState = SessionStateBuilder::new_from_existing(base.state())
        .with_physical_optimizer_rules(rules)
        .build();
    let context = SessionContext::new_with_state(state);
    context.register_udf(squared_l2_udf());
    context.register_udf(lvq_squared_l2_udf(LvqBits::Four));
    context.register_udf(lvq_squared_l2_udf(LvqBits::Eight));
    context
}

/// Replaces `Sort(Projection(relify_squared_l2(...)))` with one vector-native
/// operator. Unsupported physical shapes retain `DataFusion`'s generic plan.
#[derive(Debug)]
struct FuseIvfTopK;

impl PhysicalOptimizerRule for FuseIvfTopK {
    fn optimize(
        &self,
        plan: Arc<dyn ExecutionPlan>,
        _config: &ConfigOptions,
    ) -> DataFusionResult<Arc<dyn ExecutionPlan>> {
        plan.transform_up(|node| {
            let Some(sort) = node.downcast_ref::<SortExec>() else {
                return Ok(Transformed::no(node));
            };
            let Some(fetch) = sort.fetch().filter(|fetch| *fetch > 0) else {
                return Ok(Transformed::no(node));
            };
            if sort.expr().len() != 1 {
                return Ok(Transformed::no(node));
            }
            let sort_expr = &sort.expr()[0];
            if sort_expr.options.descending {
                return Ok(Transformed::no(node));
            }
            let Some(sort_column) = sort_expr.expr.downcast_ref::<Column>() else {
                return Ok(Transformed::no(node));
            };
            let Some(projection) = sort.input().downcast_ref::<ProjectionExec>() else {
                return Ok(Transformed::no(node));
            };
            let distance_index = sort_column.index();
            let Some(distance_projection) = projection.expr().get(distance_index) else {
                return Ok(Transformed::no(node));
            };
            if distance_projection.alias != DISTANCE_COLUMN {
                return Ok(Transformed::no(node));
            }
            let Some(function) = distance_projection
                .expr
                .downcast_ref::<datafusion::physical_expr::ScalarFunctionExpr>()
            else {
                return Ok(Transformed::no(node));
            };
            let Some(query_expression) = function.args().last() else {
                return Ok(Transformed::no(node));
            };
            let Some(query) =
                evaluate_constant_vector(query_expression, projection.input().schema())
            else {
                return Ok(Transformed::no(node));
            };
            if query.is_empty() || query.iter().any(|value| !value.is_finite()) {
                return Ok(Transformed::no(node));
            }
            let input_schema = projection.input().schema();
            let Some(distance_input) = distance_input(function, &input_schema, query.len()) else {
                return Ok(Transformed::no(node));
            };
            let dynamic_filter = sort.dynamic_filter_expr().unwrap_or_else(|| {
                Arc::new(DynamicFilterPhysicalExpr::new(
                    vec![Arc::new(Column::new(DISTANCE_COLUMN, distance_index))],
                    lit(true),
                ))
            });
            let fused = IvfTopKExec::new(
                Arc::clone(projection.input()),
                projection.expr().to_vec(),
                projection.schema(),
                distance_index,
                distance_input,
                query,
                fetch,
                !sort.preserve_partitioning(),
                dynamic_filter,
            );
            Ok(Transformed::yes(Arc::new(fused) as Arc<dyn ExecutionPlan>))
        })
        .data()
    }

    fn name(&self) -> &'static str {
        "relify_fuse_ivf_topk"
    }

    fn schema_check(&self) -> bool {
        true
    }
}

fn distance_input(
    function: &datafusion::physical_expr::ScalarFunctionExpr,
    schema: &SchemaRef,
    dimension: usize,
) -> Option<DistanceInput> {
    let column = |index: usize| {
        function
            .args()
            .get(index)?
            .downcast_ref::<Column>()
            .map(Column::index)
    };
    match (function.name(), function.args().len()) {
        (DISTANCE_UDF, 2) => DistanceInput::dense(schema, column(0)?, dimension),
        (LVQ4_DISTANCE_UDF, 4) => DistanceInput::lvq(
            schema,
            LvqBits::Four,
            column(0)?,
            column(1)?,
            column(2)?,
            dimension,
        ),
        (LVQ8_DISTANCE_UDF, 4) => DistanceInput::lvq(
            schema,
            LvqBits::Eight,
            column(0)?,
            column(1)?,
            column(2)?,
            dimension,
        ),
        _ => None,
    }
}

fn evaluate_constant_vector(expr: &Arc<dyn PhysicalExpr>, schema: SchemaRef) -> Option<Vec<f32>> {
    let batch = RecordBatch::new_empty(schema);
    let value = expr.evaluate(&batch).ok()?.into_array(1).ok()?;
    let vector = match value.data_type() {
        DataType::List(_) => value.as_any().downcast_ref::<ListArray>()?.value(0),
        DataType::LargeList(_) => value.as_any().downcast_ref::<LargeListArray>()?.value(0),
        DataType::FixedSizeList(_, _) => value
            .as_any()
            .downcast_ref::<FixedSizeListArray>()?
            .value(0),
        _ => return None,
    };
    Some(
        vector
            .as_any()
            .downcast_ref::<Float32Array>()?
            .values()
            .to_vec(),
    )
}
