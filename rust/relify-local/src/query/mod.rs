//! `DataFusion` query compilation and reference execution.

use std::collections::HashSet;
use std::sync::Arc;

#[cfg(test)]
use arrow::array::UInt64Array;
use arrow::array::{Array, Float32Array};
#[cfg(test)]
use arrow::compute::take;
use arrow::datatypes::{DataType, Field, FieldRef};
#[cfg(test)]
use arrow::datatypes::{Schema, SchemaRef};
#[cfg(test)]
use arrow::record_batch::RecordBatch;
use datafusion::common::{DataFusionError, ScalarValue};
use datafusion::dataframe::DataFrame;
use datafusion::logical_expr::{
    ColumnarValue, ReturnFieldArgs, ScalarFunctionArgs, ScalarUDF, ScalarUDFImpl, Signature,
    Volatility,
};
use datafusion::prelude::{Expr, col, lit};
use relify_meta::{DistanceMetric, IndexSnapshot, PostingEncoding};

#[cfg(test)]
use crate::ivf::read_centroids;
#[cfg(test)]
use crate::ivf::select_clusters;
#[cfg(test)]
use crate::ivf::{
    borrow_source_vectors, candidate_source_rows, source_key_arrays, source_rows_by_key,
    validate_unique_keys,
};
use crate::{ClusterSelection, Error, ResolvedSearch, Result};
use relify_kernels::{LvqBatchView, LvqBits, detect};

pub(crate) mod batch;
mod fused_topk;
mod lvq_codes;
mod stream;

pub(crate) use lvq_codes::lvq_code_rows;
pub use stream::ManagedQueryStream;

pub(crate) use fused_topk::relify_session_context;

// Keep small selections visible to Parquet pruning without letting generated
// SQL grow with large nprobe values.
const INLINE_CLUSTER_FILTER_LIMIT: usize = 128;

// Native routing enables static Parquet predicates. Bound its centroid matrix
// to 64 MiB so storage-backed queries retain predictable memory usage.
const NATIVE_CLUSTER_ROUTING_MAX_VALUES: usize = 64 * 1024 * 1024 / size_of::<f32>();

#[derive(Debug, PartialEq, Eq, Hash)]
struct SquaredL2 {
    signature: Signature,
}

impl Default for SquaredL2 {
    fn default() -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
        }
    }
}

impl ScalarUDFImpl for SquaredL2 {
    fn name(&self) -> &'static str {
        "relify_squared_l2"
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, argument_types: &[DataType]) -> datafusion::common::Result<DataType> {
        self.coerce_types(argument_types)?;
        Ok(DataType::Float32)
    }

    fn return_field_from_args(
        &self,
        arguments: ReturnFieldArgs<'_>,
    ) -> datafusion::common::Result<FieldRef> {
        let argument_types = arguments
            .arg_fields
            .iter()
            .map(|field| field.data_type().clone())
            .collect::<Vec<_>>();
        self.coerce_types(&argument_types)?;
        Ok(Arc::new(Field::new(self.name(), DataType::Float32, false)))
    }

    fn coerce_types(
        &self,
        argument_types: &[DataType],
    ) -> datafusion::common::Result<Vec<DataType>> {
        if argument_types.len() != 2 {
            return Err(DataFusionError::Plan(format!(
                "{} requires exactly two arguments",
                self.name()
            )));
        }
        if argument_types.iter().any(|data_type| {
            !matches!(
                data_type,
                DataType::List(field) | DataType::LargeList(field)
                    if field.data_type() == &DataType::Float32
            ) && !matches!(
                data_type,
                DataType::FixedSizeList(field, dimension)
                    if *dimension > 0
                        && field.data_type() == &DataType::Float32
            )
        }) {
            return Err(DataFusionError::Plan(format!(
                "{} requires two list<float> arguments",
                self.name()
            )));
        }
        Ok(argument_types.to_vec())
    }

    fn invoke_with_args(
        &self,
        arguments: ScalarFunctionArgs,
    ) -> datafusion::common::Result<ColumnarValue> {
        if arguments.number_rows == 0 {
            return Ok(ColumnarValue::Array(Arc::new(Float32Array::from(
                Vec::<f32>::new(),
            ))));
        }
        let [vectors, queries] = arguments.args.as_slice() else {
            return Err(DataFusionError::Execution(
                "relify_squared_l2 requires exactly two arguments".into(),
            ));
        };
        let vectors = vectors.to_array_of_size(arguments.number_rows)?;
        let (vectors, vector_dimension) =
            crate::ivf::borrow_vectors_allow_nullable_elements(&vectors)
                .map_err(|error| DataFusionError::Execution(error.to_string()))?;
        let (queries, query_count) = match queries {
            ColumnarValue::Scalar(query) => (query.to_array()?, 1),
            ColumnarValue::Array(queries) => {
                if queries.len() != arguments.number_rows {
                    return Err(DataFusionError::Execution(format!(
                        "query vector array contains {} rows but the input contains {}",
                        queries.len(),
                        arguments.number_rows
                    )));
                }
                (Arc::clone(queries), arguments.number_rows)
            }
        };
        let (queries, query_dimension) =
            crate::ivf::borrow_vectors_allow_nullable_elements(&queries)
                .map_err(|error| DataFusionError::Execution(error.to_string()))?;
        if vector_dimension != query_dimension {
            return Err(DataFusionError::Execution(format!(
                "vector dimension {vector_dimension} does not match query dimension \
                 {query_dimension}"
            )));
        }
        let kernel = detect();
        let mut distances = vec![0.0; arguments.number_rows];
        if query_count == 1 {
            kernel.squared_l2_rows(vectors, queries, &mut distances);
        } else {
            kernel.squared_l2_pairs(vectors, queries, vector_dimension, &mut distances);
        }
        if distances.iter().any(|distance| !distance.is_finite()) {
            return Err(DataFusionError::Execution(
                "query produced a non-finite final distance".into(),
            ));
        }
        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(
            distances,
        ))))
    }
}

/// Returns the stateless squared-Euclidean distance UDF used by Relify queries.
#[must_use]
pub fn squared_l2_udf() -> ScalarUDF {
    ScalarUDF::new_from_impl(SquaredL2::default())
}

#[derive(Debug, PartialEq, Eq, Hash)]
struct LvqSquaredL2 {
    signature: Signature,
    bits: LvqBits,
}

impl LvqSquaredL2 {
    fn new(bits: LvqBits) -> Self {
        Self {
            signature: Signature::user_defined(Volatility::Immutable),
            bits,
        }
    }
}

impl ScalarUDFImpl for LvqSquaredL2 {
    fn name(&self) -> &'static str {
        match self.bits {
            LvqBits::Four => "relify_lvq4_l2",
            LvqBits::Eight => "relify_lvq8_l2",
        }
    }

    fn signature(&self) -> &Signature {
        &self.signature
    }

    fn return_type(&self, argument_types: &[DataType]) -> datafusion::common::Result<DataType> {
        self.coerce_types(argument_types)?;
        Ok(DataType::Float32)
    }

    fn return_field_from_args(
        &self,
        arguments: ReturnFieldArgs<'_>,
    ) -> datafusion::common::Result<FieldRef> {
        let types = arguments
            .arg_fields
            .iter()
            .map(|field| field.data_type().clone())
            .collect::<Vec<_>>();
        self.coerce_types(&types)?;
        Ok(Arc::new(Field::new(self.name(), DataType::Float32, false)))
    }

    fn coerce_types(
        &self,
        argument_types: &[DataType],
    ) -> datafusion::common::Result<Vec<DataType>> {
        let [codes, DataType::Float32, DataType::Float32, query] = argument_types else {
            return Err(DataFusionError::Plan(format!(
                "{} requires binary, float, float, and list<float> arguments",
                self.name()
            )));
        };
        if !matches!(codes, DataType::Binary | DataType::BinaryView) || !is_float_vector_type(query)
        {
            return Err(DataFusionError::Plan(format!(
                "{} requires binary, float, float, and list<float> arguments",
                self.name()
            )));
        }
        Ok(argument_types.to_vec())
    }

    fn invoke_with_args(
        &self,
        arguments: ScalarFunctionArgs,
    ) -> datafusion::common::Result<ColumnarValue> {
        if arguments.number_rows == 0 {
            return Ok(ColumnarValue::Array(Arc::new(Float32Array::from(
                Vec::<f32>::new(),
            ))));
        }
        let [codes, offsets, scales, query] = arguments.args.as_slice() else {
            return Err(DataFusionError::Execution(format!(
                "{} requires exactly four arguments",
                self.name()
            )));
        };
        let codes = codes.to_array_of_size(arguments.number_rows)?;
        let offsets = offsets.to_array_of_size(arguments.number_rows)?;
        let scales = scales.to_array_of_size(arguments.number_rows)?;
        let offsets = offsets
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| DataFusionError::Execution("invalid LVQ offset column".into()))?;
        let scales = scales
            .as_any()
            .downcast_ref::<Float32Array>()
            .ok_or_else(|| DataFusionError::Execution("invalid LVQ scale column".into()))?;
        if offsets.null_count() != 0 || scales.null_count() != 0 {
            return Err(DataFusionError::Execution(
                "LVQ posting columns must not contain nulls".into(),
            ));
        }
        let (query, query_count) = match query {
            ColumnarValue::Scalar(query) => (query.to_array()?, 1),
            ColumnarValue::Array(queries) => {
                if queries.len() != arguments.number_rows {
                    return Err(DataFusionError::Execution(format!(
                        "query vector array contains {} rows but the input contains {}",
                        queries.len(),
                        arguments.number_rows
                    )));
                }
                (Arc::clone(queries), arguments.number_rows)
            }
        };
        let (queries, dimension) = crate::ivf::borrow_vectors_allow_nullable_elements(&query)
            .map_err(|error| DataFusionError::Execution(error.to_string()))?;
        let codes = lvq_code_rows(codes.as_ref(), self.bits.code_size(dimension))?;
        let view = LvqBatchView::try_new_rows(
            self.bits,
            dimension,
            codes,
            offsets.values(),
            scales.values(),
        )
        .map_err(|error| DataFusionError::Execution(error.to_string()))?;
        let mut distances = vec![0.0; arguments.number_rows];
        let kernel = detect();
        if query_count == 1 {
            kernel.lvq_squared_l2_rows(&view, queries, &mut distances)
        } else {
            kernel.lvq_squared_l2_pairs(&view, queries, &mut distances)
        }
        .map_err(|error| DataFusionError::Execution(error.to_string()))?;
        Ok(ColumnarValue::Array(Arc::new(Float32Array::from(
            distances,
        ))))
    }
}

fn is_float_vector_type(data_type: &DataType) -> bool {
    matches!(
        data_type,
        DataType::List(field) | DataType::LargeList(field)
            if field.data_type() == &DataType::Float32
    ) || matches!(
        data_type,
        DataType::FixedSizeList(field, dimension)
            if *dimension > 0 && field.data_type() == &DataType::Float32
    )
}

fn lvq_squared_l2_udf(bits: LvqBits) -> ScalarUDF {
    ScalarUDF::new_from_impl(LvqSquaredL2::new(bits))
}

/// Compiles an index-only search without SQL parser and catalog overhead.
pub(crate) fn compile_index_only_plan(
    postings: DataFrame,
    resolved: &ResolvedSearch,
) -> Result<DataFrame> {
    let columns = datafusion_index_only_columns(resolved)?;
    let mut projection = resolved
        .projection
        .iter()
        .map(|output| {
            let input = columns
                .iter()
                .find_map(|(input, candidate)| (candidate == output).then_some(input))
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "index-only projection cannot resolve source column: {output}"
                    ))
                })?;
            Ok(if input == output {
                col(input)
            } else {
                col(input).alias(output)
            })
        })
        .collect::<Result<Vec<_>>>()?;
    let query = resolved
        .query
        .iter()
        .copied()
        .map(|value| ScalarValue::Float32(Some(value)))
        .collect::<Vec<_>>();
    let query = Expr::Literal(
        ScalarValue::List(ScalarValue::new_list(&query, &DataType::Float32, false)),
        None,
    );
    let distance = match resolved.posting_encoding {
        PostingEncoding::Lvq4 => lvq_squared_l2_udf(LvqBits::Four).call(vec![
            col("code"),
            col("offset"),
            col("scale"),
            query,
        ]),
        PostingEncoding::Lvq8 => lvq_squared_l2_udf(LvqBits::Eight).call(vec![
            col("code"),
            col("offset"),
            col("scale"),
            query,
        ]),
        PostingEncoding::Source => {
            return Err(Error::InvalidArgument(
                "source-backed postings cannot execute an index-only search".into(),
            ));
        }
    };
    projection.push(distance.alias("_distance"));
    let ranked = postings
        .select(projection)?
        .sort(vec![col("_distance").sort(true, false)])?
        .limit(0, Some(resolved.limit))?;
    scale_distance_projection(ranked, resolved)
}

/// Compiles one resolved search into `DataFusion` SQL over registered relations.
pub fn compile_datafusion_sql(
    resolved: &ResolvedSearch,
    source_name: Option<&str>,
    postings_name: Option<&str>,
    centroids_name: Option<&str>,
    selected_clusters_name: Option<&str>,
) -> Result<String> {
    let source_required = datafusion_source_relation_required(resolved)?;
    let source_name = validated_datafusion_source_name(source_required, source_name)?;
    let cluster_filter = validated_datafusion_cluster_filter(
        resolved,
        postings_name,
        centroids_name,
        selected_clusters_name,
    )?;

    let query_literal = datafusion_query_literal(resolved);
    let distance = datafusion_distance_sql(resolved, &query_literal);
    let mut ctes = Vec::new();
    if let Some(source_name) = source_name {
        let source = quote_identifier(source_name);
        let predicate = resolved
            .filter
            .as_ref()
            .map_or_else(String::new, |filter| format!("\n        WHERE ({filter})"));
        ctes.push(format!(
            "relify_source AS (\n        SELECT * FROM {source}{predicate}\n    )"
        ));
    }

    let mut selected_clusters_name = selected_clusters_name;
    if matches!(cluster_filter, DataFusionClusterFilter::NativeRelation)
        && selected_clusters_name.is_none()
    {
        ctes.push(datafusion_native_selected_clusters_cte(resolved)?);
        selected_clusters_name = Some("relify_selected_clusters");
    }

    if let Some(postings_name) = postings_name {
        if let DataFusionClusterFilter::Relational = cluster_filter {
            let centroids_name = centroids_name.ok_or_else(|| {
                Error::InvalidArgument("DataFusion centroid relation is not registered".to_owned())
            })?;
            ctes.push(datafusion_selected_clusters_cte(
                resolved,
                centroids_name,
                &query_literal,
            )?);
        }
        ctes.push(datafusion_postings_cte(
            resolved,
            postings_name,
            selected_clusters_name,
            cluster_filter,
        )?);
    }
    ctes.push(datafusion_candidates_cte(
        resolved,
        source_required,
        &distance,
    )?);

    let mut output = resolved
        .projection
        .iter()
        .map(|name| format!("c.{}", quote_identifier(name)))
        .collect::<Vec<_>>();
    output.push(if resolved.metric == DistanceMetric::Cosine {
        "c.\"_distance\" * CAST(0.5 AS REAL) AS \"_distance\"".into()
    } else {
        "c.\"_distance\"".into()
    });
    if resolved.metric == DistanceMetric::Cosine {
        ctes.push(format!(
            "relify_output AS (\n        SELECT {}\n        FROM relify_candidates AS c\n    )",
            output.join(", ")
        ));
        return Ok(format!(
            "WITH\n    {}\nSELECT *\nFROM relify_output\nORDER BY \"_distance\" ASC\nLIMIT {}",
            ctes.join(",\n    "),
            resolved.limit
        ));
    }
    Ok(format!(
        "WITH\n    {}\nSELECT {}\nFROM relify_candidates AS c\nORDER BY c.\"_distance\" ASC\nLIMIT {}",
        ctes.join(",\n    "),
        output.join(", "),
        resolved.limit
    ))
}

fn datafusion_distance_sql(resolved: &ResolvedSearch, query_literal: &str) -> String {
    let query = format!("make_array({query_literal})");
    if resolved.postings_relation_key.is_none()
        || resolved.posting_encoding == PostingEncoding::Source
    {
        let vector = format!("s.{}", quote_identifier(&resolved.vector_field));
        let vector = match resolved.metric {
            DistanceMetric::Cosine => format!("relify_normalize_vector({vector})"),
            DistanceMetric::L2Squared if resolved.source_vector_is_f64 => {
                format!("relify_vector_f32({vector})")
            }
            DistanceMetric::L2Squared => vector,
        };
        return format!("relify_squared_l2({vector}, {query})");
    }
    match resolved.posting_encoding {
        PostingEncoding::Lvq4 => {
            format!("relify_lvq4_l2(p.\"code\", p.\"offset\", p.\"scale\", {query})")
        }
        PostingEncoding::Lvq8 => {
            format!("relify_lvq8_l2(p.\"code\", p.\"offset\", p.\"scale\", {query})")
        }
        PostingEncoding::Source => unreachable!("source encoding handled above"),
    }
}

fn validated_datafusion_source_name(
    source_required: bool,
    source_name: Option<&str>,
) -> Result<Option<&str>> {
    match (source_required, source_name) {
        (true, Some(name)) if !name.is_empty() => Ok(Some(name)),
        (true, _) => Err(Error::InvalidArgument(
            "DataFusion source relation is required for this search".into(),
        )),
        (false, None) => Ok(None),
        (false, Some(_)) => Err(Error::InvalidArgument(
            "DataFusion source relation must be omitted for an index-only search".into(),
        )),
    }
}

fn datafusion_query_literal(resolved: &ResolvedSearch) -> String {
    resolved
        .query
        .iter()
        .map(|value| format!("CAST({value} AS REAL)"))
        .collect::<Vec<_>>()
        .join(", ")
}

fn datafusion_candidates_cte(
    resolved: &ResolvedSearch,
    source_required: bool,
    distance: &str,
) -> Result<String> {
    if resolved.postings_relation_key.is_none() {
        return Ok(format!(
            "relify_candidates AS (\n        SELECT s.*, {distance} AS _distance\n        \
             FROM relify_source AS s\n    )"
        ));
    }
    if source_required {
        let join = resolved
            .source_key_fields
            .iter()
            .enumerate()
            .map(|(position, key)| {
                format!(
                    "s.{} = p.{}",
                    quote_identifier(key),
                    quote_identifier(&format!("key_{}", position + 1))
                )
            })
            .collect::<Vec<_>>()
            .join(" AND ");
        return Ok(format!(
            "relify_candidates AS (\n        SELECT s.*, {distance} AS _distance\n        \
             FROM relify_source AS s\n        JOIN relify_postings AS p ON {join}\n    )"
        ));
    }

    let mut columns = datafusion_index_only_columns(resolved)?
        .into_iter()
        .map(|(input, output)| {
            format!(
                "p.{} AS {}",
                quote_identifier(&input),
                quote_identifier(&output)
            )
        })
        .collect::<Vec<_>>();
    columns.push(format!("{distance} AS _distance"));
    Ok(format!(
        "relify_candidates AS (\n        SELECT {}\n        \
         FROM relify_postings AS p\n    )",
        columns.join(", ")
    ))
}

fn datafusion_index_only_columns(resolved: &ResolvedSearch) -> Result<Vec<(String, String)>> {
    if datafusion_source_relation_required(resolved)? {
        return Err(Error::InvalidArgument(
            "index-only search requires source data".into(),
        ));
    }
    let columns = resolved
        .source_key_fields
        .iter()
        .enumerate()
        .map(|(position, key)| (format!("key_{}", position + 1), key.clone()))
        .collect::<Vec<_>>();
    Ok(columns)
}

fn scale_distance_projection(dataframe: DataFrame, resolved: &ResolvedSearch) -> Result<DataFrame> {
    if resolved.metric == DistanceMetric::L2Squared {
        return Ok(dataframe);
    }
    let mut output = resolved.projection.iter().map(col).collect::<Vec<_>>();
    output.push((col("_distance") * lit(0.5_f32)).alias("_distance"));
    Ok(dataframe.select(output)?)
}

fn validated_datafusion_cluster_filter(
    resolved: &ResolvedSearch,
    postings_name: Option<&str>,
    centroids_name: Option<&str>,
    selected_clusters_name: Option<&str>,
) -> Result<DataFusionClusterFilter> {
    if resolved.postings_relation_key.is_some() != postings_name.is_some() {
        return Err(Error::InvalidArgument(
            "DataFusion postings registration does not match the resolved search".into(),
        ));
    }
    let cluster_filter = datafusion_cluster_filter(resolved)?;
    if !matches!(cluster_filter, DataFusionClusterFilter::NativeRelation)
        && selected_clusters_name.is_some()
    {
        return Err(Error::InvalidArgument(
            "DataFusion selected-cluster relation is not used by the resolved search".into(),
        ));
    }
    if matches!(cluster_filter, DataFusionClusterFilter::Relational) != centroids_name.is_some() {
        return Err(Error::InvalidArgument(
            "DataFusion centroid registration does not match the resolved search".into(),
        ));
    }
    Ok(cluster_filter)
}

fn datafusion_postings_cte(
    resolved: &ResolvedSearch,
    postings_name: &str,
    selected_clusters_name: Option<&str>,
    cluster_filter: DataFusionClusterFilter,
) -> Result<String> {
    let postings = quote_identifier(postings_name);
    let query = match cluster_filter {
        DataFusionClusterFilter::All => format!("SELECT * FROM {postings} AS p"),
        DataFusionClusterFilter::NativeInline => {
            let clusters = native_selected_clusters(resolved)?
                .iter()
                .map(i32::to_string)
                .collect::<Vec<_>>()
                .join(", ");
            format!("SELECT * FROM {postings} AS p\n        WHERE p.\"cid\" IN ({clusters})")
        }
        DataFusionClusterFilter::NativeRelation => {
            let name = selected_clusters_name.ok_or_else(|| {
                Error::InvalidArgument(
                    "DataFusion selected-cluster relation is not registered".into(),
                )
            })?;
            let selected_clusters = quote_identifier(name);
            format!(
                "SELECT p.* FROM {postings} AS p\n        LEFT SEMI JOIN \
                 {selected_clusters} AS selected ON p.\"cid\" = selected.\"cid\""
            )
        }
        DataFusionClusterFilter::Relational => format!(
            "SELECT p.* FROM {postings} AS p\n        LEFT SEMI JOIN \
             relify_selected_clusters AS selected ON p.\"cid\" = selected.\"cid\""
        ),
        DataFusionClusterFilter::Exact => {
            return Err(Error::InvalidArgument(
                "exact search cannot compile an IVF postings relation".into(),
            ));
        }
    };
    Ok(format!("relify_postings AS (\n        {query}\n    )"))
}

fn datafusion_native_selected_clusters_cte(resolved: &ResolvedSearch) -> Result<String> {
    let rows = native_selected_clusters(resolved)?
        .iter()
        .map(|cid| format!("({cid})"))
        .collect::<Vec<_>>()
        .join(", ");
    Ok(format!(
        "relify_selected_clusters(\"cid\") AS (\n        VALUES {rows}\n    )"
    ))
}

fn datafusion_selected_clusters_cte(
    resolved: &ResolvedSearch,
    centroids_name: &str,
    query_literal: &str,
) -> Result<String> {
    let Some(ClusterSelection::Relational { nprobe, .. }) = &resolved.cluster_selection else {
        return Err(Error::InvalidArgument(
            "relational cluster routing is not configured".into(),
        ));
    };
    let centroids = quote_identifier(centroids_name);
    Ok(format!(
        "relify_selected_clusters AS (\n        SELECT c.\"cid\"\n        \
         FROM {centroids} AS c\n        ORDER BY \
         relify_squared_l2(c.\"centroid\", make_array({query_literal})) ASC, c.\"cid\" ASC\n        \
         LIMIT {nprobe}\n    )"
    ))
}

/// Returns whether a `DataFusion` search needs a registered relation of selected CIDs.
pub fn datafusion_cluster_relation_required(resolved: &ResolvedSearch) -> Result<bool> {
    Ok(matches!(
        datafusion_cluster_filter(resolved)?,
        DataFusionClusterFilter::NativeRelation
    ))
}

/// Returns whether a `DataFusion` search routes through the centroid relation.
pub fn datafusion_centroid_relation_required(resolved: &ResolvedSearch) -> Result<bool> {
    Ok(matches!(
        datafusion_cluster_filter(resolved)?,
        DataFusionClusterFilter::Relational
    ))
}

/// Returns whether a `DataFusion` search must register and scan its source relation.
pub fn datafusion_source_relation_required(resolved: &ResolvedSearch) -> Result<bool> {
    if matches!(
        datafusion_cluster_filter(resolved)?,
        DataFusionClusterFilter::Exact
    ) {
        if resolved.posting_encoding != PostingEncoding::Source {
            return Err(Error::InvalidArgument(
                "exact search cannot use stored IVF vectors".into(),
            ));
        }
        return Ok(true);
    }
    if resolved.posting_encoding == PostingEncoding::Source || resolved.filter.is_some() {
        return Ok(true);
    }
    Ok(resolved
        .projection
        .iter()
        .any(|column| !resolved.source_key_fields.contains(column)))
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum DataFusionClusterFilter {
    Exact,
    All,
    NativeInline,
    NativeRelation,
    Relational,
}

fn datafusion_cluster_filter(resolved: &ResolvedSearch) -> Result<DataFusionClusterFilter> {
    if resolved.postings_relation_key.is_none() {
        if resolved.nlist.is_some() || resolved.cluster_selection.is_some() {
            return Err(Error::InvalidArgument(
                "exact search cannot contain IVF cluster selection".into(),
            ));
        }
        return Ok(DataFusionClusterFilter::Exact);
    }
    let nlist = resolved
        .nlist
        .ok_or_else(|| Error::InvalidMetadata("indexed search is missing nlist".into()))?;
    match resolved.cluster_selection.as_ref().ok_or_else(|| {
        Error::InvalidArgument("indexed search is missing IVF cluster selection".into())
    })? {
        ClusterSelection::All => Ok(DataFusionClusterFilter::All),
        ClusterSelection::Native(selected_clusters) => {
            if selected_clusters.is_empty() || selected_clusters.len() >= nlist {
                return Err(Error::InvalidArgument(format!(
                    "native IVF routing must select between 1 and {} clusters",
                    nlist.saturating_sub(1)
                )));
            }
            let mut seen = HashSet::with_capacity(selected_clusters.len());
            for cid in selected_clusters {
                if *cid < 0
                    || usize::try_from(*cid).map_or(true, |cid| cid >= nlist)
                    || !seen.insert(*cid)
                {
                    return Err(Error::InvalidArgument(format!(
                        "selected IVF cluster IDs must be unique and in 0..{nlist}"
                    )));
                }
            }
            if selected_clusters.len() <= INLINE_CLUSTER_FILTER_LIMIT {
                Ok(DataFusionClusterFilter::NativeInline)
            } else {
                Ok(DataFusionClusterFilter::NativeRelation)
            }
        }
        ClusterSelection::Relational {
            centroids_relation_key,
            nprobe,
        } => {
            if centroids_relation_key.is_empty() {
                return Err(Error::InvalidArgument(
                    "relational IVF routing requires a centroid relation".into(),
                ));
            }
            if *nprobe == 0 || *nprobe >= nlist {
                return Err(Error::InvalidArgument(format!(
                    "relational IVF routing must probe between 1 and {} clusters",
                    nlist.saturating_sub(1)
                )));
            }
            Ok(DataFusionClusterFilter::Relational)
        }
    }
}

fn native_selected_clusters(resolved: &ResolvedSearch) -> Result<&[i32]> {
    let Some(ClusterSelection::Native(selected_clusters)) = &resolved.cluster_selection else {
        return Err(Error::InvalidArgument(
            "native cluster selection is not configured".into(),
        ));
    };
    Ok(selected_clusters)
}

pub(crate) fn use_native_cluster_routing(nlist: usize, dimension: usize) -> bool {
    nlist
        .checked_mul(dimension)
        .is_some_and(|values| values <= NATIVE_CLUSTER_ROUTING_MAX_VALUES)
}

fn quote_identifier(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

#[cfg(test)]
pub(crate) struct SearchInput<'a> {
    pub snapshot: &'a IndexSnapshot,
    pub source: &'a RecordBatch,
    pub centroids: &'a RecordBatch,
    pub postings: &'a RecordBatch,
    pub query: &'a [f32],
    pub nprobe: Option<usize>,
    pub limit: usize,
    pub projection: Option<&'a [String]>,
}

#[cfg(test)]
pub(crate) fn selected_cluster_ids(
    snapshot: &IndexSnapshot,
    centroids: &RecordBatch,
    query: &[f32],
    nprobe: Option<usize>,
) -> Result<Vec<i32>> {
    let (dimension, nlist, nprobe) = validated_cluster_search(snapshot, query, nprobe)?;
    let centroid_values = read_centroids(centroids, nlist, dimension)?;
    selected_cluster_ids_from_values(query, &centroid_values, nlist, dimension, nprobe)
}

#[cfg(test)]
pub(crate) fn selected_cluster_ids_from_values(
    query: &[f32],
    centroid_values: &[f32],
    nlist: usize,
    dimension: usize,
    nprobe: usize,
) -> Result<Vec<i32>> {
    let expected_values = nlist
        .checked_mul(dimension)
        .ok_or_else(|| Error::InvalidSchema("invalid cached centroid matrix shape".into()))?;
    if expected_values != centroid_values.len() {
        return Err(Error::InvalidSchema(
            "invalid cached centroid matrix shape".into(),
        ));
    }
    Ok(select_clusters(query, centroid_values, dimension, nprobe)
        .into_iter()
        .enumerate()
        .filter(|(_, selected)| *selected)
        .map(|(cid, _)| i32::try_from(cid).expect("nlist is bounded by int32"))
        .collect())
}

pub(crate) fn validated_cluster_search(
    snapshot: &IndexSnapshot,
    query: &[f32],
    nprobe: Option<usize>,
) -> Result<(usize, usize, usize)> {
    let dimension = snapshot.parameter_usize("dimension")?;
    let nlist = snapshot.parameter_usize("nlist")?;
    if query.len() != dimension || query.iter().any(|value| !value.is_finite()) {
        return Err(Error::InvalidArgument(format!(
            "query vector must contain exactly {dimension} finite float values"
        )));
    }
    let nprobe = nprobe.unwrap_or(nlist.min(20));
    if nprobe == 0 || nprobe > nlist {
        return Err(Error::InvalidArgument(format!(
            "nprobe must be in 1..={nlist}"
        )));
    }
    Ok((dimension, nlist, nprobe))
}

#[cfg(test)]
pub(crate) fn execute(input: &SearchInput<'_>) -> Result<(Vec<RecordBatch>, SchemaRef)> {
    if input.limit == 0 {
        return Err(Error::InvalidArgument("limit must be positive".into()));
    }
    let dimension = input.snapshot.parameter_usize("dimension")?;
    let nlist = input.snapshot.parameter_usize("nlist")?;
    let ntotal = input.snapshot.parameter_usize("ntotal")?;
    let selected_ids =
        selected_cluster_ids(input.snapshot, input.centroids, input.query, input.nprobe)?;
    let mut selected_clusters = vec![false; nlist];
    for cid in selected_ids {
        selected_clusters[usize::try_from(cid).expect("selected cid is non-negative")] = true;
    }

    if input.source.num_rows() != ntotal {
        return Err(Error::InvalidSchema(format!(
            "source row count {} does not match index ntotal {ntotal}",
            input.source.num_rows()
        )));
    }
    let (vectors, source_dimension) =
        borrow_source_vectors(input.source, &input.snapshot.vector_field)?;
    if source_dimension != dimension {
        return Err(Error::InvalidSchema(format!(
            "source vector dimension {source_dimension} does not match index dimension {dimension}"
        )));
    }
    let source_keys = source_key_arrays(input.source, &input.snapshot.source_key_fields)?;
    validate_unique_keys(&source_keys)?;

    let source_by_key = source_rows_by_key(&source_keys)?;
    let candidates = candidate_source_rows(
        input.postings,
        &selected_clusters,
        nlist,
        &source_keys,
        &source_by_key,
    )?;
    let posting_encoding = PostingEncoding::from_snapshot(input.snapshot)?;
    if posting_encoding != PostingEncoding::Source {
        return Err(Error::InvalidArgument(
            "the test-only reference executor does not support quantized postings".into(),
        ));
    }
    if input.postings.schema().field_with_name("vector").is_ok() {
        return Err(Error::InvalidSchema(
            "ivf_postings.vector is not part of IVF schema version 1".into(),
        ));
    }

    let mut hits = Vec::with_capacity(candidates.len());
    let distance_kernel = detect();
    for (_, source_row, _) in candidates {
        let start = source_row * dimension;
        let distance = distance_kernel.squared_l2(
            input.query,
            &vectors[start..start.saturating_add(dimension)],
        );
        if !distance.is_finite() {
            return Err(Error::InvalidArgument(
                "query produced a non-finite final distance".into(),
            ));
        }
        hits.push((distance, source_row));
    }
    hits.sort_unstable_by(|left, right| left.0.total_cmp(&right.0));
    hits.truncate(input.limit);
    build_result_table(input.source, &hits, input.projection)
}

#[cfg(test)]
fn build_result_table(
    source: &RecordBatch,
    hits: &[(f32, usize)],
    projection: Option<&[String]>,
) -> Result<(Vec<RecordBatch>, SchemaRef)> {
    let projected = match projection {
        Some(columns) => {
            if columns.is_empty() || columns.iter().collect::<HashSet<_>>().len() != columns.len() {
                return Err(Error::InvalidArgument(
                    "projection must contain unique source column names".into(),
                ));
            }
            columns
                .iter()
                .map(|name| {
                    source
                        .schema()
                        .index_of(name)
                        .map_err(|_| Error::InvalidArgument(format!("column not found: {name}")))
                })
                .collect::<Result<Vec<_>>>()?
        }
        None => (0..source.num_columns()).collect(),
    };
    let indices = UInt64Array::from_iter_values(hits.iter().map(|hit| hit.1 as u64));
    let mut arrays = projected
        .iter()
        .map(|&column| take(source.column(column), &indices, None))
        .collect::<std::result::Result<Vec<_>, _>>()?;
    arrays.push(Arc::new(Float32Array::from_iter_values(
        hits.iter().map(|hit| hit.0),
    )));
    let mut fields = projected
        .iter()
        .map(|&column| source.schema().field(column).clone())
        .collect::<Vec<_>>();
    fields.push(Field::new("_distance", DataType::Float32, false));
    let schema = Arc::new(Schema::new(fields));
    let batch = RecordBatch::try_new(Arc::clone(&schema), arrays)?;
    Ok((vec![batch], schema))
}

#[cfg(test)]
mod tests;
