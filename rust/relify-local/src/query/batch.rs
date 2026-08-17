//! Relational reference plan for index-only batch IVF search.

use relify_meta::{DistanceMetric, PostingEncoding};

use crate::{Error, Result};

mod exec;
mod routes;
mod selector;

pub(crate) use exec::{BatchIvfTopKExec, BatchLvqInput, BatchTopKMergeExec};
pub(crate) use routes::BatchRoutes;
pub(crate) use selector::BatchCandidateSelector;

pub(crate) const QUERY_ID_COLUMN: &str = "_query_id";

/// Builds the fused physical path over a postings relation already pruned to
/// the union of all routed clusters.
pub(crate) async fn compile_index_only_batch_plan(
    postings: datafusion::dataframe::DataFrame,
    routes: BatchRoutes,
    posting_encoding: PostingEncoding,
    metric: DistanceMetric,
    source_key_fields: &[String],
    limit: usize,
) -> Result<std::sync::Arc<dyn datafusion::physical_plan::ExecutionPlan>> {
    use std::sync::Arc;

    use datafusion::physical_plan::coalesce_partitions::CoalescePartitionsExec;
    use datafusion::prelude::col;
    use relify_kernels::LvqBits;

    if source_key_fields.is_empty() {
        return Err(Error::InvalidMetadata(
            "batch index-only search requires at least one source key".into(),
        ));
    }
    let bits = match posting_encoding {
        PostingEncoding::Lvq4 => LvqBits::Four,
        PostingEncoding::Lvq8 => LvqBits::Eight,
        PostingEncoding::Source => {
            return Err(Error::InvalidArgument(
                "source-backed postings cannot execute an index-only batch search".into(),
            ));
        }
    };
    let mut projection = vec![col("cid")];
    projection.extend(
        source_key_fields
            .iter()
            .enumerate()
            .map(|(position, output)| col(format!("key_{}", position + 1)).alias(output)),
    );
    projection.extend([col("code"), col("offset"), col("scale")]);
    let input = postings.select(projection)?.create_physical_plan().await?;
    let key_count = source_key_fields.len();
    let local: Arc<dyn datafusion::physical_plan::ExecutionPlan> =
        Arc::new(BatchIvfTopKExec::try_new(
            input,
            routes.clone(),
            BatchLvqInput {
                bits,
                cid_index: 0,
                code_index: key_count + 1,
                offset_index: key_count + 2,
                scale_index: key_count + 3,
            },
            (1..=key_count).collect(),
            source_key_fields.to_vec(),
            limit,
            metric,
        )?);
    let coalesced: Arc<dyn datafusion::physical_plan::ExecutionPlan> =
        Arc::new(CoalescePartitionsExec::new(local));
    Ok(Arc::new(BatchTopKMergeExec::try_new(
        coalesced,
        routes.query_count(),
        limit,
    )?))
}

/// Compiles the correctness baseline for a batch of routed IVF queries.
///
/// `queries` contains one row per query (`_query_id`, `query`), while `routes`
/// contains (`_query_id`, `cid`) pairs. The postings input should already be
/// pruned to the distinct CIDs present in `routes`.
#[cfg(test)]
pub(crate) fn compile_index_only_batch_sql(
    posting_encoding: PostingEncoding,
    metric: DistanceMetric,
    source_key_fields: &[String],
    postings_name: &str,
    queries_name: &str,
    routes_name: &str,
    limit: usize,
) -> Result<String> {
    if limit == 0 {
        return Err(Error::InvalidArgument(
            "batch limit must be positive".into(),
        ));
    }
    if source_key_fields.is_empty() {
        return Err(Error::InvalidMetadata(
            "batch index-only search requires at least one source key".into(),
        ));
    }
    for (role, name) in [
        ("postings", postings_name),
        ("queries", queries_name),
        ("routes", routes_name),
    ] {
        if name.is_empty() {
            return Err(Error::InvalidArgument(format!(
                "batch {role} relation name must not be empty"
            )));
        }
    }

    let distance = match posting_encoding {
        PostingEncoding::Lvq4 => {
            "relify_lvq4_l2(p.\"code\", p.\"offset\", p.\"scale\", q.\"query\")"
        }
        PostingEncoding::Lvq8 => {
            "relify_lvq8_l2(p.\"code\", p.\"offset\", p.\"scale\", q.\"query\")"
        }
        PostingEncoding::Source => {
            return Err(Error::InvalidArgument(
                "source-backed postings cannot execute an index-only batch search".into(),
            ));
        }
    };
    let postings = super::quote_identifier(postings_name);
    let queries = super::quote_identifier(queries_name);
    let routes = super::quote_identifier(routes_name);
    let keys = source_key_fields
        .iter()
        .enumerate()
        .map(|(position, output)| {
            format!(
                "p.{} AS {}",
                super::quote_identifier(&format!("key_{}", position + 1)),
                super::quote_identifier(output)
            )
        })
        .collect::<Vec<_>>();
    let mut candidate_columns = vec![format!("q.{}", super::quote_identifier(QUERY_ID_COLUMN))];
    candidate_columns.extend(keys);
    candidate_columns.push(format!("{distance} AS \"_distance\""));

    let mut output = vec![format!("r.{}", super::quote_identifier(QUERY_ID_COLUMN))];
    output.extend(
        source_key_fields
            .iter()
            .map(|field| format!("r.{}", super::quote_identifier(field))),
    );
    output.push(if metric == DistanceMetric::Cosine {
        "r.\"_distance\" * CAST(0.5 AS REAL) AS \"_distance\"".into()
    } else {
        "r.\"_distance\"".into()
    });

    Ok(format!(
        "WITH\n    relify_batch_candidates AS (\n        SELECT {}\n        \
         FROM {postings} AS p\n        JOIN {routes} AS route\n          \
         ON p.\"cid\" = route.\"cid\"\n        JOIN {queries} AS q\n          \
         ON route.{} = q.{}\n    ),\n    relify_batch_ranked AS (\n        \
         SELECT c.*, ROW_NUMBER() OVER (\n            PARTITION BY c.{}\n            \
         ORDER BY c.\"_distance\" ASC\n        ) AS \"__relify_rank\"\n        \
         FROM relify_batch_candidates AS c\n    )\nSELECT {}\nFROM relify_batch_ranked AS r\n\
         WHERE r.\"__relify_rank\" <= {limit}\nORDER BY r.{} ASC, r.\"_distance\" ASC",
        candidate_columns.join(", "),
        super::quote_identifier(QUERY_ID_COLUMN),
        super::quote_identifier(QUERY_ID_COLUMN),
        super::quote_identifier(QUERY_ID_COLUMN),
        output.join(", "),
        super::quote_identifier(QUERY_ID_COLUMN),
    ))
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use arrow::array::{
        Array, BinaryArray, FixedSizeListArray, Float32Array, Int32Array, StringArray,
    };
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;
    use datafusion::execution::runtime_env::RuntimeEnv;
    use datafusion::prelude::SessionConfig;
    use relify_kernels::{LvqBits, encode_lvq_rows};

    use super::*;

    fn queries() -> RecordBatch {
        let values = Float32Array::from(vec![0.0, 0.0, 10.0, 0.0]);
        let query = FixedSizeListArray::try_new(
            Arc::new(Field::new("element", DataType::Float32, false)),
            2,
            Arc::new(values),
            None,
        )
        .unwrap();
        RecordBatch::try_from_iter([
            (QUERY_ID_COLUMN, Arc::new(Int32Array::from(vec![0, 1])) as _),
            ("query", Arc::new(query) as _),
        ])
        .unwrap()
    }

    fn routes() -> RecordBatch {
        RecordBatch::try_from_iter([
            (QUERY_ID_COLUMN, Arc::new(Int32Array::from(vec![0, 1])) as _),
            ("cid", Arc::new(Int32Array::from(vec![0, 0])) as _),
        ])
        .unwrap()
    }

    fn postings() -> RecordBatch {
        let encoded = encode_lvq_rows(
            &[0.0, 0.0, 1.0, 0.0, 10.0, 0.0, 11.0, 0.0],
            2,
            LvqBits::Eight,
        )
        .unwrap();
        let code_size = LvqBits::Eight.code_size(2);
        let codes = encoded.codes().chunks_exact(code_size).collect::<Vec<_>>();
        RecordBatch::try_new(
            Arc::new(Schema::new(vec![
                Field::new("cid", DataType::Int32, false),
                Field::new("key_1", DataType::Utf8, false),
                Field::new("code", DataType::Binary, false),
                Field::new("offset", DataType::Float32, false),
                Field::new("scale", DataType::Float32, false),
            ])),
            vec![
                Arc::new(Int32Array::from(vec![0, 0, 0, 0])),
                Arc::new(StringArray::from(vec!["a", "b", "c", "d"])),
                Arc::new(BinaryArray::from_vec(codes)),
                Arc::new(Float32Array::from(encoded.offsets().to_vec())),
                Arc::new(Float32Array::from(encoded.scales().to_vec())),
            ],
        )
        .unwrap()
    }

    #[tokio::test]
    async fn relational_batch_returns_top_k_per_query() {
        let context = super::super::relify_session_context(
            SessionConfig::new(),
            Arc::new(RuntimeEnv::default()),
        );
        context.register_batch("postings", postings()).unwrap();
        context.register_batch("queries", queries()).unwrap();
        context.register_batch("routes", routes()).unwrap();
        let sql = compile_index_only_batch_sql(
            PostingEncoding::Lvq8,
            DistanceMetric::L2Squared,
            &["id".into()],
            "postings",
            "queries",
            "routes",
            2,
        )
        .unwrap();

        let batches = context.sql(&sql).await.unwrap().collect().await.unwrap();
        let result = arrow::compute::concat_batches(&batches[0].schema(), &batches).unwrap();
        let query_ids = result
            .column(0)
            .as_any()
            .downcast_ref::<Int32Array>()
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

    #[test]
    fn relational_batch_rejects_source_backed_postings() {
        assert!(
            compile_index_only_batch_sql(
                PostingEncoding::Source,
                DistanceMetric::L2Squared,
                &["id".into()],
                "postings",
                "queries",
                "routes",
                10,
            )
            .is_err()
        );
    }
}
