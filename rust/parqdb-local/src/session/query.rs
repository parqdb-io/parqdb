//! Embedded query resolution and `DataFusion` plan construction.

use std::sync::Arc;

use arrow::array::{Array, ArrayRef, Int32Array};
use arrow::compute::concat_batches;
use arrow::datatypes::{Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::common::ScalarValue;
use datafusion::execution::FunctionRegistry;
use datafusion::execution::context::SQLOptions;
use datafusion::logical_expr::Expr;
use datafusion::prelude::{SessionContext, col, lit};
use parqdb_index::LoadedIndex;
use parqdb_meta::{DistanceMetric, PostingEncoding, RelationReference};
use uuid::Uuid;

use super::LocalSession;
use super::index_relation::IndexRelationLayout;
use super::source::{
    SourceBinding, exact_vector_field, relation_key, resolve_search_projection,
    validate_index_source_schema, vector_elements_are_f64,
};
use crate::query::{
    ManagedQueryStream, compile_datafusion_sql, compile_index_only_plan,
    datafusion_centroid_relation_required, datafusion_cluster_relation_required,
    datafusion_source_relation_required, use_native_cluster_routing, validated_cluster_search,
};
use crate::{ClusterSelection, Error, ResolvedSearch, Result, SearchRequest};

struct RegisteredSearchRelations {
    execution: ResolvedSearch,
    source: Option<String>,
    postings: Option<String>,
    centroids: Option<String>,
    selected_clusters: Option<String>,
}

impl RegisteredSearchRelations {
    fn deregister_temporary(&self, context: &SessionContext) -> Result<()> {
        for name in [
            self.postings.as_deref(),
            self.centroids.as_deref(),
            self.selected_clusters.as_deref(),
        ]
        .into_iter()
        .flatten()
        {
            context.deregister_table(name)?;
        }
        Ok(())
    }
}

impl LocalSession {
    /// Executes one IVF query and returns Arrow batches and their schema.
    pub async fn search(&self, request: &SearchRequest) -> Result<(Vec<RecordBatch>, SchemaRef)> {
        let plan = self.plan_search(request).await?;
        let schema = Arc::clone(plan.schema().inner());
        Ok((plan.collect().await?, schema))
    }

    /// Executes one search as a cancellable, admission-controlled Arrow stream.
    pub async fn stream_search(&self, request: &SearchRequest) -> Result<ManagedQueryStream> {
        let permit = self.runtime.admit_query().await?;
        let stream = self.plan_search(request).await?.execute_stream().await?;
        Ok(ManagedQueryStream::new(stream, permit))
    }

    /// Explains one search through the same cancellable, admission-controlled path.
    pub async fn stream_explain_search(
        &self,
        request: &SearchRequest,
        verbose: bool,
        analyze: bool,
    ) -> Result<ManagedQueryStream> {
        let permit = self.runtime.admit_query().await?;
        let stream = self
            .plan_search(request)
            .await?
            .explain(verbose, analyze)?
            .execute_stream()
            .await?;
        Ok(ManagedQueryStream::new(stream, permit))
    }

    /// Executes one read-only SQL statement as a cancellable, admission-controlled Arrow stream.
    pub async fn stream_sql(&self, sql: &str) -> Result<ManagedQueryStream> {
        let permit = self.runtime.admit_query().await?;
        let plan = self.context.state().create_logical_plan(sql).await?;
        SQLOptions::new()
            .with_allow_ddl(false)
            .with_allow_dml(false)
            .with_allow_statements(false)
            .verify_plan(&plan)
            .map_err(|error| {
                Error::InvalidArgument(format!("SQL execution is read-only: {error}"))
            })?;
        let stream = self
            .context
            .execute_logical_plan(plan)
            .await?
            .execute_stream()
            .await?;
        Ok(ManagedQueryStream::new(stream, permit))
    }

    /// Plans one IVF query in this session's `DataFusion` context.
    pub async fn plan_search(
        &self,
        request: &SearchRequest,
    ) -> Result<datafusion::dataframe::DataFrame> {
        let resolved = self.resolve_search(request).await?;
        self.datafusion_plan(&resolved).await
    }

    /// Compiles one search into executable SQL for this session's registered tables.
    pub async fn search_sql(&self, request: &SearchRequest) -> Result<String> {
        let resolved = self.resolve_search(request).await?;
        let mut execution = resolved.clone();
        let source_name = if datafusion_source_relation_required(&resolved)? {
            Some(
                self.source_binding(&resolved.source_relation_key)?
                    .ok_or_else(|| {
                        Error::InvalidArgument(
                            "resolved source is not bound to this session".into(),
                        )
                    })?
                    .table_name,
            )
        } else {
            None
        };
        let postings_name = match (&resolved.postings_relation_key, &resolved.cluster_selection) {
            (Some(key), Some(ClusterSelection::Native(cids))) => {
                execution.cluster_selection = Some(ClusterSelection::All);
                Some(self.register_sql_manifested_cids(key, cids).await?)
            }
            (Some(key), _) => Some(
                self.register_sql_relation("postings", key, IndexRelationLayout::ManifestedCid)
                    .await?,
            ),
            (None, _) => None,
        };
        let centroids_name = match &resolved.cluster_selection {
            Some(ClusterSelection::Relational {
                centroids_relation_key,
                ..
            }) => Some(
                self.register_sql_relation(
                    "centroids",
                    centroids_relation_key,
                    IndexRelationLayout::Plain,
                )
                .await?,
            ),
            _ => None,
        };
        compile_datafusion_sql(
            &execution,
            source_name.as_deref(),
            postings_name.as_deref(),
            centroids_name.as_deref(),
            None,
        )
    }

    /// Resolves one query without binding it to a `DataFusion` execution context.
    pub async fn resolve_search(&self, request: &SearchRequest) -> Result<ResolvedSearch> {
        let _guard = if request.bypass_index {
            None
        } else {
            Some(self.coordination.read()?)
        };
        validate_search_request(request)?;
        let source = self.bind_relation(&request.source).await?;
        let projection =
            resolve_search_projection(source.schema.as_ref(), request.projection.as_deref())?;

        if request.bypass_index {
            return resolve_exact_search(request, &source, projection);
        }
        self.resolve_indexed_search(request, &source, projection)
            .await
    }

    async fn resolve_indexed_search(
        &self,
        request: &SearchRequest,
        source: &SourceBinding,
        projection: Vec<String>,
    ) -> Result<ResolvedSearch> {
        let loaded = self
            .select_index_in(
                &request.index_namespace,
                &source.reference,
                request.index.as_deref(),
                request.column.as_deref(),
            )
            .await?;
        let snapshot = loaded.metadata.current_snapshot()?;
        validate_index_source_schema(source.schema.as_ref(), snapshot)?;
        let metric = DistanceMetric::from_metadata(&snapshot.metric).ok_or_else(|| {
            Error::InvalidMetadata(format!("unsupported IVF metric: {}", snapshot.metric))
        })?;
        let query = crate::vector::transform_query(&request.query, metric)?;
        let centroid_artifact = self.indexes.load_snapshot_ivf_centroids(&loaded).await?;
        let postings_relation = index_relation(&self.warehouse, &loaded, "ivf_postings")?;
        let postings_relation_key = relation_key(&postings_relation);
        let centroid_cache_key = format!(
            "{}\0ivf_centroids",
            centroid_artifact.entry.metadata_location
        );
        let (cluster_selection, nlist) = self
            .resolve_cluster_selection(&loaded, &centroid_cache_key, &query, request.nprobe)
            .await?;
        Ok(ResolvedSearch {
            source_relation_key: source.key.clone(),
            query,
            metric,
            vector_field: snapshot.vector_field.clone(),
            source_vector_is_f64: source_vector_uses_f64(
                source.schema.as_ref(),
                &snapshot.vector_field,
            )?,
            source_key_fields: snapshot.source_key_fields.clone(),
            postings_relation_key: Some(postings_relation_key),
            posting_encoding: PostingEncoding::from_snapshot(snapshot)?,
            cluster_selection: Some(cluster_selection),
            nlist: Some(nlist),
            ntotal: Some(snapshot.parameter_usize("ntotal")?),
            projection,
            filter: request.filter.clone(),
            limit: request.limit,
        })
    }

    /// Returns the optimized logical and physical plans for one IVF query.
    pub async fn explain_search(&self, request: &SearchRequest, verbose: bool) -> Result<String> {
        let plan = self.plan_search(request).await?;
        let logical = plan.clone().into_optimized_plan()?;
        let physical = plan.create_physical_plan().await?;
        let logical = if verbose {
            logical.display_indent_schema().to_string()
        } else {
            logical.display_indent().to_string()
        };
        let display =
            datafusion::physical_plan::displayable(physical.as_ref()).set_show_schema(verbose);
        let physical = display.indent(verbose);
        Ok(format!(
            "logical_plan\n{logical}\nphysical_plan\n{physical}"
        ))
    }

    async fn datafusion_plan(
        &self,
        resolved: &ResolvedSearch,
    ) -> Result<datafusion::dataframe::DataFrame> {
        let context = self.parquet.context();
        if let Some(postings) = self.index_only_postings(resolved).await? {
            return compile_index_only_plan(postings, resolved);
        }

        let registered = self.register_search_relations(resolved, &context).await?;
        let result = compile_registered_search(
            &context,
            &registered.execution,
            registered.source.as_deref(),
            registered.postings.as_deref(),
            registered.centroids.as_deref(),
            registered.selected_clusters.as_deref(),
        )
        .await;
        let cleanup = registered.deregister_temporary(&context);
        let plan = result?;
        cleanup?;
        Ok(plan)
    }

    async fn register_search_relations(
        &self,
        resolved: &ResolvedSearch,
        context: &SessionContext,
    ) -> Result<RegisteredSearchRelations> {
        let mut execution = resolved.clone();
        let token = Uuid::new_v4().simple().to_string();
        let source = if datafusion_source_relation_required(resolved)? {
            Some(
                self.source_binding(&resolved.source_relation_key)?
                    .ok_or_else(|| {
                        Error::InvalidArgument(
                            "resolved source is not bound to this session".into(),
                        )
                    })?
                    .table_name,
            )
        } else {
            None
        };
        let postings = resolved
            .postings_relation_key
            .as_ref()
            .map(|_| format!("parqdb_postings_{token}"));
        if let (Some(key), Some(name)) = (&resolved.postings_relation_key, &postings) {
            let dataframe = match &resolved.cluster_selection {
                Some(ClusterSelection::Native(cids)) => {
                    execution.cluster_selection = Some(ClusterSelection::All);
                    select_clusters(
                        self.index_relation_dataframe_for_cids(key, cids).await?,
                        cids,
                    )?
                }
                _ => {
                    self.index_relation_dataframe(key, IndexRelationLayout::ManifestedCid)
                        .await?
                }
            };
            context.register_table(name.clone(), dataframe.into_view())?;
        }
        let centroids = if datafusion_centroid_relation_required(&execution)? {
            let name = format!("parqdb_centroids_{token}");
            let Some(ClusterSelection::Relational {
                centroids_relation_key,
                ..
            }) = &resolved.cluster_selection
            else {
                unreachable!("validated relational cluster selection")
            };
            context.register_table(
                name.clone(),
                self.index_relation_dataframe(centroids_relation_key, IndexRelationLayout::Plain)
                    .await?
                    .into_view(),
            )?;
            Some(name)
        } else {
            None
        };
        let selected_clusters = if datafusion_cluster_relation_required(&execution)? {
            let name = format!("parqdb_selected_clusters_{token}");
            let Some(ClusterSelection::Native(selected_clusters)) = &resolved.cluster_selection
            else {
                unreachable!("validated native cluster selection")
            };
            let batch = RecordBatch::try_from_iter([(
                "cid",
                Arc::new(Int32Array::from(selected_clusters.clone())) as ArrayRef,
            )])?;
            context.register_batch(name.clone(), batch)?;
            Some(name)
        } else {
            None
        };
        Ok(RegisteredSearchRelations {
            execution,
            source,
            postings,
            centroids,
            selected_clusters,
        })
    }

    async fn read_index_relation(&self, loaded: &LoadedIndex, role: &str) -> Result<RecordBatch> {
        let reference = index_relation(&self.warehouse, loaded, role)?;
        let dataframe = self
            .index_relation_dataframe(
                &relation_key(&reference),
                if role == "ivf_postings" {
                    IndexRelationLayout::ManifestedCid
                } else {
                    IndexRelationLayout::Plain
                },
            )
            .await?;
        let schema = Arc::clone(dataframe.schema().inner());
        Ok(concat_batches(&schema, &dataframe.collect().await?)?)
    }

    async fn index_relation_dataframe(
        &self,
        key: &str,
        layout: IndexRelationLayout,
    ) -> Result<datafusion::dataframe::DataFrame> {
        let provider = self
            .index_relation_providers
            .get_or_create_parquet(key, layout, &self.context.state())
            .await?;
        Ok(self.context.read_table(provider)?)
    }

    async fn index_relation_dataframe_for_cids(
        &self,
        key: &str,
        cids: &[i32],
    ) -> Result<datafusion::dataframe::DataFrame> {
        let provider = self
            .index_relation_providers
            .manifested_cid_provider(key, cids, &self.context.state())
            .await?;
        Ok(self.context.read_table(provider)?)
    }

    async fn register_sql_relation(
        &self,
        role: &str,
        relation_key: &str,
        layout: IndexRelationLayout,
    ) -> Result<String> {
        let key = format!("{role}\0{relation_key}");
        if let Some(name) = self
            .sql_relations
            .read()
            .map_err(|_| sql_relation_lock_error())?
            .get(&key)
            .cloned()
        {
            return Ok(name);
        }

        let provider = self
            .index_relation_providers
            .deferred_parquet_provider(relation_key, layout, &self.context.state())
            .await?;
        let identifier = Uuid::new_v5(&Uuid::NAMESPACE_URL, key.as_bytes());
        let name = format!("__parqdb_{role}_{}", identifier.simple());
        let mut relations = self
            .sql_relations
            .write()
            .map_err(|_| sql_relation_lock_error())?;
        if let Some(existing) = relations.get(&key) {
            return Ok(existing.clone());
        }
        self.context.register_table(name.clone(), provider)?;
        relations.insert(key, name.clone());
        Ok(name)
    }

    async fn register_sql_manifested_cids(&self, key: &str, cids: &[i32]) -> Result<String> {
        let mut identity = Vec::with_capacity(
            key.len()
                .saturating_add(cids.len().saturating_mul(4).saturating_add(1)),
        );
        identity.extend_from_slice(key.as_bytes());
        identity.push(0);
        for cid in cids {
            identity.extend_from_slice(&cid.to_le_bytes());
        }
        let token = Uuid::new_v5(&Uuid::NAMESPACE_OID, &identity).simple();
        let name = format!("parqdb_postings_selected_{token}");
        let cache_key = format!("postings-selected\0{token}");
        if let Some(existing) = self
            .sql_relations
            .read()
            .map_err(|_| sql_relation_lock_error())?
            .get(&cache_key)
            .cloned()
        {
            return Ok(existing);
        }
        let dataframe = select_clusters(
            self.index_relation_dataframe_for_cids(key, cids).await?,
            cids,
        )?;
        let mut relations = self
            .sql_relations
            .write()
            .map_err(|_| sql_relation_lock_error())?;
        if let Some(existing) = relations.get(&cache_key) {
            return Ok(existing.clone());
        }
        self.context
            .register_table(name.clone(), dataframe.into_view())?;
        relations.insert(cache_key, name.clone());
        Ok(name)
    }

    async fn index_only_postings(
        &self,
        resolved: &ResolvedSearch,
    ) -> Result<Option<datafusion::dataframe::DataFrame>> {
        if datafusion_source_relation_required(resolved)? {
            return Ok(None);
        }
        let Some(key) = &resolved.postings_relation_key else {
            return Ok(None);
        };
        match &resolved.cluster_selection {
            Some(ClusterSelection::Native(cids)) => Ok(Some(select_clusters(
                self.index_relation_dataframe_for_cids(key, cids).await?,
                cids,
            )?)),
            Some(ClusterSelection::All) => Ok(Some(
                self.index_relation_dataframe(key, IndexRelationLayout::ManifestedCid)
                    .await?,
            )),
            Some(ClusterSelection::Relational { .. }) | None => Ok(None),
        }
    }

    async fn resolve_cluster_selection(
        &self,
        loaded: &LoadedIndex,
        cache_key: &str,
        query: &[f32],
        requested_nprobe: Option<usize>,
    ) -> Result<(ClusterSelection, usize)> {
        let snapshot = loaded.metadata.current_snapshot()?;
        let (dimension, nlist, nprobe) =
            validated_cluster_search(snapshot, query, requested_nprobe)?;
        let selection = if nprobe == nlist {
            ClusterSelection::All
        } else if use_native_cluster_routing(nlist, dimension) {
            let centroids = self
                .index_relation_providers
                .get_or_load_centroids(cache_key, || async {
                    let centroids = self.read_index_relation(loaded, "ivf_centroids").await?;
                    let values = crate::ivf::read_centroids(&centroids, nlist, dimension)?;
                    super::index_relation::CentroidNavigator::new(
                        nlist,
                        dimension,
                        Arc::from(values),
                    )
                })
                .await?;
            centroids.validate_shape(nlist, dimension)?;
            let selected = centroids
                .route(query, nprobe)?
                .into_iter()
                .map(|cid| {
                    i32::try_from(cid).map_err(|_| {
                        Error::InvalidSchema("centroid cid exceeds the INT32 index domain".into())
                    })
                })
                .collect::<Result<Vec<_>>>()?;
            ClusterSelection::Native(selected)
        } else {
            let centroids_relation_key =
                relation_key(&index_relation(&self.warehouse, loaded, "ivf_centroids")?);
            ClusterSelection::Native(
                self.route_centroids_with_datafusion(&centroids_relation_key, query, nprobe)
                    .await?,
            )
        };
        Ok((selection, nlist))
    }

    pub(super) async fn route_centroids_with_datafusion(
        &self,
        relation_key: &str,
        query: &[f32],
        nprobe: usize,
    ) -> Result<Vec<i32>> {
        let query = query
            .iter()
            .copied()
            .map(|value| ScalarValue::Float32(Some(value)))
            .collect::<Vec<_>>();
        let query = Expr::Literal(
            ScalarValue::List(ScalarValue::new_list(
                &query,
                &arrow::datatypes::DataType::Float32,
                false,
            )),
            None,
        );
        let distance = self.context.udf("parqdb_squared_l2")?;
        let batches = self
            .index_relation_dataframe(relation_key, IndexRelationLayout::Plain)
            .await?
            .select(vec![
                col("cid"),
                distance
                    .call(vec![col("centroid"), query])
                    .alias("__parqdb_distance"),
            ])?
            .sort(vec![
                col("__parqdb_distance").sort(true, false),
                col("cid").sort(true, false),
            ])?
            .limit(0, Some(nprobe))?
            .select_columns(&["cid"])?
            .collect()
            .await?;
        let mut selected = Vec::with_capacity(nprobe);
        for batch in batches {
            let cids = batch
                .column(0)
                .as_any()
                .downcast_ref::<Int32Array>()
                .ok_or_else(|| Error::InvalidSchema("ivf_centroids.cid must be INT32".into()))?;
            if cids.null_count() != 0 {
                return Err(Error::InvalidSchema(
                    "ivf_centroids.cid must be required".into(),
                ));
            }
            selected.extend(cids.values().iter().copied());
        }
        if selected.len() != nprobe {
            return Err(Error::InvalidSchema(format!(
                "ivf_centroids returned {} rows for nprobe {nprobe}",
                selected.len()
            )));
        }
        Ok(selected)
    }
}

fn validate_search_request(request: &SearchRequest) -> Result<()> {
    if request.limit == 0 {
        return Err(Error::InvalidArgument("limit must be positive".into()));
    }
    if request
        .filter
        .as_ref()
        .is_some_and(|filter| filter.trim().is_empty())
    {
        return Err(Error::InvalidArgument(
            "filter must not be empty".to_owned(),
        ));
    }
    Ok(())
}

fn resolve_exact_search(
    request: &SearchRequest,
    source: &SourceBinding,
    projection: Vec<String>,
) -> Result<ResolvedSearch> {
    if request.index.is_some() {
        return Err(Error::InvalidArgument(
            "index cannot be set when bypassing the vector index".into(),
        ));
    }
    if request.nprobe.is_some() {
        return Err(Error::InvalidArgument(
            "nprobes cannot be set when bypassing the vector index".into(),
        ));
    }
    let query = crate::vector::transform_query(&request.query, DistanceMetric::L2Squared)?;
    let vector_field = exact_vector_field(source.schema.as_ref(), request.column.as_deref())?;
    Ok(ResolvedSearch {
        source_relation_key: source.key.clone(),
        query,
        metric: DistanceMetric::L2Squared,
        source_vector_is_f64: source_vector_uses_f64(source.schema.as_ref(), &vector_field)?,
        vector_field,
        source_key_fields: Vec::new(),
        postings_relation_key: None,
        posting_encoding: PostingEncoding::Source,
        cluster_selection: None,
        nlist: None,
        ntotal: None,
        projection,
        filter: request.filter.clone(),
        limit: request.limit,
    })
}

fn source_vector_uses_f64(schema: &Schema, vector_field: &str) -> Result<bool> {
    let field = schema
        .field_with_name(vector_field)
        .map_err(|_| Error::InvalidSchema(format!("vector column not found: {vector_field}")))?;
    Ok(vector_elements_are_f64(field.data_type()))
}

fn select_clusters(
    dataframe: datafusion::dataframe::DataFrame,
    cids: &[i32],
) -> Result<datafusion::dataframe::DataFrame> {
    Ok(dataframe
        .filter(col("cid").in_list(cids.iter().copied().map(lit).collect::<Vec<_>>(), false))?)
}

async fn compile_registered_search(
    context: &SessionContext,
    resolved: &ResolvedSearch,
    source_name: Option<&str>,
    postings_name: Option<&str>,
    centroids_name: Option<&str>,
    selected_clusters_name: Option<&str>,
) -> Result<datafusion::dataframe::DataFrame> {
    if let Some(filter) = &resolved.filter {
        let source = source_name.ok_or_else(|| {
            Error::InvalidArgument("a filtered search requires the source relation".into())
        })?;
        let source = quote_identifier(source);
        context
            .sql(&format!("SELECT * FROM {source} WHERE ({filter}) LIMIT 0"))
            .await
            .map_err(|error| Error::InvalidArgument(format!("invalid filter: {error}")))?;
    }
    let sql = compile_datafusion_sql(
        resolved,
        source_name,
        postings_name,
        centroids_name,
        selected_clusters_name,
    )?;
    Ok(context.sql(&sql).await?)
}

fn quote_identifier(identifier: &str) -> String {
    format!("\"{}\"", identifier.replace('"', "\"\""))
}

fn index_relation(
    warehouse: &parqdb_storage::Warehouse,
    loaded: &LoadedIndex,
    role: &str,
) -> Result<RelationReference> {
    let relative = loaded
        .metadata
        .current_snapshot()?
        .index_relations
        .get(role)
        .ok_or_else(|| Error::InvalidMetadata(format!("missing relation role: {role}")))?;
    Ok(RelationReference::Parquet {
        uri: warehouse.location(relative, relative.ends_with('/'))?,
    })
}

fn sql_relation_lock_error() -> Error {
    Error::InvalidArgument("SQL relation lock is poisoned".into())
}
