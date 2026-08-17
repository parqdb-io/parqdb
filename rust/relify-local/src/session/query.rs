//! Embedded query resolution and `DataFusion` plan construction.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array};
use arrow::compute::concat_batches;
use arrow::datatypes::{Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use datafusion::execution::context::SQLOptions;
use datafusion::physical_plan::{ExecutionPlan, collect, execute_stream};
use datafusion::prelude::{SessionContext, col, lit};
use relify_meta::{DistanceMetric, IndexSnapshot, PostingEncoding, RelationReference};
use uuid::Uuid;

use super::LocalSession;
use super::index_relation::IndexRelationLayout;
use super::source::{
    SourceBinding, exact_vector_field, relation_key, resolve_search_projection,
    validate_index_source_schema, vector_elements_are_f64,
};
use crate::query::batch::{BatchRoutes, compile_index_only_batch_plan};
use crate::query::{
    ManagedQueryStream, compile_datafusion_sql, compile_index_only_plan,
    datafusion_centroid_relation_required, datafusion_cluster_relation_required,
    datafusion_source_relation_required, use_native_cluster_routing, validated_cluster_search,
};
use crate::{BatchSearchRequest, ClusterSelection, Error, ResolvedSearch, Result, SearchRequest};

struct RegisteredSearchRelations {
    execution: ResolvedSearch,
    source: Option<String>,
    postings: Option<String>,
    centroids: Option<String>,
    selected_clusters: Option<String>,
}

struct ResolvedBatchSearch {
    routes: BatchRoutes,
    postings_relation_key: String,
    posting_encoding: PostingEncoding,
    metric: DistanceMetric,
    source_key_fields: Vec<String>,
    nlist: usize,
    limit: usize,
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

    /// Executes an index-only batch search and returns source keys and distances per query.
    pub async fn batch_search(
        &self,
        request: &BatchSearchRequest,
    ) -> Result<(Vec<RecordBatch>, SchemaRef)> {
        let plan = self.plan_batch_search(request).await?;
        let schema = plan.schema();
        Ok((collect(plan, self.context.task_ctx()).await?, schema))
    }

    /// Executes an index-only batch search as one admission-controlled Arrow stream.
    pub async fn stream_batch_search(
        &self,
        request: &BatchSearchRequest,
    ) -> Result<ManagedQueryStream> {
        let permit = self.runtime.admit_query().await?;
        let plan = self.plan_batch_search(request).await?;
        let stream = execute_stream(plan, self.context.task_ctx())?;
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

    /// Plans one index-only batch search over a single postings scan.
    pub async fn plan_batch_search(
        &self,
        request: &BatchSearchRequest,
    ) -> Result<Arc<dyn ExecutionPlan>> {
        let resolved = self.resolve_batch_search(request).await?;
        let cids = resolved.routes.distinct_cids().collect::<Vec<_>>();
        let postings = self
            .index_relation_dataframe(
                &resolved.postings_relation_key,
                IndexRelationLayout::HiveCid,
            )
            .await?;
        let postings = if cids.len() < resolved.nlist {
            select_clusters(postings, &cids)?
        } else {
            postings
        };
        compile_index_only_batch_plan(
            postings,
            resolved.routes,
            resolved.posting_encoding,
            resolved.metric,
            &resolved.source_key_fields,
            resolved.limit,
        )
        .await
    }

    /// Compiles one search into executable SQL for this session's registered tables.
    pub async fn search_sql(&self, request: &SearchRequest) -> Result<String> {
        let resolved = self.resolve_search(request).await?;
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
        let postings_name = match &resolved.postings_relation_key {
            Some(key) => Some(
                self.register_sql_relation("postings", key, IndexRelationLayout::HiveCid)
                    .await?,
            ),
            None => None,
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
            &resolved,
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

    async fn resolve_batch_search(
        &self,
        request: &BatchSearchRequest,
    ) -> Result<ResolvedBatchSearch> {
        let _guard = self.coordination.read()?;
        validate_batch_search_request(request)?;
        let source = self.bind_relation(&request.source).await?;
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
        let posting_encoding = PostingEncoding::from_snapshot(snapshot)?;
        if posting_encoding == PostingEncoding::Source {
            return Err(Error::InvalidArgument(
                "batch search currently requires an LVQ4 or LVQ8 index".into(),
            ));
        }
        let centroid_artifact = self.indexes.load_snapshot_ivf_centroids(snapshot).await?;
        let centroid_cache_key = format!(
            "{}\0ivf_centroids",
            centroid_artifact.entry.metadata_location
        );
        let mut queries = Vec::with_capacity(request.queries.len());
        let mut cids_by_query = Vec::with_capacity(request.queries.len());
        let mut nlist = None;
        for query in &request.queries {
            let query = crate::vector::transform_query(query, metric)?;
            let (selection, query_nlist) = self
                .resolve_cluster_selection(snapshot, &centroid_cache_key, &query, request.nprobe)
                .await?;
            let cids = match selection {
                ClusterSelection::Native(cids) => cids,
                ClusterSelection::All => (0..query_nlist)
                    .map(|cid| {
                        i32::try_from(cid).map_err(|_| {
                            Error::InvalidSchema(
                                "centroid cid exceeds the INT32 index domain".into(),
                            )
                        })
                    })
                    .collect::<Result<Vec<_>>>()?,
                ClusterSelection::Relational { .. } => {
                    return Err(Error::InvalidArgument(
                        "batch search currently requires native IVF routing".into(),
                    ));
                }
            };
            queries.push(query);
            cids_by_query.push(cids);
            nlist = Some(query_nlist);
        }
        Ok(ResolvedBatchSearch {
            routes: BatchRoutes::try_new(queries, cids_by_query)?,
            postings_relation_key: relation_key(index_relation(snapshot, "ivf_postings")?),
            posting_encoding,
            metric,
            source_key_fields: snapshot.source_key_fields.clone(),
            nlist: nlist.expect("validated non-empty batch"),
            limit: request.limit,
        })
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
        let centroid_artifact = self.indexes.load_snapshot_ivf_centroids(snapshot).await?;
        let postings_relation_key = relation_key(index_relation(snapshot, "ivf_postings")?);
        let centroid_cache_key = format!(
            "{}\0ivf_centroids",
            centroid_artifact.entry.metadata_location
        );
        let (cluster_selection, nlist) = self
            .resolve_cluster_selection(snapshot, &centroid_cache_key, &query, request.nprobe)
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
            .map(|_| format!("relify_postings_{token}"));
        if let (Some(key), Some(name)) = (&resolved.postings_relation_key, &postings) {
            let dataframe = match &resolved.cluster_selection {
                Some(ClusterSelection::Native(cids)) => {
                    execution.cluster_selection = Some(ClusterSelection::All);
                    select_clusters(
                        self.index_relation_dataframe(key, IndexRelationLayout::HiveCid)
                            .await?,
                        cids,
                    )?
                }
                _ => {
                    self.index_relation_dataframe(key, IndexRelationLayout::HiveCid)
                        .await?
                }
            };
            context.register_table(name.clone(), dataframe.into_view())?;
        }
        let centroids = if datafusion_centroid_relation_required(&execution)? {
            let name = format!("relify_centroids_{token}");
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
            let name = format!("relify_selected_clusters_{token}");
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

    async fn read_index_relation(
        &self,
        snapshot: &IndexSnapshot,
        role: &str,
    ) -> Result<RecordBatch> {
        let reference = index_relation(snapshot, role)?;
        let dataframe = self
            .index_relation_dataframe(
                &relation_key(reference),
                if role == "ivf_postings" {
                    IndexRelationLayout::HiveCid
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
        let name = format!("__relify_{role}_{}", identifier.simple());
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
                self.index_relation_dataframe(key, IndexRelationLayout::HiveCid)
                    .await?,
                cids,
            )?)),
            Some(ClusterSelection::All) => Ok(Some(
                self.index_relation_dataframe(key, IndexRelationLayout::HiveCid)
                    .await?,
            )),
            Some(ClusterSelection::Relational { .. }) | None => Ok(None),
        }
    }

    async fn resolve_cluster_selection(
        &self,
        snapshot: &IndexSnapshot,
        cache_key: &str,
        query: &[f32],
        requested_nprobe: Option<usize>,
    ) -> Result<(ClusterSelection, usize)> {
        let (dimension, nlist, nprobe) =
            validated_cluster_search(snapshot, query, requested_nprobe)?;
        let selection = if nprobe == nlist {
            ClusterSelection::All
        } else if use_native_cluster_routing(nlist, dimension) {
            let centroids = self
                .index_relation_providers
                .get_or_load_centroids(cache_key, || async {
                    let centroids = self.read_index_relation(snapshot, "ivf_centroids").await?;
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
            ClusterSelection::Relational {
                centroids_relation_key: relation_key(index_relation(snapshot, "ivf_centroids")?),
                nprobe,
            }
        };
        Ok((selection, nlist))
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

fn validate_batch_search_request(request: &BatchSearchRequest) -> Result<()> {
    if request.limit == 0 {
        return Err(Error::InvalidArgument("limit must be positive".into()));
    }
    if request.queries.is_empty() {
        return Err(Error::InvalidArgument(
            "batch search requires at least one query vector".into(),
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

fn index_relation<'a>(snapshot: &'a IndexSnapshot, role: &str) -> Result<&'a RelationReference> {
    snapshot
        .index_relations
        .get(role)
        .ok_or_else(|| Error::InvalidMetadata(format!("missing relation role: {role}")))
}

fn sql_relation_lock_error() -> Error {
    Error::InvalidArgument("SQL relation lock is poisoned".into())
}
