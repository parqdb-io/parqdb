//! Embedded query resolution and `DataFusion` plan construction.

use std::sync::Arc;

use arrow::array::{ArrayRef, Int32Array};
use arrow::compute::concat_batches;
use arrow::datatypes::SchemaRef;
use arrow::record_batch::RecordBatch;
use datafusion::prelude::{SessionContext, col, lit};
use relify_meta::{IndexSnapshot, PostingEncoding, RelationReference};
use uuid::Uuid;

use super::LocalSession;
use super::index_relation::IndexRelationLayout;
use super::source::{
    exact_vector_field, relation_key, resolve_search_projection, validate_index_source_schema,
};
use crate::query::{
    compile_datafusion_sql, compile_index_only_plan, datafusion_centroid_relation_required,
    datafusion_cluster_relation_required, datafusion_source_relation_required,
    selected_cluster_ids_from_values, use_native_cluster_routing, validated_cluster_search,
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

        let source = self.bind_relation(&request.source).await?;
        let source_reference = source.reference.clone();
        let source_relation_key = source.key;
        let source_schema = source.schema;
        let projection =
            resolve_search_projection(source_schema.as_ref(), request.projection.as_deref())?;

        if request.bypass_index {
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
            if request.query.is_empty() || request.query.iter().any(|value| !value.is_finite()) {
                return Err(Error::InvalidArgument(
                    "query vector must contain finite float values and must not be empty".into(),
                ));
            }
            let vector_field =
                exact_vector_field(source_schema.as_ref(), request.column.as_deref())?;
            return Ok(ResolvedSearch {
                source_relation_key,
                query: request.query.clone(),
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
            });
        }

        let loaded = self
            .select_index(
                &source_reference,
                request.index.as_deref(),
                request.column.as_deref(),
            )
            .await?;
        let snapshot = loaded.metadata.current_snapshot()?;
        validate_index_source_schema(source_schema.as_ref(), snapshot)?;
        let postings_relation_key = relation_key(index_relation(snapshot, "ivf_postings")?);
        let centroid_cache_key = format!("{}\0ivf_centroids", loaded.entry.metadata_location);
        let (cluster_selection, nlist) = self
            .resolve_cluster_selection(
                snapshot,
                &centroid_cache_key,
                &request.query,
                request.nprobe,
            )
            .await?;
        Ok(ResolvedSearch {
            source_relation_key,
            query: request.query.clone(),
            vector_field: snapshot.vector_field.clone(),
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
            .get_or_create_parquet(key, layout, &self.parquet)
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
            .index_relation_dataframe(relation_key, layout)
            .await?
            .into_view();
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
                    super::index_relation::CentroidMatrix::new(nlist, dimension, values)
                })
                .await?;
            let selected = selected_cluster_ids_from_values(
                query,
                centroids.values(nlist, dimension)?,
                nlist,
                dimension,
                nprobe,
            )?;
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
