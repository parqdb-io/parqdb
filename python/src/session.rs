use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::{Arc, OnceLock};

use arrow::datatypes::{DataType, Schema};
use arrow::pyarrow::PyArrowType;
use datafusion_python::context::PySessionContext;
use datafusion_python::context::{PyRuntimeEnvBuilder, PySessionConfig};
use datafusion_python::dataframe::PyDataFrame;
use pyo3::prelude::*;
use relify_local::{
    DistanceMetric, IvfConfig, LocalBuildOptions, LocalBuildProgress, LocalSession,
    LocalSessionOptions, ParquetWriterOptions, PersistentParquetOptions, PostingEncoding,
    SearchRequest, relify_session_config,
};
use relify_meta::RelationReference;
use tokio::runtime::Runtime;

use crate::errors::{InvalidArgumentError, core_error, runtime_error};
use crate::index::PyNativeIndexRepository;

type PySourceField = (String, String, bool);
type PyIndexInfo = (
    String,
    String,
    String,
    String,
    BTreeMap<String, String>,
    i64,
);
type PyParquetPageCacheStats = (usize, usize, usize, usize, u64, u64, u64, u64, u64, u64);
static SHARED_RUNTIME: OnceLock<std::result::Result<Arc<Runtime>, String>> = OnceLock::new();

fn parse_posting_encoding(value: &str) -> PyResult<PostingEncoding> {
    PostingEncoding::from_metadata(value).ok_or_else(|| {
        InvalidArgumentError::new_err(format!("unsupported posting_encoding: {value}"))
    })
}

fn parse_metric(value: &str) -> PyResult<DistanceMetric> {
    DistanceMetric::from_metadata(value)
        .ok_or_else(|| InvalidArgumentError::new_err(format!("unsupported metric: {value}")))
}

pub(crate) fn shared_runtime() -> PyResult<Arc<Runtime>> {
    match SHARED_RUNTIME.get_or_init(|| {
        Runtime::new()
            .map(Arc::new)
            .map_err(|error| error.to_string())
    }) {
        Ok(runtime) => Ok(Arc::clone(runtime)),
        Err(error) => Err(runtime_error(error)),
    }
}

#[pyclass(name = "_ParquetWriterOptions", frozen)]
pub(crate) struct PyParquetWriterOptions {
    options: ParquetWriterOptions,
}

#[pyclass(name = "_NativeBuildProgress", frozen)]
pub(crate) struct PyNativeBuildProgress {
    progress: LocalBuildProgress,
}

#[pymethods]
impl PyNativeBuildProgress {
    #[new]
    fn new() -> Self {
        Self {
            progress: LocalBuildProgress::default(),
        }
    }

    fn snapshot(&self) -> (&'static str, u64, u64, f64) {
        let snapshot = self.progress.snapshot();
        (
            snapshot.phase,
            snapshot.completed,
            snapshot.total,
            snapshot.fraction,
        )
    }
}

#[pymethods]
impl PyParquetWriterOptions {
    #[new]
    #[pyo3(signature = (
        compression,
        max_row_group_rows,
        target_file_size,
        write_batch_rows
    ))]
    fn new(
        compression: String,
        max_row_group_rows: Option<usize>,
        target_file_size: usize,
        write_batch_rows: usize,
    ) -> Self {
        Self {
            options: ParquetWriterOptions {
                compression,
                max_row_group_rows,
                target_file_size,
                write_batch_rows,
            },
        }
    }
}

#[pyclass(name = "_NativeSession")]
pub(crate) struct PyNativeSession {
    session: Arc<LocalSession>,
    runtime: Arc<Runtime>,
}

#[pymethods]
impl PyNativeSession {
    #[new]
    #[pyo3(signature = (
        state_root,
        warehouse=None,
        storage_options=None,
        catalog_path=None,
        config=None,
        runtime=None
    ))]
    fn new(
        py: Python<'_>,
        state_root: PathBuf,
        warehouse: Option<String>,
        storage_options: Option<HashMap<String, String>>,
        catalog_path: Option<PathBuf>,
        config: Option<PySessionConfig>,
        runtime: Option<PyRuntimeEnvBuilder>,
    ) -> PyResult<Self> {
        let options = LocalSessionOptions::new(
            config.map_or_else(relify_session_config, |config| config.config),
            runtime.map_or_else(Default::default, |runtime| runtime.builder),
        );
        let session = match (catalog_path, warehouse) {
            (Some(catalog_path), Some(warehouse)) => LocalSession::open_sqlite_with_options(
                catalog_path,
                &warehouse,
                storage_options.unwrap_or_default(),
                options,
            ),
            (Some(_), None) => {
                return Err(InvalidArgumentError::new_err(
                    "an explicit catalog requires a warehouse",
                ));
            }
            (None, Some(warehouse)) => LocalSession::open_with_warehouse_options(
                state_root,
                &warehouse,
                storage_options.unwrap_or_default(),
                options,
            ),
            (None, None) if storage_options.as_ref().is_none_or(HashMap::is_empty) => {
                LocalSession::open_with_options(state_root, options)
            }
            (None, None) => {
                return Err(InvalidArgumentError::new_err(
                    "storage_options requires an explicit warehouse",
                ));
            }
        }
        .map_err(|error| core_error(&error))?;
        let session = Arc::new(session);
        let runtime = shared_runtime()?;
        let restore_session = Arc::clone(&session);
        let restore_runtime = Arc::clone(&runtime);
        py.detach(move || restore_runtime.block_on(restore_session.restore_table_definitions()))
            .map_err(|error| core_error(&error))?;
        Ok(Self { session, runtime })
    }

    fn warehouse_root(&self) -> String {
        self.session.warehouse_root().to_owned()
    }

    fn index_repository(&self) -> PyNativeIndexRepository {
        PyNativeIndexRepository::from_repository(
            self.session.index_repository(),
            Arc::clone(&self.runtime),
        )
    }

    fn context(&self) -> PySessionContext {
        PySessionContext::from(self.session.context())
    }

    #[pyo3(signature = (
        table_name,
        source,
        table_partition_cols,
        parquet_pruning,
        file_extension,
        skip_metadata,
        schema,
        file_sort_order
    ))]
    #[allow(clippy::too_many_arguments)]
    fn register_parquet(
        &self,
        py: Python<'_>,
        table_name: String,
        source: &str,
        table_partition_cols: Vec<(String, PyArrowType<DataType>)>,
        parquet_pruning: bool,
        file_extension: String,
        skip_metadata: bool,
        schema: Option<PyArrowType<Schema>>,
        file_sort_order: Vec<Vec<String>>,
    ) -> PyResult<(String, Vec<PySourceField>)> {
        let session = Arc::clone(&self.session);
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || {
            runtime.block_on(
                session.register_parquet_table(
                    &table_name,
                    source,
                    PersistentParquetOptions {
                        table_partition_cols: table_partition_cols
                            .into_iter()
                            .map(|(name, data_type)| (name, data_type.0))
                            .collect(),
                        parquet_pruning,
                        file_extension,
                        skip_metadata,
                        schema: schema.map(|schema| schema.0),
                        file_sort_order,
                    },
                ),
            )
        })
        .map(|description| {
            (
                description.uri,
                description
                    .fields
                    .into_iter()
                    .map(|field| (field.name, field.data_type, field.nullable))
                    .collect(),
            )
        })
        .map_err(|error| core_error(&error))
    }

    #[allow(clippy::type_complexity)]
    fn persistent_table(
        &self,
        table_name: &str,
    ) -> PyResult<Option<(String, Vec<String>, String, String)>> {
        self.session
            .persistent_table(table_name)
            .map(|binding| {
                binding.map(|(identifier, source)| {
                    (
                        identifier.catalog().to_owned(),
                        identifier.namespace().to_vec(),
                        identifier.name().to_owned(),
                        source,
                    )
                })
            })
            .map_err(|error| core_error(&error))
    }

    fn persistent_table_source_by_identifier(
        &self,
        catalog: String,
        namespace: Vec<String>,
        name: String,
    ) -> PyResult<Option<String>> {
        let identifier = relify_catalog::TableIdentifier::new(catalog, namespace, name)
            .map_err(|error| core_error(&error.into()))?;
        self.session
            .persistent_table_source_by_identifier(identifier)
            .map_err(|error| core_error(&error))
    }

    fn drop_table_definition_if_exists(&self, table_name: &str) -> PyResult<bool> {
        self.session
            .drop_table_definition_if_exists(table_name)
            .map_err(|error| core_error(&error))
    }

    fn index_exists(&self, name: &str) -> PyResult<bool> {
        self.session
            .index_exists(name)
            .map_err(|error| core_error(&error))
    }

    fn list_indexes(&self) -> PyResult<Vec<String>> {
        self.session
            .list_indexes()
            .map_err(|error| core_error(&error))
    }

    fn load_index_entry(&self, py: Python<'_>, name: String) -> PyResult<(String, String)> {
        let session = Arc::clone(&self.session);
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || runtime.block_on(session.load_index_entry(&name)))
            .map_err(|error| core_error(&error))
    }

    fn register_index(
        &self,
        py: Python<'_>,
        name: String,
        metadata_location: String,
    ) -> PyResult<()> {
        let session = Arc::clone(&self.session);
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || runtime.block_on(session.register_index(&name, &metadata_location)))
            .map_err(|error| core_error(&error))
    }

    fn select_index_metadata(
        &self,
        py: Python<'_>,
        source: &str,
        index: Option<String>,
        column: Option<String>,
    ) -> PyResult<String> {
        let source = parse_relation_reference(source)?;
        let session = Arc::clone(&self.session);
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || {
            runtime.block_on(session.select_index_metadata(
                &source,
                index.as_deref(),
                column.as_deref(),
            ))
        })
        .map_err(|error| core_error(&error))
    }

    #[pyo3(signature = (source, index=None, column=None))]
    fn select_index(
        &self,
        py: Python<'_>,
        source: &str,
        index: Option<String>,
        column: Option<String>,
    ) -> PyResult<(String, String, String)> {
        let source = parse_relation_reference(source)?;
        let session = Arc::clone(&self.session);
        let runtime = Arc::clone(&self.runtime);
        let loaded = py
            .detach(move || {
                runtime.block_on(session.select_index(&source, index.as_deref(), column.as_deref()))
            })
            .map_err(|error| core_error(&error))?;
        Ok((
            loaded.entry.identifier.name().to_owned(),
            loaded.entry.metadata_location,
            serde_json::to_string_pretty(&loaded.metadata).map_err(runtime_error)?,
        ))
    }

    fn register_iceberg_relation(
        &self,
        py: Python<'_>,
        reference: &str,
        metadata_location: String,
        file_io_properties: HashMap<String, String>,
    ) -> PyResult<PyDataFrame> {
        let reference = parse_relation_reference(reference)?;
        let session = Arc::clone(&self.session);
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || {
            runtime.block_on(session.register_iceberg_relation(
                reference,
                &metadata_location,
                file_io_properties,
            ))
        })
        .map(PyDataFrame::new)
        .map_err(|error| core_error(&error))
    }

    fn drop_index(&self, name: &str) -> PyResult<()> {
        self.session
            .drop_index(name)
            .map_err(|error| core_error(&error))
    }

    fn parquet_page_cache_stats(&self) -> PyParquetPageCacheStats {
        let stats = self.session.parquet_page_cache_stats();
        (
            stats.capacity,
            stats.resident_bytes,
            stats.retired_bytes,
            stats.page_count,
            stats.hits,
            stats.misses,
            stats.admissions,
            stats.evictions,
            stats.capacity_bypasses,
            stats.oversized_bypasses,
        )
    }

    fn clear_parquet_page_cache(&self) {
        self.session.clear_parquet_page_cache();
    }

    fn remove_orphans(
        &self,
        py: Python<'_>,
        older_than_ms: i64,
        dry_run: bool,
    ) -> PyResult<Vec<(String, String, i64)>> {
        let session = Arc::clone(&self.session);
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || runtime.block_on(session.remove_orphans(older_than_ms, dry_run)))
            .map(|objects| {
                objects
                    .into_iter()
                    .map(|object| {
                        (
                            object.kind.as_str().to_owned(),
                            object.reference,
                            object.modified_ms,
                        )
                    })
                    .collect()
            })
            .map_err(|error| core_error(&error))
    }

    fn list_source_indexes(&self, py: Python<'_>, source: &str) -> PyResult<Vec<PyIndexInfo>> {
        let source = parse_relation_reference(source)?;
        let session = Arc::clone(&self.session);
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || runtime.block_on(session.list_relation_indexes(&source)))
            .map(|indexes| {
                indexes
                    .into_iter()
                    .map(|index| {
                        (
                            index.name,
                            index.column,
                            index.family,
                            index.metric,
                            index.parameters,
                            index.current_snapshot_id,
                        )
                    })
                    .collect()
            })
            .map_err(|error| core_error(&error))
    }

    fn drop_source_index(&self, py: Python<'_>, source: &str, name: String) -> PyResult<()> {
        let source = parse_relation_reference(source)?;
        let session = Arc::clone(&self.session);
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || runtime.block_on(session.drop_relation_index(&source, &name)))
            .map_err(|error| core_error(&error))
    }

    #[pyo3(signature = (
        source,
        index_name,
        vector_field,
        source_key_fields,
        nlist,
        posting_encoding,
        metric,
        writer_options,
        partitions,
        threads,
        progress=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn create_index(
        &self,
        py: Python<'_>,
        source: String,
        index_name: String,
        vector_field: String,
        source_key_fields: Vec<String>,
        nlist: usize,
        posting_encoding: &str,
        metric: &str,
        writer_options: &PyParquetWriterOptions,
        partitions: Option<usize>,
        threads: Option<usize>,
        progress: Option<&PyNativeBuildProgress>,
    ) -> PyResult<String> {
        let session = Arc::clone(&self.session);
        let runtime = Arc::clone(&self.runtime);
        let writer_options = writer_options.options.clone();
        let progress = progress.map(|tracker| tracker.progress.clone());
        let posting_encoding = parse_posting_encoding(posting_encoding)?;
        let metric = parse_metric(metric)?;
        py.detach(move || {
            runtime.block_on(session.create_index_with_options(
                &source,
                &index_name,
                &vector_field,
                &source_key_fields,
                IvfConfig::with_metric(nlist, posting_encoding, metric),
                &LocalBuildOptions {
                    writer_options,
                    partitions,
                    threads,
                    progress,
                },
            ))
        })
        .map(|result| result.metadata_location)
        .map_err(|error| core_error(&error))
    }

    #[pyo3(signature = (
        source,
        index_name,
        nlist,
        posting_encoding,
        metric,
        writer_options,
        partitions,
        threads,
        progress=None
    ))]
    #[allow(clippy::too_many_arguments)]
    fn refresh_index(
        &self,
        py: Python<'_>,
        source: String,
        index_name: String,
        nlist: Option<usize>,
        posting_encoding: Option<String>,
        metric: Option<String>,
        writer_options: &PyParquetWriterOptions,
        partitions: Option<usize>,
        threads: Option<usize>,
        progress: Option<&PyNativeBuildProgress>,
    ) -> PyResult<String> {
        let session = Arc::clone(&self.session);
        let runtime = Arc::clone(&self.runtime);
        let writer_options = writer_options.options.clone();
        let progress = progress.map(|tracker| tracker.progress.clone());
        let config = match (nlist, posting_encoding, metric) {
            (Some(nlist), Some(posting_encoding), Some(metric)) => Some(IvfConfig::with_metric(
                nlist,
                parse_posting_encoding(&posting_encoding)?,
                parse_metric(&metric)?,
            )),
            (None, None, None) => None,
            _ => {
                return Err(InvalidArgumentError::new_err(
                    "nlist, posting_encoding, and metric must be provided together",
                ));
            }
        };
        py.detach(move || {
            runtime.block_on(session.refresh_index_with_options(
                &source,
                &index_name,
                config,
                &LocalBuildOptions {
                    writer_options,
                    partitions,
                    threads,
                    progress,
                },
            ))
        })
        .map(|result| result.metadata_location)
        .map_err(|error| core_error(&error))
    }

    #[pyo3(signature = (
        source,
        query,
        index=None,
        column=None,
        nprobe=None,
        limit=10,
        projection=None,
        filter=None,
        bypass_index=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn plan_search(
        &self,
        py: Python<'_>,
        source: &str,
        query: Vec<f32>,
        index: Option<String>,
        column: Option<String>,
        nprobe: Option<usize>,
        limit: usize,
        projection: Option<Vec<String>>,
        filter: Option<String>,
        bypass_index: bool,
    ) -> PyResult<PyDataFrame> {
        let session = Arc::clone(&self.session);
        let runtime = Arc::clone(&self.runtime);
        let request = SearchRequest {
            source: parse_relation_reference(source)?,
            index,
            column,
            query,
            nprobe,
            limit,
            projection,
            filter,
            bypass_index,
        };
        py.detach(move || runtime.block_on(session.plan_search(&request)))
            .map(PyDataFrame::new)
            .map_err(|error| core_error(&error))
    }

    #[pyo3(signature = (
        source,
        query,
        index=None,
        column=None,
        nprobe=None,
        limit=10,
        projection=None,
        filter=None,
        bypass_index=false
    ))]
    #[allow(clippy::too_many_arguments)]
    fn search_sql(
        &self,
        py: Python<'_>,
        source: &str,
        query: Vec<f32>,
        index: Option<String>,
        column: Option<String>,
        nprobe: Option<usize>,
        limit: usize,
        projection: Option<Vec<String>>,
        filter: Option<String>,
        bypass_index: bool,
    ) -> PyResult<String> {
        let session = Arc::clone(&self.session);
        let runtime = Arc::clone(&self.runtime);
        let request = SearchRequest {
            source: parse_relation_reference(source)?,
            index,
            column,
            query,
            nprobe,
            limit,
            projection,
            filter,
            bypass_index,
        };
        py.detach(move || runtime.block_on(session.search_sql(&request)))
            .map_err(|error| core_error(&error))
    }
}

#[pyfunction(name = "_new_session_config")]
fn new_session_config() -> PySessionConfig {
    relify_session_config().into()
}

fn parse_relation_reference(value: &str) -> PyResult<RelationReference> {
    let reference: RelationReference = serde_json::from_str(value)
        .map_err(|error| InvalidArgumentError::new_err(error.to_string()))?;
    reference
        .validate()
        .map_err(|error| InvalidArgumentError::new_err(error.to_string()))?;
    Ok(reference)
}

pub(crate) fn add_session_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(new_session_config, module)?)?;
    module.add_class::<PyParquetWriterOptions>()?;
    module.add_class::<PyNativeBuildProgress>()?;
    module.add_class::<PyNativeSession>()?;
    Ok(())
}
