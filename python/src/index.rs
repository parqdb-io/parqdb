use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;
use std::sync::Arc;

use parqdb_catalog::{Error as CatalogError, IndexIdentifier, SqliteCatalog};
use parqdb_core::{IndexArtifacts, IndexFormat};
use parqdb_index::{
    IndexRepository, InitialIndex, MetadataStore, new_snapshot_id, publish_initial,
};
use parqdb_meta::{DistanceMetric, IndexMetadata, RelationReference};
use parqdb_storage::{StorageRegistry, Warehouse};
use pyo3::prelude::*;
use tokio::runtime::Runtime;
use uuid::Uuid;

use crate::errors::{index_error, runtime_error};
use crate::session::shared_runtime;

type PyIndexInfo = (
    String,
    String,
    String,
    String,
    BTreeMap<String, String>,
    i64,
);

/// Catalog and metadata access that does not instantiate an execution engine.
#[pyclass(name = "_NativeIndexRepository")]
pub(crate) struct PyNativeIndexRepository {
    repository: Arc<IndexRepository>,
    runtime: Arc<Runtime>,
}

impl PyNativeIndexRepository {
    pub(crate) fn from_repository(repository: IndexRepository, runtime: Arc<Runtime>) -> Self {
        Self {
            repository: Arc::new(repository),
            runtime,
        }
    }
}

#[pymethods]
impl PyNativeIndexRepository {
    #[new]
    #[pyo3(signature = (catalog_path, warehouse, storage_options=None))]
    fn new(
        catalog_path: PathBuf,
        warehouse: &str,
        storage_options: Option<HashMap<String, String>>,
    ) -> PyResult<Self> {
        let catalog = Arc::new(SqliteCatalog::open(catalog_path).map_err(runtime_error)?);
        let warehouse = Warehouse::open(
            warehouse,
            StorageRegistry::new(storage_options.unwrap_or_default()),
        )
        .map_err(runtime_error)?;
        Ok(Self::from_repository(
            IndexRepository::new(catalog, MetadataStore::open(warehouse)),
            shared_runtime()?,
        ))
    }

    fn index_exists(&self, name: &str) -> PyResult<bool> {
        let identifier = root_identifier(name)?;
        self.repository
            .exists(&identifier)
            .map_err(|error| index_error(&error))
    }

    fn list_indexes(&self) -> PyResult<Vec<String>> {
        self.list_indexes_in(Vec::new())
    }

    #[allow(clippy::needless_pass_by_value)]
    fn list_indexes_in(&self, namespace: Vec<String>) -> PyResult<Vec<String>> {
        match self.repository.list(&namespace) {
            Ok(identifiers) => Ok(identifiers
                .into_iter()
                .map(|identifier| identifier.name().to_owned())
                .collect()),
            Err(parqdb_index::Error::Catalog(CatalogError::NamespaceNotFound(_))) => Ok(Vec::new()),
            Err(error) => Err(index_error(&error)),
        }
    }

    fn load_index_entry(&self, py: Python<'_>, name: &str) -> PyResult<(String, String)> {
        self.load_index_entry_in(py, Vec::new(), name)
    }

    fn load_index_entry_in(
        &self,
        py: Python<'_>,
        namespace: Vec<String>,
        name: &str,
    ) -> PyResult<(String, String)> {
        let identifier = IndexIdentifier::new(namespace, name).map_err(runtime_error)?;
        let repository = Arc::clone(&self.repository);
        let runtime = Arc::clone(&self.runtime);
        let loaded = py
            .detach(move || runtime.block_on(repository.load(&identifier)))
            .map_err(|error| index_error(&error))?;
        loaded_json(loaded)
    }

    fn drop_index(&self, name: &str) -> PyResult<()> {
        self.drop_index_in(Vec::new(), name)
    }

    fn drop_index_in(&self, namespace: Vec<String>, name: &str) -> PyResult<()> {
        let identifier = IndexIdentifier::new(namespace, name).map_err(runtime_error)?;
        IndexRepository::drop(self.repository.as_ref(), &identifier)
            .map_err(|error| index_error(&error))
    }

    fn list_source_indexes(
        &self,
        py: Python<'_>,
        source_json: &str,
        namespace: Vec<String>,
    ) -> PyResult<Vec<PyIndexInfo>> {
        let source = relation(source_json)?;
        let repository = Arc::clone(&self.repository);
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || runtime.block_on(repository.find_by_source(&namespace, &source)))
            .map_err(|error| index_error(&error))?
            .into_iter()
            .map(|loaded| {
                let snapshot = loaded.metadata.current_snapshot().map_err(runtime_error)?;
                Ok((
                    loaded.entry.identifier.name().to_owned(),
                    snapshot.vector_field.clone(),
                    snapshot.index_family.clone(),
                    snapshot.metric.clone(),
                    snapshot.parameters.clone(),
                    snapshot.snapshot_id,
                ))
            })
            .collect()
    }

    #[pyo3(signature = (source_json, namespace, index=None, column=None))]
    fn select_index(
        &self,
        py: Python<'_>,
        source_json: &str,
        namespace: Vec<String>,
        index: Option<&str>,
        column: Option<String>,
    ) -> PyResult<(String, String, String)> {
        let source = relation(source_json)?;
        let identifier = index
            .map(|name| IndexIdentifier::new(namespace.clone(), name))
            .transpose()
            .map_err(runtime_error)?;
        let repository = Arc::clone(&self.repository);
        let runtime = Arc::clone(&self.runtime);
        let loaded = py
            .detach(move || {
                runtime.block_on(async move {
                    let loaded = repository
                        .select(&namespace, &source, identifier.as_ref(), column.as_deref())
                        .await?;
                    repository.load_snapshot_ivf_centroids(&loaded).await?;
                    Ok::<_, parqdb_index::Error>(loaded)
                })
            })
            .map_err(|error| index_error(&error))?;
        let name = loaded.entry.identifier.name().to_owned();
        let (metadata_location, metadata) = loaded_json(loaded)?;
        Ok((name, metadata_location, metadata))
    }

    #[pyo3(signature = (
        index_name,
        source_json,
        vector_field,
        source_key_fields,
        builder,
        metric,
        parameters,
        index_relations
    ))]
    #[allow(clippy::too_many_arguments)]
    fn publish_initial(
        &self,
        py: Python<'_>,
        index_name: &str,
        source_json: &str,
        vector_field: String,
        source_key_fields: Vec<String>,
        builder: String,
        metric: &str,
        parameters: BTreeMap<String, String>,
        index_relations: BTreeMap<String, String>,
    ) -> PyResult<String> {
        let identifier = root_identifier(index_name)?;
        let source = relation(source_json)?;
        let metric = DistanceMetric::from_metadata(metric)
            .ok_or_else(|| runtime_error("unsupported IVF metric"))?;
        let index_relations = index_relations
            .into_iter()
            .map(|(role, reference)| Ok((role, relation(&reference)?)))
            .collect::<PyResult<BTreeMap<_, _>>>()?;
        let repository = Arc::clone(&self.repository);
        let runtime = Arc::clone(&self.runtime);
        py.detach(move || {
            runtime.block_on(publish_initial(
                repository.catalog(),
                repository.metadata_store(),
                InitialIndex {
                    identifier,
                    index_uuid: Uuid::new_v4(),
                    snapshot_id: new_snapshot_id(),
                    source,
                    vector_field: &vector_field,
                    source_key_fields: &source_key_fields,
                    builder: &builder,
                    build: IndexArtifacts {
                        format: IndexFormat::ivf(metric),
                        parameters,
                        index_relations,
                    },
                },
            ))
        })
        .map(|published| published.metadata_location)
        .map_err(|error| index_error(&error))
    }
}

fn root_identifier(name: &str) -> PyResult<IndexIdentifier> {
    IndexIdentifier::root(name).map_err(runtime_error)
}

fn relation(value: &str) -> PyResult<RelationReference> {
    let reference: RelationReference = serde_json::from_str(value).map_err(runtime_error)?;
    reference.validate().map_err(runtime_error)?;
    Ok(reference)
}

fn loaded_json(loaded: parqdb_index::LoadedIndex) -> PyResult<(String, String)> {
    Ok((
        loaded.entry.metadata_location,
        metadata_json(&loaded.metadata)?,
    ))
}

fn metadata_json(metadata: &IndexMetadata) -> PyResult<String> {
    serde_json::to_string_pretty(metadata).map_err(runtime_error)
}

pub(crate) fn add_index_bindings(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyNativeIndexRepository>()?;
    Ok(())
}
