//! Persistent Parquet source registration and restoration.

use std::collections::{BTreeMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow_ipc::convert::try_schema_from_ipc_buffer;
use arrow_ipc::writer::{DictionaryTracker, IpcDataGenerator, IpcWriteOptions, write_message};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use datafusion::catalog::{TableProvider, TableProviderFactory};
use datafusion::common::{DFSchema, TableReference};
use datafusion::logical_expr::expr::Sort;
use datafusion::logical_expr::{CreateExternalTable, SortExpr};
use datafusion::prelude::{ParquetReadOptions, col};
use parqdb_catalog::Error as CatalogError;
use parqdb_catalog::{TableDefinition, TableIdentifier};
use parqdb_meta::IndexSnapshot;
use parqdb_storage::StorageRegistry;
use url::Url;
use uuid::Uuid;

use super::{LocalSession, SourceDescription, SourceField};
use crate::local_uri::{directory_to_file_uri, file_uri_to_path, path_to_file_uri};
use crate::{Error, Result};

/// Persistent options for one Parquet table registered in a local session.
#[derive(Debug, Clone)]
pub struct PersistentParquetOptions {
    /// Partition columns appended to the physical Parquet schema.
    pub table_partition_cols: Vec<(String, DataType)>,
    /// Whether `DataFusion` may prune Parquet row groups.
    pub parquet_pruning: bool,
    /// File extension selected below directory locations.
    pub file_extension: String,
    /// Whether Arrow metadata embedded in Parquet files is ignored.
    pub skip_metadata: bool,
    /// Optional caller-provided physical Parquet schema.
    pub schema: Option<Schema>,
    /// Declared file ordering as ascending, nulls-first column names.
    pub file_sort_order: Vec<Vec<String>>,
}

impl Default for PersistentParquetOptions {
    fn default() -> Self {
        Self {
            table_partition_cols: Vec::new(),
            parquet_pruning: true,
            file_extension: ".parquet".into(),
            skip_metadata: true,
            schema: None,
            file_sort_order: Vec::new(),
        }
    }
}

#[derive(Clone)]
pub(super) struct SourceBinding {
    pub(super) key: String,
    pub(super) reference: TableDefinition,
    pub(super) schema: SchemaRef,
    pub(super) table_name: String,
    pub(super) provider: Arc<dyn TableProvider>,
}

impl LocalSession {
    /// Registers a `DataFusion` table-provider factory for persisted table definitions.
    pub fn register_table_provider_factory(
        &self,
        name: impl Into<String>,
        factory: Arc<dyn TableProviderFactory>,
    ) -> Result<()> {
        let name = name.into().trim().to_ascii_uppercase();
        if name.is_empty() {
            return Err(Error::InvalidArgument(
                "table provider name must not be empty".into(),
            ));
        }
        let state = self.context.state_ref();
        let mut state = state.write();
        if state.table_factories().contains_key(&name) {
            return Err(Error::InvalidArgument(format!(
                "table provider factory is already registered: {name}"
            )));
        }
        state.table_factories_mut().insert(name, factory);
        Ok(())
    }

    /// Resolves one persistent source location into current execution inputs.
    pub async fn resolve_source_locations(&self, source: &str) -> Result<Vec<String>> {
        let registry = self.warehouse.registry();
        if !source.contains('*') {
            if Url::parse(source).is_ok() {
                let resolved = registry.resolve(source)?;
                self.context
                    .runtime_env()
                    .register_object_store(resolved.base_url(), resolved.store());
            }
            return Ok(vec![source.to_owned()]);
        }
        let uri = canonical_source(&registry, source)?;
        let resolved = registry.resolve(&uri)?;
        self.context
            .runtime_env()
            .register_object_store(resolved.base_url(), resolved.store());
        Ok(registry.expand(&uri).await?)
    }

    /// Persists one table definition in the session's combined catalog.
    pub fn create_table_definition(
        &self,
        table_name: &str,
        provider: &str,
        properties: std::collections::BTreeMap<String, String>,
    ) -> Result<TableDefinition> {
        let identifier = self.resolve_table_identifier(table_name)?;
        let definition = TableDefinition::new(identifier, provider, properties)?;
        self.table_catalog.create_table(&definition)?;
        Ok(definition)
    }

    /// Lists persistent table definitions in `DataFusion`'s default namespace.
    pub fn list_table_definitions(&self) -> Result<Vec<TableDefinition>> {
        let options = self.context.state().config_options().catalog.clone();
        Ok(self
            .table_catalog
            .list_tables(&options.default_catalog, &[options.default_schema])
            .and_then(|identifiers| {
                identifiers
                    .into_iter()
                    .map(|identifier| self.table_catalog.load_table(&identifier))
                    .collect()
            })?)
    }

    /// Drops one persistent table definition.
    pub fn drop_table_definition(&self, table_name: &str) -> Result<()> {
        let identifier = self.resolve_table_identifier(table_name)?;
        Ok(self.table_catalog.drop_table(&identifier)?)
    }

    /// Returns the exact definition of a registered table.
    pub fn persistent_table_definition(&self, table_name: &str) -> Result<Option<TableDefinition>> {
        let identifier = self.resolve_table_identifier(table_name)?;
        self.persistent_table_definition_by_identifier(&identifier)
    }

    /// Returns the exact definition for a persistent table identifier.
    pub fn persistent_table_definition_by_identifier(
        &self,
        identifier: &TableIdentifier,
    ) -> Result<Option<TableDefinition>> {
        match self.table_catalog.load_table(identifier) {
            Ok(definition) => Ok(Some(definition)),
            Err(CatalogError::TableNotFound(_)) => Ok(None),
            Err(error) => Err(error.into()),
        }
    }

    /// Drops a persistent table definition and its matching source binding.
    pub fn drop_table_definition_if_exists(&self, table_name: &str) -> Result<bool> {
        let identifier = self.resolve_table_identifier(table_name)?;
        let definition = match self.table_catalog.load_table(&identifier) {
            Ok(definition) => definition,
            Err(CatalogError::TableNotFound(_)) => return Ok(false),
            Err(error) => return Err(error.into()),
        };
        let mut bindings = self
            .source_bindings
            .write()
            .map_err(|_| source_binding_lock_error())?;
        self.table_catalog.drop_table(&identifier)?;
        if bindings
            .get(&table_key(&definition))
            .is_some_and(|binding| binding.table_name == identifier.name())
        {
            bindings.remove(&table_key(&definition));
        }
        Ok(true)
    }

    /// Registers and persists one Parquet table in the default `DataFusion` namespace.
    pub async fn register_parquet_table(
        &self,
        table_name: &str,
        source: &str,
        options: PersistentParquetOptions,
    ) -> Result<SourceDescription> {
        let identifier = self.resolve_default_table_identifier(table_name)?;
        match self.table_catalog.load_table(&identifier) {
            Ok(_) => return Err(CatalogError::TableAlreadyExists(identifier).into()),
            Err(CatalogError::TableNotFound(_)) => {}
            Err(error) => return Err(error.into()),
        }
        if self.context.table_exist(table_reference(&identifier)?)? {
            return Err(CatalogError::TableAlreadyExists(identifier).into());
        }

        let definition = PersistentParquetDefinition {
            location: source.to_owned(),
            partition_schema: Schema::new(
                options
                    .table_partition_cols
                    .iter()
                    .map(|(name, data_type)| {
                        arrow::datatypes::Field::new(name, data_type.clone(), false)
                    })
                    .collect::<Vec<_>>(),
            ),
            parquet_pruning: options.parquet_pruning,
            file_extension: options.file_extension,
            skip_metadata: options.skip_metadata,
            provided_schema: options.schema,
            resolved_schema: Schema::empty(),
            file_sort_order: options.file_sort_order,
        };
        let provider = self.parquet_provider(&definition).await?;
        let definition = PersistentParquetDefinition {
            location: canonical_source(&self.warehouse.registry(), source)?,
            resolved_schema: provider.schema().as_ref().clone(),
            ..definition
        };
        let table_definition = definition.to_table_definition(identifier.clone())?;
        self.table_catalog.create_table(&table_definition)?;

        let reference = table_reference(&identifier)?;
        if let Err(error) = self
            .context
            .register_table(reference, Arc::clone(&provider))
        {
            let _ = self.table_catalog.drop_table(&identifier);
            return Err(error.into());
        }
        match self.bind_provider_reference(table_definition, provider, Some(identifier.name())) {
            Ok(binding) => Ok(binding.description()),
            Err(error) => {
                let _ = self.context.deregister_table(table_reference(&identifier)?);
                let _ = self.table_catalog.drop_table(&identifier);
                Err(error)
            }
        }
    }

    /// Restores persistent table providers into this session's `DataFusion` catalog.
    pub async fn restore_table_definitions(&self) -> Result<()> {
        for table_definition in self.list_table_definitions()? {
            let identifier = table_definition.identifier.clone();
            let provider = self
                .provider_from_table_definition(&table_definition)
                .await?;
            self.context
                .register_table(table_reference(&identifier)?, Arc::clone(&provider))?;
            self.bind_provider_reference(table_definition, provider, Some(identifier.name()))?;
        }
        Ok(())
    }

    pub(super) fn resolve_table_identifier(&self, table_name: &str) -> Result<TableIdentifier> {
        let reference = TableReference::from(table_name);
        let options = self.context.state().config_options().catalog.clone();
        let resolved = reference.resolve(&options.default_catalog, &options.default_schema);
        Ok(TableIdentifier::new(
            resolved.catalog.to_string(),
            vec![resolved.schema.to_string()],
            resolved.table.to_string(),
        )?)
    }

    fn resolve_default_table_identifier(&self, table_name: &str) -> Result<TableIdentifier> {
        let identifier = self.resolve_table_identifier(table_name)?;
        let options = self.context.state().config_options().catalog.clone();
        if identifier.catalog() != options.default_catalog
            || identifier.namespace() != std::slice::from_ref(&options.default_schema)
        {
            return Err(Error::InvalidArgument(
                "persistent Parquet tables currently require the default DataFusion catalog and schema"
                    .into(),
            ));
        }
        Ok(identifier)
    }

    /// Binds a Parquet source in this session and returns its canonical schema.
    pub async fn describe(&self, source: &str) -> Result<SourceDescription> {
        let binding = self.bind_source(source).await?;
        Ok(binding.description())
    }

    /// Binds an existing `DataFusion` table to a persistent Parquet source reference.
    pub async fn bind_registered_source(
        &self,
        table_name: &str,
        source: &str,
    ) -> Result<SourceDescription> {
        let provider = self.context.table_provider(table_name).await?;
        let identifier = self.resolve_default_table_identifier(table_name)?;
        let uri = canonical_source(&self.warehouse.registry(), source)?;
        let definition = minimal_parquet_definition(identifier.clone(), uri)?;
        let binding =
            self.bind_provider_reference(definition, provider, Some(identifier.name()))?;
        Ok(binding.description())
    }

    pub(super) async fn bind_source(&self, source: &str) -> Result<SourceBinding> {
        let uri = canonical_source(&self.warehouse.registry(), source)?;
        let identifier = anonymous_table_identifier(&uri)?;
        let reference = minimal_parquet_definition(identifier, uri.clone())?;
        if let Some(binding) = self.source_binding(&table_key(&reference))? {
            return Ok(binding);
        }

        let dataframe = self.parquet.dataframe(&uri).await?;
        let provider: Arc<dyn TableProvider> = dataframe.into_view();
        self.bind_provider_reference(reference, provider, None)
    }

    fn bind_provider_reference(
        &self,
        reference: TableDefinition,
        provider: Arc<dyn TableProvider>,
        registered_name: Option<&str>,
    ) -> Result<SourceBinding> {
        reference.validate()?;
        let key = table_key(&reference);
        let mut bindings = self
            .source_bindings
            .write()
            .map_err(|_| source_binding_lock_error())?;
        if let Some(existing) = bindings.get(&key) {
            if existing.schema.as_ref() != provider.schema().as_ref() {
                return Err(Error::InvalidSchema(format!(
                    "the same table state was registered with different schemas: {key}"
                )));
            }
            return Ok(existing.clone());
        }
        let binding = SourceBinding {
            key: key.clone(),
            reference,
            schema: provider.schema(),
            table_name: registered_name.map_or_else(
                || format!("__parqdb_source_{}", Uuid::new_v4().simple()),
                str::to_owned,
            ),
            provider,
        };
        if registered_name.is_none() {
            self.context
                .register_table(binding.table_name.clone(), Arc::clone(&binding.provider))?;
        }
        bindings.insert(key, binding.clone());
        Ok(binding)
    }

    /// Binds an already resolved runtime provider to its exact durable definition.
    pub fn bind_table_provider(
        &self,
        reference: TableDefinition,
        provider: Arc<dyn TableProvider>,
    ) -> Result<datafusion::dataframe::DataFrame> {
        let binding = self.bind_provider_reference(reference, provider, None)?;
        Ok(self.context.read_table(binding.provider)?)
    }

    pub(super) async fn bind_table(&self, reference: &TableDefinition) -> Result<SourceBinding> {
        reference.validate()?;
        if let Some(binding) = self.source_binding(&table_key(reference))? {
            return Ok(binding);
        }
        let provider = self.provider_from_table_definition(reference).await?;
        self.bind_provider_reference(reference.clone(), provider, None)
    }

    async fn provider_from_table_definition(
        &self,
        definition: &TableDefinition,
    ) -> Result<Arc<dyn TableProvider>> {
        if definition.provider == "parquet" {
            return self
                .parquet_provider(&PersistentParquetDefinition::from_table_definition(
                    definition,
                )?)
                .await;
        }
        let state = self.context.state();
        let factory = state
            .table_factories()
            .get(&definition.provider.to_ascii_uppercase())
            .cloned()
            .ok_or_else(|| {
                Error::InvalidArgument(format!(
                    "table provider factory is not registered: {}",
                    definition.provider
                ))
            })?;
        let command = external_table_command(definition)?;
        Ok(factory.create(&state, &command).await?)
    }

    async fn parquet_provider(
        &self,
        definition: &PersistentParquetDefinition,
    ) -> Result<Arc<dyn TableProvider>> {
        let locations = self.resolve_source_locations(&definition.location).await?;
        let partition_columns = definition
            .partition_schema
            .fields()
            .iter()
            .map(|field| (field.name().clone(), field.data_type().clone()))
            .collect();
        let file_sort_order = definition
            .file_sort_order
            .iter()
            .map(|order| {
                order
                    .iter()
                    .map(|name| col(name).sort(true, true))
                    .collect::<Vec<SortExpr>>()
            })
            .collect();
        let mut options = ParquetReadOptions::default()
            .table_partition_cols(partition_columns)
            .parquet_pruning(definition.parquet_pruning)
            .skip_metadata(definition.skip_metadata)
            .file_sort_order(file_sort_order);
        options.file_extension = &definition.file_extension;
        options.schema = definition.provided_schema.as_ref();
        Ok(self
            .context
            .read_parquet(locations, options)
            .await?
            .into_view())
    }

    pub(super) fn source_binding(&self, uri: &str) -> Result<Option<SourceBinding>> {
        Ok(self
            .source_bindings
            .read()
            .map_err(|_| source_binding_lock_error())?
            .get(uri)
            .cloned())
    }

    #[cfg(test)]
    pub(super) fn source_binding_count(&self) -> Result<usize> {
        Ok(self
            .source_bindings
            .read()
            .map_err(|_| source_binding_lock_error())?
            .len())
    }
}

struct PersistentParquetDefinition {
    location: String,
    partition_schema: Schema,
    parquet_pruning: bool,
    file_extension: String,
    skip_metadata: bool,
    provided_schema: Option<Schema>,
    resolved_schema: Schema,
    file_sort_order: Vec<Vec<String>>,
}

impl PersistentParquetDefinition {
    fn from_table_definition(definition: &TableDefinition) -> Result<Self> {
        let required = |name: &str| {
            definition.properties.get(name).ok_or_else(|| {
                invalid_table_definition(format!("missing persistent property: {name}"))
            })
        };
        if required("definition-version")? != "1" {
            return Err(invalid_table_definition(
                "unsupported persistent Parquet definition version".into(),
            ));
        }
        Ok(Self {
            location: required("location")?.clone(),
            partition_schema: decode_schema(required("partition-schema")?)?,
            parquet_pruning: decode_bool(
                required("option.format.pruning")?,
                "option.format.pruning",
            )?,
            file_extension: required("file-extension")?.clone(),
            skip_metadata: decode_bool(
                required("option.format.skip_metadata")?,
                "option.format.skip_metadata",
            )?,
            provided_schema: definition
                .properties
                .get("provided-schema")
                .map(|encoded| decode_schema(encoded))
                .transpose()?,
            resolved_schema: decode_schema(required("resolved-schema")?)?,
            file_sort_order: serde_json::from_str(required("file-sort-order")?).map_err(
                |error| invalid_table_definition(format!("invalid file sort order: {error}")),
            )?,
        })
    }

    fn to_table_definition(&self, identifier: TableIdentifier) -> Result<TableDefinition> {
        let mut properties = BTreeMap::from([
            ("definition-version".into(), "1".into()),
            ("location".into(), self.location.clone()),
            ("table-identity".into(), self.location.clone()),
            (
                "partition-schema".into(),
                encode_schema(&self.partition_schema)?,
            ),
            (
                "option.format.pruning".into(),
                encode_bool(self.parquet_pruning).into(),
            ),
            ("file-extension".into(), self.file_extension.clone()),
            (
                "option.format.skip_metadata".into(),
                encode_bool(self.skip_metadata).into(),
            ),
            (
                "resolved-schema".into(),
                encode_schema(&self.resolved_schema)?,
            ),
            (
                "file-sort-order".into(),
                serde_json::to_string(&self.file_sort_order)?,
            ),
        ]);
        if let Some(schema) = &self.provided_schema {
            properties.insert("provided-schema".into(), encode_schema(schema)?);
        }
        Ok(TableDefinition::new(identifier, "parquet", properties)?)
    }
}

impl SourceBinding {
    fn description(&self) -> SourceDescription {
        let uri = self
            .reference
            .properties
            .get("location")
            .expect("Parquet source definitions contain a location");
        SourceDescription {
            uri: uri.clone(),
            fields: self
                .schema
                .fields()
                .iter()
                .map(|field| SourceField {
                    name: field.name().clone(),
                    data_type: field.data_type().to_string(),
                    nullable: field.is_nullable(),
                })
                .collect(),
        }
    }
}

pub(super) fn table_key(reference: &TableDefinition) -> String {
    reference.exact_state_key()
}

fn anonymous_table_identifier(location: &str) -> Result<TableIdentifier> {
    let name = format!(
        "__parqdb_{}",
        Uuid::new_v5(&Uuid::NAMESPACE_URL, location.as_bytes()).simple()
    );
    Ok(TableIdentifier::new(
        "datafusion",
        vec!["public".into()],
        name,
    )?)
}

fn minimal_parquet_definition(
    identifier: TableIdentifier,
    location: String,
) -> Result<TableDefinition> {
    let definition = PersistentParquetDefinition {
        location,
        partition_schema: Schema::empty(),
        parquet_pruning: true,
        file_extension: ".parquet".into(),
        skip_metadata: true,
        provided_schema: None,
        resolved_schema: Schema::empty(),
        file_sort_order: Vec::new(),
    };
    definition.to_table_definition(identifier)
}

pub(super) fn parquet_table_definition(location: String) -> Result<TableDefinition> {
    minimal_parquet_definition(anonymous_table_identifier(&location)?, location)
}

fn table_reference(identifier: &TableIdentifier) -> Result<TableReference> {
    let [schema] = identifier.namespace() else {
        return Err(Error::InvalidArgument(
            "DataFusion tables require exactly one schema namespace segment".into(),
        ));
    };
    Ok(TableReference::full(
        identifier.catalog().to_owned(),
        schema.clone(),
        identifier.name().to_owned(),
    ))
}

fn external_table_command(definition: &TableDefinition) -> Result<CreateExternalTable> {
    let version = definition
        .properties
        .get("definition-version")
        .ok_or_else(|| {
            invalid_table_definition("missing persistent property: definition-version".into())
        })?;
    if version != "1" {
        return Err(invalid_table_definition(format!(
            "unsupported persistent table definition version: {version}"
        )));
    }
    let location = definition
        .properties
        .get("location")
        .ok_or_else(|| invalid_table_definition("missing persistent property: location".into()))?;
    let schema = definition
        .properties
        .get("resolved-schema")
        .map(|encoded| decode_schema(encoded))
        .transpose()?
        .unwrap_or_else(Schema::empty);
    let partition_columns = definition
        .properties
        .get("partition-schema")
        .map(|encoded| decode_schema(encoded))
        .transpose()?
        .unwrap_or_else(Schema::empty)
        .fields()
        .iter()
        .map(|field| field.name().clone())
        .collect();
    let order_exprs = definition
        .properties
        .get("file-sort-order")
        .map(|encoded| serde_json::from_str::<Vec<Vec<String>>>(encoded))
        .transpose()?
        .unwrap_or_default()
        .into_iter()
        .map(|order| {
            order
                .into_iter()
                .map(|name| Sort::new(col(name), true, true))
                .collect()
        })
        .collect();
    let options = definition
        .properties
        .iter()
        .filter_map(|(name, value)| {
            name.strip_prefix("option.")
                .map(|name| (name.to_owned(), value.clone()))
        })
        .collect();
    let schema = Arc::new(DFSchema::try_from(schema)?);
    Ok(CreateExternalTable::builder(
        table_reference(&definition.identifier)?,
        location,
        definition.provider.clone(),
        schema,
    )
    .with_partition_cols(partition_columns)
    .with_order_exprs(order_exprs)
    .with_options(options)
    .build())
}

fn encode_schema(schema: &Schema) -> Result<String> {
    let options = IpcWriteOptions::default();
    let mut dictionaries = DictionaryTracker::new(true);
    let encoded = IpcDataGenerator {}.schema_to_bytes_with_dictionary_tracker(
        schema,
        &mut dictionaries,
        &options,
    );
    let mut bytes = Vec::new();
    write_message(&mut bytes, encoded, &options)?;
    Ok(BASE64.encode(bytes))
}

fn decode_schema(encoded: &str) -> Result<Schema> {
    let bytes = BASE64
        .decode(encoded)
        .map_err(|error| invalid_table_definition(format!("invalid Arrow schema: {error}")))?;
    Ok(try_schema_from_ipc_buffer(&bytes)?)
}

fn encode_bool(value: bool) -> &'static str {
    if value { "true" } else { "false" }
}

fn decode_bool(value: &str, name: &str) -> Result<bool> {
    value.parse().map_err(|_| {
        invalid_table_definition(format!("persistent property {name} must be true or false"))
    })
}

fn invalid_table_definition(message: String) -> Error {
    CatalogError::InvalidTableDefinition(message).into()
}

fn source_binding_lock_error() -> Error {
    Error::InvalidArgument("source binding lock is poisoned".into())
}

pub(super) fn canonical_source(registry: &StorageRegistry, source: &str) -> Result<String> {
    if let Ok(uri) = Url::parse(source) {
        if uri.scheme() == "file" {
            let path = file_uri_to_path(source)?;
            return canonical_file_location(&path);
        }
        return Ok(registry.resolve(source)?.uri().to_string());
    }
    let path = PathBuf::from(source);
    if !path.is_absolute() {
        return Err(Error::InvalidArgument(
            "table location must be an absolute path or URI".into(),
        ));
    }
    canonical_file_location(&path)
}

pub(super) fn resolve_search_projection(
    schema: &Schema,
    projection: Option<&[String]>,
) -> Result<Vec<String>> {
    if schema.field_with_name("_distance").is_ok() {
        return Err(Error::InvalidSchema(
            "source table must not contain reserved column _distance".into(),
        ));
    }
    let projection = match projection {
        Some(projection) => projection.to_vec(),
        None => schema
            .fields()
            .iter()
            .map(|field| field.name().clone())
            .collect(),
    };
    if projection.is_empty() || projection.iter().collect::<HashSet<_>>().len() != projection.len()
    {
        return Err(Error::InvalidArgument(
            "projection must contain unique source column names".into(),
        ));
    }
    for name in &projection {
        schema
            .field_with_name(name)
            .map_err(|_| Error::InvalidArgument(format!("column not found: {name}")))?;
    }
    Ok(projection)
}

pub(super) fn exact_vector_field(schema: &Schema, requested: Option<&str>) -> Result<String> {
    if let Some(requested) = requested {
        let field = schema
            .field_with_name(requested)
            .map_err(|_| Error::InvalidSchema(format!("vector column not found: {requested}")))?;
        validate_vector_field(field, None)?;
        return Ok(requested.to_owned());
    }
    let candidates = schema
        .fields()
        .iter()
        .filter(|field| is_float_vector_type(field.data_type()))
        .map(|field| field.name().clone())
        .collect::<Vec<_>>();
    match candidates.as_slice() {
        [vector_field] => Ok(vector_field.clone()),
        [] => Err(Error::InvalidArgument(
            "column is required because the source has no vector column".into(),
        )),
        _ => Err(Error::InvalidArgument(
            "column is required because the source has multiple vector columns".into(),
        )),
    }
}

pub(super) fn validate_index_source_schema(
    schema: &Schema,
    snapshot: &IndexSnapshot,
) -> Result<()> {
    let vector = schema
        .field_with_name(&snapshot.vector_field)
        .map_err(|_| {
            Error::InvalidSchema(format!(
                "vector column not found: {}",
                snapshot.vector_field
            ))
        })?;
    validate_vector_field(vector, Some(snapshot.parameter_usize("dimension")?))?;
    for key in &snapshot.source_key_fields {
        schema
            .field_with_name(key)
            .map_err(|_| Error::InvalidSchema(format!("source key column not found: {key}")))?;
    }
    Ok(())
}

fn canonical_file_location(path: &std::path::Path) -> Result<String> {
    if path
        .components()
        .any(|component| component.as_os_str().to_string_lossy().contains('*'))
    {
        return canonical_file_pattern(path);
    }
    let canonical = path.canonicalize()?;
    if canonical.is_dir() {
        directory_to_file_uri(&canonical)
    } else {
        path_to_file_uri(&canonical)
    }
}

fn canonical_file_pattern(path: &std::path::Path) -> Result<String> {
    let mut prefix = PathBuf::new();
    let mut pattern = PathBuf::new();
    let mut in_pattern = false;
    for component in path.components() {
        if !in_pattern && component.as_os_str().to_string_lossy().contains('*') {
            in_pattern = true;
        }
        if in_pattern {
            pattern.push(component.as_os_str());
        } else {
            prefix.push(component.as_os_str());
        }
    }
    if !in_pattern || prefix.as_os_str().is_empty() {
        return Err(Error::InvalidArgument(format!(
            "invalid Parquet source pattern: {}",
            path.display()
        )));
    }
    path_to_file_uri(&prefix.canonicalize()?.join(pattern))
}

fn validate_vector_field(
    field: &arrow::datatypes::Field,
    expected_dimension: Option<usize>,
) -> Result<()> {
    if !is_float_vector_type(field.data_type()) {
        return Err(Error::InvalidSchema(
            "source vector column must be list<float> or list<double>".into(),
        ));
    }
    if let (Some(expected), DataType::FixedSizeList(_, actual)) =
        (expected_dimension, field.data_type())
        && usize::try_from(*actual).ok() != Some(expected)
    {
        return Err(Error::InvalidSchema(format!(
            "source vector dimension {actual} does not match index dimension {expected}"
        )));
    }
    Ok(())
}

fn is_float_vector_type(data_type: &DataType) -> bool {
    crate::vector::canonical_vector_type(data_type).is_some()
}

pub(super) fn vector_elements_are_f64(data_type: &DataType) -> bool {
    match data_type {
        DataType::List(field) | DataType::LargeList(field) => {
            field.data_type() == &DataType::Float64
        }
        DataType::FixedSizeList(field, _) => field.data_type() == &DataType::Float64,
        _ => false,
    }
}
