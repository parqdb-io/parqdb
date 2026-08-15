//! Persistent Parquet source registration and restoration.

use std::collections::{BTreeMap, HashMap, HashSet};
use std::path::PathBuf;
use std::sync::Arc;

use arrow::datatypes::{DataType, Schema, SchemaRef};
use arrow_ipc::convert::try_schema_from_ipc_buffer;
use arrow_ipc::writer::{DictionaryTracker, IpcDataGenerator, IpcWriteOptions, write_message};
use base64::Engine;
use base64::engine::general_purpose::STANDARD as BASE64;
use datafusion::catalog::TableProvider;
use datafusion::common::TableReference;
use datafusion::logical_expr::SortExpr;
use datafusion::prelude::{ParquetReadOptions, col};
use relify_catalog::Error as CatalogError;
use relify_catalog::{TableDefinition, TableIdentifier};
use relify_meta::{IndexSnapshot, RelationReference};
use relify_storage::StorageRegistry;
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
    pub(super) reference: RelationReference,
    pub(super) schema: SchemaRef,
    pub(super) table_name: String,
    pub(super) provider: Arc<dyn TableProvider>,
}

impl LocalSession {
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

    /// Returns the resolved identifier and source URI for a registered table.
    pub fn persistent_table(&self, table_name: &str) -> Result<Option<(TableIdentifier, String)>> {
        let identifier = self.resolve_table_identifier(table_name)?;
        self.persistent_table_by_identifier(identifier)
    }

    /// Returns the source URI for an exact persistent table identifier.
    pub fn persistent_table_source_by_identifier(
        &self,
        identifier: TableIdentifier,
    ) -> Result<Option<String>> {
        Ok(self
            .persistent_table_by_identifier(identifier)?
            .map(|(_, location)| location))
    }

    fn persistent_table_by_identifier(
        &self,
        identifier: TableIdentifier,
    ) -> Result<Option<(TableIdentifier, String)>> {
        match self.table_catalog.load_table(&identifier) {
            Ok(definition) => Ok(Some((
                identifier,
                PersistentParquetDefinition::from_table_definition(definition)?.location,
            ))),
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
        let location = PersistentParquetDefinition::from_table_definition(definition)?.location;
        let mut bindings = self
            .source_bindings
            .write()
            .map_err(|_| source_binding_lock_error())?;
        self.table_catalog.drop_table(&identifier)?;
        if bindings
            .get(&relation_key(&RelationReference::Parquet {
                uri: location.clone(),
            }))
            .is_some_and(|binding| binding.table_name == identifier.name())
        {
            bindings.remove(&relation_key(&RelationReference::Parquet { uri: location }));
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
        match self.bind_provider_source(&definition.location, provider, Some(identifier.name())) {
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
            let definition = PersistentParquetDefinition::from_table_definition(table_definition)?;
            let provider = self.parquet_provider(&definition).await?;
            if provider.schema().as_ref() != &definition.resolved_schema {
                return Err(Error::InvalidSchema(format!(
                    "persistent table schema changed: {}",
                    identifier.name()
                )));
            }
            self.context
                .register_table(table_reference(&identifier)?, Arc::clone(&provider))?;
            self.bind_provider_source(&definition.location, provider, Some(identifier.name()))?;
        }
        Ok(())
    }

    fn resolve_table_identifier(&self, table_name: &str) -> Result<TableIdentifier> {
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
        let binding = self.bind_provider_source(source, provider, Some(identifier.name()))?;
        Ok(binding.description())
    }

    pub(super) async fn bind_source(&self, source: &str) -> Result<SourceBinding> {
        let uri = canonical_source(&self.warehouse.registry(), source)?;
        let reference = RelationReference::Parquet { uri: uri.clone() };
        if let Some(binding) = self.source_binding(&relation_key(&reference))? {
            return Ok(binding);
        }

        let dataframe = self.parquet.dataframe(&uri).await?;
        let provider: Arc<dyn TableProvider> = dataframe.into_view();
        self.bind_provider_reference(reference, provider, None)
    }

    fn bind_provider_source(
        &self,
        source: &str,
        provider: Arc<dyn TableProvider>,
        registered_name: Option<&str>,
    ) -> Result<SourceBinding> {
        let uri = canonical_source(&self.warehouse.registry(), source)?;
        self.bind_provider_reference(
            RelationReference::Parquet { uri },
            provider,
            registered_name,
        )
    }

    fn bind_provider_reference(
        &self,
        reference: RelationReference,
        provider: Arc<dyn TableProvider>,
        registered_name: Option<&str>,
    ) -> Result<SourceBinding> {
        reference.validate()?;
        let key = relation_key(&reference);
        let mut bindings = self
            .source_bindings
            .write()
            .map_err(|_| source_binding_lock_error())?;
        if let Some(existing) = bindings.get(&key) {
            if existing.schema.as_ref() != provider.schema().as_ref() {
                return Err(Error::InvalidSchema(format!(
                    "the same relation state was registered with different schemas: {key}"
                )));
            }
            return Ok(existing.clone());
        }
        let binding = SourceBinding {
            key: key.clone(),
            reference,
            schema: provider.schema(),
            table_name: registered_name.map_or_else(
                || format!("__relify_source_{}", Uuid::new_v4().simple()),
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

    /// Registers one exact Iceberg snapshot in this session's `DataFusion` context.
    pub async fn register_iceberg_relation(
        &self,
        reference: RelationReference,
        metadata_location: &str,
        file_io_properties: HashMap<String, String>,
    ) -> Result<datafusion::dataframe::DataFrame> {
        reference.validate()?;
        let key = relation_key(&reference);
        if let Some(provider) = self.index_relation_providers.registered(&key)? {
            return Ok(self.context.read_table(provider)?);
        }
        let provider = relify_iceberg::exact_snapshot_provider(
            &reference,
            metadata_location,
            file_io_properties,
        )
        .await?;
        let provider = self.index_relation_providers.register(key, provider)?;
        self.bind_provider_reference(reference, Arc::clone(&provider), None)?;
        Ok(self.context.read_table(provider)?)
    }

    pub(super) async fn bind_relation(
        &self,
        reference: &RelationReference,
    ) -> Result<SourceBinding> {
        reference.validate()?;
        match reference {
            RelationReference::Parquet { uri } => self.bind_source(uri).await,
            RelationReference::Iceberg { .. } => self
                .source_binding(&relation_key(reference))?
                .ok_or_else(|| {
                    Error::InvalidArgument(format!(
                        "Iceberg relation is not registered in this session: {}",
                        relation_key(reference)
                    ))
                }),
        }
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
    fn to_table_definition(&self, identifier: TableIdentifier) -> Result<TableDefinition> {
        let mut properties = BTreeMap::from([
            ("definition-version".into(), "1".into()),
            ("location".into(), self.location.clone()),
            (
                "partition-schema".into(),
                encode_schema(&self.partition_schema)?,
            ),
            (
                "parquet-pruning".into(),
                encode_bool(self.parquet_pruning).into(),
            ),
            ("file-extension".into(), self.file_extension.clone()),
            (
                "skip-metadata".into(),
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

    fn from_table_definition(definition: TableDefinition) -> Result<Self> {
        if definition.provider != "parquet" {
            return Err(invalid_table_definition(format!(
                "unsupported persistent table provider: {}",
                definition.provider
            )));
        }
        let mut properties = definition.properties;
        let version = take_property(&mut properties, "definition-version")?;
        if version != "1" {
            return Err(invalid_table_definition(format!(
                "unsupported persistent table definition version: {version}"
            )));
        }
        let location = take_property(&mut properties, "location")?;
        let partition_schema = decode_schema(&take_property(&mut properties, "partition-schema")?)?;
        let parquet_pruning = decode_bool(&take_property(&mut properties, "parquet-pruning")?)?;
        let file_extension = take_property(&mut properties, "file-extension")?;
        let skip_metadata = decode_bool(&take_property(&mut properties, "skip-metadata")?)?;
        let resolved_schema = decode_schema(&take_property(&mut properties, "resolved-schema")?)?;
        let file_sort_order =
            serde_json::from_str(&take_property(&mut properties, "file-sort-order")?)?;
        let provided_schema = properties
            .remove("provided-schema")
            .map(|encoded| decode_schema(&encoded))
            .transpose()?;
        if !properties.is_empty() {
            return Err(invalid_table_definition(format!(
                "unknown persistent Parquet properties: {}",
                properties.keys().cloned().collect::<Vec<_>>().join(", ")
            )));
        }
        Ok(Self {
            location,
            partition_schema,
            parquet_pruning,
            file_extension,
            skip_metadata,
            provided_schema,
            resolved_schema,
            file_sort_order,
        })
    }
}

impl SourceBinding {
    fn description(&self) -> SourceDescription {
        let RelationReference::Parquet { uri } = &self.reference else {
            unreachable!("source descriptions are exposed only for Parquet tables")
        };
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

pub(super) fn relation_key(reference: &RelationReference) -> String {
    match reference {
        RelationReference::Parquet { uri } => uri.clone(),
        RelationReference::Iceberg { .. } => reference.exact_state_key(),
    }
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

fn decode_bool(value: &str) -> Result<bool> {
    match value {
        "true" => Ok(true),
        "false" => Ok(false),
        _ => Err(invalid_table_definition(format!(
            "invalid persistent boolean: {value}"
        ))),
    }
}

fn take_property(properties: &mut BTreeMap<String, String>, name: &str) -> Result<String> {
    properties
        .remove(name)
        .ok_or_else(|| invalid_table_definition(format!("missing persistent property: {name}")))
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
            "Parquet source must be an absolute path or URI".into(),
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
