//! Native Apache Iceberg table resolution for `ParqDB` runtimes.

use std::collections::{BTreeMap, HashMap};
use std::hash::BuildHasher;
use std::sync::Arc;

use async_trait::async_trait;
use datafusion::catalog::{Session, TableProvider, TableProviderFactory};
use datafusion::common::DataFusionError;
use datafusion::logical_expr::CreateExternalTable;
use iceberg::TableIdent;
use iceberg::io::FileIOBuilder;
use iceberg::table::{StaticTable, Table};
use iceberg_datafusion::table::IcebergStaticTableProvider;
use iceberg_storage_opendal::OpenDalResolvingStorageFactory;
use parqdb_meta::{TableDefinition, TableIdentifier};
use uuid::Uuid;

const TABLE_UUID_OPTION: &str = "table-uuid";
const SNAPSHOT_ID_OPTION: &str = "snapshot-id";

/// Iceberg table-resolution failures.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The table locator or exact-state property is invalid.
    #[error("invalid Iceberg table definition: {0}")]
    InvalidDefinition(String),
    /// Iceberg metadata or data could not be loaded.
    #[error("Iceberg error: {0}")]
    Iceberg(#[from] iceberg::Error),
    /// The persistent table definition is invalid.
    #[error("metadata error: {0}")]
    Metadata(#[from] parqdb_meta::Error),
}

/// Result returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Native `DataFusion` factory for exact Iceberg table snapshots.
#[derive(Debug, Clone, Default)]
pub struct IcebergTableProviderFactory {
    file_io_properties: HashMap<String, String>,
}

impl IcebergTableProviderFactory {
    /// Creates a factory using runtime-only object-store properties.
    #[must_use]
    pub fn new(file_io_properties: HashMap<String, String>) -> Self {
        Self { file_io_properties }
    }

    /// Creates a factory from `ParqDB`'s process-level object-store options.
    #[must_use]
    pub fn from_storage_options<S: BuildHasher>(
        storage_options: &HashMap<String, String, S>,
    ) -> Self {
        Self::new(file_io_properties(storage_options))
    }
}

/// Converts `ParqDB` object-store option names to Iceberg `FileIO` properties.
#[must_use]
pub fn file_io_properties<S: BuildHasher>(
    storage_options: &HashMap<String, String, S>,
) -> HashMap<String, String> {
    let mut properties = storage_options
        .iter()
        .filter(|(name, _)| name.starts_with("s3."))
        .map(|(name, value)| (name.clone(), value.clone()))
        .collect::<HashMap<_, _>>();
    for (source, target) in [
        ("aws_access_key_id", "s3.access-key-id"),
        ("aws_secret_access_key", "s3.secret-access-key"),
        ("aws_session_token", "s3.session-token"),
        ("aws_region", "s3.region"),
        ("aws_endpoint", "s3.endpoint"),
    ] {
        if let Some(value) = storage_options.get(source) {
            properties.insert(target.into(), value.clone());
        }
    }
    if let Some(value) = storage_options.get("aws_virtual_hosted_style_request") {
        let path_style = match value.as_str() {
            "true" => "false",
            "false" => "true",
            other => other,
        };
        properties.insert("s3.path-style-access".into(), path_style.into());
    }
    properties
}

#[async_trait]
impl TableProviderFactory for IcebergTableProviderFactory {
    async fn create(
        &self,
        _state: &dyn Session,
        command: &CreateExternalTable,
    ) -> datafusion::common::Result<Arc<dyn TableProvider>> {
        let table_uuid =
            required_uuid(&command.options, TABLE_UUID_OPTION).map_err(external_error)?;
        let snapshot_id = required_snapshot_id(&command.options).map_err(external_error)?;
        reject_unknown_options(&command.options).map_err(external_error)?;
        let identifier = command_identifier(command).map_err(external_error)?;
        exact_snapshot_provider(
            &identifier,
            &command.location,
            table_uuid,
            snapshot_id,
            self.file_io_properties.clone(),
        )
        .await
        .map_err(external_error)
    }
}

/// Reads one metadata file natively and produces the exact persistent table
/// definition that can reopen it without consulting a Python catalog.
pub async fn table_definition<S: BuildHasher>(
    identifier: TableIdentifier,
    metadata_location: &str,
    snapshot_id: Option<i64>,
    file_io_properties: HashMap<String, String, S>,
) -> Result<TableDefinition> {
    let table = load_table(
        &identifier,
        metadata_location,
        file_io_properties.into_iter().collect(),
    )
    .await?;
    table_definition_from_table(identifier, &table, snapshot_id)
}

fn table_definition_from_table(
    identifier: TableIdentifier,
    table: &Table,
    snapshot_id: Option<i64>,
) -> Result<TableDefinition> {
    let metadata = table.metadata();
    let snapshot_id = snapshot_id
        .or_else(|| metadata.current_snapshot_id())
        .ok_or_else(|| Error::InvalidDefinition("table has no current snapshot".into()))?;
    if snapshot_id <= 0 || metadata.snapshot_by_id(snapshot_id).is_none() {
        return Err(Error::InvalidDefinition(format!(
            "snapshot is not retained: {snapshot_id}"
        )));
    }
    let table_uuid = metadata.uuid();
    let metadata_location = table
        .metadata_location()
        .ok_or_else(|| Error::InvalidDefinition("catalog table has no metadata location".into()))?;
    TableDefinition::new(
        identifier,
        "iceberg",
        BTreeMap::from([
            ("definition-version".into(), "1".into()),
            ("location".into(), metadata_location.to_owned()),
            ("table-identity".into(), table_uuid.to_string()),
            ("option.table-uuid".into(), table_uuid.to_string()),
            ("option.snapshot-id".into(), snapshot_id.to_string()),
        ]),
    )
    .map_err(Into::into)
}

async fn exact_snapshot_provider(
    identifier: &TableIdentifier,
    metadata_location: &str,
    table_uuid: Uuid,
    snapshot_id: i64,
    file_io_properties: HashMap<String, String>,
) -> Result<Arc<dyn TableProvider>> {
    let table = load_table(identifier, metadata_location, file_io_properties).await?;
    verify_table_uuid(table.metadata().uuid(), table_uuid)?;
    let provider =
        IcebergStaticTableProvider::try_new_from_table_snapshot(table, snapshot_id).await?;
    Ok(Arc::new(provider))
}

async fn load_table(
    identifier: &TableIdentifier,
    metadata_location: &str,
    file_io_properties: HashMap<String, String>,
) -> Result<Table> {
    let identifier = iceberg_identifier(identifier)?;
    let factory = Arc::new(OpenDalResolvingStorageFactory::new());
    let file_io = FileIOBuilder::new(factory)
        .with_props(file_io_properties)
        .build();
    Ok(
        StaticTable::from_metadata_file(metadata_location, identifier, file_io)
            .await?
            .into_table(),
    )
}

fn command_identifier(command: &CreateExternalTable) -> Result<TableIdentifier> {
    let namespace = command
        .name
        .schema()
        .map_or_else(|| vec!["public".into()], |schema| vec![schema.into()]);
    TableIdentifier::new(
        command.name.catalog().unwrap_or("datafusion"),
        namespace,
        command.name.table(),
    )
    .map_err(Into::into)
}

fn iceberg_identifier(identifier: &TableIdentifier) -> Result<TableIdent> {
    let parts = identifier
        .namespace()
        .iter()
        .map(String::as_str)
        .chain(std::iter::once(identifier.name()));
    TableIdent::from_strs(parts).map_err(|error| Error::InvalidDefinition(error.to_string()))
}

fn required_uuid(options: &HashMap<String, String>, name: &str) -> Result<Uuid> {
    options
        .get(name)
        .and_then(|value| Uuid::parse_str(value).ok())
        .ok_or_else(|| Error::InvalidDefinition(format!("missing or invalid {name}")))
}

fn required_snapshot_id(options: &HashMap<String, String>) -> Result<i64> {
    options
        .get(SNAPSHOT_ID_OPTION)
        .and_then(|value| value.parse::<i64>().ok())
        .filter(|value| *value > 0)
        .ok_or_else(|| Error::InvalidDefinition("missing or invalid snapshot-id".into()))
}

fn reject_unknown_options(options: &HashMap<String, String>) -> Result<()> {
    let unknown = options
        .keys()
        .filter(|name| !matches!(name.as_str(), TABLE_UUID_OPTION | SNAPSHOT_ID_OPTION))
        .cloned()
        .collect::<Vec<_>>();
    if unknown.is_empty() {
        Ok(())
    } else {
        Err(Error::InvalidDefinition(format!(
            "unknown options: {}",
            unknown.join(", ")
        )))
    }
}

fn verify_table_uuid(actual: Uuid, expected: Uuid) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::InvalidDefinition(format!(
            "table UUID mismatch: expected {expected}, found {actual}"
        )))
    }
}

fn external_error(error: Error) -> DataFusionError {
    DataFusionError::External(Box::new(error))
}
