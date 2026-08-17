//! Exact Apache Iceberg snapshot resolution for `ParqDB` runtimes.

use std::collections::HashMap;
use std::sync::Arc;

use datafusion::catalog::TableProvider;
use iceberg::TableIdent;
use iceberg::io::FileIOBuilder;
use iceberg::table::StaticTable;
use iceberg_datafusion::table::IcebergStaticTableProvider;
use iceberg_storage_opendal::OpenDalResolvingStorageFactory;
use parqdb_meta::RelationReference;
use uuid::Uuid;

/// Iceberg relation-resolution failures.
#[derive(Debug, thiserror::Error)]
pub enum Error {
    /// The supplied relation did not use the Iceberg profile.
    #[error("relation is not an Iceberg relation")]
    NotIceberg,
    /// The relation locator is invalid.
    #[error("invalid Iceberg table identifier: {0}")]
    InvalidIdentifier(String),
    /// Iceberg metadata or data could not be loaded.
    #[error("Iceberg error: {0}")]
    Iceberg(#[from] iceberg::Error),
}

/// Result returned by this crate.
pub type Result<T> = std::result::Result<T, Error>;

/// Resolves one `ParqDB` Iceberg reference into a read-only, exact-snapshot
/// `DataFusion` provider.
pub async fn exact_snapshot_provider(
    reference: &RelationReference,
    metadata_location: &str,
    file_io_properties: impl IntoIterator<Item = (String, String)>,
) -> Result<Arc<dyn TableProvider>> {
    let RelationReference::Iceberg {
        namespace,
        name,
        table_uuid,
        snapshot_id,
        ..
    } = reference
    else {
        return Err(Error::NotIceberg);
    };
    let identifier = namespace
        .iter()
        .chain(std::iter::once(name))
        .map(String::as_str);
    let identifier = TableIdent::from_strs(identifier)
        .map_err(|error| Error::InvalidIdentifier(error.to_string()))?;
    let file_io_properties = file_io_properties.into_iter().collect::<HashMap<_, _>>();
    let factory = Arc::new(OpenDalResolvingStorageFactory::new());
    let file_io = FileIOBuilder::new(factory)
        .with_props(file_io_properties)
        .build();
    let table = StaticTable::from_metadata_file(metadata_location, identifier, file_io)
        .await?
        .into_table();
    verify_table_uuid(table.metadata().uuid(), *table_uuid)?;
    let provider =
        IcebergStaticTableProvider::try_new_from_table_snapshot(table, *snapshot_id).await?;
    Ok(Arc::new(provider))
}

fn verify_table_uuid(actual: Uuid, expected: Uuid) -> Result<()> {
    if actual == expected {
        Ok(())
    } else {
        Err(Error::Iceberg(iceberg::Error::new(
            iceberg::ErrorKind::DataInvalid,
            format!("Iceberg table UUID mismatch: expected {expected}, found {actual}"),
        )))
    }
}
