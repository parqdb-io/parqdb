use std::collections::BTreeMap;

use crate::{Error, IndexCatalog, Result, TableIdentifier};

/// Persistent definition of one table exposed through a runtime catalog.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableDefinition {
    /// Fully qualified runtime identifier.
    pub identifier: TableIdentifier,
    /// Provider profile used to reconstruct the table, such as `parquet`.
    pub provider: String,
    /// Provider-defined, versioned string properties.
    pub properties: BTreeMap<String, String>,
}

impl TableDefinition {
    /// Creates and validates a persistent table definition.
    pub fn new(
        identifier: TableIdentifier,
        provider: impl Into<String>,
        properties: BTreeMap<String, String>,
    ) -> Result<Self> {
        let provider = provider.into();
        if provider.is_empty() {
            return Err(Error::InvalidTableDefinition(
                "provider must not be empty".into(),
            ));
        }
        if properties.keys().any(String::is_empty) {
            return Err(Error::InvalidTableDefinition(
                "property names must not be empty".into(),
            ));
        }
        Ok(Self {
            identifier,
            provider,
            properties,
        })
    }
}

/// Persistent table-definition operations used by the runtime catalog.
pub trait TableCatalog: Send + Sync {
    /// Creates a table definition only when its identifier is absent.
    fn create_table(&self, definition: &TableDefinition) -> Result<()>;

    /// Loads one table definition.
    fn load_table(&self, identifier: &TableIdentifier) -> Result<TableDefinition>;

    /// Lists table identifiers directly in a catalog namespace.
    fn list_tables(&self, catalog: &str, namespace: &[String]) -> Result<Vec<TableIdentifier>>;

    /// Drops a table definition without deleting its external data.
    fn drop_table(&self, identifier: &TableIdentifier) -> Result<()>;
}

/// One catalog implementation that owns both table definitions and `ParqDB` indexes.
pub trait ParqDBCatalog: IndexCatalog + TableCatalog {}

impl<T> ParqDBCatalog for T where T: IndexCatalog + TableCatalog {}
