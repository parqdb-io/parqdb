use parqdb_meta::{TableDefinition, TableIdentifier};

use crate::Result;

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
