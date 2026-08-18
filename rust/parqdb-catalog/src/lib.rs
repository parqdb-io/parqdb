#![warn(missing_docs)]
//! Catalog abstractions and the `SQLite` catalog implementation for `ParqDB`.

mod catalog;
mod error;
mod identifier;
#[cfg(feature = "sqlite")]
mod sqlite;
mod table;

pub use catalog::{
    CatalogEntry, CatalogTombstone, IndexCatalog, IvfCentroidsCatalogEntry, IvfCentroidsClaim,
    IvfCentroidsClaimResult,
};
pub use error::{Error, Result};
pub use identifier::{IndexIdentifier, TableIdentifier};
#[cfg(feature = "sqlite")]
pub use sqlite::SqliteCatalog;
pub use table::{ParqDBCatalog, TableCatalog, TableDefinition};
