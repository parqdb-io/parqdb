use crate::{IndexIdentifier, TableIdentifier};

/// Errors returned by catalog operations.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// An index identifier contains an empty name or namespace segment.
    #[error("invalid index identifier: {0}")]
    InvalidIdentifier(String),
    /// The requested namespace does not exist.
    #[error("namespace not found: {0:?}")]
    NamespaceNotFound(Vec<String>),
    /// The requested index does not exist.
    #[error("index not found: {0}")]
    IndexNotFound(IndexIdentifier),
    /// An index with the same identifier already exists.
    #[error("index already exists: {0}")]
    AlreadyExists(IndexIdentifier),
    /// The requested table does not exist.
    #[error("table not found: {0}")]
    TableNotFound(TableIdentifier),
    /// A table with the same identifier already exists.
    #[error("table already exists: {0}")]
    TableAlreadyExists(TableIdentifier),
    /// A persistent table definition is invalid.
    #[error("invalid table definition: {0}")]
    InvalidTableDefinition(String),
    /// The supplied index UUID differs from the current catalog entry.
    #[error("index UUID does not match current catalog entry: {0}")]
    IndexUuidMismatch(IndexIdentifier),
    /// A compare-and-swap catalog commit lost a concurrent update.
    #[error("catalog commit conflict: {0}")]
    CommitConflict(IndexIdentifier),
    /// An index metadata document is invalid.
    #[error("invalid index metadata: {0}")]
    InvalidMetadata(String),
    /// The catalog implementation does not support an optional operation.
    #[error("catalog operation is not supported: {0}")]
    UnsupportedOperation(&'static str),
    /// A catalog implementation reported an implementation-specific failure.
    #[error("catalog implementation error: {0}")]
    Implementation(String),
    /// The `SQLite` database uses an unsupported catalog schema version.
    #[error("unsupported SQLite catalog schema version: {0}")]
    UnsupportedSchemaVersion(i64),
    /// An underlying filesystem operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// An underlying `SQLite` operation failed.
    #[cfg(feature = "sqlite")]
    #[error("SQLite error: {0}")]
    Sqlite(#[from] rusqlite::Error),
    /// A structured identifier could not be encoded.
    #[cfg(feature = "sqlite")]
    #[error("identifier encoding error: {0}")]
    Json(#[from] serde_json::Error),
}

/// Result type returned by catalog operations.
pub type Result<T> = std::result::Result<T, Error>;

impl From<relify_meta::Error> for Error {
    fn from(error: relify_meta::Error) -> Self {
        Self::InvalidMetadata(error.into_message())
    }
}
