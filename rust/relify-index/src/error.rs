use relify_catalog::IndexIdentifier;

/// Errors returned by the backend-neutral index repository.
#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum Error {
    /// Index metadata violates the Relify specification.
    #[error("invalid index metadata: {0}")]
    InvalidMetadata(String),
    /// A timestamp cannot be represented by the metadata model.
    #[error("invalid publication timestamp: {0}")]
    InvalidTimestamp(String),
    /// No published index matched the requested source and vector field.
    #[error("index not found: {0}")]
    IndexNotFound(String),
    /// More than one published index matched an implicit selection.
    #[error("ambiguous index selection: {0:?}")]
    AmbiguousIndex(Vec<IndexIdentifier>),
    /// An index catalog operation failed.
    #[error(transparent)]
    Catalog(#[from] relify_catalog::Error),
    /// A managed storage operation failed.
    #[error(transparent)]
    Storage(#[from] relify_storage::Error),
    /// Metadata JSON encoding or decoding failed.
    #[error("metadata JSON error: {0}")]
    Json(#[from] serde_json::Error),
}

impl From<relify_meta::Error> for Error {
    fn from(error: relify_meta::Error) -> Self {
        Self::InvalidMetadata(error.into_message())
    }
}

/// Result returned by index repository operations.
pub type Result<T> = std::result::Result<T, Error>;
