use relify_catalog::IndexIdentifier;
use thiserror::Error;

/// Error returned by embedded Relify operations.
#[derive(Debug, Error)]
pub enum Error {
    /// A caller supplied an invalid argument.
    #[error("invalid argument: {0}")]
    InvalidArgument(String),
    /// A source or index table has an invalid schema or value.
    #[error("invalid source schema: {0}")]
    InvalidSchema(String),
    /// No index matched the request.
    #[error("index not found: {0}")]
    IndexNotFound(String),
    /// The requested index already exists.
    #[error("index already exists: {0}")]
    AlreadyExists(String),
    /// Another process is already building the requested index.
    #[error("an index build is already running: {0}")]
    BuildAlreadyRunning(IndexIdentifier),
    /// More than one index matched an implicit selection.
    #[error("ambiguous index for source: {0}")]
    AmbiguousIndex(String),
    /// Index metadata is invalid.
    #[error("invalid index metadata: {0}")]
    InvalidMetadata(String),
    /// An underlying filesystem operation failed.
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    /// An index catalog operation failed.
    #[error(transparent)]
    Catalog(#[from] relify_catalog::Error),
    /// A managed storage operation failed.
    #[error(transparent)]
    Storage(#[from] relify_storage::Error),
    /// JSON serialization or deserialization failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// `DataFusion` planning or execution failed.
    #[error("DataFusion error: {0}")]
    DataFusion(#[from] datafusion::error::DataFusionError),
    /// Arrow validation or computation failed.
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    /// Parquet encoding or decoding failed.
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// An Iceberg table could not be resolved at its referenced snapshot.
    #[error("Iceberg error: {0}")]
    Iceberg(#[from] relify_iceberg::Error),
    /// A shared numerical kernel rejected its matrix inputs.
    #[error(transparent)]
    Kernel(#[from] relify_kernels::KernelError),
}

/// Result type returned by native Relify operations.
pub type Result<T> = std::result::Result<T, Error>;

impl From<relify_meta::Error> for Error {
    fn from(error: relify_meta::Error) -> Self {
        Self::InvalidMetadata(error.into_message())
    }
}

impl From<relify_index::Error> for Error {
    fn from(error: relify_index::Error) -> Self {
        match error {
            relify_index::Error::InvalidMetadata(message) => Self::InvalidMetadata(message),
            relify_index::Error::InvalidTimestamp(message) => Self::InvalidArgument(message),
            relify_index::Error::IndexNotFound(index) => Self::IndexNotFound(index),
            relify_index::Error::AmbiguousIndex(indexes) => Self::AmbiguousIndex(
                indexes
                    .iter()
                    .map(relify_catalog::IndexIdentifier::name)
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
            relify_index::Error::Catalog(error) => Self::Catalog(error),
            relify_index::Error::Storage(error) => Self::Storage(error),
            relify_index::Error::Json(error) => Self::Json(error),
            other => Self::InvalidMetadata(other.to_string()),
        }
    }
}

impl From<relify_kmeans::Error> for Error {
    fn from(error: relify_kmeans::Error) -> Self {
        match error {
            relify_kmeans::Error::InvalidArgument(message) => Self::InvalidArgument(message),
            relify_kmeans::Error::Kernel(error) => Self::Kernel(error),
        }
    }
}

impl From<parallite::ParalliteError<Error>> for Error {
    fn from(error: parallite::ParalliteError<Error>) -> Self {
        match error {
            parallite::ParalliteError::User(error) => error,
            other => Self::InvalidArgument(other.to_string()),
        }
    }
}
