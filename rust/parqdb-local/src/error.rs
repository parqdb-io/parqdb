use parqdb_catalog::IndexIdentifier;
use std::time::Duration;
use thiserror::Error;

/// Stable error category retained for a completed background build.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BuildFailureKind {
    /// Invalid request or configuration.
    InvalidArgument,
    /// Invalid source or index schema.
    InvalidSchema,
    /// Missing source-scoped index.
    IndexNotFound,
    /// Existing index publication.
    AlreadyExists,
    /// Conflicting build reservation.
    BuildAlreadyRunning,
    /// Ambiguous index selection.
    AmbiguousIndex,
    /// Invalid persisted metadata.
    InvalidMetadata,
    /// Catalog operation failure.
    Catalog,
    /// Storage or serialization failure.
    Storage,
    /// Compute, Arrow, or kernel failure.
    Backend,
}

impl BuildFailureKind {
    /// Returns the stable transport code for this failure category.
    #[must_use]
    pub const fn code(self) -> &'static str {
        match self {
            Self::InvalidArgument => "invalid_argument",
            Self::InvalidSchema => "invalid_schema",
            Self::IndexNotFound => "index_not_found",
            Self::AlreadyExists => "already_exists",
            Self::BuildAlreadyRunning => "build_already_running",
            Self::AmbiguousIndex => "ambiguous_index",
            Self::InvalidMetadata => "invalid_metadata",
            Self::Catalog => "catalog_error",
            Self::Storage => "storage_error",
            Self::Backend => "backend_error",
        }
    }
}

#[derive(Debug, Error)]
#[error("{0}")]
pub(crate) struct InvalidSchemaDataFusionError(String);

/// Error returned by embedded `ParqDB` operations.
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
    /// A background index build failed after it was accepted.
    #[error("{message}")]
    BuildFailed {
        /// Stable error category for transport mapping.
        kind: BuildFailureKind,
        /// Original safe error message.
        message: String,
    },
    /// The runtime's bounded query queue has no free entry.
    #[error("query queue is full (capacity {0})")]
    QueryQueueFull(usize),
    /// A query waited longer than the configured admission timeout.
    #[error("query admission timed out after {0:?}")]
    QueryQueueTimeout(Duration),
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
    Catalog(#[from] parqdb_catalog::Error),
    /// A managed storage operation failed.
    #[error(transparent)]
    Storage(#[from] parqdb_storage::Error),
    /// JSON serialization or deserialization failed.
    #[error("JSON error: {0}")]
    Json(#[from] serde_json::Error),
    /// `DataFusion` planning or execution failed.
    #[error("DataFusion error: {0}")]
    DataFusion(datafusion::error::DataFusionError),
    /// Arrow validation or computation failed.
    #[error("Arrow error: {0}")]
    Arrow(#[from] arrow::error::ArrowError),
    /// Parquet encoding or decoding failed.
    #[error("Parquet error: {0}")]
    Parquet(#[from] parquet::errors::ParquetError),
    /// An Iceberg table could not be resolved at its referenced snapshot.
    #[error("Iceberg error: {0}")]
    Iceberg(#[from] parqdb_iceberg::Error),
    /// A shared numerical kernel rejected its matrix inputs.
    #[error(transparent)]
    Kernel(#[from] parqdb_kernels::KernelError),
}

impl Error {
    pub(crate) fn retained_build_failure(&self) -> Self {
        let kind = match self {
            Self::InvalidArgument(_) => BuildFailureKind::InvalidArgument,
            Self::InvalidSchema(_) => BuildFailureKind::InvalidSchema,
            Self::IndexNotFound(_) => BuildFailureKind::IndexNotFound,
            Self::AlreadyExists(_) => BuildFailureKind::AlreadyExists,
            Self::BuildAlreadyRunning(_) => BuildFailureKind::BuildAlreadyRunning,
            Self::BuildFailed { kind, .. } => *kind,
            Self::AmbiguousIndex(_) => BuildFailureKind::AmbiguousIndex,
            Self::InvalidMetadata(_) => BuildFailureKind::InvalidMetadata,
            Self::Catalog(_) => BuildFailureKind::Catalog,
            Self::Storage(_) | Self::Io(_) | Self::Json(_) | Self::Parquet(_) => {
                BuildFailureKind::Storage
            }
            Self::DataFusion(_) | Self::Arrow(_) | Self::Iceberg(_) | Self::Kernel(_) => {
                BuildFailureKind::Backend
            }
            Self::QueryQueueFull(_) | Self::QueryQueueTimeout(_) => BuildFailureKind::Backend,
        };
        Self::BuildFailed {
            kind,
            message: self.to_string(),
        }
    }
}

/// Result type returned by native `ParqDB` operations.
pub type Result<T> = std::result::Result<T, Error>;

impl From<parqdb_meta::Error> for Error {
    fn from(error: parqdb_meta::Error) -> Self {
        Self::InvalidMetadata(error.into_message())
    }
}

impl From<datafusion::error::DataFusionError> for Error {
    fn from(error: datafusion::error::DataFusionError) -> Self {
        if let Some(message) = invalid_schema_message(&error) {
            Self::InvalidSchema(message.to_owned())
        } else {
            Self::DataFusion(error)
        }
    }
}

pub(crate) fn invalid_schema_datafusion(
    message: impl Into<String>,
) -> datafusion::error::DataFusionError {
    datafusion::error::DataFusionError::External(Box::new(InvalidSchemaDataFusionError(
        message.into(),
    )))
}

fn invalid_schema_message(error: &datafusion::error::DataFusionError) -> Option<&str> {
    use datafusion::error::DataFusionError;

    match error {
        DataFusionError::External(source) => source
            .downcast_ref::<InvalidSchemaDataFusionError>()
            .map(|source| source.0.as_str()),
        DataFusionError::Context(_, source) | DataFusionError::Diagnostic(_, source) => {
            invalid_schema_message(source)
        }
        DataFusionError::Collection(errors) => errors.iter().find_map(invalid_schema_message),
        DataFusionError::Shared(source) => invalid_schema_message(source),
        _ => None,
    }
}

impl From<parqdb_index::Error> for Error {
    fn from(error: parqdb_index::Error) -> Self {
        match error {
            parqdb_index::Error::InvalidMetadata(message) => Self::InvalidMetadata(message),
            parqdb_index::Error::InvalidTimestamp(message) => Self::InvalidArgument(message),
            parqdb_index::Error::IndexNotFound(index) => Self::IndexNotFound(index),
            parqdb_index::Error::AmbiguousIndex(indexes) => Self::AmbiguousIndex(
                indexes
                    .iter()
                    .map(parqdb_catalog::IndexIdentifier::name)
                    .collect::<Vec<_>>()
                    .join(", "),
            ),
            parqdb_index::Error::Catalog(error) => Self::Catalog(error),
            parqdb_index::Error::Storage(error) => Self::Storage(error),
            parqdb_index::Error::Json(error) => Self::Json(error),
            other => Self::InvalidMetadata(other.to_string()),
        }
    }
}

impl From<parqdb_kmeans::Error> for Error {
    fn from(error: parqdb_kmeans::Error) -> Self {
        match error {
            parqdb_kmeans::Error::InvalidArgument(message) => Self::InvalidArgument(message),
            parqdb_kmeans::Error::Kernel(error) => Self::Kernel(error),
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
