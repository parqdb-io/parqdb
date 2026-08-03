use thiserror::Error;

/// Error returned by Relify storage operations.
#[derive(Debug, Error)]
pub enum Error {
    /// A URI is malformed or uses an unsupported scheme.
    #[error("invalid storage location: {0}")]
    InvalidLocation(String),
    /// An object-store operation failed.
    #[error("storage operation failed: {0}")]
    ObjectStore(#[from] object_store::Error),
    /// A storage registry lock was poisoned.
    #[error("storage registry lock was poisoned")]
    Poisoned,
}

/// Result type returned by Relify storage operations.
pub type Result<T> = std::result::Result<T, Error>;
