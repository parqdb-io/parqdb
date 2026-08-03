//! Portable Relify index metadata types and format validation.

mod error;
mod metadata;
mod relation;
mod serde_helpers;

pub use error::{Error, Result};
pub use metadata::{IndexMetadata, IndexSnapshot, SnapshotLogEntry};
pub use relation::RelationReference;
