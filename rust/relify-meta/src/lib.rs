//! Portable Relify index metadata types and format validation.

mod error;
mod family;
mod metadata;
mod relation;
mod serde_helpers;

pub use error::{Error, Result};
pub use family::PostingEncoding;
pub use metadata::{IndexMetadata, IndexSnapshot, SnapshotLogEntry};
pub use relation::RelationReference;
