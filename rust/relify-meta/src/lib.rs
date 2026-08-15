//! Portable Relify index metadata types and format validation.

mod error;
mod family;
mod metadata;
mod relation;
mod serde_helpers;
mod shared_ivf;

pub use error::{Error, Result};
pub use family::{IVF_SCHEMA_VERSION, PostingEncoding, shared_ivf_reference};
pub use metadata::{IndexMetadata, IndexSnapshot, SnapshotLogEntry};
pub use relation::RelationReference;
pub use shared_ivf::{
    DistanceMetric, IVF_CLUSTERING_PROFILE_VERSION, SharedIvfDescriptor, SharedIvfMetadata,
    SharedIvfReference,
};
