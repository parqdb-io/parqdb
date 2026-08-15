//! Portable Relify index metadata types and format validation.

mod error;
mod family;
mod ivf_centroids;
mod metadata;
mod relation;
mod serde_helpers;

pub use error::{Error, Result};
pub use family::{IVF_SCHEMA_VERSION, PostingEncoding, ivf_centroids_reference};
pub use ivf_centroids::{
    DistanceMetric, IVF_CLUSTERING_PROFILE_VERSION, IvfCentroidsDescriptor, IvfCentroidsMetadata,
    IvfCentroidsReference,
};
pub use metadata::{IndexMetadata, IndexSnapshot, SnapshotLogEntry};
pub use relation::RelationReference;
