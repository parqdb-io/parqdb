//! Portable `ParqDB` index metadata types and format validation.

mod artifact_manifest;
mod error;
mod family;
mod ivf_centroids;
mod metadata;
mod postings_manifest;
mod relation;
mod serde_helpers;

pub use artifact_manifest::{
    IndexArtifactManifest, JSON_SAFE_INTEGER_MAX, StaticEmbeddingDescriptor,
    StaticEmbeddingParityProbe, StaticIndexDescriptor, StaticIndexHierarchy, StaticIndexObject,
    StaticIndexPostings, StaticPostingsFile, StaticSourceDescriptor, StaticSourceFile,
    StaticSourceKeyField,
};
pub use error::{Error, Result};
pub use family::{IVF_SCHEMA_VERSION, PostingEncoding, ivf_centroids_reference};
pub use ivf_centroids::{
    DistanceMetric, IVF_CLUSTERING_PROFILE_VERSION, IvfCentroidsDescriptor, IvfCentroidsMetadata,
    IvfCentroidsReference,
};
pub use metadata::{
    IndexMetadata, IndexSnapshot, SnapshotLogEntry, validate_absolute_location,
    validate_relative_location,
};
pub use postings_manifest::{IvfPostingsFile, IvfPostingsManifest};
pub use relation::RelationReference;
