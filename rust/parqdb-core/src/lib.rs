//! Backend-neutral `ParqDB` index and query models.

mod model;

pub use model::{IndexArtifacts, IndexFormat, IvfConfig, PublishedIndex, SearchRequest};
pub use parqdb_catalog::IndexIdentifier;
pub use parqdb_meta::{
    DistanceMetric, IndexMetadata, IndexSnapshot, PostingEncoding, RelationReference,
};
