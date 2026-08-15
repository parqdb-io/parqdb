//! Backend-neutral Relify index and query models.

mod model;

pub use model::{IndexArtifacts, IndexFormat, IvfConfig, PublishedIndex, SearchRequest};
pub use relify_catalog::IndexIdentifier;
pub use relify_meta::{
    DistanceMetric, IndexMetadata, IndexSnapshot, PostingEncoding, RelationReference,
};
