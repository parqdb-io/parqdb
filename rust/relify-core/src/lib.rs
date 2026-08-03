//! Backend-neutral Relify index and query models.

mod model;

pub use model::{IndexArtifacts, IvfConfig, PublishedIndex, SearchRequest};
pub use relify_catalog::IndexIdentifier;
pub use relify_meta::{IndexMetadata, IndexSnapshot, RelationReference};
