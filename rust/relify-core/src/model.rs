use std::collections::BTreeMap;

use relify_catalog::IndexIdentifier;
use relify_meta::{IndexMetadata, RelationReference};

/// Logical IVF construction options shared by backend implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfConfig {
    /// Number of IVF clusters.
    pub nlist: usize,
    /// Whether postings contain exact source vectors.
    pub store_vectors: bool,
}

impl IvfConfig {
    /// Creates an IVF configuration.
    #[must_use]
    pub const fn new(nlist: usize, store_vectors: bool) -> Self {
        Self {
            nlist,
            store_vectors,
        }
    }
}

/// User-supplied inputs for one vector search.
#[derive(Debug, Clone)]
pub struct SearchRequest {
    /// Exact portable reference for the source table.
    pub source: RelationReference,
    /// Explicit index name, or `None` for source-based discovery.
    pub index: Option<String>,
    /// Vector column used to disambiguate implicit index discovery.
    pub column: Option<String>,
    /// Query vector.
    pub query: Vec<f32>,
    /// Number of IVF clusters to probe.
    pub nprobe: Option<usize>,
    /// Maximum number of results.
    pub limit: usize,
    /// Optional ordered source-column projection.
    pub projection: Option<Vec<String>>,
    /// Optional backend expression evaluated against source rows before Top-K.
    pub filter: Option<String>,
    /// Whether to scan the source exactly without resolving a vector index.
    pub bypass_index: bool,
}

/// Immutable index relations and family parameters produced by a backend builder.
#[derive(Debug, Clone)]
pub struct IndexArtifacts {
    /// Family-defined canonical parameter values.
    pub parameters: BTreeMap<String, String>,
    /// Relation roles and their exact portable references.
    pub index_relations: BTreeMap<String, RelationReference>,
}

/// Result of publishing one immutable index metadata document.
#[derive(Debug, Clone)]
pub struct PublishedIndex {
    /// Structured catalog identifier that was published.
    pub identifier: IndexIdentifier,
    /// Immutable metadata file made current by the catalog.
    pub metadata_location: String,
    /// Validated metadata stored at `metadata_location`.
    pub metadata: IndexMetadata,
}
