use std::collections::BTreeMap;

use relify_catalog::IndexIdentifier;
use relify_meta::{IndexMetadata, PostingEncoding, RelationReference};

/// Portable identity of one index representation.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct IndexFormat {
    /// Index-family identifier.
    pub family: String,
    /// Family-defined schema version.
    pub schema_version: i32,
    /// Distance metric evaluated by the index.
    pub metric: String,
}

impl IndexFormat {
    /// Returns the legacy IVF format used by current builders.
    #[must_use]
    pub fn ivf_v1() -> Self {
        Self {
            family: "ivf".into(),
            schema_version: 1,
            metric: "l2_squared".into(),
        }
    }

    /// Returns IVF schema version 2.
    #[must_use]
    pub fn ivf_v2() -> Self {
        Self {
            family: "ivf".into(),
            schema_version: 2,
            metric: "l2_squared".into(),
        }
    }
}

/// Logical IVF construction options shared by backend implementations.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IvfConfig {
    /// Number of IVF clusters.
    pub nlist: usize,
    /// Vector representation stored in IVF postings.
    pub posting_encoding: PostingEncoding,
}

impl IvfConfig {
    /// Creates an IVF configuration with one canonical postings encoding.
    #[must_use]
    pub const fn new(nlist: usize, posting_encoding: PostingEncoding) -> Self {
        Self {
            nlist,
            posting_encoding,
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
    /// Portable family, schema version, and metric produced by the builder.
    pub format: IndexFormat,
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn posting_encodings_have_canonical_metadata_names() {
        for (encoding, name, legacy) in [
            (PostingEncoding::Source, "source", Some(false)),
            (PostingEncoding::Flat, "flat", Some(true)),
            (PostingEncoding::Lvq4, "lvq4", None),
            (PostingEncoding::Lvq8, "lvq8", None),
        ] {
            assert_eq!(encoding.as_str(), name);
            assert_eq!(encoding.v1_store_vectors(), legacy);
        }
    }
}
